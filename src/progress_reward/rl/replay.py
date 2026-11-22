"""Disk-backed DrQ-v2 replay: whole episodes appended to memory-mapped files; batches with n-step
returns and frame stacks, sampled from the newest `capacity` transitions (RARM default 750,000).

A step's frames (C x 84 x 84 uint8, all policy cameras) take ~21 KB per camera, so a full buffer is
tens of GB and lives on disk (`--replay-dir`; keep it on a local disk, not a synced folder).
Sampler processes (`samplers`) build batches into shared-memory slots. They import only NumPy, so
each stays small (the earlier torch-importing samplers grew to ~1.7 GB each).

Sampling follows DrQ-v2: transition t of an episode with T transitions is valid if t + n <= T;
reward = sum_i gamma^i r_{t+i} prod_{k<i} d_{t+k}, discount = gamma^n prod d; observation = frames
t-(stack-1)..t (repeating the episode's first frame), next observation = frames up to t + n.
"""

import json
import multiprocessing as mp
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np


def _open(folder, frame_shape, frame_cap, trans_cap, action_dim, mode):
    mm = lambda name, dtype, shape: np.memmap(Path(folder) / f"{name}.bin", dtype=dtype, mode=mode, shape=shape)
    return dict(
        frames=mm("frames", np.uint8, (frame_cap,) + tuple(frame_shape)),
        ep_start=mm("ep_start", np.int64, (frame_cap,)),       # per frame: first frame of its episode
        obs_frame=mm("obs_frame", np.int64, (trans_cap,)),     # per transition
        action=mm("action", np.float32, (trans_cap, action_dim)),
        reward=mm("reward", np.float32, (trans_cap,)),
        disc=mm("disc", np.float32, (trans_cap,)),
        valid=mm("valid", np.bool_, (trans_cap,)),
    )


def sample(arrays, n, rng, hi, capacity, nstep, discount, stack):
    lo = max(0, hi - capacity)
    idx = []
    while len(idx) < n:
        cand = rng.integers(lo, hi, 2 * n)
        idx.extend(cand[arrays["valid"][cand]][: n - len(idx)])
    g = np.asarray(idx, np.int64)
    steps = g[:, None] + np.arange(nstep)[None]
    rewards, discs = arrays["reward"][steps], arrays["disc"][steps]
    ret = np.zeros(n, np.float32)
    factor = np.ones(n, np.float32)
    for i in range(nstep):
        ret += factor * rewards[:, i]
        factor *= discs[:, i] * discount
    f = arrays["obs_frame"][g]
    start = arrays["ep_start"][f]
    offsets = np.arange(-(stack - 1), nstep + 1)
    frame_idx = np.maximum(f[:, None] + offsets[None], start[:, None])
    block = arrays["frames"][frame_idx.ravel()].reshape(n, len(offsets), *arrays["frames"].shape[1:])
    h, w = block.shape[-2:]
    obs = block[:, :stack].reshape(n, -1, h, w)
    nxt = block[:, -stack:].reshape(n, -1, h, w)
    return obs, arrays["action"][g], ret[:, None], factor[:, None], nxt


class Replay:
    FIELDS = ("obs", "action", "reward", "disc", "next")

    def __init__(self, folder, total_steps, frame_shape, action_dim, capacity=750_000, nstep=3,
                 discount=0.99, batch_size=256, stack=3, resume=False):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.frame_shape, self.action_dim = tuple(frame_shape), action_dim
        self.capacity, self.nstep, self.discount = capacity, nstep, discount
        self.batch_size, self.stack = batch_size, stack
        self.frame_cap, self.trans_cap = total_steps + 50_000, total_steps + 1
        self.arrays = _open(self.folder, frame_shape, self.frame_cap, self.trans_cap, action_dim,
                            "r+" if resume else "w+")
        self.n_frames = self.n_trans = 0
        if resume:
            counts = json.loads((self.folder / "counts.json").read_text())
            self.n_frames, self.n_trans = counts["n_frames"], counts["n_trans"]
        self.processes, self.shms = [], []

    def __len__(self):
        return self.n_trans

    def add_episode(self, frames, actions, rewards, discounts):
        """frames [T+1, C, H, W] uint8, actions [T, A], rewards [T], discounts [T] (0 at a terminal)."""
        a, t = self.arrays, len(actions)
        f0, g0 = self.n_frames, self.n_trans
        if g0 + t > self.trans_cap or f0 + t + 1 > self.frame_cap:
            raise RuntimeError("replay files full")
        a["frames"][f0:f0 + t + 1] = frames
        a["ep_start"][f0:f0 + t + 1] = f0
        a["obs_frame"][g0:g0 + t] = np.arange(f0, f0 + t)
        a["action"][g0:g0 + t] = actions
        a["reward"][g0:g0 + t] = rewards
        a["disc"][g0:g0 + t] = discounts
        valid = np.zeros(t, bool)
        valid[: max(0, t - self.nstep + 1)] = True
        a["valid"][g0:g0 + t] = valid
        self.n_frames, self.n_trans = f0 + t + 1, g0 + t
        for array in a.values():
            array.flush()
        if self.processes:
            self.shared_n.value = self.n_trans

    def save_counts(self):
        for array in self.arrays.values():
            array.flush()
        (self.folder / "counts.json").write_text(json.dumps(dict(n_frames=self.n_frames, n_trans=self.n_trans)))

    # ---- sampler processes ----

    def _shapes(self):
        b, c = self.batch_size, self.frame_shape[0] * self.stack
        h, w = self.frame_shape[1:]
        return dict(obs=((b, c, h, w), np.uint8), action=((b, self.action_dim), np.float32),
                    reward=((b, 1), np.float32), disc=((b, 1), np.float32), next=((b, c, h, w), np.uint8))

    def start(self, samplers=2, slots=4, seed=0):
        ctx = mp.get_context("spawn")
        self.shared_n = ctx.Value("q", self.n_trans)
        self.free, self.ready = ctx.Queue(), ctx.Queue()
        shapes = self._shapes()
        self.slots = []
        for _ in range(slots):
            slot = {}
            for name, (shape, dtype) in shapes.items():
                shm = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)) * np.dtype(dtype).itemsize)
                self.shms.append(shm)
                slot[name] = (shm.name, np.ndarray(shape, dtype, buffer=shm.buf))
            self.slots.append(slot)
        for k in range(slots):
            self.free.put(k)
        config = dict(folder=str(self.folder), frame_shape=self.frame_shape, frame_cap=self.frame_cap,
                      trans_cap=self.trans_cap, action_dim=self.action_dim, capacity=self.capacity,
                      nstep=self.nstep, discount=self.discount, batch_size=self.batch_size, stack=self.stack,
                      slots=[{n: (name, shapes[n]) for n, (name, _) in s.items()} for s in self.slots])
        for i in range(samplers):
            p = ctx.Process(target=_sampler, args=(config, self.shared_n, self.free, self.ready, seed * 100 + i),
                            daemon=True)
            p.start()
            self.processes.append(p)

    def sample(self, device):
        """Next batch as torch tensors on `device`."""
        import torch

        k = self.ready.get()
        slot = self.slots[k]
        batch = tuple(torch.from_numpy(slot[name][1]).to(device) for name in self.FIELDS)
        if str(device).startswith("cuda"):
            torch.cuda.current_stream().synchronize()
        self.free.put(k)
        return batch

    def stop(self):
        for p in self.processes:
            p.terminate()
        for shm in self.shms:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass


def _sampler(config, shared_n, free, ready, seed):
    arrays = _open(config["folder"], config["frame_shape"], config["frame_cap"], config["trans_cap"],
                   config["action_dim"], "r")
    slots = []
    for slot in config["slots"]:
        views = {}
        for name, (shm_name, (shape, dtype)) in slot.items():
            shm = shared_memory.SharedMemory(name=shm_name)
            views[name] = (shm, np.ndarray(shape, dtype, buffer=shm.buf))
        slots.append(views)
    rng = np.random.default_rng(seed)
    while True:
        k = free.get()
        while shared_n.value < config["batch_size"]:
            time.sleep(0.5)
        out = sample(arrays, config["batch_size"], rng, shared_n.value, config["capacity"], config["nstep"],
                     config["discount"], config["stack"])
        for name, value in zip(Replay.FIELDS, out):
            slots[k][name][1][:] = value
        del out
        ready.put(k)
