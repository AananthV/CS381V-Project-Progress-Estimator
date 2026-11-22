"""Record the fixed episodes of an offline evaluation (configs/eval/<task>.yaml, `episodes`).

Every episode keeps its simulator states, so any camera/resolution can be re-rendered later:

    outputs/eval/<task>/episodes/<name>/
        agentview_left.mp4    robot0_agentview_left, 384 px, every env step (20 fps); frame 0 = start
        eye_in_hand.mp4       robot0_eye_in_hand, 256 px, every env step
        states.npz            states [T+1, S], actions [T, 7] (empty for demos)
        meta.json             kind, scene, start, noise seed, per-step success, gripper events
    outputs/eval/<task>/episodes/index.json

Kinds: expert (a demo's own states in its own scene), replay (the reference demo's actions, open loop),
random (uniform actions in [-1, 1], as RL's first 4,000 steps), rl (a checkpoint of
outputs/rl/<task>/<run>/seed_0; the benchmark's evaluation loop: 3 stacked frames, deterministic actions).

The simulator is not deterministic across resets (the same restored state and actions drift apart by
~0.05 within 20 steps), so a policy's outcome from a given start is partly chance. `want_success`
episodes try several noise seeds (for step-0 starts: those whose evaluation trial succeeded) and keep
the first success. Labels always come from the recording.

    pr-record --task add_sweetener

Run it with no other simulator jobs: each scene build holds about 2 GB of RAM. Existing episodes are
kept; delete a folder to re-record it.
"""

import argparse
import gc
import json

import numpy as np
import yaml

from ..data.task import load_task, run_dir
from ..paths import CONFIGS, EVAL
from ..sim.env import ACTION_DIM, Demo, SceneEnv

FPS = 20
MAIN_CAMERA, MAIN_SIZE = "robot0_agentview_left", 384
WRIST_CAMERA, WRIST_SIZE = "robot0_eye_in_hand", 256
GRIPPER = 6  # gripper command in the 7-d agent action (> 0 closes)
ENV_SEED = 10_000 * 100  # pr-eval-rl's first worker
MAX_ATTEMPTS = 6


def load_eval_config(task_name):
    return yaml.safe_load((CONFIGS / "eval" / f"{task_name}.yaml").read_text())


def write_video(path, images, fps=FPS):
    import av

    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.height, stream.width = images[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        for image in images:
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def gripper_events(closed):
    change = np.flatnonzero(np.diff(np.asarray(closed, int))) + 1
    return [(int(t), "grasp" if closed[t] else "release") for t in change]


def stack(frames, k):  # rl/evaluation.py
    last = frames[-k:]
    return np.concatenate([last[0]] * (k - len(last)) + last, axis=0)


class Recorder:
    def __init__(self, env):
        self.env = env
        self.main, self.wrist, self.states, self.successes = [], [], [], []

    def capture(self, success):
        env = self.env
        self.main.append(env.render(MAIN_CAMERA, MAIN_SIZE))
        self.wrist.append(env.render(WRIST_CAMERA, WRIST_SIZE))
        self.states.append(env.env.sim.get_state().flatten().copy())
        self.successes.append(bool(success))


def run_demo(env):
    rec = Recorder(env)
    for state in env.demo.states:
        env.set_state(state)
        rec.capture(env.success())
    return rec, np.zeros((0, ACTION_DIM)), env.demo.actions[:, GRIPPER] > 0


def run_policy(env, task, start, noise_seed, act):
    """The benchmark evaluation loop (one environment), plus renders and a state copy per step."""
    obs = env.reset(start, noise_seed)
    rec = Recorder(env)
    rec.capture(obs["success"])
    frames, actions, held = [obs["policy"]], [], 0
    for step in range(task.horizon):
        action = np.clip(act(step, frames), -1, 1)
        obs = env.step(action)
        actions.append(action)
        frames = frames[-(task.frame_stack - 1):] + [obs["policy"]]
        rec.capture(obs["success"])
        held = held + 1 if obs["success"] else 0
        if held >= task.success_hold_steps:
            break
    actions = np.asarray(actions)
    return rec, actions, actions[:, GRIPPER] > 0


def make_policy(task, spec, agents):
    kind = spec["kind"]
    if kind == "random":
        rng = np.random.default_rng(spec["seed"])
        return lambda step, frames: rng.uniform(-1, 1, ACTION_DIM)
    if kind == "replay":
        demo_actions = Demo(task.data_dir, task.reference_episode).agent_actions()
        return lambda step, frames: demo_actions[min(step, len(demo_actions) - 1)]
    import torch

    from ..rl.drqv2 import DrQV2

    key = (spec["run"], spec["checkpoint"])
    if key not in agents:
        agent = DrQV2(task.channels, ACTION_DIM, "cuda")
        path = run_dir(task.name, spec["run"], 0) / "checkpoints" / f"{spec['checkpoint']}.pt"
        state = torch.load(path, map_location="cuda", weights_only=False)
        agent.encoder.load_state_dict(state["encoder"])
        agent.actor.load_state_dict(state["actor"])
        agents[key] = agent
    agent = agents[key]

    def act(step, frames):
        with torch.no_grad():
            return agent.act(stack(frames, task.frame_stack)[None], step=10**9, eval_mode=True)[0]

    return act


def attempt_seeds(task, spec):
    """Noise seeds to try: step-0 starts that should succeed use the seeds whose evaluation trial did."""
    if not spec.get("want_success"):
        return [spec.get("noise_seed")]
    if spec.get("noise_seed") is None:
        return [None] * MAX_ATTEMPTS
    results = json.loads((run_dir(task.name, spec["run"], 0) / "eval" / "results.json").read_text())
    good = [t["noise_seed"] for t in results[spec["checkpoint"]]["trials"] if t["success"]]
    return (good or [spec["noise_seed"]])[:MAX_ATTEMPTS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="add_sweetener")
    parser.add_argument("--only", nargs="*", help="record only these episode names")
    args = parser.parse_args()
    task = load_task(args.task)
    out = EVAL / task.name / "episodes"
    out.mkdir(parents=True, exist_ok=True)
    specs = load_eval_config(task.name)["episodes"]
    for spec in specs.values():
        spec.setdefault("scene", task.reference_episode)

    todo = [n for n in specs if (not args.only or n in args.only) and not (out / n / "meta.json").exists()]
    agents, env = {}, None
    for name in sorted(todo, key=lambda n: specs[n]["scene"] != task.reference_episode):  # reference scene first
        spec = specs[name]
        if env is None or env.demo.episode != spec["scene"]:
            if env is not None:
                env.close()
                del env
                gc.collect()
            env = SceneEnv(task, seed=ENV_SEED, episode=spec["scene"])
            print(f"built scene of demo {spec['scene']}", flush=True)
        tries = []
        if spec["kind"] == "expert":
            rec, actions, closed = run_demo(env)
        else:
            act = make_policy(task, spec, agents)
            for noise_seed in attempt_seeds(task, spec):
                rec, actions, closed = run_policy(env, task, spec.get("start", 0), noise_seed, act)
                tries.append(dict(noise_seed=noise_seed, success=any(rec.successes)))
                if not spec.get("want_success") or any(rec.successes):
                    break
                print(f"{name}: attempt {len(tries)} (noise seed {noise_seed}) failed", flush=True)
        folder = out / name
        folder.mkdir(exist_ok=True)
        write_video(folder / "agentview_left.mp4", rec.main)
        write_video(folder / "eye_in_hand.mp4", rec.wrist)
        np.savez_compressed(folder / "states.npz", states=np.asarray(rec.states), actions=actions)
        successes = rec.successes
        demo = env.demo
        meta = dict(name=name, task=task.name, **spec,
                    scene_meta=dict(demo=demo.episode, layout=demo.metadata["layout_id"],
                                    style=demo.metadata["style_id"], cubes=demo.metadata["refs"]["num_sugar_cubes"]),
                    reference_demo=task.reference_episode,
                    rl_run=dict(reward=spec["run"], seed=0) if spec["kind"] == "rl" else None,
                    attempts=tries, frames=len(rec.main), steps=len(rec.main) - 1, fps=FPS,
                    success=any(successes), first_success=successes.index(True) if any(successes) else None,
                    successes=successes, gripper_events=gripper_events(closed),
                    cameras={MAIN_CAMERA: MAIN_SIZE, WRIST_CAMERA: WRIST_SIZE})
        if tries:
            meta["noise_seed"] = tries[-1]["noise_seed"]
        (folder / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"{name}: {meta['steps']} steps, success {meta['success']}"
              + (f", {len(tries)} attempt(s)" if len(tries) > 1 else ""), flush=True)
    if env is not None:
        env.close()

    index = []
    for name in specs:
        path = out / name / "meta.json"
        if path.exists():
            m = json.loads(path.read_text())
            index.append({k: m.get(k) for k in ("name", "kind", "scene", "start", "noise_seed", "checkpoint",
                                                "expected", "steps", "success", "first_success")})
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print("wrote", out / "index.json")


if __name__ == "__main__":
    main()
