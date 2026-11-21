"""RARM reward (Yang et al., 2026), as in its released RL code (github.com/pengzhi1998/rarm, branch
`policy_training_RARM`: BinaryProgressRewardModel, drqv2_runner._compute_episode_rewards):

- Images: the RARM camera at 384 px, one every 4 env steps, the reset included.
- Reference: the reference demo's states rendered every 4 steps; a 5-image clip at every image
  (stride 1); clip j has progress (j + 2) / images. Gate: delta_j = max_k g(r_j, r_k) - 0.5.
- Each new image (once 5 exist) closes a clip of the last 5 images. j* = argmax g; if g >= delta_j*,
  pi_j* > p and p < final progress, then p = pi_j* and the clip reward is 1; otherwise 0. Progress
  starts at 0 (step-0 start) or (t + 1) / demo states (demo start).
- Per episode: a clip's reward goes to its middle image (i - 2); a reward of 1 also sets the two
  preceding images to 1; each env step takes the latest image at or before it. Shaping = that * 1.0.

Weights: Hugging Face `pengzhi1998/RARM`, checkpoint_epoch_45_generic.pt, used as released.
Matmuls use TF32 (scores change by < 0.04; the gate slack is 0.5).
"""

import hashlib
import json
import subprocess

import numpy as np
import torch

from .base import Episode, Reward
from .rarm_model import CHECKPOINT, CHECKPOINT_SHA256, CLIP, DIM, DINOV3_COMMIT, DINOV3_REPO, MID, OFFSET, RARM

INTERVAL = 4
CAMERA = "robot0_agentview_left"
SIZE = 384
RM_COEF = 1.0


def download():
    """Checkpoint (SHA-checked) and Meta's DINOv3 code at RARM's pinned commit."""
    from huggingface_hub import hf_hub_download

    RARM.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT.exists():
        hf_hub_download("pengzhi1998/RARM", CHECKPOINT.name, local_dir=RARM)
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise SystemExit(f"{CHECKPOINT} SHA-256 {digest} != expected {CHECKPOINT_SHA256}")
    if not DINOV3_REPO.exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/facebookresearch/dinov3.git", str(DINOV3_REPO)],
                       check=True)
    subprocess.run(["git", "-C", str(DINOV3_REPO), "checkout", "-q", DINOV3_COMMIT], check=True)


class RarmEpisode(Episode):
    def __init__(self, owner, start_progress):
        self.owner = owner
        self.progress = float(start_progress)
        self.window, self.clip_rewards, self.pending = [], [], []

    def add_image(self, image):
        self.pending.append(image)
        self.owner.queue.append(self)

    def finish(self, steps):
        self.owner.process()
        images = np.zeros(len(self.clip_rewards), np.float32)
        for i, r in enumerate(self.clip_rewards):
            if i < CLIP - 1:
                continue
            middle = max(0, i - MID)
            images[middle] = r
            if r == 1.0:
                images[max(0, middle - 2):middle] = 1.0
        image_steps = np.arange(len(images)) * INTERVAL
        which = np.clip(np.searchsorted(image_steps, np.arange(steps), side="right") - 1, 0, len(images) - 1)
        return images[which] * RM_COEF

    def summary(self):
        return dict(final_progress=round(self.progress, 3), clip_rewards=int(sum(self.clip_rewards)))


class RarmReward(Reward):
    name = "rarm"
    image_request = dict(camera=CAMERA, size=SIZE, interval=INTERVAL)

    def setup(self, render_reference):
        from .rarm_model import RarmModel

        download()
        torch.backends.cuda.matmul.allow_tf32 = True
        self.model = RarmModel(self.device)
        images = render_reference(CAMERA, SIZE, INTERVAL)
        with torch.inference_mode():
            tokens = self.model.frames(images)
            starts = list(range(len(images) - CLIP + 1))
            self.embeddings = self.model.clips(tokens, starts)
            del tokens
            self.progress = np.array([(s + MID) / len(images) for s in starts], np.float32)
            best = [float(self.model.scores(self.embeddings[i], self.embeddings).max()) for i in range(len(starts))]
        self.thresholds = np.asarray(best, np.float32) - OFFSET
        self.final = float(self.progress.max())
        from ..sim import Demo

        self.n_states = len(Demo(self.task.data_dir, self.task.reference_episode).states)
        self.queue = []
        torch.cuda.empty_cache()

    def new_episode(self, start, from_demo):
        return RarmEpisode(self, (start + 1) / self.n_states if from_demo else 0.0)

    @torch.inference_mode()
    def process(self):
        if not self.queue:
            return
        batch, self.queue = self.queue, []
        owners = []
        for ep in batch:  # each queued entry is one image of `ep`, in order
            owners.append((ep, ep.pending.pop(0)))
        tokens = self.model.frames(np.stack([img for _, img in owners]))
        for (ep, _), tok in zip(owners, tokens):
            ep.window = (ep.window + [tok])[-CLIP:]
            if len(ep.window) < CLIP:
                ep.clip_rewards.append(0.0)
                continue
            clip = self.model.temporal_processor(torch.stack(ep.window).float().reshape(1, -1, DIM))[0]
            scores = self.model.scores(clip, self.embeddings)
            best = int(scores.argmax())
            score = float(scores[best])
            matched = float(self.progress[best])
            reward = 0.0
            if ep.progress < self.final and score >= self.thresholds[best] and matched > ep.progress:
                ep.progress, reward = matched, 1.0
            ep.clip_rewards.append(reward)
