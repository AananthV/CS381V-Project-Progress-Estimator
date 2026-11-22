"""RARM-family rewards: a comparator (RARM as released, or with fine-tuned heads) and a matcher (RARM's
rule, or the HMM with the absolute or relative emission), with RARM's images and credit assignment
(released RL code: github.com/pengzhi1998/rarm, branch `policy_training_RARM`,
BinaryProgressRewardModel and drqv2_runner._compute_episode_rewards):

- Images: the RARM camera at 384 px, one every 4 env steps, the reset included.
- Reference: the reference demo's states rendered every 4 steps; a 5-image clip at every image
  (stride 1); clip j has progress (j + 2) / images; m_j = max_k g(r_j, r_k).
- Each new image (once 5 exist) closes a query clip of the last 5 images; the matcher gives a position
  and whether it matched; progress ratchets up to the matched clip's progress and the clip reward is 1
  on each rise. Progress starts at 0 (step-0 start) or (t + 1) / demo states (demo start t).
- Per episode: a clip's reward goes to its middle image (i - 2); a reward of 1 also sets the two
  preceding images to 1; each env step takes the latest image at or before it. Shaping = that * 1.0.

The configuration `rarm.yaml` (original comparator, rule) is RARM as released. Matmuls use TF32
(scores change by < 0.04; the gate slack is 0.5).
"""

import numpy as np
import torch

from ..comparator.rarm import CLIP, DIM, MID
from ..matching import make_matcher
from .base import Episode, Reward

INTERVAL = 4
CAMERA = "robot0_agentview_left"
SIZE = 384
RM_COEF = 1.0


class RarmEpisode(Episode):
    def __init__(self, owner, start_progress):
        self.owner = owner
        self.progress = float(start_progress)
        self.state = owner.matcher.initial(start_progress)
        self.window, self.clip_rewards, self.pending = [], [], []
        self.unmatched = 0

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
        return dict(final_progress=round(self.progress, 3), clip_rewards=int(sum(self.clip_rewards)),
                    unmatched_clips=self.unmatched)


class RarmReward(Reward):
    image_request = dict(camera=CAMERA, size=SIZE, interval=INTERVAL)

    def __init__(self, task, device="cuda", config=None):
        super().__init__(task, device)
        self.config = dict(config or {}, task=task.name)

    def setup(self, render_reference):
        from ..comparator.heads import load_comparator
        from ..sim.env import Demo

        torch.backends.cuda.matmul.allow_tf32 = True
        self.model = load_comparator(self.config, self.device)
        images = render_reference(CAMERA, SIZE, INTERVAL)
        with torch.inference_mode():
            tokens = self.model.frames(images)
            starts = list(range(len(images) - CLIP + 1))
            self.embeddings = self.model.clips(tokens, starts)
            del tokens
            self.progress = np.array([(s + MID) / len(images) for s in starts], np.float32)
            best = [float(self.model.scores(self.embeddings[i], self.embeddings).max()) for i in range(len(starts))]
        self.self_best = np.asarray(best, np.float32)
        self.matcher = make_matcher(self.config, self.progress, self.self_best)
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
        owners = [(ep, ep.pending.pop(0)) for ep in batch]  # each queued entry is one image of `ep`, in order
        tokens = self.model.frames(np.stack([img for _, img in owners]))
        for (ep, _), tok in zip(owners, tokens):
            ep.window = (ep.window + [tok])[-CLIP:]
            if len(ep.window) < CLIP:
                ep.clip_rewards.append(0.0)
                continue
            clip = self.model.temporal_processor(torch.stack(ep.window).float().reshape(1, -1, DIM))[0]
            scores = self.model.scores(clip, self.embeddings).float().cpu().numpy()
            ep.state, j, matched = self.matcher.step(ep.state, scores)
            reward = 0.0
            if matched and self.progress[j] > ep.progress:
                ep.progress, reward = float(self.progress[j]), 1.0
            ep.unmatched += int(not matched)
            ep.clip_rewards.append(reward)
