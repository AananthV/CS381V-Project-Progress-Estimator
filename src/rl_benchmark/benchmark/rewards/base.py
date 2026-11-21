"""The reward interface. A reward supplies only a per-step *shaping* term; the benchmark adds the
fixed terms (task YAML `reward_terms`): step reward = shaping + success * 5 - 0.5.

Lifecycle, per training run:

    reward = make_reward(name, task, device)
    reward.setup(render_reference)            # once; may render reference images from the demo
    ep = reward.new_episode(start, from_demo) # per episode, per environment
    ep.add_image(image)                       # every `image_request["interval"]` env steps
    reward.process()                          # trainer calls this often: batch GPU work here
    shaping = ep.finish(steps)                # at episode end -> float array [steps]

Rewards that need no images set `image_request = None`. Shaping is computed at episode end, so a
reward may assign credit to earlier steps (RARM does). Episodes from several environments are
open at once; `process()` should handle all of them together.
"""

import numpy as np


class Episode:
    def add_image(self, image: np.ndarray) -> None:
        """uint8 [H, W, 3] image, in order; the first is the start state."""

    def finish(self, steps: int) -> np.ndarray:
        """Per-step shaping for the episode's `steps` transitions."""
        return np.zeros(steps, np.float32)

    def summary(self) -> dict:
        """Optional numbers logged with the episode (e.g. final progress)."""
        return {}


class Reward:
    name = "base"
    # dict(camera=..., size=..., interval=...) if the reward needs images, else None
    image_request = None

    def __init__(self, task, device="cuda"):
        self.task, self.device = task, device

    def setup(self, render_reference) -> None:
        """`render_reference(camera, size, interval)` -> uint8 [N, H, W, 3] images of the reference
        demo's states every `interval` steps, rendered in the simulator."""

    def new_episode(self, start: int, from_demo: bool) -> Episode:
        """`start`: the reference-demo step the episode starts from; `from_demo`: a demo-state start
        (else demo step 0 + noise)."""
        return Episode()

    def process(self) -> None:
        """Do pending work (e.g. score queued images on the GPU)."""
