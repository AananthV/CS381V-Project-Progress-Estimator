"""Sparse baseline: no shaping. The step reward is only the fixed terms (success bonus, step cost)."""

from .base import Reward


class SparseReward(Reward):
    name = "sparse"
