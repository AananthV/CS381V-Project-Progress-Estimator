"""Reward registry. To add a reward: subclass `base.Reward` in a new module, then add it here."""

REWARDS = {
    "sparse": ("sparse", "SparseReward"),
    "rarm": ("rarm", "RarmReward"),
}


def make_reward(name, task, device="cuda"):
    import importlib

    if name not in REWARDS:
        raise SystemExit(f"unknown reward {name!r}; known: {sorted(REWARDS)}")
    module, cls = REWARDS[name]
    return getattr(importlib.import_module(f"{__name__}.{module}"), cls)(task, device)
