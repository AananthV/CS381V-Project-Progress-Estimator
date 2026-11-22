"""Reward registry: a reward is configs/rewards/<name>.yaml.

    type: sparse | rarm
    comparator: original | finetuned        (rarm)
    matcher: rule | hmm                     (rarm)
    hmm: {emission: absolute | relative, tau: ..., ...}   (matcher hmm; progress_reward.matching.hmm.HmmParams)
    heads: outputs/finetune/{task}/model/heads.pt          (comparator finetuned)
    heads_sha256: {<task>: <sha256>}

To add a reward: write its YAML (and, for a new type, a `Reward` subclass registered in TYPES).
"""

import yaml

from ..paths import CONFIGS

REWARDS = CONFIGS / "rewards"
TYPES = {"sparse": ("sparse", "SparseReward"), "rarm": ("rarm", "RarmReward")}


def known():
    return sorted(p.stem for p in REWARDS.glob("*.yaml"))


def reward_config(name, task_name=None):
    path = REWARDS / f"{name}.yaml"
    if not path.exists():
        raise SystemExit(f"unknown reward {name!r}; known: {known()}")
    config = yaml.safe_load(path.read_text()) or {}
    config["name"] = name
    if task_name and "heads" in config:
        config["heads"] = config["heads"].format(task=task_name)
        config["heads_sha256"] = (config.get("heads_sha256") or {}).get(task_name)
        config["heads_url"] = (config.get("heads_url") or {}).get(task_name)
    return config


def make_reward(name, task, device="cuda"):
    import importlib

    config = reward_config(name, task.name)
    module, cls = TYPES[config.get("type", "rarm")]
    reward_cls = getattr(importlib.import_module(f"{__name__}.{module}"), cls)
    reward = reward_cls(task, device, config) if config.get("type", "rarm") == "rarm" else reward_cls(task, device)
    reward.name = name
    return reward
