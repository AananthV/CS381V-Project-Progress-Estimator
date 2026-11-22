"""Configs load and are consistent: rewards, the task, the evaluation set."""

import yaml

from progress_reward.data.task import load_task
from progress_reward.matching.hmm import HmmParams
from progress_reward.paths import CONFIGS
from progress_reward.rewards import known, reward_config

ABLATION = {"rarm", "rarm_hmm", "rarm_hmm_rel", "rarm_ft", "rarm_ft_hmm", "rarm_ft_hmm_rel"}


def test_reward_grid_is_complete():
    assert ABLATION | {"sparse"} == set(known())
    seen = set()
    for name in ABLATION:
        c = reward_config(name, "add_sweetener")
        emission = c.get("hmm", {}).get("emission") if c["matcher"] == "hmm" else None
        seen.add((c["comparator"], c["matcher"], emission))
        if c["matcher"] == "hmm":
            HmmParams(**c["hmm"])  # every key is a real parameter
        if c["comparator"] == "finetuned":
            assert c["heads"] == "outputs/finetune/add_sweetener/model/heads.pt"
            assert len(c["heads_sha256"]) == 64
    assert len(seen) == 6


def test_hmm_settings_differ_only_in_emission():
    rel, ab = reward_config("rarm_ft_hmm_rel")["hmm"], reward_config("rarm_ft_hmm")["hmm"]
    assert {k for k in rel if rel[k] != ab[k]} == {"emission"}
    assert reward_config("rarm_hmm")["hmm"] == ab


def test_task_and_eval_config():
    task = load_task("add_sweetener")
    split = task.load_split()
    cfg = yaml.safe_load((CONFIGS / "eval" / "add_sweetener.yaml").read_text())
    assert cfg["references"]["same"] == [task.reference_episode]
    other = cfg["references"]["test"] + cfg["references"]["dev"]
    assert set(other) <= set(split["reward_heldout"])
    assert not set(other) & set(split["reward_train"])
    assert set(cfg["settings"]) == ABLATION and set(cfg["headline"]) <= ABLATION
