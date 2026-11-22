"""Matchers turn comparator scores into progress along the reference demo.

A matcher has `initial(start_progress) -> state` and `step(state, scores) -> (state, position, matched)`.
Progress follows RARM's ratchet for every matcher: it moves to the matched clip's progress when that is
above every earlier value, and each move is a clip reward of 1. The online reward (rewards/rarm.py) and
the offline evaluation (`track`) run exactly this code.
"""

import numpy as np

from .hmm import Hmm, HmmParams
from .rule import RarmRule


def make_matcher(config, progress, self_best):
    """config: a reward config (`matcher: rule | hmm`, optional `hmm:` parameters)."""
    kind = config.get("matcher", "rule")
    if kind == "rule":
        return RarmRule(progress, self_best, config.get("gate_slack", 0.5))
    if kind == "hmm":
        return Hmm(progress, self_best, HmmParams(**config.get("hmm", {})))
    raise ValueError(f"unknown matcher {kind!r}")


def track(matcher, S, progress, start_progress=0.0):
    """Offline: run a matcher over a score matrix S [query clips, reference clips].
    Returns one row per query clip: position, matched, tracked (its progress), progress, reward."""
    state, best, rows = matcher.initial(start_progress), float(start_progress), []
    for scores in S:
        state, j, matched = matcher.step(state, scores)
        reward = int(matched and progress[j] > best)
        if reward:
            best = float(progress[j])
        rows.append(dict(position=j, matched=matched, tracked=float(progress[j]), progress=best, reward=reward))
    return rows


def summary(rows):
    p = np.array([r["progress"] for r in rows])
    jumps = np.diff(np.concatenate([[p[0]], p])) if len(p) else np.zeros(0)
    return dict(final=float(p[-1]) if len(p) else 0.0, rewards=int(sum(r["reward"] for r in rows)),
                max_jump=float(jumps.max()) if len(jumps) else 0.0)
