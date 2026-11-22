"""progress_reward.matching reproduces the pre-reorganization matchers exactly (tests/reference_matching.py)."""

import dataclasses

import numpy as np
import pytest

import reference_matching as ref
from progress_reward.matching import make_matcher, summary, track
from progress_reward.matching.hmm import Hmm, HmmParams


def problem(seed, T=60, M=40):
    rng = np.random.default_rng(seed)
    progress = ((np.arange(M) + 2) / (M + 4)).astype(np.float32)
    self_best = rng.uniform(3, 6, M).astype(np.float32)
    # a noisy diagonal: matches move forward with the query, plus look-alikes elsewhere
    S = rng.normal(0, 1.5, (T, M)).astype(np.float32)
    for t in range(T):
        j = min(M - 1, int(t * M / T))
        S[t, j] += self_best[j] + rng.normal(0, 0.5)
    return S, self_best, progress


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("start", [0.0, 0.33])
def test_rule_matches_reference(seed, start):
    S, sb, progress = problem(seed)
    new = track(make_matcher({"matcher": "rule"}, progress, sb), S, progress, start)
    old = ref.rarm(S, sb, progress, start)
    assert [r["position"] for r in new] == [r["best"] for r in old]
    assert [r["matched"] for r in new] == [r["confident"] for r in old]
    assert [r["progress"] for r in new] == [r["progress"] for r in old]
    assert [r["reward"] for r in new] == [r["reward"] for r in old]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("start", [0.0, 0.33])
@pytest.mark.parametrize("emission", ["absolute", "relative"])
def test_hmm_matches_reference(seed, start, emission):
    S, sb, progress = problem(seed)
    new = track(Hmm(progress, sb, HmmParams(emission=emission)), S, progress, start)
    old = ref.hmm(S, sb, progress, dataclasses.replace(ref.HmmParams(), emission=emission), start)
    for key in ("position", "matched", "tracked", "progress", "reward"):
        assert [r[key] for r in new] == [r[key] for r in old], key


def test_relative_ignores_a_constant_shift():
    """The relative emission depends only on scores relative to the best clip (another kitchen)."""
    S, sb, progress = problem(0)
    params = HmmParams(emission="relative")
    a = track(Hmm(progress, sb, params), S, progress)
    b = track(Hmm(progress, sb, params), S - 5.0, progress)
    assert [r["position"] for r in a] == [r["position"] for r in b]
    absolute = track(Hmm(progress, sb), S - 5.0, progress)
    assert summary(absolute)["final"] < summary(a)["final"]


def test_make_matcher_from_config():
    S, sb, progress = problem(1)
    m = make_matcher({"matcher": "hmm", "hmm": {"emission": "relative", "k": 3}}, progress, sb)
    assert isinstance(m, Hmm) and m.params.emission == "relative"
    with pytest.raises(ValueError):
        make_matcher({"matcher": "nope"}, progress, sb)
