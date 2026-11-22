"""Frozen copy of the matching code used for the results before the reorganization
(src/rarm-subtask-finetune/matching.py, 2026-11-22). Tests check that progress_reward.matching reproduces it."""

from dataclasses import asdict, dataclass

import numpy as np

GATE_SLACK = 0.5  # RARM's Delta
NEG = -1e9


@dataclass
class HmmParams:
    tau: float = 0.3  # emission temperature (score units per nat)
    # log eps for any move outside [-B, K]. Any finite value only delays a skip: a policy hovering in
    # a look-alike state piles up evidence (e.g. -60 was overcome after ~36 images), so no skips at all
    log_skip: float = -1e6
    p_stay_pos: float = 0.5  # P(position stays), see forward_moves
    k: int = 3  # largest forward move (reference clips per query image), all moves 1..k equally likely
    p_back: float = 0.01  # per backward step size, |d| <= back
    back: int = 2
    p_leave: float = 0.02  # matched -> unmatched
    p_stay: float = 0.9  # unmatched -> unmatched
    gate_slack: float = GATE_SLACK
    unmatched: bool = True
    emission: str = "absolute"  # or "relative" (see module docstring)

    def to_dict(self):
        return asdict(self)


# ---------------- RARM (paper) ----------------

def rarm(S, self_best, progress, start_progress=0.0):
    p, final = float(start_progress), float(progress.max())
    rows = []
    for t in range(len(S)):
        j = int(S[t].argmax())
        confident = S[t, j] >= self_best[j] - GATE_SLACK
        reward = int(p < final and confident and progress[j] > p)
        if reward:
            p = float(progress[j])
        rows.append(dict(best=j, confident=bool(confident), progress=p, reward=reward))
    return rows


# ---------------- HMM ----------------

def forward_moves(params):
    """P(d) for d = 0..K: P(stay) = params.p_stay_pos, the rest split equally over 1..K.
    Equal odds mean every move costs the same, so the path moves only on visual evidence but
    can take up to K clips per image when the evidence is there (a fast policy is not penalized)."""
    k = params.k
    return np.concatenate([[params.p_stay_pos], np.full(k, (1.0 - params.p_stay_pos) / k)])


def _log_moves(M, params):
    """[M, M] log P(position i -> j), rows normalized."""
    d = np.arange(M)[None, :] - np.arange(M)[:, None]
    logA = np.full((M, M), float(params.log_skip))
    forward_mass = 1.0 - params.p_back * params.back
    for k, p in enumerate(forward_moves(params)):
        if p > 0:
            logA[d == k] = np.log(forward_mass * p)
    for k in range(1, params.back + 1):
        logA[d == -k] = np.log(params.p_back)
    logA -= np.logaddexp.reduce(logA, axis=1, keepdims=True)
    return logA


def hmm(S, self_best, progress, params=HmmParams(), start_progress=0.0):
    T, M = S.shape
    logA = _log_moves(M, params)
    if params.emission == "relative":
        em_matched = (S - S.max(1, keepdims=True)) / params.tau
    else:
        em_matched = (S - self_best[None, :]) / params.tau
    em_unmatched = -params.gate_slack / params.tau
    j0 = int(np.abs(progress - start_progress).argmin()) if start_progress > 0 else 0
    if params.unmatched:
        logF = np.log(np.array([[1 - params.p_leave, params.p_leave], [1 - params.p_stay, params.p_stay]]))
    else:
        logF = np.array([[0.0, NEG], [NEG, NEG]])
    delta = np.full((2, M), NEG)
    delta[0] = params.log_skip  # soft start: anywhere is possible, at the cost of one skip
    delta[0, j0] = 0.0
    # entering or staying in "unmatched" keeps the position exactly: the position can only advance
    # through matched states, so a path cannot slide through clips it never matched
    hold = np.full((M, M), NEG)
    np.fill_diagonal(hold, 0.0)
    moves = (logA, hold)
    rows, best_progress = [], float(start_progress)
    for t in range(T):
        new = np.empty((2, M))
        for f2 in range(2):
            cand = np.maximum(delta[0][:, None] + logF[0, f2], delta[1][:, None] + logF[1, f2]) + moves[f2]
            new[f2] = cand.max(0)
        new[0] += em_matched[t]
        new[1] += em_unmatched if params.unmatched else NEG
        new -= new.max()
        delta = new
        f, j = np.unravel_index(int(delta.argmax()), delta.shape)
        matched = f == 0
        reward = int(matched and progress[j] > best_progress)
        if reward:
            best_progress = float(progress[j])
        rows.append(dict(position=int(j), matched=bool(matched), tracked=float(progress[j]),
                         progress=best_progress, reward=reward))
    return rows


