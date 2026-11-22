"""Online max-product HMM (filtered Viterbi) tracking of the position along the reference demo.

States: (reference clip j, matched) and (j, unmatched). Per query clip with comparator scores g[j]:

- emission, matched j:   absolute  (g[j] - m_j) / tau          m_j = max_k g(r_j, r_k), RARM's gate scale
                         relative  (g[j] - max_k g[k]) / tau    judged against this query's best clip, so
                                                                 uniformly low scores (another kitchen) can match
  unmatched:             -gate_slack / tau                       at RARM's gate
- position moves: stay p_stay_pos; forward 1..k equally likely; back 1..back with p_back each; any
  other move log_skip (no skipping). Moves into an unmatched state keep the position, so progress only
  advances through matched states.
- matched -> unmatched p_leave, unmatched -> unmatched p_stay.
- start: the reference clip whose progress is closest to the start progress (anywhere else costs a skip).

After each clip the best state is read off. With the absolute emission "unmatched" wins whenever the
clip looks unlike the demo; with the relative one only when every clip reachable from the current
position is more than gate_slack below the best clip.
"""

from dataclasses import asdict, dataclass

import numpy as np

NEG = -1e9


@dataclass(frozen=True)
class HmmParams:
    emission: str = "absolute"  # or "relative"
    tau: float = 0.3  # emission temperature (score units per nat)
    k: int = 3  # largest forward move, reference clips per query clip; moves 1..k equally likely
    p_stay_pos: float = 0.5  # P(position stays)
    p_back: float = 0.01  # per backward step size, 1..back
    back: int = 2
    p_leave: float = 0.02  # matched -> unmatched
    p_stay: float = 0.9  # unmatched -> unmatched
    gate_slack: float = 0.5  # unmatched emission, RARM's Delta
    log_skip: float = -1e6  # any other move: effectively impossible

    def to_dict(self):
        return asdict(self)


class Hmm:
    """Shared precomputation for one reference; `step` advances one episode's state by one clip."""

    def __init__(self, progress, self_best, params=HmmParams()):
        if params.emission not in ("absolute", "relative"):
            raise ValueError(f"unknown emission {params.emission!r}")
        self.params, self.progress, self.self_best = params, np.asarray(progress), np.asarray(self_best)
        M = len(progress)
        d = np.arange(M)[None, :] - np.arange(M)[:, None]
        logA = np.full((M, M), params.log_skip)
        forward_mass = 1.0 - params.p_back * params.back
        logA[d == 0] = np.log(forward_mass * params.p_stay_pos)
        for k in range(1, params.k + 1):
            logA[d == k] = np.log(forward_mass * (1.0 - params.p_stay_pos) / params.k)
        for k in range(1, params.back + 1):
            logA[d == -k] = np.log(params.p_back)
        logA -= np.logaddexp.reduce(logA, axis=1, keepdims=True)
        hold = np.full((M, M), NEG)
        np.fill_diagonal(hold, 0.0)
        self.moves = (logA, hold)
        self.logF = np.log(np.array([[1 - params.p_leave, params.p_leave], [1 - params.p_stay, params.p_stay]]))

    def initial(self, start_progress):
        delta = np.full((2, len(self.progress)), NEG)
        delta[0] = self.params.log_skip
        j0 = int(np.abs(self.progress - start_progress).argmin()) if start_progress > 0 else 0
        delta[0, j0] = 0.0
        return delta

    def step(self, delta, scores):
        """scores: g of the new query clip vs every reference clip [M] -> (state, position, matched)."""
        p = self.params
        new = np.empty_like(delta)
        for f2 in range(2):
            cand = np.maximum(delta[0][:, None] + self.logF[0, f2], delta[1][:, None] + self.logF[1, f2])
            new[f2] = (cand + self.moves[f2]).max(0)
        top = scores.max() if p.emission == "relative" else self.self_best
        new[0] += (scores - top) / p.tau
        new[1] += -p.gate_slack / p.tau
        new -= new.max()
        f, j = np.unravel_index(int(new.argmax()), new.shape)
        return new, int(j), bool(f == 0)
