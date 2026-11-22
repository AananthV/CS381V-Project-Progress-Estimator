"""RARM's matching rule (Yang et al., 2026, Eq. 2-3 and App. C; released `BinaryProgressRewardModel`).

Each query clip is matched on its own: j* = argmax_j g(q, r_j); the match counts if
g(q, r_j*) >= m_j* - gate_slack, with m_j = max_k g(r_j, r_k) the reference clip's best self-score.
There is no memory across clips, so one look-alike clip can move progress anywhere ahead.
"""

import numpy as np


class RarmRule:
    def __init__(self, progress, self_best, gate_slack=0.5):
        self.progress, self.gate = progress, np.asarray(self_best, np.float32) - np.float32(gate_slack)

    def initial(self, start_progress):
        return None

    def step(self, state, scores):
        """scores: g of the new query clip vs every reference clip [M] -> (state, position, matched)."""
        j = int(np.argmax(scores))
        return state, j, bool(scores[j] >= self.gate[j])
