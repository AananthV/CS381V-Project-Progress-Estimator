"""Metrics of one reward setting on one reference, over final progress of labeled episodes.

- auroc: successes vs failures (ties count 1/2)
- gap: lowest success minus highest failure (> 0: a threshold separates them)
- success_final: mean final progress of the successes
- tracking_mae: successes only, mean |progress - true| over scored images, with true = start progress +
  (1 - start progress) * min(1, step / first success step)
- random_max: highest final progress of a random-action episode
"""

import numpy as np


def auroc(pos, neg):
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def tracking_mae(progress, steps, start_progress, first_success):
    true = start_progress + (1 - start_progress) * np.minimum(1.0, np.asarray(steps) / max(1, first_success))
    return float(np.abs(np.asarray(progress) - true).mean())


def metrics(rows):
    """rows: dicts with success, kind, final and (successes) mae."""
    s = [r["final"] for r in rows if r["success"]]
    f = [r["final"] for r in rows if not r["success"]]
    rnd = [r["final"] for r in rows if r["kind"] == "random"]
    return dict(auroc=auroc(s, f), gap=min(s) - max(f) if s and f else None,
                success_final=float(np.mean(s)) if s else None,
                tracking_mae=float(np.mean([r["mae"] for r in rows if r["success"]])) if s else None,
                random_max=max(rnd) if rnd else None, successes=len(s), failures=len(f))


def aggregate(values, lower_is_better=False):
    """Median and worst over references."""
    values = [v for v in values if v is not None]
    if not values:
        return dict(median=None, worst=None)
    return dict(median=float(np.median(values)), worst=(max if lower_is_better else min)(values))
