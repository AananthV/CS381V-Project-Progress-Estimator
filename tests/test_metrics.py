import numpy as np

from progress_reward.data.labels import subtask_labels
from progress_reward.evaluation.metrics import aggregate, auroc, metrics, tracking_mae


def test_auroc_and_gap():
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auroc([0.5], [0.5]) == 0.5
    rows = [dict(success=True, kind="rl", final=0.9, mae=0.1), dict(success=False, kind="random", final=0.3, mae=None),
            dict(success=False, kind="rl", final=0.6, mae=None)]
    m = metrics(rows)
    assert m["auroc"] == 1.0 and abs(m["gap"] - 0.3) < 1e-9 and m["random_max"] == 0.3


def test_tracking_mae_is_zero_on_the_linear_target():
    steps = np.arange(0, 400, 4)
    true = np.minimum(1.0, steps / 300)
    assert tracking_mae(true, steps, 0.0, 300) < 1e-9


def test_aggregate():
    assert aggregate([0.2, 0.9, 0.5]) == dict(median=0.5, worst=0.2)
    assert aggregate([0.2, 0.9, 0.5], lower_is_better=True)["worst"] == 0.9


def test_subtask_labels_uniform_progress():
    K, P = subtask_labels(np.array([0, 50, 99, 100, 150]), [100, 200, 300, 400], 500)
    assert list(K[:3]) == [1, 1, 1] and K[3] == 2
    assert P[0] == 0.0 and abs(P[1] - 0.5) < 1e-9
