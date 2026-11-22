"""Gather every run's step-0 evaluation into the dashboard.

    pr-dashboard

Reads outputs/rl/<task>/<reward>/seed_<k>/eval/results.json and writes
outputs/rl/dashboard/{index.html, data.js}. Open index.html in a browser (no server needed).
Per task, reward and checkpoint: mean success over seeds, a 95% bootstrap interval of that mean,
min-max over seeds, and every seed's own value.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from ..paths import RL as OUTPUT

VIEWER = Path(__file__).resolve().parents[1] / "viewer"


def bootstrap_ci(values, rng, n=10_000):
    values = np.asarray(values, float)
    if len(values) < 2:
        return [float(values.mean())] * 2
    means = rng.choice(values, (n, len(values))).mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    rng = np.random.default_rng(0)
    tasks = {}
    for path in sorted(OUTPUT.glob("*/*/seed_*/eval/results.json")):
        seed_dir = path.parent.parent
        task, reward, seed = seed_dir.parent.parent.name, seed_dir.parent.name, int(seed_dir.name.split("_")[1])
        results = json.loads(path.read_text())
        curve = {int(v["step"]): v["success_rate"] for v in results.values()}
        notes = (seed_dir / "PROVENANCE.md").read_text() if (seed_dir / "PROVENANCE.md").exists() else ""
        tasks.setdefault(task, {}).setdefault(reward, {})[seed] = dict(curve=curve, notes=notes,
                                                                     trials=len(next(iter(results.values()))["trials"]))
    data = dict(generated=datetime.now().isoformat(timespec="minutes"), tasks={})
    for task, rewards in tasks.items():
        entry = {}
        for reward, seeds in rewards.items():
            steps = sorted({s for run in seeds.values() for s in run["curve"]})
            points = []
            for step in steps:
                vals = {seed: run["curve"][step] for seed, run in seeds.items() if step in run["curve"]}
                v = list(vals.values())
                points.append(dict(step=step, n=len(v), mean=float(np.mean(v)), ci=bootstrap_ci(v, rng),
                                   min=float(min(v)), max=float(max(v)), seeds={str(k): x for k, x in vals.items()}))
            per_seed = {str(seed): dict(final=run["curve"][max(run["curve"])], best=max(run["curve"].values()),
                                        auc=float(np.mean(list(run["curve"].values()))), last_step=max(run["curve"]),
                                        notes=run["notes"], trials=run["trials"])
                        for seed, run in seeds.items()}
            entry[reward] = dict(points=points, seeds=per_seed)
        data["tasks"][task] = entry
    target = OUTPUT / "dashboard"
    target.mkdir(parents=True, exist_ok=True)
    (target / "data.js").write_text("window.BENCHMARK = " + json.dumps(data, indent=1) + ";\n")
    shutil.copy(VIEWER / "dashboard.html", target / "index.html")
    runs = sum(len(s) for r in tasks.values() for s in r.values())
    print(f"{runs} runs -> {target / 'index.html'}")


if __name__ == "__main__":
    main()
