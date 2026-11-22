"""Offline reward evaluation report: every reward setting of configs/eval/<task>.yaml on every
(reference, episode) pair, from the scores of pr-score. Each setting is a reward config
(configs/rewards/<name>.yaml): its comparator picks the score matrix, its matcher (progress_reward.matching,
the same code the RL reward runs) turns it into progress. Metrics: progress_reward/evaluation/metrics.py.

    outputs/eval/<task>/results.json
    outputs/eval/<task>/viewer/index.html   (serve over HTTP: pr-view --task add_sweetener)

    pr-report --task add_sweetener
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from ..comparator.rarm import CLIP, MID
from ..data.task import load_task
from ..matching import make_matcher, summary, track
from ..paths import EVAL
from ..rewards import reward_config
from ..rewards.rarm import INTERVAL
from ..sim.env import Demo
from .metrics import aggregate, metrics, tracking_mae
from .record import load_eval_config
from .score import COMPARATORS, episodes, references

VIEWER = Path(__file__).resolve().parents[1] / "viewer"
VIEW_SIZE = 256
GROUPS = ("same", "test", "dev")


def write_frames(video, folder):
    import av
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "0000.jpg").exists():
        return
    with av.open(str(video)) as v:
        for i, f in enumerate(v.decode(v.streams.video[0])):
            f.to_image().resize((VIEW_SIZE, VIEW_SIZE), Image.BILINEAR).save(folder / f"{i:04d}.jpg", quality=82)


def describe(name, config):
    matcher = config.get("matcher", "rule")
    emission = config.get("hmm", {}).get("emission", "absolute") if matcher == "hmm" else None
    label = {"rule": "RARM rule", "hmm": f"HMM {emission}"}[matcher]
    comparator = config.get("comparator", "original")
    return dict(key=name, comparator=comparator, matcher=matcher, emission=emission,
                label=f"{label} · {'fine-tuned' if comparator == 'finetuned' else 'original'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="add_sweetener")
    args = parser.parse_args()
    task = load_task(args.task)
    cfg = load_eval_config(task.name)
    base, view = EVAL / task.name, EVAL / task.name / "viewer"
    refs, eps = references(task, cfg), episodes(task)
    configs = {name: reward_config(name, task.name) for name in cfg["settings"]}
    settings = [describe(name, c) for name, c in configs.items()]
    excluded = set(cfg.get("excluded_from_metrics", []))
    n_states = len(Demo(task.data_dir, task.reference_episode).states)

    ep_meta = {}
    for name, folder in eps.items():
        meta = json.loads((folder / "meta.json").read_text())
        start = meta.get("start") or 0
        ep_meta[name] = dict(name=name, kind=meta["kind"], run=(meta.get("rl_run") or {}).get("reward"),
                             checkpoint=meta.get("checkpoint"), start=start,
                             start_progress=(start + 1) / n_states if start > 0 else 0.0,
                             success=meta["success"], first_success=meta["first_success"], length=meta["frames"],
                             events=meta["gripper_events"])
        write_frames(folder / "agentview_left.mp4", view / "frames" / name)

    finals, scored, per_ref = {}, {}, {}
    for r, group in refs.items():
        sdir = base / "scores" / f"ref_{r}"
        R = np.load(sdir / "ref.npz")
        progress = R["progress"]
        matchers = {name: make_matcher(c, progress, R[f"self_best_{c.get('comparator', 'original')}"])
                    for name, c in configs.items()}
        view_eps = {}
        for name, em in ep_meta.items():
            z = np.load(sdir / f"{name}.npz")
            steps = z["query_end_image"] * INTERVAL
            methods = {}
            for key, c in configs.items():
                rows = track(matchers[key], z[f"S_{c.get('comparator', 'original')}"], progress, em["start_progress"])
                p = [x["progress"] for x in rows]
                mae = tracking_mae(p, steps, em["start_progress"], em["first_success"]) if em["success"] else None
                s = summary(rows)
                finals[(r, name, key)] = dict(final=s["final"], max_jump=s["max_jump"], mae=mae)
                methods[key] = dict(pos=[x["position"] for x in rows], ok=[int(x["matched"]) for x in rows],
                                    progress=[round(x, 4) for x in p])
                if c.get("matcher") == "hmm":
                    methods[key]["tracked"] = [round(x["tracked"], 4) for x in rows]
            view_eps[name] = dict(t=[int(x) for x in steps], methods=methods,
                                  scores={cm: np.round(z[f"S_{cm}"] * 100).astype(int).tolist() for cm in COMPARATORS})
        per_ref[r] = dict(ref=r, group=group, progress=progress.round(4).tolist(),
                          self_best={cm: np.round(R[f"self_best_{cm}"] * 100).astype(int).tolist() for cm in COMPARATORS},
                          episodes=view_eps)
        for key in configs:
            rows = [dict(success=em["success"], kind=em["kind"], **finals[(r, name, key)])
                    for name, em in ep_meta.items() if name not in excluded]
            scored[(r, key)] = metrics(rows)
        print(f"ref {r:3d} ({group}): " + "  ".join(
            f"{k} auroc {scored[(r, k)]['auroc']:.2f} gap {scored[(r, k)]['gap']:+.2f}" for k in cfg["headline"]),
            flush=True)

    groups = {}
    for g in GROUPS:
        rs = [r for r, gg in refs.items() if gg == g]
        groups[g] = {key: {m: aggregate([scored[(r, key)][m] for r in rs], m in ("tracking_mae", "random_max"))
                           for m in ("auroc", "gap", "success_final", "tracking_mae", "random_max")}
                     for key in configs}

    results = dict(task=task.name, references={str(r): g for r, g in refs.items()}, settings=cfg["settings"],
                   headline=cfg["headline"], reward_configs=configs, excluded=sorted(excluded),
                   episodes=[{k: em[k] for k in ("name", "kind", "run", "checkpoint", "start", "success")}
                             for em in ep_meta.values()],
                   per_reference={str(r): {k: scored[(r, k)] for k in configs} for r in refs}, groups=groups,
                   finals={f"{r}|{name}|{key}": v for (r, name, key), v in finals.items()})
    (base / "results.json").write_text(json.dumps(results, indent=1))

    view.mkdir(parents=True, exist_ok=True)
    if (view / "references").exists():
        shutil.rmtree(view / "references")
    shutil.copytree(base / "references", view / "references")
    (view / "refs").mkdir(exist_ok=True)
    for r, d in per_ref.items():
        (view / "refs" / f"ref_{r}.js").write_text(
            f"window.REF_DATA = window.REF_DATA || {{}}; window.REF_DATA[{r}] = " + json.dumps(d) + ";\n")
    data = dict(task=task.name, fps=20, clip=CLIP, mid=MID, interval=INTERVAL, ref_same=task.reference_episode,
                settings=settings, headline=cfg["headline"], reward_configs=configs,
                references=[dict(ref=r, group=g) for r, g in refs.items()], episodes=list(ep_meta.values()),
                excluded=sorted(excluded), per_reference=results["per_reference"], groups=groups,
                finals={f"{r}|{name}|{key}": v["final"] for (r, name, key), v in finals.items()})
    (view / "data.js").write_text("window.DATA = " + json.dumps(data) + ";\n")
    shutil.copy(VIEWER / "progress.html", view / "index.html")
    print("wrote", view / "index.html")


if __name__ == "__main__":
    main()
