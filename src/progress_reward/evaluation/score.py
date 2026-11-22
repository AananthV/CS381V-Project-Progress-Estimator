"""Score every RL-kitchen episode against every reference demo with both comparators (original and
fine-tuned). Images as the RARM reward sees them (agentview-left, 384 px, every 4 steps), rendered from
saved simulator states; each reference demo is rendered in its own kitchen.

    outputs/eval/<task>/images/{ref_<r>,ep_<name>}.npy   rendered frames (cache, ~4 GB, deletable)
    outputs/eval/<task>/scores/ref_<r>/ref.npz          progress [M], self_best_<cmp> [M], group
    outputs/eval/<task>/scores/ref_<r>/<episode>.npz    S_<cmp> [Q, M], query_end_image [Q]
    outputs/eval/<task>/references/ref_<r>/*.jpg        reference images (viewer)

    pr-score --task add_sweetener
"""

import argparse
import gc
import json

import numpy as np
import torch

from ..comparator.heads import HeadSwitcher, heads_file
from ..comparator.rarm import CLIP, MID, RarmModel, download
from ..data.task import load_task
from ..paths import EVAL
from ..rewards import reward_config
from ..rewards.rarm import CAMERA, INTERVAL, SIZE
from ..sim.env import SceneEnv
from .record import ENV_SEED, load_eval_config

COMPARATORS = ("original", "finetuned")


def references(task, config):
    split = task.load_split()["reward_heldout"]
    refs = {}
    for group, eps in config["references"].items():
        for r in eps:
            assert group == "same" or r in split, f"reference {r} is not a reward_heldout demo"
            refs[r] = group
    return refs


def episodes(task):
    """name -> folder of every recorded episode in the reference scene."""
    found = {}
    for meta in sorted((EVAL / task.name / "episodes").glob("*/meta.json")):
        if json.loads(meta.read_text())["scene"] == task.reference_episode:
            found[meta.parent.name] = meta.parent
    return found


def render(env, states):
    images = []
    for s in states[::INTERVAL]:
        env.set_state(s)
        images.append(env.render(CAMERA, SIZE))
    return np.stack(images)


def render_all(task, refs, eps, cache):
    """Render whatever is not cached yet, one scene at a time (each scene build holds ~2 GB RAM)."""
    cache.mkdir(parents=True, exist_ok=True)
    jobs = {}
    for r in refs:
        if not (cache / f"ref_{r}.npy").exists():
            jobs.setdefault(r, []).append(("ref", r))
    for name in eps:
        if not (cache / f"ep_{name}.npy").exists():
            jobs.setdefault(task.reference_episode, []).append(("ep", name))
    for scene, items in jobs.items():
        env = SceneEnv(task, seed=ENV_SEED, episode=scene)
        for kind, key in items:
            states = env.demo.states if kind == "ref" else np.load(eps[key] / "states.npz")["states"]
            np.save(cache / f"{kind}_{key}.npy", render(env, states))
        env.close()
        del env
        gc.collect()
        print(f"rendered scene {scene}: {[k for _, k in items]}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="add_sweetener")
    args = parser.parse_args()
    task = load_task(args.task)
    torch.backends.cuda.matmul.allow_tf32 = True
    base = EVAL / task.name
    refs, eps = references(task, load_eval_config(task.name)), episodes(task)
    print(f"{len(eps)} episodes, references {refs}", flush=True)
    render_all(task, refs, eps, base / "images")

    from PIL import Image

    for r in refs:
        folder = base / "references" / f"ref_{r}"
        if not (folder / "0000.jpg").exists():
            folder.mkdir(parents=True, exist_ok=True)
            for k, im in enumerate(np.load(base / "images" / f"ref_{r}.npy")):
                Image.fromarray(im).resize((256, 256), Image.BILINEAR).save(folder / f"{k:04d}.jpg", quality=82)

    download()
    model = RarmModel()
    heads = HeadSwitcher(model, heads_file(reward_config("rarm_ft", task.name)))

    with torch.inference_mode():
        ref_clips = {}
        for r, group in refs.items():
            imgs = np.load(base / "images" / f"ref_{r}.npy")
            tok = model.frames(imgs)
            starts = list(range(len(imgs) - CLIP + 1))
            info = dict(progress=np.array([(s + MID) / len(imgs) for s in starts], np.float32), images=len(imgs),
                        group=group)
            for cmp in COMPARATORS:
                heads.use(cmp)
                clips = model.clips(tok, starts)
                ref_clips[(r, cmp)] = clips
                info[f"self_best_{cmp}"] = np.array(
                    [float(model.scores(clips[i], clips).max()) for i in range(len(starts))], np.float32)
            out = base / "scores" / f"ref_{r}"
            out.mkdir(parents=True, exist_ok=True)
            np.savez(out / "ref.npz", **info)
            del tok
        torch.cuda.empty_cache()
        print("reference clips ready", flush=True)

        for name in eps:
            tok = model.frames(np.load(base / "images" / f"ep_{name}.npy"))
            ends = np.arange(CLIP - 1, len(tok))
            result = {r: dict(query_end_image=ends) for r in refs}
            for cmp in COMPARATORS:
                heads.use(cmp)
                clips = model.clips(tok, list(ends - (CLIP - 1)))
                for r in refs:
                    result[r][f"S_{cmp}"] = np.stack(
                        [model.scores(c, ref_clips[(r, cmp)]).float().cpu().numpy() for c in clips])
                del clips
            for r in refs:
                np.savez(base / "scores" / f"ref_{r}" / f"{name}.npz", **result[r])
            del tok
            torch.cuda.empty_cache()
            print(f"scored {name}", flush=True)


if __name__ == "__main__":
    main()
