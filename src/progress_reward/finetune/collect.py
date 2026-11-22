"""Render the fine-tuning data: every clean 2-cube `reward_train` demo in its own kitchen, plus random-action
rollouts there (negatives). Images are exactly the RARM reward's: agentview-left, 384 px, every 4 env
steps, rendered from simulator states (lossless; video compression changes the comparator's scores).

    outputs/finetune/<task>/data/demo_<ep>.npz
        images [N, 384, 384, 3] uint8, steps [N], subtask [N], progress [N], events [4], length
    outputs/finetune/<task>/data/random_<ep>.npz
        images [R, n, 384, 384, 3] uint8, start [R]   (R rollouts from random demo states)
    outputs/finetune/<task>/data/index.json   demos, layouts, styles, validation demos

    pr-finetune-collect --task add_sweetener
"""

import argparse
import gc
import json

import numpy as np

from ..data.labels import clean_two_cube, subtask_labels
from ..data.task import load_task
from ..paths import FINETUNE as OUTPUT
from ..sim.env import ACTION_DIM, SceneEnv, to_native

CAMERA, SIZE, INTERVAL = "robot0_agentview_left", 384, 4
ENV_SEED = 10_000 * 100
RANDOM_ROLLOUTS, RANDOM_STEPS = 2, 64
N_VALIDATION = 6  # reward_train demos kept out of training for early stopping (distinct layouts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="add_sweetener")
    args = parser.parse_args()
    task = load_task(args.task)
    out = OUTPUT / task.name / "data"
    out.mkdir(parents=True, exist_ok=True)
    demos = clean_two_cube(task)
    assert task.reference_episode not in [d["episode"] for d in demos]

    index_path = out / "index.json"
    if index_path.exists():  # fixed once; never redrawn
        validation = json.loads(index_path.read_text())["validation"]
    else:
        rng = np.random.default_rng(0)
        by_layout = {}
        for d in demos:
            by_layout.setdefault(d["layout"], []).append(d["episode"])
        layouts = sorted(by_layout)
        picked = rng.choice(layouts, N_VALIDATION, replace=False)
        validation = sorted(int(rng.choice(by_layout[l])) for l in picked)
    index_path.write_text(json.dumps(dict(
        task=task.name, camera=CAMERA, size=SIZE, interval=INTERVAL, random_rollouts=RANDOM_ROLLOUTS,
        random_steps=RANDOM_STEPS, validation=validation,
        demos=[{k: d[k] for k in ("episode", "length", "layout", "style", "events")} for d in demos]), indent=1))

    for d in demos:
        ep = d["episode"]
        if (out / f"demo_{ep}.npz").exists() and (out / f"random_{ep}.npz").exists():
            continue
        env = SceneEnv(task, seed=ENV_SEED, episode=ep)
        states = env.demo.states
        steps = np.arange(0, len(states), INTERVAL)
        images = []
        for s in steps:
            env.set_state(states[s])
            images.append(env.render(CAMERA, SIZE))
        k, p = subtask_labels(steps, d["events"], len(states))
        np.savez_compressed(out / f"demo_{ep}.npz", images=np.stack(images), steps=steps, subtask=k, progress=p,
                            events=np.array(d["events"]), length=len(states))
        rng = np.random.default_rng(ep)
        starts = rng.integers(0, len(states) - 1, RANDOM_ROLLOUTS)
        rollouts = []
        for t in starts:
            env.reset(int(t))
            frames = [env.render(CAMERA, SIZE)]
            for i in range(1, RANDOM_STEPS + 1):
                env.env.step(to_native(rng.uniform(-1, 1, ACTION_DIM)))
                if i % INTERVAL == 0:
                    frames.append(env.render(CAMERA, SIZE))
            rollouts.append(np.stack(frames))
        np.savez_compressed(out / f"random_{ep}.npz", images=np.stack(rollouts), start=starts)
        env.close()
        del env
        gc.collect()
        print(f"demo {ep}: {len(images)} images, subtask lengths {np.diff([0] + d['events'] + [len(states)])}, "
              f"random starts {starts.tolist()}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
