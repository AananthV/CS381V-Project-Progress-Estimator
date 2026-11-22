"""Subtask labels from gripper events (no simulator state).

A clean 2-cube AddSweetener demo has gripper events grasp, release, grasp, release, which split it
into 5 subtasks: K 1 reach cube 1, 2 carry it to the blender, 3 reach cube 2, 4 carry it,
5 retreat. Within a subtask, progress P in [0, 1) assumes uniform speed: P = (step - start) / length.
A clip is labeled by its middle image.
"""

import json

import numpy as np

GRIPPER_COLUMN = 11  # gripper_close in the LeRobot action vector
PATTERN = ["grasp", "release", "grasp", "release"]


def gripper_events(data, episode):
    import pyarrow.parquet as pq

    table = pq.read_table(data / f"data/chunk-000/episode_{episode:06d}.parquet", columns=["action"])
    closed = np.asarray(table["action"].to_pylist())[:, GRIPPER_COLUMN] > 0
    change = np.flatnonzero(np.diff(closed.astype(int))) + 1
    return [(int(t), "grasp" if closed[t] else "release") for t in change]


def demo_rows(data):
    rows = []
    for line in (data / "meta/episodes.jsonl").read_text().splitlines():
        e = json.loads(line)
        meta = json.loads((data / f"extras/episode_{e['episode_index']:06d}/ep_meta.json").read_text())
        rows.append(dict(episode=e["episode_index"], length=e["length"], cubes=meta["refs"]["num_sugar_cubes"],
                         layout=meta["layout_id"], style=meta["style_id"]))
    return rows


def clean_two_cube(task, split_name="reward_train"):
    """Demos of `split_name` with 2 cubes and exactly grasp/release/grasp/release."""
    data, keep = task.data_dir, set(task.load_split()[split_name])
    out = []
    for r in demo_rows(data):
        if r["episode"] not in keep or r["cubes"] != 2:
            continue
        events = gripper_events(data, r["episode"])
        if [k for _, k in events] == PATTERN:
            out.append(dict(r, events=[t for t, _ in events]))
    return out


def subtask_labels(steps, events, length):
    """steps: env step of each image -> (K [1..5], P [0, 1)) arrays."""
    bounds = np.array([0] + list(events) + [length])
    steps = np.asarray(steps)
    k = np.clip(np.searchsorted(bounds, steps, side="right"), 1, len(bounds) - 1)
    p = (steps - bounds[k - 1]) / (bounds[k] - bounds[k - 1])
    return k.astype(np.int64), p.astype(np.float32)
