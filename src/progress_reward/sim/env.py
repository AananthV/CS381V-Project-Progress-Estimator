"""Simulation: a RoboCasa task rebuilt in the reference demo's own scene, and a worker process.

The scene, objects and states are the reference demo's (its saved model XML and simulator states).
- Agent action: 7-d arm action in [-1, 1] (end-effector position delta 3, rotation delta 3,
  gripper 1), RoboCasa PandaOmron's arm part; base and torso stay still, control mode is "arm".
- Observation: the task's policy cameras at `policy_image_size`, rendered every env step.
- Reward images: rendered every `image_interval` steps (the reset counts as step 0) when a reward
  asks for them (camera, size, interval).
- Resets (as RARM's released LIBERO wrapper): restore demo state t; for t = 0 optionally add
  N(0, init_noise_std) to the arm joints and free objects' positions (noise seed per reset).

Caveat for new tasks: restoring a state restores physics only. Fixture Python state that is set
while stepping (e.g. a coffee machine's `_turned_on`) is not restored; check such tasks with
`check_task.py` before trusting mid-demo starts.
"""

import gzip
import json
import os
from copy import deepcopy
from functools import wraps

import numpy as np

ACTION_DIM = 7


def normalize_windows_object_paths():
    """Upstream mixes slash styles when looking up a recorded object on Windows."""
    if os.name != "nt":
        return
    import robocasa.models.objects.kitchen_object_utils as objects

    original = objects.sample_kitchen_object_helper
    if getattr(original, "_windows_paths_normalized", False):
        return
    for category in objects.OBJ_CATEGORIES.values():
        for entry in category.values():
            if hasattr(entry, "mjcf_paths"):
                entry.mjcf_paths = [os.path.normpath(path) for path in entry.mjcf_paths]

    @wraps(original)
    def sample(groups, *args, **kwargs):
        if isinstance(groups, str) and groups.endswith(".xml"):
            groups = os.path.normpath(groups)
        return original(groups, *args, **kwargs)

    sample._windows_paths_normalized = True
    objects.sample_kitchen_object_helper = sample


def sync_fixture_poses(env, xml):
    """`env.reset()` places fixtures anew (some at random) and fixture objects read their pose from
    their own XML element, which `reset_from_xml_string` does not update. Copy every fixture body's
    pos/quat/euler from the demo XML so success checks look at the real fixtures."""
    import xml.etree.ElementTree as ET

    bodies = {b.get("name"): b for b in ET.fromstring(xml).iter("body")}
    synced = 0
    for fixture in env.fixtures.values():
        element = getattr(fixture, "_obj", None)
        source = bodies.get(element.get("name")) if element is not None else None
        if source is None:
            continue
        for key in ("pos", "quat", "euler"):
            if source.get(key) is not None:
                element.set(key, source.get(key))
            elif key in element.attrib:
                del element.attrib[key]
        synced += 1
    return synced


class Demo:
    """One demo of a task's LeRobot dataset: states, native 12-d actions, scene XML, metadata."""

    ORDER = ("end_effector_position", "end_effector_rotation", "gripper_close", "base_motion", "control_mode")

    def __init__(self, data_dir, episode):
        import pyarrow.parquet as pq

        extras = data_dir / f"extras/episode_{episode:06d}"
        with np.load(extras / "states.npz") as archive:
            self.states = archive["states"].copy()
        table = pq.read_table(data_dir / f"data/chunk-000/episode_{episode:06d}.parquet", columns=["action"])
        stored = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        modality = json.loads((data_dir / "meta/modality.json").read_text())["action"]
        # LeRobot storage order differs from the PandaOmron controller order
        self.actions = np.concatenate([stored[:, modality[k]["start"]:modality[k]["end"]] for k in self.ORDER], 1)
        with gzip.open(extras / "model.xml.gz", "rt") as source:
            self.xml = source.read()
        self.metadata = json.loads((extras / "ep_meta.json").read_text())
        self.env_args = json.loads((data_dir / "extras/dataset_meta.json").read_text())["env_args"]
        self.episode = episode
        if len(self.states) != len(self.actions):
            raise ValueError("state/action lengths differ")

    def agent_actions(self):
        return self.actions[:, :ACTION_DIM].copy()


def to_native(action):
    native = np.zeros(12)
    native[:ACTION_DIM] = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
    native[11] = -1.0  # arm control mode
    return native


class SceneEnv:
    def __init__(self, task, seed=0, image_request=None, episode=None):
        import robosuite

        import robocasa  # noqa: F401  (registers the kitchen environments)

        normalize_windows_object_paths()
        demo = Demo(task.data_dir, task.reference_episode if episode is None else episode)
        kwargs = deepcopy(demo.env_args["env_kwargs"])
        kwargs.update(
            env_name=demo.env_args["env_name"], has_renderer=False, has_offscreen_renderer=True,
            use_camera_obs=False, camera_names=list(task.policy_cameras), camera_heights=task.policy_image_size,
            camera_widths=task.policy_image_size, ignore_done=True, horizon=100000, seed=seed, layout_ids=None,
            style_ids=None, layout_and_style_ids=[(demo.metadata["layout_id"], demo.metadata["style_id"])],
        )
        self.task, self.demo = task, demo
        self.noise_std = task.starts["init_noise_std"]
        self.image_request = image_request  # dict(camera, size, interval) or None
        self.env = robosuite.make(**kwargs)
        self.env.set_ep_meta(demo.metadata)
        self.env.reset()
        self.env.reset_from_xml_string(demo.xml)
        self.env.sim.reset()
        self.synced_fixtures = sync_fixture_poses(self.env, demo.xml)
        self.steps = 0
        model = self.env.sim.model
        self._object_qpos = np.array([i for j in range(model.njnt) if model.jnt_type[j] == 0
                                      and not (model.joint_id2name(j) or "").startswith(("robot", "mobilebase"))
                                      for i in range(model.jnt_qposadr[j], model.jnt_qposadr[j] + 3)], int)
        self._arm_qpos = np.array(self.env.robots[0]._ref_joint_pos_indexes, int)

    def set_state(self, state):
        self.env.sim.set_state_from_flattened(state)
        self.env.sim.forward()
        self.env.update_state()

    def _reset_controllers(self, t):
        """Controllers keep state across episodes: reset goals, replay the gripper ramp up to t."""
        for robot in self.env.robots:
            composite = robot.composite_controller
            composite.update_state()
            for name, controller in composite.part_controllers.items():
                if name != "base":
                    controller.reset_goal()
            for name, gripper in composite.grippers.items():
                start, end = composite._action_split_indexes[name]
                gripper.current_action = np.zeros(gripper.dof)
                for action in self.demo.actions[:t]:
                    gripper.format_action(np.asarray(action[start:end], np.float64))

    def reset(self, t=0, noise_seed=None):
        """Start at demo state t; with a noise seed, add the init noise (step-0 starts)."""
        self.set_state(self.demo.states[t])
        if noise_seed is not None:
            rng = np.random.default_rng(noise_seed)
            qpos = self.env.sim.data.qpos
            qpos[self._arm_qpos] += rng.normal(0, self.noise_std, len(self._arm_qpos))
            qpos[self._object_qpos] += rng.normal(0, self.noise_std, len(self._object_qpos))
            self.env.sim.forward()
            self.env.update_state()
        self._reset_controllers(t)
        self.steps = 0
        return self.observe()

    def success(self):
        return bool(self.env._check_success())

    def render(self, camera, size):
        return self.env.sim.render(width=size, height=size, camera_name=camera)[::-1].copy()

    def observe(self):
        size = self.task.policy_image_size
        policy = np.concatenate([self.render(c, size) for c in self.task.policy_cameras], axis=2)
        out = dict(policy=np.ascontiguousarray(policy.transpose(2, 0, 1)), success=self.success())
        req = self.image_request
        if req and self.steps % req["interval"] == 0:
            out["image"] = self.render(req["camera"], req["size"])
        return out

    def step(self, action):
        self.env.step(to_native(action))
        self.steps += 1
        return self.observe()

    def render_demo(self, camera, size, interval):
        """Reference images: the demo's states every `interval` steps, rendered from `camera`."""
        images = []
        for state in self.demo.states[::interval]:
            self.set_state(state)
            images.append(self.render(camera, size))
        return np.stack(images)

    def close(self):
        self.env.close()


def worker(conn, task_name, seed, image_request):
    """Process loop. Commands: ('reset', t, noise_seed) | ('step', action) | ('success_at', t) |
    ('render_demo', camera, size, interval) | ('close',)."""
    from ..data.task import load_task

    env = SceneEnv(load_task(task_name), seed=seed, image_request=image_request)
    conn.send(("ready", env.synced_fixtures))
    while True:
        command, *args = conn.recv()
        if command == "reset":
            conn.send(("obs", env.reset(*args)))
        elif command == "step":
            conn.send(("obs", env.step(args[0])))
        elif command == "success_at":
            env.set_state(env.demo.states[args[0]])
            conn.send(("success", env.success()))
        elif command == "render_demo":
            conn.send(("images", env.render_demo(*args)))
        elif command == "close":
            env.close()
            conn.send(("closed", None))
            return


def start_workers(task_name, n, seed, image_request=None):
    """Build `n` worker processes one at a time (parallel builds can exhaust RAM)."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    conns = []
    for i in range(n):
        parent, child = ctx.Pipe()
        ctx.Process(target=worker, args=(child, task_name, seed * 100 + i, image_request), daemon=True).start()
        kind, _ = parent.recv()
        assert kind == "ready"
        conns.append(parent)
    return conns


def close_workers(conns):
    for c in conns:
        try:
            c.send(("close",))
        except (BrokenPipeError, OSError):
            pass
