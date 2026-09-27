"""MuJoCo quadrotor tasks.

    race2 (the study)  laps around a fixed 3-D course with a steep chute; altitude-hold stabiliser
    race3              race2 with a deadlier chute (up to 50% lift lost instead of 35%)
    race               the same course, raw rotor commands
    delivery, delivery2  land a package of random weight on a pad
    "" (cage)          land inside an open-top cage

Hidden process noise (OU wind gusts, per-rotor thrust noise) acts only on the real dynamics.
A fast, powered, upright descent loses lift (a simplified vortex-ring state).
Reward and termination are pure functions of (obs, action), so real and imagined transitions
are scored by the same code (known_reward_fn / termination_fn).

Observation: pad-relative position (3), height, quaternion (4), linear and angular velocity (6),
then per task: obstacle-relative xy (2 each), payload / drone mass, steady wind xy,
racing-line sensor (vector to the line + direction 0.5 m ahead).
Action (4): rotor commands (thrust = empty-drone hover x (1 + a)), or for ctrl_mode
"attitude" / "althold": [accel x, accel y, vertical accel or climb rate, yaw rate].
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None

import gymnasium as gym
from gymnasium import spaces


_ASSET = os.path.join(os.path.dirname(__file__), "assets", "drone.xml")

_REL = slice(0, 3)
_H = 3
_QW, _QX, _QY, _QZ = 4, 5, 6, 7
_LV = slice(8, 11)
_VZ = 10
_AV = slice(11, 14)
_OBS0 = 14
_WINDSOCK_AT = (0.9, 0.3)


# ── reward / termination terms (vectorised, shared by real and imagined transitions) ──────────────
def _drone_reward(obs, act, rw, n_obs, obs_radius, obs_scale, cage=None) -> np.ndarray:
    """Land-on-pad reward: closeness, upright, settle bonus; penalties for spin, speed, effort,
    obstacle and cage-wall proximity."""
    obs = np.atleast_2d(np.asarray(obs, dtype=np.float64))
    act = np.atleast_2d(np.asarray(act, dtype=np.float64))
    dist = np.linalg.norm(obs[:, _REL], axis=-1)
    up_z = 1.0 - 2.0 * (obs[:, _QX] ** 2 + obs[:, _QY] ** 2)
    spin = np.linalg.norm(obs[:, _AV], axis=-1)
    speed = np.linalg.norm(obs[:, _LV], axis=-1)

    r_pos = float(rw["w_pos"]) * np.exp(-dist / max(float(rw["pos_scale"]), 1e-6))
    r_level = float(rw["w_level"]) * up_z
    r_spin = float(rw["w_spin"]) * spin
    r_vel = float(rw["w_vel"]) * speed
    r_ctrl = float(rw["w_ctrl"]) * np.sum(act ** 2, axis=-1)

    w_settle = float(rw.get("w_settle", 0.0))
    if w_settle > 0.0:
        settle_dist = max(float(rw.get("settle_dist", 0.3)), 1e-6)
        settle_speed = max(float(rw.get("settle_speed", 0.4)), 1e-6)
        r_settle = (w_settle * np.exp(-dist / settle_dist)
                    * np.exp(-speed / settle_speed) * np.clip(up_z, 0.0, 1.0))
    else:
        r_settle = 0.0

    r_obs = 0.0
    if n_obs > 0:
        oxy = obs[:, _OBS0:_OBS0 + 2 * n_obs].reshape(obs.shape[0], n_obs, 2)
        d = np.linalg.norm(oxy, axis=-1)
        surface = np.maximum(d - float(obs_radius), 0.0)
        r_obs = float(rw["w_obs"]) * np.exp(-surface / max(float(obs_scale), 1e-6)).sum(axis=-1)

    r_cage = 0.0
    if cage is not None and float(rw.get("w_cage", 0.0)) > 0.0:
        radius, height, margin = cage
        dxy = np.linalg.norm(obs[:, 0:2], axis=-1)
        gap = np.maximum(np.abs(dxy - radius) - margin, 0.0)
        below = 1.0 / (1.0 + np.exp((obs[:, _H] - height) / 0.08))   # ~1 below the cage top
        r_cage = (float(rw["w_cage"]) * below
                  * np.exp(-gap / max(float(rw.get("cage_scale", 0.15)), 1e-6)))

    return r_pos + r_level + r_settle - r_spin - r_vel - r_ctrl - r_obs - r_cage


def _race_reward(obs, act, rw, course) -> np.ndarray:
    """Speed along the racing direction while near the line, minus metres outside the course tube,
    plus small upright / spin / effort terms."""
    from src.envs.race_course import project
    obs = np.atleast_2d(np.asarray(obs, dtype=np.float64))
    act = np.atleast_2d(np.asarray(act, dtype=np.float64))
    k, d = project(course, -obs[:, _REL])                          # course centre = world origin
    prog = np.sum(obs[:, _LV] * course["tangent"][k], axis=-1)
    off = np.maximum(d - float(rw.get("race_tube", 0.35)), 0.0)
    near = np.exp(-(off / max(float(rw.get("race_tube_soft", 0.35)), 1e-6)) ** 2)
    up_z = 1.0 - 2.0 * (obs[:, _QX] ** 2 + obs[:, _QY] ** 2)
    spin = np.linalg.norm(obs[:, _AV], axis=-1)
    return (float(rw.get("w_prog", 1.0)) * prog * near - float(rw.get("w_track", 1.0)) * off
            + float(rw["w_level"]) * up_z - float(rw["w_spin"]) * spin
            - float(rw["w_ctrl"]) * np.sum(act ** 2, axis=-1))


def _cage_params(cfg: dict):
    if not cfg.get("cage", False):
        return None
    return (float(cfg.get("cage_radius", 0.8)), float(cfg.get("cage_height", 1.0)),
            float(cfg.get("cage_margin", 0.15)))


def _cage_hit(obs, cage) -> np.ndarray:
    obs = np.atleast_2d(np.asarray(obs))
    radius, height, margin = cage
    dxy = np.linalg.norm(obs[:, 0:2], axis=-1)
    return (np.abs(dxy - radius) < margin) & (obs[:, _H] < height)


# ── lift loss in a fast descent ───────────────────────────────────────────────────────────────────
def _smooth01(x) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _lift_factor(descent, v_horiz, loss, v_on, v_full, v_escape) -> float:
    """Unpowered variant: lift falls from 1 at `v_on` to 1 - loss at `v_full` m/s of sink;
    sideways speed `v_escape` clears it."""
    x = float(np.clip((descent - v_on) / max(v_full - v_on, 1e-6), 0.0, 1.0))
    s = x * x * (3.0 - 2.0 * x)
    return 1.0 - loss * s * float(np.exp(-(v_horiz / max(v_escape, 1e-6)) ** 2))


def _powered_lift_factor(vel, quat, thrust_ratio, loss, v_on, v_full, v_escape, up_on, thrust_on) -> float:
    """Powered variant: needs an upright drone, a fast sink along the rotor axis and rotors pushing
    at least `thrust_on` x hover; flying across the rotor axis clears it."""
    w, x, y, z = (float(q) for q in quat)
    zb = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    v = np.asarray(vel, dtype=float)
    axial = float(v @ zb)
    across = float(np.linalg.norm(v - axial * zb))
    s_desc = _smooth01((-axial - v_on) / max(v_full - v_on, 1e-6))
    s_up = _smooth01((zb[2] - up_on) / 0.10)
    s_pow = _smooth01((thrust_ratio - thrust_on) / 0.3)
    return 1.0 - loss * s_desc * s_up * s_pow * float(np.exp(-(across / max(v_escape, 1e-6)) ** 2))


def _quat_to_mat(quat) -> np.ndarray:
    w, x, y, z = (float(q) for q in quat)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# ── scene decoration (visual only: no mass, no collisions) ───────────────────────────────────────
def _add_race_course(root, course):
    """Gate rings, racing-line dots (red = chute), a see-through chute column, red pillars."""
    wb = root.find("worldbody")
    for gi, (c, t) in enumerate(course["gates"]):
        t = t / (np.linalg.norm(t) + 1e-9)
        e1 = np.cross(t, [0.0, 0.0, 1.0])
        e1 = e1 / (np.linalg.norm(e1) + 1e-9) if np.linalg.norm(e1) > 1e-6 else np.array([1.0, 0.0, 0.0])
        e2 = np.cross(t, e1)
        ring = [c + 0.45 * (np.cos(a) * e1 + np.sin(a) * e2) for a in np.linspace(0, 2 * np.pi, 17)]
        for i in range(16):
            a, b = ring[i], ring[i + 1]
            ET.SubElement(wb, "geom", {
                "name": f"gate{gi}_{i}", "type": "capsule", "size": "0.025",
                "fromto": f"{a[0]:.4f} {a[1]:.4f} {a[2]:.4f} {b[0]:.4f} {b[1]:.4f} {b[2]:.4f}",
                "contype": "0", "conaffinity": "0", "rgba": "1.0 0.55 0.10 1" if i % 2 else "0.95 0.95 0.95 1"})
    pts, chute = course["points"], course["chute"]
    for i in range(0, len(pts), 5):
        ET.SubElement(wb, "geom", {
            "name": f"course{i}", "type": "sphere", "size": "0.035",
            "pos": f"{pts[i][0]:.4f} {pts[i][1]:.4f} {pts[i][2]:.4f}", "contype": "0", "conaffinity": "0",
            "rgba": "0.95 0.25 0.15 1" if chute[i] else "0.80 0.90 1.0 0.9"})
    top, bot = pts[np.flatnonzero(chute)[0]], pts[np.flatnonzero(chute)[-1]]
    ET.SubElement(wb, "geom", {
        "name": "chute", "type": "cylinder", "size": "0.35",
        "fromto": f"{top[0]:.4f} {top[1]:.4f} {top[2]:.4f} {bot[0]:.4f} {bot[1]:.4f} {bot[2]:.4f}",
        "contype": "0", "conaffinity": "0", "rgba": "0.95 0.20 0.10 0.18"})
    for g in root.iter("geom"):
        if (g.get("name") or "").startswith("obs"):
            g.set("rgba", "0.80 0.30 0.20 1")
    vis = root.find("visual")
    if vis is None:
        vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "headlight", {"ambient": "0.35 0.35 0.38", "diffuse": "0.55 0.55 0.55",
                                     "specular": "0.1 0.1 0.1"})


def _add_windsock(root):
    """Pole + striped sock as mocap bodies; src/viz/scenery.py points the sock along the wind."""
    wb = root.find("worldbody")
    pole = ET.SubElement(wb, "body", {"name": "windsock_pole", "mocap": "true", "pos": "0 0 0"})
    ET.SubElement(pole, "geom", {"name": "windsock_pole_geom", "type": "cylinder", "size": "0.012 0.5",
                                 "pos": "0 0 0.5", "mass": "0", "contype": "0", "conaffinity": "0",
                                 "rgba": "0.75 0.75 0.78 1"})
    sock = ET.SubElement(wb, "body", {"name": "windsock", "mocap": "true", "pos": "0 0 1"})
    for i, (a, b, r, c) in enumerate(((0.0, 0.1, 0.040, "0.95 0.45 0.10 1"),
                                      (0.1, 0.2, 0.034, "0.95 0.95 0.95 1"),
                                      (0.2, 0.3, 0.028, "0.95 0.45 0.10 1"))):
        ET.SubElement(sock, "geom", {"name": f"windsock_{i}", "type": "capsule", "size": f"{r}",
                                     "fromto": f"{a} 0 0 {b} 0 0", "mass": "0", "contype": "0",
                                     "conaffinity": "0", "rgba": c})


def _add_cage(root, radius, height, n_bars):
    wb = root.find("worldbody")
    for b in list(wb.findall("body")):
        if (b.get("name") or "").startswith("obs"):
            wb.remove(b)
    cage = ET.SubElement(wb, "body", {"name": "cage", "mocap": "true", "pos": "0 0 0"})
    pts = [(radius * np.cos(2 * np.pi * i / n_bars), radius * np.sin(2 * np.pi * i / n_bars))
           for i in range(n_bars)]
    for i, (x, y) in enumerate(pts):
        ET.SubElement(cage, "geom", {
            "name": f"cage_bar{i}", "type": "cylinder", "size": f"0.018 {height / 2}",
            "pos": f"{x:.4f} {y:.4f} {height / 2}", "mass": "0", "contype": "0",
            "conaffinity": "0", "rgba": "0.82 0.82 0.88 1"})
        x2, y2 = pts[(i + 1) % n_bars]
        ET.SubElement(cage, "geom", {
            "name": f"cage_hoop{i}", "type": "capsule", "size": "0.015",
            "fromto": f"{x:.4f} {y:.4f} {height} {x2:.4f} {y2:.4f} {height}", "mass": "0",
            "contype": "0", "conaffinity": "0", "rgba": "0.82 0.82 0.88 1"})


class DroneTargetEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: dict, seed: int = 0, render: bool = False):
        super().__init__()
        if mujoco is None:
            raise ImportError("mujoco is required to instantiate DroneTargetEnv")
        self.cfg = cfg
        g = cfg.get
        self.action_repeat = int(g("action_repeat", 2))
        self.max_episode_steps = int(g("max_episode_steps", 250))
        self.init_height = float(g("init_height", 2.0))
        self.init_noise = float(g("init_noise", 0.05))
        self.init_tilt = float(g("init_tilt", 0.0))
        self.init_spin = float(g("init_spin", 0.0))
        self.target_range_xy = float(g("target_range_xy", 1.8))
        self.pad_min_dist = float(g("pad_min_dist", 1.4))
        self.pad_z = float(g("pad_z", 0.2))
        self.thrust_gain = float(g("thrust_gain", 1.0))
        self.max_dist = float(g("max_dist", 8.0))
        self.fail_tilt = float(g("fail_tilt", 0.0))

        # hidden process noise
        self.wind_force = float(g("wind_force", 0.0))
        self.wind_corr = float(g("wind_correlation", 0.95))
        self.act_noise = float(g("actuator_noise", 0.0))
        # visible steady wind, new every episode
        self.wind_mean_max = float(g("wind_mean_max", 0.0))
        self._wind_mean = np.zeros(3)

        # obstacles: a ring of no-fly columns around the pad (race: fixed pillars)
        self.n_obstacles = int(g("n_obstacles", 2))
        self.obs_radius = float(g("obstacle_radius", 0.4))
        self.min_clear = float(g("obstacle_min_clear", 0.7))
        self.ring_radius = float(g("ring_radius", 0.75))
        self.ring_gap_half = np.radians(float(g("ring_gap_half_deg", 55.0)))

        # landing envelope
        self.land_radius = float(g("land_radius", 0.35))
        self.soft_speed = float(g("soft_speed", 0.5))
        self.impact_height = float(g("impact_height", 0.06))
        self.hard_speed = float(g("hard_speed", 1.0))

        # cage / delivery
        self.cage = bool(g("cage", False))
        self._cage = _cage_params(cfg)
        self.cage_radius = float(g("cage_radius", 0.8))
        self.cage_height = float(g("cage_height", 1.0))
        self.cage_bars = int(g("cage_bars", 16))
        self.start_height = (float(g("start_height_min", 2.2)), float(g("start_height_max", 2.6)))
        self.start_offset = float(g("start_offset", 1.0))
        self.start_offset_min = float(g("start_offset_min", 0.0))
        self.spawn_above = bool(g("spawn_above_pad", False)) or self.cage
        self.payload_max = float(g("payload_max", 0.0))
        self.payload = 0.0

        # lift loss in a fast descent
        self.vrs_loss = float(g("vrs_loss", 0.0))
        self.vrs_speed = float(g("vrs_speed", 0.8))
        self.vrs_full = float(g("vrs_full", 1.6))
        self.vrs_escape = float(g("vrs_escape", 1.0))
        self.vrs_powered = bool(g("vrs_powered", False))
        self.vrs_upright = float(g("vrs_upright", 0.85))
        self.vrs_thrust = float(g("vrs_thrust", 0.7))
        self.thrust_eff = 1.0                                     # last step's lift factor
        self.scenery = bool(g("scenery", False))

        # race
        self.race = bool(g("race", False))
        self.course = None
        if self.race:
            from src.envs.race_course import build_course
            self.course = build_course(cfg)
        self.race_max_off = float(g("race_max_off", 1.5))
        self.race_floor = float(g("race_floor", 0.12))
        self.race_pillars = [tuple(float(v) for v in p) for p in g("race_pillars", [])]
        self.race_progress = 0.0
        self._race_k = 0
        self.race_obs_course = bool(g("race_obs_course", False)) and self.race
        self._race_ahead = (int(round(float(g("race_obs_ahead", 0.5)) / self.course["params"]["spacing"]))
                            if self.race_obs_course else 0)

        # action interface: "rotor" | "attitude" | "althold" (on-board stabiliser, see attitude_to_rotors)
        self.ctrl_mode = str(g("ctrl_mode", "rotor")).lower()
        self.att_max_tilt = np.radians(float(g("att_max_tilt_deg", 40.0)))
        self.att_yaw_rate = float(g("att_yaw_rate", 2.0))
        self.alt_vz_max = float(g("alt_vz_max", 3.0))
        self.alt_kz = float(g("alt_kz", 3.0))
        self.rw = cfg["reward"]
        self.obs_scale = float(self.rw.get("obs_scale", g("obstacle_scale", 0.5)))

        self._build_model(g("mjcf_scene") or _ASSET)
        if self.ctrl_mode in ("attitude", "althold"):
            self._init_attitude_controller()
        elif self.ctrl_mode != "rotor":
            raise ValueError(f"unknown ctrl_mode {self.ctrl_mode!r} (use 'rotor', 'attitude' or 'althold')")

        obs_dim = (14 + 2 * self.n_obstacles + (1 if self.payload_max > 0.0 else 0)
                   + (2 if self.wind_mean_max > 0.0 else 0) + (6 if self.race_obs_course else 0))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_act,), dtype=np.float32)

        self.render_enabled = render
        self._renderer = None
        if render:
            try:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            except Exception as e:
                self.render_enabled = False
                print(f"[drone_env] rendering disabled (GL unavailable): {e}")

        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._target = np.zeros(3)
        self._obstacles = np.zeros((self.n_obstacles, 2))
        self._wind = np.zeros(3)

    def _build_model(self, scene):
        spare_markers = self.n_obstacles < 4 and not self.cage
        if self.cage or self.payload_max > 0.0 or spare_markers or self.scenery or self.race:
            root = ET.parse(scene).getroot()
            if self.cage:
                _add_cage(root, self.cage_radius, self.cage_height, self.cage_bars)
            if spare_markers:
                wb = root.find("worldbody")
                for b in list(wb.findall("body")):
                    name = b.get("name") or ""
                    if name.startswith("obs") and int(name[3:]) >= self.n_obstacles:
                        wb.remove(b)
            if self.payload_max > 0.0:                  # drawn at max size; reset() scales it to the weight
                core = next(b for b in root.iter("body") if b.get("name") == "core")
                ET.SubElement(core, "geom", {
                    "name": "package", "type": "box", "size": "0.04 0.04 0.024", "pos": "0 0 -0.054",
                    "mass": "0", "contype": "0", "conaffinity": "0", "rgba": "0.72 0.52 0.30 1"})
            if self.scenery:
                _add_windsock(root)
            if self.race:
                _add_race_course(root, self.course)
            self.mjcf_xml = ET.tostring(root, encoding="unicode")
            self.model = mujoco.MjModel.from_xml_string(self.mjcf_xml)
        else:
            with open(scene) as f:
                self.mjcf_xml = f.read()
            self.model = mujoco.MjModel.from_xml_path(scene)
        self.data = mujoco.MjData(self.model)
        self.n_act = self.model.nu

        self._core_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "core")
        grav = float(-self.model.opt.gravity[2])
        self._hover = float(self.model.body_subtreemass[self._core_id]) * grav / self.n_act   # empty drone
        self._ctrl_hi = float(self.model.actuator_ctrlrange[0][1])
        self._mass0 = float(self.model.body_mass[self._core_id])
        self._pkg = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "package")
        self._mocap = {}
        for name in ["pad", "cage", "windsock_pole", "windsock"] + [f"obs{i}" for i in range(self.n_obstacles)]:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0 and self.model.body_mocapid[bid] >= 0:
                self._mocap[name] = int(self.model.body_mocapid[bid])

    # ── observation ──────────────────────────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        q = self.data.qpos
        v = self.data.qvel
        pos = q[0:3]
        rel = self._target - pos
        base = [rel[0], rel[1], rel[2], q[2],
                q[3], q[4], q[5], q[6],
                v[0], v[1], v[2], v[3], v[4], v[5]]
        for o in self._obstacles:
            base += [o[0] - pos[0], o[1] - pos[1]]
        if self.payload_max > 0.0:
            base.append(self.payload / self._mass0)
        if self.wind_mean_max > 0.0:
            base += [self._wind_mean[0], self._wind_mean[1]]
        if self.race_obs_course:
            from src.envs.race_course import project
            k = int(project(self.course, pos)[0][0])
            c = self.course
            base += list(c["points"][k] - pos) + list(c["tangent"][(k + self._race_ahead) % len(c["points"])])
        return np.array(base, dtype=np.float32)

    def _place_obstacles(self):
        """Race: the fixed pillars. Otherwise columns spread evenly around the pad, leaving one entry
        gap that faces the start."""
        if self.race:
            return np.array(self.race_pillars, dtype=float).reshape(self.n_obstacles, 2)
        pad = self._target[:2]
        gap_dir = -pad
        gap_theta = float(np.arctan2(gap_dir[1], gap_dir[0]))
        if self.n_obstacles <= 0:
            return np.zeros((0, 2))
        arc_lo = gap_theta + self.ring_gap_half
        arc_hi = gap_theta + 2.0 * np.pi - self.ring_gap_half
        thetas = (np.linspace(arc_lo, arc_hi, self.n_obstacles)
                  if self.n_obstacles > 1 else np.array([gap_theta + np.pi]))
        obs = [pad + self.ring_radius * np.array([np.cos(t), np.sin(t)]) for t in thetas]
        return np.array(obs).reshape(self.n_obstacles, 2)

    # ── gym API ──────────────────────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.payload_max > 0.0:
            self.payload = (float(options["payload"]) if options and "payload" in options
                            else float(self._rng.uniform(0.0, self.payload_max)))
            self.model.body_mass[self._core_id] = self._mass0 + self.payload
            mujoco.mj_setConst(self.model, self.data)
            if self._pkg >= 0:
                s = 0.015 + 0.025 * self.payload / self.payload_max
                self.model.geom_size[self._pkg] = [s, s, 0.6 * s]
                self.model.geom_pos[self._pkg] = [0.0, 0.0, -0.03 - 0.6 * s]
        self.thrust_eff = 1.0
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._wind = np.zeros(3)

        q = np.zeros(self.model.nq)
        q[0:3] = self._rng.uniform(-self.init_noise, self.init_noise, size=3)
        q[2] = self.init_height + self._rng.uniform(-self.init_noise, self.init_noise)
        if self.init_tilt > 0.0:
            ang = self._rng.uniform(0.0, self.init_tilt)
            axis = self._rng.normal(size=3)
            axis /= (np.linalg.norm(axis) + 1e-9)
            q[3:7] = [np.cos(ang / 2.0), *(np.sin(ang / 2.0) * axis)]
        else:
            q[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[:] = q
        v = self._rng.uniform(-self.init_noise, self.init_noise, size=self.model.nv)
        if self.init_spin > 0.0:
            v[3:6] = self._rng.uniform(-self.init_spin, self.init_spin, size=3)
        self.data.qvel[:] = v

        for _try in range(20):
            pad_xy = self._rng.uniform(-self.target_range_xy, self.target_range_xy, size=2)
            if np.linalg.norm(pad_xy) >= self.pad_min_dist:
                break
        self._target = np.array([pad_xy[0], pad_xy[1], self.pad_z])
        if self.spawn_above:
            ang = self._rng.uniform(0.0, 2.0 * np.pi)
            rad = self._rng.uniform(self.start_offset_min, self.start_offset)
            self.data.qpos[0:2] = self._target[:2] + rad * np.array([np.cos(ang), np.sin(ang)])
            self.data.qpos[2] = self._rng.uniform(*self.start_height)
        if self.race:                                   # start on the course at a random point
            self._target = np.zeros(3)
            j = int(self._rng.integers(len(self.course["points"])))
            self.data.qpos[0:3] = (self.course["points"][j]
                                   + self._rng.uniform(-self.init_noise, self.init_noise, size=3))
            self._race_k, self.race_progress = j, 0.0
        if self.wind_mean_max > 0.0:
            ang = self._rng.uniform(0.0, 2.0 * np.pi)
            mag = self._rng.uniform(0.0, self.wind_mean_max)
            self._wind_mean = np.array([mag * np.cos(ang), mag * np.sin(ang), 0.0])
        self._obstacles = self._place_obstacles()
        self._sync_markers()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {"target": self._target.copy()}

    def _sync_markers(self):
        if "pad" in self._mocap:
            if self.race:
                self.data.mocap_pos[self._mocap["pad"]] = [0.0, 0.0, -5.0]
            elif self.spawn_above:
                self.data.mocap_pos[self._mocap["pad"]] = [self._target[0], self._target[1], 0.0]
            else:
                self.data.mocap_pos[self._mocap["pad"]] = self._target
        if "cage" in self._mocap:
            self.data.mocap_pos[self._mocap["cage"]] = [self._target[0], self._target[1], 0.0]
        if "windsock_pole" in self._mocap:
            x, y = self._target[0] + _WINDSOCK_AT[0], self._target[1] + _WINDSOCK_AT[1]
            self.data.mocap_pos[self._mocap["windsock_pole"]] = [x, y, 0.0]
            self.data.mocap_pos[self._mocap["windsock"]] = [x, y, 1.0]
        for i in range(self.n_obstacles):
            key = f"obs{i}"
            if key in self._mocap:
                self.data.mocap_pos[self._mocap[key]] = [self._obstacles[i, 0], self._obstacles[i, 1], 1.5]

    def _init_attitude_controller(self):
        m = self.model
        xy = np.array([m.site_pos[m.actuator_trnid[i, 0]][:2] for i in range(m.nu)])
        cyaw = np.array([float(m.actuator_gear[i, 5]) for i in range(m.nu)])
        self._att_minv = np.linalg.inv(np.vstack([np.ones(m.nu), xy[:, 1], -xy[:, 0], cyaw]))
        inertia = np.asarray(m.body_inertia[self._core_id], dtype=np.float64)
        wn, zeta = float(self.cfg.get("att_wn", 12.0)), float(self.cfg.get("att_zeta", 0.8))
        self._att_kp, self._att_kd = inertia * wn ** 2, inertia * 2.0 * zeta * wn
        self._att_kyaw = float(self.cfg.get("att_yaw_gain", 0.004))
        self._g = float(-m.opt.gravity[2])
        self.att_acc_h = self._g * float(np.tan(self.att_max_tilt))     # sideways m/s^2 at |action| = 1

    def attitude_to_rotors(self, action) -> np.ndarray:
        """On-board stabiliser: [ax, ay, az, yaw rate] in [-1, 1] -> rotor commands in [-1, 1].

        Sideways accel = (ax, ay) x g tan(max tilt); vertical accel = g (1 + az) for the current mass,
        or in "althold" az = kz (climb-rate command - vz) / g. Tilt is capped at att_max_tilt_deg;
        attitude PD torques get priority over the collective when a rotor saturates."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        mass = float(self.model.body_subtreemass[self._core_id])
        if self.ctrl_mode == "althold":
            az = float(np.clip(self.alt_kz * (a[2] * self.alt_vz_max - float(self.data.qvel[2])) / self._g,
                               -1.0, 1.0))
        else:
            az = float(a[2])
        acc = np.array([a[0] * self.att_acc_h, a[1] * self.att_acc_h, self._g * (1.0 + az)])
        h, h_max = float(np.hypot(acc[0], acc[1])), float(np.tan(self.att_max_tilt)) * max(acc[2], 0.0)
        if h > h_max:
            acc[:2] *= h_max / h
        rot = _quat_to_mat(self.data.qpos[3:7])
        zb, w = rot[:, 2], np.asarray(self.data.qvel[3:6], dtype=np.float64)
        thrust = mass * max(float(acc @ zb), 0.0)
        n = float(np.linalg.norm(acc))
        z_des = acc / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
        tau = self._att_kp * (rot.T @ np.cross(zb, z_des)) - self._att_kd * w
        tau[2] = self._att_kyaw * (a[3] * self.att_yaw_rate - w[2])
        f_col = self._att_minv[:, 0] * thrust
        f_tau = self._att_minv[:, 1:] @ tau
        lo = max(self._hover * (1.0 - self.thrust_gain), 0.0)
        hi = min(self._hover * (1.0 + self.thrust_gain), self._ctrl_hi)
        span = float(f_tau.max() - f_tau.min())
        if span > hi - lo:
            f_tau = f_tau * (hi - lo) / span
        f = f_col + f_tau
        f = f + max(0.0, lo - float(f.min()))
        f = f - max(0.0, float(f.max()) - hi)
        return np.clip((f / self._hover - 1.0) / self.thrust_gain, -1.0, 1.0)

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        rotor = self.attitude_to_rotors(action) if self.ctrl_mode != "rotor" else action

        thrust = self._hover * (1.0 + rotor * self.thrust_gain)
        if self.act_noise > 0.0:
            thrust = thrust * (1.0 + self._rng.normal(0.0, self.act_noise, size=self.n_act))
        if self.vrs_loss > 0.0:
            v = self.data.qvel[0:3]
            if self.vrs_powered:
                ratio = float(np.mean(1.0 + rotor * self.thrust_gain))
                self.thrust_eff = _powered_lift_factor(v, self.data.qpos[3:7], ratio, self.vrs_loss,
                                                       self.vrs_speed, self.vrs_full, self.vrs_escape,
                                                       self.vrs_upright, self.vrs_thrust)
            else:
                self.thrust_eff = _lift_factor(-float(v[2]), float(np.hypot(v[0], v[1])), self.vrs_loss,
                                               self.vrs_speed, self.vrs_full, self.vrs_escape)
            thrust = thrust * self.thrust_eff
        self.data.ctrl[:] = np.clip(thrust, 0.0, self._ctrl_hi)

        if self.wind_force > 0.0:                       # OU gusts
            self._wind = (self.wind_corr * self._wind +
                          np.sqrt(1.0 - self.wind_corr ** 2) *
                          self._rng.normal(0.0, self.wind_force, size=3))
        force = self._wind + self._wind_mean if self.wind_mean_max > 0.0 else self._wind
        self.data.xfrc_applied[self._core_id, 0:3] = force

        for _ in range(self.action_repeat):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()
        if self.race:
            reward = float(_race_reward(obs, action, self.rw, self.course)[0])
        else:
            reward = float(_drone_reward(obs, action, self.rw, self.n_obstacles,
                                         self.obs_radius, self.obs_scale, cage=self._cage)[0])

        dist = float(np.linalg.norm(obs[_REL]))
        up_z = 1.0 - 2.0 * (float(obs[_QX]) ** 2 + float(obs[_QY]) ** 2)
        speed = float(np.linalg.norm(obs[_LV]))
        height = float(obs[_H])
        hit_obs = self._nearest_obstacle(obs) < self.obs_radius
        hard = height < self.impact_height and abs(float(obs[_VZ])) > self.hard_speed
        hit_cage = self._cage is not None and bool(_cage_hit(obs, self._cage)[0])
        crashed = bool(up_z < self.fail_tilt or hit_obs or hard or hit_cage or dist > self.max_dist)
        if self.race:
            from src.envs.race_course import project
            k, d_course = project(self.course, -obs[_REL])
            k, d_course = int(k[0]), float(d_course[0])
            crashed = crashed or d_course > self.race_max_off or height < self.race_floor
            L = self.course["length"]
            ds = (self.course["s"][k] - self.course["s"][self._race_k] + L / 2.0) % L - L / 2.0
            if d_course < 1.0:
                self.race_progress += ds
            self._race_k = k
            landed = bool(self.race_progress >= self.course["length"])     # a full lap this flight
            if crashed:
                reward -= float(self.rw.get("w_crash", 0.0))
        else:
            landed = bool(dist < self.land_radius and speed < self.soft_speed and up_z > 0.9)

        truncated = self._step_count >= self.max_episode_steps
        info = {"failure": bool(crashed), "dist": dist, "up_z": up_z, "reached": landed,
                "thrust_eff": self.thrust_eff, "payload": self.payload}
        if self.race:
            info["laps"] = self.race_progress / self.course["length"]
        return obs, reward, crashed, truncated, info

    def _nearest_obstacle(self, obs) -> float:
        if self.n_obstacles == 0:
            return np.inf
        oxy = np.asarray(obs)[_OBS0:_OBS0 + 2 * self.n_obstacles].reshape(self.n_obstacles, 2)
        return float(np.linalg.norm(oxy, axis=-1).min())

    def render(self):
        if self._renderer is None:
            raise RuntimeError("rendering unavailable (env made with render=False, or no headless GL backend)")
        self._renderer.update_scene(self.data, camera=-1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_env(cfg: dict, seed: int = 0, render: bool = False):
    """cfg = the `env` sub-config. Returns (env, obs_dim, act_dim)."""
    env = DroneTargetEnv(cfg, seed=seed, render=render)
    return env, env.observation_space.shape[0], env.action_space.shape[0]


def known_reward_fn(cfg: dict):
    """reward_fn(obs, act) -> (batch,): the env's own reward (race: including the crash cost)."""
    rw = cfg["reward"]
    if cfg.get("race", False):
        from src.envs.race_course import build_course
        course, crashed, w_crash = build_course(cfg), termination_fn(cfg), float(rw.get("w_crash", 0.0))
        return lambda obs, act: _race_reward(obs, act, rw, course) - w_crash * crashed(obs)
    n_obs = int(cfg.get("n_obstacles", 2))
    obs_radius = float(cfg.get("obstacle_radius", 0.4))
    obs_scale = float(rw.get("obs_scale", cfg.get("obstacle_scale", 0.5)))
    cage = _cage_params(cfg)

    def reward_fn(obs, act):
        return _drone_reward(obs, act, rw, n_obs, obs_radius, obs_scale, cage=cage)

    return reward_fn


def termination_fn(cfg: dict):
    """done_fn(obs) -> (batch,) bool: the env's crash rule."""
    fail_tilt = float(cfg.get("fail_tilt", 0.0))
    max_dist = float(cfg.get("max_dist", 8.0))
    n_obs = int(cfg.get("n_obstacles", 2))
    obs_radius = float(cfg.get("obstacle_radius", 0.4))
    impact_height = float(cfg.get("impact_height", 0.06))
    hard_speed = float(cfg.get("hard_speed", 1.0))
    cage = _cage_params(cfg)
    course = None
    if cfg.get("race", False):
        from src.envs.race_course import build_course
        course = build_course(cfg)
        race_max_off = float(cfg.get("race_max_off", 1.5))
        race_floor = float(cfg.get("race_floor", 0.12))

    def done_fn(obs):
        obs = np.atleast_2d(np.asarray(obs))
        up_z = 1.0 - 2.0 * (obs[:, _QX] ** 2 + obs[:, _QY] ** 2)
        dist = np.linalg.norm(obs[:, _REL], axis=-1)
        hard = (obs[:, _H] < impact_height) & (np.abs(obs[:, _VZ]) > hard_speed)
        done = (up_z < fail_tilt) | (dist > max_dist) | hard
        if n_obs > 0:
            oxy = obs[:, _OBS0:_OBS0 + 2 * n_obs].reshape(obs.shape[0], n_obs, 2)
            done = done | (np.linalg.norm(oxy, axis=-1).min(axis=-1) < obs_radius)
        if cage is not None:
            done = done | _cage_hit(obs, cage)
        if course is not None:
            from src.envs.race_course import project
            done = done | (project(course, -obs[:, _REL])[1] > race_max_off) | (obs[:, _H] < race_floor)
        return done

    return done_fn
