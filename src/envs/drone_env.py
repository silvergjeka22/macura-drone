"""Drone in wind: FLY THROUGH obstacles and LAND softly on a pad, under turbulence.

A free-flying quadrotor starts at altitude and must reach a landing pad on the ground and
settle on it softly and upright, while a temporally-correlated WIND (gusts) and ACTUATOR
NOISE push it around and it must avoid a couple of no-fly obstacles. This is deliberately
built to be MACURA's strongest honest case on the drone:

  * the flight between gusts is smooth and learnable -> model-based RL is sample-efficient;
  * the PROCESS NOISE (wind + actuator noise) makes the learned model UNCERTAIN across the
    whole trajectory, so a fixed-horizon rollout (MBPO) over-imagines and its predictions
    diverge, while MACURA truncates the imagined rollout where the ensemble disagrees;
  * obstacles + a soft-landing envelope PUNISH that over-imagination with crashes.

Pure-function library: this module only DEFINES things. Public entry points:
  make_env(cfg, seed, render) -> (env, obs_dim(14 + 2*n_obstacles), act_dim(4))
  known_reward_fn(cfg) / termination_fn(cfg)   (analytic in (obs, action) - real == imagined)

The wind and actuator noise live ONLY in the real env dynamics; the reward and termination
stay pure functions of (obs, action), so real and imagined transitions are scored identically
and the ensemble simply learns the noisy dynamics (its predictive spread IS the uncertainty
MACURA acts on).

Observation (14 + 2*n_obstacles):
    [ pad_rel_x, pad_rel_y, pad_rel_z,     # landing-pad position MINUS drone position
      height,                               # drone world z
      qw, qx, qy, qz,                       # body orientation quaternion
      vx, vy, vz,                           # linear velocity
      wx, wy, wz,                           # angular velocity
      o1_dx, o1_dy, o2_dx, o2_dy, ... ]     # each obstacle's xy position MINUS drone xy
Action (4-dim): normalized thrust delta per rotor in [-1, 1] (0 = hover).
"""

from __future__ import annotations

import os
import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None

import gymnasium as gym
from gymnasium import spaces


_ASSET = os.path.join(os.path.dirname(__file__), "assets", "drone.xml")

# observation column indices for the fixed first 14 dims (obstacles are appended after)
_REL = slice(0, 3)
_H = 3
_QW, _QX, _QY, _QZ = 4, 5, 6, 7
_LV = slice(8, 11)
_VZ = 10
_AV = slice(11, 14)
_OBS0 = 14                       # obstacle-relative xy entries start here
_CORE_HALF_H = 0.02              # drone core box half-height: resting core height = surface + this


# ── the dense fly-to-pad-and-land reward (vectorized; real and imagined use THIS) ──────
def _drone_reward(obs, act, rw, n_obs, obs_radius, obs_scale, cage=None) -> np.ndarray:
    """Shaped land-on-pad reward from (obs, action). Returns (batch,).

    Works on a single obs or a batched (B, D) obs; always returns a 1-D array. Depends only
    on quantities present in obs (pad-relative pose, velocities, obstacle-relative xy), so it
    scores real and imagined transitions identically."""
    obs = np.atleast_2d(np.asarray(obs, dtype=np.float64))
    act = np.atleast_2d(np.asarray(act, dtype=np.float64))
    dist = np.linalg.norm(obs[:, _REL], axis=-1)                 # 3-D distance to the pad
    up_z = 1.0 - 2.0 * (obs[:, _QX] ** 2 + obs[:, _QY] ** 2)     # body-z world component (upright)
    spin = np.linalg.norm(obs[:, _AV], axis=-1)
    speed = np.linalg.norm(obs[:, _LV], axis=-1)

    r_pos = float(rw["w_pos"]) * np.exp(-dist / max(float(rw["pos_scale"]), 1e-6))
    r_level = float(rw["w_level"]) * up_z
    r_spin = float(rw["w_spin"]) * spin
    r_vel = float(rw["w_vel"]) * speed                            # penalize speed -> soft landing
    r_ctrl = float(rw["w_ctrl"]) * np.sum(act ** 2, axis=-1)

    # settle bonus: a reward "well" that peaks inside the landing envelope (close AND slow AND
    # upright) -> actively drives the drone to REACH & hold the pad. Pure fn of obs, so real == imagined.
    w_settle = float(rw.get("w_settle", 0.0))
    if w_settle > 0.0:
        settle_dist = max(float(rw.get("settle_dist", 0.3)), 1e-6)
        settle_speed = max(float(rw.get("settle_speed", 0.4)), 1e-6)
        r_settle = (w_settle * np.exp(-dist / settle_dist)
                    * np.exp(-speed / settle_speed) * np.clip(up_z, 0.0, 1.0))
    else:
        r_settle = 0.0

    # obstacle-avoidance: smooth penalty that grows as the drone nears an obstacle surface
    r_obs = 0.0
    if n_obs > 0:
        oxy = obs[:, _OBS0:_OBS0 + 2 * n_obs].reshape(obs.shape[0], n_obs, 2)
        d = np.linalg.norm(oxy, axis=-1)                         # (B, n_obs) horizontal dist
        surface = np.maximum(d - float(obs_radius), 0.0)
        r_obs = float(rw["w_obs"]) * np.exp(-surface / max(float(obs_scale), 1e-6)).sum(axis=-1)

    # cage-wall avoidance (cage task): smooth penalty near the cage wall, only BELOW its top, so
    # flying over the cage is free and the drone is pushed to come in over the top, not through it.
    r_cage = 0.0
    if cage is not None and float(rw.get("w_cage", 0.0)) > 0.0:
        radius, height, margin = cage
        dxy = np.linalg.norm(obs[:, 0:2], axis=-1)                  # horizontal distance to the pad
        gap = np.maximum(np.abs(dxy - radius) - margin, 0.0)       # distance to the wall surface
        below = 1.0 / (1.0 + np.exp((obs[:, _H] - height) / 0.08))  # ~1 below the cage top, ~0 above
        r_cage = (float(rw["w_cage"]) * below
                  * np.exp(-gap / max(float(rw.get("cage_scale", 0.15)), 1e-6)))

    return r_pos + r_level + r_settle - r_spin - r_vel - r_ctrl - r_obs - r_cage


def _cage_params(cfg: dict):
    """(radius, height, margin) of the cage wall, or None when the cage task is off. Shared by the
    env and by known_reward_fn / termination_fn, so real and imagined use identical numbers."""
    if not cfg.get("cage", False):
        return None
    return (float(cfg.get("cage_radius", 0.8)), float(cfg.get("cage_height", 1.0)),
            float(cfg.get("cage_margin", 0.15)))


def _cage_hit(obs, cage) -> np.ndarray:
    """Crash into the cage wall: drone within `margin` of the wall ring AND below its top.
    Pure function of obs (pad-relative xy + height), so imagined rollouts crash the same way."""
    obs = np.atleast_2d(np.asarray(obs))
    radius, height, margin = cage
    dxy = np.linalg.norm(obs[:, 0:2], axis=-1)
    return (np.abs(dxy - radius) < margin) & (obs[:, _H] < height)


class DroneTargetEnv(gym.Env):
    """Quadrotor flying through wind + obstacles to land on a pad (Drone MJCF model)."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: dict, seed: int = 0, render: bool = False):
        super().__init__()
        if mujoco is None:
            raise ImportError("mujoco is required to instantiate DroneTargetEnv")

        self.cfg = cfg
        self.action_repeat = int(cfg.get("action_repeat", 2))
        self.max_episode_steps = int(cfg.get("max_episode_steps", 250))
        self.init_height = float(cfg.get("init_height", 2.0))
        self.init_noise = float(cfg.get("init_noise", 0.05))
        self.init_tilt = float(cfg.get("init_tilt", 0.0))
        self.init_spin = float(cfg.get("init_spin", 0.0))
        self.target_range_xy = float(cfg.get("target_range_xy", 1.8))
        self.pad_min_dist = float(cfg.get("pad_min_dist", 1.4))   # min start->pad distance (room for the gate)
        self.pad_z = float(cfg.get("pad_z", 0.2))                 # landing-pad height
        self.thrust_gain = float(cfg.get("thrust_gain", 1.0))
        self.max_dist = float(cfg.get("max_dist", 8.0))
        self.fail_tilt = float(cfg.get("fail_tilt", 0.0))        # up_z below this = flipped = crash

        # PROCESS NOISE (the MACURA lever) -------------------------------------------------
        # Everything here lives ONLY in the real dynamics; reward + termination stay analytic in
        # (obs, agent action), so real == imagined scoring is unchanged.
        self.wind_force = float(cfg.get("wind_force", 0.0))      # CALM-transit OU gust std (N)
        self.wind_corr = float(cfg.get("wind_correlation", 0.95))# gust temporal correlation (0..1)
        self.act_noise = float(cfg.get("actuator_noise", 0.0))   # multiplicative per-rotor thrust noise (frac std)
        # LANDING-ZONE turbulence: extra choppy gusts that switch on near the pad, so the model is
        # reliable in transit but NOT where the landing is decided (uneven model trust).
        self.turb_force = float(cfg.get("turb_force", 0.0))      # zone gust std (N)
        self.turb_corr = float(cfg.get("turb_correlation", 0.8)) # choppier than the calm wind
        self.turb_radius = float(cfg.get("turb_radius", 1.3))    # horizontal distance to pad (m)
        self.turb_ramp = float(cfg.get("turb_ramp", 0.15))       # soft edge width of the zone (m)
        # hidden ADDITIVE action noise (paper App. D.4 method), weaker in transit, stronger in the zone
        self.action_noise = float(cfg.get("action_noise", 0.0))
        self.action_noise_zone = float(cfg.get("action_noise_zone", self.action_noise))
        # GROUND EFFECT: extra lift close to the ground (nonlinear, only felt while landing)
        self.ge_gain = float(cfg.get("ge_gain", 0.0))            # +gain*exp(-h/scale) thrust factor
        self.ge_scale = float(cfg.get("ge_scale", 0.25))         # decay height (m)
        # COLUMN WAKE + PAD DOWNWASH: a DETERMINISTIC, spatially complex air flow around the ring.
        # Unlike random turbulence (which the ensemble learns as noise and then AGREES about), a
        # complex deterministic field in a rarely-visited region is something the members fit
        # DIFFERENTLY -> real epistemic disagreement exactly where the drone must land.
        self.wake_gamma = float(cfg.get("wake_gamma", 0.0))      # swirl strength around each column (m^2/s)
        self.wake_core = float(cfg.get("wake_core", 0.35))       # vortex core radius (m)
        self.wake_decay = float(cfg.get("wake_decay", 0.8))      # swirl fades beyond ~this distance (m)
        self.pad_downwash = float(cfg.get("pad_downwash", 0.0))  # vertical air-speed pattern over the pad (m/s)
        self.air_drag = float(cfg.get("air_drag", 0.8))          # force per unit air speed (N per m/s)

        # OBSTACLES (virtual no-fly columns forming a RING around the pad; analytic) --------
        self.n_obstacles = int(cfg.get("n_obstacles", 2))
        self.obs_radius = float(cfg.get("obstacle_radius", 0.4))
        self.min_clear = float(cfg.get("obstacle_min_clear", 0.7))
        self.ring_radius = float(cfg.get("ring_radius", 0.75))            # column distance from the pad
        self.ring_gap_half = np.radians(float(cfg.get("ring_gap_half_deg", 55.0)))  # entry-gap half-angle

        # LANDING / crash envelope ---------------------------------------------------------
        self.land_radius = float(cfg.get("land_radius", 0.35))   # within this of the pad = "at the pad"
        self.soft_speed = float(cfg.get("soft_speed", 0.5))      # land softly below this speed
        self.impact_height = float(cfg.get("impact_height", 0.06))
        self.hard_speed = float(cfg.get("hard_speed", 1.0))      # hitting the ground faster than this = crash
        # TOUCHDOWN (off by default): the pad becomes a SOLID raised platform the drone must actually
        # settle on. Contact is an abrupt change in the physics (impact, bounce, friction, edges) that
        # smooth ensemble members fit DIFFERENTLY, in a region rarely visited early - a candidate for
        # real model DISAGREEMENT right where the landing is decided.
        self.touchdown = bool(cfg.get("touchdown", False))
        self.pad_height = float(cfg.get("pad_height", 0.15))     # platform top above the floor (m)
        self.pad_radius = float(cfg.get("pad_radius", 0.35))     # platform radius (m)
        self.touch_speed = float(cfg.get("touch_speed", 0.3))    # "resting" below this speed (m/s)
        if self.touchdown:                                        # target = drone resting ON the platform
            self.pad_z = self.pad_height + _CORE_HALF_H
        # CAGE task: the pad is surrounded by a circular cage of vertical bars, open at the top. The
        # drone spawns HIGH above it and must come in over the top and settle in the middle without
        # touching the wall. The wall is analytic (pad-relative xy + height are in the obs), so the
        # crash check and the penalty are identical for real and imagined transitions.
        self.cage = bool(cfg.get("cage", False))
        self._cage = _cage_params(cfg)
        self.cage_radius = float(cfg.get("cage_radius", 0.8))
        self.cage_height = float(cfg.get("cage_height", 1.0))
        self.cage_bars = int(cfg.get("cage_bars", 16))
        self.start_height = (float(cfg.get("start_height_min", 2.2)),
                             float(cfg.get("start_height_max", 2.6)))
        self.start_offset = float(cfg.get("start_offset", 1.0))   # max spawn distance from the pad (m)
        self.rw = cfg["reward"]
        self.obs_scale = float(self.rw.get("obs_scale", cfg.get("obstacle_scale", 0.5)))

        scene = cfg.get("mjcf_scene") or _ASSET
        if self.touchdown or self.cage:
            import xml.etree.ElementTree as ET
            root = ET.parse(scene).getroot()
            if self.touchdown:
                # Make the pad a SOLID platform in the XML *before* compiling, so MuJoCo derives every
                # collision structure (bounding volumes, body contype/affinity) for it. (Editing geom
                # fields after compile does NOT enable the collision - the drone fell through.)
                for g in root.iter("geom"):
                    if g.get("name") == "pad_geom":
                        g.set("size", f"{self.pad_radius} {self.pad_height / 2.0}")
                        g.set("contype", "1")               # drone geoms have contype 1 ->
                        g.set("conaffinity", "1")           # they now collide with the pad
                        g.set("rgba", "0.10 0.80 0.35 1")
            if self.cage:
                # visual cage (no physics - the wall is enforced analytically): one mocap body at the
                # pad with vertical bars on a circle + a hoop along the top. The ring columns go away.
                wb = root.find("worldbody")
                for b in list(wb.findall("body")):
                    if (b.get("name") or "").startswith("obs"):
                        wb.remove(b)
                cage = ET.SubElement(wb, "body", {"name": "cage", "mocap": "true", "pos": "0 0 0"})
                r, hgt, n = self.cage_radius, self.cage_height, self.cage_bars
                pts = [(r * np.cos(2 * np.pi * i / n), r * np.sin(2 * np.pi * i / n)) for i in range(n)]
                for i, (x, y) in enumerate(pts):
                    ET.SubElement(cage, "geom", {
                        "name": f"cage_bar{i}", "type": "cylinder", "size": f"0.018 {hgt / 2}",
                        "pos": f"{x:.4f} {y:.4f} {hgt / 2}", "mass": "0", "contype": "0",
                        "conaffinity": "0", "rgba": "0.82 0.82 0.88 1"})
                    x2, y2 = pts[(i + 1) % n]
                    ET.SubElement(cage, "geom", {
                        "name": f"cage_hoop{i}", "type": "capsule", "size": "0.015",
                        "fromto": f"{x:.4f} {y:.4f} {hgt} {x2:.4f} {y2:.4f} {hgt}", "mass": "0",
                        "contype": "0", "conaffinity": "0", "rgba": "0.82 0.82 0.88 1"})
            self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        else:
            self.model = mujoco.MjModel.from_xml_path(scene)
        self.data = mujoco.MjData(self.model)
        self.n_act = self.model.nu

        self._core_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "core")
        g = float(-self.model.opt.gravity[2])
        self._hover = float(self.model.body_subtreemass[self._core_id]) * g / self.n_act
        self._ctrl_hi = float(self.model.actuator_ctrlrange[0][1])

        # mocap ids for the visual pad + obstacle markers (rendering only; guarded)
        self._mocap = {}
        for name in ["pad", "cage"] + [f"obs{i}" for i in range(self.n_obstacles)]:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0 and self.model.body_mocapid[bid] >= 0:
                self._mocap[name] = int(self.model.body_mocapid[bid])

        obs_dim = 14 + 2 * self.n_obstacles
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
        self._turb = np.zeros(3)

    # ── observation ────────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        q = self.data.qpos
        v = self.data.qvel
        pos = q[0:3]
        rel = self._target - pos
        base = [rel[0], rel[1], rel[2], q[2],
                q[3], q[4], q[5], q[6],
                v[0], v[1], v[2], v[3], v[4], v[5]]
        for o in self._obstacles:
            base += [o[0] - pos[0], o[1] - pos[1]]          # obstacle xy relative to the drone
        return np.array(base, dtype=np.float32)

    def _place_obstacles(self):
        """Place the obstacles as a RING of no-fly columns AROUND the pad, leaving ONE entry gap
        that faces the start. The drone must thread the gap and land in the middle of the ring.

        This concentrates the model uncertainty at the gap and the tight pocket - exactly where a
        fixed-horizon rollout (MBPO) over-imagines (a path that clips a column) and MACURA's
        uncertainty-triggered truncation avoids the bad data. The columns are evenly spaced around
        the arc OUTSIDE the entry gap, so the pad is enclosed on every side except the opening the
        drone comes in through."""
        pad = self._target[:2]
        # the gap faces the start (origin), i.e. the direction from the pad back toward the start
        gap_dir = -pad
        gap_theta = float(np.arctan2(gap_dir[1], gap_dir[0]))
        if self.n_obstacles <= 0:
            return np.zeros((0, 2))
        # spread the columns evenly over the arc that EXCLUDES the entry gap
        arc_lo = gap_theta + self.ring_gap_half
        arc_hi = gap_theta + 2.0 * np.pi - self.ring_gap_half
        thetas = (np.linspace(arc_lo, arc_hi, self.n_obstacles)
                  if self.n_obstacles > 1 else np.array([gap_theta + np.pi]))
        obs = [pad + self.ring_radius * np.array([np.cos(t), np.sin(t)]) for t in thetas]
        return np.array(obs).reshape(self.n_obstacles, 2)

    # ── gym API ────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._wind = np.zeros(3)
        self._turb = np.zeros(3)

        n = self.model.nq
        q = np.zeros(n)
        q[0:3] = self._rng.uniform(-self.init_noise, self.init_noise, size=3)
        q[2] = self.init_height + self._rng.uniform(-self.init_noise, self.init_noise)
        if self.init_tilt > 0.0:                              # mild random start tilt
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

        # landing pad on the ground, a MIN distance from the start so there is room for the
        # obstacle gate between them (and so the demo shows a real traversal, not a spawn-on-pad).
        for _try in range(20):
            pad_xy = self._rng.uniform(-self.target_range_xy, self.target_range_xy, size=2)
            if np.linalg.norm(pad_xy) >= self.pad_min_dist:
                break
        self._target = np.array([pad_xy[0], pad_xy[1], self.pad_z])
        if self.cage:                          # spawn HIGH, above the cage, within start_offset of the pad
            ang = self._rng.uniform(0.0, 2.0 * np.pi)
            rad = self._rng.uniform(0.0, self.start_offset)
            self.data.qpos[0:2] = self._target[:2] + rad * np.array([np.cos(ang), np.sin(ang)])
            self.data.qpos[2] = self._rng.uniform(*self.start_height)
        self._obstacles = self._place_obstacles()
        self._sync_markers()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {"target": self._target.copy()}

    def _sync_markers(self):
        """Move the visual pad + obstacle markers (rendering only; no effect on physics)."""
        if "pad" in self._mocap:
            if self.touchdown:                # solid platform spanning floor .. pad_height
                self.data.mocap_pos[self._mocap["pad"]] = [self._target[0], self._target[1],
                                                           self.pad_height / 2.0]
            else:
                self.data.mocap_pos[self._mocap["pad"]] = self._target
        if "cage" in self._mocap:
            self.data.mocap_pos[self._mocap["cage"]] = [self._target[0], self._target[1], 0.0]
        for i in range(self.n_obstacles):
            key = f"obs{i}"
            if key in self._mocap:
                self.data.mocap_pos[self._mocap[key]] = [self._obstacles[i, 0], self._obstacles[i, 1], 1.5]

    def air_velocity(self, pos) -> np.ndarray:
        """Deterministic local air flow (m/s): alternating swirls around each column (the wake of
        flow past the pillars) + a downdraft/updraft ring pattern over the pad. Zero in transit."""
        v = np.zeros(3)
        if self.wake_gamma > 0.0:
            for i, c in enumerate(self._obstacles):
                d = np.asarray(pos[:2]) - c
                r = float(np.linalg.norm(d)) + 1e-6
                tangent = np.array([-d[1], d[0]]) / r
                speed = (self.wake_gamma * r / (r * r + self.wake_core ** 2)
                         * np.exp(-(r / self.wake_decay) ** 2))
                v[:2] += (1.0 if i % 2 == 0 else -1.0) * speed * tangent
        if self.pad_downwash > 0.0:
            rp = float(np.linalg.norm(np.asarray(pos[:2]) - self._target[:2]))
            v[2] -= self.pad_downwash * np.cos(np.pi * rp / 0.6) * np.exp(-(rp / 0.9) ** 2)
        return v

    def zone_weight(self, pos_xy) -> float:
        """0 in calm transit, 1 inside the landing-zone turbulence (soft edge at turb_radius)."""
        d = float(np.linalg.norm(self._target[:2] - np.asarray(pos_xy)[:2]))
        return float(1.0 / (1.0 + np.exp((d - self.turb_radius) / max(self.turb_ramp, 1e-6))))

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)   # the AGENT's action
        pos = np.array(self.data.qpos[0:3])
        s = self.zone_weight(pos[:2])                         # how deep in the landing zone

        # hidden additive action noise (unobserved -> process noise; paper App. D.4)
        applied = action
        sigma = self.action_noise + s * (self.action_noise_zone - self.action_noise)
        if sigma > 0.0:
            applied = np.clip(action + self._rng.normal(0.0, sigma, size=self.n_act), -1.0, 1.0)
        thrust = self._hover * (1.0 + applied * self.thrust_gain)
        if self.act_noise > 0.0:                              # per-rotor actuator (process) noise
            thrust = thrust * (1.0 + self._rng.normal(0.0, self.act_noise, size=self.n_act))
        if self.ge_gain > 0.0:                                # ground effect: extra lift near the floor
            thrust = thrust * (1.0 + self.ge_gain * np.exp(-max(pos[2], 0.0) / max(self.ge_scale, 1e-6)))
        self.data.ctrl[:] = np.clip(thrust, 0.0, self._ctrl_hi)

        # wind = calm OU gusts everywhere + choppy OU turbulence weighted by the landing zone
        if self.wind_force > 0.0:
            self._wind = (self.wind_corr * self._wind +
                          np.sqrt(1.0 - self.wind_corr ** 2) *
                          self._rng.normal(0.0, self.wind_force, size=3))
        if self.turb_force > 0.0:
            self._turb = (self.turb_corr * self._turb +
                          np.sqrt(1.0 - self.turb_corr ** 2) *
                          self._rng.normal(0.0, self.turb_force, size=3))
        force = self._wind + s * self._turb
        if self.wake_gamma > 0.0 or self.pad_downwash > 0.0:   # deterministic column wake + pad downwash
            force = force + self.air_drag * self.air_velocity(pos)
        self.data.xfrc_applied[self._core_id, 0:3] = force

        for _ in range(self.action_repeat):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()
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
        if self.touchdown:                    # success = actually RESTING on the platform
            on_pad = (float(np.linalg.norm(obs[_REL][:2])) < self.pad_radius
                      and -0.02 < height - self.pad_z < 0.04)
            landed = bool(on_pad and speed < self.touch_speed and up_z > 0.9)
        else:
            landed = bool(dist < self.land_radius and speed < self.soft_speed and up_z > 0.9)

        terminated = crashed
        truncated = self._step_count >= self.max_episode_steps
        info = {"failure": bool(crashed), "dist": dist, "up_z": up_z, "reached": landed}
        return obs, reward, terminated, truncated, info

    def _nearest_obstacle(self, obs) -> float:
        if self.n_obstacles == 0:
            return np.inf
        oxy = np.asarray(obs)[_OBS0:_OBS0 + 2 * self.n_obstacles].reshape(self.n_obstacles, 2)
        return float(np.linalg.norm(oxy, axis=-1).min())

    def render(self):
        if self._renderer is None:
            raise RuntimeError("rendering unavailable (env made with render=False, "
                               "or no headless GL backend)")
        self._renderer.update_scene(self.data, camera=-1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ── builders / analytic reward + termination (shared by the model-based rollouts) ─
def make_env(cfg: dict, seed: int = 0, render: bool = False):
    """Build the Drone env. `cfg` is the `env:` sub-config. Returns (env, obs_dim, act_dim)."""
    env = DroneTargetEnv(cfg, seed=seed, render=render)
    return env, env.observation_space.shape[0], env.action_space.shape[0]


def known_reward_fn(cfg: dict):
    """Return reward_fn(obs, act) -> (batch,), the SAME dense reward the env uses.
    `cfg` is the `env:` sub-config. Used to score imagined transitions identically."""
    rw = cfg["reward"]
    n_obs = int(cfg.get("n_obstacles", 2))
    obs_radius = float(cfg.get("obstacle_radius", 0.4))
    obs_scale = float(rw.get("obs_scale", cfg.get("obstacle_scale", 0.5)))
    cage = _cage_params(cfg)

    def reward_fn(obs, act):
        return _drone_reward(obs, act, rw, n_obs, obs_radius, obs_scale, cage=cage)

    return reward_fn


def termination_fn(cfg: dict):
    """Return done_fn(obs) -> (batch,) bool: the drone crashed (flipped, hit an obstacle,
    hard ground impact, or flew away). Pure function of the observation, so imagined rollouts
    truncate on the same envelope as the real environment."""
    fail_tilt = float(cfg.get("fail_tilt", 0.0))
    max_dist = float(cfg.get("max_dist", 8.0))
    n_obs = int(cfg.get("n_obstacles", 2))
    obs_radius = float(cfg.get("obstacle_radius", 0.4))
    impact_height = float(cfg.get("impact_height", 0.06))
    hard_speed = float(cfg.get("hard_speed", 1.0))
    cage = _cage_params(cfg)

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
        return done

    return done_fn
