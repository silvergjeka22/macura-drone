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
  make_env(cfg, seed, render) -> (env, obs_dim(14 + 2*n_obstacles [+1 payload]), act_dim(4))
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
      o1_dx, o1_dy, o2_dx, o2_dy, ...,      # each obstacle's xy position MINUS drone xy
      payload ]                             # DELIVERY task only: package mass / drone mass
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


def _race_reward(obs, act, rw, course) -> np.ndarray:
    """RACE task reward (vectorized, real == imagined): speed ALONG the course, counted only while near it,
    minus distance outside the course tube, plus the small upright / spin / effort terms. There is no finish
    line: every metre of course flown pays, so flying faster always pays more (like "run as fast as you can")."""
    from src.envs.race_course import project
    obs = np.atleast_2d(np.asarray(obs, dtype=np.float64))
    act = np.atleast_2d(np.asarray(act, dtype=np.float64))
    k, d = project(course, -obs[:, _REL])                         # the course centre is the world origin
    prog = np.sum(obs[:, _LV] * course["tangent"][k], axis=-1)    # m/s along the racing direction
    off = np.maximum(d - float(rw.get("race_tube", 0.35)), 0.0)   # outside the tube (m)
    near = np.exp(-(off / max(float(rw.get("race_tube_soft", 0.35)), 1e-6)) ** 2)
    up_z = 1.0 - 2.0 * (obs[:, _QX] ** 2 + obs[:, _QY] ** 2)
    spin = np.linalg.norm(obs[:, _AV], axis=-1)
    return (float(rw.get("w_prog", 1.0)) * prog * near - float(rw.get("w_track", 1.0)) * off
            + float(rw["w_level"]) * up_z - float(rw["w_spin"]) * spin
            - float(rw["w_ctrl"]) * np.sum(act ** 2, axis=-1))


def _add_race_course(root, course):
    """RACE task visuals (no physics): a ring at each gate P1..P4 and dots along the racing line (red dots =
    the chute). Plain world geoms with contype/conaffinity 0 - the course is enforced analytically."""
    import xml.etree.ElementTree as ET
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
    ET.SubElement(wb, "geom", {                       # the chute: a see-through red column
        "name": "chute", "type": "cylinder", "size": "0.35",
        "fromto": f"{top[0]:.4f} {top[1]:.4f} {top[2]:.4f} {bot[0]:.4f} {bot[1]:.4f} {bot[2]:.4f}",
        "contype": "0", "conaffinity": "0", "rgba": "0.95 0.20 0.10 0.18"})
    for g in root.iter("geom"):                       # solid pillars
        if (g.get("name") or "").startswith("obs"):
            g.set("rgba", "0.80 0.30 0.20 1")
    vis = root.find("visual")                         # brighter scene for the wide race camera
    if vis is None:
        vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "headlight", {"ambient": "0.35 0.35 0.38", "diffuse": "0.55 0.55 0.55",
                                     "specular": "0.1 0.1 0.1"})


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


_WINDSOCK_AT = (0.9, 0.3)        # windsock position relative to the pad (m), visual only


def _add_windsock(root):
    """Add a windsock (pole + striped sock) to the MJCF tree - VISUAL ONLY: two mocap bodies with
    massless, non-colliding geoms. The videos point the sock along the gust (src/viz/scenery.py)."""
    import xml.etree.ElementTree as ET
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


def _lift_factor(descent, v_horiz, loss, v_on, v_full, v_escape) -> float:
    """Thrust factor in a fast vertical descent (a simplified vortex-ring-state): 1 below `v_on` m/s of
    descent, dropping smoothly to 1 - `loss` at `v_full`; flying sideways (`v_escape` m/s) clears it.
    A deterministic function of the OBSERVED velocity, so the model can learn it - but only once the
    drone has actually flown that fast (slow-descent data never shows it)."""
    x = float(np.clip((descent - v_on) / max(v_full - v_on, 1e-6), 0.0, 1.0))
    s = x * x * (3.0 - 2.0 * x)                                   # smoothstep 0..1
    return 1.0 - loss * s * float(np.exp(-(v_horiz / max(v_escape, 1e-6)) ** 2))


def _smooth01(x) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _quat_to_mat(quat) -> np.ndarray:
    w, x, y, z = (float(q) for q in quat)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _powered_lift_factor(vel, quat, thrust_ratio, loss, v_on, v_full, v_escape, up_on, thrust_on) -> float:
    """POWERED vortex-ring state (delivery2): the thrust loss needs all three, like the real thing -
      * the drone is UPRIGHT (body-z world component above ~up_on),
      * it sinks fast ALONG ITS ROTOR AXIS (axial descent v_on -> v_full m/s),
      * its rotors are PUSHING (total thrust >= ~thrust_on x the empty drone's hover thrust).
    Flying across the rotor axis (v_escape m/s) clears it. A tumbling or free-falling drone (early crashes)
    never triggers it, so the danger first appears when a competent policy starts to hurry - it is still
    NEW to the model then. Deterministic in (observed pose + velocity, agent action)."""
    w, x, y, z = (float(q) for q in quat)
    zb = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])   # body z in world
    v = np.asarray(vel, dtype=float)
    axial = float(v @ zb)                                    # < 0 = sinking along the rotor axis
    across = float(np.linalg.norm(v - axial * zb))
    s_desc = _smooth01((-axial - v_on) / max(v_full - v_on, 1e-6))
    s_up = _smooth01((zb[2] - up_on) / 0.10)                 # fully on ~0.1 above up_on
    s_pow = _smooth01((thrust_ratio - thrust_on) / 0.3)      # fully on 0.3 above thrust_on
    return 1.0 - loss * s_desc * s_up * s_pow * float(np.exp(-(across / max(v_escape, 1e-6)) ** 2))


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
        # DELIVERY task (all off by default): spawn high above the pad, carry a package whose weight
        # changes EVERY EPISODE and is in the observation, and lose thrust in a fast vertical descent.
        # A heavy package leaves less spare thrust to brake, so the safe descent speed depends on it.
        self.spawn_above = bool(cfg.get("spawn_above_pad", False)) or self.cage
        self.payload_max = float(cfg.get("payload_max", 0.0))     # kg, uniform 0..max per episode; 0 = off
        self.vrs_loss = float(cfg.get("vrs_loss", 0.0))           # max fraction of thrust lost; 0 = off
        self.vrs_speed = float(cfg.get("vrs_speed", 0.8))         # descent speed where the loss starts (m/s)
        self.vrs_full = float(cfg.get("vrs_full", 1.6))           # descent speed of the full loss (m/s)
        self.vrs_escape = float(cfg.get("vrs_escape", 1.0))       # sideways speed that flies out of it (m/s)
        self.vrs_powered = bool(cfg.get("vrs_powered", False))    # delivery2: upright + axial + thrusting only
        self.vrs_upright = float(cfg.get("vrs_upright", 0.85))    # body-z world component where it starts
        self.vrs_thrust = float(cfg.get("vrs_thrust", 0.7))       # thrust / empty-hover thrust where it starts
        self.start_offset_min = float(cfg.get("start_offset_min", 0.0))   # min spawn distance from the pad (m)
        self.payload = 0.0
        self.thrust_eff = 1.0                                     # last step's lift factor (diagnostics/videos)
        self.scenery = bool(cfg.get("scenery", False))            # windsock for the videos (visual only)
        # RACE task (off by default): race laps around a fixed 3-D course (src/envs/race_course.py) with a
        # steep chute; a visible mean wind per episode; crash = ground, flip, pillar, or leaving the course.
        self.race = bool(cfg.get("race", False))
        self.course = None
        if self.race:
            from src.envs.race_course import build_course
            self.course = build_course(cfg)
        self.race_max_off = float(cfg.get("race_max_off", 1.5))  # farther than this from the course = out
        self.race_floor = float(cfg.get("race_floor", 0.12))     # below this height = hit the ground
        self.race_pillars = [tuple(float(v) for v in p) for p in cfg.get("race_pillars", [])]
        self.race_progress = 0.0                                  # course metres flown this episode
        self._race_k = 0
        # RACING-LINE SENSOR (off by default): 6 extra obs dims = the vector from the drone to the nearest point of
        # the racing line + the racing direction `race_obs_ahead` m further along it (a racing drone sees the next
        # gate). A pure function of the position (already in the obs): it adds no information about the physics.
        self.race_obs_course = bool(cfg.get("race_obs_course", False)) and self.race
        self._race_ahead = (int(round(float(cfg.get("race_obs_ahead", 0.5)) / self.course["params"]["spacing"]))
                            if self.race_obs_course else 0)
        # VISIBLE mean wind (off by default): a steady push of random direction and strength every episode,
        # IN the observation (unlike the hidden gusts) - new combinations for the model, not noise.
        self.wind_mean_max = float(cfg.get("wind_mean_max", 0.0))  # N
        self._wind_mean = np.zeros(3)
        # ACTION INTERFACE: "rotor" (default) = the agent sets each rotor's thrust directly (0 = hover of the EMPTY
        # drone). "attitude" = an on-board attitude stabiliser, like a real flight controller: the agent commands
        # [sideways acceleration x, y (m/s^2 via the tilt, world frame), vertical acceleration (0 = hover of the
        # CURRENT mass, package included), yaw rate] and the stabiliser turns that into rotor thrusts. The rotors,
        # their limits and all the physics (lift loss, wind, noise) are unchanged - only who holds the drone level.
        # "althold" = the same stabiliser in ALTITUDE-HOLD mode (the usual flight mode of consumer drones): the third
        # action is a CLIMB RATE (m/s, 0 = hold the height) instead of a vertical acceleration, so random stick
        # inputs average out and a beginner does not build up a fast sink by accident - only a deliberate
        # command does. The height loop is the autopilot's (vertical accel = kz x climb-rate error).
        self.ctrl_mode = str(cfg.get("ctrl_mode", "rotor")).lower()
        self.att_max_tilt = np.radians(float(cfg.get("att_max_tilt_deg", 40.0)))   # tilt limit (rad)
        self.att_yaw_rate = float(cfg.get("att_yaw_rate", 2.0))                     # rad/s at |action| = 1
        self.alt_vz_max = float(cfg.get("alt_vz_max", 3.0))                         # climb/sink rate at |az| = 1
        self.alt_kz = float(cfg.get("alt_kz", 3.0))                                 # height loop gain (1/s)
        self.rw = cfg["reward"]
        self.obs_scale = float(self.rw.get("obs_scale", cfg.get("obstacle_scale", 0.5)))

        scene = cfg.get("mjcf_scene") or _ASSET
        spare_markers = self.n_obstacles < 4 and not self.cage     # column markers the task doesn't use
        if self.touchdown or self.cage or self.payload_max > 0.0 or spare_markers or self.scenery or self.race:
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
            if spare_markers:
                # drop the column markers this task doesn't place (they would float in the scene)
                wb = root.find("worldbody")
                for b in list(wb.findall("body")):
                    name = b.get("name") or ""
                    if name.startswith("obs") and int(name[3:]) >= self.n_obstacles:
                        wb.remove(b)
            if self.payload_max > 0.0:
                # visual package under the drone (massless geom: the real mass is set on the body at
                # every reset). Drawn at max size here; reset() shrinks it to the episode's weight.
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
        g = float(-self.model.opt.gravity[2])
        self._hover = float(self.model.body_subtreemass[self._core_id]) * g / self.n_act   # EMPTY drone
        self._ctrl_hi = float(self.model.actuator_ctrlrange[0][1])
        self._mass0 = float(self.model.body_mass[self._core_id])
        self._pkg = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "package")
        if self.ctrl_mode in ("attitude", "althold"):
            self._init_attitude_controller()
        elif self.ctrl_mode != "rotor":
            raise ValueError(f"unknown ctrl_mode {self.ctrl_mode!r} (use 'rotor', 'attitude' or 'althold')")

        # mocap ids for the visual pad + obstacle markers (rendering only; guarded)
        self._mocap = {}
        for name in ["pad", "cage", "windsock_pole", "windsock"] + [f"obs{i}" for i in range(self.n_obstacles)]:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0 and self.model.body_mocapid[bid] >= 0:
                self._mocap[name] = int(self.model.body_mocapid[bid])

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
        if self.payload_max > 0.0:
            base.append(self.payload / self._mass0)         # package weight, visible to the agent
        if self.wind_mean_max > 0.0:
            base += [self._wind_mean[0], self._wind_mean[1]]   # the episode's steady wind (N), visible
        if self.race_obs_course:                               # racing-line sensor (see __init__)
            from src.envs.race_course import project
            k = int(project(self.course, pos)[0][0])
            c = self.course
            base += list(c["points"][k] - pos) + list(c["tangent"][(k + self._race_ahead) % len(c["points"])])
        return np.array(base, dtype=np.float32)

    def _place_obstacles(self):
        """Place the obstacles as a RING of no-fly columns AROUND the pad, leaving ONE entry gap
        that faces the start. The drone must thread the gap and land in the middle of the ring.

        This concentrates the model uncertainty at the gap and the tight pocket - exactly where a
        fixed-horizon rollout (MBPO) over-imagines (a path that clips a column) and MACURA's
        uncertainty-triggered truncation avoids the bad data. The columns are evenly spaced around
        the arc OUTSIDE the entry gap, so the pad is enclosed on every side except the opening the
        drone comes in through."""
        if self.race:                                  # fixed pillars inside two corners of the course
            return np.array(self.race_pillars, dtype=float).reshape(self.n_obstacles, 2)
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
        if self.payload_max > 0.0:                   # new package every episode (options can force one)
            self.payload = (float(options["payload"]) if options and "payload" in options
                            else float(self._rng.uniform(0.0, self.payload_max)))
            self.model.body_mass[self._core_id] = self._mass0 + self.payload
            mujoco.mj_setConst(self.model, self.data)          # refresh the derived mass constants
            if self._pkg >= 0:                                  # visual size follows the weight
                s = 0.015 + 0.025 * self.payload / self.payload_max
                self.model.geom_size[self._pkg] = [s, s, 0.6 * s]
                self.model.geom_pos[self._pkg] = [0.0, 0.0, -0.03 - 0.6 * s]
        self.thrust_eff = 1.0
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
        if self.spawn_above:                   # spawn HIGH above the pad (cage/delivery), within start_offset
            ang = self._rng.uniform(0.0, 2.0 * np.pi)
            rad = self._rng.uniform(self.start_offset_min, self.start_offset)
            self.data.qpos[0:2] = self._target[:2] + rad * np.array([np.cos(ang), np.sin(ang)])
            self.data.qpos[2] = self._rng.uniform(*self.start_height)
        if self.race:                          # start ON the course at a random point, course centre = origin
            self._target = np.zeros(3)
            j = int(self._rng.integers(len(self.course["points"])))
            self.data.qpos[0:3] = (self.course["points"][j]
                                   + self._rng.uniform(-self.init_noise, self.init_noise, size=3))
            self._race_k, self.race_progress = j, 0.0
        if self.wind_mean_max > 0.0:           # this episode's steady wind: random direction and strength
            ang = self._rng.uniform(0.0, 2.0 * np.pi)
            mag = self._rng.uniform(0.0, self.wind_mean_max)
            self._wind_mean = np.array([mag * np.cos(ang), mag * np.sin(ang), 0.0])
        self._obstacles = self._place_obstacles()
        self._sync_markers()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {"target": self._target.copy()}

    def _sync_markers(self):
        """Move the visual pad + obstacle markers (rendering only; no effect on physics)."""
        if "pad" in self._mocap and self.race:        # no landing pad in the race
            self.data.mocap_pos[self._mocap["pad"]] = [0.0, 0.0, -5.0]
        elif "pad" in self._mocap:
            if self.touchdown:                # solid platform spanning floor .. pad_height
                self.data.mocap_pos[self._mocap["pad"]] = [self._target[0], self._target[1],
                                                           self.pad_height / 2.0]
            elif self.spawn_above:            # cage/delivery: pad drawn ON the floor under the landing point
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

    def _init_attitude_controller(self):
        """The on-board stabiliser of ctrl_mode "attitude" / "althold" (the hand-written autopilot's cascaded PD):
        mixing matrix from the actual rotor sites / yaw gears, attitude gains from the body inertia."""
        m = self.model
        xy = np.array([m.site_pos[m.actuator_trnid[i, 0]][:2] for i in range(m.nu)])
        cyaw = np.array([float(m.actuator_gear[i, 5]) for i in range(m.nu)])
        self._att_minv = np.linalg.inv(np.vstack([np.ones(m.nu), xy[:, 1], -xy[:, 0], cyaw]))
        inertia = np.asarray(m.body_inertia[self._core_id], dtype=np.float64)
        wn, zeta = float(self.cfg.get("att_wn", 12.0)), float(self.cfg.get("att_zeta", 0.8))
        self._att_kp, self._att_kd = inertia * wn ** 2, inertia * 2.0 * zeta * wn
        self._att_kyaw = float(self.cfg.get("att_yaw_gain", 0.004))     # yaw torque per rad/s of yaw-rate error
        self._g = float(-m.opt.gravity[2])
        self.att_acc_h = self._g * float(np.tan(self.att_max_tilt))     # sideways m/s^2 at |action| = 1

    def attitude_to_rotors(self, action) -> np.ndarray:
        """ctrl_mode "attitude"/"althold": action [ax, ay, az, yaw rate] in [-1, 1] -> rotor command in [-1, 1] (the
        same units as ctrl_mode="rotor": thrust = empty-drone hover x (1 + command)). Deterministic in the current
        pose, angular velocity and package weight (all in the obs) and the action; no randomness.
          * wanted thrust acceleration = (ax, ay) x g tan(max tilt) sideways + g (1 + az) up: action 0 = level hover
            for the CURRENT mass; az = -1 cuts the thrust (free fall); az = +1 asks for 2 g (the rotors top out
            at 2x the EMPTY drone's weight, so a heavy package leaves less to brake with - as before);
            "althold": az = kz (climb-rate command - vertical speed) / g instead, clipped to +-1 (0 = hold height);
          * the tilt is capped at `att_max_tilt_deg` from vertical: the drone never flips on its own;
          * attitude PD -> torques; mixing with attitude priority (the collective gives way when a rotor
            saturates, so the drone stays level even at full or zero throttle)."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        mass = float(self.model.body_subtreemass[self._core_id])        # includes the package
        if self.ctrl_mode == "althold":        # az = climb-rate command: vertical accel = kz (vz_cmd - vz), +-1 g
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
        if span > hi - lo:                                          # torques alone exceed the rotor range
            f_tau = f_tau * (hi - lo) / span
        f = f_col + f_tau
        f = f + max(0.0, lo - float(f.min()))                       # attitude first: move the collective
        f = f - max(0.0, float(f.max()) - hi)
        return np.clip((f / self._hover - 1.0) / self.thrust_gain, -1.0, 1.0)

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)   # the AGENT's action
        pos = np.array(self.data.qpos[0:3])
        s = self.zone_weight(pos[:2])                         # how deep in the landing zone
        # rotor command: the agent's action itself, or the on-board stabiliser's output ("attitude" / "althold")
        rotor = self.attitude_to_rotors(action) if self.ctrl_mode != "rotor" else action

        # hidden additive action noise (unobserved -> process noise; paper App. D.4)
        applied = rotor
        sigma = self.action_noise + s * (self.action_noise_zone - self.action_noise)
        if sigma > 0.0:
            applied = np.clip(rotor + self._rng.normal(0.0, sigma, size=self.n_act), -1.0, 1.0)
        thrust = self._hover * (1.0 + applied * self.thrust_gain)
        if self.act_noise > 0.0:                              # per-rotor actuator (process) noise
            thrust = thrust * (1.0 + self._rng.normal(0.0, self.act_noise, size=self.n_act))
        if self.ge_gain > 0.0:                                # ground effect: extra lift near the floor
            thrust = thrust * (1.0 + self.ge_gain * np.exp(-max(pos[2], 0.0) / max(self.ge_scale, 1e-6)))
        if self.vrs_loss > 0.0:                               # lift loss in a fast vertical descent
            v = self.data.qvel[0:3]
            if self.vrs_powered:                              # delivery2: only in a powered, upright descent
                ratio = float(np.mean(1.0 + rotor * self.thrust_gain))    # commanded rotor thrust / hover
                self.thrust_eff = _powered_lift_factor(v, self.data.qpos[3:7], ratio, self.vrs_loss,
                                                       self.vrs_speed, self.vrs_full, self.vrs_escape,
                                                       self.vrs_upright, self.vrs_thrust)
            else:
                self.thrust_eff = _lift_factor(-float(v[2]), float(np.hypot(v[0], v[1])), self.vrs_loss,
                                               self.vrs_speed, self.vrs_full, self.vrs_escape)
            thrust = thrust * self.thrust_eff
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
        if self.wind_mean_max > 0.0:
            force = force + self._wind_mean
        if self.wake_gamma > 0.0 or self.pad_downwash > 0.0:   # deterministic column wake + pad downwash
            force = force + self.air_drag * self.air_velocity(pos)
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
            k, d_course = project(self.course, -obs[_REL])     # same numbers as termination_fn
            k, d_course = int(k[0]), float(d_course[0])
            crashed = crashed or d_course > self.race_max_off or height < self.race_floor
            L = self.course["length"]                     # course metres flown (wraps around the lap)
            ds = (self.course["s"][k] - self.course["s"][self._race_k] + L / 2.0) % L - L / 2.0
            if d_course < 1.0:
                self.race_progress += ds
            self._race_k = k
        if self.race:                         # "reached" = completed a full lap this episode
            landed = bool(self.race_progress >= self.course["length"])
            if crashed:                       # a crash costs (same rule as known_reward_fn)
                reward -= float(self.rw.get("w_crash", 0.0))
        elif self.touchdown:                    # success = actually RESTING on the platform
            on_pad = (float(np.linalg.norm(obs[_REL][:2])) < self.pad_radius
                      and -0.02 < height - self.pad_z < 0.04)
            landed = bool(on_pad and speed < self.touch_speed and up_z > 0.9)
        else:
            landed = bool(dist < self.land_radius and speed < self.soft_speed and up_z > 0.9)

        terminated = crashed
        truncated = self._step_count >= self.max_episode_steps
        info = {"failure": bool(crashed), "dist": dist, "up_z": up_z, "reached": landed,
                "thrust_eff": self.thrust_eff, "payload": self.payload}
        if self.race:
            info["laps"] = self.race_progress / self.course["length"]
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
    if cfg.get("race", False):                 # + the crash penalty, from the SAME crash rule as the env
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
        if course is not None:                         # race: left the course or touched the ground
            from src.envs.race_course import project
            done = done | (project(course, -obs[:, _REL])[1] > race_max_off) | (obs[:, _H] < race_floor)
        return done

    return done_fn
