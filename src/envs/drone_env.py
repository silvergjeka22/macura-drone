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


# ── the dense fly-to-pad-and-land reward (vectorized; real and imagined use THIS) ──────
def _drone_reward(obs, act, rw, n_obs, obs_radius, obs_scale) -> np.ndarray:
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

    return r_pos + r_level + r_settle - r_spin - r_vel - r_ctrl - r_obs


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
        self.pad_z = float(cfg.get("pad_z", 0.2))                 # landing-pad height
        self.thrust_gain = float(cfg.get("thrust_gain", 1.0))
        self.max_dist = float(cfg.get("max_dist", 8.0))
        self.fail_tilt = float(cfg.get("fail_tilt", 0.0))        # up_z below this = flipped = crash

        # PROCESS NOISE (the MACURA lever) -------------------------------------------------
        self.wind_force = float(cfg.get("wind_force", 0.0))      # OU stationary std of the gust force (N)
        self.wind_corr = float(cfg.get("wind_correlation", 0.95))# gust temporal correlation (0..1)
        self.act_noise = float(cfg.get("actuator_noise", 0.0))   # multiplicative per-rotor thrust noise (frac std)

        # OBSTACLES (virtual no-fly cylinders; analytic, so imagined rollouts see the same) -
        self.n_obstacles = int(cfg.get("n_obstacles", 2))
        self.obs_radius = float(cfg.get("obstacle_radius", 0.4))
        self.min_clear = float(cfg.get("obstacle_min_clear", 0.7))

        # LANDING / crash envelope ---------------------------------------------------------
        self.land_radius = float(cfg.get("land_radius", 0.35))   # within this of the pad = "at the pad"
        self.soft_speed = float(cfg.get("soft_speed", 0.5))      # land softly below this speed
        self.impact_height = float(cfg.get("impact_height", 0.06))
        self.hard_speed = float(cfg.get("hard_speed", 1.0))      # hitting the ground faster than this = crash
        self.rw = cfg["reward"]
        self.obs_scale = float(self.rw.get("obs_scale", cfg.get("obstacle_scale", 0.5)))

        self.model = mujoco.MjModel.from_xml_path(cfg.get("mjcf_scene") or _ASSET)
        self.data = mujoco.MjData(self.model)
        self.n_act = self.model.nu

        self._core_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "core")
        g = float(-self.model.opt.gravity[2])
        self._hover = float(self.model.body_subtreemass[self._core_id]) * g / self.n_act
        self._ctrl_hi = float(self.model.actuator_ctrlrange[0][1])

        # mocap ids for the visual pad + obstacle markers (rendering only; guarded)
        self._mocap = {}
        for name in ["pad"] + [f"obs{i}" for i in range(self.n_obstacles)]:
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
        """Sample obstacle xy, keeping them clear of the start (origin) and the pad."""
        obs = []
        for _ in range(self.n_obstacles):
            for _try in range(20):
                p = self._rng.uniform(-self.target_range_xy, self.target_range_xy, size=2)
                clear = (np.linalg.norm(p) > self.min_clear + self.obs_radius and
                         np.linalg.norm(p - self._target[:2]) > self.min_clear + self.obs_radius and
                         all(np.linalg.norm(p - q) > 2 * self.obs_radius + 0.2 for q in obs))
                if clear:
                    break
            obs.append(p)
        return np.array(obs).reshape(self.n_obstacles, 2)

    # ── gym API ────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._wind = np.zeros(3)

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

        # landing pad on the ground, obstacles between start and pad
        self._target = np.array([
            self._rng.uniform(-self.target_range_xy, self.target_range_xy),
            self._rng.uniform(-self.target_range_xy, self.target_range_xy),
            self.pad_z,
        ])
        self._obstacles = self._place_obstacles()
        self._sync_markers()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {"target": self._target.copy()}

    def _sync_markers(self):
        """Move the visual pad + obstacle markers (rendering only; no effect on physics)."""
        if "pad" in self._mocap:
            self.data.mocap_pos[self._mocap["pad"]] = self._target
        for i in range(self.n_obstacles):
            key = f"obs{i}"
            if key in self._mocap:
                self.data.mocap_pos[self._mocap[key]] = [self._obstacles[i, 0], self._obstacles[i, 1], 1.5]

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        thrust = self._hover * (1.0 + action * self.thrust_gain)
        if self.act_noise > 0.0:                              # per-rotor actuator (process) noise
            thrust = thrust * (1.0 + self._rng.normal(0.0, self.act_noise, size=self.n_act))
        self.data.ctrl[:] = np.clip(thrust, 0.0, self._ctrl_hi)

        if self.wind_force > 0.0:                             # temporally-correlated gust force (OU)
            self._wind = (self.wind_corr * self._wind +
                          np.sqrt(1.0 - self.wind_corr ** 2) *
                          self._rng.normal(0.0, self.wind_force, size=3))
            self.data.xfrc_applied[self._core_id, 0:3] = self._wind

        for _ in range(self.action_repeat):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()
        reward = float(_drone_reward(obs, action, self.rw, self.n_obstacles,
                                     self.obs_radius, self.obs_scale)[0])

        dist = float(np.linalg.norm(obs[_REL]))
        up_z = 1.0 - 2.0 * (float(obs[_QX]) ** 2 + float(obs[_QY]) ** 2)
        speed = float(np.linalg.norm(obs[_LV]))
        height = float(obs[_H])
        hit_obs = self._nearest_obstacle(obs) < self.obs_radius
        hard = height < self.impact_height and abs(float(obs[_VZ])) > self.hard_speed
        crashed = bool(up_z < self.fail_tilt or hit_obs or hard or dist > self.max_dist)
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

    def reward_fn(obs, act):
        return _drone_reward(obs, act, rw, n_obs, obs_radius, obs_scale)

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

    def done_fn(obs):
        obs = np.atleast_2d(np.asarray(obs))
        up_z = 1.0 - 2.0 * (obs[:, _QX] ** 2 + obs[:, _QY] ** 2)
        dist = np.linalg.norm(obs[:, _REL], axis=-1)
        hard = (obs[:, _H] < impact_height) & (np.abs(obs[:, _VZ]) > hard_speed)
        done = (up_z < fail_tilt) | (dist > max_dist) | hard
        if n_obs > 0:
            oxy = obs[:, _OBS0:_OBS0 + 2 * n_obs].reshape(obs.shape[0], n_obs, 2)
            done = done | (np.linalg.norm(oxy, axis=-1).min(axis=-1) < obs_radius)
        return done

    return done_fn
