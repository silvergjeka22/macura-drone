"""Drone (quadrotor) go-to-target env: a free-flying Skydio-X2-style quadcopter learns
RECOVER from a tumble, FLY TO A TARGET, and hold a stable hover:

  * a smooth, learnable flight regime (great for model-based sample efficiency), and
  * an aggressive regime (high-angular-rate recovery, fast accel + decel) where a
    fixed-horizon rollout over-imagines - the phase MACURA truncates.

Pure-function library: this module only DEFINES things. Public entry points:
  make_env(cfg, seed, render) -> (env, obs_dim(14), act_dim(4))
  known_reward_fn(cfg) / termination_fn(cfg)   (analytic - real == imagined)

Body (envs/assets/drone.xml): a free-joint rigid body with 4 rotor thrust motors in an
X layout. Each motor's control is a thrust; we map the policy's action in [-1, 1] to a
thrust CENTERED ON HOVER (action 0 -> the drone holds altitude), so learning is about
small corrections rather than discovering hover from scratch.

Observation (14-dim):
    [ target_rel_x, target_rel_y, target_rel_z,   # target position MINUS drone position
      height,                                       # drone world z (for ground-crash test)
      qw, qx, qy, qz,                               # body orientation quaternion
      vx, vy, vz,                                   # linear velocity
      wx, wy, wz ]                                  # angular velocity
Action (4-dim): normalized thrust delta per rotor in [-1, 1] (0 = hover).

Reward (dense, analytic in (obs, action)):
    + w_pos * exp(-dist/scale)   reward being AT the target (max at the target)
    + w_level * up_z             reward staying level (up_z = body-z world component)
    - w_spin * |ang_vel|         penalize spinning
    - w_vel  * |lin_vel|         penalize speed (so it STOPS at the target, not overshoot)
    - w_ctrl * |action|^2        mild control penalty (deviation from hover)
Termination: crashed if the body flips past 90 (up_z < 0), hits the ground
(height < fail_height), or flies away (dist > max_dist) - all pure functions of obs.
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

# observation column indices (single source of truth for reward/termination)
_REL = slice(0, 3)
_H = 3
_QW, _QX, _QY, _QZ = 4, 5, 6, 7
_LV = slice(8, 11)
_AV = slice(11, 14)


# ── the dense go-to-target reward (vectorized; real and imagined use THIS) ──────
def _drone_reward(obs: np.ndarray, act: np.ndarray, rw: dict) -> np.ndarray:
    """Shaped go-to-target reward from (obs, action). Returns (batch,).

    Works on a single (14,) obs or a batched (B, 14) obs; always returns a 1-D array.
    """
    obs = np.atleast_2d(np.asarray(obs, dtype=np.float64))
    act = np.atleast_2d(np.asarray(act, dtype=np.float64))
    dist = np.linalg.norm(obs[:, _REL], axis=-1)
    up_z = 1.0 - 2.0 * (obs[:, _QX] ** 2 + obs[:, _QY] ** 2)   # body-z world component
    spin = np.linalg.norm(obs[:, _AV], axis=-1)
    speed = np.linalg.norm(obs[:, _LV], axis=-1)

    r_pos = float(rw["w_pos"]) * np.exp(-dist / max(float(rw["pos_scale"]), 1e-6))
    r_level = float(rw["w_level"]) * up_z
    r_spin = float(rw["w_spin"]) * spin
    r_vel = float(rw["w_vel"]) * speed
    r_ctrl = float(rw["w_ctrl"]) * np.sum(act ** 2, axis=-1)
    return r_pos + r_level - r_spin - r_vel - r_ctrl


class DroneTargetEnv(gym.Env):
    """Quadrotor flying to a target and holding hover, on the Drone MJCF model."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: dict, seed: int = 0, render: bool = False):
        super().__init__()
        if mujoco is None:
            raise ImportError("mujoco is required to instantiate DroneTargetEnv")

        self.cfg = cfg
        self.action_repeat = int(cfg.get("action_repeat", 2))
        self.max_episode_steps = int(cfg.get("max_episode_steps", 250))
        self.init_height = float(cfg.get("init_height", 1.0))
        self.init_noise = float(cfg.get("init_noise", 0.05))
        self.init_tilt = float(cfg.get("init_tilt", 0.0))       # aggressive: max random start tilt (rad)
        self.init_spin = float(cfg.get("init_spin", 0.0))       # aggressive: max random start spin (rad/s)
        self.target_range_xy = float(cfg.get("target_range_xy", 1.5))
        self.target_z = (float(cfg.get("target_z_min", 0.8)), float(cfg.get("target_z_max", 1.8)))
        self.thrust_gain = float(cfg.get("thrust_gain", 1.0))   # action*gain*hover about hover
        self.max_dist = float(cfg.get("max_dist", 8.0))
        self.fail_tilt = float(cfg.get("fail_tilt", 0.0))       # up_z below this = crashed
        self.fail_height = float(cfg.get("fail_height", 0.15))
        self.rw = cfg["reward"]

        self.model = mujoco.MjModel.from_xml_path(cfg.get("mjcf_scene") or _ASSET)
        self.data = mujoco.MjData(self.model)
        self.n_act = self.model.nu

        g = float(-self.model.opt.gravity[2])
        self._hover = self.model.body_mass.sum() * g / self.n_act   # per-rotor hover thrust
        self._ctrl_hi = float(self.model.actuator_ctrlrange[0][1])

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32)
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

    # ── observation ────────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        q = self.data.qpos
        v = self.data.qvel
        pos = q[0:3]
        rel = self._target - pos
        return np.array(
            [rel[0], rel[1], rel[2], q[2],
             q[3], q[4], q[5], q[6],
             v[0], v[1], v[2], v[3], v[4], v[5]],
            dtype=np.float32,
        )

    # ── gym API ────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0

        n = self.model.nq
        q = np.zeros(n)
        q[0:3] = self._rng.uniform(-self.init_noise, self.init_noise, size=3)
        q[2] = self.init_height + self._rng.uniform(-self.init_noise, self.init_noise)
        # AGGRESSIVE start (recovery task): random tilt (axis-angle -> quaternion) so the
        # drone must recover before it can reach the target. init_tilt=0 -> level start.
        if self.init_tilt > 0.0:
            ang = self._rng.uniform(0.0, self.init_tilt)
            axis = self._rng.normal(size=3)
            axis /= (np.linalg.norm(axis) + 1e-9)
            q[3:7] = [np.cos(ang / 2.0), *(np.sin(ang / 2.0) * axis)]
        else:
            q[3:7] = [1.0, 0.0, 0.0, 0.0]                 # level orientation
        self.data.qpos[:] = q
        v = self._rng.uniform(-self.init_noise, self.init_noise, size=self.model.nv)
        if self.init_spin > 0.0:                          # random initial angular velocity (tumble)
            v[3:6] = self._rng.uniform(-self.init_spin, self.init_spin, size=3)
        self.data.qvel[:] = v

        self._target = np.array([
            self._rng.uniform(-self.target_range_xy, self.target_range_xy),
            self._rng.uniform(-self.target_range_xy, self.target_range_xy),
            self._rng.uniform(self.target_z[0], self.target_z[1]),
        ])
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {"target": self._target.copy()}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        thrust = np.clip(self._hover * (1.0 + action * self.thrust_gain), 0.0, self._ctrl_hi)
        self.data.ctrl[:] = thrust
        for _ in range(self.action_repeat):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()
        reward = float(_drone_reward(obs, action, self.rw)[0])

        dist = float(np.linalg.norm(obs[_REL]))
        up_z = 1.0 - 2.0 * (float(obs[_QX]) ** 2 + float(obs[_QY]) ** 2)
        crashed = bool(up_z < self.fail_tilt or float(obs[_H]) < self.fail_height or dist > self.max_dist)
        terminated = crashed
        truncated = self._step_count >= self.max_episode_steps

        info = {
            "failure": bool(crashed),
            "dist": dist,
            "up_z": up_z,
            "reached": bool(dist < 0.3 and up_z > 0.9),   # near target and level
        }
        return obs, reward, terminated, truncated, info

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

    def reward_fn(obs, act):
        return _drone_reward(obs, act, rw)

    return reward_fn


def termination_fn(cfg: dict):
    """Return done_fn(obs) -> (batch,) bool: the drone crashed (flipped, hit ground, or
    flew away). Pure function of the observation, so imagined rollouts truncate on the
    same envelope as the real environment."""
    fail_tilt = float(cfg.get("fail_tilt", 0.0))
    fail_height = float(cfg.get("fail_height", 0.15))
    max_dist = float(cfg.get("max_dist", 8.0))

    def done_fn(obs):
        obs = np.atleast_2d(np.asarray(obs))
        up_z = 1.0 - 2.0 * (obs[:, _QX] ** 2 + obs[:, _QY] ** 2)
        dist = np.linalg.norm(obs[:, _REL], axis=-1)
        return (up_z < fail_tilt) | (obs[:, _H] < fail_height) | (dist > max_dist)

    return done_fn
