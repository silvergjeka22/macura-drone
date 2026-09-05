"""Pogo the pogo-stick gymnast: a planar one-legged body that learns to BACKFLIP.

Pure-function library: this module only DEFINES things. The Colab notebook does
all orchestration. The public entry point is `make_env(cfg, seed, render)`, plus
the analytic `known_reward_fn(cfg)` / `termination_fn(cfg)` the model-based
rollouts use (so imagined transitions are scored exactly like real ones).

Why this task fits MACURA (the whole point of the project):
  A backflip is an AGGRESSIVE, unstable maneuver. Early in training the learned
  ensemble is very uncertain about the airborne / landing phase, so a FIXED-length
  imagined rollout (MBPO) over-imagines the flight, the policy exploits fantasy
  dynamics, and Pogo face-plants. MACURA truncates the rollout the instant the
  ensemble disagrees (high GJS) and learns the flip from trustworthy pieces only.

Body (envs/assets/pogo.xml): motion is confined to the x-z plane. The root is
two slide joints (x, z) plus one UNLIMITED hinge `rooty` (the flip axis, free to
wrap through 360). Three actuated leg hinges (hip / knee / ankle) crouch, launch
and tuck. qpos = [x, z, rooty, hip, knee, ankle]; qvel is the matching 6 rates.

Observation (13-dim, translation-invariant in x):
    [ torso_z, sin(rooty), cos(rooty), hip, knee, ankle, foot_clearance,
      x_vel, z_vel, rooty_vel, hip_vel, knee_vel, ankle_vel ]
  * the torso angle enters as sin/cos so it is smooth across the 360 wrap;
  * `foot_clearance` (how far the foot is off the floor) is the airborne signal
    the reward keys off — the torso rotates in place during a flip, so its height
    is a poor "in the air" cue; the foot leaving the ground is the reliable one.

Action (3-dim): normalized torque in [-1, 1] on hip / knee / ankle.

Reward is a DENSE shaped backflip reward, computed by `_backflip_reward` purely
from (obs, action) so it is identical for real and imagined transitions:
    + alive bonus                (keeps it positive)
    + jump      = foot clearance (reward getting airborne so it CAN flip)
    + rotation  gated by airborne (reward spinning in the flip direction WHILE
                                   airborne -> integrated over the flight = a flip)
    + upright   gated by ground   (reward being upright when NOT airborne -> land
                                   and hold, instead of spinning forever)
    - control effort.
`w_rotation` is the curriculum knob (0 = just jump & balance; large = full flip).

Failure / termination: the torso dropping below `fail_torso_height` (a collapse /
face-plant flat on the mat) ends the episode and is logged as a failure. This is
a pure function of the observation (torso_z), so `termination_fn` matches inside
model rollouts.
"""

from __future__ import annotations

import os
import numpy as np

try:
    import mujoco
except ImportError:  # allow importing the module for inspection without MuJoCo
    mujoco = None

import gymnasium as gym
from gymnasium import spaces


_ASSET = os.path.join(os.path.dirname(__file__), "assets", "pogo.xml")

# observation column indices (single source of truth for reward/termination)
_Z, _SIN, _COS, _FOOTZ, _ROOTY_VEL = 0, 1, 2, 6, 9


# ── the dense backflip reward (vectorized; real and imagined use THIS) ─────────
def _backflip_reward(obs: np.ndarray, act: np.ndarray, rw: dict) -> np.ndarray:
    """Shaped backflip reward from (obs, action). Returns (batch,).

    Works on a single (13,) obs or a batched (B, 13) obs; always returns a 1-D
    array so the env takes [0] and the rollouts use the whole vector.
    """
    obs = np.atleast_2d(np.asarray(obs, dtype=np.float64))
    act = np.atleast_2d(np.asarray(act, dtype=np.float64))
    cos_t = obs[:, _COS]
    foot = np.maximum(obs[:, _FOOTZ], 0.0)
    rooty_vel = obs[:, _ROOTY_VEL]

    airborne = np.clip(foot / max(float(rw["foot_air"]), 1e-6), 0.0, 1.0)
    r_jump = float(rw["w_height"]) * np.minimum(foot, float(rw["height_cap"]))
    r_rotate = float(rw["w_rotation"]) * float(rw["flip_dir"]) * rooty_vel * airborne
    r_upright = float(rw["w_upright"]) * cos_t * (1.0 - airborne)
    r_ctrl = float(rw["w_action"]) * np.sum(act ** 2, axis=-1)

    return float(rw["alive_bonus"]) + r_jump + r_rotate + r_upright - r_ctrl


class PogoBackflipEnv(gym.Env):
    """Planar pogo-stick gymnast learning a backflip on the Pogo MJCF model."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: dict, seed: int = 0, render: bool = False):
        super().__init__()
        if mujoco is None:
            raise ImportError("mujoco is required to instantiate PogoBackflipEnv")

        self.cfg = cfg
        self.action_repeat = int(cfg.get("action_repeat", 5))
        self.max_episode_steps = int(cfg.get("max_episode_steps", 200))
        self.fail_torso_height = float(cfg.get("fail_torso_height", 0.10))
        self.init_height = float(cfg.get("init_height", 0.78))
        self.init_noise = float(cfg.get("init_noise", 0.04))
        self.rw = cfg["reward"]

        self.model = mujoco.MjModel.from_xml_path(cfg.get("mjcf_scene") or _ASSET)
        self.data = mujoco.MjData(self.model)
        self.n_act = self.model.nu
        self._foot_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot_geom")
        self._foot_radius = float(self.model.geom_size[self._foot_gid][0])

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(13,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_act,), dtype=np.float32)

        self.render_enabled = render
        self._renderer = None
        if render:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._prev_rooty = 0.0
        self.cum_flip = 0.0            # cumulative torso rotation (rad) for the success metric

    # ── observation ────────────────────────────────────────────────────────────
    def _foot_clearance(self) -> float:
        return max(float(self.data.geom_xpos[self._foot_gid][2]) - self._foot_radius, 0.0)

    def _get_obs(self) -> np.ndarray:
        q = self.data.qpos
        v = self.data.qvel
        rooty = float(q[2])
        return np.array(
            [q[1], np.sin(rooty), np.cos(rooty), q[3], q[4], q[5], self._foot_clearance(),
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
        self.data.qpos[:] = self._rng.uniform(-self.init_noise, self.init_noise, size=n)
        self.data.qpos[1] = self.init_height + self._rng.uniform(-self.init_noise, self.init_noise)
        self.data.qvel[:] = self._rng.uniform(-self.init_noise, self.init_noise, size=self.model.nv)
        mujoco.mj_forward(self.model, self.data)

        self._prev_rooty = float(self.data.qpos[2])
        self.cum_flip = 0.0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self.data.ctrl[:] = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        for _ in range(self.action_repeat):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()
        reward = float(_backflip_reward(obs, action, self.rw)[0])

        rooty = float(self.data.qpos[2])
        self.cum_flip += rooty - self._prev_rooty          # unwrapped, so full flips accumulate
        self._prev_rooty = rooty

        height = float(self.data.qpos[1])
        terminated = height < self.fail_torso_height        # collapse / face-plant flat
        truncated = self._step_count >= self.max_episode_steps

        flips = abs(self.cum_flip) / (2.0 * np.pi)
        upright = bool(np.cos(rooty) > 0.85 and abs(self.data.qvel[2]) < 1.5
                       and self._foot_clearance() < 0.1)
        info = {
            "failure": bool(terminated),
            "cum_flip_rad": self.cum_flip,
            "flips": float(flips),
            "upright": upright,
            "landed_flip": bool(flips >= 1.0 and upright),   # a full flip, landed and stable
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            raise RuntimeError("env was created with render=False")
        self._renderer.update_scene(self.data, camera=-1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ── builders / analytic reward + termination (shared by the model-based rollouts) ─
def make_env(cfg: dict, seed: int = 0, render: bool = False):
    """Build the Pogo env. `cfg` is the `env:` sub-config. Returns (env, obs_dim, act_dim)."""
    env = PogoBackflipEnv(cfg, seed=seed, render=render)
    return env, env.observation_space.shape[0], env.action_space.shape[0]


def known_reward_fn(cfg: dict):
    """Return reward_fn(obs, act) -> (batch,), the SAME dense reward the env uses.

    `cfg` is the `env:` sub-config. Used to score imagined transitions in the
    model-based rollouts, so fantasy and reality are scored identically.
    """
    rw = cfg["reward"]

    def reward_fn(obs, act):
        return _backflip_reward(obs, act, rw)

    return reward_fn


def termination_fn(cfg: dict):
    """Return done_fn(obs) -> (batch,) bool: the torso dropped below the fail height.

    A pure function of the observation (torso_z), so imagined rollouts truncate on
    the same collapse condition as the real environment.
    """
    fail_h = float(cfg.get("fail_torso_height", 0.10))

    def done_fn(obs):
        obs = np.atleast_2d(np.asarray(obs))
        return obs[:, _Z] < fail_h

    return done_fn
