"""Skydio X2 quadrotor hover / path-following environment (MuJoCo).

Pure-function library: this module only DEFINES things. The Colab notebook does
all orchestration. The public entry point is `make_env(cfg, seed, render)`.

Design choices (all to keep training-step count low — see project plan):
  * near-hover initialization: episodes start close to the target pose with
    small random perturbations, so "competence" is reachable in few steps;
  * action repeat (frame-skip): each agent action is held for several physics
    steps, so fewer agent steps cover the same flight time;
  * dense quadratic reward: smooth, model-predictable cost on pose / velocity /
    tilt / control effort, plus an alive bonus to keep it positive near hover.

The drone model is the Skydio X2 MJCF from `mujoco_menagerie` (cloned by
`bash/setup_colab.sh`). We read the actuator count and control ranges directly
from the model, so this code is robust to the exact actuator definitions.

Observation (13-dim, all relative to the hover target):
    [ pos_error(3), quaternion(4), linear_vel(3), angular_vel(3) ]
Action (model.nu, normally 4): per-rotor thrust command in [-1, 1], linearly
    mapped to each actuator's MuJoCo ctrlrange.
"""

from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError:  # allows importing the module for inspection without MuJoCo
    mujoco = None

import gymnasium as gym
from gymnasium import spaces


# ── small math helpers (pure functions) ───────────────────────────────────────
def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = quat
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


def tilt_angle(quat: np.ndarray) -> float:
    """Angle (rad) between the drone's body-up axis and world-up. 0 = level."""
    R = quat_to_rotmat(quat)
    cos_tilt = float(np.clip(R[2, 2], -1.0, 1.0))
    return float(np.arccos(cos_tilt))


def small_random_quat(rng: np.random.Generator, max_angle: float) -> np.ndarray:
    """Quaternion near identity: random axis, small angle in [0, max_angle]."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-9
    angle = rng.uniform(0.0, max_angle)
    half = angle / 2.0
    w = np.cos(half)
    xyz = axis * np.sin(half)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


# ── the environment ───────────────────────────────────────────────────────────
class DroneHoverEnv(gym.Env):
    """Quadrotor hover (and optional path-following) on the Skydio X2 model."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: dict, seed: int = 0, render: bool = False):
        super().__init__()
        if mujoco is None:
            raise ImportError("mujoco is required to instantiate DroneHoverEnv")

        self.cfg = cfg
        self.task = cfg.get("task", "hover")
        self.target = np.asarray(cfg["target_position"], dtype=np.float64)
        self.action_repeat = int(cfg.get("action_repeat", 1))
        self.max_episode_steps = int(cfg.get("max_episode_steps", 250))
        self.tilt_limit = float(cfg.get("tilt_limit", 1.0))
        self.position_bound = float(cfg.get("position_bound", 1.5))
        self.floor_z = float(cfg.get("floor_z", 0.05))
        self.rw = cfg["reward"]
        self.track_cfg = cfg.get("track", {})

        # load the Skydio X2 model
        self.model = mujoco.MjModel.from_xml_path(cfg["mjcf_scene"])
        self.data = mujoco.MjData(self.model)

        # control ranges read directly from the model (robust to actuator naming)
        self.n_act = self.model.nu
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

        # free-joint addresses (Skydio X2 root body is a free joint)
        self.qpos_adr = 0   # [x,y,z, qw,qx,qy,qz]
        self.qvel_adr = 0   # [vx,vy,vz, wx,wy,wz]

        # spaces
        obs_dim = 13
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_act,), dtype=np.float32
        )

        self.render_enabled = render
        self._renderer = None
        if render:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

        self._rng = np.random.default_rng(seed)
        self._step_count = 0

    # ── action mapping ───────────────────────────────────────────────────────
    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Map normalized action in [-1, 1] to per-actuator ctrlrange."""
        a = np.clip(action, -1.0, 1.0)
        return self.ctrl_low + 0.5 * (a + 1.0) * (self.ctrl_high - self.ctrl_low)

    # ── observation ──────────────────────────────────────────────────────────
    def _current_target(self) -> np.ndarray:
        if self.task == "track":
            r = float(self.track_cfg.get("radius", 0.5))
            period = float(self.track_cfg.get("period_steps", 200))
            phase = 2.0 * np.pi * (self._step_count / max(period, 1.0))
            return self.target + np.array([r * np.cos(phase), r * np.sin(phase), 0.0])
        return self.target

    def _get_obs(self) -> np.ndarray:
        pos = self.data.qpos[0:3]
        quat = self.data.qpos[3:7]
        linvel = self.data.qvel[0:3]
        angvel = self.data.qvel[3:6]
        pos_err = pos - self._current_target()
        return np.concatenate([pos_err, quat, linvel, angvel]).astype(np.float32)

    # ── reward (dense quadratic) ─────────────────────────────────────────────
    def _reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        pos_err = obs[0:3]
        quat = obs[3:7]
        linvel = obs[7:10]
        angvel = obs[10:13]
        cost = (
            self.rw["w_position"] * float(np.sum(pos_err ** 2))
            + self.rw["w_velocity"] * float(np.sum(linvel ** 2))
            + self.rw["w_tilt"] * tilt_angle(quat) ** 2
            + self.rw["w_ang_vel"] * float(np.sum(angvel ** 2))
            + self.rw["w_action"] * float(np.sum(np.asarray(action) ** 2))
        )
        return float(self.rw["alive_bonus"] - cost)

    # ── termination (failure) ────────────────────────────────────────────────
    def _failed(self, obs: np.ndarray) -> bool:
        pos = obs[0:3] + self._current_target()
        if tilt_angle(obs[3:7]) > self.tilt_limit:
            return True
        if np.linalg.norm(obs[0:3]) > self.position_bound:
            return True
        if pos[2] < self.floor_z:
            return True
        return False

    # ── gym API ──────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0

        c = self.cfg
        # near-hover position
        pos = self.target + self._rng.uniform(
            -c["init_pos_noise"], c["init_pos_noise"], size=3
        )
        quat = small_random_quat(self._rng, c["init_ang_noise"])
        self.data.qpos[0:3] = pos
        self.data.qpos[3:7] = quat
        self.data.qvel[0:3] = self._rng.uniform(
            -c["init_lin_vel_noise"], c["init_lin_vel_noise"], size=3
        )
        self.data.qvel[3:6] = self._rng.uniform(
            -c["init_ang_vel_noise"], c["init_ang_vel_noise"], size=3
        )
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        ctrl = self._scale_action(np.asarray(action, dtype=np.float64))
        self.data.ctrl[:] = ctrl
        # action repeat / frame-skip
        for _ in range(self.action_repeat):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()
        reward = self._reward(obs, action)
        terminated = self._failed(obs)
        truncated = self._step_count >= self.max_episode_steps
        info = {
            "tilt": tilt_angle(obs[3:7]),
            "pos_error": float(np.linalg.norm(obs[0:3])),
            "failure": bool(terminated),
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


# ── PyBullet backend (gym-pybullet-drones) ────────────────────────────────────
class PyBulletDroneHoverEnv(gym.Env):
    """Same hover task as DroneHoverEnv, but on PyBullet via gym-pybullet-drones.

    Deliberately exposes the IDENTICAL 13-dim observation and the SAME dense
    reward / termination as the MuJoCo env, so the ONLY difference between the two
    backends is the physics engine — exactly the cross-simulator comparison.

    Built on the low-level `CtrlAviary` (direct per-rotor RPM control); we read the
    drone state via `_getDroneStateVector` (a stable API across versions) and
    compute our own observation/reward/termination on top.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: dict, seed: int = 0, render: bool = False):
        super().__init__()
        CtrlAviary, DroneModel, Physics = _import_pybullet_drones()
        self.cfg = cfg
        self.target = np.asarray(cfg["target_position"], dtype=np.float64)
        self.action_repeat = int(cfg.get("action_repeat", 1))
        self.max_episode_steps = int(cfg.get("max_episode_steps", 250))
        pyb = cfg.get("pybullet", {})
        self.action_scale = float(pyb.get("action_scale", 0.1))
        self._rng = np.random.default_rng(seed)
        self._render = render
        self._reward_fn = known_reward_fn(cfg)
        self._done_fn = termination_fn(cfg)

        self.base = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=self._init_xyz(),
            physics=Physics.PYB,
            pyb_freq=int(pyb.get("pyb_freq", 240)),
            ctrl_freq=int(pyb.get("ctrl_freq", 60)),
            gui=False,
        )
        self.HOVER_RPM = float(self.base.HOVER_RPM)
        self.MAX_RPM = float(self.base.MAX_RPM)
        self.client = getattr(self.base, "CLIENT", 0)

        self.observation_space = spaces.Box(-np.inf, np.inf, (13,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self._step_count = 0

    def _current_target(self):
        return self.target            # hover target (track not implemented on PyBullet)

    def _init_xyz(self):
        c = self.cfg
        pos = self.target + self._rng.uniform(
            -c["init_pos_noise"], c["init_pos_noise"], size=3)
        return np.array([pos], dtype=np.float64)

    def _state(self):
        return np.asarray(self.base._getDroneStateVector(0))

    def _get_obs(self):
        s = self._state()
        pos = s[0:3]
        quat_xyzw = s[3:7]                       # PyBullet order (x, y, z, w)
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        linvel = s[10:13]
        angvel = s[13:16]
        return np.concatenate([pos - self.target, quat_wxyz, linvel, angvel]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.base.INIT_XYZS = self._init_xyz()
        try:
            self.base.reset(seed=seed)
        except TypeError:
            self.base.reset()
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        rpm = np.clip(self.HOVER_RPM * (1.0 + self.action_scale * a),
                      0.0, self.MAX_RPM).reshape(1, 4)
        for _ in range(self.action_repeat):
            self.base.step(rpm)
        self._step_count += 1
        obs = self._get_obs()
        reward = float(self._reward_fn(obs[None], a[None])[0])
        terminated = bool(self._done_fn(obs[None])[0])
        truncated = self._step_count >= self.max_episode_steps
        info = {
            "tilt": tilt_angle(obs[3:7]),
            "pos_error": float(np.linalg.norm(obs[0:3])),
            "failure": bool(terminated),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        """Headless RGB frame via the PyBullet camera (DIRECT client)."""
        import pybullet as p
        w, h = 640, 480
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.target, distance=2.0, yaw=45, pitch=-30,
            roll=0, upAxisIndex=2, physicsClientId=self.client)
        proj = p.computeProjectionMatrixFOV(
            fov=60, aspect=w / h, nearVal=0.1, farVal=100.0, physicsClientId=self.client)
        _, _, rgb, _, _ = p.getCameraImage(
            w, h, view, proj, physicsClientId=self.client)
        return np.asarray(rgb, dtype=np.uint8)[:, :, :3]

    def close(self):
        try:
            self.base.close()
        except Exception:
            pass


class NoisyObsWrapper(gym.Wrapper):
    """Adds Gaussian sensor noise to observations and/or action noise — used for
    the robustness/noise study (Part 3). Backend-agnostic (works on either env)."""

    def __init__(self, env, obs_noise_std=0.0, action_noise_std=0.0, seed=0):
        super().__init__(env)
        self.obs_noise_std = float(obs_noise_std)
        self.action_noise_std = float(action_noise_std)
        self._rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._noisy(obs), info

    def step(self, action):
        if self.action_noise_std > 0:
            action = np.clip(action + self._rng.normal(0, self.action_noise_std,
                                                       size=np.shape(action)), -1, 1)
        obs, r, terminated, truncated, info = self.env.step(action)
        return self._noisy(obs), r, terminated, truncated, info

    def _noisy(self, obs):
        if self.obs_noise_std > 0:
            obs = obs + self._rng.normal(0, self.obs_noise_std, size=np.shape(obs))
        return obs.astype(np.float32)


def _import_pybullet_drones():
    """Import gym-pybullet-drones pieces, tolerant of module-path differences."""
    from gym_pybullet_drones.utils.enums import DroneModel, Physics
    try:
        from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
    except ImportError:
        from gym_pybullet_drones.envs import CtrlAviary
    return CtrlAviary, DroneModel, Physics


# ── public builders (function-only contract) ──────────────────────────────────
def make_env(cfg: dict, seed: int = 0, render: bool = False):
    """Build the drone env. Returns (env, obs_dim, act_dim).

    `cfg` is the `env:` sub-config. `cfg['backend']` selects the physics engine:
    'pybullet' (gym-pybullet-drones) or 'mujoco' (Skydio X2). Both expose the
    same 13-dim observation and dense reward, so only the physics differs.
    Optional `cfg['noise']` wraps the env with sensor/action noise (Part 3).
    """
    backend = cfg.get("backend", "mujoco")
    if backend == "pybullet":
        env = PyBulletDroneHoverEnv(cfg, seed=seed, render=render)
    else:
        env = DroneHoverEnv(cfg, seed=seed, render=render)

    noise = cfg.get("noise", {})
    if noise.get("enabled", False):
        env = NoisyObsWrapper(
            env, obs_noise_std=noise.get("obs_noise_std", 0.0),
            action_noise_std=noise.get("action_noise_std", 0.0), seed=seed)
    return env, env.observation_space.shape[0], env.action_space.shape[0]


def known_reward_fn(cfg: dict):
    """Return a vectorized reward(obs, action) callable for model-based rollouts.

    The paper's theory assumes a KNOWN reward r(s,a). We expose the analytic
    quadratic reward so all model-based algorithms score imagined transitions
    identically (fairness), instead of learning a separate reward head.

    obs:    (..., 13) array of [pos_err, quat, linvel, angvel]
    action: (..., n_act) array
    returns (...,) reward array.
    """
    rw = cfg["reward"]

    def reward(obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs)
        action = np.asarray(action)
        pos_err = obs[..., 0:3]
        quat = obs[..., 3:7]
        linvel = obs[..., 7:10]
        angvel = obs[..., 10:13]
        # vectorized tilt = arccos(R22); R22 from quat = 1 - 2(x^2 + y^2)
        x, y = quat[..., 1], quat[..., 2]
        norm = np.sum(quat ** 2, axis=-1) + 1e-9
        r22 = np.clip(1.0 - 2.0 * (x ** 2 + y ** 2) / norm, -1.0, 1.0)
        tilt = np.arccos(r22)
        cost = (
            rw["w_position"] * np.sum(pos_err ** 2, axis=-1)
            + rw["w_velocity"] * np.sum(linvel ** 2, axis=-1)
            + rw["w_tilt"] * tilt ** 2
            + rw["w_ang_vel"] * np.sum(angvel ** 2, axis=-1)
            + rw["w_action"] * np.sum(action ** 2, axis=-1)
        )
        return rw["alive_bonus"] - cost

    return reward


def termination_fn(cfg: dict):
    """Return a vectorized done(obs) callable matching DroneHoverEnv._failed.

    Model-based rollouts must respect the same failure envelope as the real env,
    otherwise the ensemble hallucinates flight past a crash.
    """
    tilt_limit = float(cfg.get("tilt_limit", 1.0))
    position_bound = float(cfg.get("position_bound", 1.5))
    floor_z = float(cfg.get("floor_z", 0.05))
    target = np.asarray(cfg["target_position"], dtype=np.float64)

    def done(obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs)
        quat = obs[..., 3:7]
        x, y = quat[..., 1], quat[..., 2]
        norm = np.sum(quat ** 2, axis=-1) + 1e-9
        r22 = np.clip(1.0 - 2.0 * (x ** 2 + y ** 2) / norm, -1.0, 1.0)
        tilt = np.arccos(r22)
        pos_err_norm = np.linalg.norm(obs[..., 0:3], axis=-1)
        world_z = obs[..., 2] + target[2]
        return (tilt > tilt_limit) | (pos_err_norm > position_bound) | (world_z < floor_z)

    return done
