"""The race3 drone task in MuJoCo: laps around a 3-D course with wind, gusts, a package and a chute.

Observation (27) and action (4: sideways acceleration x, y, climb rate, yaw rate in [-1, 1]).
Reward and crashes depend only on (obs, action), so real and imagined steps are scored the same way.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from src.envs.race_course import build_course, project

ASSET = os.path.join(os.path.dirname(__file__), "assets", "drone.xml")

POS = slice(0, 3)          # observation columns
HEIGHT = 3
QX, QY = 5, 6
VEL = slice(8, 11)
VZ = 10
SPIN = slice(11, 14)
PILLARS = 14
WINDSOCK_AT = (0.9, 0.3)


# reward and crashes: shared by real and imagined transitions
def race_reward(obs, act, rw, course):
    """Speed along the course near the racing line, minus distance outside the tube, plus small terms."""
    obs = np.atleast_2d(np.asarray(obs, dtype=np.float64))
    act = np.atleast_2d(np.asarray(act, dtype=np.float64))
    k, dist = project(course, -obs[:, POS])
    progress = np.sum(obs[:, VEL] * course["tangent"][k], axis=-1)
    outside = np.maximum(dist - rw["race_tube"], 0.0)
    near = np.exp(-(outside / rw["race_tube_soft"]) ** 2)
    upright = 1.0 - 2.0 * (obs[:, QX] ** 2 + obs[:, QY] ** 2)
    spin = np.linalg.norm(obs[:, SPIN], axis=-1)
    return (rw["w_prog"] * progress * near - rw["w_track"] * outside + rw["w_level"] * upright
            - rw["w_spin"] * spin - rw["w_ctrl"] * np.sum(act ** 2, axis=-1))


def crashed(obs, cfg, course):
    """Flipped, too far away, hit a pillar, more than race_max_off from the course, or below the floor."""
    obs = np.atleast_2d(np.asarray(obs))
    upright = 1.0 - 2.0 * (obs[:, QX] ** 2 + obs[:, QY] ** 2)
    far = np.linalg.norm(obs[:, POS], axis=-1) > cfg["max_dist"]
    pillars = obs[:, PILLARS:PILLARS + 4].reshape(len(obs), 2, 2)
    hit = np.linalg.norm(pillars, axis=-1).min(axis=-1) < cfg["obstacle_radius"]
    off = project(course, -obs[:, POS])[1] > cfg["race_max_off"]
    return (upright < 0.0) | far | hit | off | (obs[:, HEIGHT] < cfg["race_floor"])


def known_reward_fn(cfg):
    course = build_course(cfg)
    return lambda obs, act: race_reward(obs, act, cfg["reward"], course) - cfg["reward"]["w_crash"] * crashed(obs, cfg, course)


def termination_fn(cfg):
    course = build_course(cfg)
    return lambda obs: crashed(obs, cfg, course)


# lift loss 
def _smooth01(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def lift_factor(vel, quat, thrust_ratio, loss, v_on, v_full, v_escape, up_on, thrust_on):
    """Share of thrust left: drops (up to `loss`) when the upright drone sinks faster than v_on."""
    w, x, y, z = (float(q) for q in quat)
    axis = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    v = np.asarray(vel, dtype=float)
    axial = float(v @ axis)
    across = float(np.linalg.norm(v - axial * axis))
    sinking = _smooth01((-axial - v_on) / max(v_full - v_on, 1e-6))
    upright = _smooth01((axis[2] - up_on) / 0.10)
    pushing = _smooth01((thrust_ratio - thrust_on) / 0.3)
    return 1.0 - loss * sinking * upright * pushing * float(np.exp(-(across / max(v_escape, 1e-6)) ** 2))


def _quat_to_mat(quat):
    w, x, y, z = (float(q) for q in quat)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# scene decoration (visual only: no mass, no collisions)
def _add_course(root, course):
    """Gate rings, racing-line dots (red = chute), a see-through chute column, red pillars."""
    world = root.find("worldbody")
    for gi, (centre, t) in enumerate(course["gates"]):
        t = t / (np.linalg.norm(t) + 1e-9)
        e1 = np.cross(t, [0.0, 0.0, 1.0])
        e1 = e1 / (np.linalg.norm(e1) + 1e-9) if np.linalg.norm(e1) > 1e-6 else np.array([1.0, 0.0, 0.0])
        e2 = np.cross(t, e1)
        ring = [centre + 0.45 * (np.cos(a) * e1 + np.sin(a) * e2) for a in np.linspace(0, 2 * np.pi, 17)]
        for i in range(16):
            a, b = ring[i], ring[i + 1]
            ET.SubElement(world, "geom", {
                "name": f"gate{gi}_{i}", "type": "capsule", "size": "0.025",
                "fromto": f"{a[0]:.4f} {a[1]:.4f} {a[2]:.4f} {b[0]:.4f} {b[1]:.4f} {b[2]:.4f}",
                "contype": "0", "conaffinity": "0", "rgba": "1.0 0.55 0.10 1" if i % 2 else "0.95 0.95 0.95 1"})
    points, chute = course["points"], course["chute"]
    for i in range(0, len(points), 5):
        ET.SubElement(world, "geom", {
            "name": f"course{i}", "type": "sphere", "size": "0.035",
            "pos": f"{points[i][0]:.4f} {points[i][1]:.4f} {points[i][2]:.4f}", "contype": "0", "conaffinity": "0",
            "rgba": "0.95 0.25 0.15 1" if chute[i] else "0.80 0.90 1.0 0.9"})
    top, bottom = points[np.flatnonzero(chute)[0]], points[np.flatnonzero(chute)[-1]]
    ET.SubElement(world, "geom", {
        "name": "chute", "type": "cylinder", "size": "0.35",
        "fromto": f"{top[0]:.4f} {top[1]:.4f} {top[2]:.4f} {bottom[0]:.4f} {bottom[1]:.4f} {bottom[2]:.4f}",
        "contype": "0", "conaffinity": "0", "rgba": "0.95 0.20 0.10 0.18"})
    for g in root.iter("geom"):
        if (g.get("name") or "").startswith("obs"):
            g.set("rgba", "0.80 0.30 0.20 1")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", {"ambient": "0.35 0.35 0.38", "diffuse": "0.55 0.55 0.55",
                                        "specular": "0.1 0.1 0.1"})


def _add_windsock(root):
    """Pole + striped sock as mocap bodies; src/viz/scenery.py points the sock along the wind."""
    world = root.find("worldbody")
    pole = ET.SubElement(world, "body", {"name": "windsock_pole", "mocap": "true", "pos": "0 0 0"})
    ET.SubElement(pole, "geom", {"name": "windsock_pole_geom", "type": "cylinder", "size": "0.012 0.5",
                                 "pos": "0 0 0.5", "mass": "0", "contype": "0", "conaffinity": "0",
                                 "rgba": "0.75 0.75 0.78 1"})
    sock = ET.SubElement(world, "body", {"name": "windsock", "mocap": "true", "pos": "0 0 1"})
    for i, (a, b, r, c) in enumerate(((0.0, 0.1, 0.040, "0.95 0.45 0.10 1"),
                                      (0.1, 0.2, 0.034, "0.95 0.95 0.95 1"),
                                      (0.2, 0.3, 0.028, "0.95 0.45 0.10 1"))):
        ET.SubElement(sock, "geom", {"name": f"windsock_{i}", "type": "capsule", "size": f"{r}",
                                     "fromto": f"{a} 0 0 {b} 0 0", "mass": "0", "contype": "0",
                                     "conaffinity": "0", "rgba": c})


def _build_scene(course):
    """The drone asset + 2 pillar markers, the package, a windsock and the course."""
    root = ET.parse(ASSET).getroot()
    world = root.find("worldbody")
    for body in list(world.findall("body")):
        if body.get("name") in ("obs2", "obs3"):
            world.remove(body)
    core = next(b for b in root.iter("body") if b.get("name") == "core")
    ET.SubElement(core, "geom", {"name": "package", "type": "box", "size": "0.04 0.04 0.024", "pos": "0 0 -0.054",
                                 "mass": "0", "contype": "0", "conaffinity": "0", "rgba": "0.72 0.52 0.30 1"})
    _add_windsock(root)
    _add_course(root, course)
    return ET.tostring(root, encoding="unicode")


class DroneRaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg, seed=0, render=False):
        super().__init__()
        self.cfg = cfg
        self.rw = cfg["reward"]
        self.action_repeat = cfg["action_repeat"]
        self.max_episode_steps = cfg["max_episode_steps"]
        self.thrust_gain = cfg["thrust_gain"]
        self.payload_max = cfg["payload_max"]
        self.payload = 0.0
        self.thrust_eff = 1.0            # the last step's share of lift
        self.race = True
        self.ctrl_mode = "althold"
        self.course = build_course(cfg)
        self.pillars = np.array(cfg["race_pillars"], dtype=float)
        self._ahead = int(round(cfg["race_obs_ahead"] / self.course["params"]["spacing"]))

        self.mjcf_xml = _build_scene(self.course)
        self.model = mujoco.MjModel.from_xml_string(self.mjcf_xml)
        self.data = mujoco.MjData(self.model)
        self.n_act = self.model.nu
        self._core_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "core")
        self._g = float(-self.model.opt.gravity[2])
        self._hover = float(self.model.body_subtreemass[self._core_id]) * self._g / self.n_act   # per rotor, empty
        self._ctrl_hi = float(self.model.actuator_ctrlrange[0][1])
        self._mass0 = float(self.model.body_mass[self._core_id])
        self._pkg = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "package")
        self._mocap = {}
        for name in ("pad", "windsock_pole", "windsock", "obs0", "obs1"):
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self._mocap[name] = int(self.model.body_mocapid[body])
        self._init_stabiliser()

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(27,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_act,), dtype=np.float32)
        self._renderer = mujoco.Renderer(self.model, height=480, width=640) if render else None
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._target = np.zeros(3)
        self._wind = np.zeros(3)
        self._wind_mean = np.zeros(3)
        self._race_k = 0
        self.race_progress = 0.0

    # observation
    def _get_obs(self):
        pos, quat, vel = self.data.qpos[0:3], self.data.qpos[3:7], self.data.qvel
        k = int(project(self.course, pos)[0][0])
        ahead = self.course["tangent"][(k + self._ahead) % len(self.course["points"])]
        obs = [*(self._target - pos), pos[2], *quat, *vel[0:6]]
        for px, py in self.pillars:
            obs += [px - pos[0], py - pos[1]]
        obs += [self.payload / self._mass0, self._wind_mean[0], self._wind_mean[1]]
        obs += [*(self.course["points"][k] - pos), *ahead]
        return np.array(obs, dtype=np.float32)

    # gym API
    def reset(self, *, seed=None, options=None):
        """Random start on the course, package and steady wind (random draws kept in the original order)."""
        rng = self._rng = np.random.default_rng(seed) if seed is not None else self._rng
        cfg, noise = self.cfg, self.cfg["init_noise"]
        self.payload = float(options["payload"]) if options and "payload" in options else float(rng.uniform(0.0, self.payload_max))
        self.model.body_mass[self._core_id] = self._mass0 + self.payload
        mujoco.mj_setConst(self.model, self.data)
        size = 0.015 + 0.025 * self.payload / self.payload_max
        self.model.geom_size[self._pkg] = [size, size, 0.6 * size]
        self.model.geom_pos[self._pkg] = [0.0, 0.0, -0.03 - 0.6 * size]
        self.thrust_eff = 1.0
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._wind = np.zeros(3)

        rng.uniform(-noise, noise, size=3)                                       # old start position (unused)
        rng.uniform(-noise, noise)
        tilt, axis = rng.uniform(0.0, cfg["init_tilt"]), rng.normal(size=3)
        axis /= np.linalg.norm(axis) + 1e-9
        self.data.qpos[3:7] = [np.cos(tilt / 2.0), *(np.sin(tilt / 2.0) * axis)]
        vel = rng.uniform(-noise, noise, size=self.model.nv)
        vel[3:6] = rng.uniform(-cfg["init_spin"], cfg["init_spin"], size=3)
        self.data.qvel[:] = vel
        for _ in range(20):                                                      # old landing pad (unused)
            if np.linalg.norm(rng.uniform(-1.8, 1.8, size=2)) >= 1.7:
                break

        self._target = np.zeros(3)
        start = int(rng.integers(len(self.course["points"])))
        self.data.qpos[0:3] = self.course["points"][start] + rng.uniform(-noise, noise, size=3)
        self._race_k, self.race_progress = start, 0.0
        angle, strength = rng.uniform(0.0, 2.0 * np.pi), rng.uniform(0.0, cfg["wind_mean_max"])
        self._wind_mean = np.array([strength * np.cos(angle), strength * np.sin(angle), 0.0])
        self._place_markers()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {"target": self._target.copy()}

    def _place_markers(self):
        self.data.mocap_pos[self._mocap["pad"]] = [0.0, 0.0, -5.0]
        self.data.mocap_pos[self._mocap["windsock_pole"]] = [*WINDSOCK_AT, 0.0]
        self.data.mocap_pos[self._mocap["windsock"]] = [*WINDSOCK_AT, 1.0]
        for i, (px, py) in enumerate(self.pillars):
            self.data.mocap_pos[self._mocap[f"obs{i}"]] = [px, py, 1.5]

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        rotor = self.attitude_to_rotors(action)
        cfg = self.cfg

        thrust = self._hover * (1.0 + rotor * self.thrust_gain)
        thrust = thrust * (1.0 + self._rng.normal(0.0, cfg["actuator_noise"], size=self.n_act))
        self.thrust_eff = lift_factor(self.data.qvel[0:3], self.data.qpos[3:7],
                                      float(np.mean(1.0 + rotor * self.thrust_gain)), cfg["vrs_loss"], cfg["vrs_speed"],
                                      cfg["vrs_full"], cfg["vrs_escape"], cfg["vrs_upright"], cfg["vrs_thrust"])
        self.data.ctrl[:] = np.clip(thrust * self.thrust_eff, 0.0, self._ctrl_hi)

        corr = cfg["wind_correlation"]                                            # gusts: an OU process
        self._wind = corr * self._wind + np.sqrt(1.0 - corr ** 2) * self._rng.normal(0.0, cfg["wind_force"], size=3)
        self.data.xfrc_applied[self._core_id, 0:3] = self._wind + self._wind_mean
        for _ in range(self.action_repeat):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()
        crash = bool(crashed(obs, cfg, self.course)[0])
        reward = float(race_reward(obs, action, self.rw, self.course)[0]) - (self.rw["w_crash"] if crash else 0.0)

        k, dist = project(self.course, -obs[POS])
        k, dist = int(k[0]), float(dist[0])
        length = self.course["length"]
        if dist < 1.0:                                                            # lap progress along the course
            self.race_progress += (self.course["s"][k] - self.course["s"][self._race_k] + length / 2.0) % length - length / 2.0
        self._race_k = k

        truncated = self._step_count >= self.max_episode_steps
        info = {"failure": crash, "reached": self.race_progress >= length, "thrust_eff": self.thrust_eff,
                "payload": self.payload, "laps": self.race_progress / length}
        return obs, reward, crash, truncated, info

    # on-board stabiliser
    def _init_stabiliser(self):
        m = self.model
        xy = np.array([m.site_pos[m.actuator_trnid[i, 0]][:2] for i in range(m.nu)])
        yaw = np.array([float(m.actuator_gear[i, 5]) for i in range(m.nu)])
        self._mix_inv = np.linalg.inv(np.vstack([np.ones(m.nu), xy[:, 1], -xy[:, 0], yaw]))
        inertia = np.asarray(m.body_inertia[self._core_id], dtype=np.float64)
        self._att_kp, self._att_kd = inertia * 12.0 ** 2, inertia * 2.0 * 0.8 * 12.0
        self.att_max_tilt = np.radians(self.cfg["att_max_tilt_deg"])
        self.att_acc_h = self._g * float(np.tan(self.att_max_tilt))           # sideways m/s^2 at |action| = 1
        self.alt_vz_max, self.alt_kz = self.cfg["alt_vz_max"], self.cfg["alt_kz"]

    def attitude_to_rotors(self, action):
        """[accel x, accel y, climb rate, yaw rate] in [-1, 1] -> rotor commands in [-1, 1]."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        mass = float(self.model.body_subtreemass[self._core_id])
        az = float(np.clip(self.alt_kz * (a[2] * self.alt_vz_max - float(self.data.qvel[2])) / self._g, -1.0, 1.0))
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
        tau[2] = 0.004 * (a[3] * 2.0 - w[2])
        f_col = self._mix_inv[:, 0] * thrust
        f_tau = self._mix_inv[:, 1:] @ tau
        lo = max(self._hover * (1.0 - self.thrust_gain), 0.0)
        hi = min(self._hover * (1.0 + self.thrust_gain), self._ctrl_hi)
        span = float(f_tau.max() - f_tau.min())
        if span > hi - lo:
            f_tau = f_tau * (hi - lo) / span
        f = f_col + f_tau
        f = f + max(0.0, lo - float(f.min()))
        f = f - max(0.0, float(f.max()) - hi)
        return np.clip((f / self._hover - 1.0) / self.thrust_gain, -1.0, 1.0)

    def render(self):
        self._renderer.update_scene(self.data, camera=-1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_env(cfg, seed=0, render=False):
    """Returns (env, obs_dim, act_dim)."""
    env = DroneRaceEnv(cfg, seed=seed, render=render)
    return env, env.observation_space.shape[0], env.action_space.shape[0]
