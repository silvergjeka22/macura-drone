"""Hand-written AUTOPILOT for the drone - NOT a learned policy.

Used only to (1) check that a task is physically solvable (can a sensible controller land it under
the same wind?) and (2) render a demo of the environment. The learning algorithms never see it.

Cascaded PD, the classic quadrotor recipe:
    position error -> desired acceleration (+ gravity) -> thrust along the body z-axis and the
    desired body tilt -> attitude PD -> torques -> per-rotor thrusts (X-quad mixing) -> env action.
Cage task: it stays above the cage until it is over the opening, then descends slowly.
Race task: it follows the racing line at `speed` m/s, and sinks at most `descent` m/s in the chute.
"""

from __future__ import annotations

import numpy as np
import mujoco


def _rotmat(quat) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(quat, dtype=np.float64))
    return m.reshape(3, 3)


class Autopilot:
    def __init__(self, env, kp=4.0, kd=3.2, ki=2.0, kz=3.0, kz_pos=1.2, max_acc=4.0, descent=0.55,
                 att_wn=12.0, att_zeta=0.8, yaw_kd=0.004, clearance=0.35, hover_above=0.6, speed=2.0):
        m = env.model
        self.env = env
        self.mass = float(m.body_subtreemass[env._core_id])
        self.g = float(-m.opt.gravity[2])
        inertia = np.asarray(m.body_inertia[env._core_id], dtype=np.float64)
        self.att_kp = inertia * att_wn ** 2                 # per-axis attitude stiffness
        self.att_kd = inertia * 2.0 * att_zeta * att_wn     # per-axis attitude damping
        # mixing matrix from the actual rotor sites / gears: [T, tau_x, tau_y, tau_z] = M @ f
        xy, cyaw = [], []
        for i in range(m.nu):
            site = m.actuator_trnid[i, 0]
            xy.append(np.array(m.site_pos[site][:2]))
            cyaw.append(float(m.actuator_gear[i, 5]))
        xy = np.array(xy)
        M = np.vstack([np.ones(m.nu), xy[:, 1], -xy[:, 0], np.array(cyaw)])
        self.Minv = np.linalg.inv(M)
        self.kp, self.kd, self.ki, self.kz, self.kz_pos = kp, kd, ki, kz, kz_pos
        self.dt = float(m.opt.timestep) * env.action_repeat
        self._i = np.zeros(2)                                # integral of the xy error (wind trim)
        self.max_acc, self.descent, self.yaw_kd = max_acc, descent, yaw_kd
        self.clearance, self.hover_above = clearance, hover_above
        self.speed = speed                                   # race: cruise speed along the course (m/s)

    def act(self) -> np.ndarray:
        env, d = self.env, self.env.data
        if env._step_count == 0:                             # new episode -> reset the wind trim
            self._i[:] = 0.0
            self.mass = float(env.model.body_subtreemass[env._core_id])   # knows the package weight
        p, v, w = np.array(d.qpos[0:3]), np.array(d.qvel[0:3]), np.array(d.qvel[3:6])
        Rm = _rotmat(d.qpos[3:7])
        if getattr(env, "race", False):
            return self._attitude(self._race_accel(p, v), Rm, w)
        tgt = env._target
        dxy = tgt[:2] - p[:2]
        r = float(np.linalg.norm(dxy))

        # altitude goal (cage task):
        #   centred over the opening          -> descend to the pad
        #   above the cage but off-centre     -> stay above the cage while moving over the opening
        #   outside the cage and below its top-> climb above the cage first
        #   INSIDE the cage but off-centre    -> hold height and re-centre (never climb back out)
        z_goal = tgt[2]
        if getattr(env, "cage", False) and r > env.cage_radius - self.clearance:
            above = p[2] >= env.cage_height
            if above:
                z_goal = max(p[2], env.cage_height + self.hover_above)
            elif r >= env.cage_radius:
                z_goal = env.cage_height + self.hover_above
            else:
                z_goal = p[2]
        vz_des = float(np.clip(self.kz_pos * (z_goal - p[2]), -self.descent, 0.8))

        a = np.zeros(3)
        self._i = np.clip(self._i + dxy * self.dt, -1.0, 1.0)   # integral: leans into steady wind
        a[:2] = np.clip(self.kp * dxy - self.kd * v[:2] + self.ki * self._i, -self.max_acc, self.max_acc)
        a[2] = self.kz * (vz_des - v[2]) + self.g

        return self._attitude(a, Rm, w)

    def _race_accel(self, p, v) -> np.ndarray:
        """Race: fly along the course tangent (taken 0.5 m ahead, to turn in time) at `speed`, capped so the
        sink speed in the chute stays <= `descent`, pulled back onto the racing line; the steady wind is
        visible, so it is cancelled by feed-forward."""
        from src.envs.race_course import project
        c = self.env.course
        k = int(project(c, p)[0][0])
        n = len(c["points"])
        t = c["tangent"][(k + int(round(0.5 / c["params"]["spacing"]))) % n]
        along = self.speed
        if t[2] < -0.3:                                    # steep descent: limit the sink speed
            along = min(along, self.descent / max(-t[2], 1e-3))
        v_cmd = along * t + 2.0 * (c["points"][k] - p)
        a = 3.0 * (v_cmd - v)
        a[:2] = np.clip(a[:2], -8.0, 8.0)
        a[2] = float(np.clip(a[2], -6.0, 8.0)) + self.g
        a -= np.asarray(getattr(self.env, "_wind_mean", np.zeros(3))) / self.mass
        return a

    def _attitude(self, a, Rm, w) -> np.ndarray:
        """Desired acceleration -> collective thrust + attitude torques -> per-rotor action."""
        env = self.env
        zb = Rm[:, 2]
        thrust = self.mass * float(np.dot(a, zb))
        z_des = a / (np.linalg.norm(a) + 1e-9)
        e_body = Rm.T @ np.cross(zb, z_des)                 # tilt error in the body frame
        tau = self.att_kp * e_body - self.att_kd * w
        tau[2] = -self.yaw_kd * w[2]                         # just damp yaw
        f = self.Minv @ np.array([thrust, tau[0], tau[1], tau[2]])
        return np.clip(f / env._hover - 1.0, -1.0, 1.0).astype(np.float32)
