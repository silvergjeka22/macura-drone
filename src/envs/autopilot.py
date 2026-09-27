"""A hand-written AUTOPILOT for the race (not learned): shows the task is solvable and flies the demo videos.

It follows the racing line at `speed` m/s and sinks at most `descent` m/s in the chute, through the same
stabiliser interface as the learning agents (sideways acceleration, climb rate).
"""

import numpy as np

from src.envs.race_course import project


class Autopilot:
    def __init__(self, env, descent=0.55, speed=2.0):
        self.env = env
        self.descent = descent               # max sink speed in the chute (m/s)
        self.speed = speed                   # cruise speed along the course (m/s)
        self.g = float(-env.model.opt.gravity[2])
        self.mass = float(env.model.body_subtreemass[env._core_id])

    def act(self):
        env = self.env
        if env._step_count == 0:             # new flight: the package weight may have changed
            self.mass = float(env.model.body_subtreemass[env._core_id])
        a = self._desired_acceleration(np.array(env.data.qpos[0:3]), np.array(env.data.qvel[0:3]))
        climb = float(env.data.qvel[2]) + (a[2] - self.g) / env.alt_kz
        return np.clip([a[0] / env.att_acc_h, a[1] / env.att_acc_h, climb / env.alt_vz_max, 0.0], -1.0, 1.0).astype(np.float32)

    def _desired_acceleration(self, p, v):
        """Fly along the course direction 0.5 m ahead at `speed` (slower in steep descents), pulled back onto the
        racing line; the steady wind is visible, so it is cancelled."""
        course = self.env.course
        k = int(project(course, p)[0][0])
        direction = course["tangent"][(k + int(round(0.5 / course["params"]["spacing"]))) % len(course["points"])]
        along = self.speed
        if direction[2] < -0.3:
            along = min(along, self.descent / max(-direction[2], 1e-3))
        v_cmd = along * direction + 2.0 * (course["points"][k] - p)
        a = 3.0 * (v_cmd - v)
        a[:2] = np.clip(a[:2], -8.0, 8.0)
        a[2] = float(np.clip(a[2], -6.0, 8.0)) + self.g
        a -= np.asarray(self.env._wind_mean) / self.mass
        return a
