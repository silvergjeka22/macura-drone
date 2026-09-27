"""Real-time 3-D simulator: the trained drones race together in MuJoCo's native viewer.

Every pilot flies its own copy of the environment (the exact training physics, same scenario seed, so the
same package, wind and gusts). A separate display scene holds one drone per pilot as a mocap body, colour
coded, and copies each pilot's pose into it every control step (50 Hz). MACURA's panel line shows its own
saved world model's verdict on every step, exactly its trust rule (GJS < kappa).

Keys in the viewer window: SPACE pause/resume, N next scenario, R restart the scenario.
"""

from __future__ import annotations

import copy
import time
import xml.etree.ElementTree as ET

import numpy as np

COLORS = {"MACURA": (0.12, 0.47, 0.71), "MBPO": (0.84, 0.15, 0.16), "M2AC": (0.17, 0.63, 0.17),
          "SAC": (0.50, 0.50, 0.50)}
EXTRA_COLORS = [(0.58, 0.40, 0.74), (0.55, 0.34, 0.29), (0.89, 0.47, 0.76), (0.74, 0.74, 0.13)]
_SCALE = 2.0          # drones drawn 2x larger so they stay visible over the whole course
_KEY_SPACE, _KEY_N, _KEY_R = 32, 78, 82


class Pilot:
    """One drone: its own environment, a policy (SB3 .zip or the hand-written autopilot) and an optional
    trust check {"ensemble": path, "kappa": float}."""

    def __init__(self, label, cfg, policy, trust=None, device="cpu"):
        from src.envs import drone_env
        self.label, self.cfg = label, cfg
        self.env, obs_dim, act_dim = drone_env.make_env(cfg["env"], seed=0)
        self.agent, self.auto_args = None, None
        if str(policy).startswith("autopilot"):
            parts = str(policy).split(":")
            self.auto_args = (float(parts[1]) if len(parts) > 1 else 0.8, float(parts[2]) if len(parts) > 2 else 1.5)
        else:
            from stable_baselines3 import SAC
            self.agent = SAC.load(policy, device=device)
        self.kappa, self.check = None, None
        if trust and trust.get("ensemble") and trust.get("kappa") is not None:
            from src.viz.video import _trust_check
            self.kappa = float(trust["kappa"])
            self.check = _trust_check(cfg, obs_dim, act_dim, trust, device)

    def reset(self, seed):
        from src.envs.autopilot import Autopilot
        self.obs, _ = self.env.reset(seed=int(seed))
        self.auto = Autopilot(self.env, descent=self.auto_args[0], speed=self.auto_args[1]) if self.auto_args else None
        self.done, self.crashed, self.laps, self.t = False, False, 0.0, 0
        self.trusted, self.n_checked, self.ok = 0, 0, None

    def step(self, max_steps):
        if self.done:
            return
        act = self.auto.act() if self.auto else self.agent.predict(np.asarray(self.obs, np.float32), deterministic=True)[0]
        if self.check:
            self.ok = self.check(self.obs, act) < self.kappa
            self.trusted += self.ok; self.n_checked += 1
        self.obs, _, term, _, info = self.env.step(act)
        self.t += 1
        self.laps = float(info.get("laps", 0.0))
        self.crashed = bool(term)
        self.done = term or self.t >= max_steps

    def status(self):
        e, o = self.env, self.obs
        state = "CRASHED" if self.crashed else ("done" if self.done else "flying")
        line = (f"laps {self.laps:4.2f}  speed {np.linalg.norm(o[8:11]):3.1f} m/s  sink {max(-o[10], 0):3.1f} m/s  "
                f"lift {100 * e.thrust_eff:3.0f}%  {state}")
        if self.check and self.n_checked:
            verdict = "agree -> trusted" if self.ok else "DISAGREE -> would stop imagining"
            line += f"\n      world model: {verdict}  ({100 * self.trusted / self.n_checked:.0f}% of steps trusted)"
        return line


def _scaled(geom, s):
    g = copy.deepcopy(geom)
    for key in ("size", "pos", "fromto"):
        if g.get(key):
            g.set(key, " ".join(f"{float(v) * s:.5f}" for v in g.get(key).split()))
    return g


def display_scene(env_xml, labels):
    """The env's scene (course, gates, pillars, windsock) without the physical drone and its motors, plus one
    mocap drone per pilot in the pilot's colour."""
    import mujoco
    root = ET.fromstring(env_xml)
    wb = root.find("worldbody")
    core = next(b for b in wb.findall("body") if b.get("name") == "core")
    wb.remove(core)
    act = root.find("actuator")
    if act is not None:
        root.remove(act)
    for i, label in enumerate(labels):
        rgb = COLORS.get(label.split()[0].upper(), EXTRA_COLORS[i % len(EXTRA_COLORS)])
        b = ET.SubElement(wb, "body", {"name": f"drone{i}", "mocap": "true", "pos": "0 0 -5"})
        for g in core.findall("geom"):
            g2 = _scaled(g, _SCALE)
            g2.set("name", f"{g.get('name')}_{i}")
            g2.set("contype", "0"); g2.set("conaffinity", "0"); g2.set("mass", "0")
            if not (g.get("name") or "").startswith("rotor"):
                g2.set("rgba", f"{rgb[0]} {rgb[1]} {rgb[2]} 1")
            b.append(g2)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model, mujoco.MjData(model)


def run(cfg, pilots, scenarios=(1000, 1001, 1002), seconds=None, speed=1.0, trust=None, device="cpu",
        trail=True, follow=None):
    """Fly `pilots` ({label: .zip | "autopilot:<sink>:<speed>"}) together in a live viewer, one scenario after
    another (loops). `seconds`: flight length (default = the 10 s training episode)."""
    import mujoco
    import mujoco.viewer
    from src.viz import scenery

    trust = trust or {}
    fleet = [Pilot(k, cfg, p, trust.get(k), device) for k, p in pilots.items()]
    labels = [p.label for p in fleet]
    model, data = display_scene(fleet[0].env.mjcf_xml, labels)
    ids = scenery.scenery_ids(model)
    mocap = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b): int(model.body_mocapid[b])
             for b in range(model.nbody) if model.body_mocapid[b] >= 0}
    env0 = fleet[0].env
    env_mocap = {mujoco.mj_id2name(env0.model, mujoco.mjtObj.mjOBJ_BODY, b): int(env0.model.body_mocapid[b])
                 for b in range(env0.model.nbody) if env0.model.body_mocapid[b] >= 0}
    pkg_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"package_{i}") for i in range(len(fleet))]
    dt = float(env0.model.opt.timestep) * env0.action_repeat
    max_steps = int(round(seconds / dt)) if seconds else env0.max_episode_steps
    rgb = [np.array((*COLORS.get(k.split()[0].upper(), EXTRA_COLORS[i % 4]), 1.0), np.float32)
           for i, k in enumerate(labels)]
    keys = {"pause": False, "next": False, "restart": False}

    def on_key(code):
        if code == _KEY_SPACE:
            keys["pause"] = not keys["pause"]
        elif code == _KEY_N:
            keys["next"] = True
        elif code == _KEY_R:
            keys["restart"] = True

    def place(trails):
        for i, p in enumerate(fleet):
            data.mocap_pos[mocap[f"drone{i}"]] = p.env.data.qpos[0:3]
            data.mocap_quat[mocap[f"drone{i}"]] = p.env.data.qpos[3:7]
            if trail and not p.done:
                trails[i].append(np.array(p.env.data.qpos[0:3]))
        for name, j in env_mocap.items():
            if name in mocap and not name.startswith("drone"):
                data.mocap_pos[mocap[name]] = env0.data.mocap_pos[j]
        scenery.decorate(model, data, ids, env0._wind + env0._wind_mean, 1.0)
        mujoco.mj_forward(model, data)

    def draw_trails(scn, trails):
        scn.ngeom = 0
        for i, tr in enumerate(trails):
            for q in tr[-600::3]:
                if scn.ngeom >= scn.maxgeom:
                    break
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.025, 0, 0]),
                                    q, np.eye(3).flatten(), rgb[i])
                scn.ngeom += 1

    def show_text(viewer, scenario):          # set_texts takes the viewer lock itself: call it outside
        if not hasattr(viewer, "set_texts"):
            return
        head = f"scenario {scenario}   t = {max(p.t for p in fleet) * dt:4.1f} s{'   PAUSED' if keys['pause'] else ''}"
        body = "\n".join(f"{p.label:9s} {p.status()}" for p in fleet)
        viewer.set_texts([(None, mujoco.mjtGridPos.mjGRID_TOPLEFT, head + "\n" + body, None),
                          (None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, "SPACE pause   N next scenario   R restart", None)])

    def set_camera(cam):
        cam.lookat[:] = [0.3, 0.6, 2.0]
        cam.distance, cam.elevation, cam.azimuth = 10.5, -28.0, 115.0
        if follow is not None and follow in labels:
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"drone{labels.index(follow)}")
            cam.distance = 3.5

    try:                                          # side panels hidden (Tab / Shift+Tab bring them back)
        handle = mujoco.viewer.launch_passive(model, data, key_callback=on_key, show_left_ui=False,
                                              show_right_ui=False)
    except TypeError:                             # older mujoco
        handle = mujoco.viewer.launch_passive(model, data, key_callback=on_key)
    with handle as viewer:
        frame = 0
        k = 0
        while viewer.is_running():
            scenario = scenarios[k % len(scenarios)]
            for p in fleet:
                p.reset(scenario)
            for i, p in enumerate(fleet):
                if pkg_ids[i] >= 0 and p.env._pkg >= 0:
                    model.geom_size[pkg_ids[i]] = p.env.model.geom_size[p.env._pkg] * _SCALE
                    model.geom_pos[pkg_ids[i]] = p.env.model.geom_pos[p.env._pkg] * _SCALE
            trails = [[] for _ in fleet]
            keys["next"] = keys["restart"] = False
            print(f"scenario {scenario}: " + ", ".join(labels))
            end_wait = None
            while viewer.is_running() and not keys["next"] and not keys["restart"]:
                tick = time.perf_counter()
                if not keys["pause"]:
                    for p in fleet:
                        p.step(max_steps)
                with viewer.lock():
                    place(trails)
                    draw_trails(viewer.user_scn, trails)
                    if frame == 2:                # after the viewer has loaded the model (it resets the camera)
                        set_camera(viewer.cam)
                show_text(viewer, scenario)
                viewer.sync()
                frame += 1
                if all(p.done for p in fleet):
                    end_wait = end_wait or time.perf_counter()
                    if time.perf_counter() - end_wait > 2.5:
                        break
                time.sleep(max(0.0, dt / speed - (time.perf_counter() - tick)))
            for p in fleet:
                print(f"  {p.label:9s} " + p.status().replace("\n", "\n  "))
            if not keys["restart"]:
                k += 1
    for p in fleet:
        p.env.close()
