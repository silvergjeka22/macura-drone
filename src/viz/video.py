"""Race videos: several pilots on the same scenarios, tiled, with a live trust panel for MACURA.

The flights run here (no graphics); the frames are drawn in a mujoco-only subprocess, so software
rendering never shares a process with torch (a known segfault on headless GPU boxes).

    record_race(cfg, pilots, save_path, seeds, trust=..., notes=...)
    pilots = {label: policy .zip | "autopilot:<max sink m/s>:<speed m/s>" (hand-written, not learned)}
    trust  = {label: {"ensemble": <best_ensemble.pt>, "kappa": float}}: at every step the pilot's own saved
             world model is asked whether its members agree about (obs, action), exactly MACURA's rule
    notes  = {label: one line shown instead of the trust panel}
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NOTES = {"MBPO": "MBPO: imagines 10 steps ahead, no trust check",
         "M2AC": "M2AC: keeps its 25% most certain imagined steps",
         "SAC": "SAC: no world model, learns from real flights only"}
WIDE_CAMERA = {"lookat": [0.3, 0.6, 2.0], "distance": 10.5, "elevation": -28.0, "azimuth": 115.0}


def record_race(cfg, pilots, save_path, seeds=(1000, 1001), trust=None, notes=None, cols=None,
                size=(480, 360), fps=25, frame_step=2, max_steps=None, device="cpu", camera=None):
    """`frame_step`: draw every n-th control step (fps = 50 / frame_step plays in real time).
    `max_steps`: fly longer than the 10 s episode (a demo, not a score). Returns save_path or None."""
    trust, notes = trust or {}, {**NOTES, **(notes or {})}
    flights, xml = {}, None
    for label, pilot in pilots.items():
        try:
            flights[label], xml = _fly(cfg, pilot, seeds, device, max_steps, trust.get(label))
        except Exception as e:
            print(f"flight failed for {label}: {e}")
            return None
    job = {"flights": flights, "labels": list(pilots), "fps": int(fps), "step": int(frame_step),
           "camera": camera or WIDE_CAMERA, "size": tuple(size), "cols": int(cols or (len(pilots) if len(pilots) <= 3 else 2)),
           "notes": {k: notes.get(k, "") for k in pilots}, "kappa": {k: v.get("kappa") for k, v in trust.items()}}
    data_file, asset = tempfile.mktemp(suffix=".pkl"), tempfile.mktemp(suffix=".xml")
    with open(data_file, "wb") as f:
        pickle.dump(job, f)
    with open(asset, "w") as f:
        f.write(xml)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    backend = "osmesa" if os.path.isdir("/kaggle") else os.environ.get("MUJOCO_GL_VIDEO", "glfw")
    child = _CHILD % dict(backend=backend, root=ROOT, data=data_file, asset=asset, out=save_path)
    try:
        res = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=3600)
    except Exception as e:
        print("race video error:", e)
        return None
    if res.returncode != 0 or not os.path.exists(save_path):
        print("race video render failed:", (res.stderr or "")[-800:])
        return None
    for label in pilots:
        eps = flights[label]
        laps = sum(float(e["laps"][-1]) for e in eps)
        line = f"  {label}: {laps:.1f} laps in {len(eps)} flights, crashed {sum(e['outcome'] == 'crash' for e in eps)}"
        if "trusted" in eps[0]:
            line += f", model trusted {100 * np.mean(np.concatenate([e['trusted'] for e in eps])):.0f}% of the steps"
        print(line)
    return save_path


def _fly(cfg, pilot, seeds, device, max_steps, trust):
    from src.envs import drone_env
    from src.envs.autopilot import Autopilot
    env, obs_dim, act_dim = drone_env.make_env(cfg["env"], seed=0)
    agent, descent, speed = None, 0.55, 2.0
    if str(pilot).startswith("autopilot"):
        parts = str(pilot).split(":")
        descent = float(parts[1]) if len(parts) > 1 else descent
        speed = float(parts[2]) if len(parts) > 2 else speed
    else:
        from src.algorithms.sac import load_agent
        agent = load_agent(pilot, obs_dim, act_dim, cfg["sac"], device)
    check = _trust_check(cfg, obs_dim, act_dim, trust, device) if trust else None
    eps = []
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        auto = Autopilot(env, descent=descent, speed=speed) if agent is None else None
        fr = {k: [] for k in ("qpos", "mocap", "wind", "eff", "vz", "speed", "laps", "gjs", "trusted")}
        outcome = "timeout"
        for t in range(int(max_steps or env.max_episode_steps)):
            act = auto.act() if agent is None else agent.predict(np.asarray(obs, np.float32), deterministic=True)[0]
            if check:
                g = check(obs, act)
                fr["gjs"].append(g); fr["trusted"].append(g < trust["kappa"])
            obs, _, term, _, info = env.step(act)
            fr["qpos"].append(np.array(env.data.qpos)); fr["mocap"].append(np.array(env.data.mocap_pos))
            fr["wind"].append(np.array(env._wind) + env._wind_mean); fr["eff"].append(float(env.thrust_eff))
            fr["vz"].append(float(obs[10])); fr["speed"].append(float(np.linalg.norm(obs[8:11])))
            fr["laps"].append(float(info.get("laps", 0.0)))
            if term:
                outcome = "crash"
                break
        pkg = (np.r_[env.model.geom_size[env._pkg], env.model.geom_pos[env._pkg]] if env._pkg >= 0 else None)
        ep = {k: np.array(v) for k, v in fr.items() if v}
        eps.append({**ep, "pkg": pkg, "outcome": outcome, "payload": float(env.payload)})
    xml = env.mjcf_xml
    env.close()
    return eps, xml


def _trust_check(cfg, obs_dim, act_dim, trust, device):
    """(obs, act) -> GJS disagreement of the saved ensemble (the number MACURA compares with kappa)."""
    from src.models import ensemble as ens
    from src.algorithms.macura import compute_gjs
    model = ens.build_ensemble(cfg["ensemble"], obs_dim, act_dim, device)
    ens.load_ensemble(model, trust["ensemble"])
    return lambda o, a: float(compute_gjs(*ens.member_gaussians(model, np.asarray(o)[None], np.asarray(a)[None]))[0])


_CHILD = r'''
import os, sys, pickle
os.environ["MUJOCO_GL"] = "%(backend)s"
sys.path.insert(0, "%(root)s")
import numpy as np, mujoco, imageio
from PIL import Image, ImageDraw, ImageFont
from src.viz import scenery
d = pickle.load(open("%(data)s", "rb"))
m = mujoco.MjModel.from_xml_path("%(asset)s"); data = mujoco.MjData(m)
W, H = d["size"]; HEAD, FOOT = 78, 30
for g in ("core_geom", "arm_fr", "arm_bl", "arm_fl", "arm_br", "rotor1", "rotor2", "rotor3", "rotor4", "package"):
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
    if gid >= 0:
        m.geom_size[gid] *= 2.0; m.geom_pos[gid] *= 2.0          # wide view: draw the drone 2x larger
r = mujoco.Renderer(m, height=H, width=W)
ids = scenery.scenery_ids(m)
pkg_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "package")
def font(size):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()
big, small = font(18), font(14)
labels, flights, cols = d["labels"], d["flights"], d["cols"]
rows = (len(labels) + cols - 1) // cols
GREEN, ORANGE, RED, WHITE, GREY = (90, 220, 120), (255, 150, 40), (255, 80, 60), (255, 255, 255), (170, 170, 170)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.elevation, cam.azimuth = d["camera"]["distance"], d["camera"]["elevation"], d["camera"]["azimuth"]
cam.lookat[:] = d["camera"]["lookat"]
score = {k: {"laps": 0.0, "crash": 0, "n": 0, "tr": 0, "tn": 0} for k in labels}
w = imageio.get_writer("%(out)s", fps=d["fps"], macro_block_size=16)
for sc in range(len(flights[labels[0]])):
    eps = {k: flights[k][sc] for k in labels}
    T = max(len(e["qpos"]) for e in eps.values())
    trails, done = {k: [] for k in labels}, {k: False for k in labels}
    for i in list(range(0, T, d["step"])) + [T - 1] * int(d["fps"] * 1.5):
        tiles = []
        for k in labels:
            ep = eps[k]; L = len(ep["qpos"]); j = min(i, L - 1)
            if i < L and (not trails[k] or trails[k][-1][1] != i):
                trails[k].append((ep["qpos"][i][0:3].copy(), i, scenery.trail_color(ep["eff"][i])))
            if j == L - 1 and not done[k]:
                done[k] = True; s = score[k]; s["n"] += 1; s["crash"] += ep["outcome"] == "crash"
                s["laps"] += float(ep["laps"][-1])
                if "trusted" in ep:
                    s["tr"] += int(ep["trusted"].sum()); s["tn"] += len(ep["trusted"])
            data.qpos[:] = ep["qpos"][j]; data.mocap_pos[:] = ep["mocap"][j]
            if pkg_id >= 0 and ep["pkg"] is not None:
                m.geom_size[pkg_id] = ep["pkg"][:3] * 2.0; m.geom_pos[pkg_id] = ep["pkg"][3:] * 2.0
            scenery.decorate(m, data, ids, ep["wind"][j], ep["eff"][j])
            mujoco.mj_forward(m, data)
            r.update_scene(data, camera=cam)
            scenery.add_trail(r.scene, [p for p, _, _ in trails[k]], [c for _, _, c in trails[k]], radius=0.03)
            im = Image.new("RGB", (W, H + HEAD + FOOT), (0, 0, 0))
            im.paste(Image.fromarray(r.render()), (0, HEAD))
            dr = ImageDraw.Draw(im)
            dr.text((8, 3), f"{k}   package {ep['payload']:.2f} kg   t = {j * 0.02:.1f} s", font=big, fill=WHITE)
            eff = ep["eff"][j]
            dr.text((8, 26), f"speed {ep['speed'][j]:.1f} m/s   sink {max(-ep['vz'][j], 0.0):.1f} m/s   lift {100 * eff:.0f}%%",
                    font=small, fill=WHITE if eff > 0.97 else RED)
            if "trusted" in ep:
                ok = bool(ep["trusted"][j])
                n_ok = int(ep["trusted"][:j + 1].sum())
                dr.ellipse([8, 49, 20, 61], fill=GREEN if ok else ORANGE)
                dr.text((26, 46), ("model: agree -> trusted" if ok else "model: DISAGREE -> stops imagining")
                        + f"   ({100 * n_ok / (j + 1):.0f}%% trusted so far)", font=small, fill=GREEN if ok else ORANGE)
                x0, x1 = 8, W - 8
                for t in range(0, j + 1, 2):
                    x = x0 + (x1 - x0) * t / max(T - 1, 1)
                    dr.line([x, 67, x, 74], fill=GREEN if ep["trusted"][t] else ORANGE)
            elif d["notes"].get(k):
                dr.text((8, 48), d["notes"][k], font=small, fill=GREY)
            s = score[k]
            foot = f"laps {s['laps']:.1f}   crashes {s['crash']}/{s['n']} flights"
            if s["tn"]:
                foot += f"   trusted {100 * s['tr'] / s['tn']:.0f}%%"
            dr.rectangle([0, H + HEAD, W, H + HEAD + FOOT], fill=(22, 22, 28))
            dr.text((8, H + HEAD + 6), foot, font=small, fill=WHITE)
            if j == L - 1:
                txt = "CRASH" if ep["outcome"] == "crash" else f"TIME UP  {float(ep['laps'][-1]):.1f} laps"
                dr.rectangle([W // 2 - 110, HEAD + H - 44, W // 2 + 110, HEAD + H - 8], fill=(0, 0, 0))
                dr.text((W // 2 - 100, HEAD + H - 40), txt, font=big, fill=RED if ep["outcome"] == "crash" else GREEN)
            tiles.append(np.asarray(im))
        tiles += [np.zeros_like(tiles[0])] * (rows * cols - len(tiles))
        grid = np.vstack([np.hstack(tiles[q * cols:(q + 1) * cols]) for q in range(rows)])
        for q in range(1, cols):
            grid[:, q * W - 1:q * W + 1] = 255
        for q in range(1, rows):
            y = q * (H + HEAD + FOOT); grid[y - 1:y + 1] = 255
        h16, w16 = grid.shape[0] // 16 * 16, grid.shape[1] // 16 * 16
        w.append_data(np.ascontiguousarray(grid[:h16, :w16]))
w.close()
print("OK")
'''
