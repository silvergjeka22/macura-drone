"""Visual-only decorations for videos and the simulator: windsock along the wind, drone tinted and
trail coloured by lift loss (green = full lift, red = losing it). numpy + mujoco only."""

from __future__ import annotations

import numpy as np
import mujoco

WIND_FULL = 0.8                                  # gust force (N) that fully inflates the sock
_DRONE_GEOMS = ("core_geom", "arm_fr", "arm_bl", "arm_fl", "arm_br")
_RED = np.array([0.95, 0.20, 0.12, 1.0])
_GREEN = np.array([0.20, 0.80, 0.40, 1.0])


def scenery_ids(model) -> dict:
    """Geom/mocap ids the decorations touch (missing ones are simply skipped)."""
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "windsock")
    drone = [g for g in (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in _DRONE_GEOMS)
             if g >= 0]
    return {"sock": int(model.body_mocapid[body]) if body >= 0 else -1,
            "drone": drone, "drone_rgba": {g: model.geom_rgba[g].copy() for g in drone}}


def _danger(eff) -> float:
    """0 at full thrust .. 1 once 20% or more of the thrust is lost."""
    return float(np.clip((1.0 - float(eff)) / 0.2, 0.0, 1.0))


def decorate(model, data, ids, wind, eff):
    """Point the windsock downwind (limp in calm air, straight out in a strong gust) and tint the drone
    from its own colour to red as it loses thrust. Only rendering fields are written."""
    if ids["sock"] >= 0:
        w = np.asarray(wind, dtype=float)[:2]
        mag = float(np.hypot(w[0], w[1]))
        yaw = float(np.arctan2(w[1], w[0])) if mag > 1e-6 else 0.0
        droop = np.radians(75.0) * (1.0 - min(mag / WIND_FULL, 1.0))
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
                           np.array([np.cos(droop / 2), 0.0, np.sin(droop / 2), 0.0]))
        data.mocap_quat[ids["sock"]] = q
    k = _danger(eff)
    for g in ids["drone"]:
        model.geom_rgba[g] = (1.0 - k) * ids["drone_rgba"][g] + k * _RED


def trail_color(eff) -> np.ndarray:
    k = _danger(eff)
    return ((1.0 - k) * _GREEN + k * _RED).astype(np.float32)


def add_trail(scene, points, colors, radius=0.012):
    """Append the trail dots to an already-updated mjvScene (call after renderer.update_scene)."""
    for p, c in zip(points, colors):
        if scene.ngeom >= scene.maxgeom:
            break
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([radius, 0.0, 0.0]), np.asarray(p, dtype=np.float64),
                            np.eye(3).flatten(), np.asarray(c, dtype=np.float32))
        scene.ngeom += 1
