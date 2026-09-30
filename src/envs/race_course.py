"""The race course: a closed 3-D loop through four gates with a near-vertical 3 m chute.
project(course, pos) -> (nearest course point, distance); course["tangent"] is the racing direction.
"""

from __future__ import annotations

import numpy as np

GATE_U = (0.0, 0.25, 0.5, 0.75)          # lap fraction of the gates P1..P4 (by horizontal distance)


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def course_params(cfg: dict) -> dict:
    """The course numbers from the `env:` config (defaults = the RACE preset)."""
    return {"a": float(cfg.get("race_size", 3.0)), "n": float(cfg.get("race_power", 4.0)),
            "h": tuple(float(v) for v in cfg.get("race_heights", (2.0, 4.0, 1.0, 1.5))),
            "chute_u": float(cfg.get("race_chute_u", 0.30)), "chute_len": float(cfg.get("race_chute_len", 0.25)),
            "spacing": float(cfg.get("race_spacing", 0.05))}


def build_course(cfg: dict) -> dict:
    """Dense course polyline (points, tangents, 3-D arc length) + a lookup table for fast projection."""
    c = course_params(cfg)
    a, n = c["a"], c["n"]
    # fine horizontal loop, clockwise from P1 (left): t = pi -> -pi
    t = np.linspace(np.pi, -np.pi, 40001)[:-1]
    x = a * np.sign(np.cos(t)) * np.abs(np.cos(t)) ** (2.0 / n)
    y = a * np.sign(np.sin(t)) * np.abs(np.sin(t)) ** (2.0 / n)
    seg = np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[y, y[0]]))
    u = np.r_[0.0, np.cumsum(seg)[:-1]] / seg.sum()               # lap fraction by horizontal distance
    perim = float(seg.sum())
    # height profile: smooth steps between the gate heights, plus the chute (a steep 3 m drop)
    h1, h2, h3, h4 = c["h"]
    cu, cl = c["chute_u"], c["chute_len"] / perim                # chute start / horizontal length (lap frac)
    z = np.where(u < 0.25, h1 + (h2 - h1) * _smooth(u / 0.25), h2)
    z = np.where(u >= cu, h2 + (h3 - h2) * _smooth((u - cu) / cl), z)
    z = np.where(u >= 0.5, h3 + (h4 - h3) * _smooth((u - 0.5) / 0.25), z)
    z = np.where(u >= 0.75, h4 + (h1 - h4) * _smooth((u - 0.75) / 0.25), z)
    fine = np.stack([x, y, z], axis=1)
    # resample uniformly in 3-D arc length (so the vertical chute gets its fair share of points)
    d3 = np.linalg.norm(np.diff(np.vstack([fine, fine[:1]]), axis=0), axis=1)
    s_fine = np.r_[0.0, np.cumsum(d3)[:-1]]
    length = float(d3.sum())
    k = int(np.ceil(length / c["spacing"]))
    s = np.linspace(0.0, length, k, endpoint=False)
    idx = np.searchsorted(s_fine, s, side="right") - 1
    pts = fine[idx]
    tang = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
    u_pts = u[idx]
    chute = (u_pts >= cu) & (u_pts < cu + cl)
    # projection lookup: for each 1-degree bin of the horizontal angle, the course points within +-25 deg
    theta = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    cand = []
    for b in range(360):
        diff = np.abs((theta - (b - 180 + 0.5) + 180.0) % 360.0 - 180.0)
        cand.append(np.flatnonzero(diff < 25.0))
    width = max(len(ci) for ci in cand)
    table = np.stack([np.r_[ci, np.full(width - len(ci), ci[0])] for ci in cand]).astype(np.int64)
    gates = []
    for gu in GATE_U:
        j = int(np.argmin(np.abs(((u_pts - gu) + 0.5) % 1.0 - 0.5)))
        gates.append((pts[j].copy(), tang[j].copy()))
    return {"points": pts.astype(np.float64), "points32": pts.astype(np.float32),
            "tangent": tang.astype(np.float64), "s": s, "length": length, "table": table, "chute": chute,
            "gates": gates, "perimeter": perim, "params": c}


_CACHE: dict = {}                        # the last few projections (the same batch is asked for 3 times per step)


def project(course: dict, pos, chunk: int = 8192):
    """Nearest course point for each position (3,) or (B, 3). Returns (index, distance)."""
    p = np.ascontiguousarray(np.atleast_2d(np.asarray(pos, dtype=np.float32)))
    key = (len(course["points32"]), course["length"], p.shape, hash(p.tobytes()))
    if key in _CACHE:
        return _CACHE[key][0].copy(), _CACHE[key][1].copy()
    pts, table = course["points32"], course["table"]
    out_i = np.empty(len(p), dtype=np.int64)
    out_d = np.empty(len(p))
    for lo in range(0, len(p), chunk):
        q = p[lo:lo + chunk]
        b = (np.floor(np.degrees(np.arctan2(q[:, 1], q[:, 0]))).astype(np.int64) + 180) % 360
        cand = table[b]                                            # (b, W)
        d2 = ((pts[cand] - q[:, None, :]) ** 2).sum(-1)
        j = np.argmin(d2, axis=1)
        out_i[lo:lo + chunk] = cand[np.arange(len(q)), j]
        out_d[lo:lo + chunk] = np.sqrt(d2[np.arange(len(q)), j])
    if len(_CACHE) >= 4:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = (out_i, out_d)
    return out_i.copy(), out_d.copy()
