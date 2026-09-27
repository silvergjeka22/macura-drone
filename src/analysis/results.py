"""Run logs -> scores, confidence intervals, tables and trust numbers (numpy only; plots live in src/viz).

A run is the JSON log written by train_one (logs/<algo>_seed<seed>.json). Logs from several sessions
(one seed each) are merged with load_runs(dir0, dir1, ...). Across seeds every score is the IQM with a
95% bootstrap confidence interval (Agarwal et al. 2021), as recommended for few-seed RL comparisons.
"""

from __future__ import annotations

import glob
import json
import os
import warnings

import numpy as np

ALGOS = ("macura", "mbpo", "m2ac", "sac")
NAMES = {"macura": "MACURA", "mbpo": "MBPO", "m2ac": "M2AC", "sac": "SAC"}

# (key, label, higher is better, format) - the scores fixed before the runs, main ones first.
# success = share of a full lap flown in a 10 s flight (0-100%; a crash ends the flight, so it counts how far
# the drone got). It replaces "a full lap within one flight", which needs > 2.4 m/s on average and stays 0 for
# every learner (the autopilot at 3 m/s manages it in ~10% of flights): it grades how well each one flies.
SCORES = (
    ("avg_return", "avg return while learning", True, "{:.0f}"),
    ("broken", "drones broken after warm-up", False, "{:.0f}"),
    ("test_return", "final test return", True, "{:.0f}"),
    ("test_success", "success, % of a lap (test)", True, "{:.0%}"),
    ("test_crash", "final test crash rate", False, "{:.0%}"),
    ("last10_return", "return, last 10 evals", True, "{:.0f}"),
)
SMOOTH = 5                                   # learning curves: rolling mean over this many evaluations


# ── loading ──────────────────────────────────────────────────────────────────────────────────────
def run_roots(path: str) -> list:
    """Every folder under `path` (itself included) that holds logs/<algo>_seed<n>.json."""
    hits = glob.glob(os.path.join(path, "**", "logs", "*_seed*.json"), recursive=True)
    return sorted({os.path.dirname(os.path.dirname(h)) for h in hits})


def run_length(run) -> int:
    """Planned real steps of a run (older logs: last evaluation step + 1000)."""
    return int(run.get("total_env_steps") or ((run["steps"][-1] + 1000) if run.get("steps") else 0))


def load_runs(*paths, algorithms=None, verbose=True, same_length=True) -> list:
    """Merge the run logs found under `paths` (output roots, or any folder above them). A duplicate
    (algo, seed) keeps the first one found. Each run gets `_root`, its output folder. With `same_length`
    only the runs of the longest planned length are kept (curves of different lengths do not mix)."""
    found = []
    for p in paths:
        for root in run_roots(p):
            for path in sorted(glob.glob(os.path.join(root, "logs", "*_seed*.json"))):
                with open(path) as f:
                    r = json.load(f)
                if not algorithms or r.get("algo") in algorithms:
                    r["_root"] = root
                    found.append(r)
    if same_length and found:
        longest = max(run_length(r) for r in found)
        skipped = [r for r in found if run_length(r) != longest]
        found = [r for r in found if run_length(r) == longest]
        if skipped and verbose:
            print(f"  kept the {longest:,}-step runs; skipped " + ", ".join(
                f"{NAMES.get(r['algo'], r['algo'])} seed {r['seed']} ({run_length(r):,} steps)" for r in skipped))
    runs, seen = [], {}
    for r in found:
        key = (r.get("algo"), r.get("seed"))
        if key in seen:
            if verbose:
                print(f"  duplicate {key[0]} seed {key[1]}: keeping {seen[key]}, skipping {r['_root']}")
            continue
        seen[key] = r["_root"]
        runs.append(r)
    runs.sort(key=lambda r: (ALGOS.index(r["algo"]) if r["algo"] in ALGOS else 99, r["seed"]))
    if verbose:
        for a, rs in by_algo(runs).items():
            print(f"  {NAMES.get(a, a):7s} seeds {[r['seed'] for r in rs]}")
    return runs


def by_algo(runs) -> dict:
    out = {}
    for a in list(ALGOS) + sorted({r["algo"] for r in runs} - set(ALGOS)):
        rs = [r for r in runs if r["algo"] == a]
        if rs:
            out[a] = rs
    return out


def best_run(runs) -> dict:
    """The seed with the highest return on the selection scenarios (never picked on the test scenarios)."""
    return max(runs, key=lambda r: r.get("best_return", -1e18))


def checkpoint(run) -> str:
    """The run's best policy .zip, found next to its log (logs downloaded from Kaggle keep Kaggle paths)."""
    local = os.path.join(run.get("_root", ""), "checkpoints", f"{run['algo']}_seed{run['seed']}_best.zip")
    return local if os.path.exists(local) else run.get("checkpoint")


def ensemble_checkpoint(run):
    p = checkpoint(run)
    p = p[:-4] + "_ensemble.pt" if p and p.endswith(".zip") else None
    return p if p and os.path.exists(p) else None


def kappa_at(run, step=None):
    """MACURA's trust threshold at `step` (default: the best checkpoint's step)."""
    step = run.get("best_step") if step is None else step
    ks = [(s, k) for s, k in run.get("kappa", []) if step is None or s <= step]
    return float(ks[-1][1]) if ks else None


# ── statistics ───────────────────────────────────────────────────────────────────────────────────
def iqm(vals) -> float:
    v = np.sort(np.asarray(vals, dtype=float))
    k = int(len(v) * 0.25)
    core = v[k:len(v) - k] if len(v) - 2 * k >= 1 else v
    return float(core.mean())


def ci(vals, stat="iqm", n_boot=2000, alpha=0.05, seed=0):
    """(center, lo, hi): IQM or mean over seeds with a percentile-bootstrap CI (lo = hi = center below 3 seeds;
    with 2 seeds the IQM is their mean)."""
    vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    f = iqm if stat == "iqm" else (lambda a: float(np.mean(a)))
    center = f(vals)
    if len(vals) < 3:
        return center, center, center
    rng = np.random.default_rng(seed)
    boots = [f(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return center, float(lo), float(hi)


def curves(run_list, key):
    """(steps, matrix n_seeds x n_evals) of a per-eval series; a shorter run (stopped early) is padded with NaN."""
    rs = [r for r in run_list if r.get(key)]
    if not rs:
        return None, None
    lens = [min(len(r["steps"]), len(r[key])) for r in rs]
    n = max(lens)
    mat = np.full((len(rs), n), np.nan)
    for i, (r, k) in enumerate(zip(rs, lens)):
        mat[i, :k] = r[key][:k]
    return np.asarray(rs[lens.index(n)]["steps"][:n]), mat


def band(mat, stat="iqm"):
    """Per-eval-point center and band across seeds: 95% CI with 3 or more seeds, else the seeds' range."""
    out = []
    for j in range(mat.shape[1]):
        v = mat[:, j][np.isfinite(mat[:, j])]
        out.append(ci(v, stat, seed=j) if len(v) >= 3 else
                   ((float(np.mean(v)), float(v.min()), float(v.max())) if len(v) else (np.nan,) * 3))
    c = np.array(out)
    return c[:, 0], c[:, 1], c[:, 2]


def series(run_list, key):
    """(steps, nan-mean over seeds) of a per-model-round [(step, value)] diagnostic."""
    ss = [np.asarray(r[key], dtype=float) for r in run_list if r.get(key)]
    if not ss:
        return None, None
    n = min(len(s) for s in ss)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return ss[0][:n, 0], np.nanmean(np.stack([s[:n, 1] for s in ss]), axis=0)


# ── scores ───────────────────────────────────────────────────────────────────────────────────────
def success(laps):
    """Share of a full lap flown per 10 s flight, 0..1 (mean laps of the evaluation flights, clipped)."""
    return np.clip(np.asarray(laps, dtype=float), 0.0, 1.0)


def smooth(y, window=SMOOTH):
    """Trailing rolling mean over `window` evaluations (display only; the scores use the raw values)."""
    y = np.asarray(y, dtype=float)
    if window <= 1 or len(y) == 0:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    n = np.minimum(np.arange(1, len(y) + 1), window)
    return (c[1:] - c[np.maximum(np.arange(1, len(y) + 1) - window, 0)]) / n


def run_scores(run) -> dict:
    ret = np.asarray(run.get("eval_return", []), dtype=float)
    fe = run.get("final_eval", {})
    tc = run.get("train_crashes") or [0]
    return {
        "avg_return": float(ret.mean()) if len(ret) else float("nan"),
        "last10_return": float(ret[-10:].mean()) if len(ret) else float("nan"),
        "test_return": fe.get("eval_return", float("nan")),
        "test_crash": fe.get("eval_failure_rate", float("nan")),
        "test_laps": fe.get("eval_laps", float("nan")),
        "test_success": success(fe.get("eval_laps", float("nan"))),
        "test_full_lap": fe.get("eval_success_rate", float("nan")),
        "broken": float(run.get("train_crashes_total", 0) - tc[0]),
        "best_step": run.get("best_step"),
    }


def summary(runs) -> dict:
    """{algo: {score: (IQM, lo, hi, per-seed values)}}"""
    out = {}
    for a, rs in by_algo(runs).items():
        sc = [run_scores(r) for r in rs]
        out[a] = {k: (*ci([s[k] for s in sc]), [s[k] for s in sc]) for k, *_ in SCORES}
        out[a]["seeds"] = [r["seed"] for r in rs]
    return out


def best_per_score(s) -> dict:
    """{score: algo} for the scores where one algorithm is best at the shown precision (ties: none)."""
    best = {}
    for k, _, hib, fmt in SCORES:
        vals = {a: v[k][0] for a, v in s.items() if np.isfinite(v[k][0])}
        if len(vals) > 1:
            top = (max if hib else min)(vals.values())
            winners = [a for a, x in vals.items() if fmt.format(x) == fmt.format(top)]
            if len(winners) == 1:
                best[k] = winners[0]
    return best


def summary_markdown(runs) -> str:
    s = summary(runs)
    n_seeds = max((len(v["seeds"]) for v in s.values()), default=0)
    head = "| | seeds | " + " | ".join(lbl for _, lbl, _, _ in SCORES) + " |"
    rows = [head, "|" + "---|" * (len(SCORES) + 2)]
    best = best_per_score(s)
    for a, v in s.items():
        cells = []
        for k, _, _, fmt in SCORES:
            c, lo, hi, _ = v[k]
            txt = fmt.format(c) if np.isfinite(c) else "-"
            if len(v["seeds"]) >= 3 and np.isfinite(c):
                txt += f" [{fmt.format(lo)}, {fmt.format(hi)}]"
            elif len(v["seeds"]) == 2 and np.isfinite(c):
                txt += " (" + " / ".join(fmt.format(x) for x in v[k][3]) + ")"
            cells.append(f"**{txt}**" if best.get(k) == a and len(s) > 1 else txt)
        rows.append(f"| {NAMES.get(a, a)} | {len(v['seeds'])} | " + " | ".join(cells) + " |")
    note = ("IQM over seeds [95% bootstrap CI]" if n_seeds >= 3 else
            "mean of the 2 seeds (each seed in brackets); a confidence interval needs 3 or more" if n_seeds == 2 else
            "one seed per algorithm: no confidence interval")
    return "\n".join(rows) + (f"\n\n{note}; bold = best. Test = the best checkpoint on 30 fresh scenarios. "
                              "Success = share of a full lap flown per 10 s flight (autopilot on the same test flights: "
                              "54% at 1.5 m/s, 95% at 3 m/s).")


def per_seed_markdown(runs) -> str:
    lines = ["| algorithm | seed | " + " | ".join(lbl for _, lbl, _, _ in SCORES) + " | best step |",
             "|" + "---|" * (len(SCORES) + 3)]
    for r in runs:
        sc = run_scores(r)
        lines.append(f"| {NAMES.get(r['algo'], r['algo'])} | {r['seed']} | "
                     + " | ".join(fmt.format(sc[k]) if np.isfinite(sc[k]) else "-" for k, _, _, fmt in SCORES)
                     + f" | {sc['best_step']} |")
    return "\n".join(lines)


def trust_summary(runs) -> str:
    """How much of its imagination each model-based method trusted over the run."""
    lines = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for a, rs in by_algo(runs).items():
            if a == "macura":
                fast = np.nanmean([v for r in rs for _, v in r.get("trust_fast", [])])
                slow = np.nanmean([v for r in rs for _, v in r.get("trust_slow", [])])
                trip = np.nanmean([v for r in rs for _, v in r.get("rollout_length", [])])
                lines.append(f"MACURA trusted {10 * trip:.0f}% of a full 10-step imagination (average trip "
                             f"{trip:.1f}/10); steps passing its trust check: fast descents {100 * fast:.0f}%, "
                             f"normal flight {100 * slow:.0f}%")
            elif a == "mbpo":
                unt = np.nanmean([v for r in rs for _, v in r.get("untrusted_frac", [])])
                lines.append(f"MBPO trusted 100% of what it imagined (no trust check); {100 * unt:.0f}% of it was "
                             "above MACURA's threshold")
            elif a == "m2ac":
                unt = np.nanmean([v for r in rs for _, v in r.get("untrusted_frac", [])])
                lines.append(f"M2AC kept the 25% least uncertain imagined steps; {100 * unt:.0f}% of those were "
                             "above MACURA's threshold")
    return "\n".join(lines)


# ── references and model checks (these fly the simulator) ─────────────────────────────────────────
AUTOPILOTS = {"autopilot 1.5 m/s": (0.8, 1.5), "autopilot 3 m/s": (1.1, 3.0)}


def autopilot_reference(env_cfg, seed_base=100, episodes=20, pilots=None, cache=None) -> dict:
    """Return / laps / crash rate of the hand-written autopilot (NOT learned) on the given scenarios."""
    if cache and os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    from src.envs import drone_env
    from src.envs.autopilot import Autopilot
    env, _, _ = drone_env.make_env(env_cfg, seed=0)
    out = {}
    for name, (descent, speed) in (pilots or AUTOPILOTS).items():
        rets, laps, crashes = [], [], []
        for i in range(episodes):
            env.reset(seed=seed_base + i)
            ap = Autopilot(env, descent=descent, speed=speed)
            ret, done, info, crashed = 0.0, False, {}, False
            while not done:
                _, r, te, tr, info = env.step(ap.act())
                ret += r
                crashed = crashed or info.get("failure", False)
                done = te or tr
            rets.append(ret); laps.append(info.get("laps", 0.0)); crashes.append(float(crashed))
        out[name] = {"return": float(np.mean(rets)), "laps": float(np.mean(laps)), "crash": float(np.mean(crashes))}
    env.close()
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        with open(cache, "w") as f:
            json.dump(out, f, indent=2)
    return out


def model_check(run, cfg, episodes=10, seed_base=1000, device="cpu") -> dict:
    """Fly a model-based run's best policy in the real simulator and, at every step, compare its saved
    world model's disagreement (GJS) with the model's actual one-step error (paper App. C.4 / Fig. 10).

    Returns per-step arrays: gjs, error (RMS over dims, normalised units), fast (sinking faster than the
    lift-loss onset), plus kappa (MACURA's threshold at that checkpoint) and the share of steps trusted."""
    import torch
    from src.envs import drone_env
    from src.algorithms.sac import load_agent
    from src.models import ensemble as ens
    from src.algorithms.macura import compute_gjs, fast_threshold
    ens_path = ensemble_checkpoint(run)
    if ens_path is None:
        raise FileNotFoundError(f"no ensemble checkpoint for {run['algo']} seed {run['seed']}")
    env, obs_dim, act_dim = drone_env.make_env(cfg["env"], seed=0)
    agent = load_agent(checkpoint(run), obs_dim, act_dim, cfg["sac"], device)
    model = ens.build_ensemble(cfg["ensemble"], obs_dim, act_dim, device)
    ens.load_ensemble(model, ens_path)
    tn = model["target_normalizer"]
    mu, sd = tn.mean.cpu().numpy(), tn.std.cpu().numpy()
    dyn = model.get("dynamic")
    dyn = np.ones(obs_dim, bool) if dyn is None else dyn.cpu().numpy().astype(bool)
    O, A, N = [], [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed_base + i)
        done = False
        while not done:
            act = agent.predict(np.asarray(obs, np.float32), deterministic=True)[0]
            nxt, _, te, tr, _ = env.step(act)
            O.append(obs); A.append(act); N.append(nxt)
            obs, done = nxt, te or tr
    env.close()
    O, A, N = np.array(O, np.float32), np.array(A, np.float32), np.array(N, np.float32)
    gjs, err = [], []
    with torch.no_grad():
        for s in range(0, len(O), 4096):
            m, v = ens.member_gaussians(model, O[s:s + 4096], A[s:s + 4096])
            gjs.append(compute_gjs(m, v))
            real = ((N[s:s + 4096] - O[s:s + 4096]) - mu) / sd
            err.append(np.sqrt(np.mean((m.mean(axis=0) - real[:, dyn]) ** 2, axis=-1)))
    gjs, err = np.concatenate(gjs), np.concatenate(err)
    kappa = kappa_at(run)
    fast = -O[:, 10] > fast_threshold(cfg)
    return {"algo": run["algo"], "seed": run["seed"], "gjs": gjs, "error": err, "fast": fast, "kappa": kappa,
            "trusted": float(np.mean(gjs < kappa)) if kappa is not None else float("nan"), "steps": len(gjs)}
