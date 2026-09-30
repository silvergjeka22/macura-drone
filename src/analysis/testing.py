"""Test every trained model on new scenarios (seeds 3000+, never used before)."""

from __future__ import annotations

import os

import numpy as np

from src.analysis import results as R

TEST_SEED_BASE = 3000


def test_model(run, cfg, episodes=50, seed_base=TEST_SEED_BASE, traces=3, device="cpu") -> dict:
    """Fly one run's best checkpoint on `episodes` new scenarios; the first `traces` flights are recorded."""
    import torch
    from src.envs import drone_env
    from src.algorithms.sac import load_agent
    torch.set_num_threads(1)
    env, obs_dim, act_dim = drone_env.make_env(cfg["env"], seed=0)
    agent = load_agent(R.checkpoint(run), obs_dim, act_dim, cfg["sac"], device)
    model = None
    if R.ensemble_checkpoint(run):
        from src.models import ensemble as ens
        model = ens.build_ensemble(cfg["ensemble"], obs_dim, act_dim, device)
        ens.load_ensemble(model, R.ensemble_checkpoint(run))
    eps, trs = [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed_base + i)
        wind = float(np.linalg.norm(getattr(env, "_wind_mean", np.zeros(3))[:2]))
        ret, done, info, crashed, max_sink, lost, n = 0.0, False, {}, False, 0.0, 0, 0
        rec = {k: [] for k in ("pos", "vel", "sink", "eff", "gust", "reward", "laps")} if i < traces else None
        O, A = [], []
        while not done:
            act = agent.predict(np.asarray(obs, np.float32), deterministic=True)[0]
            if rec is not None:
                O.append(np.asarray(obs, np.float32)); A.append(np.asarray(act, np.float32))
            obs, r, te, tr, info = env.step(act)
            ret += r; n += 1
            crashed = crashed or te
            sink = -float(env.data.qvel[2])
            max_sink = max(max_sink, sink)
            lost += env.thrust_eff < 0.99
            if rec is not None:
                rec["pos"].append(np.array(env.data.qpos[0:3])); rec["vel"].append(np.array(env.data.qvel[0:3]))
                rec["sink"].append(sink); rec["eff"].append(float(env.thrust_eff))
                rec["gust"].append(np.array(env._wind)); rec["reward"].append(float(r))
                rec["laps"].append(float(info.get("laps", 0.0)))
            done = te or tr
        eps.append({"scenario": seed_base + i, "return": ret, "laps": float(info.get("laps", 0.0)),
                    "crashed": bool(crashed), "payload": float(env.payload), "wind": wind, "max_sink": max_sink,
                    "liftloss": lost / max(n, 1), "length": n})
        if rec is not None:
            rec = {k: np.array(v) for k, v in rec.items()}
            rec.update(scenario=seed_base + i, payload=float(env.payload), crashed=bool(crashed),
                       wind_mean=np.array(getattr(env, "_wind_mean", np.zeros(3))))
            if model is not None and O:
                from src.models import ensemble as ens
                from src.algorithms.macura import compute_gjs
                rec["gjs"] = compute_gjs(*ens.member_gaussians(model, np.array(O), np.array(A)))
            trs.append(rec)
    env.close()
    return {"algo": run["algo"], "seed": run["seed"], "kappa": R.kappa_at(run), "episodes": eps, "traces": trs}


def _job(args):
    return test_model(*args)


def test_all(runs, cfg, episodes=50, seed_base=TEST_SEED_BASE, traces=3, workers=None) -> list:
    """Every run on the same new scenarios; several runs at the same time (one process each)."""
    jobs = [(r, cfg, episodes, seed_base, traces) for r in runs]
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    if workers > 1:
        try:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
                return list(pool.map(_job, jobs))
        except Exception as e:                        # e.g. a restricted environment: fall back to one at a time
            print("parallel test failed, testing one model at a time:", e)
    return [_job(j) for j in jobs]


# ── tables ───────────────────────────────────────────────────────────────────────────────────────
TEST_SCORES = (("return", "test return", True, "{:.0f}"), ("lap", "lap progress per flight", True, "{:.0%}"),
               ("crash", "crash rate", False, "{:.0%}"), ("liftloss", "time losing lift", False, "{:.1%}"),
               ("max_sink", "fastest descent (m/s)", None, "{:.2f}"))


def seed_scores(res) -> dict:
    e = res["episodes"]
    return {"return": float(np.mean([x["return"] for x in e])),
            "lap": float(np.mean([np.clip(x["laps"], 0, 1) for x in e])),
            "crash": float(np.mean([x["crashed"] for x in e])),
            "liftloss": float(np.mean([x["liftloss"] for x in e])),
            "max_sink": float(np.median([x["max_sink"] for x in e]))}


def test_tables(results) -> dict:
    """{'summary': md, 'per_seed': md, 'paired': [md, ...]}"""
    by = {}
    for res in results:
        by.setdefault(res["algo"], []).append(res)
    n_ep = len(results[0]["episodes"]) if results else 0
    head = "| | seeds | " + " | ".join(l for _, l, _, _ in TEST_SCORES) + " |"
    rows = [head, "|" + "---|" * (len(TEST_SCORES) + 2)]
    for a in [x for x in R.ALGOS if x in by]:
        sc = [seed_scores(r) for r in by[a]]
        cells = []
        for k, _, _, fmt in TEST_SCORES:
            c, lo, hi = R.ci([s[k] for s in sc])
            cells.append(fmt.format(c) + (f" [{fmt.format(lo)}, {fmt.format(hi)}]" if len(sc) >= 3 else ""))
        rows.append(f"| {R.NAMES[a]} | {len(sc)} | " + " | ".join(cells) + " |")
    summary = "\n".join(rows) + (f"\n\nIQM over seeds [95% bootstrap CI]; each seed = its best checkpoint on the same "
                                 f"{n_ep} new scenarios (seeds {TEST_SEED_BASE}-{TEST_SEED_BASE + n_ep - 1}).")
    per = ["| algorithm | seed | " + " | ".join(l for _, l, _, _ in TEST_SCORES) + " |", "|" + "---|" * (len(TEST_SCORES) + 2)]
    for a in [x for x in R.ALGOS if x in by]:
        for r in sorted(by[a], key=lambda x: x["seed"]):
            s = seed_scores(r)
            per.append(f"| {R.NAMES[a]} | {r['seed']} | " + " | ".join(fmt.format(s[k]) for k, _, _, fmt in TEST_SCORES) + " |")
    paired = []
    if "macura" in by:
        mac = {r["seed"]: seed_scores(r) for r in by["macura"]}
        for other in [x for x in ("mbpo", "m2ac", "sac") if x in by]:
            oth = {r["seed"]: seed_scores(r) for r in by[other]}
            seeds = sorted(set(mac) & set(oth))
            t = [f"| MACURA - {R.NAMES[other]} | " + " | ".join(f"seed {s}" for s in seeds) + " | mean | MACURA better in |",
                 "|---|" + "---|" * (len(seeds) + 2)]
            for k, label, hib, fmt in TEST_SCORES:
                if hib is None:
                    continue
                d = [mac[s][k] - oth[s][k] for s in seeds]
                fm = fmt.replace("{:", "{:+")
                t.append(f"| {label} | " + " | ".join(fm.format(x) for x in d) + f" | {fm.format(np.mean(d))} | "
                         f"{sum((x > 0) if hib else (x < 0) for x in d)} of {len(d)} |")
            paired.append(f"#### Test: MACURA vs {R.NAMES[other]}\n\n" + "\n".join(t))
    return {"summary": summary, "per_seed": "\n".join(per), "paired": paired}


def final_table(runs, results) -> str:
    """One row per algorithm: learning (IQM over seeds) and the new test (IQM over seeds)."""
    train = R.summary(runs)
    test = {}
    for res in results:
        test.setdefault(res["algo"], []).append(seed_scores(res))
    rows = ["| algorithm | avg return while learning | drones broken | test return | test lap progress | test crash rate |",
            "|---|---|---|---|---|---|"]
    for algo, t in test.items():
        rows.append(f"| {R.NAMES[algo]} | {train[algo]['avg_return'][0]:.0f} | {train[algo]['broken'][0]:.0f} | "
                    f"{R.iqm([s['return'] for s in t]):.0f} | {R.iqm([s['lap'] for s in t]):.0%} | "
                    f"{R.iqm([s['crash'] for s in t]):.0%} |")
    return "\n".join(rows)


def wins(results) -> str:
    """In how many seeds MACURA's test return beats each other method (same seeds, same scenarios)."""
    by = {}
    for res in results:
        by.setdefault(res["algo"], {})[res["seed"]] = seed_scores(res)["return"]
    lines = []
    for other in [x for x in ("mbpo", "m2ac", "sac") if x in by and "macura" in by]:
        seeds = sorted(set(by["macura"]) & set(by[other]))
        d = [by["macura"][s] - by[other][s] for s in seeds]
        lines.append(f"- MACURA vs {R.NAMES[other]}: better in {sum(x > 0 for x in d)} of {len(d)} seeds, "
                     f"test return {np.mean(d):+.0f} on average")
    return "\n".join(lines)
