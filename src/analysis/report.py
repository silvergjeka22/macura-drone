"""One call that turns run logs into the full comparison: figures, summary.md and (optionally) videos.
Used by the training notebooks (Kaggle) and by testing/test.ipynb (your computer)."""

from __future__ import annotations

import os

import numpy as np

from src.analysis import results as R


def build(runs, cfg, out_dir, check=True, videos=False, close=False, device="cpu") -> dict:
    """Write everything for `runs` into out_dir (plots/, videos/, summary.md). Returns {name: path}."""
    import matplotlib.pyplot as plt
    from src.viz import figures as F

    plots = os.path.join(out_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    out = {}
    refs = R.autopilot_reference(cfg["env"], seed_base=cfg["experiment"]["eval_seeds"][0],
                                 episodes=cfg["experiment"]["eval_episodes"],
                                 cache=os.path.join(out_dir, "autopilot_reference.json"))
    test_refs = R.autopilot_reference(cfg["env"], seed_base=cfg["selection"]["final_eval_seed_base"],
                                      episodes=cfg["selection"]["final_eval_episodes"],
                                      cache=os.path.join(out_dir, "autopilot_reference_test.json"))
    figs = [("learning_curves", lambda p: F.plot_learning_curves(runs, refs, p)),
            ("summary_table", lambda p: F.plot_summary_table(runs, p)),
            ("scores", lambda p: F.plot_scores(runs, p)),
            ("model_trust", lambda p: F.plot_model_trust(runs, p)),
            ("kappa", lambda p: F.plot_kappa(runs, p) if any(r["algo"] == "macura" for r in runs) else None),
            ("untrusted_data", lambda p: F.plot_untrusted_data(runs, p)),
            ("crash_by_payload", lambda p: F.plot_crash_by_payload(runs, p))]
    checks = []
    if check:
        for algo in ("macura", "mbpo"):
            rs = R.by_algo(runs).get(algo)
            if rs and R.ensemble_checkpoint(R.best_run(rs)):
                run = R.best_run(rs)
                c = R.model_check(run, cfg, device=device)
                if c["kappa"] is None:                        # MBPO: judged with MACURA's threshold
                    mac = R.by_algo(runs).get("macura")
                    c["kappa"] = R.kappa_at(R.best_run(mac)) if mac else None
                    c["trusted"] = float(np.mean(c["gjs"] < c["kappa"])) if c["kappa"] else float("nan")
                checks.append(c)
        if checks:
            figs.append(("model_check", lambda p: F.plot_model_check(checks, p)))
    for name, make in figs:
        path = os.path.join(plots, f"{name}.png")
        fig = make(path)
        if fig is not None:
            out[name] = path
            if close:
                plt.close(fig)

    if videos:
        from src.viz import video as V
        best = {R.NAMES[a]: R.best_run(rs) for a, rs in R.by_algo(runs).items()}
        pilots = {k: R.checkpoint(r) for k, r in best.items()}
        trust = {k: {"ensemble": R.ensemble_checkpoint(r), "kappa": R.kappa_at(r)}
                 for k, r in best.items() if r["algo"] == "macura" and R.ensemble_checkpoint(r)}
        seeds = [cfg["selection"]["final_eval_seed_base"] + i for i in range(3)]
        out["race_video"] = V.record_race(cfg, pilots, os.path.join(out_dir, "videos", "race_all.mp4"), seeds=seeds,
                                          trust=trust, frame_step=3, fps=17, size=(480, 360), device=device)
        duo = {k: v for k, v in pilots.items() if k in ("MACURA", "MBPO")}
        if len(duo) == 2:
            out["long_video"] = V.record_race(cfg, duo, os.path.join(out_dir, "videos", "long_flight.mp4"),
                                              seeds=[seeds[0] + 10, seeds[0] + 11], trust=trust, frame_step=3,
                                              fps=17, max_steps=1500, device=device)

    md = markdown(runs, refs, test_refs, checks)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(md)
    out["summary"] = os.path.join(out_dir, "summary.md")
    return out


def markdown(runs, refs=None, test_refs=None, checks=()) -> str:
    lines = ["# Results", "", R.summary_markdown(runs), "", "## Per seed", "", R.per_seed_markdown(runs), "",
             "## Trust in the world model", "", R.trust_summary(runs).replace("\n", "  \n")]
    for c in checks:
        t = c["gjs"] < c["kappa"] if c["kappa"] else None
        if t is None:
            continue
        e_in = float(np.median(c["error"][t])) if t.any() else float("nan")
        e_out = float(np.median(c["error"][~t])) if (~t).any() else float("nan")
        lines.append(f"- {R.NAMES[c['algo']]} seed {c['seed']}, best world model on {c['steps']} real test steps: "
                     f"{100 * c['trusted']:.0f}% below kappa; median model error {e_in:.3f} there vs {e_out:.3f} "
                     f"on the rest; fast descents trusted {100 * np.mean(t[c['fast']]) if c['fast'].any() else float('nan'):.0f}%")
    if refs:
        lines += ["", "## Hand-written autopilot (reference, not learned)", "",
                  "| | return (selection scenarios) | laps | crash | return (test scenarios) | laps | crash |",
                  "|---|---|---|---|---|---|---|"]
        for k, v in refs.items():
            t = (test_refs or {}).get(k, {})
            lines.append(f"| {k} | {v['return']:.0f} | {v['laps']:.2f} | {v['crash']:.2f} | "
                         f"{t.get('return', float('nan')):.0f} | {t.get('laps', float('nan')):.2f} | "
                         f"{t.get('crash', float('nan')):.2f} |")
    return "\n".join(lines) + "\n"
