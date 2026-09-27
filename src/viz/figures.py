"""Figures: the comparison (paper-style), MACURA's trust in its model, and the task itself.

Every function takes run logs (or a model_check result / the config), optionally saves a PNG and
returns the matplotlib Figure.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

from src.analysis import results as R

COLORS = {"macura": "#1f77b4", "mbpo": "#d62728", "m2ac": "#2ca02c", "sac": "#7f7f7f"}
REF_STYLE = {"autopilot 1.5 m/s": ":", "autopilot 3 m/s": "-."}


def _name(a):
    return R.NAMES.get(a, a.upper())


def _save(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


def _warmup(runs):
    starts = [r["steps"][0] for r in runs if r.get("steps")]
    return min(starts) if starts else None


def _curve_panel(ax, runs, key, stat, ylabel, title, refs=None, ref_key=None, fn=None, window=1):
    """Center + 95% CI across seeds; with `window` > 1 each seed is smoothed first and the raw center is
    drawn faintly behind. `fn` transforms the values (and the reference lines)."""
    fn = fn or (lambda v: v)
    for a, rs in R.by_algo(runs).items():
        steps, mat = R.curves(rs, key)
        if steps is None:
            continue
        mat, col = fn(mat), COLORS.get(a)
        if window > 1:
            ax.plot(steps, R.band(mat, stat)[0], color=col, lw=0.8, alpha=0.3)
            mat = np.array([R.smooth(row, window) for row in mat])
        c, lo, hi = R.band(mat, stat)
        ax.plot(steps, c, color=col, lw=2, label=f"{_name(a)} ({len(rs)})")
        ax.fill_between(steps, lo, hi, color=col, alpha=0.18)
    for name, ref in (refs or {}).items():
        if ref_key in ref:
            ax.axhline(fn(np.asarray(ref[ref_key])), color="k", lw=1.1, ls=REF_STYLE.get(name, ":"), alpha=0.7,
                       label=name)
    ax.set_xlabel("real environment steps"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=0.3)


def plot_learning_curves(runs, refs=None, save_path=None, window=R.SMOOTH):
    """Paper Fig. 4 for this task: IQM return with 95% CI, crash rate, success (% of a lap per flight) and
    real crashes while learning. `refs` = autopilot_reference() (a hand-written controller, not learned).
    A-C are smoothed over `window` evaluations (faint = raw); the scores always use the raw values."""
    n = max(len(rs) for rs in R.by_algo(runs).values())
    pct = (lambda m: 100.0 * np.asarray(m, dtype=float))
    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
    _curve_panel(ax[0, 0], runs, "eval_return", "iqm", "evaluation return (IQM)",
                 "A. Return while learning", refs, "return", window=window)
    _curve_panel(ax[0, 1], runs, "eval_failure_rate", "mean", "crash rate (%)", "B. Crashes in the evaluations",
                 fn=pct, window=window)
    _curve_panel(ax[1, 0], runs, "eval_laps", "mean", "success: % of a lap per 10 s flight",
                 "C. Success (how far each 10 s flight gets)", refs, "laps",
                 fn=lambda m: 100.0 * R.success(m), window=window)
    _curve_panel(ax[1, 1], runs, "train_crashes", "mean", "real crashes so far",
                 "D. Drones broken while learning (training flights)")
    ax[0, 1].set_ylim(-2, 102); ax[1, 0].set_ylim(-2, 102)
    w = _warmup(runs)
    for a in ax.flat:
        if w:
            a.axvspan(0, w, color="0.85", alpha=0.5, lw=0)
        a.set_xlim(left=0)
        a.legend(fontsize=8, loc="best")
    band = ("IQM / mean over seeds, shaded = 95% bootstrap CI" if n >= 3 else
            "mean of 2 seeds, shaded = range between them" if n == 2 else "one seed per algorithm")
    smooth = f"; A-C: rolling mean over {window} evaluations, faint = raw" if window > 1 else ""
    fig.suptitle(f"MACURA vs MBPO vs M2AC vs SAC  ({band}{smooth}; grey = random warm-up; (n) = seeds)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, save_path)


def plot_summary_table(runs, save_path=None):
    """The results table: one row per algorithm, IQM over seeds (95% CI below it with 3 or more seeds),
    the best value of each column in bold."""
    import textwrap
    s = R.summary(runs)
    best = R.best_per_score(s)
    many = any(len(v["seeds"]) >= 2 for v in s.values())
    n = max(len(v["seeds"]) for v in s.values())
    head = ["algorithm", "seeds"] + [textwrap.fill(lbl, 14) for _, lbl, _, _ in R.SCORES]
    rows, bold = [], []
    for a, v in s.items():
        cells = []
        for k, _, _, fmt in R.SCORES:
            c, lo, hi, _ = v[k]
            txt = fmt.format(c) if np.isfinite(c) else "-"
            if len(v["seeds"]) >= 3 and np.isfinite(c):
                txt += f"\n[{fmt.format(lo)}, {fmt.format(hi)}]"
            elif len(v["seeds"]) == 2 and np.isfinite(c):
                txt += "\n(" + " / ".join(fmt.format(x) for x in v[k][3]) + ")"
            cells.append(txt)
        rows.append([_name(a), str(len(v["seeds"]))] + cells)
        bold.append([False, False] + [best.get(k) == a for k, *_ in R.SCORES])
    head_h, row_h = 0.8, (0.62 if many else 0.42)
    total = head_h + len(rows) * row_h
    fig_h = total + 1.05
    fig = plt.figure(figsize=(13.5, fig_h))
    ax = fig.add_axes([0.01, 0.42 / fig_h, 0.98, total / fig_h])
    ax.axis("off")
    widths = [0.12, 0.06] + [0.82 / len(R.SCORES)] * len(R.SCORES)
    tbl = ax.table(cellText=rows, colLabels=head, colWidths=widths, cellLoc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_height((head_h if r == 0 else row_h) / total)
        cell.set_edgecolor("0.7")
        if r == 0:
            cell.set_facecolor("#eeeeee"); cell.set_text_props(weight="bold", fontsize=10.5)
        elif c == 0:
            cell.set_facecolor(COLORS.get(list(s)[r - 1], "0.5"))
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("white" if r % 2 else "#f7f7f7")
            if bold[r - 1][c]:
                cell.set_text_props(weight="bold")
    title = (f"Results: IQM over {n} seeds, 95% bootstrap CI below; bold = best" if n >= 3 else
             "Results: mean of 2 seeds, each seed below (a confidence interval needs 3 or more); bold = best"
             if n == 2 else "Results: one seed per algorithm (no confidence interval); bold = best")
    fig.text(0.5, 1 - 0.18 / fig_h, title, ha="center", va="top", fontsize=12, weight="bold")
    fig.text(0.01, 0.12 / fig_h, "Test = the best checkpoint on 30 fresh scenarios.  Success = share of a full lap "
             "flown per 10 s test flight (autopilot: 54% at 1.5 m/s, 95% at 3 m/s).  Drones broken = real crashes "
             "while learning, after the warm-up.", ha="left", va="bottom", fontsize=9, color="0.3")
    return _save(fig, save_path)


def plot_scores(runs, save_path=None):
    """The scores fixed before the runs: IQM bar, 95% CI whisker (3+ seeds), one dot per seed."""
    s = R.summary(runs)
    algos = list(s)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (k, label, hib, fmt) in zip(axes.flat, R.SCORES):
        for i, a in enumerate(algos):
            c, lo, hi, vals = s[a][k]
            ax.bar(i, c, color=COLORS.get(a), alpha=0.85, width=0.65)
            if len(vals) >= 3:
                ax.errorbar(i, c, yerr=[[c - lo], [hi - c]], color="k", capsize=6, lw=1.5)
            ax.scatter(np.full(len(vals), i) + np.linspace(-0.15, 0.15, len(vals)), vals, color="k", s=14, zorder=3)
            ax.annotate(fmt.format(c), (i, c), textcoords="offset points", xytext=(0, 4 if c >= 0 else -12),
                        ha="center", fontsize=9)
        ax.axhline(0, color="k", lw=0.6)
        if "%" in fmt:
            from matplotlib.ticker import PercentFormatter
            ax.yaxis.set_major_formatter(PercentFormatter(1.0)); ax.set_ylim(0, 1.05)
        ax.set_xticks(range(len(algos))); ax.set_xticklabels([_name(a) for a in algos])
        ax.set_title(f"{label} ({'higher' if hib else 'lower'} = better)", fontsize=11)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Scores (bar = IQM over seeds, whisker = 95% bootstrap CI, dots = seeds)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, save_path)


def plot_algo(runs, refs=None, save_path=None, window=R.SMOOTH):
    """Quick look at one algorithm: return and success of each seed (rolling mean, faint = raw)."""
    a = runs[0]["algo"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for i, r in enumerate(runs):
        c = f"C{i}"
        for x, y in ((ax[0], np.asarray(r["eval_return"], float)),
                     (ax[1], 100.0 * R.success(r.get("eval_laps") or np.zeros(len(r["steps"]))))):
            x.plot(r["steps"], y, color=c, lw=0.8, alpha=0.3)
            x.plot(r["steps"], R.smooth(y, window), color=c, lw=2, label=f"seed {r['seed']}")
    for name, ref in (refs or {}).items():
        ls = REF_STYLE.get(name, ":")
        ax[0].axhline(ref["return"], color="k", lw=1.1, ls=ls, alpha=0.7, label=name)
        ax[1].axhline(100.0 * R.success(ref["laps"]), color="k", lw=1.1, ls=ls, alpha=0.7, label=name)
    ax[0].set_ylabel("evaluation return"); ax[1].set_ylabel("success: % of a lap per 10 s flight")
    ax[1].set_ylim(-2, 102)
    for x in ax:
        x.set_xlabel("real environment steps"); x.grid(alpha=0.3); x.legend(fontsize=8)
    fig.suptitle(f"{_name(a)}: learning curve per seed (rolling mean over {window} evaluations, faint = raw)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, save_path)


def _rolling(steps, vals, window):
    vals = np.asarray(vals, dtype=float)
    out = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        w = vals[max(0, i - window + 1): i + 1]
        if np.isfinite(w).any():
            out[i] = np.nanmean(w)
    return np.asarray(steps), out


def plot_model_trust(runs, save_path=None, smooth=8, t_max=10):
    """Does MACURA trust its world model 100%? A: share of a full imagination each method trusts;
    B: MACURA's trust in fast descents vs normal flight; C: imagined steps per trip; D: SAC updates per
    real step (MACURA's follow its trust, Eq. 22). Mean over seeds, rolling mean over `smooth` rounds."""
    by = R.by_algo(runs)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    ax = axes[0]
    for a in by:
        if a == "macura":
            st, trip = R.series(by[a], "rollout_length")
            if st is not None:
                ax.plot(*_rolling(st, 100.0 * trip / t_max, smooth), color=COLORS[a], lw=2,
                        label="MACURA: until its models disagree")
        elif a == "mbpo":
            ax.axhline(100.0, color=COLORS[a], lw=2, ls="--", label="MBPO: keeps all (no check)")
        elif a == "m2ac":
            ax.axhline(25.0, color=COLORS[a], lw=2, ls="--", label="M2AC: keeps the 25% most certain")
    ax.set_ylim(0, 105); ax.set_ylabel("imagined steps trusted (%)"); ax.set_title("A. How much imagination it trusts")
    ax = axes[1]
    for key, color, lab in (("trust_slow", COLORS["macura"], "normal flight"),
                            ("trust_fast", "#ff7f0e", "fast descents (the chute)")):
        st, m = R.series(by.get("macura", []), key)
        if st is not None:
            ax.plot(*_rolling(st, 100.0 * m, smooth), color=color, lw=2, label=lab)
    ax.set_ylim(0, 105); ax.set_ylabel("imagined steps MACURA trusts (%)"); ax.set_title("B. Where MACURA trusts it")
    ax = axes[2]
    for a in by:
        st, m = R.series(by[a], "rollout_length")
        if st is not None and a != "sac":
            ax.plot(*_rolling(st, m, smooth if a == "macura" else 1), color=COLORS[a], lw=2,
                    ls="--" if a == "m2ac" else "-", label=_name(a))
    ax.set_ylim(0, t_max + 0.5); ax.set_ylabel(f"imagined steps per trip (max {t_max})")
    ax.set_title("C. How far ahead it imagines")
    ax = axes[3]
    for a, rs in by.items():
        rs = [r for r in rs if r.get("utd")]
        if rs:
            n = min(len(r["utd"]) for r in rs)
            ax.plot(rs[0]["steps"][:n], np.mean([r["utd"][:n] for r in rs], axis=0), color=COLORS[a], lw=2,
                    ls="--" if a == "m2ac" else "-", label=_name(a))
        elif a == "sac":
            ax.axhline(1, color=COLORS[a], lw=2, label="SAC")
    ax.set_ylim(bottom=0); ax.set_ylabel("SAC updates per real step"); ax.set_title("D. Updates per real step")
    for ax in axes:
        ax.set_xlabel("real environment steps"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Does MACURA trust its world model 100%?  (mean over seeds, rolling mean over {smooth} model rounds)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, save_path)


def plot_kappa(runs, save_path=None):
    """Paper Fig. 8: MACURA's threshold kappa and the zeta-quantile of first-step disagreement per model
    round, and the resulting imagined trip length, per seed."""
    rs = [r for r in runs if r["algo"] == "macura" and r.get("kappa")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for i, r in enumerate(rs):
        c = plt.cm.Blues(0.45 + 0.5 * i / max(1, len(rs) - 1))
        k, b, t = (np.asarray(r[key], dtype=float) for key in ("kappa", "base_uncertainty", "rollout_length"))
        ax[0].plot(k[:, 0], k[:, 1], color=c, lw=2, label=f"kappa, seed {r['seed']}")
        ax[0].plot(b[:, 0], b[:, 1], color=c, lw=0.8, alpha=0.6, label=f"95% quantile of first-step GJS, seed {r['seed']}")
        ax[1].plot(*_rolling(t[:, 0], t[:, 1], 8), color=c, lw=2, label=f"seed {r['seed']}")
    ax[0].set_yscale("log"); ax[0].set_ylabel("GJS disagreement")
    ax[0].set_title("Trust threshold kappa (Eq. 21) vs this round's disagreement")
    ax[1].set_ylim(0, 10.5); ax[1].set_ylabel("average imagined trip (steps of 10)")
    ax[1].set_title("Resulting trip length (rolling mean, 8 rounds)")
    for a in ax:
        a.set_xlabel("real environment steps"); a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.suptitle("MACURA's adaptive threshold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, save_path)


def plot_untrusted_data(runs, save_path=None, smooth=8):
    """Imagined training data above MACURA's trust threshold (MBPO / M2AC measured with MACURA's rule;
    MACURA trains on none of it, dashed = what it generated and dropped) and the share of imagined
    training data that is a fast descent."""
    by = R.by_algo(runs)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for a in ("mbpo", "m2ac"):
        st, m = R.series(by.get(a, []), "untrusted_frac")
        if st is not None:
            ax[0].plot(*_rolling(st, 100 * m, smooth), color=COLORS[a], lw=2, label=f"{_name(a)}: trains on it")
    st, m = R.series(by.get("macura", []), "discarded_frac")
    if st is not None:
        ax[0].axhline(0, color=COLORS["macura"], lw=2, label="MACURA: trains on none")
        ax[0].plot(*_rolling(st, 100 * m, smooth), color=COLORS["macura"], lw=1.5, ls="--",
                   label="MACURA: imagined, then dropped")
    ax[0].set_ylim(-2, 102); ax[0].set_ylabel("% of imagined steps")
    ax[0].set_title("Imagined data the models disagree about")
    for a in by:
        st, m = R.series(by[a], "fast_frac")
        if st is not None:
            ax[1].plot(*_rolling(st, 100 * m, smooth), color=COLORS[a], lw=2, label=_name(a))
    ax[1].set_ylabel("% of imagined training data"); ax[1].set_title("Imagined fast descents it trains on")
    for a in ax:
        a.set_xlabel("real environment steps"); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_model_check(checks, save_path=None):
    """Paper Fig. 10 on real flights: each step's ensemble disagreement (GJS) vs the model's actual one-step
    error, with MACURA's threshold kappa. `checks` = model_check() results (one per run)."""
    checks = [c for c in checks if c]
    fig, axes = plt.subplots(1, len(checks) + 1, figsize=(6.2 * (len(checks) + 1), 4.8), squeeze=False)
    axes = axes[0]
    for ax, c in zip(axes, checks):
        g, e = np.maximum(c["gjs"], 1e-4), np.maximum(c["error"], 1e-4)
        ax.scatter(g[~c["fast"]], e[~c["fast"]], s=3, alpha=0.25, color="#1f77b4", label="normal flight")
        ax.scatter(g[c["fast"]], e[c["fast"]], s=6, alpha=0.6, color="#ff7f0e", label="fast descent")
        if c["kappa"]:
            ax.axvline(c["kappa"], color="k", lw=1.5, ls="--", label=f"kappa = {c['kappa']:.2g}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("ensemble disagreement GJS (what MACURA sees)")
        ax.set_ylabel("actual one-step model error (normalised RMS)")
        ax.set_title(f"{_name(c['algo'])} seed {c['seed']}: {c['steps']} real steps, "
                     f"{100 * c['trusted']:.0f}% below kappa", fontsize=10)
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, markerscale=3)
    ax = axes[-1]
    labels, vals, cols = [], [], []
    for c in checks:
        if not c["kappa"]:
            continue
        t = c["gjs"] < c["kappa"]
        for sel, lab in ((t, "trusted"), (~t, "not trusted")):
            labels.append(f"{_name(c['algo'])} s{c['seed']}\n{lab}")
            vals.append(float(np.median(c["error"][sel])) if sel.any() else np.nan)
            cols.append(COLORS.get(c["algo"]) if lab == "trusted" else "#ff7f0e")
    ax.bar(range(len(vals)), vals, color=cols)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("median one-step model error"); ax.set_title("Model error: trusted vs not trusted steps")
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Does the disagreement flag the steps where the world model is actually wrong?", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, save_path)


def plot_crash_by_payload(runs, save_path=None):
    """Final-test crash rate with a light vs a heavy package (split at half the maximum)."""
    by = R.by_algo(runs)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    xs = np.arange(len(by))
    for off, key, lab, hatch in ((-0.18, "eval_failure_light", "light package", ""),
                                 (0.18, "eval_failure_heavy", "heavy package", "//")):
        vals = [[r["final_eval"].get(key, np.nan) for r in rs] for rs in by.values()]
        ax.bar(xs + off, [np.nanmean(v) for v in vals], 0.34, color=[COLORS.get(a) for a in by], hatch=hatch,
               edgecolor="white", label=lab)
    ax.set_xticks(xs); ax.set_xticklabels([_name(a) for a in by]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("final test crash rate"); ax.set_title("Crashes with a light vs a heavy package")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    return _save(fig, save_path)


# ── the task ──────────────────────────────────────────────────────────────────────────────────────
def plot_race_course(cfg, save_path=None):
    """The course from above (colour = height, red = the chute, gates, pillars) and its height along a lap."""
    from src.envs.race_course import build_course
    c = build_course(cfg["env"])
    P, ch = c["points"], c["chute"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.35]})
    sc = ax[0].scatter(P[:, 0], P[:, 1], c=P[:, 2], cmap="viridis", s=9)
    ax[0].scatter(P[ch, 0], P[ch, 1], color="red", s=30, label="chute (steep 3 m drop)")
    for i, (g, t) in enumerate(c["gates"]):
        ax[0].plot(g[0], g[1], "o", ms=13, mfc="none", mec="darkorange", mew=2.5)
        ax[0].annotate(f"P{i + 1}  {g[2]:.1f} m", g[:2], textcoords="offset points",
                       xytext=(10, 8) if g[0] < 1.0 else (-80, 8), fontsize=10)
        ax[0].annotate("", g[:2] + 0.8 * t[:2], g[:2], arrowprops=dict(arrowstyle="->", lw=1.6, color="k"))
    for (x, y) in cfg["env"].get("race_pillars", []):
        ax[0].add_patch(plt.Circle((x, y), float(cfg["env"].get("obstacle_radius", 0.3)), color="firebrick",
                                   alpha=0.85))
    ax[0].set_aspect("equal"); ax[0].grid(alpha=0.3); ax[0].legend(loc="center", fontsize=9)
    ax[0].set_title("Course from above (clockwise), colour = height")
    fig.colorbar(sc, ax=ax[0], fraction=0.046, label="height (m)")
    ax[1].plot(c["s"], P[:, 2], color="0.3", lw=2)
    ax[1].plot(c["s"][ch], P[ch, 2], color="red", lw=4, label="chute: drop it too fast -> lift loss")
    for i in range(4):
        j = int(np.argmin(np.linalg.norm(P - c["gates"][i][0], axis=1)))
        ax[1].axvline(c["s"][j], color="darkorange", ls="--", lw=1)
        ax[1].text(c["s"][j] + 0.2, 4.25, f"P{i + 1}", color="darkorange")
    ax[1].axhline(float(cfg["env"].get("race_floor", 0.12)), color="k", lw=1)
    ax[1].set_ylim(0, 4.6); ax[1].set_xlabel("distance along the lap (m)"); ax[1].set_ylabel("height (m)")
    ax[1].set_title(f"Height along one lap ({c['length']:.1f} m); reward = speed along the course")
    ax[1].legend(loc="center right"); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_lift_loss(cfg, save_path=None):
    """The danger: lift available vs sink speed, and the best braking left with each package weight."""
    from src.envs import drone_env
    env, _, _ = drone_env.make_env(cfg["env"], seed=0)
    c = cfg["env"]
    top = env.n_act * min(env._hover * (1.0 + env.thrust_gain), env._ctrl_hi)
    full, m0 = 1.0 + env.thrust_gain, env._mass0
    env.close()

    def lift(sink, sideways=0.0, ratio=full):
        return drone_env.lift_factor([sideways, 0.0, -sink], [1.0, 0.0, 0.0, 0.0], ratio, c["vrs_loss"], c["vrs_speed"],
                                     c["vrs_full"], c["vrs_escape"], c["vrs_upright"], c["vrs_thrust"])

    v = np.linspace(0.0, 3.0, 200)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for sideways, ratio, ls, label in ((0.0, full, "-", "straight down, rotors pushing"),
                                       (c["vrs_escape"], full, "--", f"also {c['vrs_escape']:.1f} m/s sideways"),
                                       (0.0, 0.0, ":", "straight down, motors off")):
        ax[0].plot(v, [100 * lift(x, sideways, ratio) for x in v], ls, color="#d62728", lw=2, label=label)
    ax[0].axvspan(c["vrs_speed"], c["vrs_full"], color="#ff7f0e", alpha=0.12, label="lift-loss zone")
    ax[0].set_xlabel("sink speed (m/s)"); ax[0].set_ylabel("lift available (%)")
    ax[0].set_title("Sinking fast while pushing loses lift"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    for p, col in ((0.0, "#2ca02c"), (0.1, "#ff7f0e"), (0.2, "#d62728")):
        m = m0 + p
        ax[1].plot(v, (top * np.array([lift(x) for x in v]) - m * 9.81) / m, color=col, lw=2, label=f"package {p:.1f} kg")
    ax[1].axhline(0.0, color="k", lw=0.8)
    ax[1].set_xlabel("sink speed (m/s)"); ax[1].set_ylabel("max braking (m/s²)")
    ax[1].set_title("A heavy package leaves less to brake with"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_wind(cfg, steps=250, seed=0, save_path=None):
    """Hidden process noise: OU wind gusts (and per-rotor thrust noise), real dynamics only."""
    ec = cfg["env"]
    wf, wc, act = float(ec.get("wind_force", 0.0)), float(ec.get("wind_correlation", 0.95)), float(ec.get("actuator_noise", 0.0))
    hover = 4.3
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for sd in (seed, seed + 1, seed + 2):
        rng = np.random.default_rng(sd); w = 0.0; trace = []
        for _ in range(steps):
            w = wc * w + np.sqrt(1 - wc ** 2) * rng.normal(0, wf); trace.append(w)
        ax[0].plot(trace, lw=1.3, alpha=0.85, label=f"gust sample {sd}")
    ax[0].axhspan(-wf, wf, color="0.6", alpha=0.15, label=f"±1 std ({wf} N)")
    ax[0].set_xlabel("control step (20 ms)"); ax[0].set_ylabel("gust force, one axis (N)")
    ax[0].set_title(f"Hidden wind gusts (OU, correlation {wc})"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    rng = np.random.default_rng(seed); w = np.zeros(3); mags = []
    for _ in range(20000):
        w = wc * w + np.sqrt(1 - wc ** 2) * rng.normal(0, wf, 3); mags.append(np.linalg.norm(w))
    ax[1].hist(100 * np.array(mags) / hover, bins=40, color="#1f77b4", edgecolor="white")
    ax[1].set_xlabel("gust force as % of hover thrust"); ax[1].set_ylabel("count")
    ax[1].set_title(f"Gust strength (+ {act * 100:.0f}% random thrust noise per rotor)"); ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _save(fig, save_path)
