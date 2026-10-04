"""Slide figures for Part II of the presentation (large text, one message per figure).
Run from the repository root:  python3 paper/presentation/make_figures.py"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.analysis import results as R, testing as T
from src.config import config as cfg
from src.envs.race_course import build_course

OUT = "paper/presentation/figures/project"
ALGOS = ["macura", "m2ac", "mbpo", "sac"]
COLORS = {"macura": "#1565C0", "mbpo": "#C62828", "m2ac": "#2E7D32", "sac": "#8A8A8A"}
NAME = R.NAMES
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.labelsize": 13,
                     "legend.fontsize": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 200, "savefig.bbox": "tight"})

RUNS = R.load_runs(*R.find_downloads("downloads"), verbose=False)
BY = R.by_algo(RUNS)
TESTS = json.load(open("results/test_results.json"))
AUTOPILOT = json.load(open("results/autopilot_reference.json"))["autopilot 1.5 m/s"]
# return while learning, MACURA - MBPO, seeds 0-3: the race3 test (docs/EXPERIMENTS.md; those runs are not downloaded)
EARLIER_SEEDS = {0: 152, 1: -106, 2: 107, 3: 31}


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)
    print("saved", name)


def rolling(steps, vals, window=8):
    vals = np.asarray(vals, float)
    out = np.array([np.nanmean(vals[max(0, i - window + 1): i + 1]) for i in range(len(vals))])
    return np.asarray(steps), out


# 1. learning curves: return, lap progress, drones broken
def learning():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    panels = [("eval_return", "iqm", lambda m: m, "Return while learning", "evaluation return"),
              ("eval_laps", "iqm", lambda m: 100 * R.success(m), "Lap progress", "% of a lap per 10 s flight"),
              ("train_crashes", "mean", lambda m: m, "Drones broken while learning", "crashes so far")]
    for a in ALGOS:
        for x, (key, stat, fn, _, _) in zip(ax, panels):
            steps, mat = R.curves(BY[a], key)
            mat = fn(mat)
            if key != "train_crashes":
                mat = np.array([R.smooth(row, 7) for row in mat])
            c, lo, hi = R.band(mat, stat)
            x.plot(steps / 1000, c, color=COLORS[a], lw=2.5, label=NAME[a])
            x.fill_between(steps / 1000, lo, hi, color=COLORS[a], alpha=0.15, lw=0)
    ax[0].axhline(AUTOPILOT["return"], color="k", ls=":", lw=1.5, label="autopilot 1.5 m/s")
    ax[1].axhline(100 * AUTOPILOT["laps"], color="k", ls=":", lw=1.5)
    ax[1].set_ylim(0, 60)
    for x, (_, _, _, title, ylab) in zip(ax, panels):
        x.set_title(title); x.set_ylabel(ylab); x.set_xlabel("real steps (thousands)")
        x.axvspan(0, 5, color="0.9", lw=0)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save(fig, "slide_learning")


# 2. test return, MACURA minus each baseline per seed, and all 8 seeds against MBPO while learning
def results():
    scores = {a: [T.seed_scores(t) for t in TESTS if t["algo"] == a] for a in ALGOS}
    by = {(t["algo"], t["seed"]): T.seed_scores(t)["return"] for t in TESTS}
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1, 1.1, 1.35]})
    ticks = []
    for i, a in enumerate(ALGOS):
        v = np.array([s["return"] for s in scores[a]])
        c = R.iqm(v)
        ax[0].bar(i, c, color=COLORS[a], alpha=0.85, width=0.65)
        ax[0].scatter(i + np.linspace(-0.15, 0.15, len(v)), v, color="k", s=18, zorder=3)
        ticks.append(f"{NAME[a]}\n{c:.0f}")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xticks(range(len(ALGOS)), ticks); ax[0].grid(axis="x", visible=False)
    ax[0].get_xticklabels()[0].set_fontweight("bold")
    ax[0].set_title("A. Test return (50 new scenarios)")
    sd, w = [4, 5, 6, 7], 0.38
    for j, b in enumerate(["mbpo", "m2ac"]):
        d = [by[("macura", s)] - by[(b, s)] for s in sd]
        bars = ax[1].bar(np.arange(4) + (j - 0.5) * w, d, w, color=COLORS[b], label=f"vs {NAME[b]}")
        for bar, v in zip(bars, d):
            ax[1].text(bar.get_x() + bar.get_width() / 2, v + (5 if v >= 0 else -5), f"{v:+.0f}", ha="center",
                       va="bottom" if v >= 0 else "top", fontsize=10)
    ax[1].axhline(0, color="k", lw=1); ax[1].set_ylim(-60, 290)
    ax[1].set_xticks(range(4), [f"seed {s}" for s in sd]); ax[1].grid(axis="x", visible=False)
    ax[1].set_title("B. Test return, MACURA minus ..."); ax[1].legend(frameon=False, loc="upper right")
    learn = {}
    for s in sd:
        m = [r for r in RUNS if r["algo"] == "macura" and r["seed"] == s][0]
        b = [r for r in RUNS if r["algo"] == "mbpo" and r["seed"] == s][0]
        learn[s] = round(np.mean(m["eval_return"]) - np.mean(b["eval_return"]))
    allseeds = {**EARLIER_SEEDS, **learn}
    xs, vals = list(allseeds), list(allseeds.values())
    ax[2].bar(xs, vals, color=[COLORS["macura"] if v > 0 else COLORS["mbpo"] for v in vals], width=0.65)
    for x, v in zip(xs, vals):
        ax[2].text(x, v + (8 if v >= 0 else -8), f"{v:+d}", ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
    ax[2].axhline(0, color="k", lw=1); ax[2].axvline(3.5, color="0.5", ls="--", lw=1)
    ax[2].text(1.5, 455, "race3 test", ha="center", color="0.35", fontsize=11)
    ax[2].text(5.5, 455, "final study", ha="center", color="0.35", fontsize=11)
    ax[2].set_xticks(xs, [str(s) for s in xs]); ax[2].set_xlabel("seed"); ax[2].grid(axis="x", visible=False)
    ax[2].set_ylim(-150, 500)
    ax[2].set_title(f"C. While learning, MACURA $-$ MBPO: {sum(v > 0 for v in vals)} of 8")
    fig.tight_layout()
    save(fig, "slide_results")


# 3. trust: where MACURA trusts its model, and how far it imagines
def trust():
    runs = sorted(BY["macura"], key=lambda r: r["seed"])
    fast = [100 * np.nanmean([v for _, v in r["trust_fast"]]) for r in runs]
    slow = [100 * np.nanmean([v for _, v in r["trust_slow"]]) for r in runs]
    labels = [f"seed {r['seed']}" for r in runs] + ["mean"]
    fast.append(np.mean(fast)); slow.append(np.mean(slow))
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
    x, w = np.arange(len(labels)), 0.38
    b1 = ax[0].bar(x - w / 2, slow, w, color=COLORS["macura"], label="normal flight")
    b2 = ax[0].bar(x + w / 2, fast, w, color="#EF6C00", label="fast descents (the chute)")
    for bars in (b1, b2):
        for bar in bars:
            ax[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.0f}",
                       ha="center", va="bottom", fontsize=11)
    ax[0].set_xticks(x, labels); ax[0].set_ylim(0, 115); ax[0].grid(axis="x", visible=False)
    ax[0].set_ylabel("% of imagined steps trusted"); ax[0].set_title("Where MACURA trusts its model")
    ax[0].legend(frameon=False, loc="upper center", ncol=2)
    for r in runs:
        st, v = np.asarray(r["rollout_length"], float).T
        ax[1].plot(*rolling(st / 1000, v), color=COLORS["macura"], lw=1, alpha=0.35)
    st, m = R.series(runs, "rollout_length")
    ax[1].plot(*rolling(st / 1000, m), color=COLORS["macura"], lw=3, label="MACURA (mean of seeds)")
    sched = cfg.ROLLOUT["mbpo"]["rollout_schedule"]
    xs = np.linspace(5, 50, 200)
    ax[1].plot(xs, [min(sched[1], max(sched[0], sched[0] + (x * 1000 - sched[2]) / (sched[3] - sched[2]) * (sched[1] - sched[0])))
                    for x in xs], color=COLORS["mbpo"], lw=2.5, ls="--", label="MBPO (fixed schedule)")
    ax[1].set_ylim(0, 10.5); ax[1].set_xlabel("real steps (thousands)"); ax[1].set_ylabel("steps per imagined trip")
    ax[1].set_title("How far each method imagines"); ax[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "slide_trust")


# 4. what each method trains on
def untrusted():
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
    for a in ("mbpo", "m2ac"):
        st, m = R.series(BY[a], "untrusted_frac")
        ax[0].plot(*rolling(st / 1000, 100 * m), color=COLORS[a], lw=2.5, label=f"{NAME[a]}: trains on it")
    ax[0].axhline(0, color=COLORS["macura"], lw=3, label="MACURA: trains on none")
    st, m = R.series(BY["macura"], "discarded_frac")
    ax[0].plot(*rolling(st / 1000, 100 * m), color=COLORS["macura"], lw=1.8, ls="--", label="MACURA: imagined, then dropped")
    ax[0].set_ylim(-1, 32); ax[0].set_ylabel("% of imagined steps")
    ax[0].set_title("Imagined data the networks disagree about")
    for a in ("mbpo", "m2ac", "macura"):
        st, m = R.series(BY[a], "fast_frac")
        ax[1].plot(*rolling(st / 1000, 100 * m), color=COLORS[a], lw=2.5, label=NAME[a])
    ax[1].set_ylabel("% of imagined training data"); ax[1].set_title("Imagined fast descents it trains on")
    for x in ax:
        x.set_xlabel("real steps (thousands)"); x.legend(frameon=False)
    fig.tight_layout()
    save(fig, "slide_untrusted")


# 5. one test flight, best seed of each algorithm (scenario 3000)
def flight():
    best = {}
    for a in ALGOS:
        ts = [t for t in TESTS if t["algo"] == a]
        s = max(ts, key=lambda t: T.seed_scores(t)["return"])["seed"]
        best[a] = [r for r in RUNS if r["algo"] == a and r["seed"] == s][0]
    traces = {a: T.test_model(r, cfg.CFG, episodes=1, traces=1)["traces"][0] for a, r in best.items()}
    course = build_course(cfg.ENV)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.0, 1.25, 1.25]})
    ax[0].plot(course["points"][:, 0], course["points"][:, 1], color="0.82", lw=7, zorder=0)
    ch = course["chute"]
    ax[0].plot(course["points"][ch, 0], course["points"][ch, 1], color="#F4A582", lw=7, zorder=0)
    for (px, py) in cfg.ENV["race_pillars"]:
        ax[0].add_patch(plt.Circle((px, py), cfg.ENV["obstacle_radius"], color="#B71C1C", alpha=0.8))
    ax[0].text(1.3, 2.35, "chute", color="#D6604D", fontsize=12, ha="center")
    for a, t in traces.items():
        time = np.arange(len(t["sink"])) * 0.02
        ax[0].plot(t["pos"][:, 0], t["pos"][:, 1], color=COLORS[a], lw=2.2)
        ax[0].plot(*t["pos"][-1, :2], "o", color=COLORS[a], ms=7)
        ax[1].plot(time, t["sink"], color=COLORS[a], lw=1.8, label=f"{NAME[a]} (seed {best[a]['seed']})")
        ax[2].plot(time, 100 * t["eff"], color=COLORS[a], lw=2.2, label=NAME[a])
    ax[0].set_aspect("equal"); ax[0].set_title("Path from above (dot = end)"); ax[0].grid(False)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[1].axhspan(1.2, 2.2, color="#F4A582", alpha=0.3, lw=0)
    ax[1].text(0.2, 1.3, "lift-loss zone", color="#D6604D", fontsize=11)
    ax[1].set_ylim(-1.6, 2.2); ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("sink speed (m/s)")
    ax[1].set_title("Sink speed")
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    ax[2].set_ylim(70, 102); ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("lift available (%)")
    ax[2].set_title("Lift")
    fig.tight_layout()
    save(fig, "slide_flight")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    learning(); results(); trust(); untrusted(); flight()
