"""Figures of the test on new scenarios (results of testing.test_all)."""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

from src.analysis import results as R
from src.analysis.testing import seed_scores

COLORS = {"macura": "#1f77b4", "mbpo": "#d62728", "m2ac": "#2ca02c", "sac": "#7f7f7f"}


def _save(fig, path):
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def _by_algo(results):
    by = {}
    for r in results:
        by.setdefault(r["algo"], []).append(r)
    return {a: sorted(by[a], key=lambda x: x["seed"]) for a in R.ALGOS if a in by}


def _best(rs):
    """The seed with the highest mean test return (for the example flights)."""
    return max(rs, key=lambda r: seed_scores(r)["return"])


def plot_test_scores(results, save_path=None):
    """Test return, crash rate and lap progress per algorithm (dots = seeds)."""
    by = _by_algo(results)
    names = [R.NAMES[a] for a in by]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    data = [[e["return"] for r in rs for e in r["episodes"]] for rs in by.values()]
    bp = ax[0].boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    for patch, a in zip(bp["boxes"], by):
        patch.set_facecolor(COLORS[a]); patch.set_alpha(0.35)
    for i, rs in enumerate(by.values(), start=1):
        ax[0].scatter(np.full(len(rs), i) + np.linspace(-0.12, 0.12, len(rs)), [seed_scores(r)["return"] for r in rs],
                      color="k", s=18, zorder=3)
    ax[0].set_xticks(range(1, len(by) + 1)); ax[0].set_xticklabels(names)
    ax[0].set_title("Test return per flight (box) and per seed (dots)"); ax[0].set_ylabel("return (10 s flight)")
    for axi, key, title in ((ax[1], "crash", "Crash rate"), (ax[2], "lap", "Lap progress per 10 s flight")):
        for i, (a, rs) in enumerate(by.items()):
            v = [100 * seed_scores(r)[key] for r in rs]
            axi.bar(i, np.mean(v), color=COLORS[a], alpha=0.8, width=0.6)
            axi.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color="k", s=18, zorder=3)
            axi.annotate(f"{np.mean(v):.0f}%", (i, np.mean(v)), textcoords="offset points", xytext=(0, 4), ha="center")
        axi.set_xticks(range(len(by))); axi.set_xticklabels(names); axi.set_ylabel("%"); axi.set_title(title)
    for a in ax:
        a.grid(alpha=0.3, axis="y")
    n = len(results[0]["episodes"]) if results else 0
    fig.suptitle(f"Test on {n} new scenarios (best checkpoint of every seed)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, save_path)


def plot_test_conditions(results, save_path=None):
    """Crash rate by package weight and by steady-wind strength (all seeds together)."""
    by = _by_algo(results)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for axi, key, edges, label in ((ax[0], "payload", [0, 0.067, 0.133, 0.21], "package (kg)"),
                                   (ax[1], "wind", [0, 0.167, 0.333, 0.51], "steady wind (N)")):
        mids = [f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in range(len(edges) - 1)]
        for a, rs in by.items():
            eps = [e for r in rs for e in r["episodes"]]
            rate = []
            for i in range(len(edges) - 1):
                sel = [e["crashed"] for e in eps if edges[i] <= e[key] < edges[i + 1]]
                rate.append(100 * np.mean(sel) if sel else np.nan)
            axi.plot(mids, rate, "o-", color=COLORS[a], lw=2, label=R.NAMES[a])
        axi.set_xlabel(label); axi.set_ylabel("crash rate (%)"); axi.grid(alpha=0.3); axi.legend(fontsize=8)
    ax[0].set_title("Crashes by package weight"); ax[1].set_title("Crashes by steady wind")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_test_flights(results, cfg, trace=0, save_path=None):
    """The same test flight for every algorithm's best seed: path, height, sink speed and lift."""
    from src.envs.race_course import build_course
    by = _by_algo(results)
    c = build_course(cfg["env"])
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.8), gridspec_kw={"width_ratios": [1.1, 1, 1, 1]})
    ax[0].plot(c["points"][:, 0], c["points"][:, 1], color="0.75", lw=6, zorder=0, label="course")
    ch = c["chute"]
    ax[0].plot(c["points"][ch, 0], c["points"][ch, 1], color="salmon", lw=6, zorder=0, label="chute")
    for (x, y) in cfg["env"].get("race_pillars", []):
        ax[0].add_patch(plt.Circle((x, y), float(cfg["env"].get("obstacle_radius", 0.3)), color="firebrick", alpha=0.8))
    scen = None
    for a, rs in by.items():
        r = _best(rs)
        if len(r["traces"]) <= trace:
            continue
        t = r["traces"][trace]
        scen = t["scenario"]
        time = np.arange(len(t["sink"])) * 0.02
        lab = f"{R.NAMES[a]} (seed {r['seed']}{', crash' if t['crashed'] else ''})"
        ax[0].plot(t["pos"][:, 0], t["pos"][:, 1], color=COLORS[a], lw=2, label=lab)
        ax[0].plot(*t["pos"][0, :2], "o", color=COLORS[a])
        ax[1].plot(time, t["pos"][:, 2], color=COLORS[a], lw=2, label=R.NAMES[a])
        ax[2].plot(time, t["sink"], color=COLORS[a], lw=1.5, label=R.NAMES[a])
        ax[3].plot(time, 100 * t["eff"], color=COLORS[a], lw=1.5, label=R.NAMES[a])
    ax[0].set_aspect("equal"); ax[0].set_title("Path from above (dot = start)")
    ax[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax[1].set_ylabel("height (m)"); ax[1].set_title("Height")
    ax[2].axhline(1.2, color="k", ls="--", lw=1, label="lift loss starts"); ax[2].set_ylabel("sink speed (m/s)")
    ax[2].set_title("Sink speed")
    ax[3].set_ylabel("lift available (%)"); ax[3].set_title("Lift (drops in the lift-loss zone)")
    for a in ax[1:]:
        a.set_xlabel("time (s)"); a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.suptitle(f"One test flight (scenario {scen}), best seed of each algorithm", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, save_path)


def plot_test_noise(results, trace=0, save_path=None):
    """The hidden gusts, steady wind and package of one test flight."""
    r = next((x for x in results if len(x["traces"]) > trace), None)
    if r is None:
        return None
    t = r["traces"][trace]
    time = np.arange(len(t["gust"])) * 0.02
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    for i, (lab, col) in enumerate((("x", "#1f77b4"), ("y", "#ff7f0e"), ("z", "#2ca02c"))):
        ax[0].plot(time, t["gust"][:, i], color=col, lw=1.2, label=f"gust {lab}")
    for i, lab in enumerate(("x", "y")):
        ax[0].axhline(t["wind_mean"][i], color="k", ls="--" if i == 0 else ":", lw=1, label=f"steady wind {lab}")
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("force (N)"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7, ncol=2)
    ax[0].set_title(f"Wind in scenario {t['scenario']} (package {t['payload']:.2f} kg)")
    gust = np.concatenate([np.linalg.norm(x["traces"][trace]["gust"], axis=1) for x in results if len(x["traces"]) > trace])
    ax[1].hist(gust, bins=40, color="#1f77b4", edgecolor="white")
    ax[1].set_xlabel("gust strength (N)"); ax[1].set_ylabel("steps"); ax[1].grid(alpha=0.3, axis="y")
    ax[1].set_title("Gust strength over the recorded test flights (+4% random thrust noise per rotor)")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_test_trust(results, save_path=None):
    """MACURA's disagreement vs kappa during its test flights, and the share of trusted steps."""
    mac = [r for r in results if r["algo"] == "macura" and r.get("kappa") and r["traces"] and "gjs" in r["traces"][0]]
    if not mac:
        return None
    best = _best(mac)
    t = best["traces"][0]
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    time = np.arange(len(t["gjs"])) * 0.02
    sc = ax[0].scatter(time, np.maximum(t["gjs"], 1e-3), c=t["sink"], cmap="coolwarm", s=8, vmin=-1.5, vmax=1.5)
    ax[0].axhline(best["kappa"], color="k", ls="--", lw=1.5, label=f"kappa = {best['kappa']:.2g}")
    ax[0].set_yscale("log"); ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("disagreement GJS")
    ax[0].set_title(f"MACURA seed {best['seed']}, scenario {t['scenario']}: its models' disagreement at every step")
    fig.colorbar(sc, ax=ax[0], label="sink speed (m/s)"); ax[0].legend(); ax[0].grid(alpha=0.3, which="both")
    for i, r in enumerate(mac):
        v = [100 * np.mean(tr["gjs"] < r["kappa"]) for tr in r["traces"] if "gjs" in tr]
        ax[1].bar(i, np.mean(v), color=COLORS["macura"], alpha=0.8)
        ax[1].scatter(np.full(len(v), i), v, color="k", s=14, zorder=3)
    ax[1].set_xticks(range(len(mac))); ax[1].set_xticklabels([f"seed {r['seed']}" for r in mac])
    ax[1].set_ylim(0, 105); ax[1].set_ylabel("steps below kappa (%)"); ax[1].grid(alpha=0.3, axis="y")
    ax[1].set_title("How much of the real test flights MACURA's model trusts")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_test_seeds(results, save_path=None):
    """Every seed's mean test return with its standard error (over the test scenarios)."""
    by = _by_algo(results)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = 0
    ticks, labels = [], []
    for a, rs in by.items():
        for r in rs:
            v = np.array([e["return"] for e in r["episodes"]])
            ax.errorbar(x, v.mean(), yerr=v.std() / np.sqrt(len(v)), fmt="o", color=COLORS[a], capsize=4)
            ticks.append(x); labels.append(f"{R.NAMES[a]}\ns{r['seed']}")
            x += 1
        x += 0.8
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("mean test return (± s.e.)")
    ax.set_title("Test return of every trained model"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _save(fig, save_path)
