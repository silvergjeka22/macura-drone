"""Compare all seeds on your own machine after downloading the Kaggle outputs.

    python compare.py out_seed0 out_seed1 out_seed2 --out results
    python compare.py out_seed0 out_seed1 out_seed2 --out results --videos     # + race videos (a few minutes)

Writes results/summary.md (tables), results/plots/*.png and, with --videos, results/videos/*.mp4.
"""

import argparse
import os
import sys

os.environ.setdefault("MACURA_TASK", "race2")
os.environ.setdefault("MUJOCO_GL", "disable")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description="Merge the seeds and build the comparison.")
    p.add_argument("runs", nargs="+", help="downloaded output folders (each holds runs/logs, runs/checkpoints)")
    p.add_argument("--out", default="results")
    p.add_argument("--videos", action="store_true", help="also render the race videos")
    p.add_argument("--no-check", action="store_true", help="skip the world-model check on real flights")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    from src.config import config as cfg
    from src.analysis import results as R, report

    runs = R.load_runs(*args.runs)
    if not runs:
        raise SystemExit(f"no run logs under {args.runs}")
    out = report.build(runs, cfg.CFG, args.out, check=not args.no_check, videos=args.videos, close=True)
    print()
    print(R.summary_markdown(runs))
    print()
    print(R.trust_summary(runs))
    print()
    for k, v in out.items():
        if v:
            print(f"{k:18s} {v}")


if __name__ == "__main__":
    main()
