"""Watch the trained drones race live in MuJoCo's 3-D viewer (macOS: mjpython simulate.py).

    mjpython simulate.py
    mjpython simulate.py --algos macura mbpo --seconds 30 --follow MACURA
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description="Race the trained drones live in the MuJoCo viewer.")
    p.add_argument("--runs", nargs="*", help="downloaded output folders (default: everything in ./downloads)")
    p.add_argument("--algos", nargs="*", default=["macura", "mbpo", "m2ac", "sac"])
    p.add_argument("--seed", type=int, help="fly this seed (default: each algorithm's best seed)")
    p.add_argument("--scenarios", nargs="*", type=int, default=[1000, 1001, 1002, 1003, 1004],
                   help="scenario seeds (1000+ = the fresh test scenarios)")
    p.add_argument("--seconds", type=float, help="flight length (default 10 s, as in training)")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed (0.5 = slow motion)")
    p.add_argument("--autopilot", action="store_true", help="add the hand-written autopilot (not learned)")
    p.add_argument("--follow", help="camera follows this drone, e.g. MACURA")
    p.add_argument("--no-trust", action="store_true", help="skip MACURA's live world-model check")
    args = p.parse_args()

    from src.config import config as cfg
    from src.analysis import results as R
    from src.viz import simulator

    roots = args.runs or sorted(glob.glob("downloads/*")) or sorted(glob.glob("out*")) or ["."]
    runs = R.load_runs(*roots, algorithms=args.algos)
    if not runs and not args.autopilot:
        raise SystemExit(f"no run logs found under {roots}: pass --runs <downloaded output folder>")
    pilots, trust = {}, {}
    for algo, rs in R.by_algo(runs).items():
        if args.seed is not None:
            rs = [r for r in rs if r["seed"] == args.seed]
            if not rs:
                print(f"  {algo}: no seed {args.seed}, skipped")
                continue
        run = R.best_run(rs)
        ckpt = R.checkpoint(run)
        if not ckpt or not os.path.exists(ckpt):
            print(f"  {algo} seed {run['seed']}: checkpoint not found ({ckpt}), skipped")
            continue
        label = R.NAMES.get(algo, algo)
        pilots[label] = ckpt
        print(f"  {label:7s} seed {run['seed']}  (best at step {run.get('best_step')})  {ckpt}")
        if algo == "macura" and not args.no_trust and R.ensemble_checkpoint(run):
            trust[label] = {"ensemble": R.ensemble_checkpoint(run), "kappa": R.kappa_at(run)}
    if args.autopilot:
        pilots["autopilot 1.5 m/s"] = "autopilot:0.8:1.5"
        pilots["autopilot 3 m/s"] = "autopilot:1.1:3.0"
    if not pilots:
        raise SystemExit("nothing to fly")
    simulator.run(cfg.CFG, pilots, scenarios=args.scenarios, seconds=args.seconds, speed=args.speed,
                  trust=trust, follow=args.follow)


if __name__ == "__main__":
    main()
