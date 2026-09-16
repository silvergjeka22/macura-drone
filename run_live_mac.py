"""Entry point: watch a trained drone policy fly live on macOS.

This is the one executable script in the repo (the local analogue of the notebook):
the library modules stay pure functions; this file only orchestrates.

macOS: run it with `mjpython` (bundled with the `mujoco` pip package), because
MuJoCo's interactive viewer must own the main thread:

    mjpython run_live_mac.py --ckpt runs/checkpoints/macura_seed0_best.zip
    mjpython run_live_mac.py --compare runs/checkpoints          # fly all four in turn

Setup on the Mac (once):
    python -m pip install mujoco stable-baselines3 torch gymnasium numpy
Then copy the best .zip checkpoints trained on Kaggle into a local folder and
point --ckpt / --compare at them.
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import config as cfg
from src.viz import live_viewer


def main():
    p = argparse.ArgumentParser(description="Watch a trained drone policy fly live.")
    p.add_argument("--ckpt", help="path to one best .zip checkpoint to fly")
    p.add_argument("--compare", help="a folder of *_best.zip checkpoints to fly one after another")
    p.add_argument("--seconds", type=float, default=20.0, help="seconds to fly each policy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.compare:
        paths = sorted(glob.glob(os.path.join(args.compare, "*_best.zip")))
        ckpts = {os.path.basename(pth).replace("_best.zip", ""): pth for pth in paths}
        if not ckpts:
            raise SystemExit(f"no *_best.zip checkpoints found in {args.compare}")
        live_viewer.fly_sequence(ckpts, cfg.ENV, seconds_each=args.seconds,
                                 seed=args.seed, device=args.device)
    elif args.ckpt:
        dist = live_viewer.fly_policy(args.ckpt, cfg.ENV, seconds=args.seconds,
                                      seed=args.seed, device=args.device)
        print(f"best distance-to-target: {dist:.2f} m")
    else:
        raise SystemExit("pass --ckpt <best.zip> or --compare <checkpoints_dir>")


if __name__ == "__main__":
    main()
