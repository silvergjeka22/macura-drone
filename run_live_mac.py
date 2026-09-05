"""Entry point: watch a trained Pogo policy backflip live on macOS.

This is the ONE executable script in the repo (the local analogue of the Colab
notebook): the library modules stay pure functions; this file only orchestrates.

macOS: run it with `mjpython` (bundled with the `mujoco` pip package), because
MuJoCo's interactive viewer must own the main thread:

    mjpython run_live_mac.py --ckpt runs/checkpoints/macura_seed0_best.zip
    mjpython run_live_mac.py --compare runs/checkpoints          # fly all four in turn

Setup on the Mac (once):
    python -m pip install mujoco stable-baselines3 torch gymnasium pyyaml numpy
Then copy the best .zip checkpoints trained on Colab into a local folder and
point --ckpt / --compare at them.
"""

import argparse
import glob
import os

import yaml

from viz import live_viewer

_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "macura_backflip.yaml")


def _env_cfg(config_path):
    cfg = yaml.safe_load(open(config_path))
    return cfg["env"]


def main():
    p = argparse.ArgumentParser(description="Watch a trained Pogo policy backflip live.")
    p.add_argument("--ckpt", help="path to one best .zip checkpoint to fly")
    p.add_argument("--compare", help="a folder of *_best.zip checkpoints to fly one after another")
    p.add_argument("--config", default=_CONFIG, help="path to macura_backflip.yaml")
    p.add_argument("--seconds", type=float, default=20.0, help="seconds to fly each policy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    env_cfg = _env_cfg(args.config)

    if args.compare:
        paths = sorted(glob.glob(os.path.join(args.compare, "*_best.zip")))
        ckpts = {os.path.basename(pth).replace("_best.zip", ""): pth for pth in paths}
        if not ckpts:
            raise SystemExit(f"no *_best.zip checkpoints found in {args.compare}")
        live_viewer.fly_sequence(ckpts, env_cfg, seconds_each=args.seconds,
                                 seed=args.seed, device=args.device)
    elif args.ckpt:
        flips = live_viewer.fly_policy(args.ckpt, env_cfg, seconds=args.seconds,
                                       seed=args.seed, device=args.device)
        print(f"peak rotation: {flips:.2f} turns")
    else:
        raise SystemExit("pass --ckpt <best.zip> or --compare <checkpoints_dir>")


if __name__ == "__main__":
    main()
