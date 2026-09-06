# SHARED IMPORTS -> %run /content/macura-backflip/src/imports.py
# Loads every project symbol into the notebook namespace. The repo root goes on
# sys.path and config is imported through src, so there is one cfg object.

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("MUJOCO_GL", "disable")   # safe: import works, no GL, no egl+CUDA segfault

import copy
import glob
import json
import numpy as np
import torch
from IPython.display import Video, display

# config - one import root
import src.config.config as cfg
from src.config.config import *          # noqa: F403

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# project modules - %run so every symbol lands in the notebook namespace
ipython = get_ipython()

def run_module(rel):
    ipython.run_line_magic("run", os.path.join(ROOT, rel))

run_module("src/envs/pogo_env.py")
run_module("src/models/ensemble.py")
run_module("src/algorithms/sac.py")
run_module("src/algorithms/macura.py")
run_module("src/algorithms/mbpo.py")
run_module("src/algorithms/m2ac.py")
run_module("src/training/seed.py")
run_module("src/training/train.py")
run_module("src/viz/plots.py")

from src.training.seed import set_seed
set_seed(cfg.SEED)

print("ROOT   :", ROOT)
print("Device :", device)
print("Steps  :", cfg.TOTAL_ENV_STEPS, "| seeds", cfg.SEEDS, "| ->", cfg.DRIVE_ROOT)
