---
name: macura-backflip
description: Write or edit the macura-backflip notebook (notebooks/backflip.ipynb), the macOS live viewer (run_live_mac.py / src/viz/live_viewer.py), and the src/ pure-function modules. Use whenever changing the notebook, adding or editing functions under src/ (envs, models, algorithms, training, viz), the Pogo model src/envs/assets/pogo.xml, or src/config/config.py. Enforces the src/ layout with a Python config module, the bootstrap + imports (%run) pattern, thin bold-header notebooks with no def/class, the four-algorithm fairness invariant, and the analytic-reward invariant (real and imagined transitions scored identically).
---

# macura-backflip style

Pogo (a planar one-legged pogo-stick gymnast) learns to **backflip**, and four
algorithms - MACURA, MBPO, M2AC, SAC - are compared with a shared SAC backbone and a
shared ensemble, so only the rollout strategy differs. The project follows the same
layout as the ULAL repo: a domain-organized `src/`, a Python config module (no YAML),
a `bootstrap.py` + `imports.py`, and one thin notebook driven by `%run src/imports.py`.

## Layout (mirror ULAL)

```
src/
  bootstrap.py            mount Drive, install deps, make Drive folders (setup())
  imports.py              %run target: sys.path, libs, cfg, device, %run every module, set_seed
  config/config.py        ALL quantities as constants + section dicts (ENV, ENSEMBLE, SAC, ROLLOUT...) + CFG
  envs/pogo_env.py        make_env, known_reward_fn, termination_fn   (+ envs/assets/pogo.xml)
  models/ensemble.py
  algorithms/{sac,macura,mbpo,m2ac}.py
  training/{seed,train}.py
  viz/{plots,live_viewer}.py
notebooks/backflip.ipynb
run_live_mac.py           macOS live viewer (the one executable entry point)
```
No `__init__.py` (namespace packages). Inter-module imports use the `src.` prefix
(`from src.models import ensemble as ens`).

## The three hard contracts

1. **Every `.py` under `src/` DEFINES only** - no top-level execution, no printing
   except where printing IS the job (the training progress line, `set_seed` prints
   nothing). The notebook orchestrates; `run_live_mac.py` is the only executable.
2. **Fairness invariant** - the four algorithms share the SAC backbone, the ensemble,
   pink-noise exploration, the eval seeds, and the fixed `real_ratio`. Only the
   rollout strategy differs. Never give one algorithm's path an edge the others lack.
3. **Analytic-reward invariant** - the backflip reward and termination are pure
   functions of `(obs, action)` / `obs` (`src/envs/pogo_env._backflip_reward`,
   `known_reward_fn`, `termination_fn`), so real and imagined transitions are scored
   identically. Change the single shared function - never fork env vs rollout reward.

## Config = a Python module (no YAML)

Every quantity lives in `src/config/config.py` as UPPERCASE constants and section
dicts (`ENV`, `REWARD`, `ENSEMBLE`, `SAC`, `SELECTION`, `ROLLOUT`, `EXPLORATION`),
assembled into `CFG` (the nested dict the training functions consume). To change a
hyperparameter, edit `config.py` - never hard-code it in a module or the notebook.
`w_rotation` in `REWARD` is the curriculum knob (0 = jump & balance -> 1 = full flip).

## imports.py (the %run aggregator)

`imports.py` puts ROOT on `sys.path`, imports libs, does `import src.config.config as
cfg` + `from src.config.config import *`, sets `device`, then `%run`s every module so
all symbols (`make_env`, `train_one`, `plot_*`, ...) land in the notebook namespace,
and finally `set_seed(cfg.SEED)`. Add a new module by adding one `run_module(...)`
line. Keep it notebook-only (it uses `get_ipython()`); for local runs, import modules
directly.

## Notebook rules (thin, bold headers, no def/class)

1. Markdown headers are bold: `## **Connect Colab**`, `## **Setup**`, `## **Meet
   Pogo**`, `## **Train**`, `## **Results**`, `## **Flight demo**`, `## **Save**`.
2. **No `def` or `class` in the notebook.** Helpers go in `src/`.
3. **One idea per cell.** Comments are rare and short.
4. Cells call bare functions from the `%run` namespace (`train_one(...)`,
   `make_env(cfg.ENV, ...)`, `plot_success_rate(runs)`), read config as `cfg.X`, and
   `set_seed(cfg.SEED)` after imports.
5. **ASCII only.** No emoji, no unicode arrows.
6. Keep it minimal - setup, one Pogo look, train, the key result figures, videos.
   No smoke flag, no synthetic explainer figures.
7. Idempotent: skip a run whose `logs/<algo>_seed<seed>.json` already exists on Drive.

## The Connect Colab cell - keep its contract

The first code cell clones the private repo with a `getpass` token, `git reset
--hard FETCH_HEAD` on re-run, and puts ROOT on `sys.path`. The token is never written
to disk or git config. `## **Setup**` then calls `from src.bootstrap import setup;
setup(...)`, and the next cell is `%run /content/macura-backflip/src/imports.py`.

## Pogo env (src/envs/pogo_env.py + assets/pogo.xml)

Planar body: root = slide-x, slide-z, one **UNLIMITED** hinge `rooty` (the flip axis
- never range-limit it), plus hip/knee/ankle. Obs (13): `[torso_z, sin(rooty),
cos(rooty), hip, knee, ankle, foot_clearance, x_vel, z_vel, rooty_vel, hip_vel,
knee_vel, ankle_vel]` - angle as sin/cos (smooth across the wrap); foot_clearance is
the airborne signal the reward gates on. If you change the obs layout, update the
`_Z/_SIN/_COS/_FOOTZ/_ROOTY_VEL` indices, `_get_obs`, and `_backflip_reward` together.

## Real signatures (bare in the notebook after %run)

```
make_env(cfg.ENV, seed=0, render=False)              -> (env, obs_dim(13), act_dim(3))
known_reward_fn(cfg.ENV) / termination_fn(cfg.ENV)
train_one(algo, cfg.CFG, cfg.DRIVE_ROOT, seed, init_ckpt=None)  -> run_dict
evaluate / evaluate_best / record_best_videos(cfg.CFG, cfg.DRIVE_ROOT, device, seeds, seconds=8)
plot_sample_efficiency / plot_success_rate / plot_failure_rate / plot_final_quality / plot_rollout_depth
render_filmstrip / stitch_videos_grid
set_seed(cfg.SEED)
live_viewer.fly_policy / fly_sequence   # macOS, via mjpython run_live_mac.py
```

## Checklist before saving

- [ ] File lives under `src/` (pure functions) or is the notebook / run_live_mac.py
- [ ] Inter-module imports use the `src.` prefix; no `__init__.py` added
- [ ] Every new hyperparameter is a constant in `src/config/config.py`
- [ ] Reward/termination change touches the single shared function (real == imagined)
- [ ] Fairness invariant holds - change touches only the rollout path, or all four equally
- [ ] Notebook stays thin: bold `## **Header**` cells, no def/class, one idea per cell, ASCII
- [ ] New module is added to `imports.py` with a `run_module(...)` line
- [ ] New figure is a function in `viz/plots.py` that a result cell actually uses
