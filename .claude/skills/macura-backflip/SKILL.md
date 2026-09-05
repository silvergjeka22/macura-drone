---
name: macura-backflip
description: Write or edit the macura-backflip Colab notebook (colab.ipynb), the macOS live viewer (run_live_mac.py / viz/live_viewer.py), and the src/ pure-function modules. Use whenever changing colab.ipynb, adding or editing functions in envs/, models/, algorithms/, training/, viz/, the Pogo model envs/assets/pogo.xml, or configs/macura_backflip.yaml. Enforces the "every .py is a library of pure functions, the notebook orchestrates" contract, the four-algorithm fairness invariant, the analytic-reward invariant (real and imagined transitions scored identically), a minimal notebook structure, and the config-single-source-of-truth rule.
---

# macura-backflip notebook + module style

This repo teaches **Pogo** (a planar one-legged pogo-stick gymnast) to **backflip**,
and compares four algorithms - MACURA, MBPO, M2AC, SAC - with a shared SAC backbone
and a shared probabilistic ensemble, so any difference is attributable to the
rollout strategy alone. A backflip is an aggressive, unstable maneuver: it is where
a fixed-horizon rollout over-imagines and MACURA's uncertainty-adaptive truncation
earns its keep. Code runs in two places only: **`colab.ipynb`** (train on GPU) and
**`run_live_mac.py`** (watch a saved policy fly on macOS). Every `.py` is a library
of pure functions those two orchestrate.

## The three hard contracts

1. **Every `.py` DEFINES only** - no top-level execution, no `argparse`/`main`,
   no printing except where printing IS the job (the training progress line in
   `train.py`). The one executable file is `run_live_mac.py` (the Mac entry point,
   the local analogue of the notebook); keep its logic in `viz/live_viewer.py`.
2. **Fairness invariant** - the four algorithms share the SAC backbone, the
   ensemble, pink-noise exploration, the eval seeds, and the fixed `real_ratio`.
   **Only the rollout strategy differs.** Never add anything to one algorithm's
   path (extra updates, different noise, a tuned reward) the others do not get.
3. **Analytic-reward invariant** - the backflip reward and the termination
   condition are pure functions of `(obs, action)` / `obs`
   (`envs/pogo_env._backflip_reward`, `known_reward_fn`, `termination_fn`), so a
   REAL transition and an IMAGINED one are scored identically. If you change the
   reward or termination, change the single shared function - never fork a
   separate "env reward" and "rollout reward".

## The Pogo env (`envs/pogo_env.py` + `envs/assets/pogo.xml`)

- Planar body: root = slide-x, slide-z, and one **UNLIMITED** hinge `rooty` (the
  flip axis - do not add a range limit or it cannot flip), plus hip/knee/ankle.
- **Observation (13):** `[torso_z, sin(rooty), cos(rooty), hip, knee, ankle,
  foot_clearance, x_vel, z_vel, rooty_vel, hip_vel, knee_vel, ankle_vel]`.
  The angle enters as **sin/cos** (smooth across the 360 wrap - a raw angle would
  jump and the ensemble cannot model it). `foot_clearance` is the airborne signal
  the reward keys off (the torso rotates in place, so its height is a poor cue).
- **Reward** (`_backflip_reward`): alive + foot-clearance jump + spin-while-airborne
  + upright-while-grounded - control. `reward.w_rotation` is the curriculum knob
  (0 = jump & balance, large = full flip). Keep it a function of obs columns
  `_Z / _COS / _FOOTZ / _ROOTY_VEL` only, so `known_reward_fn` stays exact.
- **Termination:** `torso_z < fail_torso_height` (a collapse). Keep it a pure
  function of the observation so imagined rollouts truncate on the same condition.
- If you change the obs layout, update the `_Z/_SIN/_COS/_FOOTZ/_ROOTY_VEL`
  indices, `_get_obs`, and `_backflip_reward` together.

## Notebook structure (keep it short - this order)

```
Setup       drive mount + device; token; setup_colab.sh; imports + config (deep-merge fast profile)
Meet Pogo   make_env + render_filmstrip (one look at the untrained body)
Train       idempotent matrix (4 algos x seeds); skip a run whose log already exists
Results     sample efficiency, SUCCESS (landed flips), faceplant rate, final quality, MACURA signature
Flight demo record_best_videos, inline clips, stitched reel
Ordering    measured final ordering + the `mjpython run_live_mac.py` viewer command
```

The notebook is deliberately minimal (no smoke flag, no explainer/synthetic figures).
Keep it that way: new conceptual figures belong in a separate notebook, not this one.

## Notebook + code rules

1. **Single source of truth = `configs/macura_backflip.yaml`.** No hard-coded
   hyperparameters in the notebook; deep-merge `profiles.fast` and let `make_cfg`
   apply the `SMOKE` overrides. New quantities go in the YAML.
2. **No algorithm / env / model / training logic in the notebook** - it calls
   `src/` functions. Only glue (`deep_merge`, `make_cfg`, display loops) lives there.
3. **Every code cell is preceded by a plain-language markdown cell** (why + what
   to look for).
4. **Every figure is both saved and shown** - `save_path=f'{PLOTS}/<name>.png'`
   AND rendered inline. New plotting logic goes in `viz/plots.py`, never inline.
5. **Keep the notebook minimal** - setup, one Pogo look, train, the key result
   figures, videos. No smoke flag, no explainer/synthetic figures. A new figure
   goes in `viz/plots.py` and only if a result cell actually uses it.
6. **Idempotent / resumable** - skip a run if its `logs/<algo>_seed<seed>.json`
   already exists (compute, not samples, is the Colab bottleneck).
7. **ASCII only.** Cite the paper for paper claims (GJS Eq. 15-19, kappa Eq. 21,
   UTD Eq. 22). Use `viz/plots.ALGO_COLORS` (MACURA blue, MBPO red, M2AC green,
   SAC gray) in every comparison figure.

## src/ rules (the pure-function library)

1. One job per function; positional args first; pass the relevant **cfg sub-dict**
   (`cfg["env"]`, `cfg["ensemble"]`, `cfg["sac"]`, `cfg["rollout"]["macura"]`).
2. A one-line docstring saying what it returns; return plain values / arrays / dicts.
3. No global state; thread cross-round state in an explicit dict (`kappa_state`).
4. No printing / no defensive `try/except`-and-print noise (a real fallback with a
   comment is fine, e.g. `_bulk_add`).
5. New figures are new functions in `viz/plots.py` taking `save_path=None`.

## Real signatures (do not rewrite them into the notebook)

```
pogo_env.make_env(cfg["env"], seed=0, render=False)          -> (env, obs_dim(13), act_dim(3))
pogo_env.known_reward_fn(cfg["env"])                          -> reward_fn(obs, act) -> (batch,)
pogo_env.termination_fn(cfg["env"])                           -> done_fn(obs) -> (batch,) bool

ens.build_ensemble(cfg["ensemble"], obs_dim, act_dim, device) / train_ensemble / predict / member_gaussians
macura_mod.compute_gjs / update_kappa / macura_rollout(model, agent, start, reward_fn, done_fn, kappa_state, cfg) / gradient_steps
sac_mod.build_sac / select_action / select_actions / sac_update / sac_update_mixed

trainer.train_one(algo, cfg, drive_dir, seed, init_ckpt=None) -> run_dict   (init_ckpt warm-starts a curriculum stage)
trainer.evaluate(agent, env, eval_episodes)                  -> metrics (incl. eval_success_rate, eval_flips, eval_failure_rate)
trainer.evaluate_best / record_best_videos(cfg, drive_dir, device, seeds, seconds=8)

live_viewer.fly_policy(ckpt, env_cfg, seconds, seed, device) / fly_sequence(ckpts, env_cfg, ...)   # macOS, via mjpython
```

The ensemble predicts the next-state **delta** in **normalized** target space;
`member_gaussians` returns normalized Gaussians so `compute_gjs` is scale-balanced.
Do not undo that. Models are driven manually (SB3 `replay_buffer.add` + `.train`);
the SAC backbone is identical for all four - SAC just calls no rollout.

## The bootstrap + the Mac viewer - keep their contracts

- Part 0: `drive.mount`, `MUJOCO_GL=egl`, device print; token via `userdata`/`getpass`
  (never written to disk/git); `curl` the raw `setup_colab.sh` (clones the private
  repo - nothing external to fetch, Pogo ships in the repo); imports + config +
  `SMOKE` + `make_cfg`.
- macOS viewer: `mjpython run_live_mac.py --ckpt <best.zip>` (or `--compare <dir>`).
  Trains on Colab (CUDA) -> saved `.zip` loads on the Mac (CPU) -> live MuJoCo
  window. Use `mjpython`, never plain `python` (the viewer must own the main thread).

## Checklist before saving

- [ ] Notebook stays short (setup / meet Pogo / train / results / demo / ordering)
- [ ] No algorithm / env / model / training logic added to the notebook
- [ ] Every new hyperparameter is in `configs/macura_backflip.yaml`
- [ ] Reward/termination change touches the single shared function (real == imagined)
- [ ] Fairness invariant holds - change touches only the rollout path, or all four equally
- [ ] Obs layout change updates indices + `_get_obs` + reward together
- [ ] New figure is in `viz/plots.py`, saves AND shows, uses ALGO_COLORS
- [ ] No emoji / non-ASCII; paper claims carry an Eq./Sec. cite
