# tasks.md — MACURA-on-Drones roadmap

Actionable checklist from scaffold to results. Ordered by dependency. Each task
notes why it matters. Markers: `[ ]` = not started, `[/]` = in progress,
`[x]` = done. Priority is a plain-text tag: (high) / (med) / (low).

Current state: backbone is **Stable-Baselines3 SAC** (mbrl-lib is not installable
on Colab Python 3.12). Primary env is **PyBullet via gym-pybullet-drones**
(`CtrlAviary`, Crazyflie CF2X); MuJoCo (Skydio X2) is an optional backend. Both
expose the identical 13-dim obs + dense reward, so only physics differs.

---

## Phase 0 — Scaffold
- [x] Repo structure, pure-function modules, config, bash setup, notebook, docs.
- [x] Drone hover env (PyBullet + MuJoCo backends) with near-hover init, action
  repeat, dense reward; shared `known_reward_fn` / `termination_fn`.
- [x] GJS uncertainty + adaptive-kappa + UTD scaling (math unit-checked locally,
  matches Eq. 15-22 — see audit).
- [x] Conceptual + environment-study plots; 6-section study-guide notebook.

---

## Phase 1 — Environment & dependency bring-up (Colab)
The first time anything runs on Colab. Nothing downstream works until this passes.

- [x] (high) Core stack installs on Colab Python 3.12 + numpy 2 (no numpy/scipy/
  torch reinstall; numpy repaired in `setup_colab.sh` step 4c).
- [x] (high) gym-pybullet-drones is not on PyPI — installed from source
  `--no-deps`; `transforms3d` listed explicitly in requirements.
- [/] (high) Confirm a clean Colab run reaches the env probe: print pybullet
  version via `getattr(pybullet,'__version__',pybullet.getAPIVersion())`, obs/
  action shapes, HOVER_RPM/MAX_RPM.
- [ ] (high) Private clone via token works and the token never lands in git config
  or on disk.
- [ ] (med) gym-pybullet-drones `CtrlAviary` constructor args match the installed
  version (`pyb_freq`/`ctrl_freq` vs older `freq`); `_getDroneStateVector` layout
  confirmed (pos[0:3], quat[3:7] xyzw, vel[10:13], angvel[13:16]).
- [ ] (low) PyBullet headless camera render works for videos (CPU TinyRenderer).

---

## Phase 2 — SB3 backbone verification
Confirm the Stable-Baselines3 manual-training calls against the installed version.

- [ ] (high) `algorithms/sac.py`: confirm `SAC._setup_learn(total_timesteps=0)`
  configures `self._logger` so `agent.train(gradient_steps, batch_size)` runs
  without `.learn()`. If the logger is unset, `train()` will crash — set a logger
  explicitly.
- [x] (med) `_store_model_transitions` vectorized via `_bulk_add` (circular insert
  unit-tested); falls back to per-row add if the SB3 buffer layout differs.
- [x] (med) `select_actions` batches SB3 `predict` over the rollout (was per-row).
- [ ] (med) Confirm `replay_buffer.add(obs, next_obs, act, reward, done, infos)`
  ordering and `optimize_memory_usage=False` on the installed SB3 (assumed by
  `_bulk_add`).

---

## Phase 3 — Smoke test (1 seed, tiny steps)
Validate the full loop end-to-end before spending compute.

- [ ] (high) `train_one('macura', cfg_smoke, ...)` runs without error, logs curves,
  saves a best checkpoint.
- [ ] (high) Repeat for `mbpo`, `m2ac`, `sac`.
- [ ] (med) Confirm kappa updates and is finite; rollout length recorded.
- [ ] (med) Confirm model rollouts respect `termination_fn` (no flight past crash).
- [ ] (low) Confirm best checkpoints + run JSON land in Drive and reload.

---

## Phase 4 — Correctness & fairness fixes (from the audit)
The comparison is not trustworthy until these are addressed.

- [ ] (high) **Pink noise is not implemented.** `_explore` in `training/train.py`
  adds WHITE Gaussian noise on top of the already-stochastic SAC policy for any
  `exploration.type` except `deterministic`; `pink_noise`/`white_noise` behave
  identically. Either implement temporally-correlated pink noise (per-episode 1/f
  / AR process, like the Part-1 plot) or rename the config and fix the notebook
  narrative. The illustrative PSD plot is correct; the training is not.
- [ ] (high) **SAC baseline UTD.** Model-free SAC runs at `gradient_steps_max`
  (=20 full / 8 fast) updates per real step. Standard SAC uses ~1. UTD=20 plain
  SAC usually destabilizes -> the baseline may be unfairly weak (or non-standard).
  Set a separate `sac.baseline_gradient_steps` (~1) or document the choice.
- [ ] (high) **GJS computed in unnormalized delta space.** `train_ensemble`
  normalizes inputs only; targets `y = next_obs - obs` are raw, so
  `member_gaussians` returns raw-scale deltas and `compute_gjs` sums over obs dims
  with different physical scales (position vs angular velocity) -> uncertainty is
  scale-dominated by large-magnitude dims. mbrl-lib normalizes targets. Normalize
  the delta targets (and un-normalize for `predict`) so uGJS is scale-balanced.
- [ ] (med) **UTD confound.** MACURA uses adaptive UTD (Eq. 22) while MBPO/M2AC use
  fixed 20. Part of MACURA's gain may be UTD, not rollout adaptation. Report MACURA
  with both adaptive and fixed G.
- [ ] (med) **Tune MBPO's schedule** (`rollout.mbpo.rollout_schedule`) on the drone
  — the default [1,15,1,15000] is near length-1 for short runs; an untuned ramp is
  a strawman. Sweep and pick the best for MBPO.
- [ ] (med) **Faithful M2AC.** Current version pools all rollout steps and keeps the
  global lowest-uncertainty fraction using GJS (paper uses per-step rank-based
  masking with OvR + a penalty). Document or align.
- [ ] (low) `sac.critic_lr` in the config is ignored (SB3 SAC uses a single
  `learning_rate`). Remove it or note it.

---

## Phase 5 — Low-step optimization
- [ ] (low) Tune `action_repeat` / `max_episode_steps` for fastest signal.
- [ ] (low) Tune near-hover init noise (smaller = faster, larger = more stress).
- [ ] (low) Tune MACURA `xi`; sweep {1,3,5,10,20} as in the paper.
- [ ] (low) Confirm warmup + obs/target normalization help early take-off.

---

## Phase 6 — Full comparison & figures
- [ ] (med) Run 4 algorithms x 3 seeds to `total_env_steps` (5 for final).
- [ ] (med) Produce: sample-efficiency, failure-rate, final-quality, MACURA
  rollout-depth/kappa, GJS-over-training, rollout-length histogram.
- [ ] (med) Check the expected ordering (MACURA >= M2AC >= MBPO > SAC early; MBPO
  wobble) and report honestly if it does not hold.
- [ ] (med) MuJoCo/Hopper sanity anchor: run the same code on a stock task and
  confirm it reproduces the paper qualitatively — de-risks attributing drone
  failures to bugs. (Flip `env.backend` and/or add a Gym-MuJoCo make_env variant.)

---

## Phase 7 — Robustness & extensions (optional)
- [ ] (low) Process/sensor-noise robustness (paper App. D.4) via `NoisyObsWrapper`:
  does GJS still isolate epistemic uncertainty?
- [ ] (low) Path-following task (`env.task = track`) once hover is solid (PyBullet
  `_current_target` currently returns the static hover target — implement moving
  target for the PyBullet backend).
- [ ] (low) Cross-engine A/B: run the identical notebook with `backend: mujoco`.
- [ ] (low) Aerial-manipulation extension (welded-grasp + mass-change story).

---

## Known risks (watch throughout)
1. (high) SB3 manual-training API (`_setup_learn` logger, `replay_buffer` layout)
   — version-sensitive; the main untested runtime risk.
2. (high) gym-pybullet-drones source install + API drift on Python 3.12.
3. (high) Compute, not samples, is the Colab bottleneck; rely on best-checkpoint
   + idempotent resume.
4. (med) Ensemble variance collapse -> meaningless kappa. Monitor uGJS magnitude
   (logvar bounds currently guard this).
5. (med) Accidental MACURA advantage via exploration/UTD/strawman baselines (see
   Phase 4).
6. (med) High RL variance — need >=3 (ideally 5) seeds for legible bands.
