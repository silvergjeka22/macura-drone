# tasks.md — MACURA-on-Drones roadmap

Actionable checklist from scaffold → results. Ordered by dependency. Each task
notes **why** it matters so priorities stay clear. `[ ]` = todo, `[~]` = in
progress, `[x]` = done.

Legend of risk: 🔴 blocker · 🟠 fairness/correctness · 🟢 polish.

---

## Phase 0 — Scaffold (DONE)
- [x] Repo structure, pure-function modules, config, bash setup, notebook, docs.
- [x] Skydio X2 hover env with near-hover init, action repeat, dense reward.
- [x] GJS uncertainty + adaptive-κ + UTD scaling (math unit-checked locally).
- [x] Environment-study plots (Part 1 of the notebook).

---

> **Backbone change (resolved):** `mbrl-lib` is unmaintained and not installable
> on Colab's Python 3.12. We switched the SAC backbone to **Stable-Baselines3**
> and implemented the ensemble + rollout strategies ourselves. Phases below
> reflect that.

## Phase 1 — Environment & dependency bring-up (Colab)  🔴
The first time anything runs on Colab. Nothing downstream works until this passes.

- [x] 🔴 **Dependency import chain.** Whole stack (SB3 + mujoco + gymnasium +
  torch) installs on Colab Python 3.12 + numpy 2 via `requirements.txt`. *(Do NOT
  pin numpy<2 — it breaks the gymnasium ABI.)*
- [ ] 🔴 **Private clone via token.** Verify the `getpass`/Secrets → `setup_colab.sh`
  clone works and the token never lands in git config or disk.
- [ ] 🔴 **Headless MuJoCo render.** Confirm `MUJOCO_GL=egl` lets `env.render()`
  produce frames on the Colab GPU.
- [ ] 🟠 **Skydio X2 actuator check.** Print `model.nu` and `actuator_ctrlrange`;
  confirm 4 thrust actuators and that the hover thrust lies inside the range.
  Adjust action scaling if hover sits at an extreme of `[-1,1]`.
- [ ] 🟢 Run notebook Part 1 end-to-end; confirm all five study figures save to Drive.

---

## Phase 2 — SB3 backbone verification  🔴🟠
Confirm the Stable-Baselines3 manual-training calls against the installed version.

- [ ] 🔴 `algorithms/sac.py`: confirm `SAC._setup_learn(total_timesteps=0, ...)`
  signature and that `agent.train(gradient_steps, batch_size)` runs without
  `.learn()`. Confirm `replay_buffer.add(obs, next_obs, act, reward, done, infos)`
  ordering and `replay_buffer.size()`.
- [ ] 🟠 `models/ensemble.py`: confirm `member_gaussians` returns per-member
  `(means, variances)` with the ensemble axis first. *Why: GJS needs ALL members.*
  Unit-tested locally for finite, sensible values; re-check on GPU.
- [ ] 🟠 Decide **known vs learned reward** consistently (current: known analytic
  reward via `drone_env.known_reward_fn`). Keep identical across all four.
- [ ] 🟢 Perf: `_store_model_transitions` adds rows one-by-one — vectorize if the
  rollout round becomes a bottleneck.

---

## Phase 3 — Smoke test (1 seed, ~3k steps)  🔴
Validate the full loop end-to-end before spending compute.

- [ ] 🔴 `train_one('macura', cfg_smoke, ...)` runs without error and logs curves.
- [ ] 🔴 Repeat for `mbpo`, `m2ac`, `sac`.
- [ ] 🟠 Confirm κ updates and rollout length is recorded (MACURA) and is finite.
- [ ] 🟠 Confirm model rollouts respect `termination_fn` (no flight past a crash).
- [ ] 🟢 Confirm checkpoints + run JSON land in Drive and reload.

---

## Phase 4 — Fairness controls (the comparison is meaningless without these)  🟠
- [ ] 🟠 **Exploration confound.** Apply ONE exploration scheme to all four (config
  `exploration.type`). The paper uses pink noise for MACURA only — do NOT copy
  that asymmetry. Replace the AR(1) pink-noise proxy in `_explore` with a proper
  pink-noise generator if pink is chosen.
- [ ] 🟠 **UTD confound.** Either give all model-based algos the same gradient-step
  rule, or report MACURA both with adaptive (Eq. 22) and fixed `G` so the gain is
  attributable to rollout adaptation, not UTD.
- [ ] 🟠 **Tune MBPO's schedule** (`rollout.mbpo.rollout_schedule`) on the drone —
  an untuned ramp is a strawman. Sweep a few schedules; pick the best for MBPO.
- [ ] 🟠 **Faithful M2AC.** Current version reuses GJS + a reward penalty. Compare
  against the paper's rank-based masking; document the deviation.
- [ ] 🟠 **Ensemble uncertainty calibration.** Verify `uGJS` is finite and not
  blowing up (covariance inversion). Tune `logvar_bounds` if members collapse.

---

## Phase 5 — Low-step optimization (get clear curves fast)  🟢
- [ ] 🟢 Tune **action_repeat** (2–4) and **max_episode_steps** for fastest signal.
- [ ] 🟢 Tune **near-hover init noise** — smaller = faster results, larger = more
  instability stress. Pick a value that still shows the MBPO-vs-MACURA contrast.
- [ ] 🟢 Tune **MACURA ξ** (the one real knob); sweep {1,3,5,10,20} as in the paper.
- [ ] 🟢 Confirm warmup + observation normalization help early take-off.

---

## Phase 6 — Full comparison & figures  🟢
- [ ] 🟢 Run 4 algorithms × 3 seeds to `total_env_steps` (raise to 5 seeds for final).
- [ ] 🟢 Produce the four figures: sample-efficiency, failure-rate, final-quality,
  MACURA rollout-depth/κ.
- [ ] 🟢 Check the expected ordering (MACURA ≳ M2AC ≳ MBPO > SAC early; MBPO wobble).
- [ ] 🟢 Add MuJoCo **sanity-anchor**: run the same code on one stock MuJoCo task
  (e.g., Hopper) and confirm the curve reproduces the paper — proves the
  implementation, de-risks attributing drone failures to bugs.

---

## Phase 7 — Robustness & extensions (optional)  🟢
- [ ] 🟢 Process-noise robustness (paper App. D.4): add observation/process noise,
  re-run — does GJS still isolate epistemic uncertainty?
- [ ] 🟢 Path-following task (`env.task = track`) once hover is solid.
- [ ] 🟢 Save rollout videos of the best policy per algorithm to Drive.
- [ ] 🟢 Aerial-manipulation extension (welded-grasp + mass-change story) — the
  most vivid MACURA demo, reusing the same algorithm code. Separate milestone.

---

## Known risks (watch throughout)
1. **mbrl-lib / gym version drift** — Phase 1/2 are where this bites. 🔴
2. **Compute, not samples** — Colab wall-clock + 12h limits; rely on checkpoint/resume. 🔴
3. **Ensemble variance collapse** → meaningless κ. Monitor `uGJS` magnitude. 🟠
4. **Accidental MACURA advantage** via exploration/UTD/strawman baselines. 🟠
5. **High RL variance** — need ≥3 (ideally 5) seeds for legible ±std bands. 🟢
