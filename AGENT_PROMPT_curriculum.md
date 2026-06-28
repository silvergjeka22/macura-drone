# Task prompt — port SafeMACURA-Drive's training structure + a shared real-data-first curriculum to `macura-drone`, faithfully

You are working in the **`macura-drone`** repo (Skydio X2 quadrotor, MuJoCo, four paper
algorithms MACURA / MBPO / M2AC / SAC on a shared SB3 SAC backbone + a shared probabilistic
ensemble). Your job has two parts: **(A)** bring the training/eval *structure* up to the standard
of the sister repo `SafeMACURA-Drive` (`training/train.py`, `algorithms/model_based.py`,
`training/curriculum.py`), and **(B)** add the shared **real-data-first curriculum** (start
~100% real, phase in imagined rollouts) — implemented as a config-gated option whose **default is
the paper's fixed `real_ratio`**.

Read this whole prompt before writing code. Do not change any algorithm's update rule.

═══════════════════════════════════════════════════════════════════════════════
## 0. CONTEXT YOU MUST KNOW BEFORE TOUCHING ANYTHING
═══════════════════════════════════════════════════════════════════════════════

**The current drone data flow is off-paper.** In `training/train.py::train_one`, model-based
agents (`macura`/`mbpo`/`m2ac`) push **only imagined transitions** into the SB3 replay buffer that
the SAC learner trains on; real transitions go into the SB3 buffer **only for the SAC baseline**.
So the model-based agents currently train on **0% real data** (pure imagination). That is a
deviation from the paper, which uses **batch-level real mixing with a fixed `real_ratio` ≈ 0.05**
(Janner/MBPO; MACURA inherits it). On an *unstable* drone this 0%-real setting is the worst case:
the policy learns from a model that early on has only seen near-crash data.

**The sister repo already solves this cleanly.** In `SafeMACURA-Drive`:
- `algorithms/model_based.py` keeps a **real buffer** and a **model buffer** and, for each SAC
  update, draws the *entire* batch from the real buffer **with probability `real_ratio`**, else
  from the model buffer (**batch-level mixing**, exactly as in the official MACURA/mbrl-lib code).
- `training/curriculum.py` defines `RealRatioSchedule` — a piecewise-linear `real_ratio(step)`
  read inside that update loop. When the curriculum is disabled it degenerates to
  `RealRatioSchedule.constant(real_ratio)`, i.e. the paper's fixed value (behaviour identical to
  before).
- `training/train.py` selects the best checkpoint by **highest periodic greedy-eval return on
  fixed shared seeds** (with a `start_step` gate), saves **policy + ensemble + meta** on each new
  best, and runs **one larger final greedy eval** as a summary. It logs rich MBRL telemetry
  (`real_pct`/`imagined_pct`, `kappa`, `rollout_length`, `model_term_frac`).

Your task is to reproduce that structure in `macura-drone` **without changing any algorithm
logic** and add the curriculum the same way: shared across the three model-based agents, applied
to `real_ratio` only, never to rollout length.

═══════════════════════════════════════════════════════════════════════════════
## 1. DO NOT CHANGE ALGORITHM LOGIC (paper fidelity — hard rule)
═══════════════════════════════════════════════════════════════════════════════

These stay **behaviourally identical** — md5-stable except where this prompt explicitly edits:
- `algorithms/macura.py` — GJS uncertainty (`compute_gjs`), adaptive-κ truncation
  (`macura_rollout`, `update_kappa`), and the Eq. 22 UTD scaling (`gradient_steps`).
- `algorithms/mbpo.py` — fixed truncated-linear rollout schedule.
- `algorithms/m2ac.py` — fixed-length rollout + uncertainty masking.
- `models/ensemble.py` — ensemble build/train/predict and the per-member Gaussians.
- `algorithms/sac.py` — the shared SAC backbone math.

The ONLY new shared mechanism is the **real_ratio curriculum** in §3. You may **NOT** hand-schedule
MACURA's rollout length — that would replace MACURA's contribution with MBPO's. If a change would
touch any algorithm's update/rollout rule, STOP and flag it instead of proceeding.

═══════════════════════════════════════════════════════════════════════════════
## 2. STRUCTURAL CHANGE — two buffers + batch-level real/imagined mixing
═══════════════════════════════════════════════════════════════════════════════

This is the prerequisite that makes any `real_ratio` (fixed *or* curriculum) possible. Today the
SAC learner has one buffer; you need the update to choose its source per batch.

**2.1 Keep two buffers for model-based agents.**
- A **real buffer** of real transitions `(obs, act, rew, next_obs, done)` — used both to train the
  ensemble (already done via `_RealBuffer`, which currently stores only `obs/act/next_obs`; extend
  it to also store `rew` and `done`) **and** to supply real batches to the SAC update.
- The existing **model buffer** (the SB3 replay buffer the rollouts are written into) — imagined
  transitions only.
- For the **SAC baseline**, behaviour is unchanged: one buffer, 100% real, no mixing.

**2.2 Batch-level mixing in the SAC update (match the official code + sister repo).**
Replace the single `sac_mod.sac_update(agent, num_updates, batch)` call (which calls
`agent.train(gradient_steps=num_updates, ...)` on one buffer) with a loop that does, per gradient
step:
```
use_real = rng.random() < rr            # rr = real_ratio(step) from §3
buf      = real_buffer if use_real else model_buffer
if len(buf) < batch: break
sac_one_update(agent, buf, batch)       # ONE gradient step sampling from `buf`
```
Implementation notes:
- This is **batch-level** mixing (a whole batch from one buffer), not row-level — identical to
  `SafeMACURA-Drive/algorithms/model_based.py` and the official MACURA repo. Do not blend within a
  batch.
- You must be able to run a single SAC gradient step sampling from a chosen buffer. With SB3 this
  means either (a) temporarily pointing the learner at the chosen buffer and calling
  `agent.train(gradient_steps=1, batch_size=batch)`, or (b) sampling the batch yourself and calling
  the lower-level update. Pick the approach that keeps `algorithms/sac.py`'s math untouched; add a
  thin helper `sac_update_mixed(agent, real_buf, model_buf, num_updates, batch, rr, rng)` in
  `algorithms/sac.py` (or `training/train.py`) rather than editing the SAC core.
- Count and return the **realised** split for logging: `n_real`, `n_model`, and derive
  `real_pct` / `imagined_pct` (see §5).

**2.3 Fairness invariant.** The mixing code path is the **same** for MACURA/MBPO/M2AC — no
per-algorithm branching. SAC is model-free and is left exactly as it is (already 100% real).

═══════════════════════════════════════════════════════════════════════════════
## 3. THE SHARED CURRICULUM — fixed `real_ratio` by default, schedule optional
═══════════════════════════════════════════════════════════════════════════════

**3.1 Default = paper-faithful fixed `real_ratio`.** Add `sac.real_ratio: 0.05` (paper value) to
the config and feed it through the §2 mixing as a **constant**. This alone fixes the current
0%-real deviation and is the headline, paper-faithful setting. **The four-algorithm comparison you
report as the main result must use this fixed value**, so MACURA's win is attributable to its
rollout adaptation alone (see §6).

**3.2 Optional curriculum (config-gated, OFF by default).** Port `RealRatioSchedule` from
`SafeMACURA-Drive/training/curriculum.py` verbatim into `macura-drone/training/curriculum.py`
(piecewise-linear `real_ratio(step)`, clamped outside the knot range, with `.constant(r)` and
`.from_config(cfg)` constructors). Wire it into the §2 loop:
```
cur = cfg.get("curriculum") or {}
if cur.get("enabled") and cur.get("real_ratio_knots"):
    schedule = RealRatioSchedule.from_config(cur)
else:
    schedule = RealRatioSchedule.constant(cfg["sac"]["real_ratio"])   # paper fixed value
...
rr = schedule(step)
```
Default config block (curriculum **disabled**, so default behaviour = fixed 0.05):
```yaml
# configs/macura_drone.yaml
sac:
  real_ratio: 0.05          # paper-faithful fixed batch-level real mixing (DEFAULT)

curriculum:                 # OPTIONAL real-data-first ablation; OFF by default
  enabled: false            # set true to run the curriculum ablation
  # shared by MACURA/MBPO/M2AC; controls real_ratio ONLY (never rollout length)
  real_ratio_knots:
    - {step: 0,     real_ratio: 0.90}   # phase 1: ~90% real — model matures, agent bootstraps
    - {step: 8000,  real_ratio: 0.90}
    - {step: 16000, real_ratio: 0.20}   # phase 2: phase imagined rollouts in as model gets reliable
    - {step: 24000, real_ratio: 0.05}   # phase 3: paper-ish steady state (~5% real / 95% imagined)
```
Notes:
- The user asked for "start 100% (or ~90%) real and learn to lean on imagined." `0.90` start is a
  good default for the drone (a touch of imagined data from step 0 keeps the ensemble's gradient
  signal alive); `1.00` is also fine — expose it as a knot, do not hard-code.
- Scale the knot steps to `experiment.total_env_steps` (the config uses 20k–30k). The schedule
  clamps outside its range, so the last knot is the steady state.
- **Rollout length stays per-algorithm and at paper values** — MACURA `t_max=10` truncated by κ
  (never forced), MBPO truncated-linear, M2AC full+mask. The curriculum is the `real_ratio` ramp,
  nothing else.

═══════════════════════════════════════════════════════════════════════════════
## 4. PORT THE TRAINING/EVAL STRUCTURE FROM SafeMACURA-Drive
═══════════════════════════════════════════════════════════════════════════════

Bring `macura-drone/training/train.py` up to the sister repo's protocol (keep the drone's
pure-function, notebook-driven convention — no argparse/top-level execution):

1. **Periodic greedy eval on FIXED shared seeds** → the headline sample-efficiency curve. The
   drone already evaluates every `eval_every_steps` on `experiment.eval_seeds`; keep that, but make
   it the *selection* signal, not just a print.
2. **Best checkpoint = highest periodic greedy-eval return**, gated by a new
   `selection.start_step` (don't select during the noisy early phase). On each new best, save
   **policy + ensemble + meta.json** and mirror to Drive (the drone currently saves SB3 policy
   only via `_save_best`; also persist the ensemble `state_dict` for model-based agents, as
   `SafeMACURA-Drive/algorithms/model_based.py::save` does).
3. **One larger final greedy eval** at the end on the reloaded best checkpoint, recorded as the
   run summary (mirror `train.py`'s `final_eval` + `_meta.json`).
4. **Rich MBRL telemetry** in the run log (see §5).

Do **not** create a separate `train_macura.py`; all four algorithms must run through the **one**
`train_one` loop and the identical eval protocol.

═══════════════════════════════════════════════════════════════════════════════
## 5. TELEMETRY & PLOTS (so the curriculum and the method are visible)
═══════════════════════════════════════════════════════════════════════════════

Log per eval/diagnostic row (model-based agents): `real_ratio_target` (scheduled rr), realised
`real_pct` / `imagined_pct`, `kappa` (MACURA), `mean_rollout_length` (MACURA), and the existing
`eval_return` / `eval_return_std` / `eval_failure_rate`. Print the realised `real/imag` split in
the step line (as `SafeMACURA-Drive/training/train.py` does).

Add/confirm `viz/plots.py` figures: `plot_sample_efficiency` (return vs real steps, mean±std over
seeds), `plot_failure_rate`, `plot_rollout_depth` (MACURA κ & rollout length — the signature
figure), and a new `plot_real_ratio` (scheduled vs realised real_pct over steps) so the curriculum
is auditable. When you run the curriculum ablation, also plot fixed-vs-curriculum side by side.

═══════════════════════════════════════════════════════════════════════════════
## 6. FAIRNESS — keep the comparison honest
═══════════════════════════════════════════════════════════════════════════════

- The **headline four-algorithm result uses the fixed `real_ratio` (§3.1)** — paper-faithful, so
  MACURA's advantage is attributable to its adaptive rollout alone.
- When enabled, the curriculum is applied to **all three model-based agents equally** (one shared
  schedule, no per-algorithm branching) → it cannot bias MACURA vs MBPO vs M2AC.
- SAC gets **no curriculum** — it is the model-free baseline, already 100% real.
- Report the curriculum as a **documented deviation** from the paper's fixed `real_ratio`, run as a
  **separate ablation**, not folded silently into the main numbers.
- Compare on greedy-eval return vs **real env steps**; ≥3 seeds, mean ± std. Disclose the per-step
  compute gap (model-based do ~G× the gradient work).
- Same eval seeds across all algorithms (the drone already enforces this via
  `experiment.eval_seeds`).

═══════════════════════════════════════════════════════════════════════════════
## 7. FILES TO TOUCH
═══════════════════════════════════════════════════════════════════════════════

- `training/curriculum.py` — **new**; port `RealRatioSchedule` from the sister repo verbatim.
- `training/train.py` — two-buffer + batch-level mixing in the update; `start_step` selection;
  save policy+ensemble+meta on best; final summary eval; telemetry.
- `algorithms/sac.py` — add a thin `sac_update_mixed(...)` helper (do NOT change the SAC math).
- `models/ensemble.py` — no logic change; only ensure a `state_dict`/save path exists for the best
  checkpoint.
- `configs/macura_drone.yaml` — add `sac.real_ratio: 0.05`, a `selection:` block
  (`start_step`, `eval_every`, `final_eval_episodes`), and the OFF-by-default `curriculum:` block.
- `viz/plots.py` — add `plot_real_ratio` and confirm the sample-efficiency / rollout-depth figures.
- `tests/` — add the §8 tests.

═══════════════════════════════════════════════════════════════════════════════
## 8. FAST TESTS (correctness, not performance — minutes, CPU)
═══════════════════════════════════════════════════════════════════════════════

- `RealRatioSchedule`: `constant(0.05)(x)==0.05` for any `x`; with the default knots
  `real_ratio(0)==0.90`, `real_ratio` at the mid-knot step ≈ 0.20, and the tail clamps to 0.05.
- Mixing: with `real_ratio=1.0`, **zero** model-buffer batches are drawn; with `0.0`, zero real
  batches; with `0.5` over many draws the split is ≈50/50 (seeded RNG).
- Default config has `curriculum.enabled=false` → schedule is `constant(sac.real_ratio)` →
  realised `real_pct` ≈ 5% (the paper-faithful default), proving the structural fix landed.
- MACURA's `t_max` / rollout strategy is **unchanged** (κ-truncated, not hand-scheduled).
- Best-checkpoint save writes **both** policy and ensemble for model-based agents.
- ONE tiny smoke run per model-based algorithm (≤500 steps, warmup ≤200, 1 seed) with curriculum
  ON: runs without error and the realised `real_pct` falls over steps as scheduled.
- Print PASS/FAIL + wall-time. No long runs locally; the full 20k–30k × 3-seed run is Colab/GPU.

═══════════════════════════════════════════════════════════════════════════════
## DEFINITION OF DONE
═══════════════════════════════════════════════════════════════════════════════

1. Model-based agents train via **two buffers + batch-level real/imagined mixing**; default
   `real_ratio=0.05` (paper-faithful) — the current 0%-real deviation is gone.
2. `RealRatioSchedule` ported; curriculum is **config-gated and OFF by default**, shared across
   MACURA/MBPO/M2AC, controls `real_ratio` only (never rollout length).
3. `train.py` selects best by periodic greedy eval (with `start_step`), saves policy+ensemble+meta,
   runs one final summary eval, logs `real_pct`/`imagined_pct`/`kappa`/`rollout_length`.
4. No algorithm update/rollout rule changed (MACURA/MBPO/M2AC/SAC math md5-stable).
5. §8 tests pass; plots include sample-efficiency, rollout-depth (MACURA signature), and
   scheduled-vs-realised real_ratio.
6. Headline comparison reported at fixed `real_ratio`; curriculum reported separately as a
   documented deviation/ablation, with the fixed-vs-curriculum figure.
