# tasks.md — MACURA-backflip roadmap

Actionable checklist. Markers: `[ ]` not started, `[/]` in progress, `[x]` done.
Priority tags: (high) / (med) / (low).

Backbone: **Stable-Baselines3 SAC** + our probabilistic ensemble + our rollout
strategies. Env: **Pogo**, a custom planar one-legged MuJoCo body
(`src/envs/assets/pogo.xml`) learning a backflip. The four algorithms share
everything except the rollout strategy.

---

## Phase 0 — Scaffold & env (done, validated locally)
- [x] Pure-function modules, config, bash setup, notebook, docs.
- [x] Pogo MuJoCo body + env with the shaped, **analytic** backflip reward and
  face-plant termination (real and imagined transitions scored identically).
- [x] Confirmed physically feasible: Pogo launches clear of the airborne gate and
  a crude scripted push already rotates ~half a flip.
- [x] GJS uncertainty + adaptive-κ + UTD scaling (math unit-checked).
- [x] Full pipeline smoke-tested end-to-end for all four algorithms on CPU
  (`tests/test_mixing.py`, 9/9 pass).

## Phase 1 — Colab bring-up (high)
- [ ] (high) Rename the GitHub repo to `macura-backflip` (or set `REPO_NAME`) so
  the notebook clone works; confirm the private clone + token scrub.
- [ ] (high) Run `notebooks/backflip.ipynb` top-to-bottom on a GPU runtime with zero
  errors; confirm at least one `.mp4` renders inline.
- [ ] (med) Confirm `MUJOCO_GL=egl` offscreen rendering works for the videos.

## Phase 2 — Learn the flip (high)
- [ ] (high) Train the full matrix (3 seeds) and check that Pogo reaches a real
  rotation (`eval_flips` climbing, some `landed_flip`).
- [ ] (high) If a full flip is not discovered from scratch, run the **curriculum**
  (`curriculum.stage1_boing` → `stage2_halfflip` → `stage3_flip`), warm-starting
  each stage with `train_one(..., init_ckpt=<prev best>)`.
- [ ] (med) Tune `reward.w_rotation`, `foot_air`, `w_height`, `exploration.scale`
  if the flip stalls or it spins without landing.

## Phase 3 — The comparison & figures (med)
- [ ] (med) Run 4 algorithms × 3 seeds (5 for final figures).
- [ ] (med) Produce (the notebook makes these five): sample efficiency, stuck-backflip
  rate, face-plant rate, final quality, MACURA rollout-depth/κ.
- [ ] (med) Check the expected ordering (MACURA ≳ M2AC ≳ MBPO > SAC; MBPO
  over-imagines the flight and face-plants more) and **report honestly** if it does
  not hold.

## Phase 4 — The demo (med)
- [ ] (med) Record best-checkpoint videos per algorithm; stitch the side-by-side reel.
- [ ] (med) Watch the saved policies live on macOS: `mjpython run_live_mac.py
  --compare runs/checkpoints`.

## Phase 5 — Extensions (low)
- [ ] (low) Sweep MACURA `xi` {1,3,5,10,20} as in the paper.
- [ ] (low) Harder variants: bigger init perturbation, double backflip, forward flip.
- [ ] (low) Process/sensor-noise robustness — does GJS still isolate epistemic
  uncertainty?

## Known risks (watch throughout)
1. (high) A backflip is hard to discover from scratch — rely on the curriculum and
   vigorous (pink-noise) exploration; a mixed result is a valid, honest finding.
2. (high) Compute, not samples, is the Colab bottleneck; rely on best-checkpoint +
   idempotent resume.
3. (med) Reward shaping can produce a "helicopter" (endless spinning) instead of a
   landed flip — the airborne gate + grounded-upright term guard this; watch
   `eval_success_rate` vs `eval_flips`.
4. (med) High RL variance — need ≥3 (ideally 5) seeds for legible bands.
