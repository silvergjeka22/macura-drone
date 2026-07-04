# Prompt — Build the definitive `colab.ipynb`: MACURA wins in 20k steps, with video demos

Copy everything below the line into the coding agent. It is self-contained and grounded in this repo's real APIs. The agent must read the listed files before writing a single cell — do not invent function signatures.

---

## ROLE

You are a senior RL research engineer. You are rebuilding `colab.ipynb` in the `macura-drone` repo into **one clean, correct, top-to-bottom notebook** that (a) explains the MACURA paper, (b) runs all four algorithms on the unstable-quadrotor hover task in **20,000 real environment steps**, (c) **demonstrates that MACURA wins** — best sample efficiency and stability — and (d) at the end **records a flight video per algorithm and saves them to Google Drive** so the results can be watched as a demo.

## NON-NEGOTIABLE OUTCOME

By the end of the full (`SMOKE = False`) run at `total_env_steps = 20000`, the results must support this ordering, and the notebook must state it explicitly with the figures that prove it:

> **MACURA ≳ M2AC ≳ MBPO, all above SAC in sample efficiency; MACURA is the most stable (lowest failure rate) and reaches the highest final return.**

This win must be **earned honestly**, not faked. It is currently NOT guaranteed because `tasks.md` **Phase 4** lists real correctness/fairness bugs that can hand MACURA an accidental win *or* cripple it. You must fix those bugs (see "CORRECTNESS GATE" below) so that MACURA wins because its adaptive rollout is genuinely better — not because of a broken baseline or a scale artifact. If, after the fixes, MACURA does not win, you must say so plainly and diagnose why; do not doctor numbers.

---

## STEP 1 — READ THESE FIRST (do not skip)

Read and stay faithful to the real APIs in:

- `project.md`, `README.md`, `ARCHITECTURE_macura-drone.md`, `tasks.md`
- `paper/2405.19014v3.pdf` — the MACURA paper. Cite exact equation/section numbers when you state a paper fact (GJS = Eq. 15–19, adaptive κ = Eq. 21, UTD scaling = Eq. 22, pink noise = Sec. 6/7).
- `configs/macura_drone.yaml` — the single source of truth for all hyperparameters. The `profiles.fast` block already sets `total_env_steps: 20000` — that is the 20k run.
- `envs/drone_env.py`, `models/ensemble.py`
- `algorithms/{sac,macura,mbpo,m2ac}.py`
- `training/train.py`, `viz/plots.py`

**Real signatures you MUST call (do not rewrite them into the notebook):**

```python
env, obs_dim, act_dim = drone_env.make_env(cfg["env"], seed=SEED, render=False)
reward_fn = drone_env.known_reward_fn(cfg["env"]); done_fn = drone_env.termination_fn(cfg["env"])
run = trainer.train_one(algo_name, cfg, drive_dir, seed)      # returns run_dict, saves best ckpt + logs to Drive
metrics = trainer.evaluate(agent, env, eval_episodes)
results = trainer.evaluate_best(cfg, drive_dir, device, seeds, eval_episodes)
video_paths = trainer.record_best_videos(cfg, drive_dir, device, seeds, seconds=12)  # -> {"<algo>_seed<seed>": mp4_path}
```

`viz/plots.py` already provides: `plot_sample_efficiency`, `plot_sample_efficiency_spaghetti`, `plot_failure_rate`, `plot_final_quality`, `plot_rollout_depth`, `plot_gjs_over_training`, `plot_rollout_length_hist`, `plot_steps_to_target`, `plot_real_ratio`, `plot_return_vs_wallclock`, plus the conceptual figures (`plot_model_error_growth`, `plot_ensemble_toy_1d`, `plot_gjs_vs_spread`, `plot_rollout_truncation_cartoon`, `plot_noise_comparison`, `plot_rollout_strategy_schematic`, `plot_reward_heatmap_2d`) and env-study figures. **Reuse them.** Only add a new plotting function if a required figure is missing, and add it to `viz/plots.py` (never put logic in the notebook).

---

## STEP 2 — CORRECTNESS GATE (fix before claiming any win)

Every `.py` stays a library of pure functions; make these fixes **in the modules**, not the notebook. Each maps to a `tasks.md` Phase 4 item. After each fix, add a one-line note in the notebook explaining what was fixed and why it makes the comparison fair.

1. **Pink noise is not actually implemented in training** (`tasks.md` Phase 4, high). `_explore` in `training/train.py` adds white Gaussian noise for every non-deterministic type, so `pink_noise` == `white_noise` at train time even though the PSD *plot* is correct. Wire `_make_noise`/`_PinkNoise` (already written in `training/train.py`) into `_explore` so the configured `exploration.type: pink_noise` genuinely produces temporally-correlated 1/f action noise, reset per episode. Apply the **same** scheme to all four algorithms (fairness — the paper gives pink noise to MACURA only; we give it to everyone so no algorithm gets a free exploration edge).

2. **GJS computed in unnormalized delta space** (Phase 4, high). `train_ensemble` normalizes inputs but not the delta targets `y = next_obs - obs`, so `compute_gjs` sums uncertainty across obs dims with wildly different physical scales (position vs angular velocity) and is dominated by large-magnitude dims. Normalize the delta targets in `models/ensemble.py` (store the target mean/std, predict in normalized space, un-normalize in `predict`) so `u_GJS` is scale-balanced. This is what makes κ and adaptive truncation meaningful — without it MACURA's core signal is noise.

3. **SAC baseline UTD fairness** (Phase 4, high). Model-free SAC currently runs `gradient_steps_max` (8 in fast / 20 full) updates per real step; standard SAC uses ~1, and UTD-20 plain SAC often destabilizes → an unfairly weak baseline. Add `sac.baseline_gradient_steps: 1` to the config and use it for the SAC run only. A legitimate MACURA win must beat a *properly tuned* SAC, not a sabotaged one.

4. **Tune MBPO's schedule so it is not a strawman** (Phase 4, med). The default `rollout.mbpo.rollout_schedule: [1, 15, 1, 15000]` barely leaves length 1 within a 20k run. Pick a schedule that actually ramps within 20k (e.g. end_step ≈ 12000) so MBPO is a real fixed-horizon competitor that then over-imagines on the unstable drone — that is the honest way MACURA beats it.

5. **UTD confound** (Phase 4, med). MACURA uses adaptive UTD (Eq. 22) while MBPO/M2AC use fixed. Run MACURA **both ways** (adaptive and fixed-G) at least for one seed and show a small figure, so the reader sees the rollout-adaptation win is not just extra gradient steps.

6. **SB3 manual-training sanity** (Phase 2, high risk). Before the full run, verify on the installed Stable-Baselines3 that `SAC._setup_learn` sets a logger (else `agent.train()` crashes) and that `replay_buffer.add` ordering / `optimize_memory_usage=False` match what `_bulk_add` assumes. Do this in the smoke test; fix `algorithms/sac.py` if the version differs.

**Do not** claim MACURA wins until items 1–4 are fixed and the smoke test passes.

---

## HARD CONSTRAINTS

- **GPU-only.** Early cell: assert `torch.cuda.is_available()`, print `torch.cuda.get_device_name(0)`, and **stop with a clear message if absent**. Build SAC + ensemble with `device="cuda"`. Set `MUJOCO_GL=egl` for headless render. No silent CPU fallback.
- **Single source of truth = `configs/macura_drone.yaml`.** No hard-coded hyperparameters in the notebook. Deep-merge `profiles.fast` for the 20k run; the notebook only selects the profile.
- **`SMOKE` flag.** `SMOKE = True` → tiny steps (e.g. `total_env_steps ≈ 1500`, 1 seed) so the whole notebook runs end-to-end in minutes for review. `SMOKE = False` → the real 20k matrix (4 algos × ≥3 seeds; 5 for final figures). The notebook must execute top-to-bottom with `SMOKE = True` without error before you hand it back.
- **Idempotent / resumable.** Skip a run if its `logs/<algo>_seed<seed>.json` already exists in Drive (compute, not samples, is the Colab bottleneck — see README §7).
- **Every figure both saved and shown.** Save to `{DRIVE}/plots/<name>.png` AND display inline.
- **Keep the working bootstrap** (Drive mount, `GITHUB_TOKEN` clone via `bash/setup_colab.sh`, config load, `MUJOCO_GL=egl`). Improve its narration; don't break it.
- **Every code cell is preceded by a markdown cell** in plain language: why this cell exists and what to look for in its output. Prose, not bullet soup.

---

## NOTEBOOK STRUCTURE

**Part 0 — Front matter & setup.** Title + one-paragraph abstract (MACURA in a sentence, the drone transfer, the four-way comparison, the 20k-step claim, and that a video demo comes at the end). "How to read this / the `SMOKE` flag / where outputs land." Then the existing Drive-mount → token → `setup_colab.sh` → config-load cells, cleaned and explained. Assert GPU here.

**Part 1 — What the paper introduces (conceptual, mostly markdown + the synthetic figures).** Model exploitation (`plot_model_error_growth`); the four-algorithm table from `project.md` with the file/function that implements each; the probabilistic ensemble and why disagreement = epistemic uncertainty (`plot_ensemble_toy_1d`); GJS divergence Eq. 15–19 (`plot_gjs_vs_spread`, point to `compute_gjs`); adaptive κ Eq. 21 and the truncation rule (`plot_rollout_truncation_cartoon`, point to `update_kappa`/`macura_rollout`); UTD scaling Eq. 22 (`gradient_steps`). Close with an imports-and-versions sanity cell that prints the resolved 20k config.

**Part 2 — Environment study.** Keep the existing study figures (rollout traces, state distributions, reward landscape/`plot_reward_heatmap_2d`, `plot_action_response`, a rendered Skydio X2 frame/filmstrip). Narrate each toward one point: **the drone is unstable → model error compounds fast → this is exactly where adaptive truncation should matter.**

**Part 3 — Paper vs project design choices (the fairness story).** For each of: pink vs white exploration noise (`plot_noise_comparison`; note the Phase-4 fix you made), rollout strategy per algorithm (`plot_rollout_strategy_schematic`), known analytic reward/termination shared by all, SB3 backbone vs paper's mbrl-lib (Python 3.12/numpy 2 reason), and MuJoCo-locomotion vs unstable-quadrotor domain — write a short *Paper does X / This project does Y / Why / Caveat* block. State the hypothesis the 20k experiment tests.

**Part 4 — Run the comparison.**
- 4.1 **Smoke test**: `train_one('macura', cfg_smoke, DRIVE, 0)` then mbpo, m2ac, sac. Confirm curves log, κ finite and updating, rollouts respect `termination_fn`, checkpoints reload. This is also where you verify the SB3 API (Correctness Gate item 6).
- 4.2 **Full 20k matrix**: 4 algorithms × seeds to `total_env_steps=20000`, idempotent, checkpointing best to Drive. Print a live per-eval line (return, failure rate, real/imagined split) so progress is visible.

**Part 5 — Results (the payoff).** Generate, each with a caption stating the claim it tests and the expected ordering:
- 5.1 Sample-efficiency curve, mean ± std (`plot_sample_efficiency`) — **the headline win figure.**
- 5.2 Per-seed spaghetti + mean (`plot_sample_efficiency_spaghetti`) — honest variance.
- 5.3 Failure-rate over training (`plot_failure_rate`) — MACURA lowest, MBPO wobble/spike.
- 5.4 Final-policy quality bar chart (`plot_final_quality`) — MACURA tallest.
- 5.5 Steps-to-target bar chart (`plot_steps_to_target`, pick a target return) — the "does the model help / how much faster" number.
- 5.6 **MACURA signature**: rollout length & κ vs step (`plot_rollout_depth`) — length grows as the model becomes reliable, κ falls.
- 5.7 GJS over training (`plot_gjs_over_training`) — uncertainty falls as the ensemble learns.
- 5.8 Rollout-length histogram early vs late (`plot_rollout_length_hist`) — adaptivity made visible.
- 5.9 UTD control: MACURA adaptive-G vs fixed-G overlay (Correctness Gate item 5) — the win survives holding UTD fixed.
- 5.10 Real/imagined mixing audit (`plot_real_ratio`) and return-vs-wall-clock (`plot_return_vs_wallclock`, the compute caveat).

**Part 5b — VIDEO DEMO (required, the thing the user asked for).** At the end, after training, load each algorithm's best checkpoint and record a deterministic flight clip:
- Call `trainer.record_best_videos(cfg, DRIVE, device, seeds=[best_seed], seconds=12)`. It renders the Skydio X2 with `MUJOCO_GL=egl` and saves `{DRIVE}/videos/<algo>_seed<seed>.mp4` via `viz.plots.record_policy_video` (imageio). Print each Drive path.
- Add a cell that **displays every video inline** (`IPython.display.Video` or an HTML5 `<video>` grid) so the notebook itself is the demo, and offer `google.colab.files.download(path)` for each.
- If time permits, build a **side-by-side "MACURA vs MBPO vs M2AC vs SAC" reel** so the viewer literally sees MACURA hovering stably while the others drift/tumble. Put any new stitching logic in `viz/plots.py`, not the notebook.
- Confirm the videos actually landed in Drive (list `{DRIVE}/videos/`).

**Part 6 — Discussion & verification.**
- 6.1 Read results against the Part-3 hypotheses: did adaptivity matter (MACURA vs MBPO)? did the model help (MACURA vs SAC)? truncation vs masking (MACURA vs M2AC)? State the win — or, honestly, any part that didn't hold and the likely cause.
- 6.2 Fairness audit: confirm the Correctness-Gate fixes held (one exploration scheme, tuned MBPO, SAC UTD documented, GJS finite and scale-balanced, MACURA win survives fixed-G).
- 6.3 Reproducibility footer: seeds, config hash, library versions, and the Drive paths of every plot, checkpoint, log, and video produced.

---

## STYLE & QUALITY BAR

- Narrate like a study guide: 2–4 sentences of plain context before each cell. Assume RL basics, not this paper.
- Label every axis, title every figure, give each a one-line takeaway caption. Use the per-algorithm colors in `viz/plots.ALGO_COLORS` consistently (MACURA blue, MBPO red, M2AC green, SAC gray).
- Prefer many small clear figures over a few cramped ones.
- Any paper claim must be verifiable in `paper/2405.19014v3.pdf` with a section/equation cite. Do not overclaim.

## VERIFICATION BEFORE HANDOFF (do this, report the result)

1. Run the whole notebook with `SMOKE = True` end-to-end — zero errors, every figure renders, at least one `.mp4` is written and displays inline.
2. Confirm the six Correctness-Gate fixes are in the modules and referenced in the notebook.
3. Sanity-check that the pipeline *can* produce the target ordering: on the smoke run, verify curves are monotone-ish and κ/GJS are finite and moving. (Ordering only becomes meaningful at 20k, but broken pipelines show up here.)
4. Print a final "what to run next / known gaps" list tied to `tasks.md` (e.g. raise to 5 seeds, MuJoCo/Hopper anchor Phase 6, xi sweep Phase 5).

## DELIVERABLE

A rewritten `colab.ipynb` plus any new pure functions added **only** to `viz/plots.py` (and the Correctness-Gate fixes in `envs/`, `models/`, `algorithms/`, `training/`). It must run top-to-bottom under `SMOKE = True` before handoff, and under `SMOKE = False` it must produce the 20k-step figures and Drive-saved videos that demonstrate MACURA's win.
