# Prompt — Rebuild `colab.ipynb` into a paper-vs-project study guide

Copy everything below the line into the agent. It is written to be self-contained and grounded in this repo.

---

## ROLE & GOAL

You are a senior RL research engineer. Rebuild the notebook `colab.ipynb` in the
`macura-drone` repo into a **complete, self-explanatory study notebook** that reads
like a guide. It must do two things, in this order:

1. **Explain the paper** — *Trust the Model Where It Trusts Itself: Model-Based
   Actor-Critic with Uncertainty-Aware Rollout Adaption* (MACURA),
   Frauenknecht et al., ICML 2024, arXiv:2405.19014 — what problem it solves, the
   four algorithms it studies, the math it introduces (GJS divergence, adaptive κ,
   UTD scaling), and how each piece maps onto a file in this repo.
2. **Compare paper vs project** — run the four algorithms (MACURA, MBPO, M2AC, SAC)
   across environments, and for every design choice (e.g. **pink vs white
   exploration noise**, fixed vs adaptive rollout length, masking vs truncation)
   explain *what the paper does*, *what this project does*, *why*, and *what the
   plots show*.

Read these files first and stay faithful to them — do not invent APIs:
`project.md`, `README.md`, `tasks.md`, `ARCHITECTURE_macura-drone.md`,
`paper/2405.19014v3.pdf`, `configs/macura_drone.yaml`,
`algorithms/{macura,mbpo,m2ac,sac}.py`, `models/ensemble.py`,
`envs/drone_env.py`, `training/train.py`, `viz/plots.py`.

**Hard constraints**
- **All training MUST run on Colab GPU.** The notebook is a Colab notebook: assume
  `Runtime → GPU`. Early in the notebook, assert a GPU is present
  (`torch.cuda.is_available()` is `True`, print `torch.cuda.get_device_name(0)`) and
  **stop with a clear message if not**. Build SAC/ensemble with `device="cuda"`,
  set `MUJOCO_GL=egl` for headless rendering on the Colab GPU, and confirm models +
  batches actually live on the GPU. Do not silently fall back to CPU.
- Every `.py` file stays a library of pure functions — put **no** algorithm logic
  in the notebook; the notebook only imports, orchestrates, narrates, and plots.
  If a plot you need does not exist, add a pure function to `viz/plots.py` and call
  it from the notebook.
- Keep the existing Colab bootstrap working (Drive mount, GITHUB_TOKEN clone,
  `bash/setup_colab.sh`, config load, `MUJOCO_GL=egl`). Do not break cells 0–3.
- Use `configs/macura_drone.yaml` as the single source of truth — no hard-coded
  hyperparameters in the notebook.
- Every figure must be saved to `{DRIVE}/plots/...` AND displayed inline.
- Make it runnable top-to-bottom: a `SMOKE = True` flag uses tiny step counts so the
  whole notebook executes quickly for review; `SMOKE = False` runs the full matrix.
- Every code cell is preceded by a markdown cell that explains *why* the cell exists
  and *what to look for* in its output. Write prose, not bullet soup.

---

## REQUIRED NOTEBOOK STRUCTURE

Build the notebook in these numbered sections. Sections 0–3 already exist — keep
them and improve their narration.

### Part 0 — Front matter & setup (keep + polish)
- **0.0 Title + abstract of the project** — one paragraph: MACURA in one sentence,
  the drone transfer, the four-way comparison, what the reader will see.
- **0.1 How to read this notebook** — the paper→project mapping, the `SMOKE` flag,
  where outputs are saved.
- **0.2–0.3** Drive mount, secure token, `setup_colab.sh`, config load (existing
  cells 0–9, cleaned up and explained).

### Part 1 — What the paper introduces (NEW, the conceptual core)
This part is mostly markdown + small illustrative plots; it must stand alone as a
study guide even before any training runs.
- **1.1 The problem: model exploitation.** Explain MBRL sample efficiency vs the
  danger of compounding model error. Include a small schematic/plot illustrating
  how error grows with rollout horizon (a synthetic toy curve is fine, clearly
  labelled as illustrative).
- **1.2 The four algorithms.** Reproduce the comparison table from `project.md`
  (MACURA / MBPO / M2AC / SAC: type, rollout strategy, uncertainty use, role).
  For each, name the file and key functions in this repo that implement it.
- **1.3 The probabilistic ensemble.** Explain what `models/ensemble.py` learns
  (per-member Gaussian next-state deltas), and why member *disagreement* =
  epistemic uncertainty. **Plot:** ensemble predictions vs ground truth on a toy
  1-D function showing members agreeing in-distribution and fanning out
  out-of-distribution.
- **1.4 GJS uncertainty (paper Eq. 15–19).** State the geometric Jensen–Shannon
  divergence, show it is closed-form for diagonal Gaussians, and point to
  `compute_gjs` in `algorithms/macura.py`. **Plot:** GJS as a function of ensemble
  spread (synthetic) so the reader sees what "high uncertainty" looks like.
- **1.5 Adaptive threshold κ (Eq. 21) and the rollout rule.** Explain the
  ζ-quantile × ξ running-mean κ, and the "continue while uncertainty < κ, else
  truncate" rule (`update_kappa`, `macura_rollout`). **Plot:** a cartoon of one
  rollout being truncated at the step where GJS crosses κ.
- **1.6 UTD scaling (Eq. 22).** Explain gradient-steps scaling with model-buffer
  fullness (`gradient_steps`) and why it is a confound to control.
- **1.7 Imports & sanity.** Import every project module, print versions
  (torch, sb3, mujoco, gymnasium, numpy), confirm the import chain, and print the
  resolved config so the reader sees the exact hyperparameters used downstream.

### Part 2 — The environment study (keep existing, expand)
Keep existing study cells (random-policy instability, free dynamics, state
distribution, reward landscape, action response, rendered frame) but add narration
explaining what each shows about *why a drone is a hard test for MACURA*
(instability → fast error compounding). Add if missing:
- **2.x Action→state response grid** confirming thrust maps to climb (sanity).
- **2.x Reward landscape** as a 2-D heatmap over position error (smooth ⇒
  model-friendly).
- **2.x A short rendered GIF/frame** of the Skydio X2 hovering.

### Part 3 — Paper vs project: design choices, explained (NEW)
A dedicated comparison section. For **each** of the following, write a markdown
sub-section structured as *Paper does X / This project does Y / Why / Caveat*, and
where possible back it with a plot:
- **Exploration noise — pink vs white.** Explain what pink (1/f, temporally
  correlated) noise is vs white (uncorrelated) noise, that the **paper uses pink
  noise for MACURA only**, and that *this project applies one scheme to all four to
  avoid an unfair advantage* (see `tasks.md` Phase 4 and `_explore` in
  `training/train.py`). **Plot:** time series + power spectral density (PSD) of a
  pink vs white noise signal, showing the 1/f slope — so the reader literally sees
  the difference.
- **Rollout strategy** — adaptive truncation (MACURA) vs fixed truncated-linear
  schedule (MBPO) vs masking (M2AC) vs none (SAC). **Plot:** rollout length vs
  training step for each strategy on the same axes (schematic + later the real
  data from runs).
- **Reward & termination** — known analytic dense reward vs learned reward; why
  this project shares one `known_reward_fn`/`termination_fn` across all four.
- **Backbone** — paper builds on `mbrl-lib`; this project uses Stable-Baselines3
  (explain the Python 3.12 / numpy 2 reason from `README.md`/`tasks.md`).
- **Domain** — paper validates on MuJoCo locomotion (stable); this project targets
  an unstable quadrotor. State the hypothesis the experiments will test.

### Part 4 — Running the comparison (training)
- **4.1 Smoke test** — `train_one('macura', cfg_smoke, DRIVE, seed=0)`, then mbpo,
  m2ac, sac; confirm curves log, κ updates and is finite, rollouts respect
  `termination_fn`.
- **4.2 Full matrix** — 4 algorithms × N seeds (3 for review, 5 for final) to
  `total_env_steps`; checkpoint/resume to Drive. Make this cell idempotent (skip a
  run if its JSON already exists in Drive).

### Part 5 — Results & figures (the payoff — print as many useful plots as possible)
Generate **all** of the following, each with a markdown caption stating the claim it
tests and the expected ordering (MACURA ≳ M2AC ≳ MBPO > SAC early; MBPO expected to
wobble/collapse on the unstable drone). Add new pure functions to `viz/plots.py` as
needed.
- **5.1 Sample-efficiency curve** — eval return vs *real* env steps, mean ± std
  bands over seeds, all four algorithms (`plot_sample_efficiency`).
- **5.2 Failure-rate curve** — fraction of episodes that crash/leave envelope over
  training (`plot_failure_rate`).
- **5.3 Final-policy quality** — best policy per algorithm: hover/tracking error bar
  chart (`plot_final_quality`).
- **5.4 MACURA signature** — rollout length AND κ vs training step on twin axes
  (`plot_rollout_depth`); this is the headline figure showing length grows as the
  model becomes reliable and κ falls.
- **5.5 GJS / uncertainty over training** — mean first-step GJS vs step; show it
  decreasing as the ensemble learns.
- **5.6 Rollout-length distributions** — histogram of MACURA rollout lengths early
  vs late in training (adaptivity made visible).
- **5.7 Sample efficiency to a target return** — bar chart of real steps each
  algorithm needs to hit a fixed return threshold (the "does the model help?" number).
- **5.8 Per-seed spaghetti + mean** — for at least the sample-efficiency plot, show
  individual seeds faintly behind the mean so variance is honest.
- **5.9 Wall-clock vs sample efficiency** — note the caveat from `README.md` that
  compute, not samples, is the Colab bottleneck; plot return vs wall-clock too.
- **5.10 MuJoCo sanity anchor (if time)** — run the same code on a stock task
  (e.g. Hopper) and overlay against the paper's qualitative curve to prove the
  implementation (`tasks.md` Phase 6).
- **5.11 Best-policy rollout GIF** per algorithm, saved to Drive (see Part 5b).

### Part 5b — Record & download performance videos (REQUIRED)
At the **end of training, during the final evaluation**, render and save a video of
how each trained model actually flies — these are the presentation/representation
artifacts.
- For **each** algorithm (and at least the best seed), load the best/final
  checkpoint and run a deterministic evaluation episode while capturing frames from
  the MuJoCo Skydio X2 (`env.render()` with `MUJOCO_GL=egl` on the Colab GPU).
- Encode each rollout to an **`.mp4`** (and an optional `.gif`) using `imageio`
  (already in requirements), e.g. `imageio.mimsave(path, frames, fps=30)`.
- **Save every video to Google Drive** under
  `{DRIVE}/videos/<algo>_seed<seed>.mp4`, and print the Drive path for each so it
  can be downloaded.
- Add a final cell that displays the videos inline in the notebook
  (`IPython.display.Video` or HTML5 `<video>`), and offer a direct download via
  `google.colab.files.download(path)` for each saved video.
- Put the recording logic in a pure helper (e.g. `record_policy_video(agent, env,
  cfg, save_path)` in `viz/plots.py` or `training/train.py`) — the notebook only
  calls it. Optionally label this as a side-by-side "MACURA vs MBPO vs M2AC vs SAC"
  comparison reel if time permits.

### Part 6 — Discussion & verification (NEW, required)
- **6.1 Read the results back against the hypotheses** in Part 3 — did adaptivity
  matter (MACURA vs MBPO)? Did the model help (MACURA vs SAC)? Truncation vs
  masking (MACURA vs M2AC)? Be honest: a mixed/negative result on the drone is a
  valid finding.
- **6.2 Fairness audit** — confirm the controls from `tasks.md` Phase 4 held: one
  exploration scheme for all, UTD reported both ways, MBPO schedule tuned (not a
  strawman), GJS finite (no covariance blow-up). State any deviations.
- **6.3 Reproducibility footer** — print seeds, config hash, library versions,
  and the Drive paths of every artifact produced.

---

## STYLE & QUALITY BAR
- Narrate like a guide: each section opens with 2–4 sentences of plain-language
  context before any code. Assume the reader knows RL basics but not this paper.
- Label every plot axis, give every figure a title and a one-line takeaway caption.
- Use consistent per-algorithm colors across all figures.
- Prefer many small, well-labelled figures over a few cramped ones.
- Where you state a paper fact (equation number, "uses pink noise", "MuJoCo
  locomotion"), it must be verifiable in `paper/2405.19014v3.pdf` — cite the
  section/equation. Do not overclaim.
- At the end, print a short "what to run next / known gaps" list tied to `tasks.md`.

## DELIVERABLE
A rewritten `colab.ipynb` plus any new pure functions added to `viz/plots.py`
(and only `viz/plots.py` — no logic elsewhere in the notebook). It must execute
top-to-bottom with `SMOKE = True` without errors before you hand it back.
