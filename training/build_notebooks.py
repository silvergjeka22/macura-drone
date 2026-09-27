"""Builds the 4 training notebooks from ONE template, so they are identical except for their seeds and algorithms.

    python training/build_notebooks.py        -> training/<name>/<name>.ipynb + kernel-metadata.json (x4)

Edit the template here, never the notebooks themselves, then run this again.
"""

import json
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
# (folder / notebook name, Kaggle account, seeds, algorithms trained at the same time)
NOTEBOOKS = [
    ("macura_mbpo_seeds01", "nouradon", [0, 1], ["macura", "mbpo"]),
    ("m2ac_sac_seeds01", "nouradon", [0, 1], ["m2ac", "sac"]),
    ("macura_mbpo_seeds23", "silvergjeka01", [2, 3], ["macura", "mbpo"]),
    ("m2ac_sac_seeds23", "silvergjeka01", [2, 3], ["m2ac", "sac"]),
]
NAMES = {"macura": "MACURA", "mbpo": "MBPO", "m2ac": "M2AC", "sac": "SAC"}


def kernel_id(name, account):
    return f"{account}/train-{name.replace('_', '-')}"


def build(name, account, seeds, algos):
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    what = " and ".join(NAMES[a] for a in algos)
    table = "\n".join(f"| `{n}` | {', '.join(NAMES[a] for a in al)} | {s} | {acc} |"
                      + (" **<- this one**" if n == name else "") for n, acc, s, al in NOTEBOOKS)
    cells = [
        md(f"""# Training: {what}, seeds {seeds}

Four reinforcement-learning algorithms learn the same MuJoCo drone race from scratch: **MACURA**, **MBPO**, **M2AC**
and **SAC**. They share the SAC learner, the world-model ensemble, the data and the evaluation scenarios; only the way
they use the learned world model to imagine extra practice differs. MACURA stops each imagined trip where its models
disagree, so it does not trust its model everywhere (*Trust the Model Where It Trusts Itself*, ICML 2024).

The training is split into 4 identical notebooks on 2 Kaggle accounts that run at the same time:

| notebook | algorithms | seeds | account |
|---|---|---|---|
{table}

This notebook: **{what}**, **seeds {seeds}**, every training at the same time. It shows the training results, the
figures, a 30 s video of each seed's best checkpoint (with MACURA's live trust in its world model) and saves
everything in one file, **`{name}.zip`**, for the testing notebook (`testing/test.ipynb`).

1. Setup · 2. The task · 3. The protocol · 4. Training · 5. Results · 6. The world model and trust ·
7. Videos · 8. The results file"""),
        md("## 1. Setup"),
        code(f"""import os, sys, time, subprocess

NAME     = "{name}"
SEEDS    = {seeds}
ALGOS    = {algos}        # trained at the same time, for every seed
TASK     = "race3"
STEPS    = 50000                  # per training: 5,000 random warm-up + 45,000 learning
UPDATES  = 12                     # SAC updates per real step for MBPO and M2AC (MACURA: adaptive, up to 16)
VIDEOS   = True                   # a 30 s flight of each seed's best checkpoint
VIDEO_SECONDS = 30
REPO, BRANCH = "silvergjeka22/macura-drone", "race3"
ROOT = "/tmp/macura-drone"        # the code (not part of the saved output)
OUT  = "/kaggle/working/runs"     # everything this notebook produces

T0 = time.time()
DEADLINE = T0 + 11.0 * 3600       # Kaggle stops a session at 12 h and then saves nothing: training stops at 11 h
print("=" * 78)
print(f"  {{NAME}}   |   SEEDS {{SEEDS}}   |   {{' + '.join(a.upper() for a in ALGOS)}}   |   task {{TASK}}   |   {{STEPS:,}} steps")
print("=" * 78)
os.environ.update({{"MACURA_TASK": TASK, "MACURA_STEPS": str(STEPS), "MACURA_SEEDS": " ".join(map(str, SEEDS)),
                   "MACURA_ALGOS": " ".join(ALGOS), "MBPO_UTD": str(UPDATES), "M2AC_UTD": str(UPDATES),
                   "MACURA_OUTPUT_ROOT": OUT, "MACURA_DEADLINE": str(DEADLINE)}})

token = os.environ.get("GITHUB_TOKEN")
if not token:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
if os.path.isdir(os.path.join(ROOT, ".git")):
    subprocess.run(["git", "-C", ROOT, "fetch", "--depth", "1", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", ROOT, "reset", "--hard", "FETCH_HEAD"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                    f"https://{{token}}@github.com/{{REPO}}.git", ROOT], check=True)
sys.path.insert(0, ROOT)
print("code:", subprocess.run(["git", "-C", ROOT, "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip())"""),
        code("""from src.bootstrap import setup
setup(root=ROOT)

import json, shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
from IPython.display import Image, Markdown, Video, display
import src.config.config as cfg
from src.analysis import results as R
from src.viz import figures as F, video as V

N_JOBS = min(4, os.cpu_count() or 1)
N_GPU = torch.cuda.device_count()
print(f"config: task {cfg.TASK} | seeds {cfg.SEEDS} | max lift loss {cfg.ENV['vrs_loss']:.0%} | updates per step: "
      f"MBPO {cfg.ROLLOUT['mbpo']['fixed_gradient_steps']}, M2AC {cfg.ROLLOUT['m2ac']['fixed_gradient_steps']}, MACURA up to "
      f"{cfg.ROLLOUT['macura']['gradient_steps_max']} (xi {cfg.ROLLOUT['macura']['xi']}) | warm-up {cfg.WARMUP_RANDOM_STEPS}")
print(f"machine: {os.cpu_count()} CPU cores, {N_GPU} GPU(s) -> {N_JOBS} trainings at the same time")
os.makedirs(OUT, exist_ok=True)
json.dump({"notebook": NAME, "seeds": SEEDS, "algorithms": ALGOS, "task": TASK, "steps": STEPS, "updates": UPDATES,
           "started": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(T0))}, open(f"{OUT}/RUN_INFO.json", "w"), indent=2)

CHILD = '''
import sys
sys.path.insert(0, {root!r})
from src.config import config as cfg
from src.training.train import train_algo
train_algo({algo!r}, cfg.CFG, cfg.OUTPUT_ROOT, seeds=[{seed}])
'''


def _clock():
    return f"{(time.time() - T0) / 3600:4.1f} h"


def run_jobs(jobs, n_jobs, poll=20):
    # every training is its own process; all their printouts (evaluations, what the world model did, how the
    # policy flies) are shown here as they come, each line tagged with its algorithm and seed
    os.makedirs(f"{OUT}/joblogs", exist_ok=True)
    pending, running, seen = list(jobs), {}, {}

    def show(key, log):
        with open(log) as f:
            f.seek(seen.get(key, 0)); text = f.read(); seen[key] = f.tell()
        for line in text.splitlines():
            if line.startswith("["):
                print(line, flush=True)
            elif line.startswith("    model:") or line.startswith("    policy:"):
                print(f"[{key[0]} seed{key[1]}] {line.strip()}", flush=True)

    while pending or running:
        while pending and len(running) < n_jobs:
            algo, seed = pending.pop(0)
            if os.path.exists(f"{OUT}/logs/{algo}_seed{seed}.json"):
                print(f"[{_clock()}] {algo} seed {seed}: already finished, reloaded"); continue
            if time.time() > DEADLINE:
                print(f"[{_clock()}] {algo} seed {seed}: no session time left, skipped"); continue
            env = dict(os.environ, MACURA_SEEDS=str(seed), MACURA_ALGOS=algo, PYTHONUNBUFFERED="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            if N_GPU > 1:
                env["CUDA_VISIBLE_DEVICES"] = str(len(running) % N_GPU)
            log = f"{OUT}/joblogs/{algo}_seed{seed}.txt"
            p = subprocess.Popen([sys.executable, "-u", "-c", CHILD.format(root=ROOT, algo=algo, seed=seed)],
                                 env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
            running[(algo, seed)] = (p, log, time.time())
            print(f"[{_clock()}] started {algo} seed {seed}", flush=True)
        for key, (p, log, t0) in list(running.items()):
            show(key, log)
            if p.poll() is None:
                continue
            show(key, log)
            del running[key]
            if p.returncode == 0:
                print(f"[{_clock()}] >>> {key[0]} seed {key[1]} done in {(time.time() - t0) / 60:.0f} min", flush=True)
            else:
                print(f"[{_clock()}] >>> {key[0]} seed {key[1]} FAILED (exit {p.returncode}); last lines:\\n"
                      + "\\n".join(open(log).read().splitlines()[-15:]), flush=True)
        time.sleep(poll)
    print(f"[{_clock()}] all trainings done", flush=True)"""),
        md("""## 2. The task
A 6 x 6 m course flown clockwise through four gates: **P1 2 m -> P2 4 m -> the chute (an almost vertical 3 m drop)
-> P3 1 m -> P4 1.5 m** (24.3 m per lap), with two pillars. Every flight lasts 10 s from a random point of the course,
with a package of random weight (0-0.2 kg) and a steady wind (up to 0.5 N), both visible to the drone, plus hidden
gusts and rotor noise.

- **Flying**: an on-board stabiliser in altitude-hold mode; actions = sideways acceleration x, y, climb rate, yaw rate.
  A racing-line sensor gives the vector to the line and the direction 0.5 m ahead.
- **Reward**: speed along the course while near it, minus the distance outside a 0.35 m tube; a crash (flip, ground,
  pillar, more than 2.5 m off the course) ends the flight and costs 500.
- **The trap (race3)**: sinking faster than 1.2 m/s while upright and pushing loses lift, up to **50%**: dropping the
  chute too fast crashes. The learned models are wrong there and disagree about it, which MACURA can detect.
- **Lap progress** = how much of a full lap a 10 s flight covers (0-100%); a crash ends the flight.

Below: the course, the lift loss, the hidden wind, and the hand-written autopilot (not learned) as a reference."""),
        code("""F.plot_race_course(cfg.CFG, f"{OUT}/plots/task_course.png"); plt.show()
F.plot_lift_loss(cfg.CFG, f"{OUT}/plots/task_lift_loss.png"); plt.show()
F.plot_wind(cfg.CFG, save_path=f"{OUT}/plots/task_wind.png"); plt.show()
REFS = R.autopilot_reference(cfg.ENV, seed_base=100, episodes=20, cache=f"{OUT}/autopilot_reference.json")
for k, v in REFS.items():
    print(f"{k}: return {v['return']:.0f}, lap progress {100 * min(max(v['laps'], 0), 1):.0f}%, crash {100 * v['crash']:.0f}%")"""),
        md("""## 3. The protocol (the same in all 4 notebooks)
- **World model** (MACURA, MBPO, M2AC): 7 probabilistic networks, the 5 best used; retrained every 250 real steps;
  25,000 imagined trips from real states each time; each SAC batch = 5% real + 95% imagined; imagined data expires
  after 4 model rounds.
- **MACURA**: a trip stops once the models' disagreement (GJS) exceeds kappa = xi x running mean of the 95% quantile of
  first-step disagreement (xi = 0.5); its SAC updates follow how much it kept (up to 16 per real step).
- **MBPO**: 10-step trips; **M2AC**: keeps the least uncertain imagined steps (paper mode); both **12 updates per real
  step** (= MACURA's average in the race2 pilot), so no method simply trains more. **SAC**: no model, 1 update.
- **Exploration** (as in the MACURA paper, disclosed): MACURA pink noise, MBPO and M2AC without noise, SAC samples its
  policy. Everyone starts with 5,000 random steps.
- **Evaluation**: every 1,000 steps the greedy policy flies 20 fixed scenarios (seeds 100-119); the best one is saved
  and finally flies 30 fresh scenarios (seeds 1000-1029). The testing notebook later uses new ones (3000+)."""),
        md("""## 4. Training
Every training runs in its own process, all at the same time. Every 1,000 steps each one prints its evaluation
(return, **lap progress in %**, crash rate, drones broken so far), what its world model did (MACURA: trip length and
how much it trusts fast descents vs normal flight; MBPO / M2AC: how much of their imagined data MACURA's rule would
reject) and how the policy flies (fastest descent, time losing lift, time per phase). Each line starts with
`[algorithm seedN]`."""),
        code("""JOBS = [(a, s) for s in SEEDS for a in ALGOS]
print(NAME, "| SEEDS", SEEDS, "->", ", ".join(f"{a} seed {s}" for a, s in JOBS))
run_jobs(JOBS, N_JOBS)"""),
        md("""## 5. Results
The scores fixed before the runs: **average return while learning** and **drones broken after the warm-up**; then the
best checkpoint's test on 30 fresh scenarios (return, **lap progress**, crash rate). Curves: rolling mean over 5
evaluations (raw values faint); the dotted lines are the autopilot."""),
        code("""RUNS = R.load_runs(OUT)
missing = [(a, s) for a, s in JOBS if not any(r["algo"] == a and r["seed"] == s for r in RUNS)]
stopped = [f"{r['algo']} seed {r['seed']} at step {r['setup']['stopped_early_at']}" for r in RUNS
           if r.get("setup", {}).get("stopped_early_at")]
print(NAME, "| SEEDS", sorted({r["seed"] for r in RUNS}), "| missing:", missing or "none",
      "| stopped by the time limit:", stopped or "none")
display(Markdown(R.summary_markdown(RUNS)))
display(Markdown("### Per seed\\n\\n" + R.per_seed_markdown(RUNS)))

by = {(r["algo"], r["seed"]): R.run_scores(r) for r in RUNS}
if len(ALGOS) == 2 and all((a, s) in by for a in ALGOS for s in SEEDS):
    a, b = ALGOS
    GAP = [("avg_return", "avg return while learning", "{:+.0f}"), ("broken", "drones broken after warm-up", "{:+.0f}"),
           ("test_return", "final test return", "{:+.0f}"), ("test_success", "lap progress (test)", "{:+.0%}"),
           ("test_crash", "final test crash rate", "{:+.0%}")]
    rows = [f"| {R.NAMES[a]} - {R.NAMES[b]} | " + " | ".join(f"seed {s}" for s in SEEDS) + " |", "|---|" + "---|" * len(SEEDS)]
    rows += [f"| {lab} | " + " | ".join(fmt.format(by[(a, s)][k] - by[(b, s)][k]) for s in SEEDS) + " |" for k, lab, fmt in GAP]
    display(Markdown("### Seed by seed\\n\\n" + "\\n".join(rows)))

UTD = ["| algorithm | seed | updates per real step (mean) |", "|---|---|---|"]
UTD += [f"| {R.NAMES[r['algo']]} | {r['seed']} | " + (f"{np.mean(r['utd']):.1f}" if r.get("utd") else "1 (model-free)") + " |"
        for r in RUNS]
display(Markdown("### Updates per real step\\n\\n" + "\\n".join(UTD)))

CHUTE = ["| algorithm | seed | fastest descent, 2nd half (m/s) | losing lift, 2nd half | test: fastest descent | "
         "test: losing lift |", "|---|---|---|---|---|---|"]
for r in RUNS:
    late = np.asarray(r["steps"]) >= STEPS / 2
    sink, lift, fe = np.asarray(r["eval_max_sink"])[late], np.asarray(r["eval_liftloss"])[late], r["final_eval"]
    CHUTE.append(f"| {R.NAMES[r['algo']]} | {r['seed']} | {sink.max() if len(sink) else float('nan'):.2f} | "
                 f"{100 * lift.mean() if len(lift) else float('nan'):.1f}% | {fe.get('eval_max_sink', float('nan')):.2f} | "
                 f"{100 * fe.get('eval_liftloss', 0):.1f}% |")
display(Markdown("### The chute (lift loss starts at 1.2 m/s of sink)\\n\\n" + "\\n".join(CHUTE)))"""),
        code("""F.plot_learning_curves(RUNS, REFS, f"{OUT}/plots/learning_curves.png"); plt.show()
for a in ALGOS:
    rs = [r for r in RUNS if r["algo"] == a]
    if rs:
        F.plot_algo(rs, REFS, f"{OUT}/plots/{a}_per_seed.png"); plt.show()
F.plot_summary_table(RUNS, f"{OUT}/plots/summary_table.png"); plt.show()
F.plot_scores(RUNS, f"{OUT}/plots/scores.png"); plt.show()
F.plot_crash_by_payload(RUNS, f"{OUT}/plots/crash_by_payload.png"); plt.show()"""),
        md("""## 6. The world model and trust
How much of its imagination each method trusts (MACURA stops where its models disagree), MACURA's threshold kappa over
training, how much imagined data above MACURA's threshold MBPO / M2AC trained on, and the **world-model check**: each
model-based run's best policy flies 10 fresh scenarios and, at every step, the model's disagreement is compared with
its real one-step error (paper Fig. 10)."""),
        code("""print(R.trust_summary(RUNS))
model_based = [a for a in ALGOS if a != "sac"]
if model_based:
    F.plot_model_trust(RUNS, f"{OUT}/plots/model_trust.png"); plt.show()
if "macura" in ALGOS:
    F.plot_kappa(RUNS, f"{OUT}/plots/kappa.png"); plt.show()
if any(a in ALGOS for a in ("mbpo", "m2ac")):
    F.plot_untrusted_data(RUNS, f"{OUT}/plots/untrusted_data.png"); plt.show()
CHECKS = []
for a in model_based:
    for s in SEEDS:
        run = next((r for r in RUNS if r["algo"] == a and r["seed"] == s), None)
        if run is not None and R.ensemble_checkpoint(run):
            CHECKS.append(R.model_check(run, cfg.CFG))
if CHECKS:
    F.plot_model_check(CHECKS, f"{OUT}/plots/model_check.png"); plt.show()"""),
        md(f"""## 7. Videos: each seed's best checkpoint, {30} s
For every seed, the best checkpoint of each algorithm flies the same fresh scenario side by side for 30 s (longer than
the 10 s training flights; the policy just keeps flying). The trail turns red while a drone loses lift. MACURA's panel
shows its own world model's verdict at every step: **green = its models agree** (it would trust an imagined step
there), **orange = they disagree** (it would stop imagining)."""),
        code("""if VIDEOS:
    for s in SEEDS:
        runs_s = {R.NAMES[r["algo"]]: r for r in RUNS if r["seed"] == s}
        trust = {k: {"ensemble": R.ensemble_checkpoint(r), "kappa": R.kappa_at(r)}
                 for k, r in runs_s.items() if r["algo"] == "macura" and R.ensemble_checkpoint(r)}
        path = V.record_race(cfg.CFG, {f"{k}": R.checkpoint(r) for k, r in runs_s.items()},
                             f"{OUT}/videos/seed{s}_best_{VIDEO_SECONDS}s.mp4", seeds=(1010 + s,), trust=trust,
                             cols=len(runs_s), frame_step=3, fps=17, max_steps=int(VIDEO_SECONDS * 50))
        print(f"seed {s}:", path)
        if path:
            display(Video(path, embed=True, width=480 * len(runs_s)))"""),
        md(f"""## 8. The results file
Everything this notebook produced (logs, best checkpoints and world models, figures, videos, tables) goes into one
file, **`{name}.zip`**, in the notebook's output. Download all four notebooks on the computer with
`./kaggle.sh get`, then open `testing/test.ipynb`."""),
        code("""summary = "\\n\\n".join([f"# {NAME}: {', '.join(R.NAMES[a] for a in ALGOS)}, seeds {SEEDS}",
                         R.summary_markdown(RUNS), "## Per seed", R.per_seed_markdown(RUNS),
                         "## Updates per real step", "\\n".join(UTD), "## The chute", "\\n".join(CHUTE),
                         "## Trust", R.trust_summary(RUNS)])
open(f"{OUT}/summary.md", "w").write(summary + "\\n")
json.dump({"notebook": NAME, "seeds": SEEDS, "algorithms": ALGOS, "task": TASK, "steps": STEPS,
           "runs": [{"algo": r["algo"], "seed": r["seed"], **{k: v for k, v in R.run_scores(r).items()}} for r in RUNS],
           "finished": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())},
          open(f"{OUT}/results.json", "w"), indent=2, default=float)
zip_path = shutil.make_archive(f"/kaggle/working/{NAME}", "zip", OUT)
print(f"results file: {zip_path} ({os.path.getsize(zip_path) / 1e6:.1f} MB)")
for d in ("logs", "checkpoints", "plots", "videos"):
    print(f"  {d}/:", ", ".join(sorted(os.listdir(f"{OUT}/{d}"))) if os.path.isdir(f"{OUT}/{d}") else "-")
print(f"total time: {(time.time() - T0) / 3600:.1f} h")"""),
    ]
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}})
    folder = os.path.join(HERE, name)
    os.makedirs(folder, exist_ok=True)
    nbf.write(nb, os.path.join(folder, f"{name}.ipynb"))
    meta = {"id": kernel_id(name, account), "title": f"train-{name.replace('_', '-')}", "code_file": f"{name}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    with open(os.path.join(folder, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"{folder}/{name}.ipynb  ->  {meta['id']}")


if __name__ == "__main__":
    for spec in NOTEBOOKS:
        build(*spec)
