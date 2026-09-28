"""Builds the 4 training notebooks from one template (they differ only in name, seeds and algorithms).
Run: python training/build_notebooks.py"""

import json
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))

NOTEBOOKS = [
    # name,                 Kaggle account,  seeds,  algorithms
    ("macura_mbpo_seeds45", "nouradon", [4, 5], ["macura", "mbpo"]),
    ("m2ac_sac_seeds45", "nouradon", [4, 5], ["m2ac", "sac"]),
    ("macura_mbpo_seeds67", "silvergjeka01", [6, 7], ["macura", "mbpo"]),
    ("m2ac_sac_seeds67", "silvergjeka01", [6, 7], ["m2ac", "sac"]),
]

SETUP = '''import os, sys, time, subprocess

NAME = "{name}"
SEEDS = {seeds}
ALGOS = {algos}
STEPS = 50000
UPDATES = 12            # SAC updates per real step for MBPO and M2AC
VIDEO_SECONDS = 30
REPO, BRANCH = "silvergjeka22/macura-drone", "main"
ROOT = "/tmp/macura-drone"
OUT = "/kaggle/working/runs"
T0 = time.time()
DEADLINE = T0 + 11 * 3600    # stop training at 11 h (Kaggle stops at 12 h)

print(f"{{NAME}} | seeds {{SEEDS}} | {{ALGOS}} | {{STEPS}} steps")

os.environ.update({{
    "MACURA_STEPS": str(STEPS), "MACURA_SEEDS": " ".join(map(str, SEEDS)),
    "MACURA_ALGOS": " ".join(ALGOS), "MBPO_UTD": str(UPDATES), "M2AC_UTD": str(UPDATES),
    "MACURA_OUTPUT_ROOT": OUT, "MACURA_DEADLINE": str(DEADLINE),
}})

token = os.environ.get("GITHUB_TOKEN")
if not token:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                f"https://{{token}}@github.com/{{REPO}}.git", ROOT], check=True)
sys.path.insert(0, ROOT)'''

IMPORTS = '''from src.bootstrap import setup
setup(root=ROOT)

import json, shutil
import matplotlib.pyplot as plt
from IPython.display import Markdown, Video, display
import src.config.config as cfg
from src.analysis import results as R
from src.training.parallel import run_parallel
from src.viz import figures as F, video as V

os.makedirs(OUT, exist_ok=True)
json.dump({"notebook": NAME, "seeds": SEEDS, "algorithms": ALGOS, "steps": STEPS},
          open(f"{OUT}/RUN_INFO.json", "w"))'''

TASK_PLOTS = '''F.plot_race_course(cfg.CFG, f"{OUT}/plots/task_course.png"); plt.show()
F.plot_lift_loss(cfg.CFG, f"{OUT}/plots/task_lift_loss.png"); plt.show()
F.plot_wind(cfg.CFG, save_path=f"{OUT}/plots/task_wind.png"); plt.show()
REFS = R.autopilot_reference(cfg.ENV, cache=f"{OUT}/autopilot_reference.json")
REFS'''

TRAIN = '''jobs = [(algo, seed) for seed in SEEDS for algo in ALGOS]
run_parallel(jobs, OUT, ROOT, DEADLINE)'''

RESULTS = '''RUNS = R.load_runs(OUT)
display(Markdown(R.summary_markdown(RUNS)))
display(Markdown(R.per_seed_markdown(RUNS)))
display(Markdown(R.pair_table(RUNS, *ALGOS)))
display(Markdown(R.updates_table(RUNS)))
display(Markdown(R.chute_table(RUNS)))'''

CURVES = '''F.plot_learning_curves(RUNS, REFS, f"{OUT}/plots/learning_curves.png"); plt.show()
for algo in ALGOS:
    F.plot_algo([r for r in RUNS if r["algo"] == algo], REFS, f"{OUT}/plots/{algo}_per_seed.png"); plt.show()
F.plot_scores(RUNS, f"{OUT}/plots/scores.png"); plt.show()
F.plot_best(RUNS, REFS, f"{OUT}/plots/best_results.png"); plt.show()
F.plot_crash_by_payload(RUNS, f"{OUT}/plots/crash_by_payload.png"); plt.show()'''

TRUST = '''print(R.trust_summary(RUNS))
if any(algo != "sac" for algo in ALGOS):
    F.plot_model_trust(RUNS, f"{OUT}/plots/model_trust.png"); plt.show()
if "macura" in ALGOS:
    F.plot_kappa(RUNS, f"{OUT}/plots/kappa.png"); plt.show()
if "mbpo" in ALGOS or "m2ac" in ALGOS:
    F.plot_untrusted_data(RUNS, f"{OUT}/plots/untrusted_data.png"); plt.show()

checks = [R.model_check(r, cfg.CFG) for r in RUNS if R.ensemble_checkpoint(r)]
if checks:
    F.plot_model_check(checks, f"{OUT}/plots/model_check.png"); plt.show()'''

VIDEOS = '''def trust_of(run):
    return {"ensemble": R.ensemble_checkpoint(run), "kappa": R.kappa_at(run)}

for seed in SEEDS:
    runs = {R.NAMES[r["algo"]]: r for r in RUNS if r["seed"] == seed}
    pilots = {name: R.checkpoint(r) for name, r in runs.items()}
    trust = {name: trust_of(r) for name, r in runs.items() if r["algo"] == "macura"}
    path = V.record_race(cfg.CFG, pilots, f"{OUT}/videos/seed{seed}_best_{VIDEO_SECONDS}s.mp4", seeds=(1010 + seed,),
                         trust=trust, cols=len(pilots), frame_step=3, fps=17, max_steps=VIDEO_SECONDS * 50)
    if path:
        display(Video(path, embed=True, width=480 * len(pilots)))'''

SAVE = '''summary = [R.summary_markdown(RUNS), R.per_seed_markdown(RUNS), R.pair_table(RUNS, *ALGOS),
           R.updates_table(RUNS), R.chute_table(RUNS), R.trust_summary(RUNS)]
open(f"{OUT}/summary.md", "w").write("\\n\\n".join(summary))
zip_file = shutil.make_archive(f"/kaggle/working/{NAME}", "zip", OUT)
print(zip_file, f"{os.path.getsize(zip_file) / 1e6:.0f} MB")
print(f"total time {(time.time() - T0) / 3600:.1f} h")'''


def build(name, account, seeds, algos):
    algo_names = " + ".join(a.upper() for a in algos)
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md(f"# Training: {algo_names}, seeds {seeds}\n"
           f"Race3 drone task, {algo_names} trained at the same time for seeds {seeds}. "
           f"Results are saved in `{name}.zip` for `testing/test.ipynb`."),
        md("## 1. Setup"),
        code(SETUP.format(name=name, seeds=seeds, algos=algos)),
        code(IMPORTS),
        md("## 2. The task"),
        code(TASK_PLOTS),
        md("## 3. Training\nEvery line of every run, and a progress plot every 30 minutes."),
        code(TRAIN),
        md("## 4. Results"),
        code(RESULTS),
        code(CURVES),
        md("## 5. World model and trust"),
        code(TRUST),
        md(f"## 6. Videos: best checkpoint of each seed, 30 s"),
        code(VIDEOS),
        md("## 7. Save"),
        code(SAVE),
    ]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}})
    folder = os.path.join(HERE, name)
    os.makedirs(folder, exist_ok=True)
    nbf.write(notebook, os.path.join(folder, f"{name}.ipynb"))

    slug = "train-" + name.replace("_", "-")
    metadata = {"id": f"{account}/{slug}", "title": slug, "code_file": f"{name}.ipynb", "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
                "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    json.dump(metadata, open(os.path.join(folder, "kernel-metadata.json"), "w"), indent=2)
    print(folder, "->", metadata["id"])


if __name__ == "__main__":
    for notebook in NOTEBOOKS:
        build(*notebook)
