"""Builds testing/test.ipynb. Run: python testing/build_test_notebook.py"""

import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))

SETUP = '''import os, sys

ROOT = os.path.abspath(".." if os.path.basename(os.getcwd()) == "testing" else ".")
os.chdir(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault("MUJOCO_GL", "disable")

DOWNLOADS = "downloads"
OUT = "results"
EPISODES = 50           # new test scenarios per model
VIDEOS = True'''

IMPORTS = '''import json
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, Video, display
from src.config import config as cfg
from src.analysis import results as R, report, testing as T
from src.viz import figures as F, video as V, test_figures as TF'''

LOAD = '''FOLDERS = R.find_downloads(DOWNLOADS)
RUNS = R.load_runs(*FOLDERS)
display(Markdown(R.where_table(RUNS)))'''

TRAIN_TABLES = '''display(Markdown(R.summary_markdown(RUNS)))
display(Markdown(R.per_seed_markdown(RUNS)))
for other in ["mbpo", "m2ac", "sac"]:
    display(Markdown(R.pair_table(RUNS, "macura", other)))
display(Markdown(R.updates_table(RUNS)))
display(Markdown(R.chute_table(RUNS)))'''

TRAIN_PLOTS = '''REFS = R.autopilot_reference(cfg.ENV, cache=f"{OUT}/autopilot_reference.json")
F.plot_learning_curves(RUNS, REFS, f"{OUT}/plots/learning_curves_smooth.png", window=7); plt.show()
for algo, runs in R.by_algo(RUNS).items():
    F.plot_algo(runs, REFS, f"{OUT}/plots/{algo}_per_seed.png", window=7); plt.show()
F.plot_scores(RUNS, f"{OUT}/plots/scores.png"); plt.show()
F.plot_summary_table(RUNS, f"{OUT}/plots/summary_table.png"); plt.show()'''

TRAIN_TRUST = '''print(R.trust_summary(RUNS))
made = report.build(RUNS, cfg.CFG, OUT, check=True)
for name in ["model_trust", "kappa", "untrusted_data", "model_check", "crash_by_payload"]:
    if name in made:
        display(Image(made[name]))'''

TEST = '''TESTS = T.test_all(RUNS, cfg.CFG, episodes=EPISODES)
tables = T.test_tables(TESTS)
display(Markdown(tables["summary"]))
display(Markdown(tables["per_seed"]))
for table in tables["paired"]:
    display(Markdown(table))
R.autopilot_reference(cfg.ENV, seed_base=T.TEST_SEED_BASE, episodes=EPISODES, cache=f"{OUT}/autopilot_new_test.json")'''

TEST_PLOTS = '''TF.plot_test_scores(TESTS, f"{OUT}/plots/test_scores.png"); plt.show()
TF.plot_test_seeds(TESTS, f"{OUT}/plots/test_seeds.png"); plt.show()
TF.plot_test_conditions(TESTS, f"{OUT}/plots/test_conditions.png"); plt.show()
TF.plot_test_flights(TESTS, cfg.CFG, trace=0, save_path=f"{OUT}/plots/test_flight_0.png"); plt.show()
TF.plot_test_flights(TESTS, cfg.CFG, trace=1, save_path=f"{OUT}/plots/test_flight_1.png"); plt.show()
TF.plot_test_noise(TESTS, save_path=f"{OUT}/plots/test_wind.png"); plt.show()
TF.plot_test_trust(TESTS, f"{OUT}/plots/test_trust.png"); plt.show()'''

VIDEOS = '''def trust_of(run):
    return {"ensemble": R.ensemble_checkpoint(run), "kappa": R.kappa_at(run)}

def show(path):
    if path:
        display(Video(path, embed=True, width=960))

if VIDEOS:
    best = {R.NAMES[a]: R.best_run(runs) for a, runs in R.by_algo(RUNS).items()}
    trust = {name: trust_of(r) for name, r in best.items() if r["algo"] == "macura"}
    pilots = {name: R.checkpoint(r) for name, r in best.items()}
    scenarios = (T.TEST_SEED_BASE, T.TEST_SEED_BASE + 1, T.TEST_SEED_BASE + 2)
    show(V.record_race(cfg.CFG, pilots, f"{OUT}/videos/test_all.mp4", seeds=scenarios, trust=trust, frame_step=3, fps=17))

    for algo, runs in R.by_algo(RUNS).items():
        pilots = {f"{R.NAMES[algo]} seed {r['seed']}": R.checkpoint(r) for r in runs}
        seed_trust = {f"{R.NAMES[algo]} seed {r['seed']}": trust_of(r) for r in runs if algo == "macura"}
        show(V.record_race(cfg.CFG, pilots, f"{OUT}/videos/test_{algo}_seeds.mp4", seeds=(T.TEST_SEED_BASE + 3,),
                           trust=seed_trust, frame_step=3, fps=17))

    duo = {name: R.checkpoint(r) for name, r in best.items() if name in ("MACURA", "MBPO")}
    show(V.record_race(cfg.CFG, duo, f"{OUT}/videos/test_30s.mp4", seeds=(T.TEST_SEED_BASE + 4,), trust=trust,
                       frame_step=3, fps=17, max_steps=1500))'''

FINAL = '''train = R.summary(RUNS)
test = {}
for result in TESTS:
    test.setdefault(result["algo"], []).append(T.seed_scores(result))

rows = ["| algorithm | avg return while learning | drones broken | test return | test lap progress | test crash rate |",
        "|---|---|---|---|---|---|"]
for algo in test:
    t = test[algo]
    rows.append(f"| {R.NAMES[algo]} | {train[algo]['avg_return'][0]:.0f} | {train[algo]['broken'][0]:.0f} | "
                f"{R.iqm([s['return'] for s in t]):.0f} | {R.iqm([s['lap'] for s in t]):.0%} | "
                f"{R.iqm([s['crash'] for s in t]):.0%} |")
final_table = "\\n".join(rows)
display(Markdown(final_table))

summary = [final_table, R.summary_markdown(RUNS), R.per_seed_markdown(RUNS), tables["summary"], tables["per_seed"],
           *tables["paired"], R.where_table(RUNS)]
open(f"{OUT}/summary.md", "w").write("\\n\\n".join(summary))
json.dump([{"algo": r["algo"], "seed": r["seed"], "episodes": r["episodes"]} for r in TESTS],
          open(f"{OUT}/test_results.json", "w"), default=float)'''

SIMULATOR = '''from src.envs import drone_env
from src.algorithms.sac import load_agent

env, obs_dim, act_dim = drone_env.make_env(cfg.ENV)
for r in RUNS:
    load_agent(R.checkpoint(r), obs_dim, act_dim, cfg.SAC)
print(f"all {len(RUNS)} models load. In a terminal:")
print("  mjpython simulate.py")
print("  mjpython simulate.py --scenarios 3000 3001 3002")
print("  mjpython simulate.py --algos macura mbpo --seconds 30 --follow MACURA")'''


def build():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md("# Testing: all four algorithms together\n"
           "Run after `./kaggle.sh get` has downloaded the 4 training notebooks into `downloads/`. "
           "Everything is saved in `results/`."),
        md("## 1. Load"),
        code(SETUP),
        code(IMPORTS),
        code(LOAD),
        md("## 2. Training results"),
        code(TRAIN_TABLES),
        code(TRAIN_PLOTS),
        code(TRAIN_TRUST),
        md("## 3. Test on 50 new scenarios"),
        code(TEST),
        code(TEST_PLOTS),
        md("## 4. Test videos"),
        code(VIDEOS),
        md("## 5. Final comparison"),
        code(FINAL),
        md("## 6. Simulator"),
        code(SIMULATOR),
    ]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}})
    nbf.write(notebook, os.path.join(HERE, "test.ipynb"))
    print("written testing/test.ipynb")


if __name__ == "__main__":
    build()
