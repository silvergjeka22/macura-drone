"""Train several runs at the same time, one process each, and show their progress."""

import os
import re
import subprocess
import sys
import time

import matplotlib.pyplot as plt

CHILD = """
import sys
sys.path.insert(0, {root!r})
from src.config import config as cfg
from src.training.train import train_algo
train_algo({algo!r}, cfg.CFG, cfg.OUTPUT_ROOT, seeds=[{seed}])
"""
EVAL = re.compile(r"step\s+(\d+)\s+return\s+(-?[\d.]+)\s+lap\s+([\d.]+)%\s+crash\s+([\d.]+)")


def read_curve(log):
    """(steps, returns, lap %, crash rate) from a training printout."""
    rows = [EVAL.search(line) for line in open(log)]
    rows = [[float(x) for x in m.groups()] for m in rows if m]
    return list(zip(*rows)) if rows else ([], [], [], [])


def plot_progress(out, t0):
    logs = sorted(f for f in os.listdir(f"{out}/joblogs") if f.endswith(".txt"))
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
    for f in logs:
        steps, ret, lap, crash = read_curve(f"{out}/joblogs/{f}")
        ax[0].plot(steps, ret, "o-", ms=3, label=f[:-4])
        ax[1].plot(steps, lap, "o-", ms=3, label=f[:-4])
        ax[2].plot(steps, crash, "o-", ms=3, label=f[:-4])
    for a, title in zip(ax, ["return", "lap progress (%)", "crash rate"]):
        a.set_title(title)
        a.set_xlabel("step")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle(f"Training so far ({(time.time() - t0) / 3600:.1f} h)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()


def run_parallel(jobs, out, root, deadline, n_jobs=4, plot_every=1800):
    """jobs = [(algo, seed), ...]. Every printed line of every run is shown, tagged [algo seedN];
    a progress plot every `plot_every` seconds."""
    os.makedirs(f"{out}/joblogs", exist_ok=True)
    t0 = time.time()
    n_gpu = _gpu_count()
    pending, running, read_until = list(jobs), {}, {}
    last_plot = time.time()

    while pending or running:
        while pending and len(running) < n_jobs:
            algo, seed = pending.pop(0)
            if os.path.exists(f"{out}/logs/{algo}_seed{seed}.json"):
                print(f"{algo} seed {seed}: already done")
                continue
            if time.time() > deadline:
                print(f"{algo} seed {seed}: no time left, skipped")
                continue
            env = dict(os.environ, MACURA_SEEDS=str(seed), MACURA_ALGOS=algo, PYTHONUNBUFFERED="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            if n_gpu > 1:
                env["CUDA_VISIBLE_DEVICES"] = str(len(running) % n_gpu)
            log = f"{out}/joblogs/{algo}_seed{seed}.txt"
            code = CHILD.format(root=root, algo=algo, seed=seed)
            proc = subprocess.Popen([sys.executable, "-u", "-c", code], env=env,
                                    stdout=open(log, "w"), stderr=subprocess.STDOUT)
            running[(algo, seed)] = (proc, log, time.time())
            print(f"started {algo} seed {seed}")

        for (algo, seed), (proc, log, start) in list(running.items()):
            _print_new_lines(algo, seed, log, read_until)
            if proc.poll() is None:
                continue
            _print_new_lines(algo, seed, log, read_until)
            del running[(algo, seed)]
            minutes = (time.time() - start) / 60
            if proc.returncode == 0:
                print(f">>> {algo} seed {seed} done in {minutes:.0f} min")
            else:
                print(f">>> {algo} seed {seed} FAILED:\n" + "\n".join(open(log).read().splitlines()[-15:]))

        if running and time.time() - last_plot > plot_every:
            plot_progress(out, t0)
            last_plot = time.time()
        time.sleep(20)

    plot_progress(out, t0)
    print(f"all done in {(time.time() - t0) / 3600:.1f} h")


def _print_new_lines(algo, seed, log, read_until):
    with open(log) as f:
        f.seek(read_until.get(log, 0))
        text = f.read()
        read_until[log] = f.tell()
    for line in text.splitlines():
        if line.startswith("["):
            print(line, flush=True)
        elif line.startswith("    model:") or line.startswith("    policy:"):
            print(f"[{algo} seed{seed}] {line.strip()}", flush=True)


def _gpu_count():
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0
