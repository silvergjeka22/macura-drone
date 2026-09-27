# How to run the study

All commands run in the Terminal, from the repo folder (`cd ~/Desktop/macura-drone-cage`). Never share or upload
`kaggle.json` or your GitHub token.

## What runs where

| notebook (folder in `training/`) | algorithms | seeds | Kaggle account | key file |
|---|---|---|---|---|
| `macura_mbpo_seeds01` | MACURA + MBPO | 0, 1 | nouradon | `~/.kaggle/kaggle.json` |
| `m2ac_sac_seeds01` | M2AC + SAC | 0, 1 | nouradon | `~/.kaggle/kaggle.json` |
| `macura_mbpo_seeds23` | MACURA + MBPO | 2, 3 | silvergjeka01 | `~/.kaggle/silvergjeka01/kaggle.json` |
| `m2ac_sac_seeds23` | M2AC + SAC | 2, 3 | silvergjeka01 | `~/.kaggle/silvergjeka01/kaggle.json` |

Each notebook trains its 4 runs at the same time (about 4-5 h). The two notebooks of an account run at the same time
if Kaggle allows it; otherwise the second one waits in the queue and starts when the first one is done.
Kaggle limits: 12 h per session (the notebooks stop training at 11 h and save), about 30 GPU hours per week per account.

## 1. Start the training

The notebooks download the code from GitHub (branch `race3`): if you change anything in `src/`, commit and push it
first. Then:

    ./kaggle.sh push                  # all 4 (or name some: ./kaggle.sh push m2ac_sac_seeds01)
    ./kaggle.sh status

`push` never starts a notebook that is still running, and first downloads a finished one's previous output into
`downloads/previous/`, so nothing is lost.

**The first time only**, each of the 4 notebooks is new on Kaggle and needs 3 settings on the website (the first run
stops at the "git clone" step without them):
1. Open the notebook on kaggle.com (logged in as its account), click **Edit**.
2. **Add-ons -> Secrets**: switch **GITHUB_TOKEN** on for this notebook.
3. Right-hand panel: **Accelerator = GPU**, **Internet = On**; then close the editor.

Then start it again: `./kaggle.sh push <name>`.

## 2. Download the results

    ./kaggle.sh get

Every finished notebook goes into `downloads/<name>/` (logs, checkpoints, figures, videos, `<name>.zip`, the notebook
itself). If the folder already exists, a new one with the date is made: nothing is overwritten.
You can also download a notebook's `<name>.zip` from its Kaggle page (Output) and put it into `downloads/<name>/`.

## 3. Test and compare (on your computer)

Once (Python 3 from python.org or Homebrew):

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt torch notebook

Then:

    jupyter notebook testing/test.ipynb        # Run All

It finds everything in `downloads/` (the newest copy of each notebook), shows which run comes from which download,
compares all training results, tests every model on 50 new scenarios, makes the test figures and videos, and saves it
all in `results/` (`summary.md`, `plots/`, `videos/`, `test_results.json`).

## 4. Watch them race

    mjpython simulate.py                              # best seed of each algorithm
    mjpython simulate.py --scenarios 3000 3001 3002   # the new test scenarios
    mjpython simulate.py --algos macura mbpo --seconds 30 --follow MACURA
    mjpython simulate.py --seed 2                     # seed 2 of every algorithm

Keys in the viewer: Space pause, N next scenario, R restart. `mjpython` comes with the `mujoco` package (macOS needs it
instead of `python` for the window).

## Changing the Kaggle account of a notebook

The account is part of `training/build_notebooks.py` (`NOTEBOOKS`) and `kaggle.sh` (`keys`). Change both, run
`python3 training/build_notebooks.py`, and put the new account's key in `~/.kaggle/<name>/kaggle.json`
(Kaggle -> Settings -> API -> Create New Token downloads it). A new account also needs a verified phone number (for GPU
and Internet) and the GITHUB_TOKEN secret.

## If something goes wrong

- `./kaggle.sh status` says ERROR: open the notebook's page on Kaggle -> Logs to see why. Most common: the
  GITHUB_TOKEN secret is not switched on for that notebook (step 1).
- "Permission 'kernelSessions.enableInternet' was denied": the account's phone number is not verified.
- 401 / unauthorized: the key in `~/.kaggle/...` is old; create a new token.
- A run "stopped by the time limit": it hit the 11 h safety stop; its results up to there are saved.
- The testing notebook says "missing": that notebook is not downloaded yet (`./kaggle.sh get`).
