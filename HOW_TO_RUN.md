# How to run the study, step by step

Type every command in the Terminal, in the project folder:

    cd ~/Desktop/macura-drone-cage

Never share or upload a `kaggle.json` file or your GitHub token.

## What runs where

| training notebook (`training/`) | trains | seeds | Kaggle account | Kaggle key |
|---|---|---|---|---|
| `macura_mbpo_seeds01` | MACURA + MBPO | 0, 1 | nouradon | `~/.kaggle/kaggle.json` |
| `m2ac_sac_seeds01` | M2AC + SAC | 0, 1 | nouradon | `~/.kaggle/kaggle.json` |
| `macura_mbpo_seeds23` | MACURA + MBPO | 2, 3 | silvergjeka01 | `~/.kaggle/silvergjeka01/kaggle.json` |
| `m2ac_sac_seeds23` | M2AC + SAC | 2, 3 | silvergjeka01 | `~/.kaggle/silvergjeka01/kaggle.json` |

Each notebook trains its 4 runs at the same time and takes about 4-5 hours. The notebooks download the code from
GitHub (`silvergjeka22/macura-drone`, branch `main`), so any change to `src/` must be pushed to GitHub first.

## Step 1 - once: the Kaggle keys

Both keys are already on this Mac. To check them:

    kaggle datasets list --mine
    KAGGLE_CONFIG_DIR=~/.kaggle/silvergjeka01 kaggle datasets list --mine

"No datasets found" means the key works. If one says 401 / unauthorized: log in to that account on kaggle.com,
Settings -> API -> Create New Token, and move the downloaded `kaggle.json` to the key path in the table above
(`chmod 600` it).

## Step 2 - start the 4 training notebooks

    ./kaggle.sh push

Wait one minute, then:

    ./kaggle.sh status

## Step 3 - only the first time: switch on the GitHub secret

The 4 notebooks are new on Kaggle, so the first run of each stops after a few seconds (it cannot download the code).
For each notebook, on kaggle.com, logged in as its account (see the table):

1. Open **Code -> Your Work -> train-...** (for example `train-macura-mbpo-seeds01`) and click **Edit**.
2. Menu **Add-ons -> Secrets**: switch **GITHUB_TOKEN** on for this notebook.
3. Right-hand panel **Session options**: **Accelerator = GPU**, **Internet = On**.
4. Close the editor (top right, the notebook is saved).

Then start them again:

    ./kaggle.sh push

## Step 4 - wait and check

    ./kaggle.sh status

RUNNING = training, COMPLETE = done, ERROR = open the notebook on kaggle.com -> Logs (most often: step 3 was missed).
You can watch a notebook live on kaggle.com: open it -> the running version -> Logs. Every evaluation line and a
progress plot every 30 minutes appear there. Your computer can be off while they run.

## Step 5 - download the results

When all 4 say COMPLETE:

    ./kaggle.sh get

Each notebook goes into `downloads/<name>/` (logs, best models, figures, videos, `<name>.zip`). Nothing is ever
overwritten: if a folder already exists, a new one with the date is made.

## Step 6 - once: Python on your Mac

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt torch notebook

(Next time only `source .venv/bin/activate`.)

## Step 7 - the testing notebook

    jupyter notebook testing/test.ipynb

Click **Run -> Run All Cells**. It loads the 4 downloads, compares all training results, tests every model on 50 new
scenarios, makes the test figures and videos, and saves everything in `results/` (`summary.md`, `plots/`, `videos/`).

## Step 8 - watch them race

    mjpython simulate.py
    mjpython simulate.py --scenarios 3000 3001 3002
    mjpython simulate.py --algos macura mbpo --seconds 30 --follow MACURA

In the window: Space = pause, N = next scenario, R = restart.

## If something goes wrong

| problem | fix |
|---|---|
| status ERROR a few seconds after the start | step 3 (GITHUB_TOKEN secret, GPU, Internet) |
| "Permission 'kernelSessions.enableInternet' was denied" | verify a phone number on that Kaggle account |
| 401 / unauthorized | the key is old: step 1 |
| "no GPU quota" / QUEUED for a long time | the account used its ~30 GPU hours this week, or its other notebook is still running |
| a run says "stopped by the time limit" | it reached the 11 h safety stop; its results up to there are saved |
| the testing notebook says **missing** | that notebook is not downloaded yet: step 5 |
