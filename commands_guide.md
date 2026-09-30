# How to run the whole project

Every command runs in the Terminal, in the project folder. Never share or upload a `kaggle.json` file.

```bash
cd ~/Desktop/macura-drone
git pull
```

| step | where | time |
|---|---|---|
| 1. once: keys and Python | your computer | 5 min |
| 2. put code changes on GitHub | your computer | seconds |
| 3. train: 4 notebooks on 2 Kaggle accounts | Kaggle (GPU) | about 4-5 h, all at the same time |
| 4. download the results | your computer | 1 min |
| 5. test all models together | your computer (`testing/test.ipynb`) | about 30-60 min |
| 6. watch the drones race | your computer (MuJoCo viewer) | as long as you like |
| 7. put the plots and videos on GitHub | your computer | 1 min |

The 4 training notebooks (folder `training/`):

| notebook | trains | seeds | Kaggle account | key |
|---|---|---|---|---|
| `macura_mbpo_seeds45` | MACURA + MBPO | 4, 5 | nouradon | `~/.kaggle/kaggle.json` |
| `m2ac_sac_seeds45` | M2AC + SAC | 4, 5 | nouradon | `~/.kaggle/kaggle.json` |
| `macura_mbpo_seeds67` | MACURA + MBPO | 6, 7 | silvergjeka01 | `~/.kaggle/silvergjeka01/kaggle.json` |
| `m2ac_sac_seeds67` | M2AC + SAC | 6, 7 | silvergjeka01 | `~/.kaggle/silvergjeka01/kaggle.json` |

---

## 1. Once: keys and Python

### Kaggle keys

Check that both keys work. Each line prints the account's username (only the name, never the key):

```bash
python3 -c "import json; print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])"
python3 -c "import json; print(json.load(open('$HOME/.kaggle/silvergjeka01/kaggle.json'))['username'])"
```

A key is missing or old? Log in to that account on kaggle.com, **Settings -> API -> Create New Token**, then:

```bash
mkdir -p ~/.kaggle/silvergjeka01
mv ~/Downloads/kaggle.json ~/.kaggle/silvergjeka01/kaggle.json
chmod 600 ~/.kaggle/silvergjeka01/kaggle.json
```

(For the main account the key goes to `~/.kaggle/kaggle.json`.) Each account needs a verified phone number on
kaggle.com, otherwise it gets no GPU and no Internet.

### Python on your computer

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt torch notebook
```

Next time only `source .venv/bin/activate`.

---

## 2. Put code changes on GitHub

The notebooks download the code from GitHub (`silvergjeka22/macura-drone`, branch `main`) when they start, so a
change in `src/` must be on GitHub first:

```bash
git add src
git commit -m "what you changed"
git push
```

(Nothing changed in `src/`? Skip this step. `./kaggle.sh push` warns if `src/` is not on GitHub yet.)

---

## 3. Train on Kaggle

### Account 1: nouradon

```bash
unset KAGGLE_CONFIG_DIR
python3 -c "import json; print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])"
```

It must print `nouradon`. Then start the two notebooks:

```bash
kaggle kernels push -p training/macura_mbpo_seeds45
kaggle kernels push -p training/m2ac_sac_seeds45
```

### Account 2: silvergjeka01

```bash
export KAGGLE_CONFIG_DIR=~/.kaggle/silvergjeka01
python3 -c "import json; print(json.load(open('$KAGGLE_CONFIG_DIR/kaggle.json'))['username'])"
```

It must print `silvergjeka01`. From now on every `kaggle` command in this Terminal window uses silvergjeka01
(a new window starts again on nouradon).

```bash
kaggle kernels push -p training/macura_mbpo_seeds67
kaggle kernels push -p training/m2ac_sac_seeds67
```

### Check

```bash
kaggle kernels status silvergjeka01/train-macura-mbpo-seeds67
kaggle kernels status silvergjeka01/train-m2ac-sac-seeds67
unset KAGGLE_CONFIG_DIR
kaggle kernels status nouradon/train-macura-mbpo-seeds45
kaggle kernels status nouradon/train-m2ac-sac-seeds45
```

- RUNNING: training (you can watch it on kaggle.com -> the notebook -> Logs; a progress plot every 30 min).
- COMPLETE: done.
- ERROR: open the notebook on kaggle.com -> Logs, and see "If something goes wrong" below.

Your computer can be off while they run.

---

## 4. Download the results (when all 4 are COMPLETE)

```bash
unset KAGGLE_CONFIG_DIR
kaggle kernels output nouradon/train-macura-mbpo-seeds45 -p downloads/macura_mbpo_seeds45
kaggle kernels output nouradon/train-m2ac-sac-seeds45 -p downloads/m2ac_sac_seeds45

export KAGGLE_CONFIG_DIR=~/.kaggle/silvergjeka01
kaggle kernels output silvergjeka01/train-macura-mbpo-seeds67 -p downloads/macura_mbpo_seeds67
kaggle kernels output silvergjeka01/train-m2ac-sac-seeds67 -p downloads/m2ac_sac_seeds67
unset KAGGLE_CONFIG_DIR
```

Each folder gets the run logs, the best checkpoints, the plots, the videos and `<name>.zip`.

---

## 5. Test all models together

```bash
source .venv/bin/activate
jupyter notebook testing/test.ipynb
```

Click **Run -> Run All Cells**. The notebook loads the 4 downloads, shows all training results, tests every model
on 50 new scenarios, records the test videos, makes the final comparison and ends with a recap. Everything is saved
in `results/` (`summary.md`, `plots/`, `videos/`).

---

## 6. Watch the drones race

```bash
mjpython simulate.py
mjpython simulate.py --scenarios 3000 3001 3002
mjpython simulate.py --algos macura mbpo --seconds 30 --follow MACURA
```

Keys in the viewer: Space = pause, N = next scenario, R = restart.

---

## 7. Put the plots and videos on GitHub

Only the plots, videos and summaries go to GitHub; the checkpoints, logs and zips stay on your computer
(`.gitignore` does this).

```bash
git add downloads results
git commit -m "Results: plots and videos"
git push
```

---

## Shortcut: kaggle.sh

`kaggle.sh` does steps 3-4 for both accounts and picks each notebook's key by itself:

```bash
./kaggle.sh push      # start all 4 (a running one is not started again)
./kaggle.sh status    # the status of all 4
./kaggle.sh get       # download the finished ones into downloads/<name>
```

One notebook only: `./kaggle.sh push macura_mbpo_seeds45`.

---

## Change the seeds or the account of a notebook

1. In `training/<name>/<name>.ipynb`, first code cell: change `SEEDS` (and `ALGOS` if needed).
2. The account is the first part of `"id"` in `training/<name>/kernel-metadata.json`
   (e.g. `"silvergjeka01/train-macura-mbpo-seeds67"`). Put the account's key in `~/.kaggle/<account>/kaggle.json`.
3. A new name? Rename the folder, the notebook and the `id`, and update the list `NOTEBOOKS=` at the top of
   `kaggle.sh`.

---

## If something goes wrong

| problem | fix |
|---|---|
| ERROR after a few seconds | open its Logs on kaggle.com; "No GPU" or "Internet": verify the phone number on that account |
| "Permission 'kernelSessions.enableInternet' was denied" | verify the phone number on that account |
| 401 / unauthorized | the key is old or in the wrong folder: step 1 |
| QUEUED for a long time / "no GPU quota" | the account used its ~30 GPU hours this week |
| "stopped by the time limit" | the 11 h safety stop; the results up to there are saved |
| KeyError when checking a key | type `['username']` exactly: it is the name of the field in the key file, not your account name |
| the testing notebook says **missing** | that notebook is not downloaded yet: step 4 |
