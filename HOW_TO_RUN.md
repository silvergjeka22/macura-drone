# How to run the study

Everything runs from the main project folder, on branch `main`. Type the commands in the Terminal:

    cd ~/Desktop/macura-drone
    git pull

Never share or upload a `kaggle.json` file or your GitHub token.

## The 4 training notebooks

| notebook (`training/`) | trains | seeds | Kaggle account | key |
|---|---|---|---|---|
| `macura_mbpo_seeds45` | MACURA + MBPO | 4, 5 | nouradon | `~/.kaggle/kaggle.json` |
| `m2ac_sac_seeds45` | M2AC + SAC | 4, 5 | nouradon | `~/.kaggle/kaggle.json` |
| `macura_mbpo_seeds67` | MACURA + MBPO | 6, 7 | silvergjeka01 | `~/.kaggle/silvergjeka01/kaggle.json` |
| `m2ac_sac_seeds67` | M2AC + SAC | 6, 7 | silvergjeka01 | `~/.kaggle/silvergjeka01/kaggle.json` |

- All 4 run at the same time, 2 per account, about 4-5 hours each.
- They download the code from GitHub (`silvergjeka22/macura-drone`, branch `main`).
- Seeds 0-3 were used in the race3 test (`docs/EXPERIMENTS.md`), so this study uses new seeds.

## 1. Check the keys (once)

    kaggle datasets list --mine
    KAGGLE_CONFIG_DIR=~/.kaggle/silvergjeka01 kaggle datasets list --mine

"No datasets found" means the key works. If you get 401 / unauthorized, get a new key: kaggle.com -> Settings -> API -> Create New Token. Move it to the key path in the table and run `chmod 600` on it.

## 2. Start the notebooks

    ./kaggle.sh push

## 3. First time only: switch on the GitHub secret

The 4 notebooks are new on Kaggle, so their first run stops after a few seconds. For each notebook, log in to kaggle.com with its account:

1. **Code -> Your Work -> train-...** (e.g. `train-macura-mbpo-seeds45`) -> **Edit**.
2. **Add-ons -> Secrets**: switch **GITHUB_TOKEN** on.
3. **Session options**: **Accelerator = GPU**, **Internet = On**.
4. Close the editor (it saves by itself).

Then start them again:

    ./kaggle.sh push

## 4. Wait

    ./kaggle.sh status

- RUNNING means it is training. COMPLETE means it is done.
- ERROR means open the notebook on kaggle.com -> Logs. Most often, step 3 was missed.
- To watch live: open the notebook on kaggle.com -> the running version -> Logs. Every evaluation is printed there, with a progress plot every 30 minutes.
- Your computer can be off meanwhile.

## 5. Download the results

When all 4 say COMPLETE:

    ./kaggle.sh get

Each notebook goes to `downloads/<name>/`. Nothing is overwritten: if the folder already exists, the new download gets a date suffix.

## 6. Python on your Mac (once)

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt torch notebook

Next time you only need `source .venv/bin/activate`.

## 7. The testing notebook

    jupyter notebook testing/test.ipynb

Click **Run -> Run All Cells**. The notebook:

- merges the 4 downloads;
- shows all the training results;
- tests every model on 50 new scenarios, with plots and videos;
- ends with the final comparison and the best results plot (`best_results.png`: the best seed of each algorithm, its best checkpoint and its test return).

Everything is saved in `results/`.

## 8. Watch them race

    mjpython simulate.py
    mjpython simulate.py --algos macura mbpo --seconds 30 --follow MACURA

Keys: Space = pause, N = next scenario, R = restart.

## If something goes wrong

| problem | fix |
|---|---|
| ERROR a few seconds after the start | step 3 |
| "Permission 'kernelSessions.enableInternet' was denied" | verify a phone number on that Kaggle account |
| 401 / unauthorized | step 1 |
| QUEUED for a long time / "no GPU quota" | the account used its ~30 GPU hours this week |
| "stopped by the time limit" | the 11 h safety stop; results up to there are saved |
| the testing notebook says **missing** | that notebook is not downloaded yet: step 5 |
