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

## Kaggle accounts: how they work and how to change them

### How the script knows which account to use

A notebook's account is the first part of `"id"` in `training/<name>/kernel-metadata.json`, for example:

    "id": "silvergjeka01/train-macura-mbpo-seeds67"      -> account silvergjeka01

`./kaggle.sh` then uses that account's key:

- `~/.kaggle/<account>/kaggle.json` if that file exists (e.g. `~/.kaggle/silvergjeka01/kaggle.json`);
- otherwise `~/.kaggle/kaggle.json` (the main key, now nouradon).

To see which account a key belongs to (this prints only the username, never the key):

    python3 -c "import json; print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])"
    python3 -c "import json; print(json.load(open('$HOME/.kaggle/silvergjeka01/kaggle.json'))['username'])"

### Add a new account (or renew a key)

1. Open kaggle.com in a **private / incognito window** and log in with the account. Your normal window stays on the
   other account.
2. The account must have a **verified phone number** (Settings -> Phone verification), otherwise no GPU and no Internet.
3. Settings -> **API -> Create New Token**. A `kaggle.json` file is downloaded.
4. Put it in a folder named after the account (here `myaccount` stands for the Kaggle username):

       mkdir -p ~/.kaggle/myaccount
       mv ~/Downloads/kaggle.json ~/.kaggle/myaccount/kaggle.json
       chmod 600 ~/.kaggle/myaccount/kaggle.json

5. Check it works:

       KAGGLE_CONFIG_DIR=~/.kaggle/myaccount kaggle datasets list --mine

6. In the same browser window, the account also needs the **GITHUB_TOKEN** secret (step 3 above), once per notebook.

### Move notebooks to another account

1. Open `training/build_notebooks.py` and change the account in the `NOTEBOOKS` list:

       ("macura_mbpo_seeds67", "myaccount", [6, 7], ["macura", "mbpo"]),

   The seeds and algorithms can be changed in the same line. If you change the seeds, change the name too
   (e.g. `macura_mbpo_seeds89` with `[8, 9]`), and update the list `NOTEBOOKS=` in `kaggle.sh` to match.
2. Rebuild the notebooks:

       python3 training/build_notebooks.py

3. Push the change to GitHub. The notebooks download the code from GitHub, and `kaggle.sh` reads the new account
   from the rebuilt files:

       git add -A && git commit -m "notebooks on another account" && git push

4. Start them: `./kaggle.sh push`. On the new account they are new notebooks, so do step 3 (GITHUB_TOKEN secret, GPU,
   Internet) once, then `./kaggle.sh push` again.

### Which account am I looking at on kaggle.com?

Top right: click your picture. The username is shown there. Use **one browser window per account**: a normal window
for nouradon and a private window for silvergjeka01. Then you never change a notebook on the wrong account.

### GPU hours

Each account has about 30 GPU hours per week. It resets every week; the remaining hours are shown on kaggle.com ->
your picture -> Settings (Quotas), or at the top of the notebook editor. The 2 notebooks of one account together
use about 8-10 hours.

## If something goes wrong

| problem | fix |
|---|---|
| ERROR a few seconds after the start | step 3 |
| "Permission 'kernelSessions.enableInternet' was denied" | verify a phone number on that Kaggle account |
| 401 / unauthorized | step 1 |
| QUEUED for a long time / "no GPU quota" | the account used its ~30 GPU hours this week |
| "stopped by the time limit" | the 11 h safety stop; results up to there are saved |
| the testing notebook says **missing** | that notebook is not downloaded yet: step 5 |
