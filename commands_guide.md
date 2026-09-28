# Training & Execution Commands Guide

Run all commands from your terminal inside the main project directory (`~/Desktop/macura-drone`).

---

## 0. Project Setup & Code Packing

```bash
cd ~/Desktop/macura-drone
git pull
python3 training/build_notebooks.py
```

---

## 1. Run Account 1 Notebooks (`nouradon`)

### Verify active account (Must print `nouradon`):
```bash
unset KAGGLE_CONFIG_DIR
python3 -c "import json; print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])"
```

### Push and check Notebook 1 (`macura_mbpo_seeds45`):
```bash
kaggle kernels push -p training/macura_mbpo_seeds45
kaggle kernels status nouradon/train-macura-mbpo-seeds45
```

### Push and check Notebook 2 (`m2ac_sac_seeds45`):
```bash
kaggle kernels push -p training/m2ac_sac_seeds45
kaggle kernels status nouradon/train-m2ac-sac-seeds45
```

---

## 2. Run Account 2 Notebooks (`silvergjeka01`)

### Switch account & verify (Must print `silvergjeka01`):
```bash
export KAGGLE_CONFIG_DIR=~/.kaggle/silvergjeka01
python3 -c "import json; print(json.load(open('$KAGGLE_CONFIG_DIR/kaggle.json'))['username'])"
```

### Push and check Notebook 3 (`macura_mbpo_seeds67`):
```bash
kaggle kernels push -p training/macura_mbpo_seeds67
kaggle kernels status silvergjeka01/train-macura-mbpo-seeds67
```

### Push and check Notebook 4 (`m2ac_sac_seeds67`):
```bash
kaggle kernels push -p training/m2ac_sac_seeds67
kaggle kernels status silvergjeka01/train-m2ac-sac-seeds67
```

---

## 3. Reset Terminal Session Back to Default

```bash
unset KAGGLE_CONFIG_DIR
```

---

## 4. Download Results (When Status is COMPLETE)

### Download from `nouradon`:
```bash
unset KAGGLE_CONFIG_DIR
kaggle kernels output nouradon/train-macura-mbpo-seeds45 -p downloads/macura_mbpo_seeds45
kaggle kernels output nouradon/train-m2ac-sac-seeds45 -p downloads/m2ac_sac_seeds45
```

### Download from `silvergjeka01`:
```bash
export KAGGLE_CONFIG_DIR=~/.kaggle/silvergjeka01
kaggle kernels output silvergjeka01/train-macura-mbpo-seeds67 -p downloads/macura_mbpo_seeds67
kaggle kernels output silvergjeka01/train-m2ac-sac-seeds67 -p downloads/m2ac_sac_seeds67
```

### Reset terminal environment:
```bash
unset KAGGLE_CONFIG_DIR
```

---

## 5. Local Setup & Testing

### Create virtual environment & install requirements (Once):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt torch notebook
```

### Run testing notebook:
```bash
jupyter notebook testing/test.ipynb
```

### Run simulation:
```bash
mjpython simulate.py
```

---

## Shortcut (Using Helper Script)

Alternatively, run all steps with the automated helper script:

```bash
# Push notebooks across both accounts
./kaggle.sh push

# Check status of all runs
./kaggle.sh status

# Download all complete runs
./kaggle.sh get
```