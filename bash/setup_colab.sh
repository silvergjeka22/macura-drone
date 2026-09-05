#!/usr/bin/env bash
# ── setup_colab.sh ───────────────────────────────────────────────────────────
# One-shot Colab bootstrap for macura-backflip:
#   1. clone THIS (private) repo into the Colab workspace using a GitHub token
#   2. pip install -r requirements.txt
#   3. repair numpy consistency if a reinstall left it inconsistent
#   4. create the Google Drive results folders and print all paths
#
# The Pogo body ships in the repo (envs/assets/pogo.xml), so there is NOTHING
# external to fetch — no mujoco_menagerie, no PyBullet source build.
#
# SECURITY: the GitHub token is read from the environment variable GITHUB_TOKEN,
# set by the notebook at runtime (getpass / Colab Secrets). It is NEVER written
# to disk or baked into this script, and is scrubbed from the git remote.
#
# Usage (from the notebook):
#   !GITHUB_TOKEN=$TOKEN bash macura-backflip/bash/setup_colab.sh

set -euo pipefail

GH_USER="${GH_USER:-silvergjeka22}"
REPO_NAME="${REPO_NAME:-macura-backflip}"   # <-- rename the GitHub repo to this (see README)
WORKSPACE="${WORKSPACE:-/content}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/macura-backflip}"
REPO_DIR="${WORKSPACE}/${REPO_NAME}"

echo "==================================================================="
echo " macura-backflip Colab setup"
echo "   workspace : ${WORKSPACE}"
echo "   repo      : ${GH_USER}/${REPO_NAME}"
echo "   drive     : ${DRIVE_ROOT}"
echo "==================================================================="

# ── 1. clone (or update) the private project repo ────────────────────────────
if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "ERROR: GITHUB_TOKEN is not set. The notebook must export it before" >&2
  echo "       running this script (it is a PRIVATE repository)." >&2
  exit 1
fi

if [[ -d "${REPO_DIR}/.git" ]]; then
  echo "[1/3] repo present -> git pull"
  git -C "${REPO_DIR}" pull --ff-only \
    "https://${GITHUB_TOKEN}@github.com/${GH_USER}/${REPO_NAME}.git" main
else
  echo "[1/3] cloning private repo ${GH_USER}/${REPO_NAME}"
  git clone "https://${GITHUB_TOKEN}@github.com/${GH_USER}/${REPO_NAME}.git" "${REPO_DIR}"
  # scrub the token from the remote URL
  git -C "${REPO_DIR}" remote set-url origin "https://github.com/${GH_USER}/${REPO_NAME}.git"
fi

# ── 2. install python dependencies ───────────────────────────────────────────
echo "[2/3] installing requirements"
pip install -q -r "${REPO_DIR}/requirements.txt"

# 2b. Repair numpy consistency (a reinstall can leave numpy's .py newer than its
# compiled .so -> AttributeError). Force a single consistent numpy without
# touching anything else. Non-fatal.
pip install -q --force-reinstall --no-deps "numpy>=2.1" || true
python - <<'PY' || echo "WARNING: stack inconsistent -> Runtime > Restart session, then re-run"
import numpy, mujoco  # noqa
print("      numpy", numpy.__version__, "mujoco", mujoco.__version__, "OK")
PY

# ── 3. create Drive results folders ──────────────────────────────────────────
echo "[3/3] creating Drive results folders"
for sub in checkpoints videos plots logs; do
  mkdir -p "${DRIVE_ROOT}/${sub}"
done

echo "==================================================================="
echo " DONE.  Paths:"
echo "   repo          : ${REPO_DIR}"
echo "   drive results : ${DRIVE_ROOT}"
echo "==================================================================="
