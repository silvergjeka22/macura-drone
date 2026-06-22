#!/usr/bin/env bash
# ── setup_colab.sh ───────────────────────────────────────────────────────────
# One-shot Colab bootstrap:
#   1. clone THIS (private) repo into the Colab workspace using a GitHub token
#   2. clone mujoco_menagerie (public) to get the Skydio X2 MJCF model
#   3. (optionally) clone the authors' MACURA reference code + mbrl-lib source
#   4. pip install -r requirements.txt
#   5. create the Google Drive results folders and print all paths
#
# SECURITY: the GitHub token is read from the environment variable GITHUB_TOKEN,
# which the notebook sets at runtime via getpass / Colab Secrets. The token is
# NEVER written to disk and NEVER baked into this script. The clone URL with the
# token is used only for the single git command and not stored in git config.
#
# Usage (from the notebook):
#   !GITHUB_TOKEN=$TOKEN bash macura-drone/bash/setup_colab.sh
# or, if the repo is not yet cloned, the notebook downloads this script first.

set -euo pipefail

# ── configurable paths ───────────────────────────────────────────────────────
GH_USER="${GH_USER:-silvergjeka22}"
REPO_NAME="${REPO_NAME:-macura-drone}"
WORKSPACE="${WORKSPACE:-/content}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/macura-drone}"
CLONE_REFERENCE_CODE="${CLONE_REFERENCE_CODE:-1}"   # set to 0 to skip authors' code

REPO_DIR="${WORKSPACE}/${REPO_NAME}"
MENAGERIE_DIR="${WORKSPACE}/mujoco_menagerie"

echo "==================================================================="
echo " macura-drone Colab setup"
echo "   workspace : ${WORKSPACE}"
echo "   repo      : ${GH_USER}/${REPO_NAME}"
echo "   drive     : ${DRIVE_ROOT}"
echo "==================================================================="

# ── 1. clone (or update) the private project repo ────────────────────────────
if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "ERROR: GITHUB_TOKEN is not set. The notebook must export it before"
  echo "       running this script (it is a PRIVATE repository)." >&2
  exit 1
fi

if [[ -d "${REPO_DIR}/.git" ]]; then
  echo "[1/5] repo already present -> git pull"
  # the remote URL is tokenless (scrubbed after clone), so authenticate inline
  # for this one pull without persisting the credential.
  git -C "${REPO_DIR}" pull --ff-only \
    "https://${GITHUB_TOKEN}@github.com/${GH_USER}/${REPO_NAME}.git" main
else
  echo "[1/5] cloning private repo ${GH_USER}/${REPO_NAME}"
  # token is interpolated only for this one command; nothing persisted.
  git clone "https://${GITHUB_TOKEN}@github.com/${GH_USER}/${REPO_NAME}.git" "${REPO_DIR}"
  # scrub any credential that git might have cached in the remote URL
  git -C "${REPO_DIR}" remote set-url origin "https://github.com/${GH_USER}/${REPO_NAME}.git"
fi

# ── 2. clone mujoco_menagerie (public) for the Skydio X2 model ────────────────
if [[ -d "${MENAGERIE_DIR}/.git" ]]; then
  echo "[2/5] mujoco_menagerie already present -> skip"
else
  echo "[2/5] cloning mujoco_menagerie (Skydio X2 MJCF)"
  git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git "${MENAGERIE_DIR}"
fi
echo "      Skydio X2 scene: ${MENAGERIE_DIR}/skydio_x2/scene.xml"

# ── 3. clone the authors' MACURA reference code (optional) ────────────────────
if [[ "${CLONE_REFERENCE_CODE}" == "1" ]]; then
  if [[ -d "${WORKSPACE}/macura_reference/.git" ]]; then
    echo "[3/5] authors' MACURA code already present -> skip"
  else
    echo "[3/5] cloning authors' MACURA reference code"
    git clone --depth 1 \
      https://github.com/Data-Science-in-Mechanical-Engineering/macura.git \
      "${WORKSPACE}/macura_reference" || \
      echo "      (warning: could not clone reference code; continuing)"
  fi
else
  echo "[3/5] skipping authors' reference code (CLONE_REFERENCE_CODE=0)"
fi

# ── 4. install python dependencies ───────────────────────────────────────────
# Core stack (SB3 + torch + pybullet + mujoco + gymnasium); Python-3.12 friendly.
echo "[4/5] installing requirements (core + SB3 backbone)"
pip install -q -r "${REPO_DIR}/requirements.txt"

# 4b. gym-pybullet-drones is NOT on PyPI -> install from GitHub source.
# Non-fatal: if it fails, the rest of the stack still works and we print guidance
# (the MuJoCo backend remains usable for the cross-engine comparison).
echo "[4b/5] installing gym-pybullet-drones from source"
set +e
pip install -q "git+https://github.com/utiasDSL/gym-pybullet-drones.git"
if python -c "import gym_pybullet_drones" 2>/dev/null; then
  echo "      gym-pybullet-drones import OK"
else
  echo "      WARNING: gym-pybullet-drones did not install/import. Options:"
  echo "        - try a pinned release, e.g. add @v1.0.0 to the git URL above"
  echo "        - or set env.backend: mujoco in the config to use the MuJoCo backend"
fi
set -e

# ── 5. create Drive results folders ──────────────────────────────────────────
echo "[5/5] creating Drive results folders"
for sub in checkpoints videos plots logs; do
  mkdir -p "${DRIVE_ROOT}/${sub}"
done

echo "==================================================================="
echo " DONE. Paths:"
echo "   repo            : ${REPO_DIR}"
echo "   menagerie (X2)  : ${MENAGERIE_DIR}/skydio_x2/scene.xml"
echo "   drive results   : ${DRIVE_ROOT}"
echo "==================================================================="
