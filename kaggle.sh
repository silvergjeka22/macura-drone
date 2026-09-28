#!/bin/bash
# The 4 training notebooks on 2 Kaggle accounts (see training/ and HOW_TO_RUN.md). The account of a notebook is the
# first part of "id" in training/<name>/kernel-metadata.json; its key is ~/.kaggle/<account>/kaggle.json if that
# file exists, otherwise ~/.kaggle/kaggle.json.
#   nouradon      (key ~/.kaggle/kaggle.json):              macura_mbpo_seeds45, m2ac_sac_seeds45
#   silvergjeka01 (key ~/.kaggle/silvergjeka01/kaggle.json): macura_mbpo_seeds67, m2ac_sac_seeds67
#
#   ./kaggle.sh push [name ...]   start the training notebooks (all 4 if no name is given); a notebook that is still
#                                 running is not started again, and a finished one's output is downloaded first
#   ./kaggle.sh status            the status of all 4
#   ./kaggle.sh get               download every finished notebook into downloads/<name> (a new folder with the date
#                                 if it already exists, so nothing is ever overwritten); then open testing/test.ipynb
set -e
cd "$(dirname "$0")"
NOTEBOOKS="macura_mbpo_seeds45 m2ac_sac_seeds45 macura_mbpo_seeds67 m2ac_sac_seeds67"

kernel() { python3 -c "import json; print(json.load(open('training/$1/kernel-metadata.json'))['id'])"; }
keys()   { a="$(kernel "$1")"; a="${a%%/*}"; if [ -f "$HOME/.kaggle/$a/kaggle.json" ]; then echo "$HOME/.kaggle/$a"; else echo "$HOME/.kaggle"; fi; }
kg()     { KAGGLE_CONFIG_DIR="$(keys "$1")" kaggle "${@:2}"; }
status() { kg "$1" kernels status "$(kernel "$1")" 2>&1 | tail -1; }
fresh()  { if [ -e "$1" ]; then echo "$1_$(date +%Y%m%d_%H%M)"; else echo "$1"; fi; }

download() {   # $1 name, $2 folder
  mkdir -p "$(dirname "$2")"
  kg "$1" kernels output "$(kernel "$1")" -p "$2" > /dev/null
  echo "   $1 -> $2: $(ls "$2"/runs/logs 2>/dev/null | tr '\n' ' ')"
}

case "${1:-help}" in
  push)
    shift; names="${*:-$NOTEBOOKS}"
    for n in $names; do
      [ -d "training/$n" ] || { echo "unknown notebook: $n"; exit 1; }
      s=$(status "$n")
      case "$s" in
        *RUNNING*|*QUEUED*) echo "$n: still running, not started again ($s)"; continue;;
        *COMPLETE*) echo "$n: saving the previous output first"; download "$n" "$(fresh "downloads/previous/$n")";;
      esac
      echo "$n: starting on $(kernel "$n")"
      kg "$n" kernels push -p "training/$n"
    done;;
  status)
    for n in $NOTEBOOKS; do printf "%-22s %s\n" "$n" "$(status "$n")"; done;;
  get)
    for n in $NOTEBOOKS; do
      s=$(status "$n")
      case "$s" in
        *COMPLETE*) download "$n" "$(fresh "downloads/$n")";;
        *) echo "   $n: not finished ($s)";;
      esac
    done;;
  *) sed -n '2,10p' "$0";;
esac
