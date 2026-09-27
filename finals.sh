#!/bin/bash
# The final study on two Kaggle accounts at the same time:
#   notebook A = MACURA + MBPO on nouradon (key ~/.kaggle/kaggle.json), notebook B = M2AC + SAC on silvergjeka01
#   (key ~/.kaggle/silvergjeka01/kaggle.json). Seeds 0-3, 50,000 steps, race3.
#
#   ./finals.sh push      start A and B (refuses while a run is still going; first saves each account's previous output)
#   ./finals.sh push-a    start only A           ./finals.sh push-b    start only B
#   ./finals.sh status    both statuses
#   ./finals.sh get       download A into out_final_a and B into out_final_b (a date is added if the folder exists,
#                         so nothing is ever overwritten); then open notebooks/final_merge.ipynb
set -e
cd "$(dirname "$0")"
A_KERNEL="nouradon/macura-drone";      A_KEYS="$HOME/.kaggle";              A_DIR="kaggle_final_a"; A_OUT="out_final_a"
B_KERNEL="silvergjeka01/macura-drone"; B_KEYS="$HOME/.kaggle/silvergjeka01"; B_DIR="kaggle_final_b"; B_OUT="out_final_b"

kg() { KAGGLE_CONFIG_DIR="$1" kaggle "${@:2}"; }
status() { kg "$2" kernels status "$1" 2>&1 | tail -1; }
fresh() { if [ -e "$1" ]; then echo "$1_$(date +%Y%m%d_%H%M)"; else echo "$1"; fi; }

save_previous() {   # $1 kernel, $2 keys: never lose the output of the account's previous run
  local s d
  s=$(status "$1" "$2")
  case "$s" in
    *RUNNING*|*QUEUED*) echo "STOP: $1 is still running ($s). Wait until it is COMPLETE, then try again."; exit 1;;
  esac
  d=$(fresh "out_previous_${1%%/*}")
  echo "saving the previous output of $1 into $d ..."
  kg "$2" kernels output "$1" -p "$d" > /dev/null 2>&1 && echo "   saved" || echo "   (nothing to save)"
}

get_one() {   # $1 label, $2 kernel, $3 keys, $4 folder
  local s d
  s=$(status "$2" "$3")
  case "$s" in
    *COMPLETE*) ;;
    *) echo "$1 ($2): not finished yet: $s"; return;;
  esac
  d=$(fresh "$4")
  kg "$3" kernels output "$2" -p "$d" > /dev/null
  echo "$1 ($2) -> $d : $(ls "$d"/runs/logs 2>/dev/null | tr '\n' ' ')"
}

case "${1:-help}" in
  push)   save_previous "$A_KERNEL" "$A_KEYS"; save_previous "$B_KERNEL" "$B_KEYS"
          kg "$A_KEYS" kernels push -p "$A_DIR"; kg "$B_KEYS" kernels push -p "$B_DIR" ;;
  push-a) save_previous "$A_KERNEL" "$A_KEYS"; kg "$A_KEYS" kernels push -p "$A_DIR" ;;
  push-b) save_previous "$B_KERNEL" "$B_KEYS"; kg "$B_KEYS" kernels push -p "$B_DIR" ;;
  status) echo "A (MACURA + MBPO): $(status "$A_KERNEL" "$A_KEYS")"
          echo "B (M2AC + SAC):    $(status "$B_KERNEL" "$B_KEYS")" ;;
  get)    get_one A "$A_KERNEL" "$A_KEYS" "$A_OUT"; get_one B "$B_KERNEL" "$B_KEYS" "$B_OUT" ;;
  *)      sed -n '2,10p' "$0" ;;
esac
