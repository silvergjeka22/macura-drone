#!/usr/bin/env bash
# Run the MACURA drone training on a Kaggle GPU kernel, from your PC (VS Code terminal).
#
# One-time local setup (see README):
#   1) pip install kaggle
#   2) put your kaggle.json token in  ~/.kaggle/kaggle.json   (Windows: C:\Users\<you>\.kaggle\)
#
# EDIT THIS: replace MYUSERNAME with your Kaggle username. It must match the "id" field
# in kernel-metadata.json (e.g. jane123/macura-drone).
KERNEL="silvergjeka01/macura-drone"

set -e
case "${1:-help}" in
  push)    # upload the notebook + kernel-metadata.json and START the run on Kaggle's servers.
           # After this returns, the job runs on Kaggle - you can close VS Code and turn off your PC.
    kaggle kernels push -p . ;;

  status)  # check whether the run is queued / running / complete / errored.
    kaggle kernels status "$KERNEL" ;;

  get)     # download the finished output (checkpoints, logs, plots) into ./out
    kaggle kernels output "$KERNEL" -p ./out ;;

  *)
    echo "usage: ./run.sh [push|status|get]"
    echo "  push    - send the job to Kaggle and start it (then you can turn your PC off)"
    echo "  status  - see if it is queued, running, or done"
    echo "  get     - download results into ./out once status says 'complete'" ;;
esac
