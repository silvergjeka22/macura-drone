#!/usr/bin/env bash
# Run the MACURA drone training on a Kaggle GPU kernel, from your PC (VS Code terminal).
#
# One-time local setup (see README):
#   1) pip install kaggle
#   2) put kaggle.json in  ~/.kaggle/kaggle.json   (Windows: C:\Users\<you>\.kaggle\)
#
# KERNEL is your Kaggle username + kernel slug; it must match the "id" in kernel-metadata.json.
KERNEL="silvergjeka01/macura-drone"

set -e
case "${1:-help}" in
  push)    # upload the notebook + kernel-metadata.json and start the run on Kaggle.
           # After this returns, the job runs on Kaggle, so you can turn off your PC.
    kaggle kernels push -p . ;;

  status)  # queued / running / complete / error
    kaggle kernels status "$KERNEL" ;;

  get)     # download the finished output (checkpoints, logs, plots, videos) into ./out
    kaggle kernels output "$KERNEL" -p ./out ;;

  stop)    # Kaggle has no CLI stop, so open the kernel page to click "Stop Session".
    URL="https://www.kaggle.com/code/${KERNEL}"
    echo "Kaggle cannot stop a kernel from the CLI. Opening the kernel page:"
    echo "  $URL"
    echo "On that page click 'Stop Session' (or the running version's stop) to free the GPU."
    { command -v open >/dev/null && open "$URL"; } \
      || { command -v xdg-open >/dev/null && xdg-open "$URL"; } || true ;;

  *)
    echo "usage: ./run.sh [push|status|get|stop]"
    echo "  push    - send the job to Kaggle and start it (then you can turn your PC off)"
    echo "  status  - see if it is queued, running, or done"
    echo "  get     - download results into ./out once status says 'complete'"
    echo "  stop    - open the kernel page so you can stop it (Kaggle has no CLI stop)" ;;
esac
