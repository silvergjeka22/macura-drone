#!/bin/bash
# The 4 training notebooks on 2 Kaggle accounts (commands_guide.md has every step).
#   ./kaggle.sh pack [name ...]   put the current src/ into the notebooks' code cell (push does this itself)
#   ./kaggle.sh push [name ...]   pack and start the notebooks (all 4 if no name); running ones are skipped
#   ./kaggle.sh status            the status of all 4
#   ./kaggle.sh get               download the finished ones into downloads/<name> (never overwrites)
# A notebook's account is the first part of "id" in training/<name>/kernel-metadata.json; its key is
# ~/.kaggle/<account>/kaggle.json if that exists, otherwise ~/.kaggle/kaggle.json.
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

pack() {   # src/ + requirements.txt as a base64 tar.gz in the cell with id "code" (same bytes on every computer)
  python3 - "$@" <<'PY'
import base64, gzip, io, json, os, subprocess, sys, tarfile

files = ["requirements.txt"] + sorted(os.path.join(d, f) for d, _, fs in os.walk("src") for f in fs
                                      if "__pycache__" not in d and not f.endswith(".pyc") and not f.startswith("."))
buffer = io.BytesIO()
with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
    for f in files:
        info = tar.gettarinfo(f, arcname=f)
        info.mtime, info.uid, info.gid, info.uname, info.gname, info.mode = 0, 0, 0, "", "", 0o644
        tar.addfile(info, open(f, "rb"))
code = base64.b64encode(buffer.getvalue()).decode()

git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
version = git("log", "--oneline", "-1", "--", "src", "requirements.txt").replace('"', "'")
version += " (with local changes)" if git("status", "--porcelain", "src", "requirements.txt") else ""
lines = "\n".join(code[i:i + 100] for i in range(0, len(code), 100))
source = f'''# The project code (src/ and requirements.txt), packed by ./kaggle.sh pack: Kaggle needs no GitHub access.
import base64, io, tarfile

CODE_VERSION = "{version}"
CODE = """
{lines}
"""
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE))).extractall(ROOT)
sys.path.insert(0, ROOT)
print("code:", CODE_VERSION)'''

for name in sys.argv[1:]:
    path = f"training/{name}/{name}.ipynb"
    nb = json.load(open(path))
    cell = next(c for c in nb["cells"] if c.get("id") == "code")
    cell["source"] = source.splitlines(keepends=True)
    cell["metadata"] = {"_kg_hide-input": True, "jupyter": {"source_hidden": True}}
    open(path, "w").write(json.dumps(nb, sort_keys=True, indent=1, ensure_ascii=False) + "\n")
print("code:", version)
PY
}

case "${1:-help}" in
  pack)
    shift; pack ${*:-$NOTEBOOKS};;
  push)
    shift; names="${*:-$NOTEBOOKS}"
    pack $names
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
  *) sed -n '2,8p' "$0";;
esac
