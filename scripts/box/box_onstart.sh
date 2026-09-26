#!/usr/bin/env bash
# box_onstart.sh SCRIPT LIBRARY: a box's bring-up (docs/box_runbook.md
# "Bring-up"). The onstart of scripts/rent_box.py fetches this file from
# LIBRARY, the side copy of the box library next to the code stage on HF
# (scripts/box/box_stage.py `library_dir`), and runs it as
#     bash /workspace/box_onstart.sh SCRIPT LIBRARY
# It fetches box_stop.py first, then the rest of the library into
# /workspace/box/, then the run script from tier-b/staging/SCRIPT, and
# becomes the script (same process, same log). When a file does not
# arrive it stops the instance with box_stop.py, if that one arrived;
# otherwise nothing on the box can stop it, and the laptop must.
set -uo pipefail
[ ! -x /venv/main/bin/python ] || export PATH=/venv/main/bin:$PATH
SCRIPT=${1:?usage: box_onstart.sh SCRIPT LIBRARY}
LIBRARY=${2:?usage: box_onstart.sh SCRIPT LIBRARY}
WORKDIR=${BOX_WORKDIR:-/workspace}
BOX_LIB=$WORKDIR/box
REPO=momom2/wesnoth-model-checkpoints
LIBRARY_FILES=(box_stop.py box_onstart.sh boxlib.sh box_upload.py box_stage.py)  # as box_stage.LIBRARY_FILES

fetch() {                        # fetch HF_PATH DEST: three attempts; DEST whole or not at all
    timeout -k 30s 10m python - "$REPO" "$1" "$2" <<'EOF'
import os
import shutil
import sys
import time

from huggingface_hub import hf_hub_download

repo, source, target = sys.argv[1:4]
for attempt in range(1, 4):
    try:
        path = hf_hub_download(repo, source)
    except Exception as exc:  # noqa: BLE001 -- the type name only
        print(f"box_onstart: {source}: attempt {attempt} {type(exc).__name__}", flush=True)
        if attempt < 3:
            time.sleep(20)
        continue
    shutil.copyfile(path, target + ".tmp")
    os.replace(target + ".tmp", target)
    print(f"box_onstart: fetched {source}", flush=True)
    sys.exit(0)
sys.exit(1)
EOF
}

echo "box_onstart $(date -u +%FT%TZ): $SCRIPT with the library of $LIBRARY"
mkdir -p "$BOX_LIB" || exit 1
missing=()
for f in "${LIBRARY_FILES[@]}"; do
    fetch "$LIBRARY/$f" "$BOX_LIB/$f" || missing+=("$f")
done
fetch "tier-b/staging/$SCRIPT" "$WORKDIR/$SCRIPT" || missing+=("$SCRIPT")
if [ "${#missing[@]}" -eq 0 ]; then
    export BOX_LIB
    exec bash "$WORKDIR/$SCRIPT"
fi
echo "box_onstart: not fetched: ${missing[*]}; stopping the instance"
if [ -f "$BOX_LIB/box_stop.py" ]; then
    exec python "$BOX_LIB/box_stop.py" --outcome "$WORKDIR/onstart_stop.jsonl"
fi
echo "box_onstart: box_stop.py did not arrive either: only the laptop can stop this instance"
exit 1
