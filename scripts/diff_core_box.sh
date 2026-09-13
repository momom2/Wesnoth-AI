#!/usr/bin/env bash
# The phase-4 certification sweep (docs/rust_port_plan.md): every replay
# of a corpus through the Rust-owned state and through the Python
# applier, the two states compared after every command
# (tools/diff_core.py), sharded over the box's cores. Prints one summary
# and writes the shard logs and `summary.txt` under $OUT.
#
#   CORPUS=replays_dataset_imitation SHARDS=16 OUT=/workspace/diff_core_out \
#       bash scripts/diff_core_box.sh
set -euo pipefail
CORPUS="${CORPUS:-replays_dataset_imitation}"
SHARDS="${SHARDS:-$(nproc)}"
OUT="${OUT:-/workspace/diff_core_out}"
mkdir -p "$OUT"
rm -f "$OUT"/shard_* "$OUT"/files.txt
ls "$CORPUS"/*.json.gz > "$OUT/files.txt"
split -n "l/$SHARDS" -d -a 2 "$OUT/files.txt" "$OUT/shard_"
export OMP_NUM_THREADS=1
start=$(date +%s)
for f in "$OUT"/shard_??; do
    ( xargs -a "$f" python tools/diff_core.py > "$f.log" 2>&1 || true ) &
done
wait
end=$(date +%s)
python - "$OUT" "$((end - start))" <<'PYEOF'
import glob, re, sys
from collections import Counter
out, wall = sys.argv[1], int(sys.argv[2])
replays = clean = divergent = rust = py = 0
kinds = Counter()
divergences = []
for log in sorted(glob.glob(f"{out}/shard_??.log")):
    after_summary = False
    for line in open(log, encoding="utf-8", errors="replace"):
        m = re.match(r"diff_core: (\d+) replays, (\d+) clean, (\d+) with divergences; commands rust=(\d+) python=(\d+)", line)
        if m:
            replays += int(m[1]); clean += int(m[2]); divergent += int(m[3])
            rust += int(m[4]); py += int(m[5])
            after_summary = True
            continue
        if after_summary:          # the per-kind counts follow the summary line
            after_summary = False
            for part in line.strip().split(", "):
                k, _, v = part.partition("=")
                if v.isdigit():
                    kinds[k] += int(v)
            continue
        if line.startswith("  ") and ".json.gz" in line:
            divergences.append(line.strip())
lines = [f"diff_core sweep: {replays} replays, {clean} clean, {divergent} with divergences; "
         f"commands rust={rust} python={py}; wall {wall} s",
         "  " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))]
lines += ["  " + d[:400] for d in divergences[:300]]
text = "\n".join(lines)
print(text)
open(f"{out}/summary.txt", "w", encoding="utf-8").write(text + "\n")
PYEOF
