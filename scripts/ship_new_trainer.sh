#!/usr/bin/env bash
# On the box: a second checkout for the patched trainer next to the
# arm's, without copying the corpus: every top-level entry of
# /workspace/Wesnoth-AI is linked, wesnoth_ai/ and tools/ are copied
# so the patched files (scp'd afterwards) replace only their own.
set -euo pipefail
SRC=/workspace/Wesnoth-AI
DST=/workspace/Wesnoth-AI-new
mkdir -p "$DST"
for e in "$SRC"/* "$SRC"/.[!.]*; do
    name=$(basename "$e")
    case "$name" in wesnoth_ai|tools) cp -r "$e" "$DST/$name" ;; *) ln -sfn "$e" "$DST/$name" ;; esac
done
ls "$DST" | wc -l
