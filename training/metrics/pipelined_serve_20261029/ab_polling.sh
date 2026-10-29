#!/usr/bin/env bash
# The pipelined serve loop's waiter on a polled done event (exp/pipelined-
# blocking-wait) against the default event (feature/local-gpu-stage1):
# ab_gpu.sh's 96-in-flight pipelined arm, two rounds, the second in reverse
# order, each run with a sampler of the server process's CPU time.
set -u
export HIP_VISIBLE_DEVICES=1
PY=/c/Users/momom2/venvs/wesnoth_rocm/Scripts/python.exe
CK='C:\Users\momom2\Desktop\perso\projects\wesnoth_ai-stage1\training\checkpoints\own2.pt'
OUT=/c/Users/momom2/wesnoth_runs/pipelined_polling_20261029
WOUT='C:\Users\momom2\wesnoth_runs\pipelined_polling_20261029'
SAMPLER="$(cygpath -w "$(dirname "$0")/server_cpu_sampler.ps1")"
mkdir -p $OUT
run() {  # tag dir rep
  local tag=$1 dir=$2 rep=$3
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$SAMPLER" -Out "$WOUT\\${tag}_${rep}.cpu.txt" -Stop "$WOUT\\${tag}_${rep}.stop" &
  local sp=$!
  echo "$(date +%H:%M:%S) start $tag rep $rep ($dir)" | tee -a $OUT/ab.log
  (cd "$dir" && $PY -u tools/run_elo_batch.py --label-a own2a --spec-a "$CK" --memory-a 64 \
     --label-b own2b --spec-b "$CK" --memory-b 64 --mcts-sims 0 --raw-temperature-a 1 \
     --raw-temperature-b 1 --log-decisions --outdir "$WOUT\\${tag}_${rep}" --games 192 --max-extra-games 0 \
     --seed-base 872700 --persistent-workers --shared-inference --no-infer-compile --device cuda \
     --jobs 96 --worker-games 8 --min-free-mb 300 --infer-fp16 --inference-timeline-s 10 --inference-pipelined \
     > "$OUT/${tag}_${rep}.log" 2>&1)
  echo "$(date +%H:%M:%S) end $tag rep $rep rc=$?" | tee -a $OUT/ab.log
  touch "$OUT/${tag}_${rep}.stop"; wait $sp
}
DEF=/c/Users/momom2/Desktop/perso/projects/wesnoth_ai-stage1
BLK=/c/Users/momom2/Desktop/perso/projects/wesnoth_ai-blockwait
run default $DEF 1; run polling $BLK 1
run polling $BLK 2; run default $DEF 2
for f in $OUT/*.cpu.txt; do echo "== $f"; tail -1 "$f"; done
echo "$(date +%H:%M:%S) all done" | tee -a $OUT/ab.log
