#!/usr/bin/env bash
# Idle-GPU A/B while stage 1 is stopped: self-play decision rate of five
# arms, 192 games of own2 at temperature 1, 96 in flight, seeds 872700:
# old (wesnoth_ai-stage1 at 041bd7ca), bias (feature/hip-bias-fold 98e1ab1b),
# async (feature/async-replies 124ad8b4, --inference-async-replies), pipe
# (feature/pipe-buffer 766022a6), biaspipe (bias and pipe merged, detached
# 6d0713a6). Two rounds, the second in reverse order.
set -u
export HIP_VISIBLE_DEVICES=1
PY=/c/Users/momom2/venvs/wesnoth_rocm/Scripts/python.exe
CK='C:\Users\momom2\Desktop\perso\projects\wesnoth_ai-stage1\training\checkpoints\own2.pt'
OUT=/c/Users/momom2/wesnoth_runs/ab_idle_20261029
P=/c/Users/momom2/Desktop/perso/projects
mkdir -p $OUT
run() {  # tag checkout rep [extra flags...]
  local tag=$1 dir=$2 rep=$3; shift 3
  local od="C:\\Users\\momom2\\wesnoth_runs\\ab_idle_20261029\\${tag}_${rep}"
  echo "$(date +%H:%M:%S) start $tag rep $rep in $dir $*" | tee -a $OUT/ab.log
  (cd "$dir" && $PY -u tools/run_elo_batch.py --label-a own2a --spec-a "$CK" --memory-a 64 \
     --label-b own2b --spec-b "$CK" --memory-b 64 --mcts-sims 0 --raw-temperature-a 1 \
     --raw-temperature-b 1 --log-decisions --outdir "$od" --games 192 --max-extra-games 0 \
     --seed-base 872700 --persistent-workers --shared-inference --no-infer-compile --device cuda \
     --jobs 96 --worker-games 8 --min-free-mb 300 --infer-fp16 --inference-timeline-s 10 "$@" \
     > "$OUT/${tag}_${rep}.log" 2>&1)
  echo "$(date +%H:%M:%S) end $tag rep $rep rc=$?" | tee -a $OUT/ab.log
}
old()   { run old      $P/wesnoth_ai-stage1      $1; }
bias()  { run bias     $P/wesnoth_ai-biasfold    $1; }
async() { run async    $P/wesnoth_ai-asyncreply  $1 --inference-async-replies; }
pipe()  { run pipe     $P/wesnoth_ai-pipebuf     $1; }
bp()    { run biaspipe $P/wesnoth_ai-abbiaspipe  $1; }
old 1; bias 1; async 1; pipe 1; bp 1
bp 2; pipe 2; async 2; bias 2; old 2
echo "$(date +%H:%M:%S) all done" | tee -a $OUT/ab.log
