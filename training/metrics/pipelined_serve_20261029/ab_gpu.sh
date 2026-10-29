#!/usr/bin/env bash
# Idle-GPU A/B of the pipelined serve loop (feature/pipelined-serve), for the
# lead, while stage 1 is stopped: self-play decision rate, 192 games of own2
# at temperature 1, seeds 872700, stage 1's flags (ab_idle.sh's form,
# training/metrics/pipe_buffer_20261029). Four arms from one checkout of this
# branch: the serial loop and the pipelined one (--inference-pipelined), at
# 96 games in flight (12 workers of 8) and at 144 (12 workers of 12). Two
# rounds, the second in reverse order; about 2.7 minutes a run.
# Read: python training/metrics/pipe_buffer_20261029/ab_steady.py $OUT
set -u
export HIP_VISIBLE_DEVICES=1
PY=/c/Users/momom2/venvs/wesnoth_rocm/Scripts/python.exe
CK='C:\Users\momom2\Desktop\perso\projects\wesnoth_ai-stage1\training\checkpoints\own2.pt'
OUT=/c/Users/momom2/wesnoth_runs/pipelined_ab_20261029
DIR=/c/Users/momom2/Desktop/perso/projects/wesnoth_ai-pipeline
mkdir -p $OUT
run() {  # tag jobs worker_games rep [extra flags...]
  local tag=$1 jobs=$2 wg=$3 rep=$4; shift 4
  local od="C:\\Users\\momom2\\wesnoth_runs\\pipelined_ab_20261029\\${tag}_${rep}"
  echo "$(date +%H:%M:%S) start $tag rep $rep $*" | tee -a $OUT/ab.log
  (cd "$DIR" && $PY -u tools/run_elo_batch.py --label-a own2a --spec-a "$CK" --memory-a 64 \
     --label-b own2b --spec-b "$CK" --memory-b 64 --mcts-sims 0 --raw-temperature-a 1 \
     --raw-temperature-b 1 --log-decisions --outdir "$od" --games 192 --max-extra-games 0 \
     --seed-base 872700 --persistent-workers --shared-inference --no-infer-compile --device cuda \
     --jobs $jobs --worker-games $wg --min-free-mb 300 --infer-fp16 --inference-timeline-s 10 "$@" \
     > "$OUT/${tag}_${rep}.log" 2>&1)
  echo "$(date +%H:%M:%S) end $tag rep $rep rc=$?" | tee -a $OUT/ab.log
}
s96()  { run serial96  96  8  $1 --no-inference-pipelined; }
p96()  { run piped96   96  8  $1 --inference-pipelined; }
s144() { run serial144 144 12 $1 --no-inference-pipelined; }
p144() { run piped144  144 12 $1 --inference-pipelined; }
s96 1; p96 1; s144 1; p144 1
p144 2; s144 2; p96 2; s96 2
echo "$(date +%H:%M:%S) all done" | tee -a $OUT/ab.log
