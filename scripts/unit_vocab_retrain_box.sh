#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# The reference's recipe with every reachable unit type on its own row of
# the type embedding (tools/unit_vocab.py) and the neutral side's commands
# out of the imitation pairs. `obs8` has 157 of its 356 type names on the
# overflow row, 76 of the 190 unit types our games can field among them.
# Everything else is `scripts/observation_retrain_box.sh` to the letter:
# the relevant-set basis, --terrain-multi-hot, configs/imitation.json,
# batch 64, lr 1e-4, cosine over 4 epochs stopped after STOP_AFTER_EPOCH
# passes, the deduplicated corpus with the manifest split, fog gate on,
# pre-encoded records, the same run seed. Then:
#   the per-phase value evaluation of the checkpoint;
#   the arm against the reference player (`obs8`), both at the reference
#     decode (raw:t0+eo-1.5), PURE, 800 decisive games, each side served
#     in its own encoding by its own inference server; the match is read
#     only once it holds its 800 decisive games.
# Bars and predictions: docs/unit_vocab_retrain_prereg_20260925.md.
#
# Runs on the box library (scripts/box/boxlib.sh, docs/box_runbook.md):
# the onstart of scripts/rent_box.py fetches the library of STAGE, then
# this script. Every step has a bound: the pass is cut at TRAIN_CUT_MIN and
# ended when train.log stays silent for TRAIN_STALL_MIN, and the dead-man's
# switch finishes the entry after BOX_MAX_H hours whatever it is doing.
# Records go to HF $HF_DIR every 30 minutes and at the end, the match's
# games as one tarball. Re-entry, on this machine or a new one (files
# absent here come back from HF), skips finished steps and continues a cut
# pass where it stood (tools/supervised_train.py `PassPosition`). Every
# exit, clean or not, uploads the records with ALL_DONE last and stops the
# instance. Never `set -x`: the HF token and the instance key are in the
# environment.
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/vocabretrain
ENC=$WORKDIR/encoded_vocab
STAGE="${STAGE:-}"               # the code stage built from main at rental (tools/stage_code.py)
EPOCHS="${EPOCHS:-4}"
STOP_AFTER_EPOCH="${STOP_AFTER_EPOCH:-1}"
RUN_SEED="${RUN_SEED:-20260909}"
WORKERS="${WORKERS:-30}"
GAMES="${GAMES:-800}"
JOBS="${JOBS:-20}"
export HF_DIR="${HF_DIR:-tier-b/unit_vocab_retrain_20260925}"
ARCH=(--d-model 384 --num-layers 8 --num-heads 12 --d-ff 1536)
EVAL=(--eval-every 50000 --eval-pairs 1200 --eval-pairs-per-game 8 --eval-sample-seed 0)
# Bounds in minutes, from the pre-registration's "Cost".
PREENCODE_CUT_MIN="${PREENCODE_CUT_MIN:-60}"     # estimated 20
TRAIN_CUT_MIN="${TRAIN_CUT_MIN:-480}"            # the pass: 3-5 hours
TRAIN_STALL_MIN="${TRAIN_STALL_MIN:-30}"         # the trainer logs every 100 steps, about 40 s
RESUME_STALL_MIN="${RESUME_STALL_MIN:-100}"      # a resumed pass re-reads its trained pairs without a
                                                 # line: 1.79M pairs took 49 minutes on 2026-09-25
PHASE_CUT_MIN="${PHASE_CUT_MIN:-20}"             # estimated 5
MATCH_CUT_MIN="${MATCH_CUT_MIN:-75}"             # the match stops itself at 60 (--time-budget-min)
BOX_MAX_H="${BOX_MAX_H:-9}"                      # twice the 4.5 box-hours estimated
BOX_OUT=$OUT
# shellcheck source=box/boxlib.sh
. "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library (docs/box_runbook.md)"; exit 1; }

CKPT="$OUT/arm_epoch$((STOP_AFTER_EPOCH - 1)).pt"
E="e$STOP_AFTER_EPOCH"
NAME="vocab_${E}_vs_obs8"

decisive_results() {             # decisive_results DIR: the games in DIR that ended in a win or a loss
    timeout 2m python - "$1" <<'EOF'
import json, pathlib, sys
n = 0
for p in pathlib.Path(sys.argv[1]).glob("game_*.json"):
    try:
        n += json.loads(p.read_text(encoding="utf-8")).get("outcome_a") in ("win", "loss")
    except Exception:
        pass
print(n)
EOF
}
notes() {                        # the pass's pair count and the match's decisive games, for the finish reason
    local pairs
    pairs=$(grep -o "chain total [0-9]*" "$OUT/train.log" 2>/dev/null | tail -1)
    echo "pairs: ${pairs:-none}, decisive: $(decisive_results "$OUT/games_$NAME")"
}
box_notes() { notes; }
box_on_round() {                 # progress.txt, before each upload round
    { date -u
      grep -o "epoch=[0-9]* step=[0-9]*.*pairs=[0-9]* rate=[0-9.]*/s wall=[0-9.]*m" "$OUT/train.log" 2>/dev/null \
          | tail -1 | sed "s/avg_loss.*pairs=/pairs=/"
      grep "EVAL\[epoch" "$OUT/train.log" 2>/dev/null | cut -c1-160
      find "$OUT" -maxdepth 1 -type f -printf '%f ' 2>/dev/null; echo
      tail -n 2 "$OUT"/*.log 2>/dev/null | tail -n 8
    } > "$OUT/progress.txt.tmp" && mv -f "$OUT/progress.txt.tmp" "$OUT/progress.txt"
}

box_init
# The stage the run was pre-registered with predates the signal telemetry
# (0.7.0): a rental names the stage it built, or the box stops here.
[ -n "$STAGE" ] || box_finish "NO_STAGE: build the code stage from main (tools/stage_code.py) and pass STAGE" 1
box_restore DONE STOPPED_AFTER_EPOCH "${CKPT##*/}" arm.pt fresh_vocab.pt train.log \
    arm_eval.jsonl arm_signal.jsonl arm_prof.json phase_obs.json phase_obs.md \
    || box_finish "RESTORE_FAILED (restore.log)" 1
box_upload_hold DONE "${CKPT##*/}"
box_upload_hold STOPPED_AFTER_EPOCH "${CKPT##*/}"
box_pip huggingface_hub psutil pytest scipy requests || echo "pip install failed (pip.log)"

# ---- the code and the Rust wheel (the pre-encoder, the trainer's probe and eval all run its kernels)
box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED (no $BOX_REPO)" 1
box_build_wheel || box_finish "BUILD_FAILED (build.log)" 1
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
timeout 2m python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || box_finish "NO_CUDA (box.txt)" 1
box_upload_async
box_monitor_start

# ---- the encoder and the observation against the Rust kernels, before anything trains
if ! box_marked_this_stage "$BOX_STATE/TESTED"; then
    : > "$OUT/tests_obs.log"
    box_bounded tests 30 tests_obs.log python -m pytest tests/test_unit_vocab.py tests/test_game_record.py \
        tests/test_time_of_day_features.py tests/test_terrain_multi_hot.py tests/test_rust_encode_raw.py \
        tests/test_game_core.py tests/test_vision.py tests/test_rust_observe.py \
        -q -p no:cacheprovider -m ""
    tail -n 3 "$OUT/tests_obs.log"
    [ "$BOX_RC" -eq 0 ] || box_finish "TESTS_FAILED rc=$BOX_RC: vocabulary, pair, encoder or observation tests (tests_obs.log)" 1
    box_mark "$BOX_STATE/TESTED"
fi

# ---- the reference, the corpus, the fresh vocabulary, the pre-encoded records
box_bounded reference 15 staging.log python tools/reference_player.py --ensure \
    || box_finish "REFERENCE_MISSING rc=$BOX_RC (staging.log)" 1
if [ ! -f replays_dataset_imitation/manifest.jsonl ]; then
    { rm -rf .corpus_tmp && mkdir .corpus_tmp; } || box_finish "CORPUS_STAGING_FAILED (no .corpus_tmp)" 1
    box_bounded corpus 20 staging.log python - <<'EOF' || box_finish "CORPUS_STAGING_FAILED rc=$BOX_RC (staging.log)" 1
import json, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download("momom2/wesnoth-model-checkpoints",
                    "tier-b/replays_dataset_imitation_dedup_20260908.tar.gz")
with tarfile.open(p, "r:gz") as tf:
    tf.extractall(".corpus_tmp")
rows = [json.loads(l) for l in open(".corpus_tmp/replays_dataset_imitation/manifest.jsonl")]
print("corpus", len(rows), "games, holdout", sum(1 for r in rows if r.get("holdout")), flush=True)
EOF
    { mv .corpus_tmp/replays_dataset_imitation replays_dataset_imitation && rm -rf .corpus_tmp; } \
        || box_finish "CORPUS_STAGING_FAILED (the tarball holds no replays_dataset_imitation)" 1
fi

if [ ! -f "$OUT/fresh_vocab.pt" ]; then
    box_bounded vocab 10 staging.log python - "$OUT/fresh_vocab.pt" <<'EOF' || box_finish "VOCAB_FAILED rc=$BOX_RC (staging.log)" 1
import os, sys
import torch
from tools.unit_vocab import seed_vocab
from wesnoth_ai.encoder import GameStateEncoder, names_on_overflow_row
enc = GameStateEncoder(d_model=32)
seed_vocab(enc)                                 # refuses a set that reaches the overflow row
assert len(enc.unit_type_to_id) == 190, len(enc.unit_type_to_id)
assert names_on_overflow_row(enc.unit_type_to_id) == []
torch.save({"unit_type_to_id": dict(enc.unit_type_to_id),
            "faction_to_id": dict(enc.faction_to_id)}, sys.argv[1] + ".tmp")
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
print("fresh vocab:", len(enc.unit_type_to_id), "types,", len(enc.faction_to_id), "factions", flush=True)
EOF
fi

if [ ! -f "$ENC/PREENCODE_DONE" ]; then
    from=$(box_size "$OUT/preencode.log")
    box_bounded --stall "$OUT/preencode.log" 15 preencode "$PREENCODE_CUT_MIN" preencode.log \
        python tools/preencode_corpus.py --dataset replays_dataset_imitation --out "$ENC" \
        --vocab-from "$OUT/fresh_vocab.pt" --fog-hides-enemy-villages --relevant-set-hexes \
        --terrain-multi-hot --workers "$WORKERS"
    tail -n 3 "$OUT/preencode.log"
    if tail -c "+$(( from + 1 ))" "$OUT/preencode.log" | grep -q "PREENCODE_DONE"; then
        box_mark "$ENC/PREENCODE_DONE"
    else
        box_finish "PREENCODE_FAILED rc=$BOX_RC $BOX_WHY (preencode.log)" 1
    fi
fi

# ---- the pass: stopped once epoch STOP_AFTER_EPOCH's checkpoint and holdout eval are written
arm_epochs_done() {              # the completed epochs arm.pt records; -1 when unreadable
    timeout 5m python -c 'import sys, torch
print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("supervised_epoch", -1)))' \
        "$OUT/arm.pt" 2>/dev/null || echo -1
}
train_attempt() {                # train_attempt MINUTES: the pass, continuing arm.pt when present; sets BOX_RC, BOX_WHY
    local resume=() stall=$TRAIN_STALL_MIN
    if [ -f "$OUT/arm.pt" ]; then
        resume=(--resume "$OUT/arm.pt")
        stall=$RESUME_STALL_MIN
    fi
    box_bounded --stall "$OUT/train.log" "$stall" \
        --until "$OUT/train.log" "EVAL[epoch$((STOP_AFTER_EPOCH - 1))-end]" \
        train "$1" train.log \
        python tools/supervised_train.py replays_dataset_imitation \
        --checkpoint "$OUT/arm.pt" ${resume[@]+"${resume[@]}"} \
        --imitation-config configs/imitation.json "${ARCH[@]}" --relevant-set-hexes --terrain-multi-hot \
        --epochs "$EPOCHS" --seed "$RUN_SEED" \
        --bs 64 --lr 1e-4 --device cuda --workers 0 --preencoded "$ENC" \
        "${EVAL[@]}" --ckpt-every 2000 --log-every 100
}
if [ ! -f "$CKPT" ]; then
    # A checkpoint past the pass without the pass's own file cannot be
    # trained back to it: say so rather than train the next epoch.
    if [ -f "$OUT/arm.pt" ] && [ "$(arm_epochs_done)" -ge "$STOP_AFTER_EPOCH" ]; then
        box_finish "PASS_CHECKPOINT_MISSING: arm.pt completed epoch $((STOP_AFTER_EPOCH - 1)), ${CKPT##*/} is absent" 1
    fi
    deadline=$(( $(date +%s) + TRAIN_CUT_MIN * 60 ))
    train_attempt "$TRAIN_CUT_MIN"
    left=$(( (deadline - $(date +%s)) / 60 ))
    if [ ! -f "$CKPT" ] && [ "$BOX_WHY" = failed ] && [ "$left" -ge 60 ]; then
        train_attempt "$left"            # a crash retries once, continuing from the last periodic checkpoint
    fi
    [ -f "$CKPT" ] || box_finish "TRAINING_${BOX_WHY^^} rc=$BOX_RC (train.log; the pass continues from arm.pt on re-entry) $(notes)" 1
    [ "$BOX_WHY" != until ] || box_mark "$OUT/STOPPED_AFTER_EPOCH" "the trainer stopped after epoch $STOP_AFTER_EPOCH"
fi
[ -f "$OUT/DONE" ] || box_mark "$OUT/DONE"
box_upload_skip arm.pt                   # the pass's checkpoint is ${CKPT##*/}
box_upload_async

[ -f "$OUT/phase_obs.json" ] || box_bounded phase "$PHASE_CUT_MIN" phase.log \
    python tools/analysis/value_head_by_phase.py --checkpoint "$CKPT" \
    --jobs 20 --device cuda --out "$OUT/phase_obs.json" \
    || echo "the per-phase evaluation failed: rc=$BOX_RC $BOX_WHY (phase.log)"

# ---- the match
match() {                        # match NAME SPEC_A SPEC_B GAMES SEED_BASE MAX_EXTRA: one attempt, resumed in its directory
    local name="$1" a="$2" b="$3" games="$4" sb="$5" extra="$6"
    local dir="$OUT/games_$name" t0 f
    if [ -f "$OUT/$name.fit.json" ] || [ -f "$OUT/timing_$name.txt" ]; then echo "match $name done"; return 0; fi
    t0=$(date +%s)
    box_bounded "match $name" "$MATCH_CUT_MIN" "$name.log" \
        python tools/run_elo_batch.py --label-a "${name%%_vs_*}" --spec-a "$a" --label-b "${name##*_vs_}" --spec-b "$b" \
        --outdir "$dir" --games "$games" --max-extra-games "$extra" --seed-base "$sb" \
        --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 \
        --raw-end-turn-offset-a "$EO" --raw-end-turn-offset-b "$EO" \
        --persistent-workers --shared-inference --no-infer-compile --device cuda --jobs "$JOBS" \
        --time-budget-min 60
    echo "$name: $(( $(date +%s) - t0 )) s, $(find "$dir" -maxdepth 1 -name 'game_*.json' 2>/dev/null | wc -l) games," \
         "$(decisive_results "$dir") decisive, rc=$BOX_RC $BOX_WHY" | tee "$OUT/timing_$name.txt" | tee -a "$OUT/match.walls"
    for f in "$dir"/.inference_server_*.json; do     # one per checkpoint: _0 the arm's, _1 the reference's
        [ -f "$f" ] && cp -f "$f" "$OUT/$name.server_${f##*/.inference_server_}"
    done
    if [ "$games" -ge 100 ]; then
        box_bounded "fit $name" 10 "$name.log" \
            python tools/elo_collect.py "$dir" --no-catalog --save-json "$OUT/$name.fit.json"
    fi
}
EO=$(timeout 1m python -c "import json; print(json.load(open('configs/reference_player.json'))['decode']['raw_end_turn_offset'])") \
    || box_finish "REFERENCE_CONFIG_UNREADABLE (configs/reference_player.json)" 1
REF=$(timeout 1m python -c "import json; print(json.load(open('configs/reference_player.json'))['checkpoint_local'])") \
    || box_finish "REFERENCE_CONFIG_UNREADABLE (configs/reference_player.json)" 1
ARM="training/checkpoints/vocab_$E.pt"
mkdir -p training/checkpoints
cp "$CKPT" "$ARM" || box_finish "ARM_COPY_FAILED ($CKPT)" 1
box_upload_dir "games_$NAME" "$OUT/games_$NAME"
box_upload_hold "$NAME.fit.json" "games_$NAME.tar.gz"
box_upload_hold "timing_$NAME.txt" "games_$NAME.tar.gz"
match "$NAME" "$ARM" "$REF" "$GAMES" 70000 1500
# A match short of its decisive games (failed games, a dead server, the
# time budget) runs once more in the same directory, which resumes it;
# a match still short is recorded as cut and its fit left unread.
if [ "$(decisive_results "$OUT/games_$NAME")" -lt "$GAMES" ]; then
    rm -f "$OUT/$NAME.fit.json" "$OUT/timing_$NAME.txt"
    match "$NAME" "$ARM" "$REF" "$GAMES" 70000 1500
fi
n_decisive=$(decisive_results "$OUT/games_$NAME")
match_note=""
if [ "$n_decisive" -lt "$GAMES" ]; then
    echo "MATCH_CUT: $n_decisive of $GAMES decisive games; the fit is not the verdict" | tee -a "$OUT/match.walls"
    match_note=" MATCH_CUT"
fi
box_on_round
box_finish "UNIT_VOCAB_RETRAIN_DONE$match_note $(notes)"
