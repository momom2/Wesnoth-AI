#!/bin/bash
# Prereg departure 43's hand steps before the lean line's launch (run once, after final_match_r18 lands).
# Every check runs before any change; the first change is the run worktree's checkout.
set -euo pipefail
R=/c/Users/momom2/wesnoth_runs
RL=$R/rounds_local
RUN=/c/Users/momom2/Desktop/perso/projects/wesnoth_ai-run
HEAD=5d67c834
PY=/c/Users/momom2/Desktop/perso/projects/wesnoth_ai/.venv/Scripts/python.exe
$PY -c "import json; r = json.load(open(r'C:\Users\momom2\wesnoth_runs\final_match_r18\result.json')); assert r['complete'] and r['elo'] is not None and r['se'] is not None and r['checkpoint_sha256'].startswith('c5796941'), r; print('final', r['elo'], r['se'], r['decisive'])"
test ! -e $RL/STOP
test ! -e $RL/round_19/play_failed42
test -d $RL/round_19/play
test -f $RL/r19_play.failed
test ! -e $RL/round_18/argmax
test -z "$(git -C $RUN status --porcelain)"
git -C $RUN checkout -q --detach $HEAD
git -C $RUN log --oneline -1
# round 24's argmax base: departure 42's final on round 18
mkdir -p $RL/round_18/argmax
cp $R/final_match_r18/result.json $RL/round_18/argmax/result.json
echo "copied from $R/final_match_r18/result.json on $(date '+%F %T'): round 24's argmax base (prereg departure 43)"   > $RL/round_18/argmax/README.txt
# round 19's replay (departure 42's retry rule, departure 43's |H|)
mv $RL/round_19/play $RL/round_19/play_failed42
echo 29500000 > $RL/round_19/seed_base.txt
cp $RL/round_19/held.txt $RL/round_19/held_failed42.txt
echo 1250 > $RL/round_19/held.txt
mv $RL/r19_play.failed $RL/r19_play.failed42
# the replay's wall is its own: round 19's start was the failed attempt's
cp $RL/times.json $RL/times_pre43.json
$PY - <<'PYEOF'
import json
p = r"C:\Users\momom2\wesnoth_runs\rounds_local\times.json"
t = json.load(open(p, encoding="utf-8"))
print("round 19's old entry:", t.pop("19", None))
open(p, "w", encoding="utf-8").write(json.dumps(t, indent=1))
PYEOF
echo "REDEPLOY at $HEAD (prereg departure 43): the lean line, rebased after round 3, resumes from round 19's replay" > $RL/STOP
ls $RL/round_19; cat $RL/round_19/seed_base.txt $RL/round_19/held.txt $RL/round_19/n_train.txt
echo DEPLOY_READY
