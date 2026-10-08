#!/usr/bin/env python3
"""The free readouts of step 1 (docs/selfplay_program_20261008.md, "Step 1"):
how far apart the reference's top actions are under its decode, at its own
decisions in recorded games. They set step 2's k (actions considered) and c
(the clip).

    python tools/prior_gaps.py --games games_cand64_vs_ref --player cand64 \\
        --checkpoint training/checkpoints/parity3.pt --out OUT/prior_gaps.json [--device cuda]

`--count` games of a match directory, drawn with `--seed` from its records,
are replayed decision by decision (tools/game_record.walk, fingerprints
checked). At each decision of the player's side, the reference
(configs/reference_player.json: the checkpoint must be the one the match
recorded for that side, the memory slots the record's) reads the position
with its memory carried from its previous decision, as in play
(tools/raw_player.RawPolicyPlayer.legal_priors), and its priors over the
legal actions are taken under the decode's end_turn offset:
  - the log-prior gap between its top action and each of the next seven;
  - the share of decisions where a clip c of 0.5, 1 or 2 could flip the
    argmax: the gap to the second action under 2c (Muesli's target moves a
    log-prior by at most c either way);
  - the share whose top eight hold two or more attacks.

A decision is one command, except an attack from a distance, which the
simulator plays as a move and then the attack: a move whose next command
is an attack from its last hex is one decision with it when the replayed
argmax is that attack. How often the replayed argmax is the recorded
choice (its type and hexes) is reported: it says how closely the replay
follows the game the bf16 inference server played.

Each game's row goes to `<out stem>.games.jsonl` as it finishes; --out
holds the summary.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("prior_gaps")

TOP = 8
CLIPS = (0.5, 1.0, 2.0)
DECISION_KINDS = frozenset({"move", "attack", "recruit", "recall", "end_turn"})
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


# ---------------------------------------------------------------------
# One decision
# ---------------------------------------------------------------------

def decision_reading(legal: Sequence, priors: np.ndarray) -> Dict:
    """The gaps, in nats, from the top prior to each of the next TOP - 1,
    and the attacks among the top TOP."""
    if len(legal) == 0:
        return {"legal": 0, "gaps": [], "attacks_top": 0, "top_type": None}
    order = np.argsort(-priors, kind="stable")[:TOP]
    logp = np.log(np.maximum(priors[order], 1e-300))
    return {"legal": int(len(legal)), "gaps": [float(logp[0] - x) for x in logp[1:]],
            "attacks_top": int(sum(legal[i].action.get("type") == "attack" for i in order)),
            "top_type": legal[int(order[0])].action.get("type")}


def _xy(pos) -> Tuple[int, int]:
    return int(pos.x), int(pos.y)


def is_premove(cmd: list, nxt: Optional[list]) -> bool:
    """A move whose next command is an attack from its last hex."""
    return (cmd[0] == "move" and nxt is not None and nxt[0] == "attack"
            and (int(nxt[1]), int(nxt[2])) == (int(cmd[1][-1]), int(cmd[2][-1])))


def matches(action: Dict, cmd: list, nxt: Optional[list]) -> bool:
    """Whether a chosen action is the recorded command (or, for an attack
    from a distance, the recorded move and attack)."""
    kind = action.get("type")
    if kind == "end_turn":
        return cmd[0] == "end_turn"
    if kind == "recruit":
        return cmd[0] == "recruit" and cmd[1] == action.get("unit_type") \
            and (int(cmd[2]), int(cmd[3])) == _xy(action["target_hex"])
    if kind == "move":
        return cmd[0] == "move" and (int(cmd[1][0]), int(cmd[2][0])) == _xy(action["start_hex"]) \
            and (int(cmd[1][-1]), int(cmd[2][-1])) == _xy(action["target_hex"])
    if kind == "attack":
        if cmd[0] == "move":
            attack, start = nxt, (int(cmd[1][0]), int(cmd[2][0]))
        elif cmd[0] == "attack":
            attack, start = cmd, (int(cmd[1]), int(cmd[2]))
        else:
            return False
        return attack is not None and attack[0] == "attack" and start == _xy(action["start_hex"]) \
            and (int(attack[3]), int(attack[4])) == _xy(action["target_hex"]) \
            and int(attack[5]) == int(action.get("attack_index", -1))
    return False


def read_game(rec: Dict, player, side: int, label: str) -> Tuple[List[Dict], Dict[str, int]]:
    """The player's readings at each of its decisions in one record, and
    the replay's agreement with the record."""
    from tools.game_record import walk
    commands = rec["commands"]
    readings: List[Dict] = []
    tally = {"decisions": 0, "agree": 0, "premoves": 0}
    skip = -1
    for k, gs, cmd in walk(rec):
        if k == skip or int(gs.global_info.current_side) != side or cmd[0] not in DECISION_KINDS:
            continue
        nxt = commands[k + 1] if k + 1 < len(commands) else None
        legal, priors = player.legal_priors(gs, game_label=label)
        reading = decision_reading(legal, priors)
        readings.append(reading)
        tally["decisions"] += 1
        if len(legal):
            top = legal[int(np.argmax(priors))].action
            tally["agree"] += int(matches(top, cmd, nxt))
            if is_premove(cmd, nxt) and top.get("type") == "attack":
                skip = k + 1
                tally["premoves"] += 1
    player.drop_pending(label)
    return readings, tally


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def summarize(readings: Sequence[Dict]) -> Dict:
    """The distribution of each gap, the clip shares and the attack share,
    over the decisions with two or more legal actions."""
    multi = [r for r in readings if r["legal"] >= 2]
    out: Dict = {"decisions": len(readings), "decisions_multi": len(multi),
                 "single_legal": sum(1 for r in readings if r["legal"] == 1),
                 "no_legal": sum(1 for r in readings if r["legal"] == 0)}
    if not multi:
        return out
    gaps = {}
    for j in range(1, TOP):
        values = np.array([r["gaps"][j - 1] for r in multi if len(r["gaps"]) >= j])
        if len(values):
            gaps[f"top1_minus_{j + 1}"] = {"n": int(len(values)),
                                           **{f"q{int(q * 100)}": float(np.quantile(values, q)) for q in QUANTILES}}
    second = np.array([r["gaps"][0] for r in multi])
    out["gaps"] = gaps
    out["flip_possible"] = {f"c={c}": float(np.mean(second < 2 * c)) for c in CLIPS}
    out["two_or_more_attacks_in_top8"] = float(np.mean([r["attacks_top"] >= 2 for r in multi]))
    out["top_is_end_turn"] = float(np.mean([r["top_type"] == "end_turn" for r in multi]))
    return out


def pick_games(games_dir: Path, count: int, seed: int) -> List[Path]:
    paths = sorted(Path(games_dir).glob("*.game.jsonl.gz"))
    return sorted(random.Random(seed).sample(paths, min(count, len(paths))))


def player_side(result: Dict, label: str) -> Tuple[int, str]:
    """(side, checkpoint SHA-256) of `label` in a match result."""
    for slot in ("a", "b"):
        if result.get(f"label_{slot}") == label:
            side_a = int(result["side_a"])
            return (side_a if slot == "a" else 3 - side_a), result.get(f"checkpoint_sha256_{slot}")
    raise ValueError(f"no player labelled {label!r} in this result")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--games", type=Path, required=True, help="an extracted match directory")
    ap.add_argument("--player", required=True, help="the reference's label in that match")
    ap.add_argument("--checkpoint", type=Path, default=None, help="default: the reference's (configs)")
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    import torch
    from tools import reference_player
    from tools.eval_players import _load_policy
    from tools.game_record import read_records
    from tools.raw_player import RawPolicyPlayer
    from wesnoth_ai import __version__
    from wesnoth_ai.critic import file_sha256
    ref = reference_player.load()
    checkpoint = args.checkpoint or Path(reference_player.ensure_checkpoint(ref))
    sha = file_sha256(checkpoint)
    if sha != ref["checkpoint_sha256"]:
        raise SystemExit(f"{checkpoint} is not the reference ({ref['label']})")
    decode = ref["decode"]
    if decode["raw_end_turn"] != "joint" or float(decode["raw_temperature"]) != 0.0:
        raise SystemExit(f"the reference's decode {decode} is not the argmax with the joint end_turn this reads")
    policy = _load_policy(checkpoint, torch.device(args.device), label=ref["label"])
    player = RawPolicyPlayer(policy, 0.0, end_turn_offset=float(decode["raw_end_turn_offset"]),
                             memory_slots=int(ref["memory_slots"]))
    games = pick_games(args.games, args.count, args.seed)
    rows_path = args.out.with_suffix("").with_suffix(".games.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    readings: List[Dict] = []
    totals = {"decisions": 0, "agree": 0, "premoves": 0, "games": 0, "skipped": {}}
    t0 = time.time()
    for path in games:
        result = json.loads(Path(str(path).replace(".game.jsonl.gz", ".json")).read_text(encoding="utf-8"))
        side, played = player_side(result, args.player)
        [rec] = list(read_records(path))
        memory = rec["players"]["a" if rec["players"]["a"]["label"] == args.player else "b"].get("memory")
        if played != sha or int(memory or 0) != int(ref["memory_slots"]):
            totals["skipped"][path.name] = f"played by {played} with {memory} slots"
            continue
        game_readings, tally = read_game(rec, player, side, rec["game_label"])
        readings.extend(game_readings)
        totals["games"] += 1
        for key in ("decisions", "agree", "premoves"):
            totals[key] += tally[key]
        with rows_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"game": path.name, "side": side, **tally,
                                "readings": game_readings}) + "\n")
        log.info("%s: %d decisions, %d agree, %.0f s so far", path.name, tally["decisions"], tally["agree"],
                 time.time() - t0)
    summary = {"summary": summarize(readings), "replay": totals,
               "agreement": totals["agree"] / max(1, totals["decisions"]),
               "provenance": {"checkpoint": str(checkpoint), "sha256": sha, "decode": decode,
                              "memory_slots": ref["memory_slots"], "games_dir": str(args.games),
                              "player": args.player, "seed": args.seed, "count": args.count,
                              "code_version": __version__, "games": [p.name for p in games]},
               "seconds": round(time.time() - t0, 1)}
    args.out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log.info("PRIOR_GAPS_DONE %s", json.dumps(summary["summary"])[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
