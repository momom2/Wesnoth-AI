#!/usr/bin/env python3
"""What the memory changes in a memory checkpoint's decisions
(docs/memory_in_play_parity3_prereg_20261009.md): along a game-side's
decisions, the checkpoint with its memory carried at K slots against the
same checkpoint at 0 slots, on the trajectories a match player produced and
on human corpus games.

One row per decision: the game, the side, the turn and the decision's index
in the side-turn, the recorded action's kind, the mover's standing (its HP
margin over the two player sides, `hp_margin`, and its bucket, `standing`)
and the side's eventual outcome (`outcome`: won, lost, or capped for a game
without a winner); and for each of the two players (`m`: K slots carried,
`z`: 0 slots) the end_turn prior, the largest prior of any other action, the
entropy of the other actions' priors, the action the decode chooses (the
joint argmax after the end_turn logit offset, `tools/raw_player.py`) and the
recorded action's prior. Then whether the two choices agree, the memory
state's mean absolute value after the decision, and, on a match trajectory,
whether the choice of the reading that played reproduces the recorded
command (`repro`).

A match decision is one forward of the player that made it: an attack on a
unit the attacker does not stand next to is recorded as a move and an
attack, and a refused recruit leaves no command (its memory update was
undone in play, tools/raw_player.py). A human decision is a corpus pair
(`replay_dataset.iter_record_pairs`, the sequence trainer's positions),
timeouts included: the memory reads them, the policy has no label there.

    python tools/analysis/memory_counterfactual.py --checkpoint CKPT --slots 64 \\
        --games DIR --player parity3 --offset -1.5 --out rows.jsonl
    python tools/analysis/memory_counterfactual.py --checkpoint CKPT --slots 64 \\
        --corpus DATASET --holdout --offset -1.5 --out human.jsonl

`--procs N` runs N shards of the games as processes of their own and joins
their rows into OUT; it fails when any shard fails.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from tools.raw_player import pick_index, shift_end_turn_prior  # noqa: E402
from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors  # noqa: E402
from wesnoth_ai.classes import PLAYER_SIDES, opponent_of  # noqa: E402
from wesnoth_ai.memory import MemoryState  # noqa: E402

log = logging.getLogger("memory_counterfactual")

PLAYER_COMMANDS = ("move", "attack", "recruit", "end_turn")
# The mover's standing: behind when its HP margin over the opponent is below
# -STANDING_MARGIN of the two player sides' total HP on the board, ahead
# above +STANDING_MARGIN, level otherwise (the pre-registration's W readings).
STANDING_MARGIN = 0.10


# ---------------------------------------------------------------------------
# Actions as comparable keys
# ---------------------------------------------------------------------------

def action_key(action: Dict) -> Tuple:
    """A legal action dict as a key a recorded command or a label can match."""
    kind = action["type"]
    if kind == "end_turn":
        return ("end_turn",)
    if kind == "recruit":
        hex_ = action["target_hex"]
        return ("recruit", action["unit_type"], hex_.x, hex_.y)
    start, target = action["start_hex"], action["target_hex"]
    if kind == "attack":
        return ("attack", start.x, start.y, target.x, target.y, int(action.get("attack_index", 0)))
    return ("move", start.x, start.y, target.x, target.y)


def command_key(cmd: List) -> Tuple:
    """A match record's command as the key of the action that produced it."""
    kind = cmd[0]
    if kind == "end_turn":
        return ("end_turn",)
    if kind == "recruit":
        return ("recruit", cmd[1], int(cmd[2]), int(cmd[3]))
    if kind == "attack":
        return ("attack", int(cmd[1]), int(cmd[2]), int(cmd[3]), int(cmd[4]), int(cmd[5]))
    xs, ys = cmd[1], cmd[2]
    return ("move", int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1]))


def label_key(ai) -> Optional[Tuple]:
    """A corpus label as an action key; None for a timeout."""
    kind = ai.action_type
    if kind == "end_turn":
        return ("end_turn",)
    if kind == "recruit":
        return ("recruit", ai.recruit_type, *ai.target_hex)
    if kind == "attack":
        return ("attack", *ai.source_hex, *ai.target_hex, int(ai.weapon_idx or 0))
    if kind == "move":
        return ("move", *ai.source_hex, *ai.target_hex)
    return None


# ---------------------------------------------------------------------------
# The mover's standing and the side's outcome
# ---------------------------------------------------------------------------

def hp_margin(gs, side: int) -> float:
    """(the side's HP - the opponent's HP) / both player sides' HP, over the
    units on the board in the state of record (seen or not by the mover);
    0 when neither has any."""
    hp = {s: 0 for s in PLAYER_SIDES}
    for u in gs.map.units:
        if u.side in hp:
            hp[u.side] += int(u.current_hp)
    total = sum(hp.values())
    return (hp[side] - hp[opponent_of(side)]) / total if total else 0.0


def standing_of(margin: float) -> str:
    if margin < -STANDING_MARGIN:
        return "behind"
    return "ahead" if margin > STANDING_MARGIN else "level"


def outcome_of(side: int, winner: Optional[int]) -> str:
    """won, lost, or capped for a game that ended without a winner."""
    if winner not in PLAYER_SIDES:
        return "capped"
    return "won" if int(winner) == int(side) else "lost"


# ---------------------------------------------------------------------------
# One decision, two players
# ---------------------------------------------------------------------------

class Decision:
    """One player's reading of a position: the legal actions, their priors,
    and the decode's choice."""

    def __init__(self, legal, offset: float):
        self.legal = legal
        self.keys = [action_key(la.action) for la in legal]
        self.priors = np.array([la.prior for la in legal], dtype=np.float64)
        self.is_end = np.array([la.action["type"] == "end_turn" for la in legal], dtype=bool)
        self.choice = pick_index(shift_end_turn_prior(self.priors, self.is_end, offset), 0.0, None) \
            if len(legal) else None

    def summary(self, rec_key: Optional[Tuple]) -> Dict:
        act = self.priors[~self.is_end]
        mass = act.sum()
        q = act[act > 0] / mass if mass > 0 else np.zeros(0)
        return {"p_end": float(self.priors[self.is_end].sum()),
                "max_act": float(act.max()) if act.size else 0.0,
                "ent_act": float(-(q * np.log(q)).sum()) if q.size else 0.0,
                "choice": None if self.choice is None else self.legal[self.choice].action["type"],
                "rec_prior": self._prior_of(rec_key)}

    def _prior_of(self, key: Optional[Tuple]) -> Optional[float]:
        if key is None:
            return None
        try:
            return float(self.priors[self.keys.index(key)])
        except ValueError:
            return None

    @property
    def chosen(self) -> Optional[Dict]:
        return None if self.choice is None else self.legal[self.choice].action


class Pair:
    """The two players over one game-side: the memory player's state is
    carried from each of its decisions to the next."""

    def __init__(self, policy, slots: int, offset: float):
        self.policy, self.slots, self.offset = policy, int(slots), float(offset)
        self.state: Optional[torch.Tensor] = None

    def decide(self, gs) -> Tuple[Decision, Decision]:
        encoded = self.policy._inference_encoder.encode(gs)
        model = self.policy._inference_model
        with torch.no_grad():
            out_m = model(encoded, memory=MemoryState(self.slots, self.state))
            legal_m = enumerate_legal_actions_with_priors(encoded, out_m, gs, decision_step=0)
            out_z = model(encoded, memory=MemoryState(0, None))
            legal_z = enumerate_legal_actions_with_priors(encoded, out_z, gs, decision_step=0)
        if out_m.memory is None:
            raise RuntimeError("the checkpoint returned no memory: it has none")
        self.state = out_m.memory.detach().float()
        m, z = Decision(legal_m, self.offset), Decision(legal_z, self.offset)
        if m.keys != z.keys:
            raise RuntimeError("the two players enumerate different legal actions at one position")
        return m, z

    def memory_abs(self) -> float:
        return float(self.state.abs().mean()) if self.state is not None and self.state.numel() else 0.0


def row_of(base: Dict, gs, m: Decision, z: Decision, rec_key: Optional[Tuple], pair: Pair) -> Dict:
    row = dict(base)
    margin = hp_margin(gs, int(base["side"]))
    row["hp_margin"] = round(margin, 5)
    row["standing"] = standing_of(margin)
    for tag, d in (("m", m), ("z", z)):
        for name, value in d.summary(rec_key).items():
            row[f"{name}_{tag}"] = value
    row["agree"] = m.choice == z.choice
    row["mem_abs"] = pair.memory_abs()
    return row


# ---------------------------------------------------------------------------
# Match trajectories
# ---------------------------------------------------------------------------

def match_rows(policy, rec: Dict, player: str, slots: int, offset: float) -> Iterator[Dict]:
    """The rows of one match game for the side `player` played. The
    recorded commands are segmented, and `repro` read, by the choices of
    the reading that played: the memory player's when `player` played with
    `slots` slots, the 0-slot player's when it played without a memory."""
    from tools.game_record import _walk_core
    from wesnoth_ai.game_core import bind_view
    entry = next(p for p in rec["players"].values() if p["label"] == player)
    side, played = int(entry["side"]), int(entry.get("memory") or 0)
    if played not in (0, slots):
        raise SystemExit(f"{player} played with {played} slots, not {slots} or 0")
    outcome = outcome_of(side, rec.get("winner"))
    pair = Pair(policy, slots, offset)
    commands = rec["commands"]
    skip: set = set()
    turn = t = 0
    for i, cs, cmd in _walk_core(rec, verify=True):
        if cmd[0] == "init_side":
            if int(cmd[1]) == side:
                turn, t = turn + 1, 0
            continue
        if i in skip or cmd[0] not in PLAYER_COMMANDS or int(cs.core.current_side) != side:
            continue
        snapshot = cs.fork()
        gs = snapshot.to_state()
        bind_view(gs, snapshot)
        m, z = pair.decide(gs)
        own = m if played else z
        rec_key, kind = _recorded(commands, i, own.chosen, skip)
        row = row_of({"src": "match", "game": rec["game_label"], "player": player, "side": side,
                      "played": played, "outcome": outcome, "turn": turn, "t": t, "rec": kind},
                     gs, m, z, rec_key, pair)
        row["repro"] = own.chosen is not None and action_key(own.chosen) == rec_key
        yield row
        t += 1


def _recorded(commands: List, i: int, chosen: Optional[Dict], skip: set) -> Tuple[Tuple, str]:
    """The key and kind of the decision recorded from command i: a move
    followed by an attack from the move's end on the target the player
    chose is one attack decision (the attacker walked first), and its
    second command is skipped."""
    cmd = commands[i]
    key = command_key(cmd)
    if cmd[0] != "move" or chosen is None or chosen["type"] != "attack":
        return key, cmd[0]
    start, target = chosen["start_hex"], chosen["target_hex"]
    if (start.x, start.y) != (key[1], key[2]):
        return key, cmd[0]
    nxt = commands[i + 1] if i + 1 < len(commands) else None
    if nxt is not None and nxt[0] == "attack" and (int(nxt[1]), int(nxt[2])) == (key[3], key[4]) \
            and (int(nxt[3]), int(nxt[4])) == (target.x, target.y):
        skip.add(i + 1)
        return ("attack", key[1], key[2], target.x, target.y, int(nxt[5])), "attack"
    return key, cmd[0]


def match_games(directory: Path) -> List[Path]:
    return sorted(Path(directory).glob("*.game.jsonl.gz"))


# ---------------------------------------------------------------------------
# Human trajectories
# ---------------------------------------------------------------------------

def human_rows(policy, data: Dict, name: str, winner: int, slots: int, offset: float) -> Iterator[Dict]:
    """The rows of one corpus game, both player sides, each side's memory
    carried along its own decisions."""
    from tools.replay_dataset import iter_record_pairs
    pairs: Dict[int, Pair] = {}
    counts: Dict[Tuple[int, int], int] = {}
    for gs, ai in iter_record_pairs(data, relevant_set=False, timeouts=True):
        side, turn = int(gs.global_info.current_side), int(gs.global_info.turn_number)
        pair = pairs.setdefault(side, Pair(policy, slots, offset))
        t = counts.get((side, turn), 0)
        counts[(side, turn)] = t + 1
        m, z = pair.decide(gs)
        yield row_of({"src": "human", "game": name, "player": "winner" if side == winner else "loser",
                      "side": side, "outcome": outcome_of(side, winner), "turn": turn, "t": t,
                      "rec": ai.action_type}, gs, m, z, label_key(ai), pair)


def corpus_games(dataset: Path, holdout: bool) -> List[Tuple[Path, int]]:
    out = []
    with open(Path(dataset) / "manifest.jsonl", encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line)
            if holdout and not entry.get("holdout"):
                continue
            out.append((Path(dataset) / entry["file"], int(entry["winner_side"])))
    return out


# ---------------------------------------------------------------------------

def load_policy(checkpoint: Path, device: torch.device, bf16: bool):
    from tools.eval_players import _load_policy
    policy = _load_policy(checkpoint, device, label="counterfactual", infer_bf16=bf16)
    slots = int(getattr(policy._inference_model, "memory_slots", 0) or 0)
    if not slots:
        raise SystemExit(f"{checkpoint} has no memory")
    return policy, slots


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--slots", type=int, default=None, help="the memory player's slots (default: all)")
    ap.add_argument("--offset", type=float, required=True, help="the decode's end_turn logit offset")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--games", type=Path, help="a match directory of game records")
    src.add_argument("--corpus", type=Path, help="an imitation dataset directory")
    ap.add_argument("--player", help="with --games: the label whose decisions are read")
    ap.add_argument("--holdout", action="store_true", help="with --corpus: holdout games only")
    ap.add_argument("--limit", type=int, default=None, help="the first N games")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--procs", type=int, default=1, help="shards run as processes, joined into --out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32", action="store_true", help="no bf16 autocast on CUDA")
    ap.add_argument("--out", type=Path, required=True)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = ap.parse_args(raw_argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.games is not None and not args.player:
        ap.error("--games needs --player")
    if args.procs > 1:
        return run_shards(raw_argv, args.procs, args.out)
    if args.shards > 1:
        torch.set_num_threads(2)                 # one of several processes sharing the cores
    device = torch.device(args.device)
    policy, slots = load_policy(args.checkpoint, device, bf16=device.type == "cuda" and not args.fp32)
    k = slots if args.slots is None else args.slots
    if not 0 < k <= slots:
        ap.error(f"--slots must be in 1..{slots}")
    if args.games is not None:
        jobs = [(p, None) for p in match_games(args.games)]
    else:
        jobs = corpus_games(args.corpus, args.holdout)
    jobs = jobs[:args.limit] if args.limit else jobs
    jobs = jobs[args.shard::args.shards]
    t0, n_rows = time.time(), 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for j, (path, winner) in enumerate(jobs):
            data = json.loads(gzip.open(path, "rt", encoding="utf-8").read())
            rows = (match_rows(policy, data, args.player, k, args.offset) if args.games is not None
                    else human_rows(policy, data, path.name, winner, k, args.offset))
            repro = []
            for row in rows:
                fh.write(json.dumps(row) + "\n")
                n_rows += 1
                if "repro" in row:
                    repro.append(row["repro"])
            fh.flush()
            log.info("%d/%d %s: %s, %.0f s", j + 1, len(jobs), path.name,
                     f"reproduced {sum(repro)}/{len(repro)}" if repro else "done", time.time() - t0)
    log.info("%d rows in %.0f s", n_rows, time.time() - t0)
    return 0


def run_shards(argv: List[str], procs: int, out: Path) -> int:
    """`procs` processes of this tool, shard i of `procs` each, their rows
    joined into `out` in shard order once every one has exited 0."""
    base = _without(argv, ("--procs", "--out", "--shard", "--shards"))
    parts = [out.with_name(f"{out.name}.part{i}") for i in range(procs)]
    children = [subprocess.Popen([sys.executable, __file__, *base, "--shard", str(i), "--shards", str(procs),
                                  "--out", str(part)]) for i, part in enumerate(parts)]
    codes = [child.wait() for child in children]
    if any(codes):
        log.error("shards exited %s", codes)
        return 1
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as fh:
        for part in parts:
            fh.write(part.read_bytes())
    tmp.replace(out)
    for part in parts:
        part.unlink()
    return 0


def _without(argv: List[str], flags: Tuple[str, ...]) -> List[str]:
    """argv less each of `flags` and its value (as `--flag value` or `--flag=value`)."""
    out, skip = [], False
    for arg in argv:
        if skip:
            skip = False
        elif arg in flags:
            skip = True
        elif not any(arg.startswith(f + "=") for f in flags):
            out.append(arg)
    return out


if __name__ == "__main__":
    sys.exit(main())
