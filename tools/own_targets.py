"""The memory player's own trajectories for the sequence trainer
(docs/memory_own_trajectories_prereg_20261011.md), step two of two: which
candidates of tools/own_trajectories.py were decisions, and the teacher's
distribution at each. The records it writes sit beside the corpus's and the
sequence trainer reads them with `--own-sequences`.

A recorded command is one decision of the player, one forward of its
network, but for one case: the simulator turns an attack on a unit the
attacker does not stand next to into its approach move and the attack
(`wesnoth_sim.attack_hex`), so a move followed by an attack from its end is
one attack decision, or a move decision then an attack decision. The teacher
at the match player's slots, its memory carried along the game-side from
decision to decision as in play, tells them apart: the pair is one decision
when the teacher puts the attack above the move at the move's position (the
player's argmax took one of the two, and the end_turn offset scales both
alike; on a tie the attack, which comes first in the reference order). Each
game-side is then checked against its match's count of the player's
forwards: its decisions plus its refused recruits (each decided again from
the memory before it, tools/raw_player.py) must make that count, else the
game-side is dropped and counted.

At every decision the teacher at 0 slots gives the target: its raw
distribution over the legal actions (no end_turn offset), its 16 most
probable actions renormalised (`own_loss.top_target`). Both readings are the
match's own forward: the inference server's batched path
(`InferenceServer.infer_batch`, each position with its masks and memory),
bf16 and the packed trunk on CUDA, as tools/eval_inference_server.py serves
a match; game-sides run side by side.

Readings counted on the way: how often the teacher at the player's slots,
decoding at the player's offset, picks the recorded decision (`repro`), and
how often its choice at the reference offset agrees with the target's
(`agree`), each also from turn 6 on (`_late`).

    OUT/<name>.seq.zpkl         one GameSequence, its own sides' positions
    OUT/sequence_manifest.json  what the trainer reads, and the counts

    python tools/own_targets.py --candidates own_candidates --teacher CKPT --out own_sequences
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from tools.own_trajectories import CANDIDATE_SUFFIX, read_candidates_manifest  # noqa: E402
from tools.preencode_sequences import (MANIFEST_NAME, read_record, record_path,  # noqa: E402
                                       write_record)
from wesnoth_ai.memory import MemoryState  # noqa: E402
from wesnoth_ai.own_loss import (TOP_ACTIONS, decode_key, label_key, target_key,  # noqa: E402
                                 top_target)
from wesnoth_ai.sequence_records import GameSequence, OwnCandidateGame, SequencePosition  # noqa: E402
from wesnoth_ai.server_priors import KIND_ATTACK, KIND_END_TURN  # noqa: E402

log = logging.getLogger("own_targets")

# Game-sides stepped side by side (two leaves each: the player's slots and 0).
WALK_BATCH = 256
LOADER_THREADS = 4
LOAD_AHEAD = 16
# The turn from which the agreement readings are also reported (the
# memory-in-play readout's "turns 6 and later").
LATE_TURN = 6
# Counts that are a game's facts, not summed over games.
GAME_FACTS = ("winner", "n_commands")


def teacher_server(spec: Path, device: torch.device):
    """The inference server a match serves `spec` with
    (tools/eval_inference_server.py): bf16 and the packed trunk on CUDA;
    and the model's memory slots."""
    from tools.eval_players import _load_policy
    from tools.inference_seam import InferenceServer
    from wesnoth_ai.packed_trunk import check_packed_trunk_supported, flash_varlen_applies
    cuda = device.type == "cuda"
    policy = _load_policy(spec, device, label="own_targets", infer_bf16=cuda, infer_compile=False)
    model, encoder = policy._inference_model, policy._inference_encoder
    model.eval()
    encoder.eval()
    packed = cuda and flash_varlen_applies(device, torch.bfloat16)
    if packed:
        check_packed_trunk_supported(model.encoder)
        model.infer_packed_trunk = True
    server = InferenceServer(model, encoder, device=device, output_device=torch.device("cpu"),
                             autocast_bf16=cuda, packed_embed=packed)
    return server, int(getattr(model, "memory_slots", 0) or 0)


# ---------------------------------------------------------------------------
# Game-sides walked in order, their memory carried
# ---------------------------------------------------------------------------

class SideWalk:
    """One game-side walked in order: its positions, the next one's index,
    and the memory its last decision wrote (None before its first)."""

    def __init__(self, key, positions: List[SequencePosition], payload=None):
        self.key, self.positions, self.payload = key, positions, payload
        self.t = 0
        self.state: Optional[torch.Tensor] = None
        self.kept: List[SequencePosition] = []
        self.counts: Counter = Counter()

    @property
    def done(self) -> bool:
        return self.t >= len(self.positions)


def walk_sides(server, walks: Iterable[SideWalk], slots: int, on_position: Callable,
               zero_slots: bool, batch: int = WALK_BATCH) -> None:
    """Each walk's positions in order, `batch` walks side by side (a
    finished one's row taken by the next): per position one forward at
    `slots` with the walk's memory, and one at 0 slots when `zero_slots`;
    then `on_position(walk, output, zero_output)` returns how many
    positions the walk moves on by, and the walk's memory becomes what the
    first forward wrote. `on_position(walk, None, None)` is called once when
    a walk ends."""
    queue: Iterator[SideWalk] = iter(walks)
    running: List[SideWalk] = []
    step = 2 if zero_slots else 1
    while True:
        while len(running) < batch:
            w = next(queue, None)
            if w is None:
                break
            if w.done:
                on_position(w, None, None)
            else:
                running.append(w)
        if not running:
            return
        leaves = []
        for w in running:
            p = w.positions[w.t]
            leaves.append((p.raw, p.masks, MemoryState(slots, w.state)))
            if zero_slots:
                leaves.append((p.raw, p.masks, MemoryState(0, None)))
        outs = server.infer_batch(leaves)
        for i, w in enumerate(running):
            out = outs[i * step]
            w.t += on_position(w, out, outs[i * step + 1] if zero_slots else None)
            w.state = out.memory
            if w.done:
                on_position(w, None, None)
        running = [w for w in running if not w.done]


def compact_key_prior(compact, key) -> float:
    """The prior of the action `key` (actor, kind, target, weapon) among a
    position's compact legal actions; 0 when it is not legal or names no
    action (None)."""
    if key is None:
        return 0.0
    actor, kind, target, weapon = key
    hit = (compact.actor == actor) & (compact.kind == kind)
    if kind != KIND_END_TURN:
        hit &= compact.target == target
    if kind == KIND_ATTACK:
        hit &= compact.weapon == weapon
    return float(np.asarray(compact.prior)[hit].sum())


def end_actor_of(raw) -> int:
    return len(raw.unit_positions) + len(raw.recruit_types)


def decode_of(compact, offset: float):
    return decode_key(compact.actor, compact.kind, compact.target, compact.weapon, compact.prior, offset)


# ---------------------------------------------------------------------------
# The targets
# ---------------------------------------------------------------------------

class OwnLabeller:
    """`on_position` for the candidates' walk (module docstring): keeps
    each decision with its target, checks a game-side's forward count when
    it ends, and writes a game's record once all its sides have ended."""

    def __init__(self, out_dir: Path, reference_offset: float, top: int = TOP_ACTIONS,
                 on_game: Optional[Callable[[str, Dict[str, int]], None]] = None):
        self.out_dir = Path(out_dir)
        self.reference_offset = float(reference_offset)
        self.top = int(top)
        self.on_game = on_game
        self.games: Dict[str, Tuple[OwnCandidateGame, List[SideWalk]]] = {}
        self.totals: Counter = Counter()

    def walks(self, games: Iterable[OwnCandidateGame]) -> Iterator[SideWalk]:
        for game in games:
            self.games[game.file] = (game, [])
            for side in sorted(game.sides):
                positions = [c.position for c in game.sides[side].candidates]
                yield SideWalk(game.file, positions, payload=game.sides[side])

    def __call__(self, w: SideWalk, out, zero) -> int:
        if out is None:
            self._finish(w)
            return 0
        cand = w.payload.candidates[w.t]
        pos = cand.position
        merged = False
        if cand.attack_label is not None:
            p_attack = compact_key_prior(out.legal_compact, label_key(cand.attack_label))
            p_move = compact_key_prior(out.legal_compact, label_key(pos.label))
            merged = p_attack > 0.0 and p_attack >= p_move
            w.counts["approach_pairs"] += 1
            w.counts["approach_merged"] += int(merged)
            w.counts["approach_neither_legal"] += int(p_attack == 0.0 and p_move == 0.0)
        label = cand.attack_label if merged else pos.label
        z = zero.legal_compact
        target = top_target(z.actor, z.kind, z.target, z.weapon, z.prior, self.top)
        late = pos.turn >= LATE_TURN
        key = label_key(label)
        repro = key is not None and decode_of(out.legal_compact, w.payload.offset) == key
        agree = decode_of(out.legal_compact, self.reference_offset) == \
            target_key(target, self.reference_offset, end_actor_of(pos.raw))
        for tag, hit in (("repro", repro), ("agree", agree), ("decisions", True)):
            w.counts[tag] += int(hit)
            w.counts[f"{tag}_late"] += int(hit and late)
        w.kept.append(dataclasses.replace(pos, label=label, own_target=target))
        return 2 if merged else 1

    def _finish(self, w: SideWalk) -> None:
        side = w.payload
        made = len(w.kept) + side.rejections
        w.counts["candidates"] = len(side.candidates)
        if side.forwards is None:
            w.counts["dropped: no forward count"] = 1
        elif made != side.forwards:
            w.counts["dropped: forward count mismatch"] = 1
            w.counts["forward_count_off_by"] = abs(made - side.forwards)
        game, ended = self.games[w.key]
        ended.append(w)
        if len(ended) == len(game.sides):
            del self.games[w.key]
            counts = write_game(self.out_dir, game, ended)
            self.totals.update({k: v for k, v in counts.items() if k not in GAME_FACTS})
            if self.on_game is not None:
                self.on_game(game.file, counts)


def write_game(out_dir: Path, game: OwnCandidateGame, walks: List[SideWalk]) -> Dict[str, int]:
    """The game's own record, its kept game-sides' positions, and its
    counts; every game gets a record, so a resumed run skips it."""
    counts: Counter = Counter()
    seq = GameSequence(file=game.file, winner=game.winner, n_commands=game.n_commands, sides={})
    for w in walks:
        s = w.payload.side
        counts.update(w.counts)
        counts["game_sides"] += 1
        if any(k.startswith("dropped: ") for k in w.counts):
            counts["positions_dropped"] += len(w.kept)
            continue
        seq.sides[s] = w.kept
        counts[f"positions_side{s}"] = len(w.kept)
        counts["positions"] += len(w.kept)
        counts["game_sides_kept"] += 1
    seq.counts = {**{k: int(v) for k, v in counts.items()}, "winner": int(game.winner),
                  "n_commands": int(game.n_commands)}
    write_record(record_path(out_dir, game.file), seq)
    return seq.counts


def label_games(server, slots: int, games: Iterable[OwnCandidateGame], out_dir: Path,
                reference_offset: float, top: int = TOP_ACTIONS, batch: int = WALK_BATCH,
                on_game: Optional[Callable[[str, Dict[str, int]], None]] = None) -> Counter:
    """Walks every own game-side of `games`, writing each game's record once
    its sides have ended; the counts summed over the games."""
    labeller = OwnLabeller(out_dir, reference_offset, top, on_game)
    walk_sides(server, labeller.walks(games), slots, labeller, zero_slots=True, batch=batch)
    return labeller.totals


def candidate_games(cand_dir: Path, names: List[str]) -> Iterator[OwnCandidateGame]:
    """The candidate games in order, read ahead on threads."""
    def load(name: str) -> OwnCandidateGame:
        return read_record(Path(cand_dir) / (name + CANDIDATE_SUFFIX))

    with ThreadPoolExecutor(LOADER_THREADS) as pool:
        ahead = [pool.submit(load, n) for n in names[:LOAD_AHEAD]]
        for i in range(len(names)):
            if i + LOAD_AHEAD < len(names):
                ahead.append(pool.submit(load, names[i + LOAD_AHEAD]))
            yield ahead[i].result()
            ahead[i] = None


def own_manifest(cand: dict, games: Dict[str, Dict[str, int]], top: int, reference_offset: float) -> dict:
    """What the sequence trainer reads of the own records
    (`preencode_sequences.load_manifest`), and the counts."""
    totals: Counter = Counter()
    for c in games.values():
        totals.update({k: v for k, v in c.items() if k not in GAME_FACTS})
    rows = {name: {**{k: v for k, v in c.items() if k.startswith("positions_side")},
                   "holdout": bool(cand["games"][name]["holdout"]),
                   "n_commands": int(c["n_commands"]), "winner": int(c["winner"])}
            for name, c in games.items()}
    keys = ("fingerprint", "encoding", "observation_epoch", "core_phase", "unit_type_to_id", "faction_to_id",
            "teacher", "teacher_sha256", "slots", "selection")
    return {**{k: cand[k] for k in keys}, "source": "own", "top": top, "reference_offset": reference_offset,
            "n_games": len(games), "totals": dict(totals), "errors": cand["errors"], "games": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--candidates", type=Path, required=True, help="tools/own_trajectories.py's output")
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference-offset", type=float, default=None,
                    help="the decode's end_turn offset the agreement reading uses "
                         "(default: configs/reference_player.json's)")
    ap.add_argument("--top", type=int, default=TOP_ACTIONS)
    ap.add_argument("--batch", type=int, default=WALK_BATCH)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from tools.eval_provenance import file_sha256
    cand = read_candidates_manifest(args.candidates)
    if file_sha256(args.teacher) != cand["teacher_sha256"]:
        raise SystemExit(f"{args.teacher} is not the checkpoint the candidates were selected for")
    offset = args.reference_offset
    if offset is None:
        offset = float(json.loads(Path("configs/reference_player.json").read_text(encoding="utf-8"))
                       ["decode"]["raw_end_turn_offset"])
    server, slots = teacher_server(args.teacher, torch.device(args.device))
    if slots != int(cand["slots"]):
        raise SystemExit(f"{args.teacher} has {slots} memory slots; the candidates were played at {cand['slots']}")
    args.out.mkdir(parents=True, exist_ok=True)
    written = [n for n in cand["games"] if record_path(args.out, n).exists()]
    games: Dict[str, Dict[str, int]] = {n: read_record(record_path(args.out, n)).counts for n in written}
    todo = sorted(n for n, info in cand["games"].items()
                  if info.get("counts") and n not in games and n not in cand["errors"])
    t0, n_positions = time.time(), [0]

    def flush() -> None:
        tmp = args.out / (MANIFEST_NAME + ".tmp")
        tmp.write_text(json.dumps(own_manifest(cand, games, args.top, offset), indent=0), encoding="utf-8")
        os.replace(tmp, args.out / MANIFEST_NAME)

    def on_game(name: str, counts: Dict[str, int]) -> None:
        games[name] = counts
        n_positions[0] += counts.get("positions", 0)
        if len(games) % 100 == 0:
            el = time.time() - t0
            log.info("%d/%d games, %d positions, %.0f s, %.1f positions/s", len(games), len(cand["games"]),
                     n_positions[0], el, n_positions[0] / max(el, 1e-9))
            flush()

    log.info("labelling %d own games (%d already written) with %s at %d slots, batch %d", len(todo),
             len(written), args.teacher, slots, args.batch)
    with torch.no_grad():
        label_games(server, slots, candidate_games(args.candidates, todo), args.out, offset, args.top,
                    args.batch, on_game)
    flush()
    totals = own_manifest(cand, games, args.top, offset)["totals"]
    log.info("OWN_TARGETS_DONE %d games, %d positions in %.0f s: %s", len(games), totals.get("positions", 0),
             time.time() - t0, json.dumps(totals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
