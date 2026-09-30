"""Pre-encode the corpus for the sequence trainer
(docs/parity_memory_design_20260929.md "Training"): every decision of both
player sides of every manifest game, in order, under the parity observation,
with its label and its belief targets.

    OUT/<game>.seq.zpkl         one GameSequence (zlib pickle)
    OUT/sequence_manifest.json  the fingerprint and the counts

A decision is a position the player acted from: its encoding (the
observation that carries the true state is left out, so the truth never
enters an input), its label (`timeout` for a turn that ran out of time,
which names no action), and the belief head's targets (the hex tokens where
an enemy unit the side cannot see stands, and the tokens with no visible
unit, the loss's domain). The engine's own moves are applied and are no
decision. The encoding is the recipe's: relevant set version 2, the
enemy-village fog gate, the terrain set, `observation_parity`, through the
Rust core. The vocabulary is a fresh network's (`tools/unit_vocab.seed_vocab`),
or a checkpoint's with `--vocab-from`.

Idempotent: a game whose record exists is skipped, so a killed pass
continues; progress and the manifest are written on the run.

    python tools/preencode_sequences.py --dataset replays_dataset_imitation \\
        --out replays_dataset_sequences --workers 60
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import logging
import multiprocessing as mp
import os
import pickle
import sys
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from wesnoth_ai import unpickle  # noqa: E402
from wesnoth_ai.constants import OBSERVATION_EPOCH  # noqa: E402
from wesnoth_ai.paths import IMITATION_DATASET_DIR  # noqa: E402

log = logging.getLogger("preencode_sequences")

RECORD_SUFFIX = ".seq.zpkl"
MANIFEST_NAME = "sequence_manifest.json"
PLAYER_SIDES = (1, 2)

# The recipe's encoding (docs/parity_memory_design_20260929.md "Model interface").
ENCODING = {"relevant_set": True, "fog_hides_enemy_villages": True, "terrain_multi_hot": True,
            "observation_parity": True, "relevant_set_version": 2}


@dataclass
class SequencePosition:
    raw: object                     # RawEncoded, observation left out
    label: object                   # ActionIndices; action_type "timeout" names no action
    hidden_tokens: np.ndarray       # int64 [K]: hex tokens holding an enemy unit the side cannot see
    no_visible_unit: np.ndarray     # bool [H]: hex tokens with no visible unit


@dataclass
class GameSequence:
    file: str
    winner: int                     # the winning side
    n_commands: int
    sides: Dict[int, List[SequencePosition]] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)


def encoding_fingerprint(type_to_id: Dict[str, int], faction_to_id: Dict[str, int],
                         corpus_version: int, core_phase: int) -> str:
    """Everything the records depend on: the vocabularies, the encoding's
    switches, the observation epoch, the corpus's version (its labels and
    cuts) and the Rust core's phase (the rules that built each position)."""
    h = hashlib.sha1()
    h.update(json.dumps(sorted(type_to_id.items())).encode("utf-8"))
    h.update(json.dumps(sorted(faction_to_id.items())).encode("utf-8"))
    h.update(json.dumps(sorted(ENCODING.items())).encode("utf-8"))
    h.update(b"|obs=%d|corpus=%d|core=%d" % (int(OBSERVATION_EPOCH), int(corpus_version), int(core_phase)))
    return h.hexdigest()


def encode_game_sequence(data: dict, file: str, winner: int, type_to_id: Dict[str, int],
                         faction_to_id: Dict[str, int]) -> GameSequence:
    """One extracted game's decisions, per side, in order. A label whose
    slots do not point at what its command names raises
    (`encode_worker.LabelSlotMismatch`), so the game is left out rather
    than trained on a wrong label."""
    from tools.encode_worker import LabelSlotMismatch, label_in_raw_basis, label_slot_mismatch
    from tools.replay_dataset import TIMEOUT, iter_record_pairs
    from wesnoth_ai.belief_targets import belief_targets
    from wesnoth_ai.encoder import encode_raw
    from wesnoth_ai.faction_posterior import posterior_counts
    stats: Counter = Counter()
    before = posterior_counts()
    seq = GameSequence(file=file, winner=int(winner), n_commands=len(data.get("commands", [])),
                       sides={s: [] for s in PLAYER_SIDES})
    for gs, ai in iter_record_pairs(data, relevant_set=False, stats=stats, timeouts=True):
        side = int(gs.global_info.current_side)
        raw = encode_raw(gs, type_to_id=type_to_id, faction_to_id=faction_to_id, **ENCODING)
        if ai.action_type == TIMEOUT:
            stats["timeout_positions"] += 1
        else:
            ai = label_in_raw_basis(ai, raw)
            why = label_slot_mismatch(raw, ai)
            if why is not None:
                raise LabelSlotMismatch(f"{file}, side {side} decision {len(seq.sides[side])}: {why}")
            stats["target_off_subset"] += int(bool(ai.target_off_subset))
        bt = belief_targets(raw, side)
        stats["hidden_units"] += int(bt.hidden_tokens.shape[0]) + bt.n_untokened
        stats["hidden_untokened"] += bt.n_untokened
        stats["sighting_tokens"] += int(raw.sight_type_ids.shape[0])
        seq.sides[side].append(SequencePosition(
            raw=dataclasses.replace(raw, observation=None), label=ai,
            hidden_tokens=bt.hidden_tokens, no_visible_unit=bt.no_visible_unit))
    after = posterior_counts()
    stats["posteriors"] += after["posteriors"] - before["posteriors"]
    stats["posterior_errors"] += after["inconsistent"] - before["inconsistent"]
    stats["positions"] = sum(len(v) for v in seq.sides.values())
    stats["n_commands"] = seq.n_commands
    for s, positions in seq.sides.items():
        stats[f"positions_side{s}"] = len(positions)
    seq.counts = {k: int(v) for k, v in stats.items()}
    return seq


def record_path(out_dir: Path, file: str) -> Path:
    return Path(out_dir) / (file + RECORD_SUFFIX)


def write_record(path: Path, seq: GameSequence) -> None:
    """Atomic: one pickle, zlib level 1."""
    blob = zlib.compress(pickle.dumps(seq, protocol=pickle.HIGHEST_PROTOCOL), 1)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, path)


def read_record(path: Path) -> GameSequence:
    return unpickle.loads(zlib.decompress(Path(path).read_bytes()))


def load_manifest(out_dir: Path) -> Optional[Dict]:
    p = Path(out_dir) / MANIFEST_NAME
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fresh_vocab() -> Tuple[Dict[str, int], Dict[str, int]]:
    """A fresh network's vocabularies (`unit_vocab.seed_vocab`)."""
    from tools.unit_vocab import seed_vocab
    from wesnoth_ai.encoder import GameStateEncoder
    enc = GameStateEncoder(d_model=8, relevant_set_hexes=True, fog_hides_enemy_villages=True,
                           terrain_multi_hot=True, observation_parity=True, relevant_set_version=2)
    seed_vocab(enc)
    return dict(enc.unit_type_to_id), dict(enc.faction_to_id)


_W: Dict = {}


def _worker_init(type_to_id, faction_to_id, dataset, out_dir) -> None:
    _W.update(type_to_id=type_to_id, faction_to_id=faction_to_id, dataset=Path(dataset),
              out_dir=Path(out_dir))


def _worker_encode(row: dict) -> Tuple[str, str, Dict[str, int]]:
    """(file, status, the game's counts)."""
    file = row["file"]
    dst = record_path(_W["out_dir"], file)
    if dst.exists():
        return file, "exists", {}
    try:
        with gzip.open(_W["dataset"] / file, "rt", encoding="utf-8") as f:
            data = json.load(f)
        seq = encode_game_sequence(data, file, int(row["winner_side"]), _W["type_to_id"],
                                   _W["faction_to_id"])
    except Exception as e:  # noqa: BLE001 - one bad game must not stop the pass
        return file, f"error: {type(e).__name__}: {e}"[:300], {}
    write_record(dst, seq)
    return file, "ok", seq.counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", type=Path, default=IMITATION_DATASET_DIR)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vocab-from", type=Path, default=None,
                    help="a checkpoint whose vocabularies the encoding uses (default: a fresh network's)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=None, help="first N manifest games")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    import wesnoth_core
    from tools.preencode_corpus import vocab_from_checkpoint
    from tools.replay_dataset import corpus_version_of
    if args.vocab_from is not None:
        type_to_id, faction_to_id = vocab_from_checkpoint(args.vocab_from)
    else:
        type_to_id, faction_to_id = fresh_vocab()
    corpus = corpus_version_of(args.dataset)
    phase = int(wesnoth_core.__phase__)
    fp = encoding_fingerprint(type_to_id, faction_to_id, corpus, phase)
    args.out.mkdir(parents=True, exist_ok=True)
    existing = load_manifest(args.out) or {}
    if existing and existing.get("fingerprint") != fp:
        raise SystemExit(f"{args.out} holds records of another encoding ({existing.get('fingerprint')} "
                         f"against {fp}); use another --out")
    rows = [json.loads(line) for line in
            (args.dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()][:args.limit]
    games: Dict[str, Dict[str, int]] = dict(existing.get("games", {}))
    errors: Dict[str, str] = dict(existing.get("errors", {}))
    t0 = time.time()

    def flush() -> None:
        totals: Counter = Counter()
        for c in games.values():
            totals.update(c)
        args.out.joinpath(MANIFEST_NAME).write_text(json.dumps({
            "fingerprint": fp, "encoding": ENCODING, "observation_epoch": int(OBSERVATION_EPOCH),
            "corpus_version": corpus, "core_phase": phase, "dataset": str(args.dataset),
            "vocab_from": str(args.vocab_from) if args.vocab_from else "fresh",
            "unit_type_to_id": type_to_id, "faction_to_id": faction_to_id,
            "n_manifest_games": len(rows), "n_games": len(games), "totals": dict(totals),
            "errors": errors, "games": games,
        }, indent=0), encoding="utf-8")

    log.info("pre-encoding %d games with %d workers into %s (%d unit types, %d factions, core phase %d)",
             len(rows), args.workers, args.out, len(type_to_id), len(faction_to_id), phase)
    with mp.get_context("spawn").Pool(args.workers, initializer=_worker_init,
                                      initargs=(type_to_id, faction_to_id, str(args.dataset),
                                                str(args.out))) as pool:
        for i, (file, status, counts) in enumerate(pool.imap_unordered(_worker_encode, rows, chunksize=2), 1):
            if status == "ok":
                games[file] = counts
                errors.pop(file, None)
            elif status == "exists":
                if file not in games:
                    games[file] = read_record(record_path(args.out, file)).counts
            else:
                errors[file] = status
            if i % 200 == 0 or i == len(rows):
                n = sum(c.get("positions", 0) for c in games.values())
                el = time.time() - t0
                log.info("%d/%d games, %d positions, %d errors, %.0f s, %.1f positions/s",
                         i, len(rows), n, len(errors), el, n / max(el, 1e-9))
                flush()
    flush()
    n = sum(c.get("positions", 0) for c in games.values())
    log.info("SEQUENCES_DONE %d games, %d positions, %d errors in %.0f s", len(games), n, len(errors),
             time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
