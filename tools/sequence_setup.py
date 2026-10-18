"""The sequence trainer's setup (tools/sequence_train.py): the recipe's
architecture and defaults, the loader of the pre-encoded records, the games'
facts, the modules, the learning rate's schedule and the checkpoint."""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch

from tools.preencode_sequences import read_record, record_path
from wesnoth_ai.checkpoint_structure import checkpoint_structure
from wesnoth_ai.constants import OBSERVATION_EPOCH
from wesnoth_ai.encoder import GameStateEncoder
from wesnoth_ai.model import WesnothModel
from wesnoth_ai.sequence_columns import ColumnarSide, columns_path, read_side, read_sides
from wesnoth_ai.sequence_loss import GameFacts
from wesnoth_ai.sequence_streams import GameSide, StreamSchedule

# The recipe's architecture; the flags of the same names change it (a smoke run).
ARCH = {"d_model": 384, "num_layers": 8, "num_heads": 12, "d_ff": 1536}
MEMORY_SLOTS = 64
# The recipe (docs/parity_memory_design_20260929.md "Training").
DEFAULTS = {"streams": 32, "window": 16, "lr": 2.8e-4, "warmup_steps": 300, "weight_decay": 1e-4,
            "grad_clip": 1.0, "value_states_per_game": 16, "belief_weight": 1.0,
            "seed": 20260929, "probe_every": 500_000, "barrier_positions": 500_000,
            "checkpoint_every": 250_000, "probe_ks": (0, 16, 64), "signal_every": 100_000}
# Game-sides the loader fetches ahead of the slots, and its threads.
PREFETCH_SIDES = 64
LOADER_THREADS = 4


class SequenceLoader:
    """The game records the slots read, fetched ahead on threads (the zlib
    half releases the GIL) and dropped once no slot or upcoming game-side
    needs them. The records named in `own_files` are read from
    `own_directory`. Its clocks (tools/sequence_timing.py): `wait_seconds`,
    the time the caller waited for a record still being read, and
    `decode_seconds`, the CPU time its threads spent reading records."""

    def __init__(self, directory: Path, own_directory: Optional[Path] = None, own_files=frozenset()):
        self.directory = Path(directory)
        self.own_directory = None if own_directory is None else Path(own_directory)
        self.own_files = frozenset(own_files)
        self._pool = ThreadPoolExecutor(LOADER_THREADS)
        self._records: Dict[str, object] = {}
        self._held: Set[str] = set()
        self._lock = threading.Lock()
        self.wait_seconds = 0.0
        self.decode_seconds = 0.0

    def _path(self, f: str) -> Path:
        return record_path(self.own_directory if f in self.own_files else self.directory, f)

    def _read(self, path: Path):
        t0 = time.thread_time()
        rec = read_record(path)
        spent = time.thread_time() - t0
        with self._lock:
            self.decode_seconds += spent
        return rec

    def prefetch(self, game_sides) -> None:
        """Start reading the records of `game_sides` (both sides of a game
        are one record)."""
        for f in dict.fromkeys(g.file for g in game_sides):
            if f not in self._records:
                self._records[f] = self._pool.submit(self._read, self._path(f))

    def sides(self, game_side: GameSide):
        rec = self._records.get(game_side.file)
        if rec is None:
            self.prefetch([game_side])
            rec = self._records[game_side.file]
        if not hasattr(rec, "sides"):
            if not rec.done():
                t0 = time.perf_counter()
                rec.result()
                self.wait_seconds += time.perf_counter() - t0
            rec = self._records[game_side.file] = rec.result()
        return rec.sides[game_side.side]

    def hold(self, files) -> None:
        """Read `files` now, on the calling thread, and keep them whatever
        `keep_only` is told (a benchmark's preloaded records)."""
        for f in files:
            if f not in self._held:
                self._records[f] = read_record(self._path(f))
                self._held.add(f)

    def keep_only(self, game_sides) -> None:
        """Drop every record but those of `game_sides` and the held ones."""
        keep = {g.file for g in game_sides} | self._held
        for f in [f for f in self._records if f not in keep]:
            del self._records[f]


class ColumnLoader:
    """`SequenceLoader` for columnar records (wesnoth_ai/sequence_columns.py),
    one game-side at a time: a game-side's read is one zlib inflate, which
    releases the GIL, and its positions are slices of a few arrays, so the
    loader's threads build no Python object per position and the collector
    has none to walk. Same clocks."""

    def __init__(self, directory: Path, own_directory: Optional[Path] = None, own_files=frozenset()):
        self.directory = Path(directory)
        self.own_directory = None if own_directory is None else Path(own_directory)
        self.own_files = frozenset(own_files)
        self._pool = ThreadPoolExecutor(LOADER_THREADS)
        self._sides: Dict[GameSide, object] = {}
        self._held: Set[GameSide] = set()
        self._lock = threading.Lock()
        self.wait_seconds = 0.0
        self.decode_seconds = 0.0

    def _path(self, f: str) -> Path:
        return columns_path(self.own_directory if f in self.own_files else self.directory, f)

    def _read(self, game_side: GameSide) -> ColumnarSide:
        t0 = time.thread_time()
        side = read_side(self._path(game_side.file), game_side.side)
        spent = time.thread_time() - t0
        with self._lock:
            self.decode_seconds += spent
        return side

    def prefetch(self, game_sides) -> None:
        for g in game_sides:
            if g not in self._sides:
                self._sides[g] = self._pool.submit(self._read, g)

    def side(self, game_side: GameSide) -> ColumnarSide:
        side = self._sides.get(game_side)
        if side is None:
            self.prefetch([game_side])
            side = self._sides[game_side]
        if not isinstance(side, ColumnarSide):
            if not side.done():
                t0 = time.perf_counter()
                side.result()
                self.wait_seconds += time.perf_counter() - t0
            side = self._sides[game_side] = side.result()
        return side

    def hold(self, files) -> None:
        """Read both sides of `files` now and keep them whatever `keep_only`
        is told (a benchmark's preloaded records)."""
        for f in files:
            for s, side in read_sides(self._path(f)).items():
                g = GameSide(f, s)
                self._sides[g] = side
                self._held.add(g)

    def keep_only(self, game_sides) -> None:
        keep = set(game_sides) | self._held
        for g in [g for g in self._sides if g not in keep]:
            del self._sides[g]


def game_facts(dataset: Path, games: Dict[str, Dict[str, int]]) -> Tuple[Dict[str, GameFacts], List[str], List[str]]:
    """Each pre-encoded game's facts, and the train and holdout files, from
    the dataset's manifest. The policy weight is `obs8`'s per-game weight:
    the median winner's action count over the game's, clipped to [0.25, 4]."""
    rows = [json.loads(line) for line in (Path(dataset) / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    acts = sorted(r["winner_actions"] for r in rows if r["winner_actions"] > 0)
    median = acts[len(acts) // 2] if acts else 1
    facts, train, holdout = {}, [], []
    for r in rows:
        f = r["file"]
        if f not in games:
            continue
        w = max(0.25, min(4.0, median / r["winner_actions"])) if r["winner_actions"] > 0 else 1.0
        facts[f] = GameFacts(winner=int(r["winner_side"]), n_commands=int(games[f]["n_commands"]),
                             policy_weight=float(w))
        (holdout if r.get("holdout") else train).append(f)
    return facts, sorted(train), sorted(holdout)


def build_modules(unit_type_to_id: Dict[str, int], faction_to_id: Dict[str, int],
                  device: torch.device, arch: Dict[str, int]) -> Tuple[GameStateEncoder, WesnothModel]:
    encoder = GameStateEncoder(d_model=arch["d_model"], relevant_set_hexes=True,
                               fog_hides_enemy_villages=True, terrain_multi_hot=True,
                               observation_parity=True, relevant_set_version=2,
                               unit_type_to_id=unit_type_to_id, faction_to_id=faction_to_id).to(device)
    encoder.freeze_vocab()
    model = WesnothModel(observation_parity=True, memory_slots=MEMORY_SLOTS, **arch).to(device)
    return encoder, model


def own_stream_count(args) -> int:
    """The slots own game-sides take: none without `--own-sequences`, else
    `--own-streams`, by default half of `--streams` (equal numbers of own
    and corpus positions in every step)."""
    if args.own_sequences is None:
        return 0
    n = args.streams // 2 if args.own_streams is None else int(args.own_streams)
    if not 0 < n < args.streams:
        raise SystemExit(f"--own-streams {n}: some of the {args.streams} streams, not all, walk own game-sides")
    return n


def lr_factor(positions: int, total: int, decay_from: Optional[float]) -> float:
    """The learning rate's multiplier at `positions` of a pass of `total`:
    1, then from the fraction `decay_from` of the pass falling linearly to
    0 at its end."""
    if decay_from is None:
        return 1.0
    start = decay_from * total
    if positions < start:
        return 1.0
    return max(0.0, (total - positions) / max(1.0, total - start))


def save_checkpoint(path: Path, encoder, model, opt, state: Dict, schedule: StreamSchedule,
                    memories: Dict[int, torch.Tensor], meta: Dict, arch: Dict[str, int],
                    own_schedule: Optional[StreamSchedule] = None, extra: Optional[Dict] = None) -> None:
    """Atomic: a temporary file, then a rename. Loadable by the policy
    loader (`eval_players.peek_checkpoint_arch`, `TransformerPolicy`).
    `extra`: entries beside the weights (the trainer's weight average)."""
    payload = {
        "observation_epoch": int(OBSERVATION_EPOCH), "arch": dict(arch),
        "relevant_set_hexes": True, "fog_hides_enemy_villages": True, "terrain_multi_hot": True,
        **checkpoint_structure(model, encoder),
        "model_state": model.state_dict(), "encoder_state": encoder.state_dict(),
        "unit_type_to_id": dict(encoder.unit_type_to_id), "faction_to_id": dict(encoder.faction_to_id),
        "optimizer_state": opt.state_dict(), "training_meta": dict(meta),
        "sequence_resume": {"state": dict(state), "schedule": schedule.state_dict(),
                            "memories": {int(s): m.detach().cpu() for s, m in memories.items()},
                            "rng": {"cpu": torch.get_rng_state(),
                                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}},
    }
    if own_schedule is not None:
        payload["sequence_resume"]["own_schedule"] = own_schedule.state_dict()
    payload.update(extra or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
