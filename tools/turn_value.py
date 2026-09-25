#!/usr/bin/env python3
"""The turn-ranking value function (docs/turn_value_prereg_20260925.md).

  features  rebuild every candidate turn's pre-end_turn position from a
            position file (a tools/turn_value_data.py log, or a
            tools/turn_gap.py result) and cache, per candidate, the
            frozen reference trunk's global token (the value head's
            input), the reference value head's read, and the playout
            truth.
  fit       fit the arms on a cache's fit split, early-stopped on its
            stop split (tools/turn_value_fit.py).
  evaluate  the arms and the baselines against the playout truth of
            turn_gap validation files (the pre-registered rule), and the
            proxy barrier on a cache's proxy split.

A candidate's pre-end_turn position is its boundary position with the
candidate's recorded commands and recruit rejections applied
(`pre_end_turn` in the record, tools/turn_gap.py), checked against the
recorded digest. A candidate turn that ended the game has none; the
analysis reads its outcome instead (tools/analysis/turn_gap_pregrader.py).

Usage (box):
  python tools/turn_value.py features --reference --positions data.jsonl.gz \\
      --games-dir DIR --out train.pt --device cuda --jobs 24
  python tools/turn_value.py features --reference --positions confirm.json --out confirm.pt
  python tools/turn_value.py fit --train train.pt --out-dir heads
  python tools/turn_value.py evaluate --heads heads --train train.pt \\
      --validation confirm.json=confirm.pt --validation screen.json=screen.pt \\
      --out verdict.json
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import logging
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.bench_pipeline import DEFAULT_DATASET
from tools.game_record import read_records, turn_starts, walk

log = logging.getLogger("turn_value")

SPLIT_VALIDATION = "validation"


# ---------------------------------------------------------------------
# Position files
# ---------------------------------------------------------------------

def load_positions(path: Path) -> List[Dict]:
    """The position records of a turn_value_data log (.jsonl.gz: the
    finished games' records, the last copy of a repeated index kept) or
    of a turn_gap result (.json, full or partial)."""
    path = Path(path)
    if path.name.endswith(".jsonl.gz"):
        from tools.turn_value_data import read_log
        _, positions, done = read_log(path)
        by_index = {p["index"]: p for p in positions if p["meta"].get("game") in done}
        return [by_index[i] for i in sorted(by_index)]
    return json.loads(path.read_text(encoding="utf-8"))["positions"]


def candidates(record: Dict) -> List[Dict]:
    """Slot 0 is the base, slot j the j-th alternative of the record."""
    return [record["base"]] + list(record.get("alternatives", []))


def split_of(record: Dict) -> str:
    return record.get("meta", {}).get("split", SPLIT_VALIDATION)


def group_of(record: Dict) -> str:
    """What the proxy's bootstrap resamples: the game a position comes
    from (its positions share a game), else the position itself."""
    meta = record.get("meta", {})
    return str(meta.get("game") or meta.get("file") or record["index"])


# ---------------------------------------------------------------------
# Rebuilding positions and pre-end_turn states
# ---------------------------------------------------------------------

def apply_snapshot(gs, snapshot: Dict, label: str) -> None:
    """Apply a candidate's pre-end_turn commands and recruit rejections
    to `gs` (its boundary position), checked against the recorded
    digest (tools/game_record.RecordMismatch when they differ)."""
    turn = {"game_label": label, "commands": snapshot["commands"],
            "rejections": snapshot.get("rejections", []), "final_digest": snapshot["digest"]}
    for _ in walk(turn, gs):
        pass


def _record_positions(games_dir: Path, game: str,
                      command_indices: Sequence[int]) -> Iterator[Tuple[int, object]]:
    """(command index, turn-start state) of one recorded game at the
    wanted init_side commands, in game order. The state is the walk's."""
    [rec] = list(read_records(Path(games_dir) / game))
    wanted = set(command_indices)
    for k, gs in turn_starts(rec):
        if k in wanted:
            wanted.discard(k)
            yield k, gs
            if not wanted:
                return
    raise RuntimeError(f"{game}: turn starts {sorted(wanted)} never reached")


def _manifest_position(dataset: Path, entry: Dict):
    """The boundary state of a manifest entry, as turn_gap rebuilds it."""
    from tools.bench_pipeline import reconstruct_boundary
    with gzip.open(Path(dataset) / entry["file"], "rt", encoding="utf-8") as f:
        data = json.load(f)
    res = reconstruct_boundary(data, entry["cut_turn"])
    if res is None or res[1] != entry["begin_side"]:
        raise RuntimeError(f"manifest position failed to reconstruct: {entry}")
    return res[0]


_ENCODE_ARGS: Optional[Dict] = None


def _init_worker(encode_args: Dict) -> None:
    global _ENCODE_ARGS
    _ENCODE_ARGS = encode_args
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")


def _encode_group(task) -> List[Tuple[int, int, object]]:
    """(position index, slot, RawEncoded) for every candidate of a group
    of positions that shares one source, the ones without a pre-end_turn
    snapshot left out."""
    from wesnoth_ai.encoder import encode_raw
    kind, source, items = task
    out = []
    if kind == "game_record":
        games_dir, game = source
        states = _record_positions(games_dir, game, [k for _, k, _ in items])
        by_k = {k: (index, snaps) for index, k, snaps in items}
        pairs = ((by_k[k], gs) for k, gs in states)
    else:
        dataset, entry = source
        (index, _, snaps), = items
        pairs = [((index, snaps), _manifest_position(dataset, entry))]
    for (index, snaps), gs in pairs:
        for slot, snap in snaps:
            state = copy.deepcopy(gs)
            apply_snapshot(state, snap, f"position {index} slot {slot}")
            out.append((index, slot, encode_raw(state, **_ENCODE_ARGS)))
    return out


def encode_tasks(records: Sequence[Dict], games_dir: Optional[Path],
                 dataset: Path) -> Tuple[List, Counter]:
    """One task per source game (turn_value_data logs) or per position
    (manifest positions), each listing its candidates' snapshots; plus
    the count of candidates left out, by reason."""
    skipped: Counter = Counter()
    games: Dict[str, List] = defaultdict(list)
    tasks = []
    for rec in records:
        snaps = []
        for slot, cand in enumerate(candidates(rec)):
            snap = cand.get("pre_end_turn")
            if snap:
                snaps.append((slot, snap))
            else:
                skipped["terminal_in_turn" if cand.get("terminal_in_turn") else "no_snapshot"] += 1
        if not snaps:
            continue
        meta = rec.get("meta", {})
        if meta.get("source") == "game_record":
            if games_dir is None:
                raise SystemExit("positions from recorded games need --games-dir")
            games[meta["game"]].append((rec["index"], int(meta["command_index"]), snaps))
        else:
            tasks.append(("manifest", (str(dataset), meta), [(rec["index"], None, snaps)]))
    for game, items in games.items():
        items.sort(key=lambda it: it[1])
        tasks.append(("game_record", (str(games_dir), game), items))
    return tasks, skipped


# ---------------------------------------------------------------------
# The frozen trunk
# ---------------------------------------------------------------------

def load_reference_model(checkpoint: Path, device):
    """(model, encoder) of a checkpoint as the eval path builds it, in
    eval mode, its value head checked against the file (a load that
    fell back to a random init must not be measured under its name)."""
    import torch
    from tools.eval_sim import _load_policy
    policy = _load_policy(Path(checkpoint), device, label="turn_value")
    model, encoder = policy._trainer.model, policy._trainer.encoder
    model.eval()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"]
    for name, tensor in model.value_head.state_dict().items():
        if not torch.equal(tensor.detach().cpu(), saved[f"value_head.{name}"]):
            raise RuntimeError(f"{checkpoint}: the loaded value head differs from the file's")
    return model, encoder


def encoder_switches(encoder) -> Dict:
    """What encode_raw needs to encode as `encoder.raw_of` does."""
    return {"type_to_id": dict(encoder.unit_type_to_id),
            "faction_to_id": dict(encoder.faction_to_id),
            "relevant_set": bool(encoder.relevant_set_hexes),
            "fog_hides_enemy_villages": bool(encoder.fog_hides_enemy_villages),
            "terrain_multi_hot": bool(encoder.terrain_multi_hot)}


def head_value(model, feats):
    """The value head's expected value on global tokens [B, d]."""
    import torch
    probs = torch.softmax(model.value_head(feats), dim=-1)
    return (probs * model._value_atoms).sum(dim=-1)


class TrunkReader:
    """Batches encoded states through the frozen trunk and keeps the
    value head's input (the global token) of each."""

    def __init__(self, model, encoder, device, batch: int):
        import torch
        self.model, self.encoder, self.device, self.batch = model, encoder, device, batch
        self.pending: List[Tuple[Tuple[int, int], object]] = []
        self.keys: List[Tuple[int, int]] = []
        self.feats: List = []
        self._captured: List = []
        self._hook = model.value_head.register_forward_hook(
            lambda _m, inp, _out: self._captured.append(inp[0].detach().float().cpu()))
        self._torch = torch

    def add(self, key: Tuple[int, int], raw) -> None:
        self.pending.append((key, raw))
        if len(self.pending) >= self.batch:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        keys, raws = zip(*self.pending)
        self.pending = []
        self._captured.clear()
        with self._torch.no_grad():
            encoded = self.encoder.encode_from_raw_batch(list(raws), device=self.device)
            self.model.forward_batch(encoded, autocast_bf16=False)
        feats = self._torch.cat(self._captured, dim=0)
        if feats.shape[0] != len(keys):
            raise RuntimeError(f"captured {feats.shape[0]} global tokens for {len(keys)} states")
        self.keys.extend(keys)
        self.feats.append(feats)

    def close(self):
        self.flush()
        self._hook.remove()
        return self.keys, (self._torch.cat(self.feats, dim=0) if self.feats
                           else self._torch.zeros(0, self.model.d_model))


def build_cache(records: Sequence[Dict], model, encoder, device, *, games_dir: Optional[Path],
                dataset: Path, jobs: int, batch: int) -> Dict:
    """The feature cache of every candidate with a pre-end_turn state."""
    import numpy as np
    import torch
    tasks, skipped = encode_tasks(records, games_dir, dataset)
    reader = TrunkReader(model, encoder, device, batch)
    t0 = time.time()
    n_tasks = 0

    def consume(rows):
        nonlocal n_tasks
        n_tasks += 1
        for index, slot, raw in rows:
            reader.add((index, slot), raw)
        if n_tasks % 50 == 0:
            log.info("features: %d/%d sources, %d candidates, %.0f s", n_tasks, len(tasks),
                     len(reader.keys) + len(reader.pending), time.time() - t0)

    enc_args = encoder_switches(encoder)
    if jobs <= 1:
        _init_worker(enc_args)
        for task in tasks:
            consume(_encode_group(task))
    else:
        with mp.get_context("spawn").Pool(jobs, initializer=_init_worker,
                                          initargs=(enc_args,)) as pool:
            for rows in pool.imap_unordered(_encode_group, tasks):
                consume(rows)
    keys, feats = reader.close()
    with torch.no_grad():
        value = head_value(model, feats.to(device)).float().cpu()
    by_index = {rec["index"]: rec for rec in records}
    y, n, split, group = [], [], [], []
    for index, slot in keys:
        rec = by_index[index]
        outcomes = candidates(rec)[slot]["outcomes"]
        y.append(float(np.mean(outcomes)))
        n.append(len(outcomes))
        split.append(split_of(rec))
        group.append(group_of(rec))
    log.info("features: %d candidates in %.0f s; left out %s", len(keys), time.time() - t0,
             dict(skipped))
    return {"feats": feats, "value_reference": value,
            "index": torch.tensor([k[0] for k in keys], dtype=torch.int64),
            "slot": torch.tensor([k[1] for k in keys], dtype=torch.int64),
            "y": torch.tensor(y, dtype=torch.float32), "n": torch.tensor(n, dtype=torch.float32),
            "split": split, "group": group, "skipped": dict(skipped),
            "switches": {k: v for k, v in enc_args.items() if not k.endswith("_to_id")}}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _checkpoint_of(args) -> Tuple[Path, Optional[Dict]]:
    from tools import reference_player
    if args.reference == (args.checkpoint is not None):
        raise SystemExit("pass exactly one of --reference and --checkpoint")
    if args.checkpoint is not None:
        return Path(args.checkpoint), None
    ref = reference_player.load()
    return (Path(reference_player.ensure_checkpoint(ref)),
            {k: ref.get(k) for k in ("label", "checkpoint_hf", "procedure_tag")})


def cmd_features(args) -> int:
    import torch
    checkpoint, reference = _checkpoint_of(args)
    device = torch.device(args.device)
    model, encoder = load_reference_model(checkpoint, device)
    records = load_positions(args.positions)
    log.info("%s: %d positions", args.positions, len(records))
    cache = build_cache(records, model, encoder, device, games_dir=args.games_dir,
                        dataset=args.dataset, jobs=args.jobs, batch=args.batch)
    cache.update(positions=str(args.positions), checkpoint=str(checkpoint), reference=reference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.out)
    log.info("wrote %s: %d candidates, splits %s", args.out, len(cache["split"]),
             dict(Counter(cache["split"])))
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("features", help="cache the trunk's global token per candidate")
    f.add_argument("--positions", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--reference", action="store_true")
    f.add_argument("--checkpoint", default=None)
    f.add_argument("--games-dir", type=Path, default=None)
    f.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                   help="The replay dataset manifest positions are rebuilt from.")
    f.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    f.add_argument("--jobs", type=int, default=1)
    f.add_argument("--batch", type=int, default=128)
    from tools.turn_value_fit import add_fit_parsers
    add_fit_parsers(sub)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.command == "features":
        return cmd_features(args)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
