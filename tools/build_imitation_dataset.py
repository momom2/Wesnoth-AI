"""Build the imitation-learning corpus from the raw replays.

Every candidate the dispositions ledger accepts is extracted with the
game cut where it stopped being one between its two players
(tools/replay_control: the first action after a surrender, or a side's
turn taken by its opponent), labelled by tools/replay_outcome (leader
death, else the surrendering side loses; a surrender by the side ahead
on material leaves the game unlabelled), and kept when its outcome class
is one of configs/imitation.json's `outcome_classes`. Copies of one match
(re-saved, re-uploaded or reloaded games) keep only their longest.

Inputs:
  - training/logs/replay_dispositions.jsonl.gz  the accepted pool (paths
    `replays_raw\\<date>\\<file>.bz2`, resolved under --raw-root)
  - configs/imitation.json                      the selection knobs

Outputs, under the config's dataset_dir:
  - <date>_<stem>.json.gz   one per kept game (the date prefix because
                            Wesnoth server game ids RECYCLE across days)
  - manifest.jsonl          one row per kept game: file, source, the
                            outcome (class, winner, n_turns, material),
                            n_commands, winner_actions (the winner's
                            move/attack/recruit/recall commands, the
                            per-game weight denominator), holdout
                            (sha1 of the ledger path: rebuilds keep the
                            split), fog, shroud, match_key and
                            corpus_version
  - value_corpus_index.jsonl  file / winner / n_commands, for the value loss
  - outcomes.jsonl          every candidate's outcome, kept or not
  - quarantined.jsonl, duplicates.jsonl, errors.jsonl

Usage (a CPU box; 0.23-0.34 s per candidate on one core, measured on the
laptop 2026-09-26, so 19,367 candidates are 1.2-1.8 core-hours):
    python tools/build_imitation_dataset.py --workers 30
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path, PurePosixPath
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("build_imitation_dataset")

# The rules a corpus is built under, stamped on every manifest row and
# read back by `replay_dataset.corpus_version_of`. 2 (2026-09-26): a
# stopped move is labelled with the hex its player clicked, games are cut
# at their end, the outcome comes from tools/replay_outcome, and
# reloaded games deduplicate on a 30-command prefix.
CORPUS_VERSION = 2

ACCEPT_MOD_CLASSES = ("mod_free", "kept_cosmetic", "kept_plan_unit_advance")
DISPOSITIONS = Path("training/logs/replay_dispositions.jsonl.gz")


def load_candidates(dispositions: Path) -> List[str]:
    """The ledger paths of every replay the dispositions accept (era
    accepted, mods absent or harmless), in ledger order."""
    out = []
    with gzip.open(dispositions, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["era_class"] == "accept" and d["mod_class"] in ACCEPT_MOD_CLASSES:
                out.append(d["path"])
    return out


def raw_path(ledger_path: str, raw_root: Path) -> Path:
    """A ledger path (written on Windows, backslashes) under raw_root."""
    return raw_root / Path(PurePosixPath(ledger_path.replace("\\", "/")))


def corpus_file_name(ledger_path: str) -> str:
    p = PurePosixPath(ledger_path.replace("\\", "/"))
    return f"{p.parent.name}_{p.stem}.json.gz"


def is_holdout(ledger_path: str, holdout_fraction: float) -> bool:
    """Deterministic game-level split, stable across rebuilds."""
    h = int(hashlib.sha1(ledger_path.encode("utf-8")).hexdigest()[:8], 16)
    return (h % 10_000) < holdout_fraction * 10_000


def _winner_action_count(commands: list, winner_side: int) -> int:
    """Static count of the winner's actionable commands (move / attack
    / recruit / recall while it is the winner's turn). Approximates
    the trainer's pair count (which drops the rare unmappable action)
    closely enough for per-game weighting."""
    side = 0
    n = 0
    for c in commands:
        if not c:
            continue
        if c[0] == "init_side":
            side = c[1]
        elif side == winner_side and c[0] in ("move", "attack",
                                              "recruit", "recall"):
            n += 1
    return n


def quarantine_of(rec: dict, config: dict):
    """Why an extracted record is not a game of the corpus, or None."""
    from tools.replay_dataset import _player_sides, quarantine_reason
    sides = rec.get("starting_sides", [])
    why = quarantine_reason(sides)
    if why is not None:
        return why
    if config.get("quarantine_ai_player_sides", True) and any(
            str(s.get("controller", "")).lower() == "ai" for s in _player_sides(sides)):
        return "ai_player_side"
    if (rec.get("game_end") or {}).get("cut_before_play"):
        return "one_player_both_sides"
    return None


def build_one(job: Tuple[str, str, str, dict]) -> dict:
    """One candidate: extract, quarantine, label, and write the kept
    game. The returned row says which of those it came to."""
    ledger_path, raw_root, out_dir, config = job
    from tools.replay_dataset import fog_on_for, match_key
    from tools.replay_extract import extract_replay
    from tools.replay_outcome import label_outcome
    row: dict = {"source": ledger_path}
    try:
        rec = extract_replay(raw_path(ledger_path, Path(raw_root)), cut_at_game_end=True)
        if rec is None:
            return {**row, "error": "extract_none"}
        why = quarantine_of(rec, config)
        if why is not None:
            return {**row, "quarantined": why}
        outcome = label_outcome(rec)
    except Exception as e:                          # noqa: BLE001 - one bad replay must not stop the build
        return {**row, "error": f"{type(e).__name__}: {e}"[:160]}
    row.update(outcome.as_row())
    row["game_end"] = rec.get("game_end")
    if outcome.outcome not in config["outcome_classes"]:
        return row
    fname = corpus_file_name(ledger_path)
    with gzip.open(Path(out_dir) / fname, "wt", encoding="utf-8") as f:
        json.dump(rec, f)
    sides = rec.get("starting_sides", [])
    row.update({
        "file": fname,
        "n_commands": len(rec["commands"]),
        "winner_actions": _winner_action_count(rec["commands"], outcome.winner_side),
        "holdout": is_holdout(ledger_path, float(config["holdout_fraction"])),
        "fog": fog_on_for(sides),
        "shroud": any(bool(s.get("shroud", False)) for s in sides),
        "match_key": match_key(rec),
        "corpus_version": CORPUS_VERSION,
    })
    return row


def dedup_rows(rows):
    """One game per match (tools/replay_dataset.match_key): the copy
    with the most commands survives (ties: the first file name), it
    is holdout when any copy was, and the others are listed with
    their survivor. The 2026-09-08 review found 79 clusters of
    re-saved or re-uploaded games and one straddling the split."""
    by_key = {}
    for r in rows:
        by_key.setdefault(r.get("match_key"), []).append(r)
    kept, dropped = [], []
    for key, group in by_key.items():
        if key is None or len(group) == 1:
            kept.extend(group)
            continue
        group = sorted(group, key=lambda r: (-int(r.get("n_commands", 0)), r["file"]))
        survivor = dict(group[0])
        survivor["holdout"] = any(bool(r.get("holdout")) for r in group)
        kept.append(survivor)
        for r in group[1:]:
            dropped.append(dict(r, duplicate_of=survivor["file"]))
    return kept, dropped


def _results(jobs: list, workers: int):
    """build_one over the jobs: in this process for one worker, else in
    a pool, in completion order."""
    if workers <= 1:
        yield from map(build_one, jobs)
        return
    with Pool(workers) as pool:
        yield from pool.imap_unordered(build_one, jobs, chunksize=8)


def write_jsonl(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build(candidates: List[str], raw_root: Path, out_dir: Path, config: dict,
          workers: int) -> Counter:
    """Build the corpus into out_dir from the candidates' raw replays;
    returns the counts the summary line prints."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(p, str(raw_root), str(out_dir), config) for p in candidates]
    kept, outcomes, quarantined, errors = [], [], [], []
    t0 = time.time()
    for i, row in enumerate(_results(jobs, workers), 1):
        if "error" in row:
            errors.append(row)
        elif "quarantined" in row:
            quarantined.append(row)
        else:
            outcomes.append(row)
            if "file" in row:
                kept.append(row)
        if i % 1000 == 0 or i == len(jobs):
            rate = i / max(time.time() - t0, 1e-9)
            log.info(f"[{i}/{len(jobs)}] kept {len(kept)}, quarantined {len(quarantined)}, "
                     f"errors {len(errors)}, {rate:.1f}/s, "
                     f"{(len(jobs) - i) / max(rate, 1e-9) / 60:.0f} min left")
    rows, duplicates = dedup_rows(kept)
    dup_dir = out_dir.parent / (out_dir.name + "_duplicates")
    for r in duplicates:
        src = out_dir / r["file"]
        if src.exists():
            dup_dir.mkdir(parents=True, exist_ok=True)
            src.replace(dup_dir / r["file"])
    rows.sort(key=lambda r: r["file"])
    write_jsonl(out_dir / "manifest.jsonl", rows)
    write_jsonl(out_dir / "value_corpus_index.jsonl", (
        {"file": r["file"], "winner": r["winner_side"], "n_commands": r["n_commands"]}
        for r in rows))
    write_jsonl(out_dir / "duplicates.jsonl", duplicates)
    write_jsonl(out_dir / "quarantined.jsonl", quarantined)
    write_jsonl(out_dir / "errors.jsonl", errors)
    write_jsonl(out_dir / "outcomes.jsonl", sorted(outcomes, key=lambda r: r["source"]))
    counts: Counter = Counter(r["outcome"] for r in outcomes)
    counts.update({"games": len(rows), "duplicates": len(duplicates),
                   "quarantined": len(quarantined), "errors": len(errors),
                   "fog_off": sum(1 for r in rows if not r["fog"])})
    return counts


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=Path("configs/imitation.json"))
    ap.add_argument("--dispositions", type=Path, default=DISPOSITIONS)
    ap.add_argument("--raw-root", type=Path, default=Path("."),
                    help="directory the ledger's replays_raw/... paths resolve under")
    ap.add_argument("--out", type=Path, default=None,
                    help="dataset directory (default: the config's dataset_dir)")
    ap.add_argument("--limit", type=int, default=None, help="first N candidates only")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args(argv[1:])
    config = json.loads(args.config.read_text(encoding="utf-8"))
    out_dir = args.out or Path(config["dataset_dir"])
    candidates = load_candidates(args.dispositions)[:args.limit]
    log.info(f"imitation corpus v{CORPUS_VERSION}: {len(candidates)} candidates, "
             f"classes {config['outcome_classes']} -> {out_dir}")
    t0 = time.time()
    counts = build(candidates, args.raw_root, out_dir, config, args.workers)
    summary: Dict[str, int] = dict(sorted(counts.items()))
    log.info(f"BUILD_DONE in {(time.time() - t0) / 60:.1f} min: {summary}")
    return 1 if counts["errors"] else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    sys.exit(main(sys.argv))
