"""Build the imitation-learning dataset from the certified corpus.

Selects games from the dispositions ledger + outcome labels per
configs/imitation.json, extracts each raw .bz2 into the standard
replays_dataset json.gz format (the exact input `iter_replay_pairs`
consumes), and writes a manifest the trainer uses for winner
filtering and per-game weighting.

Inputs:
  - training/logs/replay_dispositions.jsonl.gz  (accepted pool)
  - training/logs/replay_outcomes.jsonl.gz      (outcome classes)
  - configs/imitation.json                      (selection knobs)

Outputs (under the config's dataset_dir):
  - <date>_<stem>.json.gz   one per selected game (date prefix because
                            Wesnoth server game ids RECYCLE across
                            days -- same id, unrelated games; proven
                            during the 2026-08-07 dedup work)
  - manifest.jsonl          one row per game: file, source path,
                            winner_side, outcome class, n_turns,
                            approx winner-side action count (static
                            count of move/attack/recruit/recall
                            commands during the winner's turns --
                            the per-game weight denominator), and a
                            holdout flag (deterministic split by
                            sha1(path), so rebuilds keep the split).

Usage:
    python tools/build_imitation_dataset.py [--config configs/imitation.json]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("build_imitation_dataset")

ACCEPT_MOD_CLASSES = ("mod_free", "kept_cosmetic", "kept_plan_unit_advance")


def _load_selection(config: dict) -> list:
    """Return [(path, outcome_row)] for games matching the config."""
    disp = {}
    with gzip.open("training/logs/replay_dispositions.jsonl.gz",
                   "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            disp[d["path"]] = d
    classes = set(config["outcome_classes"])
    ratio_max = float(config.get("inferred_ratio_max", 0.5))
    out = []
    with gzip.open("training/logs/replay_outcomes.jsonl.gz",
                   "rt", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            d = disp.get(o["path"])
            if d is None or d["era_class"] != "accept":
                continue
            if d["mod_class"] not in ACCEPT_MOD_CLASSES:
                continue
            if o["outcome"] not in classes:
                continue
            if (o["outcome"] == "inferred"
                    and o["material_ratio"] > ratio_max):
                continue
            if not o.get("winner_side"):
                continue        # outcome class without a usable winner
            out.append((o["path"], o))
    return out


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


def _build_one(args) -> Optional[dict]:
    path_str, outcome, out_dir_str, holdout_fraction = args
    from tools.replay_extract import extract_replay
    src = Path(path_str)
    try:
        rec = extract_replay(src)
    except Exception as e:                          # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:160],
                "source": path_str}
    if rec is None:
        return {"error": "extract_none", "source": path_str}
    date = src.parent.name
    fname = f"{date}_{src.stem}.json.gz"
    out_path = Path(out_dir_str) / fname
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(rec, f)
    # Deterministic game-level holdout split: stable across rebuilds
    # and independent of selection order.
    h = int(hashlib.sha1(path_str.encode("utf-8")).hexdigest()[:8], 16)
    holdout = (h % 10_000) < holdout_fraction * 10_000
    return {
        "file": fname,
        "source": path_str,
        "winner_side": outcome["winner_side"],
        "outcome": outcome["outcome"],
        "n_turns": outcome["n_turns"],
        "n_commands": len(rec["commands"]),
        "winner_actions": _winner_action_count(
            rec["commands"], outcome["winner_side"]),
        "holdout": holdout,
    }


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path,
                    default=Path("configs/imitation.json"))
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args(argv[1:])
    config = json.loads(args.config.read_text(encoding="utf-8"))

    selection = _load_selection(config)
    out_dir = Path(config["dataset_dir"])
    out_dir.mkdir(exist_ok=True)
    print(f"imitation dataset: {len(selection)} games "
          f"(classes={config['outcome_classes']}) -> {out_dir}",
          flush=True)

    t0 = time.time()
    jobs = [(p, o, str(out_dir), float(config["holdout_fraction"]))
            for p, o in selection]
    n_err = 0
    rows = []
    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as mf, \
            Pool(args.workers) as pool:
        for i, row in enumerate(
                pool.imap_unordered(_build_one, jobs, chunksize=20), 1):
            if row is None or "error" in row:
                n_err += 1
                log.warning("build error: %s", row)
                continue
            rows.append(row)
            mf.write(json.dumps(row) + "\n")
            if i % 2000 == 0:
                rate = i / (time.time() - t0)
                print(f"  [{i}/{len(jobs)}] err={n_err} {rate:.1f}/s "
                      f"eta={int((len(jobs)-i)/rate/60)}min", flush=True)
    # The trainer's existing value-subsampling path reads
    # value_corpus_index.jsonl (file / winner / n_commands); emit it
    # from the same rows so outcome-supervised value training works
    # with zero trainer-side special-casing.
    with open(out_dir / "value_corpus_index.jsonl", "w",
              encoding="utf-8") as vf:
        for r in rows:
            vf.write(json.dumps({
                "file": r["file"],
                "winner": r["winner_side"],
                "n_commands": r["n_commands"],
            }) + "\n")
    print(f"BUILD_DONE in {(time.time()-t0)/60:.1f}min "
          f"({len(rows)} games, errors={n_err})", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main(sys.argv))
