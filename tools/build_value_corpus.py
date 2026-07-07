"""Build a value-head training corpus from downloaded human replays.

Pipeline (raw .bz2 from replays.wesnoth.org -> labeled game index):

  1. Header gates (cheap, no full parse; reuses filter_replays):
     - version 1.18.x
     - era_id default / era_default, no campaign, no mods requested
     - exactly 2 sides, both human-controlled
     - scenario_id in the competitive-2p whitelist (mainline default
       2p maps = the ladder pool's mainline subset)
  2. Outcome scan on the RAW text (server metadata never reaches the
     extracted command stream):
     - [surrender] command  -> the surrendering side loses
     - server [speak] "<player> has left the game" -> the FIRST
       leaver loses (standard MP-corpus convention; resigning by
       leaving). Guarded by --min-turns so aborted lobbies/remakes
       stay unlabeled.
  3. Extraction: tools.replay_extract.extract_replay -> compact
     .json.gz in the dataset dir (same format the BC pipeline and
     diff_replay consume).
  4. Corruption gate: tools.diff_replay.diff_replay must return NO
     divergences (bit-exact reconstruction machinery; anything that
     desyncs is dropped, per the "non-corrupted" requirement).
  5. Authoritative outcome: re-walk the reconstruction and check
     which side's leader is dead at the end. A leader kill OVERRIDES
     the raw-scan heuristics (it is the engine's own victory
     condition); games with neither signal are excluded.

Output: <dataset-dir>/value_corpus_index.jsonl, one row per accepted
game:

  { "file": "ab12cd34ef56.json.gz", "game_id": ..., "scenario_id": ...,
    "factions": [f1, f2], "winner": 1|2, "label_source":
    "leader_kill"|"surrender"|"leaver", "n_commands": int,
    "n_turns": int, "src": "replays_raw/2026-07-03/..." }

tools/value_corpus.py consumes this index to yield (state, z) pairs.

Usage:
    python tools/build_value_corpus.py replays_raw \
        [--out replays_dataset] [--min-turns 8] [--limit N]
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.filter_replays import parse_header
from tools.scenarios import COMPETITIVE_2P_SCENARIOS

log = logging.getLogger("build_value_corpus")

# Server-side announcement when a player disconnects/quits. Observed
# verbatim in 1.18 server replays (2026-07-07 sample):
#   message="arketiyo has left the game."
_LEFT_RE = re.compile(r'message="([^"]+) has left the game\.?"')
# A [surrender] command carries the surrendering side INSIDE the tag
# (verified on a 2026-07-01 sample file):
#   [command] undo=no [surrender] side_number=1 [/surrender] [/command]
_SURRENDER_RE = re.compile(
    r'\[surrender\]\s*\n\s*side_number="?(\d)"?', re.MULTILINE)


def _decisive_winner_from_reconstruction(gz_path: Path) -> Tuple[
        Optional[int], int]:
    """Walk the extracted game; return (winner_by_leader_death | None,
    final_turn_number). Reuses the bit-exact reconstruction machinery
    (same walk diff_replay does, without the divergence checkers --
    the caller has already required a clean diff)."""
    from tools.replay_dataset import (_apply_command,
                                      _build_initial_gamestate,
                                      _setup_scenario_events)

    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    gs = _build_initial_gamestate(data)
    _setup_scenario_events(gs, data.get("scenario_id", ""))
    for cmd in data.get("commands", []):
        _apply_command(gs, cmd)
    # Units live on the map (classes.py Map.units: Set[Unit]); dead
    # units are REMOVED from the set by the reconstruction, so
    # "leader present" == "leader alive".
    alive_leaders = {u.side for u in gs.map.units if u.is_leader}
    turn = gs.global_info.turn_number
    if alive_leaders == {1}:
        return 1, turn
    if alive_leaders == {2}:
        return 2, turn
    return None, turn        # both alive (unfinished) or both dead (?)


def _raw_outcome_scan(raw_text: str,
                      side_players: Dict[str, int]) -> Tuple[
        Optional[int], Optional[str]]:
    """(winner, label_source) from raw-file signals, or (None, None).

    Precedence here: surrender > first-leaver. (Leader-kill precedence
    over BOTH is applied by the caller.)"""
    m = _SURRENDER_RE.search(raw_text)
    if m:
        loser = int(m.group(1))
        if loser in (1, 2):
            return 3 - loser, "surrender"
    for m in _LEFT_RE.finditer(raw_text):
        side = side_players.get(m.group(1))
        if side in (1, 2):
            return 3 - side, "leaver"
        # A spectator leaving also produces this message; keep
        # scanning for the first PLAYER leaver.
    return None, None


def _side_player_map(header: dict) -> Dict[str, int]:
    """player-name -> side-number from the parsed [side] blocks."""
    out: Dict[str, int] = {}
    for i, s in enumerate(header.get("sides", []), start=1):
        side_num = int(s.get("side", i) or i)
        for key in ("player_id", "current_player", "save_id"):
            name = (s.get(key) or "").strip()
            if name:
                out.setdefault(name, side_num)
    return out


def _human_sides(header: dict) -> int:
    """Count sides a human (local or network) controls."""
    n = 0
    for s in header.get("sides", []):
        ctrl = (s.get("controller") or "").lower()
        if ctrl in ("human", "network"):
            n += 1
    return n


def build(raw_dir: Path, out_dir: Path, *, min_turns: int,
          limit: Optional[int] = None) -> Counter:
    from tools.replay_extract import extract_replay
    from tools.diff_replay import diff_replay

    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "value_corpus_index.jsonl"
    # Resume-friendly: skip sources already in the index.
    done_src = set()
    if index_path.exists():
        with index_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done_src.add(json.loads(line)["src"])
                except Exception:               # noqa: BLE001
                    pass

    stats: Counter = Counter()
    files = sorted(raw_dir.glob("**/*.bz2"))
    if limit:
        files = files[:limit]
    log.info(f"scanning {len(files)} raw replays "
             f"({len(done_src)} already indexed)")

    with index_path.open("a", encoding="utf-8") as idx:
        for i, p in enumerate(files, 1):
            if i % 200 == 0:
                log.info(f"  [{i}/{len(files)}] "
                         + " ".join(f"{k}={v}"
                                    for k, v in sorted(stats.items())))
            src = str(p.as_posix())
            if src in done_src:
                stats["skip_done"] += 1
                continue
            try:
                raw = bz2.decompress(p.read_bytes())
            except Exception:                   # noqa: BLE001
                stats["reject_bz2"] += 1
                continue
            raw_text = raw.decode("utf-8", errors="replace")

            header = parse_header(raw)
            top = header.get("top", {})
            version = top.get("version", "")
            if not version.startswith("1.18"):
                stats["reject_version"] += 1
                continue
            era = (top.get("era_id") or "").strip()
            if era not in ("default", "era_default"):
                stats["reject_era"] += 1
                continue
            if (top.get("campaign") or "").strip():
                stats["reject_campaign"] += 1
                continue
            sides = header.get("sides", [])
            if len(sides) != 2 or _human_sides(header) != 2:
                stats["reject_sides"] += 1
                continue
            scen = header.get("scenario_id") or ""
            if scen not in COMPETITIVE_2P_SCENARIOS:
                stats["reject_map"] += 1
                continue

            raw_winner, raw_source = _raw_outcome_scan(
                raw_text, _side_player_map(header))

            try:
                rep = extract_replay(p)
            except Exception as e:              # noqa: BLE001
                stats["reject_extract"] += 1
                log.debug(f"  extract err {p.name}: {e}")
                continue
            if rep is None:
                stats["reject_no_commands"] += 1
                continue

            gid = hashlib.sha1(rep["game_id"].encode()).hexdigest()[:12]
            gz_path = out_dir / f"{gid}.json.gz"
            if not gz_path.exists():
                with gzip.open(gz_path, "wt", encoding="utf-8",
                               compresslevel=6) as fw:
                    json.dump(rep, fw, separators=(",", ":"))

            try:
                divergences = diff_replay(gz_path, stop_on_first=True)
            except Exception:                   # noqa: BLE001
                divergences = ["reconstruction exception"]
            if divergences:
                stats["reject_desync"] += 1
                gz_path.unlink(missing_ok=True)
                continue

            try:
                kill_winner, n_turns = (
                    _decisive_winner_from_reconstruction(gz_path))
            except Exception:                   # noqa: BLE001
                stats["reject_outcome_walk"] += 1
                gz_path.unlink(missing_ok=True)
                continue

            if kill_winner is not None:
                winner, source = kill_winner, "leader_kill"
            elif raw_winner is not None and n_turns >= min_turns:
                winner, source = raw_winner, raw_source
            else:
                stats["reject_unlabeled" if raw_winner is None
                      else "reject_short"] += 1
                gz_path.unlink(missing_ok=True)
                continue

            idx.write(json.dumps({
                "file": gz_path.name,
                "game_id": rep["game_id"],
                "scenario_id": scen,
                "factions": rep["factions"],
                "winner": winner,
                "label_source": source,
                "n_commands": len(rep["commands"]),
                "n_turns": n_turns,
                "src": src,
            }) + "\n")
            stats["accepted"] += 1
            stats[f"label_{source}"] += 1
    return stats


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("replays_dataset"))
    ap.add_argument("--min-turns", type=int, default=8,
                    help="Heuristic labels (surrender/leaver) below "
                         "this many turns are dropped as remakes/"
                         "aborts (default 8). Leader kills always "
                         "count.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only scan the first N raw files (smoke).")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    stats = build(args.raw_dir, args.out, min_turns=args.min_turns,
                  limit=args.limit)
    log.info("== done ==")
    for k, v in sorted(stats.items()):
        log.info(f"  {k}: {v}")
    total = sum(v for k, v in stats.items() if k.startswith("reject")) \
        + stats["accepted"]
    if total:
        log.info(f"  yield: {stats['accepted']}/{total}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
