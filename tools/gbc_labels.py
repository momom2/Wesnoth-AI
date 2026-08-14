"""GBC hindsight label machinery + rung-0a yield measurement.

Goal-Basis Completion (docs/gbc_spec.md) trains heads that predict
P(event within k game turns | state). The labels are FREE: a forward
scan over stored trajectories marks every death / village flip /
level-up with its turn, position, and per-side observability. This
module is the single source of those labels (rung 0a's yield stats,
rung 1's training stream, and the exact-oracle gates all read it).

Contracts (docs/gbc_spec.md par.2, review amendments):
  * Fog-censored CONFIRMED achievement, amendment A1: the observer is
    the SIDE-TO-MOVE at the anchor state, for every goal regardless
    of entity ownership. An event counts for an observer only if the
    event's hex is visible to that side when it happens (the
    you-saw-it-happen definition; a kill in fog is not a label).
  * Entities keyed by `unit.id` / village (x, y) -- NEVER slot
    indices (the fa95da5 stored-index failure class). Advancement
    preserves unit.id (`_rebuild_unit` copies base fields), so a
    level-up is not mistaken for a death.
  * k counts GAME turns (global_info.turn_number), k=1 meaning
    "within the anchor's own turn": window = turns [t0, t0+k-1].

Rung-0a CLI (laptop, CPU-only, ~1-3 s/game):

    python tools/gbc_labels.py --games 500 \
        --dataset replays_dataset_imitation
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from wesnoth_ai.visibility import (  # noqa: E402
    units_visible_to, visible_hexes_for,
)
# The event/fingerprint/diff core moved to wesnoth_ai/gbc.py
# (2026-08-14, production integration) -- this scanner and the
# training-time labeler share ONE implementation. Aliases keep this
# module's public names stable for tests/callers.
from wesnoth_ai.gbc import (  # noqa: E402
    Event, diff_events as _diff_events_core,
    unit_fingerprint as _unit_fp, village_fingerprint as _village_fp,
)

log = logging.getLogger("gbc_labels")

PREDICATES = ("dies", "flips", "levels")
K_HORIZONS = (1, 2, 3)

__all__ = ["Event", "Anchor", "GameScan", "scan_game",
           "labels_for_anchor", "rung0a", "_unit_fp", "_village_fp",
           "_diff_events"]


@dataclass
class Anchor:
    """One side-turn start: the state class GBC heads anchor on.
    `goals` maps goal key -> (predicate, entity_side,
    visible_at_anchor) for the mover's fog-honest goal roster."""
    seq:   int
    turn:  int
    side:  int                       # side to move = the observer (A1)
    goals: Dict[Tuple, Tuple[str, int, bool]] = field(
        default_factory=dict)


@dataclass
class GameScan:
    anchors: List[Anchor] = field(default_factory=list)
    events:  List[Event] = field(default_factory=list)
    n_turns: int = 0


def _village_hexes(gs) -> List[Tuple[int, int]]:
    """All village positions from terrain truth (owned or not) --
    the goal roster must include never-captured villages, or first
    captures would be unlabelable."""
    from wesnoth_ai.classes import Terrain
    return [(h.position.x, h.position.y) for h in gs.map.hexes
            if Terrain.VILLAGE in h.terrain_types]


def _diff_events(seq: int, prev_u: Dict, prev_v: Dict,
                 gs) -> List[Event]:
    """Alias over the shared core (wesnoth_ai/gbc.py)."""
    return _diff_events_core(seq, prev_u, prev_v, gs)


def scan_game(gz_path: Path, on_anchor=None) -> GameScan:
    """One pass over a stored replay: side-turn-start anchors with
    fog-honest goal rosters, plus every event with per-side
    observability. Pure CPU (bit-exact reconstruction; no model).

    `on_anchor(gs, anchor)`: optional callback fired at each anchor
    with the LIVE (mutating) state -- consumers needing per-anchor
    model quantities (rung 0b's value read) compute them here
    instead of snapshotting states."""
    from tools.replay_dataset import iter_replay_pairs_with_state

    scan = GameScan()
    prev_turnside: Optional[Tuple[int, int]] = None
    prev_u: Dict = {}
    prev_v: Dict = {}
    villages: Optional[List[Tuple[int, int]]] = None
    seq = 0
    for gs, _ai in iter_replay_pairs_with_state(gz_path):
        if seq > 0:
            scan.events.extend(_diff_events(seq, prev_u, prev_v, gs))
        if villages is None:
            villages = _village_hexes(gs)
        gi = gs.global_info
        turnside = (int(gi.turn_number), int(gi.current_side))
        if turnside != prev_turnside and gi.current_side in (1, 2):
            prev_turnside = turnside
            anchor = Anchor(seq=seq, turn=turnside[0],
                            side=turnside[1])
            vis_hex = frozenset(visible_hexes_for(gs, anchor.side))
            for u in units_visible_to(gs, anchor.side):
                anchor.goals[("u", u.id)] = ("dies", u.side, True)
            owners = _village_fp(gs)
            for pos in villages:
                anchor.goals[("v",) + tuple(pos)] = (
                    "flips", int(owners.get(pos, 0)),
                    tuple(pos) in vis_hex)
            scan.anchors.append(anchor)
            if on_anchor is not None:
                on_anchor(gs, anchor)
        prev_u = _unit_fp(gs)
        prev_v = _village_fp(gs)
        scan.n_turns = int(gs.global_info.turn_number)
        seq += 1
    return scan


def labels_for_anchor(anchor: Anchor, events: List[Event],
                      ) -> List[Tuple[Tuple, str, int, int, bool]]:
    """(goal_key, predicate, k, y, visible_at_anchor) rows for one
    anchor, fog-censored per A1: only events OBSERVED by the
    anchor's side-to-move count as achieved. `levels` goals ride the
    unit roster (any visible unit is a levels goal too)."""
    rows = []
    by_key = defaultdict(list)
    for e in events:
        by_key[(e.predicate, e.key)].append(e)
    for key, (pred, _eside, vis0) in anchor.goals.items():
        preds = (("dies", "levels") if key[0] == "u" else ("flips",))
        for p in preds:
            for k in K_HORIZONS:
                hit = any(
                    e.seq > anchor.seq
                    and anchor.turn <= e.turn <= anchor.turn + k - 1
                    and anchor.side in e.observed_by
                    for e in by_key[(p, key)])
                rows.append((key, p, k, int(hit), vis0))
    return rows


# ---------------------------------------------------------------------
# Rung 0a: label yield per (predicate, k, fog stratum)
# ---------------------------------------------------------------------

def rung0a(dataset: Path, n_games: int, seed: int = 0) -> Dict:
    files = sorted(dataset.glob("*.json.gz"))
    if not files:
        raise SystemExit(f"no replays under {dataset}")
    rng = random.Random(seed)
    rng.shuffle(files)
    files = files[:n_games]
    pos = defaultdict(int)
    tot = defaultdict(int)
    scanned = errors = 0
    for i, gz in enumerate(files):
        try:
            scan = scan_game(gz)
        except Exception as e:  # noqa: BLE001
            errors += 1
            log.warning(f"{gz.name}: scan failed: {e!r}")
            continue
        scanned += 1
        for anchor in scan.anchors:
            for _key, p, k, y, vis0 in labels_for_anchor(
                    anchor, scan.events):
                stratum = "vis" if vis0 else "fog"
                tot[(p, k, stratum)] += 1
                pos[(p, k, stratum)] += y
        if (i + 1) % 50 == 0:
            log.info(f"scanned {i + 1}/{len(files)}")
    out = {"scanned": scanned, "errors": errors, "buckets": {}}
    for key in sorted(tot):
        p, k, stratum = key
        out["buckets"][f"{p}/k{k}/{stratum}"] = {
            "n": tot[key], "positives": pos[key],
            "rate": round(pos[key] / tot[key], 4) if tot[key] else None,
        }
    return out


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path,
                    default=Path("replays_dataset_imitation"))
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s "
                               "%(message)s", datefmt="%H:%M:%S")
    stats = rung0a(args.dataset, args.games, args.seed)
    text = json.dumps(stats, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    # Pre-registered gate (docs/gbc_spec.md par.6 rung 0a): every
    # (predicate, k) bucket needs >=5% positives after stratification,
    # else prune the vocabulary.
    weak = [b for b, d in stats["buckets"].items()
            if d["rate"] is not None and d["rate"] < 0.05]
    print(f"\n0a gate: buckets under 5% positives: {weak or 'NONE'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
