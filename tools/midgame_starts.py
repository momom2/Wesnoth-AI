"""Mid-game self-play starts from the human corpus (2026-07-12).

The stalemate is an equilibrium of self-play at current skill: fresh
ladder games never reach contact, so decisive terminals -- the only
honest value signal -- never occur. Rather than relabel draws
(distorting the objective), we change WHICH STATES get experienced:
a fraction of training games starts from a human game's position at
a uniform-random turn and is played out by self-play. Decisive
outcomes become reachable at ladder scale, and the learnable
frontier walks backward from contact toward the opening.

Hyperparameters (user 2026-07-12): the mix fraction lives in the
rollout loops (--midgame-ratio); the cut turn is uniform over
[1, end_turn of the sampled game].

The corpus is the value-corpus export (replays_dataset/): one
json.gz of extracted commands per game + value_corpus_index.jsonl.
Training boxes already download it (onstart anchor build).
"""

from __future__ import annotations

import copy
import gzip
import json
import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger("midgame_starts")

_INDEX_CACHE: dict = {}


def _load_index(dataset_dir: Path) -> List[dict]:
    key = str(dataset_dir)
    if key not in _INDEX_CACHE:
        idx = dataset_dir / "value_corpus_index.jsonl"
        rows = []
        if idx.is_file():
            with idx.open(encoding="utf-8") as f:
                rows = [json.loads(l) for l in f if l.strip()]
        _INDEX_CACHE[key] = rows
    return _INDEX_CACHE[key]


def midgame_available(dataset_dir: Path) -> bool:
    return bool(_load_index(dataset_dir))


def sample_midgame_start(
    rng: random.Random,
    dataset_dir: Path,
) -> Optional[Tuple["GameState", str, int, int]]:
    """Return (game_state, scenario_id, cut_turn, begin_side) for
    one uniformly
    sampled corpus game cut at a uniform-random turn in
    [1, end_turn], or None if the corpus is unavailable / the game
    fails to reconstruct (caller falls back to a fresh start).

    The cut lands at the first init_side of a PLAYER side with
    turn_number >= cut_turn, i.e. a clean side-turn boundary; the
    returned state is exactly what the side to move would face.
    Scenario events are ALREADY APPLIED (reconstruction runs them);
    wrap in WesnothSim(..., apply_scenario_events=False).
    """
    from tools.replay_dataset import (_apply_command,
                                      _build_initial_gamestate,
                                      _setup_scenario_events)
    rows = _load_index(dataset_dir)
    if not rows:
        return None
    row = rng.choice(rows)
    gz = dataset_dir / row["file"]
    try:
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            data = json.load(f)
        cmds = data.get("commands", [])
        # Pass 1 (cheap, no state): the game's turn count = number of
        # side-1 init_side commands.
        end_turn = sum(1 for c in cmds if c and c[0] == "init_side"
                       and len(c) > 1 and c[1] == 1)
        if end_turn < 2:
            return None
        cut = rng.randint(1, end_turn)
        gs = _build_initial_gamestate(data)
        _setup_scenario_events(gs, data.get("scenario_id", ""))
        # Pass 2: apply commands until the cut boundary. Record WHICH
        # side's init_side we stop at: the sim must resume THAT
        # side's turn (WesnothSim(begin_side=...)) or the position
        # gets a tempo bias (review 2026-07-12 C1). Because
        # turn_number increments on init_side(1), the boundary is in
        # practice always an init_side(2).
        begin_side = None
        boundary_idx = None
        for i, cmd in enumerate(cmds):
            if (cmd and cmd[0] == "init_side"
                    and gs.global_info.turn_number >= cut
                    and gs.global_info.current_side in (1, 2)
                    and len(cmd) > 1 and cmd[1] in (1, 2)):
                begin_side = int(cmd[1])
                boundary_idx = i
                break
            _apply_command(gs, cmd)
        if begin_side is None:
            # cut == end_turn on a draw/timeout game: the loop never
            # breaks and the position is 0-2 turns from the cap --
            # worthless as a start (review m2). Caller falls back.
            return None
        # Both leaders must be alive to continue (decisive human games
        # cut at the very end could hand the sim a finished position).
        alive = {u.side for u in gs.map.units if u.is_leader}
        if not {1, 2} <= alive:
            return None
        # Provenance for validation exports: enough to re-walk the
        # human prefix (commands[:boundary_idx]) and splice it in
        # front of the sim continuation
        # (tools/validation_exports.export_midgame_replay).
        provenance = {
            "dataset_dir": str(dataset_dir),
            "file": row["file"],
            "boundary_idx": boundary_idx,
            "begin_side": begin_side,
        }
        return (copy.deepcopy(gs), data.get("scenario_id", ""),
                gs.global_info.turn_number, begin_side, provenance)
    except Exception as e:                            # noqa: BLE001
        log.warning(f"midgame sample failed for {row.get('file')}: {e}")
        return None
