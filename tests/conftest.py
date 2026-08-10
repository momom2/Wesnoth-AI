"""Path bootstrap for the test suite.

The tests moved out of the repo root into tests/ (2026-07-23). Core
library code now lives in the `wesnoth_ai` package at the repo root and
the scripts under `tools/`; both are imported as top-level names by the
tests (`from wesnoth_ai.model import ...`, `import mcts`). Put the repo
root, `tools/`, and this tests/ dir on sys.path so those imports resolve
no matter how pytest is invoked. The per-file `sys.path.insert` lines
remain for running a test file directly as a script; this makes them
redundant under pytest, not required.

The repo-root conftest.py stays responsible for `collect_ignore_glob`
(vendored trees that must not be collected).
"""
import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
_ROOT = _TESTS.parent

for _p in (_ROOT, _ROOT / "tools", _TESTS):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


def recruit_action_for(unit_type: str):
    """Test helper: build a recruit-action dict for `unit_type` on a
    free castle hex adjacent to the side-to-move's leader, or None if
    the leader isn't on a keep / no free castle hex / not in the
    recruit list. (Salvaged from the deleted tools/openers.py
    `recruit_type` -- 2026-08-10 F6 ruling -- because two RNG/chance
    tests use it to engineer deterministic recruit actions.)"""
    from typing import Dict, Optional
    from wesnoth_ai.classes import GameState, Position, TerrainModifiers
    from tools.abilities import hex_neighbors

    def _move(state: GameState, side: int) -> Optional[Dict]:
        leader = next(
            (u for u in state.map.units if u.side == side and u.is_leader),
            None,
        )
        if leader is None:
            return None
        on_keep = False
        for h in state.map.hexes:
            if (h.position.x, h.position.y) == (leader.position.x,
                                                leader.position.y):
                on_keep = TerrainModifiers.KEEP in h.modifiers
                break
        if not on_keep:
            return None
        side_idx = side - 1
        if not (0 <= side_idx < len(state.sides)):
            return None
        if unit_type not in state.sides[side_idx].recruits:
            return None
        occupied = {(u.position.x, u.position.y) for u in state.map.units}
        for nx, ny in hex_neighbors(leader.position.x, leader.position.y):
            if (nx, ny) in occupied:
                continue
            for h in state.map.hexes:
                if (h.position.x, h.position.y) != (nx, ny):
                    continue
                if TerrainModifiers.CASTLE in h.modifiers:
                    return {
                        "type":       "recruit",
                        "unit_type":  unit_type,
                        "target_hex": Position(nx, ny),
                    }
                break
        return None
    return _move
