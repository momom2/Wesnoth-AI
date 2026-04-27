"""Auto-derived list of competitive 2p Wesnoth scenarios.

Single source of truth for "what counts as a competitive 2p duel
replay" in the supervised-bootstrap pipeline. Built at import time by
scanning the ships-with `2p_*.cfg` files in
`wesnoth_src/data/multiplayer/scenarios/` and pulling each file's
top-level `[multiplayer] id=` value. Two PvE scenarios that share the
`2p_` filename prefix (Dark Forecast, Isle of Mists) are filtered out
by id.

This replaces a hand-curated set that drifted from reality — it had
a `multiplayer_Wilderlands` typo (the real 5p map id is
`multiplayer_The_Wilderlands`), an `multiplayer_Isars_Cross` entry
that's actually a 4p scenario, and was missing
`multiplayer_Ruined_Passage`. Now the list IS the source tree:
re-checking out a different Wesnoth tag (or pinning to a later 1.18
patch) automatically refreshes the allowlist.

Pinning: `wesnoth_src/` is checked out at 1.18.4 (see CLAUDE.md
"Wesnoth source pinning").
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Set


log = logging.getLogger("scenarios")

WESNOTH_SRC = Path(__file__).resolve().parent.parent / "wesnoth_src"
SCENARIO_DIR = WESNOTH_SRC / "data" / "multiplayer" / "scenarios"


# Co-op / PvE scenarios that ship as `2p_*.cfg` but are objectively
# non-competitive (player + AI co-op defending against waves of enemy
# AI). Excluded by id.
_PVE_2P_IDS: Set[str] = {
    "multiplayer_2p_Dark_Forecast",
    "multiplayer_2p_Isle_of_Mists",
}


# Match `id=multiplayer_*` anywhere in the file. Every ships-with 2p
# scenario sets `[multiplayer] id=multiplayer_<MapName>` once, and
# unrelated `id=` attrs (e.g. `[unit] id=Foo`, `[status] id=bar`) use
# different shapes that don't start with `multiplayer_`. So a global
# search is safe and bypasses macro-definition false stops.
_MP_ID_RE = re.compile(r'\bid\s*=\s*"?(multiplayer_[A-Za-z0-9_]+)"?')


def _scrape_ships_with_2p_ids() -> Set[str]:
    """Walk `wesnoth_src/data/multiplayer/scenarios/2p_*.cfg`, harvest
    each file's `[multiplayer] id=multiplayer_*`, drop the two PvE
    entries. Returns an empty set (and warns once) if wesnoth_src
    isn't accessible — that'd mean the whole pipeline is
    misconfigured."""
    if not SCENARIO_DIR.exists():
        log.warning(
            "scenarios: %s not found — competitive_2p allowlist is "
            "EMPTY. Check the wesnoth_src checkout.", SCENARIO_DIR,
        )
        return set()
    ids: Set[str] = set()
    for cfg in sorted(SCENARIO_DIR.glob("2p_*.cfg")):
        try:
            text = cfg.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            log.warning("scenarios: skip %s: %s", cfg.name, e)
            continue
        # First match wins. There's only one `id=multiplayer_*` per
        # ships-with cfg.
        m = _MP_ID_RE.search(text)
        if not m:
            log.debug("scenarios: no multiplayer_* id in %s", cfg.name)
            continue
        ids.add(m.group(1))
    ids -= _PVE_2P_IDS
    return ids


COMPETITIVE_2P_SCENARIOS: Set[str] = _scrape_ships_with_2p_ids()


def is_competitive_2p(scenario_id: str) -> bool:
    return scenario_id in COMPETITIVE_2P_SCENARIOS


__all__ = ["COMPETITIVE_2P_SCENARIOS", "is_competitive_2p"]
