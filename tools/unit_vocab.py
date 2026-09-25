"""The unit-type vocabulary a fresh network starts from.

The type embedding has `MAX_UNIT_TYPES` rows, and the last is the
overflow row every name without a row of its own shares
(wesnoth_ai/encoder.py). A fresh vocabulary holds exactly the unit types
that can take part in a game we build or rebuild, so each has its own
row: the default era's recruits and leaders, the units the scenarios of
the pool and of the corpus place, and every type they advance into.
Seeding refuses a set that would reach the overflow row.

    python tools/unit_vocab.py        # the set's size and the rows left
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from wesnoth_ai.encoder import MAX_UNIT_TYPES, names_on_overflow_row  # noqa: E402

UNIT_STATS = ROOT / "unit_stats.json"


def _faction_types() -> Set[str]:
    """Every recruit and every possible leader of the default era."""
    from tools.scenario_pool import load_factions
    factions = load_factions()
    names: Set[str] = set()
    for f in (factions if isinstance(factions, list) else factions.values()):
        names.update(f.recruit)
        names.update(f.leader_pool)
        names.update(f.random_leader_pool)
    return names


def _placed_types() -> Set[str]:
    """Every unit type a [unit] or [side] of a scenario we build or
    rebuild names (tools/analysis/scenario_surface.py's list)."""
    from tools.analysis.expansion_diff import _scenario_block
    from tools.analysis.scenario_surface import CORPUS_SCENARIOS
    from tools.scenario_events import load_scenario_wml
    names: Set[str] = set()

    def walk(node) -> None:
        for child in node.children:
            if child.tag in ("unit", "side") and "type" in child.attrs:
                names.update(t.strip() for t in str(child.attrs["type"]).split(",") if t.strip())
            walk(child)

    for scenario_id in CORPUS_SCENARIOS:
        block = _scenario_block(load_scenario_wml(scenario_id))
        if block is not None:
            walk(block)
    return names


def advancement_closure(names: Iterable[str], units: Dict[str, Dict]) -> Set[str]:
    """`names` and every type they advance into, over `advances_to`;
    names without a unit_stats entry are left out."""
    out: Set[str] = set()
    todo = list(names)
    while todo:
        name = todo.pop()
        if name in out or name not in units:
            continue
        out.add(name)
        targets = units[name].get("advances_to") or []
        if isinstance(targets, str):
            targets = [t.strip() for t in targets.split(",")]
        todo.extend(t for t in targets if t and t != "null")
    return out


def reachable_unit_types(unit_stats: Path = UNIT_STATS) -> List[str]:
    """The unit types that can take part in a game we build or rebuild,
    sorted by name."""
    units = json.loads(Path(unit_stats).read_text(encoding="utf-8"))["units"]
    return sorted(advancement_closure(_faction_types() | _placed_types(), units))


def seed_vocab(encoder, unit_stats: Path = UNIT_STATS) -> None:
    """Give every reachable unit type of a fresh encoder its own row, in
    name order. Refuses a set that would reach the overflow row."""
    type_to_id = encoder.unit_type_to_id
    for name in reachable_unit_types(unit_stats):
        if name not in type_to_id:
            type_to_id[name] = len(type_to_id)
    shared = names_on_overflow_row(type_to_id)
    if shared:
        raise ValueError(f"{len(type_to_id)} unit types for {MAX_UNIT_TYPES - 1} rows of the "
                         f"type embedding: {len(shared)} would share the overflow row "
                         f"({', '.join(shared[:5])}, ...)")


def main() -> int:
    names = reachable_unit_types()
    print(f"{len(names)} reachable unit types; {MAX_UNIT_TYPES - 1 - len(names)} of the "
          f"{MAX_UNIT_TYPES - 1} named rows left")
    return 0


if __name__ == "__main__":
    sys.exit(main())
