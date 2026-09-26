"""The unit-type vocabulary a fresh network starts from.

The type embedding has `MAX_UNIT_TYPES` rows, and the last is the
overflow row every name without a row of its own shares
(wesnoth_ai/encoder.py). A fresh vocabulary holds exactly the unit types
that can take part in a game we build or rebuild, so each has its own
row: the default era's recruits and leaders, the units the scenarios of
the pool and of the corpus place, and every type they advance into.
Seeding refuses a set that would reach the overflow row.

A variation of a reachable type (`Walking Corpse:swimmer`, raised by a
plague kill of a merman; `Soulless:bat`) shares its base type's row: the
unit features carry its hit points and moves, and the 48 variations of
unit_stats.json would not fit the rows left. The seeded vocabulary is
then frozen, so a name it does not know takes the overflow row, with a
warning, on every encode path; before 2026-09-26 the pre-encoder sent a
plague corpse's variation to the overflow row while the live path gave
it a fresh row that training never updated.

    python tools/unit_vocab.py        # the set's size and the rows left
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from wesnoth_ai.encoder import MAX_UNIT_TYPES, names_on_overflow_row  # noqa: E402
from wesnoth_ai.paths import UNIT_STATS_PATH  # noqa: E402


def _faction_types() -> Set[str]:
    """Every recruit and every possible leader of the default era."""
    from wesnoth_ai.rules.scenario_pool import load_factions
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


def reachable_unit_types(unit_stats: Path = UNIT_STATS_PATH) -> List[str]:
    """The unit types that can take part in a game we build or rebuild,
    sorted by name."""
    units = json.loads(Path(unit_stats).read_text(encoding="utf-8"))["units"]
    return sorted(advancement_closure(_faction_types() | _placed_types(), units))


def variation_aliases(names: Iterable[str], units: Dict[str, Dict]) -> Dict[str, str]:
    """Each `Base:variation` type of `units` whose base is in `names`, to
    its base."""
    bases = set(names)
    return {name: name.split(":", 1)[0] for name in units
            if ":" in name and name.split(":", 1)[0] in bases}


def seed_vocab(encoder, unit_stats: Path = UNIT_STATS_PATH) -> None:
    """Give every reachable unit type of a fresh encoder its own row, in
    name order, point each variation of one at its base type's row, and
    freeze the vocabulary. Refuses a set that would reach the overflow
    row."""
    type_to_id = encoder.unit_type_to_id
    names = reachable_unit_types(unit_stats)
    for name in names:
        if name not in type_to_id:
            type_to_id[name] = len(type_to_id)
    shared = names_on_overflow_row(type_to_id)
    if shared:
        raise ValueError(f"{len(type_to_id)} unit types for {MAX_UNIT_TYPES - 1} rows of the "
                         f"type embedding: {len(shared)} would share the overflow row "
                         f"({', '.join(shared[:5])}, ...)")
    units = json.loads(Path(unit_stats).read_text(encoding="utf-8"))["units"]
    for alias, base in sorted(variation_aliases(names, units).items()):
        type_to_id.setdefault(alias, type_to_id[base])
    encoder.freeze_vocab()


def main() -> int:
    names = reachable_unit_types()
    units = json.loads(UNIT_STATS_PATH.read_text(encoding="utf-8"))["units"]
    aliases = variation_aliases(names, units)
    print(f"{len(names)} reachable unit types; {MAX_UNIT_TYPES - 1 - len(names)} of the "
          f"{MAX_UNIT_TYPES - 1} named rows left; {len(aliases)} variations share their "
          f"base type's row")
    return 0


if __name__ == "__main__":
    sys.exit(main())
