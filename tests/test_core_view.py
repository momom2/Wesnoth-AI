"""The view's units the core builds (`GameCore.units_view`,
rust/wesnoth_core/src/core_view.rs) are the ones `unit_from_fields`
builds from `units_export`: the same classes, the same `__dict__` keys in
the same order, the same values of the same Python types, the enum
members themselves, on played positions and on a unit carrying every
underscore attribute."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from wesnoth_ai.game_core import game_core_class  # noqa: E402

needs_core = pytest.mark.skipif(game_core_class() is None, reason="wesnoth_core wheel absent or stale")


def _same(a, b, where: str) -> None:
    """Deep equality with the types: dataclass instances by their
    `__dict__` in order, containers by type and elements, enum members
    by identity."""
    import enum
    assert type(a) is type(b), (where, type(a), type(b))
    if isinstance(a, enum.Enum):
        assert a is b, (where, a, b)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), (where, a, b)
        for k, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{where}[{k}]")
    elif isinstance(a, dict):
        assert list(a) == list(b), (where, list(a), list(b))
        for k in a:
            _same(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, (set, frozenset)):
        assert a == b and all(type(x) is str for x in a), (where, a, b)
        assert list(a) == list(b), (where, "iteration order")
    elif hasattr(a, "__dict__"):
        _same(vars(a), vars(b), f"{where}<{type(a).__name__}>")
    elif hasattr(type(a), "__slots__"):
        for slot in type(a).__slots__:
            _same(getattr(a, slot), getattr(b, slot), f"{where}.{slot}")
    else:
        assert a == b, (where, a, b)


def _check(cs, where: str) -> dict:
    """Compare the two constructions over every unit of `cs`; the
    underscore attributes seen, counted."""
    from wesnoth_ai.game_core import _view_types, unit_from_fields
    got = cs.core.units_view(_view_types())
    want = [unit_from_fields(d) for d in cs.core.units_export()]
    assert len(got) == len(want) > 0
    seen = {}
    for g, w in zip(got, want):
        _same(g, w, f"{where}:{w.id}")
        for k in vars(w):
            if k.startswith("_"):
                seen[k] = seen.get(k, 0) + 1
    return seen


@needs_core
def test_the_core_builds_the_units_unit_from_fields_builds():
    from helpers.core_positions import core_positions
    from wesnoth_ai.game_core import core_of
    seen = {}
    for k, gs in enumerate(core_positions()):
        for key, n in _check(core_of(gs), f"position {k}").items():
            seen[key] = seen.get(key, 0) + n
    assert seen.get("_trait_order"), seen


@needs_core
def test_every_underscore_attribute_rides_the_view():
    from helpers.core_positions import core_positions
    from tools.replay_extract import WMLNode
    from wesnoth_ai.classes import Position
    from wesnoth_ai.game_core import core_of, unit_fields, unit_from_fields
    cs = core_of(core_positions()[0]).fork()
    probe = unit_from_fields(cs.core.units_export()[0])
    occupied = {(u.position.x, u.position.y) for u in cs.to_state().map.units}
    probe.id = "probe"
    probe.position = Position(*next(p for p in cs.geometry().keys if p not in occupied))
    effect = WMLNode("effect")
    effect.attrs = {"apply_to": "hitpoints", "increase_total": "5"}
    obj = WMLNode("object")
    obj.attrs = {"id": "probe_object"}
    obj.children = [effect]
    probe._object_effects = [obj]
    probe._defense_table = {"Gg": 40, "Hh": 50}
    probe._pickadvance = ["Swordsman"]
    probe._trait_order = ["quick", "resilient"]
    probe._feeding_count = 2
    probe._wml_role = "hero"
    probe._ai_guardian = True
    probe.statuses = {"poisoned", "slowed"}
    cs.core.add_unit(unit_fields(probe))
    seen = _check(cs, "probe")
    assert set(seen) == {"_object_effects", "_defense_table", "_pickadvance", "_trait_order",
                         "_feeding_count", "_wml_role", "_ai_guardian"}, seen
