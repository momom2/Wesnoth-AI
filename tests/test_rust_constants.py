"""Constants the Rust core writes out a second time, compared with the
Python values they copy.

The Rust kernels (rust/wesnoth_core/src) restate tables that Python
owns -- the combat kernel's flag order, the hide abilities, the damage
types, the time-of-day cycle, the healing amounts -- and nothing
compared the two copies: a Python edit that missed its Rust twin would
surface only as a divergence in a corpus sweep. These tests read the
Rust SOURCE, so they run without a wheel and catch a drift before any
wheel is built (the pattern of tests/test_time_of_day_features.py).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import pathfind_sim, replay_dataset, terrain_resolver  # noqa: E402
from wesnoth_ai import classes, combat, visibility  # noqa: E402

RUST_SRC = Path(__file__).parent.parent / "rust" / "wesnoth_core" / "src"


def _source(name: str) -> str:
    return (RUST_SRC / name).read_text(encoding="utf-8")


def _const(file: str, name: str) -> str:
    """The value text of `[pub] const NAME: TYPE = VALUE;` in a Rust file."""
    match = re.search(rf"\bconst {name}: [^=]+= (.*?);", _source(file), re.S)
    assert match, f"{file} declares no const {name}"
    return match.group(1)


def _strings(file: str, name: str) -> list:
    return re.findall(r'"([^"]*)"', _const(file, name))


def _ints(file: str, name: str) -> list:
    return [int(v) for v in re.findall(r"-?\d+", _const(file, name))]


def _int(file: str, name: str) -> int:
    return int(_const(file, name).strip())


def _ratio(file: str, name: str) -> float:
    """A float written as `a / b` or a plain literal."""
    parts = [float(p) for p in _const(file, name).split("/")]
    return parts[0] / parts[1] if len(parts) == 2 else parts[0]


@pytest.mark.parametrize("rust, python", [
    (lambda: _strings("core_attack.rs", "UNIT_FLAG_NAMES"), lambda: list(combat._UNIT_FLAGS)),
    (lambda: _int("combat.rs", "UNIT_FLAGS"), lambda: len(combat._UNIT_FLAGS)),
    (lambda: _strings("core_attack.rs", "DAMAGE_TYPES"), lambda: list(combat.DAMAGE_TYPES)),
    (lambda: set(_strings("core_attack.rs", "UNPLAGUEABLE_TRAITS")),
     lambda: set(replay_dataset._UNPLAGUEABLE_TRAITS)),
    (lambda: _int("core_attack.rs", "ILLUMINATION"), lambda: replay_dataset.ILLUMINATES_VALUE),
    (lambda: set(_strings("core_move.rs", "HIDE_ABILITIES")), lambda: set(visibility._AMBUSH_ABILITIES)),
    (lambda: set(_strings("core_observe.rs", "HIDE_ABILITIES")), lambda: set(visibility._AMBUSH_ABILITIES)),
    (lambda: _int("core_step.rs", "REST_HEAL_AMOUNT"), lambda: combat.REST_HEAL_AMOUNT),
    (lambda: _int("core_step.rs", "REGENERATE_AMOUNT"), lambda: combat.REGENERATE_AMOUNT),
    (lambda: _int("core_step.rs", "POISON_AMOUNT"), lambda: combat.POISON_AMOUNT),
    (lambda: _int("combat.rs", "COMBAT_EXPERIENCE"), lambda: combat.COMBAT_EXPERIENCE),
    (lambda: _int("combat.rs", "KILL_EXPERIENCE"), lambda: combat.KILL_EXPERIENCE),
    (lambda: _int("combat.rs", "MAX_LIMINAL_BONUS"), lambda: combat.MAX_LIMINAL_BONUS),
    (lambda: _ints("core.rs", "DEFAULT_CYCLE"), lambda: [lb for _, lb in combat.TOD_DEFAULT_CYCLE]),
    (lambda: _strings("core.rs", "TOD_NAMES"), lambda: [name for name, _ in combat.TOD_DEFAULT_CYCLE]),
    (lambda: _int("lib.rs", "UNREACHABLE"), lambda: pathfind_sim.UNREACHABLE),
    (lambda: _int("core_move.rs", "UNREACHABLE"), lambda: pathfind_sim.UNREACHABLE),
    (lambda: _int("core_move.rs", "UNREACHABLE"), lambda: terrain_resolver.UNREACHABLE_COST),
    (lambda: _ratio("lib.rs", "SUBCOST_SCALE"), lambda: pathfind_sim._SUBCOST_SCALE),
], ids=[
    "unit flag names", "unit flag count", "damage types", "unplagueable traits",
    "illumination", "hide abilities (move)", "hide abilities (observe)", "rest heal",
    "regeneration", "poison", "combat experience", "kill experience", "liminal bonus",
    "time-of-day bonuses", "time-of-day names", "unreachable (reach)",
    "unreachable (core move)", "unreachable (terrain resolver)", "subcost scale",
])
def test_the_rust_copy_equals_the_python_value(rust, python):
    assert rust() == python()


def test_each_flag_index_names_its_python_flag():
    """combat.rs reads the flag vector by `F_<NAME>` indices; each must
    point at the flag of that name in `combat._UNIT_FLAGS`, and every
    flag must have one."""
    indices = {name.lower(): int(i) for name, i in
               re.findall(r"\bconst F_(\w+): usize = (\d+);", _source("combat.rs"))}
    assert indices == {flag: i for i, flag in enumerate(combat._UNIT_FLAGS)}


def test_the_alignment_codes_mean_the_same_alignments():
    """The kernel's `combat_modifier` switches on the alignment code the
    Python side sends (`combat._ALIGNMENT_CODE` for the combat snapshot,
    the Alignment enums for GameCore's unit and type fields)."""
    body = re.search(r"fn combat_modifier\(.*?\n\}", _source("combat.rs"), re.S).group(0)
    arms = {
        "LAWFUL": r"(\d+) => lawful_bonus,",
        "NEUTRAL": r"(\d+) => 0,",
        "CHAOTIC": r"(\d+) => -lawful_bonus,",
        "LIMINAL": r"(\d+) => MAX_LIMINAL_BONUS - lawful_bonus\.abs\(\),",
    }
    rust = {name: int(re.search(arm, body).group(1)) for name, arm in arms.items()}
    assert rust == combat._ALIGNMENT_CODE
    assert rust == {a.name: a.value for a in combat.Alignment}
    assert rust == {a.name: a.value for a in classes.Alignment}


def test_the_default_cycle_length_is_the_turn_modulus():
    """`GameCore::tod_index` writes the cycle's length as a literal,
    as `% 6` or `.rem_euclid(6)`."""
    body = re.search(r"fn tod_index\(.*?\n    \}", _source("core_step.rs"), re.S).group(0)
    modulus = re.search(r"(?:% |rem_euclid\()(\d+)\)", body)
    assert modulus, body
    assert int(modulus.group(1)) == len(combat.TOD_DEFAULT_CYCLE)
