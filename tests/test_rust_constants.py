"""Constants the Rust core writes out a second time, compared with the
Python values they copy.

The Rust core (rust/wesnoth_core/src) restates tables that Python also
reads -- the damage types, the time-of-day cycle, the alignment codes,
the unreachable movement cost -- and keeps the hide abilities in two of
its files; nothing compared the copies: an edit that missed its twin
would surface only as a divergence in a corpus sweep. These tests read the
Rust SOURCE, so they run without a wheel and catch a drift before any
wheel is built (the pattern of tests/test_time_of_day_features.py).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai.rules import terrain_resolver  # noqa: E402
from wesnoth_ai import classes, combat  # noqa: E402

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


@pytest.mark.parametrize("rust, python", [
    (lambda: _strings("core_attack.rs", "DAMAGE_TYPES"), lambda: list(combat.DAMAGE_TYPES)),
    (lambda: set(_strings("core_move.rs", "HIDE_ABILITIES")),
     lambda: set(_strings("core_observe.rs", "HIDE_ABILITIES"))),
    (lambda: _ints("core.rs", "DEFAULT_CYCLE"), lambda: [lb for _, lb in combat.TOD_DEFAULT_CYCLE]),
    (lambda: _strings("core.rs", "TOD_NAMES"), lambda: [name for name, _ in combat.TOD_DEFAULT_CYCLE]),
    (lambda: _int("lib.rs", "UNREACHABLE"), lambda: terrain_resolver.UNREACHABLE_COST),
    (lambda: _int("core_move.rs", "UNREACHABLE"), lambda: terrain_resolver.UNREACHABLE_COST),
], ids=[
    "damage types", "hide abilities (move and observe)",
    "time-of-day bonuses", "time-of-day names", "unreachable (reach)",
    "unreachable (core move)",
])
def test_the_rust_copy_equals_the_python_value(rust, python):
    assert rust() == python()


def test_the_alignment_codes_mean_the_same_alignments():
    """The core's `combat_modifier` switches on the alignment code the
    Python side sends (`classes.Alignment` for GameCore's unit fields,
    `game_core.unit_fields`)."""
    body = re.search(r"fn combat_modifier\(.*?\n\}", _source("combat.rs"), re.S).group(0)
    arms = {
        "LAWFUL": r"(\d+) => lawful_bonus,",
        "NEUTRAL": r"(\d+) => 0,",
        "CHAOTIC": r"(\d+) => -lawful_bonus,",
        "LIMINAL": r"(\d+) => MAX_LIMINAL_BONUS - lawful_bonus\.abs\(\),",
    }
    rust = {name: int(re.search(arm, body).group(1)) for name, arm in arms.items()}
    assert rust == {a.name: a.value for a in classes.Alignment}


def test_the_default_cycle_length_is_the_turn_modulus():
    """`GameCore::tod_index` writes the cycle's length as a literal,
    as `% 6` or `.rem_euclid(6)`."""
    body = re.search(r"fn tod_index\(.*?\n    \}", _source("core_step.rs"), re.S).group(0)
    modulus = re.search(r"(?:% |rem_euclid\()(\d+)\)", body)
    assert modulus, body
    assert int(modulus.group(1)) == len(combat.TOD_DEFAULT_CYCLE)
