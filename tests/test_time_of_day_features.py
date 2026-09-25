"""The network can see the time of day (2026-09-22).

Until this landed, `GLOBAL_FEAT_DIM` was 6 -- turn, side to move, our
gold, our income, our villages, theirs -- and no time-of-day or lawful
bonus signal appeared anywhere in the global, unit or hex features,
while combat applied the bonus in `wesnoth_ai/combat.py` and in the
Rust kernel. A lawful unit's damage swung by half for reasons the
policy could not observe.

Two features rather than one, because the bonus alone cannot tell dawn
from dusk: both are 0, and they are strategically opposite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.scenario_pool import ScenarioSetup, build_scenario_gamestate  # noqa: E402
from wesnoth_ai import encoder as enc_mod  # noqa: E402
from wesnoth_ai.encoder import (GLOBAL_FEAT_DIM, LAWFUL_BONUS_NORM,  # noqa: E402
                                GameStateEncoder)

TOD_NOW, TOD_NEXT = 6, 7          # the two new global slots


def _state(scenario_id: str, turn: int):
    gs = build_scenario_gamestate(ScenarioSetup(
        scenario_id=scenario_id, faction1="Rebels", leader1="Elvish Captain",
        faction2="Loyalists", leader2="Lieutenant", fogless=False, tod_start=None))
    gs.global_info.turn_number = turn
    return gs


def _globals(encoder, scenario_id, turn):
    return encoder.raw_of(_state(scenario_id, turn)).global_feats


@pytest.fixture(scope="module")
def encoder():
    return GameStateEncoder()


def test_the_cycle_reaches_the_features(encoder):
    """Hamlets starts at dawn, so the first four turns run
    dawn, morning, afternoon, dusk: 0, +25, +25, 0."""
    got = [_globals(encoder, "multiplayer_Hamlets", t)[TOD_NOW] for t in (1, 2, 3, 4)]
    assert got == pytest.approx([0.0, 1.0, 1.0, 0.0])


def test_dawn_and_dusk_are_distinguishable(encoder):
    """The whole reason for the second feature. Both carry a zero
    bonus; one is followed by day and the other by night, and they ask
    for opposite play."""
    dawn = _globals(encoder, "multiplayer_Hamlets", 1)
    dusk = _globals(encoder, "multiplayer_Hamlets", 4)
    assert dawn[TOD_NOW] == dusk[TOD_NOW] == 0.0
    assert dawn[TOD_NEXT] == pytest.approx(1.0)
    assert dusk[TOD_NEXT] == pytest.approx(-1.0)
    assert list(dawn) != list(dusk)


def test_a_scenario_that_starts_at_night_says_so(encoder):
    """Fallenstar Lake and Ruined Passage declare `current_time=5`
    ({DEFAULT_SCHEDULE_SECOND_WATCH}), so their turn 1 is second
    watch, not dawn. The turn number alone cannot express that, which
    is why it was never a stand-in for this feature."""
    for scenario_id in ("multiplayer_Fallenstar_Lake", "multiplayer_Ruined_Passage"):
        first = _globals(encoder, scenario_id, 1)
        assert first[TOD_NOW] == pytest.approx(-1.0), scenario_id
        assert first[TOD_NEXT] == pytest.approx(0.0), scenario_id
    # Against a dawn map at the same turn number.
    assert _globals(encoder, "multiplayer_Hamlets", 1)[TOD_NOW] == 0.0


def test_the_bonus_is_normalised_into_the_other_features_range(encoder):
    seen = {_globals(encoder, "multiplayer_Hamlets", t)[TOD_NOW] for t in range(1, 7)}
    assert seen == {-1.0, 0.0, 1.0}
    assert LAWFUL_BONUS_NORM == 25.0


def test_an_old_checkpoint_observes_what_it_always_did():
    """A checkpoint trained on six globals loads through
    `pad_legacy_encoder_state` with a zero column for each new
    feature, so its output cannot move no matter what the time of day
    is. Without the pad, `strict=False` does not tolerate the shape
    change and the load raises."""
    encoder = GameStateEncoder(d_model=16)
    state = {k: v.clone() for k, v in encoder.state_dict().items()}
    # An encoder from before the change: six global inputs.
    legacy = state["global_proj.weight"][:, :6].clone()
    state["global_proj.weight"] = legacy
    with pytest.raises(RuntimeError):
        encoder.load_state_dict(state, strict=False)

    padded = enc_mod.pad_legacy_encoder_state(state, encoder)
    encoder.load_state_dict(padded, strict=False)
    assert torch.equal(encoder.global_proj.weight[:, :6], legacy)
    assert torch.count_nonzero(encoder.global_proj.weight[:, 6:]) == 0
    # The padded columns are exactly the time-of-day ones, so two
    # states differing only in the time of day project identically.
    day = torch.zeros(1, GLOBAL_FEAT_DIM)
    night = day.clone()
    night[0, TOD_NOW], night[0, TOD_NEXT] = -1.0, -1.0
    assert torch.allclose(encoder.global_proj(day), encoder.global_proj(night))


def test_a_stale_wheel_is_refused_rather_than_silently_narrow(monkeypatch):
    """A kernel from before the widths changed composes six globals
    where the encoder expects eight. numpy would broadcast or raise far
    from the cause, so the phase is checked and the Python builders
    take over."""
    from tools import pathfind_sim

    class _Stale:
        __phase__ = enc_mod._ENCODE_KERNEL_PHASE - 1

        @staticmethod
        def encode_raw_streams(*a, **k):
            raise AssertionError("the stale kernel must not be called")

    monkeypatch.setattr(pathfind_sim, "_RUST", _Stale)
    monkeypatch.setattr(enc_mod, "_warned_stale_kernel", False)
    assert enc_mod._rust_encode_kernel() is None

    class _Current(_Stale):
        __phase__ = enc_mod._ENCODE_KERNEL_PHASE

    monkeypatch.setattr(pathfind_sim, "_RUST", _Current)
    assert enc_mod._rust_encode_kernel() is not None


def test_a_stale_core_is_refused_at_the_point_of_encoding():
    """GameCore's own gate accepts any wheel from phase 7, but only
    phase 11 emits the time-of-day globals. A narrower core must fail
    with a message naming the wheel, not as a shape error inside the
    model's first forward pass."""
    import numpy as np

    from wesnoth_ai.game_core import _require_global_width

    _require_global_width(np.zeros(GLOBAL_FEAT_DIM, dtype=np.float32))
    with pytest.raises(RuntimeError, match="phase"):
        _require_global_width(np.zeros(6, dtype=np.float32))


def test_the_rust_kernel_declares_the_same_widths():
    """The Rust composition mirrors these constants; a wheel that
    drifts emits a differently shaped row. Checked on the source,
    which is readable whether or not the wheel is current."""
    import re

    from helpers.source_tree import source_files

    source = "\n".join(p.read_text(encoding="utf-8")
                       for p in source_files("rust/wesnoth_core/src", pattern="*.rs"))
    declared = re.findall(r"GLOBAL_FEAT_DIM: usize = (\d+)", source)
    assert [int(v) for v in declared] == [GLOBAL_FEAT_DIM], declared
    norms = re.findall(r"LAWFUL_BONUS_NORM: f64 = ([\d.]+)", source)
    assert [float(v) for v in norms] == [LAWFUL_BONUS_NORM], norms
