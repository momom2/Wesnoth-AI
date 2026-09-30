"""Our games do without shroud (user ruling 2026-09-30): the scenario
builder refuses a side that declares it, and the corpus leaves out every
game in which a player side had it."""
from __future__ import annotations

import random

import pytest


def test_the_scenario_builder_refuses_a_side_with_shroud(monkeypatch):
    from wesnoth_ai.rules import scenario_pool as sp
    setup = sp.random_setup(random.Random(1))
    sp.build_scenario_gamestate(setup)                      # the pool's own scenarios build
    real = sp.read_side

    def with_shroud(node, **kw):
        side = real(node, **kw)
        return None if side is None else {**side, "shroud": True}

    monkeypatch.setattr(sp, "read_side", with_shroud)
    with pytest.raises(ValueError, match="declares shroud"):
        sp.build_scenario_gamestate(setup)
