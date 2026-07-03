#!/usr/bin/env python3
"""Elo-ladder evaluation must sample LADDER maps only.

Mini-map / drill games have different dynamics (engagement by turn
3-5, tiny economies) — mixing them into strength evaluation would
pollute the Elo-vs-compute measurement (user requirement, 2026-07-03).
`random_setup`'s defaults already exclude them; this test pins those
defaults so a future curriculum flag can't silently leak into the
eval path.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import glob
import pytest

from tools.scenario_events import SCENARIO_DIR
from tools.scenario_pool import LADDER_SCENARIO_IDS, random_setup

pytestmark = pytest.mark.skipif(
    not SCENARIO_DIR.exists(),
    reason="wesnoth_src scenario dir not present",
)


def test_default_random_setup_is_ladder_only():
    """With no curriculum kwargs (exactly how tools/elo_ladder.py
    calls it), every sampled scenario must be a ladder map."""
    rng = random.Random(42)
    seen = set()
    for _ in range(300):
        setup = random_setup(rng, forced_faction=None)
        assert setup.scenario_id in LADDER_SCENARIO_IDS, (
            f"non-ladder scenario {setup.scenario_id!r} sampled by the "
            f"default (evaluation) path -- mini/drill maps must never "
            f"reach the Elo ladder")
        seen.add(setup.scenario_id)
    # Sanity: the sampler actually spreads over the pool (not one map).
    assert len(seen) >= 10, f"suspiciously few distinct maps: {seen}"
