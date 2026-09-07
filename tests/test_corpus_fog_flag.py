"""The corpus records each side's fog and shroud settings and the
reconstruction sets the encoder's fog switch from them (2026-09-06:
18.9% of the corpus was played fog-off while the encoder hid enemy
units on every game). Rule: shroud counts as fog; fog off with shroud
on is quarantined."""
import bz2
import copy
import glob
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools.replay_dataset import fog_on_for, quarantine_reason  # noqa: E402
from tools.replay_extract import _wml_bool  # noqa: E402


def _sides(fog, shroud):
    return [{"side": 1, "fog": fog, "shroud": shroud}, {"side": 2, "fog": fog, "shroud": shroud}]


def test_fog_switch_and_quarantine_from_the_recorded_sides():
    assert fog_on_for(_sides(True, False)) is True
    assert fog_on_for(_sides(False, False)) is False
    assert fog_on_for(_sides(True, True)) is True
    assert fog_on_for(_sides(False, True)) is True          # shroud counts as fog
    assert fog_on_for([{"side": 1}, {"side": 2}]) is True   # files from before the flags
    assert fog_on_for([]) is True
    assert quarantine_reason(_sides(False, True)) == "fog_off_shroud_on"
    for fog, shroud in ((True, False), (False, False), (True, True)):
        assert quarantine_reason(_sides(fog, shroud)) is None
    assert quarantine_reason([{"side": 1}]) is None


def test_wml_bool_reads_both_spellings_and_keeps_the_default():
    assert _wml_bool("yes", False) and _wml_bool('"true"', False) and _wml_bool("1", False)
    assert not _wml_bool("no", True) and not _wml_bool("false", True)
    assert _wml_bool(None, True) is True and _wml_bool("maybe", False) is False


def test_extractor_records_the_sides_fog_and_the_encoder_honours_it():
    """On a real raw replay (skipped when the corpus is absent): the
    extractor records fog/shroud per side, the reconstruction sets the
    switch, and on a fog-off copy the mover sees every enemy unit."""
    from tools.replay_dataset import _build_initial_gamestate
    from tools.replay_extract import extract_replay
    from wesnoth_ai.visibility import units_visible_to
    raws = sorted(glob.glob("replays_raw/*/2p_*.bz2"))[:40]
    if not raws:
        pytest.skip("replays_raw not present")
    rec = None
    for p in raws:
        try:
            txt = bz2.open(p, "rt", encoding="utf-8", errors="replace").read(200000)
        except OSError:
            continue
        if "fog=yes" in txt or 'fog="yes"' in txt:
            rec = extract_replay(Path(p))
            if rec is not None:
                break
    if rec is None:
        pytest.skip("no readable fog-on 2p replay among the first 40")
    sides = rec["starting_sides"]
    assert all("fog" in s and "shroud" in s for s in sides)
    assert fog_on_for(sides) is True
    gs = _build_initial_gamestate(rec)
    assert gs.global_info._fog is True
    off = copy.deepcopy(rec)
    for s in off["starting_sides"]:
        s["fog"], s["shroud"] = False, False
    gs_off = _build_initial_gamestate(off)
    assert gs_off.global_info._fog is False
    mover = 1
    seen_off = {u.id for u in units_visible_to(gs_off, mover)}
    assert seen_off == {u.id for u in gs_off.map.units}, "fog off: every unit is visible"
    seen_on = {u.id for u in units_visible_to(gs, mover)}
    assert seen_on <= seen_off
