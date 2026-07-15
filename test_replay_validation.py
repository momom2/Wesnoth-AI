#!/usr/bin/env python3
"""Replay-validity harness (2026-07-06, after real-Wesnoth playback
of a self-play export desynced on its first action).

Layer 1 (always runs): map parsing handles the header lines add-on
maps carry (`border_size=` / `usage=` — the root cause: parsers
counted them as terrain rows, shifting the sim's whole coordinate
frame vs Wesnoth's), and a self-play game exported from scratch
passes tools/validate_replay's ground-truth checks on BOTH map
classes.

Layer 2 (opt-in, WESNOTH_E2E=1 + a real install): plays the export
back in actual Wesnoth and fails on the engine's own OOS error
lines. Ladder maps are asserted clean; mini maps are xfail pending
the side-3 scenery sequencing fix (first catch of this harness).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import glob
import pytest

from tools.replay_dataset import split_map_grid
from tools.scenario_pool import extract_player_starts
from tools.sim_to_replay import _scrape_map_keep_positions

_HEADERFUL = """border_size=1
usage=map

Xu, Xu, Xu, Xu, Xu
Xu, Gg, 1 Kh, Gg, Xu
Xu, Gg, Gg, 2 Kh, Xu
Xu, Xu, Xu, Xu, Xu
"""
_HEADERLESS = """Xu, Xu, Xu, Xu, Xu
Xu, Gg, 1 Kh, Gg, Xu
Xu, Gg, Gg, 2 Kh, Xu
Xu, Xu, Xu, Xu, Xu
"""


def test_split_map_grid_strips_headers():
    rows_h, border_h = split_map_grid(_HEADERFUL)
    rows_p, border_p = split_map_grid(_HEADERLESS)
    assert border_h == border_p == 1
    assert rows_h == rows_p, (
        "header lines must not survive into the terrain grid")


def test_player_starts_identical_with_and_without_headers():
    a = extract_player_starts(_HEADERFUL)
    b = extract_player_starts(_HEADERLESS)
    assert a == b, f"header lines shifted the start frame: {a} vs {b}"
    # 0-indexed border-stripped: the '1 Kh' sits at grid (2,1) ->
    # playable (1,0).
    assert (a[1].x, a[1].y) == (1, 0)
    assert (a[2].x, a[2].y) == (2, 1)


def test_keep_scrape_matches_wml_coordinates():
    for md in (_HEADERFUL, _HEADERLESS):
        keeps = _scrape_map_keep_positions(md)
        # WML (1,1) = first playable hex; '1 Kh' at playable (1,0)
        # -> WML (2,1).
        assert keeps[1] == (2, 1), keeps
        assert keeps[2] == (3, 2), keeps


def _export_and_validate(mini: bool, tmp_path: Path,
                         policy_kind: str = "dummy",
                         max_turns: int = 8):
    """Play + export one game, return (path, layer-1 violations).

    policy_kind='mcts' drives the PRODUCTION self-play path (real
    MCTSPolicy search on a small random-init net) — the user-facing
    replays come from self-play, so the e2e verification must too
    (dummy-verified ladder exports had been trusted while self-play
    minis desynced; never let the verified path diverge from the
    production path again)."""
    from sim_test_helpers import fresh_scenario_sim
    from tools.sim_self_play import _recruit_cost_lookup, play_one_game
    from tools.sim_to_replay import export_replay_from_scratch
    from tools.validate_replay import validate_replay

    if policy_kind == "mcts":
        import torch
        from tools.draw_tiebreak import DrawTiebreakConfig
        from tools.mcts import MCTSConfig
        from tools.mcts_policy import MCTSPolicy
        from transformer_policy import TransformerPolicy
        base = TransformerPolicy(device=torch.device("cpu"),
                                 d_model=48, num_layers=2,
                                 num_heads=4, d_ff=96)
        policy = MCTSPolicy(base, MCTSConfig(
            n_simulations=4,
            draw_tiebreak=DrawTiebreakConfig(cap=0.3)))
    else:
        from dummy_policy import DummyPolicy

        class _Stub:
            uses_step_rewards = False
            def __init__(self): self._d = DummyPolicy()
            def select_action(self, gs, *, game_label="d", sim=None):
                return self._d.select_action(
                    gs, game_label=game_label, sim=sim)
            def observe(self, *a, **kw): pass
            def drop_pending(self, *a, **kw): pass
            def finalize_game(self, *a, **kw): pass
        policy = _Stub()

    sim = fresh_scenario_sim(seed=7, max_turns=max_turns, mini=mini)
    play_one_game(sim, policy, lambda *a, **kw: 0.0,
                  game_label="val", cost_lookup=_recruit_cost_lookup())
    out = tmp_path / (f"validate_{policy_kind}_"
                      f"{'mini' if mini else 'ladder'}.bz2")
    export_replay_from_scratch(sim, out)
    return out, validate_replay(out)


def test_selfplay_export_validates_ladder(tmp_path):
    _, problems = _export_and_validate(mini=False, tmp_path=tmp_path)
    assert problems == [], "\n".join(problems)


def test_selfplay_export_validates_mini(tmp_path):
    _, problems = _export_and_validate(mini=True, tmp_path=tmp_path)
    assert problems == [], "\n".join(problems)


def test_mcts_selfplay_export_validates_ladder(tmp_path):
    """The PRODUCTION path: real MCTS search, exported, statically
    validated. Small net + 4 sims keeps it suite-friendly."""
    _, problems = _export_and_validate(mini=False, tmp_path=tmp_path,
                                       policy_kind="mcts")
    assert problems == [], "\n".join(problems)


# ---- Layer 2: real Wesnoth (opt-in) ---------------------------------

_e2e = pytest.mark.skipif(
    not os.environ.get("WESNOTH_E2E"),
    reason="set WESNOTH_E2E=1 (needs a real Wesnoth install; "
           "launches a minimized game window)")


@_e2e
def test_ladder_export_plays_back_oos_free_in_wesnoth(tmp_path):
    from tools.validate_replay_wesnoth import validate_in_wesnoth
    out, problems = _export_and_validate(mini=False, tmp_path=tmp_path)
    assert problems == []
    bad = validate_in_wesnoth(out, timeout=300)
    assert bad == [], "\n".join(bad)


@_e2e
@pytest.mark.xfail(reason="mini scenarios carry a scenery side 3 the "
                          "export's turn sequencing doesn't model — "
                          "first catch of this harness (2026-07-06)")
def test_mini_export_plays_back_oos_free_in_wesnoth(tmp_path):
    from tools.validate_replay_wesnoth import validate_in_wesnoth
    out, problems = _export_and_validate(mini=True, tmp_path=tmp_path)
    assert problems == []
    bad = validate_in_wesnoth(out, timeout=300)
    assert bad == [], "\n".join(bad)


def test_event_unit_scan_reaches_switch_case_spawns():
    """Layer-1 occupancy seeding must find units spawned via
    [event]->[switch]->[case]->[unit] (Hornshark Island's faction
    starters). WMLNode.all() is non-recursive, so the flat
    ev.all("unit") scan finds ZERO of them -- the recursive walk
    (validate_replay.py, 2026-07-15) must find them all."""
    from tools.scenario_events import load_scenario_wml
    root = load_scenario_wml("multiplayer_Hornshark_Island")
    assert root is not None
    mp = root.first("multiplayer") or root.first("scenario")
    assert mp is not None
    flat, recursive = 0, 0
    for ev in mp.all("event"):
        flat += sum(1 for u in ev.all("unit")
                    if u.attrs.get("x") and u.attrs.get("y"))
        stack = [ev]
        while stack:
            node = stack.pop()
            for child in node.children:
                if (child.tag == "unit" and child.attrs.get("x")
                        and child.attrs.get("y")):
                    recursive += 1
                stack.append(child)
    assert recursive > flat, (
        f"recursive walk must find the [switch]/[case] spawns the "
        f"flat scan misses (flat={flat}, recursive={recursive})")
    assert recursive >= 8, f"Hornshark spawns >= 8 (got {recursive})"
