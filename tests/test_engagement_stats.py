#!/usr/bin/env python3
"""Engagement telemetry (user spec 2026-07-12).

Pins:
  1. Healing attribution: actual (post-cap) HP only; rest first,
     remainder to the main source; village/oasis bucket beats
     regen/healer ("ability") on ties.
  2. The sim-level attack gate: attacks on petrified/scenery targets
     are refused (Wesnoth-as-played, mouse_events.cpp:753) and every
     counter fires (attempted / invalid / rejected) while the statue
     stays unharmed.
  3. End-to-end: play_one_game produces GameOutcome.engagement with
     map constants and search diagnostics.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import numpy as np
import torch

from wesnoth_ai.classes import Position
from test_inference_snapshot import _gs, _u
from tools.engagement_stats import clear_event_sink, set_event_sink
from tools.replay_dataset import _apply_command


def _heal_events(gs, side=1):
    events = []
    set_event_sink(lambda k, p: events.append((k, p)))
    try:
        _apply_command(gs, ["init_side", side])
    finally:
        clear_event_sink()
    return [p for k, p in events if k == "heal" and p["side"] == side]


def test_heal_attribution_village_with_cap():
    """Missing 3 HP, resting, on a village: total applied = 3;
    rest gets its 2 first, the village gets the remaining 1."""
    gs = _gs()
    u1 = next(u for u in gs.map.units if u.id == "u1")
    u1.current_hp = u1.max_hp - 3
    u1.statuses.add("resting")
    setattr(gs.global_info, "_terrain_codes",
            {(u1.position.x, u1.position.y): "Gg^Vh"})
    evs = _heal_events(gs)
    assert sum(e["rest"] for e in evs) == 2
    assert sum(e["village"] for e in evs) == 1
    assert sum(e["ability"] for e in evs) == 0
    # The healing loop REBUILDS unit objects; re-fetch before asserting.
    u1_after = next(u for u in gs.map.units if u.id == "u1")
    assert u1_after.current_hp == u1_after.max_hp


def test_heal_attribution_oasis_is_village_bucket():
    """Oasis (^Do): heals 8 like a village and lands in the village
    bucket, despite not being village terrain."""
    gs = _gs()
    u1 = next(u for u in gs.map.units if u.id == "u1")
    u1.current_hp = u1.max_hp - 20
    setattr(gs.global_info, "_terrain_codes",
            {(u1.position.x, u1.position.y): "Dd^Do"})
    evs = _heal_events(gs)
    assert sum(e["village"] for e in evs) == 8
    assert sum(e["rest"] for e in evs) == 0
    assert sum(e["ability"] for e in evs) == 0


def test_heal_attribution_regen_is_ability_bucket():
    gs = _gs()
    u1 = next(u for u in gs.map.units if u.id == "u1")
    u1.current_hp = u1.max_hp - 20
    u1.abilities.add("regenerate")
    evs = _heal_events(gs)
    assert sum(e["ability"] for e in evs) == 8
    assert sum(e["village"] for e in evs) == 0


def test_poison_cure_counted_and_heals_only_rest():
    """Cure turn: poisoned status cleared, poison_cured fires, HP
    healed shows only the rest bucket (+2)."""
    gs = _gs()
    u1 = next(u for u in gs.map.units if u.id == "u1")
    u1.current_hp = u1.max_hp - 10
    u1.statuses.update({"poisoned", "resting"})
    setattr(gs.global_info, "_terrain_codes",
            {(u1.position.x, u1.position.y): "Gg^Vh"})
    events = []
    set_event_sink(lambda k, p: events.append((k, p)))
    try:
        _apply_command(gs, ["init_side", 1])
    finally:
        clear_event_sink()
    heals = [p for k, p in events if k == "heal" and p["side"] == 1]
    poisons = [p for k, p in events if k == "poison" and p["side"] == 1]
    assert sum(e["rest"] for e in heals) == 2
    assert sum(e["village"] for e in heals) == 0, \
        "curing replaces healing"
    assert poisons == [{"side": 1, "cured": True, "damage": 0}]
    u1_after = next(u for u in gs.map.units if u.id == "u1")
    assert "poisoned" not in u1_after.statuses


def test_poison_damage_net_of_rest():
    """Poison-normal turn while resting: net -6, one event."""
    gs = _gs()
    u1 = next(u for u in gs.map.units if u.id == "u1")
    u1.statuses.update({"poisoned", "resting"})
    events = []
    set_event_sink(lambda k, p: events.append((k, p)))
    try:
        _apply_command(gs, ["init_side", 1])
    finally:
        clear_event_sink()
    poisons = [p for k, p in events if k == "poison" and p["side"] == 1]
    assert poisons == [{"side": 1, "cured": False, "damage": 6}]
    u1_after = next(u for u in gs.map.units if u.id == "u1")
    assert u1_after.current_hp == u1_after.max_hp - 6


def test_sim_gate_rejects_statue_attack_and_counts():
    from sim_test_helpers import fresh_scenario_sim
    from tools.abilities import hex_neighbors
    sim = fresh_scenario_sim(seed=5, max_turns=5, mini=True)
    es = sim.enable_engagement_stats()
    side = sim.gs.global_info.current_side
    occupied = {(x.position.x, x.position.y) for x in sim.gs.map.units}
    hexes = {(h.position.x, h.position.y) for h in sim.gs.map.hexes}
    actor, nb = None, None
    for u in sim.gs.map.units:
        if u.side != side:
            continue
        nb = next((p for p in hex_neighbors(u.position.x, u.position.y)
                   if p not in occupied and p in hexes), None)
        if nb is not None:
            actor = u
            break
    assert actor is not None, "fixture: need a free neighbor hex"
    statue = _u("statue1", 2, nb[0], nb[1])
    statue.statuses.add("petrified")
    sim.gs.map.units.add(statue)

    sim.step({"type": "attack", "start_hex": actor.position,
              "target_hex": Position(*nb), "attack_index": 0})

    assert es.attacks_attempted[side] == 1
    assert es.attacks_invalid_wesnoth[side] == 1, \
        "the Wesnoth-validity classifier must flag it independently"
    assert es.attacks_rejected_sim[side] == 1, \
        "the sim gate must refuse it (defense in depth)"
    st = next(x for x in sim.gs.map.units if x.id == "statue1")
    assert st.current_hp == st.max_hp, "statue must be unharmed"


def test_outcome_engagement_end_to_end():
    from sim_test_helpers import fresh_scenario_sim
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy
    from tools.sim_self_play import _recruit_cost_lookup, play_one_game
    from wesnoth_ai.transformer_policy import TransformerPolicy

    pol = TransformerPolicy(device=torch.device("cpu"), d_model=32,
                            num_layers=1, num_heads=4, d_ff=64)
    mp = MCTSPolicy(pol, MCTSConfig(n_simulations=4, batch_size=1,
                                    add_root_noise=False))
    mp._rng = np.random.default_rng(3)
    sim = fresh_scenario_sim(seed=3, max_turns=4, mini=True)
    out = play_one_game(sim, mp, lambda d: 0.0, game_label="g",
                        cost_lookup=_recruit_cost_lookup())
    e = out.engagement
    assert e is not None
    assert e["map_total_villages"] > 0
    assert set(e["attacks_attempted"]) == {1, 2}
    assert e["attacks_invalid_wesnoth"] == {1: 0, 2: 0}
    assert e["attacks_rejected_sim"] == {1: 0, 2: 0}
    assert e["start_gold"][1] > 0
    assert e["material_end"][1] > 0
    # Both sides ended at least one turn in a 4-turn game.
    assert e["unused_mp_frac"][1] is not None
    # Gold-hoarding watch (2026-07-20): per-side time-averaged bank.
    assert e["gold_bank_mean"][1] is not None
    assert e["gold_bank_mean"][1] >= 0.0
    assert e["gold_bank_mean"][2] is not None
    # Search diagnostics from the MCTS policy.
    assert e["search"] is not None
    assert 0.0 <= e["search"]["overturn_frac"] <= 1.0
    assert e["search"]["n_searches"] > 0
    # Tree-shape / reuse / end_turn-context diagnostics (2026-07-21).
    assert 0.0 <= e["search"]["reuse_frac"] <= 1.0
    assert e["search"]["depth_max"] >= 1
    assert 0.0 < e["search"]["depth_w_mean"] <= e["search"]["depth_max"]
    assert e["search"]["nodes_mean"] > 1.0
    # A finished game contains at least one chosen end_turn.
    assert e["search"]["et_n"] > 0
    assert 0.0 <= e["search"]["et_visit_frac_mean"] <= 1.0


def test_first_contact_is_player_vs_player_only():
    """Poking a side-3 tentacle is NOT "contact" (user 2026-07-15):
    first_contact_turn only sets when the target is the OPPOSING
    PLAYER side. Attacking side 3 still counts as an attempted (and
    Wesnoth-valid) attack."""
    from tools.engagement_stats import EngagementStats
    gs = _gs()
    gs.map.units.add(_u("tent", 3, 3, 4, name="Tentacle of the Deep"))
    es = EngagementStats()
    # side 1 attacks the armed side-3 tentacle: no contact.
    es.note_attack_attempt(gs, {"type": "attack",
                                "target_hex": Position(3, 4)})
    assert es.first_contact_turn is None
    assert es.attacks_attempted[1] == 1
    assert es.attacks_invalid_wesnoth[1] == 0   # armed side-3 = valid
    # side 1 attacks the side-2 unit: contact.
    es.note_attack_attempt(gs, {"type": "attack",
                                "target_hex": Position(4, 3)})
    assert es.first_contact_turn == 1
