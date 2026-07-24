"""Validation for tools/combat_outcomes.py: the exact outcome DP
must agree with the bit-exact sim sampled under independent seed
salts -- the strongest available cross-check, since the two paths
share parameters (build_attack_context) but compute the
distribution by entirely different means (probability propagation
vs actual MT-seeded strike resolution).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools.abilities import hex_neighbors  # noqa: E402
from tools.combat_outcomes import (  # noqa: E402
    enumerate_attack_outcomes, outcome_key_for_child,
)
from tools.scenario_pool import (  # noqa: E402
    random_setup, build_scenario_gamestate, load_factions,
)
from tools.wesnoth_sim import WesnothSim  # noqa: E402


def _engineered_fight():
    """A real game state with one side-1 unit teleported adjacent to
    a side-2 unit, plus the attack action between them. Returns
    (sim, action) or skips if the setup can't be arranged."""
    load_factions()
    setup = random_setup(random.Random(3), forced_faction=None)
    gs = build_scenario_gamestate(setup)
    sim = WesnothSim(gs, scenario_id=setup.scenario_id, max_turns=30)

    side = sim.gs.global_info.current_side
    # At game start each side has only its leader -- leaders fight
    # fine for this purpose.
    att = next((u for u in sim.gs.map.units
                if u.side == side and u.attacks),
               None)
    dfd = next((u for u in sim.gs.map.units
                if u.side != side and u.attacks
               and u.side in (1, 2)
               and "petrified" not in (u.statuses or set())),
               None)
    if att is None or dfd is None:
        pytest.skip("setup lacks a usable attacker/defender pair")

    # Teleport the attacker next to the defender (state surgery is
    # fine: combat reads positions/terrain, not how we got there).
    occupied = {(u.position.x, u.position.y) for u in sim.gs.map.units}
    spot = next(
        ((nx, ny) for nx, ny in hex_neighbors(dfd.position.x,
                                              dfd.position.y)
         if (nx, ny) not in occupied
         and 0 <= nx < sim.gs.map.size_x
         and 0 <= ny < sim.gs.map.size_y),
        None)
    if spot is None:
        pytest.skip("no free hex adjacent to the defender")
    att.position.x, att.position.y = spot

    action = {
        "type": "attack",
        "start_hex": att.position,
        "target_hex": dfd.position,
        "attack_index": 0,
    }
    return sim, action


def test_dp_is_a_distribution_and_matches_sampling():
    sim, action = _engineered_fight()
    dist = enumerate_attack_outcomes(sim.gs, action)
    assert dist is not None, "plain level-1 fight must be enumerable"
    total = sum(dist.probs.values())
    assert abs(total - 1.0) < 1e-9
    assert all(p > 0 for p in dist.probs.values())
    # Dead units carry canonicalized (False) flags.
    for (a_hp, d_hp, a_sl, d_sl, a_po, d_po, a_pe, d_pe, a_ty, d_ty) in dist.probs:
        if a_hp == 0:
            assert not a_sl and not a_po and not a_pe
        if d_hp == 0:
            assert not d_sl and not d_po and not d_pe

    # Empirical cross-check: salted sim forks ARE the ground truth.
    N = 400
    counts = {}
    for i in range(N):
        f = sim.fork()
        f._seed_salt = f"xcheck-{i}"
        f.step(action)
        key = outcome_key_for_child(f.gs, dist.attacker_id,
                                    dist.defender_id)
        counts[key] = counts.get(key, 0) + 1

    # Every sampled outcome must be in the exact support.
    unknown = set(counts) - set(dist.probs)
    assert not unknown, (
        f"sampled outcomes missing from DP support: {unknown}\n"
        f"DP support: {sorted(dist.probs)}"
    )
    # Total-variation distance within sampling noise. E[TV] for a
    # ~10-atom distribution at N=400 is ~0.08; 0.15 gives ~comfortable
    # headroom while still catching any real modeling error (a wrong
    # cth or strike count shifts mass by far more).
    tv = 0.5 * sum(
        abs(counts.get(k, 0) / N - p) for k, p in dist.probs.items()
    ) + 0.5 * sum(counts.get(k, 0) / N for k in unknown)
    assert tv < 0.15, (
        f"TV distance {tv:.3f} between DP and {N} sim samples\n"
        f"DP:        {sorted(dist.probs.items())}\n"
        f"empirical: {sorted((k, c / N) for k, c in counts.items())}"
    )


def test_advancement_risk_refuses_enumeration():
    sim, action = _engineered_fight()
    att = next(u for u in sim.gs.map.units
               if u.position is action["start_hex"])
    att.current_exp = max(0, att.max_exp - 1)
    assert enumerate_attack_outcomes(sim.gs, action) is None, (
        "a fight that could advance a unit must fall back to sampling"
    )


def test_advancement_exact_path_matches_sampling():
    """With advancement_choice set, the exact path RESOLVES an advancing
    fight (no None bail) and its outcomes match the sim's uniformly-
    advanced sampled children -- so MCTS's exact path never silently
    falls back to sampling for advancement. If advancement weren't
    modelled, the advanced children's keys would be MISSING from the
    support and the coverage assert below would fire."""
    sim, action = _engineered_fight()
    att = next(u for u in sim.gs.map.units
               if u.position is action["start_hex"])
    dfd = next(u for u in sim.gs.map.units
               if u.position is action["target_hex"])
    if int(getattr(dfd, "level", 1) or 1) < 1:
        pytest.skip("defender is level 0 -> no combat XP -> no forced advance")
    att.current_exp = max(0, att.max_exp - 1)   # any combat XP crosses
    dfd.current_hp = 1                          # near-certain kill
    sim.enable_uniform_advancement()

    dist = enumerate_attack_outcomes(sim.gs, action,
                                     advancement_choice="uniform")
    assert dist is not None, "advancing fight must enumerate with a choice"

    N = 600
    counts = {}
    for i in range(N):
        f = sim.fork()
        f._seed_salt = f"adv-xcheck-{i}"
        f.step(action)
        key = outcome_key_for_child(f.gs, dist.attacker_id, dist.defender_id)
        counts[key] = counts.get(key, 0) + 1

    assert any(k[0] > 0 for k in counts), (
        "attacker never survived -- can't exercise advancement")
    # COVERAGE: every sampled outcome (incl. the advanced ones) is in the
    # exact support -- the whole point of routing MCTS through it.
    unknown = set(counts) - set(dist.probs)
    assert not unknown, (
        f"exact path missed sampled advancement outcomes: {unknown}\n"
        f"DP support: {sorted(dist.probs)}")
    tv = 0.5 * sum(
        abs(counts.get(k, 0) / N - p) for k, p in dist.probs.items()
    ) + 0.5 * sum(counts.get(k, 0) / N for k in unknown)
    assert tv < 0.15, (
        f"TV {tv:.3f} between DP and {N} advancing samples\n"
        f"DP:        {sorted(dist.probs.items())}\n"
        f"empirical: {sorted((k, c / N) for k, c in counts.items())}")


def test_mcts_edge_exact_outcome_bookkeeping():
    """Drive _select_one over a real attack edge: sampled children
    register against the exact distribution, seen_mass accumulates,
    and once coverage crosses the threshold no NEW outcome children
    appear (selection switches to the exact known set)."""
    import numpy as np
    from wesnoth_ai.action_sampler import LegalActionPrior
    from tools.mcts import (
        MCTSNode, MCTSEdge, _select_one, _STEP_ERROR_KEY,
        _EXACT_COVERAGE_EPSILON,
    )

    sim, action = _engineered_fight()
    root = MCTSNode(sim.fork())
    root.expanded = True
    lap = LegalActionPrior(action=action, prior=1.0, actor_idx=0,
                           target_idx=None, weapon_idx=None,
                           type_idx=None)
    edge = MCTSEdge(lap)
    root.edges = [edge]

    rng = np.random.default_rng(123)
    for _ in range(300):
        _select_one(root, c_puct=1.5, virtual_loss=0.0,
                    chance_nodes=True, sample_rng=rng,
                    exact_outcomes=True)
        if edge.seen_mass >= 1.0 - _EXACT_COVERAGE_EPSILON:
            break

    dist = edge.outcome_probs
    assert dist is not None and dist != "unset", (
        "attack edge should have an exact outcome distribution"
    )
    # Bookkeeping invariants.
    real_children = [k for k in edge.children if k != _STEP_ERROR_KEY]
    assert len(edge.outcome_keys) == len(real_children)
    assert all(ok in dist.probs for ok in edge.outcome_keys.values())
    expected_mass = sum(dist.probs[ok]
                        for ok in set(edge.outcome_keys.values()))
    assert abs(edge.seen_mass - expected_mass) < 1e-9
    assert edge.seen_mass >= 1.0 - _EXACT_COVERAGE_EPSILON, (
        f"300 samples should cover a typical fight; seen "
        f"{edge.seen_mass:.4f} of mass over "
        f"{len(real_children)} outcomes"
    )

    # Past the threshold: further traversals select among KNOWN
    # children without inventing new ones.
    n_children_before = len(edge.children)
    for _ in range(50):
        _select_one(root, c_puct=1.5, virtual_loss=0.0,
                    chance_nodes=True, sample_rng=rng,
                    exact_outcomes=True)
    assert len(edge.children) == n_children_before


def test_selfplay_attack_resolves_counter_and_retaliates():
    """Regression for the retaliation-free self-play bug (fixed
    2026-06-12): the sim used to emit d_weapon=-1, which
    resolve_attack maps to None = NO counter-attack, while Wesnoth
    playback of the exported replay AUTO-SELECTS a counter
    (battle_context::choose_defender_weapon on -1) -- silent
    sim-vs-playback divergence that empty [checkup] blocks never
    surface. The emitted command must carry a concrete counter
    index whenever the defender has a matching-range weapon, and
    the attacker must actually take damage in some outcomes."""
    sim, action = _engineered_fight()
    att = next(u for u in sim.gs.map.units
               if u.position is action["start_hex"])
    dfd = next(u for u in sim.gs.map.units
               if u.position is action["target_hex"])
    a_ranged = att.attacks[action["attack_index"]].is_ranged
    if not any(w.is_ranged == a_ranged for w in dfd.attacks):
        pytest.skip("defender has no matching-range counter here")

    hp0 = att.current_hp
    damaged = 0
    for i in range(40):
        f = sim.fork()
        f._seed_salt = f"retal-{i}"
        f.step(action)
        assert f.command_history, "attack command must be recorded"
        cmd = f.command_history[-1].cmd
        assert cmd[0] == "attack"
        assert cmd[6] >= 0, (
            "self-play attack must carry a concrete defender weapon "
            "index (d_weapon=-1 means NO retaliation in our resolver "
            "but AUTO-retaliation in Wesnoth playback)"
        )
        a2 = next((u for u in f.gs.map.units if u.id == att.id), None)
        if a2 is None or a2.current_hp < hp0:
            damaged += 1
    assert damaged > 0, (
        "across 40 independent rolls the defender never landed a "
        "counter-strike -- retaliation is not being applied"
    )


def test_advancement_multi_option_exact_matches_sampling():
    """2-target advancement (Skeleton -> Revenant/Deathblade): the DP's
    uniform split AND the advanced full-HP/type keys must match the sim's
    salted forks. The single-target test above can't exercise the uniform
    branch, the multi-advance recursion, or the type-in-key matching --
    this closes that coverage gap (2026-07-24 adversarial review)."""
    from tools.replay_dataset import (_build_recruit_unit, _stats_for,
                                      _rebuild_unit)
    assert len(_stats_for("Skeleton").get("advances_to", [])) == 2
    sim, action = _engineered_fight()
    xpmod = int(getattr(sim.gs.global_info, "_experience_modifier", 100) or 100)
    a_pos, d_pos = action["start_hex"], action["target_hex"]
    old_att = next(u for u in sim.gs.map.units
                   if u.position.x == a_pos.x and u.position.y == a_pos.y)
    old_dfd = next(u for u in sim.gs.map.units
                   if u.position.x == d_pos.x and u.position.y == d_pos.y)
    # Side-1 Skeleton at the brink of leveling; frail defender so the
    # Skeleton reliably survives and crosses its XP threshold (+1 combat
    # XP alone suffices) -> advances uniformly to Revenant or Deathblade.
    sk = _build_recruit_unit("Skeleton", side=old_att.side, x=a_pos.x,
                             y=a_pos.y, next_uid=9001, game_id="t",
                             trait_seed_hex="12345678", exp_modifier=xpmod)
    sk = _rebuild_unit(sk, current_exp=sk.max_exp - 1)
    sim.gs.map.units.discard(old_att)
    sim.gs.map.units.add(sk)
    sim.gs.map.units.discard(old_dfd)
    sim.gs.map.units.add(_rebuild_unit(old_dfd, current_hp=1))
    sim.enable_uniform_advancement()

    dist = enumerate_attack_outcomes(sim.gs, action,
                                     advancement_choice="uniform")
    assert dist is not None, "advancing fight must enumerate with a choice"
    adv_types = {k[8] for k in dist.probs} | {k[9] for k in dist.probs}
    assert {"Revenant", "Deathblade"} <= adv_types, (
        f"both advances must appear in DP support, got {adv_types}")

    N = 1500
    counts = {}
    for i in range(N):
        f = sim.fork()
        f._seed_salt = f"adv2-{i}"
        f.step(action)
        key = outcome_key_for_child(f.gs, dist.attacker_id, dist.defender_id)
        counts[key] = counts.get(key, 0) + 1
    unknown = set(counts) - set(dist.probs)
    assert not unknown, (
        f"sampled advancement outcomes missing from DP support "
        f"(silent fallback to sampling): {unknown}")
    tv = 0.5 * sum(abs(counts.get(k, 0) / N - p) for k, p in dist.probs.items()) \
       + 0.5 * sum(counts.get(k, 0) / N for k in unknown)
    assert tv < 0.15, (
        f"TV {tv:.3f} between DP and {N} salted sim forks\n"
        f"DP:        {sorted(dist.probs.items())}\n"
        f"empirical: {sorted((k, c / N) for k, c in counts.items())}")
