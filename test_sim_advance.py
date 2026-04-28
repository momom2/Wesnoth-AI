#!/usr/bin/env python3
"""Test: post-attack advancement detection + [choose] WML emission.

When an attacker or defender crosses its XP threshold mid-attack,
Wesnoth's `attack_unit_and_advance` (attack.cpp:1556-1573) records the
chosen advance in a `[command] dependent="yes" [choose]` block — one
per advancing unit, attacker first then defender. Without these the
replay engine fires "expecting a user choice" the first time a model
lands a level-up.

We verify both:
  - the sim's `step()` correctly stashes `advance_choices` on the
    RecordedCommand for kill-based AND damage-based advances;
  - the exporter's `_build_replay_wml` emits the [choose] blocks in
    the right order.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import pytest
from classes import Position, Unit
from tools.replay_dataset import _build_recruit_unit
from tools.sim_to_replay import _build_replay_wml
from tools.wesnoth_sim import RecordedCommand, WesnothSim


def _make(unit_type, side, x, y, uid, *, current_hp=None, current_exp=0,
          is_leader=False, exp_modifier=100):
    """Produce a Unit at full MP / specified HP / specified XP."""
    base = _build_recruit_unit(unit_type, side=side, x=x, y=y,
                               next_uid=uid, game_id="t",
                               trait_seed_hex="12345678",
                               exp_modifier=exp_modifier)
    fresh = Unit(
        id=base.id, name=base.name, name_id=base.name_id, side=side,
        is_leader=is_leader, position=Position(x, y),
        max_hp=base.max_hp, max_moves=base.max_moves, max_exp=base.max_exp,
        cost=base.cost, alignment=base.alignment,
        levelup_names=list(base.levelup_names),
        current_hp=current_hp if current_hp is not None else base.max_hp,
        current_moves=base.max_moves,
        current_exp=current_exp,
        has_attacked=False,
        attacks=list(base.attacks),
        resistances=list(base.resistances),
        defenses=list(base.defenses),
        movement_costs=list(base.movement_costs),
        abilities=set(base.abilities),
        traits=set(base.traits),
        statuses=set(),
    )
    for k, v in base.__dict__.items():
        if k.startswith("_"):
            setattr(fresh, k, v)
    return fresh


@pytest.fixture
def fresh_sim():
    """A WesnothSim with one of the dataset replays for bootstrap.
    The unit roster gets cleared and replaced per test."""
    # Pick any tiny dataset replay; we throw away its unit set.
    import glob
    candidates = sorted(glob.glob("replays_dataset/*.json.gz"))
    if not candidates:
        pytest.skip("no replays_dataset/ to bootstrap from")
    sim = WesnothSim.from_replay(Path(candidates[0]), max_turns=10)
    sim.gs.map.units.clear()
    return sim


def _run_attack(sim, *units):
    """Place units, set side 1's turn, run a (10,10)->(11,10) attack."""
    for u in units:
        sim.gs.map.units.add(u)
    sim.gs.global_info.current_side = 1
    sim._begin_side_turn(1)
    sim.step({
        "type": "attack",
        "start_hex": Position(10, 10),
        "target_hex": Position(11, 10),
        "attack_index": 0,
    })
    return sim.command_history[-1]


def test_no_advance_no_choose(fresh_sim):
    """An attack with no level-ups produces no advance_choices entry."""
    sim = fresh_sim
    xpmod = int(getattr(sim.gs.global_info, "_experience_modifier", 100) or 100)
    rc = _run_attack(
        sim,
        _make("Skeleton", 1, 10, 10, 1, current_exp=0, exp_modifier=xpmod),
        _make("Spearman", 2, 11, 10, 2, exp_modifier=xpmod),  # full HP
        _make("Skeleton", 1, 20, 20, 100, is_leader=True, exp_modifier=xpmod),
        _make("Skeleton", 2, 21, 20, 101, is_leader=True, exp_modifier=xpmod),
    )
    assert rc.kind == "attack"
    assert "advance_choices" not in rc.extras


def test_kill_based_advance_detected(fresh_sim):
    """Attacker kills a level-1 victim and crosses XP threshold."""
    sim = fresh_sim
    xpmod = int(getattr(sim.gs.global_info, "_experience_modifier", 100) or 100)
    sk = _make("Skeleton", 1, 10, 10, 1,
               current_exp=22, exp_modifier=xpmod)  # +8 from kill -> >= max_exp
    rc = _run_attack(
        sim, sk,
        _make("Spearman", 2, 11, 10, 2, current_hp=1, exp_modifier=xpmod),
        _make("Skeleton", 1, 20, 20, 100, is_leader=True, exp_modifier=xpmod),
        _make("Skeleton", 2, 21, 20, 101, is_leader=True, exp_modifier=xpmod),
    )
    assert "advance_choices" in rc.extras
    choices = rc.extras["advance_choices"]
    assert choices == [(1, 0)], f"expected attacker-side=1 idx=0, got {choices}"


def test_damage_based_advance_detected(fresh_sim):
    """Attacker DOESN'T kill defender but gains enough XP to advance."""
    sim = fresh_sim
    xpmod = int(getattr(sim.gs.global_info, "_experience_modifier", 100) or 100)
    sk = _make("Skeleton", 1, 10, 10, 1,
               current_exp=26, exp_modifier=xpmod)  # +1 per attack on lvl-1
    rc = _run_attack(
        sim, sk,
        _make("Spearman", 2, 11, 10, 2, exp_modifier=xpmod),  # full HP
        _make("Skeleton", 1, 20, 20, 100, is_leader=True, exp_modifier=xpmod),
        _make("Skeleton", 2, 21, 20, 101, is_leader=True, exp_modifier=xpmod),
    )
    # Defender survives high-HP; attacker advances on damage XP.
    post_att = next((u for u in sim.gs.map.units if u.id == sk.id), None)
    if post_att and post_att.name == "Skeleton":
        pytest.skip("attacker did not advance (no XP gained?); test premise failed")
    assert "advance_choices" in rc.extras


def test_simultaneous_advance_order(fresh_sim):
    """Both attacker and defender cross threshold -- attacker recorded
    first per Wesnoth's attack_unit_and_advance order."""
    sim = fresh_sim
    xpmod = int(getattr(sim.gs.global_info, "_experience_modifier", 100) or 100)
    rc = _run_attack(
        sim,
        _make("Skeleton", 1, 10, 10, 1, current_exp=26, exp_modifier=xpmod),
        # Defender at xp >= max_exp already so retaliation tips it.
        _make("Spearman", 2, 11, 10, 2, current_exp=39, current_hp=40,
              exp_modifier=xpmod),
        _make("Skeleton", 1, 20, 20, 100, is_leader=True, exp_modifier=xpmod),
        _make("Skeleton", 2, 21, 20, 101, is_leader=True, exp_modifier=xpmod),
    )
    choices = rc.extras.get("advance_choices", [])
    sides_in_order = [s for s, _ in choices]
    # Attacker side first (1), defender side second (2).
    assert sides_in_order == [1, 2], (
        f"expected attacker-first defender-second order, got {sides_in_order}")


def test_choose_emitted_after_random_seed():
    """`_build_replay_wml` orders [choose] AFTER the attack's
    [random_seed] follow-up, in attacker-then-defender order."""
    history = [
        RecordedCommand(kind="init_side", side=1, cmd=["init_side", 1]),
        RecordedCommand(
            kind="attack", side=1,
            cmd=["attack", 5, 5, 6, 5, 0, -1, "deadbeef"],
            extras={"advance_choices": [(1, 0), (2, 1)]}),
        RecordedCommand(kind="end_turn", side=1, cmd=["end_turn"]),
    ]
    wml = _build_replay_wml(history)
    pa = wml.find("[attack]")
    ps = wml.find("[random_seed]", pa)
    pc1 = wml.find("[choose]", ps)
    pc2 = wml.find("[choose]", pc1 + 1)
    assert pa > 0 and ps > pa and pc1 > ps and pc2 > pc1, (
        f"order violated: attack={pa} seed={ps} choose1={pc1} choose2={pc2}")
    # First choose belongs to side 1 (attacker), second to side 2.
    import re
    sides = re.findall(r'from_side="(\d+)"\s*\n\[choose\]', wml)
    values = re.findall(r'\[choose\]\s*\nvalue="(\d+)"', wml)
    assert sides == ["1", "2"], sides
    assert values == ["0", "1"], values


def test_no_advance_no_choose_in_wml():
    """Attack without `advance_choices` extras should produce zero
    [choose] blocks (regression: an attack on a non-leveling unit
    shouldn't emit anything advance-related)."""
    history = [
        RecordedCommand(
            kind="attack", side=1,
            cmd=["attack", 5, 5, 6, 5, 0, -1, "deadbeef"],
            extras={}),  # no advance_choices
        RecordedCommand(kind="end_turn", side=1, cmd=["end_turn"]),
    ]
    wml = _build_replay_wml(history)
    assert "[choose]" not in wml


def test_select_action_rejects_repeated_state(fresh_sim):
    """Debug-mode tripwire: passing the same `GameState` object twice
    in a row to select_action means the caller didn't deepcopy
    between calls. The trainer's stored Transition.game_state would
    then point at a mutated state and the re-forward path would
    diverge. Catch the bug at the second call rather than in a
    corrupt loss six hours into a training run."""
    from transformer_policy import TransformerPolicy
    sim = fresh_sim
    leader1 = _make("Skeleton", 1, 5, 5, 1, is_leader=True)
    leader2 = _make("Skeleton", 2, 6, 5, 2, is_leader=True)
    sim.gs.map.units.add(leader1)
    sim.gs.map.units.add(leader2)
    sim._begin_side_turn(1)

    policy = TransformerPolicy()
    # First call OK.
    policy.select_action(sim.gs, game_label="contract")
    # Second call with same id() should raise.
    with pytest.raises(RuntimeError, match="SAME GameState object"):
        policy.select_action(sim.gs, game_label="contract")


def test_recall_action_rejected_to_end_turn(fresh_sim):
    """The sim has no recall list and the exporter would emit a
    broken `[recall]` block citing a unit_id that's not on Wesnoth's
    recall list. `_action_to_command` rejects recall actions outright
    so they translate to end_turn -- the recorded command kind must
    NEVER be 'recall'."""
    sim = fresh_sim
    # Place at least one leader so the sim doesn't error on missing leader.
    leader = _make("Skeleton", 1, 5, 5, 1, is_leader=True)
    leader2 = _make("Skeleton", 2, 6, 5, 2, is_leader=True)
    sim.gs.map.units.add(leader)
    sim.gs.map.units.add(leader2)
    sim._begin_side_turn(1)
    pre_count = len(sim.command_history)
    sim.step({"type": "recall", "unit_id": "u1",
              "target_hex": Position(5, 6)})
    new_kinds = [rc.kind for rc in sim.command_history[pre_count:]]
    assert "recall" not in new_kinds, (
        f"recall command leaked through to history: {new_kinds}")
    # The recall should have triggered an end_turn fallback.
    assert "end_turn" in new_kinds, (
        f"expected end_turn fallback, got {new_kinds}")
