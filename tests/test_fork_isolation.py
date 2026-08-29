"""Fork isolation: a search fork must never rewrite the parent state.

The bug class (three live instances found 2026-07-18 .. 2026-07-29):
`WesnothSim.fork()` clones the game via `Map.__deepcopy__` /
`GlobalInfo.__deepcopy__` fast paths that deliberately ALIAS
structures assumed immutable (hexes, fog, mask, `_terrain_codes`,
Unit contents, ScenarioEvent objects). Any in-place mutation of an
aliased structure inside an MCTS fork rewrites the LIVE game and
every sibling fork -- "search imagination rewrites the real present".

Instances so far:
  - 2026-07-18: `_terrain_action` morphed shared hexes/_terrain_codes
    (Aethermaw turns 4-6) -> fixed copy-on-write.
  - 2026-07-29 (fa95da5): `_capture_village` stamped
    TerrainModifiers.VILLAGE on a shared Hex -> ownership moved to the
    per-fork `_village_owner` map.
  - 2026-07-29 (this file): `fire_event` latched `ev.fired = True` on
    ScenarioEvent objects SHARED across forks -- a fork crossing
    Aethermaw's turn-4 boundary permanently suppressed the LIVE
    game's first_time_only morph. And `_object_action` applied
    `[effect]` mutations in place on shared Unit objects (dormant:
    current-pool [object]s all fire pre-fork, but one scenario away
    from live).

These tests exercise the production mutation paths directly on
production forks -- no policy, no model, no RNG -- so they are
deterministic and cannot be flaky.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402

from tools.replay_dataset import _fire_turn_events  # noqa: E402
from tools.replay_extract import WMLNode  # noqa: E402
from tools.scenario_events import _object_action  # noqa: E402
from wesnoth_ai.classes import deep_state_fingerprint  # noqa: E402


# Aethermaw's `side 1 turn 4` [terrain] event morphs WML (13,13) ->
# Chw (2p_Aethermaw.cfg:323-339). Python 0-indexed: (12, 12).
_AETHERMAW = "multiplayer_Aethermaw"
_MORPH_HEX_PY = (12, 12)
_MORPH_CODE = "Chw"


def _attack_sig(u):
    return tuple(
        (int(a.type_id), a.number_strikes, a.damage_per_strike,
         a.is_ranged, tuple(sorted(str(s) for s in a.weapon_specials)))
        for a in u.attacks
    )


def _unit_by_id(gs, uid):
    return next(u for u in gs.map.units if u.id == uid)


# ---------------------------------------------------------------------
# 1. The live leak: first_time_only latch on fork-shared ScenarioEvents
# ---------------------------------------------------------------------

def test_fork_turn_event_latch_isolated():
    """A search fork crossing Aethermaw's turn-4 boundary fires the
    `side 1 turn 4` morph event IN THE FORK. The latch (`ev.fired`)
    must stay fork-local: the parent's event must remain unfired and
    the parent must still get its morph when ITS turn 4 arrives.

    Pre-fix failure mode: ScenarioEvent objects were shared across
    forks, so the fork's latch suppressed the live game's morph
    forever (and sibling forks saw inconsistent terrain futures)."""
    sim = fresh_scenario_sim(0, scenario_id=_AETHERMAW)
    events = getattr(sim.gs.global_info, "_scenario_events")
    parent_t4 = [ev for ev in events if ev.name == "side 1 turn 4"]
    assert parent_t4, "Aethermaw must define a `side 1 turn 4` event"
    assert all(not ev.fired for ev in parent_t4)
    codes = getattr(sim.gs.global_info, "_terrain_codes")
    baseline = codes[_MORPH_HEX_PY]
    assert baseline != _MORPH_CODE

    fork = sim.fork()
    # Production path: _apply_command("init_side") calls this at every
    # turn rotation, including turn rotations stepped inside a fork.
    _fire_turn_events(fork.gs, 1, 4)

    # Sanity: the FORK saw its morph and latched its own event.
    fork_events = getattr(fork.gs.global_info, "_scenario_events")
    fork_t4 = [ev for ev in fork_events if ev.name == "side 1 turn 4"]
    assert all(ev.fired for ev in fork_t4)
    assert getattr(fork.gs.global_info,
                   "_terrain_codes")[_MORPH_HEX_PY] == _MORPH_CODE

    # THE leak assertions: the parent is untouched...
    assert all(not ev.fired for ev in parent_t4), (
        "fork's event latch leaked to the parent: the LIVE game would "
        "never fire its first_time_only morph"
    )
    assert getattr(sim.gs.global_info,
                   "_terrain_codes")[_MORPH_HEX_PY] == baseline

    # ...and the REAL game still morphs when its own turn 4 arrives.
    _fire_turn_events(sim.gs, 1, 4)
    assert getattr(sim.gs.global_info,
                   "_terrain_codes")[_MORPH_HEX_PY] == _MORPH_CODE


# ---------------------------------------------------------------------
# 2. The dormant leak: [object] [effect] mutation on fork-shared Units
# ---------------------------------------------------------------------

def test_fork_object_effect_isolated():
    """`_object_action` applies WML [effect]s to units it pulls out of
    `gs.map.units` -- exactly the Unit objects `Map.__deepcopy__`
    shares across forks. An [object] fired inside a fork must not
    rewrite the parent's units.

    (Dormant today: every [object] in the current pools fires from
    prestart / turn-1 events, which run in `WesnothSim.__init__`
    before any fork exists. One scenario addition -- e.g. the armed-
    monster events on the mini-maps roadmap -- would make it live.)"""
    sim = fresh_scenario_sim(0)
    u = min((x for x in sim.gs.map.units if x.side == 1 and x.attacks),
            key=lambda x: x.id)
    before_attacks = _attack_sig(u)
    before_max_hp = u.max_hp
    before_hp = u.current_hp

    obj = WMLNode("object")
    filt = WMLNode("filter")
    filt.attrs = {"x": str(u.position.x + 1), "y": str(u.position.y + 1)}
    eff_atk = WMLNode("effect")
    eff_atk.attrs = {"apply_to": "attack", "increase_damage": "5"}
    eff_hp = WMLNode("effect")
    eff_hp.attrs = {"apply_to": "hitpoints", "increase_total": "10"}
    obj.children = [filt, eff_atk, eff_hp]

    fork = sim.fork()
    _object_action(fork.gs, obj)

    # Sanity: the FORK's unit took the effects.
    fu = _unit_by_id(fork.gs, u.id)
    assert [a.damage_per_strike for a in fu.attacks] == [
        sig[2] + 5 for sig in before_attacks]
    assert fu.max_hp == before_max_hp + 10
    assert fu.current_hp == before_hp + 10

    # THE leak assertions: the parent's unit is untouched.
    pu = _unit_by_id(sim.gs, u.id)
    assert _attack_sig(pu) == before_attacks, (
        "[effect] applied inside a fork leaked into the parent's unit"
    )
    assert pu.max_hp == before_max_hp
    assert pu.current_hp == before_hp
    # Post-fix the fork holds its own replacement object.
    assert fu is not pu


# ---------------------------------------------------------------------
# 3. Executable spec of the fork-shared attack surface
# ---------------------------------------------------------------------

def test_fork_alias_contract():
    """The definitive aliased-vs-copied list for `WesnothSim.fork()`
    (2026-07-29 audit). If this test fails after a deepcopy change,
    update BOTH the fast-path docstrings and the audit conclusion --
    aliasing more is a perf choice that widens the mutation attack
    surface; aliasing less is safe but slower."""
    sim = fresh_scenario_sim(0, scenario_id=_AETHERMAW)
    fork = sim.fork()
    m, fm = sim.gs.map, fork.gs.map
    gi, fgi = sim.gs.global_info, fork.gs.global_info

    # Fresh outer objects.
    assert fork.gs is not sim.gs
    assert fm is not m
    assert fgi is not gi

    # ALIASED (immutable-by-contract; mutators must copy-on-write):
    assert fm.mask is m.mask
    assert fm.fog is m.fog
    assert fm.hexes is m.hexes
    assert getattr(fgi, "_terrain_codes") is getattr(gi, "_terrain_codes")

    # COPIED containers with SHARED leaf objects:
    assert fm.units is not m.units
    parent_by_id = {u.id: u for u in m.units}
    for u in fm.units:
        assert parent_by_id[u.id] is u, (
            "Unit contents are shared by design; mutators must use the "
            "replace-unit pattern"
        )

    # FULLY per-fork:
    assert fork.gs.sides is not sim.gs.sides
    assert all(fs is not ps for fs, ps in zip(fork.gs.sides, sim.gs.sides))
    pv = getattr(gi, "_village_owner", None)
    if pv is not None:
        fv = getattr(fgi, "_village_owner")
        assert fv is not pv and fv == pv

    # Scenario events: per-fork list; UNFIRED events must be per-fork
    # objects (the `fired` latch is mutable state). Already-fired
    # events may stay shared -- their only later mutation is an
    # idempotent re-latch to True. Parsed WML actions stay shared
    # (read-only after parse).
    evs = getattr(gi, "_scenario_events")
    fevs = getattr(fgi, "_scenario_events")
    assert fevs is not evs and len(fevs) == len(evs)
    for pe, fe in zip(evs, fevs):
        if pe.fired:
            continue
        assert fe is not pe, (
            "unfired ScenarioEvent shared across forks: the fired "
            "latch would leak hypothetical futures into the live game"
        )
        assert fe.actions is pe.actions
        assert (fe.name, fe.first_time_only, fe.fired) == \
               (pe.name, pe.first_time_only, pe.fired)


# ---------------------------------------------------------------------
# 4. deep_state_fingerprint: the whole-class detector
# ---------------------------------------------------------------------

def test_deep_fingerprint_stable_across_fork_mutation():
    """Guard property: heavy mutation inside a fork leaves the
    parent's fingerprint bit-identical. This is what SIM_FORK_GUARD=1
    asserts around every mcts_search."""
    sim = fresh_scenario_sim(0, scenario_id=_AETHERMAW)
    fp0 = deep_state_fingerprint(sim.gs)

    fork = sim.fork()
    _fire_turn_events(fork.gs, 1, 4)   # terrain morph + event latch
    u = min((x for x in fork.gs.map.units if x.side == 1 and x.attacks),
            key=lambda x: x.id)
    obj = WMLNode("object")
    filt = WMLNode("filter")
    filt.attrs = {"x": str(u.position.x + 1), "y": str(u.position.y + 1)}
    eff = WMLNode("effect")
    eff.attrs = {"apply_to": "attack", "increase_damage": "5"}
    obj.children = [filt, eff]
    _object_action(fork.gs, obj)       # unit [effect] mutation
    fork.step({"type": "end_turn"})    # a real production step too

    assert deep_state_fingerprint(sim.gs) == fp0, (
        "fork-side mutation changed the parent's deep fingerprint"
    )


def test_deep_fingerprint_covers_surfaces_state_key_misses():
    """Sensitivity: the fingerprint must flip on mutations that
    `state_key` deliberately ignores (state_key answers 'same MCTS
    node?'; the fingerprint answers 'did anything leak?'). Each of
    the three real leak instances lived on such a surface."""
    from wesnoth_ai.classes import state_key

    sim = fresh_scenario_sim(0, scenario_id=_AETHERMAW)
    fp0 = deep_state_fingerprint(sim.gs)
    sk0 = state_key(sim.gs)

    # (a) Event latch (the 2026-07-29 leak surface).
    evs = getattr(sim.gs.global_info, "_scenario_events")
    target = next(ev for ev in evs if not ev.fired)
    target.fired = True
    assert state_key(sim.gs) == sk0          # invisible to state_key
    assert deep_state_fingerprint(sim.gs) != fp0
    target.fired = False
    assert deep_state_fingerprint(sim.gs) == fp0

    # (b) Unit attack table (the [object] surface).
    u = min((x for x in sim.gs.map.units if x.attacks),
            key=lambda x: x.id)
    saved = u.attacks
    u.attacks = []
    assert state_key(sim.gs) == sk0          # invisible to state_key
    assert deep_state_fingerprint(sim.gs) != fp0
    u.attacks = saved
    assert deep_state_fingerprint(sim.gs) == fp0

    # (c) Hex modifiers (the village-bit surface, fa95da5).
    h = next(iter(sim.gs.map.hexes))
    from wesnoth_ai.classes import TerrainModifiers
    added = TerrainModifiers.ILLUMINATED not in h.modifiers
    if added:
        h.modifiers.add(TerrainModifiers.ILLUMINATED)
    else:
        h.modifiers.discard(TerrainModifiers.ILLUMINATED)
    assert deep_state_fingerprint(sim.gs) != fp0
    if added:
        h.modifiers.discard(TerrainModifiers.ILLUMINATED)
    else:
        h.modifiers.add(TerrainModifiers.ILLUMINATED)
    assert deep_state_fingerprint(sim.gs) == fp0

    # (d) Per-unit `_defense_table` stash (coverage gap closed
    # 2026-08-10, user order): shallow unit copies SHARE the dict, so
    # a fork-side write is exactly the aliasing class the guard
    # exists for. Invisible to state_key by design.
    u2 = min(sim.gs.map.units, key=lambda x: x.id)
    had_tbl = hasattr(u2, "_defense_table")
    orig_tbl = getattr(u2, "_defense_table", None)
    setattr(u2, "_defense_table",
            {**(orig_tbl or {}), "village": 1})
    assert state_key(sim.gs) == sk0          # invisible to state_key
    assert deep_state_fingerprint(sim.gs) != fp0
    if had_tbl:
        setattr(u2, "_defense_table", orig_tbl)
    else:
        delattr(u2, "_defense_table")
    assert deep_state_fingerprint(sim.gs) == fp0


def test_fork_guard_no_false_positive_on_real_search(monkeypatch):
    """SIM_FORK_GUARD e2e: a real (tiny-model, few-sims) search with
    the guard forced ON must complete without tripping -- a noisy
    guard would be worse than none. True-positive coverage comes from
    the fingerprint sensitivity test above plus the pre-fix failures
    of the isolation tests."""
    import torch
    from wesnoth_ai.transformer_policy import TransformerPolicy
    import tools.mcts as mcts_mod
    from tools.mcts import MCTSConfig, mcts_search

    torch.manual_seed(0)
    policy = TransformerPolicy(
        d_model=64, num_layers=2, num_heads=4, d_ff=128,
        device=torch.device("cpu"),
    )
    sim = fresh_scenario_sim(0, scenario_id=_AETHERMAW)
    monkeypatch.setattr(mcts_mod, "_FORK_GUARD", True)
    fp0 = deep_state_fingerprint(sim.gs)
    root = mcts_search(
        sim, policy._inference_model, policy._inference_encoder,
        MCTSConfig(n_simulations=8, add_root_noise=False),
    )
    assert root is not None
    assert deep_state_fingerprint(sim.gs) == fp0


def test_fork_guard_violation_is_not_swallowable():
    """Round-34 C2: every game loop wraps games in `except
    Exception` and continues -- a guard trip must NOT be an
    Exception, or the launch smoke exits 0 on a real violation."""
    from tools.mcts import ForkGuardViolation
    assert issubclass(ForkGuardViolation, BaseException)
    assert not issubclass(ForkGuardViolation, Exception)


def test_actor_fatal_error_is_not_swallowable():
    """Round-35 C0: the pool's fatal channel must not be catchable
    by log-and-continue handlers either."""
    from tools.actor_pool import ActorFatalError
    assert issubclass(ActorFatalError, BaseException)
    assert not issubclass(ActorFatalError, Exception)
