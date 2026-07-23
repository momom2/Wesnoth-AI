"""No-progress stalemate rule (2026-07-21, chess 50-move analog).

Progress = any objective state change: unit count (kill/recruit),
net HP decrease (combat/poison damage), or village ownership. After
K consecutive FULL turns without one, the game ends as a draw
(`ended_by='no_progress'`). K=0 disables enforcement but the
tracker still records would-fire statistics for offline evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402


def _spin_turns(sim, n_turns: int) -> None:
    """Advance the game n_turns full turns via end_turn spam."""
    target = sim.gs.global_info.turn_number + n_turns
    guard = 0
    while (sim.gs.global_info.turn_number < target
           and not sim.done and guard < 500):
        sim.step({"type": "end_turn"})
        guard += 1


def test_rule_fires_after_k_quiet_turns():
    sim = fresh_scenario_sim(seed=5, max_turns=60, mini=True)
    sim.no_progress_turns = 4
    _spin_turns(sim, 10)
    assert sim.done
    assert sim.ended_by == "no_progress"
    assert sim.winner == 0
    # Fired at K quiet turns, far before the turn cap.
    assert sim.gs.global_info.turn_number <= 8


def _hook_mutation_into_step(sim, at_call: int, mutate) -> None:
    """Apply `mutate(sim)` DURING the `at_call`-th _step_inner call --
    the detector fingerprints before/after each step, so progress
    must land inside a step window, exactly like real events do."""
    orig = sim._step_inner
    calls = {"n": 0}

    def hooked(action):
        r = orig(action)
        calls["n"] += 1
        if calls["n"] == at_call:
            mutate(sim)
        return r

    sim._step_inner = hooked


def test_progress_resets_the_clock():
    sim = fresh_scenario_sim(seed=5, max_turns=60, mini=True)
    sim.no_progress_turns = 4

    # Deal damage during the 8th step (mid-quiet): net HP drop is a
    # progress event and must reset the streak.
    def dmg(s):
        u = next(iter(s.gs.map.units))
        u.current_hp = max(1, u.current_hp - 5)
    _hook_mutation_into_step(sim, 8, dmg)
    _spin_turns(sim, 3)
    assert not sim.done
    _spin_turns(sim, 3)
    assert not (sim.done and sim.ended_by == "no_progress")
    summary = sim.noprogress_summary()
    assert summary["resumed_streaks"], \
        "the pre-damage quiet streak must be recorded as resumed"


def test_observe_mode_tracks_but_never_ends():
    sim = fresh_scenario_sim(seed=5, max_turns=12, mini=True)
    assert sim.no_progress_turns == 0    # default: rule off
    _spin_turns(sim, 15)
    assert sim.ended_by == "max_turns"   # cap ends it, not the rule
    s = sim.noprogress_summary()
    assert s["max_quiet"] >= 8
    assert s["tail_quiet"] >= 8


def test_village_capture_counts_as_progress():
    sim = fresh_scenario_sim(seed=5, max_turns=60, mini=True)
    sim.no_progress_turns = 4

    def capture(s):
        vo = dict(getattr(s.gs.global_info, "_village_owner", None)
                  or {})
        vo[(1, 1)] = 1
        s.gs.global_info._village_owner = vo
    _hook_mutation_into_step(sim, 8, capture)
    _spin_turns(sim, 3)
    _spin_turns(sim, 3)
    assert not (sim.done and sim.ended_by == "no_progress")


def test_fork_carries_rule_state():
    sim = fresh_scenario_sim(seed=5, max_turns=60, mini=True)
    sim.no_progress_turns = 4
    _spin_turns(sim, 2)
    f = sim.fork()
    assert f.no_progress_turns == 4
    assert f._last_progress_turn == sim._last_progress_turn
    # The fork enforces independently of the parent.
    _spin_turns(f, 8)
    assert f.done and f.ended_by == "no_progress"
    assert not sim.done
