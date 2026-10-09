"""Regression test: combat damage seed alignment against Wesnoth.

Pinned protection for the chain
  `_action_to_command -> request_seed(N) -> [random_seed request_id=N]`
  → `mt_rng::seed_random(seed_str, 0)` → the core's Mersenne Twister.

Each per-strike (chance, hits, damage) the Rust core plays must match
Wesnoth's recorded `[mp_checkup]` ground truth bit-exactly. A silent
regression (a reseed, a request_seed increment bug, a wrong
attacker_first ordering) then fails CI rather than surfacing during a
multi-week training run.

Fixture: `tests/fixtures/strict_sync_hamlets_t9.bz2` — a 9-turn AI-vs-AI
strict-sync (oos_debug=yes) Hamlets replay with 29 [attack] commands
and 1078 mp_checkup result entries.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tools.replay_extract import extract_replay
from tools.verify_mp_checkup import parse_replay as parse_strict_replay
from wesnoth_ai import game_core as gc

FIXTURE = Path(__file__).parent / "fixtures" / "strict_sync_hamlets_t9.bz2"

pytestmark = pytest.mark.skipif(gc.game_core_class() is None,
                                reason="the installed wesnoth_core wheel is older than game_core needs")


def _played_strikes(cs) -> list:
    """(chance, hits, damage) of each strike of the last attack the core
    applied, from its [mp_checkup] record."""
    flat = cs.core.last_checkup_strikes_export()
    return [(int(flat[k]), bool(flat[k + 1]), int(flat[k + 2])) for k in range(0, len(flat), 4)]


def _first_mismatch(played, recorded) -> str:
    """The first strike where the core and the engine differ, or ""."""
    for i, (chance, hits, damage) in enumerate(played):
        if i >= len(recorded):
            return f"the core plays {len(played)} strikes, the engine {len(recorded)}"
        rec = recorded[i]
        if chance != rec.chance:
            return f"strike {i}: chance {chance} against {rec.chance}"
        if hits != rec.hits:
            return f"strike {i}: hits {hits} against {rec.hits} (chance {rec.chance})"
        if hits and damage != rec.damage:
            return f"strike {i}: damage {damage} against {rec.damage}"
    if len(recorded) > len(played):
        return f"the engine plays {len(recorded)} strikes, the core {len(played)}"
    return ""


def test_strict_sync_combat_bit_exact():
    """Every recorded strike on the fixture matches the core's fight
    bit-exactly."""
    from tools.replay_dataset import record_core
    wesnoth_attacks = parse_strict_replay(FIXTURE)
    assert wesnoth_attacks, "fixture has no [attack] commands"
    assert any(a.strikes for a in wesnoth_attacks), (
        "fixture has no mp_checkup strike data; was it recorded with oos_debug=yes?")
    data = extract_replay(FIXTURE)
    assert data is not None, "extract_replay returned None on fixture"

    cs = record_core(data)
    attack_idx = 0
    failures: list[str] = []
    n_strikes = 0
    for i, cmd in enumerate(data["commands"]):
        cs.apply_command(list(cmd))
        if cmd[0] != "attack":
            continue
        assert attack_idx < len(wesnoth_attacks), (
            f"attack #{attack_idx + 1} at cmd[{i}] but Wesnoth recorded only {len(wesnoth_attacks)}")
        recorded = wesnoth_attacks[attack_idx]
        played = _played_strikes(cs)
        n_strikes += len(played)
        mismatch = _first_mismatch(played, recorded.strikes)
        if mismatch:
            failures.append(f"cmd[{i}] attack #{attack_idx} ({recorded.attacker_type} -> "
                            f"{recorded.defender_type} weapons {cmd[5]}/{cmd[6]} seed {cmd[7]}): {mismatch}")
        attack_idx += 1

    assert attack_idx == len(wesnoth_attacks) > 0
    assert n_strikes > 0, "no strike was compared"
    assert not failures, (f"{len(failures)} of {attack_idx} attacks diverged from Wesnoth's record:\n  "
                          + "\n  ".join(failures))
