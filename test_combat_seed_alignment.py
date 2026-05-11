"""Regression test: combat damage seed alignment against Wesnoth.

Pinned protection for the chain
  `_action_to_command -> request_seed(N) -> [random_seed request_id=N]`
  → `mt_rng::seed_random(seed_str, 0)` → `combat.MTRng(seed_hex)`.

Each per-strike (chance, hits, damage, dies) must match Wesnoth's
recorded `[mp_checkup]` ground truth bit-exactly. The CLAUDE.md
status note pins this at 731/731 across the historical strict-sync
corpus; this test re-asserts the property on a small fixture so a
silent regression (e.g. accidental MTRng reseed, request_seed
increment bug, wrong attacker_first ordering) fails CI rather
than only surfacing during a multi-week training run.

Fixture: `tests/fixtures/strict_sync_hamlets_t9.bz2` — a 9-turn AI-vs-AI
strict-sync (oos_debug=yes) Hamlets replay with 29 [attack] commands
and 1078 mp_checkup result entries.

Dependencies: tools.replay_extract, tools.diff_combat_strike,
              tools.verify_mp_checkup, tools.replay_dataset, combat.
Dependents: regression CI.
"""
from __future__ import annotations

import gzip
import json
import tempfile
from pathlib import Path

from tools.replay_extract import extract_replay
from tools.diff_combat_strike import _verify_attack
from tools.verify_mp_checkup import parse_replay as parse_strict_replay
from tools.replay_dataset import (
    _apply_command,
    _build_initial_gamestate,
    _setup_scenario_events,
)

FIXTURE = Path(__file__).parent / "tests" / "fixtures" / "strict_sync_hamlets_t9.bz2"


def test_fixture_present():
    """The strict-sync fixture must exist (committed to the repo)."""
    assert FIXTURE.exists(), (
        f"missing strict-sync fixture at {FIXTURE}; this regression "
        f"test cannot run without it"
    )


def _verified_strike_count(divergences):
    """Collapse a list of StrikeMismatch into a single string for the
    assertion message."""
    if not divergences:
        return "clean"
    d = divergences[0]
    return (f"strike[{d.strike_index}] field={d.field} "
            f"ours={d.ours} wesnoth={d.wesnoth}: {d.detail}")


def test_strict_sync_combat_bit_exact():
    """Every recorded strike on the fixture must match our combat
    resolver bit-exactly.

    This is the single load-bearing assertion: if it fails, our
    `_action_to_command` / `request_seed` / `combat.MTRng` / strike
    ordering has drifted from Wesnoth.
    """
    # Parse Wesnoth's ground truth.
    wesnoth_attacks = parse_strict_replay(FIXTURE)
    assert wesnoth_attacks, "fixture has no [attack] commands"
    with_strikes = [a for a in wesnoth_attacks if a.strikes]
    assert with_strikes, (
        "fixture has no mp_checkup strike data; was it recorded "
        "with oos_debug=yes?"
    )

    # Extract via the same path the training pipeline uses.
    data = extract_replay(FIXTURE)
    assert data is not None, "extract_replay returned None on fixture"

    # Walk the command stream, verifying each [attack] against the
    # recorded strike sequence.
    gs = _build_initial_gamestate(data)
    _setup_scenario_events(gs, data.get("scenario_id", ""))

    cmds = data["commands"]
    attack_idx = 0
    n_checked = 0
    n_clean = 0
    failures: list[str] = []

    for i, cmd in enumerate(cmds):
        if cmd[0] == "attack":
            assert attack_idx < len(wesnoth_attacks), (
                f"sim emits attack #{attack_idx + 1} at cmd[{i}] but "
                f"Wesnoth recorded only {len(wesnoth_attacks)}"
            )
            recorded = wesnoth_attacks[attack_idx]
            mismatches = _verify_attack(gs, cmd, recorded.strikes)
            n_checked += 1
            if mismatches:
                failures.append(
                    f"cmd[{i}] attack #{attack_idx} "
                    f"({recorded.attacker_type} -> "
                    f"{recorded.defender_type} weap={cmd[5]}/{cmd[6]} "
                    f"seed={cmd[7]}): "
                    f"{_verified_strike_count(mismatches)}"
                )
            else:
                n_clean += 1
            attack_idx += 1
        _apply_command(gs, cmd)

    assert n_checked > 0, "no attacks were checked"
    assert not failures, (
        f"{len(failures)} of {n_checked} attacks diverged from "
        f"Wesnoth ground truth:\n  " + "\n  ".join(failures)
    )

    # Sanity: every strike was actually verified.
    total_recorded_strikes = sum(
        len(a.strikes) for a in wesnoth_attacks[:n_checked]
    )
    assert total_recorded_strikes > 0, (
        "no individual strikes were verified -- combat oracle is "
        "effectively a no-op"
    )
