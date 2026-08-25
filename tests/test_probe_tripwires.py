"""A1 value-accuracy alarm predicates (credit-assignment review
2026-08-17). Leg 3 ran 21 iterations with value_auc below chance in
a column nobody read; these predicates are the alarm. Behavioral
guards only: a silently-broken tripwire re-burns a leg.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.holdout_probe_loop import (  # noqa: E402
    qualify_verdict, redraw_verdict,
)


def _rows(*aucs):
    return [{"ce": "3.5", "value_auc": a, "decision_step": str(i)}
            for i, a in enumerate(aucs)]


def test_redraw_verdict_aborts_only_when_all_draws_fail():
    """User ruling 2026-08-25: a low reading triggers independent
    sample redraws; abort ONLY if all n draws are below threshold."""
    assert redraw_verdict([0.55, 0.58, 0.59], floor=0.60, n=3)
    assert not redraw_verdict([0.55, 0.61, 0.58], floor=0.60, n=3)
    assert not redraw_verdict([0.55, 0.58], floor=0.60, n=3)


def test_redraw_verdict_missing_reading_cannot_abort():
    """A failed probe (None) is not evidence of a broken judge."""
    assert not redraw_verdict([0.55, None, 0.58], floor=0.60, n=3)
    assert not redraw_verdict([None, None, None], floor=0.60, n=3)


def test_qualify_refuses_unmeasured_and_below_bar():
    ok, _ = qualify_verdict({"value_auc": "0.71"}, 0.60)
    assert ok
    bad, why = qualify_verdict({"value_auc": "0.309"}, 0.60)
    assert not bad and "0.309" in why
    miss, why = qualify_verdict({"value_auc": ""}, 0.60)
    assert not miss and "missing" in why


def test_qualify_gate_wired_into_launcher():
    """The entry gate ran BY HAND for legs 4 and 5 (BACKLOG gap).
    Pin the wiring: onstart must invoke --qualify on the checkpoint
    training starts from, treat a measured refusal as an ABORTED_
    marker (human decision, no silent relaunch), and keep probe
    failure (rc 2) distinct from refusal (rc 3) -- only the latter
    blocks restarts."""
    onstart = (Path(__file__).parent.parent
               / "scripts/vast_onstart.sh").read_text(encoding="utf-8")
    assert '--qualify "$CKPT_IN"' in onstart
    assert "ABORTED_qualify" in onstart
    # marker keyed by campaign identity, so a re-gated box on a new
    # leg cannot inherit the previous leg's pass
    assert '.qualified_${CAMPAIGN_FILE}' in onstart


def test_k_median_of_matches_csv_statistic():
    """--abort-k-median consumes k_median_of; it must equal the
    actions_per_turn_median CSV statistic (pooled side-turn action
    counts, lower median) and return None with no data -- the
    tripwire must not fire on an empty iteration."""
    from types import SimpleNamespace
    from tools.sim_self_play import k_median_of

    games = [SimpleNamespace(turn_action_counts=[12, 2, 15]),
             SimpleNamespace(turn_action_counts=[3, 14]),
             SimpleNamespace(turn_action_counts=None)]
    assert k_median_of(games) == 12       # sorted [2,3,12,14,15]
    assert k_median_of([]) is None
    assert k_median_of(
        [SimpleNamespace(turn_action_counts=[])]) is None
