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
    _auc_tail_trips, qualify_verdict,
)


def _rows(*aucs):
    return [{"ce": "3.5", "value_auc": a, "decision_step": str(i)}
            for i, a in enumerate(aucs)]


def test_auc_tripwire_fires_on_the_leg3_pattern():
    # Leg 3's actual probe series began 0.309, 0.401, 0.364 -- the
    # alarm must fire on exactly this, at the default floor.
    assert _auc_tail_trips(_rows("0.309", "0.401", "0.364"),
                           floor=0.52, n=3)


def test_auc_tripwire_needs_consecutive_breakage():
    # One healthy probe inside the tail resets the alarm.
    assert not _auc_tail_trips(_rows("0.30", "0.65", "0.31"),
                               floor=0.52, n=3)
    # Fewer rows than the window can't trip.
    assert not _auc_tail_trips(_rows("0.30", "0.31"), floor=0.52, n=3)


def test_auc_tripwire_missing_values_cannot_trip():
    # A probe with no value-labeled pairs (empty cell) is not
    # evidence of breakage.
    assert not _auc_tail_trips(_rows("0.30", "", "0.31"),
                               floor=0.52, n=3)



def test_qualify_refuses_unmeasured_and_below_bar():
    ok, _ = qualify_verdict({"value_auc": "0.71"}, 0.60)
    assert ok
    bad, why = qualify_verdict({"value_auc": "0.309"}, 0.60)
    assert not bad and "0.309" in why
    miss, why = qualify_verdict({"value_auc": ""}, 0.60)
    assert not miss and "missing" in why


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
