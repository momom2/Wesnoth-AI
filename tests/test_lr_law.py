"""tools/lr_law.py: the learning-rate history's areas, the fit of the loss
law, and the rule that holds the rate, lowers it, or stops to look."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import lr_law  # noqa: E402
from tools.lr_law import Areas, Point  # noqa: E402

PEAK = 2.8e-4


def _history(warmup=3, hold=40, decay=30):
    return ([PEAK * (s + 1) / warmup for s in range(warmup - 1)] + [PEAK] * (hold + 1)
            + [PEAK * (1 - (k + 1) / decay) for k in range(decay)])


def test_the_areas_follow_their_definition():
    """S1 sums every rate; S2 sums, from the end of the warm-up, each step's
    decrease of the rate counted again at every later step, fading by
    LAW_LAMBDA a step."""
    rates, warm = _history(), 2
    a = Areas()
    for s, r in enumerate(rates):
        a.step(r, warming=s < warm)
    lam = lr_law.LAW_LAMBDA
    post = rates[warm:]
    s2 = sum(sum((post[k - 1] - post[k]) * lam ** (i - k) for k in range(1, i + 1)) for i in range(len(post)))
    assert a.s1 == pytest.approx(sum(rates)) and a.s2 == pytest.approx(s2)
    assert a.s2 > 0, "the lowering counts"
    restored = Areas.from_dict(a.to_dict())
    assert restored == a


def _law_points(law, n_hold=12, n_decay=6, noise=0.0, seed=0, steps_per_point=600):
    """Probes of a run that holds the peak, then lowers it to 0, read
    through `law` = (L0, A, alpha, C)."""
    rng = np.random.default_rng(seed)
    rates = [PEAK] * (n_hold * steps_per_point) + [PEAK * (1 - (k + 1) / (n_decay * steps_per_point))
                                                   for k in range(n_decay * steps_per_point)]
    a, out = Areas(), []
    for s, r in enumerate(rates):
        a.step(r, warming=False)
        if (s + 1) % steps_per_point == 0:
            L0, A, alpha, C = law
            out.append(Point(a.s1, a.s2, L0 + A * a.s1 ** -alpha - C * a.s2 + rng.normal(0, noise)))
    return out, a


def test_a_fit_recovers_the_law_that_made_the_probes():
    truth = (2.0, 1.5, 0.25, 1.2)
    points, _ = _law_points(truth, noise=0.004)
    law = lr_law.fit(points)
    assert law.rms < 0.006 and law.n == 18
    for got, want in zip((law.L0, law.A, law.alpha, law.C), truth):
        assert got == pytest.approx(want, rel=0.15)


def test_the_rule_holds_while_an_epoch_is_worth_it_then_lowers():
    points, a = _law_points((2.0, 1.5, 0.25, 1.2), noise=0.002)
    earlier, this_pass = points[:-1], points[-1:]
    law = lr_law.fit(points)
    gain = law.epoch_gain(a.s1, PEAK, 8000)
    assert gain == pytest.approx(float(law.predict(a.s1, a.s2) - law.predict(a.s1 + PEAK * 8000, a.s2))), \
        "an epoch at the peak adds the peak rate times the epoch's steps to S1, and nothing to S2"
    assert gain > 0
    held = lr_law.decide(earlier, this_pass, a.s1, PEAK, 8000, threshold=gain / 2)
    assert held.action == "hold" and held.gain == pytest.approx(gain, rel=0.05)
    lowered = lr_law.decide(earlier, this_pass, a.s1, PEAK, 8000, threshold=gain * 2)
    assert lowered.action == "lower"
    assert lr_law.decide(points[:4], [], a.s1, PEAK, 8000, threshold=0.0).action == "hold", \
        "too few probes to fit: hold"


def test_the_rule_stops_to_look_when_the_latest_probes_leave_the_law():
    points, a = _law_points((2.0, 1.5, 0.25, 1.2), noise=0.002)
    earlier = points[:-3]
    drifted = [*points[-3:-2], *(Point(p.s1, p.s2, p.loss + 0.2) for p in points[-2:])]
    looked = lr_law.decide(earlier, drifted, a.s1, PEAK, 8000, threshold=0.0)
    assert looked.action == "review", looked.reason
    one_off = [*points[-3:-1], Point(points[-1].s1, points[-1].s2, points[-1].loss + 0.2)]
    assert lr_law.decide(earlier, one_off, a.s1, PEAK, 8000, threshold=0.0).action != "review", \
        "one probe off the law is noise, not a departure"


def test_a_lowering_the_law_has_seen_is_predicted():
    """The law fitted on a run that lowered its rate once predicts what a
    longer hold before the same lowering reaches."""
    truth = (2.0, 1.5, 0.25, 1.2)
    points, _ = _law_points(truth, noise=0.002)
    law = lr_law.fit(points)
    longer, _ = _law_points(truth, n_hold=20)
    hold_end = Areas()
    for _ in range(20 * 600):
        hold_end.step(PEAK, warming=False)
    assert lr_law.predict_lowering(law, hold_end, PEAK, 6 * 600) == pytest.approx(longer[-1].loss, abs=0.01)


def test_a_law_fitted_before_any_lowering_does_not_pretend_to_know_one():
    points, a = _law_points((2.0, 1.5, 0.25, 1.2), noise=0.002)
    before = [p for p in points if p.s2 == 0]
    law = lr_law.fit(before)
    assert not law.lowering_seen and law.C == 0.0
    with pytest.raises(ValueError, match="no probe after a lowering"):
        lr_law.predict_lowering(law, a, PEAK, 600)
    assert lr_law.fit(points).lowering_seen


def test_the_rule_reads_this_pass_s_probes():
    """Earlier probes of a run that barely progresses say lower; this pass's
    probes, falling faster, raise the fitted gain above the threshold (and,
    being below the law, do not stop the run for a look)."""
    law = lambda s1: 2.5 + 0.3 * s1 ** -0.25                       # noqa: E731
    earlier = [Point(s1, 0.0, law(s1)) for s1 in [0.5 * k for k in range(1, 13)]]
    this_pass = [Point(s1, 0.0, law(s1) - 0.08 * (i + 1)) for i, s1 in enumerate((6.5, 7.0, 7.5))]
    alone = lr_law.decide(earlier, [], 7.5, PEAK, 8000, threshold=0.018)
    both = lr_law.decide(earlier, this_pass, 7.5, PEAK, 8000, threshold=0.018)
    assert (alone.action, both.action) == ("lower", "hold") and both.gain > alone.gain


def test_the_replay_lowers_on_the_positions_trained_before_each_step():
    """A pass of 45 positions lowering from its middle (22.5): at 5 positions
    a step, step 5 starts at 25 and is lowered while step 4 (at 20) is not;
    spread evenly (4.5 a step), step 5 would start at 22 and hold."""
    rates = lr_law.pass_rates(10, 45, PEAK, 0, 0.5, positions_per_step=5)
    assert rates[4] == PEAK and rates[5] < PEAK
    assert lr_law.pass_rates(10, 45, PEAK, 0, 0.5)[5] == PEAK
