"""The serve threads' batch picker (tools/actor_pool._BatchPicker,
design note section 7): the fifo policy is the arrival-order rule the
pool always had; the length policy groups the queued requests by token
count, never delays a request by more than one batch, and applies the
gap rule; the telemetry counts what it did."""
from __future__ import annotations

import queue
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def _request(actor: int, tokens: int, leaves: int):
    """(actor id, request id, payload) with a PackedRequest-shaped
    payload: headers carrying the per-leaf hex and unit counts."""
    return (actor, actor, SimpleNamespace(
        headers=[SimpleNamespace(n_hexes=tokens - 3, n_units=3) for _ in range(leaves)]))


def _queued(specs):
    q = queue.Queue()
    for actor, (tokens, leaves) in enumerate(specs):
        q.put(_request(actor, tokens, leaves))
    return q


def _actors(batch):
    return [w.item[0] for w in batch]


def test_fifo_takes_arrival_order_until_max_batch_with_the_overshoot_rule():
    from tools.actor_pool import _BatchPicker
    q = _queued([(600, 8), (2100, 6), (620, 8), (1200, 8)])
    pk = _BatchPicker("fifo")
    # 8 < 16 so the second request joins even though it overshoots to 14;
    # 14 < 16 so the third joins too (22 leaves): the rule since 2026-07-22.
    assert _actors(pk.take(q, 16, 0.01)) == [0, 1, 2]
    assert _actors(pk.take(q, 16, 0.01)) == [3]
    assert pk.take(q, 16, 0.01) == []
    assert pk.skipped == 0 and pk.picks == 2 and pk.depth == 5


def test_length_groups_by_tokens_and_forces_a_skipped_request_next():
    from tools.actor_pool import _BatchPicker
    q = _queued([(600, 8), (2100, 8), (620, 8), (1200, 8), (610, 8)])
    pk = _BatchPicker("length")
    # Five requests wait: the oldest (600) anchors and the nearest, 610,
    # fills the batch to 16 leaves; 620, 1200 and 2100 are left behind.
    first = pk.take(q, 16, 0.01)
    assert _actors(first) == [0, 4]
    assert pk.skipped == 3
    # Every request left behind is forced into the next batch, in age
    # order, before any anchor rule: 2100, 620 fill it (16 leaves).
    second = pk.take(q, 16, 0.01)
    assert _actors(second) == [1, 2]
    assert pk.skipped == 4                    # 1200 was left behind a second time
    assert _actors(pk.take(q, 16, 0.01)) == [3]
    assert pk.take(q, 16, 0.01) == []
    assert pk.picks == 3 and pk.depth == 5 + 3 + 1


def test_length_without_depth_is_fifo():
    from tools.actor_pool import _BatchPicker
    q = _queued([(2100, 8), (600, 6)])
    pk = _BatchPicker("length")
    assert _actors(pk.take(q, 16, 0.01)) == [0, 1]     # everything fits: no choice made
    assert pk.skipped == 0


def test_gap_rule_splits_a_batch_that_would_fit():
    from tools.actor_pool import _BatchPicker
    q = _queued([(600, 4), (2100, 4), (650, 4)])
    pk = _BatchPicker("length", gap=128)
    assert _actors(pk.take(q, 16, 0.01)) == [0, 2]     # 2100 is 1500 tokens away
    assert _actors(pk.take(q, 16, 0.01)) == [1]        # and goes out next, alone
    assert pk.skipped == 1


def test_take_blocks_only_when_nothing_waits_and_records_lengths():
    from tools.actor_pool import _BatchPicker
    q = queue.Queue()
    pk = _BatchPicker("fifo")
    assert pk.take(q, 16, 0.01) == []                  # timed out on the empty queue
    q.put(_request(0, 700, 3))
    batch = pk.take(q, 16, 0.01)
    assert len(batch) == 1 and batch[0].n_leaves == 3
    assert batch[0].lens == [700, 700, 700] and batch[0].tokens == 700


def test_legacy_payloads_count_tokens_from_the_raws():
    from tools.actor_pool import _request_lengths
    raw = SimpleNamespace(hex_xs=[0] * 40, unit_xs=[0] * 5)
    assert _request_lengths([raw, (raw, None)]) == [45, 45]


def test_unknown_policy_is_refused():
    from tools.actor_pool import _BatchPicker
    with pytest.raises(ValueError, match="fifo"):
        _BatchPicker("shortest")
