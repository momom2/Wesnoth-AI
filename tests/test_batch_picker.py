"""The serve threads' batch picker (tools/serve_worker._BatchPicker,
design note section 7): the fifo policy is the arrival-order rule the
pool always had; the length policy groups the queued requests by token
count, serves a request it displaced in the very next batch, and
applies the gap rule; the telemetry counts what it did. Also the serve
loop's stop: every request parked in the picker gets the failure
reply."""
from __future__ import annotations

import queue
import random
import sys
import threading
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


def test_length_groups_by_tokens_and_serves_a_displaced_request_next():
    from tools.actor_pool import _BatchPicker
    q = _queued([(600, 8), (2100, 8), (620, 8), (1200, 8), (610, 8)])
    pk = _BatchPicker("length")
    # Five requests wait: the oldest (600) anchors and the nearest, 610,
    # fills the batch to 16 leaves. Only 2100 is displaced: the fifo
    # rule would have served it now; 620 and 1200 would have waited
    # under either rule.
    first = pk.take(q, 16, 0.01)
    assert _actors(first) == [0, 4]
    assert pk.skipped == 1
    # The displaced request goes first, then the anchor rule around it:
    # 1200 is nearer to 2100 than 620 is. 620 was inside this pick's
    # fifo batch, so it is displaced in turn ...
    second = pk.take(q, 16, 0.01)
    assert _actors(second) == [1, 3]
    assert pk.skipped == 2
    # ... and served next, alone.
    assert _actors(pk.take(q, 16, 0.01)) == [2]
    assert pk.skipped == 2
    assert pk.take(q, 16, 0.01) == []
    assert pk.picks == 3 and pk.depth == 5 + 3 + 1


def _fifo_batch(pending, max_batch):
    """The requests the fifo rule serves now: arrival order until the
    batch holds max_batch leaves, the last one overshooting."""
    n, k = 0, 0
    while k < len(pending) and n < max_batch:
        n += pending[k][1]
        k += 1
    return {actor for actor, _ in pending[:k]}


def test_length_displaces_a_request_at_most_once_and_serves_it_next():
    """The one-batch bound over random queues: a request inside the
    fifo batch of a pick that the length rule passes over is in the
    very next batch, and `skipped` counts each such request once."""
    from tools.actor_pool import _BatchPicker
    rng = random.Random(7)
    for _trial in range(40):
        specs = [(rng.choice([300, 600, 620, 900, 1200, 2100]), rng.randint(3, 16))
                 for _ in range(rng.randint(2, 9))]
        q = _queued(specs)
        pk = _BatchPicker("length", gap=rng.choice([0, 0, 128]))
        pending = [(actor, leaves) for actor, (_, leaves) in enumerate(specs)]
        displaced_last, n_displaced = set(), 0
        while pending:
            fifo = _fifo_batch(pending, 16)
            batch = set(_actors(pk.take(q, 16, 0.01)))
            assert batch, specs
            assert displaced_last <= batch, (specs, displaced_last, batch)
            displaced_last = fifo - batch
            n_displaced += len(displaced_last)
            pending = [(a, n) for a, n in pending if a not in batch]
        assert pk.skipped == n_displaced <= len(specs), specs


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


def test_serving_stop_fails_every_parked_request():
    """A request the picker lifted out of the queue but never batched
    gets the failure reply when serving stops: the picker is rebuilt
    on the next start, so nothing else would ever answer it and its
    actor would block forever (the hard-deadline abandon reaches the
    stop with actors in flight)."""
    from tools.serve_worker import _BatchPicker, _serve_loop
    stop = threading.Event()

    class Server:
        def infer_batch(self, flat, stats=None):
            stop.set()                    # serving stops with requests parked
            raise RuntimeError("stopping")

    req = queue.Queue()
    for actor in range(3):
        req.put((actor, 100 + actor, [SimpleNamespace(hex_xs=[0] * 5, unit_xs=[0])]))
    resp = [queue.Queue() for _ in range(3)]
    stats = []
    # max_batch 1: the first pick takes one request and parks two.
    _serve_loop(Server(), _BatchPicker("fifo"), req, resp, 1, 0.01, stop, stats)
    replies = {actor: resp[actor].get(timeout=1.0) for actor in range(3)}
    assert replies == {0: (100, None), 1: (101, None), 2: (102, None)}
    assert len(stats) == 1 and req.empty()
