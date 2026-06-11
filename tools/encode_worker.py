"""Worker-side encoding pipeline for supervised_train.py.

When `tools/supervised_train.py --workers N>0` is in effect, the main
process spawns N of these workers. Each one:

  1. Pops a `gz_path` from the input queue.
  2. Replays it via `replay_dataset.iter_replay_pairs` to yield
     (GameState, ActionIndices) pairs.
  3. Phase-1 encodes each pair via `encoder.encode_raw` against the
     read-only vocab dicts that were given at startup.
  4. Accumulates ALL pairs from this replay into a list and pushes one
     `("file", pairs, gz_name)` message to the output queue.

Why one message per replay (not per pair): the multiprocessing.Queue
boundary pickles every message. RawEncoded carries ~5,000 small Python
objects (hex int lists, modifier flag rows, ...), and pickling those
once per pair turned out to dominate step time on the cluster — the
first version of this worker pumped a per-pair message and ate the
encoder savings (and then some) in queue overhead. Batching by replay
amortizes pickle cost across ~50–200 pairs per put().

Memory: a typical replay's pair list is 0.5–2 MB pickled. With N
workers and a prefetch_factor of 4, the trainer holds ~N*4 messages
on average — bounded RAM, comfortably within cluster headroom.

Workers never mutate the vocab dicts and never touch nn parameters,
so vocab IDs stay stable across processes. Out-of-vocab unit / faction
names hit the overflow bucket (handled by `encode_raw`); the trainer
pre-seeds the vocab from `unit_stats.json` before spawning workers so
overflow only catches genuinely-rare names.

Vocab discipline: the dicts are passed at worker startup as plain
Python dicts. They're shared via fork() inheritance on Linux (zero
copy, copy-on-write) and via re-pickling on Windows spawn (one copy
per worker, 100s of KB). Either way, workers treat them as read-only.

Failure mode: any exception inside the worker on a single replay is
caught, logged, and turned into a `("file_error", gz_name, err_str)`
message — the trainer skips that file and moves on. A worker only
exits cleanly when it pops the sentinel `None` from the input queue.
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict


# Resolve the project root from this file's location so workers can
# `import encoder` and `import replay_dataset` regardless of how they
# were spawned (Linux fork inherits sys.path; Windows spawn re-bootstraps
# the interpreter and would otherwise miss the project root).
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parent.parent
_TOOLS_DIR    = _THIS.parent


def worker_main(
    in_q,
    out_q,
    type_to_id: Dict[str, int],
    faction_to_id: Dict[str, int],
    log_level: int = logging.WARNING,
) -> None:
    """Worker entry point.

    Each item written to `out_q` is one of:
      ("file",       pairs, gz_name)        # pairs = list of (RawEncoded, ActionIndices)
      ("file_error", gz_name, err_str)
      ("worker_exit",)

    The trainer's main loop unpacks "file" messages into individual
    ("pair", raw, ai, gz_name) events for its own consumption (see
    `_ParallelStream` in supervised_train.py). The "file_done" marker
    that drives gc / batch-flush bookkeeping is synthesized by the
    stream after a file's pairs are drained — workers don't emit it
    explicitly anymore.
    """
    # Re-bootstrap import paths for Windows spawn — fork would inherit.
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    if str(_TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(_TOOLS_DIR))

    # Local imports so they happen post-bootstrap.
    from encoder import encode_raw
    from replay_dataset import iter_replay_pairs

    logging.basicConfig(level=log_level, format="%(message)s")
    log = logging.getLogger("encode_worker")

    while True:
        item = in_q.get()
        if item is None:
            out_q.put(("worker_exit",))
            return

        gz_path: Path = item
        gz_name = gz_path.name
        try:
            pairs = []
            for state, ai in iter_replay_pairs(gz_path):
                raw = encode_raw(
                    state,
                    type_to_id=type_to_id,
                    faction_to_id=faction_to_id,
                )
                pairs.append((raw, ai))
            # Single put() amortizes pickle cost across all pairs from
            # this replay. Empty lists are valid (replay had no
            # actionable pairs) — the trainer will see file_done with
            # n=0 and move on.
            out_q.put(("file", pairs, gz_name))
        except Exception as e:
            tb = traceback.format_exception_only(type(e), e)[-1].strip()
            out_q.put(("file_error", gz_name, tb))
            log.debug(f"  worker skip {gz_name}: {e}")
            # Keep going; main thread tolerates per-file failures.
