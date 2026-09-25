"""Worker-side encoding pipeline for supervised_train.py.

When `tools/supervised_train.py --workers N>0` is in effect, the main
process spawns N of these workers. Each one:

  1. Pops a `(seq, gz_path)` from the input queue; `seq` is the file's
     position in the trainer's dispatch order.
  2. Replays it via `replay_dataset.iter_replay_pairs` to yield
     (GameState, ActionIndices) pairs.
  3. Phase-1 encodes each pair via `encoder.encode_raw` against the
     read-only vocab dicts that were given at startup.
  4. Accumulates ALL pairs from this replay into a list and pushes one
     `("file", seq, pairs, gz_name)` message to the output queue. The
     trainer emits files in `seq` order whatever order they complete
     in, so a seeded file order gives one pair stream at any worker
     count.

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
caught, logged, and turned into a `("file_error", seq, gz_name,
err_str)` message — the trainer skips that file and moves on. A
worker exits when it pops the sentinel `None` from the input queue,
or once the trainer is gone (`serve_files`).
"""

from __future__ import annotations

import functools
import logging
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, List

from wesnoth_ai.paths import REPO_ROOT, TOOLS_DIR
from tools.mp_teardown import ParentGone, get_while_parent_lives, put_while_parent_lives

# How long a read or a write waits before checking that the trainer is
# still alive (seconds); the actors' period (tools/actor_worker.py).
_PARENT_POLL = 2.0


def encode_game(gz_path: Path, type_to_id: Dict[str, int], faction_to_id: Dict[str, int],
                relevant_set: bool, fog_hides_enemy_villages: bool = False,
                terrain_multi_hot: bool = False) -> List:
    """One replay's (RawEncoded, ActionIndices) pairs: the per-file work
    of a worker, and of tools/preencode_corpus.py.

    `relevant_set`: encode the relevant hex subset and build labels in
    the same basis (label builder and `encode_raw` each compute the
    subset; the encoder has no entry point that accepts a precomputed
    one). `fog_hides_enemy_villages` and `terrain_multi_hot` are the
    trainer encoder's own switches: a worker must encode exactly what
    the encoder would, else the pairs carry another observation."""
    from tools.replay_dataset import iter_replay_pairs
    from wesnoth_ai.encoder import encode_raw
    pairs = []
    for state, ai in iter_replay_pairs(gz_path, relevant_set=relevant_set):
        pairs.append((encode_raw(state, type_to_id=type_to_id, faction_to_id=faction_to_id,
                                 relevant_set=relevant_set,
                                 fog_hides_enemy_villages=fog_hides_enemy_villages,
                                 terrain_multi_hot=terrain_multi_hot), ai))
    return pairs


def worker_main(
    in_q,
    out_q,
    type_to_id: Dict[str, int],
    faction_to_id: Dict[str, int],
    log_level: int = logging.WARNING,
    relevant_set: bool = False,
    fog_hides_enemy_villages: bool = False,
    terrain_multi_hot: bool = False,
) -> None:
    """Worker entry point: `serve_files` over `encode_game` with the
    trainer's vocab and encoder switches.

    Each item written to `out_q` is one of:
      ("file",       seq, pairs, gz_name)   # pairs = list of (RawEncoded, ActionIndices)
      ("file_error", seq, gz_name, err_str)
      ("worker_exit",)
    where `seq` echoes the input item's dispatch index.

    The trainer's main loop unpacks "file" messages into individual
    ("pair", raw, ai, gz_name) events for its own consumption (see
    `_ParallelStream` in supervised_train.py). The "file_done" marker
    that drives gc / batch-flush bookkeeping is synthesized by the
    stream after a file's pairs are drained — workers don't emit it
    explicitly anymore.
    """
    # The repo root and tools/ on the worker's import path, however it
    # was started (fork inherits the trainer's; a Windows spawn starts
    # a fresh interpreter).
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    logging.basicConfig(level=log_level, format="%(message)s")
    serve_files(in_q, out_q, functools.partial(
        encode_game, type_to_id=type_to_id, faction_to_id=faction_to_id,
        relevant_set=relevant_set, fog_hides_enemy_villages=fog_hides_enemy_villages,
        terrain_multi_hot=terrain_multi_hot))


def serve_files(in_q, out_q, encode_file: Callable[[str], List]) -> None:
    """The worker's loop: take `(seq, gz_path)` items until the sentinel
    None, and ship `encode_file(gz_path)` for each, or its error.

    Returns once the trainer is gone. A spawned worker holds both ends of
    both queues, so neither pipe ever breaks: a killed trainer (an OOM
    kill, or the SL relaunch's pkill, which matches the trainer's command
    line and not the workers') would leave it waiting forever for a
    replay, or for room on its full output queue. So every read and
    write waits in _PARENT_POLL slices and checks the trainer between
    them (tools/mp_teardown.get_while_parent_lives)."""
    log = logging.getLogger("encode_worker")
    try:
        while True:
            item = get_while_parent_lives(in_q, _PARENT_POLL)
            if item is None:
                put_while_parent_lives(out_q, ("worker_exit",), _PARENT_POLL)
                return
            seq, gz_path = item
            gz_name = Path(gz_path).name
            try:
                # One put() per replay amortizes pickle cost across all
                # its pairs. An empty list is valid (no actionable
                # pairs): the trainer sees file_done with n=0.
                msg = ("file", seq, encode_file(gz_path), gz_name)
            except Exception as e:
                tb = traceback.format_exception_only(type(e), e)[-1].strip()
                msg = ("file_error", seq, gz_name, tb)
                log.debug(f"  worker skip {gz_name}: {e}")
            put_while_parent_lives(out_q, msg, _PARENT_POLL)
    except ParentGone:
        log.warning("encode worker: the trainer process is gone; exiting")
