"""Supervised pre-training on human replays — behavior cloning loss.

Trains the same encoder+model we use for self-play against human
actions observed in the replay corpus. Loss is cross-entropy on:
  - actor head: which slot the observed action picks.
  - target head: which hex (for move/attack/recruit) the action targets.
  - weapon head: which weapon slot (attack only).

Value head is NOT trained here (we don't have clean win/loss labels
on every state). Will be initialized from the self-play phase.

Output: a checkpoint at training/checkpoints/supervised.pt that the
self-play path can `--resume` from.

Usage:
    python tools/supervised_train.py DATASET_DIR [--epochs N] [--lr 1e-4] [--bs 8]
        [--init-from CKPT | --resume CKPT] [--relevant-set-hexes]
        [--max-pairs N] [--seed N]

Simplicity first: no DataLoader, no workers. Iterate replay files
sequentially, yield pairs, batch by count. If training gets slow we
can add multi-worker prefetch. Current rate estimate: with ~2000 pairs
per replay × ~10 replays/sec encoding-only → ~20K pairs/sec, so a
10M-pair corpus is one overnight run at batch=8 on CPU.
"""

from __future__ import annotations

import argparse
import copy
import gc
import gzip
import json
import logging
import multiprocessing as mp
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Project imports — assume cwd is the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.unit_vocab import seed_vocab
from wesnoth_ai.encoder import GameStateEncoder, RawEncoded
from wesnoth_ai.constants import OBSERVATION_EPOCH
from wesnoth_ai.model import WesnothModel
from wesnoth_ai.imitation_loss import build_imitation_targets, imitation_loss_parts
# Import replay_dataset from the same tools/ dir.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools.replay_dataset import ActionIndices, filter_competitive_2p, iter_replay_pairs
from tools.encode_worker import worker_main as _encode_worker_main
from tools.mp_teardown import start_child


log = logging.getLogger("supervised_train")


def _apply_size_filters(
    files: List[Path],
    dataset_dir: Path,
    max_commands: int,
    max_starting: int,
) -> List[Path]:
    """Drop replay files whose size metrics exceed the supplied caps.

    First-pass uses the cheap `n_commands` from index.jsonl to skip
    replays without opening them. For the rest we open the gz once
    and check `starting_units` — a smaller per-replay cost than
    paying for the whole training iteration only to OOM later.
    """
    # Build index lookup: file → n_commands.
    by_file: dict = {}
    idx = dataset_dir / "index.jsonl"
    if idx.exists():
        for line in idx.open(encoding="utf-8"):
            m = json.loads(line)
            by_file[m["file"]] = m.get("n_commands", 0)

    out: List[Path] = []
    n_dropped_cmds = 0
    n_dropped_units = 0
    for p in files:
        n_cmds = by_file.get(p.name, 0)
        if max_commands and n_cmds and n_cmds > max_commands:
            n_dropped_cmds += 1
            continue
        if max_starting:
            # Need to open for starting_units. Cheap (~5KB compressed).
            try:
                with gzip.open(p, "rt", encoding="utf-8") as f:
                    data = json.load(f)
                if len(data.get("starting_units", [])) > max_starting:
                    n_dropped_units += 1
                    continue
            except Exception:
                continue
        out.append(p)
    msg = f"  size filter dropped {n_dropped_cmds} (>{max_commands} cmds)"
    if max_starting:
        msg += f", {n_dropped_units} (>{max_starting} starting units)"
    log.info(msg)
    return out


def _save_checkpoint(
    path: Path,
    model: "WesnothModel",
    encoder: "GameStateEncoder",
    opt: torch.optim.Optimizer,
    step: int,
    pairs: int,
    epoch: int = 0,
    arch: Optional[Dict[str, int]] = None,
    carry: Optional[Dict] = None,
    relevant_set_hexes: bool = False,
    training_meta: Optional[Dict] = None,
    terrain_multi_hot: bool = False,
    resume_state: Optional[Dict] = None,
) -> None:
    """Atomic-ish checkpoint write: save to .tmp then rename.

    `epoch` is the count of FULLY-COMPLETED epochs across the whole
    chain (i.e. global, not per-run). Resume reads it as
    `resumed_epoch` and starts the loop at `range(resumed_epoch,
    epochs)`. Older checkpoints didn't carry this key; resume falls
    back to counting per-epoch snapshot files when it's absent.

    `relevant_set_hexes` is a top-level key (kept OUT of `arch`, which
    the policy loader compares strictly): `eval_sim.peek_checkpoint_arch`
    reads it with the other CHECKPOINT_STRUCT_FLAGS so every eval
    entry point builds the encoder in the hex basis this checkpoint
    was trained in. `training_meta` is provenance (init_from, seed,
    max_pairs); the resume reads its seed. `resume_state` is where the
    pass stands (`_resume_state`), so a resume continues it exactly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = dict(carry or {})
    # `carry` holds fields the MCTS/self-play path owns and the SL
    # pass must transport UNCHANGED (decision_step for the
    # combat-oracle anneal, aux_score / moves_left head flags) --
    # the SL<->MCTS round-trip contract (2026-07-16). Explicit keys
    # below always win.
    payload.update({
        # The sim's observation semantics at training time.
        # Without this, every checkpoint the imitation trainer
        # writes reads as epoch 1 on load, so the cross-build
        # warning fires on the newest work and stops meaning
        # anything. NOT in `carry`, which is the SL<->MCTS
        # transport: a later session adding it to carry's copy
        # loop would silently stamp the PARENT checkpoint's epoch.
        "observation_epoch": int(OBSERVATION_EPOCH),
        "arch": dict(arch) if arch else {
            "d_model": 128, "num_layers": 3,
            "num_heads": 4, "d_ff": 256},
        "relevant_set_hexes": bool(relevant_set_hexes),
        # The hex terrain view (encoder.terrain_tokens), read back by
        # the policy loader and the eval entry points like the basis.
        "terrain_multi_hot": bool(terrain_multi_hot),
        "training_meta":   dict(training_meta or {}),
        "model_state":     model.state_dict(),
        "encoder_state":   encoder.state_dict(),
        "unit_type_to_id": dict(encoder.unit_type_to_id),
        "faction_to_id":   dict(encoder.faction_to_id),
        "optimizer_state": opt.state_dict(),
        "supervised_step":  step,
        "supervised_pairs": pairs,
        "supervised_epoch": epoch,
    })
    if resume_state is not None:
        payload["supervised_resume"] = resume_state
    torch.save(payload, tmp)
    import os
    os.replace(tmp, path)


@dataclass
class PassPosition:
    """Where a resumed run re-enters the epoch it was cut in, so the rest
    of the epoch is the stream the uncut run would have trained. The
    epoch's file order comes from the global `random` at the epoch's
    start; the pairs already trained are read again and skipped, their
    value-selection draws replayed, and training goes on from the next
    pair."""
    epoch: int
    skip_pairs: int                        # pairs of this epoch trained before the cut
    step_in_epoch: int
    epoch_rng_state: Optional[tuple]       # None: the seed's state (a first run's first epoch)
    rng_state_at_cut: Optional[tuple]      # checked once the skip ends; None: unknown


def _resume_state(epoch: int, epoch_rng_state: tuple, epoch_start_pairs: int,
                  epoch_start_step: int, last_eval_pairs: int, seed: Optional[int]) -> Dict:
    """What a checkpoint records for a later run to continue its pass:
    the epoch in progress (or just finished), the global `random` state
    at its start and now, the counters at its start, the last holdout
    evaluation (so the resumed run evaluates where the uncut one would)
    and the torch generators (dropout)."""
    state = {"epoch": int(epoch), "epoch_rng_state": epoch_rng_state,
             "epoch_start_pairs": int(epoch_start_pairs),
             "epoch_start_step": int(epoch_start_step),
             "last_eval_pairs": int(last_eval_pairs),
             "rng_state": random.getstate(), "torch_rng_state": torch.get_rng_state(),
             "seed": seed}
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    return state


def _pass_position(ckpt: Dict, resumed_epoch: int, resumed_pairs: int, resumed_step: int,
                   seed: Optional[int]) -> Tuple[Optional[PassPosition], Optional[tuple]]:
    """(position, rng state) of a resume: the position when the
    checkpoint was cut inside the epoch it resumes, the global `random`
    state the next epoch shuffles with when it was saved at an epoch's
    end. Restores the torch generators the checkpoint saved. A checkpoint
    without a saved position continues exactly only in the first epoch of
    a first run under the same --seed, where the order is the seed's;
    otherwise the epoch restarts in a new order, with a warning."""
    res = ckpt.get("supervised_resume")
    if res is not None:
        torch.set_rng_state(res["torch_rng_state"])
        if "cuda_rng_state" in res and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(res["cuda_rng_state"])
            except RuntimeError as e:
                log.warning(f"  CUDA generators not restored ({e}); dropout draws differ")
        if int(res["epoch"]) == resumed_epoch:
            return PassPosition(resumed_epoch, resumed_pairs - int(res["epoch_start_pairs"]),
                                resumed_step - int(res["epoch_start_step"]),
                                res["epoch_rng_state"], res["rng_state"]), None
        return None, res["rng_state"]
    meta = ckpt.get("training_meta") or {}
    if (resumed_epoch == 0 and resumed_pairs > 0 and seed is not None
            and meta.get("seed") == seed and meta.get("resume") is None):
        log.info("  the checkpoint predates saved pass positions: its first epoch "
                 "continues in the seed's order; dropout draws start afresh")
        return PassPosition(0, resumed_pairs, resumed_step, None, None), None
    if resumed_pairs > 0:
        log.warning("  this resume cannot continue its pass (no saved position, and "
                    "not the first epoch of a first run under the same --seed): the "
                    "epoch restarts in a new order")
    return None, None


def _log_pass_reentry(position: PassPosition, t_epoch: float) -> None:
    """The skip of a resumed pass is over: say so, and check that the
    replayed draws land on the state the checkpoint saw."""
    log.info(f"  resumed inside epoch {position.epoch}: {position.skip_pairs} pairs "
             f"read again and skipped in {time.time() - t_epoch:.0f} s")
    if position.rng_state_at_cut is None:
        return
    if random.getstate() == position.rng_state_at_cut:
        log.info("  the replayed draws land on the checkpoint's state: the pass continues exactly")
    else:
        log.error("  the replayed draws do NOT land on the checkpoint's state: the "
                  "continued pass differs from the one that was cut")


# ---------------------------------------------------------------------
# Pair streams. Both expose the same event-stream shape:
#   ("pair",       state_or_raw, ai, gz_name)
#   ("file_done",  gz_name, n_pairs)
#   ("file_error", gz_name, err_str)
# Serial does encode_raw inline (technically just yields GameState and
# the loss function calls encoder.encode); parallel does encode_raw in
# worker processes and yields RawEncoded.
# ---------------------------------------------------------------------

def _pair_stream_serial(
    files: List[Path],
    *,
    max_pairs_per_replay: int = 0,
    sample_seed: Optional[int] = None,
    relevant_set: bool = False,
):
    """Single-process pair stream — reads + encodes inline.

    With `sample_seed` set (and a per-replay cap), each file
    contributes a seeded RANDOM sample of its pairs (reservoir)
    instead of its first N — the probe's independent-redraw
    mechanism (user ruling 2026-08-25: a low reading is retried
    with different RNG before it may abort anything).

    `relevant_set` selects the label basis; the encoder that later
    encodes these states must carry the same flag."""
    rng = random.Random(sample_seed) if sample_seed is not None else None
    for gz in files:
        n = 0
        try:
            if rng is not None and max_pairs_per_replay:
                # iter_replay_pairs yields ONE GameState object,
                # mutated in place as the replay advances -- buffered
                # entries MUST be deepcopied or the whole reservoir
                # collapses onto the final state (caught 2026-08-25:
                # every sampled game read as one-sided, n_auc_games
                # 0/150).
                buf: List[Tuple] = []
                seen = 0
                for state, ai in iter_replay_pairs(
                        gz, relevant_set=relevant_set):
                    seen += 1
                    if len(buf) < max_pairs_per_replay:
                        buf.append((copy.deepcopy(state), ai))
                    else:
                        j = rng.randrange(seen)
                        if j < max_pairs_per_replay:
                            buf[j] = (copy.deepcopy(state), ai)
                for state, ai in buf:
                    n += 1
                    yield ("pair", state, ai, gz.name)
            else:
                for state, ai in iter_replay_pairs(
                        gz, relevant_set=relevant_set):
                    if max_pairs_per_replay and n >= max_pairs_per_replay:
                        break
                    n += 1
                    yield ("pair", state, ai, gz.name)
            yield ("file_done", gz.name, n)
        except Exception as e:
            yield ("file_error", gz.name, repr(e))


def _pair_stream_preencoded(files: List[Path], preencoded_dir: Path):
    """The parallel stream's events from pre-encoded records
    (tools/preencode_corpus.py): one file at a time, in the given
    order, ("pair", RawEncoded, ActionIndices, name) then
    ("file_done", name, n). No workers: a record loads in
    milliseconds against the seconds a replay takes."""
    from tools.preencode_corpus import read_record, record_path
    for gz in files:
        try:
            pairs = read_record(record_path(preencoded_dir, gz.name))
        except Exception as e:  # noqa: BLE001 - the file is skipped, loudly
            yield ("file_error", gz.name, repr(e))
            continue
        for raw, ai in pairs:
            yield ("pair", raw, ai, gz.name)
        yield ("file_done", gz.name, len(pairs))


def check_preencoded(preencoded_dir: Path, files: List[Path], encoder,
                     relevant_set: bool) -> None:
    """Refuse a pre-encoded corpus that is not this run's encoding:
    other vocab, hex basis or observation epoch, or records missing for
    files of the pass (the pre-encoder is resumable; finish it first)."""
    from tools.preencode_corpus import check_manifest_epoch, load_manifest, record_path, vocab_fingerprint
    # The epoch is checked first so a stale-world corpus does not read
    # as a vocab mismatch: what is wrong is the OBSERVATIONS, not the
    # encoding of them.
    check_manifest_epoch(preencoded_dir)
    man = load_manifest(preencoded_dir)
    # The vocab is append-only and grows during a run (the holdout
    # eval registers names the corpus seeding lacked), so the records'
    # vocab is a prefix of this run's: compare on that prefix. Names
    # past it never occur in the records (the pre-encoder mapped them
    # to the overflow bucket), so they cannot disagree.
    n_types = int(man.get("unit_types", len(encoder.unit_type_to_id)))
    n_factions = int(man.get("factions", len(encoder.faction_to_id)))
    types = {k: v for k, v in encoder.unit_type_to_id.items() if v < n_types}
    factions = {k: v for k, v in encoder.faction_to_id.items() if v < n_factions}
    fp = vocab_fingerprint(types, factions, relevant_set,
                           bool(getattr(encoder, "fog_hides_enemy_villages", False)),
                           bool(getattr(encoder, "terrain_multi_hot", False)))
    if man.get("fingerprint") != fp:
        raise RuntimeError(f"--preencoded {preencoded_dir} was encoded with another vocab or "
                           f"hex basis ({man.get('fingerprint')}; this run {fp}); "
                           f"re-run tools/preencode_corpus.py with this run's checkpoint")
    grown = (len(encoder.unit_type_to_id) - len(types), len(encoder.faction_to_id) - len(factions))
    if any(grown):
        log.info(f"  pre-encoded vocab is a prefix of this run's: {grown[0]} unit types and "
                 f"{grown[1]} factions registered since the records were made")
    missing = [gz.name for gz in files if not record_path(preencoded_dir, gz.name).exists()]
    if missing:
        raise RuntimeError(f"--preencoded {preencoded_dir}: {len(missing)} of {len(files)} "
                           f"files of this pass have no record (first: {missing[:3]}); "
                           f"finish the pre-encoding pass")


class _ParallelStream:
    """Wraps the worker pool + producer-consumer queues as an iterable.

    Spawns N worker processes that each pop a `(seq, Path)` from the
    input queue, run `iter_replay_pairs` + `encode_raw` for the entire
    replay, and push a single ("file", seq, pairs, gz_name) message —
    the pairs list is the whole replay. The main thread keeps the
    input queue topped up: a fresh file goes in for every EMITTED
    file, and once all files are dispatched, N sentinels (None)
    retire the workers.

    Dispatch order is delivery order. Workers finish in an order set
    by replay length and scheduling, so messages are held in a
    reorder buffer until the next sequence number arrives; the pair
    stream the trainer sees is therefore the seeded file order at any
    worker count (`--seed` reproduces it). The buffer is bounded
    without a separate cap: a refill is sent per emitted file, so at
    most `workers * 2` files (the priming depth) are dispatched and
    not yet emitted, in the pipeline and the buffer together. A
    worker that dies holding a file (OOM kill) would stall the head
    forever, so corpse reconciliation switches the stream to arrival
    order for the rest of the pass, loudly; that pass is already not
    reproducible (its files are lost).

    Why batched messages: the per-pair version pickled each RawEncoded
    individually across the queue and the overhead exceeded the
    encoder savings (113 → 86 pairs/sec on the cluster). One message
    per replay means ~50–200 pairs share a single pickle / put / get
    round, dropping queue overhead by ~2 orders of magnitude.

    Trainer-facing API: still per-pair. The stream unpacks each "file"
    message into N successive ("pair", ...) events plus a synthesized
    ("file_done", gz_name, n) marker — so the trainer's loop is
    identical to the serial path.

    Per-replay pair cap: workers don't know about it (keeps them
    simple). Caller can apply it on the consumer side; we already do
    that in the trainer when --workers 0, but in --workers >0 mode
    we currently don't enforce it (default off, no behavior change).

    Cleanup contract: closing the iterator (or letting it run to
    completion) drains and joins the worker pool. Always either
    iterate to exhaustion or call `.close()` on it.
    """

    def __init__(
        self,
        files: List[Path],
        *,
        workers: int,
        type_to_id,
        faction_to_id,
        prefetch_factor: int,
        relevant_set: bool = False,
        fog_hides_enemy_villages: bool = False,
        terrain_multi_hot: bool = False,
    ):
        self._files = list(files)
        self._workers_n = workers
        self._closed = False
        # Worker-message wait bound (s): how long a get() may block
        # before checking for dead workers. Overridable for tests.
        self._get_timeout = float(
            os.environ.get("WAI_STREAM_GET_TIMEOUT", "60"))

        # `spawn` works on both Linux and Windows. `fork` would be
        # slightly faster on Linux (cow-shared dicts) but inherits
        # whatever weird state lived in the parent — torch's lazy
        # CUDA init included. spawn is the portable safe default.
        self._ctx = mp.get_context("spawn")
        # Bounded queues so a fast worker doesn't OOM the main
        # process while it's busy doing forward/backward.
        self._in_q  = self._ctx.Queue(maxsize=workers * 2)
        # Output queue holds prefetched per-file batches now (one
        # message per replay), so the size is small in slot-count.
        # workers * prefetch_factor is a generous upper bound on
        # in-flight files.
        self._out_q = self._ctx.Queue(maxsize=max(workers * prefetch_factor, 8))

        # Through start_child, so a worker whose trainer was killed
        # exits without writing out results nobody will read
        # (tools/mp_teardown.run_child).
        self._procs = [
            start_child(self._ctx, _encode_worker_main,
                        (self._in_q, self._out_q, dict(type_to_id), dict(faction_to_id)),
                        name=f"encode-{i}",
                        kwargs={"relevant_set": relevant_set,
                                "fog_hides_enemy_villages": fog_hides_enemy_villages,
                                "terrain_multi_hot": terrain_multi_hot})
            for i in range(workers)]

        self._workers_alive = workers
        self._init_consumer_state()
        # Prime the input queue with up to `workers * 2` files (its
        # capacity, so this never blocks). The rest are fed one per
        # emitted file.
        for _ in range(min(workers * 2, len(self._files))):
            self._in_q.put((self._next_file, self._files[self._next_file]))
            self._next_file += 1
        self._send_sentinels()

    def _init_consumer_state(self) -> None:
        """Dispatch, reorder and per-file unpack state; the tests build
        a stream with stub workers and call this to get the real
        consumer logic on top."""
        self._next_file = 0                 # next file to dispatch; its index is its seq
        self._next_seq = 0                  # next seq to emit
        self._reorder: Dict[int, Tuple] = {}    # seq -> message arrived ahead of its turn
        self._refills_owed = 0              # emitted files not yet paid back with a dispatch
        self._sentinels_sent = 0
        self._ordered = True                # False once a corpse may have lost a seq
        # Per-file unpack state: when a "file" message is emitted we
        # iterate its pairs locally, returning one ("pair", ...) at a
        # time. After the iterator is exhausted, return a synthesized
        # ("file_done", ...) before pulling the next message.
        self._pair_iter = None              # iter over current file's pairs
        self._current_gz: Optional[str] = None
        self._pending_file_done: Optional[Tuple[str, int]] = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration

        # 1. Drain the current file's pair iterator if we're mid-replay.
        if self._pair_iter is not None:
            try:
                raw, ai = next(self._pair_iter)
                return ("pair", raw, ai, self._current_gz)
            except StopIteration:
                # File exhausted — emit file_done next call.
                self._pair_iter = None
                # fall through to step 2

        # 2. If a file just finished, emit the file_done marker now.
        if self._pending_file_done is not None:
            gz_name, n = self._pending_file_done
            self._pending_file_done = None
            return ("file_done", gz_name, n)

        # 3. Otherwise emit the next file in dispatch order, pulling
        # worker messages into the reorder buffer until it arrives.
        while True:
            msg = self._take_ready()
            if msg is not None:
                return self._emit(msg)
            if self._workers_alive <= 0:
                if self._reorder:
                    # Nothing more can arrive: what is buffered is
                    # delivered, gaps skipped.
                    self._ordered = False
                    continue
                self.close()
                raise StopIteration
            # BOUNDED get + corpse reconciliation (2026-08-10, BACKLOG
            # item 1): a worker that dies WITHOUT emitting its
            # ("worker_exit",) message (OOM-kill, segfault) leaves
            # _workers_alive overcounted, and a bare blocking get()
            # then hangs forever once the surviving workers retire --
            # the silent-stall signature of the 2026-08-08 imitation
            # run (94% done, no traceback, ~0 CPU). On timeout,
            # compare actual corpses against counted exits and
            # reconcile loudly instead of waiting forever.
            try:
                item = self._out_q.get(timeout=self._get_timeout)
            except Exception:                       # queue.Empty
                self._reconcile_corpses()
                self._send_sentinels()
                continue
            tag = item[0]
            if tag in ("file", "file_error"):
                self._reorder[int(item[1])] = item
            elif tag == "worker_exit":
                self._workers_alive -= 1
            else:
                # Unknown tag — should never happen. Log and keep pulling.
                log.warning(f"  unknown stream event {tag!r}; ignoring")

    def _take_ready(self) -> Optional[Tuple]:
        """The buffered message whose turn it is: the next seq in order,
        or the lowest buffered seq once order has been given up."""
        if not self._reorder:
            return None
        if self._ordered:
            return self._reorder.pop(self._next_seq, None)
        return self._reorder.pop(min(self._reorder))

    def _emit(self, msg: Tuple):
        """Turn a worker message into the trainer-facing event(s) and
        pay its refill."""
        tag, seq = msg[0], int(msg[1])
        self._next_seq = max(self._next_seq, seq + 1)
        self._refills_owed += 1
        self._pump_refills()
        if tag == "file_error":
            _, _, gz_name, err = msg
            return ("file_error", gz_name, err)
        _, _, pairs, gz_name = msg
        if not pairs:
            # Empty replay (no actionable pairs). Don't bother
            # setting up an iterator; emit file_done directly.
            return ("file_done", gz_name, 0)
        # Hand the pair stream off to step 1 on the next call.
        self._pair_iter = iter(pairs)
        self._current_gz = gz_name
        self._pending_file_done = (gz_name, len(pairs))
        # Return the first pair from the new buffer immediately.
        raw, ai = next(self._pair_iter)
        return ("pair", raw, ai, gz_name)

    def _reconcile_corpses(self) -> None:
        """A get() timed out: count workers that died without their
        exit message, and stop waiting on anything they held."""
        n_dead = sum(1 for p in self._procs if not p.is_alive())
        n_exited = self._workers_n - self._workers_alive
        if n_dead <= n_exited:
            return
        missing = n_dead - n_exited
        log.error(
            f"{missing} encode worker(s) died without a worker_exit "
            f"message (OOM-killed?); reconciling so the stream "
            f"terminates instead of hanging. Files those workers held "
            f"are LOST from this pass (audible in the files_seen "
            f"accounting), and the pass continues in arrival order: "
            f"its pair stream is no longer the seeded one.")
        self._workers_alive -= missing
        self._ordered = False
        # Each corpse may have held a file whose refill never comes.
        self._refills_owed += missing
        self._pump_refills()

    def _pump_refills(self) -> None:
        """Dispatch one file per emitted file while files remain, then
        the workers' sentinels. Never blocks: a full input queue (only
        possible once no worker is taking files) defers the refill to
        the next emit or timeout."""
        while self._refills_owed > 0 and self._next_file < len(self._files):
            try:
                self._in_q.put_nowait((self._next_file, self._files[self._next_file]))
            except Exception:                       # queue.Full
                break
            self._next_file += 1
            self._refills_owed -= 1
        self._send_sentinels()

    def _send_sentinels(self) -> None:
        """Once every file is dispatched, queue one None per worker so
        each retires after the files ahead of it. Never blocks: a full
        input queue is retried on the next call (every refill and every
        timeout gets here), and a stream with fewer files than workers
        still retires all of them."""
        if self._next_file < len(self._files):
            return
        while self._sentinels_sent < self._workers_n:
            try:
                self._in_q.put_nowait(None)
            except Exception:                       # queue.Full
                return
            self._sentinels_sent += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Send sentinels to anyone still waiting on input.
        for _ in range(self._workers_n):
            try:
                self._in_q.put_nowait(None)
            except Exception:
                break
        # Drain residual messages so workers don't block on a full
        # output queue while they're shutting down.
        deadline = time.time() + 5.0
        while time.time() < deadline and any(p.is_alive() for p in self._procs):
            try:
                self._out_q.get(timeout=0.1)
            except Exception:
                pass
        for p in self._procs:
            if p.is_alive():
                p.terminate()
            p.join(timeout=2.0)


def _pair_stream_parallel(
    files: List[Path],
    *,
    workers: int,
    type_to_id,
    faction_to_id,
    prefetch_factor: int = 4,
    max_pairs_per_replay: int = 0,  # currently unused in parallel mode;
                                    # added for API symmetry.
    relevant_set: bool = False,
    fog_hides_enemy_villages: bool = False,
    terrain_multi_hot: bool = False,
):
    """Multi-process pair stream — encode_raw runs in worker processes."""
    return _ParallelStream(
        files,
        workers=workers,
        type_to_id=type_to_id,
        faction_to_id=faction_to_id,
        prefetch_factor=prefetch_factor,
        relevant_set=relevant_set,
        fog_hides_enemy_villages=fog_hides_enemy_villages,
        terrain_multi_hot=terrain_multi_hot,
    )


def _encode_one(
    encoder: GameStateEncoder,
    state_or_raw,                # GameState (serial) or RawEncoded (worker)
    device: torch.device,
):
    """Run phase-2 encoding on either form. Lives on the main thread —
    touches the encoder's nn.Embedding parameters and produces tensors
    on `device`."""
    if isinstance(state_or_raw, RawEncoded):
        return encoder.encode_from_raw(state_or_raw, device=device)
    return encoder.encode(state_or_raw)


def _raw_one(encoder: GameStateEncoder, state_or_raw) -> RawEncoded:
    """The RawEncoded of a pair: a worker's or a pre-encoded record's as
    is; a GameState's through `encode_raw` after registering its names,
    as `encoder.encode` does."""
    if isinstance(state_or_raw, RawEncoded):
        return state_or_raw
    encoder.register_names(state_or_raw)
    return encoder.raw_of(state_or_raw)


@dataclass
class LossParts:
    """Per-head decomposition of one pair's CE loss.

    `total = actor + type + target + weapon` — used for backward.
    The `*_fired` flags tell the trainer whether each head was
    relevant for THIS pair (e.g. end_turn actions don't fire the
    target / type / weapon head; move actions don't fire the
    weapon head). Per-head averages in the progress log are taken
    over fired pairs only, so they're interpretable as "when this
    head HAS to predict, how good is it?"
    """
    total:        torch.Tensor   # scalar, grad-tracking
    actor:        torch.Tensor   # scalar, grad-tracking (or zero sentinel)
    type:         torch.Tensor   # scalar, grad-tracking (or zero sentinel)
    target:       torch.Tensor   # scalar, grad-tracking (or zero sentinel)
    # value: C51 CE against the game outcome z (joint value training,
    # user 2026-07-16 -- the value-frozen epoch-0 pass let policy
    # gradients reshape the trunk under the head, late AUC 0.79->0.63).
    weapon:       torch.Tensor   # scalar, grad-tracking (or zero sentinel)
    actor_fired:  bool
    type_fired:   bool
    target_fired: bool
    weapon_fired: bool
    value:        torch.Tensor = None   # zero sentinel when off
    value_fired:  bool = False
    # Policy-loss multiplier (imitation mode): 0.0 silences the policy
    # heads for a pair kept only as a value state (loser-side states
    # under winners-only training); per-game equal weighting scales it
    # by median_actions/game_actions. Head tensors above stay RAW for
    # logging; `total` and the batched re-sum apply this weight.
    policy_w:     float = 1.0


# Per-action-type loss weight on the actor head + the new type head.
# Without per-type upweighting, moves dominate (~65% of corpus
# actions) and the model learns to never recruit. Weights default to
# inverse-frequency from a 5k-replay scan of replays_dataset/
# (computed by tools/compute_action_type_weights.py); override at
# CLI time via --action-type-weights path/to/weights.json.
#
# Used in two places:
#   - actor-head CE: multiply by the weight of the recorded action
#     type (so recruit/attack actions get more gradient).
#   - type-head CE: torch.cross_entropy `weight` argument (per-class
#     weighting for the ATTACK vs MOVE softmax). Only the ATTACK and
#     MOVE entries here matter for the type head; recruit / end_turn
#     don't go through the type head at all.
_DEFAULT_ACTION_TYPE_LOSS_WEIGHT = {
    # Generated by tools/compute_action_type_weights.py
    # (5k-replay slice of replays_dataset/, ignore_recall=True).
    "move":     0.189,
    "attack":   0.628,
    "recruit":  1.748,
    "recall":   0.0,    # PvP shouldn't have recalls
    "end_turn": 1.435,
}


def _load_action_type_weights(path: Optional[Path]) -> Dict[str, float]:
    """Load per-action-type weights from a JSON config (the format
    `tools/compute_action_type_weights.py` produces). When `path` is
    None, returns the baked-in defaults."""
    if path is None:
        return dict(_DEFAULT_ACTION_TYPE_LOSS_WEIGHT)
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    weights = payload.get("weights", payload)   # accept raw dict OR full payload
    if not isinstance(weights, dict):
        raise ValueError(
            f"{path}: action-type weights JSON must contain a "
            f"'weights' dict (or be a raw dict)")
    out = dict(_DEFAULT_ACTION_TYPE_LOSS_WEIGHT)
    for k, v in weights.items():
        out[k] = float(v)
    return out

# Label smoothing on cross-entropy. 0.05 = 5% of probability mass
# spread uniformly over the non-target classes. Standard regularization;
# discourages overconfident logits without meaningfully changing the
# argmax decision boundary.
_LABEL_SMOOTHING = 0.05


def _loss_parts_for_output(
    output,                      # ModelOutput
    ai:      ActionIndices,
    device:  torch.device,
    *,
    type_loss_weights: Dict[str, float] = None,
    value_z: Optional[int] = None,      # game outcome for the mover
    value_weight: float = 0.0,          # lambda_v x per-game weight
    policy_weight: float = 1.0,         # imitation: 0 = value-only pair
) -> LossParts:
    """Per-sample CE loss decomposed into actor/type/target/weapon
    heads.

    Returns a LossParts. The `total` field is what the trainer
    backprops; the per-head fields are the same per-CE breakdown for
    diagnostic reporting.

    Action-type-conditional weighting on the actor head + type head
    is applied so rare-but-important action types (recruits) get a
    stronger learning signal. `type_loss_weights` is the dict
    produced by `tools/compute_action_type_weights.py`; defaults to
    `_DEFAULT_ACTION_TYPE_LOSS_WEIGHT`. Per-head running averages
    report the UNWEIGHTED loss so the magnitudes stay comparable
    to plain CE (random ≈ ln(num_classes)).

    Type head: only fires for unit actors (`ai.type_idx is not
    None`). The cross_entropy uses per-class weights for ATTACK vs
    MOVE (drawn from the same dict, since they're at the leaf
    level). For non-unit actors (recruit / end_turn) the type head
    output is ignored.

    Skips heads if the observed slot index doesn't land inside the
    model's output shape (very rare — the encoder sort should match;
    the guard is there so one bad replay doesn't tank a run). When a
    head is skipped, the corresponding `*_fired` flag is False and
    the loss tensor is the zero sentinel (contributes nothing to
    gradients or to the per-head running average).

    NO legality mask: the observed action in a human replay is legal
    by construction, so we don't feed the policy a legality prior
    during supervised training. Applying our approximate mask would
    also risk -inf'ing the ground-truth slot when our mask is stricter
    than actual Wesnoth rules (e.g., multi-turn moves, ZoC
    interactions) and blow the loss up to ~1e9. The legality mask IS
    still applied at rollout time in action_sampler.sample_action, so
    illegal model predictions are filtered there.
    """
    if type_loss_weights is None:
        type_loss_weights = _DEFAULT_ACTION_TYPE_LOSS_WEIGHT

    zero = torch.zeros((), device=device)
    actor_logits = output.actor_logits        # [1, A]
    A = actor_logits.size(1)
    if ai.actor_idx >= A:
        # Pathological: observed actor isn't in the model's output. The
        # whole pair contributes nothing — total=0, no head fired.
        return LossParts(zero, zero, zero, zero, zero,
                         False, False, False, False,
                         value=zero, value_fired=False)

    actor_target = torch.tensor(ai.actor_idx, device=device, dtype=torch.long)
    actor_loss_raw = F.cross_entropy(
        actor_logits, actor_target.unsqueeze(0),
        label_smoothing=_LABEL_SMOOTHING,
    )
    actor_weight = type_loss_weights.get(ai.action_type, 1.0)
    actor_loss = actor_loss_raw * actor_weight

    # Type head: only for unit actors (action_type in
    # {"attack", "move"} maps to a UnitActionType slot).
    type_loss = zero
    type_loss_raw = zero
    type_fired = False
    if ai.type_idx is not None:
        type_row = output.type_logits[0, ai.actor_idx]  # [T]
        T = type_row.numel()
        if T > 0 and 0 <= ai.type_idx < T:
            # Per-class weights: pull ATTACK / MOVE from the
            # supplied dict. F.cross_entropy(weight=...) expects a
            # tensor of length T; index 0 is ATTACK, 1 is MOVE.
            class_weights = torch.tensor(
                [type_loss_weights.get("attack", 1.0),
                 type_loss_weights.get("move", 1.0)],
                device=device, dtype=type_row.dtype,
            )
            tt = torch.tensor(ai.type_idx, device=device, dtype=torch.long)
            type_loss_raw = F.cross_entropy(
                type_row.unsqueeze(0), tt.unsqueeze(0),
                label_smoothing=_LABEL_SMOOTHING,
                weight=class_weights,
            )
            type_loss = type_loss_raw
            type_fired = True

    target_loss = zero
    target_fired = False
    if ai.target_idx is not None and ai.action_type != "end_turn":
        tgt_row = output.target_logits[0, ai.actor_idx]  # [H]
        H = tgt_row.numel()
        if H > 0 and ai.target_idx < H:
            tt = torch.tensor(ai.target_idx, device=device, dtype=torch.long)
            target_loss = F.cross_entropy(
                tgt_row.unsqueeze(0), tt.unsqueeze(0),
                label_smoothing=_LABEL_SMOOTHING,
            )
            target_fired = True

    weapon_loss = zero
    weapon_fired = False
    if ai.weapon_idx is not None:
        w_row = output.weapon_logits[0, ai.actor_idx]  # [max_attacks]
        W = w_row.numel()
        if W > 0 and ai.weapon_idx < W:
            wt = torch.tensor(ai.weapon_idx, device=device, dtype=torch.long)
            weapon_loss = F.cross_entropy(
                w_row.unsqueeze(0), wt.unsqueeze(0),
                label_smoothing=_LABEL_SMOOTHING,
            )
            weapon_fired = True

    # Joint value loss (user 2026-07-16): C51 CE against the game
    # outcome z for the side to move. Corpus games are decisive by
    # construction (z in {-1,+1}), so the projected target is a
    # ONE-HOT on the support's edge atom -- CE reduces to
    # -log p(edge). `value_weight` carries lambda_v x the per-game
    # decorrelation weight (mean_cmds / n_cmds(game)): each GAME
    # contributes ~equally regardless of length, the same
    # game_weight philosophy as trainer.step_mcts. Value CE per
    # sample is reported UNWEIGHTED in `value` for the log.
    value_loss = zero
    value_loss_raw = zero
    value_fired = False
    if value_z is not None and value_weight > 0.0             and getattr(output, "value_logits", None) is not None:
        logp = F.log_softmax(output.value_logits[0], dim=-1)   # [K]
        edge = (logp.shape[0] - 1) if value_z > 0 else 0
        value_loss_raw = -logp[edge]
        value_loss = value_weight * value_loss_raw
        value_fired = True

    # `policy_weight` scales the POLICY heads only (imitation mode:
    # winners-only sets 0.0 on loser-side pairs kept as value states;
    # per-game weighting scales by median/game action count). Value
    # supervision is weighted separately via `value_weight`.
    total = (policy_weight * (actor_loss + type_loss + target_loss
                              + weapon_loss)
             + value_loss)
    # Per-head fields report the raw CE (no action-type weight, no
    # smoothing scale baked in) so the running-average log line stays
    # interpretable -- "actor=2.5" means CE=2.5 even if the actor head
    # is internally scaled 5x for recruits.
    return LossParts(total, actor_loss_raw, type_loss_raw,
                     target_loss, weapon_loss,
                     True, type_fired, target_fired, weapon_fired,
                     value=value_loss_raw, value_fired=value_fired,
                     policy_w=policy_weight)


def _loss_for_output(
    output,
    ai:     ActionIndices,
    device: torch.device,
    *,
    type_loss_weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Backwards-compat shim: returns just the total scalar loss.

    Existing callers (parity tests, single-step debugging) didn't need
    the per-head breakdown. The trainer's hot loop uses
    `_loss_parts_for_output` directly.
    """
    return _loss_parts_for_output(
        output, ai, device, type_loss_weights=type_loss_weights).total


def _loss_for_pair(
    encoder: GameStateEncoder,
    model:   WesnothModel,
    state_or_raw,                # GameState (serial) or RawEncoded (worker)
    ai:      ActionIndices,
    device:  torch.device,
    *,
    type_loss_weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Per-pair forward + total loss. Kept for backwards compatibility
    (parity tests). The trainer's hot loop computes per-head losses
    via `_loss_parts_for_output` so it can report each head's progress
    separately."""
    encoded = _encode_one(encoder, state_or_raw, device)
    output = model(encoded)
    return _loss_for_output(output, ai, device,
                             type_loss_weights=type_loss_weights)


def _loss_parts_for_pair(
    encoder: GameStateEncoder,
    model:   WesnothModel,
    state_or_raw,
    ai:      ActionIndices,
    device:  torch.device,
    *,
    type_loss_weights: Optional[Dict[str, float]] = None,
    value_z: Optional[int] = None,
    value_weight: float = 0.0,
    policy_weight: float = 1.0,
) -> LossParts:
    """Per-pair forward + per-head loss breakdown."""
    encoded = _encode_one(encoder, state_or_raw, device)
    output = model(encoded)
    return _loss_parts_for_output(
        output, ai, device, type_loss_weights=type_loss_weights,
        value_z=value_z, value_weight=value_weight,
        policy_weight=policy_weight)


def _batch_loss(model, encoder, raws, ais, zw, device, type_loss_weights,
                autocast_dtype=None):
    """The batch's loss parts and targets: one pinned host buffer of
    token features (`encode_from_raw_embedded`), one trunk pass
    (`forward_embedded`), every head scored once
    (`wesnoth_ai.imitation_loss`). `autocast_dtype` (torch.bfloat16
    with --bf16) runs the embeddings, the trunk and the heads in that
    dtype under autocast; the cross-entropies run in fp32 (autocast
    promotes log_softmax), the weights and gradients stay fp32."""
    if autocast_dtype is None:
        return _batch_loss_impl(model, encoder, raws, ais, zw, device, type_loss_weights)
    with torch.autocast(device.type, dtype=autocast_dtype):
        return _batch_loss_impl(model, encoder, raws, ais, zw, device, type_loss_weights)


def _batch_loss_impl(model, encoder, raws, ais, zw, device, type_loss_weights):
    streams = encoder.encode_from_raw_embedded(raws, device=device)
    material = None
    if getattr(model, "has_value_material", False):
        material = torch.tensor([[float(r.material)] for r in raws],
                                dtype=torch.float32, device=device)
    padded = model.forward_embedded(streams, material=material).float32()
    targets = build_imitation_targets(
        ais, zw, streams.sizes,
        n_types=padded.type_logits.shape[2],
        n_weapons=padded.weapon_logits.shape[2],
        n_atoms=padded.value_logits.shape[1],
        type_loss_weights=type_loss_weights, device=device)
    return imitation_loss_parts(padded, targets), targets


def _accumulate_batch(model, encoder, raws, ais, zw, batch_size, device,
                      type_loss_weights, sink: List, opt, autocast_dtype=None) -> int:
    """Backward of the batch's loss / batch_size into the parameters'
    gradients, in as many equal chunks as it takes to fit the device.

    On a CUDA out-of-memory error the whole flush is REDONE from zeroed
    gradients with twice as many chunks. Retrying without zeroing would
    double-count: the peak is inside `backward()`, so an out-of-memory
    error there leaves the gradients of the layers it already walked
    accumulated, and the retry adds those same contributions again.
    A single pair that does not fit raises.

    `sink` collects (targets, per-sample log tensor) per chunk in
    order; it is cleared on every attempt. Returns the number of
    halvings (0 when the batch fit whole)."""
    n_chunks = 1
    halvings = 0
    parts = targets = None
    while True:
        opt.zero_grad(set_to_none=True)
        sink.clear()
        step = -(-len(raws) // n_chunks)          # ceil
        try:
            for start in range(0, len(raws), step):
                parts, targets = _batch_loss(
                    model, encoder, raws[start:start + step], ais[start:start + step],
                    zw[start:start + step], device, type_loss_weights, autocast_dtype)
                (parts.total / batch_size).backward()
                sink.append((targets, parts.log_tensor()))
                parts = targets = None            # the chunk's graph goes now
            return halvings
        except torch.cuda.OutOfMemoryError:
            if step <= 1:
                raise
            halvings += 1
            n_chunks *= 2
        # Past the except clause: the traceback no longer pins the
        # failed chunk's frame, and dropping its loss frees the graph
        # with the saved activations the failed backward never walked
        # (nearly all of them: the peak is at the top of backward).
        # Only then can the allocator hand the memory to the retry.
        parts = targets = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _flush_batch(
    model:           WesnothModel,
    encoder:         GameStateEncoder,
    batch_raws:      List[RawEncoded],
    batch_ais:       List[ActionIndices],
    batch_zw:        List,                    # per-sample (z, value weight, policy weight)
    opt:             torch.optim.Optimizer,
    params_for_clip: List[torch.nn.Parameter],
    batch_size:      int,
    device:          torch.device,
    running_loss:    deque,
    running_loss_actor:  deque,
    running_loss_type:   deque,
    running_loss_target: deque,
    running_loss_weapon: deque,
    running_loss_value:  deque,
    type_loss_weights: Optional[Dict[str, float]] = None,
    autocast_dtype=None,
) -> int:
    """One batched forward + summed-loss backward + opt step over B
    RawEncoded pairs; returns the number of times the batch had to be
    halved to fit the device (0 when it fit whole).

    The batch's token embeddings come from one pinned host buffer
    (`encode_from_raw_embedded`, the inference server's path), the
    trunk and the heads run once on the padded batch
    (`forward_embedded`), and every head's cross-entropy runs once over
    the PaddedOutput (`wesnoth_ai.imitation_loss`); the only
    host-device synchronization is the log transfer after the step.
    A batch that does not fit the device is split, never dropped
    (`_accumulate_batch`): before 2026-09-11 an out-of-memory batch was
    dropped with a DEBUG line, and seed2's epochs lost 12% of their
    pairs that way (docs/box_specs.md "Pair census").

    `batch_size` is the *target* batch size used as the loss-scaling
    denominator, so a partial flush at a file boundary scales the
    gradient as a full batch does: the same effective learning rate
    per pair regardless of where file boundaries fell.

    Per-sample losses (per head) go to the `running_loss*` deques for
    the progress log; per-head averages are taken over fired pairs
    only (see LossParts).
    """
    if not batch_raws:
        return 0
    if type_loss_weights is None:
        type_loss_weights = _DEFAULT_ACTION_TYPE_LOSS_WEIGHT
    zw = batch_zw if batch_zw else [(None, 0.0, 1.0)] * len(batch_ais)
    sink: List = []
    splits = _accumulate_batch(model, encoder, batch_raws, batch_ais, zw, batch_size, device,
                               type_loss_weights, sink, opt, autocast_dtype)

    torch.nn.utils.clip_grad_norm_(params_for_clip, 1.0)
    opt.step()
    opt.zero_grad()

    # One transfer for the whole flush, AFTER the optimizer step: the
    # backward, the clip and the step are all enqueued before the host
    # blocks on the device. Reading the per-sample losses straight
    # after backward() was a hard sync that serialized host and GPU.
    drained = []
    if sink:
        widths = [t.shape[1] for _, t in sink]
        flat = torch.cat([t for _, t in sink], dim=1).cpu().tolist()
        off = 0
        for (targets, _), w in zip(sink, widths):
            drained.append((targets, [row[off:off + w] for row in flat]))
            off += w

    for targets, (actor_v, type_v, target_v, weapon_v, value_v) in drained:
        ok = targets.ok
        for i in range(targets.n):
            if not ok["actor"][i]:
                continue   # actor_idx out of range: the pair contributed 0 to the gradient
            if ok["value"][i] and running_loss_value is not None:
                running_loss_value.append(value_v[i])
            if targets.policy_w[i] == 0.0:
                continue   # value-only pair: keep the policy averages clean
            running_loss.append(actor_v[i] + type_v[i] + target_v[i] + weapon_v[i])
            running_loss_actor.append(actor_v[i])
            if ok["type"][i]:
                running_loss_type.append(type_v[i])
            if ok["target"][i]:
                running_loss_target.append(target_v[i])
            if ok["weapon"][i]:
                running_loss_weapon.append(weapon_v[i])
    return splits


def _masked_target_nll(
    target_row: torch.Tensor,     # [H] target logits for the actor
    legal: torch.Tensor,          # [H] bool, mask-legal hexes
    target_idx: int,
) -> Optional[Tuple[float, bool]]:
    """-log p(target | legal hexes) and the masked top-1 hit, or None
    when the human's target is not mask-legal (counted by the caller).

    This is the distribution that plays (action_sampler restricts the
    softmax to the legal set), so it is the same quantity in every hex
    basis: the unmasked target CE's support is all H hex logits and
    shrinks with the basis (docs/model_cost_study_20260905.md 2.5).
    No label smoothing, unlike the training loss."""
    if not bool(legal[target_idx]):
        return None
    masked = target_row.masked_fill(~legal, float("-inf"))
    logp = F.log_softmax(masked, dim=-1)
    return float(-logp[target_idx]), int(masked.argmax()) == target_idx


def _legal_target_row(masks, ai: ActionIndices) -> Optional[torch.Tensor]:
    """The legality-mask row the action sampler would apply for this
    labelled action: type-conditional for unit actors, the recruit
    hex set for recruit actors. None for actions without a target."""
    if ai.target_idx is None:
        return None
    if ai.action_type == "move":
        row = masks.target_valid_move[ai.actor_idx]
    elif ai.action_type == "attack":
        row = masks.target_valid_attack[ai.actor_idx]
    elif ai.action_type == "recruit":
        row = masks.target_valid[ai.actor_idx]
    else:
        return None
    return row > 0.5


def _evaluate(
    model, encoder, holdout_files, device, *,
    eval_pairs: int = 1200,
    eval_pairs_per_game: int = 0,
    eval_sample_seed: Optional[int] = None,
    type_loss_weights: Optional[Dict[str, float]] = None,
    winner_map: Optional[Dict[str, int]] = None,
    cache: Optional[list] = None,
) -> Dict[str, float]:
    """Held-out behavior-cloning metrics: per-head top-1 accuracy +
    mean CE over the first `eval_pairs` pairs of the holdout games
    (deterministic file order -> comparable across evals). The
    encoder vocab is not intentionally grown here; unseen names hit
    the overflow bucket exactly as they would at rollout time.

    Labels are built in the encoder's hex basis
    (`encoder.relevant_set_hexes`). Besides the training CE the
    function reports the target head two ways: `target_ce` (plain
    NLL over all H hex logits, basis-dependent) and
    `target_masked_ce` (`_masked_target_nll`, basis-independent), with
    the masked top-1, the number of pairs behind it, and the count
    of holdout targets the legality mask does not offer
    (`target_off_mask`: the mask is stricter than Wesnoth in places,
    e.g. multi-turn moves).

    `cache`: a list the first call fills with the sample's pairs (their
    RawEncoded, labels, mover and legality row) and later calls read
    instead of reconstructing the holdout games through the simulator
    (about 140 s per probe of 150 games, 28% of a training run's wall
    before 2026-09-11). The sample is deterministic, so the cached
    probe is the same probe."""
    from wesnoth_ai.action_sampler import _build_legality_masks
    was_training = model.training
    model.eval()
    encoder.eval()
    hits = {"actor": 0, "type": 0, "target": 0, "weapon": 0}
    fired = {"actor": 0, "type": 0, "target": 0, "weapon": 0}
    ce_sum, n = 0.0, 0
    ces: List[float] = []         # per-pair CE, for the SE
    target_ces: List[float] = []  # plain target NLL, all H logits
    masked_ces: List[float] = []  # target NLL over the legal set
    masked_hits = 0
    off_mask = off_subset = mask_errors = 0
    ev_win, ev_loss = [], []      # E[V] samples for value AUC
    # Stratified mode (2026-08-25 instrument repair): cap pairs per
    # game so eval_pairs spans MANY games instead of running ~3
    # games deep (the leg-5 probe's 1,200 "pairs" were 60% one
    # game's internal comparisons and its Hanley-McNeil CI assumed
    # they were independent). Per-game statistics + between-game SE
    # replace the pooled AUC when the cap is on.
    per_game: Dict[str, Dict[str, list]] = {}
    _order = sorted(holdout_files)
    if eval_sample_seed is not None and eval_pairs_per_game:
        # The tripwire's "independent redraws" must redraw the GAME
        # sample, not only the within-game pair reservoir (project
        # round-1 C12: a fixed sorted order always fed the same 150
        # date-oldest games, so all three redraws sampled one fixed,
        # biased set). Separate RNG stream from the reservoir's.
        import random as _random
        _random.Random(eval_sample_seed ^ 0x5EED).shuffle(_order)
    def _items():
        """(raw, ai, name, mover, legal row, mask error) per holdout pair:
        from the cache when it is filled, else from the replay stream
        (remembered in the cache when one is given)."""
        if cache:
            yield from cache
            return
        for item in _pair_stream_serial(
                _order,
                max_pairs_per_replay=eval_pairs_per_game,
                sample_seed=eval_sample_seed,
                relevant_set=encoder.relevant_set_hexes):
            if item[0] != "pair":
                continue
            _, state, ai, name = item
            try:
                raw = _raw_one(encoder, state)
            except Exception:                     # noqa: BLE001
                continue
            legal, mask_error = None, None
            if ai.target_idx is not None and ai.action_type != "end_turn":
                try:
                    with torch.no_grad():
                        legal = _legal_target_row(_build_legality_masks(
                            encoder.encode_from_raw(raw, device=device), state), ai)
                    if legal is not None:
                        legal = legal.cpu()
                except Exception as e:            # noqa: BLE001
                    mask_error = repr(e)
            entry = (raw, ai, name, state.global_info.current_side, legal, mask_error)
            if cache is not None:
                cache.append(entry)
            yield entry

    with torch.no_grad():
        for raw, ai, _name, mover, legal, mask_error in _items():
            if n >= eval_pairs:
                break
            try:
                output = model(encoder.encode_from_raw(raw, device=device))
            except Exception:                     # noqa: BLE001
                continue
            n += 1
            off_subset += int(ai.target_off_subset)
            if ai.target_idx is not None and ai.action_type != "end_turn" \
                    and ai.actor_idx < output.actor_logits.size(1) \
                    and ai.target_idx < output.target_logits.size(2):
                tgt_row = output.target_logits[0, ai.actor_idx]
                target_ces.append(float(F.cross_entropy(
                    tgt_row.unsqueeze(0),
                    torch.tensor([ai.target_idx], device=device))))
                if mask_error is not None:
                    mask_errors += 1
                    if mask_errors <= 3:
                        log.warning(f"  legality mask failed on a "
                                    f"holdout pair ({_name}): {mask_error}")
                elif legal is not None:
                    r = _masked_target_nll(tgt_row, legal.to(device), ai.target_idx)
                    if r is None:
                        off_mask += 1
                    else:
                        masked_ces.append(r[0])
                        masked_hits += int(r[1])
            if winner_map and _name in winner_map:
                ev = float(output.value.item())
                (ev_win if winner_map[_name] == mover
                 else ev_loss).append(ev)
                g = per_game.setdefault(_name, {"w": [], "l": [],
                                                "ce": []})
                (g["w"] if winner_map[_name] == mover
                 else g["l"]).append(ev)
            parts = _loss_parts_for_output(
                output, ai, device,
                type_loss_weights=type_loss_weights)
            ce_sum += float(parts.total.item())
            ces.append(float(parts.total.item()))
            if eval_pairs_per_game:
                per_game.setdefault(_name, {"w": [], "l": [],
                                            "ce": []})["ce"].append(
                    float(parts.total.item()))
            A = output.actor_logits.size(1)
            if ai.actor_idx < A:
                fired["actor"] += 1
                if int(output.actor_logits[0].argmax()) == ai.actor_idx:
                    hits["actor"] += 1
                if ai.type_idx is not None:
                    fired["type"] += 1
                    if int(output.type_logits[0, ai.actor_idx].argmax())                             == ai.type_idx:
                        hits["type"] += 1
                if ai.target_idx is not None                         and ai.target_idx < output.target_logits.size(2):
                    fired["target"] += 1
                    if int(output.target_logits[0, ai.actor_idx].argmax())                             == ai.target_idx:
                        hits["target"] += 1
                if ai.weapon_idx is not None                         and ai.weapon_idx < output.weapon_logits.size(2):
                    fired["weapon"] += 1
                    if int(output.weapon_logits[0, ai.actor_idx].argmax())                             == ai.weapon_idx:
                        hits["weapon"] += 1
    if was_training:
        model.train()
        encoder.train()
    out = {"n": n, "ce": (ce_sum / n) if n else float("nan")}
    # Standard error of the mean CE (user ruling 2026-08-20: values
    # don't mean anything without a CI).
    if n >= 2:
        mean = ce_sum / n
        var = sum((c - mean) ** 2 for c in ces) / (n - 1)
        out["ce_se"] = (var / n) ** 0.5
    else:
        out["ce_se"] = None
    for k in hits:
        out[f"{k}_top1"] = (hits[k] / fired[k]) if fired[k] else None

    def _mean_se(xs: List[float]):
        if not xs:
            return None, None
        m = sum(xs) / len(xs)
        if len(xs) < 2:
            return m, None
        v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        return m, (v / len(xs)) ** 0.5

    out["target_n"] = len(target_ces)
    out["target_ce"], _ = _mean_se(target_ces)
    out["target_masked_ce"], out["target_masked_ce_se"] = _mean_se(masked_ces)
    out["target_masked_top1"] = (masked_hits / len(masked_ces)
                                 if masked_ces else None)
    out["target_masked_n"] = len(masked_ces)
    out["target_off_mask"] = off_mask
    out["target_off_subset"] = off_subset
    out["mask_errors"] = mask_errors
    # Value discrimination: P(E[V]_winner-to-move > E[V]_loser-to-
    # move) over holdout states -- the same AUC probe_value_head
    # reports, cheap enough to ride every eval so trunk-drift damage
    # (epoch-0 lesson: late AUC 0.79 -> 0.63) shows up in the CURVE.
    def _auc(w, ls):
        wins = sum(1 for a in w for b in ls if a > b)
        ties = sum(1 for a in w for b in ls if a == b)
        return (wins + 0.5 * ties) / (len(w) * len(ls))

    if eval_pairs_per_game:
        # Stratified statistics (2026-08-25): per-game AUC, mean
        # across games, BETWEEN-GAME SE. Only games containing both
        # winner-mover and loser-mover states contribute an AUC.
        # The Hanley-McNeil form is invalid here: comparisons within
        # a game are heavily dependent (the leg-5 lesson).
        g_aucs = [_auc(g["w"], g["l"]) for g in per_game.values()
                  if g["w"] and g["l"]]
        k = len(g_aucs)
        out["n_auc_games"] = k
        out["n_value"] = len(ev_win) + len(ev_loss)
        if k >= 2:
            m = sum(g_aucs) / k
            var = sum((a - m) ** 2 for a in g_aucs) / (k - 1)
            out["value_auc"] = m
            out["value_auc_se"] = (var / k) ** 0.5
        else:
            out["value_auc"] = g_aucs[0] if g_aucs else None
            out["value_auc_se"] = None
        # CE keeps its definition (mean over pairs) but its SE
        # becomes between-game too.
        g_ces = [sum(g["ce"]) / len(g["ce"])
                 for g in per_game.values() if g["ce"]]
        if len(g_ces) >= 2:
            mc = sum(g_ces) / len(g_ces)
            vc = sum((c - mc) ** 2 for c in g_ces) / (len(g_ces) - 1)
            out["ce_se"] = (vc / len(g_ces)) ** 0.5
        out["n_ce_games"] = len(g_ces)
    elif ev_win and ev_loss:
        n1, n2 = len(ev_win), len(ev_loss)
        auc = _auc(ev_win, ev_loss)
        out["value_auc"] = auc
        out["n_value"] = n1 + n2
        # Hanley-McNeil (1982) SE -- only meaningful when the pooled
        # samples are independent (NOT the case under file-sequential
        # uncapped streaming; kept for the legacy path only).
        q1 = auc / (2 - auc)
        q2 = 2 * auc * auc / (1 + auc)
        var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc * auc)
               + (n2 - 1) * (q2 - auc * auc)) / (n1 * n2)
        out["value_auc_se"] = max(var, 0.0) ** 0.5
    else:
        out["value_auc"] = None
        out["value_auc_se"] = None
        out["n_value"] = 0
    return out


def _log_eval(stats: Dict, epoch: int, global_step: int,
              pairs: int, checkpoint_out: Path, tag: str = "") -> None:
    """One log line + one JSONL row per held-out eval, so training
    yields a generalization CURVE (house convention: mid-epoch
    measurements, not endpoints). JSONL sits next to the checkpoint:
    <stem>_eval.jsonl."""
    def fmt(v):
        return "n/a" if v is None else f"{v:.3f}"
    log.info(
        f"EVAL{('[' + tag + ']') if tag else ''} step={global_step} "
        f"pairs={pairs} n={stats['n']} ce={stats['ce']:.4f} "
        f"actor_top1={fmt(stats['actor_top1'])} "
        f"type_top1={fmt(stats['type_top1'])} "
        f"target_top1={fmt(stats['target_top1'])} "
        f"weapon_top1={fmt(stats['weapon_top1'])} "
        f"value_auc={fmt(stats.get('value_auc'))} "
        f"target_masked_ce={fmt(stats.get('target_masked_ce'))} "
        f"masked_top1={fmt(stats.get('target_masked_top1'))} "
        f"masked_n={stats.get('target_masked_n')} "
        f"off_mask={stats.get('target_off_mask')} "
        f"off_subset={stats.get('target_off_subset')}")
    try:
        row = dict(stats)
        row.update({"epoch": epoch, "step": global_step,
                    "pairs": pairs, "tag": tag,
                    "ts": time.strftime("%FT%T")})
        out = checkpoint_out.with_name(checkpoint_out.stem + "_eval.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except Exception as e:                            # noqa: BLE001
        log.warning(f"eval log write failed: {e}")


def train(
    dataset_dir: Path,
    checkpoint_out: Path,
    epochs: int      = 1,
    batch_size: int  = 8,
    lr: float        = 1e-4,
    max_replays: int = 0,           # 0 = all
    max_pairs: int   = 0,           # 0 = all (per epoch)
    log_every: int   = 100,
    ckpt_every: int  = 2000,        # steps between periodic checkpoints
    gc_every_files:  int = 16,      # gc.collect() this often (replay files)
    max_replay_commands: int = 1500,    # skip a replay file if it exceeds
    max_starting_units:  int = 0,       # 0 = no cap (TSG ships with ~24
                                        #     statues for recruit-hex
                                        #     mechanics — legit)
    max_pairs_per_replay: int = 0,      # 0 = no cap (every replay
                                        #     weighted equally)
    device_str: str  = "cpu",
    competitive_only: bool = True,
    resume: Optional[Path] = None,
    workers: int     = 0,            # >0 = prefetch encode_raw in N
                                     # subprocesses
    prefetch_factor: int = 4,        # in-flight items per worker target
    batched_forward: Optional[bool] = None,  # None = auto (on iff GPU)
    d_model: int = 128,              # model/encoder width; the tier-a
    num_layers: int = 3,             # 5M arch is 256/6/8/1024
    num_heads: int = 4,
    d_ff: int = 256,
    holdout_games: int = 300,        # held-out GAMES (never trained on)
    value_loss_weight: float = 1.0,  # lambda_v on SELECTED value
                                     # states (0 = legacy policy-only)
    value_states_per_game: int = 16, # expected value states sampled
                                     # per game per epoch
    eval_every: int = 50_000,        # pairs between held-out evals
                                     # (0 = only at epoch ends)
    eval_pairs: int = 1200,          # held-out pairs per eval
    eval_pairs_per_game: int = 0,    # stratified probe cap (0=legacy)
    eval_sample_seed: Optional[int] = None,  # seeded random redraw
    eval_only: bool = False,         # evaluate --resume ckpt and exit
    eval_json: Optional[Path] = None,  # eval-only: also dump stats JSON
    reinit_value_head: bool = False,
    value_material: bool = False,   # the value head also reads material
    preencoded: Optional[Path] = None,  # tools/preencode_corpus.py output
    bf16: bool = False,                 # autocast the batched flow's forward in bf16
    tf32: bool = False,                 # fp32 matmuls on the tensor cores (a recipe change: opt-in)
    fused_adamw: bool = False,          # the fused optimizer step (another one: opt-in)
    fog_hides_enemy_villages: "bool | None" = None,  # global feature 5 under fog
    terrain_multi_hot: "bool | None" = None,
        # the hex's terrain as its full set from the engine's aliases
        # (encoder.terrain_tokens); None = on for a fresh network, a
        # checkpoint's own setting on a warm start. Rides the checkpoint.
        # drop value_head.* from the --resume state (and skip the
        # optimizer-state restore): warm trunk+policy, fresh value.
        # Imitation A/B 2026-08-08 verdict -- see the resume block.
    imitation_config: Optional[Path] = None,
        # configs/imitation.json: winners-only policy loss, per-game
        # weighting, manifest holdout. Requires manifest.jsonl in
        # dataset_dir (tools/build_imitation_dataset.py).
    type_loss_weights: Optional[Dict[str, float]] = None,
        # per-action-type loss weights; None -> defaults
        # (_DEFAULT_ACTION_TYPE_LOSS_WEIGHT). Pass via --action-type-weights
        # JSON; see tools/compute_action_type_weights.py.
    relevant_set_hexes: bool = False,
        # hex basis of the encoder AND the labels: the relevant subset
        # (visibility.relevant_hexes_in_slot_order) instead of the full
        # board. Recorded in the checkpoint; eval builds the same basis.
    init_from: Optional[Path] = None,
        # warm start: model + encoder weights and vocab from this
        # checkpoint, fresh optimizer and counters (step 0, epoch 0,
        # LR schedule from the start). --resume continues a run instead.
    seed: Optional[int] = None,
        # seeds `random` (file order, value-state subsampling) and torch,
        # so two runs on the same corpus see the same pair stream.
) -> None:
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        log.info(f"Seed: {seed} (file order, value subsampling, torch)")
    if resume is not None and init_from is not None:
        raise ValueError("--resume and --init-from are exclusive")
    # `--device dml` (or `dml:N`) routes through Microsoft DirectML
    # for AMD/Intel GPU acceleration on Windows. NVIDIA users keep
    # passing `cuda` which torch resolves itself.
    if device_str == "dml" or device_str.startswith("dml:"):
        try:
            import torch_directml
        except ImportError:
            raise RuntimeError(
                "torch-directml is not installed. "
                "`pip install torch-directml` to enable DirectML."
            )
        idx = 0
        if ":" in device_str:
            try:
                idx = int(device_str.split(":", 1)[1])
            except ValueError:
                idx = 0
        if idx >= torch_directml.device_count():
            raise RuntimeError(
                f"DirectML device {idx} requested but only "
                f"{torch_directml.device_count()} present"
            )
        device = torch_directml.device(idx)
        log.info(f"Device: DirectML[{idx}] = {torch_directml.device_name(idx)}")
    else:
        device = torch.device(device_str)
        log.info(f"Device: {device}")
    if tf32:
        from wesnoth_ai.train_perf import enable_tf32
        enable_tf32(True)

    # Peek the resume checkpoint FIRST: the SL<->MCTS round-trip
    # contract (2026-07-16) requires (a) constructing the OPTIONAL
    # heads (aux_score / moves_left) the checkpoint carries --
    # otherwise strict=False loading silently DROPS the campaign's
    # trained aux head as "unexpected keys"; (b) validating the arch
    # instead of loading a 256-wide checkpoint into a 128 model; and
    # (c) carrying decision_step (combat-oracle anneal) through
    # unchanged so the next MCTS resume doesn't restart the anneal.
    ckpt = None
    aux_flag = moves_flag = False
    carry: Dict = {}
    ckpt_src = resume if resume is not None else init_from
    if ckpt_src is not None and not ckpt_src.exists():
        if init_from is not None:
            raise FileNotFoundError(f"--init-from {init_from} not found")
        log.warning(f"--resume {resume} not found; starting fresh")
        ckpt_src = None
    if ckpt_src is not None:
        ckpt = torch.load(ckpt_src, map_location="cpu",
                          weights_only=False)
        saved_arch = ckpt.get("arch") or {}
        ours = {"d_model": d_model, "num_layers": num_layers,
                "num_heads": num_heads, "d_ff": d_ff}
        for k, v in ours.items():
            if saved_arch and saved_arch.get(k) != v:
                raise RuntimeError(
                    f"checkpoint arch mismatch on '{k}': {ckpt_src} "
                    f"has {saved_arch.get(k)!r}, flags say {v!r}. "
                    f"Pass the checkpoint's arch explicitly.")
        # Hex basis: a checkpoint trained on the full board warm-
        # starting the relevant-set arm is the experiment (docs/
        # model_cost_study_20260905.md 7); evaluating across bases
        # is meaningless, so eval-only refuses.
        ckpt_basis = bool(ckpt.get("relevant_set_hexes"))
        if ckpt_basis != relevant_set_hexes:
            msg = (f"hex basis switch: {ckpt_src.name} has "
                   f"relevant_set_hexes={ckpt_basis}, this run "
                   f"{relevant_set_hexes}")
            if eval_only:
                raise RuntimeError(msg + " (pass the checkpoint's basis "
                                   "for --eval-only)")
            log.warning(f"  {msg}: warm start across hex bases")
        ms = ckpt.get("model_state", {})
        aux_flag = bool(ckpt.get("aux_score")) or any(
            k.startswith("aux_score_head.") for k in ms)
        moves_flag = bool(ckpt.get("moves_left")) or any(
            k.startswith("moves_left_head.") for k in ms)
        # The material input of the value head: on when asked for
        # (--value-material, a warm start grafts the zero-initialized
        # projection) or when the checkpoint carries it.
        value_material = bool(value_material) or bool(ckpt.get("value_material")) or any(
            k.startswith("material_proj.") for k in ms)
        # The fog gate of global feature 5: an explicit flag wins,
        # else the checkpoint's own setting (a checkpoint without the
        # key was trained on the true count).
        if fog_hides_enemy_villages is None:
            fog_hides_enemy_villages = bool(ckpt.get("fog_hides_enemy_villages", False))
        # The terrain view, the same way; switching it under a warm
        # start re-reads every hex token, so it is logged, and an
        # eval-only run never measures a checkpoint in another view.
        ckpt_terrain = bool(ckpt.get("terrain_multi_hot", False))
        if terrain_multi_hot is None:
            terrain_multi_hot = ckpt_terrain
        elif bool(terrain_multi_hot) != ckpt_terrain:
            msg = (f"terrain view switch: {ckpt_src.name} has "
                   f"terrain_multi_hot={ckpt_terrain}, this run {bool(terrain_multi_hot)}")
            if eval_only:
                raise RuntimeError(msg + " (pass the checkpoint's view for --eval-only)")
            log.warning(f"  {msg}: warm start across terrain views")
        for k in ("decision_step", "aux_score", "moves_left"):
            if k in ckpt:
                carry[k] = ckpt[k]
        carry.setdefault("aux_score", aux_flag)
        carry.setdefault("moves_left", moves_flag)
        if "decision_step" in carry:
            log.info(f"  carrying decision_step="
                     f"{carry['decision_step']} through the SL pass")

    if fog_hides_enemy_villages is None:
        fog_hides_enemy_villages = True      # a fresh network never reads the hidden count
    if terrain_multi_hot is None:
        terrain_multi_hot = True             # a fresh network reads the full terrain set
    encoder = GameStateEncoder(
        d_model=d_model, relevant_set_hexes=relevant_set_hexes,
        fog_hides_enemy_villages=bool(fog_hides_enemy_villages),
        terrain_multi_hot=bool(terrain_multi_hot)).to(device)
    carry["fog_hides_enemy_villages"] = bool(fog_hides_enemy_villages)
    if fog_hides_enemy_villages:
        log.info("Global feature 5 under fog: the enemy villages the mover can see")
    log.info(f"Hex terrain: {'the full set from the engine aliases (multi-hot)' if terrain_multi_hot else 'one class per hex'}")
    model   = WesnothModel(d_model=d_model, num_layers=num_layers,
                           num_heads=num_heads, d_ff=d_ff,
                           aux_score=aux_flag,
                           moves_left=moves_flag,
                           value_material=bool(value_material)).to(device)
    carry["value_material"] = bool(value_material)
    if value_material:
        log.info("Value head input: material (wesnoth_ai/material.py)")
    model.train()
    encoder.train()
    arch_record = {"d_model": d_model, "num_layers": num_layers,
                   "num_heads": num_heads, "d_ff": d_ff}
    log.info(f"Hex basis: {'RELEVANT SET' if relevant_set_hexes else 'full board'}")
    training_meta = {
        "init_from": str(init_from) if init_from is not None else None,
        "resume": str(resume) if resume is not None else None,
        "seed": seed, "max_pairs": max_pairs,
        "dataset_dir": str(dataset_dir),
    }
    save_kwargs = dict(arch=arch_record, carry=carry,
                       relevant_set_hexes=relevant_set_hexes,
                       training_meta=training_meta,
                       terrain_multi_hot=bool(terrain_multi_hot))

    from wesnoth_ai.train_perf import adamw, reassert_step_kernel
    opt = adamw(list(model.parameters()) + list(encoder.parameters()),
                lr=lr, weight_decay=1e-4, fused=fused_adamw)

    # Cosine learning-rate decay across the planned epoch budget. With
    # the resume-from-checkpoint path, `T_max` is the TOTAL planned
    # epochs (not just the remaining ones) -- the scheduler is
    # advanced once per epoch and we use `last_epoch=resumed_epoch`
    # to skip forward to the right point on the cosine curve. The
    # late-epoch sharpening this gives is empirically helpful for
    # behavior-cloning loss to converge tightly.
    #
    # `eta_min = lr * 0.05` (i.e. final lr is 5% of initial) -- not
    # zero, so opt continues to learn through the last few hundred
    # steps; not the typical 0.1 because the corpus is small enough
    # that we benefit from a steeper decay.
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * 0.05,
    )

    # Optional resume: restore model + encoder + optimizer state from
    # a previous checkpoint, and continue the pass where it was cut
    # (`_pass_position`).
    resumed_step = 0
    resumed_pairs = 0
    resumed_epoch = 0
    pass_position: Optional[PassPosition] = None
    next_epoch_rng_state: Optional[tuple] = None
    resumed_last_eval: Optional[int] = None
    if ckpt is not None:
        log.info(f"{'Warm start (weights only) from' if init_from else 'Resuming from'} "
                 f"{ckpt_src}")
        # strict=False so an architecture-additive change (new head,
        # new embedding column) can warm-start from a prior
        # checkpoint without losing the heads that DID exist.
        # Concrete case: C.1 added model.type_head; the cluster's
        # epoch-3 checkpoint predates it. We log the deltas so a
        # silent vocab/encoder regression can't sneak through
        # disguised as "ah, that's just the new head".
        if reinit_value_head:
            # Imitation A/B verdict (2026-08-08): the warm-started
            # trunk+policy dominate everywhere (holdout CE 3.107 vs
            # 3.449), but the warm VALUE head -- trained on search-
            # backed z targets -- fights the outcome supervision all
            # run (AUC oscillating 0.52-0.89, final 0.538) while a
            # fresh head climbs cleanly to 0.951. Drop the value-head
            # weights from the checkpoint so they train from random
            # init while everything else warm-starts.
            n_drop = 0
            for k in list(ckpt["model_state"].keys()):
                if k.startswith("value_head."):
                    del ckpt["model_state"][k]
                    n_drop += 1
            log.info(f"  --reinit-value-head: dropped {n_drop} value "
                     f"head tensor(s) from the resume state")
        m_missing, m_unexpected = model.load_state_dict(
            ckpt["model_state"], strict=False)
        if m_missing:
            log.warning(f"  model: {len(m_missing)} missing key(s) "
                        f"(will train from random init): {m_missing}")
        if m_unexpected:
            log.warning(f"  model: {len(m_unexpected)} unexpected key(s) "
                        f"in checkpoint (ignored): {m_unexpected}")
        from wesnoth_ai.encoder import pad_legacy_encoder_state
        e_missing, e_unexpected = encoder.load_state_dict(
            pad_legacy_encoder_state(ckpt["encoder_state"], encoder),
            strict=False)
        if e_missing:
            log.warning(f"  encoder: {len(e_missing)} missing key(s) "
                        f"(will train from random init): {e_missing}")
        if e_unexpected:
            log.warning(f"  encoder: {len(e_unexpected)} unexpected key(s) "
                        f"in checkpoint (ignored): {e_unexpected}")
        encoder.unit_type_to_id = dict(ckpt.get("unit_type_to_id", {}))
        if "faction_to_id" in ckpt:
            encoder.faction_to_id = dict(ckpt["faction_to_id"])
        if init_from is not None:
            # Weights and vocab only: the optimizer, the step / pair
            # / epoch counters and the LR schedule start fresh, so a
            # completed-epoch checkpoint can seed a new run without
            # `range(resumed_epoch, epochs)` skipping it.
            log.info("  --init-from: fresh optimizer and counters")
        elif "optimizer_state" in ckpt and reinit_value_head:
            # Fresh value-head params must not inherit the old head's
            # Adam moments (state entries match by param order, so the
            # stale moments would land ON the re-initialized tensors).
            # A new training phase re-accumulates momentum cheaply.
            log.info("  --reinit-value-head: skipping optimizer-state "
                     "restore (fresh momentum)")
        elif "optimizer_state" in ckpt:
            try:
                opt.load_state_dict(ckpt["optimizer_state"])
                reassert_step_kernel(opt)      # the checkpoint's groups carry their own `fused`
                # Padded legacy encoder tensors need their Adam
                # moments padded too (see encoder.py helper).
                from wesnoth_ai.encoder import repair_optimizer_state_shapes
                repair_optimizer_state_shapes(opt)
            except Exception as e:
                log.warning(f"  optimizer state restore failed ({e}); "
                            f"re-accumulating momentum from scratch")
    if ckpt is not None and init_from is None:
        resumed_step  = int(ckpt.get("supervised_step", 0))
        resumed_pairs = int(ckpt.get("supervised_pairs", 0))
        # Global epoch counter -- count of fully-completed epochs
        # across the WHOLE chain (not just this link). New
        # checkpoints carry it; older ones don't. Fall back to
        # counting per-epoch snapshot files (the previous
        # heuristic) when the key is missing.
        resumed_epoch = int(ckpt.get("supervised_epoch", -1))
        if resumed_epoch < 0:
            resumed_epoch = 0
            for n in range(epochs):
                cand = resume.parent / f"supervised_epoch{n}.pt"
                if cand.exists():
                    resumed_epoch = n + 1
        log.info(
            f"  resumed at step={resumed_step} "
            f"pairs={resumed_pairs} epoch={resumed_epoch}"
        )
        pass_position, next_epoch_rng_state = _pass_position(
            ckpt, resumed_epoch, resumed_pairs, resumed_step, seed)
        if "supervised_resume" in ckpt:
            resumed_last_eval = int(ckpt["supervised_resume"]["last_eval_pairs"])
        # Fast-forward the LR scheduler past completed epochs. We
        # don't checkpoint the scheduler's state directly; we
        # reconstruct it from the epoch counter at resume time.
        for _ in range(resumed_epoch):
            lr_scheduler.step()
        log.info(f"  LR scheduler advanced {resumed_epoch} epochs; "
                 f"current lr = {opt.param_groups[0]['lr']:.2e}")

    # Competitive-2p filter: reads index.jsonl entries and only keeps
    # replays on ships-with ladder maps with two default-faction sides.
    # Drops ~97% of the raw dataset (most replays are FFA / multi-side).
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"replay dataset {dataset_dir} does not exist")
    if competitive_only:
        files = filter_competitive_2p(dataset_dir)
        log.info(f"Competitive-2p filter: {len(files)} of "
                 f"{len(list(dataset_dir.glob('*.json.gz')))} replays kept")
    else:
        files = sorted(dataset_dir.glob("*.json.gz"))
        log.info(f"No scenario filter: using all {len(files)} replays")

    # Size filter: drop pathologically large replays that, in our
    # earlier overnight run, drove RAM use to ~65% and triggered OS
    # swap thrashing for hours. The default threshold (1500 commands)
    # comes from the corpus distribution survey: p99 ≈ 1450, and the
    # worst 4 outliers (2000-3300 commands, 200-274 unit accumulations)
    # are what caused the stall. Cheap pre-filter using the index.jsonl
    # `n_commands` field — no need to open the gz.
    if max_replay_commands or max_starting_units:
        files = _apply_size_filters(
            files, dataset_dir,
            max_commands=max_replay_commands,
            max_starting=max_starting_units,
        )
        log.info(f"After size filters: {len(files)} replay files remain")
    if not files:
        raise ValueError(f"no replay in {dataset_dir} passes the filters")

    # Held-out split BY GAME (seed-0 deterministic shuffle of the
    # post-filter file list; the last `holdout_games` files are never
    # trained on). Held-out top-1 accuracy is the SL acceptance
    # metric -- logged periodically so training yields a curve, not
    # an endpoint.
    holdout_files: List[Path] = []
    # Imitation mode (configs/imitation.json + the dataset's
    # manifest.jsonl from tools/build_imitation_dataset.py):
    #   - winners-only policy loss (loser pairs keep value duty),
    #   - per-game equal weighting (median/game winner-action count,
    #     clipped to [0.25, 4] to bound E[w^2] -- see the value-
    #     subsampling comment below for why unbounded per-game
    #     weights destabilize file-sequential batches),
    #   - manifest-driven deterministic holdout split (stable across
    #     rebuilds; replaces the shuffled last-N split).
    imit_policy_w: Optional[Dict[str, float]] = None
    imit_winners_only = False
    imit_winner_map: Dict[str, int] = {}
    if imitation_config is not None:
        icfg = json.loads(
            Path(imitation_config).read_text(encoding="utf-8"))
        man_path = dataset_dir / "manifest.jsonl"
        if not man_path.is_file():
            raise RuntimeError(
                f"--imitation-config given but {man_path} is missing; "
                f"run tools/build_imitation_dataset.py first")
        man_rows = [json.loads(line) for line in
                    man_path.open(encoding="utf-8")]
        imit_winners_only = bool(icfg.get("policy_winners_only", True))
        per_game = bool(icfg.get("per_game_weight", True))
        # Config-first: the config's value_from_outcome_weight governs
        # lambda_v in imitation mode (overrides the CLI flag — the
        # techniques-inventory audit found this key silently dead,
        # its _doc promising a disable that never happened).
        if "value_from_outcome_weight" in icfg:
            value_loss_weight = float(icfg["value_from_outcome_weight"])
            log.info(f"  value_from_outcome_weight from config: "
                     f"{value_loss_weight}")
        acts = sorted(r["winner_actions"] for r in man_rows
                      if r["winner_actions"] > 0)
        med_actions = acts[len(acts) // 2] if acts else 1
        imit_policy_w = {}
        for r in man_rows:
            if per_game and r["winner_actions"] > 0:
                w = med_actions / r["winner_actions"]
                w = max(0.25, min(4.0, w))
            else:
                w = 1.0
            imit_policy_w[r["file"]] = w
            imit_winner_map[r["file"]] = int(r["winner_side"])
        # Only manifest games train: a file in the directory but not
        # in the manifest is quarantined or stale (2026-09-08: the
        # scan had kept training on the 20 quarantined games).
        manifest_names = {r["file"] for r in man_rows}
        off_manifest = [f.name for f in files if f.name not in manifest_names]
        if off_manifest:
            log.info(f"  {len(off_manifest)} files in {dataset_dir} are not in the "
                     f"manifest and are skipped (first: {off_manifest[:2]})")
            files = [f for f in files if f.name in manifest_names]
        holdout_names = {r["file"] for r in man_rows if r["holdout"]}
        holdout_files = [f for f in files if f.name in holdout_names]
        files = [f for f in files if f.name not in holdout_names]
        log.info(
            f"Imitation mode: {len(man_rows)} manifest games, "
            f"winners_only={imit_winners_only}, per_game={per_game} "
            f"(median actions {med_actions}), holdout "
            f"{len(holdout_files)}")
    elif holdout_games > 0 and len(files) > holdout_games * 2:
        _split = sorted(files)
        random.Random(0).shuffle(_split)
        holdout_files = _split[-holdout_games:]
        files = _split[:-holdout_games]
        log.info(f"Holdout: {len(holdout_files)} games held out; "
                 f"{len(files)} remain for training")

    if max_replays:
        files = files[:max_replays]
    log.info(f"Training on {len(files)} replay files")

    # Joint value training via VALUE-STATE SUBSAMPLING (2026-07-16,
    # replaces per-state weight scaling): each state carries the
    # value loss with probability k/n_cmds(game) at UNIFORM weight
    # lambda_v. Expected per-game total = k*lambda_v -- equal per
    # game like before -- but (a) per-state weight is bounded (no
    # 6.5x short-game spikes; E[w^2] controlled), and (b) since the
    # pair stream is FILE-SEQUENTIAL, a batch used to fill up with
    # 64 same-label same-weight states of one game and shove the
    # value head in one direction per step; with ~k states per game
    # selected, same-game value states rarely share a batch. This
    # was the actual instability mechanism behind value_auc
    # oscillating 0.36<->0.76: equal-total-per-epoch normalization
    # can't fix within-step correlation. (The MCTS trainer's
    # identical game_weight scheme is stable because ITS batches
    # come from a shuffled replay buffer.)
    winner_map: Dict[str, int] = {}
    value_select_p: Dict[str, float] = {}
    if value_loss_weight > 0.0:
        idx_path = dataset_dir / "value_corpus_index.jsonl"
        if idx_path.is_file():
            k = max(1, int(value_states_per_game))
            with idx_path.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    w = row.get("winner")
                    if w in (1, 2):
                        name = row["file"]
                        winner_map[name] = int(w)
                        n = max(1, int(row.get("n_commands", 1)))
                        value_select_p[name] = min(1.0, k / n)
            log.info(f"joint value loss ON (lambda_v="
                     f"{value_loss_weight}, ~{k} value states/game): "
                     f"{len(winner_map)} labeled games")
        else:
            log.warning(f"no {idx_path} -- joint value loss OFF")
            value_loss_weight = 0.0
    else:
        log.info("joint value loss OFF (value_loss_weight=0)")
    # Imitation manifest winners feed the same map: the winners-only
    # policy filter and the holdout value-AUC probe both read it,
    # independent of whether the value LOSS is enabled. (Files absent
    # from value_corpus_index keep select_p 0.0, so this cannot
    # enable value training by itself.)
    winner_map.update(imit_winner_map)
    eval_cache: list = []      # the holdout probe's sample, built once

    if eval_only:
        if not holdout_files:
            log.error("--eval-only needs a holdout split")
            return
        stats = _evaluate(model, encoder, holdout_files, device,
                          eval_pairs=eval_pairs,
                          eval_pairs_per_game=eval_pairs_per_game,
                          eval_sample_seed=eval_sample_seed,
                          type_loss_weights=type_loss_weights,
                          winner_map=winner_map, cache=eval_cache)
        stats["decision_step"] = carry.get("decision_step")
        log.info(f"EVAL-ONLY {stats}")
        if eval_json is not None:
            import json as _json
            Path(eval_json).write_text(_json.dumps(stats),
                                       encoding="utf-8")
        return

    # A fresh encoder gets every reachable unit type's row BEFORE
    # workers spawn: they receive a snapshot and never grow it, and the
    # pre-encoded records carry the seeded ids (tools/unit_vocab.py). A
    # resumed run keeps its checkpoint's vocab.
    if ckpt is None:
        seed_vocab(encoder)
        log.info(
            f"Pre-seeded encoder vocab: "
            f"{len(encoder.unit_type_to_id)} unit types, "
            f"{len(encoder.faction_to_id)} factions"
        )

    # Training loop: per-pair forward+backward with mini-batch gradient
    # accumulation. We do NOT collect pairs into a list — the streaming
    # iterator mutates the same GameState object across iterations for
    # speed, so a list would snapshot stale state references. Streaming
    # is both correct and memory-frugal (only one chunk's activation
    # graph lives at a time).
    running_loss = deque(maxlen=200)
    # Per-head loss running averages. Each only sees pairs where the
    # corresponding head fired (see LossParts) — actor is every pair,
    # target is every move/attack/recruit (i.e. not end_turn), weapon
    # is every attack. So the deques can grow at different rates.
    running_loss_actor  = deque(maxlen=200)
    running_loss_type   = deque(maxlen=200)
    running_loss_target = deque(maxlen=200)
    running_loss_weapon = deque(maxlen=200)
    running_loss_value  = deque(maxlen=200)
    # running_count is the CHAIN-cumulative pair count (seeded from
    # the checkpoint's supervised_pairs, so the recorded total keeps
    # accumulating across links -- project round-1 C16: seeding 0
    # reset it every resume while the comment claimed otherwise).
    # max_pairs is compared against THIS-RUN pairs via
    # run_start_count.
    running_count = resumed_pairs
    run_start_count = resumed_pairs
    last_eval_pairs = resumed_pairs if resumed_last_eval is None else resumed_last_eval
    global_step = resumed_step
    t_start = time.time()
    stop = False
    files_seen = 0
    file_errors = 0
    flush_failures = 0        # batches lost to an exception (warned, counted)
    autocast_dtype = torch.bfloat16 if bf16 else None
    if bf16:
        log.info("Batched flow under bf16 autocast (fp32 weights and gradients)")
    oom_halvings = 0          # out-of-memory halvings of a batch (nothing lost)
    # Relevant-set basis: labelled targets with no subset slot (kept
    # as actor/type/weapon pairs, target head silent). Expected 0;
    # every one is a superset violation worth a look.
    target_off_subset = 0

    # Stage profiling (WESNOTH_PROF=1, same env flag as the rollout
    # prof system): wall-time accumulators for the three loop stages.
    #   wait   = producer stall (worker prefetch / disk / extract)
    #   encode = phase-2 encoding on the main thread
    #   flush  = forward + backward + opt step (on CUDA the .tolist()
    #            sync inside _flush_batch absorbs async kernel time,
    #            so `flush` is an honest GPU-side total)
    # Reported in every log line and dumped to <ckpt>_prof.json at
    # each eval -- the box-sizing readout (CPU-encode vs GPU-forward
    # balance) for tier-b hardware selection.
    prof_on = bool(int(os.environ.get("WESNOTH_PROF", "0") or 0))
    prof_acc = {"wait": 0.0, "encode": 0.0, "flush": 0.0,
                "other": 0.0, "pairs": 0}
    prof_path = checkpoint_out.with_name(
        checkpoint_out.stem + "_prof.json")

    # Auto-pick batched vs per-pair forward. Batched amortizes
    # per-forward kernel-launch overhead across `bs` pairs at the cost
    # of padding-to-max-seq waste; on GPU the launch overhead clearly
    # dominates and batched wins, on CPU the padding waste over
    # 1700-hex sequences is too expensive (smoke showed 27/s →
    # 3-6/s regression). Default to "on iff GPU" so the cluster
    # benefits and local development stays responsive.
    if batched_forward is None:
        use_batched = device.type != "cpu"
    else:
        use_batched = bool(batched_forward)
    log.info(
        f"Forward mode: {'BATCHED' if use_batched else 'per-pair'} "
        f"(device={device.type}, batched_forward={batched_forward})"
    )

    # Loop counts GLOBAL epochs across the whole chain. After a
    # walltime-cut and resume, this picks up at `resumed_epoch`
    # rather than restarting from 0 -- so each link advances the
    # global counter and per-epoch snapshot filenames don't
    # collide between links.
    for epoch in range(resumed_epoch, epochs):
        if stop:
            break
        position = pass_position if (pass_position is not None
                                     and pass_position.epoch == epoch) else None
        if position is not None and position.epoch_rng_state is not None:
            random.setstate(position.epoch_rng_state)
        elif epoch == resumed_epoch and next_epoch_rng_state is not None:
            random.setstate(next_epoch_rng_state)
        epoch_rng_state = random.getstate()
        epoch_start_pairs = running_count - (position.skip_pairs if position else 0)
        epoch_start_step = global_step - (position.step_in_epoch if position else 0)
        random.shuffle(files)
        step = position.step_in_epoch if position else 0
        skip_left = position.skip_pairs if position else 0
        t_epoch = time.time()
        # Snapshot the cumulative `running_count` at epoch start so the
        # rate log uses pairs-this-epoch / elapsed-this-epoch. Without
        # this the rate report blows up at the start of each epoch
        # past the first (the cumulative count is already large but
        # the elapsed timer resets, giving e.g. 150000 pairs/sec for
        # the first log step of epoch 2).
        epoch_start_count = running_count

        # Producer: a stream of ("pair", state_or_raw, ai, gz_name)
        # events plus ("file_done", gz_name, n) markers. Either serial
        # (does encoding inline) or parallel (workers prefetch the
        # encode_raw side; main does encode_from_raw + forward + back).
        if preencoded is not None:
            check_preencoded(preencoded, files, encoder, relevant_set_hexes)
            log.info(f"Pairs from the pre-encoded corpus {preencoded} "
                     f"(workers ignored; the per-replay cap does not apply)")
            stream = _pair_stream_preencoded(files, preencoded)
        elif workers > 0:
            stream = _pair_stream_parallel(
                files,
                workers=workers,
                type_to_id=encoder.unit_type_to_id,
                faction_to_id=encoder.faction_to_id,
                prefetch_factor=prefetch_factor,
                max_pairs_per_replay=max_pairs_per_replay,
                relevant_set=relevant_set_hexes,
                fog_hides_enemy_villages=bool(encoder.fog_hides_enemy_villages),
                terrain_multi_hot=bool(encoder.terrain_multi_hot),
            )
        else:
            stream = _pair_stream_serial(
                files,
                max_pairs_per_replay=max_pairs_per_replay,
                relevant_set=relevant_set_hexes,
            )

        # Per-pair / batched: shared bookkeeping below; the differences
        # are concentrated in the "pair" event handler.
        batch_raws: List = []
        batch_ais: List[ActionIndices] = []
        batch_zw: List = []
        losses_in_batch = 0  # used by per-pair flow
        params_for_clip = list(model.parameters()) + list(encoder.parameters())
        opt.zero_grad()
        try:
            _stream_iter = iter(stream)
            while True:
                if stop:
                    break
                _tw = time.perf_counter() if prof_on else 0.0
                try:
                    event = next(_stream_iter)
                except StopIteration:
                    break
                if prof_on:
                    prof_acc["wait"] += time.perf_counter() - _tw
                kind = event[0]

                if kind == "file_done":
                    files_seen += 1
                    # In batched mode we let partial batches span file
                    # boundaries (each forward_batch is expensive). In
                    # per-pair mode, gradients are already accumulated
                    # per-pair via `(loss/bs).backward()`; flushing is
                    # a free `opt.step()` on whatever's accumulated, so
                    # we do it for crash-resilience.
                    if not use_batched and losses_in_batch > 0:
                        torch.nn.utils.clip_grad_norm_(params_for_clip, 1.0)
                        opt.step()
                        opt.zero_grad()
                        losses_in_batch = 0
                    if files_seen % gc_every_files == 0:
                        gc.collect()
                    continue

                if kind == "file_error":
                    files_seen += 1
                    file_errors += 1
                    _, gz_name, err = event
                    # WARNING, not debug: a silent per-file error rate
                    # is exactly what made the 2026-08-08 random-arm
                    # underrun (epoch "done" at 171k of 2.5M pairs)
                    # undiagnosable post-hoc. First few get the full
                    # error; the rest count silently into the epoch
                    # summary line.
                    if file_errors <= 5:
                        log.warning(f"  file_error {gz_name}: {err}")
                    # Abandon any partial gradient or batch from this
                    # file — its data is incomplete.
                    opt.zero_grad()
                    batch_raws.clear()
                    batch_ais.clear()
                    batch_zw.clear()
                    losses_in_batch = 0
                    continue

                # kind == "pair"
                _, state_or_raw, ai, _gz_name = event

                if ai.target_off_subset:
                    target_off_subset += 1
                    if target_off_subset <= 5:
                        log.warning(
                            f"  relevant-set gap: {_gz_name} "
                            f"{ai.action_type} target has no subset "
                            f"slot (pair kept, target head silent)")

                if isinstance(state_or_raw, RawEncoded):
                    mover = 1 if state_or_raw.global_feats[1] < 0 else 2
                else:
                    mover = state_or_raw.global_info.current_side

                # Imitation-mode policy weight: winners-only zeroes
                # the policy heads on loser-side pairs (they can still
                # carry value supervision below); per-game weighting
                # scales by median/game winner-action count. 1.0 when
                # no imitation manifest is loaded (legacy behavior).
                p_w = 1.0
                if imit_policy_w is not None:
                    p_w = imit_policy_w.get(_gz_name, 1.0)
                    if (imit_winners_only
                            and winner_map.get(_gz_name) != mover):
                        p_w = 0.0

                # Joint value target: subsampled -- this state
                # carries the value loss with prob k/n(game) at
                # uniform weight lambda_v (see the selection-map
                # comment above for why not per-state weights).
                v_z, v_w = None, 0.0
                if (value_loss_weight > 0.0 and _gz_name in winner_map
                        and random.random()
                        < value_select_p.get(_gz_name, 0.0)):
                    v_z = 1 if winner_map[_gz_name] == mover else -1
                    v_w = value_loss_weight

                if p_w == 0.0 and v_z is None:
                    # Loser-side pair not selected as a value state:
                    # nothing to learn from -- skip before encoding.
                    continue

                if skip_left > 0:
                    # Trained before the cut this run resumes: its draws
                    # above are replayed, the pair is not trained again.
                    if use_batched:
                        try:
                            _raw_one(encoder, state_or_raw)
                        except Exception:
                            continue
                    skip_left -= 1
                    if skip_left == 0:
                        _log_pass_reentry(position, t_epoch)
                    continue

                if use_batched:
                    # === Batched flow: accumulate B, then forward_batch.
                    _te = time.perf_counter() if prof_on else 0.0
                    try:
                        raw = _raw_one(encoder, state_or_raw)
                    except Exception as e:
                        log.debug(f"  encode failed: {e}")
                        continue
                    finally:
                        if prof_on:
                            prof_acc["encode"] += time.perf_counter() - _te
                            prof_acc["pairs"] += 1
                    batch_raws.append(raw)
                    batch_ais.append(ai)
                    batch_zw.append((v_z, v_w, p_w))
                    if len(batch_raws) < batch_size:
                        continue
                    _tf = time.perf_counter() if prof_on else 0.0
                    try:
                        oom_halvings += _flush_batch(
                            model, encoder, batch_raws, batch_ais,
                            batch_zw,
                            opt, params_for_clip, batch_size, device,
                            running_loss,
                            running_loss_actor,
                            running_loss_type,
                            running_loss_target,
                            running_loss_weapon,
                            running_loss_value,
                            type_loss_weights=type_loss_weights,
                            autocast_dtype=autocast_dtype,
                        )
                    except Exception as e:
                        flush_failures += 1
                        log.warning(f"  batch flush failed ({len(batch_raws)} pairs lost, "
                                    f"{flush_failures} so far): {e!r}"[:400])
                        opt.zero_grad()
                        batch_raws.clear()
                        batch_ais.clear()
                        batch_zw.clear()
                        continue
                    finally:
                        if prof_on:
                            prof_acc["flush"] += time.perf_counter() - _tf
                    running_count += len(batch_raws)
                    batch_raws.clear()
                    batch_ais.clear()
                    batch_zw.clear()
                    step_just_landed = True
                else:
                    # === Per-pair flow: forward+backward per pair, step
                    # at every batch_size accumulation. Lower memory
                    # peak; fewer kernel-launch savings, but on CPU the
                    # batched path's padding-to-max-seq waste is far
                    # worse so per-pair wins.
                    # (per-pair mode: encode happens inside the call,
                    # so `flush` here covers encode+forward+backward)
                    _tf = time.perf_counter() if prof_on else 0.0
                    try:
                        parts = _loss_parts_for_pair(
                            encoder, model, state_or_raw, ai, device,
                            type_loss_weights=type_loss_weights,
                            value_z=v_z, value_weight=v_w,
                            policy_weight=p_w,
                        )
                    except Exception as e:
                        log.debug(f"  loss compute failed: {e}")
                        continue
                    finally:
                        if prof_on:
                            prof_acc["flush"] += time.perf_counter() - _tf
                            prof_acc["pairs"] += 1
                    (parts.total / batch_size).backward()
                    if parts.actor_fired:
                        # 5 .item() calls per pair on CPU is fine — the
                        # per-pair flow only runs on CPU, no GPU sync.
                        if parts.value_fired:
                            running_loss_value.append(float(parts.value.item()))
                        if parts.policy_w > 0.0:
                            running_loss.append(float(parts.total.item()))
                            running_loss_actor.append(float(parts.actor.item()))
                            if parts.type_fired:
                                running_loss_type.append(float(parts.type.item()))
                            if parts.target_fired:
                                running_loss_target.append(float(parts.target.item()))
                            if parts.weapon_fired:
                                running_loss_weapon.append(float(parts.weapon.item()))
                    running_count += 1
                    losses_in_batch += 1
                    if losses_in_batch < batch_size:
                        continue
                    torch.nn.utils.clip_grad_norm_(params_for_clip, 1.0)
                    opt.step()
                    opt.zero_grad()
                    losses_in_batch = 0
                    step_just_landed = True

                # Common bookkeeping post-step.
                if step_just_landed:
                    step += 1
                    global_step += 1
                    if step % log_every == 0:
                        def _avg(d):
                            return sum(d) / len(d) if d else float("nan")
                        avg        = _avg(running_loss)
                        avg_actor  = _avg(running_loss_actor)
                        avg_type   = _avg(running_loss_type)
                        avg_target = _avg(running_loss_target)
                        avg_weapon = _avg(running_loss_weapon)
                        elapsed = time.time() - t_epoch
                        # Rate is per-epoch: pairs trained THIS EPOCH
                        # divided by elapsed THIS EPOCH. Cumulative
                        # `running_count` is reported separately as
                        # `pairs=` so the user still sees total
                        # progress.
                        rate = (running_count - epoch_start_count) / max(1e-9, elapsed)
                        total_elapsed = time.time() - t_start
                        eta_pairs = (max_pairs - running_count) if max_pairs else None
                        eta = f"{eta_pairs/rate/60:.1f}m" if eta_pairs and rate > 0 else "?"
                        # Per-head breakdown lets us see e.g. that
                        # actor loss is converging while target loss
                        # stays near ln(num_hexes) — i.e. the model
                        # learned WHAT to do but not WHERE. (Either
                        # head can be NaN early on if no pair has
                        # fired it yet.)
                        prof_str = ""
                        if prof_on:
                            tot = max(1e-9, sum(
                                prof_acc[k] for k in
                                ("wait", "encode", "flush")))
                            prof_str = (
                                f" prof[wait={prof_acc['wait']/tot:.0%}"
                                f" enc={prof_acc['encode']/tot:.0%}"
                                f" flush={prof_acc['flush']/tot:.0%}]")
                        log.info(
                            f"  epoch={epoch} step={step} "
                            f"avg_loss={avg:.3f} "
                            f"(actor={avg_actor:.3f} type={avg_type:.3f} "
                            f"target={avg_target:.3f} weapon={avg_weapon:.3f} "
                            f"value={_avg(running_loss_value):.3f}) "
                            f"pairs={running_count} rate={rate:.1f}/s "
                            f"wall={total_elapsed/60:.1f}m eta={eta}"
                            f"{prof_str}"
                        )
                    if global_step % ckpt_every == 0:
                        # Mid-epoch periodic checkpoint: save the
                        # GLOBAL completed-epoch count (= `epoch`,
                        # since this epoch hasn't finished yet) and
                        # the pass position a resume continues from.
                        _save_checkpoint(
                            checkpoint_out, model, encoder, opt,
                            global_step, running_count, epoch=epoch,
                            resume_state=_resume_state(epoch, epoch_rng_state,
                                                       epoch_start_pairs, epoch_start_step,
                                                       last_eval_pairs, seed),
                            **save_kwargs,
                        )
                        log.info(f"  periodic checkpoint @ step={global_step}")
                    if (eval_every and holdout_files
                            and running_count - last_eval_pairs
                            >= eval_every):
                        last_eval_pairs = running_count
                        stats = _evaluate(
                            model, encoder, holdout_files, device,
                            eval_pairs=eval_pairs,
                            eval_pairs_per_game=eval_pairs_per_game,
                            eval_sample_seed=eval_sample_seed,
                            type_loss_weights=type_loss_weights,
                            winner_map=winner_map, cache=eval_cache)
                        stats["train_target_off_subset"] = target_off_subset
                        _log_eval(stats, epoch, global_step,
                                  running_count, checkpoint_out)
                        if prof_on:
                            prof_path.write_text(json.dumps({
                                "pairs": running_count,
                                "wall_s": time.time() - t_start,
                                **{k: round(v, 2) for k, v in
                                   prof_acc.items()},
                            }), encoding="utf-8")
                    if (max_pairs
                            and running_count - run_start_count
                            >= max_pairs):
                        log.info(f"Reached max_pairs={max_pairs}; stopping.")
                        stop = True
                        break
        finally:
            # Make sure the parallel stream's worker pool is torn down
            # cleanly even if we break out early (max_pairs hit, KbInt,
            # exception in the inner loop).
            close = getattr(stream, "close", None)
            if close is not None:
                close()

        # Flush the residual batch at end-of-epoch so we don't lose
        # the tail (up to bs-1 pairs). Skipped on `stop=True` paths
        # (max_pairs hit, KbInt) where the user wanted to stop *now*.
        if not stop:
            if use_batched and batch_raws:
                try:
                    oom_halvings += _flush_batch(
                        model, encoder, batch_raws, batch_ais,
                        batch_zw,
                        opt, params_for_clip, batch_size, device,
                        running_loss,
                        running_loss_actor,
                        running_loss_type,
                        running_loss_target,
                        running_loss_weapon,
                        running_loss_value,
                        type_loss_weights=type_loss_weights,
                        autocast_dtype=autocast_dtype,
                    )
                    running_count += len(batch_raws)
                except Exception as e:
                    flush_failures += 1
                    log.warning(f"  end-of-epoch flush failed ({len(batch_raws)} pairs lost): {e!r}"[:400])
                    opt.zero_grad()
                batch_raws.clear()
                batch_ais.clear()
                batch_zw.clear()
            elif not use_batched and losses_in_batch > 0:
                torch.nn.utils.clip_grad_norm_(params_for_clip, 1.0)
                opt.step()
                opt.zero_grad()
                losses_in_batch = 0

        # Save after each epoch — to BOTH the canonical path (for
        # easy resumption) AND a per-epoch numbered snapshot so the
        # user can grab a stable copy of e.g. epoch-1 weights for
        # in-situ evaluation while training continues into epoch-2.
        # The per-epoch file uses the canonical path's stem with
        # `_epochN` appended (e.g. supervised.pt → supervised_epoch1.pt).
        # `epoch + 1` is the count of fully-completed epochs after
        # this save (the loop just finished epoch `epoch`). Resume
        # will read this back as `resumed_epoch` and start the next
        # link at `range(resumed_epoch, epochs)`. A max_pairs cut
        # mid-epoch did NOT complete it (project round-1 C15:
        # recording epoch+1 made the resume skip the rest of the
        # corpus).
        if skip_left > 0:
            log.error(f"epoch {epoch} ended with {skip_left} pairs of the resumed pass "
                      f"still to skip: the corpus is not the one the cut run read")
        completed = epoch if stop else epoch + 1
        resume_state = _resume_state(epoch, epoch_rng_state, epoch_start_pairs,
                                     epoch_start_step, last_eval_pairs, seed)
        if stop:
            log.info(f"max_pairs cut mid-epoch {epoch}; checkpoint "
                     f"records epoch={completed} (NOT completed; "
                     f"resume redoes it)")
        _save_checkpoint(checkpoint_out, model, encoder, opt,
                         global_step, running_count, epoch=completed,
                         resume_state=resume_state, **save_kwargs)
        epoch_path = checkpoint_out.with_name(
            f"{checkpoint_out.stem}_epoch{epoch}{checkpoint_out.suffix}"
        )
        _save_checkpoint(epoch_path, model, encoder, opt,
                         global_step, running_count, epoch=completed,
                         resume_state=resume_state, **save_kwargs)
        log.info(f"Epoch {epoch} saved to {checkpoint_out} and {epoch_path.name}")
        # Accounting line: an epoch that "completes" with a large
        # error count or far fewer pairs than the corpus holds is a
        # broken run, not a fast one (2026-08-08 random-arm underrun).
        log.info(f"  epoch accounting: files_seen={files_seen} "
                 f"file_errors={file_errors} "
                 f"pairs={running_count - run_start_count} "
                 f"(chain total {running_count}) "
                 f"target_off_subset={target_off_subset} "
                 f"flush_failures={flush_failures} oom_halvings={oom_halvings}")
        if holdout_files:
            stats = _evaluate(model, encoder, holdout_files, device,
                              eval_pairs=eval_pairs,
                              eval_pairs_per_game=eval_pairs_per_game,
                              eval_sample_seed=eval_sample_seed,
                              type_loss_weights=type_loss_weights,
                              winner_map=winner_map, cache=eval_cache)
            stats["train_target_off_subset"] = target_off_subset
            _log_eval(stats, epoch, global_step, running_count,
                      checkpoint_out, tag=f"epoch{epoch}-end")
        # Advance the LR scheduler one cosine step. Done AFTER the
        # save so the saved optimizer state still has the lr that
        # produced this epoch's gradients (recoverable for analysis).
        lr_scheduler.step()
        log.info(f"  next-epoch lr = {opt.param_groups[0]['lr']:.2e}")


def main(argv: List[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset_dir", type=Path)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("training/checkpoints/supervised.pt"),
                    help="Output checkpoint path (also the periodic-save target).")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", dest="batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-replays", type=int, default=0,
                    help="Cap replay-file count (0 = all competitive-2p).")
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="Cap total training pairs per run (0 = no cap).")
    ap.add_argument("--ckpt-every", type=int, default=2000,
                    help="Checkpoint every N gradient steps.")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--all-scenarios", action="store_true",
                    help="Skip the competitive-2p scenario filter.")
    ap.add_argument("--resume", type=Path, default=None,
                    help="Checkpoint to resume from (model + encoder + "
                         "optimizer). New shuffled file order — we don't "
                         "rewind to the exact replay we left off on.")
    ap.add_argument("--max-replay-commands", type=int, default=1500,
                    help="Skip replays with > this many commands "
                         "(catches the 4 corpus outliers at 2000+ that "
                         "drove RAM use to 65%% on the first overnight).")
    ap.add_argument("--max-starting-units", type=int, default=0,
                    help="Skip replays with > this many starting units "
                         "(0 = no cap; some legit maps like Thousand "
                         "Stings Garrison ship with ~24 statues for "
                         "recruit-hex mechanics, so this defaults off).")
    ap.add_argument("--max-pairs-per-replay", type=int, default=0,
                    help="Bail mid-replay after this many pairs "
                         "(0 = no cap; off by default to weight every "
                         "replay equally).")
    ap.add_argument("--gc-every-files", type=int, default=16,
                    help="gc.collect() after this many replay files.")
    ap.add_argument("--workers", type=int, default=0,
                    help="Encoder worker processes (0 = serial). "
                         "Each worker prefetches encode_raw of replay "
                         "pairs in parallel. On the cluster's 8-CPU "
                         "L40S node, --workers 6 leaves 2 cores for "
                         "the main thread + os and roughly doubles "
                         "throughput. Out-of-vocab unit names hit the "
                         "overflow bucket — for fresh runs we pre-seed "
                         "vocab from unit_stats.json automatically.")
    ap.add_argument("--prefetch-factor", type=int, default=4,
                    help="Output-queue depth target per worker.")
    ap.add_argument("--batched-forward", choices=("auto", "on", "off"),
                    default="auto",
                    help="Use model.forward_batch (one padded transformer "
                         "pass over `bs` pairs) instead of per-pair forward. "
                         "Big win on GPU; regresses on CPU because of "
                         "padding-to-max-seq waste. Default 'auto' is "
                         "on iff device != cpu.")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=256)
    ap.add_argument("--value-loss-weight", type=float, default=1.0,
                    help="lambda_v applied to SELECTED value states "
                         "(subsampled at ~value-states-per-game per "
                         "game, uniform weight). 0 = legacy "
                         "policy-only (the value-frozen mode that let "
                         "the trunk drift under the head).")
    ap.add_argument("--value-states-per-game", type=int, default=16,
                    help="Expected number of states per game that "
                         "carry the value loss each epoch (the "
                         "streaming form of AlphaGo's "
                         "one-position-per-game decorrelation).")
    ap.add_argument("--holdout-games", type=int, default=300,
                    help="Games held out of training (seed-0 split); "
                         "held-out top-1 accuracy is the SL "
                         "acceptance metric. 0 disables.")
    ap.add_argument("--eval-every", type=int, default=50_000,
                    help="Pairs between held-out evals (0 = epoch "
                         "ends only).")
    ap.add_argument("--eval-pairs", type=int, default=1200)
    ap.add_argument("--eval-sample-seed", type=int, default=None,
                    help="Seeded RANDOM per-game pair sample instead "
                         "of first-N (independent probe redraws).")
    ap.add_argument("--eval-pairs-per-game", type=int, default=0,
                    help="Stratified probe (2026-08-25 instrument "
                         "repair): cap pairs per holdout game so "
                         "--eval-pairs spans many games; per-game "
                         "AUC/CE with BETWEEN-game SE. 0 = legacy "
                         "pooled behavior.")
    ap.add_argument("--eval-only", action="store_true",
                    help="Evaluate the --resume checkpoint on the "
                         "holdout split and exit (baseline mode).")
    ap.add_argument("--eval-json", type=Path, default=None,
                    help="With --eval-only: also write the stats dict "
                         "as JSON here (machine-readable; the "
                         "campaign holdout-probe loop parses it).")
    ap.add_argument("--fog-hides-enemy-villages", action="store_true", default=None,
                    help="Global feature 5 under fog counts only the enemy "
                         "villages the mover can see (the engine never shows "
                         "the true count). Default: on for a fresh network, a "
                         "checkpoint's own setting on a warm start; rides the "
                         "checkpoint.")
    ap.add_argument("--no-fog-hides-enemy-villages", action="store_false",
                    dest="fog_hides_enemy_villages",
                    help="Feed the true enemy village count (the pre-2026-09-08 encoding).")
    ap.add_argument("--terrain-multi-hot", action="store_true", default=None,
                    help="Each hex carries its full terrain set from the engine's "
                         "aliases (a forested hill is HILLS and FOREST), embedded as "
                         "a multi-hot over the terrain table. Default: on for a fresh "
                         "network, a checkpoint's own setting on a warm start; rides "
                         "the checkpoint. --eval-only must match the checkpoint.")
    ap.add_argument("--no-terrain-multi-hot", action="store_false", dest="terrain_multi_hot",
                    help="One terrain class per hex (the pre-2026-09-19 encoding).")
    ap.add_argument("--tf32", action="store_true",
                    help="Run this run's fp32 matmuls on the tensor cores (TF32). A "
                         "recipe change (10-bit matmul inputs; the holdout probe runs "
                         "under it too), so off by default and one factor of its own. "
                         "The sim, the corpus sweeps and eval are never touched.")
    ap.add_argument("--fused-adamw", action="store_true",
                    help="The fused AdamW kernel (one launch per step instead of one "
                         "per tensor). Sums in another order than the reference step: "
                         "off by default, one factor of its own.")
    ap.add_argument("--bf16", action="store_true",
                    help="Run the batched flow's forward and loss under bf16 autocast "
                         "(fp32 master weights; the eval path already serves bf16).")
    ap.add_argument("--preencoded", type=Path, default=None,
                    help="Read pairs from a pre-encoded corpus "
                         "(tools/preencode_corpus.py) instead of replaying "
                         "the games; the vocab and hex basis must match.")
    ap.add_argument("--value-material", action="store_true",
                    help="Build the value head with material (cost x HP "
                         "fraction, mover minus visible enemies) as an extra "
                         "input; a warm start grafts it zero-initialized.")
    ap.add_argument("--reinit-value-head", action="store_true",
                    help="Drop value_head.* from the --resume state "
                         "and skip optimizer-state restore: warm "
                         "trunk+policy, fresh value head (imitation "
                         "A/B verdict 2026-08-08).")
    ap.add_argument("--imitation-config", type=Path, default=None,
                    help="configs/imitation.json — enables imitation "
                         "mode: winners-only policy loss, per-game "
                         "equal weighting, manifest-driven holdout. "
                         "Needs manifest.jsonl in DATASET_DIR (from "
                         "tools/build_imitation_dataset.py).")
    ap.add_argument("--action-type-weights", type=Path, default=None,
                    help="Path to action-type loss-weight JSON "
                         "(see tools/compute_action_type_weights.py). "
                         "Default: bake-in inverse-frequency weights "
                         "(_DEFAULT_ACTION_TYPE_LOSS_WEIGHT).")
    ap.add_argument("--relevant-set-hexes", action="store_true",
                    help="Encode and label in the relevant hex subset "
                         "(encoder relevant_set_hexes; docs/"
                         "model_cost_study_20260905.md 2). Recorded "
                         "in the checkpoint so evals build the same "
                         "basis. --eval-only must match the checkpoint.")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="Warm start: model + encoder weights and vocab "
                         "from this checkpoint, fresh optimizer and "
                         "counters (unlike --resume, which continues "
                         "the chain). Exclusive with --resume.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Seed for file order, value-state subsampling "
                         "and torch: two runs with the same seed and "
                         "corpus train on the same pair stream, at any "
                         "--workers count (the parallel stream delivers "
                         "files in dispatch order).")
    args = ap.parse_args(argv[1:])
    bf_arg = (None if args.batched_forward == "auto"
              else args.batched_forward == "on")

    type_loss_weights = _load_action_type_weights(args.action_type_weights)
    log.info(f"action-type loss weights: {type_loss_weights}")

    train(
        dataset_dir=args.dataset_dir,
        checkpoint_out=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_replays=args.max_replays,
        max_pairs=args.max_pairs,
        ckpt_every=args.ckpt_every,
        log_every=args.log_every,
        gc_every_files=args.gc_every_files,
        max_replay_commands=args.max_replay_commands,
        max_starting_units=args.max_starting_units,
        max_pairs_per_replay=args.max_pairs_per_replay,
        device_str=args.device,
        competitive_only=not args.all_scenarios,
        resume=args.resume,
        workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        batched_forward=bf_arg,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        holdout_games=args.holdout_games,
        value_loss_weight=args.value_loss_weight,
        value_states_per_game=args.value_states_per_game,
        eval_every=args.eval_every,
        eval_pairs=args.eval_pairs,
        eval_pairs_per_game=args.eval_pairs_per_game,
        eval_sample_seed=args.eval_sample_seed,
        eval_only=args.eval_only,
        eval_json=args.eval_json,
        reinit_value_head=args.reinit_value_head,
        value_material=args.value_material,
        preencoded=args.preencoded,
        bf16=args.bf16,
        tf32=args.tf32,
        fused_adamw=args.fused_adamw,
        fog_hides_enemy_villages=args.fog_hides_enemy_villages,
        terrain_multi_hot=args.terrain_multi_hot,
        imitation_config=args.imitation_config,
        type_loss_weights=type_loss_weights,
        relevant_set_hexes=args.relevant_set_hexes,
        init_from=args.init_from,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
