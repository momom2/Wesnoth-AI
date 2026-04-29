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

Simplicity first: no DataLoader, no workers. Iterate replay files
sequentially, yield pairs, batch by count. If training gets slow we
can add multi-worker prefetch. Current rate estimate: with ~2000 pairs
per replay × ~10 replays/sec encoding-only → ~20K pairs/sec, so a
10M-pair corpus is one overnight run at batch=8 on CPU.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import logging
import multiprocessing as mp
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Project imports — assume cwd is the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classes import GameState
from encoder import GameStateEncoder, RawEncoded, encode_raw
from model import WesnothModel
# Import replay_dataset from the same tools/ dir.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_dataset import ActionIndices, filter_competitive_2p, iter_replay_pairs
from encode_worker import worker_main as _encode_worker_main


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
            n_dropped_cmds += 1; continue
        if max_starting:
            # Need to open for starting_units. Cheap (~5KB compressed).
            try:
                with gzip.open(p, "rt", encoding="utf-8") as f:
                    data = json.load(f)
                if len(data.get("starting_units", [])) > max_starting:
                    n_dropped_units += 1; continue
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
) -> None:
    """Atomic-ish checkpoint write: save to .tmp then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "arch": {"d_model": 128, "num_layers": 3,
                 "num_heads": 4, "d_ff": 256},
        "model_state":     model.state_dict(),
        "encoder_state":   encoder.state_dict(),
        "unit_type_to_id": dict(encoder.unit_type_to_id),
        "faction_to_id":   dict(encoder.faction_to_id),
        "optimizer_state": opt.state_dict(),
        "supervised_step":  step,
        "supervised_pairs": pairs,
    }, tmp)
    import os
    os.replace(tmp, path)


def _seed_vocab_from_unit_stats(
    encoder: GameStateEncoder,
    unit_stats_path: Path,
) -> None:
    """Pre-seed the encoder's vocab from `unit_stats.json`.

    Called before workers spawn (when --workers > 0 and we're not
    resuming). Workers receive the seeded dict at startup; out-of-vocab
    names later still hit the overflow bucket via `_lookup_id` clamp,
    but the named-row hit rate is much higher.

    No-op if `unit_stats.json` is missing — workers will fall back to
    overflow more often, which only hurts rare unit types.
    """
    if not unit_stats_path.exists():
        log.warning(
            f"  vocab seed skipped: {unit_stats_path} not found"
        )
        return
    try:
        with unit_stats_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.warning(f"  vocab seed skipped: {e}")
        return
    type_to_id = encoder.unit_type_to_id
    # Stable ordering — use sorted keys so two different runs starting
    # from the same unit_stats.json produce the same id assignment.
    for name in sorted(data.get("units", {}).keys()):
        if name not in type_to_id:
            type_to_id[name] = len(type_to_id)


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
):
    """Single-process pair stream — reads + encodes inline."""
    for gz in files:
        n = 0
        try:
            for state, ai in iter_replay_pairs(gz):
                if max_pairs_per_replay and n >= max_pairs_per_replay:
                    break
                n += 1
                yield ("pair", state, ai, gz.name)
            yield ("file_done", gz.name, n)
        except Exception as e:
            yield ("file_error", gz.name, repr(e))


class _ParallelStream:
    """Wraps the worker pool + producer-consumer queues as an iterable.

    Spawns N worker processes that each pop a `Path` from the input
    queue, run `iter_replay_pairs` + `encode_raw` for the entire
    replay, and push a single ("file", pairs, gz_name) message — the
    pairs list is the whole replay. The main thread keeps the input
    queue topped up: a fresh file goes in for every consumed message,
    and once all files are dispatched, N sentinels (None) retire the
    workers.

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
    ):
        self._files = list(files)
        self._workers_n = workers
        self._closed = False

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

        self._procs = []
        for _ in range(workers):
            p = self._ctx.Process(
                target=_encode_worker_main,
                args=(self._in_q, self._out_q,
                      dict(type_to_id), dict(faction_to_id)),
                daemon=True,
            )
            p.start()
            self._procs.append(p)

        # Prime the input queue with up to `workers * 2` files. The
        # rest are fed lazily, one per consumed message.
        self._next_file = 0
        self._workers_alive = workers
        for _ in range(min(workers * 2, len(self._files))):
            self._in_q.put(self._files[self._next_file])
            self._next_file += 1

        # Per-file unpack state: when a "file" message arrives we
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

        # 3. Otherwise pull the next message from workers.
        while True:
            if self._workers_alive == 0:
                self.close()
                raise StopIteration
            item = self._out_q.get()
            tag = item[0]

            if tag == "file":
                _, pairs, gz_name = item
                # Refill the input queue: send one more file if any
                # remain, else a sentinel for one worker.
                self._refill_input()
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

            if tag == "file_error":
                _, gz_name, err = item
                self._refill_input()
                return ("file_error", gz_name, err)

            if tag == "worker_exit":
                self._workers_alive -= 1
                continue

            # Unknown tag — should never happen. Log and keep pulling.
            log.warning(f"  unknown stream event {tag!r}; ignoring")

    def _refill_input(self) -> None:
        """Top up the input queue when a worker delivers a file.

        Either feeds one more replay path or, if we've run out of
        files, sends one None sentinel so a worker can retire.
        """
        if self._next_file < len(self._files):
            self._in_q.put(self._files[self._next_file])
            self._next_file += 1
        else:
            self._in_q.put(None)

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
):
    """Multi-process pair stream — encode_raw runs in worker processes."""
    return _ParallelStream(
        files,
        workers=workers,
        type_to_id=type_to_id,
        faction_to_id=faction_to_id,
        prefetch_factor=prefetch_factor,
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
    weapon:       torch.Tensor   # scalar, grad-tracking (or zero sentinel)
    actor_fired:  bool
    type_fired:   bool
    target_fired: bool
    weapon_fired: bool


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
    from model import UnitActionType

    zero = torch.zeros((), device=device)
    actor_logits = output.actor_logits        # [1, A]
    A = actor_logits.size(1)
    if ai.actor_idx >= A:
        # Pathological: observed actor isn't in the model's output. The
        # whole pair contributes nothing — total=0, no head fired.
        return LossParts(zero, zero, zero, zero, zero,
                         False, False, False, False)

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

    total = actor_loss + type_loss + target_loss + weapon_loss
    # Per-head fields report the raw CE (no action-type weight, no
    # smoothing scale baked in) so the running-average log line stays
    # interpretable -- "actor=2.5" means CE=2.5 even if the actor head
    # is internally scaled 5x for recruits.
    return LossParts(total, actor_loss_raw, type_loss_raw,
                     target_loss, weapon_loss,
                     True, type_fired, target_fired, weapon_fired)


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
) -> LossParts:
    """Per-pair forward + per-head loss breakdown."""
    encoded = _encode_one(encoder, state_or_raw, device)
    output = model(encoded)
    return _loss_parts_for_output(
        output, ai, device, type_loss_weights=type_loss_weights)


def _flush_batch(
    model:           WesnothModel,
    encoder:         GameStateEncoder,
    batch_encoded:   List,
    batch_ais:       List[ActionIndices],
    opt:             torch.optim.Optimizer,
    params_for_clip: List[torch.nn.Parameter],
    batch_size:      int,
    device:          torch.device,
    running_loss:    deque,
    running_loss_actor:  deque,
    running_loss_type:   deque,
    running_loss_target: deque,
    running_loss_weapon: deque,
    type_loss_weights: Optional[Dict[str, float]] = None,
) -> None:
    """One batched forward + summed-loss backward + opt step.

    `batch_size` is the *target* batch size used as the loss-scaling
    denominator. We pass it (rather than `len(batch_encoded)`) so that
    a partial flush at a file boundary scales the gradient the same
    way as a full batch — this matches the per-pair flow's behavior
    of dividing each individual loss by `batch_size` and reproduces
    the same effective learning rate per pair, regardless of where
    file boundaries fell.

    Per-sample losses (total + per-head) are appended to the
    `running_loss*` deques for the progress log. We pull them after
    backward via a single `.detach().cpu().tolist()` to avoid a sync
    per pair. Per-head averages are taken over fired pairs only — see
    LossParts docstring.
    """
    if not batch_encoded:
        return

    # Single padded transformer pass over B samples.
    outputs = model.forward_batch(batch_encoded)

    parts_list = [
        _loss_parts_for_output(out, ai, device,
                               type_loss_weights=type_loss_weights)
        for out, ai in zip(outputs, batch_ais)
    ]

    # Stack each head separately so we can both backprop through the
    # sum AND retrieve per-head per-sample values for logging in one
    # post-backward sync.
    actor_stack  = torch.stack([p.actor  for p in parts_list])  # [B]
    type_stack   = torch.stack([p.type   for p in parts_list])  # [B]
    target_stack = torch.stack([p.target for p in parts_list])  # [B]
    weapon_stack = torch.stack([p.weapon for p in parts_list])  # [B]

    # backward() through the same scalar that the per-pair path
    # produces: sum of all heads divided by batch_size. Linearity
    # guarantees identical gradients to B individual
    # `(part.total / bs).backward()` calls.
    total_loss = (actor_stack.sum() + type_stack.sum()
                  + target_stack.sum()
                  + weapon_stack.sum()) / batch_size
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(params_for_clip, 1.0)
    opt.step()
    opt.zero_grad()

    # One sync per head (four small CPU transfers) — vs 4*B if we
    # called .item() per pair per head. Pre-cast to lists once.
    actor_floats  = actor_stack.detach().cpu().tolist()
    type_floats   = type_stack.detach().cpu().tolist()
    target_floats = target_stack.detach().cpu().tolist()
    weapon_floats = weapon_stack.detach().cpu().tolist()
    for i, p in enumerate(parts_list):
        if not p.actor_fired:
            continue   # actor_idx out of range — pair contributed 0 to grad
        a, ty, t, w = (actor_floats[i], type_floats[i],
                       target_floats[i], weapon_floats[i])
        running_loss.append(a + ty + t + w)   # total = sum of heads
        running_loss_actor.append(a)
        if p.type_fired:
            running_loss_type.append(ty)
        if p.target_fired:
            running_loss_target.append(t)
        if p.weapon_fired:
            running_loss_weapon.append(w)


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
    type_loss_weights: Optional[Dict[str, float]] = None,
        # per-action-type loss weights; None -> defaults
        # (_DEFAULT_ACTION_TYPE_LOSS_WEIGHT). Pass via --action-type-weights
        # JSON; see tools/compute_action_type_weights.py.
) -> None:
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

    encoder = GameStateEncoder(d_model=128).to(device)
    model   = WesnothModel(d_model=128, num_layers=3,
                           num_heads=4, d_ff=256).to(device)
    model.train(); encoder.train()

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(encoder.parameters()),
        lr=lr, weight_decay=1e-4,
    )

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
    # a previous checkpoint. Loop continues with a fresh shuffled
    # file order — we don't try to resume mid-replay precisely, since
    # behavior cloning doesn't require it.
    resumed_step = 0
    resumed_pairs = 0
    if resume is not None and resume.exists():
        log.info(f"Resuming from {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        # strict=False so an architecture-additive change (new head,
        # new embedding column) can warm-start from a prior
        # checkpoint without losing the heads that DID exist.
        # Concrete case: C.1 added model.type_head; the cluster's
        # epoch-3 checkpoint predates it. We log the deltas so a
        # silent vocab/encoder regression can't sneak through
        # disguised as "ah, that's just the new head".
        m_missing, m_unexpected = model.load_state_dict(
            ckpt["model_state"], strict=False)
        if m_missing:
            log.warning(f"  model: {len(m_missing)} missing key(s) "
                        f"(will train from random init): {m_missing}")
        if m_unexpected:
            log.warning(f"  model: {len(m_unexpected)} unexpected key(s) "
                        f"in checkpoint (ignored): {m_unexpected}")
        e_missing, e_unexpected = encoder.load_state_dict(
            ckpt["encoder_state"], strict=False)
        if e_missing:
            log.warning(f"  encoder: {len(e_missing)} missing key(s) "
                        f"(will train from random init): {e_missing}")
        if e_unexpected:
            log.warning(f"  encoder: {len(e_unexpected)} unexpected key(s) "
                        f"in checkpoint (ignored): {e_unexpected}")
        encoder.unit_type_to_id = dict(ckpt.get("unit_type_to_id", {}))
        if "faction_to_id" in ckpt:
            encoder.faction_to_id = dict(ckpt["faction_to_id"])
        if "optimizer_state" in ckpt:
            try:
                opt.load_state_dict(ckpt["optimizer_state"])
            except Exception as e:
                log.warning(f"  optimizer state restore failed ({e}); "
                            f"re-accumulating momentum from scratch")
        resumed_step  = int(ckpt.get("supervised_step", 0))
        resumed_pairs = int(ckpt.get("supervised_pairs", 0))
        log.info(f"  resumed at step={resumed_step} pairs={resumed_pairs}")
        # Fast-forward the LR scheduler past the epochs already
        # completed. We don't checkpoint the scheduler's state, so
        # we infer "epochs done" from the saved per-epoch snapshots.
        # Best-effort: if no `supervised_epoch{N}.pt` exists alongside
        # the resumed file, assume 0 epochs done and start at the
        # initial lr. Either way, scheduler.step() at end of each
        # epoch will continue from there.
        completed_epochs = 0
        for n in range(epochs):
            sibling = resume.with_name(
                f"{resume.stem.replace('_epoch' + str(n), '')}_epoch{n}{resume.suffix}"
            )
            # Resume target is usually `supervised.pt` (rolling), and
            # per-epoch snapshots live alongside as supervised_epoch0.pt
            # etc. Probe for the highest existing epoch snapshot.
            cand = resume.parent / f"supervised_epoch{n}.pt"
            if cand.exists():
                completed_epochs = n + 1
        for _ in range(completed_epochs):
            lr_scheduler.step()
        log.info(f"  LR scheduler advanced {completed_epochs} epochs; "
                 f"current lr = {opt.param_groups[0]['lr']:.2e}")

    # Competitive-2p filter: reads index.jsonl entries and only keeps
    # replays on ships-with ladder maps with two default-faction sides.
    # Drops ~97% of the raw dataset (most replays are FFA / multi-side).
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

    if max_replays:
        files = files[:max_replays]
    log.info(f"Training on {len(files)} replay files")

    # Pre-seed the encoder vocab from observed names BEFORE workers
    # spawn (when --workers > 0). Workers receive a snapshot at startup
    # and never grow it afterwards: out-of-vocab names hit the overflow
    # bucket. With a fresh encoder, this scan touches the unit names
    # in `unit_stats.json` so common types (Drake Burner, Loyalist,
    # etc.) hit named rows instead of the overflow bucket. Skip if
    # we resumed (the resumed dict already has whatever the previous
    # runs accumulated).
    if workers > 0 and resume is None:
        _seed_vocab_from_unit_stats(encoder, dataset_dir.parent / "unit_stats.json")
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
    # running_count is THIS RUN's pair count (drives max_pairs cap and
    # rate). cumulative_pairs adds the resumed-from total for the
    # progress-display + checkpoint save.
    running_count = 0
    global_step = resumed_step
    t_start = time.time()
    stop = False
    files_seen = 0

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

    for epoch in range(epochs):
        if stop: break
        random.shuffle(files)
        step = 0
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
        if workers > 0:
            stream = _pair_stream_parallel(
                files,
                workers=workers,
                type_to_id=encoder.unit_type_to_id,
                faction_to_id=encoder.faction_to_id,
                prefetch_factor=prefetch_factor,
                max_pairs_per_replay=max_pairs_per_replay,
            )
        else:
            stream = _pair_stream_serial(
                files,
                max_pairs_per_replay=max_pairs_per_replay,
            )

        # Per-pair / batched: shared bookkeeping below; the differences
        # are concentrated in the "pair" event handler.
        batch_encoded: List = []
        batch_ais: List[ActionIndices] = []
        losses_in_batch = 0  # used by per-pair flow
        params_for_clip = list(model.parameters()) + list(encoder.parameters())
        opt.zero_grad()
        try:
            for event in stream:
                if stop:
                    break
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
                    _, gz_name, err = event
                    log.debug(f"  skip {gz_name}: {err}")
                    # Abandon any partial gradient or batch from this
                    # file — its data is incomplete.
                    opt.zero_grad()
                    batch_encoded.clear(); batch_ais.clear()
                    losses_in_batch = 0
                    continue

                # kind == "pair"
                _, state_or_raw, ai, _gz_name = event

                if use_batched:
                    # === Batched flow: accumulate B, then forward_batch.
                    try:
                        encoded = _encode_one(encoder, state_or_raw, device)
                    except Exception as e:
                        log.debug(f"  encode failed: {e}")
                        continue
                    batch_encoded.append(encoded)
                    batch_ais.append(ai)
                    if len(batch_encoded) < batch_size:
                        continue
                    try:
                        _flush_batch(
                            model, encoder, batch_encoded, batch_ais,
                            opt, params_for_clip, batch_size, device,
                            running_loss,
                            running_loss_actor,
                            running_loss_type,
                            running_loss_target,
                            running_loss_weapon,
                            type_loss_weights=type_loss_weights,
                        )
                    except Exception as e:
                        log.debug(f"  batch flush failed: {e}")
                        opt.zero_grad()
                        batch_encoded.clear(); batch_ais.clear()
                        continue
                    running_count += len(batch_encoded)
                    batch_encoded.clear(); batch_ais.clear()
                    step_just_landed = True
                else:
                    # === Per-pair flow: forward+backward per pair, step
                    # at every batch_size accumulation. Lower memory
                    # peak; fewer kernel-launch savings, but on CPU the
                    # batched path's padding-to-max-seq waste is far
                    # worse so per-pair wins.
                    try:
                        parts = _loss_parts_for_pair(
                            encoder, model, state_or_raw, ai, device,
                            type_loss_weights=type_loss_weights,
                        )
                    except Exception as e:
                        log.debug(f"  loss compute failed: {e}")
                        continue
                    (parts.total / batch_size).backward()
                    if parts.actor_fired:
                        # 5 .item() calls per pair on CPU is fine — the
                        # per-pair flow only runs on CPU, no GPU sync.
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
                        log.info(
                            f"  epoch={epoch} step={step} "
                            f"avg_loss={avg:.3f} "
                            f"(actor={avg_actor:.3f} type={avg_type:.3f} "
                            f"target={avg_target:.3f} weapon={avg_weapon:.3f}) "
                            f"pairs={running_count} rate={rate:.1f}/s "
                            f"wall={total_elapsed/60:.1f}m eta={eta}"
                        )
                    if global_step % ckpt_every == 0:
                        _save_checkpoint(
                            checkpoint_out, model, encoder, opt,
                            global_step, running_count,
                        )
                        log.info(f"  periodic checkpoint @ step={global_step}")
                    if max_pairs and running_count >= max_pairs:
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
            if use_batched and batch_encoded:
                try:
                    _flush_batch(
                        model, encoder, batch_encoded, batch_ais,
                        opt, params_for_clip, batch_size, device,
                        running_loss,
                        running_loss_actor,
                        running_loss_type,
                        running_loss_target,
                        running_loss_weapon,
                        type_loss_weights=type_loss_weights,
                    )
                    running_count += len(batch_encoded)
                except Exception as e:
                    log.debug(f"  end-of-epoch flush failed: {e}")
                    opt.zero_grad()
                batch_encoded.clear(); batch_ais.clear()
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
        _save_checkpoint(checkpoint_out, model, encoder, opt,
                         global_step, running_count)
        epoch_path = checkpoint_out.with_name(
            f"{checkpoint_out.stem}_epoch{epoch}{checkpoint_out.suffix}"
        )
        _save_checkpoint(epoch_path, model, encoder, opt,
                         global_step, running_count)
        log.info(f"Epoch {epoch} saved to {checkpoint_out} and {epoch_path.name}")
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
    ap.add_argument("--action-type-weights", type=Path, default=None,
                    help="Path to action-type loss-weight JSON "
                         "(see tools/compute_action_type_weights.py). "
                         "Default: bake-in inverse-frequency weights "
                         "(_DEFAULT_ACTION_TYPE_LOSS_WEIGHT).")
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
        type_loss_weights=type_loss_weights,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
