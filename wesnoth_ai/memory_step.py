"""The self-play learner over a model with a memory
(docs/memory_everywhere_20261005.md, hole 6).

A side's memory at a decision is what its previous decision wrote, so the
learner never evaluates such a model on a position alone: it runs each
game-side's positions in order, the ones without a search target included
(the actors ship every decision, tools/memory_trace.py).

The step (`step_streams`) is the sequence trainer's scheme
(tools/sequence_train.py) on the self-play losses: `train_batch_size`
game-sides run side by side, each window of `memory_window` decisions is
back-propagated through the memory, and the memory crosses into the next
window without gradient (truncated back-propagation through time; Williams
and Peng, 1990). Each decision's forward is recomputed in the backward pass
(activation checkpointing). The step stays one optimizer update over the
whole batch, as step_mcts is: the weights are the same in every window, so
the carried memory is the current network's.

Telemetry reads a position's memory from `memory_inputs`, one in-order pass
without gradient over the game-sides the positions belong to.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.checkpoint import checkpoint

from wesnoth_ai.memory import MemoryState

# An experience's place in a game-side: (game id, side).
StreamKey = Tuple[str, int]


def reads_memory(model) -> bool:
    """The model has memory slots: every forward takes each sample's memory."""
    return int(getattr(model, "memory_slots", 0) or 0) > 0


def game_side_streams(experiences: Sequence) -> List[List]:
    """The experiences grouped by game-side, each in decision order. A
    game-side that misses a decision, holds one twice or mixes slot counts
    is refused: its memory cannot be rebuilt."""
    by_side: Dict[StreamKey, List] = {}
    for e in experiences:
        if int(getattr(e, "side_step", -1)) < 0:
            raise ValueError("an experience without its place in its game-side (side_step): "
                             "a model with a memory trains on the actors' whole game-sides")
        by_side.setdefault((str(e.game_id), int(e.side)), []).append(e)
    streams = []
    for key in sorted(by_side):
        stream = sorted(by_side[key], key=lambda e: int(e.side_step))
        steps = [int(e.side_step) for e in stream]
        if steps != list(range(len(stream))):
            raise ValueError(f"game-side {key} holds decisions {steps[:8]}..., not 0 to "
                             f"{len(stream) - 1} in order: its memory cannot be rebuilt")
        if len({int(e.memory_k) for e in stream}) != 1:
            raise ValueError(f"game-side {key} mixes memory sizes")
        streams.append(stream)
    return streams


def cap_streams(streams: List[List], cap: int, rng: random.Random) -> List[List]:
    """Whole game-sides in a random order until `cap` positions (at least
    one game-side): the positional subsample of step_mcts would cut the
    memory chains."""
    if sum(len(s) for s in streams) <= cap:
        return streams
    kept, total = [], 0
    for i in rng.sample(range(len(streams)), len(streams)):
        if kept and total + len(streams[i]) > cap:
            continue
        kept.append(streams[i])
        total += len(streams[i])
    return kept


def time_major(streams: List[List], width: int, window: int):
    """The order the step runs the positions in, and its plan: per group of
    `width` game-sides, per window of `window` decisions, each decision as
    (offset of its positions in the order, the indices of the game-sides
    still running)."""
    order: List = []
    plan: List[Tuple[List[int], List[List[Tuple[int, List[int]]]]]] = []
    for g0 in range(0, len(streams), width):
        group = list(range(g0, min(g0 + width, len(streams))))
        length = max(len(streams[i]) for i in group)
        windows = []
        for w0 in range(0, length, window):
            steps = []
            for t in range(w0, min(w0 + window, length)):
                active = [i for i in group if t < len(streams[i])]
                steps.append((len(order), active))
                order.extend(streams[i][t] for i in active)
            windows.append(steps)
        plan.append((group, windows))
    return order, plan


def _chunk_loss(trainer, chunk, raw_chunk, start, batch, timer, autocast_bf16, memory):
    return trainer._mcts_chunk_loss(chunk, raw_chunk, start, batch, timer, autocast_bf16,
                                    memory=memory)


def step_streams(trainer, experiences: Sequence, timings: Optional[Dict[str, float]] = None,
                 no_grad: bool = False):
    """step_mcts for a model with a memory (module docstring)."""
    from wesnoth_ai.trainer import TrainStats, _mcts_batch, _StageTimer, _summed_stats
    if not experiences:
        return TrainStats()
    cfg = trainer.config
    streams = cap_streams(game_side_streams(experiences), cfg.max_transitions_per_step, trainer.rng)
    dev = trainer.device or next(trainer.model.parameters()).device
    autocast_bf16 = bool(cfg.train_autocast_bf16) and dev.type == "cuda"
    order, plan = time_major(streams, max(1, cfg.train_batch_size), max(1, cfg.memory_window))
    batch = _mcts_batch(trainer, order, dev)
    if not no_grad:
        trainer.optimizer.zero_grad()
    trainer.model.eval()
    trainer.encoder.eval()
    timer = _StageTimer(dev, timings if timings is not None else trainer.stage_timings)
    with timer.stage("encode_raw"):
        raw_cache = trainer._mcts_raw_cache(order)
    losses = []
    for group, windows in plan:
        held: Dict[int, torch.Tensor] = {}
        for steps in windows:
            total = None
            for start, active in steps:
                chunk = order[start:start + len(active)]
                memory = [held[i] if i in held else trainer.model.initial_memory(int(e.memory_k))
                          for i, e in zip(active, chunk)]
                args = (trainer, chunk, raw_cache[start:start + len(active)], start, batch, timer,
                        autocast_bf16, memory)
                loss = (_chunk_loss(*args) if no_grad
                        else checkpoint(_chunk_loss, *args, use_reentrant=False))
                for i, m in zip(active, loss.memory):
                    held[i] = m
                total = loss.total if total is None else total + loss.total
                losses.append(loss.detached())
            if not no_grad and total is not None:
                with timer.stage("backward"):
                    total.backward()
            held = {i: m.detach() for i, m in held.items()}
    return _summed_stats(trainer, losses, batch, len(order), timer, no_grad)


def memory_inputs(model, encoder, experiences: Sequence,
                  width: int = 32) -> Dict[int, MemoryState]:
    """Each experience's memory at its decision (by `id`), what the side's
    previous decision wrote under `model`'s current weights: one in-order
    pass without gradient over the experiences' whole game-sides."""
    out: Dict[int, MemoryState] = {}
    streams = game_side_streams(experiences)
    with torch.no_grad():
        for g0 in range(0, len(streams), width):
            group = streams[g0:g0 + width]
            held: Dict[int, Optional[torch.Tensor]] = {}
            for t in range(max(len(s) for s in group)):
                active = [i for i, s in enumerate(group) if t < len(s)]
                chunk = [group[i][t] for i in active]
                raws = [e.raw if e.raw is not None else encoder.raw_of(e.game_state) for e in chunk]
                states = [MemoryState(int(e.memory_k), held.get(i)) for i, e in zip(active, chunk)]
                outs = model.forward_batch(encoder.encode_from_raw_batch(raws), memory=states)
                for i, e, state, o in zip(active, chunk, states, outs):
                    out[id(e)] = state
                    held[i] = o.memory
    return out


def whole_game_sides(experiences: Sequence, positions: int, rng: random.Random) -> List:
    """Whole game-sides of `experiences` in a random order until about
    `positions` positions: the sample a memory model's gradient probes
    run on."""
    streams = game_side_streams(experiences)
    return [e for s in cap_streams(streams, positions, rng) for e in s]
