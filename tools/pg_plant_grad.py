"""Gradients of the law's mass on a class of actions, summed over logged
decisions, by code separate from the gradient pass (tools/pg_plant.py, the
planted readout's forecast and its SE).

`MassGradients` sums, over groups of game-sides (the pass's splits, or the
seats of the held-out games), the gradient of

    sum_{game-sides in the group} weight * sum_t p_t(X | h_t)

at the checkpoint's weights, X one class of pg_loss.CLASSES. It reads no
label and no R. The positions are the logged decisions replayed by
tools/pg_positions.py, refused ones as shadow rows, and the memory is
carried along each game-side as the player carried it, its gradient cut at
the schedule's windows: `split_gradients` walks the planted pass's own
schedule (tools/pg_schedule.split_schedules with the pass's streams, window,
seed and slots), so it cuts the memory's gradient where the pass cuts it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from tools.pg_pass import load_network, stage_step  # noqa: E402
from tools.pg_positions import PositionLoader, record_facts, side_lengths  # noqa: E402
from tools.pg_schedule import PGMemoryRouter, split_schedules  # noqa: E402
from wesnoth_ai.param_groups import named_model_parameters  # noqa: E402
from wesnoth_ai.pg_loss import CLASSES, class_masses  # noqa: E402
from wesnoth_ai.sequence_streams import GameSide, StreamSchedule  # noqa: E402

Gradient = Dict[str, torch.Tensor]


class MassGradients:
    """The network of `checkpoint` and the logged decisions of `records`
    ({game: record path}), for gradients of class masses."""

    def __init__(self, checkpoint: Path, records: Mapping[str, Path], device: torch.device, *, streams: int,
                 window: int, seed: int, slots: int, workers: int = 0, prefetch: int = 64, verify: bool = True):
        self.device = device
        self.streams, self.window, self.seed, self.slots, self.prefetch = streams, window, seed, slots, prefetch
        self.encoder, self.model = load_network(Path(checkpoint), device)
        self.named = named_model_parameters(self.model, self.encoder)
        self.records = {g: Path(p) for g, p in records.items()}
        self.lengths = side_lengths(record_facts(self.records, workers))
        self.loader = PositionLoader(self.records, dict(self.encoder.unit_type_to_id),
                                     dict(self.encoder.faction_to_id), workers=workers, verify=verify)
        self.recompute = device.type == "cuda"

    def close(self) -> None:
        self.loader.close()

    def _zeros(self) -> Gradient:
        return {n: torch.zeros_like(p, dtype=torch.float32) for n, p in self.named}

    def _mass_step(self, staged, mems, batch, w, column: int):
        out = self.model.forward_embedded(self.encoder.embed_staged(staged), memory=mems).float32()
        return (w * class_masses(out, batch)[:, column]).sum(), out.memory_padded

    def _walk(self, schedule: StreamSchedule, weights: Mapping[Tuple[str, int], float], column: int) -> Gradient:
        """The gradient of sum weight * p(X) over the schedule's rows."""
        total = self._zeros()
        params = [p for _, p in self.named]
        memories: Dict[int, torch.Tensor] = {}
        while not schedule.exhausted():
            window = schedule.window(self.window)
            if not window:
                break
            self.loader.prefetch(g.file for g in schedule.upcoming(self.prefetch))
            router = PGMemoryRouter(self.model, self.device, memories)
            loss = torch.zeros((), device=self.device)
            for steps in window:
                positions = [self.loader.side(s.game_side.file, s.game_side.side)[s.offset] for s in steps]
                mems = router.batch(steps)
                staged = self.encoder.stage_raws([p.raw for p in positions], device=self.device)
                batch = stage_step(positions, staged, self.model, self.device)
                w = torch.tensor([weights[(s.game_side.file, s.game_side.side)] for s in steps], device=self.device)
                args = (staged, mems, batch, w, column)
                part, written = (checkpoint(self._mass_step, *args, use_reentrant=False) if self.recompute
                                 else self._mass_step(*args))
                loss = loss + part
                router.wrote(steps, written, [p.shadow for p in positions])
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            for (n, _), g in zip(self.named, grads):
                if g is not None:
                    total[n] += g.float()
            memories = {s: m.detach() for s, m in router.memories().items()}
            self.loader.keep_only(g.file for g in schedule.upcoming(self.prefetch))
        self.loader.keep_only(())
        return {n: t.cpu() for n, t in total.items()}

    def split_gradients(self, splits: Mapping[str, int], cls: str) -> List[Gradient]:
        """Per split of the planted pass (`splits`: game -> split), the
        gradient of (1/2N) sum over the split's decisions of p_t(X), N the
        number of games, on the pass's schedule."""
        weight = 1.0 / (2.0 * len(splits))
        weights = {(g, side): weight for g in splits for side in (1, 2)}
        schedules = split_schedules(dict(splits), self.lengths, self.streams, self.seed, self.slots)
        return [self._walk(s, weights, CLASSES.index(cls)) for s in schedules]

    def seat_gradients(self, seats: Mapping[str, Sequence[Tuple[str, int]]], cls: str) -> Dict[str, Gradient]:
        """Per seat (`seats`: seat -> its game-sides), the gradient of the
        mean over its game-sides of sum_t p_t(X)."""
        out = {}
        for seat, sides in seats.items():
            order = [GameSide(g, s) for g, s in sorted(sides)]
            schedule = StreamSchedule(order, self.lengths, self.streams, self.seed, k=self.slots, name=f"seat {seat}")
            weights = {(g, s): 1.0 / len(order) for g, s in sides}
            out[seat] = self._walk(schedule, weights, CLASSES.index(cls))
        return out
