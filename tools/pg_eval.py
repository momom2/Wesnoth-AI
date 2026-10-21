"""The network side of the step's readings
(docs/training_design_20261019/current_design.md §1 step 5, §5;
build_contracts.md §4): `NetworkEvaluator`, the `Evaluator` of
wesnoth_ai/pg_protocol.py on the real network.

An `EvalContext` holds what every evaluator of one checkpoint and one
held-out set shares: the network at theta_k, the held-out game-sides of the
label rows and the replay of their logged decisions through the core
(tools/pg_positions.py). `context.bind(d)` is the evaluator bound to one
direction d in weight space ({parameter name: tensor}, the names of pass.pt;
a name left out moves nothing): the protocol's `EvaluatorFactory`.

`logp(s)` is log pi at theta_k + s d of every held-out decision's sampled
action, refused ones included, forwards only: game-side after game-side in
`game_sides()` order, each side's decisions in log order, the memory carried
along each game-side as the player carried it (a shadow row reads and writes
nothing). `base_logp()` is `logp(0.0)` as this code recomputes it, not the
logged value: the replay identity compares the two. `mean_kl(s)`, s of
either sign, is the mean per-decision KL(pi_k || pi_{theta_k + s d}) over
the KL subsample, the game-sides drawn with probability `kl_fraction` by a
hash of their names; both networks carry their own memories along it.

`evaluate([s, ...])` reads several steps in one replay of the positions
(each `logp` call replays them otherwise). `class_mass_by_seat(s)` is the
teacher-forced sum over each game-side's decisions of the law's mass on
end_turn, attacks and recruits at theta_k + s d, by seat (the planted
round's offline readout). `write_candidate(s, path)` writes theta_k + s d
as a checkpoint any loader reads.

`network_factory` is the stage C driver's `--factory` (tools/pg_stage_c.py):

    python tools/pg_stage_c.py --pass pass.pt --factory tools.pg_eval:network_factory \\
        --factory-arg checkpoint=own2.pt --factory-arg games=DIR[,DIR...] \\
        --factory-arg labels=labels.jsonl --factory-arg device=cuda --factory-arg workers=30 --out stage_c.json
"""
from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from tools.pg_pass import load_network, stage_step  # noqa: E402
from tools.pg_positions import PositionLoader, RecordIndex, side_lengths  # noqa: E402
from tools.pg_schedule import PGMemoryRouter, game_order  # noqa: E402
from wesnoth_ai.param_groups import named_model_parameters  # noqa: E402
from wesnoth_ai.pg_loss import CLASSES, action_log_probs, class_masses, legal_kl  # noqa: E402
from wesnoth_ai.pg_protocol import GameSide  # noqa: E402
from wesnoth_ai.pg_records import LabelRow, labelled_games, read_labels  # noqa: E402
from wesnoth_ai.sequence_streams import GameSide as StreamSide  # noqa: E402
from wesnoth_ai.sequence_streams import StreamSchedule, unit_draw  # noqa: E402


@dataclass
class EvalConfig:
    streams: int = 64
    window: int = 16
    seed: int = 20261021
    slots: int = 64
    workers: int = 0
    prefetch: int = 128
    fp32: bool = False
    packed: Optional[bool] = None
    verify: bool = True
    kl_fraction: float = 0.05


@dataclass
class Reading:
    """One step's forwards over a set of game-sides, per decision."""
    logp: np.ndarray                  # [n] float64
    masses: np.ndarray                # [n, 3] float64, CLASSES' order
    kl: Optional[np.ndarray] = None   # [n] float64, against theta_k


class EvalContext:
    """The network at theta_k and the held-out decisions of `labels`, shared
    by the evaluators `bind` makes."""

    def __init__(self, checkpoint_path: Path, records: Mapping[str, Path], labels: Sequence[LabelRow],
                 device: torch.device, config: EvalConfig = EvalConfig()):
        from tools.eval_provenance import file_sha256
        self.checkpoint_path, self.device, self.config = Path(checkpoint_path), device, config
        self.checkpoint_sha256 = file_sha256(self.checkpoint_path)
        self.encoder, self.model = load_network(self.checkpoint_path, device)
        self.named = named_model_parameters(self.model, self.encoder)
        self.base = {n: p.detach().clone() for n, p in self.named}
        games = labelled_games(labels, heldout=True)
        if not games:
            raise ValueError("no held-out game in the labels")
        paths = {g: Path(records[g]) for g in games}
        lengths = side_lengths(paths, config.workers)
        self.loader = PositionLoader(paths, dict(self.encoder.unit_type_to_id), dict(self.encoder.faction_to_id),
                                     workers=config.workers, verify=config.verify)
        self.sides: List[GameSide] = []
        start = 0
        for g in game_order(games, config.seed):
            row, n = games[g.file][g.side], lengths[g]
            self.sides.append(GameSide(game=g.file, side=g.side, R=row.R, half=row.half, start=start, stop=start + n))
            start += n
        self.kl_sides = ([gs for gs in self.sides if unit_draw("pg_kl", config.seed, gs.game, gs.side)
                          < config.kl_fraction] or self.sides[:1])
        self.autocast = torch.bfloat16 if (device.type == "cuda" and not config.fp32) else None

    def params(self) -> Dict[str, torch.Tensor]:
        """theta_k by parameter name, on the CPU."""
        return {n: t.detach().cpu() for n, t in self.base.items()}

    def bind(self, direction: Mapping[str, torch.Tensor]) -> "NetworkEvaluator":
        return NetworkEvaluator(self, direction)

    def close(self) -> None:
        self.loader.close()

    # ---- the forwards ----------------------------------------------------
    def network(self, direction: Mapping[str, torch.Tensor], s: float):
        """(encoder, model) at theta_k + s d: the loaded modules at 0, a copy
        otherwise."""
        if s == 0.0:
            return self.encoder, self.model
        encoder, model = copy.deepcopy(self.encoder), copy.deepcopy(self.model)
        with torch.no_grad():
            for n, p in named_model_parameters(model, encoder):
                d = direction.get(n)
                p.copy_(self.base[n] if d is None else self.base[n] + s * d)
        return encoder.eval(), model.eval()

    def read(self, direction: Mapping[str, torch.Tensor], steps: Sequence[float], sides: Sequence[GameSide],
             kl: bool) -> Dict[float, Reading]:
        """Each step's reading over `sides`, theta_k's alongside when `kl`,
        in one replay of their positions."""
        keys = sorted({float(s) for s in steps} | ({0.0} if kl else set()))
        networks = {s: self.network(direction, s) for s in keys}
        local, start = {}, 0
        for gs in sides:
            local[(gs.game, gs.side)] = start
            start += gs.stop - gs.start
        out = {s: Reading(logp=np.zeros(start), masses=np.zeros((start, len(CLASSES))),
                          kl=np.zeros(start) if kl else None) for s in keys}
        order = [StreamSide(gs.game, gs.side) for gs in sides]
        lengths = {StreamSide(gs.game, gs.side): gs.stop - gs.start for gs in sides}
        schedule = StreamSchedule(order, lengths, self.config.streams, self.config.seed, k=self.config.slots)
        memories: Dict[float, Dict[int, torch.Tensor]] = {s: {} for s in keys}
        with torch.no_grad():
            while not schedule.exhausted():
                window = schedule.window(self.config.window)
                if not window:
                    break
                self.loader.prefetch(g.file for g in schedule.upcoming(self.config.prefetch))
                routers = {s: PGMemoryRouter(networks[s][1], self.device, memories[s]) for s in keys}
                for steps_t in window:
                    self._time_step(steps_t, networks, routers, local, out, kl)
                memories = {s: routers[s].memories() for s in keys}
                self.loader.keep_only(g.file for g in schedule.upcoming(self.config.prefetch))
        self.loader.keep_only(())
        return {float(s): out[float(s)] for s in steps}

    def _time_step(self, steps, networks, routers, local, out, kl: bool) -> None:
        positions = [self.loader.side(s.game_side.file, s.game_side.side)[s.offset] for s in steps]
        index = np.array([local[(s.game_side.file, s.game_side.side)] + s.offset for s in steps])
        staged = self.encoder.stage_raws([p.raw for p in positions], device=self.device)
        batch = stage_step(positions, staged, self.model, self.device)
        shadow = [p.shadow for p in positions]
        outputs = {}
        for s, (encoder, model) in networks.items():
            with torch.autocast(self.device.type, dtype=self.autocast, enabled=self.autocast is not None):
                padded = model.forward_embedded(encoder.embed_staged(staged), memory=routers[s].batch(steps),
                                                packed=self.config.packed)
            padded = padded.float32()
            routers[s].wrote(steps, padded.memory_padded, shadow)
            outputs[s] = padded
            out[s].logp[index] = action_log_probs(padded, batch).double().cpu().numpy()
            out[s].masses[index] = class_masses(padded, batch).double().cpu().numpy()
        if kl:
            for s in networks:
                out[s].kl[index] = legal_kl(outputs[0.0], outputs[s], batch.views, batch.rows).double().cpu().numpy()


class NetworkEvaluator:
    """The `Evaluator` of `context`'s network along `direction` (module
    docstring)."""

    def __init__(self, context: EvalContext, direction: Mapping[str, torch.Tensor]):
        names = {n for n, _ in context.named}
        unknown = sorted(set(direction) - names)
        if unknown:
            raise ValueError(f"the direction names parameters the network does not have: {unknown[:5]}")
        for n, d in direction.items():
            if tuple(d.shape) != tuple(context.base[n].shape):
                raise ValueError(f"the direction's {n} is {tuple(d.shape)}, the parameter "
                                 f"{tuple(context.base[n].shape)}")
        self.context = context
        self.direction = {n: d.detach().to(device=context.device, dtype=torch.float32) for n, d in direction.items()}
        self._full: Dict[float, Reading] = {}
        self._kl: Dict[float, float] = {}

    # ---- the protocol ----------------------------------------------------
    def game_sides(self) -> Sequence[GameSide]:
        return list(self.context.sides)

    def base_logp(self) -> np.ndarray:
        return self.logp(0.0)

    def logp(self, s: float) -> np.ndarray:
        return self._reading(s).logp

    def mean_kl(self, s: float) -> float:
        s = float(s)
        if s not in self._kl:
            if s == 0.0:
                self._kl[s] = 0.0
            else:
                reading = self.context.read(self.direction, [s], self.context.kl_sides, kl=True)[s]
                self._kl[s] = float(reading.kl.mean())
        return self._kl[s]

    # ---- beyond the protocol --------------------------------------------
    def evaluate(self, steps: Iterable[float]) -> None:
        """Read the steps not yet read in one replay of the positions."""
        todo = sorted({float(s) for s in steps} - set(self._full))
        if todo:
            self._full.update(self.context.read(self.direction, todo, self.context.sides, kl=False))

    def class_mass_by_seat(self, s: float, seat_of: Optional[Callable[[str, int], object]] = None
                           ) -> Dict[object, Dict[str, np.ndarray]]:
        """Per seat (`seat_of(game, side)`, by default the side), per class
        of `CLASSES`: each of its game-sides' sum over its decisions of the
        law's mass on the class at theta_k + s d, in `game_sides()` order."""
        seat_of = seat_of or (lambda game, side: side)
        masses = self._reading(s).masses
        out: Dict[object, Dict[str, List[float]]] = {}
        for gs in self.context.sides:
            sums = masses[gs.start:gs.stop].sum(axis=0)
            seat = out.setdefault(seat_of(gs.game, gs.side), {c: [] for c in CLASSES})
            for k, c in enumerate(CLASSES):
                seat[c].append(float(sums[k]))
        return {seat: {c: np.array(v) for c, v in by.items()} for seat, by in out.items()}

    def write_candidate(self, s: float, path: Path) -> str:
        """theta_k + s d as a checkpoint at `path` (the source checkpoint
        with its weights replaced and the step recorded); its SHA-256."""
        from tools.eval_provenance import file_sha256
        ck = torch.load(self.context.checkpoint_path, map_location="cpu", weights_only=True)
        encoder, model = self.context.network(self.direction, float(s))
        ck["model_state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        ck["encoder_state"] = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}
        ck["pg_candidate"] = {"base_sha256": self.context.checkpoint_sha256, "step": float(s)}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(ck, tmp)
        os.replace(tmp, path)
        return file_sha256(path)

    def _reading(self, s: float) -> Reading:
        s = float(s)
        self.evaluate([s])
        return self._full[s]


def _config_value(name: str, text: str):
    """An `EvalConfig` field from its text."""
    if name in ("fp32", "verify"):
        return str(text).strip().lower() in ("1", "true", "yes", "on")
    if name == "packed":
        return None if text.lower() == "auto" else _config_value("fp32", text)
    return float(text) if name == "kl_fraction" else int(text)


def network_factory(checkpoint: str, games: str, labels: str, device: Optional[str] = None, **config: str):
    """(evaluator factory, theta_k by name, the checkpoint's SHA-256): the
    stage C driver's `--factory` (module docstring). `games` lists the
    record directories, comma-separated; the other keyword arguments are
    `EvalConfig`'s fields, as text (`packed=auto` lets the forward pick)."""
    names = {f.name for f in fields(EvalConfig)}
    unknown = sorted(set(config) - names)
    if unknown:
        raise ValueError(f"network_factory takes no {unknown}")
    parsed = {k: _config_value(k, v) for k, v in config.items()}
    rows = read_labels(Path(labels))
    index = RecordIndex([Path(d) for d in games.split(",") if d])
    records = {r.game: index.path(r.game) for r in rows if r.heldout}
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    context = EvalContext(Path(checkpoint), records, rows, dev, EvalConfig(**parsed))
    return context.bind, context.params(), context.checkpoint_sha256
