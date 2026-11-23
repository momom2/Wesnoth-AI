"""One gradient pass over a round's logged games, through whole game-sides
(docs/simple_loop_prereg_20261117.md; the rulings are in
docs/training_dynamics_theory_20261111.md, "The user's decisions").

For every game-side the checkpoint played (theta_k, at the pass's memory
slots, under the temperature-1 procedure with no offset), the pass sums

    sum_t w_t grad log pi(a_t | h_t)  -  sum_t grad CE(value head, z)

and divides by the game-sides: z the side's result (+1 a win, -1 a loss, 0
a game without a winner), log pi the masked joint law the player sampled
from (wesnoth_ai/pg_loss.py), refused decisions included as shadow rows
(tools/pg_positions.py).

The weight w_t is generalised advantage estimation's (`lambda_weights`),
with no discount and the result as the only reward, over the value head's
readings v_t logged when each decision was taken:

    w_t = sum_{k>=0} lam^k (v_{t+k+1} - v_{t+k}),

the reading after a side's last decision being z. At lam = 1 (the default)
it is z - v_t: v_t is a function of the history before a_t, so the first
sum is an unbiased policy gradient. Under 1 the later readings stand in for
part of the result, each later change counting for lam times the one before
it: the weight is less noisy, and the gradient is biased by whatever the
readings get wrong.

The memory's gradient is not cut: a group of game-sides runs from each
side's first decision to its last as one graph, so a write is credited
with every later decision of its side. Each time step's forward is
recomputed in the backward pass (torch.utils.checkpoint), so a group holds
its steps' inputs and memory states and one step's activations. Game-sides
are dealt to groups longest first, at most `streams` to a group and at most
`group_decisions` decisions (a game-side longer than that is a group of its
own), so a group's rows stay near `streams` for most of its steps.

The value head is trained on the same games: at every decision, the
cross-entropy of its categorical law against z projected on its atoms, on
the trunk's global row detached (WesnothModel.value_logits_detached). Only
the value head and the material projection receive that gradient, and no
other parameter's gradient depends on it.

A game-side another network played (a past self, in the games against one)
is left out and counted. A game-side of this checkpoint that was not
played as the pass's player (another procedure, memory size or a memory
reset each turn) is refused.

Two exact checks fail the pass (exit 3), its sums written beside the output
as `<out stem>.refused.pt`:
- every decision the logs hold for the trained game-sides was trained;
- the replay identity: with d = logged log p - the pass's log p at each
  decision, half the mean of d squared is at most `replay_kl_limit`. It is
  the second-order reading of the KL from the law that played to the law
  the pass differentiates. Under float16 serving it read 5.7e-6 on round
  22's 2.45 million decisions and 5.8e-6 on 7,878 here; the limit is 1e-4,
  a root mean square d of 0.014 nats. The mean of d itself, the unbiased
  estimate tools/pg_pass.py holds to that limit over 5,000 games, is
  reported as `kl` and not held to it: over one round's decisions its
  sampling error is of the limit's order (it read 1.1e-4 on those 7,878).

grad.pt holds:

    names, total     the ascent direction per parameter (float32)
    game_sides, games, decisions, skipped_sides
    groups           per group: its game-sides, decisions, steps, seconds,
                     loss scale and its own gradient's norm per game-side
    norm             the total's norm
    norm_by_group    its norm over each group of parameters
                     (wesnoth_ai/param_groups.py): the value head's is the
                     cross-entropy's gradient, the others' the policy's
    replay           kl, half_mean_square, mean_abs, max_abs, limit
    value            the mean cross-entropy and the mean squared error of
                     the head's expectation against z, before the step
    weight           lam, the weights' mean square and what it is at lam = 1
    loss_scale, seconds, rate, peak_memory_mb, checkpoint_sha256, config

and grad.json its scalars.

    python tools/pg_game_pass.py --checkpoint pi.pt --games DIR [DIR ...] --out grad.pt \\
        --device cuda --fp16 --packed-trunk --workers 6
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from tools.pg_loss_scale import LossScale  # noqa: E402
from tools.pg_pass import REPLAY_KL_LIMIT, load_network, stage_step  # noqa: E402
from tools.pg_positions import (RECORD_SUFFIX, SELF_PLAY_PROCEDURE, ForeignGameError, PositionLoader,  # noqa: E402
                                foreign_sides, record_facts)
from tools.pg_schedule import PGMemoryRouter  # noqa: E402
from wesnoth_ai.model_output import VALUE_V_MAX, VALUE_V_MIN  # noqa: E402
from wesnoth_ai.param_groups import GROUP_ORDER, group_of, named_model_parameters  # noqa: E402
from wesnoth_ai.pg_loss import action_log_probs  # noqa: E402
from wesnoth_ai.sequence_streams import GameSide, Step  # noqa: E402

log = logging.getLogger("pg_game_pass")

EXIT_REFUSED = 3


class GamePassRefused(RuntimeError):
    """An exact check failed; `result` carries the pass's sums."""

    def __init__(self, result: Dict, why: str):
        super().__init__(why)
        self.result = result


@dataclass
class GamePassConfig:
    streams: int = 32                       # game-sides a group holds at most
    group_decisions: int = 8192             # decisions a group holds at most (a longer game-side goes alone)
    slots: int = 64                         # the memory size the player played at
    workers: int = 0                        # replay processes (0: on the calling thread)
    fp16: bool = False                      # the device's float16 autocast, loss-scaled (CUDA only)
    packed: Optional[bool] = None           # None: the trunk the forward picks
    recompute: Optional[bool] = None        # recompute each step in the backward pass (default: on CUDA)
    verify: bool = True                     # check the records' fingerprints while replaying
    procedure: str = SELF_PLAY_PROCEDURE    # the procedure every trained game-side was played under
    replay_kl_limit: float = REPLAY_KL_LIMIT
    value_loss: bool = True                 # train the value head on z (module docstring)
    lam: float = 1.0                        # the weight's lambda (`lambda_weights`); 1: z - v_t


def value_cross_entropy(logits: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """[B]: the cross-entropy of the value head's categorical law (logits
    [B, K] over K atoms evenly spaced on [VALUE_V_MIN, VALUE_V_MAX]) against
    each row's result z projected on the atoms: the two atoms around z share
    its mass in proportion (the categorical projection of Bellemare et al.
    2017), so z on an atom puts all of it there."""
    K = logits.size(-1)
    b = (z.clamp(VALUE_V_MIN, VALUE_V_MAX) - VALUE_V_MIN) * (K - 1) / (VALUE_V_MAX - VALUE_V_MIN)
    lo = b.floor().long().clamp(0, K - 2)
    upper = b - lo.to(b.dtype)
    logp = torch.log_softmax(logits, dim=-1)
    at_lo = logp.gather(1, lo.unsqueeze(1)).squeeze(1)
    at_hi = logp.gather(1, (lo + 1).unsqueeze(1)).squeeze(1)
    return -((1.0 - upper) * at_lo + upper * at_hi)


def lambda_weights(v: Sequence[float], z: float, lam: float) -> List[float]:
    """One game-side's weights in play order: generalised advantage
    estimation (Schulman et al. 2016) with no discount and the result as
    the only reward,

        a_t = sum_{k>=0} lam^k (v_{t+k+1} - v_{t+k}),

    v_t the estimate logged at decision t, and the one after the last
    decision the result z. Each later change of the estimate counts for lam
    times the one before it: lam = 1 sums them all to z - v_t, and lam = 0
    keeps the first alone. Written as a comparison, a_t is a blend of the
    later estimates and the result, less the present estimate (T the last
    decision):

        a_t = (1 - lam) sum_{k>=1} lam^(k-1) v_{t+k}  +  lam^(T-t) z  -  v_t
    """
    out = [0.0] * len(v)
    after, later = float(z), 0.0
    for t in range(len(v) - 1, -1, -1):
        later = (after - v[t]) + lam * later
        out[t] = later
        after = v[t]
    return out


def deal_groups(lengths: Dict[GameSide, int], streams: int, group_decisions: int) -> List[List[GameSide]]:
    """The game-sides in groups, longest first (ties by name): a group
    closes at `streams` game-sides or when the next would take it past
    `group_decisions` decisions. A game-side without a decision is left
    out."""
    if streams < 1 or group_decisions < 1:
        raise ValueError(f"streams and group_decisions must be positive, got {streams} and {group_decisions}")
    order = sorted((g for g, n in lengths.items() if n > 0), key=lambda g: (-lengths[g], g.file, g.side))
    groups: List[List[GameSide]] = []
    held = 0
    for g in order:
        if not groups or len(groups[-1]) >= streams or held + lengths[g] > group_decisions:
            groups.append([])
            held = 0
        groups[-1].append(g)
        held += lengths[g]
    return groups


def own_sides(facts: Dict[str, Dict[int, dict]], checkpoint_sha256: str, slots: int,
              procedure: str) -> Tuple[Dict[GameSide, dict], int]:
    """(the facts of the game-sides `checkpoint_sha256` played, the number
    of game-sides another network played). One of the checkpoint's own that
    was not played as the pass's player, or whose result its record's winner
    contradicts, is refused."""
    mine = {g: {s: f for s, f in sides.items() if f["checkpoint_sha256"] == checkpoint_sha256}
            for g, sides in facts.items()}
    others = sum(len(sides) for sides in facts.values()) - sum(len(sides) for sides in mine.values())
    bad = foreign_sides(mine, checkpoint_sha256, slots, procedure)
    if bad:
        shown = "; ".join(f"{g.file} side {g.side}: {why}" for g, why in sorted(bad.items())[:5])
        raise ForeignGameError(f"{len(bad)} of the checkpoint's game-sides were not played at {slots} slots "
                               f"under {procedure}: {shown}")
    out = {}
    for game, sides in mine.items():
        for side, f in sides.items():
            z = f["z"]
            if z and (int(f["winner"]) == side) != (z > 0):
                raise ForeignGameError(f"{game} side {side}: the record's winner {f['winner']} contradicts "
                                       f"its result {z:+.0f}")
            out[GameSide(game, side)] = f
    return out, others


class GamePass:
    """The pass of the module docstring over `records` (game name -> record
    path)."""

    def __init__(self, checkpoint_path: Path, records: Dict[str, Path], device: torch.device,
                 config: GamePassConfig = GamePassConfig()):
        from tools.eval_provenance import file_sha256
        self.config, self.device = config, device
        self.checkpoint_sha256 = file_sha256(checkpoint_path)
        self.encoder, self.model = load_network(checkpoint_path, device)
        self.named = named_model_parameters(self.model, self.encoder)
        facts = record_facts(dict(records), config.workers)
        self.sides, self.skipped = own_sides(facts, self.checkpoint_sha256, config.slots, config.procedure)
        self.lengths = {g: int(f["decisions"]) for g, f in self.sides.items()}
        self.z = {g: float(f["z"]) for g, f in self.sides.items()}
        self.groups = deal_groups(self.lengths, config.streams, config.group_decisions)
        if not self.groups:
            raise ValueError("the checkpoint played no decision of these games")
        games = {g.file for group in self.groups for g in group}
        self.loader = PositionLoader({g: records[g] for g in games}, dict(self.encoder.unit_type_to_id),
                                     dict(self.encoder.faction_to_id), workers=config.workers,
                                     verify=config.verify)
        cuda = device.type == "cuda"
        self.autocast = torch.float16 if cuda and config.fp16 else None
        self.recompute = cuda if config.recompute is None else bool(config.recompute)
        self.loss_scale = LossScale() if config.fp16 else None

    # ---- one time step ----------------------------------------------------
    def _step(self, staged, mems, batch, w, z, logged):
        """The step's signed loss, its new memories and its readings: the
        log probabilities' differences from the logged ones, the value
        head's cross-entropies and its squared errors."""
        sink: List[dict] = []
        with torch.autocast(self.device.type, dtype=self.autocast, enabled=self.autocast is not None):
            streams = self.encoder.embed_staged(staged)
            with self.model.keeping_head_inputs(sink):
                out = self.model.forward_embedded(streams, memory=mems, packed=self.config.packed)
            value_logits = self.model.value_logits_detached(sink[-1]) if self.config.value_loss else None
        out = out.float32()
        lp = action_log_probs(out, batch)
        loss = -(w * lp).sum()
        ce = torch.zeros(len(w), dtype=torch.float64, device=self.device)
        if value_logits is not None:
            rows = value_cross_entropy(value_logits.float(), z)
            loss = loss + rows.sum()
            ce = rows.detach().double()
        with torch.no_grad():
            diff = logged - lp.detach().double()
            error = (out.value.detach().double().squeeze(1) - z.double()) ** 2
        return loss, out.memory_padded, diff, ce, error

    def _forward_group(self, group: Sequence[GameSide]) -> Tuple[torch.Tensor, Dict]:
        """The group's loss, every game-side from its first decision to its
        last, and its readings."""
        router = PGMemoryRouter(self.model, self.device, {})
        rows = [self.loader.side(g.file, g.side) for g in group]
        for g, r in zip(group, rows):
            if len(r) != self.lengths[g]:
                raise RuntimeError(f"{g.file} side {g.side}: {len(r)} positions replayed for {self.lengths[g]} "
                                   f"logged decisions")
        # A refused decision is in its side's chain like any other: its
        # estimate was logged, and the side decided again at once.
        weights = [lambda_weights([p.v for p in r], self.z[g], self.config.lam) for g, r in zip(group, rows)]
        squares = (sum(a * a for ws in weights for a in ws),
                   sum((self.z[g] - p.v) ** 2 for g, r in zip(group, rows) for p in r))
        total = torch.zeros((), device=self.device)
        sums = torch.zeros(6, dtype=torch.float64, device=self.device)   # |d|, max |d|, d, d^2, CE, error^2
        n = shadow = steps_run = 0
        for t in range(max(len(r) for r in rows)):
            live = [i for i, r in enumerate(rows) if t < len(r)]
            steps = [Step(slot=i, game_side=group[i], offset=t, k=self.config.slots, starts=t == 0) for i in live]
            positions = [rows[i][t] for i in live]
            mems = router.batch(steps)
            staged = self.encoder.stage_raws([p.raw for p in positions], device=self.device)
            batch = stage_step(positions, staged, self.model, self.device)
            z = torch.tensor([self.z[group[i]] for i in live], device=self.device)
            w = torch.tensor([weights[i][t] for i in live], device=self.device)
            logged = torch.tensor([p.logp for p in positions], dtype=torch.float64, device=self.device)
            args = (staged, mems, batch, w, z, logged)
            loss, mem_padded, diff, ce, error = (checkpoint(self._step, *args, use_reentrant=False)
                                                 if self.recompute else self._step(*args))
            total = total + loss
            router.wrote(steps, mem_padded, [p.shadow for p in positions])
            sums[0] += diff.abs().sum()
            sums[1] = torch.maximum(sums[1], diff.abs().max())
            sums[2] += diff.sum()
            sums[3] += (diff * diff).sum()
            sums[4] += ce.sum()
            sums[5] += error.sum()
            n += len(live)
            shadow += sum(p.shadow for p in positions)
            steps_run += 1
        return total, {"sums": sums, "decisions": n, "shadow": shadow, "steps": steps_run, "squares": squares}

    def _group_gradient(self, group: Sequence[GameSide]) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """The group's ascent direction per parameter (summed over its
        decisions) and its readings. Under fp16 the loss is multiplied by
        the loss scale before the backward and the gradients divided by it
        after (a power of two: exact); a group whose gradient is not finite
        is run again at half the scale."""
        while True:
            scale = None if self.loss_scale is None else self.loss_scale.scale
            loss, readings = self._forward_group(group)
            (loss if scale is None else loss * scale).backward()
            grads = [(n, p) for n, p in self.named if p.grad is not None]
            finite = bool(torch.stack([torch.isfinite(p.grad).all() for _, p in grads]).all())
            if scale is not None and not finite:
                for _, p in grads:
                    p.grad = None
                self.loss_scale.overflowed()
                log.info("a group's gradient overflowed: run again at loss scale %g", self.loss_scale.scale)
                continue
            if scale is not None:
                self.loss_scale.finite()
            out = {}
            for n, p in grads:
                g = -p.grad.float()                       # the ascent direction
                out[n] = g if scale is None else g / scale
                p.grad = None
            readings["scale"] = scale
            readings["finite"] = finite
            return out, readings

    def _clock(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def run(self) -> Dict:
        """The pass; the grad.pt dict (module docstring). Raises
        `GamePassRefused`, which carries the dict, when an exact check
        fails."""
        t0 = time.time()
        for _, p in self.named:
            p.grad = None
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        total = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in self.named}
        sums = torch.zeros(6, dtype=torch.float64, device=self.device)
        decisions = shadow = 0
        squares = [0.0, 0.0]                     # the weights' squares: at the pass's lambda, at 1
        groups = []
        for k, group in enumerate(self.groups):
            self.loader.prefetch({g.file for g in group} | {g.file for g in self.groups[min(k + 1, len(self.groups) - 1)]})
            start = self._clock()
            grad, r = self._group_gradient(group)
            seconds = self._clock() - start
            square = torch.zeros((), dtype=torch.float64, device=self.device)
            for n, g in grad.items():
                total[n] += g
                square += (g.double() ** 2).sum()
            sums[[0, 2, 3, 4, 5]] += r["sums"][[0, 2, 3, 4, 5]]
            sums[1] = torch.maximum(sums[1], r["sums"][1])
            decisions += r["decisions"]
            shadow += r["shadow"]
            squares = [a + b for a, b in zip(squares, r["squares"])]
            groups.append({"game_sides": len(group), "decisions": r["decisions"], "steps": r["steps"],
                           "seconds": seconds, "scale": r["scale"], "finite": r["finite"],
                           "norm_per_side": float(square.sqrt()) / len(group)})
            log.info("group %d of %d: %d game-sides, %d decisions, %d steps, %.0f s (%.1f decisions/s)",
                     k + 1, len(self.groups), len(group), r["decisions"], r["steps"], seconds,
                     r["decisions"] / max(seconds, 1e-9))
            self.loader.keep_only({g.file for g in self.groups[min(k + 1, len(self.groups) - 1)]})
        self.loader.close()
        n_sides = sum(len(g) for g in self.groups)
        total = {n: (t / n_sides).cpu() for n, t in total.items()}
        by_group: Dict[str, float] = {}
        for n, t in total.items():
            by_group[group_of(n)] = by_group.get(group_of(n), 0.0) + float((t.double() ** 2).sum())
        rows = max(1, decisions)
        spent = sum(g["seconds"] for g in groups)
        result = {
            "names": [n for n, _ in self.named], "total": total,
            "norm": float(torch.sqrt(sum((t.double() ** 2).sum() for t in total.values()))),
            "norm_by_group": {g: by_group[g] ** 0.5 for g in GROUP_ORDER if g in by_group},
            "finite": all(g["finite"] for g in groups),
            "game_sides": n_sides, "games": len({g.file for group in self.groups for g in group}),
            "decisions": int(decisions), "shadow": int(shadow), "skipped_sides": int(self.skipped),
            "groups": groups,
            "replay": {"kl": float(sums[2]) / rows, "half_mean_square": 0.5 * float(sums[3]) / rows,
                       "mean_abs": float(sums[0]) / rows, "max_abs": float(sums[1]),
                       "limit": float(self.config.replay_kl_limit)},
            "value": {"cross_entropy": float(sums[4]) / rows, "squared_error": float(sums[5]) / rows},
            "weight": {"lam": float(self.config.lam), "mean_square": squares[0] / rows,
                       "mean_square_at_1": squares[1] / rows},
            "loss_scale": None if self.loss_scale is None else self.loss_scale.summary(),
            "seconds": time.time() - t0,
            "rate": {"decisions_per_second": decisions / max(spent, 1e-9), "seconds": spent},
            "peak_memory_mb": (torch.cuda.max_memory_allocated(self.device) / 2 ** 20
                               if self.device.type == "cuda" else None),
            "checkpoint_sha256": self.checkpoint_sha256, "config": asdict(self.config),
        }
        expected = sum(self.lengths.values())
        if decisions != expected:
            raise GamePassRefused(result, f"the pass trained {decisions} decisions of the logs' {expected}")
        if not result["replay"]["half_mean_square"] <= result["replay"]["limit"]:
            raise GamePassRefused(result, f"the replay's second-order KL {result['replay']['half_mean_square']:.3g} "
                                          f"nats per decision exceeds its limit {result['replay']['limit']:.3g} "
                                          f"({json.dumps(result['replay'])})")
        return result


SUMMARY_KEYS = ("game_sides", "games", "decisions", "shadow", "skipped_sides", "norm", "norm_by_group", "finite",
                "groups",
                "replay", "value", "weight", "loss_scale", "seconds", "rate", "peak_memory_mb",
                "checkpoint_sha256", "config")


def save_grad(result: Dict, out: Path) -> None:
    """Atomic: a temporary file, then a rename; beside it `<out stem>.json`,
    the result's scalars (`SUMMARY_KEYS`), readable without torch."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(result, tmp)
    os.replace(tmp, out)
    summary = out.with_suffix(".json")
    tmp = summary.with_name(summary.name + ".tmp")
    tmp.write_text(json.dumps({k: result.get(k) for k in SUMMARY_KEYS}, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, summary)


def game_records(dirs: Sequence[Path]) -> Dict[str, Path]:
    """Every game record of the match directories, by its stem; a stem two
    directories share is refused."""
    out: Dict[str, Path] = {}
    for d in dirs:
        for p in sorted(Path(d).glob("*" + RECORD_SUFFIX)):
            stem = p.name[:-len(RECORD_SUFFIX)]
            if stem in out:
                raise ValueError(f"two records answer to {stem}: {out[stem]} and {p}")
            out[stem] = p
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True, help="theta_k, the network that played the games")
    ap.add_argument("--games", type=Path, nargs="+", required=True, help="directories of the game records")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    defaults = GamePassConfig()
    ap.add_argument("--streams", type=int, default=defaults.streams, help="game-sides a group holds at most")
    ap.add_argument("--group-decisions", type=int, default=defaults.group_decisions,
                    help="decisions a group holds at most")
    ap.add_argument("--slots", type=int, default=defaults.slots, help="the memory size the player played at")
    ap.add_argument("--workers", type=int, default=defaults.workers, help="replay processes")
    ap.add_argument("--fp16", action="store_true", help="the device's float16 autocast with a loss scale")
    ap.add_argument("--packed-trunk", action="store_true",
                    help="the packed trunk (tools/pg_pass.py); default: the trunk the forward picks")
    ap.add_argument("--replay-kl-limit", type=float, default=REPLAY_KL_LIMIT)
    ap.add_argument("--no-value-loss", action="store_true", help="leave the value head untrained")
    ap.add_argument("--lam", type=float, default=defaults.lam,
                    help="the weight's lambda: each later change of the value estimate counts for this much of "
                         "the one before it (1: the result less the estimate)")
    ap.add_argument("--limit-games", type=int, default=0, help="the first N games only (a timing run; 0: all)")
    ap.add_argument("--no-verify", action="store_true", help="skip the records' fingerprints")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    if not 0.0 <= args.lam <= 1.0:
        ap.error("--lam must lie in 0..1")
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(name)s %(levelname)s %(message)s")
    records = game_records(args.games)
    if args.limit_games > 0:
        records = dict(list(records.items())[:args.limit_games])
    config = GamePassConfig(streams=args.streams, group_decisions=args.group_decisions, slots=args.slots,
                            workers=args.workers, fp16=args.fp16, packed=True if args.packed_trunk else None,
                            verify=not args.no_verify, replay_kl_limit=args.replay_kl_limit,
                            value_loss=not args.no_value_loss, lam=args.lam)
    try:
        result = GamePass(args.checkpoint, records, torch.device(args.device), config).run()
    except GamePassRefused as e:
        refused = args.out.with_name(f"{args.out.stem}.refused.pt")
        save_grad(e.result, refused)
        log.error("PG_GAME_PASS_REFUSED %s; the sums are in %s", e, refused)
        return EXIT_REFUSED
    save_grad(result, args.out)
    log.info("PG_GAME_PASS_DONE %d game-sides (%d skipped), %d decisions, %d groups in %.0f s; norm %.4g; "
             "replay %s; value %s; weight %s; peak %s MB",
             result["game_sides"], result["skipped_sides"], result["decisions"], len(result["groups"]),
             result["seconds"], result["norm"], json.dumps(result["replay"]), json.dumps(result["value"]),
             json.dumps(result["weight"]),
             None if result["peak_memory_mb"] is None else round(result["peak_memory_mb"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
