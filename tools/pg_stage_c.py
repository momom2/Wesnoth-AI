"""Stage C of the policy-gradient program
(docs/training_design_20261019/current_design.md §2 and §6): from one
gradient pass, read on held-out games; nothing is played or kept.

From pass.pt (build_contracts.md §3) and an Evaluator factory
(wesnoth_ai/pg_protocol.py) over the held-out games:
  1. the Fisher of the head-temperature directions and the gradient's slopes
     along them (the probe_span arm and the floor);
  2. each arm's direction scaled to one mean per-decision KL eps, the KL
     K_side per side-game, and read by ΔĴ and Ĝ on half A;
  3. the optimizer: another arm replaces reduced_adamw only when its ΔĴ on
     half A exceeds reduced_adamw's by more than 2 SE paired by game (§2);
  4. the ladder on the winner at K_side/2, K_side and 2 K_side, selected on
     half A and confirmed on half B;
  5. the critical batch B in reduced AdamW's metric, and in the winner's when
     its direction is linear in the gradient (every arm but muon);
  6. the gain curve: reduced AdamW's Ĝ on half A from the first 2 and 4 of
     the 8 splits (5,000 and 10,000 games of a 20,000-game pass) against the
     whole pass, beside the model's ratio from B;
  7. the floor sqrt(2K s^T F_s^-1 s), as read and less its noise.
The readout JSON is rewritten after each part, so a cut run keeps what it
read.

Usage:
    python tools/pg_stage_c.py --pass pass.pt --factory MODULE:CALLABLE \
        [--factory-arg KEY=VALUE ...] --out stage_c.json [--k-side 0.25]

CALLABLE(**factory_args) returns (evaluator_factory, params, sha256): the
factory binds an Evaluator to a direction, params are the weights at theta_k
named as in the pass, and sha256 (or None) is checked against the pass's
`checkpoint_sha256`.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

from wesnoth_ai.pg_protocol import EvaluatorFactory  # noqa: E402
from wesnoth_ai.pg_step import (ARMS, DEFAULT_ARM, Direction, Reading, critical_batch,  # noqa: E402
                                debiased_quadratic, decisions_per_side, direction, fisher_quadratic, floor,
                                gain_ratio, head_directions, kl_search, ladder, norm, paired_difference,
                                pooled_gradient, read_candidate, sides_by_half, span_fisher, span_slopes)

REPLACE_SE = 2.0
K_SIDE = 0.25                  # the arms' KL per side-game (design §6)
CURVE_SPLITS = (2, 4)          # the gain curve's points besides the whole pass
SPAN_STEP = 0.05               # the head-temperature Fisher's step: a 5% logit scale
LINEAR_ARMS = ("sgd", "reduced_adamw", "heads_only", "probe_span")


def load_pass(path: Path) -> Dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def choose_arm(exact: Mapping[str, Reading], default: str = DEFAULT_ARM,
               threshold: float = REPLACE_SE) -> Tuple[str, Dict[str, Dict]]:
    """The optimizer by design §2's criterion: `default` unless another arm's
    ΔĴ exceeds it by more than `threshold` paired SE; among several such
    arms, the largest difference. Returns the winner and each arm's paired
    difference."""
    table, winner, margin = {}, default, -math.inf
    for arm, reading in exact.items():
        if arm == default:
            continue
        diff = paired_difference(reading, exact[default])
        beyond = diff.estimate > threshold * diff.se
        table[arm] = {**diff.summary(), "beyond": bool(beyond)}
        if beyond and diff.estimate > margin:
            winner, margin = arm, diff.estimate
    return winner, table


class StageC:
    """One stage C run; each part fills its section of `readout` and writes
    the file."""

    def __init__(self, pass_: Mapping, params: Mapping[str, torch.Tensor], factory: EvaluatorFactory,
                 out: Path, k_side: float = K_SIDE, halves: Tuple[str, str] = ("A", "B"),
                 span_step: float = SPAN_STEP):
        missing = [name for name in pass_["names"] if name not in params]
        if missing:
            raise ValueError(f"the weights lack {len(missing)} parameters of the pass, e.g. {missing[:3]}")
        self.pass_, self.factory, self.out, self.halves = pass_, factory, Path(out), halves
        self.params = {name: params[name].detach().float().cpu() for name in pass_["names"]}
        self.span_step = span_step
        self.directions: Dict[str, Direction] = {}
        self.points = {}
        self.readout: Dict = {"pass": {k: pass_[k] for k in ("games", "decisions", "windows", "split_games",
                                                              "checkpoint_sha256")},
                              "settings": {"k_side": k_side, "halves": list(halves), "span_step": span_step,
                                           "replace_se": REPLACE_SE}}
        self.default_evaluator = factory(self.direction(DEFAULT_ARM, span_fisher=None))
        sides = self.default_evaluator.game_sides()
        self.by_half = sides_by_half(sides, halves)
        per_side = decisions_per_side(sides)
        self.eps = k_side / per_side
        self.readout["heldout"] = {"game_sides": len(sides), "decisions_per_side": per_side, "eps": self.eps,
                                   "games_by_half": {h: len({gs.game for gs in s}) for h, s in self.by_half.items()}}
        self.write()

    def write(self) -> None:
        tmp = self.out.with_name(self.out.name + ".tmp")
        tmp.write_text(json.dumps(self.readout, indent=1))
        os.replace(tmp, self.out)

    def direction(self, arm: str, span_fisher=None, gradient: Optional[Direction] = None) -> Direction:
        return direction(self.pass_, arm, self.params, span_fisher, gradient)

    def span(self) -> None:
        """Part 1: the head-temperature directions' Fisher and slopes."""
        self.heads = head_directions(self.params)
        self.F_s = span_fisher(self.factory, list(self.heads.values()), self.span_step)
        self.slopes = span_slopes(self.direction("sgd"), list(self.heads.values()))
        self.readout["span"] = {"heads": list(self.heads), "slopes": self.slopes.tolist(),
                                "fisher": self.F_s.tolist()}
        self.write()

    def _scaled(self, d: Direction, evaluator=None):
        evaluator = evaluator if evaluator is not None else self.factory(d)
        size = norm(d)
        if size == 0:
            raise ValueError("a zero direction")
        return evaluator, kl_search(evaluator.mean_kl, self.eps, s0=1.0 / size)

    def arms(self, arms: Sequence[str]) -> str:
        """Parts 2 and 3: every arm at KL eps on half A, and the winner."""
        select = self.halves[0]
        exact, table = {}, {}
        for arm in arms:
            d = self.direction(arm, span_fisher=self.F_s)
            evaluator, point = self._scaled(d, self.default_evaluator if arm == DEFAULT_ARM else None)
            reading = read_candidate(evaluator, point.s, point.kl, {select: self.by_half[select]})[select]
            self.directions[arm], self.points[arm] = d, point
            exact[arm] = reading.exact
            table[arm] = {"s": point.s, "kl": point.kl, "evals": point.evals, "norm": norm(d),
                          select: reading.summary()}
            self.readout["arms"] = table
            self.write()
        winner, paired = choose_arm(exact) if DEFAULT_ARM in exact else (max(exact, key=lambda a: exact[a].estimate), {})
        self.readout["optimizer"] = {"winner": winner, "paired_vs_default": paired}
        self.write()
        return winner

    def ladder(self, winner: str) -> None:
        """Part 4: the winner's ladder."""
        decision = ladder(self.factory, self.directions[winner], self.eps, self.halves,
                          s0=self.points[winner].s * math.sqrt(0.5))
        self.readout["ladder"] = {"arm": winner, **decision.summary()}
        self.write()

    def critical_batch(self, arm: str):
        """Part 5 for one arm: the Fisher quadratic forms of the eight split
        directions and of the whole pass's, each read near KL eps/4 (the
        splits' steps from the whole pass's by the ratio of norms)."""
        d_total = self.directions[arm]
        s_total = self.points[arm].s / 2.0
        q_total = fisher_quadratic(self.factory(d_total), s_total)
        q_splits = []
        for i in range(len(self.pass_["splits"])):
            d_split = self.direction(arm, self.F_s, pooled_gradient(self.pass_, [i]))
            q_splits.append(fisher_quadratic(self.factory(d_split), s_total * norm(d_total) / norm(d_split)))
        result = critical_batch(q_splits, q_total, self.pass_["split_games"])
        self.readout.setdefault("critical_batch", {})[arm] = {**result.summary(), "q_total": q_total,
                                                              "q_splits": q_splits}
        self.write()
        return result

    def gain_curve(self, B: float, splits: Sequence[int] = CURVE_SPLITS) -> None:
        """Part 6: reduced AdamW's Ĝ on half A from the first k splits."""
        select = self.halves[0]
        games = self.pass_["split_games"]
        full_games = sum(games)
        full = self.readout["arms"][DEFAULT_ARM][select]["first_order"]["estimate"]
        points = []
        for k in splits:
            if k >= len(games):
                continue
            d = self.direction(DEFAULT_ARM, gradient=pooled_gradient(self.pass_, range(k)))
            evaluator, point = self._scaled(d)
            reading = read_candidate(evaluator, point.s, point.kl, {select: self.by_half[select]})[select]
            n = sum(games[:k])
            points.append({"splits": k, "games": n, "kl": point.kl, "first_order": reading.first_order.summary(),
                           "ratio": reading.first_order.estimate / full if full else math.nan,
                           "model_ratio": gain_ratio(n, full_games, B)})
            self.readout["gain_curve"] = {"B": B, "full_games": full_games, "full_first_order": full,
                                          "points": points}
            self.write()

    def floor(self) -> None:
        """Part 7: the floor at KL eps, F_s per decision."""
        heads = list(self.heads.values())
        splits = [span_slopes(pooled_gradient(self.pass_, [i]), heads) for i in range(len(self.pass_["splits"]))]
        quadratic = debiased_quadratic(self.slopes, splits, self.pass_["split_games"], self.F_s)
        self.readout["floor"] = {"eps": self.eps, "floor": floor(self.slopes, self.F_s, self.eps),
                                 "floor_debiased": math.sqrt(2.0 * self.eps * max(quadratic, 0.0)),
                                 "debiased_quadratic": quadratic}
        self.write()


def run_stage_c(pass_: Mapping, params: Mapping[str, torch.Tensor], factory: EvaluatorFactory, out: Path,
                k_side: float = K_SIDE, arms: Sequence[str] = ARMS, halves: Tuple[str, str] = ("A", "B"),
                span_step: float = SPAN_STEP, curve_splits: Sequence[int] = CURVE_SPLITS) -> Dict:
    """The whole stage C readout (module docstring), written to `out`."""
    run = StageC(pass_, params, factory, out, k_side, halves, span_step)
    run.span()
    winner = run.arms(arms)
    run.ladder(winner)
    if DEFAULT_ARM in arms:
        B = run.critical_batch(DEFAULT_ARM).B
        run.gain_curve(B, curve_splits)
    if winner != DEFAULT_ARM and winner in LINEAR_ARMS:
        run.critical_batch(winner)
    run.floor()
    return run.readout


def _factory_args(pairs: List[str]) -> Dict[str, str]:
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--factory-arg takes KEY=VALUE, got {pair!r}")
        out[key] = value
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pass", dest="pass_path", type=Path, required=True)
    parser.add_argument("--factory", required=True, help="MODULE:CALLABLE returning (factory, params, sha256)")
    parser.add_argument("--factory-arg", action="append", default=[], help="KEY=VALUE for the factory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k-side", type=float, default=K_SIDE)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--curve-splits", nargs="*", type=int, default=list(CURVE_SPLITS))
    parser.add_argument("--span-step", type=float, default=SPAN_STEP)
    args = parser.parse_args(argv)
    module, _, name = args.factory.partition(":")
    factory, params, sha = getattr(importlib.import_module(module), name)(**_factory_args(args.factory_arg))
    pass_ = load_pass(args.pass_path)
    if sha is not None and sha != pass_["checkpoint_sha256"]:
        raise SystemExit(f"the evaluator's checkpoint {sha} is not the pass's {pass_['checkpoint_sha256']}")
    readout = run_stage_c(pass_, params, factory, args.out, args.k_side, args.arms,
                          span_step=args.span_step, curve_splits=args.curve_splits)
    print(json.dumps({"winner": readout["optimizer"]["winner"], "ladder": readout["ladder"]["reason"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
