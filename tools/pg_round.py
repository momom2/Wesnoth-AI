"""One round's step (docs/training_design_20261019/current_design.md §1 steps
4-5, §6 "Rounds"): from the round's gradient pass (tools/pg_pass.py) and an
evaluator factory over its held-out games (tools/pg_eval.py's
network_factory), the direction d = P g under the round's arm (stage C's
winner, reduced AdamW by default; wesnoth_ai/pg_step.direction), the ladder at
KL eps/2, eps and 2 eps selected on half A and confirmed on half B
(pg_step.ladder), and the round's decision (pg_step.decide). The rung is
adopted unless half B reads below -2 SE; then pi_k stays and eps halves. An
adopted rung's network theta_k + s d is written as the candidate checkpoint,
pi_{k+1}.

eps is the round's radius: the previous round's `next_eps`, from its
round.json, or round 1's from stage C (stage_c.json's ladder). With
`--adopt-from stage_c.json` no ladder runs. Stage C's ladder on stage 1's
pass is round 1's step (design §6), so its decision is taken as read and
its candidate written.

The probe_span arm steps along the natural gradient in the span of the
head-temperature directions, so it needs that span's Fisher F_s. Round 1
takes stage C's own (stage_c.json "span" "fisher", whose `heads` must be
this network's). A later round reads its own, by stage C's order of
sources: from the pass's sampled scores when it carries them
(tools/pg_head_floor.fisher_from_pass), else through the KL on the held-out
positions (pg_step.span_fisher at stage C's SPAN_STEP).

`--chosen-network`: when the decision is decide's cross-play case (the
chosen rung's importance weights fail on half B; wesnoth_ai/pg_cross_play.py),
that rung's network theta_k + s d is written there for the match that reads
it (scripts/rounds_local.py), and recorded under `chosen_network`. It is not
the candidate: the match decides.

round.json also logs what design §6 asks of each round:
- Ĝ and ΔĴ at the three radii, and the half-B reading (the decision's
  rungs);
- the KL per side-game;
- the cosine of d with the previous round's direction (`--previous`, its
  direction.pt), which reads oscillation directly, since nothing carries
  momentum.

`--lean` (prereg departure 43): the step is the KL search alone. Along the
averaged direction (d itself without --average-with, or when the averaging
cannot take its input, as below), s is searched to the radius, eps or the
range's edge if that is lower, and the step is adopted unread: no half reads
it, so no importance weight is checked and no cross-play rung is written.
The decision holds that one rung, with no halves, and `"lean": true`.
`--lean-radius R` (prereg departure 47; R at least 1) searches to R times
that radius instead; its next eps is the target, so from the next round on
the radius reads R times the edge (once R times eps passes the edge).

`--fixed-step HEADS REST` (prereg departure 50; with --lean): no KL is read
at all. The head maps move HEADS in weight space along their direction,
averaged with the earlier rounds' head maps when --average-with is given
(pg_step.averaged), and every other parameter moves REST along its own. Each
group's direction is scaled to unit norm first, so the two lengths alone
size the step, which is adopted unread. A REST above 0 needs an arm that
reads every parameter. The decision holds no rung, `"lean": true` and
`"fixed": true`, and eps is carried to the next round unchanged; the step
itself is saved beside --direction-out as step.pt.

`--average-with` (prereg departure 37; the heads-only arm): earlier rounds'
direction.pt files. Beside d the round reads their average with d
(pg_step.averaged: each scaled to unit norm, one weight each) on
pg_step.AVERAGE_MULTIPLIERS' rungs, and pg_step.paired_ladder selects among
both directions' rungs on half A and confirms on half B. The candidate (and
a cross-play rung) steps along the chosen direction; round.json's "paired"
holds both ladders and "average" the averaged direction's facts, and
--direction-out still saves d, the round's own, so no average enters
another twice; a chosen average is saved beside it as
direction_average.pt. An input the averaging cannot take (another arm, an
earlier direction off the heads or lacking d's names, a zero average) or a
failure in its ladder is recorded under "average" "error", and the round
reads d's ladder alone.

    python tools/pg_round.py --pass pass.pt --factory tools.pg_eval:network_factory \\
        --factory-arg checkpoint=pi_k.pt --factory-arg games=DIR,... --factory-arg labels=labels.jsonl \\
        --eps-from ../round_01/round.json --arm reduced_adamw --out round.json --candidate pi_k1.pt \\
        --direction-out direction.pt [--previous ../round_01/direction.pt] \\
        [--average-with ../round_01/direction.pt ../round_00/direction.pt]
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from wesnoth_ai.pg_cross_play import cross_play_case  # noqa: E402
from wesnoth_ai.pg_step import (ARMS, AVERAGE_MULTIPLIERS, DEFAULT_ARM, HEADS_ONLY_ARMS,  # noqa: E402
                                LADDER_MULTIPLIERS, SIDE_KL_RANGE, Decision, Rung, averaged, decisions_per_side,
                                direction, dot, head_directions, head_of, kl_search, ladder, norm, paired_difference,
                                paired_ladder, span_fisher)


def eps_of(path: Path) -> float:
    """The radius a round starts from: the previous round's next_eps
    (round.json), or stage C's ladder's (stage_c.json)."""
    readout = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("decision", "ladder"):
        if isinstance(readout.get(key), dict) and readout[key].get("next_eps") is not None:
            return float(readout[key]["next_eps"])
    raise ValueError(f"{path} names no next_eps (a round.json's decision or a stage_c.json's ladder)")


def arm_of(path: Path) -> str:
    """Stage C's optimizer: its winner, the default when it read none."""
    readout = json.loads(Path(path).read_text(encoding="utf-8"))
    return (readout.get("optimizer") or {}).get("winner") or DEFAULT_ARM


def round_span_fisher(pass_: Dict, params: Dict[str, torch.Tensor], factory,
                      stage_c: Optional[Dict] = None) -> tuple:
    """(F_s, its source) for the probe_span arm (module docstring): stage C's
    when `stage_c` is given, else the pass's, else the KL's."""
    from tools.pg_head_floor import fisher_from_pass
    from tools.pg_stage_c import SPAN_STEP
    heads = head_directions(params)
    if stage_c is not None:
        span = stage_c.get("span") or {}
        if span.get("fisher") is None:
            raise ValueError("stage_c.json holds no span Fisher: stage C did not read its span")
        if list(span.get("heads") or []) != list(heads):
            raise ValueError(f"stage C's span heads {span.get('heads')} are not this network's {list(heads)}")
        return np.asarray(span["fisher"], dtype=np.float64), f"stage C ({span.get('fisher_source')})"
    from_pass = fisher_from_pass(pass_, list(heads))
    if from_pass is not None:
        return from_pass, "pass"
    return span_fisher(factory, list(heads.values()), SPAN_STEP), "kl"


def _write_json(path: Path, obj: Dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run_round(pass_: Dict, params: Dict[str, torch.Tensor], factory, arm: str, eps: float,
              previous: Optional[Dict[str, torch.Tensor]] = None, adopt: Optional[Dict] = None,
              stage_c: Optional[Dict] = None, average: Optional[Sequence[Dict[str, torch.Tensor]]] = None,
              average_of: Optional[Sequence[str]] = None, lean: bool = False, radius: float = 1.0,
              fixed: Optional[Sequence[float]] = None) -> tuple:
    """(readout, d, s or None, the direction s steps along): the round's
    direction, its ladder (or stage C's decision, `adopt`, taken as read)
    and the step to write. `stage_c` is stage_c.json, whose span Fisher
    round 1's probe_span takes. `average` (the earlier rounds' directions,
    `average_of` their paths) adds the averaged direction beside d (module
    docstring), and the step may go along it. `lean` takes the step by its
    KL search alone, to `radius` times the radius (module docstring, prereg
    departures 43 and 47). `fixed` (the head maps' length, the rest's)
    takes a lean step of those lengths with no KL read (departure 50)."""
    if lean and adopt is not None:
        raise ValueError("--lean takes no stage C decision: round 1's step is stage C's")
    if radius != 1.0 and not (lean and math.isfinite(radius) and radius >= 1.0):
        raise ValueError(f"a lean radius of {radius}: it scales a lean step's radius, so it needs --lean and "
                         f"is at least 1 (below 1 the next eps would shrink every round)")
    if fixed is not None:
        check_fixed(fixed, arm, lean, radius)
    fisher, source = (round_span_fisher(pass_, params, factory, stage_c if adopt is not None else None)
                      if arm == "probe_span" else (None, None))
    d = direction(pass_, arm, params, fisher)
    size = norm(d)
    readout: Dict = {"arm": arm, "eps": eps, "direction_norm": size,
                     "pass": {k: pass_.get(k) for k in ("games", "decisions", "windows", "checkpoint_sha256")}}
    if fisher is not None:
        readout["span_fisher"] = {"fisher": np.asarray(fisher).tolist(), "source": source}
    if previous is not None:
        common = [n for n in d if n in previous]
        readout["cosine_previous"] = (dot({n: d[n] for n in common}, {n: previous[n] for n in common})
                                      / (size * norm({n: previous[n] for n in common}) or math.nan))
    if adopt is not None:
        readout["decision"] = {**adopt, "from": "stage C's ladder (round 1's step, design §6)"}
        return readout, d, (float(adopt["s"]) if adopt.get("adopt") else None), d
    if size == 0:
        readout["decision"] = {"adopt": False, "s": 0.0, "eps": eps, "next_eps": eps / 2.0, "chosen": None,
                               "reason": "the direction is zero (every label 0)", "rungs": []}
        if lean:
            readout["decision"]["lean"] = True
        return readout, d, None, d
    if fixed is not None:
        return _fixed_step(readout, eps, d, average or [], average_of, fixed)
    if lean:
        return _lean_step(readout, factory, arm, eps, d, size, average or [], average_of, radius)
    step_d, decision = d, None
    if average:
        readout["average"] = {"of": [str(p) for p in (average_of or [])]}
        try:
            decision, step_d = _paired_step(readout, factory, arm, eps, d, size, average)
        except (ValueError, RuntimeError) as e:          # the plain step goes on (the 2026-11-05 code review's M1)
            readout.pop("paired", None)
            readout["average"]["error"] = f"{type(e).__name__}: {e}"
            print(f"PG_ROUND_AVERAGE_FAILED: {readout['average']['error']}; the round reads d's ladder alone",
                  file=sys.stderr, flush=True)
            step_d = d
    if decision is None:
        decision = ladder(factory, d, eps, s0=1.0 / size)
    readout["decision"] = decision.summary()
    sides = factory(d).game_sides()
    readout["heldout"] = {"game_sides": len(sides), "decisions_per_side": decisions_per_side(sides)}
    return readout, d, (decision.s if decision.adopt else None), step_d


def _paired_step(readout: Dict, factory, arm: str, eps: float, d: Dict[str, torch.Tensor], size: float,
                 average: Sequence[Dict[str, torch.Tensor]]) -> tuple:
    """(decision, the direction it steps along) of the paired ladder over d
    and its average with `average` (module docstring), its facts written
    into `readout`. Raises ValueError on an input departure 37 does not
    cover; run_round then reads d's ladder alone."""
    d_avg = _average(readout, arm, d, size, average)
    paired = paired_ladder(factory, {"plain": d, "average": d_avg}, eps,
                           {"plain": LADDER_MULTIPLIERS, "average": AVERAGE_MULTIPLIERS}, s0=1.0 / size)
    readout["paired"] = {**paired.summary(), "average_minus_plain_A": paired_a(paired.rungs)}
    return paired.decision, (d_avg if paired.chosen == "average" else d)


def _average(readout: Dict, arm: str, d: Dict[str, torch.Tensor], size: float,
             average: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """d averaged with `average` (prereg departure 37), its norm and its
    cosine with d written into readout["average"]. Raises ValueError on an
    input departure 37 does not cover."""
    if arm != "heads_only":
        raise ValueError(f"--average-with is prereg departure 37's, on the heads_only arm, not {arm}")
    moved = sorted({n for e in average for n, t in e.items() if head_of(n) is None and bool(torch.any(t))})
    if moved:
        raise ValueError(f"an earlier direction to average moves more than the heads: {moved[:3]}")
    d_avg = averaged([d, *average])
    size_avg = norm(d_avg)
    if not size_avg > 0:
        raise ValueError("the averaged direction is zero")
    readout["average"].update({"norm": size_avg, "cosine_plain": dot(d, d_avg) / (size * size_avg)})
    return d_avg


def _lean_step(readout: Dict, factory, arm: str, eps: float, d: Dict[str, torch.Tensor], size: float,
               average: Sequence[Dict[str, torch.Tensor]], average_of: Optional[Sequence[str]],
               radius: float = 1.0) -> tuple:
    """Prereg departure 43's step (module docstring): run_round's return,
    the step searched to `radius` times the radius (departure 47) along the
    averaged direction (or d) and adopted unread."""
    step_d, along = d, "plain"
    if average:
        readout["average"] = {"of": [str(p) for p in (average_of or [])]}
        try:
            step_d, along = _average(readout, arm, d, size, average), "average"
        except ValueError as e:                          # the step goes along d, as run_round's paired path does
            readout["average"]["error"] = f"{type(e).__name__}: {e}"
            print(f"PG_ROUND_AVERAGE_FAILED: {readout['average']['error']}; the lean step goes along d",
                  file=sys.stderr, flush=True)
    evaluator = factory(step_d)
    sides = evaluator.game_sides()
    edge = SIDE_KL_RANGE / decisions_per_side(sides)
    target = radius * min(eps, edge)
    point = kl_search(evaluator.mean_kl, target, s0=1.0 / norm(step_d))
    scaled = f" at {radius:g} times the radius (prereg departure 47)" if radius != 1.0 else ""
    decision = Decision(True, point.s, eps, target, 0,
                        f"lean step (prereg departure 43): KL {point.kl:.4g} a decision along {along}{scaled}, "
                        f"unread", [Rung(radius, target, point.s, point.kl, target <= edge, {})])
    readout["decision"] = {**decision.summary(), "lean": True}
    readout["lean"] = {"along": along, "edge": edge, "radius": radius, "kl_evals": point.evals}
    readout["heldout"] = {"game_sides": len(sides), "decisions_per_side": decisions_per_side(sides)}
    return readout, d, point.s, step_d


def check_fixed(fixed: Sequence[float], arm: str, lean: bool, radius: float) -> None:
    """Raises ValueError unless `fixed` is a step --fixed-step can take
    (module docstring): two finite lengths, neither negative and not both
    0, on a lean step at radius 1, the rest's 0 on an arm that reads the
    head maps alone."""
    if len(fixed) != 2 or not all(math.isfinite(x) and x >= 0 for x in fixed) or not any(x > 0 for x in fixed):
        raise ValueError(f"a fixed step of {list(fixed)}: two lengths in weight space, the head maps' and the "
                         f"rest's, neither negative and not both 0")
    if not lean or radius != 1.0:
        raise ValueError("a fixed step is a lean step with no radius to scale: it needs --lean and no --lean-radius")
    if fixed[1] > 0 and arm in HEADS_ONLY_ARMS:
        raise ValueError(f"a fixed step moves the rest of the network by {fixed[1]}, and the {arm} arm reads the "
                         f"head maps alone: give an arm that reads every parameter")


def _fixed_step(readout: Dict, eps: float, d: Dict[str, torch.Tensor], average: Sequence[Dict[str, torch.Tensor]],
                average_of: Optional[Sequence[str]], fixed: Sequence[float]) -> tuple:
    """Prereg departure 50's step (module docstring): run_round's return.
    The head maps move fixed[0] along their direction (averaged with
    `average`'s head maps, each scaled to unit norm, when that can be
    taken) and every other parameter fixed[1] along its own, and the step
    is adopted unread at s = 1: no KL is searched or read. A group whose
    direction is zero stays where it is; when both do, pi_k stays."""
    heads = {n: t for n, t in d.items() if head_of(n)}
    rest = {n: t for n, t in d.items() if not head_of(n)}
    head_d, along = heads, "plain"
    if average and heads and norm(heads) > 0:
        readout["average"] = {"of": [str(p) for p in (average_of or [])]}
        try:
            head_d = averaged([heads, *average])
            size_avg = norm(head_d)
            if not size_avg > 0:
                raise ValueError("the averaged direction is zero")
            readout["average"].update({"norm": size_avg, "cosine_plain": dot(heads, head_d) / (norm(heads) * size_avg)})
            along = "average"
        except ValueError as e:                          # the head maps go along d, as a lean step's do
            head_d = heads
            readout["average"]["error"] = f"{type(e).__name__}: {e}"
            print(f"PG_ROUND_AVERAGE_FAILED: {readout['average']['error']}; the fixed step's head maps go along d",
                  file=sys.stderr, flush=True)
    step_d = {n: torch.zeros_like(t, dtype=torch.float32) for n, t in d.items()}
    moved = {}
    for group, part, length in (("heads", head_d, float(fixed[0])), ("rest", rest, float(fixed[1]))):
        size = norm(part) if part else 0.0
        moved[group] = length if size > 0 else 0.0
        if length > 0 and size > 0:
            step_d.update({n: t.float() * (length / size) for n, t in part.items()})
    readout["lean"] = {"along": along, "fixed": {"heads": float(fixed[0]), "rest": float(fixed[1]),
                                                 "heads_norm": norm(heads) if heads else 0.0,
                                                 "rest_norm": norm(rest) if rest else 0.0, "moved": moved}}
    if not any(moved.values()):
        readout["decision"] = {"adopt": False, "s": 0.0, "eps": eps, "next_eps": eps, "chosen": None,
                               "reason": "fixed step (prereg departure 50): no group with a length has a direction",
                               "rungs": [], "lean": True, "fixed": True}
        return readout, d, None, d
    decision = Decision(True, 1.0, eps, eps, None,
                        f"fixed step (prereg departure 50): the head maps {moved['heads']:g} along {along}, the rest "
                        f"{moved['rest']:g}, in weight space; no KL read, unread", [])
    readout["decision"] = {**decision.summary(), "lean": True, "fixed": True}
    return readout, d, 1.0, step_d


def paired_a(rungs: Dict[str, list]) -> Optional[Dict[str, float]]:
    """Departure 37's number: ΔĴ on half A of the averaged direction's eps
    rung less the plain direction's, paired by game; None unless both read
    there."""
    def at_eps(label):
        return next((r for r in rungs.get(label, []) if r.multiplier == 1.0 and r.in_range and "A" in r.halves), None)
    a, b = at_eps("average"), at_eps("plain")
    if a is None or b is None:
        return None
    return paired_difference(a.halves["A"].exact, b.halves["A"].exact).summary()


def main(argv: Optional[List[str]] = None) -> int:
    from tools.pg_stage_c import _factory_args, load_pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pass", dest="pass_path", type=Path, required=True)
    ap.add_argument("--factory", required=True, help="MODULE:CALLABLE returning (factory, params, sha256)")
    ap.add_argument("--factory-arg", action="append", default=[], help="KEY=VALUE for the factory")
    radius = ap.add_mutually_exclusive_group(required=True)
    radius.add_argument("--eps", type=float, help="the round's radius, KL a decision")
    radius.add_argument("--eps-from", type=Path, help="the previous round.json, or stage_c.json for round 2")
    radius.add_argument("--adopt-from", type=Path, help="stage_c.json: its ladder's decision is round 1's")
    arm = ap.add_mutually_exclusive_group()
    arm.add_argument("--arm", choices=ARMS, default=None)
    arm.add_argument("--arm-from", type=Path, help="stage_c.json: its optimizer's winner")
    ap.add_argument("--previous", type=Path, default=None, help="the previous round's direction.pt")
    ap.add_argument("--average-with", type=Path, nargs="+", default=None,
                    help="earlier rounds' direction.pt: read their average with d beside d (prereg departure 37)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True, help="where an adopted rung's network is written")
    ap.add_argument("--direction-out", type=Path, default=None)
    ap.add_argument("--chosen-network", type=Path, default=None,
                    help="where the chosen rung's network is written when a cross-play match must read it")
    ap.add_argument("--lean", action="store_true",
                    help="the step by its KL search alone, adopted unread (prereg departure 43)")
    ap.add_argument("--lean-radius", type=float, default=1.0, metavar="R",
                    help="a lean step searches to R times the radius (prereg departure 47)")
    ap.add_argument("--fixed-step", type=float, nargs=2, default=None, metavar=("HEADS", "REST"),
                    help="a lean step of these lengths in weight space, the head maps' and the rest's, with no KL "
                         "read (prereg departure 50)")
    args = ap.parse_args(argv)
    if args.lean and args.adopt_from is not None:
        ap.error("--lean takes no --adopt-from: round 1's step is stage C's")
    if args.lean_radius != 1.0 and not (args.lean and math.isfinite(args.lean_radius) and args.lean_radius >= 1.0):
        ap.error(f"--lean-radius {args.lean_radius}: a scale of at least 1 of a --lean step's radius")
    if args.fixed_step is not None and not (args.lean and args.lean_radius == 1.0):
        ap.error("--fixed-step is a --lean step with no --lean-radius")
    module, _, name = args.factory.partition(":")
    factory, params, sha = getattr(importlib.import_module(module), name)(**_factory_args(args.factory_arg))
    pass_ = load_pass(args.pass_path)
    if sha is not None and sha != pass_["checkpoint_sha256"]:
        raise SystemExit(f"the evaluator's checkpoint {sha} is not the pass's {pass_['checkpoint_sha256']}")
    adopt, stage_c = None, None
    if args.adopt_from is not None:
        stage_c = json.loads(args.adopt_from.read_text(encoding="utf-8"))
        ladder_readout = stage_c.get("ladder")
        if not ladder_readout:
            raise SystemExit(f"{args.adopt_from} holds no ladder: stage C did not read round 1's step")
        adopt = {k: ladder_readout.get(k) for k in ("adopt", "s", "eps", "next_eps", "chosen", "reason")}
        arm = ladder_readout.get("arm") or arm_of(args.adopt_from)
        eps = float(ladder_readout["eps"])
    else:
        arm = args.arm or (arm_of(args.arm_from) if args.arm_from else DEFAULT_ARM)
        eps = args.eps if args.eps is not None else eps_of(args.eps_from)
    previous = torch.load(args.previous, map_location="cpu", weights_only=True) if args.previous else None
    average = [torch.load(p, map_location="cpu", weights_only=True) for p in args.average_with or []]
    readout, d, s, step_d = run_round(pass_, params, factory, arm, eps, previous, adopt, stage_c, average,
                                      args.average_with, lean=args.lean, radius=args.lean_radius,
                                      fixed=args.fixed_step)
    if args.direction_out is not None:
        torch.save(d, args.direction_out)
        if step_d is not d:              # what rebuilds an averaged candidate (the code review's L4)
            fixed = "fixed" in (readout.get("lean") or {})
            averaged_out = args.direction_out.with_name("step.pt" if fixed else "direction_average.pt")
            torch.save(step_d, averaged_out)
            readout["lean" if "lean" in readout else "paired"]["direction"] = str(averaged_out)
    readout["candidate"] = None
    if s is not None:
        sha_out = factory(step_d).write_candidate(s, args.candidate)
        readout["candidate"] = {"path": str(args.candidate), "sha256": sha_out, "s": s}
    readout["chosen_network"] = None
    rungs = readout["decision"].get("rungs") or ((stage_c or {}).get("ladder") or {}).get("rungs") or []
    chosen = cross_play_case(readout["decision"], rungs)
    if args.chosen_network is not None and chosen is not None:
        s_chosen = float(rungs[chosen]["s"])
        sha_chosen = factory(step_d).write_candidate(s_chosen, args.chosen_network)
        readout["chosen_network"] = {"path": str(args.chosen_network), "sha256": sha_chosen, "s": s_chosen,
                                     "rung": chosen}
    _write_json(args.out, readout)
    dec = readout["decision"]
    along = (f" along {readout['paired']['chosen']}" if "paired" in readout else
             f" along {readout['lean']['along']} (lean)" if "lean" in readout else "")
    print(f"PG_ROUND adopt={dec['adopt']} s={dec['s']:.4g} eps={eps:.4g} next_eps={dec['next_eps']:.4g} "
          f"arm={arm}{along}: {dec['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
