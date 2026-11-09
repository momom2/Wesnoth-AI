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
from wesnoth_ai.pg_step import (ARMS, AVERAGE_MULTIPLIERS, DEFAULT_ARM, LADDER_MULTIPLIERS,  # noqa: E402
                                SIDE_KL_RANGE, Decision, Rung, averaged, decisions_per_side, direction, dot,
                                head_directions, head_of, kl_search, ladder, norm, paired_difference, paired_ladder,
                                span_fisher)


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
              average_of: Optional[Sequence[str]] = None, lean: bool = False) -> tuple:
    """(readout, d, s or None, the direction s steps along): the round's
    direction, its ladder (or stage C's decision, `adopt`, taken as read)
    and the step to write. `stage_c` is stage_c.json, whose span Fisher
    round 1's probe_span takes. `average` (the earlier rounds' directions,
    `average_of` their paths) adds the averaged direction beside d (module
    docstring), and the step may go along it. `lean` takes the step by its
    KL search alone (module docstring, prereg departure 43)."""
    if lean and adopt is not None:
        raise ValueError("--lean takes no stage C decision: round 1's step is stage C's")
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
    if lean:
        return _lean_step(readout, factory, arm, eps, d, size, average or [], average_of)
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
               average: Sequence[Dict[str, torch.Tensor]], average_of: Optional[Sequence[str]]) -> tuple:
    """Prereg departure 43's step (module docstring): run_round's return,
    the step searched to the radius along the averaged direction (or d)
    and adopted unread."""
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
    target = min(eps, edge)
    point = kl_search(evaluator.mean_kl, target, s0=1.0 / norm(step_d))
    decision = Decision(True, point.s, eps, target, 0,
                        f"lean step (prereg departure 43): KL {point.kl:.4g} a decision along {along}, unread",
                        [Rung(1.0, target, point.s, point.kl, True, {})])
    readout["decision"] = {**decision.summary(), "lean": True}
    readout["lean"] = {"along": along, "edge": edge, "kl_evals": point.evals}
    readout["heldout"] = {"game_sides": len(sides), "decisions_per_side": decisions_per_side(sides)}
    return readout, d, point.s, step_d


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
    args = ap.parse_args(argv)
    if args.lean and args.adopt_from is not None:
        ap.error("--lean takes no --adopt-from: round 1's step is stage C's")
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
                                      args.average_with, lean=args.lean)
    if args.direction_out is not None:
        torch.save(d, args.direction_out)
        if step_d is not d:              # what rebuilds an averaged candidate (the code review's L4)
            averaged_out = args.direction_out.with_name("direction_average.pt")
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
