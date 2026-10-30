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

round.json also logs what design §6 asks of each round:
- Ĝ and ΔĴ at the three radii, and the half-B reading (the decision's
  rungs);
- the KL per side-game;
- the cosine of d with the previous round's direction (`--previous`, its
  direction.pt), which reads oscillation directly, since nothing carries
  momentum.

    python tools/pg_round.py --pass pass.pt --factory tools.pg_eval:network_factory \\
        --factory-arg checkpoint=pi_k.pt --factory-arg games=DIR,... --factory-arg labels=labels.jsonl \\
        --eps-from ../round_01/round.json --arm reduced_adamw --out round.json --candidate pi_k1.pt \\
        --direction-out direction.pt [--previous ../round_01/direction.pt]
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from wesnoth_ai.pg_step import ARMS, DEFAULT_ARM, decisions_per_side, direction, dot, ladder, norm  # noqa: E402


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


def _write_json(path: Path, obj: Dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run_round(pass_: Dict, params: Dict[str, torch.Tensor], factory, arm: str, eps: float,
              previous: Optional[Dict[str, torch.Tensor]] = None, adopt: Optional[Dict] = None) -> tuple:
    """(readout, d, s or None): the round's direction, its ladder (or
    stage C's decision, `adopt`, taken as read) and the step to write."""
    d = direction(pass_, arm, params)
    size = norm(d)
    readout: Dict = {"arm": arm, "eps": eps, "direction_norm": size,
                     "pass": {k: pass_.get(k) for k in ("games", "decisions", "windows", "checkpoint_sha256")}}
    if previous is not None:
        common = [n for n in d if n in previous]
        readout["cosine_previous"] = (dot({n: d[n] for n in common}, {n: previous[n] for n in common})
                                      / (size * norm({n: previous[n] for n in common}) or math.nan))
    if adopt is not None:
        readout["decision"] = {**adopt, "from": "stage C's ladder (round 1's step, design §6)"}
        return readout, d, (float(adopt["s"]) if adopt.get("adopt") else None)
    if size == 0:
        readout["decision"] = {"adopt": False, "s": 0.0, "eps": eps, "next_eps": eps / 2.0, "chosen": None,
                               "reason": "the direction is zero (every label 0)", "rungs": []}
        return readout, d, None
    decision = ladder(factory, d, eps, s0=1.0 / size)
    readout["decision"] = decision.summary()
    sides = factory(d).game_sides()
    readout["heldout"] = {"game_sides": len(sides), "decisions_per_side": decisions_per_side(sides)}
    return readout, d, (decision.s if decision.adopt else None)


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
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True, help="where an adopted rung's network is written")
    ap.add_argument("--direction-out", type=Path, default=None)
    args = ap.parse_args(argv)
    module, _, name = args.factory.partition(":")
    factory, params, sha = getattr(importlib.import_module(module), name)(**_factory_args(args.factory_arg))
    pass_ = load_pass(args.pass_path)
    if sha is not None and sha != pass_["checkpoint_sha256"]:
        raise SystemExit(f"the evaluator's checkpoint {sha} is not the pass's {pass_['checkpoint_sha256']}")
    adopt = None
    if args.adopt_from is not None:
        stage_c = json.loads(args.adopt_from.read_text(encoding="utf-8"))
        ladder_readout = stage_c.get("ladder")
        if not ladder_readout:
            raise SystemExit(f"{args.adopt_from} holds no ladder: stage C did not read round 1's step")
        adopt = {k: ladder_readout.get(k) for k in ("adopt", "s", "eps", "next_eps", "chosen", "reason")}
        arm = ladder_readout.get("arm") or arm_of(args.adopt_from)
        if arm == "probe_span":
            raise SystemExit("stage C's ladder stepped along probe_span, whose direction needs stage C's span "
                             "Fisher: run round 1's ladder anew with --eps-from stage_c.json and --arm probe_span")
        eps = float(ladder_readout["eps"])
    else:
        arm = args.arm or (arm_of(args.arm_from) if args.arm_from else DEFAULT_ARM)
        eps = args.eps if args.eps is not None else eps_of(args.eps_from)
    previous = torch.load(args.previous, map_location="cpu", weights_only=True) if args.previous else None
    readout, d, s = run_round(pass_, params, factory, arm, eps, previous, adopt)
    if args.direction_out is not None:
        torch.save(d, args.direction_out)
    readout["candidate"] = None
    if s is not None:
        sha_out = factory(d).write_candidate(s, args.candidate)
        readout["candidate"] = {"path": str(args.candidate), "sha256": sha_out, "s": s}
    _write_json(args.out, readout)
    dec = readout["decision"]
    print(f"PG_ROUND adopt={dec['adopt']} s={dec['s']:.4g} eps={eps:.4g} next_eps={dec['next_eps']:.4g} "
          f"arm={arm}: {dec['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
