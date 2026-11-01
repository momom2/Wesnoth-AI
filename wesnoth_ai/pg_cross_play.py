"""A rung read by a cross-play match (docs/training_design_20261019/current_design.md,
§3 "Range" and the checks table: "Heavy-tailed ratios (a rare action
raised): that rung is read by a cross-play match"; prereg departure 23 in
docs/stage_c_rounds_local_prereg_20261030.md).

pg_step.decide keeps pi_k when the rung it chose on half A fails the
importance-weight check on half B. That is the cross-play case. The rung's
network theta_k + s d (tools/pg_round.py --chosen-network) plays pi_k at
temperature 1, both at the round's memory (scripts/rounds_local.py), and
the candidate's mean z over the match replaces half B's reading in decide's
rule. It is in E[z], ΔĴ's unit (pi_k against itself reads 0). The rung is
adopted unless the match reads below -REVERT_SE SE; otherwise pi_k stays
and the radius halves. The decision pg_step took is kept under
"importance".

The match is read as tools/pg_planted_play.py reads the link: a win 1, a
loss -1, a cap (the turn or action horizon, or no progress) 0, and a game
killed on the wall clock left out and counted. The candidate holds label A
in every game, at its checkpoint, memory and raw:t1; a game that does not
is refused and counted.

Stdlib only: scripts/rounds_local.py, which imports no torch, reads the
match and amends the round in its own process.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

REVERT_SE = 2.0                 # wesnoth_ai/pg_step.py's, which tests/test_pg_cross_play.py holds equal
DECIDED = {"win": 1.0, "loss": -1.0}
CAPPED = ("draw", "timeout")    # the game's own horizons: z 0 (tools/pg_planted_play.py's)
PROCEDURE = "raw:t1"
RESULT_SUFFIX = ".json"
RECORD_SUFFIX = ".game.jsonl.gz"


def cross_play_case(decision: Mapping, rungs: Sequence[Mapping], confirm: str = "B") -> Optional[int]:
    """The rung decide left to a cross-play match, from the decision's and
    its rungs' summaries (pg_step.Decision.summary): the chosen rung, not
    adopted, whose importance weights fail the check on the confirmation
    half. None when the decision is not that case."""
    chosen = decision.get("chosen")
    if decision.get("adopt") or chosen is None or not 0 <= int(chosen) < len(rungs):
        return None
    half = (rungs[int(chosen)].get("halves") or {}).get(confirm)
    return int(chosen) if half is not None and half.get("ess_ok") is False else None


def _results(games: Path) -> List[Dict]:
    out = []
    for p in sorted(Path(games).glob("game_*" + RESULT_SUFFIX)):
        if not p.name.endswith(RECORD_SUFFIX):
            out.append(json.loads(p.read_text(encoding="utf-8")) | {"_name": p.name})
    return out


def mean_z(games: Path, label: str, sha256: str, slots: int) -> Dict:
    """The candidate's mean z over a cross-play match (module docstring),
    its SE over the games read, and what was left out or refused."""
    zs: List[float] = []
    left_out, refused = 0, []
    for r in _results(games):
        if (r.get("label_a") != label or r.get("checkpoint_sha256_a") != sha256 or r.get("memory_a") != slots
                or r.get("procedure_a") != PROCEDURE or r.get("procedure_b") != PROCEDURE):
            refused.append(f"{r['_name']}: label A {r.get('label_a')} played {str(r.get('checkpoint_sha256_a'))[:12]} "
                           f"at {r.get('memory_a')} slots under {r.get('procedure_a')} against {r.get('procedure_b')}")
            continue
        outcome = r.get("outcome_a")
        if outcome in DECIDED or outcome in CAPPED:
            zs.append(DECIDED.get(outcome, 0.0))
        else:
            left_out += 1
    n = len(zs)
    mean = sum(zs) / n if n else math.nan
    se = math.sqrt(sum((z - mean) ** 2 for z in zs) / (n - 1) / n) if n > 1 else math.nan
    return {"label": label, "sha256": sha256, "estimate": mean, "se": se, "games": n, "games_left_out": left_out,
            "games_refused": len(refused), "refused_examples": refused[:5]}


def decide_by_match(decision: Mapping, rungs: Sequence[Mapping], reading: Mapping) -> Dict:
    """The decision once the cross-play case's rung is read by the match
    (module docstring): `reading` is mean_z's."""
    chosen = cross_play_case(decision, rungs)
    if chosen is None:
        raise ValueError("the decision is not the cross-play case: no rung to read by a match")
    est, se, eps = float(reading["estimate"]), float(reading["se"]), float(decision["eps"])
    out = {**decision, "importance": {k: decision[k] for k in ("adopt", "s", "next_eps", "reason") if k in decision},
           "cross_play": dict(reading)}
    if not se > 0:
        out.update(adopt=False, s=0.0, next_eps=eps / 2.0, reason=f"the cross-play match reads no SE ({se})")
    elif est < -REVERT_SE * se:
        out.update(adopt=False, s=0.0, next_eps=eps / 2.0,
                   reason=f"the cross-play match reads {est:.4g} below -{REVERT_SE:g} SE ({se:.3g})")
    else:
        rung = rungs[chosen]
        out.update(adopt=True, s=float(rung["s"]), next_eps=float(rung["target_kl"]),
                   reason=f"the cross-play match reads {est:.4g} (SE {se:.3g})")
    return out
