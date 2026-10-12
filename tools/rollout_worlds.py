"""The re-grade's worlds as tools/rollout_regrade.py runs them
(docs/belief_worlds_prereg_20261012.md; the worlds themselves are
wesnoth_ai/belief_worlds.py): which world a run plays, what each candidate
gives its world, and where an arm's records go."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class ArmInputs:
    """The world a run's rollouts start from (wesnoth_ai/belief_worlds.py
    ARMS) and, for `roster_belief`, the policy whose belief head it reads."""
    world: str = "observed"
    belief_policy: object = None


def arm_inputs(spec: Dict, base) -> ArmInputs:
    """The run's world, with the policy RB reads the belief head of: the
    local base itself, or behind a shared server (which returns no
    belief) a local CPU copy of the checkpoint."""
    world = spec.get("world", "observed")
    if world != "roster_belief":
        return ArmInputs(world)
    if not spec.get("address"):
        return ArmInputs(world, base)
    import torch
    from tools.eval_players import _load_policy
    return ArmInputs(world, _load_policy(Path(spec["checkpoint"]), torch.device("cpu"), label="belief"))


def world_inputs(arm: ArmInputs, boundary, pre, mover: int) -> Dict:
    """What the arm's world needs of a candidate, as build_arm_world's
    keywords: the mover's gold and base income at its turn start (the
    position's boundary, after its init_side), its sighting record before
    the candidate's end_turn (`pre`), and the belief head's read there."""
    from wesnoth_ai.belief_worlds import ESTIMATED_ARMS, belief_by_hex
    from wesnoth_ai.lookahead_world import side_economy
    out: Dict = {}
    if arm.world in ESTIMATED_ARMS:
        out["turn_start"] = side_economy(boundary.core, mover)
    if arm.world == "last_seen":
        out["sightings"] = [tuple(r) for r in pre.core.sightings_export(mover)]
    if arm.world == "roster_belief":
        out["belief"] = belief_by_hex(arm.belief_policy, pre)
    return out


def inputs_record(inputs: Dict, seconds: float) -> Dict:
    """The candidate record's account of its world's inputs."""
    out: Dict = {"world_inputs_seconds": round(seconds, 4)}
    if "turn_start" in inputs:
        out["turn_start"] = list(inputs["turn_start"])
    if "sightings" in inputs:
        out["sightings"] = len(inputs["sightings"])
    if "belief" in inputs:
        probs = list(inputs["belief"].values())
        out["belief"] = {"tokens": len(probs), "mass": round(float(sum(probs)), 4)}
    return out


def out_files(out: Path, world: str) -> Tuple[Path, Path]:
    """(rollouts, summary) of a world's run in `out`: the observed world
    keeps the re-grade's names."""
    if world == "observed":
        return out / "rollouts.jsonl", out / "regrade_summary.json"
    return out / f"rollouts_{world}.jsonl", out / f"regrade_summary_{world}.json"


def record_world(rec: Dict) -> str:
    """The world a record was played in (the re-grade's records name none:
    the observed world)."""
    return rec.get("world", "observed")
