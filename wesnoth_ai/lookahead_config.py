"""The look-ahead player's configuration (tools/lookahead_player.py,
docs/selfplay_program_20261008.md "The rounds", step 3) and its procedure
tag. Standard library only: run_elo_batch reads it without torch.

A configuration file (configs/lookahead.json) holds

  k                the prior's top k actions are the candidates (0: the
                   operator is off and the player is the raw player);
  c                the clip of the advantage term, in nats (0: the
                   operator runs and changes nothing, the null control);
  sigma            the fixed scale the advantage is divided by;
  kinds            "all", or "attacks": the candidates are then the prior's
                   argmax and the attacks among the top k;
  determinization  "observed" (the world the side sees, the fair player)
                   or "godview" (the true state, an upper bound whose tag
                   carries "+godview");
  evaluator        {"name": "material", "hp_scale": ...},
                   {"name": "critic", "checkpoint": ..., "view": "obs" or
                   "true", "device": "cpu" or "cuda", "batch": ...}, or
                   {"name": "rollout"} (an interface, not implemented);
                   a critic played on a box also names "checkpoint_hf"
                   and "checkpoint_sha256" (tools/lookahead_gate.py ensure);
  max_attack_leaves  hit/miss sequences an attack may take before its
                   expansion is counted as failed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

KINDS = ("all", "attacks")
DETERMINIZATIONS = ("observed", "godview")
EVALUATORS = ("material", "critic", "rollout")
CRITIC_VIEWS = ("obs", "true")
GODVIEW_SUFFIX = "+godview"
TAG_PREFIX = "la:"


@dataclass(frozen=True)
class LookaheadConfig:
    k: int
    c: float
    sigma: float
    kinds: str = "all"
    determinization: str = "observed"
    evaluator: Dict[str, Any] = field(default_factory=lambda: {"name": "material", "hp_scale": 250.0})
    max_attack_leaves: int = 512

    def __post_init__(self):
        if int(self.k) < 0:
            raise ValueError(f"k must be >= 0, got {self.k}")
        if float(self.c) < 0:
            raise ValueError(f"c must be >= 0, got {self.c}")
        if not float(self.sigma) > 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        if self.kinds not in KINDS:
            raise ValueError(f"kinds must be one of {KINDS}, got {self.kinds!r}")
        if self.determinization not in DETERMINIZATIONS:
            raise ValueError(f"determinization must be one of {DETERMINIZATIONS}, got {self.determinization!r}")
        name = self.evaluator.get("name")
        if name not in EVALUATORS:
            raise ValueError(f"evaluator name must be one of {EVALUATORS}, got {name!r}")
        if name == "material" and not float(self.evaluator.get("hp_scale", 0)) > 0:
            raise ValueError("the material evaluator needs hp_scale > 0")
        if name == "critic":
            if self.evaluator.get("view") not in CRITIC_VIEWS:
                raise ValueError(f"the critic's view must be one of {CRITIC_VIEWS}")
            if not self.evaluator.get("checkpoint"):
                raise ValueError("the critic evaluator needs a checkpoint")
            if self.evaluator.get("device", "cpu") not in ("cpu", "cuda"):
                raise ValueError("the critic's device must be cpu or cuda")
        if int(self.max_attack_leaves) < 1:
            raise ValueError("max_attack_leaves must be >= 1")

    @property
    def evaluator_name(self) -> str:
        return str(self.evaluator["name"])

    @property
    def godview(self) -> bool:
        return self.determinization == "godview"


def config_from_dict(d: Dict[str, Any]) -> LookaheadConfig:
    """A configuration from its dict; keys starting with "_" are comments."""
    known = {"k", "c", "sigma", "kinds", "determinization", "evaluator", "max_attack_leaves"}
    body = {k: v for k, v in d.items() if not k.startswith("_")}
    unknown = sorted(set(body) - known)
    if unknown:
        raise ValueError(f"unknown look-ahead config keys {unknown}")
    if "evaluator" in body:
        body["evaluator"] = {k: v for k, v in body["evaluator"].items() if not k.startswith("_")}
    return LookaheadConfig(**body)


def load_config(path) -> LookaheadConfig:
    return config_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def evaluator_tag(cfg: LookaheadConfig) -> str:
    """"material", "critic.obs", "critic.true" or "rollout"."""
    if cfg.evaluator_name == "critic":
        return f"critic.{cfg.evaluator['view']}"
    return cfg.evaluator_name


def procedure_tag(cfg: LookaheadConfig, raw_end_turn_offset: float = 0.0) -> str:
    """The procedure tag of a look-ahead player: "la:<evaluator>:k<k>c<c>s<sigma>",
    then "+atk" for an attacks-only operator, the raw decode's end_turn
    offset as the raw tag writes it ("+eo-1.5"), and "+godview" for the
    true-state upper bound."""
    tag = f"{TAG_PREFIX}{evaluator_tag(cfg)}:k{int(cfg.k)}c{float(cfg.c):g}s{float(cfg.sigma):g}"
    if cfg.kinds == "attacks":
        tag += "+atk"
    if raw_end_turn_offset:
        tag += f"+eo{float(raw_end_turn_offset):g}"
    if cfg.godview:
        tag += GODVIEW_SUFFIX
    return tag


def config_record(cfg: LookaheadConfig, critic_sha256: Optional[str] = None) -> Dict[str, Any]:
    """What a result records of a look-ahead player's configuration (an
    estimand field): every knob, and the critic checkpoint's SHA-256 in
    place of its path."""
    out = asdict(cfg)
    ev = dict(out["evaluator"])
    if ev.get("name") == "critic":
        ev.pop("checkpoint", None)
        ev["checkpoint_sha256"] = critic_sha256
    out["evaluator"] = ev
    return out
