"""Procedure provenance for eval results -- stdlib-only on purpose:
run_elo_batch is a long-lived torch-free driver, and importing the
tag helper from elo_eval_game pulled torch + the sim stack into it
(round-5 C9: +174 MB RSS in the batch process).

The raw player's decode knobs (tools/raw_player.py,
docs/decode_knobs_prereg_20261012.md) live here too, so the two drivers
parse, forward and tag them alike: each is a logit offset on the compact
distribution before the joint choice, a per-side float flag, recorded in
the result file always and in the procedure tag when it is not zero."""
from __future__ import annotations

import re
from typing import Dict, Optional

# (knob, tag code, flag stem): `--raw-attack-offset-a` reaches
# `args.raw_attack_offset_a`; the tag carries `+ao<x>`. The end_turn offset
# (`+eo<x>`) predates them and keeps its own flags.
RAW_KNOBS = (
    ("attack_offset", "ao", "--raw-attack-offset"),
    ("recruit_offset", "ro", "--raw-recruit-offset"),
    ("attack_tod_slope", "at", "--raw-attack-tod-slope"),
)
RAW_KNOB_HELP = {
    "attack_offset": "Offset added to every attack's logit in the raw player's choice "
                     "(procedure tag '+ao<x>').",
    "recruit_offset": "Offset added to every recruit's logit in the raw player's choice "
                      "(procedure tag '+ro<x>').",
    "attack_tod_slope": "An attack's logit gains this times the attacker's time-of-day "
                        "damage modifier (-0.25 to +0.25) at the hex it strikes from "
                        "(procedure tag '+at<x>').",
}


def add_raw_knob_args(ap) -> None:
    """The knob flags of both sides on an argparse parser."""
    for side in ("a", "b"):
        for name, _code, stem in RAW_KNOBS:
            ap.add_argument(f"{stem}-{side}", type=float, default=0.0,
                            help=RAW_KNOB_HELP[name] if side == "a"
                            else f"Player B (see {stem}-a).")


def raw_knobs_of(args, side: str) -> Dict[str, float]:
    """Side `side`'s knob values from parsed arguments."""
    return {name: float(getattr(args, f"raw_{name}_{side}")) for name, _c, _s in RAW_KNOBS}


def raw_knob_argv(knobs: Dict[str, float], side: str) -> list:
    """The flags that hand `knobs` to a child, the zero ones left out."""
    out = []
    for name, _code, stem in RAW_KNOBS:
        if knobs.get(name):
            out += [f"{stem}-{side}", str(float(knobs[name]))]
    return out


def raw_knob_fields(args) -> Dict[str, float]:
    """The result-file fields of both sides' knobs (`raw_attack_offset_a`, ...)."""
    return {f"raw_{name}_{side}": float(getattr(args, f"raw_{name}_{side}"))
            for side in ("a", "b") for name, _c, _s in RAW_KNOBS}


# A knob vector in config form: these keys of a decode block
# (configs/reference_player.json "decode", a search's vector or winner
# file), an absent key meaning 0.
KNOB_VECTOR_KEYS = ("raw_end_turn_offset",) + tuple(f"raw_{name}" for name, _c, _s in RAW_KNOBS)


def knob_vector(decode: dict) -> Dict[str, float]:
    """The knob vector of a decode block, every key present (absent = 0).
    Refuses a raw_* key no decode reads, which would otherwise play as 0."""
    known = set(KNOB_VECTOR_KEYS) | {"raw_temperature", "raw_end_turn"}
    unknown = sorted(k for k in decode if k.startswith("raw_") and k not in known)
    if unknown:
        raise ValueError(f"decode keys no decode reads: {unknown} (known: {sorted(known)})")
    return {k: float(decode.get(k, 0.0) or 0.0) for k in KNOB_VECTOR_KEYS}


def decode_knobs(decode: dict) -> Dict[str, float]:
    """The `raw_knobs` mapping (RAW_KNOBS names) of a decode block."""
    vec = knob_vector(decode)
    return {name: vec[f"raw_{name}"] for name, _c, _s in RAW_KNOBS}


def decode_flags(side: str, decode: dict) -> list:
    """run_elo_batch flags that play a decode block on `side`: the end_turn
    rule and offset and the knobs, each only when it differs from the
    default. The temperature and the search budget are per-match flags."""
    flags = []
    if decode.get("raw_end_turn", "joint") != "joint":
        flags += [f"--raw-end-turn-{side}", str(decode["raw_end_turn"])]
    offset = knob_vector(decode)["raw_end_turn_offset"]
    if offset:
        flags += [f"--raw-end-turn-offset-{side}", str(offset)]
    return flags + raw_knob_argv(decode_knobs(decode), side)


def decode_procedure(decode: dict) -> str:
    """The procedure tag a decode block plays under (mcts_sims 0)."""
    return procedure_of(int(decode.get("mcts_sims", 0)), False, False, decode.get("raw_temperature"),
                        raw_end_turn=decode.get("raw_end_turn", "joint"),
                        raw_end_turn_offset=knob_vector(decode)["raw_end_turn_offset"],
                        raw_knobs=decode_knobs(decode))


def end_turn_refusal(side: str, spec: str, sims: int, raw_temperature,
                     raw_end_turn: str, raw_end_turn_offset: float,
                     raw_knobs: Optional[Dict[str, float]] = None) -> Optional[str]:
    """Why side `side`'s decode flags cannot apply, or None. The end_turn
    rule, its offset and the decode knobs are knobs of the
    joint-temperature raw player (tools/raw_player.py), which a side plays
    only at sims 0 with a raw temperature and a network. The search, the
    legacy sampler and the scripted 'dummy' never read them, while the
    result file recorded them as if they had played."""
    knobs_on = any((raw_knobs or {}).values())
    if raw_end_turn == "joint" and not raw_end_turn_offset and not knobs_on:
        return None
    if sims > 0 or raw_temperature is None or spec == "dummy":
        return (f"--raw-end-turn-{side}/--raw-end-turn-offset-{side} and the decode knobs "
                f"(--raw-attack-offset-{side}, --raw-recruit-offset-{side}, "
                f"--raw-attack-tod-slope-{side}) decode the joint-temperature raw player "
                f"only: side {side} needs sims 0, --raw-temperature-{side} and a network "
                f"(the search, the legacy sampler and 'dummy' never read them).")
    return None


def procedure_of(sims: int, plan: bool, no_turn_search: bool,
                 raw_temperature=None, gumbel_root: bool = True, *,
                 raw_end_turn: str = "joint", raw_end_turn_offset: float = 0.0,
                 raw_knobs: Optional[Dict[str, float]] = None) -> str:
    """Canonical procedure tag for result provenance. Carries the
    sims budget (round-13 C2: 'mcts' alone let an outdir silently
    mix --mcts-sims 16 and 32 games -- different estimands), the
    root procedure of a plain search ('mcts' = Gumbel root, 'puct' =
    plain PUCT root, what the az legs trained with) and, for the raw
    player, its joint sampling temperature (tools/raw_player.py;
    None = the legacy factored sampler), its end_turn decode
    ('+endm' = decided at the actor level, '+eo<x>' = the end_turn
    logit offset x; panel test 1) and its decode knobs ('+ao', '+ro',
    '+at', RAW_KNOBS), so those estimands never share an outdir either."""
    if sims <= 0:
        if raw_temperature is None:
            return "raw"
        tag = f"raw:t{float(raw_temperature):g}"
        if raw_end_turn == "actor":
            tag += "+endm"
        if raw_end_turn_offset:
            tag += f"+eo{float(raw_end_turn_offset):g}"
        for name, code, _stem in RAW_KNOBS:
            value = (raw_knobs or {}).get(name)
            if value:
                tag += f"+{code}{float(value):g}"
        return tag
    if plan:
        name = "plan_tournament"
    elif no_turn_search:
        name = "mcts" if gumbel_root else "puct"
    else:
        name = "tcs"
    return f"{name}:{sims}"


_SUFFIX = re.compile(r"\+(?=[a-z])")
_CODES = {code: name for name, code, _s in RAW_KNOBS}


def parse_raw_procedure(tag: str) -> dict:
    """The raw player's decode from its procedure tag (`procedure_of`'s
    inverse): temperature, end_turn rule and offset, and every knob, the
    absent ones 0. Raises ValueError on a tag that is not a raw player's
    at a temperature, or on a suffix no decode writes."""
    head, *suffixes = _SUFFIX.split(tag)
    if not head.startswith("raw:t"):
        raise ValueError(f"{tag!r} is not a raw player's procedure at a temperature")
    out = {"raw_temperature": float(head[len("raw:t"):]), "raw_end_turn": "joint",
           "raw_end_turn_offset": 0.0, **{name: 0.0 for name, _c, _s in RAW_KNOBS}}
    for suffix in suffixes:
        if suffix == "endm":
            out["raw_end_turn"] = "actor"
        elif suffix[:2] == "eo":
            out["raw_end_turn_offset"] = float(suffix[2:])
        elif suffix[:2] in _CODES:
            out[_CODES[suffix[:2]]] = float(suffix[2:])
        else:
            raise ValueError(f"{tag!r}: unknown decode suffix {suffix!r}")
    return out
