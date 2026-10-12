#!/usr/bin/env python3
"""The look-ahead gate's helpers (scripts/lookahead_gate_box.sh): an arm's
configuration made ready to play, its evaluator's scale calibrated on timed
decisions, and a match's look-ahead telemetry pooled into one small record.

    python tools/lookahead_gate.py ensure configs/lookahead_material_gate.json
    python tools/lookahead_gate.py calibrate configs/lookahead_exposure_gate.json exposure.decision_timing.json
    python tools/lookahead_gate.py summarize eval_games/la_material --out la_material.lookahead.json

`ensure` loads a look-ahead configuration (wesnoth_ai/lookahead_config.py)
and prints its procedure tag at the reference decode's end_turn offset; it
refuses a scale still to calibrate unless told it runs before the
calibration (`--before-calibration`). A critic evaluator names its
checkpoint three ways:

    "checkpoint"         the local path the player loads (relative to the
                         working directory, as the player reads it);
    "checkpoint_hf"      its path on the model host, in "hf_repo" when the
                         evaluator names one, else the reference's
                         repository (configs/reference_player.json);
    "checkpoint_sha256"  its SHA-256.

The file is fetched when it is missing and its SHA-256 checked either way;
a fetched file that does not match is not kept.

`calibrate` replaces a scale to calibrate ("calibrate_scale": true, the
material_exposure evaluator's) by the median |M| over the states the timed
decisions valued (tools/lookahead_timing.py's `m_abs_median`), rewriting
the configuration in place with "calibrate_scale" false and the
measurement's size; a configuration with nothing to calibrate is left as it
is. Run on every entry of a box, it reads the same timing record and writes
the same scale.

`summarize` pools the per-game telemetry (`lookahead_telemetry_<side>`,
tools/lookahead_player.GameTelemetry) of every game result in a match
directory. Rates are pooled over decisions, not averaged over games: the
flip rate of a kind is the decisions of that kind played otherwise than the
prior's argmax over every decision whose argmax was of that kind.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.eval_provenance import file_sha256  # noqa: E402
from wesnoth_ai.lookahead_config import (  # noqa: E402
    LookaheadConfig, config_from_dict, load_config, procedure_tag)

SUMMED = ("decisions", "operated", "states", "terminal_states", "forwards",
          "seconds", "seconds_prior", "seconds_expand", "seconds_evaluate")
SUMMED_BY_KEY = ("by_kind", "flips_by_kind", "flips_to_kind", "candidates_by_kind", "failed", "evaluator_counts")
MAXIMA = ("states_max", "seconds_max")

Download = Callable[[str, str], str]    # (repo, path on the host) -> a local file


# ---- ensure ------------------------------------------------------------------

def _hf_download(repo: str, path: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path)


def ensure_critic(evaluator: Dict, default_repo: str, download: Optional[Download] = None) -> Path:
    """The critic checkpoint `evaluator` names, fetched when missing, its
    SHA-256 the one the configuration records (module docstring)."""
    for key in ("checkpoint", "checkpoint_hf", "checkpoint_sha256"):
        if not evaluator.get(key):
            raise ValueError(f"a critic evaluator on a box names its {key} (tools/lookahead_gate.py)")
    path = Path(evaluator["checkpoint"])
    want = str(evaluator["checkpoint_sha256"]).lower()
    if not path.exists():
        source = Path((download or _hf_download)(evaluator.get("hf_repo") or default_repo,
                                                 evaluator["checkpoint_hf"]))
        got = file_sha256(source)
        if got != want:
            raise ValueError(f"{evaluator['checkpoint_hf']} has SHA-256 {got}, the configuration "
                             f"records {want}; not kept")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        shutil.copyfile(source, partial)
        partial.replace(path)
    got = file_sha256(path)
    if got != want:
        raise ValueError(f"{path} has SHA-256 {got}, the configuration records {want}")
    return path


def ensure(config_path: Path, end_turn_offset: float, default_repo: str,
           download: Optional[Download] = None, before_calibration: bool = False) -> str:
    """The procedure tag of the configuration at `config_path`, its critic
    checkpoint (if any) on the disk and checked, its scale calibrated
    unless `before_calibration`."""
    cfg: LookaheadConfig = load_config(config_path)
    if cfg.scale_to_calibrate and not before_calibration:
        raise ValueError("its evaluator's scale is a placeholder still to calibrate "
                         "(tools/lookahead_gate.py calibrate)")
    if cfg.evaluator_name == "critic":
        ensure_critic(cfg.evaluator, default_repo, download)
    return procedure_tag(cfg, end_turn_offset)


# ---- calibrate -----------------------------------------------------------------

def calibrate(config_path: Path, timing_path: Path) -> Optional[float]:
    """The scale written into the configuration at `config_path` from the
    timing record at `timing_path` (module docstring); None when the
    configuration has no scale to calibrate (the timing is not read)."""
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not config_from_dict(raw).scale_to_calibrate:
        return None
    timing = json.loads(Path(timing_path).read_text(encoding="utf-8"))
    median, states = timing.get("m_abs_median"), int(timing.get("m_states") or 0)
    if states < 1 or not isinstance(median, (int, float)) or not median > 0:
        raise ValueError(f"{timing_path} records no positive median |M| (m_states {states}, "
                         f"m_abs_median {median!r})")
    evaluator = raw["evaluator"]
    evaluator.update({"scale": float(median), "calibrate_scale": False,
                      "scale_measured": {"states": states,
                                         "decisions": int(timing["telemetry"]["decisions"]),
                                         "placeholder": evaluator["scale"]}})
    config_from_dict(raw)
    partial = Path(config_path).with_name(Path(config_path).name + ".partial")
    partial.write_bytes((json.dumps(raw, indent=2) + "\n").encode("utf-8"))
    partial.replace(config_path)
    return float(median)


# ---- summarize -----------------------------------------------------------------

def _ratio(num: float, den: float) -> Optional[float]:
    return round(num / den, 6) if den else None


def summarize(outdir: Path, side: str = "a") -> Dict:
    """The look-ahead telemetry of side `side` pooled over the game results
    in `outdir` (module docstring)."""
    key = f"lookahead_telemetry_{side}"
    totals: Dict[str, float] = dict.fromkeys(SUMMED, 0)
    by_key: Dict[str, Counter] = {k: Counter() for k in SUMMED_BY_KEY}
    maxima: Dict[str, float] = dict.fromkeys(MAXIMA, 0)
    outcomes: Counter = Counter()
    records, procedures = set(), set()
    games = without = with_failures = 0
    for path in sorted(Path(outdir).glob("game_*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        outcomes[str(result.get("outcome_a"))] += 1
        tel = result.get(key)
        if not tel:
            without += 1
            continue
        games += 1
        records.add(json.dumps(result.get(f"lookahead_{side}"), sort_keys=True))
        procedures.add(result.get(f"procedure_{side}"))
        for k in SUMMED:
            totals[k] += tel.get(k, 0)
        for k in SUMMED_BY_KEY:
            by_key[k].update(tel.get(k) or {})
        for k in MAXIMA:
            maxima[k] = max(maxima[k], tel.get(k, 0))
        with_failures += bool(tel.get("failed"))
    if len(records) > 1 or len(procedures) > 1:
        raise ValueError(f"{outdir} holds games of several look-ahead configurations: {sorted(map(str, procedures))}")
    d, n_op = totals["decisions"], totals["operated"]
    flips = sum(by_key["flips_by_kind"].values())
    # The decision kinds as the telemetry lists them (every kind of
    # tools/raw_player.KIND_NAMES, zeros included).
    kinds = list(by_key["by_kind"])
    return {
        "outdir": Path(outdir).name,
        "side": side,
        "procedure": next(iter(procedures), None),
        "lookahead": json.loads(next(iter(records))) if records else None,
        "games": games,
        "games_without_telemetry": without,
        "outcomes_a": dict(sorted(outcomes.items())),
        "decisions": int(d),
        "operated": int(n_op),
        "operated_share": _ratio(n_op, d),
        "flips": int(flips),
        "flip_rate": _ratio(flips, d),
        "flip_rate_operated": _ratio(flips, n_op),
        "decisions_by_kind": {k: int(by_key["by_kind"][k]) for k in kinds},
        "flips_by_kind": {k: int(by_key["flips_by_kind"][k]) for k in kinds},
        "flip_rate_by_kind": {k: _ratio(by_key["flips_by_kind"][k], by_key["by_kind"][k]) for k in kinds},
        "flips_to_kind": {k: int(by_key["flips_to_kind"][k]) for k in kinds},
        "candidates_by_kind": {k: int(by_key["candidates_by_kind"][k]) for k in kinds},
        "candidates_per_operated": _ratio(sum(by_key["candidates_by_kind"].values()), n_op),
        "states_per_decision": _ratio(totals["states"], d),
        "states_per_operated": _ratio(totals["states"], n_op),
        "states_max": int(maxima["states_max"]),
        "terminal_states": int(totals["terminal_states"]),
        "forwards_per_decision": _ratio(totals["forwards"], d),
        "seconds_per_decision": {"total": _ratio(totals["seconds"], d),
                                 "prior": _ratio(totals["seconds_prior"], d),
                                 "expand": _ratio(totals["seconds_expand"], d),
                                 "evaluate": _ratio(totals["seconds_evaluate"], d)},
        "seconds_max": round(float(maxima["seconds_max"]), 6),
        "failed": dict(sorted(by_key["failed"].items())),
        "games_with_failures": with_failures,
        "evaluator_counts": dict(sorted(by_key["evaluator_counts"].items())),
    }


# ---- command line ------------------------------------------------------------------

def _reference() -> Dict:
    from tools.reference_player import load
    return load()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    en = sub.add_parser("ensure", help="check a configuration, fetch its critic, print its procedure tag")
    en.add_argument("config", type=Path)
    en.add_argument("--end-turn-offset", type=float, default=None,
                    help="the raw decode's end_turn offset (default: the reference's)")
    en.add_argument("--before-calibration", action="store_true",
                    help="accept a scale still to calibrate (the timed decisions that calibrate it)")
    ca = sub.add_parser("calibrate", help="write the timed decisions' median |M| into a scale to calibrate")
    ca.add_argument("config", type=Path)
    ca.add_argument("timing", type=Path)
    su = sub.add_parser("summarize", help="pool a match's look-ahead telemetry")
    su.add_argument("outdir", type=Path)
    su.add_argument("--side", choices=("a", "b"), default="a")
    su.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.command == "ensure":
        ref = _reference()
        offset = args.end_turn_offset
        if offset is None:
            offset = float(ref["decode"].get("raw_end_turn_offset", 0.0))
        try:
            print(ensure(args.config, offset, ref["hf_repo"], before_calibration=args.before_calibration))
        except (OSError, ValueError) as e:
            raise SystemExit(f"look-ahead config {args.config}: {e}") from e
        return 0
    if args.command == "calibrate":
        try:
            scale = calibrate(args.config, args.timing)
        except (OSError, ValueError, KeyError) as e:
            raise SystemExit(f"look-ahead config {args.config}: {e}") from e
        print("nothing to calibrate" if scale is None else f"scale {scale:g}")
        return 0
    summary = summarize(args.outdir, args.side)
    text = json.dumps(summary, indent=1)
    if args.out is not None:
        partial = args.out.with_name(args.out.name + ".partial")
        partial.write_text(text + "\n", encoding="utf-8")
        partial.replace(args.out)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
