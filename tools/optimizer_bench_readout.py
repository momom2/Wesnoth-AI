"""The optimizer bench's readout (scripts/optimizer_bench_box.sh): passes from
scratch over the same positions in the same order under the same seed, one
per optimizer and learning rate, compared on their holdout probes over the
same holdout game-sides.

    python tools/optimizer_bench_readout.py --dir OUT --reference adamw_mid --repeat adamw_mid2 \\
        --arms adamw_lo adamw_hi muon_mid ... [--out bench.json]

Each arm NAME is read from OUT/pass_NAME.*: its .probe_sides.jsonl (the
probes' per-game-side sums, tools/sequence_train.py), its .ema.probe_sides.jsonl
when its weight average was probed, its .probe.jsonl (the optimizer state's
size) and its .steps.jsonl (each window's time, positions and peak device
memory). Two probes are read: the pass's first (`mid`, at about half the
pass) and its final one.

Per probe and reading (ce_all and belief at each probe run: k0, k64,
k64_reset) every arm's pooled value and its difference from the reference
arm, with its standard error paired over holdout games
(tools/trainer_acceptance_readout.py, `paired_difference`); the repeat's
difference from the reference is the run-to-run noise. An arm's average,
when probed, is read against the arm's own live weights and against the
reference. Speed: positions a second over the windows after the first
WARM_WINDOWS (the compile), and the largest window's peak memory. An arm
whose files cannot be read, or whose probe holds other game-sides or counts
than the reference's, is set aside with its reason; the reference must be
read. The criterion is the lead's to pre-register; this gives the numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.trainer_acceptance_readout import (READINGS, SidesKey, check_same_holdout,  # noqa: E402
                                              paired_difference, per_game, pooled)

WARM_WINDOWS = 20
PROBES = ("mid", "final")
LEAD = ("ce_all_k64", "belief_k64", "ce_all_k0")    # the readings the table lists


def _fmt(d: Optional[Dict]) -> str:
    """A difference and its standard error."""
    if d is None or d["diff"] is None:
        return "-"
    return f"{d['diff']:+.5f}" + ("" if d["se"] is None else f"+-{d['se']:.5f}")


def _rows(path: Path) -> List[Dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def probe_sides(path: Path) -> Dict[str, Dict[SidesKey, Dict]]:
    """{"mid": ..., "final": ...}: the first non-final probe's game-sides and
    the last final one's (either may be missing)."""
    rows = _rows(path)
    mid = next((r for r in rows if not r.get("final")), None)
    final = next((r for r in reversed(rows) if r.get("final")), None)
    return {name: {(s["file"], int(s["side"])): s for s in row["sides"]}
            for name, row in (("mid", mid), ("final", final)) if row is not None}


def speed(path: Path) -> Dict:
    """Positions a second over the trained windows past the first
    WARM_WINDOWS, and the largest window's peak device memory (GiB)."""
    rows = [r for r in _rows(path) if not r.get("forward_only")]
    timed = rows[WARM_WINDOWS:]
    positions = sum(b["positions"] - a["positions"] for a, b in zip(timed, timed[1:]))
    seconds = sum(r["seconds"] for r in timed[1:])
    peaks = [r["peak_gb"] for r in rows if r.get("peak_gb") is not None]
    return {"positions_per_second": round(positions / seconds, 1) if seconds > 0 else None,
            "windows": len(rows), "peak_gb": max(peaks) if peaks else None}


def _state_gb(path: Path) -> Optional[float]:
    rows = [r for r in _rows(path) if "optimizer_state_gb" in r] if path.exists() else []
    return rows[-1]["optimizer_state_gb"] if rows else None


def _differences(x: Dict[SidesKey, Dict], y: Dict[SidesKey, Dict], runs: Sequence[str]) -> Dict[str, Dict]:
    """Each reading's pooled x and x minus y, paired over games."""
    out = {}
    for run in runs:
        for reading, count, total in READINGS:
            gx, gy = per_game(x, run, count, total), per_game(y, run, count, total)
            out[f"{reading}_{run}"] = {"value": pooled(gx), "minus": paired_difference(gx, gy)}
    return out


def _read_arm(directory: Path, name: str, reference: Dict[str, Dict[SidesKey, Dict]]) -> Dict:
    """One arm's probes against the reference's, its average's, its speed."""
    base = directory / f"pass_{name}"
    sides = probe_sides(base.with_suffix(".probe_sides.jsonl"))
    arm: Dict = {"probes": {}}
    for probe in PROBES:
        if probe in sides and probe in reference:
            runs = check_same_holdout({"reference": reference[probe], name: sides[probe]})
            arm["probes"][probe] = _differences(sides[probe], reference[probe], runs)
    ema_file = base.with_suffix(".ema.probe_sides.jsonl")
    if ema_file.exists() and "final" in sides:
        ema = probe_sides(ema_file)["final"]
        runs = check_same_holdout({"live": sides["final"], "ema": ema})
        arm["ema"] = {"minus_live": _differences(ema, sides["final"], runs),
                      "minus_reference": _differences(ema, reference["final"], runs)}
    arm.update(speed(base.with_suffix(".steps.jsonl")), optimizer_state_gb=_state_gb(base.with_suffix(".probe.jsonl")))
    return arm


def readout(directory: Path, reference: str, repeat: Optional[str], arms: Sequence[str]) -> Dict:
    directory = Path(directory)
    ref_sides = probe_sides(directory / f"pass_{reference}.probe_sides.jsonl")
    if "final" not in ref_sides:
        raise SystemExit(f"the reference {reference} has no final probe")
    out: Dict = {"reference": reference, "repeat": repeat, "arms": {}, "refused": {},
                 "games": len({f for f, _ in ref_sides["final"]}), "game_sides": len(ref_sides["final"])}
    names = [reference] + ([repeat] if repeat else []) + [a for a in arms if a not in (reference, repeat)]
    for name in names:
        try:
            out["arms"][name] = _read_arm(directory, name, ref_sides)
        except (OSError, ValueError, KeyError, StopIteration, SystemExit) as e:   # one bad arm leaves the others read
            out["refused"][name] = str(e) or type(e).__name__
    return out


def table(result: Dict) -> str:
    """The lead readings per arm: final and mid differences from the
    reference, the average's difference from its live weights, speed."""
    ref = result["reference"]
    head = f"{'arm':<16}" + "".join(f" {k + ' final':>26}" for k in LEAD) + f" {'ce_all_k64 mid':>22}" \
        f" {'ema-live ce_all_k64':>22} {'pos/s':>8} {'peak GB':>8} {'state GB':>8}"
    lines = [f"differences from {ref}, standard errors paired over {result['games']} holdout games", head]
    for name, arm in result["arms"].items():
        final, mid = arm["probes"].get("final", {}), arm["probes"].get("mid", {})
        ema = arm.get("ema", {}).get("minus_live", {}).get("ce_all_k64", {}).get("minus")

        def value(r, k):
            return "-" if k not in r else f"{r[k]['value']:.5f} {_fmt(r[k]['minus'])}"
        lines.append(f"{name:<16}" + "".join(f" {value(final, k):>26}" for k in LEAD)
                     + f" {_fmt(mid['ce_all_k64']['minus']) if 'ce_all_k64' in mid else '-':>22}"
                     + f" {_fmt(ema):>22} {arm['positions_per_second'] or '-':>8} {arm['peak_gb'] or '-':>8}"
                     + f" {arm['optimizer_state_gb'] if arm['optimizer_state_gb'] is not None else '-':>8}")
    for name, why in result["refused"].items():
        lines.append(f"{name}: set aside: {why}")
    return "\n".join(lines)


def main(argv: Sequence[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, required=True, help="the bench's directory of pass_NAME.* files")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--repeat", default=None, help="the reference's repeat: the run-to-run noise")
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=None)
    opts = ap.parse_args(argv)
    result = readout(opts.dir, opts.reference, opts.repeat, opts.arms)
    print(table(result))
    if opts.out:
        opts.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
