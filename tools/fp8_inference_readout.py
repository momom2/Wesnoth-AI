"""The fp8 inference round's readouts (docs/fp8_inference_prereg_20261019.md,
scripts/fp8_inference_box.sh): speed.json from the speed arms' records,
play.json from the play check's games and fit.

    python tools/fp8_inference_readout.py speed OUT [--spread 0.05]
    python tools/fp8_inference_readout.py play OUT --games-dir DIR --fit FIT --games N \\
        --fp8-label A --bf16-label B [--rc RC]

speed: per arm (OUT/arm_<name>.json and its servers' records beside it),
the match's wall and games, the servers' figures over their whole serving
wall (decisions a second, device and infer ms a batch, device ms a leaf,
mean batch, decisions a game) and over a steady window of their timeline
(tools/box_probe.steady_rate: from WARM_S after the first request, WINDOW_S
long, inside the 120 games in flight), the compiled loop's state at the end
of serving and whether the servers played fp8. An arm fails when its match
failed or lost a game, a server failed a request or died, its compiled loop
fell back to the eager loop, or an fp8 arm's servers did not play fp8. Then
the reading, on the window rates when every arm read has one, else on the
whole-run rates: the reference runs (ref_*) against their mean (valid when
each lies within --spread of it), each arm against that mean, the fp8 runs
(fp8c_*, their mean) against it, and whether the gain is resolved: above
the reference runs' largest deviation from their mean and at least MIN_GAIN.

play: the fp8 side's Elo against the bf16 side from the fit (PURE, the
decisive games), its standard error, the decisive games, each side's
decisions a game, whether every game and server carries the sides' fp8
switches, and the verdict: NOT_READ when the match holds fewer decisive
games than it was to play or its records do not carry the switches, KILL
when the fp8 side reads below -2 standard errors, PASS otherwise.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.box_probe import read_timeline, steady_rate  # noqa: E402

WARM_S = 20.0
WINDOW_S = 60.0
MIN_GAIN = 0.03
PURE = "PURE (decisive only, primary)"
KILL_SE = 2.0


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _servers(out: Path, name: str) -> List[Dict]:
    return [_json(p) for p in sorted(out.glob(f"arm_{name}.inference_server_*.json"))]


def _whole_run(servers: List[Dict], games: int) -> Dict:
    """The servers' figures over their whole serving wall, summed over the
    servers, the wall the longest one's."""
    def total(k):
        return sum(float(s.get(k) or 0.0) for s in servers)
    requests, batches, leaves = int(total("requests")), int(total("batches")), int(total("leaves"))
    wall = max(float(s.get("wall_s") or 0.0) for s in servers)
    return {"requests": requests, "batches": batches, "mean_batch": round(leaves / batches, 2) if batches else None,
            "decisions_per_s": round(requests / wall, 2) if wall else None,
            "decisions_per_game": round(requests / games, 1) if games else None,
            "device_ms_per_batch": round(total("gpu_ms") / batches, 3) if batches else None,
            "infer_ms_per_batch": round(1e3 * total("infer_s") / batches, 3) if batches else None,
            "device_ms_per_leaf": round(total("gpu_ms") / leaves, 4) if leaves else None,
            "server_wall_s": wall}


def _problems(arm: Dict, servers: List[Dict]) -> List[str]:
    """Why an arm's timing does not stand (module docstring)."""
    out = []
    if arm.get("failed_games"):
        out.append(f"{arm['failed_games']} failed games")
    if arm.get("rc") not in (0, 3, 4):
        out.append(f"match rc {arm.get('rc')}")
    if not servers:
        out.append("no server records")
    for s in servers:
        if s.get("failed_requests") or s.get("fatal"):
            out.append(f"server: {s.get('failed_requests')} failed requests, fatal {s.get('fatal')}")
        if (s.get("memory_states") or {}).get("refused"):
            out.append("server memory refused states")
        if s.get("trunk_stats_error"):
            out.append(f"server trunk stats: {s['trunk_stats_error']}")
        compiled = s.get("packed_compile_stats")
        if s.get("compile_packed") and not (compiled and compiled.get("active")):
            out.append(f"compiled loop not active: {(compiled or {}).get('fallback_reason')}")
    if "--fp8-inference" in str(arm.get("flags")) and not all(s.get("fp8") for s in servers):
        out.append("an fp8 arm's server did not play fp8")
    return out


def arm_row(out: Path, name: str) -> Dict:
    arm = _json(out / f"arm_{name}.json")
    servers = _servers(out, name)
    row = {k: arm.get(k) for k in ("jobs", "flags", "games", "failed_games", "rc", "why", "wall_s")}
    row["games_per_hour"] = round(3600.0 * arm["games"] / arm["wall_s"], 1) if arm.get("wall_s") else None
    if servers:
        row.update(_whole_run(servers, arm["games"]))
        timelines = sorted(out.glob(f"arm_{name}.inference_server_*.timeline.jsonl"))
        row["window"] = (steady_rate(read_timeline(timelines[0]), WARM_S, WINDOW_S)
                         if len(timelines) == 1 and len(servers) == 1 else None)
        row["compiled"] = [s.get("packed_compile_stats") for s in servers]
        row["fp8"] = [bool(s.get("fp8")) for s in servers]
        row["fp8_scales"] = next((s["fp8_scales"] for s in servers if s.get("fp8_scales")), None)
        row["profile"] = [s["profile"] for s in servers if s.get("profile")]
    row["problems"] = _problems(arm, servers)
    row["failed"] = bool(row["problems"])
    return row


def _rate(row: Dict, windowed: bool) -> Optional[float]:
    if row["failed"]:
        return None
    return (row.get("window") or {}).get("decisions_per_s") if windowed else row.get("decisions_per_s")


def reading(arms: Dict[str, Dict], spread: float) -> Dict:
    """The reference runs' validity and every arm against their mean
    (module docstring)."""
    timed = {n: r for n, r in arms.items() if not n.startswith("profile")}
    windowed = bool(timed) and all((r.get("window") or {}).get("decisions_per_s") for r in timed.values()
                                   if not r["failed"])
    rates = {n: _rate(r, windowed) for n, r in timed.items()}
    refs = [rates[n] for n in sorted(rates) if n.startswith("ref_") and rates[n]]
    fp8c = [rates[n] for n in sorted(rates) if n.startswith("fp8c_") and rates[n]]
    out = {"rates_from": "window" if windowed else "whole_run",
           "rates": rates, "valid": False, "ref_spread": {}, "against_ref": {}}
    if len(refs) < 2:
        return out
    mean = statistics.fmean(refs)
    deviations = [abs(r / mean - 1.0) for r in refs]
    out.update(valid=max(deviations) <= spread, ref_mean=round(mean, 2),
               ref_spread={n: round(rates[n] / mean, 4) for n in rates if n.startswith("ref_") and rates[n]},
               against_ref={n: round(r / mean, 4) for n, r in rates.items() if r})
    if fp8c:
        gain = statistics.fmean(fp8c) / mean
        out["fp8c_against_ref"] = round(gain, 4)
        out["fp8c_gain_resolved"] = bool(out["valid"] and gain - 1.0 >= max(MIN_GAIN, max(deviations)))
        leaf = [arms[n]["device_ms_per_leaf"] for n in arms if n.startswith("fp8c_") and not arms[n]["failed"]]
        ref_leaf = [arms[n]["device_ms_per_leaf"] for n in arms if n.startswith("ref_") and not arms[n]["failed"]]
        if leaf and ref_leaf:
            out["fp8c_device_ms_per_leaf_against_ref"] = round(statistics.fmean(leaf) / statistics.fmean(ref_leaf), 4)
    if rates.get("fp8e") and rates.get("bf16e"):
        out["fp8e_against_bf16e"] = round(rates["fp8e"] / rates["bf16e"], 4)
    return out


def speed(opts) -> int:
    out = Path(opts.out)
    names = sorted(p.name[len("arm_"):-len(".json")] for p in out.glob("arm_*.json") if p.name.count(".") == 1)
    arms = {n: arm_row(out, n) for n in names}
    summary = {"arms": arms, "reading": reading(arms, opts.spread),
               "failed_arms": sorted(n for n, r in arms.items() if r["failed"])}
    tmp = out / "speed.json.tmp"
    tmp.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    tmp.replace(out / "speed.json")
    r = summary["reading"]
    print(json.dumps({k: r.get(k) for k in ("rates_from", "valid", "ref_spread", "fp8c_against_ref",
                                            "fp8c_gain_resolved", "fp8c_device_ms_per_leaf_against_ref",
                                            "fp8e_against_bf16e")} | {"failed_arms": summary["failed_arms"]}),
          flush=True)
    return 0


# ---- the play check ------------------------------------------------------------

def _games(games_dir: Path) -> List[Dict]:
    found = []
    for p in sorted(games_dir.glob("game_*.json")):
        try:
            found.append(_json(p))
        except (OSError, ValueError):
            continue
    return found


def play_readout(games: List[Dict], fit: Optional[Dict], servers: List[Dict], target: int,
                 fp8_label: str, bf16_label: str, rc: Optional[int] = None) -> Dict:
    """The play check's reading (module docstring)."""
    decisive = [g for g in games if g.get("outcome_a") in ("win", "loss")]
    switches = all(g.get("infer_fp8_a") is True and not g.get("infer_fp8_b") for g in games) and bool(games)
    served = sorted(bool(s.get("fp8")) for s in servers) == [False, True]
    out: Dict = {"games": len(games), "decisive": len(decisive), "target": target, "rc": rc,
                 "game_switches_ok": switches, "servers_fp8_and_bf16": served,
                 "wins_fp8": sum(g.get("outcome_a") == "win" for g in decisive)}
    if decisive:
        for side in ("a", "b"):
            counts = [g[f"forwards_{side}"] for g in decisive if g.get(f"forwards_{side}") is not None]
            out[f"decisions_{side}_per_game"] = round(statistics.fmean(counts), 1) if counts else None
    table = ((fit or {}).get("tables") or {}).get(PURE) or {}
    if fp8_label in table and bf16_label in table:
        a, b = table[fp8_label], table[bf16_label]
        out["elo_fp8"] = round(a["elo"] - b["elo"], 1)
        out["se"] = round((a["se"] ** 2 + b["se"] ** 2) ** 0.5, 1)
    if len(decisive) < target or not (switches and served) or "elo_fp8" not in out:
        out["verdict"] = "NOT_READ"
    elif out["elo_fp8"] < -KILL_SE * out["se"]:
        out["verdict"] = "KILL"
    else:
        out["verdict"] = "PASS"
    return out


def play(opts) -> int:
    out = Path(opts.out)
    fit = _json(Path(opts.fit)) if Path(opts.fit).exists() else None
    servers = [_json(p) for p in sorted(out.glob("play.server_*.json"))]
    result = play_readout(_games(Path(opts.games_dir)), fit, servers, opts.games, opts.fp8_label,
                          opts.bf16_label, opts.rc)
    tmp = out / "play.json.tmp"
    tmp.write_text(json.dumps(result, indent=1), encoding="utf-8")
    tmp.replace(out / "play.json")
    print(json.dumps(result), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("speed")
    s.add_argument("out")
    s.add_argument("--spread", type=float, default=0.05)
    p = sub.add_parser("play")
    p.add_argument("out")
    p.add_argument("--games-dir", required=True)
    p.add_argument("--fit", required=True)
    p.add_argument("--games", type=int, required=True, help="the decisive games the match was to play")
    p.add_argument("--fp8-label", required=True)
    p.add_argument("--bf16-label", required=True)
    p.add_argument("--rc", type=int, default=None, help="the match's exit code")
    opts = ap.parse_args(argv)
    return {"speed": speed, "play": play}[opts.cmd](opts)


if __name__ == "__main__":
    sys.exit(main())
