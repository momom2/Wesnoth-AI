#!/usr/bin/env python3
"""Read a match directory of `run_elo_batch` game records for the
actor-level end_turn test (docs/endturn_rule_prereg_20260919.md):
player A's decisive score, the same with capped games scored 0.5, the
capped fraction, and each player's decisions per side-turn (the raw
player makes one forward per decision, so forwards over turns).

    python tools/analysis/endturn_readout.py GAMES_DIR [GAMES_DIR ...]
        [--require-fire 1.03]   exit 1 unless A's decisions per side-turn
                                are at least that multiple of B's (kill 1)
        [--require-pass 0.535]  exit 1 unless A's decisive score is at least that
        [--best-p]              print the directory with the highest decisive score
        [--verdict]             the pre-registered bars over every directory given
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Readout:
    name: str
    games: int
    wins: int
    losses: int
    draws: int
    capped: int
    dps_a: float            # decisions per side-turn, player A, mean over games
    dps_b: float
    procedure_a: str
    procedure_b: str

    @property
    def decisive(self) -> int:
        return self.wins + self.losses

    @property
    def p(self) -> Optional[float]:
        return self.wins / self.decisive if self.decisive else None

    @property
    def se(self) -> Optional[float]:
        if not self.decisive:
            return None
        p = self.p
        return math.sqrt(p * (1 - p) / self.decisive)

    @property
    def p_half(self) -> Optional[float]:
        n = self.games
        return (self.wins + 0.5 * (self.draws + self.capped)) / n if n else None

    @property
    def capped_frac(self) -> Optional[float]:
        return self.capped / self.games if self.games else None

    @property
    def fire_ratio(self) -> Optional[float]:
        return self.dps_a / self.dps_b if self.dps_b else None


def read_dir(path: Path) -> Readout:
    games = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(path.glob("game_*.json"))]
    wins = sum(1 for g in games if g.get("outcome_a") == "win")
    losses = sum(1 for g in games if g.get("outcome_a") == "loss")
    draws = sum(1 for g in games if g.get("outcome_a") == "draw")
    capped = sum(1 for g in games if g.get("outcome_a") == "timeout")
    dps_a = [g["forwards_a"] / g["turns"] for g in games
             if g.get("forwards_a") and g.get("turns")]
    dps_b = [g["forwards_b"] / g["turns"] for g in games
             if g.get("forwards_b") and g.get("turns")]
    procs = {(g.get("procedure_a"), g.get("procedure_b")) for g in games}
    if len(procs) > 1:
        raise SystemExit(f"{path}: mixed procedures {sorted(procs)}")
    pa, pb = next(iter(procs)) if procs else ("?", "?")
    return Readout(name=path.name, games=len(games), wins=wins, losses=losses, draws=draws,
                   capped=capped, dps_a=statistics.fmean(dps_a) if dps_a else 0.0,
                   dps_b=statistics.fmean(dps_b) if dps_b else 0.0,
                   procedure_a=str(pa), procedure_b=str(pb))


def describe(r: Readout) -> str:
    p = f"{r.p:.3f} +- {r.se:.3f}" if r.p is not None else "n/a"
    ph = f"{r.p_half:.3f}" if r.p_half is not None else "n/a"
    fire = f"{r.fire_ratio:.3f}x" if r.fire_ratio else "n/a"
    return (f"{r.name}: {r.procedure_a} vs {r.procedure_b}: {r.games} games, "
            f"W-L-D-capped {r.wins}-{r.losses}-{r.draws}-{r.capped}, p {p} over "
            f"{r.decisive} decisive, capped scored 0.5: {ph}, capped fraction "
            f"{(r.capped_frac or 0):.2f}; decisions per side-turn A {r.dps_a:.2f} "
            f"B {r.dps_b:.2f} ({fire})")


def verdict(readouts: List[Readout], fire: float, pass_p: float) -> str:
    lines = [describe(r) for r in readouts]
    by = {r.name: r for r in readouts}
    screen = by.get("games_screen_endm")
    if screen is not None:
        ok = screen.fire_ratio is not None and screen.fire_ratio >= fire
        lines.append(f"kill 1 (screen fire >= {fire:.2f}x): {'pass' if ok else 'KILL'} "
                     f"({screen.fire_ratio and round(screen.fire_ratio, 3)}x)")
    match = by.get("games_endm")
    if match is not None and match.p is not None:
        if match.p <= 0.50:
            lines.append(f"kill 2 (p <= 0.50 at {match.decisive} decisive): KILL (p {match.p:.3f})")
        elif match.p >= pass_p:
            lines.append(f"pass (p >= {pass_p} at {match.decisive} decisive): PASS (p {match.p:.3f})")
        else:
            lines.append(f"neither kill 2 nor pass: p {match.p:.3f} at {match.decisive} decisive "
                         f"(above 0.50, under {pass_p})")
        if screen is not None and screen.capped_frac is not None and match.capped_frac is not None:
            # Barrier: the rule's capped rate against the reference's
            # self-match rate, 17 of 40 on 2026-09-13 as the standing null.
            null = 17 / 40
            se = math.sqrt(null * (1 - null) / max(1, match.games))
            tilt = match.capped_frac > null + 2 * se
            lines.append(f"barrier (capped fraction {match.capped_frac:.2f} against the "
                         f"self-match null {null:.2f}, 2 SE {2 * se:.2f}): "
                         f"{'STALL TILT' if tilt else 'clear'}")
    for name, r in by.items():
        if name.startswith("games_eo") and r.p is not None and match is not None and match.p is not None:
            # The pre-registered reading: an offset that matches the rule
            # within 1 SE says the lever is "act more" and the config
            # scalar is the adopted form; one that BEATS the rule says so
            # louder; only an offset well below the rule leaves something
            # rule-specific to explain.
            se = match.se or 0.0
            if r.p >= match.p - se:
                how = ("within 1 SE" if abs(r.p - match.p) <= se
                       else f"above the rule by {(r.p - match.p) / max(se, 1e-9):.1f} SE")
                reading = f"{how}, the lever is act more; the config scalar is the adopted form"
            else:
                reading = f"below the rule by {(match.p - r.p) / max(se, 1e-9):.1f} SE: rule-specific"
            lines.append(f"attribution {name}: p {r.p:.3f} against the rule's {match.p:.3f}: {reading}")
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--require-fire", type=float, default=None)
    ap.add_argument("--require-pass", type=float, default=None)
    ap.add_argument("--best-p", action="store_true")
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--fire", type=float, default=1.03)
    ap.add_argument("--pass-p", type=float, default=0.535)
    args = ap.parse_args(argv)
    readouts = [read_dir(d) for d in args.dirs]
    if args.best_p:
        best = max((r for r in readouts if r.p is not None), key=lambda r: r.p, default=None)
        print(best.name if best else "")
        return 0
    if args.verdict:
        print(verdict(readouts, args.fire, args.pass_p))
        return 0
    for r in readouts:
        print(describe(r))
    if args.require_fire is not None:
        r = readouts[0]
        return 0 if (r.fire_ratio is not None and r.fire_ratio >= args.require_fire) else 1
    if args.require_pass is not None:
        r = readouts[0]
        return 0 if (r.p is not None and r.p >= args.require_pass) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
