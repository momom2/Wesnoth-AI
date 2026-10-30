"""Stage P of the training design on the local machine (Windows, the RX 6600;
docs/training_design_20261019/current_design.md §5 "Planted round", §6
"P"), after stage C (scripts/stage_c_local.py) and before the rounds
(scripts/rounds_local.py). It runs only when stage C's verdict says P (its
verdict.json "next").

    the planted step played: theta' (stage C's planted.pt, tools/pg_plant.py
        readout --candidate) against pi_0, P_GAMES games at temperature 1 on
        both seats, decisions logged; read back by tools/pg_planted_play.py
        planted (X per side-game rises on each seat; the weights played are
        theta''s);
    the link match: pi_0 at T = 1/(1 + e) on side A against pi_0 at
        T = 1/(1 - e) on side B, LINK_GAMES games; read back by
        tools/pg_planted_play.py link against 2 e times stage 1's sharpening
        slope. It is played only when that slope is resolved at 3 SE (stage
        1's readout.json "sharpening" "resolved", design §6: "iff the slope
        >= 3 SE"); otherwise link.json records "not run" and the rounds do
        not wait on it.

The link's e is the pre-registration's (--e, no default): the forecast is
sealed before the match is played.

The verdict, design §6:
- a planted miss stops the rounds for debugging;
- a link outside its forecast by 2 combined SE means first-order labels do
  not price play at temperature 1: stop and diagnose before rounds;
- otherwise the rounds may start.

Each step resumes from its markers. A STOP file ends the run between steps,
and a display reset during a GPU step writes STOP and ends it.

    HIP_VISIBLE_DEVICES=1 <rocm venv>/python scripts/p_local.py --stage1 <stage 1's out> --stage-c <stage C's out> \\
        --out C:/Users/<you>/wesnoth_runs/p_local --e 0.02
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.local_machine import (Steps, Stopped, below_normal_priority, default_tools_python,  # noqa: E402
                                   display_resets_since, keep_awake, log, require_visible_gpu, run_tool)
from scripts.rounds_local import POOL  # noqa: E402

P_GAMES = 5000                  # design §5: the planted step against pi_0
LINK_GAMES = 12800              # design §4: the link match ($0.33 on a box)
P_SEED_BASE = 50_000_000       # past the rounds' blocks (scripts/rounds_local.py's seed layout)
LINK_SEED_BASE = 60_000_000
PLANTED_LABEL = "planted"
MATCH_CUT_MIN = 6 * 60


class P(Steps):
    def __init__(self, args, runner: Callable[..., int] = run_tool,
                 resets: Callable[[float], int] = display_resets_since):
        super().__init__(args.out, args.python, args.tools_python, runner, resets)
        self.s1, self.sc = Path(args.stage1), Path(args.stage_c)
        self.pi0, self.e = str(args.checkpoint), float(args.e)

    def side(self, side: str, label: str, spec: str, temperature: float) -> list:
        return [f"--label-{side}", label, f"--spec-{side}", spec, f"--memory-{side}", "64",
                f"--raw-temperature-{side}", repr(temperature)]

    def match(self, part: str, games_dir: Path, flags: list, games: int, seed: int) -> bool:
        if self.skip(part):
            return self.done(part)
        self.check_stop(part)
        rc = self.step(part, ["tools/run_elo_batch.py", *flags, "--mcts-sims", "0", "--outdir", str(games_dir),
                              "--games", str(games), "--max-extra-games", "0", "--seed-base", str(seed), *POOL],
                       f"{part}.log", MATCH_CUT_MIN)
        self.mark(part, None if rc in (0, 4) else f"the match ended rc={rc} ({part}.log)")
        return self.done(part)

    def readout(self, part: str, cmd: list, out_json: Path) -> dict:
        if not self.skip(part):
            rc = self.step(part, cmd, f"{part}.log", 30, gpu=False)
            self.mark(part, None if rc == 0 and out_json.exists() else f"the readout ended rc={rc}")
        return json.loads(out_json.read_text(encoding="utf-8")) if self.done(part) else {}


def run(p: P) -> int:
    """0 when P passes and the link holds (the rounds may start), 1 on a
    failed step, 2 on STOP, 3 when P or the link stops the rounds."""
    out = p.out
    verdict = json.loads((p.sc / "verdict.json").read_text(encoding="utf-8"))
    if not str(verdict.get("next", "")).startswith("P"):
        log(out, f"P_NOT_DUE: stage C's verdict says {verdict.get('next')!r}")
        return 3
    planted = p.sc / "planted.pt"
    if not planted.exists():
        log(out, f"P_NO_CANDIDATE: {planted} is missing (stage C's plant readout writes it)")
        return 1
    try:
        games = out / "games_p"
        side_a = p.side("a", PLANTED_LABEL, str(planted), 1.0)
        side_b = p.side("b", "own2", p.pi0, 1.0)
        if not p.match("p_match", games, side_a + side_b + ["--log-decisions"], P_GAMES, P_SEED_BASE):
            return 1
        p_read = p.readout("p_readout", ["tools/pg_planted_play.py", "planted", "--games", str(games),
                                         "--planted-label", PLANTED_LABEL, "--candidate", str(planted), "--out",
                                         str(out / "p_readout.json")], out / "p_readout.json")
        if not p_read:
            return 1
        sharp = json.loads((p.s1 / "readout.json").read_text(encoding="utf-8")).get("sharpening") or {}
        if sharp.get("resolved"):
            link_games = out / "games_link"
            flags = (p.side("a", "own2c", p.pi0, 1.0 / (1.0 + p.e))
                     + p.side("b", "own2h", p.pi0, 1.0 / (1.0 - p.e)))
            if not p.match("link_match", link_games, flags, LINK_GAMES, LINK_SEED_BASE):
                return 1
            link = p.readout("link_readout", ["tools/pg_planted_play.py", "link", "--games", str(link_games),
                                              "--readout", str(p.s1 / "readout.json"), "--e", repr(p.e), "--out",
                                              str(out / "link.json")], out / "link.json")
            if not link:
                return 1
        else:
            link = {"verdict": "not run", "sharpening": sharp,
                    "reason": "stage 1's sharpening slope is not resolved at 3 SE (design §6)"}
            (out / "link.json").write_text(json.dumps(link, indent=1) + "\n", encoding="utf-8")
            log(out, f"LINK_NOT_RUN: the sharpening slope reads {sharp}")
    except Stopped as e:
        log(out, f"STOPPED: {e}; a new start resumes")
        return 2
    if p_read["verdict"] != "pass":
        why = f"P_MISS: the planted step played does not read back ({p_read['seats']}): debug before the rounds"
    elif link["verdict"] not in ("holds", "not run"):
        why = (f"LINK_MISS: measured {link['measured']:.4f} against the forecast {link['forecast']:.4f} "
               f"({link['miss_se']:.1f} combined SE): first-order labels do not price play at temperature 1; "
               f"stop and diagnose before the rounds")
    else:
        held = (f"the link holds ({link['measured']:.4f} against {link['forecast']:.4f})"
                if link["verdict"] == "holds" else "the link was not run")
        log(out, f"P_DONE: the planted step reads back on both seats and {held}: the rounds may start")
        return 0
    (out / "STOP").write_text(why + "\n", encoding="utf-8")
    log(out, why)
    return 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1", type=Path, required=True)
    ap.add_argument("--stage-c", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--e", type=float, required=True, help="the link match's temperature spread, pre-registered")
    ap.add_argument("--checkpoint", type=Path, default=REPO / "training" / "checkpoints" / "own2.pt")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--tools-python", default=None)
    ap.add_argument("--normal-priority", action="store_true")
    ap.add_argument("--keep-awake", action="store_true")
    args = ap.parse_args()
    if args.tools_python is None:
        args.tools_python = default_tools_python(REPO)
    require_visible_gpu()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.normal_priority:
        below_normal_priority()
    if args.keep_awake:
        keep_awake()
    log(args.out, f"P local: stage C {args.stage_c}, e {args.e}, out {args.out}")
    return run(P(args))


if __name__ == "__main__":
    sys.exit(main())
