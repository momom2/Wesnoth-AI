"""The design's rounds on the local machine (Windows, the RX 6600;
docs/training_design_20261019/current_design.md §1, §6 "Rounds" and
"Claims"), after stage C (scripts/stage_c_local.py), round after round:

- **Round 1** is stage C's ladder on stage 1's pass (design §6). Its
  decision is taken as read and its candidate written (tools/pg_round.py
  --adopt-from).
- **Round k >= 2**, from the policy pi the previous round left:
  - its games: N_TRAIN + N_HELD temperature-1 self-play games of pi on both
    seats, walked, labelled and read, the last N_HELD held out
    (scripts/selfplay_t1_local.py --checkpoint pi --no-measures, on seeds of
    the round's own);
  - the labels' corrections, fitted on its held-out games
    (tools/pg_corrections.py);
  - the gradient pass over its training games (tools/pg_pass.py in fp16,
    with the corrections);
  - the step (tools/pg_round.py): the ladder at the radius the previous
    round left, under stage C's arm, its decision and, when a rung is
    adopted, the next pi.
- **Every CLAIM_EVERY-th round, a claim:** pi at temperature 1 against the
  reference, CLAIM_GAMES decisive games (D0's match:
  scripts/selfplay_t1_local.py's measure), fitted PURE with the reference at 0.
- **The stops** (design §6):
  - pause, after every CLAIM_EVERY-th round, when the sum of the last
    CLAIM_EVERY confirmed readings (half B's ΔĴ of the adopted rungs, 0 for
    a round that kept pi) is not above 0 by 2 SE;
  - the claims' lower tail: claim_k - D0 below ELO_PER_Z times the sum of
    the confirmed readings since round 1 by more than 2 combined SE. Gains
    do not transfer, and the user picks a magnet or a pool.

  Either writes STOP with its reason; the chain waits for the user.

A failed part stops the chain: the next round needs its policy. A failed
part stays failed at a new start (its `rNN_<part>.failed`: delete it to run
it again), and a done part is not run again. A STOP file in --out ends the
run between steps, and a display reset during a GPU step writes STOP and
ends it.

    HIP_VISIBLE_DEVICES=1 <rocm venv>/python scripts/rounds_local.py --stage1 <stage 1's out> \\
        --stage-c <stage C's out> --out C:/Users/<you>/wesnoth_runs/rounds_local --rounds 12
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.local_machine import (Steps, Stopped, below_normal_priority, default_tools_python,  # noqa: E402
                                   display_resets_since, keep_awake, log, reference_flags, require_visible_gpu,
                                   run_tool)
from scripts.selfplay_t1_local import CHUNK_GAMES, MIN_FREE_MB, player_flags  # noqa: E402

N_TRAIN = 20000                 # design §1: N, a round's training games (stage C's gate may name another)
N_HELD = 10000                  # |H|, its held-out games
SEED_BASE = 1_000_000           # round k's games from SEED_BASE + k * SEED_STRIDE: past stage 1's (800000-848199)
CLAIM_SEED_BASE = 2_000_000     # round k's claim from CLAIM_SEED_BASE + k * SEED_STRIDE
SEED_STRIDE = 100_000
CLAIM_EVERY = 3
CLAIM_GAMES = 12800             # design §3.2: a claim's decisive games, D0's
ELO_PER_Z = 347.4               # Elo per unit of E[z] at a win rate of 1/2: 400 / ln 10 / (4 p (1 - p))
FLOOR_SIDES = 100
PASS_FP16 = ["--fp16", "--packed-trunk", "--floor-chunk", "8", "--floor-sides", str(FLOOR_SIDES)]
POOL = ["--persistent-workers", "--shared-inference", "--no-infer-compile", "--device", "cuda", "--jobs", "96",
        "--worker-games", "8", "--min-free-mb", str(MIN_FREE_MB), "--infer-fp16", "--inference-pipelined",
        "--inference-timeline-s", "30"]
PLAY_CUT_MIN = 16 * 60          # 30,000 games at about 5,000 an hour, played again once when short
CORRECTIONS_CUT_MIN = 60
PASS_CUT_MIN = 48 * 60          # 20,000 games at 80 to 150 decisions a second: 14 to 26 h, unmeasured
STEP_CUT_MIN = 36 * 60          # the ladder's forwards over 10,000 held-out games, unmeasured
CLAIM_CUT_MIN = 10 * 60
PASS_EXITS = {3: "the replay identity failed", 4: "the precision floor's bound lifts the replay limit past its cap",
              5: "the packed and padded trunks disagree in fp32"}


class Rounds(Steps):
    def __init__(self, args, runner: Callable[..., int] = run_tool,
                 resets: Callable[[float], int] = display_resets_since):
        super().__init__(args.out, args.python, args.tools_python, runner, resets)
        self.a = args
        self.s1, self.sc = Path(args.stage1), Path(args.stage_c)
        self.pi0 = str(args.checkpoint)
        self.workers = str(args.workers)

    # ---- the rounds' files
    def rdir(self, k: int) -> Path:
        d = self.out / f"round_{k:02d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def round_json(self, k: int) -> dict:
        return json.loads((self.rdir(k) / "round.json").read_text(encoding="utf-8"))

    def policy_after(self, k: int) -> str:
        """The policy round k left: its candidate when it adopted one, else
        the one it started from (pi_0 before round 1)."""
        while k >= 1:
            if (self.round_json(k).get("candidate") or {}).get("path"):
                return str(self.rdir(k) / "pi.pt")
            k -= 1
        return self.pi0

    def p_gate(self) -> Optional[str]:
        """Design §6: P before the rounds. None when stage P read back and the
        link held (scripts/p_local.py's p_readout.json and link.json in
        --p-out), or when the gate is waived (--no-p-gate); why not otherwise."""
        if self.a.p_out is None:
            return None if self.a.no_p_gate else "no --p-out: stage P (scripts/p_local.py) has not been named"
        reads = {}
        for name in ("p_readout.json", "link.json"):
            f = Path(self.a.p_out) / name
            if not f.exists():
                return f"stage P's {name} is missing from {self.a.p_out}"
            reads[name] = json.loads(f.read_text(encoding="utf-8")).get("verdict")
        if reads["p_readout.json"] != "pass" or reads["link.json"] != "holds":
            return f"stage P reads {reads}: the rounds wait (design §6)"
        return None

    def game_dirs(self, k: int) -> List[str]:
        play = self.rdir(k) / "play"
        return [str(play / f"games_s{c:02d}") for c in range((N_TRAIN + N_HELD + CHUNK_GAMES - 1) // CHUNK_GAMES)]

    def factory_args(self, checkpoint: str, dirs: List[str], labels: Path) -> List[str]:
        out = []
        for key, value in (("checkpoint", checkpoint), ("games", ",".join(dirs)), ("labels", labels),
                           ("device", "cuda"), ("workers", self.workers), ("fp16", "1")):
            out += ["--factory-arg", f"{key}={value}"]
        return out

    # ---- round 1: stage C's decision
    def round1(self) -> bool:
        part = "r01_step"
        if self.skip(part):
            return self.done(part)
        d = self.rdir(1)
        dirs = [str(p) for p in sorted(self.s1.glob("games_s*")) if p.is_dir()]
        rc = self.step("round 1 step", ["tools/pg_round.py", "--pass", str(self.sc / "pass.pt"), "--factory",
                                        "tools.pg_eval:network_factory",
                                        *self.factory_args(self.pi0, dirs, self.s1 / "labels.jsonl"),
                                        "--adopt-from", str(self.sc / "stage_c.json"), "--out", str(d / "round.json"),
                                        "--candidate", str(d / "pi.pt"), "--direction-out",
                                        str(d / "direction.pt")], "round_01.log", STEP_CUT_MIN)
        if rc != 0 or not (d / "round.json").exists():
            self.mark(part, f"round 1's step ended rc={rc} (round_01.log)")
            return False
        self.mark(part)
        return True

    # ---- round k >= 2
    def round_k(self, k: int) -> bool:
        d, pi = self.rdir(k), self.policy_after(k - 1)
        tag = f"r{k:02d}"
        labels, fits = d / "play" / "labels.jsonl", d / "play" / "fits.json"
        dirs = self.game_dirs(k)
        if not self.skip(f"{tag}_play"):
            self.check_stop(f"round {k}'s games")
            rc = self.step(f"round {k} games", ["scripts/selfplay_t1_local.py", "--out", str(d / "play"),
                                                "--checkpoint", pi, "--label", f"pi{k}", "--seed-base",
                                                str(SEED_BASE + k * SEED_STRIDE), "--games", str(N_TRAIN + N_HELD),
                                                "--heldout-from", str(N_TRAIN), "--no-measures"],
                           f"round_{k:02d}.log", PLAY_CUT_MIN)
            if rc == 2 or rc == 5 or (d / "play" / "STOP").exists():
                raise Stopped(f"round {k}'s games stopped (rc={rc}, {d / 'play'})")
            self.mark(f"{tag}_play", None if rc == 0 else f"round {k}'s games ended rc={rc} ({d / 'play'})")
        if not self.done(f"{tag}_play"):
            return False
        if not self.skip(f"{tag}_corrections"):
            rc = self.step(f"round {k} corrections", ["tools/pg_corrections.py", "--games", *dirs, "--labels",
                                                      str(labels), "--fits", str(fits), "--workers", self.workers,
                                                      "--out", str(d / "corrections.json")],
                           f"round_{k:02d}.log", CORRECTIONS_CUT_MIN, gpu=False)
            self.mark(f"{tag}_corrections", None if rc == 0 else f"the corrections ended rc={rc}")
        if not self.done(f"{tag}_corrections"):
            return False
        if not self.skip(f"{tag}_pass"):
            self.check_stop(f"round {k}'s pass")
            rc = self.step(f"round {k} pass", ["tools/pg_pass.py", "--checkpoint", pi, "--games", *dirs,
                                               "--labels", str(labels), "--out", str(d / "pass.pt"), "--device",
                                               "cuda", "--workers", self.workers, *PASS_FP16, "--corrections",
                                               str(d / "corrections.json")], f"round_{k:02d}.log", PASS_CUT_MIN)
            self.mark(f"{tag}_pass", None if rc == 0 else
                      f"{PASS_EXITS.get(rc, f'the pass ended rc={rc}')} (round_{k:02d}.log)")
        if not self.done(f"{tag}_pass"):
            return False
        if not self.skip(f"{tag}_step"):
            self.check_stop(f"round {k}'s step")
            prev = self.rdir(k - 1)
            previous = ["--previous", str(prev / "direction.pt")] if (prev / "direction.pt").exists() else []
            rc = self.step(f"round {k} step", ["tools/pg_round.py", "--pass", str(d / "pass.pt"), "--factory",
                                               "tools.pg_eval:network_factory",
                                               *self.factory_args(pi, dirs, labels), "--eps-from",
                                               str(prev / "round.json"), "--arm-from", str(self.sc / "stage_c.json"),
                                               *previous, "--out", str(d / "round.json"), "--candidate",
                                               str(d / "pi.pt"), "--direction-out", str(d / "direction.pt")],
                           f"round_{k:02d}.log", STEP_CUT_MIN)
            ok = rc == 0 and (d / "round.json").exists()
            self.mark(f"{tag}_step", None if ok else f"round {k}'s step ended rc={rc} (round_{k:02d}.log)")
        return self.done(f"{tag}_step")

    # ---- the readings and the claims
    def confirmed(self, k: int) -> Tuple[float, float]:
        """Round k's confirmed reading (half B's ΔĴ of its adopted rung, in
        units of E[z]) and its SE; (0, 0) for a round that kept pi."""
        r = self.round_json(k)
        dec = r["decision"]
        if not dec.get("adopt"):
            return 0.0, 0.0
        rungs = dec.get("rungs")
        if not rungs:                                    # round 1: stage C's ladder holds them
            rungs = json.loads((self.sc / "stage_c.json").read_text(encoding="utf-8"))["ladder"]["rungs"]
        exact = rungs[dec["chosen"]]["halves"]["B"]["exact"]
        return float(exact["estimate"]), float(exact["se"])

    def claim(self, k: int) -> Optional[Tuple[float, float]]:
        """pi at temperature 1 against the reference: (Elo, SE), or None."""
        tag, d = f"r{k:02d}_claim", self.rdir(k)
        label = f"pi{k}t1"
        fit = d / "claim.fit.json"
        if not self.skip(tag):
            self.check_stop(f"round {k}'s claim")
            ref_b = reference_flags("b", self.tools_py)
            side_a = player_flags(reference_flags("a", self.tools_py), "a", label, spec=self.policy_after(k))
            games = d / "claim"
            rc = self.step(f"round {k} claim", ["tools/run_elo_batch.py", *side_a, *ref_b, "--mcts-sims", "0",
                                                "--raw-temperature-a", "1", "--raw-temperature-b", "0", "--outdir",
                                                str(games), "--games", str(CLAIM_GAMES), "--max-extra-games",
                                                str(CLAIM_GAMES // 2), "--seed-base",
                                                str(CLAIM_SEED_BASE + k * SEED_STRIDE), *POOL],
                           f"claim_{k:02d}.log", CLAIM_CUT_MIN)
            if rc in (0, 4):
                rc = self.step(f"round {k} claim fit", ["tools/elo_collect.py", str(games), "--no-catalog",
                                                        "--anchor", ref_b[ref_b.index("--label-b") + 1],
                                                        "--save-json", str(fit)], f"claim_{k:02d}.log", 15, gpu=False)
            self.mark(tag, None if rc == 0 and fit.exists() else f"the claim ended rc={rc} (claim_{k:02d}.log)")
        if not self.done(tag):
            return None
        return pure(fit, label)

    def stops(self, k: int, claim: Optional[Tuple[float, float]]) -> Optional[str]:
        """Design §6's two stops after round k; None to go on."""
        last = [self.confirmed(j) for j in range(max(1, k - CLAIM_EVERY + 1), k + 1)]
        total, se = sum(e for e, _ in last), math.sqrt(sum(s * s for _, s in last))
        if not total > 2 * se:
            return (f"PAUSE after round {k}: the last {len(last)} confirmed readings sum to {total:.4g} "
                    f"(SE {se:.3g}), not above 0 by 2 SE (design §6)")
        if claim is not None:
            d0 = pure(self.s1 / "d0.fit.json", "own2t1")
            readings = [self.confirmed(j) for j in range(1, k + 1)]
            forecast = ELO_PER_Z * sum(e for e, _ in readings)
            gain = claim[0] - d0[0]
            combined = math.sqrt(claim[1] ** 2 + d0[1] ** 2 + ELO_PER_Z ** 2 * sum(s * s for _, s in readings))
            if gain - forecast < -2 * combined:
                return (f"STOP after round {k}'s claim: claim - D0 reads {gain:+.1f} Elo against {forecast:+.1f} "
                        f"forecast from the confirmed readings, below by more than 2 SE ({combined:.1f}): gains do "
                        f"not transfer; the user picks a magnet or a pool (design §6)")
        return None


def pure(fit_json: Path, label: str) -> Tuple[float, float]:
    """A label's PURE Elo and SE from elo_collect's --save-json."""
    table = json.loads(Path(fit_json).read_text(encoding="utf-8"))["tables"]["PURE (decisive only, primary)"]
    return float(table[label]["elo"]), float(table[label]["se"])


def run(r: Rounds, rounds: int) -> int:
    """Rounds 1..`rounds`: 0 when all ran, 1 on a failed part, 2 on STOP, 3
    when a design §6 stop fires."""
    out = r.out
    try:
        for k in range(1, rounds + 1):
            r.check_stop(f"round {k}")
            if k == 2:
                why = r.p_gate()
                if why:
                    log(out, f"ROUNDS_WAIT before round 2: {why}")
                    return 3
            if not (r.round1() if k == 1 else r.round_k(k)):
                log(out, f"ROUNDS_FAILED at round {k}: {' '.join(r.state)}")
                return 1
            dec = r.round_json(k)["decision"]
            log(out, f"round {k}: adopt={dec.get('adopt')} next_eps={dec.get('next_eps')}: {dec.get('reason')}")
            if k % CLAIM_EVERY == 0:
                claim = r.claim(k)
                if claim is None:
                    log(out, f"ROUNDS_FAILED at round {k}'s claim: {' '.join(r.state)}")
                    return 1
                log(out, f"round {k}'s claim: {claim[0]:+.1f} +- {claim[1]:.1f} Elo against the reference")
                why = r.stops(k, claim)
                if why:
                    (out / "STOP").write_text(why + "\n", encoding="utf-8")
                    log(out, why)
                    return 3
            (out / "progress.txt").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\nround {k} done; policy "
                                              f"{r.policy_after(k)}\n", encoding="utf-8")
    except Stopped as e:
        log(out, f"STOPPED: {e}; a new start resumes")
        return 2
    log(out, f"ROUNDS_DONE through round {rounds}; policy {r.policy_after(rounds)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1", type=Path, required=True, help="the local stage 1's run directory (D0, round 1)")
    ap.add_argument("--stage-c", type=Path, required=True, help="the local stage C's out (stage_c.json, pass.pt)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--p-out", type=Path, default=None, help="stage P's out (scripts/p_local.py): round 2 waits on it")
    ap.add_argument("--no-p-gate", action="store_true", help="play the rounds without stage P (the user's waiver)")
    ap.add_argument("--checkpoint", type=Path, default=REPO / "training" / "checkpoints" / "own2.pt",
                    help="pi_0, the network stage 1 played (own2)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--python", default=sys.executable, help="the GPU tools' interpreter (the ROCm venv)")
    ap.add_argument("--tools-python", default=None, help="the CPU tools' (default: the repository's .venv)")
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
    log(args.out, f"rounds local: stage 1 {args.stage1}, stage C {args.stage_c}, out {args.out}, "
                  f"rounds {args.rounds}, pi_0 {args.checkpoint}")
    return run(Rounds(args), args.rounds)


if __name__ == "__main__":
    sys.exit(main())
