"""The design's rounds on the local machine (Windows, the RX 6600;
docs/training_design_20261019/current_design.md §1, §6 "Rounds" and
"Claims"), after stage C (scripts/stage_c_local.py), round after round:

- **Stage C's gate** (its stage_c.json "gate", design §6) decides whether
  rounds run and at what size: "resolved" plays N training games a round
  (N_TRAIN, or --n-train: prereg departure 25); "rounds_at_size" plays the
  gain curve's `round_size`; "neither",
  or a gate stage C did not read, plays no round (the user chooses). The
  held-out set H stays N_HELD games either way, or --held's (prereg
  departure 34). --no-p-gate does not waive it.
  One "neither" is read on: when its ladder's chosen rung failed half B's
  importance-weight check (the cross-play case below), round 1 reads that
  rung by a cross-play match, and the rounds go on at N iff the match's
  lower 2-SE bound is above 0 (prereg departure 23). A round's N and |H|
  are recorded when its games start (its n_train.txt and held.txt), so a
  new start with another --n-train or --held finishes a round at the sizes
  it began with; a round whose games started before held.txt existed
  played N_HELD.
- **P before the rounds** (design §6): round 2 waits on stage P's planted
  readout passing and its link holding, or the link not run (stage 1's
  sharpening slope under 3 SE).

- **Round 1** is stage C's ladder on stage 1's pass (design §6). Its
  decision is taken as read and its candidate written (tools/pg_round.py
  --adopt-from).
- **Round k >= 2**, from the policy pi the previous round left:
  - its games: N + |H| temperature-1 self-play games of pi on both
    seats, walked, labelled and read, the last |H| held out
    (scripts/selfplay_t1_local.py --checkpoint pi --no-measures, on seeds of
    the round's own);
  - the labels' corrections, fitted on its held-out games
    (tools/pg_corrections.py);
  - the gradient pass over its training games (tools/pg_pass.py in fp16,
    with the corrections);
  - the step (tools/pg_round.py): the ladder at the radius the previous
    round left, under stage C's arm, its decision and, when a rung is
    adopted, the next pi.
- **The cross-play case** (any round; wesnoth_ai/pg_cross_play.py, design
  §5): when the rung the step chose fails half B's importance-weight check,
  its network (pg_round --chosen-network) plays the round's pi at
  temperature 1, XP_GAMES games, and the match's mean z takes half B's
  place in the decision, which round.json then holds (the importance
  decision under "importance").
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

  Either writes STOP with every reason that fired; the chain waits for the
  user. The claims' upper tail (claim_k - D0 above the forecast by more
  than 2 combined SE) is logged and the rounds go on (design §6: reported).

**--end-by** (a local time, "YYYY-MM-DD HH:MM"): the rounds must end by
then. Round k >= 3 starts only when END_BY_MARGIN times the longest round
so far (rounds 2 on, from `times.json`) fits before it. Otherwise the
rounds end there, done (exit 0), so the window's final match can start on
time. A round already started is finished.

**After a design stop** (prereg departure 33), the rounds can go on past
it in the same --out, and three options shape them:
- `--arm NAME`: the step's arm (tools/pg_round.py --arm) in place of
  stage C's winner; the pass is heads-only only when this arm reads the
  heads alone and --heads-only-pass is given;
- `--pass-window N`: the pass's window (tools/pg_pass.py --window), N
  decisions; one larger than any game-side carries the memory's gradient
  through the whole game-side;
- `--rebase-after K0`: the stops of rounds K0 and before were acted on and
  are not read again, and the stops after K0 read only the rounds after
  it: the pause sums the confirmed readings after K0, and a claim is tested
  against round K0's claim in place of D0.
The STOP the stop wrote is moved aside by the caller (scripts/chain_local.py
--after-stop) before the rounds resume.

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
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.local_machine import (Steps, Stopped, below_normal_priority, default_tools_python,  # noqa: E402
                                   display_resets_since, keep_awake, log, reference_flags, require_visible_gpu,
                                   run_tool, stage_c_unfinished)
from scripts.selfplay_t1_local import CHUNK_GAMES, MIN_FREE_MB, player_flags  # noqa: E402
from wesnoth_ai.pg_cross_play import cross_play_case, decide_by_match, mean_z  # noqa: E402

N_TRAIN = 20000                 # design §1: N, a round's training games (stage C's gate may name another)
N_HELD = 10000                  # |H|, its held-out games (--held: prereg departure 34)
MIN_HELD = 4                    # each half of H holds a pair of games: tools/pg_labels.py half_of
# The seeds, so that no two matches share one. A match's replacement games
# take seed + gen * 1,000,000 (tools/run_elo_batch.py REPL_SEED_OFFSET):
# - stage 1's games 800,000-848,199, its measures' replacements under 10M;
# - round k's games (no replacements) from SEED_BASE + k * SEED_STRIDE, a
#   block of 1M for each of rounds 1-MAX_ROUNDS, all under stage P's 50M;
# - stage P at 50M and the link at 60M (scripts/p_local.py, no replacements);
# - round k's cross-play match (no replacements) from XP_SEED_BASE + k *
#   XP_SEED_STRIDE, between the link's block and the claims';
# - round k's claim from CLAIM_SEED_BASE + k * CLAIM_SEED_STRIDE, which holds
#   its replacements of generations 1-9.
SEED_BASE = 10_000_000
SEED_STRIDE = 1_000_000
MAX_ROUNDS = 39                 # round 39's block ends at 50M, where stage P's starts
CLAIM_SEED_BASE = 100_000_000
XP_SEED_BASE = 70_000_000
XP_SEED_STRIDE = 100_000
# The cross-play case's match (design §3 "Range": a 10,000-game cross-play
# match against pi_k). A match that read fewer than XP_MIN_SHARE of its games
# fails its part: the rule adopts unless the reading is below -2 SE, so a
# short match must not stand in for one.
XP_GAMES = 10000
XP_MIN_SHARE = 0.9
CLAIM_SEED_STRIDE = 10_000_000
CLAIM_EVERY = 3
CLAIM_GAMES = 12800             # design §3.2: a claim's decisive games, D0's
# Elo per unit of E[z] at a win rate of 1/2: Elo = 400 log10(p / (1 - p))
# and E[z] = 2 p - 1, so dElo / dE[z] = 200 / ln 10 / (p (1 - p)) = 347.4.
ELO_PER_Z = 347.4
FLOOR_SIDES = 100
# The arms that read the head maps' gradient alone, so a heads-only pass
# serves them (--heads-only-pass): wesnoth_ai/pg_step.py HEADS_ONLY_ARMS,
# which tests/test_rounds_local.py holds equal (this file imports no torch).
HEADS_ONLY_ARMS = ("heads_only", "probe_span")
ARMS = ("sgd", "reduced_adamw", "muon", "heads_only", "probe_span")    # wesnoth_ai/pg_step.py ARMS, held equal too
END_BY_MARGIN = 1.2             # --end-by: a round may run this much longer than the longest so far
END_BY_FORMAT = "%Y-%m-%d %H:%M"
PASS_FP16 = ["--fp16", "--packed-trunk", "--floor-chunk", "8", "--floor-sides", str(FLOOR_SIDES)]
POOL = ["--persistent-workers", "--shared-inference", "--no-infer-compile", "--device", "cuda", "--jobs", "96",
        "--worker-games", "8", "--min-free-mb", str(MIN_FREE_MB), "--infer-fp16", "--inference-pipelined",
        "--inference-timeline-s", "30"]
PLAY_CUT_MIN = 16 * 60          # 30,000 games at about 5,000 an hour, played again once when short
CORRECTIONS_CUT_MIN = 60
PASS_CUT_MIN = 48 * 60          # 20,000 games at 80 to 150 decisions a second: 14 to 26 h, unmeasured
STEP_CUT_MIN = 36 * 60          # the ladder's forwards over 10,000 held-out games, unmeasured
CLAIM_CUT_MIN = 10 * 60
XP_CUT_MIN = 6 * 60             # 10,000 games at about 3,450 an hour (scripts/match_local.py's estimate)
PASS_EXITS = {3: "the replay identity failed", 4: "the precision floor's bound lifts the replay limit past its cap",
              5: "the packed and padded trunks disagree in fp32"}


class Rounds(Steps):
    def __init__(self, args, runner: Callable[..., int] = run_tool,
                 resets: Callable[[float], int] = display_resets_since, now: Callable[[], float] = time.time):
        super().__init__(args.out, args.python, args.tools_python, runner, resets)
        self.now = now
        self.a = args
        self.s1, self.sc = Path(args.stage1), Path(args.stage_c)
        self.pi0 = str(args.checkpoint)
        self.workers = str(args.workers)
        self.n_design = int(getattr(args, "n_train", None) or N_TRAIN)     # the gate's N where it names none
        self.n_train = self.n_design
        self.held = int(getattr(args, "held", None) or N_HELD)          # |H| for rounds whose games start now
        self.xp_gate = False            # round 1's cross-play match decides whether the rounds go on

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

    def stage_c_gate(self) -> Tuple[Optional[int], str]:
        """(N, its source) from stage C's gate (module docstring), or (None,
        why no round runs)."""
        unfinished = stage_c_unfinished(self.sc)
        if unfinished:
            return None, f"{unfinished}: no rounds"
        f = self.sc / "stage_c.json"
        gate = json.loads(f.read_text(encoding="utf-8")).get("gate") if f.is_file() else None
        verdict = (gate or {}).get("verdict")
        sized = "" if self.n_design == N_TRAIN else f" (--n-train {self.n_design}, prereg departure 25)"
        if verdict == "resolved":
            return self.n_design, "stage C's gate: resolved" + sized
        if verdict == "rounds_at_size":
            size = gate.get("round_size")
            if isinstance(size, (int, float)) and math.isfinite(size) and size >= 1:
                n = int(math.ceil(size))
                return n, f"stage C's gate: rounds at the gain curve's size {n}"
            return None, f"stage C's gate says rounds_at_size with no finite round_size ({size!r})"
        if verdict == "neither":
            ladder = json.loads(f.read_text(encoding="utf-8")).get("ladder") or {}
            if cross_play_case(ladder, ladder.get("rungs") or []) is not None:
                self.xp_gate = True
                return self.n_design, ("stage C's gate reads neither because its ladder's rung fails half B's "
                                       "importance-weight check: round 1 reads the rung by a cross-play match, and "
                                       "the rounds go on iff it resolves (prereg departure 23)" + sized)
            return None, f"stage C's gate reads neither ({gate.get('reason')}): no rounds; the user chooses"
        return None, f"stage C's gate is {verdict or 'missing'} ({f}): no rounds until it reads"

    def p_gate(self) -> Optional[str]:
        """Design §6: P before the rounds. None when stage P read back and the
        link held or was not run (scripts/p_local.py's p_readout.json and
        link.json in --p-out), or when the gate is waived (--no-p-gate); why
        not otherwise."""
        if self.a.p_out is None:
            return None if self.a.no_p_gate else "no --p-out: stage P (scripts/p_local.py) has not been named"
        reads = {}
        for name in ("p_readout.json", "link.json"):
            f = Path(self.a.p_out) / name
            if not f.exists():
                return f"stage P's {name} is missing from {self.a.p_out}"
            reads[name] = json.loads(f.read_text(encoding="utf-8")).get("verdict")
        if reads["p_readout.json"] != "pass" or reads["link.json"] not in ("holds", "not run"):
            return f"stage P reads {reads}: the rounds wait (design §6)"
        return None

    def round_size(self, k: int) -> int:
        """Round k's N: the one recorded when its games started, else the
        rounds' N, recorded now."""
        f = self.rdir(k) / "n_train.txt"
        if not f.exists():
            f.write_text(f"{self.n_train}\n", encoding="utf-8")
        return int(f.read_text(encoding="utf-8"))

    def held_size(self, k: int) -> int:
        """Round k's |H|: the one recorded when its games started; else
        N_HELD for a round whose games started before held.txt was written
        (every round before prereg departure 34); else the rounds' |H|,
        recorded now."""
        f = self.rdir(k) / "held.txt"
        if not f.exists():
            started = (self.rdir(k) / "play").exists()
            f.write_text(f"{N_HELD if started else self.held}\n", encoding="utf-8")
        return int(f.read_text(encoding="utf-8"))

    def game_dirs(self, k: int) -> List[str]:
        play = self.rdir(k) / "play"
        chunks = (self.round_size(k) + self.held_size(k) + CHUNK_GAMES - 1) // CHUNK_GAMES
        return [str(play / f"games_s{c:02d}") for c in range(chunks)]

    # ---- the rounds' walls (--end-by)
    def _times(self) -> dict:
        path = self.out / "times.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def timed(self, k: int, event: str) -> None:
        """Records round k's start or end, the first of each kept across
        restarts: a resume walks the finished rounds again, and must not
        move their ends (their walls bound --end-by)."""
        times = self._times()
        entry = times.setdefault(str(k), {})
        if event not in entry:
            entry[event] = self.now()
        (self.out / "times.json").write_text(json.dumps(times, indent=1), encoding="utf-8")

    def no_time(self, k: int) -> Optional[str]:
        """Why round k must not start under --end-by, or None."""
        end_by = getattr(self.a, "end_by", None)
        times = self._times()
        if end_by is None or k < 3 or "start" in times.get(str(k), {}):
            return None
        walls = [t["end"] - t["start"] for j, t in times.items() if int(j) >= 2 and "end" in t and "start" in t]
        if not walls:
            return None
        need = END_BY_MARGIN * max(walls)
        deadline = datetime.strptime(end_by, END_BY_FORMAT).timestamp()
        if self.now() + need <= deadline:
            return None
        return (f"round {k} would need about {need / 3600:.1f} h ({END_BY_MARGIN} times the longest round), "
                f"past --end-by {end_by}")

    def pass_scope(self) -> List[str]:
        """["--heads-only"] when --heads-only-pass is given and stage C's arm
        reads the head maps alone (HEADS_ONLY_ARMS); else nothing, the
        full pass."""
        if not self.a.heads_only_pass:
            return []
        return ["--heads-only"] if self.arm() in HEADS_ONLY_ARMS else []

    def arm(self) -> Optional[str]:
        """The step's arm: --arm, else stage C's winner."""
        if getattr(self.a, "arm", None):
            return self.a.arm
        stage_c = json.loads((self.sc / "stage_c.json").read_text(encoding="utf-8"))
        return (stage_c.get("optimizer") or {}).get("winner")

    def arm_args(self) -> List[str]:
        """pg_round's arm: --arm when given, else stage C's winner read there."""
        if getattr(self.a, "arm", None):
            return ["--arm", self.a.arm]
        return ["--arm-from", str(self.sc / "stage_c.json")]

    def window_args(self) -> List[str]:
        w = getattr(self.a, "pass_window", None)
        return ["--window", str(w)] if w else []

    def factory_args(self, checkpoint: str, dirs: List[str], labels: Path, cache: Path) -> List[str]:
        """The evaluator's arguments; its readings are kept in `cache`, so a
        restarted ladder resumes (tools/pg_eval_cache.py)."""
        out = []
        for key, value in (("checkpoint", checkpoint), ("games", ",".join(dirs)), ("labels", labels),
                           ("device", "cuda"), ("workers", self.workers), ("fp16", "1"), ("cache", cache)):
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
                                        *self.factory_args(self.pi0, dirs, self.s1 / "labels.jsonl", d / "eval_cache"),
                                        "--adopt-from", str(self.sc / "stage_c.json"), "--out", str(d / "round.json"),
                                        "--candidate", str(d / "pi.pt"), "--direction-out",
                                        str(d / "direction.pt"), "--chosen-network", str(d / "rung.pt")],
                         "round_01.log", STEP_CUT_MIN)
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
                                                str(SEED_BASE + k * SEED_STRIDE), "--games",
                                                str(self.round_size(k) + self.held_size(k)), "--heldout-from",
                                                str(self.round_size(k)),
                                                "--no-measures"],
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
                                               "cuda", "--workers", self.workers, *PASS_FP16, *self.pass_scope(),
                                               *self.window_args(),
                                               "--corrections", str(d / "corrections.json")], f"round_{k:02d}.log",
                                   PASS_CUT_MIN)
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
                                               *self.factory_args(pi, dirs, labels, d / "eval_cache"), "--eps-from",
                                               str(prev / "round.json"), *self.arm_args(), *previous, "--out", str(d / "round.json"), "--candidate",
                                               str(d / "pi.pt"), "--direction-out", str(d / "direction.pt"),
                                               "--chosen-network", str(d / "rung.pt")],
                           f"round_{k:02d}.log", STEP_CUT_MIN)
            ok = rc == 0 and (d / "round.json").exists()
            self.mark(f"{tag}_step", None if ok else f"round {k}'s step ended rc={rc} (round_{k:02d}.log)")
        return self.done(f"{tag}_step")

    # ---- the cross-play case
    def rungs_of(self, k: int) -> list:
        """Round k's rungs: its ladder's, or stage C's for round 1."""
        rungs = self.round_json(k)["decision"].get("rungs")
        if not rungs:
            rungs = json.loads((self.sc / "stage_c.json").read_text(encoding="utf-8"))["ladder"]["rungs"]
        return rungs

    def cross_play(self, k: int) -> bool:
        """Round k's chosen rung read by a cross-play match when its step
        left it to one (module docstring); True when round k's decision
        stands, read or not needed."""
        tag, d = f"r{k:02d}_xp", self.rdir(k)
        r = self.round_json(k)
        net = r.get("chosen_network")
        if net is None or "cross_play" in r["decision"]:
            return True
        games, label = d / "cross_play", f"rung{k}"
        ref_a = reference_flags("a", self.tools_py)
        if not self.skip(tag):
            self.check_stop(f"round {k}'s cross-play match")
            side_a = player_flags(ref_a, "a", label, spec=net["path"])
            side_b = player_flags(reference_flags("b", self.tools_py), "b", f"pi{k - 1}", spec=self.policy_after(k - 1))
            rc = self.play_match(f"round {k} cross-play", ["tools/run_elo_batch.py", *side_a, *side_b, "--mcts-sims",
                                                           "0", "--raw-temperature-a", "1", "--raw-temperature-b", "1",
                                                           "--outdir", str(games), "--games", str(XP_GAMES),
                                                           "--max-extra-games", "0", "--seed-base",
                                                           str(XP_SEED_BASE + k * XP_SEED_STRIDE), *POOL],
                                 f"round_{k:02d}.log", XP_CUT_MIN)
            self.mark(tag, None if rc in (0, 4) else f"round {k}'s cross-play match ended rc={rc} (round_{k:02d}.log)")
        if not self.done(tag) or self.skip(f"{tag}_read"):
            return False
        slots = int(ref_a[ref_a.index("--memory-a") + 1])
        reading = mean_z(games, label, net["sha256"], slots)
        if reading["games_refused"] or reading["games"] < XP_MIN_SHARE * XP_GAMES:
            self.mark(f"{tag}_read", f"round {k}'s cross-play match read {reading['games']} games of {XP_GAMES}, "
                                     f"{reading['games_refused']} refused ({reading['refused_examples'][:2]})")
            return False
        r["decision"] = decide_by_match(r["decision"], self.rungs_of(k), reading)
        if r["decision"]["adopt"]:
            pi = d / "pi.pt"
            pi.write_bytes(Path(net["path"]).read_bytes())
            r["candidate"] = {"path": str(pi), "sha256": net["sha256"], "s": net["s"], "from": "the cross-play match"}
        tmp = d / "round.json.tmp"
        tmp.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
        tmp.replace(d / "round.json")
        log(self.out, f"round {k}'s cross-play match: {reading['estimate']:+.4f} +- {reading['se']:.4f} over "
                      f"{reading['games']} games ({reading['games_left_out']} left out)")
        return True

    def xp_resolved(self) -> Optional[str]:
        """Why round 1's cross-play match stops the rounds (its lower 2-SE
        bound is not above 0), or None."""
        xp = self.round_json(1)["decision"].get("cross_play")
        if xp is None:
            return "round 1 holds no cross-play reading"
        lower = xp["estimate"] - 2.0 * xp["se"]
        if lower > 0:
            return None
        return (f"round 1's cross-play match reads {xp['estimate']:+.4f} +- {xp['se']:.4f}, its lower 2-SE bound "
                f"{lower:+.4f} not above 0: stage C's gate stays neither (prereg departure 23)")

    # ---- the readings and the claims
    def confirmed(self, k: int) -> Tuple[float, float]:
        """Round k's confirmed reading (half B's ΔĴ of its adopted rung, or
        its cross-play match's mean z, in units of E[z]) and its SE; (0, 0)
        for a round that kept pi."""
        r = self.round_json(k)
        dec = r["decision"]
        if not dec.get("adopt"):
            return 0.0, 0.0
        if "cross_play" in dec:
            return float(dec["cross_play"]["estimate"]), float(dec["cross_play"]["se"])
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
            rc = self.play_match(f"round {k} claim", ["tools/run_elo_batch.py", *side_a, *ref_b, "--mcts-sims", "0",
                                                      "--raw-temperature-a", "1", "--raw-temperature-b", "0",
                                                      "--outdir", str(games), "--games", str(CLAIM_GAMES),
                                                      "--max-extra-games", str(CLAIM_GAMES // 2), "--seed-base",
                                                      str(CLAIM_SEED_BASE + k * CLAIM_SEED_STRIDE), *POOL],
                                 f"claim_{k:02d}.log", CLAIM_CUT_MIN)
            if rc in (0, 4):
                rc = self.step(f"round {k} claim fit", ["tools/elo_collect.py", str(games), "--no-catalog",
                                                        "--anchor", ref_b[ref_b.index("--label-b") + 1],
                                                        "--save-json", str(fit)], f"claim_{k:02d}.log", 15, gpu=False)
            self.mark(tag, None if rc == 0 and fit.exists() else f"the claim ended rc={rc} (claim_{k:02d}.log)")
        if not self.done(tag):
            return None
        return pure(fit, label)

    def stops(self, k: int, claim: Optional[Tuple[float, float]]) -> Tuple[Optional[str], List[str]]:
        """Design §6's two stops after round k: (every stop that fired, joined,
        or None to go on; the notes to log, the claims' upper tail)."""
        why, notes = [], []
        k0 = getattr(self.a, "rebase_after", None) or 0
        if k <= k0:
            return None, [f"round {k}'s stops were acted on: the rounds go on after round {k0} (--rebase-after, "
                          f"prereg departure 33)"]
        last = [self.confirmed(j) for j in range(max(1, k0 + 1, k - CLAIM_EVERY + 1), k + 1)]
        total, se = sum(e for e, _ in last), math.sqrt(sum(s * s for _, s in last))
        if not total > 2 * se:
            why.append(f"PAUSE after round {k}: the last {len(last)} confirmed readings sum to {total:.4g} "
                       f"(SE {se:.3g}), not above 0 by 2 SE (design §6)")
        if claim is not None:
            if k0:
                base, base_name = pure(self.rdir(k0) / "claim.fit.json", f"pi{k0}t1"), f"round {k0}'s claim"
            else:
                base, base_name = pure(self.s1 / "d0.fit.json", "own2t1"), "D0"
            readings = [self.confirmed(j) for j in range(k0 + 1, k + 1)]
            forecast = ELO_PER_Z * sum(e for e, _ in readings)
            gain = claim[0] - base[0]
            combined = math.sqrt(claim[1] ** 2 + base[1] ** 2 + ELO_PER_Z ** 2 * sum(s * s for _, s in readings))
            read = (f"claim - {base_name} reads {gain:+.1f} Elo against {forecast:+.1f} forecast from the "
                    f"confirmed readings (combined SE {combined:.1f})")
            if gain - forecast < -2 * combined:
                why.append(f"STOP after round {k}'s claim: {read}, below by more than 2 SE: gains do not "
                           f"transfer; the user picks a magnet or a pool (design §6)")
            elif gain - forecast > 2 * combined:
                notes.append(f"UPPER_TAIL at round {k}'s claim: {read}, above by more than 2 SE: play gains "
                             f"more than the labels price (design §6: reported)")
            else:
                notes.append(f"round {k}'s claim: {read}, within 2 SE")
        return ("; ".join(why) or None), notes


def pure(fit_json: Path, label: str) -> Tuple[float, float]:
    """A label's PURE Elo and SE from elo_collect's --save-json."""
    table = json.loads(Path(fit_json).read_text(encoding="utf-8"))["tables"]["PURE (decisive only, primary)"]
    return float(table[label]["elo"]), float(table[label]["se"])


def run(r: Rounds, rounds: int) -> int:
    """Rounds 1..`rounds`: 0 when all ran, 1 on a failed part, 2 on STOP, 3
    when a design §6 stop fires."""
    out = r.out
    if rounds > MAX_ROUNDS:
        log(out, f"ROUNDS_REFUSED: {rounds} rounds, past the {MAX_ROUNDS} the seed blocks hold (SEED_BASE)")
        return 1
    n, source = r.stage_c_gate()
    if n is None:
        log(out, f"ROUNDS_WAIT before round 1: {source}")
        return 3
    if n + r.held > SEED_STRIDE:
        log(out, f"ROUNDS_REFUSED: rounds of {n} + {r.held} games overrun a round's {SEED_STRIDE} seeds")
        return 1
    r.n_train = n
    log(out, f"rounds of N {n} + |H| {r.held} games ({source}; a round started earlier keeps its own |H|)")
    try:
        for k in range(1, rounds + 1):
            r.check_stop(f"round {k}")
            late = r.no_time(k)
            if late:
                log(out, f"ROUNDS_DONE through round {k - 1}: {late}; policy {r.policy_after(k - 1)}")
                return 0
            if k == 2:
                why = r.p_gate()
                if why:
                    log(out, f"ROUNDS_WAIT before round 2: {why}")
                    return 3
            r.timed(k, "start")
            if not ((r.round1() if k == 1 else r.round_k(k)) and r.cross_play(k)):
                log(out, f"ROUNDS_FAILED at round {k}: {' '.join(r.state)}")
                return 1
            dec = r.round_json(k)["decision"]
            log(out, f"round {k}: adopt={dec.get('adopt')} next_eps={dec.get('next_eps')}: {dec.get('reason')}")
            if k == 1 and r.xp_gate:
                why = r.xp_resolved()
                if why:
                    log(out, f"ROUNDS_WAIT after round 1: {why}")
                    return 3
            if k % CLAIM_EVERY == 0:
                claim = r.claim(k)
                if claim is None:
                    log(out, f"ROUNDS_FAILED at round {k}'s claim: {' '.join(r.state)}")
                    return 1
                log(out, f"round {k}'s claim: {claim[0]:+.1f} +- {claim[1]:.1f} Elo against the reference")
                why, notes = r.stops(k, claim)
                for note in notes:
                    log(out, note)
                if why:
                    (out / "STOP").write_text(why + "\n", encoding="utf-8")
                    log(out, why)
                    return 3
            r.timed(k, "end")
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
    ap.add_argument("--n-train", type=int, default=N_TRAIN,
                    help="N, a round's training games where stage C's gate names no size (prereg departure 25)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--python", default=sys.executable, help="the GPU tools' interpreter (the ROCm venv)")
    ap.add_argument("--tools-python", default=None, help="the CPU tools' (default: the repository's .venv)")
    ap.add_argument("--end-by", default=None, metavar="'YYYY-MM-DD HH:MM'",
                    help="the rounds must end by this local time (module docstring)")
    ap.add_argument("--heads-only-pass", action="store_true",
                    help="each round's pass takes the head maps' gradient alone when stage C's arm reads no "
                         "other (HEADS_ONLY_ARMS; tools/pg_pass.py --heads-only)")
    ap.add_argument("--arm", choices=ARMS, default=None,
                    help="the step's arm in place of stage C's winner (module docstring, prereg departure 33)")
    ap.add_argument("--pass-window", type=int, default=None,
                    help="the pass's window in decisions (tools/pg_pass.py --window; prereg departure 33)")
    ap.add_argument("--rebase-after", type=int, default=None, metavar="K0",
                    help="the stops of rounds K0 and before were acted on (module docstring, prereg departure 33)")
    ap.add_argument("--held", type=int, default=N_HELD,
                    help="|H|, a round's held-out games, for rounds whose games start from now (prereg departure "
                         "34); a round keeps the |H| it started with (its held.txt)")
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
    if args.end_by is not None:
        datetime.strptime(args.end_by, END_BY_FORMAT)               # refused now, not at round 3
    if args.n_train < 1:
        ap.error(f"--n-train {args.n_train}: a round trains on at least one game")
    if args.held < MIN_HELD:
        ap.error(f"--held {args.held}: each half of H needs a pair of games (tools/pg_labels.py half_of), so "
                 f"|H| is at least {MIN_HELD}")
    if args.pass_window is not None and args.pass_window < 1:
        ap.error(f"--pass-window {args.pass_window}: a window holds at least one decision")
    if args.rebase_after is not None and args.rebase_after < 1:
        ap.error(f"--rebase-after {args.rebase_after}: name the round whose stop was acted on")
    log(args.out, f"rounds local: stage 1 {args.stage1}, stage C {args.stage_c}, out {args.out}, "
                  f"rounds {args.rounds}, pi_0 {args.checkpoint}, |H| {args.held}, end by {args.end_by}, "
                  f"heads-only pass {args.heads_only_pass}, arm {args.arm or 'from stage C'}, pass window "
                  f"{args.pass_window or 'the pass default'}, rebase after {args.rebase_after}")
    return run(Rounds(args), args.rounds)


if __name__ == "__main__":
    sys.exit(main())
