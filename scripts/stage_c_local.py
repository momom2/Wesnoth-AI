"""Stage C of the training design on the local machine (Windows, the RX 6600;
docs/training_design_20261019/current_design.md §6): the parts of
scripts/stage_c_box.sh on the local stage 1's run directory
(scripts/selfplay_t1_local.py), without the box library.

    C0: the labels' per-decision corrections fitted on stage 1's held-out
        games (tools/pg_corrections.py; every pass reads them), the pass on
        the first C0_GAMES training games (the replay identity, the
        precision floor, the trainer's rate) and one replay of C0_HELD
        held-out games (the forwards' rate and their identity);
    the plan from C0's rates (tools/stage_c_plan.py): the arms' trim and
        each step's cut;
    the pass over every training game;
    the stage C driver (tools/pg_stage_c.py on tools/pg_eval.py's factory);
    the planted offline readout (tools/pg_plant.py, tools/selfplay_rows.py,
        tools/pg_labels.py, a planted pass);
    the verdict (verdict.json), by the box script's rule.

A failed C0 or pass stops what needs it, a failed driver leaves the plant to
run and the other way round, and a part's failure stays failed at a new
start (its .failed file: delete it to run the part again); a done part is
not run again. A STOP file in --out ends the run between steps, and a
display-driver reset (System event 4101) during a GPU step writes STOP and
ends it.

What differs from the box script:
- Stage 1 is read in place: --stage1 is the local run's directory (its
  games, walks, labels and readout); the walks' join goes to --out.
- The RX 6600's arithmetic: the passes with --fp16 --packed-trunk, the
  forwards with --fp16 (pg_eval's, the plant's, the driver's fp16=1). The
  main and planted passes' floor reads FLOOR_SIDES game-sides (C0's reads
  every game-side of its games, as on the box), FLOOR_CHUNK rows a forward:
  its padded fp32 forward takes about 3 s at 64 rows here.
- No dead-man's switch: the plan's time left is unbounded, and each step's
  cut is the plan's (tools/stage_c_plan.py: 1.5 times its estimate plus 15
  minutes).
- No uploads, and no CPU tests on the box: the desktop's full suite stands
  in for them (the run's pre-registration names it).
- The driver's gate weighs the box's prices (PRICE_PER_GAME, ROUND_FIXED):
  the design's yardstick for a round's size, not this machine's costs.
- The passes weigh each decision by its corrected label (tools/pg_pass.py
  --corrections): stage 1 found the value head's correction entering, and
  the luck dealt so far enters beside it
  (training/metrics/luck_baseline_20261030); the box script predates both.

    HIP_VISIBLE_DEVICES=1 <rocm venv>/python scripts/stage_c_local.py \\
        --stage1 C:/Users/<you>/wesnoth_runs/selfplay_t1_local --out C:/Users/<you>/wesnoth_runs/stage_c_local
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.local_machine import (below_normal_priority, default_tools_python, display_resets_since,  # noqa: E402
                                   keep_awake, log, require_visible_gpu, run_tool)
from scripts.selfplay_t1_local import CHUNK_GAMES, GATES, HELDOUT_FROM, N_SELFPLAY, SP_SEED_BASE  # noqa: E402

C0_GAMES = 200
C0_HELD = 100
N_PLANT = 5000
PLANT_READOUT_FRACTION = 0.2
PLANT_CLASS = "attack"
PRICE_PER_GAME = 7.76e-5        # design §2: c_g, a generated and trained game (the box's)
ROUND_FIXED = 2.42              # design §2: c_f's high end (the box's)
FLOOR_SIDES = 100               # the main and planted passes' floor (C0's reads all of its 400)
FLOOR_CHUNK = 8                 # rows a padded fp32 forward of the floor
C0_CUT_MIN = 120                # C0's pass reads the floor on each of its 75,600 decisions
SMALL_CUT_MIN = 30              # the plant's copy, rows and labels; the plan
PARTS = ("c0", "pass", "stage_c", "plant")
FP16 = ["--fp16"]
PASS_FP16 = ["--fp16", "--packed-trunk", "--floor-chunk", str(FLOOR_CHUNK)]

# pg_pass.py's exits (PassRefused.code) and what each means for the part.
PASS_EXITS = {3: "the replay identity failed", 4: "the precision floor's bound lifts the replay limit past its cap",
              5: "the packed and padded trunks disagree in fp32"}


class Stopped(Exception):
    """The run ends here: a STOP file, or a display reset during a step."""


class Run:
    def __init__(self, args, runner: Callable[..., int] = run_tool,
                 resets: Callable[[float], int] = display_resets_since):
        self.a = args
        self.out: Path = args.out
        self.s1: Path = args.stage1
        self.ck = str(args.checkpoint)
        self.py, self.tools_py = args.python, args.tools_python
        self.workers = str(args.workers)
        self.runner, self.resets = runner, resets
        self.state: List[str] = []
        self.dirs: List[str] = []
        self.cuts: Dict[str, float] = {}
        self.trim = False

    # ---- markers
    def done(self, part: str) -> bool:
        return (self.out / f"{part}.done").exists()

    def failed_before(self, part: str) -> Optional[str]:
        f = self.out / f"{part}.failed"
        return f.read_text(encoding="utf-8").strip() if f.exists() else None

    def mark(self, part: str, why: Optional[str] = None) -> None:
        name = f"{part}.failed" if why else f"{part}.done"
        (self.out / name).write_text((why or time.strftime("%Y-%m-%d %H:%M:%S")) + "\n", encoding="utf-8")
        self.state.append(f"{part}: {'FAILED, ' + why if why else 'done'};")

    def skip(self, part: str) -> bool:
        """True when `part` is done or failed at an earlier start."""
        if self.done(part):
            self.state.append(f"{part}: done;")
            return True
        why = self.failed_before(part)
        if why:
            self.state.append(f"{part}: FAILED at an earlier start, {why};")
            return True
        return False

    def check_stop(self, before: str) -> None:
        if (self.out / "STOP").exists():
            raise Stopped(f"STOP found before {before}")

    # ---- one step
    def step(self, name: str, cmd: List[str], log_name: str, cut_min: float, gpu: bool = True) -> int:
        t0 = time.time()
        rc = self.runner(self.out, name, cmd, log_name, cut_min, self.py if gpu else self.tools_py)
        if gpu:
            n = self.resets(t0)
            if n:
                why = f"{n} display-driver reset(s) (System event 4101) during {name}"
                (self.out / "STOP").write_text(why + "\n", encoding="utf-8")
                raise Stopped(why)
        return rc

    def games(self) -> List[str]:
        return ["--games", *self.dirs]

    # ---- stage 1, read in place
    def restore(self) -> Optional[str]:
        """Stage 1's matches, walks, labels and readout; None, or why stage C
        cannot read them."""
        chunks = [f"s{c:02d}" for c in range((N_SELFPLAY + CHUNK_GAMES - 1) // CHUNK_GAMES)]
        missing = [f"games_{c}" for c in chunks if not (self.s1 / f"games_{c}").is_dir()]
        missing += [f"walk_{c}.jsonl" for c in chunks if not (self.s1 / f"walk_{c}.jsonl").is_file()]
        missing += [n for n in ("labels.jsonl", "fits.json", "readout.json") if not (self.s1 / n).is_file()]
        if missing:
            return f"missing from {self.s1}: {' '.join(missing)}"
        self.dirs = [str(self.s1 / f"games_{c}") for c in chunks]
        walk = self.out / "walk.jsonl"
        if not walk.exists():
            tmp = walk.with_name(walk.name + ".tmp")
            with open(tmp, "wb") as f:
                for c in chunks:
                    f.write((self.s1 / f"walk_{c}.jsonl").read_bytes())
            tmp.replace(walk)
        readout = json.loads((self.s1 / "readout.json").read_text(encoding="utf-8"))
        failed = [k for k in GATES if not (readout.get(k) or {}).get("pass")]
        return f"gates failed: {' '.join(failed)}" if failed else None

    # ---- the parts
    def _pass_refused(self, part: str, rc: int, what: str, log_name: str) -> bool:
        if rc == 0:
            return False
        self.mark(part, f"{PASS_EXITS.get(rc, f'the pass ended rc={rc}')} on {what} ({log_name})")
        return True

    def corrections(self) -> List[str]:
        return ["--corrections", str(self.out / "corrections.json")]

    def c0(self) -> bool:
        if self.skip("c0"):
            return self.done("c0")
        if not (self.out / "corrections.json").exists():
            rc = self.step("corrections", ["tools/pg_corrections.py", *self.games(), "--labels",
                                           str(self.s1 / "labels.jsonl"), "--fits", str(self.s1 / "fits.json"),
                                           "--workers", self.workers, "--out", str(self.out / "corrections.json")],
                           "c0.log", SMALL_CUT_MIN, gpu=False)
            if rc != 0 or not (self.out / "corrections.json").exists():
                self.mark("c0", f"the corrections' fit ended rc={rc} (c0.log)")
                return False
        rc = self.step("c0 pass", ["tools/pg_pass.py", "--checkpoint", self.ck, *self.games(), "--labels",
                                   str(self.s1 / "labels.jsonl"), "--train-limit", str(C0_GAMES), "--fresh",
                                   "--out", str(self.out / "c0_pass.pt"), "--device", "cuda", "--workers",
                                   self.workers, *PASS_FP16, *self.corrections()], "c0.log", C0_CUT_MIN)
        if self._pass_refused("c0", rc, f"{C0_GAMES} training games", "c0.log"):
            return False
        try:
            limit = float(json.loads((self.out / "c0_pass.json").read_text(encoding="utf-8"))["replay"]["limit"])
        except (OSError, KeyError, ValueError, TypeError):
            self.mark("c0", "c0_pass.json names no replay limit")
            return False
        rc = self.step("c0 forwards", ["tools/pg_eval.py", "time", "--checkpoint", self.ck, *self.games(),
                                       "--labels", str(self.s1 / "labels.jsonl"), "--held-games", str(C0_HELD),
                                       "--device", "cuda", "--workers", self.workers, *FP16,
                                       "--replay-kl-limit", repr(limit), "--out", str(self.out / "c0_forwards.json")],
                       "c0.log", C0_CUT_MIN)
        if rc == 3:
            self.mark("c0", f"the evaluator's replay identity failed on {C0_HELD} held-out games, limit {limit:g} "
                            "(c0.log, c0_forwards.json)")
            return False
        if rc != 0:
            self.mark("c0", f"the C0 forwards ended rc={rc} (c0.log)")
            return False
        self.mark("c0")
        return True

    def plan(self) -> bool:
        """The trim and the cuts from C0's rates (c0_plan.json, kept: a new
        start reads it)."""
        path = self.out / "c0_plan.json"
        if not path.exists():
            rc = self.runner(self.out, "plan", ["tools/stage_c_plan.py", "--pass-summary",
                                                str(self.out / "c0_pass.json"), "--forwards",
                                                str(self.out / "c0_forwards.json"), "--labels",
                                                str(self.s1 / "labels.jsonl"), "--n-plant", str(N_PLANT),
                                                "--readout-fraction", str(PLANT_READOUT_FRACTION), "--out", str(path)],
                             "c0_plan.log", SMALL_CUT_MIN, self.tools_py)
            if rc != 0 or not path.exists():
                self.state.append(f"plan: unpriced, C0's rates could not be read (rc={rc}, c0_plan.log);")
                return False
        plan = json.loads(path.read_text(encoding="utf-8"))
        self.trim = bool(plan["trim"])
        self.cuts = {k: float(plan["cuts"][k]) for k in ("pass", "stage_c", "plant")}
        self.state.append(f"plan: {plan['total_minutes']:.0f} min projected, trim={int(self.trim)};")
        return True

    def gradient_pass(self) -> bool:
        if self.skip("pass"):
            return self.done("pass")
        rc = self.step("pass", ["tools/pg_pass.py", "--checkpoint", self.ck, *self.games(), "--labels",
                                str(self.s1 / "labels.jsonl"), "--out", str(self.out / "pass.pt"), "--device",
                                "cuda", "--workers", self.workers, *PASS_FP16, "--floor-sides", str(FLOOR_SIDES),
                                *self.corrections()], "pass.log", self.cuts["pass"])
        if self._pass_refused("pass", rc, "every training game", "pass.log"):
            return False
        self.mark("pass")
        return True

    def stage_c(self) -> bool:
        if self.skip("stage_c"):
            return self.done("stage_c")
        arms, splits = (["sgd", "reduced_adamw", "muon", "probe_span"], ["4"]) if self.trim else \
            (["sgd", "reduced_adamw", "muon", "heads_only", "probe_span"], ["2", "4"])
        rc = self.step("stage c", ["tools/pg_stage_c.py", "--pass", str(self.out / "pass.pt"), "--factory",
                                   "tools.pg_eval:network_factory", "--factory-arg", f"checkpoint={self.ck}",
                                   "--factory-arg", "games=" + ",".join(self.dirs), "--factory-arg",
                                   f"labels={self.s1 / 'labels.jsonl'}", "--factory-arg", "device=cuda",
                                   "--factory-arg", f"workers={self.workers}", "--factory-arg", "fp16=1",
                                   "--price-per-game", str(PRICE_PER_GAME), "--round-fixed-cost", str(ROUND_FIXED),
                                   "--arms", *arms, "--curve-splits", *splits, "--out",
                                   str(self.out / "stage_c.json")], "stage_c.log", self.cuts["stage_c"])
        if rc != 0 or not (self.out / "stage_c.json").exists():
            self.mark("stage_c", f"the driver ended rc={rc} (stage_c.log; stage_c.json keeps its parts)")
            return False
        self.mark("stage_c")
        return True

    def plant(self) -> bool:
        if self.skip("plant"):
            return self.done("plant")
        half = self.cuts["plant"] / 2
        labels, planted = str(self.s1 / "labels.jsonl"), str(self.out / "labels_planted.jsonl")
        steps = [
            ("delta", half, True, self.out / "plant_grads.pt",
             ["tools/pg_plant.py", "delta", "--checkpoint", self.ck, *self.games(), "--labels", labels, "--pass",
              str(self.out / "pass.pt"), "--walk", str(self.out / "walk.jsonl"), "--seed-base", str(SP_SEED_BASE),
              "--train-games", str(N_PLANT), "--class", PLANT_CLASS, "--readout-fraction",
              str(PLANT_READOUT_FRACTION), "--device", "cuda", "--workers", self.workers, *FP16, "--out",
              str(self.out / "plant_grads.pt")]),
            ("copy", SMALL_CUT_MIN, False, self.out / "walk_planted.jsonl",
             ["tools/pg_plant.py", "plant", "--walk", str(self.out / "walk.jsonl"), *self.games(), "--grads",
              str(self.out / "plant_grads.pt"), "--out", str(self.out / "walk_planted.jsonl")]),
            ("rows", SMALL_CUT_MIN, False, None,
             ["tools/selfplay_rows.py", "--no-walk", "--walk", str(self.out / "walk_planted.jsonl"), "--out",
              str(self.out / "rows_planted.jsonl"), "--seed-base", str(SP_SEED_BASE), "--max-errors", "1000"]),
            ("labels", SMALL_CUT_MIN, False, None,
             ["tools/pg_labels.py", str(self.out / "rows_planted.jsonl"), "--heldout-from", str(HELDOUT_FROM),
              "--out", planted, "--fits", str(self.out / "fits_planted.json")]),
            ("pass", half, True, self.out / "pass_planted.pt",
             ["tools/pg_pass.py", "--checkpoint", self.ck, *self.games(), "--labels", planted, "--out",
              str(self.out / "pass_planted.pt"), "--device", "cuda", "--workers", self.workers, *PASS_FP16,
              "--floor-sides", str(FLOOR_SIDES), *self.corrections()]),
            ("readout", half, True, None,
             ["tools/pg_plant.py", "readout", "--checkpoint", self.ck, *self.games(), "--labels", planted,
              "--pass", str(self.out / "pass_planted.pt"), "--grads", str(self.out / "plant_grads.pt"),
              "--main-pass", str(self.out / "pass.pt"), "--device", "cuda", "--workers", self.workers, *FP16,
              "--out", str(self.out / "plant_readout.json")]),
        ]
        for name, cut, gpu, product, cmd in steps:
            if product is not None and product.exists():
                continue
            self.check_stop(f"plant {name}")
            rc = self.step(f"plant {name}", cmd, "plant.log", cut, gpu=gpu)
            if rc != 0 and not (name == "labels" and rc == 3):       # pg_labels: 3 when a game had no luck walk
                self.mark("plant", f"plant {name} ended rc={rc} (plant.log)")
                return False
        self.mark("plant")
        return True


def verdict(out: Path) -> str:
    """verdict.json: design §6's decision on what landed (the box script's
    rule); its line."""
    def read(name):
        p = out / name
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    c, plant, c0 = read("stage_c.json"), read("plant_readout.json"), read("c0_plan.json")
    gate = (c or {}).get("gate", {}).get("verdict")
    if gate in (None, "unread"):
        nxt = "stage C did not read its gate (stage_c.json's errors, or no driver readout): fix it and run it again"
    elif gate == "neither":
        nxt = "no rounds: the user chooses the $1,000 case, sized from the curve, or a stop"
    elif plant is None:
        nxt = "the planted readout did not land: run it before P and rounds"
    elif plant.get("verdict") != "pass":
        nxt = "the planted readout missed: debug before P and rounds"
    else:
        nxt = {"resolved": "P (the planted step played), then rounds",
               "rounds_at_size": "rounds at the size the gain curve names"}.get(gate, "no verdict")
    v = {"gate": (c or {}).get("gate"), "optimizer": (c or {}).get("optimizer"),
         "stage_c_errors": (c or {}).get("errors"),
         "plant": None if plant is None else {"verdict": plant["verdict"], "seats": plant["seats"],
                                              "delta": plant["delta"]},
         "c0": c0, "next": nxt}
    (out / "verdict.json").write_text(json.dumps(v, indent=1) + "\n", encoding="utf-8")
    return f"gate={gate} plant={None if plant is None else plant['verdict']} next: {nxt}"


def run(r: Run) -> int:
    """The parts in the box script's order; 0 when all four are done."""
    out = r.out
    why = r.restore()
    if why:
        log(out, f"STAGE1_UNUSABLE: {why}")
        return 1
    try:
        r.check_stop("C0")
        if r.c0():
            if not r.plan():
                log(out, f"C0_UNPRICED {' '.join(r.state)} verdict: {verdict(out)}")
                return 1
            r.check_stop("the pass")
            if r.gradient_pass():
                r.check_stop("the stage C driver")
                r.stage_c()
                r.check_stop("the plant")
                r.plant()
    except Stopped as e:
        log(out, f"STOPPED: {e}; a new start resumes")
        return 2
    line = verdict(out)
    reason = f"{' '.join(r.state)} verdict: {line}"
    if all(r.done(p) for p in PARTS):
        (out / "ALL_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
        log(out, f"STAGE_C_DONE {reason}")
        return 0
    log(out, f"STAGE_C_INCOMPLETE {reason}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1", type=Path, required=True, help="the local stage 1's run directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=REPO / "training" / "checkpoints" / "own2.pt",
                    help="pi_0, the network that played stage 1 (own2)")
    ap.add_argument("--workers", type=int, default=6, help="the tools' replay processes")
    ap.add_argument("--python", default=sys.executable, help="the interpreter of the GPU tools (the ROCm venv)")
    ap.add_argument("--tools-python", default=None, help="the CPU tools' (default: the repository's .venv)")
    ap.add_argument("--normal-priority", action="store_true")
    ap.add_argument("--keep-awake", action="store_true", help="ask Windows not to sleep while the runner lives")
    args = ap.parse_args()
    if args.tools_python is None:
        args.tools_python = default_tools_python(REPO)
    require_visible_gpu()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.normal_priority:
        below_normal_priority()
    if args.keep_awake:
        keep_awake()
    log(args.out, f"stage C local: stage 1 {args.stage1}, out {args.out}, checkpoint {args.checkpoint}, "
                  f"python {args.python}, tools {args.tools_python}")
    return run(Run(args))


if __name__ == "__main__":
    sys.exit(main())
