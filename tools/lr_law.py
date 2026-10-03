#!/usr/bin/env python3
"""The holdout loss of a training run as a function of its learning-rate
history, and the rule that decides from it when a run stops training at its
peak rate (docs/imitation_anneal_prereg_20261003.md).

The law (Tissue et al., "Scaling Law with Learning Rate Annealing", 2024):

    L = L0 + A * S1^-alpha - C * S2

S1, the sum of the learning rate over the optimizer steps so far, measures
the progress made. S2 measures how much of the noise that a high rate keeps
in the weights has faded since the rate came down: each step's decrease of
the rate counts again at every later step, fading by LAW_LAMBDA a step,
m_i = LAW_LAMBDA * m_(i-1) + (eta_(i-1) - eta_i), S2 = the sum of m. The
warm-up adds to S1 only. Only a lowering of the rate determines C: fitted on
probes taken before any, the law leaves C at 0 and says so
(`LawFit.lowering_seen`), and it predicts no lowering. Fitted on the parity
passes with only the first two probes of pass 2's lowering, it put that
lowering's end 0.03 below (64 slots) to 0.09 above (0 slots) what was
measured; with the whole lowering, every probe within its noise (`backtest`,
docs/imitation_anneal_prereg_20261003.md). The rule's decision reads S1's
term only.

The rule: after every holdout probe, fit the law to every probe of the
run's lineage. While one more epoch at the peak rate would lower the fitted
loss by more than a threshold, hold the rate; otherwise lower it. When the
latest probes of the pass all sit above what a fit without them predicts by
more than GUARD_SIGMAS times that fit's error, the law no longer describes
the run: stop and look.

    python tools/lr_law.py replay LINEAGE.json --out POINTS.jsonl [--areas-at PASS:STEPS]
    python tools/lr_law.py decide --points POINTS.jsonl --s1 S1 --peak 2.8e-4 \\
        --epoch-steps 7882 --threshold 0.03
    python tools/lr_law.py predict --points POINTS.jsonl --areas AREAS.json --peak 2.8e-4 \\
        --epoch-steps 7882 --lower-steps 3941 --epochs 0 1 2
    python tools/lr_law.py backtest --points POINTS.jsonl --first 15
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

# The fading of a lowering's effect per optimizer step: Tissue et al.'s
# value. Across 0.998 to 0.9995 the fit to the parity passes predicts the
# same next epochs to within 0.02 nat (docs/imitation_anneal_prereg_20261003.md).
LAW_LAMBDA = 0.999
# Fewest probes the law is fitted on.
MIN_POINTS = 6
# The guard: the latest probes of a pass, and how far above a fit without
# them, in that fit's root-mean-square errors, they must all sit.
GUARD_POINTS = 2
GUARD_SIGMAS = 2.0


@dataclass
class Areas:
    """S1 and S2 of a learning-rate history, step by step: `prev` is the
    last rate after the warm-up, `m` the fading sum of its decreases."""
    s1: float = 0.0
    s2: float = 0.0
    m: float = 0.0
    prev: Optional[float] = None

    def step(self, lr: float, warming: bool) -> None:
        """One optimizer step at rate `lr`; a warm-up step adds to S1 only."""
        self.s1 += lr
        if warming:
            return
        if self.prev is not None:
            self.m = LAW_LAMBDA * self.m + (self.prev - lr)
        self.prev = lr
        self.s2 += self.m

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "Areas":
        return cls(s1=float(d["s1"]), s2=float(d["s2"]), m=float(d["m"]),
                   prev=None if d.get("prev") is None else float(d["prev"]))


@dataclass(frozen=True)
class Point:
    """One holdout probe: where the run stood and the loss it read."""
    s1: float
    s2: float
    loss: float
    source: str = ""


@dataclass(frozen=True)
class LawFit:
    L0: float
    A: float
    alpha: float
    C: float
    rms: float
    n: int
    lowering_seen: bool = True        # some probe followed a lowering of the rate: C is fitted

    def predict(self, s1, s2):
        return self.L0 + self.A * np.power(s1, -self.alpha) - self.C * s2

    def epoch_gain(self, s1: float, peak: float, epoch_steps: float) -> float:
        """How much one more epoch at the peak rate lowers the fitted loss
        (its S1 term; S2 does not move while the rate holds)."""
        return float(self.A * (s1 ** -self.alpha - (s1 + peak * epoch_steps) ** -self.alpha))


def fit(points: Sequence[Point]) -> LawFit:
    """The least-squares fit of the law, the best of a few starts."""
    from scipy.optimize import least_squares
    s1 = np.array([p.s1 for p in points], dtype=np.float64)
    s2 = np.array([p.s2 for p in points], dtype=np.float64)
    y = np.array([p.loss for p in points], dtype=np.float64)
    if len(y) < 4 or (s1 <= 0).any():
        raise ValueError(f"the law needs at least 4 probes past the first step, got {len(y)}")

    seen = bool((s2 > 0).any())

    def residuals(th):
        L0, A, alpha = th[:3]
        C = th[3] if seen else 0.0
        return L0 + A * s1 ** -alpha - C * s2 - y

    lo, hi = [0.0, 0.0, 0.01, 0.0], [20.0, 100.0, 5.0, 1e3]
    best = None
    for alpha0 in (0.1, 0.3, 0.6):
        x0 = [max(0.0, y.min() - 0.5), 1.0, alpha0, 1.0]
        n = 4 if seen else 3
        res = least_squares(residuals, x0=x0[:n], bounds=(lo[:n], hi[:n]))
        if best is None or res.cost < best.cost:
            best = res
    L0, A, alpha = (float(v) for v in best.x[:3])
    C = float(best.x[3]) if seen else 0.0
    return LawFit(L0, A, alpha, C, rms=float(math.sqrt(np.mean(best.fun ** 2))), n=len(y), lowering_seen=seen)


@dataclass(frozen=True)
class Decision:
    action: str                      # "hold", "lower" or "review"
    reason: str
    gain: Optional[float] = None     # the fitted loss decrease of one more epoch at the peak
    law: Optional[LawFit] = None

    def to_dict(self) -> Dict:
        return {"action": self.action, "reason": self.reason, "gain": self.gain,
                "law": None if self.law is None else asdict(self.law)}


def decide(earlier: Sequence[Point], this_pass: Sequence[Point], s1_now: float, peak: float,
           epoch_steps: float, threshold: float) -> Decision:
    """The rule after a probe of this pass (see the module docstring).
    `earlier`: the probes of the passes before this one; `this_pass`: this
    pass's probes so far, in order."""
    points = list(earlier) + list(this_pass)
    if len(points) < MIN_POINTS:
        return Decision("hold", f"{len(points)} probes, the law is fitted on {MIN_POINTS} or more")
    law = fit(points)
    if len(this_pass) >= GUARD_POINTS and len(points) - GUARD_POINTS >= MIN_POINTS:
        without = fit(points[:-GUARD_POINTS])
        latest = this_pass[-GUARD_POINTS:]
        excess = [p.loss - float(without.predict(p.s1, p.s2)) for p in latest]
        if all(e > GUARD_SIGMAS * without.rms for e in excess):
            return Decision("review", f"the latest {GUARD_POINTS} probes sit "
                            f"{', '.join(f'{e:+.3f}' for e in excess)} above the law fitted without them "
                            f"(its error {without.rms:.3f})", law=law)
    gain = law.epoch_gain(s1_now, peak, epoch_steps)
    if gain > threshold:
        return Decision("hold", f"one more epoch at the peak lowers the fitted loss by {gain:.3f}, "
                        f"more than {threshold}", gain=gain, law=law)
    return Decision("lower", f"one more epoch at the peak lowers the fitted loss by {gain:.3f}, "
                    f"not more than {threshold}", gain=gain, law=law)


def predict_lowering(law: LawFit, areas: Areas, peak: float, steps: int) -> float:
    """The fitted loss after lowering the rate from `peak` in a straight
    line to 0 over `steps` steps, from `areas`. A law that has seen no
    lowering cannot tell."""
    if not law.lowering_seen:
        raise ValueError("the law was fitted on no probe after a lowering: it cannot predict one")
    a = Areas(**areas.to_dict())
    for k in range(steps):
        a.step(peak * (1.0 - (k + 1) / steps), warming=False)
    return float(law.predict(a.s1, a.s2))


# ---------------------------------------------------------------------------
# Passes trained before the trainer kept its areas, replayed from their settings
# ---------------------------------------------------------------------------

def pass_rates(steps: int, total_positions: int, lr: float, warmup_steps: int,
               decay_from: Optional[float]) -> List[float]:
    """Each step's rate as tools/sequence_train.py sets it: the warm-up on
    the step count, the lowering on the positions trained before the step
    (taken as evenly spread over the pass's steps)."""
    from tools.sequence_train import lr_factor
    per_step = total_positions / steps
    return [lr * min(1.0, (s + 1) / max(1, warmup_steps))
            * lr_factor(int(s * per_step), total_positions, decay_from) for s in range(steps)]


def replay(lineage: Sequence[Dict], key: str) -> Dict[str, object]:
    """The probe points of a lineage of passes, each pass starting where
    its parent stood after `parent_steps` steps. Returns {"points": [Point],
    "areas": {pass: [Areas after each step]}}."""
    areas_by_pass: Dict[str, List[Areas]] = {}
    points: List[Point] = []
    for p in lineage:
        start = Areas()
        if p.get("parent"):
            start = Areas(**areas_by_pass[p["parent"]][int(p["parent_steps"]) - 1].to_dict())
        rates = pass_rates(int(p["steps"]), int(p["total_positions"]), float(p["lr"]),
                           int(p["warmup_steps"]), p.get("decay_from"))
        a, trail = start, []
        for s, rate in enumerate(rates):
            a.step(rate, warming=(s + 1) < int(p["warmup_steps"]))
            trail.append(Areas(**a.to_dict()))
        areas_by_pass[p["name"]] = trail
        for row in read_probes(Path(p["probes"])):
            at = trail[int(row["steps"]) - 1]
            points.append(Point(at.s1, at.s2, float(row[key]["ce_all"]), source=p["name"]))
    return {"points": points, "areas": areas_by_pass}


def read_probes(path: Path) -> List[Dict]:
    """A .probe.jsonl's rows, one per position (a probe repeated by a
    resume keeps its last row)."""
    rows: Dict[int, Dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["positions"])] = row
    return [rows[k] for k in sorted(rows)]


def probe_points(path: Path, key: str, source: str = "") -> List[Point]:
    """The points of a probe file whose rows carry the trainer's areas."""
    out = []
    for row in read_probes(path):
        if "areas" not in row:
            raise ValueError(f"{path}: a probe at {row['positions']} positions carries no areas")
        out.append(Point(float(row["areas"]["s1"]), float(row["areas"]["s2"]), float(row[key]["ce_all"]),
                         source=source))
    return out


def read_points(path: Path) -> List[Point]:
    return [Point(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_points(path: Path, points: Iterable[Point], append: bool = False) -> None:
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for p in points:
            f.write(json.dumps(asdict(p)) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay", help="the probe points of passes trained before the trainer kept areas")
    r.add_argument("lineage", type=Path, help="a JSON list of passes (name, probes, steps, total_positions, "
                   "lr, warmup_steps, decay_from, parent, parent_steps)")
    r.add_argument("--key", default="k64", help="the probe's slot reading the law is fitted on")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--areas-at", default=None, help="PASS:STEPS: print the areas after that many steps")
    a = sub.add_parser("append", help="add a finished pass's probe points to a points file")
    a.add_argument("probes", type=Path)
    a.add_argument("--key", default="k64")
    a.add_argument("--source", default="")
    a.add_argument("--out", type=Path, required=True)
    d = sub.add_parser("decide", help="the rule on a points file, for a reader")
    d.add_argument("--points", type=Path, required=True)
    d.add_argument("--s1", type=float, required=True)
    d.add_argument("--peak", type=float, required=True)
    d.add_argument("--epoch-steps", type=float, required=True)
    d.add_argument("--threshold", type=float, required=True)
    q = sub.add_parser("predict", help="the fitted loss after holding the peak for some epochs, then lowering")
    q.add_argument("--points", type=Path, required=True)
    q.add_argument("--areas", type=Path, required=True, help="the areas where the holding starts (JSON)")
    q.add_argument("--peak", type=float, required=True)
    q.add_argument("--epoch-steps", type=int, required=True)
    q.add_argument("--lower-steps", type=int, required=True)
    q.add_argument("--epochs", type=float, nargs="+", required=True)
    b = sub.add_parser("backtest", help="fit on the first N points, predict the others")
    b.add_argument("--points", type=Path, required=True)
    b.add_argument("--first", type=int, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "backtest":
        points = read_points(args.points)
        law = fit(points[:args.first])
        print(json.dumps({"law": asdict(law), "rest": [
            {"source": p.source, "measured": p.loss,
             "predicted": float(law.predict(p.s1, p.s2)) if law.lowering_seen or p.s2 == 0 else None}
            for p in points[args.first:]]}, indent=1))
        return 0
    if args.cmd == "predict":
        law = fit(read_points(args.points))
        start = Areas.from_dict(json.loads(args.areas.read_text(encoding="utf-8")))
        out = {"law": asdict(law), "predictions": {}}
        for epochs in args.epochs:
            a = Areas(**start.to_dict())
            for _ in range(int(epochs * args.epoch_steps)):
                a.step(args.peak, warming=False)
            out["predictions"][str(epochs)] = {"epoch_gain_after": law.epoch_gain(a.s1, args.peak, args.epoch_steps),
                                               "loss_after_lowering": predict_lowering(law, a, args.peak,
                                                                                       args.lower_steps)}
        print(json.dumps(out, indent=1))
        return 0
    if args.cmd == "replay":
        lineage = json.loads(args.lineage.read_text(encoding="utf-8"))
        result = replay(lineage, args.key)
        write_points(args.out, result["points"])
        if args.areas_at:
            name, steps = args.areas_at.split(":")
            print(json.dumps(result["areas"][name][int(steps) - 1].to_dict()))
        return 0
    if args.cmd == "append":
        write_points(args.out, probe_points(args.probes, args.key, args.source), append=True)
        return 0
    points = read_points(args.points)
    print(json.dumps(decide(points, [], args.s1, args.peak, args.epoch_steps, args.threshold).to_dict()))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main())
