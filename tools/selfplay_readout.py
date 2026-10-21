"""Stage 1's offline readouts of a self-play run (docs/training_design_
20261019/current_design.md §4-§6): the run's game records with their
decision logs (tools/decision_log.py), its labels (tools/pg_labels.py) and
its luck walk (tools/selfplay_rows.py). Each readout returns its numbers
with standard errors, and a pass/fail where the design states one.

    census         decisions a game and a side-game, side-turns, action
                   shares, turns, refusals; the cap rate against 0.009
                   (above it, the user rules on cap scoring)
    i_sample       the actions follow the logged law (gate): for end_turn,
                   attack and recruit, a binned chi-square of 1[a in c]
                   against the logged p(c) (bins by p); the sampled
                   action's randomized cdf position uniform (20 bins); and
                   the inverse-temperature score sum_t (log p_t + H_t),
                   mean zero under the logged law, positive when the
                   player samples colder.
    legal_mass     every decision's prior mass is 1 within 1e-3 (gate)
    i_count        every command of a record belongs to a logged decision,
                   each refused decision is logged and decided again, and
                   the record's recruit rejections at each command are its
                   logged refused recruits re-decided there plus at most
                   one placed recruit (gate)
    label_noise    E[z^2], E[(z - b)^2] and E[R^2] per side-game; then,
                   per decision, E[R^2] under luck alone and with the
                   corrections -alpha.v (the value head) and -kappa.U (the
                   opponent's sum of log p + H), alpha and kappa fitted
                   cross-fitted (two folds by game). A correction enters
                   iff it cuts E[R^2] by 2 SE and by at least 5 %.
    sharpening    the mean over side-games of R . sum_t (log p_t + H_t),
                   the first-order value of sampling colder; resolved at
                   3 SE
    dice_control   separate code (the walk's rows, side A's frame) refits
                   the luck slope on the held-out games and must equal the
                   training labels' beta; the training game-sides' z
                   regressed on their beta_S must read slope 1 within the
                   gates' z
    i_v            each correction's mean against 1[a in X] - p(X), for X
                   end_turn, attack, recruit: 0 within the gates' z
    p1_targets     sum over decisions of R (1[a in X] - p(X)) / (2 games),
                   what the gradient pass's probes must equal (design §3.4)

The gates' 18 statistical tests (I-sample's five, dice control's slope,
I-V's twelve) share a family-wise false-alarm rate of 1 %: each passes at
p >= 0.01 / 18, |z| <= 3.45 (Bonferroni), so a sound run fails a gate at
most once in a hundred. Standard errors treat a game as the unit (its two
sides share it).

    python tools/selfplay_readout.py --games DIR... --labels labels.jsonl --fits fits.json \\
        --walk WALK.jsonl... --out readout.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.match_luck import COLUMNS, result_z  # noqa: E402
from wesnoth_ai.server_priors import KIND_ATTACK, KIND_END_TURN, KIND_MOVE, KIND_RECRUIT  # noqa: E402

FAMILY_ALPHA = 0.01            # the gates' false-alarm rate on a sound run
GATE_TESTS = 5 + 1 + 12        # I-sample's five tests, dice control's slope, I-V's twelve
ALPHA = FAMILY_ALPHA / GATE_TESTS                    # each test's p-value floor (Bonferroni)
GATE_Z = NormalDist().inv_cdf(1.0 - ALPHA / 2.0)    # its two-sided z, 3.45
P_EDGES = np.array([0.0, 0.01, 0.03, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.97, 0.99, 1.0])
MIN_BIN_VARIANCE = 5.0         # a p-bin enters the chi-square with at least this variance
PIT_BINS = 20
MASS_TOLERANCE = 1e-3
CORRECTION_SE = 2.0            # a correction enters iff it cuts E[R^2] by this many SE ...
CORRECTION_SHARE = 0.05        # ... and by at least this share (design §4)
RESOLVED_SE = 3.0
CAP_RATE_RULE = 0.009          # one claim SE / 347.4 (design §4)
# (name, the entry's column, the action kind) of the three categories.
CATEGORIES = (("end_turn", "p_end", KIND_END_TURN), ("attack", "p_att", KIND_ATTACK),
              ("recruit", "p_rec", KIND_RECRUIT))
KIND_NAMES = {KIND_ATTACK: "attack", KIND_MOVE: "move", KIND_RECRUIT: "recruit", KIND_END_TURN: "end_turn"}


# ---------------------------------------------------------------------
# One game's decisions
# ---------------------------------------------------------------------
def decision_arrays(log: Sequence[Dict]) -> Dict[str, np.ndarray]:
    """A game's decision log as columns."""
    return {"side": np.array([e["side"] for e in log], dtype=int),
            "kind": np.array([e["action"][1] for e in log], dtype=int),
            "refused": np.array([e["refused"] for e in log], dtype=bool),
            "p": np.array([[e[c] for _, c, _ in CATEGORIES] for e in log], dtype=float).reshape(len(log), 3),
            "logp": np.array([e["logp"] for e in log], dtype=float),
            "H": np.array([e["H"] for e in log], dtype=float),
            "cdf": np.array([e["cdf"] for e in log], dtype=float).reshape(len(log), 2),
            "v": np.array([e["v"] for e in log], dtype=float),
            "mass": np.array([e["mass"] for e in log], dtype=float)}


def residuals(d: Dict[str, np.ndarray]) -> np.ndarray:
    """[n, 3]: 1[a in X] - p(X) for each decision and category."""
    hit = np.stack([d["kind"] == k for _, _, k in CATEGORIES], axis=1)
    return hit - d["p"]


@dataclass
class SideGame:
    """One side's sums over its decisions in one game."""
    n: int = 0
    refused: int = 0
    side_turns: int = 0
    sum_v: float = 0.0
    sum_v2: float = 0.0
    score: float = 0.0
    D: np.ndarray = field(default_factory=lambda: np.zeros(3))     # sum 1[a in X] - p(X)
    DV: np.ndarray = field(default_factory=lambda: np.zeros(3))    # sum v (1[a in X] - p(X))
    kinds: Counter = field(default_factory=Counter)


def side_games(d: Dict[str, np.ndarray]) -> Dict[int, SideGame]:
    res = residuals(d)
    out = {}
    for side in (1, 2):
        m = d["side"] == side
        v = np.nan_to_num(d["v"][m])
        live = m & ~d["refused"]
        out[side] = SideGame(n=int(m.sum()), refused=int((m & d["refused"]).sum()),
                             side_turns=int((live & (d["kind"] == KIND_END_TURN)).sum()),
                             sum_v=float(v.sum()), sum_v2=float((v * v).sum()),
                             score=float((d["logp"][m] + d["H"][m]).sum()), D=res[m].sum(axis=0),
                             DV=(v[:, None] * res[m]).sum(axis=0),
                             kinds=Counter(KIND_NAMES[k] for k in d["kind"][live]))
    return out


# ---------------------------------------------------------------------
# I-sample and legal mass
# ---------------------------------------------------------------------
class SampleCheck:
    """The accumulators of I-sample and legal mass, fed a game at a time."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        nb = len(P_EDGES) - 1
        self.obs, self.exp, self.var = np.zeros((3, nb)), np.zeros((3, nb)), np.zeros((3, nb))
        self.pit = np.zeros(PIT_BINS, dtype=np.int64)
        self.score = self.score2 = 0.0
        self.n = 0
        self.mass_dev = 0.0
        self.mass_off = 0

    def add(self, d: Dict[str, np.ndarray]) -> None:
        n = len(d["kind"])
        if not n:
            return
        bins = np.clip(np.searchsorted(P_EDGES, d["p"], side="right") - 1, 0, len(P_EDGES) - 2)
        for i, (_, _, k) in enumerate(CATEGORIES):
            np.add.at(self.obs[i], bins[:, i], (d["kind"] == k).astype(float))
            np.add.at(self.exp[i], bins[:, i], d["p"][:, i])
            np.add.at(self.var[i], bins[:, i], d["p"][:, i] * (1.0 - d["p"][:, i]))
        u = d["cdf"][:, 0] + self.rng.random(n) * (d["cdf"][:, 1] - d["cdf"][:, 0])
        np.add.at(self.pit, np.clip((u * PIT_BINS).astype(int), 0, PIT_BINS - 1), 1)
        x = d["logp"] + d["H"]
        self.score += float(x.sum())
        self.score2 += float((x * x).sum())
        self.n += n
        dev = np.abs(d["mass"] - 1.0)
        self.mass_dev = max(self.mass_dev, float(dev.max()))
        self.mass_off += int((dev > MASS_TOLERANCE).sum())

    def i_sample(self) -> Dict:
        from scipy.stats import chi2, norm
        out: Dict = {"decisions": self.n}
        passed = True
        for i, (name, _, _) in enumerate(CATEGORIES):
            use = self.var[i] >= MIN_BIN_VARIANCE
            stat = float(((self.obs[i] - self.exp[i]) ** 2 / np.where(use, self.var[i], 1.0))[use].sum())
            dof = int(use.sum())
            p = float(chi2.sf(stat, dof)) if dof else 1.0
            passed &= p >= ALPHA
            out[name] = {"chi2": stat, "dof": dof, "p": p, "observed": float(self.obs[i].sum()),
                         "expected": float(self.exp[i].sum()), "bins": [
                             {"edges": [float(P_EDGES[b]), float(P_EDGES[b + 1])], "observed": float(self.obs[i][b]),
                              "expected": float(self.exp[i][b])} for b in range(len(P_EDGES) - 1) if use[b]]}
        want = self.n / PIT_BINS
        stat = float(((self.pit - want) ** 2 / max(want, 1e-12)).sum())
        p_pit = float(chi2.sf(stat, PIT_BINS - 1)) if self.n else 1.0
        z = self.score / math.sqrt(self.score2) if self.score2 > 0 else 0.0
        p_score = float(2 * norm.sf(abs(z)))
        out["cdf_position"] = {"chi2": stat, "dof": PIT_BINS - 1, "p": p_pit, "counts": self.pit.tolist()}
        out["temperature_score"] = {"mean": self.score / max(self.n, 1), "z": z, "p": p_score}
        out["pass"] = bool(passed and p_pit >= ALPHA and p_score >= ALPHA)
        return out

    def legal_mass(self) -> Dict:
        return {"max_deviation": self.mass_dev, "tolerance": MASS_TOLERANCE, "off": self.mass_off,
                "pass": self.mass_off == 0}


# ---------------------------------------------------------------------
# I-count: the log against the record's commands
# ---------------------------------------------------------------------
FIRST_COMMAND = {KIND_ATTACK: ("move", "attack"), KIND_MOVE: ("move",), KIND_RECRUIT: ("recruit",),
                 KIND_END_TURN: ("end_turn",)}
PLAYER_SIDES = (1, 2)


def span_ok(kind: int, span: Sequence[list]) -> bool:
    """Whether the commands a decision of `kind` started are what the
    simulator emits for it: an attack, after its approach move when it has
    one (the approach alone when it fell short); a move; a recruit; an
    end_turn, then the neutral sides' turns and the next side's start."""
    names = [c[0] for c in span]
    if not names or names[0] not in FIRST_COMMAND[kind]:
        return False
    if kind == KIND_ATTACK:
        return names in (["attack"], ["move", "attack"], ["move"])
    if kind != KIND_END_TURN:
        return len(names) == 1
    neutral = False
    for c in span[1:]:
        if c[0] == "init_side":
            neutral = int(c[1]) not in PLAYER_SIDES
        elif not neutral and c[0] != "auto_shroud":
            return False
    return True


def icount_problems(rec: Dict) -> List[str]:
    """What in the record's decision log does not account for its
    commands; empty when every decision is logged."""
    log = rec.get("decisions")
    if log is None:
        return ["no decision log"]
    cmds = rec["commands"]
    live = [e for e in log if not e["refused"]]
    starts = [e["cmd"] for e in live]
    problems = []
    if starts != sorted(set(starts)) or (starts and (starts[0] < 0 or starts[-1] >= len(cmds))):
        return ["the logged decisions' commands are not increasing within the record"]
    bounds = starts + [len(cmds)]
    if any(c[0] not in ("init_side", "auto_shroud") for c in cmds[:bounds[0]]):
        problems.append("commands before the first logged decision")
    for e, lo, hi in zip(live, bounds, bounds[1:]):
        if not span_ok(e["action"][1], cmds[lo:hi]):
            problems.append(f"decision at command {lo} ({KIND_NAMES[e['action'][1]]}) started "
                            f"{[c[0] for c in cmds[lo:hi]]}")
    for i, e in enumerate(log):
        if e["refused"] and (e["cmd"] != -1 or i + 1 >= len(log) or log[i + 1]["side"] != e["side"]):
            problems.append(f"refused decision {i} is not decided again by its side")
    return problems + rejection_problems(log, cmds, rec.get("rejections", ()))


def rejection_problems(log: Sequence[Dict], cmds: Sequence[list], rejections) -> List[str]:
    """The record's recruit rejections against the log's refused recruits.
    A rejection at command k is a bounce (the hex held a hidden unit and no
    castle hex was vacant: the harness refused the recruit, and the side
    decides again at k, where no recruit can land) or a placed recruit (the
    recruit command at k, set on the nearest vacant castle hex). So at each
    k the rejections are the refused recruits re-decided at k, plus at most
    one when a recruit decision starts at k."""
    refused = Counter()
    for i, e in enumerate(log):
        if e["refused"] and e["action"][1] == KIND_RECRUIT:
            nxt = next((x for x in log[i + 1:] if not x["refused"]), None)
            refused[len(cmds) if nxt is None else nxt["cmd"]] += 1
    rejected = Counter(int(k) for k, _x, _y in rejections)
    placed = Counter(e["cmd"] for e in log if not e["refused"] and e["action"][1] == KIND_RECRUIT
                     and 0 <= e["cmd"] < len(cmds) and cmds[e["cmd"]][0] == "recruit")
    problems = []
    for k in sorted(set(refused) | set(rejected)):
        if refused[k] > rejected[k]:
            problems.append(f"{refused[k]} refused recruits re-decided at command {k}, "
                            f"{rejected[k]} rejections there")
        elif rejected[k] - refused[k] > placed[k]:
            problems.append(f"{rejected[k]} recruit rejections at command {k}: {refused[k]} logged "
                            f"refusals and {placed[k]} placed recruits")
    return problems


# ---------------------------------------------------------------------
# Statistics over side-games
# ---------------------------------------------------------------------
def ratio_se(num: np.ndarray, den: np.ndarray) -> Tuple[float, float]:
    """sum(num) / sum(den) over games and its standard error (the
    games as independent units)."""
    G = len(num)
    r = float(num.sum() / den.sum())
    if G < 2:
        return r, math.nan
    resid = num - r * den
    return r, float(np.sqrt(np.sum(resid ** 2) * G / (G - 1)) / den.sum())


def mean_se(x: np.ndarray) -> Tuple[float, float]:
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else math.nan


@dataclass
class Table:
    """Side-game columns, two rows per game (sides 1 and 2), rows of a
    game adjacent."""
    game: np.ndarray
    fold: np.ndarray
    R: np.ndarray
    z: np.ndarray
    b: np.ndarray
    beta_s: np.ndarray
    n: np.ndarray
    sum_v: np.ndarray
    sum_v2: np.ndarray
    score: np.ndarray
    opp_score: np.ndarray
    D: np.ndarray            # [rows, 3]
    DV: np.ndarray           # [rows, 3]

    def per_game(self, x: np.ndarray) -> np.ndarray:
        """Sum of a row column over each game's two rows."""
        return x.reshape(-1, 2, *x.shape[1:]).sum(axis=1)


def fit_corrections(t: Table, train: np.ndarray, use_v: bool, use_u: bool) -> np.ndarray:
    """[alpha, kappa] minimising the decision-weighted sum of (R - alpha v_t
    - kappa U)^2 over the rows `train` (the unused one 0)."""
    U = t.opp_score
    A = np.array([[t.sum_v2[train].sum(), (U * t.sum_v)[train].sum()],
                  [(U * t.sum_v)[train].sum(), (t.n * U * U)[train].sum()]])
    y = np.array([(t.R * t.sum_v)[train].sum(), (t.n * t.R * U)[train].sum()])
    keep = [i for i, on in enumerate((use_v, use_u)) if on]
    coef = np.zeros(2)
    if keep:
        coef[keep] = np.linalg.solve(A[np.ix_(keep, keep)], y[keep])
    return coef


def squared_label(t: Table, coef: np.ndarray) -> np.ndarray:
    """Per row: the sum over its decisions of (R - alpha v_t - kappa U)^2."""
    a, k = coef
    U = t.opp_score
    return (t.n * t.R ** 2 + a * a * t.sum_v2 + k * k * t.n * U * U - 2 * a * t.R * t.sum_v
            - 2 * k * t.n * t.R * U + 2 * a * k * U * t.sum_v)


def label_noise(t: Table) -> Dict:
    zg = t.z[0::2]
    out = {"per_side_game": {"E_z2": mean_se(zg ** 2), "E_z_minus_b2": mean_se((t.z - t.b)[0::2] ** 2),
                             "E_R2": mean_se(t.R[0::2] ** 2)}}
    n_g = t.per_game(t.n.astype(float))
    luck = t.per_game(squared_label(t, np.zeros(2)))
    out["luck"] = dict(zip(("E", "se"), ratio_se(luck, n_g)))
    for name, use_v, use_u in (("v", True, False), ("U", False, True), ("v_and_U", True, True)):
        q = np.zeros_like(t.R)
        coefs = []
        for f in (0, 1):
            coef = fit_corrections(t, t.fold != f, use_v, use_u)
            coefs.append(coef.tolist())
            rows = t.fold == f
            q[rows] = squared_label(t, coef)[rows]
        qg = t.per_game(q)
        E, se = ratio_se(qg, n_g)
        cut, cut_se = ratio_se(luck - qg, n_g)
        share = cut / out["luck"]["E"] if out["luck"]["E"] else 0.0
        out[name] = {"E": E, "se": se, "cut": cut, "cut_se": cut_se, "share": share,
                     "coefficients_by_fold": coefs,
                     "enters": bool(cut >= CORRECTION_SE * cut_se and share >= CORRECTION_SHARE)}
    return out


def sharpening_slope(R: np.ndarray, score: np.ndarray) -> Dict:
    """The mean over side-games of R . score (rows of a game adjacent), its
    SE over games, and whether it is resolved at 3 SE."""
    per_game = (R * score).reshape(-1, 2).mean(axis=1)
    slope, se = mean_se(per_game)
    return {"slope": slope, "se": se, "z": slope / se if se else 0.0,
            "resolved": bool(se and abs(slope) >= RESOLVED_SE * se)}


def i_v(t: Table) -> Dict:
    """Each correction's mean against 1[a in X] - p(X) over decisions."""
    n_g = t.per_game(t.n.astype(float))
    corr = {"beta_S": t.beta_s[:, None] * t.D, "b": t.b[:, None] * t.D, "U": t.opp_score[:, None] * t.D,
            "v": t.DV}
    out: Dict = {}
    passed = True
    for name, x in corr.items():
        out[name] = {}
        xg = t.per_game(x)
        for i, (cat, _, _) in enumerate(CATEGORIES):
            m, se = ratio_se(xg[:, i], n_g)
            ok = not (se > 0) or abs(m) <= GATE_Z * se
            passed &= ok
            out[name][cat] = {"mean": m, "se": se, "pass": bool(ok)}
    out["pass"] = bool(passed)
    return out


def dice_control(labels: Sequence[Dict], walk_rows: Sequence[Dict], train_beta: Sequence[float]) -> Dict:
    """The luck slope refitted by separate code, and the training labels'
    slope of z on beta_S."""
    heldout = {lab["game"] for lab in labels if lab["heldout"]}
    held = [r for r in walk_rows if r["game"] in heldout and "error" not in r]
    z = np.array([result_z(r.get("outcome_a")) for r in held], dtype=float)
    S = np.array([[r["S"][c] for c in COLUMNS] for r in held], dtype=float).reshape(len(held), len(COLUMNS))
    # Side rows are antisymmetric, so the labels' fit has no intercept: z
    # on S through the origin in side A's frame is the same fit.
    beta = np.linalg.lstsq(S, z, rcond=None)[0] if len(held) else np.zeros(len(COLUMNS))
    diff = float(np.max(np.abs(beta - np.asarray(train_beta)))) if len(held) else math.nan
    scale = max(1.0, float(np.max(np.abs(train_beta))))
    train = [lab for lab in labels if not lab["heldout"] and lab["side"] == 1]
    x = np.array([lab["beta_S"] for lab in train])
    y = np.array([lab["z"] for lab in train])
    if len(train) > 1 and float(x @ x) > 0:
        slope = float(x @ y / (x @ x))
        se = float(np.sqrt(np.sum(x ** 2 * (y - slope * x) ** 2)) / (x @ x))
    else:
        slope, se = math.nan, math.nan
    slope_ok = bool(se > 0 and abs(slope - 1.0) <= GATE_Z * se)
    return {"beta_labels": [float(b) for b in train_beta], "beta_separate": beta.tolist(),
            "max_difference": diff, "beta_pass": bool(diff <= 1e-6 * scale),
            "slope_z_on_beta_S": slope, "slope_se": se, "slope_pass": slope_ok,
            "pass": bool(diff <= 1e-6 * scale and slope_ok)}


# ---------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------
def run_records(games: Path) -> Iterator[Dict]:
    """Every game record of a run's directory or tarball, by file name."""
    from tools.match_luck import RECORD_SUFFIX, _match_files, parse_record
    for name, data in _match_files(games):
        if name.startswith("game_") and name.endswith(RECORD_SUFFIX):
            yield parse_record(data, name[:-len(RECORD_SUFFIX)])


def read_jsonl(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def readout(records: Iterator[Dict], labels: Sequence[Dict], fits: Dict, walk_rows: Sequence[Dict],
            seed: int = 0) -> Dict:
    """Every readout of the module docstring."""
    by_side = {(lab["game"], lab["side"]): lab for lab in labels}
    check = SampleCheck(seed)
    sides: Dict[Tuple[str, int], SideGame] = {}
    problems: Dict[str, List[str]] = {}
    census = defaultdict(float)
    kinds: Counter = Counter()
    for rec in records:
        game = rec["game_label"]
        census["games"] += 1
        census["turns"] += rec["turns"]
        census["capped"] += int(rec["winner"]) == 0
        p = icount_problems(rec)
        if p:
            problems[game] = p
        if rec.get("decisions") is None:
            continue
        d = decision_arrays(rec["decisions"])
        check.add(d)
        for side, sg in side_games(d).items():
            sides[(game, side)] = sg
            kinds.update(sg.kinds)
            census["decisions"] += sg.n
            census["refused"] += sg.refused
            census["side_turns"] += sg.side_turns
    G = max(census["games"], 1)
    cap = census["capped"] / G
    out = {"census": {"games": int(census["games"]), "turns": census["turns"] / G,
                      "decisions_per_game": census["decisions"] / G,
                      "decisions_per_side_game": census["decisions"] / (2 * G),
                      "side_turns_per_side_game": census["side_turns"] / (2 * G),
                      "refused_per_game": census["refused"] / G,
                      "shares": {k: v / max(sum(kinds.values()), 1) for k, v in sorted(kinds.items())},
                      "cap_rate": cap, "cap_rate_se": math.sqrt(cap * (1 - cap) / G),
                      "cap_rate_above_rule": bool(cap > CAP_RATE_RULE)},
           "i_sample": check.i_sample(), "legal_mass": check.legal_mass(),
           "i_count": {"games_with_problems": len(problems), "pass": not problems,
                       "examples": dict(list(problems.items())[:20])},
           "gates": {"family_alpha": FAMILY_ALPHA, "tests": GATE_TESTS, "per_test_alpha": ALPHA,
                     "z": GATE_Z}}
    games = sorted({g for g, _ in sides if (g, 1) in by_side and (g, 2) in by_side and (g, 2) in sides})
    out["labelled_games"] = len(games)
    if len(games) < 4:
        return out
    t = table(games, sides, by_side)
    out["label_noise"] = label_noise(t)
    out["sharpening"] = sharpening_slope(t.R, t.score)
    out["i_v"] = i_v(t)
    out["p1_targets"] = {cat: float((t.R[:, None] * t.D).sum(axis=0)[i] / (2 * len(games)))
                         for i, (cat, _, _) in enumerate(CATEGORIES)}
    out["dice_control"] = dice_control(labels, walk_rows, fits["fits"]["train"]["beta"])
    out["gates_pass"] = bool(out["i_sample"]["pass"] and out["legal_mass"]["pass"] and out["i_count"]["pass"]
                             and out["dice_control"]["pass"] and out["i_v"]["pass"])
    return out


def table(games: Sequence[str], sides: Dict[Tuple[str, int], SideGame], by_side: Dict) -> Table:
    rows = [(g, s) for g in games for s in (1, 2)]
    sg = [sides[r] for r in rows]
    lab = [by_side[r] for r in rows]
    score = np.array([x.score for x in sg])
    return Table(game=np.array([g for g, _ in rows]),
                 fold=np.array([(int(x["split"]) // 2) % 2 for x in lab]),
                 R=np.array([x["R"] for x in lab]), z=np.array([x["z"] for x in lab]),
                 b=np.array([x["b"] for x in lab]), beta_s=np.array([x["beta_S"] for x in lab]),
                 n=np.array([x.n for x in sg]), sum_v=np.array([x.sum_v for x in sg]),
                 sum_v2=np.array([x.sum_v2 for x in sg]), score=score,
                 opp_score=score.reshape(-1, 2)[:, ::-1].reshape(-1),
                 D=np.array([x.D for x in sg]), DV=np.array([x.DV for x in sg]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=Path, nargs="+", required=True,
                    help="the run's games directories or tarballs")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--fits", type=Path, required=True, help="tools/pg_labels.py --fits")
    ap.add_argument("--walk", type=Path, nargs="+", required=True, help="tools/selfplay_rows.py's --walk files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0, help="the cdf positions' uniform draws")
    args = ap.parse_args(argv)
    records = (rec for games in args.games for rec in run_records(games))
    walk_rows = [row for path in args.walk for row in read_jsonl(path)]
    result = readout(records, read_jsonl(args.labels), json.loads(args.fits.read_text(encoding="utf-8")),
                     walk_rows, seed=args.seed)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: result[k].get("pass") for k in ("i_sample", "legal_mass", "i_count", "dice_control", "i_v")
                      if k in result}))
    print(f"the gates' {GATE_TESTS} statistical tests: family-wise false-alarm rate {FAMILY_ALPHA:g}, "
          f"each at p >= {ALPHA:.2g} (|z| <= {GATE_Z:.2f})")
    return 0 if result.get("gates_pass") else 1


if __name__ == "__main__":
    sys.exit(main())
