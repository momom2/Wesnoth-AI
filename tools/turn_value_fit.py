"""Labels, arms, the truncated-rollout grader and the verdict of the
turn-ranking value function (docs/turn_value_prereg_20260925.md); the
CLI is tools/turn_value.py.

Luck. Two fight lucks enter every playout's outcome z: the playout's own
(L, HP and kills) and the candidate turn's own (T), each the realized
minus the exactly expected change of the mover's margins, with a zero
mean given the position (tools/playout_reads.py). One least-squares fit
of z on (L, T) over the fit split's playouts gives beta. The adjusted
outcome z - beta . (L, T) keeps the expected value of the candidate
turn, taken over its own dice, and sheds noise: it is the truth every
grader is judged against, since a teacher replays a chosen turn with new
dice and cannot use the ones a grader saw.

Labels. A candidate's label is the mean over its playouts of

    lam * V_h + (1 - lam) * z'

where z' is z or the adjusted outcome, and V_h the reference head's read
at the mover's turn start HORIZON_TURNS turns after the boundary.

Arms, fitted on the fit split's candidates with the loss

    mean w (v - y)^2  +  rank_weight * mean w ((v - v_p) - (y - y_p))^2

(y the label, w the playout count, v_p and y_p the position means):
  linear   standardized token -> value in closed form, with a ridge;
  head     the reference value head's shape (d -> d -> atoms, expected
           value over the atoms), from its weights, AdamW, epochs.
Every configuration (label, rank weight, ridge or epoch) is scored on
the stop split by its within-position correlation with the adjusted
outcomes (scale-free, as the verdict is), and each arm keeps its best.
  rollout  no fitting: a candidate's grade is the mean of V_h over its
           first ROLLOUT_READS playouts, and its truth comes from the
           other playouts only.

The verdict's measure is the within-position correlation between a
grader and the candidates' mean adjusted outcomes, divided by the square
root of those means' within-position reliability (the share of their
within-position variance that is not playout noise), with a bootstrap
standard error over source games (half the 68% percentile interval).
"""
from __future__ import annotations

import copy
import json
import logging
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger("turn_value_fit")

LAMBDAS = (0.0, 0.25, 0.5, 0.75)
RANK_WEIGHTS = (0.0, 1.0, 10.0)
RIDGE_GRID = tuple(10.0 ** k for k in range(-6, 2))
# Horizon read j is taken at the j-th player turn start of a playout,
# read 0 on the position right after the candidate turn (the opponent's
# turn start): the mover's own turn starts are the odd reads, k turns
# after the boundary at read 2k - 1.
HORIZON_TURNS = 2
HORIZON_READ = 2 * HORIZON_TURNS - 1
ROLLOUT_READS = 8
ROLLOUT_SECONDARY = ((4, 3), (8, 1), (8, 7))      # (playouts read, horizon read)
PASS_AT, FAIL_AT = 0.7, 0.5
MIN_RELIABILITY = 0.2
MAX_SE = 0.12
GAP_THRESHOLD = 0.25
BOOTSTRAP_RESAMPLES = 1000
JUDGED = ("linear", "head", "rollout")


# ---------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------

def position_ids(index: torch.Tensor) -> torch.Tensor:
    """Position indices remapped to 0..G-1."""
    return torch.unique(index, return_inverse=True)[1]


def centered(x: torch.Tensor, w: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """x minus its weighted mean over its group."""
    g = int(groups.max()) + 1 if len(groups) else 0
    wsum = torch.zeros(g, dtype=x.dtype, device=x.device).index_add(0, groups, w)
    mean = torch.zeros(g, dtype=x.dtype, device=x.device).index_add(0, groups, w * x) / wsum
    return x - mean[groups]


def within_error(v, y, w, groups) -> torch.Tensor:
    """Weighted mean squared difference of within-position deviations."""
    return (w * (centered(v, w, groups) - centered(y, w, groups)) ** 2).sum() / w.sum()


def within_correlation(v, y, w, groups) -> float:
    """The weighted correlation of within-position deviations: how a
    grader ranks the candidates of a position, whatever its scale."""
    vc, yc = centered(v.double(), w, groups), centered(y.double(), w, groups)
    denominator = math.sqrt(float((w * vc * vc).sum()) * float((w * yc * yc).sum()))
    return float((w * vc * yc).sum()) / denominator if denominator > 0 else -math.inf


def turn_loss(v, y, w, groups, rank_weight: float) -> torch.Tensor:
    level = (w * (v - y) ** 2).sum() / w.sum()
    return level + rank_weight * within_error(v, y, w, groups)


# ---------------------------------------------------------------------
# Splits and labels
# ---------------------------------------------------------------------

def split_rows(cache: Dict, split: Optional[str] = None) -> Dict:
    """The cache's rows of one split (all rows when None)."""
    keep = torch.tensor([split is None or s == split for s in cache["split"]])
    idx = torch.nonzero(keep).flatten()
    part = {key: cache[key][idx] for key in ("feats", "value_reference", "index", "slot",
                                            "outcomes", "horizon_value", "luck", "turn_luck")}
    part["n"] = torch.isfinite(part["outcomes"]).sum(dim=1).double()
    part["raw"] = torch.nanmean(part["outcomes"].double(), dim=1)
    part["groups"] = position_ids(part["index"])
    part["group"] = [cache["group"][i] for i in idx.tolist()]
    part["baselines"] = {k: [v[i] for i in idx.tolist()]
                         for k, v in cache.get("baselines", {}).items()}
    return part


def _luck_columns(part: Dict) -> np.ndarray:
    """[N, P, 4]: the playout's luck (HP, kills), then the candidate
    turn's own (HP, kills), repeated over its playouts."""
    playout = part["luck"].double().numpy()
    turn = np.broadcast_to(part["turn_luck"].double().numpy()[:, None, :], playout.shape)
    return np.concatenate([playout, turn], axis=2)


def luck_coefficients(part: Dict) -> Tuple[np.ndarray, float, int]:
    """(beta over (L_hp, L_kills, T_hp, T_kills), share of the outcome
    variance the four explain, playouts used): least squares of the
    outcome on the lucks over the part's playouts."""
    z = part["outcomes"].double().numpy().ravel()
    luck = _luck_columns(part).reshape(-1, 4)
    ok = np.isfinite(z) & np.isfinite(luck).all(axis=1)
    if ok.sum() < 10:
        return np.zeros(4), 0.0, int(ok.sum())
    lc = luck[ok] - luck[ok].mean(axis=0)
    zc = z[ok] - z[ok].mean()
    beta = np.linalg.lstsq(lc, zc, rcond=None)[0]
    r2 = 1.0 - float(np.var(zc - lc @ beta) / np.var(zc))
    return beta, r2, int(ok.sum())


def adjusted_outcomes(part: Dict, beta: np.ndarray) -> torch.Tensor:
    """[N, P]: each playout's outcome minus beta . (its luck, its
    candidate turn's luck); a missing luck term is left out."""
    terms = torch.from_numpy(_luck_columns(part) * np.asarray(beta, dtype=float))
    adjust = torch.nan_to_num(terms, nan=0.0).sum(dim=2)
    return part["outcomes"].double() - adjust


@dataclass(frozen=True)
class LabelSpec:
    lam: float
    luck: bool


def label_specs(part: Dict) -> List[LabelSpec]:
    has_horizon = part["horizon_value"].shape[-1] > HORIZON_READ
    lams = LAMBDAS if has_horizon else (0.0,)
    return [LabelSpec(lam, luck) for luck in (False, True) for lam in lams]


def labels(part: Dict, spec: LabelSpec, beta: np.ndarray) -> torch.Tensor:
    """Each candidate's label under `spec` (see the module docstring)."""
    z = adjusted_outcomes(part, beta) if spec.luck else part["outcomes"].double()
    if spec.lam:
        vh = part["horizon_value"][..., HORIZON_READ].double()
        z = torch.where(torch.isfinite(vh), spec.lam * vh + (1 - spec.lam) * z, z)
    return torch.nanmean(z, dim=1)


# ---------------------------------------------------------------------
# The linear arm
# ---------------------------------------------------------------------

@dataclass
class LinearArm:
    mean: torch.Tensor
    std: torch.Tensor
    coef: torch.Tensor          # [d]
    intercept: float
    ridge: float

    def predict(self, feats: torch.Tensor) -> torch.Tensor:
        return ((feats.double() - self.mean) / self.std) @ self.coef + self.intercept

    def state(self) -> Dict:
        return {"mean": self.mean, "std": self.std, "coef": self.coef,
                "intercept": self.intercept, "ridge": self.ridge}


def centered_rows(z: torch.Tensor, w: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Each row of z minus its group's weighted mean row."""
    g = int(groups.max()) + 1
    wsum = torch.zeros(g, dtype=z.dtype).index_add(0, groups, w)
    mean = torch.zeros(g, z.shape[1], dtype=z.dtype).index_add(0, groups, w[:, None] * z)
    return z - (mean / wsum[:, None])[groups]


class LinearProblem:
    """The normal equations of turn_loss plus ridge * |coef|^2 (intercept
    unpenalized) on standardized features, built once: every label,
    rank weight and ridge is then one small solve. The equations are of
    the weighted sums, so the ridge term carries sum(w)."""

    def __init__(self, feats: torch.Tensor, w: torch.Tensor, groups: torch.Tensor):
        x = feats.double()
        self.mean, self.std = x.mean(dim=0), x.std(dim=0).clamp_min(1e-6)
        z = (x - self.mean) / self.std
        self.w, self.groups = w.double(), groups
        ones = torch.ones(len(z), 1, dtype=z.dtype)
        self.a = torch.cat([z, ones], dim=1)
        self.c = torch.cat([centered_rows(z, self.w, groups),
                            torch.zeros(len(z), 1, dtype=z.dtype)], dim=1)
        self.aa = (self.a * self.w[:, None]).T @ self.a
        self.cc = (self.c * self.w[:, None]).T @ self.c

    def solve(self, y: torch.Tensor, rank_weight: float, ridge: float) -> LinearArm:
        y = y.double()
        yc = centered(y, self.w, self.groups)
        d = self.a.shape[1] - 1
        lhs = self.aa + rank_weight * self.cc
        lhs[:d, :d] += ridge * float(self.w.sum()) * torch.eye(d, dtype=lhs.dtype)
        rhs = (self.a * self.w[:, None]).T @ y + rank_weight * (self.c * self.w[:, None]).T @ yc
        theta = torch.linalg.solve(lhs, rhs)
        return LinearArm(mean=self.mean, std=self.std, coef=theta[:-1],
                         intercept=float(theta[-1]), ridge=ridge)


def stop_score(grade: torch.Tensor, stop: Dict) -> float:
    """A configuration's score: its within-position correlation with the
    stop split's mean adjusted outcomes."""
    return within_correlation(grade, stop["truth"], stop["n"], stop["groups"])


def with_truth(stop: Dict, beta: np.ndarray) -> Dict:
    """The stop split with its candidates' mean adjusted outcomes."""
    return dict(stop, truth=torch.nanmean(adjusted_outcomes(stop, beta), dim=1))


def fit_linear_arm(train: Dict, stop: Dict, beta: np.ndarray) -> Tuple[LinearArm, Dict, List]:
    """The linear arm's best configuration by its stop score."""
    problem = LinearProblem(train["feats"], train["n"], train["groups"])
    stop = with_truth(stop, beta)
    trace, best = [], None
    for spec in label_specs(train):
        y = labels(train, spec, beta)
        for rank_weight in RANK_WEIGHTS:
            for ridge in RIDGE_GRID:
                arm = problem.solve(y, rank_weight, ridge)
                score = stop_score(arm.predict(stop["feats"]), stop)
                config = {**asdict(spec), "rank_weight": rank_weight, "ridge": ridge,
                          "stop_score": score}
                trace.append(config)
                if best is None or score > best[1]["stop_score"]:
                    best = (arm, config)
    log.info("linear arm: %s", best[1])
    return best[0], best[1], trace


# ---------------------------------------------------------------------
# The head arm
# ---------------------------------------------------------------------

class HeadArm(torch.nn.Module):
    """The reference value head's shape: expected value over its atoms."""

    def __init__(self, value_head: torch.nn.Module, atoms: torch.Tensor):
        super().__init__()
        self.net = copy.deepcopy(value_head).float()
        self.register_buffer("atoms", atoms.detach().float().clone())

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return (torch.softmax(self.net(feats), dim=-1) * self.atoms).sum(dim=-1)

    def predict(self, feats: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self(feats.float())


@dataclass
class HeadRecipe:
    lr: float = 3e-4
    weight_decay: float = 1e-2
    positions_per_batch: int = 128
    max_epochs: int = 40
    patience: int = 6
    seed: int = 0


def _batches(groups: torch.Tensor, per_batch: int, gen: torch.Generator):
    """Row indices of `per_batch` whole positions at a time, shuffled."""
    order = torch.argsort(groups)
    counts = torch.bincount(groups)
    starts = torch.cumsum(counts, 0) - counts
    perm = torch.randperm(len(counts), generator=gen)
    for lo in range(0, len(perm), per_batch):
        chosen = perm[lo:lo + per_batch]
        yield torch.cat([order[starts[p]:starts[p] + counts[p]] for p in chosen.tolist()])


def train_head(value_head, atoms, train: Dict, y: torch.Tensor, stop: Dict,
               rank_weight: float, recipe: HeadRecipe,
               device=torch.device("cpu")) -> Tuple[HeadArm, float, int]:
    """One head trained on labels `y`, kept at its epoch with the best
    stop score (epoch 0 is the reference head itself); `stop` carries
    its truth (`with_truth`). Returned on the CPU."""
    torch.manual_seed(recipe.seed)
    gen = torch.Generator().manual_seed(recipe.seed)
    arm = HeadArm(value_head, atoms).to(device)
    opt = torch.optim.AdamW(arm.parameters(), lr=recipe.lr, weight_decay=recipe.weight_decay)
    feats, y = train["feats"].float().to(device), y.float().to(device)
    w, groups = train["n"].float().to(device), train["groups"]
    stop_feats = stop["feats"].float().to(device)

    def score() -> float:
        return stop_score(arm.predict(stop_feats).cpu(), stop)

    best_score, best_state, best_epoch = score(), copy.deepcopy(arm.state_dict()), 0
    for epoch in range(1, recipe.max_epochs + 1):
        arm.train()
        for rows in _batches(groups, recipe.positions_per_batch, gen):
            loss = turn_loss(arm(feats[rows.to(device)]), y[rows.to(device)],
                             w[rows.to(device)], position_ids(groups[rows]).to(device),
                             rank_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        arm.eval()
        current = score()
        if current > best_score:
            best_score, best_state, best_epoch = current, copy.deepcopy(arm.state_dict()), epoch
        elif epoch - best_epoch >= recipe.patience:
            break
    arm.load_state_dict(best_state)
    return arm.cpu(), best_score, best_epoch


def fit_head_arm(value_head, atoms, train: Dict, stop: Dict, beta: np.ndarray,
                 recipe: HeadRecipe = HeadRecipe(),
                 device=torch.device("cpu")) -> Tuple[HeadArm, Dict, List]:
    stop = with_truth(stop, beta)
    trace, best = [], None
    for spec in label_specs(train):
        y = labels(train, spec, beta)
        for rank_weight in RANK_WEIGHTS:
            arm, score, epoch = train_head(value_head, atoms, train, y, stop, rank_weight,
                                           recipe, device)
            config = {**asdict(spec), "rank_weight": rank_weight, "epoch": epoch,
                      "stop_score": score}
            trace.append(config)
            log.info("head arm: %s", config)
            if best is None or score > best[1]["stop_score"]:
                best = (arm, config)
    log.info("head arm: selected %s", best[1])
    return best[0], best[1], trace


# ---------------------------------------------------------------------
# The measure: noise-corrected within-position correlation
# ---------------------------------------------------------------------

def _position_sums(grade: np.ndarray, truth: np.ndarray, positions: np.ndarray,
                   clusters: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per position with two or more usable candidates (a finite grade,
    two or more truth playouts): its cluster, and the sums over its
    candidates' deviations from the position's means of grade x truth,
    grade^2 and truth^2, with the truth's expected noise in the last."""
    n = np.isfinite(truth).sum(axis=1)
    usable = np.isfinite(grade) & (n >= 2)
    rows_by_position: Dict[int, List[int]] = {}
    for row in np.nonzero(usable)[0]:
        rows_by_position.setdefault(int(positions[row]), []).append(int(row))
    sums, owners = [], []
    for rows in rows_by_position.values():
        if len(rows) < 2:
            continue
        g = grade[rows]
        y = np.nanmean(truth[rows], axis=1)
        noise = np.nanvar(truth[rows], axis=1, ddof=1) / n[rows]
        gc, yc = g - g.mean(), y - y.mean()
        sums.append((gc @ yc, gc @ gc, yc @ yc, (1 - 1 / len(rows)) * noise.sum()))
        owners.append(clusters[rows[0]])
    return np.asarray(owners), np.asarray(sums, dtype=float).reshape(-1, 4)


def _ratios(s: np.ndarray) -> Tuple[float, float, float]:
    """(observed correlation, reliability, corrected correlation)."""
    sgy, sgg, syy, noise = s
    if sgg <= 0 or syy <= 0:
        return math.nan, math.nan, math.nan
    reliability = 1.0 - noise / syy
    observed = sgy / math.sqrt(sgg * syy)
    corrected = observed / math.sqrt(reliability) if reliability > 0 else math.nan
    return observed, reliability, corrected


def corrected_correlation(grade, truth, positions, clusters, seed: int = 0) -> Dict:
    """The verdict's measure (module docstring), with bootstrap standard
    errors over `clusters` (the source games)."""
    owners, sums = _position_sums(np.asarray(grade, dtype=float),
                                  np.asarray(truth, dtype=float),
                                  np.asarray(positions), np.asarray(clusters))
    if len(sums) == 0:
        return {"positions": 0, "clusters": 0, "observed": math.nan,
                "reliability": math.nan, "corrected": math.nan,
                "observed_se": math.nan, "corrected_se": math.nan}
    _, cluster_of = np.unique(owners, return_inverse=True)
    per_cluster = np.zeros((cluster_of.max() + 1, 4))
    np.add.at(per_cluster, cluster_of, sums)
    observed, reliability, corrected = _ratios(per_cluster.sum(axis=0))
    rng = np.random.default_rng(seed)
    n = len(per_cluster)
    boots = np.array([_ratios(per_cluster[rng.integers(0, n, n)].sum(axis=0))
                      for _ in range(BOOTSTRAP_RESAMPLES)])
    return {"positions": int(len(sums)), "clusters": int(n),
            "observed": observed, "reliability": reliability, "corrected": corrected,
            "observed_se": _percentile_se(boots[:, 0]),
            "corrected_se": _percentile_se(boots[:, 2])}


def _percentile_se(draws: np.ndarray) -> float:
    """Half the central 68% interval of the bootstrap draws: robust to
    the long tail of resamples whose reliability falls near zero, where
    the corrected correlation blows up."""
    finite = draws[np.isfinite(draws)]
    if len(finite) < 10:
        return math.nan
    low, high = np.percentile(finite, [15.87, 84.13])
    return float(high - low) / 2.0


def rule(stats: Dict) -> str:
    """The pre-registered rule for one judged grader."""
    rel, r, se = stats["reliability"], stats["corrected"], stats["corrected_se"]
    if not math.isfinite(rel) or rel < MIN_RELIABILITY:
        return "UNDECIDED (the playout means do not separate the candidates)"
    if not math.isfinite(se) or se > MAX_SE:
        return "UNDECIDED (standard error above 0.12)"
    if r >= PASS_AT:
        return "PASS"
    if r <= FAIL_AT:
        return "FAIL"
    return "INCONCLUSIVE"


# ---------------------------------------------------------------------
# Graders on a part
# ---------------------------------------------------------------------

def rollout_grade(part: Dict, truth: np.ndarray, reads: int,
                  horizon_read: int) -> Tuple[np.ndarray, np.ndarray]:
    """(grade, truth playouts): the mean of the horizon read over the
    first `reads` playouts, and the other playouts of `truth` as the
    truth."""
    value = part["horizon_value"].double().numpy()
    if value.shape[-1] <= horizon_read:
        return np.full(len(truth), np.nan), np.full((len(truth), 1), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # a candidate with no read: NaN
        grade = np.nanmean(value[:, :reads, horizon_read], axis=1)
    return grade, truth[:, reads:]


def fitted_grades(part: Dict, arms: Dict) -> Dict[str, np.ndarray]:
    grades = {name: arm.predict(part["feats"]).double().numpy() for name, arm in arms.items()}
    grades["value_reference"] = part["value_reference"].double().numpy()
    for key, values in part["baselines"].items():
        grades[key] = np.array([np.nan if v is None else float(v) for v in values])
    return grades


def large_gap_ranks(grade: np.ndarray, part: Dict) -> Dict:
    """Alternatives whose playout mean beats their base's by
    GAP_THRESHOLD or more: how many the grader ranks above the base."""
    index, slot = part["index"].tolist(), part["slot"].tolist()
    base = {i: row for row, (i, s) in enumerate(zip(index, slot)) if s == 0}
    raw = part["raw"].numpy()
    counts = {"above": 0, "below": 0, "tied": 0}
    for row, (i, s) in enumerate(zip(index, slot)):
        b = base.get(i)
        if s == 0 or b is None or raw[row] - raw[b] < GAP_THRESHOLD:
            continue
        if not (np.isfinite(grade[row]) and np.isfinite(grade[b])):
            continue
        key = "above" if grade[row] > grade[b] else "below" if grade[row] < grade[b] else "tied"
        counts[key] += 1
    return counts


def evaluate_part(part: Dict, arms: Dict, beta: np.ndarray, *,
                  rollout: bool) -> Dict[str, Dict]:
    """Every grader's measure on a part against the adjusted outcomes,
    with the measure against the raw outcomes beside it (`raw`); the
    rollout grader and its secondary settings when `rollout`."""
    positions, clusters = part["groups"].numpy(), np.asarray(part["group"])
    truths = {"adjusted": adjusted_outcomes(part, beta).numpy(),
              "raw": part["outcomes"].double().numpy()}

    def measure(grade_of) -> Dict:
        stats = {}
        for kind, truth in truths.items():
            grade, rest = grade_of(truth)
            stats[kind] = corrected_correlation(grade, rest, positions, clusters)
        return dict(stats["adjusted"], raw=stats["raw"])

    out = {}
    for name, grade in fitted_grades(part, arms).items():
        out[name] = measure(lambda truth, g=grade: (g, truth))
        out[name]["large_gaps"] = large_gap_ranks(grade, part)
    if rollout:
        for reads, read in ((ROLLOUT_READS, HORIZON_READ),) + ROLLOUT_SECONDARY:
            name = ("rollout" if (reads, read) == (ROLLOUT_READS, HORIZON_READ)
                    else f"rollout_r{reads}_h{read}")
            out[name] = measure(lambda truth, r=reads, h=read: rollout_grade(part, truth, r, h))
    return out


# ---------------------------------------------------------------------
# CLI (registered by tools/turn_value.py)
# ---------------------------------------------------------------------

def _load_cache(path: Path) -> Dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def cmd_fit(args) -> int:
    from tools.turn_value import load_reference_model
    cache = _load_cache(args.train)
    model, _ = load_reference_model(Path(cache["checkpoint"]), torch.device("cpu"))
    train, stop = split_rows(cache, "fit"), split_rows(cache, "stop")
    beta, r2, n_luck = luck_coefficients(train)
    log.info("fit %d candidates, stop %d; luck beta %s explains %.3f of the outcome "
             "variance over %d playouts", len(train["raw"]), len(stop["raw"]), beta, r2, n_luck)
    linear, linear_config, linear_trace = fit_linear_arm(train, stop, beta)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head, head_config, head_trace = fit_head_arm(model.value_head, model._value_atoms,
                                                 train, stop, beta, device=device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tmp = args.out_dir / "arms.pt.tmp"
    torch.save({"linear": linear.state(), "head": head.state_dict(),
                "selected": {"linear": linear_config, "head": head_config},
                "trace": {"linear": linear_trace, "head": head_trace},
                "luck": {"beta": [float(b) for b in beta], "r2": r2, "playouts": n_luck},
                "recipe": asdict(HeadRecipe()), "train_cache": str(args.train),
                "checkpoint": cache["checkpoint"]}, tmp)
    tmp.replace(args.out_dir / "arms.pt")
    log.info("wrote %s", args.out_dir / "arms.pt")
    return 0


def load_arms(path: Path) -> Tuple[Dict, Dict]:
    from tools.turn_value import load_reference_model
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model, _ = load_reference_model(Path(saved["checkpoint"]), torch.device("cpu"))
    head = HeadArm(model.value_head, model._value_atoms)
    head.load_state_dict(saved["head"])
    head.eval()
    return {"linear": LinearArm(**saved["linear"]), "head": head}, saved


def _pair(text: str) -> Tuple[Path, Path]:
    records, cache = text.split("=", 1)
    return Path(records), Path(cache)


def cmd_evaluate(args) -> int:
    arms, saved = load_arms(args.heads / "arms.pt")
    records_path, cache_path = args.validation
    validation_cache = _load_cache(cache_path)
    beta = np.asarray(saved["luck"]["beta"], dtype=float)
    validation = evaluate_part(split_rows(validation_cache), arms, beta, rollout=True)
    for name in JUDGED:
        validation[name]["verdict"] = rule(validation[name])
    out: Dict = {"rule": "docs/turn_value_prereg_20260925.md", "records": str(records_path),
                 "selected": saved["selected"], "luck": saved["luck"],
                 "validation": validation,
                 "counts": {"validation": validation_cache.get("counts"),
                            "validation_skipped": validation_cache.get("skipped"),
                            "validation_errors": validation_cache.get("errors")}}
    if args.train is not None:
        train_cache = _load_cache(args.train)
        proxy = evaluate_part(split_rows(train_cache, "proxy"), arms, beta, rollout=False)
        for stats in proxy.values():
            stats["barrier_passes"] = bool(stats["observed"] - 2 * stats["observed_se"] > 0)
        out["proxy"] = proxy
        out["counts"].update(train=train_cache.get("counts"),
                             train_skipped=train_cache.get("skipped"),
                             train_errors=train_cache.get("errors"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    tmp.replace(args.out)
    report = markdown(out)
    args.out.with_suffix(".md").write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


def _fmt(v, spec: str = ".3f") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return format(v, spec)


def markdown(out: Dict) -> str:
    lines = [f"validation: {out['records']}", "",
             "| grader | corrected r | SE | observed r | reliability | positions "
             "| against raw outcomes | large gaps above / below | verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, s in out["validation"].items():
        gaps = s.get("large_gaps")
        gap_text = "-" if gaps is None else f"{gaps['above']} / {gaps['below']}"
        raw = s["raw"]
        lines.append(f"| {name} | {_fmt(s['corrected'])} | {_fmt(s['corrected_se'])} | "
                     f"{_fmt(s['observed'])} | {_fmt(s['reliability'])} | {s['positions']} | "
                     f"{_fmt(raw['corrected'])} +- {_fmt(raw['corrected_se'])} | "
                     f"{gap_text} | {s.get('verdict', '(reported)')} |")
    if out.get("proxy"):
        lines += ["", "proxy games (in distribution):", "",
                  "| grader | corrected r | SE | observed r | SE | barrier |",
                  "|---|---|---|---|---|---|"]
        for name, s in out["proxy"].items():
            lines.append(f"| {name} | {_fmt(s['corrected'])} | {_fmt(s['corrected_se'])} | "
                         f"{_fmt(s['observed'])} | {_fmt(s['observed_se'])} | "
                         f"{'passes' if s['barrier_passes'] else 'FAILS'} |")
    lines += ["", f"selected: {json.dumps(out['selected'], default=float)}",
              f"luck: {json.dumps(out['luck'], default=float)}",
              f"counts: {json.dumps(out['counts'], default=str)}"]
    return "\n".join(lines)


def add_fit_parsers(sub) -> None:
    fit = sub.add_parser("fit", help="fit the arms on a feature cache")
    fit.add_argument("--train", type=Path, required=True)
    fit.add_argument("--out-dir", type=Path, required=True)
    fit.set_defaults(run=cmd_fit)
    ev = sub.add_parser("evaluate", help="the verdict on the validation file")
    ev.add_argument("--heads", type=Path, required=True)
    ev.add_argument("--validation", type=_pair, required=True,
                    help="RECORDS.json=CACHE.pt of the flat validation run.")
    ev.add_argument("--train", type=Path, default=None,
                    help="The training cache, for the proxy games.")
    ev.add_argument("--out", type=Path, required=True)
    ev.set_defaults(run=cmd_evaluate)

