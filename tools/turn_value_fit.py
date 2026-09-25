"""Arms, proxy barrier and verdict of the turn-ranking value function
(docs/turn_value_prereg_20260925.md); the CLI is tools/turn_value.py.

Both arms read the frozen reference trunk's global token and minimize,
over the candidates of the fit split,

    mean w (v - y)^2  +  RANK_WEIGHT * mean w ((v - v_p) - (y - y_p))^2

where y is a candidate's playout mean, w its playout count, and v_p, y_p
the weighted means over its position's candidates (the second term is
the ranking term: only differences within a position enter it).

  linear  standardized token -> value, in closed form (ridge), the ridge
          strength chosen on the stop split's loss.
  head    the reference value head's shape (d -> d -> atoms, expected
          value over the atoms), initialized from it, trained with AdamW
          and early-stopped on the stop split's loss (epoch 0 is the
          reference head itself).
"""
from __future__ import annotations

import copy
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "analysis"))

import turn_gap_pregrader as pg  # noqa: E402

log = logging.getLogger("turn_value_fit")

RANK_WEIGHT = 1.0
RIDGE_GRID = tuple(10.0 ** k for k in range(-6, 2))
ARMS = ("linear", "head")
GAP_THRESHOLD = 0.25
BOOTSTRAP_RESAMPLES = 1000


# ---------------------------------------------------------------------
# The loss
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


def turn_loss(v: torch.Tensor, y: torch.Tensor, w: torch.Tensor, groups: torch.Tensor,
              rank_weight: float = RANK_WEIGHT) -> torch.Tensor:
    level = (w * (v - y) ** 2).sum() / w.sum()
    rank = (w * (centered(v, w, groups) - centered(y, w, groups)) ** 2).sum() / w.sum()
    return level + rank_weight * rank


# ---------------------------------------------------------------------
# The arms
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


def fit_linear(feats, y, w, groups, ridge: float, mean=None, std=None,
               rank_weight: float = RANK_WEIGHT) -> LinearArm:
    """Closed-form minimizer of turn_loss plus ridge * |coef|^2 (the
    intercept unpenalized) on standardized features: the normal
    equations of the weighted sums, so the ridge term carries sum(w)."""
    x = feats.double()
    if mean is None:
        mean = x.mean(dim=0)
        std = x.std(dim=0).clamp_min(1e-6)
    z = (x - mean) / std
    y, w = y.double(), w.double()
    ones = torch.ones(len(z), 1, dtype=z.dtype)
    a = torch.cat([z, ones], dim=1)
    c = torch.cat([centered_rows(z, w, groups), torch.zeros(len(z), 1, dtype=z.dtype)], dim=1)
    yc = centered(y, w, groups)
    lhs = (a * w[:, None]).T @ a + rank_weight * (c * w[:, None]).T @ c
    lhs[:-1, :-1] += ridge * float(w.sum()) * torch.eye(z.shape[1], dtype=z.dtype)
    rhs = (a * w[:, None]).T @ y + rank_weight * (c * w[:, None]).T @ yc
    theta = torch.linalg.solve(lhs, rhs)
    return LinearArm(mean=mean, std=std, coef=theta[:-1], intercept=float(theta[-1]), ridge=ridge)


def centered_rows(z: torch.Tensor, w: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Each row of z minus its group's weighted mean row."""
    g = int(groups.max()) + 1
    wsum = torch.zeros(g, dtype=z.dtype).index_add(0, groups, w)
    mean = torch.zeros(g, z.shape[1], dtype=z.dtype).index_add(0, groups, w[:, None] * z)
    return z - (mean / wsum[:, None])[groups]


def select_linear(train: Dict, stop: Dict) -> Tuple[LinearArm, List[Dict]]:
    """The ridge strength with the lowest stop-split loss."""
    trace, best = [], None
    for ridge in RIDGE_GRID:
        arm = fit_linear(train["feats"], train["y"], train["w"], train["groups"], ridge)
        loss = float(turn_loss(arm.predict(stop["feats"]), stop["y"].double(),
                               stop["w"].double(), stop["groups"]))
        trace.append({"ridge": ridge, "stop_loss": loss})
        if best is None or loss < best[0]:
            best = (loss, arm)
    log.info("linear arm: ridge %g (stop loss %.4f)", best[1].ridge, best[0])
    return best[1], trace


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


def fit_head(value_head, atoms, train: Dict, stop: Dict,
             recipe: HeadRecipe = HeadRecipe()) -> Tuple[HeadArm, List[Dict]]:
    torch.manual_seed(recipe.seed)
    gen = torch.Generator().manual_seed(recipe.seed)
    arm = HeadArm(value_head, atoms)
    opt = torch.optim.AdamW(arm.parameters(), lr=recipe.lr, weight_decay=recipe.weight_decay)
    feats, y, w, groups = (train["feats"].float(), train["y"].float(), train["w"].float(),
                           train["groups"])

    def stop_loss() -> float:
        return float(turn_loss(arm.predict(stop["feats"]), stop["y"].float(),
                               stop["w"].float(), stop["groups"]))

    best_loss, best_state, best_epoch = stop_loss(), copy.deepcopy(arm.state_dict()), 0
    trace = [{"epoch": 0, "stop_loss": best_loss}]
    for epoch in range(1, recipe.max_epochs + 1):
        arm.train()
        for rows in _batches(groups, recipe.positions_per_batch, gen):
            loss = turn_loss(arm(feats[rows]), y[rows], w[rows], position_ids(groups[rows]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        arm.eval()
        loss = stop_loss()
        trace.append({"epoch": epoch, "stop_loss": loss})
        if loss < best_loss:
            best_loss, best_state, best_epoch = loss, copy.deepcopy(arm.state_dict()), epoch
        elif epoch - best_epoch >= recipe.patience:
            break
    arm.load_state_dict(best_state)
    log.info("head arm: epoch %d of %d (stop loss %.4f; the reference head %.4f)",
             best_epoch, trace[-1]["epoch"], best_loss, trace[0]["stop_loss"])
    return arm, trace


# ---------------------------------------------------------------------
# Splits of a cache
# ---------------------------------------------------------------------

def split_rows(cache: Dict, split: str) -> Dict:
    rows = torch.tensor([s == split for s in cache["split"]])
    index = cache["index"][rows]
    return {"feats": cache["feats"][rows], "y": cache["y"][rows], "w": cache["n"][rows],
            "index": index, "slot": cache["slot"][rows], "groups": position_ids(index),
            "value_reference": cache["value_reference"][rows],
            "group": [g for g, keep in zip(cache["group"], rows.tolist()) if keep]}


# ---------------------------------------------------------------------
# The proxy barrier
# ---------------------------------------------------------------------

def within_correlation(pred: np.ndarray, part: Dict) -> Dict:
    """Correlation of predicted and observed deviations from their
    position's mean, pooled over the positions with two or more
    candidates, with a bootstrap SE over the source games."""
    groups, w = part["groups"], part["w"].double()
    xc = centered(torch.as_tensor(pred, dtype=torch.float64), w, groups).numpy()
    yc = centered(part["y"].double(), w, groups).numpy()
    wn = w.numpy()
    games = np.unique(np.asarray(part["group"]), return_inverse=True)[1]
    sums = np.zeros((games.max() + 1, 3))
    np.add.at(sums, games, np.stack([wn * xc * yc, wn * xc * xc, wn * yc * yc], axis=1))

    def corr(s: np.ndarray) -> float:
        return float(s[0] / math.sqrt(s[1] * s[2])) if s[1] > 0 and s[2] > 0 else float("nan")

    r = corr(sums.sum(axis=0))
    rng = np.random.default_rng(0)
    boots = [corr(sums[rng.integers(0, len(sums), len(sums))].sum(axis=0))
             for _ in range(BOOTSTRAP_RESAMPLES)]
    se = float(np.nanstd(boots, ddof=1))
    return {"r": r, "se": se, "games": int(len(sums)), "candidates": int(len(xc)),
            "passes": bool(r - 2.0 * se > 0.0)}


# ---------------------------------------------------------------------
# The verdict on a validation file
# ---------------------------------------------------------------------

def annotate(records: List[Dict], cache: Dict, predictions: Dict[str, np.ndarray]) -> None:
    """Write each grader's read into the candidates it has a state for:
    `grader_<arm>`, `value_reference` (the reference head on the same
    token) and `value_pre` (the reference's read while playing)."""
    by_key = {(int(i), int(s)): row for row, (i, s) in
              enumerate(zip(cache["index"].tolist(), cache["slot"].tolist()))}
    for rec in records:
        for slot, cand in enumerate([rec["base"]] + list(rec.get("alternatives", []))):
            snap = cand.get("pre_end_turn") or {}
            cand["value_pre"] = snap.get("value_pre")
            row = by_key.get((int(rec["index"]), slot))
            if row is None:
                continue
            cand["value_reference"] = float(cache["value_reference"][row])
            for key, pred in predictions.items():
                cand[key] = float(pred[row])


def rule(p: Dict, null_sd: Optional[float]) -> Dict:
    """The pre-registered rule for one grader on one file."""
    sd = p["residual_sd_within_position"]
    n_large = p["n_confirmed"] + p["n_confirmed_unranked"]
    df = p["n"] - p["n_positions_graded"] - 1
    se = sd / math.sqrt(2 * df) if sd is not None and df > 0 else None
    out = {"residual_sd": sd, "residual_sd_se": se, "null_sd": null_sd,
           "n_large_gaps": n_large, "ranked_above": p["n_confirmed_ranked_above"],
           "ranked_below": p["n_confirmed_ranked_below"], "df": df}
    if n_large < 4:
        verdict = "UNDECIDED (fewer than 4 alternatives with a gap >= 0.25)"
    elif sd is None or se is None:
        verdict = "UNDECIDED (no within-position residuals)"
    elif sd >= 0.3 or p["n_confirmed_ranked_below"]:
        verdict = "FAIL"
    elif (sd <= 0.2 and sd + 2 * se < 0.3 and p["n_confirmed_ranked_above"] == n_large
          and null_sd is not None and sd <= null_sd - se):
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"
    out["verdict"] = verdict
    return out


def evaluate_file(records: List[Dict], cache: Dict, arms: Dict,
                  baselines: Sequence[str]) -> Dict:
    records = [r for r in records if r.get("base", {}).get("outcomes")]
    predictions = {f"grader_{name}": arm.predict(cache["feats"]).numpy()
                   for name, arm in arms.items()}
    annotate(records, cache, predictions)
    keys = list(predictions) + list(baselines)
    result = pg.analyze(records, GAP_THRESHOLD, keys=keys)
    null_sd = pg.null_within_position_sd(records)
    for key in keys:
        p = result["pregraders"][key]
        p["n_positions_graded"] = len({int(r["index"]) for r in records
                                       for c in pg.candidates_of(r)
                                       if pg.pregrader_value(c, key) is not None})
        p["rule"] = rule(p, null_sd)
    result["null_sd"] = null_sd
    return result


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
    log.info("fit %d candidates in %d positions, stop %d in %d", len(train["y"]),
             int(train["groups"].max()) + 1, len(stop["y"]), int(stop["groups"].max()) + 1)
    linear, linear_trace = select_linear(train, stop)
    head, head_trace = fit_head(model.value_head, model._value_atoms, train, stop)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"linear": linear.state(), "head": head.state_dict(),
                "rank_weight": RANK_WEIGHT, "recipe": HeadRecipe().__dict__,
                "trace": {"linear": linear_trace, "head": head_trace},
                "train_cache": str(args.train), "checkpoint": cache["checkpoint"]},
               args.out_dir / "arms.pt")
    log.info("wrote %s", args.out_dir / "arms.pt")
    return 0


def load_arms(path: Path) -> Dict:
    from tools.turn_value import load_reference_model
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model, _ = load_reference_model(Path(saved["checkpoint"]), torch.device("cpu"))
    linear = LinearArm(**saved["linear"])
    head = HeadArm(model.value_head, model._value_atoms)
    head.load_state_dict(saved["head"])
    head.eval()
    return {"linear": linear, "head": head}


def _pair(text: str) -> Tuple[Path, Path]:
    records, cache = text.split("=", 1)
    return Path(records), Path(cache)


def cmd_evaluate(args) -> int:
    arms = load_arms(args.heads / "arms.pt")
    out: Dict = {"rule": "docs/turn_value_prereg_20260925.md", "files": {}, "proxy": {}}
    baselines = ("value_reference", "value_pre", "value_post", "hp_margin_post")
    for role, pair in (("primary", args.primary), ("secondary", args.secondary)):
        if pair is None:
            continue
        records_path, cache_path = pair
        records = json.loads(records_path.read_text(encoding="utf-8"))["positions"]
        out["files"][role] = {"records": str(records_path),
                              **evaluate_file(records, _load_cache(cache_path), arms, baselines)}
    if args.train is not None:
        proxy = split_rows(_load_cache(args.train), "proxy")
        reads = {name: arm.predict(proxy["feats"]).numpy() for name, arm in arms.items()}
        reads["value_reference"] = proxy["value_reference"].numpy()
        out["proxy"] = {name: within_correlation(pred, proxy) for name, pred in reads.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    report = markdown(out)
    args.out.with_suffix(".md").write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


def markdown(out: Dict) -> str:
    def f(v, spec=".3f"):
        return "-" if v is None else format(v, spec)
    lines = []
    for role, res in out["files"].items():
        lines += [f"### {role}: {res['records']} ({res['n_positions']} positions, "
                  f"{res['n_candidates']} candidates; null grader {f(res['null_sd'])})", "",
                  "| grader | within-position residual SD | large gaps ranked above / below "
                  "| top pick agrees | verdict |", "|---|---|---|---|---|"]
        for key, p in res["pregraders"].items():
            r = p["rule"]
            lines.append(f"| {key} | {f(r['residual_sd'])} +- {f(r['residual_sd_se'])} | "
                         f"{r['ranked_above']} / {r['ranked_below']} of {r['n_large_gaps']} | "
                         f"{p['top_pick_agrees']}/{p['n_positions_ranked']} | "
                         f"{r['verdict'] if key.startswith('grader_') else '(baseline)'} |")
        lines.append("")
    if out["proxy"]:
        lines += ["### proxy barrier (within-position correlation, held-out games)", "",
                  "| grader | r | SE | games | passes |", "|---|---|---|---|---|"]
        for key, p in out["proxy"].items():
            lines.append(f"| {key} | {f(p['r'])} | {f(p['se'])} | {p['games']} | {p['passes']} |")
    return "\n".join(lines)


def add_fit_parsers(sub) -> None:
    fit = sub.add_parser("fit", help="fit the arms on a feature cache")
    fit.add_argument("--train", type=Path, required=True)
    fit.add_argument("--out-dir", type=Path, required=True)
    fit.set_defaults(run=cmd_fit)
    ev = sub.add_parser("evaluate", help="the verdict on validation files")
    ev.add_argument("--heads", type=Path, required=True)
    ev.add_argument("--primary", type=_pair, default=None,
                    help="RECORDS.json=CACHE.pt of the confirmation run (the verdict's file; "
                         "absent when the screen had no hit to confirm).")
    ev.add_argument("--secondary", type=_pair, default=None,
                    help="RECORDS.json=CACHE.pt of the screen run.")
    ev.add_argument("--train", type=Path, default=None,
                    help="The training cache, for the proxy barrier on its proxy split.")
    ev.add_argument("--out", type=Path, required=True)
    ev.set_defaults(run=cmd_evaluate)
