"""Player ratings over the imitation corpus and the top-rated subset.

Panel item 3 (docs/training_signal_panel_20260905.md, test 3): rate the
corpus's players and keep the games its strongest regulars won.

Two steps.

index: read each manifest game's raw replay header (the [side] blocks
of the start snapshot, through tools.filter_replays.parse_header) and
write players.jsonl: the ids of sides 1 and 2, the game's version and
its upload date. Needs the raw .bz2 files (the manifest's `source`
path, relative to --raw-root); the json.gz game records carry no
player names. Every ai-controlled side is the one id AI_PLAYER_ID.

fit: Bradley-Terry ratings with a Gaussian prior on every rating
(shrinkage toward the corpus mean, weaker the more games a player has)
written to ratings.json, and subset_manifest.jsonl: the non-holdout
games whose winner is a regular (at least --min-games games) rated in
the top --quantile of regulars. Every subset row is its manifest row
plus winner_id, loser_id, winner_rating_elo, winner_se_elo and
winner_games. --build-dataset writes a directory the trainer reads
like the full dataset: hardlinked game files, manifest.jsonl (the
subset rows plus the holdout games won by top-rated players, flagged
holdout so they only feed the barrier eval) and
value_corpus_index.jsonl.

Usage:
    python tools/player_ratings.py index --dataset DIR [--raw-root .] [--workers N]
    python tools/player_ratings.py fit --dataset DIR --min-games 30 --quantile 0.75 \\
        --out DIR [--report] [--build-dataset DIR]
"""
from __future__ import annotations

import argparse
import bz2
import json
import logging
import math
import os
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.filter_replays import parse_header

log = logging.getLogger("player_ratings")

# One id for every ai-controlled side (the built-in RCA AI). Wesnoth
# usernames cannot contain brackets, so it collides with no human.
AI_PLAYER_ID = "[ai]"
# Bytes of decompressed replay read for the header; the [side] blocks
# of the start snapshot sit inside it (tools/replay_manifest.py).
HEADER_BYTES = 400_000
# Elo per natural-log rating unit, as tools/whr.py.
ELO_PER_NATURAL = 400.0 / math.log(10.0)
# Prior standard deviation of a rating, in Elo: a picked knob. At 200,
# a 2-0 player lands near +140 Elo and a 20-10 regular against average
# opponents near +105 (the maximum-likelihood values would be infinity
# and +120).
DEFAULT_PRIOR_SD_ELO = 200.0
# Pre-registration kill 0 (panel test 3): an arm-1 subset under this
# many winner-side pairs switches to the soft-weight arm before renting.
KILL_MIN_PAIRS = 250_000
# Keys the subset rows add to their manifest rows.
SUBSET_EXTRA_KEYS = ("winner_id", "loser_id", "winner_rating_elo",
                     "winner_se_elo", "winner_games")


# ---------------------------------------------------------------------
# Manifest and player index
# ---------------------------------------------------------------------

def read_manifest(dataset_dir: Path) -> List[dict]:
    """Rows of <dataset>/manifest.jsonl, checked for the fields the
    trainer reads (tools/supervised_train.py, imitation mode): file,
    winner_side in {1, 2}, holdout bool; winner_actions when present."""
    rows = read_jsonl(dataset_dir / "manifest.jsonl")
    for k, r in enumerate(rows):
        if not isinstance(r.get("file"), str) or not r["file"]:
            raise ValueError(f"manifest row {k}: missing file")
        if r.get("winner_side") not in (1, 2):
            raise ValueError(f"manifest row {k} ({r['file']}): "
                             f"winner_side {r.get('winner_side')!r}")
        if not isinstance(r.get("holdout"), bool):
            raise ValueError(f"manifest row {k} ({r['file']}): "
                             f"holdout {r.get('holdout')!r}")
        if "winner_actions" in r and not isinstance(r["winner_actions"], int):
            raise ValueError(f"manifest row {k} ({r['file']}): "
                             f"winner_actions {r['winner_actions']!r}")
    return rows


def read_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            n += 1
    return n


def player_id_of_side(side: dict) -> Optional[str]:
    """The id of one parsed [side] block: AI_PLAYER_ID for an ai
    controller, the player name for a human one (player_id, then
    current_player, then save_id, the precedence of the outcome
    labeling in tools/build_value_corpus.py), None otherwise (a null
    side, or a human side without a name)."""
    controller = (side.get("controller") or "").strip().lower()
    if controller == "ai":
        return AI_PLAYER_ID
    if controller not in ("human", "network"):
        return None
    for key in ("player_id", "current_player", "save_id"):
        name = (side.get(key) or "").strip()
        if name:
            return name
    return None


def side_ids_from_header(header: dict) -> Dict[int, Optional[str]]:
    """Player ids of sides 1 and 2 from a parse_header() result. The
    corpus's outcomes name side 1 or 2 as the winner; a third side (a
    league referee, an observer slot) is ignored."""
    out: Dict[int, Optional[str]] = {1: None, 2: None}
    for side in header.get("sides", []):
        try:
            num = int(side.get("side", 0) or 0)
        except ValueError:
            continue
        if num in out and out[num] is None:
            out[num] = player_id_of_side(side)
    return out


def raw_replay_path(source: str, raw_root: Path) -> Path:
    """The manifest's `source` is written with the builder's OS
    separators (backslashes on Windows); resolve it under raw_root."""
    return raw_root / Path(PurePosixPath(source.replace("\\", "/")))


def index_one(job: Tuple[str, str, str]) -> dict:
    """players.jsonl row for one game: file, source, date (the raw
    corpus's day directory), version, sides {"1": id, "2": id}; or
    an `error` field when the header cannot be read."""
    file, source, raw_root = job
    path = raw_replay_path(source, Path(raw_root))
    row: dict = {"file": file, "source": source, "date": path.parent.name}
    try:
        with bz2.open(path, "rb") as f:
            header = parse_header(f.read(HEADER_BYTES))
    except Exception as e:                           # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}"[:160]
        return row
    row["version"] = header["top"].get("version", "")
    row["sides"] = {str(k): v for k, v in side_ids_from_header(header).items()}
    return row


def build_player_index(rows: Sequence[dict], raw_root: Path, out_path: Path,
                       *, workers: int = 1) -> Counter:
    """Write players.jsonl for `rows` (manifest rows, in order) and
    return counts (ok / error / sides_missing)."""
    jobs = [(r["file"], r["source"], str(raw_root)) for r in rows]
    stats: Counter = Counter()
    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as f:
        if workers > 1:
            pool = Pool(workers)
            results = pool.imap(index_one, jobs, chunksize=50)
        else:
            pool = None
            results = map(index_one, jobs)
        try:
            for i, row in enumerate(results, 1):
                if "error" in row:
                    stats["error"] += 1
                    log.warning("%s: %s", row["file"], row["error"])
                else:
                    stats["ok"] += 1
                    if None in row["sides"].values():
                        stats["sides_missing"] += 1
                f.write(json.dumps(row) + "\n")
                if i % 1000 == 0:
                    rate = i / (time.time() - t0)
                    log.info("[%d/%d] %.0f/s eta %.0fs", i, len(jobs), rate,
                             (len(jobs) - i) / rate)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    return stats


# ---------------------------------------------------------------------
# Rating fit
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class GameRecord:
    file: str
    winner_id: str
    loser_id: str


@dataclass
class PlayerRating:
    id: str
    rating_elo: float
    se_elo: float
    games: int
    wins: int


@dataclass
class RatingFit:
    players: List[PlayerRating]          # strongest first
    prior_sd_elo: float
    n_games: int

    def by_id(self) -> Dict[str, PlayerRating]:
        return {p.id: p for p in self.players}


def game_records(rows: Sequence[dict], index: Dict[str, dict]
                 ) -> Tuple[List[GameRecord], Counter]:
    """One record per manifest row whose two sides have distinct known
    ids; the counter says why the others were skipped."""
    out: List[GameRecord] = []
    skipped: Counter = Counter()
    for r in rows:
        entry = index.get(r["file"])
        if entry is None or "sides" not in entry:
            skipped["not_indexed"] += 1
            continue
        winner = entry["sides"].get(str(r["winner_side"]))
        loser = entry["sides"].get(str(3 - r["winner_side"]))
        if not winner or not loser:
            skipped["side_id_missing"] += 1
        elif winner == loser:
            skipped["same_id_both_sides"] += 1
        else:
            out.append(GameRecord(r["file"], winner, loser))
    return out, skipped


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def fit_bradley_terry(records: Sequence[GameRecord], *,
                      prior_sd_elo: float = DEFAULT_PRIOR_SD_ELO) -> RatingFit:
    """Posterior mode of the Bradley-Terry model P(i beats j) =
    sigmoid(r_i - r_j) with r_k ~ N(0, sigma^2) on every player
    (sigma = prior_sd_elo in natural units); the prior also fixes the
    gauge, so 0 Elo is the corpus mean. Standard errors are the
    diagonal Laplace approximation, 1 / sqrt(prior precision + sum
    over the player's games of p (1 - p)). Deterministic: players are
    indexed in sorted id order and L-BFGS runs from zero."""
    ids = sorted({g.winner_id for g in records} | {g.loser_id for g in records})
    idx = {pid: k for k, pid in enumerate(ids)}
    n = len(ids)
    pair_counts = Counter((idx[g.winner_id], idx[g.loser_id]) for g in records)
    pairs = sorted(pair_counts.items())
    winners = np.array([w for (w, _l), _c in pairs], dtype=np.int64)
    losers = np.array([lo for (_w, lo), _c in pairs], dtype=np.int64)
    counts = np.array([c for _p, c in pairs], dtype=np.float64)
    precision = 1.0 / (prior_sd_elo / ELO_PER_NATURAL) ** 2

    def objective(r: np.ndarray) -> Tuple[float, np.ndarray]:
        d = r[winners] - r[losers]
        nll = float(np.dot(counts, np.logaddexp(0.0, -d)))
        q = counts * (1.0 - _sigmoid(d))          # -d nll / d r_winner
        grad = (np.bincount(losers, weights=q, minlength=n)
                - np.bincount(winners, weights=q, minlength=n)
                + precision * r)
        return nll + 0.5 * precision * float(np.dot(r, r)), grad

    r = np.zeros(n)
    if n:
        res = minimize(objective, r, jac=True, method="L-BFGS-B",
                       options={"gtol": 1e-7, "ftol": 1e-14, "maxiter": 10_000})
        r = res.x
        if float(np.max(np.abs(res.jac))) > 1e-4:
            raise RuntimeError(f"rating fit did not converge: {res.message}")
    p = _sigmoid(r[winners] - r[losers])
    h = counts * p * (1.0 - p)
    info = (precision + np.bincount(winners, weights=h, minlength=n)
            + np.bincount(losers, weights=h, minlength=n))
    se = 1.0 / np.sqrt(info)
    games = Counter()
    wins = Counter()
    for g in records:
        games[g.winner_id] += 1
        games[g.loser_id] += 1
        wins[g.winner_id] += 1
    players = [PlayerRating(pid, float(r[k] * ELO_PER_NATURAL),
                            float(se[k] * ELO_PER_NATURAL),
                            games[pid], wins[pid])
               for pid, k in idx.items()]
    players.sort(key=lambda p: (-p.rating_elo, p.id))
    return RatingFit(players, prior_sd_elo, len(records))


# ---------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------

@dataclass
class TopSet:
    ids: Set[str]
    threshold_elo: Optional[float]      # None when there is no regular
    n_regulars: int


def select_top_regulars(fit: RatingFit, *, min_games: int,
                        quantile: float) -> TopSet:
    """Regulars are human players with at least min_games games (the
    AI is rated but never imitated); the top set is the regulars whose
    rating reaches the `quantile` quantile of the regulars' ratings."""
    regulars = [p for p in fit.players
                if p.games >= min_games and p.id != AI_PLAYER_ID]
    if not regulars:
        return TopSet(set(), None, 0)
    threshold = float(np.quantile([p.rating_elo for p in regulars], quantile))
    ids = {p.id for p in regulars if p.rating_elo >= threshold}
    return TopSet(ids, threshold, len(regulars))


def annotated_rows(rows: Sequence[dict], records: Sequence[GameRecord],
                   fit: RatingFit, top: TopSet, *, holdout: bool) -> List[dict]:
    """Manifest rows (with holdout == `holdout`) won by a top-rated
    player, each extended by SUBSET_EXTRA_KEYS; manifest order."""
    by_file = {g.file: g for g in records}
    ratings = fit.by_id()
    out = []
    for r in rows:
        g = by_file.get(r["file"])
        if r["holdout"] != holdout or g is None or g.winner_id not in top.ids:
            continue
        w = ratings[g.winner_id]
        out.append({**r, "winner_id": g.winner_id, "loser_id": g.loser_id,
                    "winner_rating_elo": round(w.rating_elo, 2),
                    "winner_se_elo": round(w.se_elo, 2),
                    "winner_games": w.games})
    return out


def estimate_pairs(rows: Sequence[dict]) -> Optional[int]:
    """Sum of the manifest's winner_actions (the trainer's per-game
    pair count estimate), or None when a row lacks it."""
    if any("winner_actions" not in r for r in rows):
        return None
    return int(sum(r["winner_actions"] for r in rows))


def build_dataset_dir(dataset_dir: Path, out_dir: Path,
                      rows: Sequence[dict]) -> int:
    """A dataset directory for `rows`: manifest.jsonl,
    value_corpus_index.jsonl (file / winner / n_commands, as
    tools/build_imitation_dataset.py writes it) and the game files,
    hardlinked (copied when the filesystem refuses). Refuses a
    non-empty directory so two subsets never mix."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty")
    write_jsonl(out_dir / "manifest.jsonl", rows)
    write_jsonl(out_dir / "value_corpus_index.jsonl",
                ({"file": r["file"], "winner": r["winner_side"],
                  "n_commands": r.get("n_commands", 0)} for r in rows))
    for r in rows:
        src, dst = dataset_dir / r["file"], out_dir / r["file"]
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return len(rows)


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def rating_table(fit: RatingFit, *, limit: int) -> str:
    lines = [f"{'rank':>4}  {'player':<24} {'elo':>7} {'se':>5} {'games':>6} {'wins':>5}"]
    for k, p in enumerate(fit.players[:limit], 1):
        lines.append(f"{k:>4}  {p.id:<24} {p.rating_elo:>7.0f} {p.se_elo:>5.0f} "
                     f"{p.games:>6} {p.wins:>5}")
    ai = fit.by_id().get(AI_PLAYER_ID)
    if ai is not None:
        lines.append(f"{'ai':>4}  {ai.id:<24} {ai.rating_elo:>7.0f} {ai.se_elo:>5.0f} "
                     f"{ai.games:>6} {ai.wins:>5}")
    return "\n".join(lines)


def subset_summary(rows: Sequence[dict], subset: Sequence[dict],
                   holdout_top: Sequence[dict], top: TopSet) -> str:
    train_rows = [r for r in rows if not r["holdout"]]
    pairs = estimate_pairs(subset)
    total = estimate_pairs(train_rows)
    lines = [f"top-rated regulars: {len(top.ids)} of {top.n_regulars} "
             f"(threshold {top.threshold_elo:.0f} Elo)" if top.threshold_elo is not None
             else "no regular at this --min-games",
             f"subset: {len(subset)} of {len(train_rows)} training games"]
    if pairs is None:
        lines.append(f"pairs: no winner_actions in the manifest; kill 0 reads games "
                     f"({len(subset)})")
    else:
        share = pairs / total if total else 0.0
        verdict = "KILL 0 (under 250k pairs: soft-weight arm)" if pairs < KILL_MIN_PAIRS \
            else "passes kill 0"
        lines.append(f"pairs: {pairs:,} of {total:,} winner-side pairs "
                     f"({share:.1%}); {verdict}")
    lines.append(f"holdout games won by top-rated players (barrier eval): "
                 f"{len(holdout_top)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def cmd_index(args: argparse.Namespace) -> int:
    rows = read_manifest(args.dataset)
    if args.limit:
        rows = rows[:args.limit]
    out = args.out or args.dataset / "players.jsonl"
    stats = build_player_index(rows, args.raw_root, out, workers=args.workers)
    print(f"indexed {stats['ok']} games -> {out} "
          f"(errors {stats['error']}, sides missing {stats['sides_missing']})")
    return 1 if stats["error"] else 0


def cmd_fit(args: argparse.Namespace) -> int:
    rows = read_manifest(args.dataset)
    players_path = args.players or args.dataset / "players.jsonl"
    index = {r["file"]: r for r in read_jsonl(players_path)}
    records, skipped = game_records(rows, index)
    log.info("%d games rated, skipped %s", len(records), dict(skipped))
    fit = fit_bradley_terry(records, prior_sd_elo=args.prior_sd_elo)
    top = select_top_regulars(fit, min_games=args.min_games, quantile=args.quantile)
    subset = annotated_rows(rows, records, fit, top, holdout=False)
    holdout_top = annotated_rows(rows, records, fit, top, holdout=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ratings.json").write_text(json.dumps({
        "prior_sd_elo": fit.prior_sd_elo, "min_games": args.min_games,
        "quantile": args.quantile, "threshold_elo": top.threshold_elo,
        "n_games": fit.n_games, "skipped": dict(skipped),
        "games_by_version": dict(Counter(
            index[g.file].get("version", "") for g in records)),
        "top_ids": sorted(top.ids),
        "players": [asdict(p) for p in fit.players],
    }, indent=1), encoding="utf-8")
    write_jsonl(args.out / "subset_manifest.jsonl", subset)
    if args.build_dataset is not None:
        n = build_dataset_dir(args.dataset, args.build_dataset, subset + holdout_top)
        print(f"dataset: {n} games -> {args.build_dataset}")
    if args.report:
        print(rating_table(fit, limit=args.table))
    print(subset_summary(rows, subset, holdout_top, top))
    return 0


def main(argv: List[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    ix = sub.add_parser("index", help="players.jsonl from the raw replay headers")
    ix.add_argument("--dataset", type=Path, required=True)
    ix.add_argument("--raw-root", type=Path, default=Path("."),
                    help="directory the manifest's source paths are relative to")
    ix.add_argument("--out", type=Path, default=None,
                    help="default <dataset>/players.jsonl")
    ix.add_argument("--workers", type=int, default=1)
    ix.add_argument("--limit", type=int, default=0, help="first N games only")
    ix.set_defaults(func=cmd_index)

    ft = sub.add_parser("fit", help="ratings.json and subset_manifest.jsonl")
    ft.add_argument("--dataset", type=Path, required=True)
    ft.add_argument("--players", type=Path, default=None,
                    help="default <dataset>/players.jsonl")
    ft.add_argument("--min-games", type=int, default=30)
    ft.add_argument("--quantile", type=float, default=0.75)
    ft.add_argument("--prior-sd-elo", type=float, default=DEFAULT_PRIOR_SD_ELO)
    ft.add_argument("--out", type=Path, required=True)
    ft.add_argument("--report", action="store_true", help="print the rating table")
    ft.add_argument("--table", type=int, default=25, help="rows of the rating table")
    ft.add_argument("--build-dataset", type=Path, default=None,
                    help="write a trainer-ready dataset directory here")
    ft.set_defaults(func=cmd_fit)

    args = ap.parse_args(argv[1:])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
