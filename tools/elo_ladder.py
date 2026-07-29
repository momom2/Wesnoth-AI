"""Internal Elo ladder over the simulator (plan §3.4-1).

Round-robins a set of PLAYERS (trained checkpoints + the random-init
baseline, optionally the scripted `DummyPolicy`) through the bit-exact
simulator and fits static Bradley-Terry / Elo ratings, giving a single
internal strength axis so every training run has a measurable
trajectory ("is iteration N actually stronger than iteration N-k?").

Why this, and what it is NOT:
  + Cheap + stable: all games run in `WesnothSim` (~1000x faster than
    spawning Wesnoth), reusing `eval_sim`'s rollout + policy loading.
  + A SINGLE number per player, jointly fit across all pairings, with
    a principled standard error -- strictly more informative than the
    pairwise win-rate `eval_sim` already prints.
  - NOT human-calibrated. The only opponents here are our own
    policies + a scripted floor; RCA AI is live-Wesnoth
    (`eval_vs_builtin.py`) and human strength is the replay-agreement
    bridge (plan §3.4-2). This ladder measures RELATIVE internal
    progress, gauge-fixed by pinning one anchor player (default the
    random-init baseline) at a chosen Elo.

Game protocol (2026-07-29): games are played as MIRRORED SETUP PAIRS
-- each sampled setup (map + factions + leaders + ToD) is played once
per side by each player, so map/faction/side advantages cancel within
the pair. Mirror-pair sweep counts are reported alongside Elo (a
sweep is paired evidence of a strength gap; a 1-1 split is a
setup-decided pair, i.e. no evidence). The eval horizon defaults to
100 turns to sit inside the training horizon band; report headers and
saved JSON always carry max_turns/seed/draw_weight so a result cannot
be quoted without its protocol.

Rating model (dependency-free; numpy only):
  Bradley-Terry  P(i beats j) = gamma_i / (gamma_i + gamma_j), draws
  counted as half a win to each side. Fit by the MM algorithm (Hunter,
  "MM algorithms for generalized Bradley-Terry models", Ann. Stat.
  2004) -- monotone, no learning rate, no scipy. A weak symmetric
  prior (`--prior-games` ghost games per pair, split 50/50) regularizes
  toward equality so an undefeated or winless player gets a finite
  rating instead of +/-inf. Standard errors come from the diagonal of
  the inverse Fisher information in the log-gamma parameterization,
  with the anchor row/col dropped to remove the gauge freedom.

  Elo_i = 400 * log10(gamma_i), recentered so the anchor sits at
  --anchor-elo. (Elo gap of 400 == 10:1 odds, the usual convention.)

Usage:
    python tools/elo_ladder.py \\
        --player random=random \\
        --player iter50=training/checkpoints/sim_selfplay_archive_50.pt \\
        --player latest=training/checkpoints/sim_selfplay.pt \\
        --games-per-pair 30 --save-json training/logs/elo_ladder.json

    # convenience: glob a directory of checkpoints + add baselines
    python tools/elo_ladder.py \\
        --checkpoints training/checkpoints/sim_selfplay*.pt \\
        --include-random --include-dummy --games-per-pair 20
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from tools.device_select import select_inference_device, describe_device
from tools.scenario_pool import build_scenario_gamestate, random_setup
from tools.sim_self_play import _recruit_cost_lookup
from tools.eval_sim import (_PolicyPair, _load_policy,
                            _play_one_eval_game, peek_checkpoint_arch)
from wesnoth_sim import WesnothSim


log = logging.getLogger("elo_ladder")

# Elo per natural-log unit of gamma: Elo = 400*log10(gamma) =
# (400/ln 10)*ln(gamma). ~173.72.
_ELO_PER_LN = 400.0 / math.log(10.0)


# =====================================================================
# Pure rating math (torch-free, sim-free -> unit-testable in isolation)
# =====================================================================

@dataclass
class PairRecord:
    """Aggregated outcomes between two players i and j (i < j by
    convention). `wins_i` / `wins_j` are decisive results; `draws`
    pools draws + timeouts (no decisive winner)."""
    wins_i: int = 0
    draws:  int = 0
    wins_j: int = 0

    @property
    def games(self) -> int:
        return self.wins_i + self.draws + self.wins_j


def _win_and_game_matrices(
    n: int, pairs: Dict[Tuple[int, int], PairRecord], prior_games: float,
    draw_weight: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the regularized win-mass vector W (length n) and games
    matrix N (n x n, symmetric, zero diagonal) the MM fit consumes.

    `draw_weight` is the fraction of a draw credited as a win to EACH
    side. Default 0.0 DROPS draws: a Wesnoth "draw" is a turn-budget
    TIMEOUT, not evidence of equality (neither side could force a win in
    the budget; even an agent vs itself almost always resolves via
    RNG/terrain/faction asymmetry). Counting it as half-a-win would
    wrongly pull drawn opponents toward equal. 0.5 = textbook half-win
    for games with legitimate draws.

    The prior adds `prior_games` ghost games to EVERY unordered pair,
    split 50/50, i.e. a uniform shrink toward 50% with strength
    `prior_games`."""
    W = np.zeros(n, dtype=float)
    N = np.zeros((n, n), dtype=float)
    for (i, j), rec in pairs.items():
        eff = rec.wins_i + rec.wins_j + 2.0 * draw_weight * rec.draws
        N[i, j] += eff
        N[j, i] += eff
        W[i] += rec.wins_i + draw_weight * rec.draws
        W[j] += rec.wins_j + draw_weight * rec.draws
    if prior_games > 0:
        for i, j in combinations(range(n), 2):
            N[i, j] += prior_games
            N[j, i] += prior_games
            W[i] += 0.5 * prior_games
            W[j] += 0.5 * prior_games
    return W, N


def fit_bradley_terry(
    W: np.ndarray, N: np.ndarray, iters: int = 10000, tol: float = 1e-10,
) -> np.ndarray:
    """MM fit for the Bradley-Terry strengths gamma (Hunter 2004).
    Returns gamma normalized to geometric mean 1. Monotone in the
    log-likelihood; no step size."""
    n = len(W)
    gamma = np.ones(n, dtype=float)
    for _ in range(iters):
        new = gamma.copy()
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if j == i or N[i, j] == 0:
                    continue
                denom += N[i, j] / (gamma[i] + gamma[j])
            if denom > 0 and W[i] > 0:
                new[i] = W[i] / denom
        # Renormalize to geometric mean 1 (kills the scale gauge,
        # prevents drift/overflow). Recentering to the anchor happens
        # downstream in Elo space.
        gm = math.exp(float(np.mean(np.log(new))))
        new /= gm
        if np.max(np.abs(new - gamma)) < tol:
            gamma = new
            break
        gamma = new
    return gamma


def elo_standard_errors(
    gamma: np.ndarray, N: np.ndarray, anchor_idx: int,
) -> np.ndarray:
    """Approximate per-player Elo SEs from the inverse Fisher
    information in beta = ln(gamma). The info matrix is singular (one
    gauge degree of freedom), so we pin the anchor: drop its row/col,
    invert the rest (pseudo-inverse fallback), and report the anchor's
    SE as 0 by construction.

    Fisher info (expected == observed here):
      I[i,i] =  sum_j N[i,j] p_ij (1 - p_ij)
      I[i,j] = -N[i,j] p_ij (1 - p_ij)
    """
    n = len(gamma)
    info = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if j == i or N[i, j] == 0:
                continue
            p = gamma[i] / (gamma[i] + gamma[j])
            w = N[i, j] * p * (1.0 - p)
            info[i, i] += w
            info[i, j] -= w
    keep = [k for k in range(n) if k != anchor_idx]
    se = np.zeros(n, dtype=float)
    if keep:
        sub = info[np.ix_(keep, keep)]
        try:
            cov = np.linalg.inv(sub)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(sub)
        var = np.clip(np.diag(cov), 0.0, None)
        for idx, k in enumerate(keep):
            se[k] = _ELO_PER_LN * math.sqrt(var[idx])
    return se


def fit_elo(
    n: int, pairs: Dict[Tuple[int, int], PairRecord],
    anchor_idx: int, anchor_elo: float = 0.0, prior_games: float = 1.0,
    draw_weight: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """End-to-end: aggregate -> MM fit -> Elo (recentered to anchor)
    + SEs. Returns (elo[n], se[n]). `draw_weight` (default 0.0) drops
    draws -- a Wesnoth draw is a timeout, not equality evidence (see
    _win_and_game_matrices)."""
    W, N = _win_and_game_matrices(n, pairs, prior_games, draw_weight)
    gamma = fit_bradley_terry(W, N)
    elo = _ELO_PER_LN * np.log(gamma)
    elo += anchor_elo - elo[anchor_idx]
    se = elo_standard_errors(gamma, N, anchor_idx)
    return elo, se


# =====================================================================
# Player construction
# =====================================================================

class _ScriptedAdapter:
    """Wrap a stateless scripted policy (e.g. DummyPolicy) so it
    satisfies the (select_action / drop_pending / reset_game) contract
    `_play_one_eval_game` expects. The no-ops are safe because the
    scripted policy holds no per-game state."""
    trainable = False

    def __init__(self, inner):
        self._inner = inner

    def select_action(self, gs, *, game_label="default", sim=None):
        return self._inner.select_action(gs, game_label=game_label, sim=sim)

    def drop_pending(self, game_label):  # no per-game queue to clear
        pass

    def reset_game(self, game_label):
        pass


@dataclass
class Player:
    label: str
    spec:  str    # "random" | "dummy" | <ckpt path> |
                  # "mcts:<sims>:<ckpt path|random>" (search-vs-prior
                  # strength tests, 2026-07-16 -- the null the
                  # overturn metric lacked). Sims comes BEFORE the
                  # path because Windows paths contain colons.
    policy: object = None            # built lazily on the device

    def build(self, device) -> None:
        if self.spec == "random":
            self.policy = _load_policy(None, device, label=self.label)
        elif self.spec == "dummy":
            from wesnoth_ai.dummy_policy import DummyPolicy
            self.policy = _ScriptedAdapter(DummyPolicy())
        elif self.spec.startswith("mcts:"):
            _, sims_s, inner = self.spec.split(":", 2)
            ckpt = None if inner == "random" else Path(inner)
            base = _load_policy(ckpt, device, label=self.label)
            from tools.mcts import MCTSConfig
            from tools.mcts_policy import MCTSPolicy
            # Eval contract: MCTSConfig defaults keep the training
            # crutches OFF (aux_value_bonus=0.0, draw_tiebreak=None)
            # -- pure search over the checkpoint's own heads.
            # Root advice is NOT a crutch: it is learned model
            # conditioning (a gated module in the checkpoint), so an
            # advice-trained checkpoint plays as trained -- advice ON
            # at the root exactly when the checkpoint carries the
            # path. For others the flag is inert either way.
            advice = bool(peek_checkpoint_arch(
                ckpt, self.label).get("advice", False))
            self.policy = MCTSPolicy(
                base, mcts_config=MCTSConfig(
                    n_simulations=int(sims_s), advice=advice))
        else:
            self.policy = _load_policy(Path(self.spec), device,
                                       label=self.label)


def _parse_players(args) -> List[Player]:
    players: List[Player] = []
    seen = set()

    def _add(label: str, spec: str):
        if label in seen:
            raise SystemExit(f"duplicate player label: {label!r}")
        seen.add(label)
        players.append(Player(label=label, spec=spec))

    for entry in (args.player or []):
        if "=" not in entry:
            raise SystemExit(
                f"--player must be LABEL=SPEC (got {entry!r}); SPEC is a "
                f"checkpoint path, 'random', 'dummy', or "
                f"'mcts:<sims>:<ckpt|random>'.")
        label, spec = entry.split("=", 1)
        _add(label.strip(), spec.strip())

    for pat in (args.checkpoints or []):
        for p in sorted(Path().glob(pat)) or [Path(pat)]:
            if p.exists():
                _add(p.stem, str(p))
    if args.include_random and "random" not in seen:
        _add("random", "random")
    if args.include_dummy and "dummy" not in seen:
        _add("dummy", "dummy")
    return players


# =====================================================================
# Round-robin
# =====================================================================

@dataclass
class LadderResult:
    labels:  List[str]
    elo:     List[float]
    se:      List[float]
    record:  List[Dict[str, int]]            # per-player W/L/D totals
    pairs:   Dict[str, Dict[str, int]]       # "i__vs__j" -> wins/draws
    anchor:  str
    n_games: int
    wall_seconds: float = 0.0
    max_turns: int = 0                       # eval horizon, so no report
                                             # can hide a train/eval
                                             # horizon mismatch
    mirror:  Dict[str, Dict[str, int]] = field(default_factory=dict)
                                             # "i__vs__j" -> mirror-pair
                                             # tallies (sweeps/splits)


@dataclass
class MirrorStats:
    """Per-pair tallies over mirrored setup pairs (same setup, sides
    swapped). Under H0 (equal strength) sweeps_i and sweeps_j are
    exchangeable, so the sweep counts feed a paired sign test that is
    more powerful than the pooled win rate when setups are lopsided
    (a map/faction draw that decides the winner produces a 1-1 split,
    which the sign test correctly treats as no evidence)."""
    sweeps_i: int = 0     # pi won both orientations of a setup
    sweeps_j: int = 0     # pj won both orientations
    splits:   int = 0     # 1-1, each side won once (setup-decided)
    mixed:    int = 0     # at least one draw/timeout/skip in the pair


def _play_single_game(
    pi: Player, pj: Player, pi_side: int, setup, max_turns: int,
    game_label: str,
) -> Optional[str]:
    """Build a fresh sim from `setup` and play one game with pi as
    pair_a on `pi_side`. Returns the outcome from pi's perspective
    ('win'/'loss'/'draw'/'timeout'/'errored'), or None if the game
    could not be built/completed (not counted)."""
    try:
        gs = build_scenario_gamestate(setup)
        sim = WesnothSim(gs, scenario_id=setup.scenario_id,
                         max_turns=max_turns)
    except Exception as e:
        log.warning(f"skip {setup.label()}: {e}")
        return None
    for pl in (pi, pj):
        if hasattr(pl.policy, "reset_game"):
            pl.policy.reset_game(game_label)
    pair_i = _PolicyPair(policy=pi.policy, label=pi.label, side=pi_side)
    pair_j = _PolicyPair(policy=pj.policy, label=pj.label,
                         side=(3 - pi_side))
    try:
        r = _play_one_eval_game(sim, pair_i, pair_j,
                                game_label=game_label)
    except Exception as e:
        log.exception(f"game crashed ({pi.label} v {pj.label}): {e}")
        pi.policy.drop_pending(game_label)
        pj.policy.drop_pending(game_label)
        return None
    return r.outcome


def _count_outcome(rec: PairRecord, outcome: Optional[str]) -> None:
    """Fold one pi-perspective outcome into the pair record.
    None / 'errored' are dropped (not counted)."""
    if outcome == "win":
        rec.wins_i += 1
    elif outcome == "loss":
        rec.wins_j += 1
    elif outcome in ("draw", "timeout"):
        rec.draws += 1


def _play_pair(
    pi: Player, pj: Player, games: int, rng: random.Random,
    max_turns: int, forced_faction, progress_prefix: str,
) -> Tuple[PairRecord, MirrorStats]:
    """Play `games` between two players as MIRRORED SETUP PAIRS: each
    sampled setup (map + factions + leaders + ToD slot) is played
    TWICE, once with pi on side 1 and once with pi on side 2, so
    map/faction/side advantages cancel within the pair instead of
    adding noise (Wesnoth side bias is real, and the forced-faction
    sampler makes matchup composition lumpy at small n). The mirror
    game is NOT a deterministic replay: policies sample their actions
    (Gumbel-max over the softmax), so paired games explore different
    lines from an identical start. An odd trailing game plays an
    unpaired fresh setup with pi on side 2 (matching the old code's
    side split for the remainder).

    pi is always pair_a, so outcomes are read from pi's perspective.

    NOTE (seed compatibility): pairing halves the number of
    `random_setup` draws per pair, so a given --seed produces a
    different setup sequence than pre-2026-07-29 unpaired runs.
    """
    rec = PairRecord()
    mir = MirrorStats()
    n_pairs = games // 2
    for g in range(n_pairs):
        setup = random_setup(rng, forced_faction=forced_faction)
        out_1 = _play_single_game(
            pi, pj, 1, setup, max_turns,
            game_label=f"elo_{pi.label}_{pj.label}_p{g}a")
        out_2 = _play_single_game(
            pi, pj, 2, setup, max_turns,
            game_label=f"elo_{pi.label}_{pj.label}_p{g}b")
        _count_outcome(rec, out_1)
        _count_outcome(rec, out_2)
        if out_1 == "win" and out_2 == "win":
            mir.sweeps_i += 1
        elif out_1 == "loss" and out_2 == "loss":
            mir.sweeps_j += 1
        elif {out_1, out_2} == {"win", "loss"}:
            mir.splits += 1
        else:
            mir.mixed += 1
    if games % 2:
        setup = random_setup(rng, forced_faction=forced_faction)
        _count_outcome(rec, _play_single_game(
            pi, pj, 2, setup, max_turns,
            game_label=f"elo_{pi.label}_{pj.label}_odd"))
    sys.stderr.write(
        f"  {progress_prefix} {pi.label} vs {pj.label}: "
        f"{rec.wins_i}-{rec.draws}-{rec.wins_j} (W-D-L)  "
        f"mirror: {mir.sweeps_i} swept-by-{pi.label} / {mir.splits} "
        f"split / {mir.sweeps_j} swept-by-{pj.label} / "
        f"{mir.mixed} mixed\n")
    return rec, mir


def run_ladder(
    players: List[Player], *, games_per_pair: int, max_turns: int,
    seed: int, forced_faction, anchor_label: Optional[str],
    anchor_elo: float, prior_games: float, draw_weight: float = 0.0,
) -> LadderResult:
    n = len(players)
    if n < 2:
        raise SystemExit("need >= 2 players for a ladder")
    rng = random.Random(seed)
    pair_recs: Dict[Tuple[int, int], PairRecord] = {}
    mirror_recs: Dict[Tuple[int, int], MirrorStats] = {}
    n_pairs = n * (n - 1) // 2
    t0 = time.perf_counter()
    done = 0
    for i, j in combinations(range(n), 2):
        done += 1
        rec, mir = _play_pair(players[i], players[j], games_per_pair, rng,
                              max_turns, forced_faction,
                              progress_prefix=f"[pair {done}/{n_pairs}]")
        pair_recs[(i, j)] = rec
        mirror_recs[(i, j)] = mir
    wall = time.perf_counter() - t0

    # Anchor: explicit label, else "random" if present, else player 0.
    if anchor_label is not None:
        labels = [p.label for p in players]
        if anchor_label not in labels:
            raise SystemExit(f"anchor {anchor_label!r} not among players")
        anchor_idx = labels.index(anchor_label)
    else:
        anchor_idx = next((k for k, p in enumerate(players)
                           if p.spec == "random"), 0)

    elo, se = fit_elo(n, pair_recs, anchor_idx, anchor_elo, prior_games,
                      draw_weight)

    # Per-player W/L/D totals and pairwise dict for the report.
    record = [{"win": 0, "loss": 0, "draw": 0} for _ in range(n)]
    pairs_out: Dict[str, Dict[str, int]] = {}
    mirror_out: Dict[str, Dict[str, int]] = {}
    for (i, j), rec in pair_recs.items():
        record[i]["win"] += rec.wins_i
        record[i]["loss"] += rec.wins_j
        record[i]["draw"] += rec.draws
        record[j]["win"] += rec.wins_j
        record[j]["loss"] += rec.wins_i
        record[j]["draw"] += rec.draws
        key = f"{players[i].label}__vs__{players[j].label}"
        pairs_out[key] = {
            "wins": rec.wins_i, "draws": rec.draws, "losses": rec.wins_j}
        mir = mirror_recs[(i, j)]
        mirror_out[key] = {
            "sweeps_i": mir.sweeps_i, "splits": mir.splits,
            "sweeps_j": mir.sweeps_j, "mixed": mir.mixed}

    return LadderResult(
        labels=[p.label for p in players],
        elo=[float(x) for x in elo], se=[float(x) for x in se],
        record=record, pairs=pairs_out,
        anchor=players[anchor_idx].label,
        n_games=sum(r.games for r in pair_recs.values()),
        wall_seconds=wall,
        max_turns=max_turns,
        mirror=mirror_out,
    )


# =====================================================================
# Reporting
# =====================================================================

def print_ladder(res: LadderResult) -> None:
    order = sorted(range(len(res.labels)), key=lambda k: res.elo[k],
                   reverse=True)
    print()
    print("=" * 72)
    print(f"Internal Elo ladder  (anchor {res.anchor!r} = "
          f"{res.elo[res.labels.index(res.anchor)]:.0f}; "
          f"{res.n_games} games, max_turns {res.max_turns}, "
          f"{res.wall_seconds:.0f}s)")
    print("=" * 72)
    print(f"{'rank':>4}  {'player':<22} {'Elo':>7}  {'+/-95%':>7}  "
          f"{'W-D-L':>12}")
    print("-" * 72)
    for rank, k in enumerate(order, 1):
        rc = res.record[k]
        ci = 1.96 * res.se[k]
        wdl = f"{rc['win']}-{rc['draw']}-{rc['loss']}"
        print(f"{rank:>4}  {res.labels[k]:<22} {res.elo[k]:>7.0f}  "
              f"{ci:>7.0f}  {wdl:>12}")
    print("-" * 72)
    # Mirror-pair summary: sweeps are the paired evidence of a
    # strength gap; splits are setup-decided games (no evidence).
    for key, m in res.mirror.items():
        a, b = key.split("__vs__")
        print(f"  {a} vs {b}: {m['sweeps_i']} swept by {a}, "
              f"{m['splits']} split, {m['sweeps_j']} swept by {b}, "
              f"{m['mixed']} with draws")
    print("=" * 72)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--player", action="append", default=[],
                    help="LABEL=SPEC, repeatable. SPEC is a checkpoint "
                         "path, 'random' (random-init baseline), or "
                         "'dummy' (scripted floor).")
    ap.add_argument("--checkpoints", nargs="+", default=[],
                    help="Glob(s) of checkpoints to add as players "
                         "(labeled by filename stem).")
    ap.add_argument("--include-random", action="store_true",
                    help="Add the random-init baseline as a player.")
    ap.add_argument("--include-dummy", action="store_true",
                    help="Add the scripted DummyPolicy as a player.")
    ap.add_argument("--games-per-pair", type=int, default=30,
                    help="Games for each unordered player pair, played "
                         "as mirrored setup pairs (each sampled setup "
                         "is played once per side by each player).")
    ap.add_argument("--max-turns", type=int, default=100,
                    help="Eval horizon. Default 100 sits inside the "
                         "training band (training rolls each game's "
                         "horizon in [--max-turns-min 60, MAX_TURNS]; "
                         "the 2026-07 box sets 100). Evaluating BELOW "
                         "the training minimum (e.g. the 30 used in "
                         "the 2026-07-28/29 evals) truncates games the "
                         "policies were trained to finish and measures "
                         "a different game.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--forced-faction", default=None,
                    help="Lock both sides to a faction (or 'none' to "
                         "force random); default = pool randomization.")
    ap.add_argument("--anchor", default=None,
                    help="Player LABEL pinned to --anchor-elo (gauge "
                         "fix). Default: 'random' if present, else the "
                         "first player.")
    ap.add_argument("--anchor-elo", type=float, default=0.0)
    ap.add_argument("--prior-games", type=float, default=1.0,
                    help="Ghost games per pair (50/50 split) "
                         "regularizing toward equality, so winless/"
                         "undefeated players stay finite.")
    ap.add_argument("--draw-weight", type=float, default=0.0,
                    help="Fraction of a draw credited as a win to each "
                         "side. Default 0.0 DROPS draws: a Wesnoth draw "
                         "is a turn-budget timeout, not equality "
                         "evidence. 0.5 = textbook half-win, and is "
                         "what tools/elo_collect.py calls PURE -- the "
                         "two tools DIFFER by default, so never quote "
                         "one as the other (the header and saved JSON "
                         "carry draw_weight for exactly this reason).")
    ap.add_argument("--save-json", type=Path, default=None)
    ap.add_argument("--log-level", default="WARNING",
                    choices=["DEBUG", "INFO", "WARNING"])
    args = ap.parse_args(argv[1:])

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S")

    players = _parse_players(args)
    if len(players) < 2:
        log.error("need >= 2 players (use --player / --checkpoints / "
                  "--include-random / --include-dummy)")
        return 1

    device = select_inference_device(args.device)
    log.info(f"device: {describe_device(device)}")
    _ = _recruit_cost_lookup()  # surface a missing unit_stats.json early
    for p in players:
        p.build(device)
        log.info(f"player {p.label!r} <- {p.spec}")

    forced_faction = ...
    if args.forced_faction is not None:
        forced_faction = (None if args.forced_faction.lower() == "none"
                          else args.forced_faction)

    res = run_ladder(
        players, games_per_pair=args.games_per_pair,
        max_turns=args.max_turns, seed=args.seed,
        forced_faction=forced_faction, anchor_label=args.anchor,
        anchor_elo=args.anchor_elo, prior_games=args.prior_games,
        draw_weight=args.draw_weight)

    print_ladder(res)

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "players": [
                {"label": res.labels[k], "elo": res.elo[k],
                 "se": res.se[k], "ci95": 1.96 * res.se[k],
                 "record": res.record[k]}
                for k in range(len(res.labels))],
            "anchor": res.anchor, "anchor_elo": args.anchor_elo,
            "prior_games": args.prior_games,
            "pairs": res.pairs, "mirror": res.mirror,
            "n_games": res.n_games,
            "games_per_pair": args.games_per_pair,
            # Protocol knobs that change what the Elo MEANS -- recorded
            # so no saved result can be quoted without them.
            "max_turns": res.max_turns, "seed": args.seed,
            "draw_weight": args.draw_weight,
            "wall_seconds": res.wall_seconds, "backend": "sim",
        }
        args.save_json.write_text(json.dumps(payload, indent=2, default=str),
                                  encoding="utf-8")
        log.info(f"ladder written to {args.save_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
