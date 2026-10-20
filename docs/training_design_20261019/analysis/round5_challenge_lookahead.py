"""Round 5 (challenge stance): what the two look-ahead matches on record say per changed decision,
and what a look-ahead match can resolve.

Inputs (read-only):
  training/metrics/lookahead_material_gate_20261009/games_material.tar.gz  (parity3 + one-step
      look-ahead with the HP-margin evaluator, against parity3 raw; 991 games)
  training/metrics/lookahead_exposure_gate_20261012/games_exposure.tar.gz  (own1 + look-ahead with
      material and exposure, against own1 raw; 800 games)
  training/metrics/match_luck_20261019/rows.jsonl  (per-game luck columns and setup, 23 matches)
  training/metrics/critic_step1_20261009/{T25,T50,T100,O100}.summary.json  (the learned critics)

Reads per game the look-ahead side's telemetry (decisions, flips by kind: a flip is a decision where
the look-ahead's choice differs from the raw argmax at the same state) and the result, joins it with
the luck walk's row, and reports:

  1. per match: decisive games, capped share, mean z of the look-ahead side (raw and corrected),
     flips per side-game by kind, decisions per side-game;
  2. the mean advantage per flip a = E[z] / E[flips per side-game]. Performance-difference lemma:
     against a fixed opponent, J(pi') - J(pi) = E_pi'[sum over pi''s decisions of A^pi(h, pi'(h))],
     and A^pi(h, pi(h)) = 0, so only flips contribute. Here J(pi) = 0 because the base side is the
     same network at the same decode as the opponent. Ratio-of-means with a game bootstrap.
  3. the corrected z: the critic-free correction of round5_provenance_match_variance.py (setup
     baseline + seven luck columns, fitted on the other 22 matches, each match's mean removed first);
  4. derived: the smallest per-flip advantage a 3,200- or 12,800-game corrected match resolves at
     2 SE, a_min = 2 sqrt(r / n) / m, r = 0.747 (provenance stance, measured);
  5. derived: the information per game of a match on a rule that flips m decisions per side-game
     against coins on n_c decisions per game at p(1-p): ratio m^2 / (n_c p(1-p));
  6. the learned critics' holdout curve (best value loss, where in the epoch it was reached).

numpy and the standard library only; no network, no simulator.
Run: python round5_challenge_lookahead.py  (writes round5_challenge_lookahead.out beside it)
"""
import json
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
METRICS = ROOT / "training/metrics"
TARS = {
    "la_material_vs_parity3": METRICS / "lookahead_material_gate_20261009/games_material.tar.gz",
    "la_exposure_vs_own1": METRICS / "lookahead_exposure_gate_20261012/games_exposure.tar.gz",
}
ROWS = METRICS / "match_luck_20261019/rows.jsonl"
CRITICS = METRICS / "critic_step1_20261009"
COLUMNS = ("hp", "material", "material_nonterminal", "terminal", "value_50", "value_100", "value_200")
KINDS = ("attack", "move", "recruit", "end_turn")
R_CORRECTED = 0.747  # round5_provenance_match_variance.out, pooled Var(G)/Var(z), setup + columns
RNG = np.random.default_rng(20261020)
OUT = Path(__file__).with_suffix(".out")
lines = []


def say(text=""):
    print(text)
    lines.append(text)


def read_telemetry(tar_path):
    games = {}
    with tarfile.open(tar_path) as t:
        for m in t.getmembers():
            name = m.name.split("/")[-1]
            if not name.endswith(".json") or name.startswith("."):
                continue
            d = json.loads(t.extractfile(m).read())
            tel = d.get("lookahead_telemetry_a")
            if tel is None:
                continue
            games[name[:-len(".json")]] = {
                "outcome": d["outcome_a"],
                "decisions": tel["decisions"],
                "flips": tel["flips"],
                "flips_by_kind": tel["flips_by_kind"],
                "flips_to_kind": tel["flips_to_kind"],
            }
    return games


def corrected_results(rows):
    """Leave-one-match-out critic-free correction, as round5_provenance_match_variance.py."""
    maps = {m: i for i, m in enumerate(sorted({r["scenario_id"] for r in rows}))}
    facs = {f: i for i, f in enumerate(sorted({r["faction_a"] for r in rows} | {r["faction_b"] for r in rows}))}
    n_ms, n_ff = 2 * len(maps), len(facs) ** 2
    D = np.zeros((len(rows), n_ms + n_ff))
    for i, r in enumerate(rows):
        D[i, maps[r["scenario_id"]] * 2 + (r["side_a"] - 1)] = 1.0
        D[i, n_ms + facs[r["faction_a"]] * len(facs) + facs[r["faction_b"]]] = 1.0
    S = np.array([[r["S"][c] for c in COLUMNS] for r in rows])
    z = np.array([float(r["z"]) for r in rows])
    match = np.array([r["match"] for r in rows])
    zc = z.copy()
    for m in set(match):
        zc[match == m] -= z[match == m].mean()
    G = np.zeros_like(z)
    J = np.hstack([D, S])
    for m in set(match):
        test, train = match == m, match != m
        c = np.linalg.solve(J[train].T @ J[train] + np.eye(J.shape[1]), J[train].T @ zc[train])
        cs, cc = c[:D.shape[1]], c[D.shape[1]:]
        uniform = cs[:n_ms].mean() + cs[n_ms:].mean()
        G[test] = z[test] - (D[test] @ cs - uniform) - S[test] @ cc
    return {r["game"]: g for r, g in zip(rows, G)}


def ratio_boot(num, den, reps=2000):
    idx = RNG.integers(0, len(num), (reps, len(num)))
    vals = num[idx].mean(1) / den[idx].mean(1)
    return float(num.mean() / den.mean()), float(vals.std()), np.percentile(vals, [2.5, 97.5])


def main():
    rows = [json.loads(line) for line in open(ROWS)]
    decisive_rows = [r for r in rows if r.get("decisive")]
    G = corrected_results(decisive_rows)
    say("== 1-3. The look-ahead matches on record, per changed decision")
    for match, tar in TARS.items():
        tel = read_telemetry(tar)
        mrows = [r for r in rows if r["match"] == match]
        capped = sum(1 for r in mrows if not r.get("decisive"))
        dec = [r for r in mrows if r.get("decisive") and r["game"] in tel]
        z = np.array([float(r["z"]) for r in dec])
        g = np.array([G[r["game"]] for r in dec])
        flips = np.array([tel[r["game"]]["flips"] for r in dec], float)
        decisions = np.array([tel[r["game"]]["decisions"] for r in dec], float)
        say(f"\n[{match}] games {len(mrows)}, decisive with telemetry {len(dec)}, capped {capped} "
            f"({capped / len(mrows):.3f})")
        say(f"  decisions per side-game {decisions.mean():.1f}; flips per side-game {flips.mean():.2f} "
            f"(share {flips.sum() / decisions.sum():.4f})")
        by = {k: np.mean([tel[r['game']]['flips_by_kind'][k] for r in dec]) for k in KINDS}
        to = {k: np.mean([tel[r['game']]['flips_to_kind'][k] for r in dec]) for k in KINDS}
        say("  flips from kind per side-game: " + ", ".join(f"{k} {v:.2f}" for k, v in by.items()))
        say("  flips to kind per side-game:   " + ", ".join(f"{k} {v:.2f}" for k, v in to.items()))
        for label, y in (("raw", z), ("corrected", g)):
            se = y.std(ddof=1) / np.sqrt(len(y))
            a, a_se, (lo, hi) = ratio_boot(y, flips)
            say(f"  {label:9s}: mean z {y.mean():+.4f} +- {se:.4f}; Var ratio {y.var() / z.var():.3f}; "
                f"advantage per flip {a:+.5f} +- {a_se:.5f} (95% {lo:+.5f} to {hi:+.5f})")
        n_dec = decisions.mean()
        say(f"  derived bound: under the greedy rule on an exact A^pi the mean advantage per decision is "
            f"at most (1 - J)/N = 1/{n_dec:.0f} = {1 / n_dec:.4f} (J = 0 in this symmetric pairing)")

    say("\n== 4. Resolution of a corrected look-ahead match (derived)")
    for n in (800, 3200, 12800):
        se_z = np.sqrt(R_CORRECTED / n)
        elo = 347.4 * se_z
        cells = ", ".join(f"m={m}: {2 * se_z / m:.4f}" for m in (5, 10, 20, 40))
        say(f"  n = {n:5d} corrected games: SE {se_z:.4f} z ({elo:.1f} Elo); 2-SE per-flip a_min  {cells}")

    say("\n== 5. Information per game: a match on the flipped decisions against coins (derived)")
    n_coins, p1mp = 180, 0.19  # provenance stance: coin-eligible decisions per game, p(1-p)
    for m in (10, 20, 40):
        say(f"  m = {m:2d} flips per side-game, same mean advantage: match / coins = m^2 / (n_c p(1-p)) "
            f"= {m * m / (n_coins * p1mp):.1f}  (n_c = {n_coins}, p(1-p) = {p1mp})")

    say("\n== 6. The learned critics of 2026-10-09 (holdout value loss, C51 cross-entropy)")
    for name in ("T25", "T50", "T100", "O100"):
        s = json.loads((CRITICS / f"{name}.summary.json").read_text())
        st = s["state"]
        reads = [json.loads(line) for line in open(CRITICS / f"{name}.holdout.jsonl")]
        last = reads[-1]["holdout_value"]
        say(f"  {name:5s} train positions {s['train_positions']:6d} (about {s['train_positions'] / 24:.0f} games at "
            f"24 a game); best {st['best']:.4f} at {st['best_positions'] / s['train_positions']:.2f} epochs; "
            f"last read {last:.4f} at {reads[-1]['positions'] / s['train_positions']:.2f} epochs")
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
