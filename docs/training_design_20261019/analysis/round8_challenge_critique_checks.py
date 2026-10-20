"""Round 8, critique of the challenge stance (the skill axis): checks of its inputs on existing records.

Inputs (read-only):
  training/metrics/player_ratings/ratings.json, players.jsonl     the Bradley-Terry fit the stance conditions on
  replays_dataset_imitation/manifest.jsonl and its *.json.gz      human games (command streams, factions, scenario)
  training/metrics/own_confirm_20261011/games_own64_selfplay.tar.gz  own1 at 64 slots, raw:t0+eo-1.5, self-play

Sections:
 1. Outcome leakage in the conditioning input s. The stance feeds each side its in-sample posterior-mean rating
    (ratings.json, fitted on every game, the side's own game included). A 5-fold out-of-fold refit (same MAP model,
    same prior sd 200 Elo, the stance's own fit_bt) gives each side a rating that has not seen its own result.
    Reported per player class: E[s | side won] - E[s | side lost], in-sample against out-of-fold; and the make-up of
    the training sides near s_mid = +84 and s_top = +253 (+-30 Elo): share from players with 1 game and with < 30
    games, their win rate, how many distinct players supply them, and the share of the three largest.
 2. Tempo: own1 against humans, own side-turns 2-4, actions (move, attack, recruit commands) per side-turn, counted
    exactly as round8_challenge_skill.py section 4. Humans on the scenarios own1's pool plays, default factions only;
    human regulars split at the stance's quartile line (+147). Faction-matched: each human tier's per-faction mean
    weighted by own1's faction mix.
 3. The within-game tempo gap with the two sides' factions controlled: (higher-rated minus lower-rated) regressed on
    an intercept and the difference of the two sides' faction indicators (one faction dropped); the intercept is the
    gap at equal factions. 2,000 games from the stance's section-4 pool with its seed 20261020; a different draw
    from the stance's, whose generator first served its section 3.
 4. Derived checks of the stance's decision rules (no data): S3's acceptance of a 2.8-Elo deficit under the rule as
    written ("not below its parent by 2 SE") and under "estimate >= 0"; the selection bias of the best of three equal
    arms at SE 3.07; the Elo of own1 at end_turn offset -2.5 against own1 (round5_provenance_stop_acts.out:
    z -0.0350 +- 0.0353 for +19.4 decisions per game); the contrast at which a tail candidate overtakes the top one
    under guidance.
numpy and the standard library only.
"""
import gzip
import io
import json
import math
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RAT = ROOT / "training/metrics/player_ratings"
DATA = ROOT / "replays_dataset_imitation"
OWN = ROOT / "training/metrics/own_confirm_20261011/games_own64_selfplay.tar.gz"
PRIOR_SD = 200.0
ELO = 400.0 / math.log(10.0)
MIN_GAMES = 30
S_MID, S_TOP, HALF = 84.0, 253.0, 30.0
TOP_LINE = 147.0
EARLY = (2, 3, 4)
FACTIONS = ("Drakes", "Knalgan Alliance", "Loyalists", "Northerners", "Rebels", "Undead")


def fit_bt(pairs, n_players, iters=400, lr=0.5):
    """The stance's MAP Bradley-Terry (round8_challenge_skill.py), natural units."""
    theta = np.zeros(n_players)
    win, lose = pairs[:, 0], pairs[:, 1]
    prior_prec = (ELO / PRIOR_SD) ** 2
    counts = np.bincount(np.concatenate([win, lose]), minlength=n_players).astype(float) + prior_prec
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(theta[win] - theta[lose])))
        g = np.bincount(win, weights=1 - p, minlength=n_players) - np.bincount(lose, weights=1 - p, minlength=n_players)
        g -= prior_prec * theta
        theta += lr * g / (0.25 * counts)
    return theta


def load_games():
    players = {p["id"]: p for p in json.load(open(RAT / "ratings.json"))["players"]}
    sides = {}
    for line in open(RAT / "players.jsonl"):
        row = json.loads(line)
        sides[row["file"]] = row["sides"]
    games = []
    for line in open(DATA / "manifest.jsonl"):
        row = json.loads(line)
        s = sides.get(row["file"])
        if not s or "1" not in s or "2" not in s or s["1"] == s["2"]:
            continue
        w = s[str(row["winner_side"])]
        lo = s["2" if row["winner_side"] == 1 else "1"]
        games.append(dict(file=row["file"], w=w, l=lo, holdout=row["holdout"], sides=s))
    return players, games


def section1(players, games):
    ids = sorted({g["w"] for g in games} | {g["l"] for g in games})
    idx = {k: i for i, k in enumerate(ids)}
    pairs = np.array([(idx[g["w"]], idx[g["l"]]) for g in games])
    fold = np.random.default_rng(7).integers(0, 5, len(games))
    oof = np.zeros((len(games), 2))                       # out-of-fold rating of winner, loser (Elo)
    for k in range(5):
        th = fit_bt(pairs[fold != k], len(ids)) * ELO
        m = fold == k
        oof[m, 0], oof[m, 1] = th[pairs[m, 0]], th[pairs[m, 1]]
    rows, pids = [], []                                   # (s_in, s_oof, won, n_games) per human training side
    for gi, g in enumerate(games):
        if g["holdout"]:
            continue
        for j, pid, won in ((0, g["w"], 1), (1, g["l"], 0)):
            if pid == "[ai]":
                continue
            p = players[pid]
            rows.append((p["rating_elo"], oof[gi, j], won, p["games"]))
            pids.append(pid)
    r = np.array(rows)
    pids = np.array(pids)
    s_in, s_oof, won, ng = r[:, 0], r[:, 1], r[:, 2].astype(bool), r[:, 3]
    print("1. outcome leakage in s (human training sides, non-holdout games)")
    print(f"  sides {len(r)}")
    for name, m in (("1 game", ng == 1), ("2-4 games", (ng >= 2) & (ng <= 4)), ("5-29 games", (ng >= 5) & (ng < 30)),
                    (">= 30 games", ng >= 30), ("all", ng > 0)):
        d_in = s_in[m & won].mean() - s_in[m & ~won].mean()
        d_oof = s_oof[m & won].mean() - s_oof[m & ~won].mean()
        print(f"  {name:>11}: share of sides {m.mean():.3f}; E[s|won] - E[s|lost] in-sample {d_in:+7.1f} Elo, "
              f"out-of-fold {d_oof:+7.1f} Elo")
    print(f"  corr(s, won): in-sample {np.corrcoef(s_in, won)[0, 1]:.3f}, out-of-fold {np.corrcoef(s_oof, won)[0, 1]:.3f}")
    for name, c in (("s_mid +84", S_MID), ("s_top +253", S_TOP)):
        m = np.abs(s_in - c) <= HALF
        print(f"  sides with in-sample s within {HALF:.0f} of {name}: {m.sum()}; from 1-game players {np.mean(ng[m] == 1):.3f}, "
              f"from < {MIN_GAMES}-game players {np.mean(ng[m] < MIN_GAMES):.3f}; won {won[m].mean():.3f}; "
              f"their out-of-fold s mean {s_oof[m].mean():+.1f}")
        _, cnt = np.unique(pids[m], return_counts=True)
        cnt = np.sort(cnt)[::-1]
        print(f"    distinct players {len(cnt)}; share of these sides from the largest one {cnt[0] / m.sum():.3f}, "
              f"the largest three {cnt[:3].sum() / m.sum():.3f}")
    one = ng == 1
    print(f"  1-game players: in-sample s if won {s_in[one & won].mean():+.1f} (sd {s_in[one & won].std():.1f}), "
          f"if lost {s_in[one & ~won].mean():+.1f}")


def early_tempo(commands):
    """{side: actions per own side-turn over own side-turns 2-4}, as round8_challenge_skill.side_tempo."""
    turns, early, side = {1: 0, 2: 0}, {1: 0, 2: 0}, None
    for cmd in commands:
        if cmd[0] == "init_side":
            side = int(cmd[1])
            if side in turns:
                turns[side] += 1
        elif cmd[0] in ("move", "attack", "recruit") and side in turns and turns[side] in EARLY:
            early[side] += 1
    return {k: early[k] / len(EARLY) for k in (1, 2) if turns[k] >= max(EARLY)}


def own1_tempo():
    out, scen = [], set()
    with tarfile.open(OWN) as tf:
        for mem in tf:
            if not mem.name.endswith(".game.jsonl.gz"):
                continue
            d = json.loads(gzip.open(io.BytesIO(tf.extractfile(mem).read())).read())
            scen.add(d["setup"]["scenario_id"])
            t = early_tempo(d["commands"])
            for k in t:
                out.append((d["setup"][f"faction{k}"], t[k]))
    return out, scen


def section2(players, games, scen_pool):
    own, scen = scen_pool
    rng = np.random.default_rng(20261021)
    cand = [g for g in games if (players.get(g["w"], {}).get("games", 0) >= MIN_GAMES
                                 or players.get(g["l"], {}).get("games", 0) >= MIN_GAMES)]
    pick = rng.choice(len(cand), size=min(3000, len(cand)), replace=False)
    hum = []                                              # (tier, faction, tempo)
    for i in pick:
        g = cand[i]
        d = json.load(gzip.open(DATA / g["file"]))
        if d["scenario_id"] not in scen:
            continue
        fac = {s["side"]: s["faction"] for s in d["starting_sides"]}
        t = early_tempo(d["commands"])
        for k in t:
            pid = g["sides"][str(k)]
            p = players.get(pid)
            if pid == "[ai]" or p is None or p["games"] < MIN_GAMES or fac.get(k) not in FACTIONS:
                continue
            hum.append(("top" if p["rating_elo"] >= TOP_LINE else "rest", fac[k], t[k]))
    own_f = {f: [t for ff, t in own if ff == f] for f in FACTIONS}
    w = {f: len(own_f[f]) / len(own) for f in FACTIONS}
    print("2. tempo at own side-turns 2-4 (actions per side-turn), own1 self-play against human regulars")
    print(f"  own1 side-games {len(own)} on {len(scen)} scenarios: mean {np.mean([t for _, t in own]):.2f}")
    for tier in ("top", "rest"):
        per = {f: [t for tt, ff, t in hum if tt == tier and ff == f] for f in FACTIONS}
        n = sum(len(v) for v in per.values())
        matched = sum(w[f] * np.mean(per[f]) for f in FACTIONS if per[f])
        print(f"  human regulars {tier:>4} (>= {TOP_LINE:.0f} or below): side-games {n}, raw mean "
              f"{np.mean([t for tt, _, t in hum if tt == tier]):.2f}, faction-matched to own1 {matched:.2f}")
    print("  per faction: own1 | top | rest")
    for f in FACTIONS:
        top = [t for tt, ff, t in hum if tt == "top" and ff == f]
        rest = [t for tt, ff, t in hum if tt == "rest" and ff == f]
        print(f"    {f:>16}: {np.mean(own_f[f]):.2f} (n {len(own_f[f])}) | {np.mean(top):.2f} (n {len(top)}) | "
              f"{np.mean(rest):.2f} (n {len(rest)})")


def section3(players, games):
    rng = np.random.default_rng(20261020)                 # the stance's seed and pool order
    pool = [g for g in games if players.get(g["w"], {}).get("games", 0) >= MIN_GAMES
            and players.get(g["l"], {}).get("games", 0) >= MIN_GAMES and "[ai]" not in (g["w"], g["l"])
            and players[g["w"]]["rating_elo"] != players[g["l"]]["rating_elo"]]
    pick = rng.choice(len(pool), size=min(2000, len(pool)), replace=False)
    y, X = [], []
    for i in pick:
        g = pool[i]
        d = json.load(gzip.open(DATA / g["file"]))
        fac = {s["side"]: s["faction"] for s in d["starting_sides"]}
        t = early_tempo(d["commands"])
        if 1 not in t or 2 not in t or fac.get(1) not in FACTIONS or fac.get(2) not in FACTIONS:
            continue
        s = g["sides"]
        hi_side = 1 if players[s["1"]]["rating_elo"] > players[s["2"]]["rating_elo"] else 2
        lo_side = 3 - hi_side
        row = [1.0] + [float(fac[hi_side] == f) - float(fac[lo_side] == f) for f in FACTIONS[1:]]
        y.append(t[hi_side] - t[lo_side])
        X.append(row)
    y, X = np.array(y), np.array(X)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    cov = np.linalg.inv(X.T @ X) * res.var(ddof=X.shape[1])
    print("3. within-game tempo gap at own side-turns 2-4, the two sides' factions controlled")
    print(f"  games {len(y)} (default factions on both sides); raw gap {y.mean():+.3f} +- {y.std(ddof=1) / math.sqrt(len(y)):.3f}")
    print(f"  gap at equal factions (intercept) {beta[0]:+.3f} +- {math.sqrt(cov[0, 0]):.3f}")
    print("  faction effects on a side's tempo, relative to Drakes: "
          + ", ".join(f"{f} {b:+.2f}" for f, b in zip(FACTIONS[1:], beta[1:])))


def section4():
    se = 800.0 / math.log(10.0) / math.sqrt(12800)
    phi = lambda x: 0.5 * math.erfc(-x / math.sqrt(2))
    print("4. derived checks of decision rules")
    print(f"  S3, SE {se:.2f}: a 2.8-Elo deficit is accepted with P {phi((2 * se - 2.8) / se):.3f} under 'not below by 2 SE', "
          f"P {phi(-2.8 / se):.3f} under 'estimate >= 0'; under the first rule a {2 * se:.1f}-Elo deficit passes half the time")
    z = np.random.default_rng(1).standard_normal((1_000_000, 3)).max(axis=1)
    print(f"  best of three equal arms: E[max] {z.mean():.3f} SE = {z.mean() * se:.2f} Elo of selection bias in the claim")
    print(f"  own1 at offset -2.5 against own1: {800 / math.log(10) * -0.0350:+.1f} +- {800 / math.log(10) * 0.0353:.1f} Elo")
    for g in (1.0, 2.0):
        print(f"  gamma {g:.0f}: a candidate at -3 nats under N(s_top) beats a zero-contrast top candidate once its contrast "
              f"l_top - l_mid exceeds {3.0 / g:.2f} nats")


def main():
    section4()
    players, games = load_games()
    section1(players, games)
    section2(players, games, own1_tempo())
    section3(players, games)


if __name__ == "__main__":
    main()
