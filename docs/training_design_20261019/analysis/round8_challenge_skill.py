"""Round 8, challenge stance: what the corpus's player ratings say about skill, tempo and the imitation corpus.

Inputs (read-only):
  training/metrics/player_ratings/ratings.json   Bradley-Terry fit over 17,074 corpus games (tools/player_ratings.py),
                                                  prior sd 200 Elo, one id per human, "[ai]" for the built-in AI
  training/metrics/player_ratings/players.jsonl  per game file: the ids of sides 1 and 2
  replays_dataset_imitation/manifest.jsonl       per game file: winner_side, n_turns, n_commands, winner_actions, holdout

Sections:
 1. Coverage and spread: regulars (>= 30 games), their rating spread; games with 0, 1, 2 regulars.
 2. Who the winners-only imitation corpus imitates: the winner-action-weighted distribution of the winner's rating.
 3. Out-of-sample skill signal: a Bradley-Terry refit (same prior) on a random 80% of games, scored on the other 20%:
    log loss against ln 2, and the share won by the higher-rated side, by |rating gap|.
 4. Tempo and skill, from the command streams of a seeded sample of 2,000 games between two human regulars
    (replays_dataset_imitation/*.json.gz): each side's actions (move, attack, recruit) per own side-turn (its init_side
    count). Within-game paired difference, higher-rated minus lower-rated regular, which holds map, factions, length
    and the game's luck fixed; split by which side won, and its slope on the rating gap. Repeated over own side-turns
    2-4 only, while both armies are still of similar size (an association: early village income can still differ).
numpy and the standard library only; 10-25 s, most of it reading the 2,000 game files.
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RAT = ROOT / "training/metrics/player_ratings"
MAN = ROOT / "replays_dataset_imitation/manifest.jsonl"
RNG = np.random.default_rng(20261020)
MIN_GAMES = 30
PRIOR_SD = 200.0
ELO = 400.0 / math.log(10.0)


def load():
    ratings = json.load(open(RAT / "ratings.json"))
    players = {p["id"]: p for p in ratings["players"]}
    sides = {}
    for line in open(RAT / "players.jsonl"):
        row = json.loads(line)
        sides[row["file"]] = row["sides"]
    games = []
    for line in open(MAN):
        row = json.loads(line)
        s = sides.get(row["file"])
        if not s or "1" not in s or "2" not in s or s["1"] == s["2"]:
            continue
        w = s[str(row["winner_side"])]
        l_side = "2" if row["winner_side"] == 1 else "1"
        games.append(dict(file=row["file"], w=w, l=s[l_side], n_turns=row["n_turns"],
                          n_cmd=row["n_commands"], w_act=row["winner_actions"], holdout=row["holdout"]))
    return ratings, players, games


def regular(players, pid):
    p = players.get(pid)
    return p is not None and p["games"] >= MIN_GAMES


def section1(players, games):
    reg = [p for p in players.values() if p["games"] >= MIN_GAMES]
    humans = np.array([p["rating_elo"] for p in reg if p["id"] != "[ai]"])
    n_reg = [regular(players, g["w"]) + regular(players, g["l"]) for g in games]
    print("1. coverage and spread")
    print(f"  games joined {len(games)}; regulars (>= {MIN_GAMES} games) {len(reg)}, of which human {len(humans)}")
    print(f"  human regulars' rating: sd {humans.std():.1f} Elo; 5/25/50/75/95 pct "
          f"{np.percentile(humans, [5, 25, 50, 75, 95]).round(0)}")
    ai = players.get("[ai]")
    if ai:
        print(f"  built-in AI: {ai['rating_elo']:.0f} +- {ai['se_elo']:.0f} over {ai['games']} games")
    print(f"  games with 0 / 1 / 2 regulars: {n_reg.count(0)} / {n_reg.count(1)} / {n_reg.count(2)}")


def section2(players, games):
    rw, wt, top = [], [], []
    thr = np.percentile([p["rating_elo"] for p in players.values() if p["games"] >= MIN_GAMES and p["id"] != "[ai]"], 75)
    for g in games:
        if g["holdout"]:
            continue
        p = players.get(g["w"])
        r = p["rating_elo"] if p else 0.0
        rw.append(r)
        wt.append(g["w_act"])
        top.append(p is not None and p["games"] >= MIN_GAMES and p["id"] != "[ai]" and r >= thr)
    rw, wt, top = np.array(rw), np.array(wt, float), np.array(top)
    order = np.argsort(rw)
    cw = np.cumsum(wt[order]) / wt.sum()
    q = [float(rw[order][np.searchsorted(cw, x)]) for x in (0.25, 0.5, 0.75)]
    reg_share = sum(w for g, w in zip([g for g in games if not g["holdout"]], wt) if regular(players, g["w"])) / wt.sum()
    print("2. the winners-only corpus, weighted by winner actions")
    print(f"  winner rating, action-weighted mean {np.average(rw, weights=wt):.1f} Elo; quartiles {np.round(q, 0)}")
    print(f"  share of winner actions by regulars {reg_share:.3f}; by top-quartile human regulars (>= {thr:.0f}) {wt[top].sum() / wt.sum():.3f}")


def fit_bt(pairs, n_players, iters=400, lr=0.5):
    """MAP Bradley-Terry: pairs (winner_idx, loser_idx); ratings in natural units, prior N(0, (PRIOR_SD/ELO)^2)."""
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


def section3(games):
    ids = sorted({g["w"] for g in games} | {g["l"] for g in games})
    idx = {k: i for i, k in enumerate(ids)}
    pairs = np.array([(idx[g["w"]], idx[g["l"]]) for g in games])
    test = RNG.random(len(pairs)) < 0.2
    theta = fit_bt(pairs[~test], len(ids))
    d = theta[pairs[test, 0]] - theta[pairs[test, 1]]          # winner minus loser
    ll = np.mean(np.log1p(np.exp(-d)))
    print("3. out-of-sample skill signal (BT refit on 80% of games, scored on the other 20%)")
    print(f"  test games {test.sum()}: log loss {ll:.4f} against ln 2 = {math.log(2):.4f} (gain {math.log(2) - ll:.4f} nats)")
    gap = np.abs(d) * ELO
    tie = d == 0
    print(f"  exact ties (both players unseen in the fit, rating 0): {tie.sum()} games, left out below")
    for lo, hi in ((0, 50), (50, 100), (100, 200), (200, 400), (400, 2000)):
        m = (gap >= lo) & (gap < hi) & ~tie
        if m.sum():
            print(f"  |gap| {lo:>3}-{hi:<4} Elo: games {m.sum():5d}, higher-rated side won {np.mean(d[m] > 0):.3f}")


EARLY = (2, 3, 4)   # own side-turns 2-4: armies of similar size, before the result shows in unit counts


def side_tempo(path):
    """Per side: actions (move, attack, recruit) per own side-turn over the game; and, over own side-turns 2-4,
    actions per side-turn and moves-plus-attacks per side-turn (recruits left out). From one game's command stream."""
    import gzip
    d = json.load(gzip.open(path))
    acts = {1: 0, 2: 0}
    turns = {1: 0, 2: 0}
    early = {1: 0, 2: 0}
    early_ma = {1: 0, 2: 0}
    side = None
    for cmd in d["commands"]:
        if cmd[0] == "init_side":
            side = int(cmd[1])
            if side in turns:
                turns[side] += 1
        elif cmd[0] in ("move", "attack", "recruit") and side in acts:
            acts[side] += 1
            if turns[side] in EARLY:
                early[side] += 1
                if cmd[0] != "recruit":
                    early_ma[side] += 1
    out = {}
    for k in (1, 2):
        if turns[k] >= max(EARLY):
            out[k] = (acts[k] / turns[k], early[k] / len(EARLY), early_ma[k] / len(EARLY))
    return out


def section4(players, games, n_sample=2000):
    pool = [g for g in games if regular(players, g["w"]) and regular(players, g["l"]) and "[ai]" not in (g["w"], g["l"])
            and players[g["w"]]["rating_elo"] != players[g["l"]]["rating_elo"]]
    pick = RNG.choice(len(pool), size=min(n_sample, len(pool)), replace=False)
    side_of = {}
    for line in open(RAT / "players.jsonl"):
        row = json.loads(line)
        side_of[row["file"]] = row["sides"]
    diff, gap, hi_won, wl, e_diff, ema_diff, e_wl = [], [], [], [], [], [], []
    for i in pick:
        g = pool[i]
        t = side_tempo(ROOT / "replays_dataset_imitation" / g["file"])
        s = side_of[g["file"]]
        if 1 not in t or 2 not in t:
            continue
        a = {s["1"]: t[1], s["2"]: t[2]}
        rw, rl = players[g["w"]]["rating_elo"], players[g["l"]]["rating_elo"]
        hi, lo = (g["w"], g["l"]) if rw > rl else (g["l"], g["w"])
        diff.append(a[hi][0] - a[lo][0])
        gap.append(abs(rw - rl))
        hi_won.append(rw > rl)
        wl.append(a[g["w"]][0] - a[g["l"]][0])
        e_diff.append(a[hi][1] - a[lo][1])
        ema_diff.append(a[hi][2] - a[lo][2])
        e_wl.append(a[g["w"]][1] - a[g["l"]][1])
    dd, gg, hw, wl = np.array(diff), np.array(gap), np.array(hi_won), np.array(wl)
    ed, emd, ewl = np.array(e_diff), np.array(ema_diff), np.array(e_wl)

    def ms(x):
        return f"{x.mean():+.3f} +- {x.std(ddof=1) / math.sqrt(len(x)):.3f} (n {len(x)})"

    print("4. tempo (actions per own side-turn, from the command streams) and skill, games between two human regulars")
    print(f"  pool {len(pool)} games, sample {len(dd)}")
    print(f"  higher-rated minus lower-rated: {ms(dd)}")
    print(f"    the higher-rated side won: {ms(dd[hw])}; it lost: {ms(dd[~hw])}")
    print(f"    the average of the won and lost cases: {0.5 * (dd[hw].mean() + dd[~hw].mean()):+.3f}")
    print(f"  winner minus loser: {ms(wl)}")
    X = np.vstack([np.ones_like(gg), gg / 100.0]).T
    beta = np.linalg.lstsq(X, dd, rcond=None)[0]
    res = dd - X @ beta
    cov = np.linalg.inv(X.T @ X) * res.var(ddof=2)
    print(f"  slope of (higher minus lower) on the rating gap: {beta[1]:+.3f} +- {math.sqrt(cov[1, 1]):.3f} actions per turn per 100 Elo")
    print("  own side-turns 2-4 only (armies of similar size):")
    print(f"    actions per side-turn, higher minus lower: {ms(ed)}; won {ms(ed[hw])}; lost {ms(ed[~hw])}")
    print(f"    moves and attacks only, higher minus lower: {ms(emd)}; won {ms(emd[hw])}; lost {ms(emd[~hw])}")
    print(f"    actions per side-turn, winner minus loser: {ms(ewl)}")
    be = np.linalg.lstsq(X, ed, rcond=None)[0]
    re_ = ed - X @ be
    ce = np.linalg.inv(X.T @ X) * re_.var(ddof=2)
    print(f"    slope on the rating gap: {be[1]:+.3f} +- {math.sqrt(ce[1, 1]):.3f} actions per side-turn per 100 Elo")


def main():
    ratings, players, games = load()
    section1(players, games)
    section2(players, games)
    section3(games)
    section4(players, games)


if __name__ == "__main__":
    main()
