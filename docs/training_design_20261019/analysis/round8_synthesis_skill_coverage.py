"""Round 8 synthesis: where the rating input can sit at the top of the human axis without naming one player.

The challenge critique (round8_challenge_critique_checks.out section 1) found that at s_top = +253 the
training sides within 30 Elo come from 46 players, the largest supplying 0.605 of them. This script
reads, for candidate top settings c, the make-up of the training sides whose OUT-OF-FOLD rating lies
near c, among rated sides only (players with >= 30 games; the repaired design feeds other sides an
"unrated" input). Out-of-fold ratings come from the critique's own 5-fold MAP Bradley-Terry refit
(same seed, same prior), so no side's input has seen its own result.

Windows: +-30 Elo (the critique's) and +-52 Elo (the median posterior sd of regulars, the jitter scale).
Rule [C] for the repaired s_top: the highest c at which, in the +-52 window, at least 10 distinct
players supply the sides and none supplies more than 0.2 of them.

Inputs (read-only): training/metrics/player_ratings/ratings.json, players.jsonl,
replays_dataset_imitation/manifest.jsonl. numpy and the standard library only.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import round8_challenge_critique_checks as C  # noqa: E402

MIN_GAMES = 30
MAX_SHARE, MIN_PLAYERS = 0.2, 10


def rated_sides(players, games):
    ids = sorted({g["w"] for g in games} | {g["l"] for g in games})
    idx = {k: i for i, k in enumerate(ids)}
    pairs = np.array([(idx[g["w"]], idx[g["l"]]) for g in games])
    fold = np.random.default_rng(7).integers(0, 5, len(games))
    oof = np.zeros((len(games), 2))
    for k in range(5):
        th = C.fit_bt(pairs[fold != k], len(ids)) * C.ELO
        m = fold == k
        oof[m, 0], oof[m, 1] = th[pairs[m, 0]], th[pairs[m, 1]]
    s, pid, won, n_all = [], [], [], 0
    for gi, g in enumerate(games):
        if g["holdout"]:
            continue
        for j, p, w in ((0, g["w"], 1), (1, g["l"], 0)):
            if p == "[ai]":
                continue
            n_all += 1
            if players[p]["games"] >= MIN_GAMES:
                s.append(oof[gi, j])
                pid.append(p)
                won.append(w)
    return np.array(s), np.array(pid), np.array(won, bool), n_all


def make_up(s, pid, won, c, half):
    m = np.abs(s - c) <= half
    if m.sum() == 0:
        return None
    _, cnt = np.unique(pid[m], return_counts=True)
    cnt = np.sort(cnt)[::-1]
    return m.sum(), len(cnt), cnt[0] / m.sum(), cnt[:3].sum() / m.sum(), won[m].mean()


def main():
    players, games = C.load_games()
    s, pid, won, n_all = rated_sides(players, games)
    print(f"human training sides {n_all}; rated (player >= {MIN_GAMES} games) {len(s)} ({len(s) / n_all:.3f})")
    q = np.percentile(s, [5, 25, 50, 75, 90, 95])
    print("rated sides' out-of-fold rating, 5/25/50/75/90/95 pct: " + " ".join(f"{v:+.0f}" for v in q))
    print(f"rated sides' win rate {won.mean():.3f}; corr(out-of-fold s, won) {np.corrcoef(s, won)[0, 1]:.3f}")
    chosen = None
    for half in (30.0, 52.0):
        print(f"\nwindow +-{half:.0f} Elo: centre | sides | players | largest share | top-3 share | win rate")
        for c in range(0, 321, 20):
            r = make_up(s, pid, won, float(c), half)
            if r is None:
                continue
            n, k, top1, top3, wr = r
            ok = k >= MIN_PLAYERS and top1 <= MAX_SHARE
            print(f"  {c:+4d} | {n:6d} | {k:4d} | {top1:.3f} | {top3:.3f} | {wr:.3f}{'  ok' if ok else ''}")
            if half == 52.0 and ok:
                chosen = c
    print(f"\nrule: highest centre with >= {MIN_PLAYERS} players and largest share <= {MAX_SHARE} in the +-52 window: "
          f"s_top = {chosen:+d}")
    n, k, top1, top3, wr = make_up(s, pid, won, float(chosen), 52.0)
    print(f"  at s_top: {n} sides from {k} players, largest {top1:.3f}, top-3 {top3:.3f}, win rate {wr:.3f}")
    mid = float(np.median(s))
    n, k, top1, top3, wr = make_up(s, pid, won, mid, 52.0)
    print(f"  rated median {mid:+.0f}: {n} sides from {k} players, largest {top1:.3f}, top-3 {top3:.3f}, win rate {wr:.3f}")
    print(f"  setting gap s_top - median: {chosen - mid:.0f} Elo; human expected score at that gap "
          f"{1 / (1 + 10 ** (-(chosen - mid) / 400)):.3f}")


if __name__ == "__main__":
    main()
