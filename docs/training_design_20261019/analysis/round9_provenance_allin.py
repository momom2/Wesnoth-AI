"""Round 9, provenance stance: the all-in price of a coin batch, and the plan's
budget at it. round9_provenance_budget.py priced games at the matches' own
walls; the box also bills the work after each match (the luck walk, coin rows,
plant rows, readout), which the label needs. Standard library only.

Measured inputs:
- training/metrics/near_tie_coins_20261020/walls.txt: every stage's wall on box
  56955777 ($0.577 an hour), 2026-10-20/21.
- training/metrics/own2_base_20261020/walls.txt: the two-network box's stages.
- round9_provenance_budget.py's measured rates (one network 22,341 games an
  hour for a 10,000-game match, start-up included; two networks 6,523).
- the distillation, about 4.8 box-hours (round9_provenance_budget.py).
Choices [C]: the stage list and sizes of round9_provenance.md.
"""
import re
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COIN_WALLS = ROOT / "training" / "metrics" / "near_tie_coins_20261020" / "walls.txt"
OWN2_WALLS = ROOT / "training" / "metrics" / "own2_base_20261020" / "walls.txt"
ONE_RATE, TWO_RATE = 22_341, 6_523
LEFT = 85.0 - 2.66
DISTILL_H, SETUP_H = 2.1 / 0.44, 0.1
BATCH, BATCHES, SMOKE = 100_000, 8, 2_000
LINK, MATCH, TWO_NET = 12_800, 37_800, 12_800
GAMES = {"b01": 10_000, "b02": 10_000, "b03": 10_000, "b04": 10_000, "b05": 10_000, "b06": 8_000}


def stages(path):
    out = []
    for line in path.read_text().splitlines():
        m = re.match(r"(\S+)Z (.+?) rc=\d+ ok (\d+) s", line)
        if m:
            out.append((datetime.fromisoformat(m.group(1)), m.group(2), int(m.group(3))))
    return out


def post_share():
    rows = stages(COIN_WALLS)
    match_s = post_s = 0
    current = None
    for _t, name, sec in rows:
        if name.startswith("match "):
            current = name.split()[1]
            if current in GAMES:
                match_s += sec
        elif current in GAMES and name.split()[0] in ("walk", "coin", "plant", "readout", "gpu"):
            post_s += sec
    games = sum(GAMES.values())
    return match_s, post_s, games


def box_span():
    rows = stages(COIN_WALLS)
    start = datetime.fromisoformat("2026-10-20T21:42:39")   # bring-up start (round9_provenance_budget.py)
    end = rows[-1][0]
    return (end - start).total_seconds()


def plan(rate, post, feature_overhead=0.0):
    p1, p2 = rate / ONE_RATE, rate / TWO_RATE
    batch = (1 + feature_overhead) * (1 + post) * p1
    one, two = (lambda n: n * p1), (lambda n: n * p2)
    rnd = (SMOKE + BATCHES * BATCH) * batch
    s = {
        "L + T (+ C if T adopts)": SETUP_H * rate + one(2 * LINK) + two(TWO_NET),
        "S1": SETUP_H * rate + rnd,
        "S2": one(MATCH) + two(TWO_NET),
        "E reserve": 2 * one(MATCH) + two(TWO_NET),
        "S3": SETUP_H * rate + rnd + one(MATCH) + two(TWO_NET),
        "S4": DISTILL_H * rate + two(TWO_NET),
        "S5": SETUP_H * rate + rnd + 2 * two(TWO_NET),
    }
    return s


def ceiling(post, h):
    lo, hi = 0.05, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sum(plan(mid, post, h).values()) <= LEFT else (lo, mid)
    return lo


def main():
    match_s, post_s, games = post_share()
    post = post_s / match_s
    span = box_span()
    print("1. post-match work on the coin box (b01-b06: walk, coin rows, plant rows, readout, gpu check)")
    print(f"  matches {match_s} s, post {post_s} s for {games} games: post adds {post:.3f} of match wall "
          f"({post_s / games * 1000:.1f} s per 1,000 games)")
    print(f"  whole box: {span / 3600:.2f} h for 66,400 decisive games = {66_400 / span * 3600:,.0f} games/h all-in, "
          f"${0.577 * span / 3600 / 66_400:.3e} a game at $0.577/h (record: about $2.0)")
    two = stages(OWN2_WALLS)
    two_span = sum(sec for _t, _n, sec in two)
    print(f"  two-network box, staged work only: {two_span} s for 6,400 decisive games = "
          f"{6_400 / two_span * 3600:,.0f} games/h (match alone {TWO_RATE:,})")
    print("\n2. the plan with the post-match work on every coin batch")
    for rate in (0.42, 0.577):
        s = plan(rate, post)
        total = sum(s.values())
        print(f"  at ${rate:.3f}/h: " + "; ".join(f"{k} ${v:.2f}" for k, v in s.items()))
        print(f"    total ${total:.2f} of ${LEFT:.2f} (left ${LEFT - total:+.2f}); without S5 ${total - s['S5']:.2f}; "
              f"a round's coin batches ${s['S1']:.2f}; a null round 1 after L and T "
              f"${s['L + T (+ C if T adopts)'] - TWO_NET * rate / TWO_RATE + s['S1'] + MATCH * rate / ONE_RATE:.2f}")
    print("\n3. the hourly rate at which the whole path fits, by the feature dump's overhead h (unmeasured)")
    for h in (0.0, 0.05, 0.10, 0.20, 0.30):
        print(f"  h {h:.2f}: rate <= ${ceiling(post, h):.3f}/h")
    print(f"  without the post-match work (round9_provenance_budget.py's model): ${ceiling(0.0, 0.0):.3f}/h")


if __name__ == "__main__":
    main()
