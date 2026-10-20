"""Round 5 (provenance stance): the mean advantage of a decision switched toward a class of
actions, from the decode-knob search's whole game records.

The 24 knob vectors of training/metrics/decode_knobs_20261012/ (end_turn, attack and recruit
logit offsets and an attack time-of-day slope, each vector against own1 at its own decode)
change which action the argmax picks at the decisions where the offset crosses the top-two
margin. Each changed decision is a deviation from own1's choice; by the performance-difference
lemma the match result's mean shift is the sum, over the deviating player's decisions, of each
deviation's advantage under own1's play. Per vector the script counts the deviating side's extra
attacks, recruits and other acts per side-game (side A minus side B in the same games, B being
own1 at its decode), and fits, across vectors, z_v on the extra decisions
(weighted by games, no intercept: kv00 is own1 against itself).

Attack and recruit offsets change WHICH action is taken far more than how many, so counts do not
identify their deviations; only the decision count (driven by the end_turn offset) does. The fit
is therefore z_v = a_minus * min(extra_v, 0) + a_plus * max(extra_v, 0) + b * knobs_v, with the
attack, recruit and time-of-day knobs as covariates: a_minus is the mean advantage of a decision
the decode stops making when it ends turns earlier, a_plus of one it adds when it ends them later.

Reverse causation (a side that wins has more units, so more acts) is measured on own1's
self-play records (gen_own1_selfplay, own64_selfplay: A minus B counts regressed on z per game)
and removed from each vector's counts before the fit.

Decision counting as round5_provenance_stop_acts.py: a move immediately followed by an attack
from its last hex counts once (as the attack). Decisive games only. numpy and the standard
library only.
"""
import gzip
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

METRICS = Path(__file__).resolve().parents[3] / "training/metrics"
KNOBS = METRICS / "decode_knobs_20261012"
CONTROLS = {
    "gen_own1_selfplay": ("memory_own_round2_20261011/games_gen_own1_selfplay.tar.gz", "own1a"),
    "own64_selfplay": ("own_confirm_20261011/games_own64_selfplay.tar.gz", "own64a"),
}
RNG = np.random.default_rng(20261020)
CLASSES = ("attack", "recruit", "move")


def side_counts(commands):
    """{side: {class: decisions}} plus side-turn counts, from a record's command list."""
    counts, turns, side = {}, {}, None
    for i, c in enumerate(commands):
        kind = c[0]
        if kind == "init_side":
            side = c[1]
            turns[side] = turns.get(side, 0) + 1
            counts.setdefault(side, dict.fromkeys(CLASSES, 0))
        elif kind in ("attack", "recruit"):
            counts[side][kind] += 1
        elif kind == "move":
            nxt = commands[i + 1] if i + 1 < len(commands) else None
            last = (c[1][-1], c[2][-1])
            if not (nxt and nxt[0] == "attack" and (nxt[1], nxt[2]) == last):
                counts[side]["move"] += 1
    return counts, turns


def read_games(tarball, label_a=None):
    """Per decisive game: z for side A and A-minus-B counts per class. Side A is the player
    whose label is label_a, or the player that is not own1 when label_a is None."""
    out = []
    with tarfile.open(tarball) as t:
        for member in t.getmembers():
            if not member.name.endswith(".game.jsonl.gz"):
                continue
            rec = json.loads(gzip.decompress(t.extractfile(member).read()).decode().splitlines()[0])
            if str(rec.get("winner")) not in ("1", "2"):
                continue
            players = rec["players"]
            if isinstance(players, str):
                import ast
                players = ast.literal_eval(players)
            commands = rec["commands"]
            if label_a is None:
                a = players["a"] if players["a"]["label"] != "own1" else players["b"]
            else:
                a = players["a"] if players["a"]["label"] == label_a else players["b"]
            b = players["b"] if a is players["a"] else players["a"]
            counts, _turns = side_counts(commands)
            ca = counts.get(a["side"], dict.fromkeys(CLASSES, 0))
            cb = counts.get(b["side"], dict.fromkeys(CLASSES, 0))
            out.append((1.0 if int(rec["winner"]) == a["side"] else -1.0,
                        [ca[k] - cb[k] for k in CLASSES], a.get("procedure", "")))
    return out


def reverse_slope():
    """Per class, the A-minus-B count per unit of z in self-play (both sides the same player)."""
    z, d = [], []
    for name, (tb, label) in CONTROLS.items():
        for zi, di, _ in read_games(METRICS / tb, label):
            z.append(zi)
            d.append(di)
    z, d = np.array(z), np.array(d, float)
    slope = (z[:, None] * (d - d.mean(0))).mean(0) / z.var()
    se = d.std(0) / np.sqrt(len(z)) / z.std()
    return slope, se, len(z)


def knob_values():
    vec = [json.loads(line) for line in open(KNOBS / "search.jsonl") if '"vectors"' in line][0]["vectors"]
    return {v["label"]: (v["raw_end_turn_offset"], v["raw_attack_offset"], v["raw_recruit_offset"],
                         v["raw_attack_tod_slope"]) for v in vec}


def wls(X, y, w):
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    cov = np.linalg.inv((X * w[:, None]).T @ X)          # binomial noise: var(z_v) = 1 / n_v
    return coef, np.sqrt(np.diag(cov))


def main():
    slope, slope_se, n_ctrl = reverse_slope()
    total_slope = slope.sum()
    print(f"reverse causation, self-play controls ({n_ctrl} decisive games): A-minus-B decisions per "
          f"side-game per unit z: " + ", ".join(f"{k} {s:+.2f} +- {e:.2f}" for k, s, e in zip(CLASSES, slope, slope_se))
          + f"; all decisions {total_slope:+.1f}")
    knobs = knob_values()
    rows = []
    for v in range(24):
        label = f"kv{v:02d}"
        games = read_games(KNOBS / f"games_{label}.tar.gz")
        z = np.array([g[0] for g in games])
        d = np.array([sum(g[1]) for g in games], float)
        rows.append((label, len(z), z.mean(), d.mean(), d.std() / np.sqrt(len(z))) + knobs[label])
    print("\nvector, decisive games, z, extra decisions per side-game (raw, then without the winning share), knobs e a r t:")
    for r in rows:
        print(f"  {r[0]} {r[1]:5d} {r[2]:+.3f} {r[3]:+6.1f} {r[3] - total_slope * r[2]:+6.1f}   "
              f"e {r[5]:+.2f} a {r[6]:+.2f} r {r[7]:+.2f} t {r[8]:.2f}")
    n = np.array([r[1] for r in rows], float)
    z = np.array([r[2] for r in rows])
    extra = np.array([r[3] for r in rows]) - total_slope * z
    e, a, rr, t = (np.array([r[k] for r in rows]) for k in (5, 6, 7, 8))
    fewer, more = np.minimum(extra, 0.0), np.maximum(extra, 0.0)
    # (1) first stage: what the end_turn offset does to the decision count (the other knobs as covariates)
    X1 = np.column_stack([np.ones(24), e + 1.5, a, rr, t])
    c1, _ = wls(X1, extra, n)
    r1 = extra - X1 @ c1
    sigma2 = float(np.sum(n * r1 ** 2) / (24 - X1.shape[1]))     # per-game count variance, from residuals
    s1 = np.sqrt(np.diag(np.linalg.inv((X1 * n[:, None]).T @ X1)) * sigma2)
    print("\nfirst stage, extra decisions per side-game on (1, e + 1.5, a, r, t): "
          + ", ".join(f"{c:+.2f} +- {s:.2f}" for c, s in zip(c1, s1)))
    # (2) the result on acts removed and acts added, the other knobs as covariates, no intercept
    #     (a vector with no extra decisions and zero offsets is own1 against itself)
    X2 = np.column_stack([fewer, more, a, rr, t])
    c2, s2 = wls(X2, z, n)
    resid = z - X2 @ c2
    disp = float(np.sum(n * resid ** 2) / (24 - X2.shape[1]))
    names = ("per decision removed", "per decision added", "attack offset", "recruit offset", "tod slope")
    print("result fit, z on (min(extra, 0), max(extra, 0), a, r, t), weights n, binomial SEs:")
    for nm, c, s in zip(names, c2, s2):
        print(f"  {nm:22s} {c:+.5f} +- {s:.5f}")
    print(f"  residual dispersion {disp:.2f} on {24 - X2.shape[1]} df (1 = binomial noise only)")
    # (3) Match A predicted: own1 at offset 0 against own1 at -1.5, the other knobs at 0
    d0 = c1[0] + c1[1] * 1.5
    zA = c2[0] * min(d0, 0.0) + c2[1] * max(d0, 0.0)
    g = np.array([min(d0, 0.0), max(d0, 0.0), 0, 0, 0])
    cov = np.linalg.inv((X2 * n[:, None]).T @ X2)
    se = float(np.sqrt(g @ cov @ g))
    print(f"\nMatch A predicted (offset 0 against -1.5, other knobs 0): extra decisions {d0:+.1f} per side-game, "
          f"z {zA:+.3f} +- {se:.3f} (decision-count uncertainty not propagated), Elo {347.4 * zA:+.0f} +- {347.4 * se:.0f}")


if __name__ == "__main__":
    sys.exit(main())
