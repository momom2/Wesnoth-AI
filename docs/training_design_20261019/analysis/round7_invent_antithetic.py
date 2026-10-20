"""Round 7, invent stance: antithetic chance siblings (ACS).

A luck term needs, at fight k, a price h of the post-fight state with an exactly known mean under
the fight's law. Enumerating every outcome child costs ~12 network forwards per fight
(round5_synthesis_outcomes.out). ACS evaluates ONE extra child: the fight replayed on a fork with
every strike's draw reflected (r -> R - 1 - r, so u -> 1 - u). The realized child's price comes
free from the next decision's forward. L_k = (h(o) - h(o')) / 2 has mean zero exactly, because the
reflected stream has the same law as the real one. Against a full enumeration it keeps the share
kappa = (1 - rho) / 2 of the price's variance, rho = Corr(h(o), h(o')).

Part A (derived, stated fight model): kappa for strike-level reflection on random fights drawn
from a grid of default-era-like parameters. Strikes alternate attacker, defender, attacker, ...;
a unit with more strikes continues alone; the fight stops at a death. One uniform per strike slot;
hit iff u < c; the reflected fight hits iff 1 - u < c. Per slot the pair (hit, reflected hit) has
P(1,1) = max(0, 2c - 1), P(1,0) = P(0,1) = min(c, 1 - c), P(0,0) = max(0, 1 - 2c). Exact DP over
the pair of fights. Prices (side A's view): hp margin; kill difference; hp-fraction margin plus a
half-unit kill bonus. Harris's inequality predicts rho <= 0 for prices monotone in each strike.

Part B (measured, records): the concentration of a proxy price's variance across real fights, which
decides how many fights ACS must evaluate. Input: the enumerated outcome tables of own1 self-play's
fights whose counter weapon was a choice (training/metrics/memory_own_round2_20261011/
games_gen_own1_selfplay.tar.gz, outcomes[...]["counter_weapon"], the chosen weapon's table; rows
[att_hp, def_hp, six status flags, att_damaged, def_damaged, p]; pre-fight hp is the hp of the
undamaged rows). That subsample is not random (the defender had a choice of weapon).

Part C (derived): cost multiple and break-even lines. numpy and the standard library only.
"""
import gzip
import json
import tarfile
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TARBALL = ROOT / "training/metrics/memory_own_round2_20261011/games_gen_own1_selfplay.tar.gz"
RNG = np.random.default_rng(20261020)


# ---------------------------------------------------------------- Part A
def slot_order(n_att, n_def):
    order = []
    for i in range(max(n_att, n_def)):
        if i < n_att:
            order.append("a")
        if i < n_def:
            order.append("d")
    return tuple(order)


def pair_law(c):
    both = max(0.0, 2 * c - 1)
    one = min(c, 1 - c)
    return ((1, 1, both), (1, 0, one), (0, 1, one), (0, 0, max(0.0, 1 - 2 * c)))


def joint_outcomes(f):
    """Exact joint law of (real outcome, reflected outcome) as {((ha, hd), (ha', hd')): p}."""
    ha0, hd0, da, dd, na, nd, ca, cd = f
    order = slot_order(na, nd)

    @lru_cache(maxsize=None)
    def go(j, ha, hd, ha2, hd2):
        if j == len(order):
            return {((ha, hd), (ha2, hd2)): 1.0}
        attacker_strikes = order[j] == "a"
        c = ca if attacker_strikes else cd
        out = {}
        for h, h2, p in pair_law(c):
            if p == 0.0:
                continue
            a, d, a2, d2 = ha, hd, ha2, hd2
            live1 = a > 0 and d > 0
            live2 = a2 > 0 and d2 > 0
            if live1 and h:
                if attacker_strikes:
                    d = max(0, d - da)
                else:
                    a = max(0, a - dd)
            if live2 and h2:
                if attacker_strikes:
                    d2 = max(0, d2 - da)
                else:
                    a2 = max(0, a2 - dd)
            for k, q in go(j + 1, a, d, a2, d2).items():
                out[k] = out.get(k, 0.0) + p * q
        return out

    return go(0, ha0, hd0, ha0, hd0)


def prices(o, f):
    ha0, hd0 = f[0], f[1]
    ha, hd = o
    hp = ha - hd
    kill = float(hd == 0) - float(ha == 0)
    frac = ha / ha0 - hd / hd0 + 0.5 * kill
    return np.array([hp, kill, frac])


PRICE_NAMES = ("hp margin", "kill difference", "hp fraction + 0.5 kill")


def random_fight():
    da, dd = RNG.choice([4, 5, 6, 7, 8, 9, 11, 14], size=2)
    na, nd = RNG.choice([1, 2, 3, 4, 5], size=2, p=[0.05, 0.4, 0.35, 0.15, 0.05])
    ca, cd = RNG.choice([0.3, 0.4, 0.5, 0.6, 0.7], size=2)
    ha0 = int(RNG.choice([33, 38, 45]) if RNG.random() < 0.5 else RNG.integers(4, 25))
    hd0 = int(RNG.choice([33, 38, 45]) if RNG.random() < 0.5 else RNG.integers(4, 25))
    return (ha0, hd0, int(da), int(dd), int(na), int(nd), float(ca), float(cd))


def part_a(n_fights=1500):
    var_full = np.zeros(3)
    var_kept = np.zeros(3)
    rhos = [[], [], []]
    for _ in range(n_fights):
        f = random_fight()
        law = joint_outcomes(f)
        ps = np.array(list(law.values()))
        g1 = np.array([prices(k[0], f) for k in law])
        g2 = np.array([prices(k[1], f) for k in law])
        m = ps @ g1
        e1, e2 = g1 - m, g2 - m
        v = ps @ (e1 ** 2)
        cov = ps @ (e1 * e2)
        for i in range(3):
            if v[i] < 1e-12:
                continue
            rho = cov[i] / v[i]
            rhos[i].append(rho)
            var_full[i] += v[i]
            var_kept[i] += v[i] * (1 - rho) / 2
    print(f"Part A: {n_fights} random fights, strike-level reflection")
    for i, name in enumerate(PRICE_NAMES):
        r = np.array(rhos[i])
        print(f"  {name:24s} kappa (variance-weighted) {var_kept[i] / var_full[i]:.3f}; "
              f"rho quantiles 50/90/max {np.percentile(r, 50):+.3f} {np.percentile(r, 90):+.3f} "
              f"{r.max():+.3f}; share rho <= 1e-9: {np.mean(r <= 1e-9):.3f}")
    print("  closed form, one strike, additive price: kappa(c) = (1 - rho)/2, rho = (max(0,2c-1) - c^2)/(c(1-c))")
    for c in (0.3, 0.4, 0.5, 0.6, 0.7):
        rho = (max(0.0, 2 * c - 1) - c * c) / (c * (1 - c))
        print(f"    c = {c}: rho {rho:+.3f}, kappa {(1 - rho) / 2:.3f}")
    return var_kept[2] / var_full[2]


# ---------------------------------------------------------------- Part B
def read_tables():
    tables = []
    with tarfile.open(TARBALL) as t:
        for m in t.getmembers():
            if not m.name.endswith(".game.jsonl.gz"):
                continue
            rec = json.loads(gzip.decompress(t.extractfile(m).read()).decode().splitlines()[0])
            for entry in rec.get("outcomes", {}).values():
                cw = entry.get("counter_weapon")
                if not cw:
                    continue
                tab = cw["tables"].get(str(cw["chosen"]))
                if tab:
                    tables.append(tab)
    return tables


def table_variance(tab):
    rows = np.array([[r[0], r[1], float(r[8]), float(r[9]), r[-1]] for r in tab], float)
    ha, hd, a_dmg, d_dmg, p = rows.T
    p = p / p.sum()
    ha0 = ha[a_dmg == 0].max() if np.any(a_dmg == 0) else ha.max()
    hd0 = hd[d_dmg == 0].max() if np.any(d_dmg == 0) else hd.max()
    kill = (hd == 0).astype(float) - (ha == 0).astype(float)
    g = np.stack([ha - hd, kill, ha / max(ha0, 1) - hd / max(hd0, 1) + 0.5 * kill])
    mean = g @ p
    return ((g - mean[:, None]) ** 2) @ p, kill


def part_b():
    tables = read_tables()
    v = np.array([table_variance(t)[0] for t in tables])
    out = None
    print(f"\nPart B: {len(tables)} recorded fight laws (own1 self-play, counter-weapon subsample); "
          f"support mean {np.mean([len(t) for t in tables]):.2f}")
    for i, name in enumerate(PRICE_NAMES):
        s = np.sort(v[:, i])[::-1]
        tot = s.sum()
        shares = [s[: int(np.ceil(q * len(s)))].sum() / tot for q in (0.1, 0.3, 0.5)]
        print(f"  {name:24s} share of the summed variance in the top 10/30/50% of fights: "
              f"{shares[0]:.3f} {shares[1]:.3f} {shares[2]:.3f}")
        if i == 2:
            out = [(0.3, shares[1]), (0.5, shares[2]), (1.0, 1.0)]
    return out


# ---------------------------------------------------------------- Part C
def part_c():
    fights, decisions = 109.7, 449.1           # own1 self-play per game, round5_synthesis_outcomes.out
    support = 11.86                            # same file, counter-weapon subsample
    r_ref = 0.688                              # setup cell + seven columns, current_design.md
    print("\nPart C: cost multiple m and the kept ratio r a corrector must reach to carry more label "
          "information per dollar than the cell + columns (r * m < 0.688)")
    m_full = 1 + fights * support / decisions
    print(f"  full enumeration: m {m_full:.2f}, line r < {r_ref / m_full:.3f}, "
          f"so the price must explain more than {r_ref - r_ref / m_full:.3f} of Var(z) beyond the columns")
    for f_sel in (1.0, 0.5, 0.3):
        m = 1 + f_sel * fights / decisions
        line = r_ref / m
        print(f"  ACS on a share {f_sel:.1f} of fights: m {m:.3f}, line r < {line:.3f}; "
              f"needed explained share E beyond the columns at kappa 0.5 / 0.75 / 1.0: "
              f"{(r_ref - line) / 0.5:.3f} {(r_ref - line) / 0.75:.3f} {(r_ref - line):.3f}")
    print("  a batch saved per round (current design's DPI line, r <= 0.602 at m = 1): "
          f"ACS on all fights needs r <= {0.602 / (1 + fights / decisions):.3f}")
    return fights, decisions, r_ref


def part_d(kappa, shares):
    """Selection by a pre-fight cutoff on the proxy price's variance under the law: evaluating the
    share f of fights keeps the share s(f) of the proxy's summed variance (Part B, mixed proxy),
    so the effective capture is kappa x s(f) (kappa from Part A, mixed price)."""
    fights, decisions, r_ref = 109.7, 449.1, 0.688
    print(f"\nPart D: effective capture with selection (kappa {kappa:.3f} from Part A, shares from Part B)")
    for f_sel, s in shares:
        m = 1 + f_sel * fights / decisions
        line = r_ref / m
        k_eff = kappa * s
        print(f"  evaluate {f_sel:.1f} of fights: m {m:.3f}, effective capture {k_eff:.3f}, "
              f"pays iff the full-enumeration price explains more than {(r_ref - line) / k_eff:.3f} of Var(z) "
              f"beyond the columns; saves a batch per round iff it explains more than "
              f"{(r_ref - 0.602 / m) / k_eff:.3f}")


def part_e():
    """S0 cost of ACS logging on the luck walk's games, and of a full enumeration on own1 self-play.
    Inputs: training/metrics/match_luck_20261019/rows.jsonl (per game: decisive, counts.attacks,
    turns); replay speed 2,440 forwards a second and $0.42 an hour (current_design.md S0(a))."""
    rows = [json.loads(line) for line in open(ROOT / "training/metrics/match_luck_20261019/rows.jsonl")]
    dec = [r for r in rows if r.get("decisive")]
    attacks = np.array([r["counts"]["attacks"] for r in dec], float)
    turns = np.array([r["turns"] for r in dec], float)
    fwd_per_s, dollars_per_h = 2440.0, 0.42
    acs = attacks.sum() / fwd_per_s / 3600 * dollars_per_h
    enum = 800 * 109.7 * 11.86 / fwd_per_s / 3600 * dollars_per_h
    print(f"\nPart E: walked decisive games {len(dec)}, attacks per game {attacks.mean():.1f}, turns {turns.mean():.2f}")
    print(f"  ACS: one reflected-child forward per fight on all of them: {attacks.sum() / 1e6:.2f} M forwards, ${acs:.3f}")
    print(f"  full enumeration on own1 self-play (800 games x 109.7 fights x 11.86 children): ${enum:.3f}")


if __name__ == "__main__":
    kappa = part_a()
    shares = part_b()
    part_c()
    part_d(kappa, shares)
    part_e()
