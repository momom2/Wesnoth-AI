"""Round 5 (provenance stance): what the end_turn offset changes in own1's and parity3's play,
and the mean advantage per extra act it implies.

Reads whole game records (tools/game_record.py format, commands only) from the match tarballs
under training/metrics/, and per game counts each side's decisions per side-turn:
recruits + attacks + moves not immediately followed by an attack from the move's last hex
(a move-then-attack counts once). Pairs each match's result (side A's mean z, decisive games)
with A's extra decisions per side-game over B's, against a control match where both sides play
the same decode, and reports

    a = (z_A - z_control) / (extra decisions per side-game of A),

the mean advantage, in result units (win +1, loss -1), of one extra act in the window the
offset change opens (performance-difference lemma: the result's gain is the sum of the
advantages of the decisions the new decode changes). Standard errors by the game bootstrap.
"""
import gzip
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

METRICS = Path(__file__).resolve().parents[3] / "training/metrics"
RNG = np.random.default_rng(20261020)

MATCHES = {
    # name: (tarball, label of side A's player)
    "own1_eo25_vs_own1": ("rollout_regrade_20261012/games_own1_eo25_vs_own1.tar.gz", "own1_eo25"),
    "gen_own1_selfplay": ("memory_own_round2_20261011/games_gen_own1_selfplay.tar.gz", "own1a"),
    "a1_eo25 (parity3 -2.5 vs slots0 -1.5)": ("memory_in_play_parity3_20261009/games_a1_eo25.tar.gz", "parity3_eo25"),
    "a3_eo35 (parity3 -3.5 vs slots0 -1.5)": ("memory_in_play_parity3_20261009/games_a3_eo35.tar.gz", "parity3_eo35"),
    "slots64_vs_slots0 (parity3 -1.5 vs slots0 -1.5)": ("parity3_baselines_20261008/games_slots64_vs_slots0.tar.gz", "parity3"),
}


def side_turn_decisions(commands):
    """{side: [decisions in each of its side-turns]} from a record's command list."""
    per_side, side, count = {}, None, 0
    for i, c in enumerate(commands):
        kind = c[0]
        if kind == "init_side":
            side, count = c[1], 0
        elif kind == "end_turn":
            per_side.setdefault(side, []).append(count)
        elif kind == "recruit" or kind == "attack":
            count += 1
        elif kind == "move":
            nxt = commands[i + 1] if i + 1 < len(commands) else None
            last = (c[1][-1], c[2][-1])
            if not (nxt and nxt[0] == "attack" and (nxt[1], nxt[2]) == last):
                count += 1
    return per_side


def read_match(tarball, label_a):
    games = []
    with tarfile.open(METRICS / tarball) as t:
        for member in t.getmembers():
            if not member.name.endswith(".game.jsonl.gz"):
                continue
            rec = json.loads(gzip.decompress(t.extractfile(member).read()).decode().splitlines()[0])
            if rec.get("winner") not in (1, 2):
                continue  # capped: not decisive
            players = rec["players"]
            a = players["a"] if players["a"]["label"] == label_a else players["b"]
            b = players["b"] if a is players["a"] else players["a"]
            per_side = side_turn_decisions(rec["commands"])
            da, db = per_side.get(a["side"], []), per_side.get(b["side"], [])
            games.append({
                "z": 1.0 if rec["winner"] == a["side"] else -1.0,
                "acts_a": sum(da), "acts_b": sum(db),
                "turns_a": len(da), "turns_b": len(db),
            })
    return games


def summarise(games):
    z = np.array([g["z"] for g in games])
    extra = np.array([g["acts_a"] - g["acts_b"] for g in games], dtype=float)
    per_turn_a = sum(g["acts_a"] for g in games) / max(1, sum(g["turns_a"] for g in games))
    per_turn_b = sum(g["acts_b"] for g in games) / max(1, sum(g["turns_b"] for g in games))
    side_turns = np.mean([g["turns_a"] for g in games])
    return z, extra, per_turn_a, per_turn_b, side_turns


def main():
    data = {}
    for name, (tarball, label) in MATCHES.items():
        data[name] = summarise(read_match(tarball, label))
        z, extra, pa, pb, st = data[name]
        print(f"{name}: decisive {len(z)}, z_A {z.mean():+.4f} +- {z.std() / np.sqrt(len(z)):.4f}, "
              f"decisions per side-turn A {pa:.2f} B {pb:.2f}, A side-turns per game {st:.1f}, "
              f"A minus B decisions per game {extra.mean():+.1f} +- {extra.std() / np.sqrt(len(z)):.1f}")
    pairs = [("own1_eo25_vs_own1", "gen_own1_selfplay"),
             ("a1_eo25 (parity3 -2.5 vs slots0 -1.5)", "slots64_vs_slots0 (parity3 -1.5 vs slots0 -1.5)"),
             ("a3_eo35 (parity3 -3.5 vs slots0 -1.5)", "a1_eo25 (parity3 -2.5 vs slots0 -1.5)")]
    print("\nmean advantage per extra act (treatment minus control):")
    for treat, ctrl in pairs:
        zt, et, *_ = data[treat]
        zc, ec, *_ = data[ctrl]
        boots = []
        for _ in range(2000):
            it = RNG.integers(0, len(zt), len(zt))
            ic = RNG.integers(0, len(zc), len(zc))
            dz = zt[it].mean() - zc[ic].mean()
            de = et[it].mean() - ec[ic].mean()
            boots.append(dz / de if de > 0 else np.nan)
        dz = zt.mean() - zc.mean()
        de = et.mean() - ec.mean()
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        sd_dz = np.sqrt(zt.var() / len(zt) + zc.var() / len(zc))
        print(f"  {treat} | {ctrl}: dz {dz:+.4f} +- {sd_dz:.4f}, extra acts per side-game {de:+.1f}, "
              f"a = {dz / de:+.5f} (95% bootstrap {lo:+.5f} to {hi:+.5f})")
    symmetric_reading(data)


def symmetric_reading(data, treat="own1_eo25_vs_own1"):
    """own1 against itself has E[z_A] = 0 and E[A minus B acts] = 0 by symmetry, so the
    treatment match alone reads the offset change; the control only adds noise."""
    z, extra, *_ = data[treat]
    a = z.mean() / extra.mean()
    boots = []
    for _ in range(2000):
        i = RNG.integers(0, len(z), len(z))
        boots.append(z[i].mean() / extra[i].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\n{treat} alone (symmetric control): z {z.mean():+.4f} +- {z.std() / np.sqrt(len(z)):.4f}, "
          f"extra acts per side-game {extra.mean():+.1f} +- {extra.std() / np.sqrt(len(z)):.1f}, "
          f"a = {a:+.5f} (95% bootstrap {lo:+.5f} to {hi:+.5f})")


if __name__ == "__main__":
    sys.exit(main())
