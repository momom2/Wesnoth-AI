"""Round 11 (provenance stance): the census a learning-from-sampled-play regimen is priced on.

Reads, read-only:
  training/metrics/own2_base_20261020/games_own2_vs_own1.tar.gz   (S0(b): own2 v own1, both at the
      reference decode raw:t0+eo-1.5, 64 slots; every game recorded whole)
  training/metrics/near_tie_coins_20261020/coins_b00.jsonl, b06   (own1 coin self-play: the gap G
      between the top two joint log-priors at every band decision, G < 0.35)

Reports:
  1. decisions per game and per side-game at the reference decode, by kind (a decision is any command
     but init_side), turns per game, side-turns per side-game, decisions per side-turn;
  2. from the coin gaps, a lower bound on how often a temperature-1 player leaves the argmax on own1's
     argmax trajectories: at a band decision with gap G the argmax's probability is at most
     1 / (1 + exp(-G)) (the top two alone), so P(leave argmax) >= exp(-G) / (1 + exp(-G)); summed per
     side-game. Decisions outside the band are counted as 0 (a lower bound).
numpy and the standard library only.
"""
import gzip
import json
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
GAMES = ROOT / "training/metrics/own2_base_20261020/games_own2_vs_own1.tar.gz"
COINS = [ROOT / f"training/metrics/near_tie_coins_20261020/coins_b0{i}.jsonl" for i in (0, 6)]  # the two on disk


def game_census():
    per_game, per_side, side_turns, kinds, turns = [], [], [], Counter(), []
    capped = 0
    with tarfile.open(GAMES) as tar:
        for member in tar:
            if not member.name.endswith(".game.jsonl.gz"):
                continue
            rec = json.loads(gzip.decompress(tar.extractfile(member).read()).splitlines()[0])
            if rec.get("ended_by") != "leader_killed":
                capped += 1
            side = None
            counts = Counter()
            ends = Counter()
            for cmd in rec["commands"]:
                if cmd[0] == "init_side":
                    side = cmd[1]
                    continue
                counts[side] += 1
                kinds[cmd[0]] += 1
                if cmd[0] == "end_turn":
                    ends[side] += 1
            per_game.append(sum(counts.values()))
            for s in (1, 2):
                per_side.append(counts[s])
                side_turns.append(ends[s])
            turns.append(rec["turns"])
    return np.array(per_game), np.array(per_side), np.array(side_turns), kinds, np.array(turns), capped


def coin_bound():
    per_side_game = []
    band = []
    for path in COINS:
        with open(path) as fh:
            for line in fh:
                row = json.loads(line)
                if not row.get("decisive"):
                    continue
                sides = np.array(row["band"]["side"])
                gaps = np.array(row["band"]["gap"], dtype=float)
                leave = np.exp(-gaps) / (1.0 + np.exp(-gaps))
                for s in (1, 2):
                    per_side_game.append(leave[sides == s].sum())
                    band.append(int((sides == s).sum()))
    return np.array(per_side_game), np.array(band)


def main():
    per_game, per_side, side_turns, kinds, turns, capped = game_census()
    n = len(per_game)
    print(f"1. own2 v own1 at raw:t0+eo-1.5: games {n}, not ended by a leader kill {capped}")
    print(f"   decisions per game mean {per_game.mean():.1f} (SE {per_game.std(ddof=1) / np.sqrt(n):.1f}),"
          f" quartiles {np.percentile(per_game, [25, 50, 75]).tolist()}")
    print(f"   decisions per side-game mean {per_side.mean():.1f}; side-turns per side-game mean"
          f" {side_turns.mean():.2f}; decisions per side-turn {per_side.sum() / max(side_turns.sum(), 1):.2f}")
    print(f"   turns per game mean {turns.mean():.2f}, quartiles {np.percentile(turns, [25, 50, 75]).tolist()}")
    total = sum(kinds.values())
    print("   decisions by kind: " + ", ".join(f"{k} {v / total:.3f}" for k, v in kinds.most_common()))
    leave, band = coin_bound()
    m = len(leave)
    print(f"2. own1 coin self-play, side-games {m}: band decisions per side-game {band.mean():.1f}")
    print(f"   lower bound on argmax departures per side-game at temperature 1 (band only):"
          f" mean {leave.mean():.1f} (SE {leave.std(ddof=1) / np.sqrt(m):.2f}),"
          f" quartiles {np.percentile(leave, [25, 50, 75]).round(1).tolist()}")
    print(f"   as a share of the side-game's decisions (census of section 1): {leave.mean() / per_side.mean():.3f}")


if __name__ == "__main__":
    main()
