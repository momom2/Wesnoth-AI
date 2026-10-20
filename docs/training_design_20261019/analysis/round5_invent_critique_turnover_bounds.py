"""Round 5, critique of the invent stance: both-sided bounds on where the turn-over rule
(end iff raw P(end) >= 1/2) parts from the -1.5 offset, from the same record the invent
stance read (training/metrics/critic_step1_20261009/prior_gaps.games.jsonl; per decision the
number of legal joint actions, the top-1 minus top-k log-prior gaps for k = 2..8 under the
offset, the top action's type; tools/prior_gaps.py at ea8536f).

Masses are relative to the shifted top action, m_i = exp(-gap_i). Raw end mass = e^1.5 times
its shifted mass; acting masses are unchanged by the offset.

(1) End-top decisions (the offset ends). The turn-over rule continues iff the acting mass
    S = sum over acting actions of m exceeds e^1.5 = 4.48.
      lower bound on S: the top seven (all acting, since end is the top);
      upper bound on S: the top seven plus (legal - 8) * m_7 (every unread action at most
      the eighth's mass).
(2) Act-top decisions (the offset acts). End is some unread or read non-top entry with
    shifted mass m_end <= m_1 (the second). The turn-over rule ends iff
    e^1.5 m_end >= 1 + (acting masses other than the top). Ending is possible only if some
    read entry j satisfies e^1.5 m_j >= 1 + sum_{i != j, i read} m_i (taking the unread tail
    as zero, which favours ending), or end is unread (m_end <= m_7) and
    e^1.5 m_7 >= 1 + sum of read m; otherwise it is impossible.
    The share where it is possible is an upper bound on this direction's disagreement.
numpy and the standard library only.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[3] / "training/metrics/critic_step1_20261009/prior_gaps.games.jsonl"
E = math.exp(1.5)


def main():
    n_sg = 0
    lo, hi, act_possible, act_n = [], [], 0, 0
    for line in open(SRC):
        row = json.loads(line)
        n_sg += 1
        for rd in row["readings"]:
            if rd["legal"] < 2 or not rd["gaps"]:
                continue
            m = np.exp(-np.array(rd["gaps"]))
            if rd["top_type"] == "end_turn":
                s_lo = m.sum()
                tail = max(rd["legal"] - 1 - len(m), 0) * m[-1]
                lo.append(s_lo)
                hi.append(s_lo + tail)
            else:
                act_n += 1
                total = 1.0 + m.sum()
                possible = any(E * mj >= total - mj for mj in m)
                if not possible and E * m[-1] >= total:
                    possible = True  # end could be unread with mass up to m_7
                act_possible += possible
    lo, hi = np.array(lo), np.array(hi)
    n_end = len(lo)
    print(f"side-games {n_sg}; end-top decisions {n_end} ({n_end / n_sg:.1f} per side-game); "
          f"act-top multi-legal decisions {act_n}")
    print(f"(1) turn-over continues where the offset ends: certain {np.mean(lo > E):.4f}, "
          f"possible {np.mean(hi > E):.4f} of end-top decisions "
          f"({np.sum(lo > E) / n_sg:.2f} to {np.sum(hi > E) / n_sg:.2f} per side-game)")
    print("    upper-bound sum quantiles 10/50/90: "
          + ", ".join(f"{q:.2f}" for q in np.percentile(hi, [10, 50, 90])))
    print(f"(2) turn-over could end where the offset acts: at most {act_possible / act_n:.4f} "
          f"of act-top decisions ({act_possible / n_sg:.1f} per side-game)")


if __name__ == "__main__":
    sys.exit(main())
