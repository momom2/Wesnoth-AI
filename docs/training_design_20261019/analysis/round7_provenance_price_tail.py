"""Round 7, provenance stance: the generation price a long coin batch pays, and how much
per-decision work it can absorb.

round6_provenance_prices.py priced a pool game at the arm's whole wall (600 games in 129 s for
ref_p120x6), which includes worker start-up, the last wave's tail and the server's shutdown. A
100,000-game batch pays almost none of those. This script reads every completion line
("HH:MM:SS INFO game N done in ...") of each 120-job arm log in
training/metrics/generation_speed_r5_20261015/ and reports:
  - the arm's wall (walls.txt) and the price it implies (round 6's reading);
  - the steady-state rate: games completed between the 120th and the 480th completion, over the
    seconds between them (one-second timestamps; both ends are inside the run, so start-up and the
    last wave are excluded);
  - the start-up (arm start to first completion) and tail (480th to 600th) in seconds.
It then reads the worker stage table of the same arm (speed.json) to state the per-decision
budget: the worker's own wall per decision against its decision cycle, so that X ms of added
worker work per decision slows a worker by about X / cycle (an upper bound when the server, not
the worker, sets the pace).
Prices at $0.42 an hour (seed section 1). numpy and the standard library only.
"""
import json
import re
from pathlib import Path


RUN = Path(__file__).resolve().parents[3] / "training/metrics/generation_speed_r5_20261015"
PRICE_H = 0.42
ARMS = ("ref_p120x6", "priors_p120x6", "compile_p120x6", "bf16res_p120x6", "compile_bf16res_p120x6",
        "best_repeat", "ref_p160x6", "bf16res_p160x6")


def seconds(hms):
    h, m, s = (int(x) for x in hms.split(":"))
    return 3600 * h + 60 * m + s


def completions(log):
    start, done = None, []
    for line in log.read_text().splitlines():
        m = re.match(r"(\d\d:\d\d:\d\d) INFO (.*)", line)
        if not m:
            continue
        t = seconds(m.group(1))
        if start is None:
            start = t
        r = re.search(r"results=(\d+)/(\d+)", m.group(2))
        if "done in" in m.group(2) and r:
            done.append((int(r.group(1)), t))
    by_count = {}
    for k, t in done:
        by_count.setdefault(k, t)
    return start, by_count


def walls():
    out = {}
    for line in (RUN / "walls.txt").read_text().splitlines():
        m = re.search(r" arm (\S+) rc=0 ok (\d+) s", line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def main():
    arms = json.loads((RUN / "speed.json").read_text())["arms"]
    wall = walls()
    print("arm                      games  wall_s  $/game(wall)  steady games/h  $/game(steady)  start_s  tail_s  steady/wall")
    steady_prices = {}
    for name in ARMS:
        log = RUN / f"arm_{name}.log"
        if not log.exists() or name not in wall:
            continue
        start, c = completions(log)
        n = max(c)
        lo, hi = n // 5, 4 * n // 5
        rate = (hi - lo) / (c[hi] - c[lo])                  # games per second in the middle window
        p_wall = PRICE_H / (n / wall[name] * 3600)
        p_steady = PRICE_H / (rate * 3600)
        steady_prices[name] = p_steady
        print(f"{name:24s} {n:5d}  {wall[name]:6d}  {p_wall:.3e}     {rate * 3600:9.0f}      {p_steady:.3e}   "
              f"{c[1] - start:6d}  {c[n] - c[hi]:6d}  {p_wall / p_steady:.2f}")
    ref = arms["ref_p120x6"]
    w = ref["worker"]
    cycle = w["decision_wall_ms"]
    print(f"\nref_p120x6 worker: decision cycle {cycle:.2f} ms (own wall {w['own_wall_ms']:.2f}, "
          f"round trip {w['stage_ms_per_decision']['round_trip']:.2f}, cpu {w['decision_cpu_ms']:.2f}); "
          f"server busy {ref['server']['infer_s'] / ref['server']['wall_s']:.2f} of its wall, "
          f"mean batch {ref['server']['mean_batch']:.1f} of max {ref['server']['max_batch']}")
    cpu_cores = ref["decisions_per_s"] * w["decision_cpu_ms"] / 1000
    print(f"worker CPU in use: {cpu_cores:.1f} cores at {ref['decisions_per_s']:.0f} decisions/s "
          f"(box.txt: 31 usable cores)")
    print("added worker work per decision -> slowdown bound (added / cycle), and per game at 465 decisions:")
    for added_ms in (0.1, 0.5, 0.7, 1.0, 2.0):
        print(f"  +{added_ms:.1f} ms/decision: <= {added_ms / cycle:.3f} slower; "
              f"= {added_ms * 465 / 1000:.2f} s of worker time per game")
    walk = 0.321                                               # round6_simplify_walk_cost.out, s per game
    print(f"the luck walk at {walk} s a game, run in the worker at game end: {walk * 1000 / 465:.2f} ms per decision, "
          f"<= {walk * 1000 / 465 / cycle:.3f} slower")
    p = steady_prices["ref_p120x6"]
    print(f"\nsteady ref price {p:.3e} a game; the round-6 three-round line $2.92e-5 then allows "
          f"overhead up to {2.92e-5 / p - 1:.2f}; with the 1.41x host spread: {2.92e-5 / (1.41 * p) - 1:+.2f}")


if __name__ == "__main__":
    main()
