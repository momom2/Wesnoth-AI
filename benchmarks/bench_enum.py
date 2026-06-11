"""A/B micro-benchmark for enumerate_legal_actions_with_priors.

Builds a fixed game (seeded scenario_pool setup), advances K
policy-sampled decisions to a representative mid-turn state, then
times N enumeration calls on that frozen state. Run on two code
versions (git stash A/B) for a like-for-like comparison.
"""
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from transformer_policy import TransformerPolicy  # noqa: E402
from action_sampler import enumerate_legal_actions_with_priors  # noqa: E402
from tools.scenario_pool import (  # noqa: E402
    random_setup, build_scenario_gamestate, load_factions,
)
from tools.wesnoth_sim import WesnothSim, PvPDefaults  # noqa: E402


def main() -> int:
    rng = random.Random(42)
    torch.manual_seed(0)
    policy = TransformerPolicy(
        d_model=128, num_layers=3, num_heads=4, d_ff=256,
        device=torch.device("cpu"),
    )
    model = policy._inference_model
    encoder = policy._inference_encoder

    load_factions()
    setup = random_setup(rng, forced_faction=None)
    gs = build_scenario_gamestate(setup)
    sim = WesnothSim(gs, scenario_id=setup.scenario_id, max_turns=60)

    # Advance ~30 decisions sampling from the enumeration itself so
    # the benched state has recruits on the board and real branching.
    nrng = random.Random(7)
    for _ in range(30):
        if sim.done:
            break
        with torch.no_grad():
            encoded = encoder.encode(sim.gs)
            output = model(encoded)
            laps = enumerate_legal_actions_with_priors(
                encoded, output, sim.gs)
        if not laps:
            break
        lap = nrng.choices(laps, weights=[l.prior for l in laps])[0]
        try:
            sim.step(lap.action)
        except Exception:
            break

    with torch.no_grad():
        encoded = encoder.encode(sim.gs)
        output = model(encoded)
        n_actions = len(enumerate_legal_actions_with_priors(
            encoded, output, sim.gs))

        # Warmup, then timed runs.
        for _ in range(3):
            enumerate_legal_actions_with_priors(encoded, output, sim.gs)
        times = []
        for _ in range(60):
            t0 = time.perf_counter()
            enumerate_legal_actions_with_priors(encoded, output, sim.gs)
            times.append((time.perf_counter() - t0) * 1000.0)

    print(f"state: turn={sim.gs.global_info.turn_number} "
          f"units={len(sim.gs.map.units)} legal_actions={n_actions}")
    print(f"enumerate: median={statistics.median(times):.2f}ms "
          f"mean={statistics.mean(times):.2f}ms "
          f"min={min(times):.2f}ms over {len(times)} calls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
