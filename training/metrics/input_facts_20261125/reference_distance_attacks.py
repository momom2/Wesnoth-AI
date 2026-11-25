"""Count how often a checkpoint orders an attack on an enemy its unit is not
next to (the simulator then moved the unit first), at a given decode. Run on
main at 0.35.6, before that attack left the legal actions
(docs/input_facts_prereg_20261125.md); on later code the count is 0 by rule.

    python -I -X utf8 reference_distance_attacks.py <checkpoint> <out.json> <first seed> <games> <temperature> <end_turn_offset>
"""
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch  # noqa: E402

torch.set_num_threads(4)

import tools.raw_player as rp  # noqa: E402
from tools.abilities import hex_neighbors  # noqa: E402
from tools.eval_players import _load_policy, _play_one_eval_game, _PolicyPair  # noqa: E402
from tools.wesnoth_sim import PvPDefaults, WesnothSim  # noqa: E402
from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate, random_setup  # noqa: E402

OUT = {"games": [], "attacks": 0, "distance": 0, "decisions": 0, "by_game": []}


def play(policy, seed, temperature, offset):
    rng = random.Random(seed)
    setup = random_setup(rng, category="ladder")
    pvp = PvPDefaults(starting_gold=100, village_gold=2, village_support=1, experience_modifier=70)
    gs = build_scenario_gamestate(setup, base_income=pvp.base_income)
    sim = WesnothSim(gs, scenario_id=setup.scenario_id, max_turns=40)
    label = f"g{seed}"
    count = {"attacks": 0, "distance": 0, "decisions": 0}

    class Logging(rp.RawPolicyPlayer):
        def select_action(self, game_state, *, game_label="default", sim=None):
            action = super().select_action(game_state, game_label=game_label, sim=sim)
            count["decisions"] += 1
            if action.get("type") == "attack":
                count["attacks"] += 1
                s, t = action["start_hex"], action["target_hex"]
                if (t.x, t.y) not in hex_neighbors(s.x, s.y):
                    count["distance"] += 1
            return action

    player = Logging(policy, temperature, seed=seed, end_turn_offset=offset, memory_slots=64)
    t0 = time.perf_counter()
    _play_one_eval_game(sim, _PolicyPair(player, label, 1), _PolicyPair(player, label, 2), game_label=label)
    row = {"label": label, "scenario": setup.scenario_id, "winner": int(sim.winner),
           "turns": int(sim.gs.global_info.turn_number), "seconds": round(time.perf_counter() - t0, 1), **count}
    OUT["by_game"].append(row)
    for k in count:
        OUT[k] += count[k]
    print(json.dumps(row), flush=True)


def main():
    ckpt, out, first, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    temperature, offset = float(sys.argv[5]), float(sys.argv[6])
    policy = _load_policy(ckpt, torch.device("cpu"), label="count")
    for seed in range(first, first + n):
        play(policy, seed, temperature, offset)
        out.write_text(json.dumps(OUT), encoding="utf-8")
    print("total", OUT["attacks"], "attacks,", OUT["distance"], "from a distance,", OUT["decisions"], "decisions",
          flush=True)


if __name__ == "__main__":
    main()
