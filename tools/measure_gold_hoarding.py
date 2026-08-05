"""Gold-hoarding A/B across checkpoints (re-run of the 2026-07-20 probe).

Measures, per checkpoint, on shared fogged-ladder seeds with the RAW policy
(no MCTS, so we see the prior's own behaviour, not search's correction):
  bank      mean gold held by the side at each of its turn starts
  end_gold  gold at game end
  recruits  recruit actions per game
  turns     game length
"""
import copy
import pathlib
import random
import sys
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))
ROOT = pathlib.Path(r"C:\Users\amaur\Desktop\Perso\projects\Wesnoth_AI")
sys.path.insert(0, str(ROOT))

from tools.scenario_pool import (random_setup, build_scenario_gamestate,
                                 load_factions)
from tools.wesnoth_sim import WesnothSim
from wesnoth_ai.transformer_policy import TransformerPolicy


def load_policy(path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    arch = raw.get("arch", {}) or {}
    kw = {k: int(arch[k]) for k in
          ("d_model", "num_layers", "num_heads", "d_ff") if k in arch}
    p = TransformerPolicy(aux_score=bool(raw.get("aux_score")),
                          moves_left=bool(raw.get("moves_left")),
                          # Build the advice path when the checkpoint has
                          # one, else its 12 advice_* tensors load as
                          # "unexpected keys" and are DROPPED. Harmless for
                          # this probe (no advice tokens are attached at
                          # act time, so the path is inert either way), but
                          # a probe that silently discards weights is one
                          # bad assumption away from lying.
                          advice=bool(raw.get("advice", False)),
                          relevant_set_hexes=bool(
                              raw.get("relevant_set_hexes", False)),
                          **kw)
    p.load_checkpoint(pathlib.Path(path))
    return p


def run(policy, seeds, max_turns=30):
    banks, ends, recs, turns = [], [], [], []
    for sd in seeds:
        setup = random_setup(random.Random(sd), forced_faction=None)
        sim = WesnothSim(build_scenario_gamestate(setup),
                         scenario_id=setup.scenario_id, max_turns=max_turns)
        seen_turn, bank_samples, n_rec, steps = set(), [], 0, 0
        while not sim.done and steps < 1200:
            gs = sim.gs
            side = gs.global_info.current_side
            if side in (1, 2):
                key = (gs.global_info.turn_number, side)
                if key not in seen_turn:          # sample once per side-turn
                    seen_turn.add(key)
                    bank_samples.append(int(gs.sides[side - 1].current_gold))
            act = policy.select_action(copy.deepcopy(gs),
                                       game_label=f"s{sd}", sim=sim)
            if act is None:
                break
            if isinstance(act, dict) and act.get("type") == "recruit":
                n_rec += 1
            sim.step(act)
            steps += 1
        banks.append(sum(bank_samples) / max(1, len(bank_samples)))
        ends.append(int(sim.gs.sides[0].current_gold))
        recs.append(n_rec)
        turns.append(sim.gs.global_info.turn_number)
    n = len(seeds)
    return (sum(banks) / n, sum(ends) / n, sum(recs) / n, sum(turns) / n)


if __name__ == "__main__":
    load_factions()
    seeds = [201, 202, 203]
    ck = ROOT / "training" / "checkpoints"
    for label, f in [("0719 (2.75M, pre-fix)", "tier_a_campaign_20260719.pt"),
                     ("0722 (3.74M, POST-fix)", "tier_a_campaign_20260722.pt")]:
        pol = load_policy(ck / f)
        b, e, r, t = run(pol, seeds)
        print(f"{label:26s} bank={b:6.1f}  end_gold={e:6.1f}  "
              f"recruits/game={r:5.1f}  turns={t:4.1f}", flush=True)
