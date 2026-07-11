"""One Elo-ladder evaluation game between two player specs; result to
a JSON file. Designed to be launched N-way parallel (each game is an
independent process — the pattern that saturated a 4090 where the
central-server pool could not; see BACKLOG 2026-07-03).

Usage:
    python tools/elo_eval_game.py LABEL_A SPEC_A LABEL_B SPEC_B \
        SIDE_A SEED OUTDIR [--max-turns 200] [--mcts-sims 32]

SPEC is a checkpoint .pt path or the literal 'dummy' (scripted
baseline). Checkpoint players play through MCTS at --mcts-sims
(training-matched, 32) unless 0 (raw policy). Maps come from the
LADDER-ONLY default `random_setup` (pinned by test_elo_ladder_maps).

The result file records BOTH the outcome and the final material
margin from A's perspective, so the collector can fit Elo under the
PURE (primary -- draws are draws; material advantage is a training
crutch and does not factor into evaluation, user 2026-07-11) and
material-sign (diagnostic) draw conventions from one set of games.
Eval search likewise runs WITHOUT the material shapers
(draw_tiebreak, aux_value_bonus) regardless of training config.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.draw_tiebreak import DrawTiebreakConfig, material_margin
from tools.elo_ladder import _ScriptedAdapter
from tools.eval_sim import _PolicyPair, _load_policy, _play_one_eval_game
from tools.scenario_pool import build_scenario_gamestate, random_setup
from wesnoth_sim import WesnothSim

log = logging.getLogger("elo_eval_game")


def _build_player(spec: str, label: str, sims: int, device):
    if spec == "dummy":
        from dummy_policy import DummyPolicy
        return _ScriptedAdapter(DummyPolicy())
    policy = _load_policy(Path(spec), device, label=label)
    if sims > 0:
        from tools.mcts import MCTSConfig
        from tools.mcts_policy import MCTSPolicy
        import os
        # EVALUATION CONTRACT (user, 2026-07-11): valuing material
        # advantage is a TRAINING crutch, not part of what policy
        # performance means -- so the material-based search shapers
        # (draw_tiebreak, aux_value_bonus) are OFF here regardless of
        # what the checkpoint trained with. Eval search sees the real
        # game: win +1, loss -1, draw 0. moves_left_utility (time
        # preference among equal outcomes, no material content) stays
        # env-configurable.
        return MCTSPolicy(policy, MCTSConfig(
            n_simulations=sims,
            moves_left_utility=float(
                os.environ.get("ELO_MOVES_LEFT_UTILITY", "0") or 0)))
    return policy


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("label_a"); ap.add_argument("spec_a")
    ap.add_argument("label_b"); ap.add_argument("spec_b")
    ap.add_argument("side_a", type=int, choices=(1, 2))
    ap.add_argument("seed", type=int)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--mcts-sims", type=int, default=32)
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level))

    import torch
    torch.set_num_threads(2)
    device = (torch.device("cuda") if torch.cuda.is_available() else None)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / (
        f"game_{args.label_a}_{args.label_b}_s{args.side_a}"
        f"_{args.seed}.json")
    if out_path.exists():
        print(f"exists, skipping: {out_path.name}")
        return 0

    pa = _build_player(args.spec_a, args.label_a, args.mcts_sims, device)
    pb = _build_player(args.spec_b, args.label_b, args.mcts_sims, device)

    rng = random.Random(args.seed)
    setup = random_setup(rng)
    gs = build_scenario_gamestate(setup)
    sim = WesnothSim(gs, scenario_id=setup.scenario_id,
                     max_turns=args.max_turns)
    game_label = out_path.stem
    t0 = time.time()
    r = _play_one_eval_game(
        sim,
        _PolicyPair(policy=pa, label=args.label_a, side=args.side_a),
        _PolicyPair(policy=pb, label=args.label_b, side=3 - args.side_a),
        game_label=game_label)
    margin_a = material_margin(sim.gs, args.side_a,
                               DrawTiebreakConfig(cap=0.3))
    result = {
        "label_a": args.label_a, "label_b": args.label_b,
        "side_a": args.side_a, "seed": args.seed,
        "scenario_id": setup.scenario_id,
        "outcome_a": r.outcome,          # win/loss/draw/timeout from A
        "margin_a": float(margin_a),     # final material, A's view
        "turns": sim.gs.global_info.turn_number,
        "ended_by": sim.ended_by,
        "secs": round(time.time() - t0, 1),
    }
    out_path.write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
