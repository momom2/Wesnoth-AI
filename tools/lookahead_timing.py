"""Time the look-ahead player on a few decisions of one game.

The raw player plays the game's first `--warmup` decisions (both sides),
then the look-ahead player (configs/lookahead.json or `--config`) decides
the next `--decisions`, and the decision's telemetry is printed: seconds per
decision split into the prior's forward, the expansions and the evaluator,
the evaluator's states and forwards per decision, the candidates by kind.

`--critic-arch full|small` replaces the configuration's evaluator by an
untrained critic of the reference's architecture or step 1's small one
(wesnoth_ai/critic.py), on `--critic-device`: the same forwards as a
trained critic, for pricing before step 1's critics exist.

    python tools/lookahead_timing.py --checkpoint training/checkpoints/parity3.pt \\
        --warmup 120 --decisions 20 [--critic-arch full]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from wesnoth_ai.lookahead_config import load_config  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "lookahead.json"


def untrained_critic(arch_name: str, view: str, policy, out: Path) -> Path:
    """An untrained critic at FULL_ARCH or SMALL_ARCH on the policy's
    vocabularies, saved as tools/critic_train.py saves one."""
    from wesnoth_ai.critic import FULL_ARCH, SMALL_ARCH, build_critic
    arch = FULL_ARCH if arch_name == "full" else SMALL_ARCH
    enc = policy._inference_encoder
    encoder, model = build_critic(arch, dict(enc.unit_type_to_id), dict(enc.faction_to_id), torch.device("cpu"))
    torch.save({"arch": arch, "model_state": model.state_dict(), "encoder_state": encoder.state_dict(),
                "unit_type_to_id": dict(encoder.unit_type_to_id), "faction_to_id": dict(encoder.faction_to_id),
                "critic": {"view": view}}, out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=120)
    ap.add_argument("--decisions", type=int, default=20)
    ap.add_argument("--end-turn-offset", type=float, default=-1.5)
    ap.add_argument("--critic-arch", choices=("full", "small"), default=None)
    ap.add_argument("--critic-view", choices=("obs", "true"), default="obs")
    ap.add_argument("--critic-device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    from tools.eval_players import _load_policy, peek_checkpoint_arch
    from tools.lookahead_player import lookahead_player
    from tools.raw_player import RawPolicyPlayer
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.game_core import snapshot_view
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate, random_setup

    policy = _load_policy(args.checkpoint, None, "prior")
    slots = int(peek_checkpoint_arch(args.checkpoint).get("memory_slots", 0) or 0) or None
    cfg = load_config(args.config)
    with tempfile.TemporaryDirectory() as tmp:
        if args.critic_arch:
            path = untrained_critic(args.critic_arch, args.critic_view, policy, Path(tmp) / "critic.pt")
            cfg = dataclasses.replace(cfg, evaluator={"name": "critic", "checkpoint": str(path),
                                                      "view": args.critic_view, "device": args.critic_device,
                                                      "batch": 64})
        raw = RawPolicyPlayer(policy, 0.0, end_turn_offset=args.end_turn_offset, memory_slots=slots)
        player = lookahead_player(raw, cfg)
        setup = random_setup(random.Random(args.seed))
        sim = WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id)
        sim._seed_salt = f"elo:{args.seed}"
        t0 = time.perf_counter()
        n = 0
        while not sim.done and n < args.warmup:
            sim.step(raw.select_action(snapshot_view(sim.gs), game_label="t"))
            n += 1
        warm = time.perf_counter() - t0
        n = 0
        while not sim.done and n < args.decisions:
            sim.step(player.select_action(snapshot_view(sim.gs), game_label="t", sim=sim))
            n += 1
    tel = player.telemetry("t")
    print(json.dumps({"scenario": setup.scenario_id, "turn": sim.turn_number, "warmup_decisions": args.warmup,
                      "warmup_seconds_per_decision": round(warm / max(1, args.warmup), 4),
                      "evaluator": cfg.evaluator.get("name"), "critic_arch": args.critic_arch,
                      "evaluator_seconds_encode": round(player.evaluator.seconds_encode, 4),
                      "evaluator_seconds_forward": round(player.evaluator.seconds_forward, 4),
                      "telemetry": tel}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
