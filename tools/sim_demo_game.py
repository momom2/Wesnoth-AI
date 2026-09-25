"""Play one game in the simulator and export it as a Wesnoth replay.

Both sides play the same checkpoint, loaded as the evaluation path loads
it (`tools.eval_sim._load_policy`: its architecture and every structural
flag, the hex basis included). Each side decides with the reference
player's decode by default (configs/reference_player.json: argmax with
its end_turn logit offset); `--temperature` and `--end-turn-offset`
change the decode, and `--mcts` plays the search player instead. The
game is written as a .bz2 replay that Wesnoth 1.18 opens (Load Game) and
copied into Wesnoth's saves directory. No Wesnoth install is needed to
play the game, only to watch it.

    python tools/reference_player.py --ensure
    python tools/sim_demo_game.py --checkpoint $(python tools/reference_player.py --path)

Without `--checkpoint` the demo plays the reference player's local
checkpoint. The replay lands at `--out` (default
`logs/sim_demo_<timestamp>.bz2`), and the log names both copies.
"""

from __future__ import annotations

import argparse
import copy
import logging
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

# Make project root importable when run as a script.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from tools import reference_player
from tools.eval_sim import _load_policy
from tools.sim_to_replay import export_replay, find_source_bz2
from tools.wesnoth_sim import PvPDefaults, WesnothSim


log = logging.getLogger("sim_demo_game")


def load_player(ckpt: Path, device, *, mcts_sims: int, temperature: float,
                end_turn_offset: float, end_turn_rule: str = "joint",
                seed: Optional[int] = None):
    """The player both sides use: the checkpoint loaded as the eval path
    loads it (its arch and every structural flag, so a relevant-set
    checkpoint chooses its hex targets in the relevant-set basis), under
    the eval-contract search when `mcts_sims` > 0 (MCTSConfig defaults:
    no material shapers), else the raw player at the given decode."""
    policy = _load_policy(ckpt, device, label="demo")
    if mcts_sims > 0:
        from tools.mcts import MCTSConfig
        from tools.mcts_policy import MCTSPolicy
        return MCTSPolicy(policy, mcts_config=MCTSConfig(n_simulations=int(mcts_sims)),
                          rng_seed=seed)
    from tools.raw_player import RawPolicyPlayer
    return RawPolicyPlayer(policy, temperature, seed=seed, end_turn_rule=end_turn_rule,
                           end_turn_offset=end_turn_offset)


def _pick_replay_seed(pool: Path, rng: random.Random) -> Optional[Path]:
    """Random `replays_dataset/*.json.gz` to bootstrap the sim from.

    Filters to 2p game_ids via index.jsonl when present (mirrors
    `sim_self_play._gather_replay_pool`). Without the filter, the
    demo can land on a campaign-scenario replay whose runtime state
    used Lua / WML paths the sim doesn't model -- e.g. a
    `[modify_unit]` that pushed hp past max_hp via `violate_maximum=yes`,
    which trips `_assert_invariants` on the first sim step.
    """
    if not pool.is_dir():
        return None
    files = list(pool.glob("*.json.gz"))
    if not files:
        return None
    # 2p filter (matches sim_self_play._gather_replay_pool).
    idx_path = pool / "index.jsonl"
    if idx_path.exists():
        import json as _json
        keep_names: set = set()
        with idx_path.open() as f:
            for line in f:
                try:
                    e = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if e.get("game_id", "").startswith("2p"):
                    keep_names.add(e.get("file", ""))
        if keep_names:
            files = [f for f in files if f.name in keep_names]
    if not files:
        return None
    return rng.choice(files)


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Model checkpoint .pt. Default: the reference "
                         "player's local checkpoint "
                         "(configs/reference_player.json).")
    ap.add_argument("--replay-pool", type=Path, default=None,
                    help="Optional pool of .json.gz replays to seed the "
                         "initial state from. If omitted, uses the "
                         "from-scratch scenario builder (random ladder "
                         "scenario + random factions/leaders from the "
                         "Default era). Pass a directory to instead "
                         "sample initial state from a real replay.")
    ap.add_argument("--scenario", type=str, default=None,
                    help="When using the from-scratch path, force this "
                         "scenario_id (e.g. multiplayer_Hamlets). "
                         "Default: random from the 21-map ladder pool.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output .bz2 path. Default: "
                         "logs/sim_demo_<local timestamp>.bz2.")
    ap.add_argument("--saves-dir", type=Path,
                    default=Path.home() / "Documents" / "My Games"
                            / "Wesnoth1.18" / "saves",
                    help="Wesnoth's saves directory; the exported "
                         ".bz2 is copied here so it shows up in "
                         "File -> Load Game without manual copying. "
                         "Default: standard Windows path. Pass an "
                         "empty string to skip the copy.")
    ap.add_argument("--fogless", action="store_true",
                    help="Play (and export) the game with fog of war "
                         "OFF -- the ladder pool's fogless condition "
                         "(ScenarioSetup.fogless).")
    ap.add_argument("--mcts", action="store_true",
                    help="Drive both sides with MCTS in the eval-contract "
                         "configuration (MCTSConfig defaults: no material "
                         "shapers, search over the checkpoint's own heads) "
                         "instead of the raw player. Much slower.")
    ap.add_argument("--mcts-sims", type=int, default=32,
                    help="Simulations per decision for --mcts "
                         "(32 = training/eval convention).")
    decode = reference_player.load()["decode"]
    ap.add_argument("--temperature", type=float,
                    default=float(decode["raw_temperature"]),
                    help="The raw player's joint temperature (0 = argmax). "
                         "Default: the reference player's.")
    ap.add_argument("--end-turn-offset", type=float,
                    default=float(decode.get("raw_end_turn_offset", 0.0)),
                    help="Added to the end_turn actor logit before the "
                         "choice. Default: the reference player's.")
    ap.add_argument("--max-turns", type=int, default=40,
                    help="Per-game turn cap.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Seed of the scenario draw and of the player's "
                         "choices. Default: time-based.")
    # The multiplayer defaults, for a game seeded from a replay
    # (--replay-pool); a from-scratch game plays its scenario's own
    # gold, village economy and experience modifier, as self-play does.
    ap.add_argument("--starting-gold", type=int, default=100)
    ap.add_argument("--village-gold", type=int, default=2)
    ap.add_argument("--village-support", type=int, default=1)
    ap.add_argument("--exp-modifier", type=int, default=70)
    ap.add_argument("--device", default="auto",
                    help="Torch device. 'auto' = DML (discrete) > "
                         "CUDA > CPU.")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING"])
    args = ap.parse_args(argv[1:])

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # 1. The checkpoint: the reference player's unless one is passed.
    ckpt = args.checkpoint or reference_player.local_path()
    if not ckpt.exists():
        log.error(f"checkpoint not found: {ckpt}"
                  + ("" if args.checkpoint else
                     " (fetch it: python tools/reference_player.py --ensure)"))
        return 2
    log.info(f"checkpoint: {ckpt}")

    # 2. Build the initial GameState.
    rng = random.Random(args.seed if args.seed is not None
                        else int(time.time()))
    pvp = PvPDefaults(
        starting_gold=args.starting_gold,
        village_gold=args.village_gold,
        village_support=args.village_support,
        experience_modifier=args.exp_modifier,
    )
    from_scratch = args.replay_pool is None
    if from_scratch:
        # From-scratch path: pick a ladder scenario + random factions/
        # leaders and build the GameState directly from wesnoth_src.
        # No source bz2 needed -- export_replay_from_scratch composes
        # the save WML from templates + the .cfg + .map.
        from tools.scenario_pool import (
            random_setup, build_scenario_gamestate, LADDER_SCENARIO_IDS,
        )
        if args.scenario:
            if args.scenario not in LADDER_SCENARIO_IDS:
                log.warning(
                    f"--scenario {args.scenario!r} not in "
                    f"LADDER_SCENARIO_IDS; proceeding anyway")
            # Sample factions/leaders through the PRODUCTION sampler
            # (which applies the FORCED_FACTION rule -- every
            # self-play game has a Knalgan side by default), then
            # override only the map. Hand-rolling the faction draw
            # here used to sample both sides uniformly, producing
            # replays OFF the training distribution (a Loy-vs-Loy
            # mirror that training can never generate; caught by the
            # user 2026-08-15).
            import dataclasses
            from tools.scenario_pool import sample_tod_start
            setup = random_setup(
                rng, category="fogless" if args.fogless else "ladder")
            setup = dataclasses.replace(
                setup,
                scenario_id=args.scenario,
                tod_start=sample_tod_start(args.scenario, rng),
            )
        else:
            setup = random_setup(
                rng, category="fogless" if args.fogless else "ladder")
        log.info(
            f"from-scratch setup: scenario={setup.scenario_id} "
            f"factions={setup.faction1} vs {setup.faction2} "
            f"leaders={setup.leader1} / {setup.leader2}")
        # The scenario's own economy (None), as sim_self_play reads it
        # since 2026-09-21: five of the seven minis pay 3 gold a village.
        gs = build_scenario_gamestate(setup, base_income=pvp.base_income)
        sim = WesnothSim(gs, scenario_id=setup.scenario_id,
                         max_turns=args.max_turns)
        src_bz2 = None  # unused in the from-scratch path
    else:
        seed_replay = _pick_replay_seed(args.replay_pool, rng)
        if seed_replay is None:
            log.error(f"no .json.gz replays under {args.replay_pool}")
            return 2
        log.info(f"seed replay: {seed_replay.name}")
        src_bz2 = find_source_bz2(seed_replay)
        if src_bz2 is None:
            log.error(
                f"could not auto-locate the .bz2 matching {seed_replay}. "
                f"Pick a different --replay-pool entry, check that "
                f"replays_raw/ has the matching source, or drop "
                f"--replay-pool to use the from-scratch path.")
            return 2
        log.info(f"source bz2:  {src_bz2}")
        sim = WesnothSim.from_replay(
            seed_replay, max_turns=args.max_turns, pvp_defaults=pvp,
        )

    # 3. Load the player and play the game.
    from tools.device_select import select_inference_device, describe_device
    device = select_inference_device(args.device)
    log.info(f"device: {describe_device(device)}")
    policy = load_player(
        ckpt, device, mcts_sims=args.mcts_sims if args.mcts else 0,
        temperature=args.temperature, end_turn_offset=args.end_turn_offset,
        seed=args.seed)
    log.info(f"player: {type(policy).__name__} "
             + (f"({args.mcts_sims} simulations per decision)" if args.mcts else
                f"(temperature {args.temperature}, end_turn offset "
                f"{args.end_turn_offset})"))
    log.info("running one game (this is headless -- progress in stderr)...")
    t0 = time.perf_counter()
    game_label = "demo"
    while not sim.done:
        # Deepcopy the state before each select_action: a search player
        # keeps references to the states it decided on, and `sim.step`
        # mutates `sim.gs` in place (the contract in
        # `transformer_policy.select_action`'s docstring).
        pre_state = copy.deepcopy(sim.gs)
        action = policy.select_action(pre_state, game_label=game_label,
                                      sim=sim)
        sim.step(action)
    # Nothing trains here: drop whatever the player recorded.
    policy.drop_pending(game_label)
    dt = time.perf_counter() - t0

    log.info(
        f"game over in {dt:.1f}s: winner={sim.winner} "
        f"turns={sim.gs.global_info.turn_number} "
        f"ended_by={sim.ended_by} "
        f"actions={len(sim.command_history)}")

    # 4. Export the replay.
    if args.out is None:
        Path("logs").mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = Path("logs") / f"sim_demo_{ts}.bz2"
    else:
        out_path = args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)

    if src_bz2 is None:
        # From-scratch path: no source bz2 to splice; compose the
        # whole save WML from templates + scenario .cfg.
        from tools.sim_to_replay import export_replay_from_scratch
        export_replay_from_scratch(sim, out_path, pvp_defaults=pvp)
    else:
        export_replay(sim, source_bz2=src_bz2, out_path=out_path,
                      pvp_defaults=pvp)
    log.info(f"wrote {out_path}")

    # Copy into Wesnoth's saves dir so the file shows up under
    # File -> Load Game without the user having to navigate to
    # logs/. shutil.copy2 preserves mtime so Wesnoth's "most
    # recent" sort orders correctly. Failure is non-fatal -- the
    # logs/ copy is still there.
    # Path("") stringifies to "." (truthy), so test the RAW string
    # (project round-2 C9: the documented "" escape copied into the
    # cwd instead of skipping).
    saves_dir = (Path(args.saves_dir)
                 if str(args.saves_dir) not in ("", ".") else None)
    if saves_dir is not None:
        try:
            saves_dir.mkdir(parents=True, exist_ok=True)
            saves_path = saves_dir / out_path.name
            shutil.copy2(out_path, saves_path)
            log.info(f"copied to Wesnoth saves: {saves_path}")
            log.info(
                f"To watch: open Wesnoth -> Load Game -> "
                f"{out_path.name}")
        except OSError as e:
            log.warning(
                f"could not copy to {saves_dir}: {e}. "
                f"Open the file manually from {out_path}.")
    else:
        log.info(
            f"To watch: open {out_path} in Wesnoth (File -> Load "
            f"Game -> pick this file).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
