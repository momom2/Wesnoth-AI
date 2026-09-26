"""Evaluation players: a policy built from a checkpoint, and a game
between two of them.

`_load_policy` builds a TransformerPolicy at the architecture and the
structural flags a checkpoint was trained with (`peek_checkpoint_arch`,
`CHECKPOINT_STRUCT_FLAGS`) and loads its weights. `_PolicyPair` binds a
policy to the side it plays, and `_play_one_eval_game` plays one
simulator game between two pairs, without training, into a
`GameResult`. The eval command lines (tools/eval_sim.py,
tools/elo_eval_game.py, tools/elo_ladder.py), turn_gap, the benchmarks
and the probes build their players and play their games through here.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from wesnoth_ai.sim.classes import GameState
from tools.selfplay_game import _update_closest_approach, _would_recruit_bounce
from wesnoth_ai.transformer_policy import TransformerPolicy
from tools.game_record import note_search_outcomes
from tools.wesnoth_sim import WesnothSim


log = logging.getLogger("eval_players")


# ---------------------------------------------------------------------
# Per-game record + side dispatch
# ---------------------------------------------------------------------

@dataclass
class GameResult:
    """One game's outcome, in the same shape eval_vs_builtin emits so
    eval_daily's parser doesn't have to branch on backend."""
    scenario_id:   str
    our_faction:   str
    opp_faction:   str
    our_side:      int            # 1 or 2 -- which side OUR policy played
    outcome:       str            # win / loss / draw / timeout / errored
    turns:         int
    our_actions:   int
    wall_seconds:  float
    # Sim-specific extras useful for diagnosing the no-kills phase.
    # eval_vs_builtin doesn't have these, eval_daily ignores unknown
    # keys when summarizing -- so no schema break.
    closest_approach_ours: Optional[int] = None
    closest_approach_opp:  Optional[int] = None
    attack_count_ours:     int = 0
    # Turns handed to a side neither policy plays, each ended at once.
    # The simulator gives turns to the two players and plays the
    # neutral sides itself, so a nonzero count is a defect to report.
    unplayed_side_turns:   int = 0


@dataclass
class _PolicyPair:
    """Bundle a policy + its label + which side it plays. Two of these
    drive each game; the rollout loop picks the right one based on
    `gs.global_info.current_side`."""
    policy:       TransformerPolicy
    label:        str             # for game_label scoping (per-game)
    side:         int             # 1 or 2

    def select_action(self, gs: GameState, game_label: str, sim) -> Dict:
        return self.policy.select_action(gs, game_label=game_label, sim=sim)

    def drop_pending(self, game_label: str) -> None:
        self.policy.drop_pending(game_label)

    def drop_last_pending(self, game_label: str) -> bool:
        drop = getattr(self.policy, "drop_last_pending", None)
        return bool(callable(drop) and drop(game_label))


# ---------------------------------------------------------------------
# Two-policy rollout
# ---------------------------------------------------------------------

def _play_one_eval_game(
    sim: WesnothSim,
    pair_a: _PolicyPair,
    pair_b: _PolicyPair,
    *,
    game_label: str,
) -> GameResult:
    """Drive `sim` to completion. On each step the acting side is
    looked up in `(pair_a, pair_b)` -- one of them owns side 1, the
    other side 2 -- and that policy picks the action.

    No training. Pending Transitions are dropped at game end so the
    policies' internal queues stay clean for the next game.
    """
    by_side: Dict[int, _PolicyPair] = {pair_a.side: pair_a,
                                       pair_b.side: pair_b}
    # Track metrics from OUR policy's view. Caller passes which side
    # is "ours" via pair_a (convention: pair_a is the candidate being
    # evaluated; pair_b is the reference).
    our_side = pair_a.side
    closest_approach: Dict[int, Optional[int]] = {}
    _update_closest_approach(sim.gs, closest_approach)
    attack_count_ours = 0
    our_actions = 0
    unplayed_side_turns = 0
    t0 = time.perf_counter()

    while not sim.done:
        acting_side = sim.gs.global_info.current_side
        actor = by_side.get(acting_side)
        if actor is None:
            # The simulator hands turns only to the players
            # (WesnothSim._next_player_side), so this is a defect: the
            # turn is ended, counted and reported, and the game goes on.
            if not unplayed_side_turns:
                log.warning("%s: side %d is to move at turn %d and no policy plays it; "
                            "ending its turns (counted in unplayed_side_turns)",
                            game_label, acting_side, sim.gs.global_info.turn_number)
            unplayed_side_turns += 1
            sim.step({"type": "end_turn"})
            continue
        # Stable snapshot for select_action (see play_one_game in
        # tools/selfplay_game.py for why this deepcopy is
        # load-bearing).
        pre_state = copy.deepcopy(sim.gs)
        from tools.mcts import fork_guard
        with fork_guard(sim):
            action = actor.select_action(pre_state, game_label, sim)

        # Recruit-bounce retry (god-view occupied hex). Same pattern
        # as play_one_game in tools/selfplay_game.py.
        while _would_recruit_bounce(action, sim.gs):
            tgt = action["target_hex"]
            sim.reject_recruit_hex(tgt.x, tgt.y)
            # Discard the bounced decision AND any cached plan
            # (round-24 C6: without this, a search policy that
            # serves from a cached plan -- PlanTournamentPolicy,
            # TurnCommitPolicy -- kept serving the SAME plan and
            # silently forfeited the recruit; play_one_game in
            # tools/selfplay_game.py has always done this).
            actor.drop_last_pending(game_label)
            pre_state = copy.deepcopy(sim.gs)
            action = actor.select_action(pre_state, game_label, sim)

        atype = action.get("type", "end_turn")
        if acting_side == our_side and atype == "attack":
            attack_count_ours += 1
        if acting_side == our_side:
            our_actions += 1

        commands_before = len(sim.command_history)
        sim.step(action)
        note_search_outcomes(sim, actor.policy, game_label, commands_before)
        _update_closest_approach(sim.gs, closest_approach)

    wall = time.perf_counter() - t0

    # Map sim winner to OUR-side perspective.
    if sim.winner == our_side:
        outcome = "win"
    elif sim.winner == 0:
        outcome = ("timeout" if sim.ended_by in ("max_turns",
                                                  "max_actions")
                   else "draw")
    else:
        outcome = "loss"

    # Drop pending Transitions from both policies so neither leaks
    # into a future train_step (though training isn't in scope here,
    # this keeps the policies clean for any subsequent caller).
    pair_a.drop_pending(game_label)
    pair_b.drop_pending(game_label)

    # Faction/scenario labels — pulled from the live state's sides.
    fa = (sim.gs.sides[0].faction if len(sim.gs.sides) >= 1 else "")
    fb = (sim.gs.sides[1].faction if len(sim.gs.sides) >= 2 else "")
    our_faction = fa if our_side == 1 else fb
    opp_faction = fb if our_side == 1 else fa

    return GameResult(
        scenario_id=sim.scenario_id,
        our_faction=our_faction,
        opp_faction=opp_faction,
        our_side=our_side,
        outcome=outcome,
        turns=sim.gs.global_info.turn_number,
        our_actions=our_actions,
        wall_seconds=wall,
        closest_approach_ours=closest_approach.get(our_side),
        closest_approach_opp=closest_approach.get(3 - our_side),
        attack_count_ours=attack_count_ours,
        unplayed_side_turns=unplayed_side_turns,
    )


# ---------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------

# Structural flags a checkpoint may carry. Each one gates a MODULE
# that owns weights, so a policy built without it silently drops
# those tensors on load (they arrive as "unexpected keys") and then
# evaluates a different model than the one that trained.
CHECKPOINT_STRUCT_FLAGS = ("aux_score", "moves_left",
                           "relevant_set_hexes", "gbc", "value_material",
                           "fog_hides_enemy_villages", "terrain_multi_hot")


def peek_checkpoint_arch(
    ckpt_path: Optional[Path], label: str = "ckpt",
) -> Dict[str, object]:
    """Read the constructor kwargs a checkpoint was trained with:
    the arch ints plus `CHECKPOINT_STRUCT_FLAGS`. ONE read, so every
    eval entry point agrees on what a checkpoint is.

    On an unreadable checkpoint this logs and returns {} -- callers
    then build default paths, which is the pre-2026-07-29 behaviour.
    It is logged rather than swallowed because a silent {} here means
    "evaluated a structurally different model", which is exactly the
    failure this function exists to prevent."""
    out: Dict[str, object] = {}
    if not (ckpt_path and ckpt_path.exists()):
        return out
    import torch
    try:
        raw = torch.load(ckpt_path, map_location="cpu",
                         weights_only=False)
    except Exception as e:
        log.warning(f"[{label}] couldn't peek arch from {ckpt_path}: "
                    f"{e!r}; building DEFAULT paths, which may not "
                    f"match this checkpoint")
        return out
    for k in ("d_model", "num_layers", "num_heads", "d_ff"):
        v = (raw.get("arch") or {}).get(k)
        if v is not None:
            out[k] = int(v)
    for k in CHECKPOINT_STRUCT_FLAGS:
        if raw.get(k):
            out[k] = True
    return out


def _load_policy(
    ckpt_path: Optional[Path], device, label: str,
    infer_bf16: bool = False, infer_compile: bool = False,
) -> TransformerPolicy:
    """Build a TransformerPolicy at the checkpoint's saved arch and
    load weights. `ckpt_path=None` means "random init" -- useful as
    a lowest-bar reference. Mirrors the arch-peek pattern in
    sim_self_play.main.

    Structural flags (aux_score / moves_left / relevant_set_hexes)
    are peeked from the checkpoint too, so the policy is BUILT with
    the paths the checkpoint carries and every weight loads --
    without this, an eval would silently measure a structurally
    different model than the one that trained (the probe-bug class
    from the 2026-07-29 hoarding probe).

    A checkpoint that does not load raises, an arch mismatch included:
    the policy it was loading into holds the random init."""
    if ckpt_path is not None and not Path(ckpt_path).exists():
        # A typo'd path used to measure a random-init net under the
        # checkpoint's name (2026-09-06: a value-head study ran on one
        # for two hours). Random init is only ever explicit (None).
        raise FileNotFoundError(f"[{label}] checkpoint {ckpt_path} does not exist; "
                                f"pass None for a deliberate random init")
    arch_kwargs = peek_checkpoint_arch(ckpt_path, label)
    policy = TransformerPolicy(device=device, infer_bf16=infer_bf16,
                               infer_compile=infer_compile,
                               **arch_kwargs)
    if ckpt_path is not None:
        try:
            policy.load_checkpoint(ckpt_path)
        except RuntimeError as e:
            raise RuntimeError(f"[{label}] cannot load {ckpt_path}: {e}") from e
        log.info(f"[{label}] loaded {Path(ckpt_path).name}")
    else:
        log.info(f"[{label}] no checkpoint -> random init")
    return policy
