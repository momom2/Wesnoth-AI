"""Procedure provenance for eval results -- stdlib-only on purpose:
run_elo_batch is a long-lived torch-free driver, and importing the
tag helper from elo_eval_game pulled torch + the sim stack into it
(round-5 C9: +174 MB RSS in the batch process)."""
from __future__ import annotations

from typing import Optional


def end_turn_refusal(side: str, spec: str, sims: int, raw_temperature,
                     raw_end_turn: str, raw_end_turn_offset: float) -> Optional[str]:
    """Why side `side`'s end_turn decode flags cannot apply, or None.
    They are knobs of the joint-temperature raw player
    (tools/raw_player.py), which a side plays only at sims 0 with a raw
    temperature and a network. The search, the legacy sampler and the
    scripted 'dummy' never read them, while the result file recorded
    them as if they had played."""
    if raw_end_turn == "joint" and not raw_end_turn_offset:
        return None
    if sims > 0 or raw_temperature is None or spec == "dummy":
        return (f"--raw-end-turn-{side}/--raw-end-turn-offset-{side} decode the "
                f"joint-temperature raw player only: side {side} needs sims 0, "
                f"--raw-temperature-{side} and a network (the search, the legacy "
                f"sampler and 'dummy' never read them).")
    return None


def memory_reset_refusal(side: str, sims: int, raw_temperature, memory: Optional[int],
                         memory_reset: bool) -> Optional[str]:
    """Why side `side`'s per-turn memory reset cannot apply, or None. It
    is a decode of the joint-temperature raw player of a model with a
    memory (tools/raw_player.py): a side at sims 0 with a raw temperature,
    playing at least one memory slot."""
    if not memory_reset:
        return None
    if sims > 0 or raw_temperature is None:
        return (f"--memory-reset-{side} decodes the joint-temperature raw player only: "
                f"side {side} needs sims 0 and --raw-temperature-{side}.")
    if not memory:
        return (f"--memory-reset-{side} needs a player with memory slots: side {side} plays "
                f"{'no memory' if memory is None else '0 slots'}.")
    return None


def procedure_of(sims: int, plan: bool, no_turn_search: bool,
                 raw_temperature=None, gumbel_root: bool = True, *,
                 raw_end_turn: str = "joint", raw_end_turn_offset: float = 0.0,
                 memory_reset: bool = False) -> str:
    """Canonical procedure tag for result provenance. Carries the
    sims budget (round-13 C2: 'mcts' alone let an outdir silently
    mix --mcts-sims 16 and 32 games -- different estimands), the
    root procedure of a plain search ('mcts' = Gumbel root, 'puct' =
    plain PUCT root, what the az legs trained with) and, for the raw
    player, its joint sampling temperature (tools/raw_player.py;
    None = the legacy factored sampler) and its end_turn decode
    ('+endm' = decided at the actor level, '+eo<x>' = the end_turn
    logit offset x; panel test 1) and its memory decode ('+mr' = the
    memory reset to the learned initial state at each of the side's
    turns), so those estimands never share an outdir either."""
    if sims <= 0:
        if raw_temperature is None:
            return "raw"
        tag = f"raw:t{float(raw_temperature):g}"
        if raw_end_turn == "actor":
            tag += "+endm"
        if raw_end_turn_offset:
            tag += f"+eo{float(raw_end_turn_offset):g}"
        if memory_reset:
            tag += "+mr"
        return tag
    if plan:
        name = "plan_tournament"
    elif no_turn_search:
        name = "mcts" if gumbel_root else "puct"
    else:
        name = "tcs"
    return f"{name}:{sims}"
