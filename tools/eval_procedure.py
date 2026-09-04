"""Procedure provenance for eval results -- stdlib-only on purpose:
run_elo_batch is a long-lived torch-free driver, and importing the
tag helper from elo_eval_game pulled torch + the sim stack into it
(round-5 C9: +174 MB RSS in the batch process)."""
from __future__ import annotations


def procedure_of(sims: int, plan: bool, no_turn_search: bool,
                 raw_temperature=None) -> str:
    """Canonical procedure tag for result provenance. Carries the
    sims budget (round-13 C2: 'mcts' alone let an outdir silently
    mix --mcts-sims 16 and 32 games -- different estimands) and,
    for the raw player, its joint sampling temperature
    (tools/raw_player.py; None = the legacy factored sampler)."""
    if sims <= 0:
        if raw_temperature is None:
            return "raw"
        return f"raw:t{float(raw_temperature):g}"
    name = ("plan_tournament" if plan
            else ("mcts" if no_turn_search else "tcs"))
    return f"{name}:{sims}"
