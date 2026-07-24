"""Detector -> trainable signal: the ADVISOR (propose + dispose).

See docs/detector_training_signal.md. This is the detector-side spine of
the training-signal pipeline, deliberately independent of the model:

  PROPOSE  -- run the Tier-1 (product-order certificate) generators on a
              played side-turn; each finding carries the two reordered
              action indices (the gaining attack + the move that sets it
              up).
  DISPOSE  -- reconstruct the played vs proposed orderings into their exact
              end-state distributions and score each with the MODEL'S OWN
              value function -> delta_v = V(proposed) - V(played). A
              stronger value net gives a better delta_v, so the model
              learns to IGNORE the signal where it deviates deliberately
              (exp management): delta_v <= 0 there.

The trainer (later) distills toward the proposed action weighted by
max(0, delta_v). Tier-1 only for the MVP -- product-order certificates are
dominant on every tracked dimension (incl. XP), so they are the safest to
couple; banking-tier motifs (which trade dimensions) come with the learned
gate. `delta_v is None` (reconstruction bailed) -> advice-token-only, no
distillation push.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from tools.swap_detector import (                                 # noqa: E402
    SideTurn, Finding, reconstruct_side_turn_dist,
    backstab_setup_findings, leadership_setup_findings,
)
from wesnoth_ai.classes import GameState                          # noqa: E402

# gs -> scalar value from gs's acting-side perspective (e.g. the C51 value
# head's expected value in [-1, +1]). reconstruct_side_turn_dist applies a
# side-turn's actions but NOT end_turn, so the acting side is unchanged and
# the two orderings' end-states are directly comparable.
ValueFn = Callable[[GameState], float]

# Tier-1 = product-order certificates only (design doc: safest to couple).
TIER1_GENERATORS = {
    "backstab_setup":   backstab_setup_findings,
    "leadership_setup": leadership_setup_findings,
}


@dataclass
class AdviceSignal:
    """One value-net-judged piece of advice for a played side-turn."""
    motif:             str
    tier:              str                     # "tier1"
    game_id:           str
    turn:              int
    side:              int
    # the action the model should have done FIRST (the setup move) and the
    # action it did first instead (the gaining attack) -- the divergence.
    proposed_action:   list
    divergence_action: list
    # board-localized refs (0-indexed hexes) the model can attend to.
    attacker_pos:      Tuple[int, int]
    defender_pos:      Tuple[int, int]
    gain_vector:       Dict[str, str]          # detector's claimed deltas
    delta_v:           Optional[float]         # V(proposed) - V(played); None if unjudged


def _reorder_before(actions: List[list], move_idx: int,
                    attack_idx: int) -> List[list]:
    """The proposed ordering: relocate the setup move (at `move_idx`) to
    just BEFORE the gaining attack (at `attack_idx`). For Tier-1 setups the
    move is recorded AFTER the attack (move_idx > attack_idx), so it bubbles
    up to the attack's position."""
    acts = list(actions)
    mv = acts.pop(move_idx)
    insert_at = attack_idx if move_idx < attack_idx else attack_idx
    acts.insert(insert_at, mv)
    return acts


def _expected_value(dist: Optional[List[Tuple[GameState, float]]],
                    value_fn: ValueFn) -> Optional[float]:
    """Probability-weighted value over a reconstructed end-state
    distribution (normalized to 1 by the reconstructor)."""
    if dist is None:
        return None
    return sum(p * value_fn(st) for st, p in dist)


def delta_v_for_finding(
    st: SideTurn, finding: Finding, value_fn: ValueFn, *,
    advancement_choice: str = "uniform",
) -> Optional[float]:
    """V(proposed ordering) - V(played ordering), both reconstructed to the
    side-turn end and scored by `value_fn`. None if the finding carries no
    reorder indices or either reconstruction bails (advancement past cap /
    blow-up) -- the caller then falls back to advice-token-only."""
    if finding.attack_idx is None or finding.move_idx is None:
        return None
    played = reconstruct_side_turn_dist(
        st.pre_state, st.actions, advancement_choice=advancement_choice)
    proposed_actions = _reorder_before(
        st.actions, finding.move_idx, finding.attack_idx)
    proposed = reconstruct_side_turn_dist(
        st.pre_state, proposed_actions, advancement_choice=advancement_choice)
    v_played = _expected_value(played, value_fn)
    v_proposed = _expected_value(proposed, value_fn)
    if v_played is None or v_proposed is None:
        return None
    return v_proposed - v_played


def advice_signals(st: SideTurn, value_fn: ValueFn, *,
                   advancement_choice: str = "uniform") -> List[AdviceSignal]:
    """All Tier-1 advice signals for a played side-turn, each with its
    value-net-judged delta_v (None where unjudgeable)."""
    out: List[AdviceSignal] = []
    for motif, gen in TIER1_GENERATORS.items():
        findings, _inc = gen(st)
        for f in findings:
            if f.attack_idx is None or f.move_idx is None:
                continue
            dv = delta_v_for_finding(st, f, value_fn,
                                     advancement_choice=advancement_choice)
            out.append(AdviceSignal(
                motif=f.motif, tier="tier1",
                game_id=f.game_id, turn=f.turn, side=f.side,
                proposed_action=st.actions[f.move_idx],
                divergence_action=st.actions[f.attack_idx],
                attacker_pos=f.attacker_pos, defender_pos=f.defender_pos,
                gain_vector=f.vector, delta_v=dv))
    return out
