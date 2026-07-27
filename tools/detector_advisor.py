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

import copy as _copy                                              # noqa: E402

from tools.swap_detector import (                                 # noqa: E402
    SideTurn, Finding, reconstruct_side_turn_dist,
    backstab_setup_findings, leadership_setup_findings,
)
from tools.replay_dataset import _apply_command                   # noqa: E402
from wesnoth_ai.classes import GameState, Position                 # noqa: E402

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


def model_value_fn(model, encoder) -> ValueFn:
    """Wrap a WesnothModel + GameStateEncoder into a `ValueFn`: gs -> the
    C51 value head's mean (in [-1, +1]) from gs's ACTING-side perspective.

    - RAW value, NOT the MCTS aux-adjusted one: the aux bonus is a material
      training crutch (see the eval contract) and would contaminate ΔV.
    - Acting-side perspective is exactly what the advisor wants: the encoder
      frames the state from `current_side`, and reconstruct_side_turn_dist
      does not end the turn, so played and proposed end-states share the
      acting side -> ΔV needs no sign flip.
    - Pass the INFERENCE (eval-mode) model so dropout doesn't make ΔV
      stochastic; this wrapper does not toggle the model's mode (no side
      effects on a possibly-shared module)."""
    import torch

    def f(gs: GameState) -> float:
        with torch.no_grad():
            out = model(encoder.encode(gs))
        return float(out.value.squeeze().item())
    return f


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
    advancement_choice: str = "uniform", window: bool = True,
) -> Optional[float]:
    """V(proposed ordering) - V(played ordering) scored by `value_fn`. None
    if the finding carries no reorder indices or either reconstruction bails
    (advancement past cap / blow-up) -> the caller falls back to
    advice-token-only.

    `window=True` (default) reconstructs only the REORDER WINDOW
    [min(attack,move) .. max(attack,move)] distributionally, CONDITIONING on
    the recorded prefix (applied deterministically with its recorded seeds).
    Actions before the window are identical in both orderings, so the prefix
    is a common factor; the suffix is identical too and its value-to-go is
    what value_fn estimates at the window end. Full-side-turn reconstruction
    (`window=False`) blows up on real games (the joint over every combat in
    the turn) -- offline validation measured 0/10 findings judgeable -- so
    windowing is what makes the signal have coverage."""
    ai, mi = finding.attack_idx, finding.move_idx
    if ai is None or mi is None:
        return None
    if window:
        lo, hi = min(ai, mi), max(ai, mi)
        start = _copy.deepcopy(st.pre_state)
        for cmd in st.actions[:lo]:                 # recorded prefix (realized)
            _apply_command(start, cmd)
        played_actions = st.actions[lo:hi + 1]
        proposed_actions = _reorder_before(played_actions, mi - lo, ai - lo)
    else:
        start = st.pre_state
        played_actions = st.actions
        proposed_actions = _reorder_before(st.actions, mi, ai)
    played = reconstruct_side_turn_dist(
        start, played_actions, advancement_choice=advancement_choice)
    proposed = reconstruct_side_turn_dist(
        start, proposed_actions, advancement_choice=advancement_choice)
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


# =====================================================================
# Prospective advisor (decision-time) + model bridge
# =====================================================================
# The retrospective advice_signals() above judges a PLAYED side-turn (for
# offline validation / the exploration seed). At decision time the model
# instead needs PROSPECTIVE advice: among the currently available actions,
# which setup move would enable a Tier-1 certificate? These become the
# encoder advice tokens the model conditions on (with its learnable gate).

# Motif -> id for the model's advice_motif_embed (must stay < N_ADVICE_MOTIFS).
ADVICE_MOTIF_IDS = {"backstab_setup": 0, "leadership_setup": 1}


@dataclass
class AdviceOpportunity:
    """One decision-time setup opportunity: doing the setup move (mover ->
    dest) now would enable a Tier-1 certificate for attacker -> target."""
    motif:        str
    mover_pos:    Tuple[int, int]     # the setup move's mover, current hex
    dest_pos:     Tuple[int, int]     # the setup move's destination hex
    target_pos:   Tuple[int, int]     # the enemy the setup helps attack
    attacker_pos: Tuple[int, int]     # the attacker that gains
    gain:         float               # expected enemy-HP drop (certificate mag)
    delta_v:      Optional[float] = None


def prospective_backstab_opportunities(
    gs: GameState, side: Optional[int] = None,
) -> List[AdviceOpportunity]:
    """Own backstab-weapon unit adjacent to an attackable enemy with the
    OPPOSITE hex free and reachable this turn by another own unit -> moving
    that flanker onto the opposite hex first activates the backstab (a
    Tier-1 certificate, DP-verified). Cheap: DP + reach, no value net."""
    from tools.swap_detector import (
        _unit_at, _weapon_has_backstab, opposite_hex, hex_neighbors,
        enumerate_attack_outcomes, compare_distributions, Verdict, _reach,
        _marginal, ATTACK_DIMS)
    from tools.abilities import is_backstab_active
    side = side if side is not None else gs.global_info.current_side
    own = [u for u in gs.map.units if u.side == side]
    enemy_hp = next(d for d in ATTACK_DIMS if d.name == "enemy_hp")
    opps: List[AdviceOpportunity] = []
    for u in own:
        if not _weapon_has_backstab(u.name, 0):
            continue
        for (ex, ey) in hex_neighbors(u.position.x, u.position.y):
            e = _unit_at(gs, (ex, ey))
            if (e is None or e.side == side
                    or is_backstab_active(u, e, gs.map.units)):
                continue
            opp = opposite_hex((ex, ey), (u.position.x, u.position.y))
            if (opp is None or _unit_at(gs, opp) is not None
                    or not (0 <= opp[0] < gs.map.size_x
                            and 0 <= opp[1] < gs.map.size_y)):
                continue
            action = {"type": "attack", "start_hex": u.position,
                      "target_hex": e.position, "attack_index": 0}
            d_base = enumerate_attack_outcomes(gs, action,
                                               advancement_choice="uniform")
            if d_base is None:
                continue
            g2 = _copy.deepcopy(gs)
            ph = _copy.deepcopy(u)
            ph.position = Position(opp[0], opp[1])
            ph.id = "adv_phantom_flanker"
            g2.map.units.add(ph)
            d_cand = enumerate_attack_outcomes(g2, action,
                                               advancement_choice="uniform")
            if d_cand is None:
                continue
            if compare_distributions(d_base, d_cand).verdict \
                    is not Verdict.STRICTLY_BETTER:
                continue
            mover = next(
                (c for c in own if c.id != u.id and int(c.current_moves) > 0
                 and opp in _reach(gs, c).landable), None)
            if mover is None:
                continue
            base_e = sum(v * p for v, p in _marginal(d_base, enemy_hp.value).items())
            cand_e = sum(v * p for v, p in _marginal(d_cand, enemy_hp.value).items())
            opps.append(AdviceOpportunity(
                "backstab_setup",
                (mover.position.x, mover.position.y), opp, (ex, ey),
                (u.position.x, u.position.y), max(0.0, base_e - cand_e)))
    # Deterministic order: gs.map.units iterates in set order, so sort the
    # opportunities (the model's cross-attention is order-invariant, but a
    # stable order keeps runs reproducible + tests robust).
    opps.sort(key=lambda o: (o.attacker_pos, o.dest_pos, o.mover_pos))
    return opps


def prospective_leadership_opportunities(
    gs: GameState, side: Optional[int] = None,
) -> List[AdviceOpportunity]:
    """Own attacker adjacent to an attackable enemy, NOT currently under
    leadership, with a higher-level leadership ally that can reach a free
    hex adjacent to the attacker this turn -> moving that leader adjacent
    first activates the +25%/level leadership bonus (a Tier-1 certificate,
    DP-verified). Same shape as backstab; cheap (DP + reach)."""
    from tools.swap_detector import (
        _unit_at, hex_neighbors, enumerate_attack_outcomes,
        compare_distributions, Verdict, _reach, _marginal, ATTACK_DIMS,
        _has_leadership, _unit_level)
    from tools.abilities import leadership_bonus
    side = side if side is not None else gs.global_info.current_side
    own = [u for u in gs.map.units if u.side == side]
    leaders = [u for u in own
               if _has_leadership(u.name) and int(u.current_moves) > 0]
    if not leaders:
        return []
    enemy_hp = next(d for d in ATTACK_DIMS if d.name == "enemy_hp")
    opps: List[AdviceOpportunity] = []
    for u in own:
        if leadership_bonus(u, gs.map.units) != 0:      # already boosted
            continue
        adj_enemy = next(
            ((ex, ey) for (ex, ey) in hex_neighbors(u.position.x, u.position.y)
             if (e := _unit_at(gs, (ex, ey))) is not None and e.side != side),
            None)
        if adj_enemy is None:
            continue
        free_adj = [h for h in hex_neighbors(u.position.x, u.position.y)
                    if _unit_at(gs, h) is None
                    and 0 <= h[0] < gs.map.size_x and 0 <= h[1] < gs.map.size_y]
        u_lvl = _unit_level(u.name)
        for lead in leaders:
            if lead.id == u.id or _unit_level(lead.name) <= u_lvl:
                continue
            r = _reach(gs, lead)
            dest = next((h for h in free_adj if h in r.landable), None)
            if dest is None:
                continue
            action = {"type": "attack", "start_hex": u.position,
                      "target_hex": Position(adj_enemy[0], adj_enemy[1]),
                      "attack_index": 0}
            d_base = enumerate_attack_outcomes(gs, action,
                                               advancement_choice="uniform")
            if d_base is None:
                continue
            g2 = _copy.deepcopy(gs)
            ph = _copy.deepcopy(lead)
            ph.position = Position(dest[0], dest[1])
            ph.id = "adv_phantom_leader"
            g2.map.units.add(ph)
            d_cand = enumerate_attack_outcomes(g2, action,
                                               advancement_choice="uniform")
            if d_cand is None or compare_distributions(d_base, d_cand).verdict \
                    is not Verdict.STRICTLY_BETTER:
                continue
            base_e = sum(v * p for v, p in _marginal(d_base, enemy_hp.value).items())
            cand_e = sum(v * p for v, p in _marginal(d_cand, enemy_hp.value).items())
            opps.append(AdviceOpportunity(
                "leadership_setup",
                (lead.position.x, lead.position.y), dest, adj_enemy,
                (u.position.x, u.position.y), max(0.0, base_e - cand_e)))
            break                                       # one leader per attacker
    opps.sort(key=lambda o: (o.attacker_pos, o.dest_pos, o.mover_pos))
    return opps


def prospective_opportunities(
    gs: GameState, side: Optional[int] = None,
) -> List[AdviceOpportunity]:
    """All Tier-1 decision-time opportunities (backstab + leadership setups)."""
    return (prospective_backstab_opportunities(gs, side)
            + prospective_leadership_opportunities(gs, side))


def opportunities_to_features(encoded, opps: List[AdviceOpportunity]):
    """Resolve opportunities against `encoded` into the model builder's
    inputs: (motif_ids[A], feats[A,4], mover_uidx[A], dest_hidx[A]). Drops
    any opportunity whose mover/dest isn't in the encoded frame (fogged
    mover, off-board dest). Returns tensors on the encoded's device."""
    import torch
    dev = encoded.unit_tokens.device
    upos = {(p.x, p.y): i for i, p in enumerate(encoded.unit_positions)}
    hpos = encoded.pos_to_hex
    motif_ids, feats, muidx, dhidx = [], [], [], []
    for o in opps:
        ui = upos.get(o.mover_pos)
        hi = hpos.get(o.dest_pos)
        mid = ADVICE_MOTIF_IDS.get(o.motif)
        if ui is None or hi is None or mid is None:
            continue
        dv = 0.0 if o.delta_v is None else float(o.delta_v)
        motif_ids.append(mid)
        feats.append([1.0, float(o.gain), dv, 0.0 if o.delta_v is None else 1.0])
        muidx.append(ui)
        dhidx.append(hi)
    return (torch.tensor(motif_ids, dtype=torch.long, device=dev),
            torch.tensor(feats, dtype=torch.float32, device=dev).reshape(-1, 4),
            torch.tensor(muidx, dtype=torch.long, device=dev),
            torch.tensor(dhidx, dtype=torch.long, device=dev))


def encode_with_advice(encoder, model, gs: GameState):
    """Encode `gs` and, if the model has the advice path, attach prospective
    advice tokens. Shared by self-play (acting) and the trainer reforward
    (learning) so advice is consistent between the two. Returns (encoded,
    opportunities)."""
    encoded = encoder.encode(gs)
    if not getattr(model, "has_advice", False):
        return encoded, []
    opps = prospective_opportunities(gs)
    motif_ids, feats, muidx, dhidx = opportunities_to_features(encoded, opps)
    encoded.advice_tokens = model.build_advice_tokens(
        encoded, motif_ids, feats, muidx, dhidx)
    return encoded, opps
