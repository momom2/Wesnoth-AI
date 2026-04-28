"""Opener-gating policy wrapper.

A scripted opener that delegates to a learned policy after the
opener finishes. The user's design goal: forcing specific openers
(rush, defensive, village-grab, ...) is a config flip, not a
re-trained model.

Pattern::

    from tools.openers import Opener, OpenerPolicy
    from transformer_policy import TransformerPolicy

    def grab_first_village(state, side):
        # Return an action dict if this opener wants to act this
        # decision, else None (falls through to the base policy).
        from classes import Position
        for u in state.map.units:
            if u.side == side and u.is_leader:
                # Move leader one hex south as a placeholder example.
                return {
                    "type": "move",
                    "start_hex": u.position,
                    "target_hex": Position(u.position.x, u.position.y + 1),
                }
        return None

    opener = Opener(
        name="grab_village",
        moves=[grab_first_village, grab_first_village],  # one move/side/turn
        sides=(1, 2),                                    # both sides
    )
    base = TransformerPolicy()
    policy = OpenerPolicy(base=base, opener=opener)

    while not sim.done:
        action = policy.select_action(sim.gs, game_label="g0")
        sim.step(action)

How the wrapper decides whether to fire its scripted move:

  - It tracks the per-(game_label, side) opener-step index. On the
    first call for a (game, side), index=0; each successful opener
    move increments it.
  - At index k, if k < len(opener.moves), call opener.moves[k]
    on the state. If that returns an action, USE it (and bump the
    index). If it returns None, fall through to base for THIS
    decision (no index bump -- gives the opener a chance to try
    again next decision, useful if the opener is gated on a
    condition like "leader on keep").
  - When the index reaches len(opener.moves), the opener is "done"
    for that game-side and every subsequent call goes to the base.
  - `sides`: only the listed sides get the scripted opener. Side-1-
    only openers (e.g. testing rush plays) just pass `sides=(1,)`.

Design choices:

  - The opener's moves are functions (state, side) -> Optional[dict],
    not pre-baked action dicts, because legality depends on state
    that isn't known until we get there. A "move leader to (5,5)"
    opener might be illegal if an enemy is on (5,5) -- the function
    can detect that and return None.

  - We forward `game_label` to the base policy unchanged. Trainable
    policies (TransformerPolicy) need it for trajectory bookkeeping.
    Crucially, when the opener fires its OWN move (not delegating
    to base), the base policy doesn't see that step at all -- so
    the opener's gradient-tracked log_prob can't fire either, which
    is the point: scripted moves bypass the learned policy.

  - "Trainable" hooks (observe, train_step, save_checkpoint,
    load_checkpoint, drop_pending) pass straight through to base.
    The wrapper itself has no learnable parameters.

  - The fired-set is keyed by (game_label, side); calling
    `reset_game(game_label)` between games clears that game's state
    so the opener fires fresh on the next run.

CLI plumbing (TODO): once an opener registry is built, expose
`--opener-spec NAME` in `tools/sim_self_play.py` so cluster jobs
can flip openers without code edits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from classes import GameState


# Type alias for an opener move: takes (state, acting_side), returns
# either an action dict (the wrapper uses it) or None (delegate to
# base for this decision).
OpenerMove = Callable[[GameState, int], Optional[Dict]]


@dataclass
class Opener:
    """A scripted sequence of moves applied for the first
    `len(moves)` opener-fires per (game_label, side)."""
    name:  str
    moves: List[OpenerMove]
    # Sides this opener runs for. Default both sides (most openers
    # are symmetric "village grab" / "rush" plays).
    sides: Tuple[int, ...] = (1, 2)


class OpenerPolicy:
    """Policy wrapper: scripted opener for the first K decisions of
    each game-side, then delegate to `base`.

    Mirrors the `Policy` Protocol's `select_action`. Forwards every
    other policy method (observe / train_step / save_checkpoint /
    load_checkpoint / drop_pending) to `base` so trainable wrappers
    keep working unchanged.
    """

    def __init__(self, *, base, opener: Opener):
        self._base   = base
        self._opener = opener
        # (game_label, side) -> next opener-move index to try.
        self._cursor: Dict[Tuple[str, int], int] = {}

    # ------------------------------------------------------------------
    # Policy Protocol
    # ------------------------------------------------------------------

    def select_action(
        self,
        game_state: GameState,
        *,
        game_label: str = "default",
    ) -> Dict:
        side = game_state.global_info.current_side
        if side in self._opener.sides:
            key = (game_label, side)
            idx = self._cursor.get(key, 0)
            if idx < len(self._opener.moves):
                action = self._opener.moves[idx](game_state, side)
                if action is not None:
                    # Opener fired. Advance the cursor so the next
                    # decision tries the NEXT scripted move. The
                    # learned policy is bypassed for this step --
                    # no gradient bookkeeping happens on it.
                    self._cursor[key] = idx + 1
                    return action
                # Move returned None -- gate not satisfied (e.g. the
                # leader isn't on the keep yet). Fall through to base
                # WITHOUT advancing the cursor; we'll retry this
                # opener-step next decision.
        return self._base.select_action(game_state, game_label=game_label)

    # ------------------------------------------------------------------
    # Per-game lifecycle
    # ------------------------------------------------------------------

    def reset_game(self, game_label: str) -> None:
        """Clear opener-cursor state for `game_label`. Call between
        games when re-using the same OpenerPolicy instance so the
        next game starts fresh from move 0."""
        stale = [k for k in self._cursor if k[0] == game_label]
        for k in stale:
            del self._cursor[k]

    # ------------------------------------------------------------------
    # Forwarding for trainable-policy hooks (duck-typed; only forward
    # if the base actually has them so a scripted base doesn't see
    # surprise method calls).
    # ------------------------------------------------------------------

    def observe(self, *args, **kwargs):
        fn = getattr(self._base, "observe", None)
        if fn is not None:
            return fn(*args, **kwargs)

    def drop_pending(self, *args, **kwargs):
        fn = getattr(self._base, "drop_pending", None)
        if fn is not None:
            return fn(*args, **kwargs)

    def train_step(self, *args, **kwargs):
        fn = getattr(self._base, "train_step", None)
        if fn is not None:
            return fn(*args, **kwargs)

    def save_checkpoint(self, *args, **kwargs):
        fn = getattr(self._base, "save_checkpoint", None)
        if fn is not None:
            return fn(*args, **kwargs)

    def load_checkpoint(self, *args, **kwargs):
        fn = getattr(self._base, "load_checkpoint", None)
        if fn is not None:
            return fn(*args, **kwargs)

    @property
    def trainable(self) -> bool:
        return bool(getattr(self._base, "trainable", False))


# ---------------------------------------------------------------------
# Built-in opener moves
# ---------------------------------------------------------------------
# A small library of building-block moves that opener authors can
# compose. Each is an `OpenerMove` -- (state, side) -> Optional[Dict].
# Keep this minimal; users can write their own arbitrary moves.

def recruit_type(unit_type: str) -> OpenerMove:
    """Opener move: recruit `unit_type` on a free castle hex adjacent
    to our leader. Returns None if our leader isn't on a keep, no
    free castle adjacent, or recruit list doesn't include unit_type."""
    from classes import Position, TerrainModifiers
    from tools.abilities import hex_neighbors

    def _move(state: GameState, side: int) -> Optional[Dict]:
        # Leader.
        leader = next(
            (u for u in state.map.units if u.side == side and u.is_leader),
            None,
        )
        if leader is None:
            return None
        # Leader must be on a keep hex.
        on_keep = False
        for h in state.map.hexes:
            if (h.position.x, h.position.y) == (leader.position.x,
                                                leader.position.y):
                on_keep = TerrainModifiers.KEEP in h.modifiers
                break
        if not on_keep:
            return None
        # unit_type must be in our recruit list.
        side_idx = side - 1
        if not (0 <= side_idx < len(state.sides)):
            return None
        if unit_type not in state.sides[side_idx].recruits:
            return None
        # Find a free castle hex adjacent to the leader.
        occupied = {(u.position.x, u.position.y) for u in state.map.units}
        for nx, ny in hex_neighbors(leader.position.x, leader.position.y):
            if (nx, ny) in occupied:
                continue
            for h in state.map.hexes:
                if (h.position.x, h.position.y) != (nx, ny):
                    continue
                if TerrainModifiers.CASTLE in h.modifiers:
                    return {
                        "type":       "recruit",
                        "unit_type":  unit_type,
                        "target_hex": Position(nx, ny),
                    }
                break
        return None
    return _move


def end_turn() -> OpenerMove:
    """Opener move: unconditionally end the turn."""
    def _move(state: GameState, side: int) -> Optional[Dict]:
        return {"type": "end_turn"}
    return _move
