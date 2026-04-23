"""Hard-coded policy used to exercise the end-to-end IPC path.

This is not an AI. It picks deterministic, visible actions so we can
verify that state flows from Wesnoth to Python, actions flow back, and
the Lua action executor dispatches them correctly. The transformer
(Phase 2 material) isn't involved.

Rules, evaluated in order every time we're asked for an action:

  1. If our leader still has moves and we have enough gold and at least
     one empty adjacent hex, recruit the first recruit type there.
  2. Else pick the first owned unit with moves remaining and step it to
     some empty adjacent hex.
  3. Else end the turn.

Adjacency is approximated with a cartesian ring — not all six hex
neighbors are represented, but at least one usually resolves to a real
neighbor and Wesnoth's check_move/check_recruit will reject any that
don't. That rejection showing up in `.out.log` is itself useful Phase 1
signal.

Action format is the *internal* shape: a dict with `start_hex` /
`target_hex` as `Position` objects (0-indexed). The caller
(`game_manager.convert_action_to_json`) turns those into wire-format
1-indexed flat `start_x` / `start_y` fields.
"""

from typing import Dict, Iterable, List, Optional

from classes import GameState, Position, SideInfo, Unit

# Cheapest dwarvish/saurian recruit is 14g, pad a bit so we don't
# accidentally try to recruit without the gold.
_MIN_RECRUIT_GOLD = 20

# Cartesian neighborhood (not true hex adjacency — Wesnoth uses
# offset-axial which depends on column parity). Good enough for a
# "find SOMEWHERE legal to step" policy.
_ADJACENT_OFFSETS = [(1, 0), (-1, 0), (0, 1), (0, -1),
                     (1, 1), (-1, -1), (1, -1), (-1, 1)]


class DummyPolicy:
    """Stateless scripted policy. Not trainable."""

    # Flag we'll teach game_manager / training-loop code to look at.
    # A real policy will be an object that both select_action and
    # train_step in one place; this one has nothing to learn.
    trainable = False

    def select_action(self, game_state: GameState) -> Dict:
        current_side = game_state.global_info.current_side
        my_units = [u for u in game_state.map.units if u.side == current_side]
        leader = next((u for u in my_units if u.is_leader), None)
        if leader is None:
            return {'type': 'end_turn'}

        side_info = game_state.sides[current_side - 1]

        recruit = self._try_recruit(leader, side_info, game_state)
        if recruit is not None:
            return recruit

        move = self._try_move(my_units, game_state)
        if move is not None:
            return move

        return {'type': 'end_turn'}

    def _try_recruit(
        self,
        leader: Unit,
        side_info: SideInfo,
        game_state: GameState,
    ) -> Optional[Dict]:
        if leader.current_moves <= 0:
            return None
        if not side_info.recruits:
            return None
        if side_info.current_gold < _MIN_RECRUIT_GOLD:
            return None
        target = self._first_empty_adjacent(leader.position, game_state)
        if target is None:
            return None
        return {
            'type': 'recruit',
            'unit_type': side_info.recruits[0],
            'target_hex': target,
        }

    def _try_move(
        self,
        my_units: Iterable[Unit],
        game_state: GameState,
    ) -> Optional[Dict]:
        for u in my_units:
            if u.current_moves <= 0:
                continue
            target = self._first_empty_adjacent(u.position, game_state)
            if target is None:
                continue
            return {
                'type': 'move',
                'start_hex': u.position,
                'target_hex': target,
            }
        return None

    def _first_empty_adjacent(
        self,
        pos: Position,
        game_state: GameState,
    ) -> Optional[Position]:
        on_map = {(h.position.x, h.position.y) for h in game_state.map.hexes}
        occupied = {(u.position.x, u.position.y) for u in game_state.map.units}
        for dx, dy in _ADJACENT_OFFSETS:
            nx, ny = pos.x + dx, pos.y + dy
            if (nx, ny) in on_map and (nx, ny) not in occupied:
                return Position(nx, ny)
        return None
