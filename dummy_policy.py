"""Hard-coded policy used to exercise the end-to-end IPC path.

This is not an AI. It picks deterministic, visible actions so we can
verify that state flows from Wesnoth to Python, actions flow back, and
the Lua action executor dispatches them correctly. The transformer
(Phase 2 material) isn't involved.

IMPORTANT constraint discovered during Phase 1 testing: Wesnoth's AI
scheduler blacklists a CA whose execution failed to change game state —
that includes rejected actions (invalid_move, invalid_recruit_location,
etc.). So a failed action burns the turn. This policy therefore has ONE
shot per turn and must pick an action with a high probability of
succeeding.

Rules, evaluated in order:

  1. If we have fewer than BOOTSTRAP_UNITS units, the leader has moves,
     we have the gold, and there's an empty *castle* hex near the
     leader's keep, recruit the first recruit type there.
  2. Else pick the first owned unit with moves remaining and step it
     to some empty adjacent hex.
  3. Else end the turn.

Adjacency is approximated with a Cartesian ring — not all six hex
neighbors are represented, but most resolve to real ones. Wesnoth's
check_move has its own nearest-valid-hex fallback so misfires usually
succeed anyway. Castle targeting uses a Chebyshev-distance box around
the leader: Wesnoth's real rule is path-connectivity through castle
tiles, but the small box approximates that well for the layouts we
care about.

Action format is the *internal* shape: a dict with `start_hex` /
`target_hex` as `Position` objects (0-indexed). The caller
(`game_manager.convert_action_to_json`) turns those into wire-format
1-indexed flat `start_x` / `start_y` fields.
"""

from typing import Dict, Iterable, List, Optional

from classes import GameState, Position, SideInfo, TerrainModifiers, Unit

# Cheapest dwarvish/saurian recruit is 14g. Pad so we don't try to
# recruit without the cash.
_MIN_RECRUIT_GOLD = 20

# Stop recruiting once we've bootstrapped to this many units. Beyond
# that, our single-attempt-per-turn is better spent on moves.
_BOOTSTRAP_UNITS = 3

# How far from the leader we'll consider a castle tile a valid recruit
# target (Chebyshev distance). Real rule is "connected through castle
# tiles to the keep"; this is a cheap approximation good enough for the
# starting-keep layouts on 2p_Caves_of_the_Basilisk.
_CASTLE_SEARCH_RADIUS = 3

# Cartesian neighborhood (not true hex axial adjacency).
_ADJACENT_OFFSETS = [(1, 0), (-1, 0), (0, 1), (0, -1),
                     (1, 1), (-1, -1), (1, -1), (-1, 1)]


class DummyPolicy:
    """Stateless scripted policy. Not trainable."""

    trainable = False

    def select_action(self, game_state: GameState) -> Dict:
        current_side = game_state.global_info.current_side
        my_units = [u for u in game_state.map.units if u.side == current_side]
        leader = next((u for u in my_units if u.is_leader), None)
        if leader is None:
            return {'type': 'end_turn'}

        side_info = game_state.sides[current_side - 1]

        recruit = self._try_recruit(leader, side_info, game_state, my_units)
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
        my_units: List[Unit],
    ) -> Optional[Dict]:
        if len(my_units) >= _BOOTSTRAP_UNITS:
            return None
        if leader.current_moves <= 0:
            return None
        if not side_info.recruits:
            return None
        if side_info.current_gold < _MIN_RECRUIT_GOLD:
            return None
        target = self._find_nearby_empty_castle(leader.position, game_state)
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

    def _find_nearby_empty_castle(
        self,
        leader_pos: Position,
        game_state: GameState,
    ) -> Optional[Position]:
        """An unoccupied castle hex near the leader's keep."""
        occupied = {(u.position.x, u.position.y) for u in game_state.map.units}
        lx, ly = leader_pos.x, leader_pos.y

        # Outward ring-by-ring so closer castles win.
        candidates = [
            h for h in game_state.map.hexes
            if TerrainModifiers.CASTLE in h.modifiers
            and (h.position.x, h.position.y) not in occupied
        ]
        for radius in range(1, _CASTLE_SEARCH_RADIUS + 1):
            for h in candidates:
                p = h.position
                if max(abs(p.x - lx), abs(p.y - ly)) == radius:
                    return Position(p.x, p.y)
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
