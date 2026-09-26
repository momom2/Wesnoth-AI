"""Small 1.18 multiplayer replays written from scratch, for tests of the
extraction, the game-end cuts and the labels that need no corpus: a flat
grass board, two human sides and the [command]s a test gives.

Coordinates are WML's, 1-indexed, as a replay carries them.
"""
import bz2
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

WIDTH, HEIGHT = 12, 8           # playable hexes; the board adds a border of 1


def _map_data() -> str:
    rows = [", ".join(["Gg"] * (WIDTH + 2)) for _ in range(HEIGHT + 2)]
    return "border_size=1\nusage=map\n\n" + "\n".join(rows)


def side_block(side: int, player: str, units: Sequence[Tuple[str, int, int, bool]], *,
               controller: str = "human", recruit: str = "Spearman,Cavalryman",
               gold: int = 100) -> str:
    """A [side]: `units` are (type, x, y, is_leader)."""
    lines = ["    [side]", f'        side="{side}"', f'        controller="{controller}"',
             f'        current_player="{player}"', f'        name="{player}"',
             f'        player_id="{player}"', '        faction="Loyalists"',
             f'        gold="{gold}"', '        fog="no"', '        shroud="no"',
             f'        recruit="{recruit}"']
    for unit_type, x, y, leader in units:
        lines += ["        [unit]", f'            type="{unit_type}"', f'            x="{x}"',
                  f'            y="{y}"', f'            canrecruit="{"yes" if leader else "no"}"',
                  "        [/unit]"]
    lines.append("    [/side]")
    return "\n".join(lines)


def two_sides(p1: str = "alice", p2: str = "bob", *, controller2: str = "human",
              extra1: Sequence[Tuple[str, int, int, bool]] = (),
              extra2: Sequence[Tuple[str, int, int, bool]] = ()) -> List[str]:
    """Two sides with a Lieutenant leader each (x 2 and x 11 of row 4)."""
    return [side_block(1, p1, [("Lieutenant", 2, 4, True), *extra1]),
            side_block(2, p2, [("Lieutenant", 11, 4, True), *extra2], controller=controller2)]


def replay_text(sides: Sequence[str], commands: Sequence[str],
                scenario_id: str = "test_board") -> str:
    body = "\n".join(f"    [command]\n{c}\n    [/command]" for c in commands)
    return "\n".join([
        'version="1.18.4"', 'era_id="era_default"',
        "[replay_start]", f'    id="{scenario_id}"', '    random_start_time="no"',
        f'    map_data="{_map_data()}"', *sides, "[/replay_start]",
        "[replay]", body, "[/replay]", ""])


def write_replay(path: Path, sides: Sequence[str], commands: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bz2.compress(replay_text(sides, commands).encode("utf-8")))
    return path


# ---- [command] bodies ------------------------------------------------

def init_side(side: int) -> str:
    return f'        [init_side]\n            side_number="{side}"\n        [/init_side]'


def end_turn() -> str:
    return "        [end_turn]\n        [/end_turn]"


def move(side: int, path: Sequence[Tuple[int, int]],
         final: Optional[Tuple[int, int]] = None,
         stopped_early: Optional[str] = None) -> str:
    """A [move] along `path` whose [checkup] records `final` (the path's
    end by default) and, when given, `stopped_early`."""
    xs = ",".join(str(x) for x, _ in path)
    ys = ",".join(str(y) for _, y in path)
    fx, fy = final if final is not None else path[-1]
    early = f"\n                stopped_early={stopped_early}" if stopped_early else ""
    return (f"        from_side={side}\n"
            f'        [move]\n            x="{xs}"\n            y="{ys}"\n        [/move]\n'
            f"        [checkup]\n            [result]\n                final_hex_x={fx}\n"
            f"                final_hex_y={fy}{early}\n            [/result]\n"
            f"        [/checkup]")


def server(message: str) -> str:
    return (f'        [speak]\n            id="server"\n            message="{message}"\n'
            f"        [/speak]")


def surrender(side: int) -> str:
    """The [surrender] command of the player of `side` (the command
    carries the 0-based viewing team)."""
    return f"        [surrender]\n            side_number={side - 1}\n        [/surrender]"


def turn(side: int, *moves: str) -> List[str]:
    """One side turn: init_side, the given commands, end_turn."""
    return [init_side(side), *moves, end_turn()]

