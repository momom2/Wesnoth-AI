"""The live mirror (tools/live_mirror.py): a game replayed in the simulator
from the log a live engine writes for it."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_test_helpers import scenario_setup  # noqa: E402
from tools.live_mirror import (SYNC_MARKER, AiStop, EngineLogReader, EngineMessage,  # noqa: E402
                               LiveMirror, LoggedCommand, MirrorDivergence, SyncMarker,
                               engine_commands)
from wesnoth_ai import game_core  # noqa: E402

pytestmark = pytest.mark.skipif(game_core.game_core_class() is None,
                                reason="needs the wesnoth_core wheel of this source's phase")

STAMP = "20261004 19:37:45"

# The engine's own text for two commands, from a 1.18.8 game run with
# --log-info=replay,random (2026-10-04), the draws shortened.
REAL_LOG = f"""{STAMP} info replay: add_synced_command:
from_side = 1
[recruit]
	type = Elvish Scout
	x = 15
	y = 23
	[from]
		x = 15
		y = 24
	[/from]
[/recruit]

{STAMP} info replay: set_scontext_synced_base::set_scontext_synced_base
{STAMP} info random: randomness::rng::next_random_impl returned 1555323485
{STAMP} info random: randomness::rng::next_random_impl returned 2355955719
{STAMP} error config: Multiple [unit_type]s with id=ROLDune Airist encountered.
{STAMP} error wml: Invalid WML found: [message] missing required speaker
{STAMP} info ai/actions: start of execution of:  stopunit by side 2 : remove movement and  remove attacks from unit on location 40,5
{STAMP} info replay: add_synced_command:
from_side = 2
[attack]
	defender_type = Elvish Archer
	defender_weapon = 0
	weapon = 1
	[source]
		x = 15
		y = 11
	[/source]
	[destination]
		x = 16
		y = 11
	[/destination]
[/attack]

{STAMP} info random: randomness::rng::next_random_impl returned 741086739
{STAMP} info engine: unit at position 16,11 chose advancement number 1
{STAMP} error scripting/lua/user: {SYNC_MARKER} 3
"""


def test_the_reader_reads_the_engines_log_in_any_chunks():
    whole = EngineLogReader().feed(REAL_LOG)
    split = EngineLogReader()
    chunked = split.feed(REAL_LOG[:157]) + split.feed(REAL_LOG[157:400]) + split.feed(REAL_LOG[400:])
    for events in (whole, chunked):
        message, stop, recruit, attack, marker = events
        assert message == EngineMessage("error", "wml", "Invalid WML found: [message] missing required speaker")
        assert stop == AiStop(2, 40, 5, movement=True, attacks=True)
        assert (recruit.from_side, recruit.tag, recruit.block.attrs["type"]) == (1, "recruit", "Elvish Scout")
        assert recruit.draws == [1555323485, 2355955719]
        assert (attack.from_side, attack.tag, attack.draws, attack.advancements) == (2, "attack", [741086739], [1])
        assert attack.block.child("destination").attrs == {"x": "16", "y": "11"}
        assert marker == SyncMarker(3)


def _draws(seed_hex: str, n: int):
    from wesnoth_ai.combat import MTRng
    rng = MTRng(seed_hex)
    return [rng.get_next_random() for _ in range(n)]


def _command_block(rc) -> str:
    """The WML of one simulator command as the engine logs it (1-based)."""
    cmd = rc.cmd
    if rc.kind == "move":
        xs = ",".join(str(v + 1) for v in cmd[1])
        ys = ",".join(str(v + 1) for v in cmd[2])
        return f"[move]\n\tskip_sighted = all\n\tx = {xs}\n\ty = {ys}\n[/move]"
    if rc.kind == "attack":
        return (f"[attack]\n\tdefender_weapon = {cmd[6]}\n\tweapon = {cmd[5]}\n"
                f"\t[source]\n\t\tx = {cmd[1] + 1}\n\t\ty = {cmd[2] + 1}\n\t[/source]\n"
                f"\t[destination]\n\t\tx = {cmd[3] + 1}\n\t\ty = {cmd[4] + 1}\n\t[/destination]\n[/attack]")
    return f"[recruit]\n\ttype = {cmd[1]}\n\tx = {cmd[2] + 1}\n\ty = {cmd[3] + 1}\n[/recruit]"


def engine_log_of(history):
    """The log a live engine writes for a simulated game: each command's
    WML, the numbers its seed draws (one a strike, the checkup's entries
    with a chance; a recruit's gender, traits and name), the advancement choices; and, where a side's turn
    begins, a sync marker standing for a decision's frame. Returns the
    text and the (turn, side) of each marker."""
    lines, frames, turn = [], [], 0
    for rc in history:
        if rc.kind == "init_side":
            side = int(rc.cmd[1])
            turn += side == 1
            frames.append((turn, side))
            lines.append(f"{STAMP} error scripting/lua/user: {SYNC_MARKER} {len(frames)}")
            continue
        if rc.kind not in ("move", "attack", "recruit"):
            continue
        lines += [f"{STAMP} info replay: add_synced_command: ", f"from_side = {rc.side}",
                  _command_block(rc), ""]
        if rc.kind == "attack":
            n = sum("chance" in entry for entry in rc.extras.get("checkup_strikes") or [])
            lines += [f"{STAMP} info random: randomness::rng::next_random_impl returned {d}"
                      for d in _draws(rc.cmd[7], n)]
            lines += [f"{STAMP} info engine: unit at position 1,1 chose advancement number {choice}"
                      for _, choice in rc.extras.get("advance_choices") or []]
        if rc.kind == "recruit":
            lines += [f"{STAMP} info random: randomness::rng::next_random_impl returned {d}"
                      for d in _draws(rc.cmd[4], 22)]
    return "\n".join(lines) + "\n", frames


def _fresh_sim():
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate
    setup = scenario_setup(1)
    gs = build_scenario_gamestate(setup, experience_modifier=10)
    return WesnothSim(gs, scenario_id=setup.scenario_id, max_turns=8)


@pytest.fixture(scope="module")
def played_game():
    """Den of Onis, to a leader's death on turn 7, between two samplers of a small random
    network, at 10% experience so units level, the simulator drawing
    each advancement among its options."""
    from tools.eval_players import _PolicyPair, _play_one_eval_game
    from tools.raw_player import RawPolicyPlayer
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(0)
    policy = TransformerPolicy(device=torch.device("cpu"), d_model=32, num_layers=1, num_heads=2, d_ff=64)
    sim = _fresh_sim()
    sim.enable_uniform_advancement()
    player = RawPolicyPlayer(policy, 1.0, seed=6, end_turn_offset=-4.0)
    _play_one_eval_game(sim, _PolicyPair(player, "a", 1), _PolicyPair(player, "a", 2), game_label="a")
    kinds = Counter(rc.kind for rc in sim.command_history)
    choices = [c for rc in sim.command_history for _, c in rc.extras.get("advance_choices") or []]
    assert kinds["recruit"] and kinds["move"] and kinds["attack"] and 1 in choices, (kinds, choices)
    return sim


def _mirror(text, frames):
    mirror = LiveMirror(_fresh_sim())
    reader = EngineLogReader()
    for start in range(0, len(text), 997):
        for event in reader.feed(text[start:start + 997]):
            if isinstance(event, SyncMarker):
                mirror.reach_turn(*frames[event.seq - 1])
            else:
                mirror.apply(event)
    for event in reader.close():
        mirror.apply(event)
    return mirror


def test_a_game_mirrored_from_its_engine_log_is_the_game(played_game):
    from wesnoth_ai.classes import state_digest
    text, frames = engine_log_of(played_game.command_history)
    mirror = _mirror(text, frames)
    sim = mirror.sim
    mirror.reach_turn(played_game.turn_number, played_game.current_side)
    assert mirror.commands == sum(rc.kind in ("move", "attack", "recruit") for rc in played_game.command_history)
    assert (sim.done, sim.turn_number) == (played_game.done, played_game.turn_number)
    assert state_digest(sim.gs) == state_digest(played_game.gs)


def test_the_mirror_refuses_a_fight_short_of_a_draw(played_game):
    text, frames = engine_log_of(played_game.command_history)
    attack_at = text.index("[attack]")
    first_draw = text.index("next_random_impl returned", attack_at)
    line_start = text.rindex("\n", 0, first_draw) + 1
    line_end = text.index("\n", first_draw) + 1
    with pytest.raises(MirrorDivergence, match="draws"):
        _mirror(text[:line_start] + text[line_end:], frames)


def test_a_policy_step_becomes_the_engine_commands_it_plays(played_game):
    """A move-to-attack on a fork: the move then the attack, in the
    engine's 1-based hexes and the Lua AI's 1-based weapon index."""
    from tools.wesnoth_sim import RecordedCommand
    history = [RecordedCommand(kind="move", side=1, cmd=["move", [4, 5, 6], [7, 7, 8], 1], extras={}),
               RecordedCommand(kind="attack", side=1, cmd=["attack", 6, 8, 7, 8, 0, 1, "ab", []], extras={}),
               RecordedCommand(kind="end_turn", side=1, cmd=["end_turn"], extras={}),
               RecordedCommand(kind="init_side", side=2, cmd=["init_side", 2], extras={})]
    assert engine_commands(history) == [
        {"type": "move", "from_x": 5, "from_y": 8, "to_x": 7, "to_y": 9},
        {"type": "attack", "from_x": 7, "from_y": 9, "to_x": 8, "to_y": 9, "weapon": 1},
        {"type": "end_turn"}]
    assert isinstance(LoggedCommand(1, "move", None), LoggedCommand)


def test_the_driver_ends_the_game_on_the_command_still_open(played_game, tmp_path):
    """The attack that kills a leader is the engine's last command: no
    later one releases it from the reader, so the driver tries it on a
    fork and applies it when it ends the game."""
    from types import SimpleNamespace
    from tools.live_vs_rca import LiveGame
    assert played_game.ended_by == "leader_killed"
    text, frames = engine_log_of(played_game.command_history)
    game = LiveGame(1, played_game.scenario_id, ("", ""), 1, None, None, tmp_path, 4.0, 8)
    game.mirror = LiveMirror(_fresh_sim())
    game.engine_log = SimpleNamespace(reader=EngineLogReader())
    for event in game.engine_log.reader.feed(text):
        if isinstance(event, SyncMarker):
            game.mirror.reach_turn(*frames[event.seq - 1])
        else:
            game.mirror.apply(event)
    assert game.engine_log.reader.open_command is not None and not game.mirror.sim.done
    game._drain()
    assert (game.mirror.sim.done, game.mirror.sim.winner) == (True, played_game.winner)
