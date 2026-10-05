#!/usr/bin/env python3
"""The reference player against Wesnoth's default AI, live, in a window a
person can watch.

Each game is a real multiplayer game started from the command line (the
launch of tools/scenario_init_oracle.py: the scenario's own sides, the
default era, both factions named, the lobby's side settings): our side is
played by lua/live_stage.lua, the other by the default (RCA) AI. The
simulator mirrors the game from the engine's log (tools/live_mirror.py), so
the player observes it as in a simulated match, its memory included. At
each decision the engine's board and the simulator's are compared field by
field; any difference stops the game as a fidelity defect. The player's
action is planned on a fork of the simulator (a move-to-attack becomes the
move and the attack the simulator would play) and the engine plays those
commands.

    python tools/live_vs_rca.py [--games 10] [--speed 4] [--seed S]

One known difference with a hosted game: a command-line start plays 100%
experience where a lobby plays 70% (scenario_init_oracle's docstring); the
simulator is built at the engine's value, so the two boards agree.
Live Wesnoth runs on this machine, in a window, one game at a time.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools import reference_player  # noqa: E402
from tools.live_mirror import (ENGINE_DEBUG_DOMAINS, ENGINE_LOG_DOMAINS, AiCandidateAction,  # noqa: E402
                               AiStop, EngineLog, EngineMessage, LiveMirror, LoggedCommand,
                               MirrorDivergence, SyncMarker, engine_commands)
from tools.scenario_init_oracle import (applied_experience_modifier, compare, engine_declarations,  # noqa: E402
                                        lobby_parms, our_record, our_setup)
from wesnoth_ai.constants import GAMES_PATH, WESNOTH_USERDATA_PATH  # noqa: E402

log = logging.getLogger("live_vs_rca")

LIVE_AI_CONFIG = "~add-ons/wesnoth_ai/live_ai.cfg"
IPC_GAME_ID = "live"
FRAME_TIMEOUT_S = 600.0        # the default AI's turn, at the watched speed
END_WAIT_S = 20.0              # for the engine's end of scenario before closing it


def launch_args(scenario_id: str, factions, our_side: int, decl) -> list:
    args = ["--multiplayer", f"--scenario={scenario_id}", "--era=era_default",
            "--side", f"1:{factions[0]}", "--side", f"2:{factions[1]}",
            "--controller", "1:ai", "--controller", "2:ai",
            "--ai-config", f"{our_side}:{LIVE_AI_CONFIG}",
            "--exit-at-end", "--log-to-file", f"--log-info={ENGINE_LOG_DOMAINS}",
            f"--log-debug={ENGINE_DEBUG_DOMAINS}"]
    for side, attr, value in lobby_parms(decl):
        args += ["--parm", f"{side}:{attr}:{value}"]
    return args


def board_differences(frame: dict, sim, stopped: set) -> tuple:
    """The fields where the engine's board and the simulator's disagree,
    and the known differences set apart (`known_difference`). `stopped`:
    the (side, x, y) of the units the default AI stopped in its last turn,
    and ("leaders", side) when it took its leaders' movement."""
    fields = compare(frame, our_record(sim.gs))
    diffs, known = {}, []
    for name, f in fields.items():
        if f["agree"] == f["total"]:
            continue
        rest = [d for d in f["diffs"] if not known_difference(name, d, frame, stopped)]
        known += [(name, d) for d in f["diffs"] if known_difference(name, d, frame, stopped)]
        if rest or len(f["diffs"]) < f["total"] - f["agree"]:
            diffs[name] = rest or f["diffs"]
    return diffs, known


def known_difference(field: str, diff: dict, frame: dict, stopped: set) -> bool:
    """A difference the check lets through, understood: a unit the
    default AI stopped (`AiStop`), and its leaders once its
    leader_shares_keep action ran (ai/default/ca.cpp:1688), have no
    movement left and, if they had not moved, the not_moved state that
    keeps their rest (unit.cpp:2784-2791); neither is in the replay, so
    the simulator keeps their movement. Both end at the side's next turn
    start."""
    item = tuple(diff["item"][:3]) if isinstance(diff["item"], (list, tuple)) else ()
    leader = any(u["canrecruit"] and (u["side"], u["x"], u["y"]) == item for u in frame["units"])
    if item not in stopped and not (leader and ("leaders", item[0]) in stopped):
        return False
    if field == "unit.moves":
        return diff["engine"] == 0
    if field == "unit.status":
        return set(diff["engine"]) - set(diff["ours"]) == {"not_moved"} and set(diff["ours"]) <= set(diff["engine"])
    return False


class LiveGame:
    """One live game: the engine, its log, the mirror, the player."""

    def __init__(self, index: int, scenario_id: str, factions, our_side: int, decl, player, out_dir: Path,
                 speed: float, max_turns: int, watched: bool = True, player_label: str = "the network"):
        self.index, self.scenario_id, self.factions, self.our_side = index, scenario_id, factions, our_side
        self.decl, self.player, self.speed, self.max_turns = decl, player, speed, max_turns
        self.watched = watched
        self.player_label = player_label
        self.label = f"live_{index}"
        self.record_path = out_dir / f"game_{index:02d}.jsonl"
        self.game = None
        self.engine_log: Optional[EngineLog] = None
        self.mirror: Optional[LiveMirror] = None
        self.pending: list = []          # engine events read ahead of the mirror
        self.known_differences: dict = {}    # field -> differences `known_difference` let through
        self.messages: dict = {}         # the engine's warnings and errors: text -> count
        self.stopped: set = set()        # (side, x, y) the default AI stopped this turn
        self.preferences: Optional[Path] = None
        self.saved_preferences: Optional[bytes] = None

    def _note(self, **entry) -> None:
        with self.record_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def play(self) -> dict:
        from wesnoth_ai.wesnoth_interface import WesnothGame
        ipc = GAMES_PATH / IPC_GAME_ID
        self.game = WesnothGame(label=self.label, watched=self.watched,
                                launch_args=launch_args(self.scenario_id, self.factions, self.our_side, self.decl))
        self.game.adopt_game_id(IPC_GAME_ID)
        # The stage sets this process's display preferences; Wesnoth may
        # write them back on exit, so the user's file is restored after.
        self.preferences = WESNOTH_USERDATA_PATH / "preferences"
        self.saved_preferences = self.preferences.read_bytes() if self.preferences.exists() else None
        (ipc / "settings.lua").write_text(
            f"return {{ turbo_speed = {self.speed}, animate = {str(self.watched).lower()}, "
            f"player = {json.dumps(self.player_label)} }}\n", encoding="utf-8")
        self._note(event="start", scenario=self.scenario_id, factions=self.factions, our_side=self.our_side)
        self.game.start_wesnoth()
        try:
            return self._loop()
        finally:
            self._close()

    def _loop(self) -> dict:
        decisions = 0
        while True:
            frame = self._next_frame()
            if frame is None:
                return self._result(f"engine_stopped (exit code {self.game.process.returncode})", decisions)
            if self.mirror is None:
                self._build(frame)
            self._catch_up(frame["seq"])
            if self.mirror.sim.done:
                return self._result("over", decisions)
            self.mirror.reach_turn(frame["turn"], frame["current_side"])
            diffs, known = board_differences(frame, self.mirror.sim, self.stopped)
            for field, _ in known:
                self.known_differences[field] = self.known_differences.get(field, 0) + 1
            if diffs:
                self._note(event="divergence", seq=frame["seq"], turn=frame["turn"], diffs=diffs)
                raise MirrorDivergence(f"game {self.index}, decision {frame['seq']} (turn {frame['turn']}): "
                                       f"the boards differ in {sorted(diffs)}: {json.dumps(diffs, default=str)[:2000]}")
            commands = self._decide()
            decisions += 1
            if commands[-1]["type"] == "end_turn":
                self.stopped.clear()     # the default AI's next turn starts after ours
            self._note(event="decision", seq=frame["seq"], turn=frame["turn"], commands=commands)
            if not self.game.send_action({"commands": commands}, seq=frame["seq"]):
                raise RuntimeError(f"game {self.index}: the commands of decision {frame['seq']} were not written")

    def _build(self, frame: dict) -> None:
        """The simulator's game, built as the scenario-init oracle builds
        it: the engine's leaders and start time, its experience modifier."""
        from tools.wesnoth_sim import WesnothSim
        from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate
        setup = our_setup(self.scenario_id, self.factions, frame)
        gs = build_scenario_gamestate(setup, experience_modifier=applied_experience_modifier(frame))
        sim = WesnothSim(gs, scenario_id=self.scenario_id, max_turns=self.max_turns)
        if sim.core is None:
            raise RuntimeError("the live mirror needs the Rust core (wesnoth_core of this source's phase)")
        self.mirror = LiveMirror(sim)
        self.engine_log = EngineLog(self.game.engine_log_path())
        log.info(f"game {self.index}: {self.scenario_id}, {self.factions[0]} ({setup.leader1}) vs "
                 f"{self.factions[1]} ({setup.leader2}), we play side {self.our_side}")

    def _next_frame(self) -> Optional[dict]:
        """The next frame of ours, the engine's commands mirrored as they
        complete meanwhile, so the end of the game is seen when it comes.
        None when the engine stopped; a frame of seq None when the game
        is over."""
        deadline = time.time() + FRAME_TIMEOUT_S
        while time.time() < deadline:
            raw = self.game.poll_state()
            if raw is not None:
                return json.loads(raw)
            if self.mirror is not None:
                self.pending.extend(self.engine_log.poll())
                self._drain()
                if self.mirror.sim.done:
                    return {"seq": None}
            if self.game.process is not None and self.game.process.poll() is not None:
                if self.mirror is not None:
                    self.pending.extend(self.engine_log.poll() + self.engine_log.reader.close())
                    self._drain()
                return None
            time.sleep(0.1)
        raise TimeoutError(f"game {self.index}: no frame in {FRAME_TIMEOUT_S:.0f} s")

    def _drain(self, until_seq: Optional[int] = None) -> bool:
        """Mirror the engine's events in order (every command the reader
        released is complete); then the command still open, if applied on
        a fork it ends the game with every draw logged. Without
        `until_seq` a sync marker stops the drain: the frame of its
        decision is read next. Returns True when the marker `until_seq`
        was reached."""
        while self.pending and not self.mirror.sim.done:
            head = self.pending[0]
            if isinstance(head, EngineMessage):
                self._message(self.pending.pop(0))
                continue
            if isinstance(head, AiStop):
                stop = self.pending.pop(0)
                if stop.side != self.our_side:
                    self.stopped.add((stop.side, stop.x, stop.y))
                continue
            if isinstance(head, AiCandidateAction):
                if "leader_shares_keep" in self.pending.pop(0).name:
                    self.stopped.add(("leaders", 3 - self.our_side))
                continue
            if isinstance(head, SyncMarker):
                if until_seq is None:
                    return False
                self.pending.pop(0)
                if head.seq == until_seq:
                    return True
                continue
            self._apply(self.pending.pop(0))
        reader = self.engine_log.reader
        if (not self.mirror.sim.done and reader.open_command is not None
                and self._ends_game(reader.open_command)):
            self._apply(reader.take_open())
        return False

    def _message(self, msg: EngineMessage) -> None:
        key = f"{msg.level} {msg.domain}: {msg.text}"
        if key not in self.messages:
            log.warning(f"game {self.index}: Wesnoth {key}")
            self._note(event="engine_message", level=msg.level, domain=msg.domain, text=msg.text,
                       turn=self.mirror.sim.turn_number if self.mirror else None)
        self.messages[key] = self.messages.get(key, 0) + 1

    def _ends_game(self, cmd: LoggedCommand) -> bool:
        fork = LiveMirror(self.mirror.sim.fork())
        try:
            fork.apply(cmd)
        except MirrorDivergence:
            return False
        return fork.sim.done

    def _catch_up(self, seq: Optional[int]) -> None:
        """Mirror every command logged before decision `seq`'s marker."""
        if seq is None:
            return
        deadline = time.time() + 30.0
        while True:
            self.pending.extend(self.engine_log.poll())
            if self._drain(until_seq=seq):
                return
            if time.time() > deadline:
                raise MirrorDivergence(f"game {self.index}: no sync marker {seq} in the engine log")
            time.sleep(0.02)

    def _apply(self, cmd: LoggedCommand) -> None:
        self._note(event="command", side=cmd.from_side, tag=cmd.tag, attrs=cmd.block.attrs,
                   children={t: n.attrs for t, n in cmd.block.children}, draws=cmd.draws,
                   advancements=cmd.advancements)
        self.mirror.apply(cmd)

    def _decide(self) -> list:
        """The player's action at the simulator's state, as in a match
        (tools/eval_players._play_one_eval_game), planned on a fork."""
        from tools.mcts import fork_guard
        from tools.selfplay_game import _would_recruit_bounce
        from wesnoth_ai.game_core import snapshot_view
        sim = self.mirror.sim
        with fork_guard(sim):
            action = self.player.select_action(snapshot_view(sim.gs), game_label=self.label, sim=sim)
        while _would_recruit_bounce(action, sim.gs):
            tgt = action["target_hex"]
            sim.reject_recruit_hex(tgt.x, tgt.y)
            self.player.drop_last_pending(self.label)
            with fork_guard(sim):
                action = self.player.select_action(snapshot_view(sim.gs), game_label=self.label, sim=sim)
        fork = sim.fork()
        fork.step(action)
        if fork.last_step_refusal == "mask_disagreement":
            raise MirrorDivergence(f"game {self.index}: the simulator refused {action!r}, which the "
                                   f"legality mask offered")
        commands = engine_commands(fork.command_history)
        return commands or [{"type": "end_turn"}]

    def _result(self, how: str, decisions: int) -> dict:
        sim = self.mirror.sim if self.mirror else None
        winner = sim.winner if sim is not None and sim.done else None
        result = {"game": self.index, "scenario": self.scenario_id, "factions": list(self.factions),
                  "our_side": self.our_side, "how": how, "winner": winner,
                  "outcome": ("win" if winner == self.our_side else "loss" if winner in (1, 2)
                              else "draw" if sim is not None and sim.done else "unfinished"),
                  "turns": sim.turn_number if sim is not None else None, "decisions": decisions,
                  "commands": self.mirror.commands if self.mirror else 0,
                  "known_differences": self.known_differences,
                  "engine_messages": self.messages}
        self._note(event="result", **result)
        return result

    def _close(self) -> None:
        if self.game is None:
            return
        deadline = time.time() + END_WAIT_S
        while time.time() < deadline and self.game.process and self.game.process.poll() is None:
            time.sleep(0.5)
        self.game.terminate()
        if self.saved_preferences is not None and self.preferences.read_bytes() != self.saved_preferences:
            self.preferences.write_bytes(self.saved_preferences)
            log.info("restored the Wesnoth preferences the game had rewritten")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--speed", type=float, default=4.0, help="Wesnoth's accelerated speed for the game")
    ap.add_argument("--unwatched", action="store_true",
                    help="a check run: the window minimized, no animations, no delays")
    ap.add_argument("--seed", type=int, default=None, help="the draw of scenarios, factions and sides")
    ap.add_argument("--max-turns", type=int, default=200, help="the matches' turn limit")
    ap.add_argument("--out", type=Path, default=None, help="directory of the game records")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    import torch
    from tools.sim_demo_game import load_player
    from wesnoth_ai.rules.scenario_pool import random_setup
    ref = reference_player.load()
    decode = ref["decode"]
    player = load_player(reference_player.local_path(), torch.device("cpu"), mcts_sims=0,
                         temperature=float(decode["raw_temperature"]),
                         end_turn_offset=float(decode.get("raw_end_turn_offset", 0.0)),
                         memory=ref.get("memory_slots"))
    seed = args.seed if args.seed is not None else int(time.time())
    rng = random.Random(seed)
    out = args.out or ROOT / "logs" / "live_vs_rca" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    log.info(f"{ref['label']} against the default AI, {args.games} games, seed {seed}, records in {out}")
    results = []
    for index in range(1, args.games + 1):
        setup = random_setup(rng, category="ladder")
        our_side = 1 + (index + seed) % 2
        decl = engine_declarations([setup.scenario_id])[setup.scenario_id]
        game = LiveGame(index, setup.scenario_id, (setup.faction1, setup.faction2), our_side, decl, player,
                        out, args.speed, args.max_turns, watched=not args.unwatched, player_label=ref["label"])
        try:
            result = game.play()
        except (MirrorDivergence, TimeoutError, RuntimeError) as e:
            # A divergence is a fidelity defect to investigate; the record
            # keeps the boards' differences, and the series goes on.
            log.error(f"game {index} stopped: {e}")
            result = {"game": index, "scenario": setup.scenario_id, "our_side": our_side,
                      "outcome": "stopped", "how": f"{type(e).__name__}: {str(e)[:500]}"}
        results.append(result)
        log.info(f"game {index}: {result['outcome']} on turn {result.get('turns')} ({result['how']})")
        (out / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    wins = sum(r["outcome"] == "win" for r in results)
    losses = sum(r["outcome"] == "loss" for r in results)
    log.info(f"{wins} wins, {losses} losses, {len(results) - wins - losses} other, of {len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
