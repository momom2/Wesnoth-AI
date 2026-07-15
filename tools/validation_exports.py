#!/usr/bin/env python3
"""Strict-sync validation exports from online training (2026-07-15).

User spec: pick EVERY 100th game (deterministic, not random 1%),
counted SEPARATELY per category -- "mini", "ladder",
"ladder_fogless", "midgame" (plus "drill" if that pool is ever
mixed in) -- and export each pick as a Wesnoth-loadable .bz2
replay while training runs. The box's HF uploader ships the
exports; `tools/run_validation_batch.py` plays each back in real
Wesnoth under strict sync locally and reports OOS.

Counters are per-process: each spool worker counts its own game
stream, so the aggregate pick rate stays 1/100 per category and
picks spread evenly across workers. Filenames carry pid + counter
so parallel workers never collide.

Fresh games export via `export_replay_from_scratch` (the path
engine-verified on the tentacle maps, 2026-07-15). Midgame games
(human-corpus starts) need the HUMAN PREFIX spliced in front of
the sim's continuation: we re-walk the source dataset commands up
to the cut boundary through `_apply_command`, harvesting the same
side-channels `WesnothSim.step` records (checkup strikes,
advancement choices, recruit [from] positions), wrap them as
RecordedCommands, and compose the save with the dataset's own
starting sides. Playback then reproduces the human half bit-exact
(the dataset commands carry the original [random_seed] values)
and continues into ours.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

log = logging.getLogger("validation_exports")


def category_of(sim) -> str:
    """Validation category of a finished sim game. Midgame trumps
    map class (a midgame game on a ladder map is validated as
    "midgame" -- its replay shape is the spliced one)."""
    if getattr(sim, "_midgame_start", False):
        return "midgame"
    from tools.scenario_pool import classify_scenario
    cls = classify_scenario(getattr(sim, "scenario_id", "") or "")
    if cls == "ladder" and not getattr(sim.gs.global_info, "_fog", True):
        return "ladder_fogless"
    return cls or "unknown"


class ValidationExporter:
    """Every-Nth-per-category picker + exporter. Thread-safe within
    one process; per-process counters across spool workers."""

    def __init__(self, out_dir: Path, every: int = 100):
        self.out_dir = Path(out_dir)
        self.every = int(every)
        self._counts: Dict[str, int] = {}
        self._lock = threading.Lock()

    def maybe_export(self, sim, game_label: str = "") -> Optional[Path]:
        """Count the game; export if it's the 1st of its category's
        current block of `every`. Never raises -- a failed export
        logs and returns None (training must not die for telemetry).
        """
        if self.every <= 0:
            return None
        try:
            cat = category_of(sim)
            with self._lock:
                self._counts[cat] = self._counts.get(cat, 0) + 1
                n = self._counts[cat]
            if n % self.every != 1 and self.every != 1:
                return None
            sub = self.out_dir / cat
            sub.mkdir(parents=True, exist_ok=True)
            sid = getattr(sim, "scenario_id", "") or "unknown"
            fname = (f"{cat}_n{n:05d}_p{os.getpid()}_{sid}"
                     f"{('_' + game_label) if game_label else ''}.bz2")
            out = sub / fname
            if cat == "midgame":
                export_midgame_replay(sim, out)
            else:
                from tools.sim_to_replay import export_replay_from_scratch
                export_replay_from_scratch(sim, out)
            log.info(f"validation export: {out}")
            return out
        except Exception as e:                        # noqa: BLE001
            log.warning(f"validation export failed ({game_label}): {e}")
            return None


def _walk_prefix_commands(data: dict, boundary_idx: int):
    """Re-apply the source game's commands[:boundary_idx] on a fresh
    reconstruction, harvesting per-command extras exactly like
    `WesnothSim.step` does. Returns (RecordedCommand list, final gs).
    """
    from classes import Position  # noqa: F401  (Position via gs)
    from tools.replay_dataset import (_apply_command,
                                      _build_initial_gamestate,
                                      _setup_scenario_events)
    from tools.wesnoth_sim import RecordedCommand

    gs = _build_initial_gamestate(data)
    _setup_scenario_events(gs, data.get("scenario_id", ""))
    history = []
    side_now = gs.global_info.current_side or 1
    for cmd in data.get("commands", [])[:boundary_idx]:
        if not cmd:
            continue
        kind = cmd[0]
        if kind == "init_side" and len(cmd) > 1:
            side_now = int(cmd[1])
        extras: dict = {}
        pre_att = pre_dfd = None
        if kind == "attack":
            ax, ay, dx, dy = cmd[1], cmd[2], cmd[3], cmd[4]
            for u in gs.map.units:
                if u.position.x == ax and u.position.y == ay:
                    pre_att = (u.id, u.name, u.side)
                elif u.position.x == dx and u.position.y == dy:
                    pre_dfd = (u.id, u.name, u.side)
        elif kind in ("recruit", "recall"):
            for u in gs.map.units:
                if u.is_leader and u.side == side_now:
                    extras["leader_pos"] = (u.position.x, u.position.y)
                    break
        _apply_command(gs, cmd)
        if kind == "attack":
            # Advancement choices, attacker first (mirrors sim.step).
            by_id = {u.id: u for u in gs.map.units}
            advance_choices = []
            for pre in (pre_att, pre_dfd):
                if pre is None:
                    continue
                pre_id, pre_name, pre_side = pre
                post = by_id.get(pre_id)
                if post is not None and post.name != pre_name:
                    advance_choices.append((pre_side, 0))
            if advance_choices:
                extras["advance_choices"] = advance_choices
            strikes = getattr(gs.global_info,
                              "_last_checkup_strikes", None)
            if strikes:
                extras["checkup_strikes"] = strikes
                setattr(gs.global_info, "_last_checkup_strikes", None)
        history.append(RecordedCommand(
            kind=kind, side=side_now, cmd=list(cmd), extras=extras))
    return history, gs


def export_midgame_replay(sim, out_path: Path) -> None:
    """Compose a full Wesnoth replay for a midgame-start game:
    human prefix (from the source dataset game, bit-exact seeds) +
    the sim's continuation, over a save whose [side] blocks come
    from the dataset's STARTING sides/leaders (the sim's own state
    is mid-game and describes the wrong starting setup).
    Requires `sim._midgame_provenance` (set by play_one_game from
    sample_midgame_start).
    """
    import bz2
    import gzip
    import json

    from classes import Position
    from tools.sim_to_replay import build_save_wml
    from tools.wesnoth_sim import PvPDefaults

    prov = getattr(sim, "_midgame_provenance", None)
    if not prov:
        raise RuntimeError("midgame sim carries no provenance")
    gz = Path(prov["dataset_dir"]) / prov["file"]
    with gzip.open(gz, "rt", encoding="utf-8") as f:
        data = json.load(f)

    prefix, _gs_cut = _walk_prefix_commands(
        data, int(prov["boundary_idx"]))

    # Starting leaders/positions from the dataset (0-indexed).
    initial_leaders = {}
    uid_to_type = {}
    for u in data.get("starting_units", []):
        uid_to_type[u.get("uid")] = u.get("type")
        if u.get("is_leader"):
            initial_leaders[int(u["side"])] = (
                u["type"], Position(x=int(u["x"]), y=int(u["y"])))
    if not {1, 2} <= set(initial_leaders):
        raise RuntimeError(f"{gz.name}: missing starting leaders")

    # Starting-side economy for the save header. Sides can in
    # principle differ in gold; the composer takes side 1's and
    # warns -- [side] gold is per-side in _render_side_block only
    # via cfg scrape, so a differing side 2 would need work.
    sides = data.get("starting_sides", [])
    golds = [int(s.get("gold", 100) or 100) for s in sides[:2]]
    if len(set(golds)) > 1:
        log.warning(f"{gz.name}: per-side starting gold differs "
                    f"{golds}; using {golds[0]}")
    pvp = PvPDefaults(
        starting_gold=golds[0] if golds else 100,
        experience_modifier=int(
            data.get("experience_modifier", 70) or 70),
    )

    # A fresh reconstruction of the INITIAL state gives build_save_wml
    # the starting sides (gold/recruits/faction) + the ToD offset.
    from tools.replay_dataset import _build_initial_gamestate
    gs0 = _build_initial_gamestate(data)

    shim = SimpleNamespace(
        gs=gs0,
        scenario_id=data.get("scenario_id", ""),
        initial_leaders=initial_leaders,
        command_history=list(prefix) + list(sim.command_history),
    )
    save_wml = build_save_wml(shim, pvp_defaults=pvp)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with bz2.open(out_path, "wt", encoding="utf-8") as f:
        f.write(save_wml)
