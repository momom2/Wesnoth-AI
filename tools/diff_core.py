#!/usr/bin/env python3
"""The Rust-owned game state against the Python applier, over replays.

Every command of each replay goes to the Python state through
`replay_dataset._apply_command` and to a `wesnoth_ai.game_core.CoreState`
built from a copy of the initial state, the scenario set up by each
(`_setup_scenario_events`, `CoreState.setup_scenario`); after the setup
and after every command (or every `--every` commands) the two states
are compared over their modeled content (`core_compare.state_differences`). A difference is a divergence,
reported with the command's index and kind and the path the core took
("rust" or the Python fallback). A replay on which the core raises or
panics (a Rust panic reaches Python as pyo3's PanicException, a
BaseException) is listed as that replay's divergence and the sweep goes
on. Certification of the port's step kernels (docs/rust_port_plan.md
phase 4).

    python tools/diff_core.py replays_dataset_imitation/*.json.gz --limit 200
    python tools/diff_core.py DIR --every 1 --stop-on-first

With `--encode-every N`, every Nth player decision is also encoded by
the Python encoder and by the core (`encoding_differences`) in three
views: the full board, the full board with the terrain set, and obs8's
(the relevant set, the enemy-village gate, the terrain set). With
`--outcomes`, before every attack command the defender's weapon choice
with its strike tables and the attack's outcome distributions, without
and with advancement branches, are computed by `tools/combat_outcomes.py`
on the Python state and by the core, and must be equal to the last bit.
With `--sightings`, after every command the core notes sightings after,
each player side's sighting record, seen types and gone entries are
compared with `tools/sighting_oracle.py`, which follows the Python state;
and at every encoded decision, the parity encoding's sighting stream with
the stream the oracle's record gives. The parity encoding's other columns
have no Python builder: they are covered by
tests/test_parity_observation.py and tests/test_sighting_record.py.

Prints one summary line: `diff_core: N replays, M clean, K with
divergences; commands rust=A python=B`, then the divergences.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("diff_core")


def is_rust_panic(exc: BaseException) -> bool:
    """A Rust panic reaches Python as pyo3's `PanicException`, which
    derives from BaseException precisely so that `except Exception`
    does not catch it. It has no importable home (`pyo3_runtime` is
    not a module), so it is recognised by its name."""
    return type(exc).__name__ == "PanicException"


def _describe_failure(exc: BaseException) -> str:
    """A replay's failure as one listed line, or the exception re-raised
    when it is not the replay's (KeyboardInterrupt, SystemExit)."""
    if is_rust_panic(exc):
        return f"panicked {exc!r}"
    if isinstance(exc, Exception):
        return f"raised {exc!r}"
    raise exc


# (relevant set, enemy-village gate, terrain multi-hot) per encoding compared.
ENCODINGS = ((False, False, False), (False, False, True), (True, True, True))
_VOCAB: dict = {}


def _vocab():
    """A vocabulary over every unit type of the database and every
    default-era faction: any fixed one serves, both encoders read it."""
    if not _VOCAB:
        from wesnoth_ai.paths import UNIT_STATS_PATH
        from wesnoth_ai.rules.scenario_pool import load_factions
        names = sorted(json.loads(UNIT_STATS_PATH.read_text(encoding="utf-8"))["units"])
        _VOCAB["types"] = {n: i for i, n in enumerate(names)}
        _VOCAB["factions"] = {f: i for i, f in enumerate(sorted(load_factions()))}
    return _VOCAB["types"], _VOCAB["factions"]


def encoding_divergences(gs, cs) -> List[str]:
    """The encodings of the Python state and of the core that differ. On
    the pure-Python path (the certification's: WESNOTH_RUST=0 and friends)
    the Python encoding carries no observation record, which only the Rust
    paths attach; the model's inputs, its arrays, are compared all the same."""
    import dataclasses
    from wesnoth_ai.core_compare import encoding_differences
    from wesnoth_ai.encoder import encode_raw
    types, factions = _vocab()
    out: List[str] = []
    for relevant, gate, multi in ENCODINGS:
        kw = dict(type_to_id=types, faction_to_id=factions, relevant_set=relevant,
                  fog_hides_enemy_villages=gate, terrain_multi_hot=multi)
        py, rs = encode_raw(gs, **kw), cs.encode_raw(**kw)
        if py.observation is None:
            rs = dataclasses.replace(rs, observation=None)
        diffs = encoding_differences(py, rs)
        if diffs:
            out.append(f"encoding {(relevant, gate, multi)}: {diffs[:6]}")
    return out


def stream_divergences(gs, cs, oracle) -> List[str]:
    """The core's parity sighting stream for the side to act against the one
    the oracle's record gives: its entries that are gone or whose unit the
    side does not see (the Python visibility), sorted by (y, x, id), at the
    recorded hex clamped to the board tensor, with the type's vocabulary row,
    hit points over maximum and maximum over HP_NORM."""
    from wesnoth_ai import encoder as enc
    from wesnoth_ai.visibility import units_visible_to_python
    types, factions = _vocab()
    side = int(gs.global_info.current_side)
    raw = cs.encode_raw(type_to_id=types, faction_to_id=factions, relevant_set=True,
                        fog_hides_enemy_villages=True, terrain_multi_hot=True,
                        observation_parity=True, relevant_set_version=2)
    seen = {u.id for u in units_visible_to_python(gs, side)}
    gone = oracle.gone[side]
    rows = sorted((r for r in oracle.sightings[side].values() if r[0] in gone or r[0] not in seen),
                  key=lambda r: (r[5], r[4], r[0]))
    limit = enc.MAX_MAP_SIZE - 1
    want = ([enc.type_row(r[1], types) for r in rows], [min(max(r[4], 0), limit) for r in rows],
            [min(max(r[5], 0), limit) for r in rows])
    got = (raw.sight_type_ids.tolist(), raw.sight_xs.tolist(), raw.sight_ys.tolist())
    feats = raw.sight_feats.tolist()
    same = got == want and len(feats) == len(rows) and all(
        abs(f[0] - r[2] / max(r[3], 1)) < 1e-6 and abs(f[1] - r[3] / enc.HP_NORM) < 1e-6
        for f, r in zip(feats, rows))
    if same:
        return []
    return [f"sighting stream of side {side}: core {list(zip(*got))[:4]}, oracle {list(zip(*want))[:4]}"]


def outcome_divergences(gs, cs, cmd: list) -> List[str]:
    """Before an attack command: the Python's and the core's defender
    weapon choice, strike tables and outcome distributions that differ."""
    from tools import combat_outcomes as co
    from wesnoth_ai.classes import Position
    ax, ay, dx, dy, a_weapon = (int(v) for v in cmd[1:6])
    units = {(u.position.x, u.position.y): u for u in gs.map.units}
    att, dfd = units.get((ax, ay)), units.get((dx, dy))
    if att is None or dfd is None:
        return []
    out: List[str] = []
    weapon, tables = co.counter_weapon_choice(gs, att, dfd, a_weapon)
    rs = cs.core.counter_weapon_choice(ax, ay, dx, dy, a_weapon)
    if rs is None or rs[0] != weapon or \
            [(i, list(t.items())) for i, t in rs[1].items()] != [(i, list(t.items())) for i, t in tables.items()]:
        out.append(f"counter weapon: python {weapon} core {None if rs is None else rs[0]}")
    action = {"type": "attack", "start_hex": Position(ax, ay), "target_hex": Position(dx, dy),
              "attack_index": a_weapon}
    for choice in (None, "uniform"):
        py = co.enumerate_attack_outcomes(gs, action, advancement_choice=choice)
        rs = cs.core.attack_outcomes(ax, ay, dx, dy, a_weapon, choice == "uniform")
        same = (py is None and rs is None) or (
            py is not None and rs is not None and list(py.probs.items()) == list(rs[0].items()))
        if not same:
            out.append(f"outcomes ({choice}): python {None if py is None else len(py.probs)} keys, "
                       f"core {None if rs is None else len(rs[0])}")
    return out


def diff_core(gz_path: Path, *, every: int = 1, stop_on_first: bool = True,
              counts: Optional[Counter] = None, encode_every: int = 0,
              outcomes: bool = False, sightings: bool = False) -> List[str]:
    from tools.replay_dataset import _apply_command, _build_initial_gamestate, _setup_scenario_events
    from wesnoth_ai.core_compare import state_differences
    from wesnoth_ai.game_core import CoreState
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    gs = _build_initial_gamestate(data)
    scenario_id = data.get("scenario_id", "")
    try:
        cs = CoreState.from_state(copy.deepcopy(gs))
        cs.setup_scenario(scenario_id)
    except BaseException as e:  # noqa: BLE001 - a panic is a divergence too
        return [f"{gz_path.name}#setup: core {_describe_failure(e)}"]
    _setup_scenario_events(gs, scenario_id)
    out: List[str] = []
    diffs = state_differences(gs, cs.to_state(), stash=False)
    if diffs:
        return [f"{gz_path.name}#setup: " + " | ".join(diffs[:4])]
    decisions = 0
    oracle = None
    if sightings:
        from tools.sighting_oracle import NOTED_KINDS, SightingOracle, record_differences
        oracle = SightingOracle()
    for idx, cmd in enumerate(data.get("commands", [])):
        kind = cmd[0] if cmd else "?"
        if encode_every and gs.global_info.current_side in (1, 2):
            if decisions % encode_every == 0:
                enc = encoding_divergences(gs, cs)
                if oracle is not None:
                    enc += stream_divergences(gs, cs, oracle)
                if counts is not None:
                    counts[("encode", "rust")] += 1
                if enc:
                    out.append(f"{gz_path.name}#{idx} before {kind}: " + " | ".join(enc))
                    if stop_on_first:
                        break
            decisions += 1
        if outcomes and kind == "attack" and len(cmd) > 5:
            diffs = outcome_divergences(gs, cs, cmd)
            if counts is not None:
                counts[("outcomes", "rust")] += 1
            if diffs:
                out.append(f"{gz_path.name}#{idx} before attack: " + " | ".join(diffs))
                if stop_on_first:
                    break
        if oracle is not None:
            oracle.before(gs, cmd)
        _apply_command(gs, list(cmd))
        try:
            path = cs.apply_command(list(cmd))
        except BaseException as e:  # noqa: BLE001 - a panic is a divergence too
            out.append(f"{gz_path.name}#{idx} {kind}: core {_describe_failure(e)}")
            break
        if counts is not None:
            counts[(kind, path)] += 1
        if oracle is not None and kind in NOTED_KINDS:
            oracle.after(gs, cmd)
            seen = record_differences(oracle, cs.core)
            if counts is not None:
                counts[("sightings", "rust")] += 1
            if seen:
                out.append(f"{gz_path.name}#{idx} {kind} sightings: " + " | ".join(seen[:4]))
                if stop_on_first:
                    break
        if idx % every == 0 or kind in ("init_side", "attack"):
            diffs = state_differences(gs, cs.to_state(), stash=False,
                                      map_and_events=kind in ("init_side", "end_turn"))
            if diffs:
                out.append(f"{gz_path.name}#{idx} {kind} via {path}: " + " | ".join(diffs[:4]))
                if stop_on_first:
                    break
    return out


def _walk(inputs: List[Path]) -> List[Path]:
    files: List[Path] = []
    for p in inputs:
        if p.is_dir():
            files.extend(sorted(p.glob("*.json.gz")))
        else:
            files.append(p)
    return files


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--every", type=int, default=1, help="compare every N commands (init_side and attack always)")
    ap.add_argument("--stop-on-first", action="store_true")
    ap.add_argument("--encode-every", type=int, default=0,
                    help="compare the two encodings every N player decisions (0: never)")
    ap.add_argument("--outcomes", action="store_true",
                    help="compare the counter weapon and the outcome distributions before every attack")
    ap.add_argument("--sightings", action="store_true",
                    help="compare each side's sighting record, and at encoded decisions its sighting "
                         "stream, with tools/sighting_oracle.py")
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING),
                        format="%(levelname)s %(name)s: %(message)s")
    files = _walk(args.inputs)
    if args.limit:
        files = files[:args.limit]
    # What the Python side ran on: independent of the core only when every
    # kernel it reaches is the Python one (the certification's setting).
    from tools.kernel_status import banner
    print(f"diff_core: the applier's {banner()}")
    counts: Counter = Counter()
    clean = 0
    divergences: List[str] = []
    for gz in files:
        try:
            d = diff_core(gz, every=args.every, stop_on_first=args.stop_on_first, counts=counts,
                          encode_every=args.encode_every, outcomes=args.outcomes,
                          sightings=args.sightings)
        except BaseException as e:  # noqa: BLE001 - one bad file, panic included, must not end the sweep
            d = [f"{gz.name}: harness {_describe_failure(e)}"]
        if d:
            divergences.extend(d)
        else:
            clean += 1
    checks = ("encode", "outcomes", "sightings")          # comparisons, not commands
    rust = sum(v for (k, p), v in counts.items() if p == "rust" and k not in checks)
    py = sum(v for (k, p), v in counts.items() if p == "python" and k not in checks)
    print(f"diff_core: {len(files)} replays, {clean} clean, {len(files) - clean} with divergences; "
          f"commands rust={rust} python={py}")
    by_kind = Counter()
    for (k, p), v in counts.items():
        by_kind[f"{k}:{p}"] += v
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))
    for line in divergences[:200]:
        print("  " + line)
    return 0 if len(files) == clean else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
