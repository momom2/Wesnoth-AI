"""Build per-scenario `[scenario]` templates from the GAME'S OWN
.cfg files, expanded by Wesnoth's OWN preprocessor (`wesnoth -p`).

Replaces the retired replays_raw-based extract_scenario_templates.py:
the replay corpus is gone for good (user decision 2026-06-12), and
replay-derived templates carried per-replay quirks anyway (e.g.
multiplayer_Hamlets shipped RUSSIAN objectives text inherited from
whichever player's save it was extracted from). Building from the
cfg gives byte-faithful macro expansion -- {DEFAULT_SCHEDULE},
{DEFAULT_MUSIC_PLAYLIST}, add-on map inclusions, even scenario-local
#define blocks -- because the actual game binary does the expanding.

Covers ALL THREE scenario pools (tools/scenario_pool.py):
  - LADDER_SCENARIO_IDS  -- mainline 2p maps under
    wesnoth_src/data/multiplayer/scenarios/
  - MINI_MAP_SCENARIO_IDS -- the Mini Maps Collection add-on under
    wesnoth_src/data/add-ons/Mini_Maps_Collection/scenarios/
    (tactical-training curriculum; templates make mini games
    exportable to the replay viewer, which replay extraction never
    could -- no human replays exist for them).
  - DRILL_SCENARIO_IDS -- our own capability drills under
    add-ons/wesnoth_ai/scenarios/drills/ (project add-on,
    junctioned into userdata so the preprocessor and the real game
    resolve the same files).

REQUIREMENT for the mini pool: the add-on must ALSO be installed in
Wesnoth's userdata (Documents/My Games/Wesnoth1.18/data/add-ons/),
because the preprocessor resolves `{~add-ons/...}` includes against
userdata. Install: extract Mini_Maps_Collection.tar.bz2 there
(https://files.wesnoth.org/addons/1.18/Mini_Maps_Collection.tar.bz2).

Transformations from preprocessed [multiplayer] to save-shaped
[scenario] (mirroring what the engine itself writes into saves,
verified against the previously user-verified replay-derived
templates):
  - [multiplayer] -> [scenario]
  - translation markers `_"..."` -> plain strings (saves carry none)
  - #textdomain lines dropped
  - [side] 1 and 2 stripped (the runtime emitter renders fresh ones
    from sim state); scenery sides 3+ KEPT (sim_to_replay's
    keep-scenery rule, commit ccc7d82 lineage)
  - ladder: map_data inlined from the .map file (map_file kept too,
    matching real saves); minis arrive with map_data already inlined
    by the preprocessor
  - save-only attrs injected when absent: turns="-1",
    experience_modifier="70" (PvP default),
    has_mod_events="yes", loaded_resources="", English objectives
  - the default era's two engine-injected lua events appended
    (quick_4mp_leaders trait + turns_over_advantage), byte-exact
    from the previously verified save extraction

Usage:
    python tools/build_scenario_templates.py            # all
    python tools/build_scenario_templates.py --only multiplayer_Hamlets 2p_mini
    python tools/build_scenario_templates.py --check    # build to tmp + diff only
"""
from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from wesnoth_ai.constants import WESNOTH_PATH
from tools.scenario_pool import (
    DRILL_SCENARIO_IDS, LADDER_SCENARIO_IDS, MINI_MAP_SCENARIO_IDS,
)
from tools.replay_extract import parse_wml

log = logging.getLogger("build_scenario_templates")

LADDER_SRC = _ROOT / "wesnoth_src" / "data" / "multiplayer" / "scenarios"
# Preprocess the add-on via its _main.cfg, NOT the scenarios dir:
# _main.cfg includes utils/ (shared #defines) before scenarios/,
# mirroring the game's load order. Preprocessing scenario files in
# isolation leaves utils macros undefined and the preprocessor
# refuses the file (observed: every enclave_* scenario).
MINI_SRC = (_ROOT / "wesnoth_src" / "data" / "add-ons"
            / "Mini_Maps_Collection" / "_main.cfg")
# Capability drills live in OUR add-on at the project root. The
# preprocessor resolves its `{~add-ons/wesnoth_ai/...}` includes
# against userdata, where add-ons/wesnoth_ai is junctioned -- so the
# files it reads ARE these files.
DRILL_SRC = _ROOT / "add-ons" / "wesnoth_ai" / "_main.cfg"
OUT_DIR = _ROOT / "tools" / "templates" / "scenarios"

# Save-only attributes the engine injects at game start. Injected
# only when the cfg doesn't define them itself. experience_modifier
# matches PvPDefaults; turns=-1 = unlimited (the sim's --max-turns
# governs actual length; Wesnoth playback follows the [replay]).
_INJECTED_ATTRS = [
    ("turns", "-1"),
    ("experience_modifier", "70"),
    ("has_mod_events", "yes"),
    ("loaded_resources", ""),
    ("objectives",
     "<big>Victory:</big>\\n"
     "<span color='#00ff00'>&#8226; Defeat enemy leader(s)</span>"),
]

# The default era injects these two events into every MP game; real
# saves carry them and playback expects them (turns_over_advantage
# fires at `time over`). Byte-exact from the previously user-verified
# replay-derived template (multiplayer_Hamlets.wml lines 385-417) --
# they are constant across all 21 ladder templates (verified
# 2026-06-12) because they come from data/multiplayer/eras.cfg's
# {QUICK_4MP_LEADERS} / {TURNS_OVER_ADVANTAGE}, not the scenario.
ERA_DEFAULT_EVENTS = '''[event]

\t\tname="prestart"
\t\t[lua]
\t\t\tcode=" wesnoth.require(""multiplayer/eras.lua"").quick_4mp_leaders(...) "
\t\t\t[args]
\t\t\t\t[trait]
\t\t\t\t\tfemale_name=_"female^quick"
\t\t\t\t\thelp_text=_"<italic>text='Quick'</italic> units have 1 extra movement point, but 5% less hitpoints than usual." +
\t\t\t\t\t\t_"

Quick is the most noticeable trait, particularly in slower moving units such as trolls or heavy infantry. Units with the quick trait often have greatly increased mobility in rough terrain, which can be important to consider when deploying your forces. Also, quick units aren’t quite as tough as units without this trait and are subsequently less good at holding contested positions."
\t\t\t\t\tid="quick"
\t\t\t\t\tmale_name=_"quick"
\t\t\t\t\t[effect]
\t\t\t\t\t\tapply_to="movement"
\t\t\t\t\t\tincrease=1
\t\t\t\t\t[/effect]
\t\t\t\t\t[effect]
\t\t\t\t\t\tapply_to="hitpoints"
\t\t\t\t\t\tincrease_total="-5%"
\t\t\t\t\t[/effect]
\t\t\t\t[/trait]
\t\t\t[/args]
\t\t[/lua]
\t[/event]
[event]

\t\tname="time over"
\t\t[lua]
\t\t\tcode=" wesnoth.require(""multiplayer/eras.lua"").turns_over_advantage() "
\t\t[/lua]
\t[/event]
'''


def run_preprocessor(src: Path, out_dir: Path) -> None:
    """Expand a cfg file or folder with the game's own preprocessor.
    Read-only use of the Steam binary; output lands in `out_dir`.
    The binary is GUI-subsystem -- it exits after -p without a
    window, but we poll for output since it detaches."""
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [str(WESNOTH_PATH), "--preprocess", str(src), str(out_dir),
         "--preprocess-defines=MULTIPLAYER"],
    )
    try:
        proc.wait(timeout=180)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError(f"wesnoth --preprocess timed out on {src}")
    # Give the filesystem a beat; -p writes synchronously but the
    # process tree can outlive the parent slightly on Windows.
    for _ in range(20):
        if any(out_dir.glob("*.cfg")):
            return
        time.sleep(0.5)
    raise RuntimeError(f"wesnoth --preprocess produced no .cfg in {out_dir}")


def _block_spans(lines: List[str], tag: str) -> List[tuple]:
    """(start, end_exclusive) line spans of top-level [tag] blocks,
    tracking quote parity so bracket-looking text inside multi-line
    quoted values (map_data) can't confuse the scan."""
    spans = []
    depth = 0
    in_quote = False
    start = None
    for i, line in enumerate(lines):
        if not in_quote:
            s = line.strip()
            if s == f"[{tag}]":
                if depth == 0:
                    start = i
                depth += 1
            elif s == f"[/{tag}]" and depth > 0:
                depth -= 1
                if depth == 0:
                    spans.append((start, i + 1))
                    start = None
        if line.count('"') % 2 == 1:
            in_quote = not in_quote
    return spans


def _strip_player_sides(body_lines: List[str]) -> List[str]:
    """Remove top-level [side] blocks whose side= is 1 or 2; keep
    scenery sides 3+ (the emitter preserves them per ccc7d82).

    Kept blocks get their [side] / [/side] boundary lines dedented
    to column 0: build_save_wml locates its player-side splice point
    with a column-0-anchored `^\\[side\\]` regex (anchoring on
    indentation is how it avoids matching [side]-shaped tags nested
    inside [event] filters), and scenario cfgs indent their scenery
    sides."""
    out = list(body_lines)
    for start, end in reversed(_block_spans(out, "side")):
        block = "\n".join(out[start:end])
        m = re.search(r"^\s*side\s*=\s*\"?(\d+)\"?", block, re.M)
        side_n = int(m.group(1)) if m else 0
        if side_n in (1, 2):
            del out[start:end]
        else:
            out[start] = out[start].strip()
            out[end - 1] = out[end - 1].strip()
    return out


def _inline_map_data(body_lines: List[str], scenario_id: str) -> List[str]:
    """Ladder cfgs reference map_file=...; real saves additionally
    inline map_data. Insert map_data before the map_file line (keep
    map_file, matching the engine's saves). Minis arrive with
    map_data already inlined by the preprocessor."""
    for i, line in enumerate(body_lines):
        m = re.match(r'(\s*)map_file\s*=\s*"?([^"\s]+)"?\s*$', line)
        if m:
            indent, rel = m.group(1), m.group(2)
            map_path = _ROOT / "wesnoth_src" / "data" / rel
            if not map_path.is_file():
                raise RuntimeError(
                    f"{scenario_id}: map file not found: {map_path}")
            content = map_path.read_text(
                encoding="utf-8", errors="replace").rstrip("\n")
            map_lines = content.split("\n")
            map_lines[0] = f'{indent}map_data="{map_lines[0]}'
            map_lines[-1] = f'{map_lines[-1]}"'
            return body_lines[:i] + map_lines + body_lines[i:]
    return body_lines


def transform(pp_text: str, scenario_id: str, source_note: str) -> str:
    """Preprocessed [multiplayer] cfg text -> save-shaped [scenario]
    template text."""
    lines = pp_text.splitlines()
    spans = _block_spans(lines, "multiplayer")
    if not spans:
        raise RuntimeError(f"{scenario_id}: no [multiplayer] block")
    start, end = spans[0]
    body = lines[start + 1:end - 1]

    # Drop #textdomain markers (saves carry none).
    body = [l for l in body if not l.strip().startswith("#textdomain")]
    # Strip translation markers everywhere in the cfg-derived body
    # (the era boilerplate below is spliced raw and keeps its own).
    body = [re.sub(r'_\s*"', '"', l) for l in body]
    body = _strip_player_sides(body)
    if (scenario_id not in MINI_MAP_SCENARIO_IDS
            and scenario_id not in DRILL_SCENARIO_IDS):
        # Minis and drills arrive with map_data already inlined by
        # the preprocessor; ladder cfgs reference map_file=.
        body = _inline_map_data(body, scenario_id)

    # Inject save-only attrs not already defined by the cfg.
    present = {m.group(1) for l in body
               for m in [re.match(r"\s*(\w+)\s*=", l)] if m}
    injected = [f'{k}="{v}"' for k, v in _INJECTED_ATTRS
                if k not in present]

    header = (
        f"# Per-scenario [scenario] template for from-scratch save\n"
        f"# WML emission. Built from the game's own scenario .cfg by\n"
        f"# Wesnoth's own preprocessor (see\n"
        f"# tools/build_scenario_templates.py). [side] 1/2 stripped\n"
        f"# at build; the runtime emitter (sim_to_replay.build_save_wml)\n"
        f"# inserts fresh [side] blocks per game from sim state.\n"
        f"#\n"
        f"# Source: {source_note}\n"
        f"# Regenerate via: python tools/build_scenario_templates.py\n"
    )
    out = (header + "[scenario]\n"
           + "\n".join(injected) + ("\n" if injected else "")
           + "\n".join(body).rstrip("\n") + "\n"
           + ERA_DEFAULT_EVENTS
           + "[/scenario]\n")
    _validate(out, scenario_id)
    return out


def _validate(text: str, scenario_id: str) -> None:
    root = parse_wml(text)
    sc = root.first("scenario")
    assert sc is not None, f"{scenario_id}: no [scenario] after transform"
    got = sc.attrs.get("id", "")
    assert got == scenario_id, f"id mismatch: {got!r} != {scenario_id!r}"
    assert sc.attrs.get("map_data"), f"{scenario_id}: missing map_data"
    times = [c for c in sc.children if c.tag == "time"]
    assert len(times) >= 4, f"{scenario_id}: schedule missing ({len(times)} [time])"
    sides = [c for c in sc.children if c.tag == "side"]
    for s in sides:
        n = int(str(s.attrs.get("side", "0")).strip('"') or 0)
        assert n >= 3, f"{scenario_id}: player side {n} not stripped"


def _index_preprocessed(pp_dir: Path) -> Dict[str, str]:
    """scenario_id -> preprocessed [multiplayer] block text, for
    every block in every cfg under pp_dir. A single output file may
    carry MANY blocks (preprocessing an add-on's _main.cfg expands
    all its scenarios into one stream); junk between blocks (e.g.
    raw map content included by `{.../maps}` directory includes) is
    ignored because only [multiplayer] spans are extracted."""
    out: Dict[str, str] = {}
    for f in pp_dir.glob("*.cfg"):
        lines = f.read_text(
            encoding="utf-8", errors="replace").splitlines()
        for s, e in _block_spans(lines, "multiplayer"):
            block = "\n".join(lines[s:e])
            m = re.search(r'^\s*id\s*=\s*"?([\w-]+)"?\s*$', block, re.M)
            if m:
                out[m.group(1)] = block
    return out


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None,
                    help="Subset of scenario ids to build.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s %(message)s")

    wanted = set(args.only) if args.only else (
        set(LADDER_SCENARIO_IDS) | set(MINI_MAP_SCENARIO_IDS)
        | set(DRILL_SCENARIO_IDS))

    with tempfile.TemporaryDirectory(prefix="wml_pp_") as td:
        tmp = Path(td)
        index: Dict[str, str] = {}
        sources: Dict[str, str] = {}
        if wanted & set(LADDER_SCENARIO_IDS):
            run_preprocessor(LADDER_SRC, tmp / "ladder")
            idx = _index_preprocessed(tmp / "ladder")
            index.update(idx)
            sources.update({k: "wesnoth_src/data/multiplayer/scenarios "
                               "(game cfg, game preprocessor)"
                            for k in idx})
        if wanted & set(MINI_MAP_SCENARIO_IDS):
            if not MINI_SRC.is_file():
                log.error(f"mini add-on missing: {MINI_SRC}")
                return 2
            run_preprocessor(MINI_SRC, tmp / "mini")
            idx = _index_preprocessed(tmp / "mini")
            index.update(idx)
            sources.update({k: "Mini_Maps_Collection add-on "
                               "(game cfg, game preprocessor)"
                            for k in idx})
        if wanted & set(DRILL_SCENARIO_IDS):
            if not DRILL_SRC.is_file():
                log.error(f"wesnoth_ai add-on missing: {DRILL_SRC}")
                return 2
            run_preprocessor(DRILL_SRC, tmp / "drills")
            idx = _index_preprocessed(tmp / "drills")
            index.update(idx)
            sources.update({k: "add-ons/wesnoth_ai drills "
                               "(game cfg, game preprocessor)"
                            for k in idx})

        args.out_dir.mkdir(parents=True, exist_ok=True)
        written, missing = [], []
        for sid in sorted(wanted):
            if sid not in index:
                missing.append(sid)
                continue
            out_text = transform(index[sid], sid, sources[sid])
            (args.out_dir / f"{sid}.wml").write_text(
                out_text, encoding="utf-8", newline="\n")
            written.append(sid)
        log.info(f"wrote {len(written)} templates to {args.out_dir}")
        if missing:
            log.error(f"NOT FOUND in preprocessed output: {missing}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
