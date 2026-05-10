"""Minimal scenario-event interpreter for Wesnoth replays.

Many maps trigger gameplay-affecting WML `[event]` blocks during play
(Aethermaw morphs impassable terrain into water at turns 4-6;
 Caves of the Basilisk spawns petrified statue units at prestart;
 etc.). The replay file does NOT carry the result of those events —
 the engine re-fires them by re-loading the scenario .cfg.

This module reads a scenario's .cfg from the Wesnoth source tree and
exposes the events as a list keyed by trigger name. Callers (the
replay-dataset reconstructor) ask for events by trigger and apply
their action tags to the running `GameState`.

Scope is deliberately minimal:
  - Trigger names supported: prestart, start, side N turn M, turn N
  - Action tags supported: [terrain], [unit], [item] (no-op),
                           [modify_side], [gold], [endlevel] (no-op),
                           [message]/[note]/[music]/[sound] (no-op)
  - Macros: a tiny lookup of the cosmetic macros that show up in 2p
    scenarios — substituted to a no-op. Anything unrecognized
    silently expands to nothing (best-effort; logged once per macro).

We do NOT preserve full WML semantics. We aim for: gameplay-affecting
state changes for the events we've audited in the 2p mainline.

Dependencies: tools.replay_extract (parse_wml), classes
Dependents: tools.replay_dataset
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classes import GameState, Hex, Position, SideInfo, TerrainModifiers
from tools.replay_extract import WMLNode, parse_wml


log = logging.getLogger("scenario_events")

WESNOTH_SRC = Path(__file__).resolve().parent.parent / "wesnoth_src"
SCENARIO_DIR = WESNOTH_SRC / "data" / "multiplayer" / "scenarios"


# ----------------------------------------------------------------------
# WML macro pre-processor
# ----------------------------------------------------------------------

# Macros whose body is purely cosmetic — sound, music, halo, image
# placement — that we substitute to nothing. The lists isn't exhaustive;
# unknown macros also expand to nothing (with a logged warning the
# first time we see them).
_COSMETIC_MACROS: Set[str] = {
    "FLASH_WHITE", "QUAKE", "PLACE_IMAGE", "PLACE_HALO",
    "DEFAULT_SCHEDULE", "DEFAULT_MUSIC_PLAYLIST",
    "UNDEAD_MUSIC", "LOYALIST_MUSIC", "REBELS_MUSIC",
    "ITM_FOREST_FOG", "BIGMAP", "IS_LAST_SCENARIO",
    "TURNS_OVER_ADVANTAGE",
}

# Track which macros we've warned about to avoid log spam.
_WARNED_MACROS: Set[str] = set()


_MACRO_DEFINE_RE = re.compile(
    # Use `[ \t]+` (NOT `\s+`) for the gap between the macro name and
    # its params — `\s+` matches newlines too, which causes parameter-
    # less macros like `#define SECOND_WATCH\n  [time]…` to swallow the
    # opening `[time]` of the body as a "parameter". That breaks every
    # ToD macro the time_area scenarios rely on.
    r'#define[ \t]+(\w+)(?:[ \t]+([^\n]+))?\n(.*?)#enddef',
    re.DOTALL,
)
# {MACRO_NAME arg1 arg2 ...} — args can be quoted strings, bare words,
# or numeric values. We capture the whole brace expression.
_MACRO_INVOKE_RE = re.compile(r'\{([A-Z_][A-Z0-9_]*)([^{}]*)\}')


def _strip_textdomain(text: str) -> str:
    return re.sub(r'^\s*#textdomain\s+\S+\s*$', '', text, flags=re.MULTILINE)


# WML parallel-assignment: `x,y=24,0` → `x=24\ny=0`. Wesnoth treats the
# key list and value list as positional pairs. Our line-based parser
# only recognizes `key=val` with key being a bare identifier, so a line
# like `x,y=24,0` is silently dropped — which makes [store_locations]
# [or] clauses lose their coordinates. Expanding this BEFORE parse
# fixes the issue cleanly.
_PARALLEL_ASSIGN_RE = re.compile(
    r'^(\s*)([a-zA-Z_][\w]*(?:\s*,\s*[a-zA-Z_][\w]*)+)\s*=\s*(.*?)\s*$',
    re.MULTILINE,
)


def _expand_parallel_assigns(text: str) -> str:
    def _do(m):
        indent  = m.group(1)
        keys    = [k.strip() for k in m.group(2).split(",")]
        # Split values on UNQUOTED commas. Quoted strings shouldn't show
        # up in coordinate lists; the simple split is fine for the
        # gameplay-affecting uses we care about (x,y=…, side,gold=…).
        vals    = [v.strip() for v in m.group(3).split(",")]
        if len(keys) != len(vals):
            return m.group(0)            # leave unchanged if shape is off
        return "\n".join(f"{indent}{k}={v}" for k, v in zip(keys, vals))
    return _PARALLEL_ASSIGN_RE.sub(_do, text)


def _strip_comments(text: str) -> str:
    """Remove '# ...' comments outside of strings. WML comments are
    line-based and not great when nested inside macros, so we just drop
    every line starting with '#' that isn't `#define`/`#enddef`/
    `#textdomain`/`#ifdef`."""
    out_lines: List[str] = []
    for line in text.splitlines():
        s = line.lstrip()
        if s.startswith("#") and not s.startswith(("#define", "#enddef",
                                                   "#textdomain", "#ifdef",
                                                   "#ifndef", "#endif",
                                                   "#else", "#undef")):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def _extract_inline_macros(text: str) -> Tuple[str, Dict[str, Tuple[List[str], str]]]:
    """Pull #define...#enddef blocks out of `text`. Returns (text_without_defines,
    {macro_name: (param_names, body_text)})."""
    macros: Dict[str, Tuple[List[str], str]] = {}
    def _strip(m):
        name   = m.group(1)
        params = (m.group(2) or "").split()
        body   = m.group(3)
        macros[name] = (params, body)
        return ""
    cleaned = _MACRO_DEFINE_RE.sub(_strip, text)
    return cleaned, macros


def _split_macro_args(arg_str: str) -> List[str]:
    """Split macro invocation arguments respecting quoted strings."""
    args: List[str] = []
    cur = ""
    in_quote = False
    for ch in arg_str:
        if ch == '"':
            in_quote = not in_quote
            cur += ch
        elif ch.isspace() and not in_quote:
            if cur:
                args.append(cur)
                cur = ""
        else:
            cur += ch
    if cur:
        args.append(cur)
    return args


def _substitute_macros(text: str, macros: Dict[str, Tuple[List[str], str]],
                       depth: int = 0) -> str:
    """Recursively substitute {MACRO arg arg ...} occurrences in `text`.
    Cosmetic macros and unknown macros expand to empty string."""
    if depth > 8:
        return text  # avoid infinite recursion

    def _do(m):
        name = m.group(1)
        argstr = m.group(2).strip()
        if name in _COSMETIC_MACROS:
            return ""
        if name in macros:
            params, body = macros[name]
            args = _split_macro_args(argstr)
            # Pad/truncate to param count.
            args = (args + [""] * len(params))[:len(params)]
            sub = body
            for p, a in zip(params, args):
                sub = re.sub(r'\{' + re.escape(p) + r'\}', a, sub)
            # Substitute nested macros in the expansion.
            return _substitute_macros(sub, macros, depth + 1)
        # Inline-include macros like {~add-ons/...} or {core/macros/...}
        if name.startswith("~") or "/" in argstr:
            return ""
        if name not in _WARNED_MACROS:
            _WARNED_MACROS.add(name)
            log.debug(f"unknown macro {{{name} ...}} → substituted as empty")
        return ""

    prev = None
    cur = text
    while prev != cur:
        prev = cur
        cur = _MACRO_INVOKE_RE.sub(_do, cur)
    return cur


# ----------------------------------------------------------------------
# Scenario WML loader
# ----------------------------------------------------------------------

def _load_core_macros() -> Dict[str, Tuple[List[str], str]]:
    """Slurp Wesnoth's data/core/macros/*.cfg for macro definitions
    that scenarios commonly invoke. We don't expand the bodies (most
    are cosmetic anyway); just need names so we don't warn on them."""
    macros: Dict[str, Tuple[List[str], str]] = {}
    macros_dir = WESNOTH_SRC / "data" / "core" / "macros"
    if not macros_dir.exists():
        return macros
    for cfg in macros_dir.glob("*.cfg"):
        try:
            txt = cfg.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        txt = _strip_comments(_strip_textdomain(txt))
        for m in _MACRO_DEFINE_RE.finditer(txt):
            name   = m.group(1)
            params = (m.group(2) or "").split()
            body   = m.group(3)
            macros[name] = (params, body)
    return macros


_CORE_MACROS_CACHE: Optional[Dict[str, Tuple[List[str], str]]] = None


def load_scenario_wml(scenario_id: str) -> Optional[WMLNode]:
    """Find and parse the scenario .cfg matching `scenario_id`. Returns
    the parsed root node (with a [multiplayer] or [scenario] child),
    or None if unmappable. `scenario_id` should match the WML id like
    "multiplayer_Aethermaw" or "multiplayer_Den_of_Onis"."""
    global _CORE_MACROS_CACHE
    if not SCENARIO_DIR.exists():
        return None

    # Map "multiplayer_Aethermaw" → "2p_Aethermaw.cfg"
    base = scenario_id
    if base.startswith("multiplayer_"):
        base = base[len("multiplayer_"):]

    # Known short-id → full-filename overrides. Some mainline
    # scenarios pick a snappier WML `id=` than their filename
    # (Caves of the Basilisk's id=multiplayer_Basilisk vs file
    # 2p_Caves_of_the_Basilisk.cfg). Add new entries as discovered.
    _BASE_OVERRIDES = {
        "Basilisk": "Caves_of_the_Basilisk",
    }
    base = _BASE_OVERRIDES.get(base, base)

    candidate = SCENARIO_DIR / f"2p_{base}.cfg"
    if not candidate.exists():
        # Try without the 2p_ prefix (some scenarios use 4p_, 8p_, etc.)
        for nplayers in (3, 4, 5, 6, 8):
            alt = SCENARIO_DIR / f"{nplayers}p_{base}.cfg"
            if alt.exists():
                candidate = alt; break
        else:
            return None

    if _CORE_MACROS_CACHE is None:
        _CORE_MACROS_CACHE = _load_core_macros()

    raw = candidate.read_text(encoding="utf-8", errors="replace")
    raw = _strip_comments(_strip_textdomain(raw))
    raw = _expand_parallel_assigns(raw)
    raw, scenario_macros = _extract_inline_macros(raw)
    # Merge core macros with this scenario's local macros (local wins).
    all_macros = dict(_CORE_MACROS_CACHE)
    all_macros.update(scenario_macros)
    expanded = _substitute_macros(raw, all_macros)
    return parse_wml(expanded)


# ----------------------------------------------------------------------
# Event extraction
# ----------------------------------------------------------------------

@dataclass
class ScenarioEvent:
    """One [event] block extracted from a scenario .cfg."""
    name: str                         # "prestart", "side 1 turn 4", etc.
    first_time_only: bool = True
    actions: List[WMLNode] = field(default_factory=list)
    fired: bool = False               # latched by the interpreter


def collect_events(root: WMLNode) -> List[ScenarioEvent]:
    """Find every [event] block under [multiplayer] / [scenario] and
    return them in WML-order so the caller can fire them sequentially."""
    out: List[ScenarioEvent] = []
    container = root.first("multiplayer") or root.first("scenario")
    if container is None:
        return out
    for ev in container.all("event"):
        name = ev.attrs.get("name", "").strip().strip('"').lower()
        first_time = ev.attrs.get("first_time_only", "yes").strip().lower() in (
            "yes", "true", "1",
        )
        # The "actions" of an event are its inner WML children except
        # nested [filter] (those are predicates, not actions).
        actions = [ch for ch in ev.children if ch.tag != "filter"]
        out.append(ScenarioEvent(
            name=name, first_time_only=first_time, actions=actions,
        ))
    return out


# ----------------------------------------------------------------------
# Action tag interpreters
# ----------------------------------------------------------------------

def _parse_int_csv(s: str) -> List[int]:
    """Parse "1,2,3" or "1..5" into [1,2,3] or [1,2,3,4,5]. Wesnoth's
    [terrain] tag uses comma-separated lists; the range form is rare
    in events but supported."""
    out: List[int] = []
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if ".." in part:
            a, b = part.split("..", 1)
            try:
                a_i, b_i = int(a), int(b)
                out.extend(range(min(a_i, b_i), max(a_i, b_i) + 1))
            except ValueError:
                pass
        else:
            try:
                out.append(int(part))
            except ValueError:
                pass
    return out


def _terrain_action(gs: GameState, action: WMLNode) -> None:
    """Apply [terrain] x=... y=... terrain=CODE [/terrain] to gs.map.

    Wesnoth allows scalar (x=5,y=3) or list (x=1,2,3 y=4,5,6) forms.
    The list form pairs them positionally: hex (1,4), (2,5), (3,6).
    Coordinates are 1-indexed in WML.

    NOTE on coordinate handling: the raw_map_data string we manipulate
    here STILL INCLUDES the 1-hex border. So WML (X, Y) lives at
    raw_map_data row index Y, col index X (no -1 needed) because the
    border row at index 0 / col 0 occupies those slots. The parsed
    Hex grid is border-stripped (Position(0,0) = WML(1,1)), so we
    convert separately for that lookup.
    """
    xs = _parse_int_csv(action.attrs.get("x", ""))
    ys = _parse_int_csv(action.attrs.get("y", ""))
    new_code = action.attrs.get("terrain", "").strip()
    if not (xs and ys and new_code):
        return
    raw = getattr(gs.global_info, "_raw_map_data", "")
    raw_lines = raw.splitlines() if raw else []
    raw_cells: List[List[str]] = [
        [c.strip() for c in row.split(",")] for row in raw_lines
    ]

    # Decode the new terrain code once — used both for raw_map_data
    # bookkeeping and to update Hex.terrain_types/modifiers on the
    # parsed grid (so combat defense queries see the change).
    from tools.replay_dataset import _parse_hex_code
    from classes import Hex, Position
    new_terr, new_mods = _parse_hex_code(new_code)

    # Strip overlay from the new terrain code; the base part is what
    # the WML defense table is keyed by (e.g., 'Wwf^Bsb/' → 'Wwf').
    new_base_code = new_code.split("^", 1)[0]

    pairs = list(zip(xs, ys))
    for wml_x, wml_y in pairs:
        # Parsed Hex set: 0-indexed playable coords → subtract 1.
        py_x, py_y = wml_x - 1, wml_y - 1
        # Update or insert the parsed Hex so `_terrain_at` reflects
        # the new terrain. Hex is hashable on position, so we discard
        # the old and add a fresh one with the new terrain set.
        old_hex = next(
            (h for h in gs.map.hexes
             if h.position.x == py_x and h.position.y == py_y),
            None,
        )
        if old_hex is not None:
            gs.map.hexes.discard(old_hex)
        gs.map.hexes.add(Hex(
            position=Position(x=py_x, y=py_y),
            terrain_types=set(new_terr),
            modifiers=set(new_mods),
        ))

        # Mirror the change into the per-game terrain-code dict that
        # `_terrain_keys_at` uses for alias resolution. Without this
        # the post-event hex still resolves through the OLD code and
        # combat defense math is wrong (Drake on Ford should get its
        # grass defense, not shallow-water defense).
        codes = getattr(gs.global_info, "_terrain_codes", None)
        if codes is not None:
            codes[(py_x, py_y)] = new_base_code

        # Raw map_data string: border-included → WML (X, Y) is at
        # raw_cells[Y][X] directly (file row Y col X with the border
        # row at index 0).
        if 0 <= wml_y < len(raw_cells) and 0 <= wml_x < len(raw_cells[wml_y]):
            old = raw_cells[wml_y][wml_x]
            prefix = ""
            if len(old) >= 2 and old[0].isdigit() and old[1] == " ":
                prefix = old[:2]
            raw_cells[wml_y][wml_x] = prefix + new_code

    if raw_cells:
        new_raw = "\n".join(", ".join(row) for row in raw_cells)
        setattr(gs.global_info, "_raw_map_data", new_raw)


def _no_op_action(gs: GameState, action: WMLNode) -> None:
    """Cosmetic action — no state change."""
    return


def _modify_side_action(gs: GameState, action: WMLNode) -> None:
    """[modify_side] side=N gold=X / income=Y / recruit=... [/modify_side]"""
    side_num = int(action.attrs.get("side", 0) or 0)
    if not (1 <= side_num <= len(gs.sides)):
        return
    s = gs.sides[side_num - 1]
    new_gold = int(action.attrs.get("gold", s.current_gold) or s.current_gold)
    new_income = int(action.attrs.get("income", s.base_income) or s.base_income)
    new_recruit = action.attrs.get("recruit", "")
    new_recruits = (
        [r.strip() for r in new_recruit.split(",") if r.strip()]
        if new_recruit
        else list(s.recruits)
    )
    gs.sides[side_num - 1] = SideInfo(
        player=s.player, recruits=new_recruits,
        current_gold=new_gold, base_income=new_income,
        nb_villages_controlled=s.nb_villages_controlled,
        faction=s.faction,
    )


# ----------------------------------------------------------------------
# [time_area] / [store_locations] — per-hex ToD overrides
# ----------------------------------------------------------------------
#
# Some 2p maps (Tombs of Kesorak, Elensefar Courtyard) declare regions
# where a different time-of-day cycle applies — e.g., a permanently
# darkened tomb that always reads as second_watch (lawful_bonus=-25)
# regardless of the global cycle. Combat damage on those hexes uses
# that override. We parse [time_area] blocks (both top-level and
# event-fired) and stash a per-hex `cycle` of lawful_bonus values on
# `gs.global_info._time_areas`. Combat consults this map before
# falling back to the default 6-step cycle.

# Lawful_bonus default values for a [time] block that has only an `id=`
# and omits `lawful_bonus`. From data/core/macros/schedules.cfg.
_DEFAULT_LAWFUL_BY_TOD_ID: Dict[str, int] = {
    "dawn":              0,
    "morning":          25,
    "midday":           25,
    "afternoon":        25,
    "dusk":              0,
    "first_watch":     -25,
    "second_watch":    -25,
    "indoors":           0,
    "underground":     -25,
    "underground_illum": 0,
    "deep_underground":-25,
}


def _parse_int_or_range(part: str) -> List[int]:
    """Parse one element of an x= / y= attribute. '5' → [5]; '1-4' →
    [1,2,3,4]; '1..4' → [1,2,3,4] (Wesnoth supports both)."""
    part = part.strip()
    if not part:
        return []
    for sep in ("..", "-"):
        if sep in part:
            a, b = part.split(sep, 1)
            try:
                a_i, b_i = int(a), int(b)
            except ValueError:
                return []
            return list(range(min(a_i, b_i), max(a_i, b_i) + 1))
    try:
        return [int(part)]
    except ValueError:
        return []


def _resolve_xy_attr(x_attr: str, y_attr: str,
                     map_w: int, map_h: int) -> Set[Tuple[int, int]]:
    """Return the set of WML hexes (1-indexed) implied by x= / y=. Two
    forms:
      - Both attrs comma-separated, equal length: pair positionally.
        x="9,10,28,29" y="4,3,20,20" → {(9,4),(10,3),(28,20),(29,20)}.
      - One or both as ranges/wildcards: form the cartesian product.
        x="22-37" (no y) → all hexes in cols 22..37, every row.
    Returns 1-indexed (WML) coords; caller subtracts 1 for Python."""
    out: Set[Tuple[int, int]] = set()
    x_parts = [p.strip() for p in (x_attr or "").split(",") if p.strip()]
    y_parts = [p.strip() for p in (y_attr or "").split(",") if p.strip()]

    # Pairwise mode: equal-length comma-lists of single integers.
    if x_parts and y_parts and len(x_parts) == len(y_parts) and all(
        "-" not in p and ".." not in p for p in x_parts + y_parts
    ):
        for xs, ys in zip(x_parts, y_parts):
            try:
                out.add((int(xs), int(ys)))
            except ValueError:
                pass
        return out

    # Cross-product mode (with ranges, missing axis = whole map).
    xs: List[int] = []
    for p in x_parts:
        xs.extend(_parse_int_or_range(p))
    if not xs:
        xs = list(range(1, map_w + 1))
    ys: List[int] = []
    for p in y_parts:
        ys.extend(_parse_int_or_range(p))
    if not ys:
        ys = list(range(1, map_h + 1))
    for x in xs:
        for y in ys:
            out.add((x, y))
    return out


def _terrain_filter_match(code: str, pattern: str) -> bool:
    """Match a terrain code against one Wesnoth filter pattern. Wesnoth
    supports `*` wildcards and comma-separated alternatives; we handle
    the common forms used in 2p maps (e.g., "R*", "Rr,Xos", "Gg^Emf")."""
    if not pattern:
        return True
    for alt in pattern.split(","):
        alt = alt.strip()
        if not alt:
            continue
        if alt == code:
            return True
        if "*" in alt:
            # Convert glob to regex: '*' → '.*'; literal-escape the rest.
            rx = re.escape(alt).replace(r"\*", ".*")
            if re.fullmatch(rx, code):
                return True
    return False


def _eval_location_clause(gs: GameState, clause: WMLNode,
                          map_w: int, map_h: int) -> Set[Tuple[int, int]]:
    """Evaluate a single [store_locations] clause (or nested [or]).
    Honors x=, y= (ranges/lists), and terrain=. Returns 1-indexed
    WML hexes."""
    hexes = _resolve_xy_attr(
        clause.attrs.get("x", ""), clause.attrs.get("y", ""), map_w, map_h,
    )
    terrain_pat = clause.attrs.get("terrain", "").strip()
    if terrain_pat:
        codes = getattr(gs.global_info, "_terrain_codes", {}) or {}
        kept: Set[Tuple[int, int]] = set()
        for (wx, wy) in hexes:
            # WML 1-indexed → Python 0-indexed for the codes dict.
            code = codes.get((wx - 1, wy - 1), "")
            if _terrain_filter_match(code, terrain_pat):
                kept.add((wx, wy))
        hexes = kept
    return hexes


def _store_locations_action(gs: GameState, action: WMLNode) -> None:
    """[store_locations] variable=NAME x=… y=… [or]…[/or] [/store_locations]
    Stores a set of hexes (Python 0-indexed) under `gs.global_info.
    _scenario_vars[NAME]` so a later [time_area] find_in=NAME can read
    them. We don't model unit-filter / radius (rare in 2p mainline)."""
    var_name = action.attrs.get("variable", "").strip()
    if not var_name:
        return
    map_w, map_h = gs.map.size_x, gs.map.size_y
    hexes = _eval_location_clause(gs, action, map_w, map_h)
    for sub in action.all("or"):
        hexes |= _eval_location_clause(gs, sub, map_w, map_h)
    # Convert to Python 0-indexed for downstream consumers and drop any
    # hexes outside the playable area (off-map clauses can sneak in
    # from WML coords like y=0, which becomes Python y=-1 and isn't a
    # real hex).
    py_hexes = {
        (wx - 1, wy - 1) for (wx, wy) in hexes
        if 0 < wx <= map_w and 0 < wy <= map_h
    }
    vars_ = getattr(gs.global_info, "_scenario_vars", None)
    if vars_ is None:
        vars_ = {}
        setattr(gs.global_info, "_scenario_vars", vars_)
    vars_[var_name] = py_hexes


def _clear_variable_action(gs: GameState, action: WMLNode) -> None:
    """[clear_variable] name=NAME [/clear_variable]"""
    var_name = action.attrs.get("name", "").strip()
    vars_ = getattr(gs.global_info, "_scenario_vars", None) or {}
    vars_.pop(var_name, None)


def _parse_time_cycle(ta_node: WMLNode) -> List[int]:
    """Read the [time] children of a [time_area] and return a list of
    lawful_bonus values, one per cycle position. Single-entry lists are
    valid (constant ToD); the engine indexes by `(turn-1) % len`.

    NOTE: We don't try to merge `[+time]` overrides — Wesnoth's `[+]`
    syntax doesn't round-trip through our line-based WML parser
    (it's silently dropped). Auditing 2p uses of [+time] (Tombs of
    Kesorak), each `[+time]` only restates the same lawful_bonus
    already set by the preceding macro, so dropping them is exact for
    those scenarios. If a future scenario uses [+time] to actually
    change lawful_bonus we'll need a real parser tweak."""
    cycle: List[int] = []
    for ch in ta_node.children:
        if ch.tag != "time":
            continue
        lb_raw = ch.attrs.get("lawful_bonus")
        if lb_raw is None or lb_raw == "":
            tod_id = ch.attrs.get("id", "").strip()
            lb = _DEFAULT_LAWFUL_BY_TOD_ID.get(tod_id, 0)
        else:
            try:
                lb = int(lb_raw)
            except ValueError:
                lb = 0
        cycle.append(lb)
    return cycle


def _time_area_action(gs: GameState, action: WMLNode) -> None:
    """[time_area] x=… y=… (or find_in=VAR) [time]…[/time] [/time_area]
    Stamps a per-hex lawful_bonus cycle onto `gs.global_info._time_areas`.
    Multiple time_areas can stack; later writes win on overlapping hexes,
    matching Wesnoth's "last [time_area] applied wins" rule."""
    cycle = _parse_time_cycle(action)
    if not cycle:
        return

    # Resolve the hex set: either explicit x=/y=, or find_in=variable.
    map_w, map_h = gs.map.size_x, gs.map.size_y
    py_hexes: Set[Tuple[int, int]] = set()
    find_in = action.attrs.get("find_in", "").strip()
    if find_in:
        vars_ = getattr(gs.global_info, "_scenario_vars", None) or {}
        py_hexes = set(vars_.get(find_in, set()))
    else:
        wml_hexes = _resolve_xy_attr(
            action.attrs.get("x", ""), action.attrs.get("y", ""),
            map_w, map_h,
        )
        py_hexes = {
            (wx - 1, wy - 1) for (wx, wy) in wml_hexes
            if 0 < wx <= map_w and 0 < wy <= map_h
        }

    if not py_hexes:
        return

    areas = getattr(gs.global_info, "_time_areas", None)
    if areas is None:
        areas = {}
        setattr(gs.global_info, "_time_areas", areas)
    for h in py_hexes:
        areas[h] = list(cycle)


def setup_static_time_areas(gs: GameState, root: WMLNode) -> None:
    """Process top-level [time_area] blocks declared directly under
    [multiplayer] / [scenario] (NOT inside an [event]). These apply
    from game start — Tombs of Kesorak's three zones are this form.
    Called once during scenario setup."""
    container = root.first("multiplayer") or root.first("scenario")
    if container is None:
        return
    for ta in container.all("time_area"):
        _time_area_action(gs, ta)


def _gold_action(gs: GameState, action: WMLNode) -> None:
    """[gold] side=N amount=X [/gold]"""
    side_num = int(action.attrs.get("side", 0) or 0)
    amount   = int(action.attrs.get("amount", 0) or 0)
    if not (1 <= side_num <= len(gs.sides)):
        return
    s = gs.sides[side_num - 1]
    gs.sides[side_num - 1] = SideInfo(
        player=s.player, recruits=s.recruits,
        current_gold=s.current_gold + amount, base_income=s.base_income,
        nb_villages_controlled=s.nb_villages_controlled,
        faction=s.faction,
    )


# ----------------------------------------------------------------------
# WML variables, control flow, and unit spawning
# ----------------------------------------------------------------------
# A handful of MP scenarios (Hornshark Island most prominently) place
# pre-game units via a `[switch] variable=pN_faction` inside an [event]
# triggered from prestart. Without these handlers, our reconstructor
# starts every Hornshark replay missing 4-6 named units per side, which
# cascades into "src_missing"/"final_occupied" failures from cmd[1]
# onward. The implementation is deliberately narrow: only the WML
# patterns we've seen in mainline 2p scenarios.

def _wml_vars(gs: GameState) -> Dict[str, str]:
    """Lazily-stash dict of WML variable name -> string value on
    `gs.global_info`. Mirrors Wesnoth's `wml.variables[]` namespace."""
    v = getattr(gs.global_info, "_wml_variables", None)
    if v is None:
        v = {}
        setattr(gs.global_info, "_wml_variables", v)
    return v


def _set_variable_action(gs: GameState, action: WMLNode) -> None:
    """Implement `[set_variable] name=X value=Y`. Wesnoth supports many
    operators (`add`, `multiply`, `to_variable`, `random`); we handle
    the common scalar-set form which is enough for Hornshark + a
    handful of similar scenarios."""
    name = action.attrs.get("name", "").strip().strip('"')
    if not name:
        return
    if "value" in action.attrs:
        _wml_vars(gs)[name] = action.attrs["value"].strip().strip('"')
    elif "literal" in action.attrs:
        _wml_vars(gs)[name] = action.attrs["literal"].strip().strip('"')
    elif "add" in action.attrs:
        try:
            cur = int(_wml_vars(gs).get(name, "0"))
        except ValueError:
            cur = 0
        try:
            inc = int(action.attrs["add"].strip().strip('"'))
        except ValueError:
            inc = 0
        _wml_vars(gs)[name] = str(cur + inc)


_FACTION_LUA_RE = re.compile(
    r'wml\.variables\s*\[\s*"p"\s*\.\.\s*tostring\(\s*i\s*\)\s*\.\.\s*"_faction"\s*\]\s*=\s*side\.faction',
    re.S,
)


def _lua_action(gs: GameState, action: WMLNode) -> None:
    """Recognise the one Lua pattern Hornshark Island uses to publish
    each side's faction as a WML variable, and emulate it. Anything
    else falls through as a no-op (we don't run a Lua interpreter)."""
    code = action.attrs.get("code", "")
    if _FACTION_LUA_RE.search(code):
        for i, s in enumerate(gs.sides, start=1):
            _wml_vars(gs)[f"p{i}_faction"] = s.faction or ""


def _fire_event_action(gs: GameState, action: WMLNode) -> None:
    """`[fire_event] name=X` triggers another named [event] from inside
    the current event's action list (Hornshark uses this from prestart
    to call into `place_units`). Honors `first_time_only` like the
    public fire_event entry point."""
    name = action.attrs.get("name", "").strip().strip('"').lower()
    if not name:
        return
    events = getattr(gs.global_info, "_scenario_events", None)
    if not events:
        return
    for ev in events:
        if ev.name != name:
            continue
        if ev.first_time_only and ev.fired:
            continue
        for child in ev.actions:
            _apply_action(gs, child)
        ev.fired = True


def _switch_action(gs: GameState, action: WMLNode) -> None:
    """`[switch] variable=X [case] value=V ... [/case] ...` selects the
    [case] whose `value=` matches the variable's current value (or
    `[else]`) and executes its inner actions. Multiple matching values
    can be comma-separated in `value=`."""
    var_name = action.attrs.get("variable", "").strip().strip('"')
    if not var_name:
        return
    cur = _wml_vars(gs).get(var_name, "")
    matched_case: Optional[WMLNode] = None
    else_case: Optional[WMLNode] = None
    for child in action.children:
        if child.tag == "case":
            vals = [v.strip() for v in
                    (child.attrs.get("value", "") or "").split(",")]
            if cur in vals:
                matched_case = child
                break
        elif child.tag == "else" and else_case is None:
            else_case = child
    target = matched_case or else_case
    if target is None:
        return
    for sub in target.children:
        _apply_action(gs, sub)


_TRAIT_MACRO_RE = re.compile(r'^TRAIT_(\w+)$')


def _trait_ids_from_modifications(node: WMLNode) -> List[str]:
    """Walk a `[modifications]` child node and pull trait ids from
    nested [trait] children. The `{TRAIT_LOYAL}` macros are pre-
    expanded by the macro substitution pass into `[trait]id=loyal[/trait]`,
    which appears here as a child node we can read. We also accept
    raw `id=loyal` attrs on the modifications node itself for
    robustness against macro-expansion edge cases."""
    out: List[str] = []
    for ch in node.children:
        if ch.tag == "trait":
            tid = (ch.attrs.get("id", "") or "").strip().strip('"').lower()
            if tid:
                out.append(tid)
    return out


def _unit_action(gs: GameState, action: WMLNode) -> None:
    """Spawn a unit on the map. Used by Hornshark-style pre-placed
    units in scenario [event]s. Reads side, type, x, y, optional name,
    and an optional `[modifications]` block of `[trait]` children.

    Coordinates in WML are 1-indexed; we convert to our internal
    0-indexed before placing on the map."""
    try:
        side = int(action.attrs.get("side", "0").strip().strip('"'))
    except ValueError:
        return
    if side <= 0:
        return
    utype = (action.attrs.get("type", "") or "").strip().strip('"')
    if not utype:
        return
    # `variation=...` (Hornshark's "Soulless variation=saurian" = the
    # named hero "Rzrrt the Dauntless" with saurian movement_type and
    # defenses, NOT the base humanoid Soulless). Wesnoth resolves this
    # to a unit-type lookup `Soulless:saurian` in our scrape (the
    # scraper expanded variations into separate units). If the
    # composite key isn't in the DB, fall back to base type.
    variation = (action.attrs.get("variation", "") or "").strip().strip('"')
    try:
        wml_x = int(action.attrs.get("x", "0").strip().strip('"'))
        wml_y = int(action.attrs.get("y", "0").strip().strip('"'))
    except ValueError:
        return
    if wml_x <= 0 or wml_y <= 0:
        return
    # Defer import to avoid circular: replay_dataset imports us.
    from tools.replay_dataset import (
        _build_unit, _UNIT_DB, _stats_for,
    )
    if variation:
        composite = f"{utype}:{variation}"
        if composite in _UNIT_DB:
            utype = composite
    from tools.traits import apply_traits_to_unit

    # Generate a fresh uid: max existing (numeric) uid + 1.
    max_uid = 0
    for u in gs.map.units:
        try:
            n = int(u.id.lstrip("u"))
            if n > max_uid:
                max_uid = n
        except ValueError:
            continue
    uid = max_uid + 1
    udict = {
        "uid": uid,
        "type": utype,
        "side": side,
        "x": wml_x - 1,
        "y": wml_y - 1,
        "is_leader": False,
    }
    # Honor the scenario's experience_modifier (default 100; common
    # ladder games run at 70%). Without this, scenario-event-placed
    # heroes (Hornshark Island's Sorrek/Rukhos Skeleton, the Drake
    # Fighter "Rawffus", etc.) keep their base max_exp and don't
    # advance at the same xp threshold real recruits hit. Concrete:
    # 2p_Hornshark_Island_Turn_12_(103721) cmd[311]: Skeleton "Sorrek"
    # advanced to Deathblade (movement=6) on turn 9 in Wesnoth at
    # xp=31/27 (39 * 0.7 = 27), but our sim kept him at Skeleton with
    # xp=31/39 because exp_modifier defaulted to 100 — Deathblade's 6
    # MP would have made the 6-cost path on turn 11 valid.
    exp_mod = int(getattr(gs.global_info, "_experience_modifier", 100) or 100)
    base_unit = _build_unit(udict, apply_leader_traits=False,
                            exp_modifier=exp_mod)
    # Pull `musthave` traits from the unit type's stats. Soulless,
    # Walking Corpse, Vampire Bat, and other undead/mechanical/elemental
    # units have musthave=['undead', 'fearless'] (or 'mechanical' /
    # 'elemental') that the type macros guarantee. _build_unit gives us
    # an empty trait set; we must merge musthaves before processing
    # the [modifications] block so the resulting trait set matches what
    # Wesnoth would have post-`new_unit_construction`. Concrete:
    # 2p__Hornshark_Island_Turn_10_(176667).bz2 cmd[234] -- a preplaced
    # Soulless:dwarf attacks a poisoned Elvish Archer at hp 12 in day
    # ToD (lawful=+25). Without `fearless`, chaotic Soulless dmg = 7 *
    # 0.75 = 5; with fearless, dmg = 7 (no penalty). 2 hits at 7 = 14
    # kills the archer (and plagues it); 2 hits at 5 = 10 leaves the
    # archer at hp=2 in our sim, surviving until turn 9 cmd[257] when
    # u28 Mage tries to attack the now-empty hex and sees friendly
    # fire on the still-living archer.
    stats = _stats_for(utype)
    musthave_ids = list((stats.get("traits", {}) or {}).get("musthave", []) or [])
    # Apply explicit [modifications]/[trait] traits.
    mods = action.first("modifications")
    explicit_ids = _trait_ids_from_modifications(mods) if mods is not None else []
    # Merge musthave + explicit; deduplicate while preserving order.
    seen = set()
    trait_ids = []
    for tid in list(musthave_ids) + list(explicit_ids):
        if tid not in seen:
            trait_ids.append(tid)
            seen.add(tid)
    if trait_ids:
        defense_table = dict(getattr(base_unit, "_defense_table", {})
                             or stats.get("defense", {}))
        # `stats.get("level", 1) or 1` was wrongly coercing level-0
        # (Walking Corpse / Vampire Bat / Mudcrawler / statue side-3
        # units) to 1 because 0 is falsy. That defeated Stage 4's fix
        # and gave level-0 units +1 HP per resilient/healthy. Use the
        # raw level instead.
        lvl_raw = stats.get("level", 1)
        try:
            lvl = int(lvl_raw)
        except (TypeError, ValueError):
            lvl = 1
        base_unit = apply_traits_to_unit(
            base_unit, trait_ids, level=lvl,
            defense_table=defense_table,
        )
        setattr(base_unit, "_defense_table", defense_table)
        # Preserve trait order through advancement. For preplaced
        # units, the order comes from the [modifications]/[trait]
        # children's document order in the scenario WML (musthave
        # traits first, then explicit ones).
        setattr(base_unit, "_trait_order", list(trait_ids))
    # Refresh current_hp = max_hp post-traits (a freshly placed unit
    # spawns at full health).
    from dataclasses import replace as _dc_replace
    base_unit = _dc_replace(
        base_unit, current_hp=base_unit.max_hp,
        current_moves=base_unit.max_moves,
    )
    # Apply CUSTOM trait [effect]s (NOT named traits already handled
    # by apply_traits_to_unit). Examples: Caves of the Basilisk's
    # `id=remove_hp` trait whose [effect]s drop hp/movement to make
    # statues 1-HP non-actors, or Sullas Ruins' identical setup. These
    # are NOT in our TRAITS registry; their behaviour lives entirely
    # in the [trait]'s [effect] children.
    # Skip named traits (loyal/quick/resilient/strong/intelligent/etc.)
    # whose [effect]s are already applied by `apply_traits_to_unit` --
    # double-applying them caused Hornshark Sergeants/Drake Fighters
    # to lose 1 movement (resilient + quick stack incorrectly).
    if mods is not None:
        from tools.traits import TRAITS as _NAMED_TRAITS
        for trait_node in mods.all("trait"):
            tid = (trait_node.attrs.get("id", "") or "").strip().strip('"').lower()
            if tid in _NAMED_TRAITS:
                continue
            for eff in trait_node.all("effect"):
                _apply_effect_to_unit(base_unit, eff)
    # Apply petrified status from `[status] petrified=yes`.
    status_node = action.first("status")
    if status_node is not None:
        petr = (status_node.attrs.get("petrified", "") or "").strip().lower()
        if petr in ("yes", "true", "1"):
            new_st = set(base_unit.statuses); new_st.add("petrified")
            base_unit = _dc_replace(base_unit, statuses=new_st,
                                    current_moves=0, has_attacked=True,
                                    attacks=[])
    # Apply [abilities] block from the [unit] action. Hornshark Island's
    # preplaced Mermaid Initiates have `[abilities] {ABILITY_HEALS}
    # [/abilities]` granting heals_4, and the Soulless heroes get
    # `{ABILITY_AMBUSH}`. Without parsing this block, the preplaced
    # Mermaid doesn't heal adjacent allies at init_side, causing the
    # turn-7 Elvish Scout to enter turn 8 at hp=26 instead of 30 (and
    # die to a 14-dmg/strike Spearman attack). Map [tag]→canonical id
    # via the same convention scrape_unit_stats uses.
    abil_node = action.first("abilities")
    if abil_node is not None:
        # Tag-name → canonical id used by tools/abilities.py and combat.
        # Mirrors the result of scrape_unit_stats's ABILITY_MACROS for
        # the post-macro-expansion form (each macro emits [heals],
        # [hides], [leadership], etc. as children).
        _TAG_TO_ABILITY = {
            "heals": None,             # value-dependent: heals_4 or heals_8
            "regenerate": "regenerate",
            "cures": "cures",
            "leadership": "leadership",
            "skirmisher": "skirmisher",
            "illuminates": "illuminates",
            "teleport": "teleport",
            "hides": "ambush",         # ABILITY_AMBUSH expands to [hides]
            "feeding": "feeding",
            "steadfast": "steadfast",
        }
        new_abilities = set(base_unit.abilities)
        for child in abil_node.children:
            tag = child.tag
            if tag == "heals":
                # ABILITY_HEALS expands to [heals] value=4. ABILITY_HEALS_8
                # / ABILITY_EXTRA_HEAL expand to [heals] value=8.
                try:
                    val = int((child.attrs.get("value", "4") or "4").strip().strip('"'))
                except (TypeError, ValueError):
                    val = 4
                new_abilities.add("heals_8" if val >= 8 else "heals_4")
            elif tag in _TAG_TO_ABILITY:
                aid = _TAG_TO_ABILITY[tag]
                if aid:
                    new_abilities.add(aid)
        if new_abilities != base_unit.abilities:
            base_unit = _dc_replace(base_unit, abilities=new_abilities)
    gs.map.units.add(base_unit)
    # Bump Wesnoth's monotonic next_unit_id counter — Wesnoth's
    # prestart [unit] events also assign sequential uids.
    cur = int(getattr(gs.global_info, "_next_uid_counter", 1) or 1)
    setattr(gs.global_info, "_next_uid_counter", cur + 1)


def _parse_increase(raw: str, base: int) -> int:
    """Mirror Wesnoth's `apply_modifier(base, increase_str)` for a
    single integer value. Accepts plain ints (`-1`, `2`) and percent
    strings (`-100%`, `50%`). Falls back to 0 on malformed input.

    For percent values Wesnoth's `apply_modifier` uses
    `div100rounded(base * pct)` (round-half-away-from-zero with +50
    bias); plain ints just add.
    """
    s = (raw or "").strip().strip('"')
    if not s:
        return 0
    if s.endswith("%"):
        try:
            pct = int(s[:-1])
        except ValueError:
            return 0
        raw_v = base * pct
        if raw_v < 0:
            return -(((-raw_v) + 50) // 100)
        return (raw_v + 50) // 100
    try:
        return int(s)
    except ValueError:
        return 0


def _apply_effect_to_unit(u, eff: WMLNode) -> None:
    """Apply a single `[effect]` block's mutation to one unit, in
    place. Supports the apply_to forms used by 2p ladder scenarios
    (Hornshark Island, Caves of the Basilisk, Sullas Ruins,
    Silverhead Crossing, Thousand Stings Garrison):

      - `apply_to=attack [+ range= +/-/set_specials/increase_attacks/
                         increase_damage/set_attack_weight]`
      - `apply_to=new_attack` (add a new Attack)
      - `apply_to=remove_attacks` (clear all attacks)
      - `apply_to=hitpoints` (increase_total, set)
      - `apply_to=movement` (set, increase)
      - `apply_to=status` (add named status flag)
      - `apply_to=ellipse` / `image_mod` / `overlay` / `profile`
        / `new_animation`: cosmetic, no-op.
    """
    apply_to = (eff.attrs.get("apply_to", "") or "").strip().strip('"')
    from classes import Attack
    from combat import DAMAGE_TYPES
    from dataclasses import replace as _dc_replace

    if apply_to == "attack":
        weapon_range = (eff.attrs.get("range", "") or "").strip().strip('"')
        weapon_name = (eff.attrs.get("name", "") or "").strip().strip('"')
        ss = eff.first("set_specials")
        new_specials = {ch.tag for ch in ss.children} if ss is not None else set()
        inc_attacks_raw = eff.attrs.get("increase_attacks", "")
        inc_damage_raw  = eff.attrs.get("increase_damage", "")
        new_attacks = []
        for atk in u.attacks:
            match = True
            if weapon_range:
                if weapon_range == "ranged" and not atk.is_ranged:
                    match = False
                elif weapon_range == "melee" and atk.is_ranged:
                    match = False
            # `name=` matches the weapon's display name. Our `Attack`
            # doesn't carry a name, so we can't filter by name here;
            # if a name filter is given, only fall back to range.
            # (Default-era 2p [object]s in scope don't filter by name
            # alone -- always with range -- so this is safe.)
            if match:
                new_dmg = atk.damage_per_strike + _parse_increase(
                    inc_damage_raw, atk.damage_per_strike)
                new_n = atk.number_strikes + _parse_increase(
                    inc_attacks_raw, atk.number_strikes)
                new_dmg = max(0, new_dmg)
                new_n = max(0, new_n)
                merged_specials = set(atk.weapon_specials) | new_specials
                new_attacks.append(Attack(
                    type_id=atk.type_id,
                    number_strikes=new_n,
                    damage_per_strike=new_dmg,
                    is_ranged=atk.is_ranged,
                    weapon_specials=merged_specials,
                ))
            else:
                new_attacks.append(atk)
        u.attacks = new_attacks
        return

    if apply_to == "new_attack":
        wrange = (eff.attrs.get("range", "") or "melee").strip().strip('"')
        wtype  = (eff.attrs.get("type", "") or "blade").strip().strip('"')
        try:
            damage = int((eff.attrs.get("damage", "0") or "0").strip().strip('"'))
        except ValueError:
            damage = 0
        try:
            number = int((eff.attrs.get("number", "1") or "1").strip().strip('"'))
        except ValueError:
            number = 1
        ss = eff.first("specials") or eff.first("set_specials")
        specials = {ch.tag for ch in ss.children} if ss is not None else set()
        # Map type string -> DamageType enum index.
        try:
            type_id_idx = DAMAGE_TYPES.index(wtype.lower())
        except ValueError:
            type_id_idx = 0  # fallback to blade
        from classes import DamageType
        try:
            type_id = list(DamageType)[type_id_idx]
        except (ValueError, IndexError):
            type_id = list(DamageType)[0]
        u.attacks = list(u.attacks) + [Attack(
            type_id=type_id,
            number_strikes=number,
            damage_per_strike=damage,
            is_ranged=(wrange == "ranged"),
            weapon_specials=specials,
        )]
        return

    if apply_to == "remove_attacks":
        u.attacks = []
        return

    if apply_to == "hitpoints":
        inc_raw = eff.attrs.get("increase_total", "")
        if inc_raw:
            delta = _parse_increase(inc_raw, u.max_hp)
            new_max = max(1, u.max_hp + delta)
            new_cur = max(1, min(u.current_hp + delta, new_max))
            u.max_hp = new_max
            u.current_hp = new_cur
        set_raw = eff.attrs.get("set", "")
        if set_raw:
            try:
                v = int(set_raw)
                u.current_hp = max(1, min(v, u.max_hp))
            except ValueError:
                pass
        heal_raw = eff.attrs.get("heal_full", "")
        if (heal_raw or "").strip().lower() in ("yes", "true", "1"):
            u.current_hp = u.max_hp
        return

    if apply_to == "movement":
        set_raw = eff.attrs.get("set", "")
        inc_raw = eff.attrs.get("increase", "")
        if set_raw:
            try:
                u.max_moves = max(0, int(set_raw))
                u.current_moves = min(u.current_moves, u.max_moves)
            except ValueError:
                pass
        elif inc_raw:
            u.max_moves = max(0, u.max_moves + _parse_increase(
                inc_raw, u.max_moves))
            u.current_moves = min(u.current_moves, u.max_moves)
        return

    if apply_to == "status":
        # `[effect] apply_to=status add=poisoned [/effect]` style.
        add = (eff.attrs.get("add", "") or "").strip().strip('"')
        rem = (eff.attrs.get("remove", "") or "").strip().strip('"')
        if add:
            new_st = set(u.statuses); new_st.add(add); u.statuses = new_st
        if rem and rem in u.statuses:
            new_st = set(u.statuses); new_st.discard(rem); u.statuses = new_st
        return

    # apply_to in {ellipse, image_mod, overlay, profile, new_animation,
    # halo, zoc} are cosmetic / display-only -- silently ignored.


def _object_action(gs: GameState, action: WMLNode) -> None:
    """Handle a top-level [object] block (Wesnoth's mid-event unit
    modifier). The [object] declares a [filter] (which units to
    affect) and one or more [effect] blocks (what to change).

    Routes each [effect] through `_apply_effect_to_unit`, which
    handles the apply_to forms used by 2p ladder scenarios
    (Hornshark Island, Silverhead Crossing, Thousand Stings
    Garrison, Caves of the Basilisk, Sullas Ruins).
    """
    filt = action.first("filter")
    if filt is None:
        return
    # Resolve the filter's hex set.
    map_w, map_h = gs.map.size_x, gs.map.size_y
    wml_hexes = _resolve_xy_attr(
        filt.attrs.get("x", ""), filt.attrs.get("y", ""),
        map_w, map_h,
    )
    py_hexes = {(wx - 1, wy - 1) for (wx, wy) in wml_hexes
                if 0 < wx <= map_w and 0 < wy <= map_h}
    type_filter = (filt.attrs.get("type", "") or "").strip().strip('"')
    side_filter_raw = (filt.attrs.get("side", "") or "").strip().strip('"')
    try:
        side_filter = int(side_filter_raw) if side_filter_raw else 0
    except ValueError:
        side_filter = 0

    targets = []
    for u in gs.map.units:
        if py_hexes and (u.position.x, u.position.y) not in py_hexes:
            continue
        if type_filter and u.name != type_filter:
            continue
        if side_filter and u.side != side_filter:
            continue
        targets.append(u)
    if not targets:
        return
    for eff in action.all("effect"):
        for u in targets:
            _apply_effect_to_unit(u, eff)


# Action-tag dispatch table.
_ACTION_HANDLERS: Dict[str, Callable[[GameState, WMLNode], None]] = {
    "terrain":         _terrain_action,
    "modify_side":     _modify_side_action,
    "gold":            _gold_action,
    "store_locations": _store_locations_action,
    "clear_variable":  _clear_variable_action,
    "time_area":       _time_area_action,
    "object":          _object_action,
    # WML control flow + variables (Hornshark Island pre-placed units).
    "set_variable":    _set_variable_action,
    "fire_event":      _fire_event_action,
    "switch":          _switch_action,
    "lua":             _lua_action,
    "unit":            _unit_action,
    # state-affecting tags we don't (yet) interpret.
    "remove_unit": _no_op_action,
    "modify_unit": _no_op_action,
    "store_unit":  _no_op_action,
    # cosmetic / no-op
    "message":     _no_op_action,
    "note":        _no_op_action,
    "objectives":  _no_op_action,
    "objective":   _no_op_action,
    "item":        _no_op_action,
    "label":       _no_op_action,
    "music":       _no_op_action,
    "sound":       _no_op_action,
    "endlevel":    _no_op_action,
    "variable":    _no_op_action,
    "case":        _no_op_action,
    "if":          _no_op_action,
}


def _apply_action(gs: GameState, action: WMLNode) -> None:
    handler = _ACTION_HANDLERS.get(action.tag, _no_op_action)
    handler(gs, action)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def fire_event(gs: GameState, events: List[ScenarioEvent], trigger: str) -> int:
    """Fire every event whose name matches `trigger`. Returns the number
    of events fired. Latches `first_time_only` so subsequent calls with
    the same trigger don't re-fire."""
    n = 0
    for ev in events:
        if ev.name != trigger:
            continue
        if ev.first_time_only and ev.fired:
            continue
        for action in ev.actions:
            _apply_action(gs, action)
        ev.fired = True
        n += 1
    return n


def load_events_for_scenario(scenario_id: str) -> List[ScenarioEvent]:
    """Convenience wrapper: load the .cfg and return its events. Returns
    an empty list if the scenario isn't found in the source tree."""
    root = load_scenario_wml(scenario_id)
    if root is None:
        return []
    return collect_events(root)


__all__ = [
    "ScenarioEvent", "load_scenario_wml", "load_events_for_scenario",
    "collect_events", "fire_event", "setup_static_time_areas",
]
