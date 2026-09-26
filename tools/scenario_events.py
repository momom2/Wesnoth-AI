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

import contextvars
import copy as _copy
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Set, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wesnoth_ai.classes import GameState, Hex, Position, SideInfo
from wesnoth_ai.paths import ADDONS_DIR, WESNOTH_SRC_DIR
from tools.replay_extract import WMLNode, parse_wml
from tools.wml_state import wml_int


log = logging.getLogger("scenario_events")

WESNOTH_SRC = WESNOTH_SRC_DIR
SCENARIO_DIR = WESNOTH_SRC / "data" / "multiplayer" / "scenarios"


# ----------------------------------------------------------------------
# WML macro pre-processor
# ----------------------------------------------------------------------

# Macros we expand to nothing, in two classes, because the reason
# matters and one set could not record it.
#
# COSMETIC: presentation only. Deleting the body changes nothing any
# consumer reads. A macro qualifies only if that is true of its WHOLE
# body. `DEFAULT_SCHEDULE` sat here until 2026-09-22 and did not
# qualify -- it carries the board's six [time] blocks, so our
# expansion emitted none where the game's emits six, and the cycle
# came off a hardcoded constant instead of the scenario. Nothing
# compared the two renderings of one file, which is why the gap went
# unseen for as long as it did; tools/analysis/expansion_diff.py is
# that comparison, and tests/test_expansion_diff.py keeps it.
_COSMETIC_MACROS: Set[str] = {
    "FLASH_WHITE", "QUAKE", "PLACE_IMAGE", "PLACE_HALO",
    "DEFAULT_MUSIC_PLAYLIST",
    "UNDEAD_MUSIC", "LOYALIST_MUSIC", "REBELS_MUSIC",
    "ITM_FOREST_FOG", "BIGMAP", "IS_LAST_SCENARIO",
}

# SUBSTITUTED: the behaviour is real and the sim implements it
# natively, so the WML body is dropped rather than ignored. Each entry
# names what stands in for it; a reader who deletes the substitute
# must put the macro back.
_SUBSTITUTED_MACROS: Dict[str, str] = {
    # The turn-cap tiebreak. wesnoth_sim scores a game that reaches
    # the turn limit itself, so the era's lua hook is not needed.
    "TURNS_OVER_ADVANTAGE": "wesnoth_sim turn-cap scoring",
}

# Unknown macros also expand to nothing, with a warning the first
# time. That path is the one that hides bugs, so it is narrow by
# design: anything reached through it is UNCLASSIFIED, not cosmetic.

# Track which macros we've warned about to avoid log spam.
_WARNED_MACROS: Set[str] = set()


_MACRO_DEFINE_RE = re.compile(
    # Use `[ \t]+` (NOT `\s+`) for the gap between the macro name and
    # its params — `\s+` matches newlines too, which causes parameter-
    # less macros like `#define SECOND_WATCH\n  [time]…` to swallow the
    # opening `[time]` of the body as a "parameter". That breaks every
    # ToD macro the time_area scenarios rely on.
    # The name may contain ':' -- the core macros declare
    # `#define INTERNAL:SPECIAL_NOTES_SUBMERGE`, and `\w+` stopped at
    # the colon, so every such macro was defined under the name
    # "INTERNAL" and every use of one expanded to nothing.
    r'#define[ \t]+([\w:]+)(?:[ \t]+([^\n]+))?\n(.*?)#enddef',
    re.DOTALL,
)
# {MACRO_NAME arg1 arg2 ...} — args can be quoted strings, bare words,
# or numeric values. We capture the whole brace expression.
_MACRO_INVOKE_RE = re.compile(r'\{([A-Z_][A-Z0-9_:]*)([^{}]*)\}')


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
                                                   "#else", "#undef",
                                                   "#arg", "#endarg")):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


class MacroDef(NamedTuple):
    """A `#define`: its positional parameters, its body, and the
    defaults of its OPTIONAL named arguments."""
    params: List[str]
    body: str
    defaults: Dict[str, str]


# `#arg NAME` / <default value> / `#endarg` inside a macro body: an
# optional named argument. Unhandled until 2026-09-22, which left
# `{OVERLAY}` unsubstituted in TRAIT_LOYAL and the default value
# stranded in the body as a bare line.
_MACRO_ARG_RE = re.compile(r'#arg[ \t]+(\w+)[ \t]*\n(.*?)#endarg[ \t]*\n?',
                           re.DOTALL)


def _split_optional_args(body: str) -> Tuple[str, Dict[str, str]]:
    defaults: Dict[str, str] = {}

    def _take(m):
        defaults[m.group(1)] = m.group(2).strip()
        return ""

    return _MACRO_ARG_RE.sub(_take, body), defaults


# ----------------------------------------------------------------------
# Preprocessor conditionals
# ----------------------------------------------------------------------
#
# `#ifdef SYM` / `#ifndef SYM` / `#else` / `#endif`, evaluated the way
# Wesnoth's preprocessor does (1.18.4 src/serialization/preprocessor.cpp
# :1322-1400): a branch is live when SYM is in the define set, `#ifndef`
# negates, `#else` flips, `#endif` closes, and they nest. The define set
# holds preprocessor symbols AND every macro `#define`d so far
# (`parent_.defines_` is one map), so `#ifdef SOME_MACRO` turns true at
# the line that defines it.
#
# Until 2026-09-23 the expander evaluated none of this: the comment
# stripper kept the directive lines, the WML parser skipped them, and
# the content of EVERY branch survived. That was right for the only
# conditional in a scenario we build -- Hornshark Island's, which tests
# its own `define=` and so is true when Wesnoth loads it -- and wrong
# for the core macros' `#ifdef EASY` / `NORMAL` / `HARD` / `NIGHTMARE`,
# `#ifdef __UNUSED` and `#ifndef MULTIPLAYER`, none of which is defined
# in a multiplayer game. No scenario we build invokes those macros,
# which the expansion diff shows for the pool.

# What a multiplayer game defines, as `--preprocess-defines=MULTIPLAYER`
# does for the template builder.
_BASE_DEFINES = frozenset({"MULTIPLAYER"})

_COND_RE = re.compile(r'^\s*#(ifdef|ifndef|else|endif|ifhave|ifnhave|ifver|ifnver)\b'
                      r'[ \t]*(\S*)')
_DEFINE_NAME_RE = re.compile(r'^\s*#define[ \t]+([\w:]+)')
# A scenario's own `define=`, which Wesnoth adds to the define set when it
# loads that scenario (the MP list itself is preprocessed without it).
_SCENARIO_DEFINE_RE = re.compile(r'^[ \t]*define[ \t]*=[ \t]*"?([^"\n]+?)"?[ \t]*$',
                                 re.MULTILINE)


class PreprocessorError(RuntimeError):
    """An unbalanced or unsupported conditional."""


def scenario_defines(text: str) -> Set[str]:
    """The symbols a scenario defines for itself with `define=`. A
    comma-separated list is accepted; Wesnoth's own scenarios use one."""
    out: Set[str] = set()
    for m in _SCENARIO_DEFINE_RE.finditer(text):
        out.update(s.strip() for s in m.group(1).split(",") if s.strip())
    return out


def evaluate_conditionals(text: str, defines: Set[str]) -> str:
    """`text` with every conditional resolved: directive lines removed,
    dead branches dropped.

    `defines` is UPDATED in place with each `#define` met in a live
    branch, because the engine keeps one map for the whole load: pass
    the same set through several files and a later file's `#ifdef`
    sees an earlier file's `#define`.

    `#ifhave` / `#ifver` and their negations test files and the game
    version; nothing we expand uses them (measured 2026-09-23), so they
    fail loudly rather than being guessed at."""
    defined = defines
    # One frame per open conditional: (this branch live?, parent live?,
    # already saw #else?)
    stack: List[Tuple[bool, bool, bool]] = []
    live = True
    out: List[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        m = _COND_RE.match(line)
        if m is None:
            if live:
                d = _DEFINE_NAME_RE.match(line)
                if d:
                    defined.add(d.group(1))
                out.append(line)
            continue
        directive, symbol = m.group(1), m.group(2)
        if directive in ("ifhave", "ifnhave", "ifver", "ifnver"):
            raise PreprocessorError(
                f"line {lineno}: #{directive} is not supported "
                f"(nothing we expand used it when this was written)")
        if directive in ("ifdef", "ifndef"):
            if not symbol:
                raise PreprocessorError(f"line {lineno}: #{directive} with no symbol")
            found = symbol in defined
            branch = found if directive == "ifdef" else not found
            stack.append((branch, live, False))
            live = live and branch
        elif directive == "else":
            if not stack or stack[-1][2]:
                raise PreprocessorError(f"line {lineno}: unexpected #else")
            branch, parent, _ = stack.pop()
            stack.append((not branch, parent, True))
            live = parent and not branch
        else:                                   # endif
            if not stack:
                raise PreprocessorError(f"line {lineno}: unexpected #endif")
            _, parent, _ = stack.pop()
            live = parent
    if stack:
        raise PreprocessorError(f"{len(stack)} #ifdef/#ifndef never closed")
    return "\n".join(out)


def _preprocess_text(text: str, defines: Set[str]) -> str:
    """The one path every expanded file takes before its macros are
    read: textdomain and comments stripped, conditionals resolved.
    Three call sites (core macros, an add-on's utility files, the
    scenario) used to repeat the two strips and skip the third step."""
    return evaluate_conditionals(_strip_comments(_strip_textdomain(text)),
                                 defines)


def _extract_inline_macros(text: str) -> Tuple[str, Dict[str, MacroDef]]:
    """Pull #define...#enddef blocks out of `text`. Returns
    (text_without_defines, {macro_name: MacroDef})."""
    macros: Dict[str, MacroDef] = {}

    def _strip(m):
        name = m.group(1)
        params = (m.group(2) or "").split()
        body, defaults = _split_optional_args(m.group(3))
        macros[name] = MacroDef(params, body, defaults)
        return ""
    cleaned = _MACRO_DEFINE_RE.sub(_strip, text)
    return cleaned, macros


def _split_macro_args(arg_str: str) -> List[str]:
    """Split macro invocation arguments respecting quoted strings AND
    Wesnoth-preprocessor parenthesized arguments: `(a b, c)` is ONE
    argument whose value is the inner text (parens stripped) -- the
    Mini_Maps_Collection spawns pass `(Tentacle of the Deep)` and
    `(2,2)` this way; splitting them on whitespace mangled x,y into
    `(2` / `2)` and the [unit] action silently dropped every enclave
    tentacle (found 2026-07-14)."""
    args: List[str] = []
    cur = ""
    in_quote = False
    depth = 0
    for ch in arg_str:
        if ch == '"' and depth == 0:
            in_quote = not in_quote
            cur += ch
        elif ch == "(" and not in_quote:
            if depth > 0:
                cur += ch
            depth += 1
        elif ch == ")" and not in_quote and depth > 0:
            depth -= 1
            if depth > 0:
                cur += ch
            else:
                args.append(cur)
                cur = ""
        elif ch.isspace() and not in_quote and depth == 0:
            if cur:
                args.append(cur)
                cur = ""
        else:
            cur += ch
    if cur:
        args.append(cur)
    return args


_UNKNOWN_MACRO_COUNTS: Dict[str, int] = {}


def unknown_macro_counts() -> Dict[str, int]:
    """Macro name to the number of times it expanded to nothing because
    we had no definition and no classification for it. A census, and
    what `tests/test_action_classification.py` reads."""
    return dict(_UNKNOWN_MACRO_COUNTS)


def reset_unknown_macros() -> None:
    _WARNED_MACROS.clear()
    _UNKNOWN_MACRO_COUNTS.clear()


def _report_unknown_macro(name: str) -> None:
    """An undefined macro deleted from the expansion. This is the
    third silent site, and it is the one that hid `DEFAULT_SCHEDULE`'s
    sibling class for months at DEBUG level: a macro that expands to
    nothing looks exactly like a macro whose body was cosmetic."""
    _UNKNOWN_MACRO_COUNTS[name] = _UNKNOWN_MACRO_COUNTS.get(name, 0) + 1
    if os.environ.get("WESNOTH_STRICT_WML"):
        raise UnmodelledWML(
            f"{{{name}}} has no definition and no recorded reason to drop it")
    if name not in _WARNED_MACROS:
        _WARNED_MACROS.add(name)
        log.warning(
            "unknown macro {%s} expanded to nothing. If its body is "
            "presentation, name it in _COSMETIC_MACROS; if the sim "
            "implements it, name it in _SUBSTITUTED_MACROS.", name)


def _substitute_macros(text: str, macros: Dict[str, MacroDef],
                       depth: int = 0) -> str:
    """Recursively substitute {MACRO arg arg ...} occurrences in `text`.
    Classified and unknown macros expand to the empty string; the
    unknown ones are reported (see `_report_unknown_macro`)."""
    if depth > 8:
        return text  # avoid infinite recursion

    def _do(m):
        name = m.group(1)
        argstr = m.group(2).strip()
        if name in _COSMETIC_MACROS or name in _SUBSTITUTED_MACROS:
            return ""
        if name in macros:
            params, body, defaults = macros[name]
            args = _split_macro_args(argstr)
            # An optional named argument is passed as `NAME=value` and
            # binds by name, so it does not consume a positional slot.
            named = dict(defaults)
            positional = []
            for arg in args:
                key, sep, value = arg.partition("=")
                if sep and key.strip() in defaults:
                    named[key.strip()] = value.strip().strip('"')
                else:
                    positional.append(arg)
            # Pad/truncate to param count.
            positional = (positional + [""] * len(params))[:len(params)]
            sub = body
            for p, a in zip(params, positional):
                sub = re.sub(r'\{' + re.escape(p) + r'\}', a, sub)
            for key, value in named.items():
                sub = re.sub(r'\{' + re.escape(key) + r'\}', value, sub)
            # Substitute nested macros in the expansion.
            return _substitute_macros(sub, macros, depth + 1)
        # Inline-include macros like {~add-ons/...} or {core/macros/...}
        if name.startswith("~") or "/" in argstr:
            return ""
        _report_unknown_macro(name)
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

def _load_core_macros() -> Dict[str, MacroDef]:
    """Slurp Wesnoth's data/core/macros/*.cfg for macro definitions
    that scenarios commonly invoke. We don't expand the bodies (most
    are cosmetic anyway); just need names so we don't warn on them."""
    macros: Dict[str, MacroDef] = {}
    macros_dir = WESNOTH_SRC / "data" / "core" / "macros"
    if not macros_dir.exists():
        # A bare git clone carries only the tracked runtime subset.
        # schedules.cfg IS tracked (2026-08-04) precisely because
        # [time_area] ToD macros must expand -- without them the
        # parser dropped {FIRST_WATCH} etc. and Kesorak's darkened
        # hex silently ran NEUTRAL instead of night (engine-verified
        # OOS: recorded 10 dmg vs engine 7). Fail loudly, not silently.
        log.warning(
            "core macros dir missing (%s): scenario ToD macros "
            "({DAWN}/{FIRST_WATCH}/...) will not expand -- "
            "[time_area] cycles WILL be wrong on maps that use them "
            "(Tombs of Kesorak, Elensefar Courtyard)", macros_dir)
        return macros
    # One define set across the core files, as the engine keeps one
    # map: a later file's `#ifdef` sees an earlier file's `#define`.
    core_defines: Set[str] = set(_BASE_DEFINES)
    for cfg in sorted(macros_dir.glob("*.cfg")):
        try:
            txt = cfg.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        txt = _preprocess_text(txt, core_defines)
        # The same extraction the scenario's own body gets: one reader,
        # so optional-argument defaults and colon-bearing names behave
        # identically whichever file a macro came from.
        _, defined = _extract_inline_macros(txt)
        macros.update(defined)
    # MODIFY_UNIT's mainline body (data/core/macros/utils.cfg:271-301)
    # is a [store_unit] kill=yes -> [foreach] set this_item.VAR ->
    # [unstore_unit] round-trip -- tags we don't run. For scalar VARs
    # that round-trip is exactly [modify_unit] semantics, which we DO
    # interpret, so expand the macro to the reduced form. Load-bearing
    # for Mini_Maps_Collection's repeating `turn refresh` event
    # {MODIFY_UNIT (role=monster) moves 0} (enclave_micro_isar.cfg:
    # 86-91): it pins every tentacle at 0 MP, which via unit::
    # end_turn's movement_!=total_movement check permanently cancels
    # its rest heal (Micro Isar 38859, 2026-08-07).
    macros["MODIFY_UNIT"] = MacroDef(
        ["FILTER", "VAR", "VALUE"],
        "[modify_unit]\n"
        "    [filter]\n"
        "        {FILTER}\n"
        "    [/filter]\n"
        "    {VAR}={VALUE}\n"
        "[/modify_unit]\n",
        {},
    )
    return macros


_CORE_MACROS_CACHE: Optional[Dict[str, MacroDef]] = None

_MACRO_DEFINITION_RE = re.compile(r"^[ \t]*#define\b.*?^[ \t]*#enddef\b",
                                  re.MULTILINE | re.DOTALL)
_TAG_LINE_RE = re.compile(r"^\s*\[(\+|/)?([a-zA-Z_][a-zA-Z0-9_]*)\]\s*$")
_ID_LINE_RE = re.compile(r'^\s*id\s*=\s*"?([^"\s]+)"?\s*$')
_SCENARIO_TAGS = ("multiplayer", "scenario", "test")


def scenario_id_of_cfg(text: str) -> Optional[str]:
    """The `id=` of the file's own [multiplayer], [scenario] or [test]
    tag: the first id that tag holds directly, macro definitions left
    out. The first `id=` of the text can belong to anything else: in
    WL_Mappack's 2p_Troll_Toll.cfg it is `id=remove_hp`, a [trait]
    inside the file's `#define UNIT_PETRIFY`."""
    depth = 0
    in_scenario = False
    for line in _MACRO_DEFINITION_RE.sub("", text).splitlines():
        tag = _TAG_LINE_RE.match(line)
        if tag is not None:
            closing = tag.group(1) == "/"
            if closing:
                depth -= 1
                in_scenario = in_scenario and depth > 0
            else:
                depth += 1
                if depth == 1:
                    in_scenario = tag.group(2) in _SCENARIO_TAGS
            continue
        if in_scenario and depth == 1:
            m = _ID_LINE_RE.match(line)
            if m:
                return m.group(1)
    return None


def find_scenario_cfg_path(scenario_id: str) -> Optional[Path]:
    """Locate the scenario .cfg whose WML id is `scenario_id`.

    Search order:
      1. wesnoth_src/data/multiplayer/scenarios/  (core 2p ladder)
      2. wesnoth_src/data/add-ons/<pkg>/scenarios/  (vendored add-ons,
         currently just Mini_Maps_Collection)
      3. <project>/add-ons/<pkg>/scenarios/ and scenarios/drills/
         (our own add-on: the capability drills)

    Shared by load_scenario_wml AND sim_to_replay's per-scenario
    gold/village/unit scrapes, so the two can't disagree on which
    file defines a scenario.
    """
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
    # The two lowercase-id maps MUST be here: their files are
    # CamelCase, and on a case-SENSITIVE filesystem (Linux training
    # nodes) the naive `2p_<id>.cfg` probe misses them -- observed
    # 2026-07-02 on Vast: both maps silently skipped from the ladder
    # pool. Windows' case-insensitive filesystem masked it locally.
    _BASE_OVERRIDES = {
        "Basilisk": "Caves_of_the_Basilisk",
        "elensefar_courtyard": "Elensefar_Courtyard",
        "thousand_stings_garrison": "Thousand_Stings_Garrison",
    }
    base = _BASE_OVERRIDES.get(base, base)

    candidate = SCENARIO_DIR / f"2p_{base}.cfg"
    if candidate.exists():
        return candidate
    # Try without the 2p_ prefix (some scenarios use 4p_, 8p_, etc.)
    for nplayers in (3, 4, 5, 6, 8):
        alt = SCENARIO_DIR / f"{nplayers}p_{base}.cfg"
        if alt.exists():
            return alt

    # Add-on fallback: scan vendored add-ons (wesnoth_src/data/
    # add-ons/) AND the project's own add-on tree (add-ons/ at the
    # repo root -- the capability drills live there, junctioned into
    # userdata for the real game). Mini-map / drill scenarios often
    # have raw ids like "2p_mini" or "drill_duel" that don't match
    # either prefix convention; look up by both filename and by the
    # [multiplayer]/[scenario] id attribute. Directories and files are
    # walked in sorted order, so every filesystem picks the same file
    # when two declare one id (WL_Troll_Toll: WL_Mappack and
    # Seamless_Map_Picker, whose scenario bodies are identical).
    addon_roots = [
        SCENARIO_DIR.parent.parent / "add-ons",        # wesnoth_src/data
        ADDONS_DIR,                                     # project
    ]
    for addons_dir in addon_roots:
        if not addons_dir.is_dir():
            continue
        for addon in sorted(addons_dir.iterdir()):
            sc_root = addon / "scenarios"
            if not sc_root.is_dir():
                continue
            for sc_dir in (sc_root, sc_root / "drills"):
                if not sc_dir.is_dir():
                    continue
                # First try filename matches (faster).
                for fname in (f"{scenario_id}.cfg",
                              f"{base}.cfg",
                              f"2p_{base}.cfg"):
                    p = sc_dir / fname
                    if p.is_file():
                        return p
                # Fall through: scan every .cfg's scenario id.
                # Add-on scenarios often pick non-filename-matching
                # ids (e.g. file Modified_Close_Relation.cfg has
                # id=Modified_Tiny_Close_Relation).
                for p in sorted(sc_dir.glob("*.cfg")):
                    try:
                        text = p.read_text(encoding="utf-8", errors="ignore")
                    except OSError:
                        continue
                    if scenario_id_of_cfg(text) == scenario_id:
                        return p
    return None


def load_scenario_wml(scenario_id: str) -> Optional[WMLNode]:
    """Find and parse the scenario .cfg matching `scenario_id`. Returns
    the parsed root node (with a [multiplayer] or [scenario] child),
    or None if unmappable. `scenario_id` should match the WML id like
    "multiplayer_Aethermaw", "multiplayer_Den_of_Onis", or for
    add-on scenarios the raw id like "2p_mini" or "drill_duel".
    Search order: see `find_scenario_cfg_path`.
    """
    candidate = find_scenario_cfg_path(scenario_id)
    if candidate is None:
        return None
    return parse_scenario_cfg(candidate)


def parse_scenario_cfg(candidate: Path) -> Optional[WMLNode]:
    """Parse a scenario .cfg at a known path: macros expanded (core,
    the file's own, and an add-on's siblings) and parallel assigns
    normalized, as `load_scenario_wml` does for an id.

    Split out 2026-09-22 so a caller holding the path -- the replay
    exporter -- reads the scenario the same way the pool does instead
    of scraping it with regex. That split is not hypothetical: the
    exporter's village scraper handled only the `x=`/`y=` form and
    silently dropped every combined `x,y=` village until 2026-07-19,
    which made Clearing Gushes playback capture a village the sim
    already owned.
    """
    global _CORE_MACROS_CACHE
    if _CORE_MACROS_CACHE is None:
        _CORE_MACROS_CACHE = _load_core_macros()

    raw = candidate.read_text(encoding="utf-8", errors="replace")
    # What Wesnoth has defined by the time it preprocesses this file:
    # the multiplayer symbol, the scenario's own `define=` (added when
    # the scenario is loaded), and every core macro.
    defines = set(_BASE_DEFINES) | scenario_defines(raw) | set(_CORE_MACROS_CACHE)
    raw, scenario_macros = _extract_inline_macros(_preprocess_text(raw, defines))
    # Merge core macros with this scenario's local macros (local wins).
    all_macros = dict(_CORE_MACROS_CACHE)
    # Add-on scenarios may define macros in SIBLING utility files
    # (Mini_Maps_Collection keeps {MI_UNIT_PLACING} in
    # utils/units-utils.cfg): without them the enclave maps' turn-1
    # tentacle spawns expanded to nothing and side 3 was silently
    # empty (found 2026-07-14). Load defines from every other .cfg
    # under the add-on root (nearest ancestor containing _main.cfg).
    addon_root = None
    for parent in candidate.parents:
        # Only genuine add-on trees: mainline multiplayer/ also has a
        # _main.cfg, and merging all 86 mainline cfgs' defines into
        # every load is broad, slow, and fragile (review 2026-07-14).
        if (parent / "_main.cfg").is_file() and "add-ons" in parent.parts:
            addon_root = parent
            break
    if addon_root is not None:
        for util_cfg in sorted(addon_root.rglob("*.cfg")):
            if util_cfg == candidate:
                continue
            try:
                util_raw = _preprocess_text(
                    util_cfg.read_text(encoding="utf-8", errors="replace"),
                    defines)
            except OSError:
                continue
            _, util_macros = _extract_inline_macros(util_raw)
            for k, v in util_macros.items():
                all_macros.setdefault(k, v)
    all_macros.update(scenario_macros)
    expanded = _substitute_macros(raw, all_macros)
    # Parallel assigns (x,y=2,2) AFTER substitution: macro bodies
    # carry `x,y={POSITION}` which only becomes expandable once the
    # args are in (the pre-substitution ordering silently mangled
    # every enclave tentacle spawn, 2026-07-14).
    expanded = _expand_parallel_assigns(expanded)
    return parse_wml(expanded)


# ----------------------------------------------------------------------
# Event extraction
# ----------------------------------------------------------------------

def standard_event_name(name: str) -> str:
    """The engine's `event_handlers::standardize_name`
    (src/game_events/manager_impl.cpp:65-76, 1.18.4): trimmed, every
    internal space an underscore, case kept. `side 1 turn` and
    `side_1_turn` are one name; `Prestart` is not `prestart`."""
    return name.strip().replace(" ", "_")


def event_names(raw: str) -> List[str]:
    """The names an [event] answers to: its `name=` is a comma-separated
    list, split with empty pieces dropped, each piece standardized
    (`event_handler::names`, src/game_events/handlers.cpp:64-88)."""
    return [standard_event_name(piece) for piece in raw.split(",") if piece.strip()]


@dataclass
class ScenarioEvent:
    """One [event] block extracted from a scenario .cfg."""
    name: str                         # "prestart", "side 1 turn 4", etc.
    first_time_only: bool = True
    actions: List[WMLNode] = field(default_factory=list)
    fired: bool = False               # latched by the interpreter
    scenario_id: str = ""             # so an unmodelled tag names its map

    @property
    def names(self) -> List[str]:
        return event_names(self.name)

    def can_fire(self) -> bool:
        return not (self.first_time_only and self.fired)


# The events the engine fires around a side's turn, in its order
# (src/play_controller.cpp, 1.18.4). `do_init_side` fires "turn N" and
# "new turn" only at the first side turn of a turn (:473-477, latched by
# tod_manager's has_turn_event_fired, which next_turn resets), then the
# four side forms (:479-482); after the refresh, healing and income, the
# four refresh forms (:519-522). `finish_side_turn_events` fires the four
# end forms after the side's units end their turn (:585-588), and
# `finish_turn` the two turn-end forms once the last side's turn is
# over (:601-602). docs/wesnoth_rules.md "Turn events".

def side_turn_event_names(side: int, turn: int, *, new_turn: bool) -> List[str]:
    names = [f"turn {turn}", "new turn"] if new_turn else []
    return names + ["side turn", f"side {side} turn", f"side turn {turn}",
                    f"side {side} turn {turn}"]


def turn_refresh_event_names(side: int, turn: int) -> List[str]:
    return ["turn refresh", f"side {side} turn refresh", f"turn {turn} refresh",
            f"side {side} turn {turn} refresh"]


def side_turn_end_event_names(side: int, turn: int) -> List[str]:
    return ["side turn end", f"side {side} turn end", f"side turn {turn} end",
            f"side {side} turn {turn} end"]


def turn_end_event_names(turn: int) -> List[str]:
    return ["turn end", f"turn {turn} end"]


def init_side_event_names(side: int, turn_before: int) -> List[str]:
    """Every name the applier's init_side fires, given the turn counter
    before it. The applier opens a turn at side 1's init_side (it counts
    turns there; side 1 always opens a turn), so that init_side also
    ends the turn before it."""
    if side != 1:
        return (side_turn_event_names(side, turn_before, new_turn=False)
                + turn_refresh_event_names(side, turn_before))
    turn = turn_before + 1
    ended = turn_end_event_names(turn_before) if turn_before >= 1 else []
    return (ended + side_turn_event_names(side, turn, new_turn=True)
            + turn_refresh_event_names(side, turn))


def any_can_fire(events: List["ScenarioEvent"], names: List[str]) -> bool:
    """Whether firing `names` would run any of `events`."""
    wanted = {standard_event_name(n) for n in names}
    return any(ev.can_fire() and not wanted.isdisjoint(ev.names) for ev in events)


def collect_events(root: WMLNode, scenario_id: str = "") -> List[ScenarioEvent]:
    """Find every [event] block under [multiplayer] / [scenario] and
    return them in WML-order so the caller can fire them sequentially."""
    out: List[ScenarioEvent] = []
    container = root.first("multiplayer") or root.first("scenario")
    if container is None:
        return out
    for ev in container.all("event"):
        name = ev.attrs.get("name", "").strip().strip('"')
        first_time = ev.attrs.get("first_time_only", "yes").strip().lower() in (
            "yes", "true", "1",
        )
        # The "actions" of an event are its inner WML children except
        # nested [filter] (those are predicates, not actions).
        actions = [ch for ch in ev.children if ch.tag != "filter"]
        out.append(ScenarioEvent(
            name=name, first_time_only=first_time, actions=actions,
            scenario_id=scenario_id,
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
    from tools.terrain_resolver import split_start_position, terrain_mask
    new_terr, new_mods = _parse_hex_code(new_code)
    new_mask = terrain_mask(new_code)

    # NOTE: `_terrain_codes` stores the FULL code including any
    # overlay, matching the map-load path (parse_terrain_codes). The
    # movement/defense resolvers (terrain_resolver.mvt_cost/def_pct
    # via _move_cost_at_hex / _terrain_def_pct) walk the alias graph
    # from that code, and an overlay can DOMINATE it: ^Xo is the
    # Impassable Overlay with mvt_alias=Xt (wesnoth_src/data/core/
    # terrain.cfg:1743-1751), so 'Chw^Xo' is impassable despite the
    # castle base. A previous version stored the overlay-stripped
    # base here ('Chw^Xo' -> 'Chw'), which made Aethermaw's turn-6
    # whirlpool walls walkable in the sim -- units moved onto them
    # and the exported replays failed strict-sync in real Wesnoth
    # ("found corrupt movement in replay", engine-verified
    # 2026-07-29). See test_terrain_overlay_resolution.py::
    # test_terrain_event_preserves_overlay_in_codes.

    # COPY-ON-WRITE (adversarial-review HIGH finding, 2026-07-18):
    # `Map.__deepcopy__` / `GlobalInfo.__deepcopy__` ALIAS
    # `map.hexes` and `_terrain_codes` across `WesnothSim.fork()`
    # (terrain was assumed immutable). Mutating them in place meant
    # an MCTS rollout fork that crossed a morph turn (Aethermaw,
    # turns 4-6) morphed the LIVE game's terrain too — reproduced:
    # 22 live hexes changed from a fork stepped to turn 13, with the
    # live `_terrain_epoch` left stale so the planner/mask served
    # pre-morph costs against post-morph combat. Build NEW
    # containers and rebind them on THIS gs only. Replacing the
    # hexes set also changes `id(gs.map.hexes)`, auto-invalidating
    # the `_hex_lookup` cache keyed on it (previously never
    # invalidated here — same latent bug, second symptom).
    new_hexes = set(gs.map.hexes)
    codes = getattr(gs.global_info, "_terrain_codes", None)
    new_codes = dict(codes) if codes is not None else None

    pairs = list(zip(xs, ys))
    for wml_x, wml_y in pairs:
        # Parsed Hex set: 0-indexed playable coords → subtract 1.
        py_x, py_y = wml_x - 1, wml_y - 1
        # Update or insert the parsed Hex so `_terrain_at` reflects
        # the new terrain. Hex is hashable on position, so we discard
        # the old and add a fresh one with the new terrain set.
        old_hex = next(
            (h for h in new_hexes
             if h.position.x == py_x and h.position.y == py_y),
            None,
        )
        if old_hex is not None:
            new_hexes.discard(old_hex)
        new_hexes.add(Hex(
            position=Position(x=py_x, y=py_y),
            terrain_types=set(new_terr),
            modifiers=set(new_mods),
            terrain_mask=new_mask,
        ))

        # Mirror the change into the per-game terrain-code dict that
        # `_terrain_keys_at` / `_move_cost_at_hex` resolve through.
        # Without this the post-event hex still resolves through the
        # OLD code and combat defense math is wrong (Drake on Ford
        # should get its grass defense, not shallow-water defense).
        # FULL code, overlay included -- see the ^Xo note above.
        if new_codes is not None:
            new_codes[(py_x, py_y)] = new_code

        # Raw map_data string: border-included → WML (X, Y) is at
        # raw_cells[Y][X] directly (file row Y col X with the border
        # row at index 0).
        # A [terrain] event replaces the terrain, never the hex's
        # starting-position label, so the label is carried over. The
        # engine writes a cell back as label + " " + code
        # (wesnoth_src/src/terrain/translation.cpp:775-782,
        # number_to_string_). Splicing a fixed two characters instead
        # only recognized a one-digit label, so a rewrite of "10 Kh"
        # or "lake Gs^Vc" silently DROPPED the start position from the
        # exported map_data.
        if 0 <= wml_y < len(raw_cells) and 0 <= wml_x < len(raw_cells[wml_y]):
            label, _old_code = split_start_position(raw_cells[wml_y][wml_x])
            raw_cells[wml_y][wml_x] = f"{label} {new_code}" if label else new_code

    gs.map.hexes = new_hexes
    if new_codes is not None:
        setattr(gs.global_info, "_terrain_codes", new_codes)
        # Invalidate the reach-planner's per-map terrain cache
        # (pathfind_sim keys on this epoch). Once per event, on THIS
        # gs only.
        from tools.pathfind_sim import next_terrain_epoch
        setattr(gs.global_info, "_terrain_epoch", next_terrain_epoch())

    if raw_cells:
        new_raw = "\n".join(", ".join(row) for row in raw_cells)
        setattr(gs.global_info, "_raw_map_data", new_raw)


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


def _area_cycle_from_turn_one(cycle: List[int], action: WMLNode,
                              turn_number: int) -> List[int]:
    """The area's cycle rotated so that turn t reads index (t - 1) % len.

    An area keeps its own slot: `tod_manager::add_time_area` starts it at
    the area's `current_time` (default 0) on the turn it is placed, and
    `resolve_random` moves only the board's slot (src/tod_manager.cpp,
    1.18.4; docs/wesnoth_rules.md "Time areas keep their own slot"). At
    turn t the area reads slot (current_time + t - placed) mod len."""
    placed = max(1, int(turn_number or 0))
    shift = (wml_int(action.attrs.get("current_time"), 0) - (placed - 1)) % len(cycle)
    return cycle[shift:] + cycle[:shift]


def _time_area_action(gs: GameState, action: WMLNode) -> None:
    """[time_area] x=… y=… (or find_in=VAR) [time]…[/time] [/time_area]
    Stamps a per-hex lawful_bonus cycle onto `gs.global_info._time_areas`,
    phased so that turn t reads index (t - 1) % len.
    Multiple time_areas can stack; later writes win on overlapping hexes,
    matching Wesnoth's "last [time_area] applied wins" rule."""
    cycle = _parse_time_cycle(action)
    if not cycle:
        return
    cycle = _area_cycle_from_turn_one(cycle, action, gs.global_info.turn_number)

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
        _wml_vars(gs)[name] = _subst_wml_vars(
            gs, action.attrs["value"]).strip().strip('"')
    elif "literal" in action.attrs:
        _wml_vars(gs)[name] = action.attrs["literal"].strip().strip('"')
    if "add" in action.attrs:
        try:
            cur = int(float(_wml_vars(gs).get(name, "0")))
        except ValueError:
            cur = 0
        try:
            inc = int(float(_subst_wml_vars(
                gs, action.attrs["add"]).strip().strip('"')))
        except ValueError:
            inc = 0
        _wml_vars(gs)[name] = str(cur + inc)
    if "sub" in action.attrs:
        # value= and sub= COMPOSE (Marshy Fill: value=9
        # sub=$leader1.moves -> 9 - moves); Wesnoth applies the ops
        # in sequence on the same variable.
        try:
            cur = int(float(_wml_vars(gs).get(name, "0")))
        except ValueError:
            cur = 0
        try:
            dec = int(float(_subst_wml_vars(
                gs, action.attrs["sub"]).strip().strip('"')))
        except ValueError:
            dec = 0
        _wml_vars(gs)[name] = str(cur - dec)


_VAR_REF_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_.\[\]]*)")


def _subst_wml_vars(gs: GameState, raw: str) -> str:
    """Substitute `$name` / `$name.path` references from the WML
    variable namespace (see `_wml_vars`). Unknown refs resolve to ""
    -- Wesnoth's own behavior for unset variables."""
    if "$" not in (raw or ""):
        return raw or ""
    v = _wml_vars(gs)
    return _VAR_REF_RE.sub(lambda m: v.get(m.group(1), ""), raw)


def _num_or_none(s: str):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


# What `_capture_village_action` reads: the side, and the location
# filter's x=, y= and terrain= (`_eval_location_clause`).
_CAPTURE_VILLAGE_READ = frozenset({"side", "x", "y", "terrain"})


def _capture_village_action(gs: GameState, action: WMLNode) -> None:
    """`[capture_village] side=N` plus a location filter: every matched
    VILLAGE changes hands as `wesnoth.map.set_owner` hands it
    (wesnoth_src/data/lua/wml-tags.lua:444-461, then
    src/scripting/game_lua_kernel.cpp:1142-1193, 1.18.4): a location
    that is not a village is skipped, the old owner loses the village
    and `side` gains it; no side (or 0) leaves it to nobody. The
    transfer is `replay_dataset.set_village_owner`, the one a move's
    capture uses, so the sides' village counts, which income reads,
    follow the owners. WL_Mappack's Cold War and Summer Frosts use this
    at prestart for asymmetric starting villages (audit 2026-08-06).

    Not modelled, and reported when a scenario uses it: the rest of the
    standard location filter (`[and]`, `[filter_side]`, radius, ...)
    and `fire_event=`, which would fire capture events. Also not
    modelled: set_owner does nothing for a side the engine counts as
    defeated (by default, one without a leader); every capture we have
    met runs at prestart on the player sides, whose leaders stand."""
    from tools.replay_dataset import _terrain_at, set_village_owner
    unread = sorted(set(action.attrs) - _CAPTURE_VILLAGE_READ)
    unread += [f"[{child.tag}]" for child in action.children]
    if unread:
        _report_unmodelled_value(
            f"[capture_village] {', '.join(unread)}: the sim reads side, x, y and terrain")
    side_raw = _subst_wml_vars(gs, action.attrs.get("side", "")).strip()
    try:
        side = int(side_raw) if side_raw else 0
    except ValueError:
        return          # the engine raises "invalid side in [capture_village]"
    hexes = _eval_location_clause(gs, action, gs.map.size_x, gs.map.size_y)
    for wx, wy in sorted(hexes):
        x, y = wx - 1, wy - 1
        if _terrain_at(gs, x, y) == "village":
            set_village_owner(gs, x, y, side)


def _units_matching_filter(gs: GameState, flt: Optional[WMLNode]):
    """Units matching a minimal [filter]: x/y hex lists, id=, side=.
    Enough for the add-on start events we support; extend as needed."""
    if flt is None:
        return list(gs.map.units)
    hexes = None
    if flt.attrs.get("x") or flt.attrs.get("y"):
        hexes = {(wx - 1, wy - 1) for wx, wy in _resolve_xy_attr(
            flt.attrs.get("x", ""), flt.attrs.get("y", ""),
            gs.map.size_x, gs.map.size_y)}
    want_id = flt.attrs.get("id", "").strip().strip('"') or None
    want_side = flt.attrs.get("side", "").strip() or None
    # role= matches the WML role assigned at spawn ([unit] role=...,
    # stashed as `_wml_role` by _unit_action). Mini Maps' tentacles
    # carry role=monster (units-utils.cfg:9).
    want_role = flt.attrs.get("role", "").strip().strip('"') or None
    out = []
    for u in gs.map.units:
        if hexes is not None and (u.position.x, u.position.y) not in hexes:
            continue
        if want_id and u.id != want_id:
            continue
        if want_side and str(u.side) != want_side:
            continue
        if want_role and getattr(u, "_wml_role", None) != want_role:
            continue
        out.append(u)
    return out


def _swap_unit(gs: GameState, old, new) -> None:
    """Put `new` in `old`'s place in the fork-local unit set.

    `gs.map.units` is a per-fork SET whose Unit ELEMENTS are shared
    with the parent and every sibling fork (`Map.__deepcopy__`), so an
    event handler that changes a unit must build a replacement and
    swap it in -- assigning to the shared object's attributes rewrites
    the live game from inside a search (tests/test_fork_isolation.py).
    Unit eq/hash is (id, side), so discard+add replaces the right
    element however many fields changed (wesnoth_ai/classes.py:120).
    """
    gs.map.units.discard(old)
    gs.map.units.add(new)


def _store_unit_action(gs: GameState, action: WMLNode) -> None:
    """`[store_unit] [filter]..[/filter] variable=V` -- snapshot the
    matched units' scalar attributes into WML variables (`V.moves`,
    `V.max_moves`, ...). kill= defaults to no in Wesnoth; we never
    remove the unit. Multiple matches store the FIRST (the add-on
    events we support filter a single hex)."""
    var = action.attrs.get("variable", "").strip().strip('"')
    if not var:
        return
    units = _units_matching_filter(gs, action.first("filter"))
    if not units:
        return
    u = units[0]
    v = _wml_vars(gs)
    v[var + ".moves"] = str(u.current_moves)
    v[var + ".max_moves"] = str(u.max_moves)
    v[var + ".hitpoints"] = str(u.current_hp)
    v[var + ".max_hitpoints"] = str(u.max_hp)
    v[var + ".experience"] = str(u.current_exp)
    v[var + ".side"] = str(u.side)
    v[var + ".id"] = u.id
    v[var + ".x"] = str(u.position.x + 1)
    v[var + ".y"] = str(u.position.y + 1)
    v[var + ".length"] = str(len(units))


def _eval_variable_cond(gs: GameState, node: WMLNode) -> bool:
    """One `[variable]` condition. Numeric comparisons when both sides
    parse as numbers; string equality for the equals forms otherwise
    (Wesnoth semantics)."""
    name = node.attrs.get("name", "").strip().strip('"')
    cur = _wml_vars(gs).get(name, "")
    for op in ("equals", "not_equals", "numerical_equals",
               "greater_than", "less_than",
               "greater_than_equal_to", "less_than_equal_to"):
        if op not in node.attrs:
            continue
        rhs = _subst_wml_vars(gs, node.attrs[op]).strip().strip('"')
        a, b = _num_or_none(cur), _num_or_none(rhs)
        if op == "equals":
            return (a == b) if (a is not None and b is not None)                 else (cur == rhs)
        if op == "not_equals":
            return (a != b) if (a is not None and b is not None)                 else (cur != rhs)
        if a is None or b is None:
            return False
        if op == "numerical_equals":
            return a == b
        if op == "greater_than":
            return a > b
        if op == "less_than":
            return a < b
        if op == "greater_than_equal_to":
            return a >= b
        return a <= b
    return False


def _eval_condition(gs: GameState, node: WMLNode) -> bool:
    """Conjunction of direct [variable] children with nested
    [and]/[or]/[not] (recursing). Empty condition is true, matching
    Wesnoth's conditional evaluation."""
    ok = True
    for child in node.children:
        if child.tag == "variable":
            ok = ok and _eval_variable_cond(gs, child)
        elif child.tag == "and":
            ok = ok and _eval_condition(gs, child)
        elif child.tag == "or":
            ok = ok or _eval_condition(gs, child)
        elif child.tag == "not":
            ok = ok and not _eval_condition(gs, child)
    return ok


def _if_action(gs: GameState, action: WMLNode) -> None:
    """`[if] <conditions> [then]...[/then] [else]...[/else]` with the
    condition forms `_eval_condition` supports. Executes the selected
    branch's actions through the normal dispatch (recursion via
    _apply_action, same pattern as [switch])."""
    branch = "then" if _eval_condition(gs, action) else "else"
    for blk in action.all(branch):
        for sub in blk.children:
            _apply_action(gs, sub)


_MODIFY_UNIT_SCALARS = {
    # WML attr -> Unit field. CURRENT-value fields only: [modify_unit]
    # moves= writes the stored unit WML `moves` attribute = remaining
    # MP this turn (wesnoth_src/data/lua/wml/modify_unit.lua:14-17,41
    # -- attributes pass straight into the stored unit; max_moves is a
    # distinct attribute we intentionally do NOT map yet).
    "moves": "current_moves",
    "hitpoints": "current_hp",
    "experience": "current_exp",
}


def _modify_unit_action(gs: GameState, action: WMLNode) -> None:
    """`[modify_unit] [filter]..[/filter] attr=value` for the scalar
    attrs in _MODIFY_UNIT_SCALARS. Marshy Fill's start event uses
    moves=$leader1_moves to shave side 1's leader MP on turn 1 (the
    anti-first-move-advantage tweak).

    Uses the replace-unit pattern (`_swap_unit`): the matched units
    are the fork-SHARED Unit objects, so the new values go on a
    shallow copy that is swapped into the fork-local set."""
    flt = action.first("filter")
    if flt is None:
        return
    changes: Dict[str, int] = {}
    for attr, ufield in _MODIFY_UNIT_SCALARS.items():
        if attr not in action.attrs:
            continue
        raw = _subst_wml_vars(gs, action.attrs[attr]).strip().strip('"')
        try:
            val = int(float(raw))
        except ValueError:
            continue
        changes[ufield] = max(0, val)
    if not changes:
        return
    # `_units_matching_filter` materialises its list, so swapping
    # elements of `gs.map.units` below cannot disturb the iteration.
    for u in _units_matching_filter(gs, flt):
        new_u = _copy.copy(u)
        for ufield, val in changes.items():
            setattr(new_u, ufield, val)
        _swap_unit(gs, u, new_u)


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
    to call into `place_units`), through the same matching and
    `first_time_only` latch as the public `fire_event`."""
    name = action.attrs.get("name", "").strip().strip('"')
    if not name:
        return
    events = getattr(gs.global_info, "_scenario_events", None)
    if not events:
        return
    fire_event(gs, events, name)


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


def own_modification_effects(mods: Optional[WMLNode]) -> List[WMLNode]:
    """The [effect]s a placed [unit] carries in its own [modifications]:
    those of every [object], and of every CUSTOM [trait] (one outside the
    named traits, tools/traits.TRAITS, whose effects
    `apply_traits_to_unit` applies; applying those twice cost Hornshark's
    Sergeants and Drake Fighters a movement point). The statues of Caves
    of the Basilisk and Sullas Ruins carry a `remove_hp` trait, those of
    Thousand Stings Garrison the same effects in an [object]: 1 hp, no
    moves."""
    if mods is None:
        return []
    from tools.traits import TRAITS
    out: List[WMLNode] = []
    for node in mods.children:
        tid = (node.attrs.get("id", "") or "").strip().strip('"').lower()
        if node.tag == "object" or (node.tag == "trait" and tid not in TRAITS):
            out.extend(node.all("effect"))
    return out


def apply_side_unit_modifications(gs: GameState, root: WMLNode) -> None:
    """Give the units a scenario places in its [side] blocks their own
    [modifications] effects, as the engine builds them at scenario init,
    before prestart. Units are matched by side, position and type, so
    this serves a state built from the scenario (generation) and one
    rebuilt from a replay record (reconstruction) alike: neither's
    starting units carry the modifications. Called once per game from
    replay_dataset._setup_scenario_events; the units are freshly built,
    so the in-place contract of _apply_effect_to_unit holds."""
    block = root.first("multiplayer") or root.first("scenario")
    if block is None:
        return
    effects: Dict[Tuple[int, int, int, str], List[WMLNode]] = {}
    for side_node in block.all("side"):
        try:
            side = int(side_node.attrs.get("side", "0"))
        except ValueError:
            continue
        for unit_node in side_node.all("unit"):
            try:
                x = int(unit_node.attrs.get("x", "0")) - 1
                y = int(unit_node.attrs.get("y", "0")) - 1
            except ValueError:
                continue
            unit_type = unit_node.attrs.get("type", "").strip().strip('"')
            own = own_modification_effects(unit_node.first("modifications"))
            if own:
                effects[(side, x, y, unit_type)] = own
    if not effects:
        return
    for unit in gs.map.units:
        for eff in effects.get((unit.side, unit.position.x, unit.position.y, unit.name), ()):
            _apply_effect_to_unit(unit, eff)


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


def _heals_ability(node: WMLNode) -> Optional[str]:
    """The sim's heal ability for one `[heals]` block: `heals_4`,
    `heals_8`, or None when it heals nothing.

    The value is READ, never assumed. The engine builds the heal
    effect with a default of 0 (1.18.4 src/actions/heal.cpp:211,
    `effect(heal_list, 0)`) and a block without `value=` contributes
    nothing (src/units/abilities.cpp:2061), so a value-less [heals]
    heals 0 in Wesnoth. This reader used to default it to 4.

    That default was never exercised by a real scenario, but it was by
    our own expander: until 2026-09-22 every `#define INTERNAL:...`
    collapsed onto the single name INTERNAL, so Hornshark Island's
    {ABILITY_HEALS} expanded to an EMPTY [heals] block. The default of
    4 happened to equal the macro, so the preplaced Mermaid Initiates
    healed correctly by accident -- and a map using {ABILITY_HEALS_8}
    would have healed half what Wesnoth does, silently.
    """
    raw = node.attrs.get("value")
    if raw is None or not str(raw).strip().strip('"'):
        _report_unmodelled_value("[heals] with no value= (the engine heals 0)")
        return None
    try:
        value = int(str(raw).strip().strip('"'))
    except ValueError:
        _report_unmodelled_value(f"[heals] value={raw!r} is not an integer")
        return None
    if value == 4:
        return "heals_4"
    if value == 8:
        return "heals_8"
    # The sim models the two mainline heal amounts only.
    _report_unmodelled_value(f"[heals] value={value}: the sim models 4 and 8")
    return "heals_8" if value > 8 else ("heals_4" if value > 0 else None)


def _report_unmodelled_value(what: str) -> None:
    scenario_id = _FIRING_SCENARIO.get()
    if os.environ.get("WESNOTH_STRICT_WML"):
        raise UnmodelledWML(what + (f" (scenario {scenario_id})" if scenario_id else ""))
    log.warning("%s%s", what, f" in {scenario_id}" if scenario_id else "")


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
    from tools import replay_dataset as _rd
    from tools.replay_dataset import _build_unit, _stats_for
    if variation:
        # Force the DB load before the membership check: `_load_unit_db`
        # REBINDS the module global, so a from-imported `_UNIT_DB` taken
        # pre-load would stay the orphaned empty dict and this check
        # would silently drop the variation (dual-import audit,
        # 2026-07-30). Reading through the module attribute post-load is
        # correct regardless of who warmed the DB first.
        _rd._load_unit_db()
        composite = f"{utype}:{variation}"
        if composite in _rd._UNIT_DB:
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
    for eff in own_modification_effects(mods):
        _apply_effect_to_unit(base_unit, eff)
    # Apply petrified status from `[status] petrified=yes`.
    status_node = action.first("status")
    if status_node is not None:
        petr = (status_node.attrs.get("petrified", "") or "").strip().lower()
        if petr in ("yes", "true", "1"):
            new_st = set(base_unit.statuses)
            new_st.add("petrified")
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
        # An ability member is named by its `id=` (ambush, submerge,
        # concealment, nightstalk, burrow all live under [hides];
        # `_effect_member_ids` has the engine citations), the tag being
        # the fallback of a block without one. [heals] is the one
        # value-dependent name: heals_4 or heals_8.
        new_abilities = set(base_unit.abilities)
        for child in abil_node.children:
            tag = child.tag
            if tag == "heals":
                heal = _heals_ability(child)
                if heal:
                    new_abilities.add(heal)
            else:
                new_abilities.add((child.attrs.get("id") or "").strip().strip('"') or tag)
        if new_abilities != base_unit.abilities:
            base_unit = _dc_replace(base_unit, abilities=new_abilities)
    # Stash the WML role so later [filter] role= matching can find
    # this unit (Mini Maps' MODIFY_UNIT (role=monster) MP-zeroing).
    role = (action.attrs.get("role", "") or "").strip().strip('"')
    if role:
        setattr(base_unit, "_wml_role", role)
    # `ai_special=guardian` sets STATE_GUARDIAN (1.18.4 unit.cpp:659),
    # and the default AI's move phase then hands the unit a move from
    # its own hex to its own hex -- "is guardian, staying still"
    # (ca_move_to_targets.cpp:269-277). It is therefore the engine's
    # own reason why a neutral unit does not roam, and tools/neutral_ai
    # relies on exactly that. Read, not assumed: the dependency is
    # asserted in `neutral_ai.run_neutral_side_turn`.
    if (action.attrs.get("ai_special", "") or "").strip().strip('"') == "guardian":
        setattr(base_unit, "_ai_guardian", True)
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
    place.

    CONTRACT: `u` must be a FORK-PRIVATE unit -- either freshly built
    and not yet inserted into `gs.map.units` (`_unit_action`), or a
    `copy.copy` swapped in via the replace-unit pattern
    (`_object_action`). Unit objects in `gs.map.units` are SHARED
    across MCTS forks (Map.__deepcopy__), so mutating one in place
    here would leak into the live game. Every branch below REBINDS
    fields with freshly-built containers (never `.append`/`.add` on
    an existing one), which is what makes a shallow copy sufficient.

    Supports the apply_to forms used by 2p ladder scenarios
    (Hornshark Island, Caves of the Basilisk, Sullas Ruins,
    Silverhead Crossing, Thousand Stings Garrison):

      - `apply_to=attack [+ range= +/-/set_specials/increase_attacks/
                         increase_damage/set_attack_weight]`
      - `apply_to=new_attack` (add a new Attack)
      - `apply_to=remove_attacks` (clear all attacks)
      - `apply_to=hitpoints` (increase_total, set)
      - `apply_to=movement` (set, increase)
      - `apply_to=status` (add named status flag)
      - `apply_to=new_ability` / `remove_ability` (by the `[abilities]`
        children's `id=`)
      - `apply_to=movement_costs`: dropped, silently for a neutral
        side's unit (it never moves), with a warning for a player's
      - cosmetic values in `_COSMETIC_APPLY_TO`: no-op.

    Anything else is logged once and dropped -- see `_COSMETIC_APPLY_TO`.
    """
    apply_to = (eff.attrs.get("apply_to", "") or "").strip().strip('"')
    from wesnoth_ai.classes import Attack
    from wesnoth_ai.combat import DAMAGE_TYPES

    if apply_to == "attack":
        weapon_range = (eff.attrs.get("range", "") or "").strip().strip('"')
        new_specials = _effect_member_ids(eff.first("set_specials"))
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
                # NOTE: this is `mode=append`. The engine DEFAULTS to
                # replace -- `[set_specials]` clears the weapon's
                # specials unless `mode=append` exactly
                # (src/units/attack_type.cpp:416-429: `if(mode !=
                # "append") { specials_.clear(); }`, with a deprecation
                # warning when mode is unset). Modelling replace needs
                # a way to say "these are ALL the specials": our
                # `Attack.weapon_specials` is an additive overlay that
                # `_to_combat_unit` unions with the scraped base
                # (tools/replay_dataset.py). No scenario in either pool
                # uses [set_specials] -- only Hornshark Island, whose
                # bow has no base specials, so append and replace agree
                # there. Recorded in BACKLOG.md rather than half-fixed.
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
        specials = _effect_member_ids(
            eff.first("specials") or eff.first("set_specials"))
        # Map type string -> DamageType enum index.
        try:
            type_id_idx = DAMAGE_TYPES.index(wtype.lower())
        except ValueError:
            type_id_idx = 0  # fallback to blade
        from wesnoth_ai.classes import DamageType
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
            new_st = set(u.statuses)
            new_st.add(add)
            u.statuses = new_st
        if rem and rem in u.statuses:
            new_st = set(u.statuses)
            new_st.discard(rem)
            u.statuses = new_st
        return

    if apply_to in ("new_ability", "remove_ability"):
        ids = _effect_member_ids(eff.first("abilities"))
        if ids:
            have = set(u.abilities or set())
            u.abilities = (have | ids) if apply_to == "new_ability" else (have - ids)
        return

    if apply_to == "movement_costs":
        # Movement costs matter only to a unit that moves. The pool's
        # carriers are the minis' neutral Tentacles, which the neutral AI
        # never moves (neutral_ai._check_units_are_stationary enforces
        # it); a player unit's costs would be a real gap.
        if u.side in (1, 2):
            _warn_unmodelled_apply_to(apply_to)
        return

    if apply_to and apply_to not in _COSMETIC_APPLY_TO:
        _warn_unmodelled_apply_to(apply_to)


def _warn_unmodelled_apply_to(apply_to: str) -> None:
    if apply_to not in _APPLY_TO_GAPS_SEEN:
        _APPLY_TO_GAPS_SEEN.add(apply_to)
        log.warning("[effect] apply_to=%r is not modelled; the effect is "
                    "dropped. Scenario behaviour will differ from Wesnoth.",
                    apply_to)


def _effect_member_ids(container: Optional[WMLNode]) -> set:
    """The ids declared by an `[effect]` container's children.

    OUR model names an ability and a weapon special by the engine's
    `id=`: `unit_stats.json` scrapes them that way, combat asks
    `"magical" in weapon.specials` (wesnoth_ai/combat.py) and the fog
    gate asks `"submerge" in unit.abilities` (wesnoth_ai/visibility.py).
    So an `[effect]`'s children must be read by `id=`, not by the tag
    carrying them. Three specials share the `[chance_to_hit]` tag
    (wesnoth_src/data/core/macros/weapon_specials.cfg):

        #define WEAPON_SPECIAL_MAGICAL
            [chance_to_hit]
                id=magical
                value=70

    and every ability is `[hides] id=submerge`, `[hides] id=ambush`
    (wesnoth_src/data/core/macros/abilities.cfg). Reading the tag gave
    `chance_to_hit` and `hides`, which nothing consumes, so the
    special or ability was created and then silently did nothing.

    Why non-obvious: the ENGINE uses both keys, for different jobs. It
    resolves an effect's numbers by TAG -- `get_specials_and_abilities
    ("chance_to_hit")` (src/actions/attack.cpp:173),
    `get_ability_bool("hides", loc)` (src/units/unit.cpp:2620-2622) --
    and identifies a member by `id=` for dedup, removal and named
    lookup (`has_ability_by_id`, unit.cpp:1414-1423;
    `remove_ability_by_id`, unit.cpp:1425-1436; `has_special` matches
    tag OR id, abilities.cpp:807-814). A unified id-keyed set is the
    right shape for us because we flatten "which rule fires" into the
    name, and the engine's ids are unique where its tags are not.

    The tag is the fallback for a block with no `id=`. The engine keeps
    such a block working by tag but makes it invisible to every
    id-keyed operation (a blank attribute never compares equal to a
    non-empty string, src/config_attribute_value.cpp:422-427).
    """
    if container is None:
        return set()
    out = set()
    for ch in container.children:
        raw = (ch.attrs.get("id") or "").strip().strip('"')
        out.add(raw or ch.tag)
    return out


# `apply_to` values that are DISPLAY ONLY, so dropping them changes
# nothing. Anything outside this set and outside the branches below is
# a gap and is logged: silence is how `new_ability` went missing (2p
# Silverhead Crossing grants submerge to its Tentacle by [object] and
# we dropped it).
#
# Keep this list strictly display-only. Several values that LOOK
# incidental are rules we model, and listing them here would re-create
# the same silence:
#   loyal     -> upkeep skips loyal units (replay_dataset, init_side
#                gold), and we test the `loyal` TRAIT, which an
#                [effect] does not set
#   zoc       -> pathfind_sim's zone of control
#   fearless  -> combat's time-of-day penalty
#   healthy   -> resting and poison
#   variation / type -> the stats the whole sim reads
# None of those appears in either scenario pool today; if one shows
# up, the warning is how we find out.
_COSMETIC_APPLY_TO = frozenset({
    "ellipse", "image_mod", "overlay", "profile", "new_animation",
    "halo", "portrait", "small_profile", "description", "usage",
})
_APPLY_TO_GAPS_SEEN: set = set()


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
    # REPLACE-UNIT pattern (fork-shared-mutable-state class, audit
    # 2026-07-29; same class as the terrain COW 2026-07-18 and the
    # village bit fa95da5): `targets` are the Unit objects that
    # `Map.__deepcopy__` SHARES across every MCTS fork, so applying
    # effects in place on them would rewrite the live game (and all
    # sibling forks) whenever an [object] event fires inside a search.
    # Apply effects to a shallow copy instead and swap it into the
    # (fork-local) unit set -- every `_apply_effect_to_unit` branch
    # REBINDS fields with freshly-built containers, so the shallow
    # copy fully isolates the original. Unit eq/hash is (id, side),
    # so discard+add replaces in place. Per-unit effect order equals
    # the original effects-outer loop: each effect touches only the
    # one unit it's applied to.
    effects = action.all("effect")
    for u in targets:
        new_u = _copy.copy(u)
        for eff in effects:
            _apply_effect_to_unit(new_u, eff)
        # Persist the effect nodes on the unit: Wesnoth stores the
        # [object] in the unit's [modifications] and RE-APPLIES it on
        # advancement (same persistence rule as traits). Our
        # advancement rebuilds attacks from the new type's base
        # stats, which silently dropped object-granted weapon
        # specials: Hornshark's MODIFY_BOWMAN firststrike vanished
        # when the (28,24) Bowman leveled to Longbowman, flipping
        # strike order in every later defense (16349, engine-clean,
        # user viewer ledger 2026-08-07). `_advance_unit_once`
        # re-applies this stash after its rebuild.
        setattr(new_u, "_object_effects",
                list(getattr(new_u, "_object_effects", []) or []) +
                list(effects))
        _swap_unit(gs, u, new_u)


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
    # WL_Mappack / Seamless start-event support (2026-08-06):
    "capture_village": _capture_village_action,
    "modify_unit":     _modify_unit_action,
    "store_unit":      _store_unit_action,
    "if":              _if_action,
}

# Tags we deliberately do nothing with, each with the reason. A tag
# reaches the no-op ONLY through this table.
#
# The default used to be the no-op itself, so an unrecognised tag was
# indistinguishable from a classified one: `[unstore_unit]`, which puts
# back a unit `[store_unit kill=yes]` has removed, was silently dropped,
# and so was `[foreach]`. Nothing said so. The default now fails.
_IGNORED_ACTIONS: Dict[str, str] = {
    "message":    "text shown to a human player",
    "note":       "text shown to a human player",
    "objectives": "text shown to a human player",
    "objective":  "text shown to a human player",
    "item":       "a map decoration; no unit or terrain state",
    "label":      "a map decoration; no unit or terrain state",
    "music":      "sound",
    "sound":      "sound",
    "scroll":     "moves the human player's viewport",
    "screen_fade": "moves the human player's viewport",
    "delay":      "waits for a human to read the screen",
    "variable":   "a [variable] PREDICATE, read by _if_action, never an action",
    "case":       "a [case] branch, read by _switch_action, never an action",
}

# Tags whose behaviour is real, and which the sim produces some other
# way. Each names what stands in for it; deleting that substitute
# without implementing the tag is a bug, which is why the reason is
# here rather than in a comment on the no-op.
_SUBSTITUTED_ACTIONS: Dict[str, str] = {
    "endlevel": (
        "wesnoth_sim decides termination itself (_check_game_over) and "
        "scores the turn cap itself, so the scenario's own end condition "
        "is not replayed"),
    "end_turn": (
        "the pool's only [end_turn] is Silverhead Crossing's `side 3 "
        "turn` event, on a controller=null side that never takes a turn "
        "in the engine (skip_empty_sides) or here (_neutral_actor_sides "
        "censuses controller != null). Inert by that precondition, which "
        "tests/test_action_classification.py pins"),
}


class UnmodelledWML(RuntimeError):
    """An event-action tag with no handler, no ignore reason and no
    substitute. Raised instead of warned when `WESNOTH_STRICT_WML` is
    set, so a test or a box run can demand the whole surface."""


_UNMODELLED_SEEN: Set[Tuple[str, str]] = set()
_UNMODELLED_COUNTS: Dict[str, int] = {}


def unmodelled_action_counts() -> Dict[str, int]:
    """Tag to the number of times it fell through, for a census or a
    box record. Cleared by `reset_unmodelled_actions`."""
    return dict(_UNMODELLED_COUNTS)


def reset_unmodelled_actions() -> None:
    _UNMODELLED_SEEN.clear()
    _UNMODELLED_COUNTS.clear()


def _report_unmodelled(tag: str, scenario_id: str) -> None:
    _UNMODELLED_COUNTS[tag] = _UNMODELLED_COUNTS.get(tag, 0) + 1
    if os.environ.get("WESNOTH_STRICT_WML"):
        raise UnmodelledWML(
            f"[{tag}] has no handler and no recorded reason to ignore it"
            + (f" (scenario {scenario_id})" if scenario_id else ""))
    key = (tag, scenario_id)
    if key not in _UNMODELLED_SEEN:
        _UNMODELLED_SEEN.add(key)
        log.warning(
            "unmodelled event action [%s]%s: doing nothing. Classify it in "
            "scenario_events._IGNORED_ACTIONS / _SUBSTITUTED_ACTIONS, or "
            "write a handler.", tag,
            f" in {scenario_id}" if scenario_id else "")


# The scenario whose event is firing. Nested actions -- the bodies of
# [if], [switch] and [fire_event] -- dispatch through `_apply_action`
# from inside handlers whose signature is (gs, action), so they cannot
# be handed the id; without this an unmodelled tag inside an [if] was
# reported as belonging to no scenario, and because reports dedupe on
# (tag, scenario) a second scenario hitting the same nested tag stayed
# silent. A ContextVar rather than a module global so concurrent sims
# in one process (MCTS forks, threaded rollout workers) cannot see each
# other's scenario.
_FIRING_SCENARIO: contextvars.ContextVar[str] = contextvars.ContextVar(
    "firing_scenario", default="")


def _apply_action(gs: GameState, action: WMLNode,
                  scenario_id: str = "") -> None:
    handler = _ACTION_HANDLERS.get(action.tag)
    if handler is not None:
        handler(gs, action)
        return
    if action.tag in _IGNORED_ACTIONS or action.tag in _SUBSTITUTED_ACTIONS:
        return
    _report_unmodelled(action.tag, scenario_id or _FIRING_SCENARIO.get())


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def fire_event(gs: GameState, events: List[ScenarioEvent], trigger: str) -> int:
    """Fire every event that answers to `trigger`, in WML order.
    Returns the number of events fired. Latches `first_time_only` so
    subsequent calls with the same trigger don't re-fire.

    Names compare as the engine standardizes them (`event_names`): the
    engine fires "turn_refresh" while scenario WML writes `name=turn
    refresh`, and they match; `name=side 1 turn,side 2 turn` answers to
    both."""
    n = 0
    trig = standard_event_name(trigger)
    for ev in events:
        if trig not in ev.names or not ev.can_fire():
            continue
        token = _FIRING_SCENARIO.set(ev.scenario_id)
        try:
            for action in ev.actions:
                _apply_action(gs, action, ev.scenario_id)
        finally:
            _FIRING_SCENARIO.reset(token)
        ev.fired = True
        n += 1
    return n


def fire_events(gs: GameState, events: List[ScenarioEvent], triggers: List[str]) -> int:
    """`fire_event` for each trigger in order, as the engine pumps a
    sequence of names (the turn events above)."""
    return sum(fire_event(gs, events, trigger) for trigger in triggers)


def load_events_for_scenario(scenario_id: str) -> List[ScenarioEvent]:
    """Convenience wrapper: load the .cfg and return its events. Returns
    an empty list if the scenario isn't found in the source tree."""
    root = load_scenario_wml(scenario_id)
    if root is None:
        return []
    return collect_events(root, scenario_id)


__all__ = [
    "ScenarioEvent", "load_scenario_wml", "load_events_for_scenario",
    "collect_events", "fire_event", "fire_events", "setup_static_time_areas",
    "standard_event_name", "event_names", "side_turn_event_names",
    "turn_refresh_event_names", "side_turn_end_event_names",
    "turn_end_event_names", "init_side_event_names", "any_can_fire",
    "UnmodelledWML", "unmodelled_action_counts", "reset_unmodelled_actions",
    "unknown_macro_counts", "reset_unknown_macros",
]
