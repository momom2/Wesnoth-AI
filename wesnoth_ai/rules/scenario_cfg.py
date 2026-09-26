"""A scenario's .cfg read the way the game reads it: found by its id,
preprocessed, parsed.

`load_scenario_wml(scenario_id)` finds the file with
`find_scenario_cfg_path` (the mainline multiplayer scenarios, the add-ons
vendored under wesnoth_src, the project's own add-on) and returns
`parse_scenario_cfg(path)`: the file with its preprocessor conditionals
resolved as the engine resolves them (`evaluate_conditionals`), its macros
expanded (the core macros, the add-on's utility files and the file's own
`#define`s), its parallel assignments split, and the result parsed into a
`WMLNode` tree.

A cosmetic macro, or one whose behaviour the simulator implements itself,
expands to nothing on purpose (`_COSMETIC_MACROS`, `_SUBSTITUTED_MACROS`).
An unknown macro expands to nothing with a warning, and raises
`UnmodelledWML` under `WESNOTH_STRICT_WML`.
tools/analysis/expansion_diff.py compares this expansion with the game's
own for every pool scenario.

Dependencies: tools.replay_extract (parse_wml), wesnoth_ai.paths
Dependents: tools.scenario_events, wesnoth_ai.rules.scenario_pool, the
replay exporter, the scenario detectors
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

from wesnoth_ai.paths import ADDONS_DIR, WESNOTH_SRC_DIR
from tools.replay_extract import WMLNode, parse_wml


log = logging.getLogger("scenario_cfg")

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


class UnmodelledWML(RuntimeError):
    """An event-action tag with no handler, no ignore reason and no
    substitute. Raised instead of warned when `WESNOTH_STRICT_WML` is
    set, so a test or a box run can demand the whole surface."""


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
