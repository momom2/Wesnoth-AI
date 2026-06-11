"""Scrape Wesnoth 1.18 unit configs into a Python lookup table.

We deliberately DON'T do general WML macro expansion — animation/sound
macros nest exponentially and the only macros that matter for stats are
the defense / resistance / movement-cost ones in
`data/core/macros/movetypes.cfg`. Strategy:

  1. Parse `data/core/macros/movetypes.cfg` for `#define` blocks that
     contain `[defense]`, `[resistance]`, `[movement_costs]` blocks.
     Also catch the bare `{MOVE_TYPE_X}` family.
  2. Parse `data/core/units.cfg` for `[movetype]` blocks. For each,
     start with the inline values then layer on whatever macros it
     calls (single-level lookup into the table from step 1).
  3. Parse `data/core/units/**/*.cfg` for `[unit_type]` blocks. Inline
     stats give us hp/exp/cost/etc. Defense / resistance default to
     the unit's `movement_type` table; explicit `[defense]` /
     `[resistance]` blocks under `[unit_type]` override.
  4. Emit one JSON: `{movement_types: {...}, units: {...}}`.

Robust to the mass of animation macros that DON'T affect stats: any
unrecognized `{...}` line is silently skipped.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set


TAG_OPEN_RE  = re.compile(r'^\s*\[([+]?[a-zA-Z_][a-zA-Z0-9_]*)\]\s*$')
TAG_CLOSE_RE = re.compile(r'^\s*\[/([a-zA-Z_][a-zA-Z0-9_]*)\]\s*$')
KEY_RE       = re.compile(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*?)\s*$')
INVOKE_RE    = re.compile(r'^\s*\{(\S+)(?:\s+([^}]*))?\}\s*$')
DEFINE_RE    = re.compile(r'^\s*#define\s+(\S+)(?:\s+(.*))?$')
ENDDEF_RE    = re.compile(r'^\s*#enddef\s*$')


DAMAGE_TYPES = ["blade", "pierce", "impact", "fire", "cold", "arcane"]
TERRAINS = [
    "deep_water", "shallow_water", "reef", "swamp_water",
    "flat", "sand", "forest", "hills", "mountains",
    "village", "castle", "cave", "fungus", "frozen",
    "unwalkable", "impassable",
]


# ---------------------------------------------------------------------
# Trait-macro registry: TRAIT_NAME → (id, availability)
# ---------------------------------------------------------------------
#
# Order matches `data/core/macros/traits.cfg` and reflects the
# canonical Wesnoth 1.18 trait set. Wesnoth's `unit_type::
# possible_traits()` iterates these in WML declaration order, so the
# result of `mt() % len(candidates)` depends on this order — keep it
# matching the source. Availability values:
#   "any"       — eligible for the random pool
#   "musthave"  — auto-applied, no random call
#   "none"      — never granted (loyal-style placeholders)
TRAIT_MACROS: Dict[str, tuple] = {
    "TRAIT_LOYAL":              ("loyal",      "any"),
    "TRAIT_LOYAL_HERO":         ("loyal",      "any"),
    "TRAIT_LOYAL_HERO_NOSLOT":  ("loyal",      "any"),
    "TRAIT_UNDEAD":             ("undead",     "musthave"),
    "TRAIT_MECHANICAL":         ("mechanical", "musthave"),
    "TRAIT_ELEMENTAL":          ("elemental",  "musthave"),
    "TRAIT_STRONG":             ("strong",     "any"),
    "TRAIT_DEXTROUS":           ("dextrous",   "any"),
    "TRAIT_QUICK":              ("quick",      "any"),
    "TRAIT_INTELLIGENT":        ("intelligent","any"),
    "TRAIT_RESILIENT":          ("resilient",  "any"),
    "TRAIT_HEALTHY":            ("healthy",    "any"),
    "TRAIT_FEARLESS":           ("fearless",   "any"),
    "TRAIT_FEARLESS_MUSTHAVE":  ("fearless",   "musthave"),
    "TRAIT_FERAL_MUSTHAVE":     ("feral",      "musthave"),
    "TRAIT_WEAK":               ("weak",       "any"),
    "TRAIT_SLOW":               ("slow",       "any"),
    "TRAIT_DIM":                ("dim",        "any"),
    "TRAIT_AGED":               ("aged",       "any"),
    "TRAIT_OCEANGOING_MUSTHAVE": ("oceangoing", "musthave"),
    "TRAIT_PUSH_POLES_MUSTHAVE": ("push_poles", "musthave"),
}

# Global traits added at the [units] level for any race that doesn't
# set ignore_global_traits=yes. Order matches data/core/units.cfg.
GLOBAL_TRAIT_MACROS = ["TRAIT_STRONG", "TRAIT_QUICK",
                       "TRAIT_INTELLIGENT", "TRAIT_RESILIENT"]


# ---------------------------------------------------------------------
# Tiny WML tree
# ---------------------------------------------------------------------

class Node:
    __slots__ = ("tag", "attrs", "children")

    def __init__(self, tag: str):
        self.tag = tag
        self.attrs: Dict[str, str] = {}
        self.children: List["Node"] = []


def _strip_value(val: str) -> str:
    if "#" in val and not (val.startswith('_') or val.startswith('"')):
        val = val.split("#", 1)[0].strip()
    if len(val) >= 2 and val[0] == '"' and val.endswith('"'):
        val = val[1:-1]
    if val.startswith("_ "):
        v2 = val[2:].strip()
        if len(v2) >= 2 and v2[0] == '"' and v2.endswith('"'):
            val = v2[1:-1]
        else:
            val = v2
    return val


def _split_macro_args(s: str) -> List[str]:
    """Split a WML macro argument string into individual args. Honors
    parenthesised expressions like `( _ "bite")` (a textdomain-tagged
    translatable string literal — kept whole) and double-quoted
    literals. Without this, `{X ( _ "foo") "bar.png" baz}` would split
    into ['(', '_', '"foo")', '"bar.png"', 'baz'] and the third
    parameter would resolve to the wrong thing."""
    args: List[str] = []
    cur: List[str] = []
    depth = 0
    in_quote = False
    for ch in s:
        if in_quote:
            cur.append(ch)
            if ch == '"':
                in_quote = False
            continue
        if ch == '"':
            cur.append(ch); in_quote = True; continue
        if ch == "(":
            cur.append(ch); depth += 1; continue
        if ch == ")":
            cur.append(ch); depth = max(0, depth - 1); continue
        if depth == 0 and ch.isspace():
            if cur:
                args.append("".join(cur)); cur = []
            continue
        cur.append(ch)
    if cur:
        args.append("".join(cur))
    return args


def parse_lines(
    lines: List[str],
    inline_macros: Optional[Dict[str, tuple[List[str], List[str]]]] = None,
) -> Node:
    """Parse WML lines into a Node tree.

    `inline_macros` is a name → (param_names, body_lines) dict. When a
    `{NAME arg1 arg2 ...}` line is encountered, the macro body is
    parsed in place with simple `{PARAM}` text substitution. Lines we
    don't recognize (animation/sound macros not in the dict) are
    silently dropped.
    """
    inline_macros = inline_macros or {}
    root = Node("__root__")
    stack: List[Node] = [root]

    def emit_lines(buf: List[str]) -> None:
        for line in buf:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            m = TAG_OPEN_RE.match(line)
            if m:
                node = Node(m.group(1).lstrip("+"))
                stack[-1].children.append(node)
                stack.append(node)
                continue
            m = TAG_CLOSE_RE.match(line)
            if m:
                if len(stack) > 1 and stack[-1].tag == m.group(1):
                    stack.pop()
                continue
            m = INVOKE_RE.match(line)
            if m:
                name = m.group(1)
                args_str = m.group(2) or ""
                spec = inline_macros.get(name)
                if spec is None:
                    continue  # unknown macro → skip silently
                params, body = spec
                args = _split_macro_args(args_str)
                # Pad args; substitute {PARAM} in body literal-string-wise.
                while len(args) < len(params):
                    args.append("")
                substituted = []
                for bline in body:
                    for pname, pval in zip(params, args):
                        bline = bline.replace("{" + pname + "}", pval)
                    substituted.append(bline)
                emit_lines(substituted)
                continue
            m = KEY_RE.match(line)
            if m:
                key = m.group(1)
                val = _strip_value(m.group(2))
                stack[-1].attrs[key] = val

    emit_lines(lines)
    return root


# ---------------------------------------------------------------------
# Movetype-macro extraction
# ---------------------------------------------------------------------

def load_macro_defs(path: Path) -> Dict[str, tuple[List[str], List[str]]]:
    """Parse `#define NAME [PARAM1 PARAM2 ...]` … `#enddef` from one
    file. Returns name → (param_names, body_lines).

    Param substitution: macro body uses `{PARAM_NAME}` to refer to the
    argument; the parser substitutes literally before re-parsing the
    expanded body. Most stat-relevant macros (LESS_NIMBLE_ELF,
    WOODLAND_RESISTANCE, ...) are argless; FLY_DEFENSE / MOUNTAIN_DEFENSE
    take a single numeric argument.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    return _scan_macro_defs(text)


def _scan_macro_defs(text: str) -> Dict[str, tuple[List[str], List[str]]]:
    """Same as load_macro_defs but on an already-loaded source string.
    Used to harvest inline `#define` blocks from per-unit cfgs (e.g.,
    Corpse_Walking.cfg defines UNIT_BODY_WALKING_CORPSE_STATS at file
    top before its [unit_type] block uses it)."""
    out: Dict[str, tuple[List[str], List[str]]] = {}
    cur_name: Optional[str] = None
    cur_args: List[str] = []
    cur_body: List[str] = []
    for line in text.splitlines():
        if cur_name is None:
            m = DEFINE_RE.match(line)
            if m:
                cur_name = m.group(1)
                cur_args = (m.group(2) or "").split() if m.group(2) else []
                cur_body = []
            continue
        if ENDDEF_RE.match(line):
            out[cur_name] = (cur_args, cur_body)
            cur_name = None
            continue
        cur_body.append(line)
    return out


# ---------------------------------------------------------------------
# Movetype + unit extraction
# ---------------------------------------------------------------------

def _table(node: Node, tag: str, keys: List[str], default: int) -> Dict[str, int]:
    out = {k: default for k in keys}
    n = next((c for c in node.children if c.tag == tag), None)
    if n is None:
        return out
    for k in keys:
        if k in n.attrs:
            try:
                out[k] = int(n.attrs[k])
            except ValueError:
                pass
    return out


def _has_block(node: Node, tag: str) -> bool:
    return any(c.tag == tag for c in node.children)


def extract_races(units_cfg: Path, macros: Dict[str, List[str]]) -> Dict[str, dict]:
    """Parse `[race]` blocks and capture the trait info Wesnoth uses
    when generating units of that race:

      num_traits           — random rolls per unit (from [race]
                             num_traits=, default 2).
      ignore_global_traits — if true, the global pool (strong/quick/
                             intelligent/resilient) is NOT included.
      trait_macros         — list of `TRAIT_X` macros invoked at race
                             level, in WML declaration order.
    """
    text = units_cfg.read_text(encoding="utf-8", errors="replace")
    root = parse_lines(text.splitlines(), inline_macros=macros)
    out: Dict[str, dict] = {}

    def collect(n: Node):
        if n.tag != "race":
            for c in n.children:
                collect(c)
            return
        race_id = n.attrs.get("id", "")
        if not race_id:
            return
        # Find {TRAIT_*} invocations in order. We re-read the raw lines
        # for THIS race block — invocations are not preserved as nodes.
        # The full race body is between the [race] open/close lines in
        # the source; trace by re-scanning the raw text.
        out[race_id] = {
            "num_traits": int(n.attrs.get("num_traits", 2) or 2),
            "ignore_global_traits":
                (n.attrs.get("ignore_global_traits", "no") or "no").lower()
                in ("yes", "true", "1"),
            # Race-level undead_variation default. Per types.cpp:209-213,
            # when a unit's own undead_variation is empty, the engine
            # falls back to the race's. This drives plague spawn type:
            # a Gryphon Rider (race=gryphon, unit undead_variation="")
            # leaves a "Walking Corpse:gryphon" corpse (fly mvt, 5 MP,
            # 21 hp), not a base "Walking Corpse" (smallfoot, 4 MP,
            # 18 hp). Used by _spawn_plague_corpse in replay_dataset.
            "undead_variation":
                (n.attrs.get("undead_variation", "") or "").strip(),
            # trait_macros filled below from raw text
            "trait_macros": [],
        }

    collect(root)

    # Re-scan raw text for TRAIT_X invocations within each [race] block.
    lines = text.splitlines()
    cur_race: Optional[str] = None
    inside_race_depth = 0
    for line in lines:
        stripped = line.strip()
        m_open  = TAG_OPEN_RE.match(line)
        m_close = TAG_CLOSE_RE.match(line)
        if m_open and m_open.group(1) == "race":
            inside_race_depth += 1
            continue
        if m_close and m_close.group(1) == "race":
            cur_race = None
            inside_race_depth = max(0, inside_race_depth - 1)
            continue
        if inside_race_depth >= 1:
            kv = KEY_RE.match(line)
            if kv and kv.group(1) == "id":
                rid = _strip_value(kv.group(2))
                if rid in out:
                    cur_race = rid
            elif cur_race:
                inv = re.match(r'^\s*\{(TRAIT_\w+)(?:\s+[^}]*)?\}\s*$', line)
                if inv:
                    out[cur_race]["trait_macros"].append(inv.group(1))
    return out


def _scan_unit_trait_macros(node: Node, raw_text: str) -> List[str]:
    """Pull TRAIT_X macros invoked at the top level of a [unit_type]
    block (e.g. {TRAIT_FEARLESS_MUSTHAVE} on Walking Corpse,
    {TRAIT_FERAL_MUSTHAVE} on bats, {TRAIT_FEARLESS} on Heavy Infantry).
    These are not preserved as nodes by our parser because they're
    macro invocations, not WML tags — so we re-scan the raw text.

    Bounds: from the line containing `id=<this unit's id>` to the
    matching [/unit_type] closer. Crude but works because unit_type
    blocks aren't typically nested.
    """
    target_id = node.attrs.get("id", "")
    if not target_id:
        return []
    out: List[str] = []
    lines = raw_text.splitlines()
    in_block = False
    depth = 0
    for line in lines:
        m_open = TAG_OPEN_RE.match(line)
        m_close = TAG_CLOSE_RE.match(line)
        kv = KEY_RE.match(line)
        if m_open and m_open.group(1) == "unit_type":
            depth += 1
            in_block = False
            continue
        if m_close and m_close.group(1) == "unit_type":
            if in_block:
                return out
            depth = max(0, depth - 1)
            continue
        if depth == 1 and kv and kv.group(1) == "id":
            in_block = (_strip_value(kv.group(2)) == target_id)
            continue
        if in_block and depth == 1:
            inv = re.match(r'^\s*\{(TRAIT_\w+)(?:\s+[^}]*)?\}\s*$', line)
            if inv:
                out.append(inv.group(1))
    return out


def build_trait_info(race_data: dict, unit_macros: List[str], *,
                     ignore_race_traits: bool = False,
                     alignment: str = "neutral") -> dict:
    """Combine race-level + unit-level trait macros into the per-unit
    trait pool that mirrors Wesnoth's `unit_type::possible_traits()`.

    Source: `wesnoth_src/src/units/types.cpp:337-363`. The engine
    builds the trait pool as:

      1. Start with global traits (passed in by advance_to).
      2. If `race->uses_global_traits()` is false: clear pool.
      3. If `cfg["ignore_race_traits"]==yes` (unit-type level): clear
         pool. Otherwise, add the race's `additional_traits`
         AND skip `fearless` if the unit is neutral-aligned
         (types.cpp:350: `if(alignment_ != neutral ||
         t["id"] != "fearless")`). The skip exists because fearless
         only matters for time-of-day-aligned units; on a neutral
         unit it's a dead trait and the engine omits it from the
         random pool.
      4. Always: add unit-type's own [trait] children
         (e.g. Dark Adept's inline {TRAIT_QUICK} {TRAIT_INTELLIGENT}
         {TRAIT_RESILIENT}).

    Critical: step 3 means the unit-type's `ignore_race_traits=yes`
    flag REPLACES the entire global+race pool with the unit's own
    list. Three units in 1.18.4 use this: Dark Adept (3-trait pool
    quick/intelligent/resilient, NO strong), Bay Horse (Horse_Black),
    Black Horse (Horse_Dark). Without this fix, our scraper gives
    Dark Adept a 4-trait pool with strong, which makes seed-driven
    trait rolls diverge from Wesnoth's behavior.

    Returns:
      {
        "num_traits": <int>,         # from race
        "musthave":  [trait_id...],  # auto-applied
        "pool":      [trait_id...],  # eligible for random pick
      }
    """
    musthave: List[str] = []
    musthave_seen: Set[str] = set()
    pool: List[str] = []
    is_neutral = (alignment or "neutral").strip().lower() == "neutral"

    def add(macro_name: str, *, skip_fearless: bool = False) -> None:
        info = TRAIT_MACROS.get(macro_name)
        if info is None:
            return
        tid, avl = info
        if skip_fearless and tid == "fearless":
            # types.cpp:350: the engine skips fearless from race
            # additional_traits when the unit is neutral. Match by
            # trait id, not macro name, to mirror the C++ check
            # `t["id"] != "fearless"`.
            return
        if avl == "musthave":
            # Musthaves dedup -- adding the same musthave twice doesn't
            # change unit behavior; we track them as a flat set on the
            # unit. Wesnoth's `add_child` would create duplicate trait
            # records, but the must-have phase of generate_traits skips
            # already-applied IDs, so the effect is a single application.
            if tid in musthave_seen:
                return
            musthave_seen.add(tid)
            musthave.append(tid)
        elif avl == "any":
            # Random-pool traits: KEEP DUPS. Wesnoth's
            # unit_type::build_full at types.cpp:337-363 calls
            # `possible_traits_.add_child("trait", t)` once per occurrence
            # in the global pool AND once per race additional_trait;
            # config::add_child appends without dedup
            # (wesnoth_src/src/config.cpp:442). The random-fill loop in
            # generate_traits (unit.cpp:813-828) then iterates the WHOLE
            # possible_traits list, building candidate_traits with
            # duplicates -- which means trolls (race adds STRONG, QUICK,
            # RESILIENT also in global pool) have those traits at 2/N
            # probability instead of 1/N. Without preserving dups our
            # seed-driven trait roll picks the wrong trait when the dup
            # appears at the rolled index.
            pool.append(tid)
        # availability == "none" (or unknown) → skip

    if ignore_race_traits:
        # types.cpp:346-347: `if(cfg["ignore_race_traits"].to_bool())
        # possible_traits_.clear();` — wipe everything race-related
        # (global pool was already added before this, but we never
        # added it). Then add only the unit-type's own traits below.
        pass
    else:
        # Global traits if race doesn't ignore them.
        if not race_data.get("ignore_global_traits", False):
            for m in GLOBAL_TRAIT_MACROS:
                add(m)
        # Race-level traits (e.g. {TRAIT_DEXTROUS} for elf,
        # {TRAIT_HEALTHY} for dwarf, {TRAIT_UNDEAD} for undead,
        # the four-trait pool {TRAIT_STRONG/QUICK/RESILIENT/FEARLESS}
        # for trolls). Race additionals append on top of global, with
        # duplicates preserved per the comment in `add` above.
        # Pass skip_fearless=is_neutral to mirror types.cpp:350: the
        # engine omits fearless from race-additional pool for neutral
        # units (default-era trolls are chaotic, so this is dormant
        # for ladder play; matters for hypothetical neutral-troll
        # eras and for any future race that adds fearless).
        for m in race_data.get("trait_macros", []):
            add(m, skip_fearless=is_neutral)

    # Unit-type-level macros (e.g. {TRAIT_FERAL_MUSTHAVE} on bats,
    # {TRAIT_FEARLESS_MUSTHAVE} on Walking Corpse, and the inline
    # {TRAIT_QUICK}/{TRAIT_INTELLIGENT}/{TRAIT_RESILIENT} on Dark
    # Adept). Always added regardless of ignore_race_traits.
    for m in unit_macros:
        add(m)

    return {
        "num_traits": int(race_data.get("num_traits", 2)),
        "musthave":   musthave,
        "pool":       pool,
    }


def extract_movetypes(units_cfg: Path, macros: Dict[str, List[str]]) -> Dict[str, dict]:
    """Walk units.cfg [movetype] blocks. Apply argless macro
    expansion in-place where the unit calls e.g. {DRAKEFLY_RESISTANCE}.
    """
    text = units_cfg.read_text(encoding="utf-8", errors="replace")
    root = parse_lines(text.splitlines(), inline_macros=macros)

    out: Dict[str, dict] = {}

    def collect(n: Node):
        if n.tag == "movetype":
            name = n.attrs.get("name", "?")
            raw_defense = _table(n, "defense", TERRAINS, 100)
            # PRESERVE the negative sign at the movetype layer.
            # Wesnoth uses negative defense values as CAPS (floor on
            # to-be-hit). E.g. mounted's `forest=-70` means "70%
            # to-be-hit minimum on any forest-aliased terrain". The
            # cap matters for composite terrains (Gg^Fet, Gg^Fp,
            # etc.) where naive min-over-aliases would pick the
            # flat value (60) and bypass the forest cap. The
            # terrain_resolver's `_collect_neg_caps` walks the
            # alias tree and applies these floors AFTER the alias
            # min, only if the negative entry is preserved here.
            # Stripping the sign was a Stage-14 bug -- 2026-05-04
            # caught via Weldyn Channel Turn 23 (40663): Horseman
            # on Gg^Fet had 40% def instead of 30%, drains
            # retaliation missed where it should have hit.
            defense = {k: int(v) for k, v in raw_defense.items()}
            out[name] = {
                "name": name,
                "defense":     defense,
                "resistance":  _table(n, "resistance",     DAMAGE_TYPES, 100),
                "movement_costs": _table(n, "movement_costs", TERRAINS, 99),
            }
            return
        for c in n.children:
            collect(c)

    collect(root)
    return out


# Map `{WEAPON_SPECIAL_X}` macros to their special id. Each macro
# expands to `[<special_id>] id=<special_id> ... [/<special_id>]`,
# so the special id matches the inner tag name. Order doesn't matter.
WEAPON_SPECIAL_MACROS: Dict[str, str] = {
    "WEAPON_SPECIAL_BERSERK":     "berserk",
    "WEAPON_SPECIAL_BACKSTAB":    "backstab",
    "WEAPON_SPECIAL_PLAGUE":      "plague",
    "WEAPON_SPECIAL_PLAGUE_TYPE": "plague",
    "WEAPON_SPECIAL_SLOW":        "slow",
    "WEAPON_SPECIAL_PETRIFY":     "petrifies",
    "WEAPON_SPECIAL_MARKSMAN":    "marksman",
    "WEAPON_SPECIAL_DEFLECT":     "deflect",
    "WEAPON_SPECIAL_MAGICAL":     "magical",
    "WEAPON_SPECIAL_SWARM":       "swarm",
    "WEAPON_SPECIAL_CHARGE":      "charge",
    "WEAPON_SPECIAL_ABSORB":      "absorb",
    "WEAPON_SPECIAL_DRAIN":       "drains",
    "WEAPON_SPECIAL_FIRSTSTRIKE": "firststrike",
    "WEAPON_SPECIAL_POISON":      "poison",
    "WEAPON_SPECIAL_STUN":        "stun",
}


def extract_attacks(node: Node, raw_text: str = "") -> List[dict]:
    """Extract [attack] blocks. We pull weapon specials from THREE
    sources, since Wesnoth WML uses all of them:
      1. `specials_list=poison` shorthand attribute on [attack]
      2. inline tag children inside [specials], e.g.
         `[specials] [poison] id=poison [/poison] [/specials]`
      3. `{WEAPON_SPECIAL_X}` macro invocations inside [specials];
         these aren't WML tags so we have to re-scan the raw text.
    """
    out = []
    # Some unit types have MULTIPLE [attack] blocks with the same
    # `name=` but different damage type / range (e.g. Drake Arbiter
    # has two halberd attacks: a blade-melee and a pierce-melee).
    # _scan_attack_special_macros disambiguated by name only,
    # so a `firststrike` macro inside the pierce-halberd's
    # [specials] block leaked into the blade-halberd entry too.
    # Pre-scrape macros once per unit, keyed by (name, type, range,
    # damage, number) so collisions can't happen.
    macros_by_attack = (
        _scan_attack_special_macros_by_index(raw_text)
        if raw_text else []
    )
    attack_idx = 0
    for c in node.children:
        if c.tag != "attack":
            continue
        specials = []
        # 1. specials_list shorthand
        sl = c.attrs.get("specials_list", "").strip()
        if sl:
            specials.extend(s.strip() for s in sl.split(",") if s.strip())
        # 2. parsed [specials] children (each child tag IS the special id)
        sp = next((s for s in c.children if s.tag == "specials"), None)
        if sp is not None:
            for sc in sp.children:
                if sc.tag not in specials:
                    specials.append(sc.tag)
        # 3. macro invocations inside [specials] block — match by
        # attack INDEX (position among [attack] blocks). The text
        # scanner walks [attack] blocks in source order, same as
        # this loop walks node children.
        if attack_idx < len(macros_by_attack):
            for sp_id in macros_by_attack[attack_idx]:
                if sp_id not in specials:
                    specials.append(sp_id)
        out.append({
            "name":    c.attrs.get("name", "?"),
            "type":    c.attrs.get("type", "blade"),
            "range":   c.attrs.get("range", "melee"),
            "damage":  int(c.attrs.get("damage", 0) or 0),
            "number":  int(c.attrs.get("number", 1) or 1),
            # `accuracy` / `parry` are direct numeric attrs on
            # [attack] (units/attack_type.cpp:87-90). They feed
            # the CTH formula at attack.cpp:168-169:
            #     cth = defender_defense_modifier
            #         + attacker.weapon.accuracy
            #         - defender.weapon.parry      (if defender retaliates)
            # then clamped to [0, 100].
            # Default era examples in 1.18.4: Elvish Champion sword
            # (accuracy=10), Marksman/Sharpshooter NOT here (those use
            # the marksman SPECIAL, not accuracy attr). Without these,
            # the Champion's sword effectively misses 10% more often
            # than reality. Witnessed 2026-05-08 in
            # 2p__Den_of_Onis_Turn_65_(135596) cmd[1161]: 5-strike
            # sword vs Troll Whelp on grass (40 cth + 10 accuracy =
            # 50 cth) lands 4 hits in Wesnoth (24 dmg, kills) vs 3
            # hits in our sim (18 dmg, leaves Troll Whelp alive).
            "accuracy": int(c.attrs.get("accuracy", 0) or 0),
            "parry":    int(c.attrs.get("parry", 0) or 0),
            "specials": specials,
        })
        attack_idx += 1
    return out


def _scan_attack_special_macros_by_index(raw_text: str) -> List[List[str]]:
    """Walk [attack] blocks in source order; for each, return the list
    of WEAPON_SPECIAL_X macro ids found inside its [specials] block.
    Returns one list per [attack] block (same order as the parser
    visits attack children). Replaces the old name-based scanner
    which collided when two attacks shared a `name=` (Drake Arbiter
    has two halberd attacks; the firststrike macro on the pierce
    halberd was leaking into the blade halberd entry, breaking
    cmd-1105 combat in
    2p__Cynsaun_Battlefield_Turn_33_(94735).bz2)."""
    out: List[List[str]] = []
    in_attack = False
    in_specials = False
    cur: List[str] = []
    for line in raw_text.splitlines():
        m_open  = TAG_OPEN_RE.match(line)
        m_close = TAG_CLOSE_RE.match(line)
        if m_open:
            tag = m_open.group(1)
            if tag == "attack" and not in_attack:
                in_attack = True
                cur = []
            elif tag == "specials" and in_attack:
                in_specials = True
            continue
        if m_close:
            tag = m_close.group(1)
            if tag == "specials":
                in_specials = False
            elif tag == "attack" and in_attack:
                out.append(cur)
                cur = []
                in_attack = False
                in_specials = False
            continue
        if in_attack and in_specials:
            inv = re.match(r'^\s*\{(WEAPON_SPECIAL_\w+)(?:\s+[^}]*)?\}\s*$', line)
            if inv:
                sp_id = WEAPON_SPECIAL_MACROS.get(inv.group(1))
                if sp_id and sp_id not in cur:
                    cur.append(sp_id)
    return out


def _scan_attack_special_macros(raw_text: str, attack_name: str) -> List[str]:
    """Return any specials_id this `name=<attack_name>` `[attack]` block
    pulls in via `{WEAPON_SPECIAL_X}` macros. Crude bracket-tracking but
    fine for the well-formatted core unit cfgs."""
    out: List[str] = []
    if not attack_name:
        return out
    lines = raw_text.splitlines()
    in_attack = False
    in_specials = False
    depth = 0
    for line in lines:
        m_open  = TAG_OPEN_RE.match(line)
        m_close = TAG_CLOSE_RE.match(line)
        kv      = KEY_RE.match(line)
        if m_open:
            tag = m_open.group(1)
            if tag == "attack":
                in_attack = False  # provisional until name= matches
                depth += 1
            elif tag == "specials" and in_attack:
                in_specials = True
            depth = depth + 1 if tag != "attack" else depth
            continue
        if m_close:
            tag = m_close.group(1)
            if tag == "specials":
                in_specials = False
            elif tag == "attack":
                in_attack = False
            continue
        if kv and kv.group(1) == "name" and not in_specials:
            # name on [attack] level
            in_attack = (_strip_value(kv.group(2)) == attack_name)
            continue
        if in_attack and in_specials:
            inv = re.match(r'^\s*\{(WEAPON_SPECIAL_\w+)(?:\s+[^}]*)?\}\s*$', line)
            if inv:
                sp_id = WEAPON_SPECIAL_MACROS.get(inv.group(1))
                if sp_id:
                    out.append(sp_id)
    return out


ABILITY_MACROS: Dict[str, str] = {
    # Wesnoth's `[abilities]` block is almost always populated with macros
    # like `{ABILITY_REGENERATES}` rather than inline tags. The parser only
    # sees expanded tag children, so we have to re-scan the raw text to
    # recover the ability ids — exactly how WEAPON_SPECIAL_X is handled.
    "ABILITY_HEALING":            "heals_4",
    "ABILITY_HEALS":              "heals_4",
    "ABILITY_HEALS_8":            "heals_8",
    "ABILITY_EXTRA_HEAL":         "heals_8",
    "ABILITY_CURES":              "cures",
    "ABILITY_REGENERATES":        "regenerate",
    "ABILITY_STEADFAST":          "steadfast",
    "ABILITY_LEADERSHIP":         "leadership",
    "ABILITY_SKIRMISHER":         "skirmisher",
    "ABILITY_ILLUMINATES":        "illuminates",
    "ABILITY_TELEPORT":           "teleport",
    "ABILITY_AMBUSH":             "ambush",
    "ABILITY_NIGHTSTALK":         "nightstalk",
    "ABILITY_CONCEALMENT":        "concealment",
    "ABILITY_SUBMERGE":           "submerge",
    "ABILITY_FEEDING":            "feeding",
}


def _scan_unit_ability_macros(raw_text: str) -> List[str]:
    """Find `{ABILITY_X}` invocations inside the unit's `[abilities]`
    block. Returns the canonical id (e.g. `regenerate`) for each known
    macro."""
    if not raw_text:
        return []
    lines = raw_text.splitlines()
    in_abil = False
    depth = 0
    out: List[str] = []
    for line in lines:
        s = line.strip()
        if s.startswith("[abilities]"):
            in_abil = True; depth = 1; continue
        if not in_abil:
            continue
        # Track nested tag depth so we don't consume macros from
        # neighboring blocks if WML formatting puts them on the same line.
        for tok in s.split():
            if tok.startswith("[/"):
                depth -= 1
                if depth <= 0:
                    in_abil = False
            elif tok.startswith("[") and not tok.startswith("[/"):
                depth += 1
        if not in_abil:
            continue
        m = re.search(r"\{(ABILITY_[A-Z0-9_]+)", s)
        if m:
            ab_id = ABILITY_MACROS.get(m.group(1))
            if ab_id and ab_id not in out:
                out.append(ab_id)
    return out


def extract_abilities(node: Node, raw_text: str = "") -> List[str]:
    out: List[str] = []
    abil = next((c for c in node.children if c.tag == "abilities"), None)
    if abil is not None:
        for c in abil.children:
            out.append(c.attrs.get("id", c.tag))
    # Re-scan raw text for macros that the parser didn't expand.
    for ab_id in _scan_unit_ability_macros(raw_text):
        if ab_id not in out:
            out.append(ab_id)
    return out


def extract_variations(node: Node, parent_stats: dict,
                       move_types: Dict[str, dict]) -> Dict[str, dict]:
    """Extract `[variation]` children of a unit_type into a dict keyed
    by variation_id. Each entry is a full per-variation stats dict —
    parent stats overlaid with the variation's overrides — ready to be
    used as a drop-in unit type at runtime.

    This matters for plague reanimation: a Cavalryman killed by plague
    becomes "Walking Corpse:mounted", which has different HP, movement
    type, and damage type from the base Walking Corpse. Without this,
    every spawned corpse used base stats and combat against it (or
    by it) diverged from Wesnoth.

    `inherit=yes` (the typical case) means the variation starts from
    parent_stats and applies its own overrides. We don't try to model
    `inherit=no`; none of the default-era variations use it.

    NB: parent_stats["resistance"] etc. are already merged (movetype
    defaults + unit overrides), so when a variation switches to a
    different movetype we can't reconstruct the parent's explicit
    overrides from parent_stats alone. We re-extract the parent unit's
    own [resistance]/[defense]/[movement_costs] blocks from `node`
    here so they layer correctly on the new movetype's defaults.
    Witnessed 2026-05-08: Walking Corpse:mounted (mounted movetype
    arcane=90) was losing the parent WC's `[resistance] arcane=140`
    override, computing arcane resistance 0.9 instead of 1.4. That
    cut faerie-fire damage in half on mounted corpses, leaving u30
    in 75a38b573ef6 alive at hp=7 instead of dropping to ≤3 at
    cmd[351], which then cascaded into the Shaman missing the
    cmd[371] retaliation kill XP and never advancing to Sorceress
    by cmd[510] (weapon_oob).
    """
    # Parent unit's explicit override blocks (applied AFTER new_mt
    # defaults inside the variation loop below).
    parent_resist_overrides: Dict[str, int] = {}
    if _has_block(node, "resistance"):
        for k, v_ in _table(node, "resistance",
                            DAMAGE_TYPES, -999).items():
            if v_ != -999:
                parent_resist_overrides[k] = v_
    parent_defense_overrides: Dict[str, int] = {}
    if _has_block(node, "defense"):
        for k, v_ in _table(node, "defense", TERRAINS, -999).items():
            if v_ != -999:
                parent_defense_overrides[k] = int(v_)
    parent_mc_overrides: Dict[str, int] = {}
    if _has_block(node, "movement_costs"):
        for k, v_ in _table(node, "movement_costs",
                            TERRAINS, -999).items():
            if v_ != -999:
                parent_mc_overrides[k] = int(v_)

    out: Dict[str, dict] = {}
    for child in node.children:
        if child.tag != "variation":
            continue
        var_id = child.attrs.get("variation_id", "").strip()
        if not var_id:
            continue
        # Start from a deep-ish copy of parent and overlay variation
        # data. Tables (defense/resistance/movement_costs) merge by
        # key — explicit entries replace, others stick.
        v = {
            "id":               parent_stats["id"],
            "variation_id":     var_id,
            "race":             parent_stats.get("race", ""),
            "alignment":        parent_stats.get("alignment", "neutral"),
            "level":            parent_stats.get("level", 0),
            "cost":             parent_stats.get("cost", 0),
            "hitpoints":        int(child.attrs.get(
                                    "hitpoints",
                                    parent_stats.get("hitpoints", 1))),
            "experience":       parent_stats.get("experience", 30),
            "moves":            int(child.attrs.get(
                                    "movement",
                                    parent_stats.get("moves", 0))),
            "vision":           int(child.attrs.get(
                                    "vision",
                                    parent_stats.get("vision",
                                                     parent_stats.get("moves", 5)))),
            "advances_to":      list(parent_stats.get("advances_to", [])),
            "movement_type":    child.attrs.get(
                                    "movement_type",
                                    parent_stats.get("movement_type", "")),
            "abilities":        list(parent_stats.get("abilities", [])),
            "usage":            parent_stats.get("usage", ""),
            "traits":           parent_stats.get("traits", {}),
            "n_genders":        parent_stats.get("n_genders", 1),
            "undead_variation": parent_stats.get("undead_variation", ""),
        }
        # Movement-type-driven defaults for the new movetype.
        new_mt = move_types.get(v["movement_type"], {})
        defense = dict(new_mt.get("defense",
                                  parent_stats.get("defense", {})))
        resistance = dict(new_mt.get("resistance",
                                     parent_stats.get("resistance", {})))
        movement_costs = dict(new_mt.get("movement_costs",
                                         parent_stats.get("movement_costs", {})))
        # Layer parent unit's explicit overrides on top of the new
        # movetype's defaults — these survive an `inherit=yes`
        # variation that switches movetypes.
        for k, val in parent_resist_overrides.items():
            resistance[k] = val
        for k, val in parent_defense_overrides.items():
            defense[k] = val
        for k, val in parent_mc_overrides.items():
            movement_costs[k] = val
        # Variation's own [defense] / [resistance] / [movement_costs]
        # overlay on top of the chosen movetype's defaults.
        if _has_block(child, "defense"):
            for k, val in _table(child, "defense", TERRAINS, -999).items():
                if val != -999:
                    defense[k] = val
        # Preserve negative-as-cap signs (see movetype scraper
        # comment for full rationale).
        defense = {k: int(v_) for k, v_ in defense.items()}
        if _has_block(child, "resistance"):
            for k, val in _table(child, "resistance",
                                 DAMAGE_TYPES, -999).items():
                if val != -999:
                    resistance[k] = val
        # Variation-level [movement_costs] override (e.g. Walking
        # Corpse:bat overrides cave/fungus to 1, deep_water to 1
        # — making bat-corpses faster on those than the base `fly`
        # movetype). Without this override, our DB used the base
        # movetype's costs and rejected legitimate moves through
        # mushroom groves on Hornshark Island as "mp_insufficient".
        if _has_block(child, "movement_costs"):
            for k, val in _table(child, "movement_costs",
                                 TERRAINS, -999).items():
                if val != -999:
                    movement_costs[k] = int(val)
        v["defense"] = defense
        v["resistance"] = resistance
        v["movement_costs"] = movement_costs
        # Attacks: variation [attack] blocks REPLACE parent's attacks.
        # If no [attack] child, fall through to parent's attacks.
        v_attacks = extract_attacks(child)   # raw_text not used for variation specials
        v["attacks"] = v_attacks if v_attacks else list(parent_stats.get("attacks", []))
        out[var_id] = v
    return out


def extract_unit(node: Node, move_types: Dict[str, dict],
                 races: Dict[str, dict], raw_text: str) -> Optional[dict]:
    if "id" not in node.attrs:
        return None
    mt_name = node.attrs.get("movement_type", "")
    mt = move_types.get(mt_name, {})

    defense = dict(mt.get("defense", {k: 100 for k in TERRAINS}))
    if _has_block(node, "defense"):
        for k, v in _table(node, "defense", TERRAINS, -999).items():
            if v != -999:
                defense[k] = v
    # Wesnoth's [defense] uses negative values to indicate a defense
    # CAP/FLOOR on def_pct (movetype.cpp::config_to_min/config_to_max).
    # The engine stores both `min_` (cap floor on def_pct) and `max_`
    # (the normal value) per terrain. Effective def_pct on aliased
    # terrain = max(cap_floor, alias_min(positive_values)).
    #
    # Negative `-50` means: when aliasing matches this terrain key,
    # the resulting def_pct is AT LEAST 50 (i.e., the unit is at
    # least 50% to be hit, regardless of base terrain). Positive 60
    # means a normal value with no special cap.
    #
    # Concrete: feral Vampire Bat on Gg^Vc (grass village).
    # Bat flat=40 (lizard movement), village=-50 (feral cap).
    # Alias-min over [flat=40, village=50_abs] = 40.
    # But village's negative sign means floor of 50 -> def_pct = 50.
    # Without preserving the sign we'd return 40 = wrong CTH.
    #
    # We preserve negative values verbatim. The terrain_resolver
    # checks the sign during def_pct lookup and applies the floor.
    # `defense` keeps signs; positive: normal, negative: cap floor.
    defense = {k: int(v) for k, v in defense.items()}

    resistance = dict(mt.get("resistance", {k: 100 for k in DAMAGE_TYPES}))
    if _has_block(node, "resistance"):
        for k, v in _table(node, "resistance", DAMAGE_TYPES, -999).items():
            if v != -999:
                resistance[k] = v

    # Per-unit [movement_costs] overrides layered on top of the
    # movetype's defaults. Several units tweak specific terrain costs
    # without changing movement_type entirely. Examples in 1.18.4:
    #   Poacher       (smallfoot)    swamp_water=2  (vs default 3)
    #   Trapper       (smallfoot)    swamp_water=1
    #   Ranger        (smallfoot)    forest=1, swamp_water=2
    #   Huntsman      (smallfoot)    forest=1, swamp_water=2
    #   Footpad       (smallfoot)    swamp_water=2
    # Without this merge, Poacher's diff_replay move on swamp goes
    # cost=3 + ford=1 + forest=2 = 6 over a 5 MP budget and our sim
    # rejects a move Wesnoth accepted (replays_dataset/0f056150f29d
    # cmd 63 turn 4). Mirrors the [defense] handling above.
    movement_costs = dict(mt.get("movement_costs", {k: 99 for k in TERRAINS}))
    if _has_block(node, "movement_costs"):
        for k, v in _table(node, "movement_costs", TERRAINS, -999).items():
            if v != -999:
                movement_costs[k] = v

    race_id = node.attrs.get("race", "")
    race_data = races.get(race_id, {"num_traits": 2,
                                    "ignore_global_traits": False,
                                    "trait_macros": []})
    unit_macros = _scan_unit_trait_macros(node, raw_text)
    # Wesnoth: unit-type level `ignore_race_traits=yes` clears the
    # global+race pool, leaving only the unit's own [trait] children.
    # 1.18.4 uses this on Dark Adept, Horse_Black (Bay Horse),
    # Horse_Dark (Black Horse). See types.cpp:346-347.
    ignore_race_traits = (
        (node.attrs.get("ignore_race_traits", "no") or "no")
        .strip().lower() in ("yes", "true", "1")
    )
    alignment_attr = (node.attrs.get("alignment", "neutral") or "neutral")
    trait_info = build_trait_info(
        race_data, unit_macros,
        ignore_race_traits=ignore_race_traits,
        alignment=alignment_attr,
    )

    # Genders count matters for the synced-RNG accounting at recruit
    # time: `unit::init` calls `generate_gender(...)` before traits and
    # consumes ONE random call from the synced pool when the unit-type
    # has multiple genders (gender=male,female). Single-gender units
    # consume zero. This shifts which random outputs become trait picks.
    gender_attr = (node.attrs.get("gender") or "").strip()
    if gender_attr:
        n_genders = len([g for g in gender_attr.split(",") if g.strip()])
    else:
        n_genders = 1   # `gender` attr absent → male-only (Wesnoth default)

    return {
        "id":         node.attrs["id"],
        "race":       race_id,
        "alignment":  node.attrs.get("alignment", "neutral"),
        "level":      int(node.attrs.get("level", 1) or 1),
        "cost":       int(node.attrs.get("cost", 0) or 0),
        "hitpoints":  int(node.attrs.get("hitpoints", 1) or 1),
        "experience": int(node.attrs.get("experience", 30) or 30),
        "moves":      int(node.attrs.get("movement", 0) or 0),
        # `vision` defaults to `movement` per Wesnoth's unit_type ctor;
        # only a handful of units (Falconer, Dragonfly, Sky Hunter,
        # Dragonfly Grand) override it. We mirror that default here so
        # downstream fog-of-war logic doesn't have to special-case the
        # missing attr.
        "vision":     int(
            node.attrs.get("vision",
                           node.attrs.get("movement", 5)) or 5
        ),
        # `undead_variation` controls plague reanimation. The killed
        # unit's value is applied as a `[variation]` effect to the new
        # Walking Corpse (e.g., "mounted" → Mounted Walking Corpse).
        # Setting it to "null" makes the unit unplagueable (Mudcrawler,
        # Jinn). Default "" inherits from the race or, ultimately, no
        # variation (plain Walking Corpse).
        "undead_variation": (
            node.attrs.get("undead_variation", "") or ""
        ).strip(),
        "advances_to": [
            x.strip() for x in (node.attrs.get("advances_to", "") or "").split(",")
            if x.strip() and x.strip().lower() != "null"
        ],
        "movement_type":  mt_name,
        "defense":        defense,
        "resistance":     resistance,
        "movement_costs": movement_costs,
        "attacks":        extract_attacks(node, raw_text),
        "abilities":     extract_abilities(node, raw_text),
        "usage":         node.attrs.get("usage", ""),
        "traits":        trait_info,
        "n_genders":     n_genders,
    }


def scrape(repo: Path, out_path: Path) -> None:
    macros = load_macro_defs(repo / "data" / "core" / "macros" / "movetypes.cfg")
    print(f"Loaded {len(macros)} macros from movetypes.cfg")

    move_types = extract_movetypes(
        repo / "data" / "core" / "units.cfg", macros,
    )
    print(f"Found {len(move_types)} movement types")

    races = extract_races(
        repo / "data" / "core" / "units.cfg", macros,
    )
    print(f"Found {len(races)} races")
    # Sanity: drakefly should have resistance fields populated.
    df = move_types.get("drakefly", {})
    if df.get("resistance", {}).get("blade") == 100:
        print("  WARNING: drakefly resistance still defaults — macro lookup may have missed.")
    else:
        print(f"  drakefly resistance: {df.get('resistance')}")

    units_dir = repo / "data" / "core" / "units"
    units: Dict[str, dict] = {}
    cfg_files = sorted(units_dir.rglob("*.cfg"))
    for p in cfg_files:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Some unit cfgs (Corpse_Walking, Corpse_Soulless, Drake_Burner)
        # `#define` macros at file-top that the unit_type block then
        # invokes — e.g., UNIT_BODY_WALKING_CORPSE_STATS sets
        # hitpoints, movement_type, movement. Without harvesting them,
        # variation extraction sees hp=1 / mt= / moves=0.
        local_macros = dict(macros)
        local_macros.update(_scan_macro_defs(text))
        root = parse_lines(text.splitlines(), inline_macros=local_macros)

        def collect_units(n: Node):
            if n.tag == "unit_type":
                u = extract_unit(n, move_types, races, text)
                if u and u["id"]:
                    units[u["id"]] = u
                    # Walking Corpse / Soulless variants: store under
                    # composite keys like "Walking Corpse:mounted".
                    # Plague spawns look these up by killed unit's
                    # `undead_variation`. Stats are computed from the
                    # variation's overrides on top of the parent.
                    variations = extract_variations(n, u, move_types)
                    for var_id, var_stats in variations.items():
                        units[f"{u['id']}:{var_id}"] = var_stats
            for c in n.children:
                collect_units(c)
        collect_units(root)

    print(f"Extracted {len(units)} unit types (incl. variations)")
    out = {
        "movement_types": move_types,
        "races":         races,
        "units":         units,
    }
    out_path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size//1024} KB)")


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path)
    ap.add_argument("out",  type=Path)
    args = ap.parse_args(argv[1:])
    scrape(args.repo, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
