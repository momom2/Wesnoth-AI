"""Scrape `wesnoth_src/data/core/terrain.cfg` into a JSON terrain DB.

Output: `terrain_db.json` at the project root, keyed by terrain code
(e.g. `Wwf`, `^Vhs`). Each entry carries the data needed to resolve
movement / defense costs against a `movement_type`'s costs table at
runtime, mirroring Wesnoth's actual semantics.

Why this exists
---------------
Hand-rolled overlay tables miss cases. Wesnoth has 280+ terrain
entries with different aliasing patterns, and movement / defense
resolution depends on per-entry `aliasof` / `mvt_alias` lists with
PLUS/MINUS markers (the markers control whether to prefer the lowest
or highest cost across aliases). Pre-2026-05 our sim hand-rolled
~30 overlay codes and got wrong answers on every other case, which
poisoned every game's movement validation.

This scraper produces an exhaustive DB. The runtime resolver
(`tools/terrain_resolver.py`) consumes it.

Wesnoth source paths
--------------------
- `wesnoth_src/src/terrain/terrain.cpp:130-170` — terrain_type ctor
  parses `aliasof` / `mvt_alias` / `def_alias`. If `mvt_alias` is
  absent, `mvt_type_` falls back to `aliasof`. If both absent,
  `mvt_type_=[self]` (terminal / indivisible).
- `wesnoth_src/src/terrain/terrain.cpp:208-244` — composite terrain
  ctor (base + overlay): starts with `overlay.mvt_type_`, then
  merges `base.mvt_type_` in via `merge_alias_lists`.
- `wesnoth_src/src/terrain/terrain.cpp:334-377` — `merge_alias_lists`
  walks the overlay's list, finds the `BASE` (`_bas`) marker, and
  splices in the base's list.
- `wesnoth_src/src/movetype.cpp:276-369` — `terrain_info::data::
  calc_value` walks the alias list with PLUS/MINUS markers,
  recursively resolving each terrain code, and aggregates
  according to `prefer_high`.

CLI
---
    python tools/scrape_terrain.py
        # writes terrain_db.json at the project root

The scraper parses terrain.cfg with a tiny WML parser tuned for the
[terrain_type] block shape — the file is structurally simple
(no nested macros within blocks), so we don't need the full WML
machinery.

Dependencies: stdlib only (json, re, pathlib)
Dependents: tools.terrain_resolver (consumes terrain_db.json)
"""
from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


log = logging.getLogger("scrape_terrain")


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TERRAIN_CFG = _PROJECT_ROOT / "wesnoth_src" / "data" / "core" / "terrain.cfg"
_OUT_PATH = _PROJECT_ROOT / "terrain_db.json"


# Marker tokens. terrain.cfg uses these as elements within an
# alias list. Wesnoth's translation.cpp encodes them as special
# t_translation values; we keep them as Python strings.
MARKER_PLUS = "+"
MARKER_MINUS = "-"
MARKER_BASE = "_bas"   # placeholder for the base terrain in an overlay's alias list


# ---------------------------------------------------------------------
# WML parsing
# ---------------------------------------------------------------------

# The terrain.cfg file is one big sequence of [terrain_type] ... [/terrain_type]
# blocks. Within a block, attributes are `key=value` lines (key is a
# bare identifier; value runs to the end of the line). The block has no
# nested children we care about. Comments start with `#` (line-level).
#
# The values we need are simple strings: `string`, `id`, `aliasof`,
# `mvt_alias`, `def_alias`, `default_base`, `hidden`. Lists of terrain
# codes are comma-separated; we split on commas and trim whitespace.

_BLOCK_RE = re.compile(r"\[terrain_type\](.*?)\[/terrain_type\]", re.DOTALL)
_COMMENT_RE = re.compile(r"^\s*#.*$", re.MULTILINE)


def _strip_comments(text: str) -> str:
    return _COMMENT_RE.sub("", text)


def _parse_attrs(block_text: str) -> Dict[str, str]:
    """Pull `key=value` pairs from a `[terrain_type]` block. Stops at
    the next `[` (we don't expect [child] tags in terrain.cfg's
    [terrain_type] blocks, but defensively skip if encountered).

    NOTE on the `_` prefix: in WML, `_ "..."` marks a translation
    string for *user-facing* fields like `name=` / `description=`.
    For the structural fields we care about (`string`, `id`,
    `aliasof`, `mvt_alias`, `def_alias`, `default_base`), the value
    `_bas` is a LITERAL token meaning "base terrain placeholder" --
    NOT a translation marker. We must NOT strip the leading
    underscore. The early version of this parser stripped `_` from
    every value, which mangled `aliasof=_bas, Vt` into `aliasof=bas,
    Vt`, breaking every overlay's runtime alias resolution.
    """
    attrs: Dict[str, str] = {}
    for line in block_text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("[") and s.startswith("[/"):
            continue
        if "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        val = val.strip()
        # Strip trailing `# ...` comments. terrain.cfg has lines
        # like `string=Gt       # wmllint: ignore` where the `#`
        # comment is mid-line. Without this, `Gt` becomes
        # `Gt       # wmllint: ignore` and lookups for the abstract
        # terrain code `Gt` fail -- which silently breaks every
        # alias resolution (Gg, Gs, Re, Wwf, etc. all alias to Gt).
        # WML's quoted strings can contain `#` legally, so the strip
        # is bounded to UNQUOTED `#`. Our values aren't quoted at
        # this point (we strip quotes below), so a simple split is
        # safe enough for terrain.cfg.
        if "#" in val:
            val = val.split("#", 1)[0].rstrip()
        # Strip the translation marker `_` ONLY for user-facing
        # fields where it's a real WML idiom (`_ "Forest"`). For
        # structural alias fields, `_bas` is a literal token and
        # leading `_` must be preserved.
        if key in ("name", "editor_name", "description", "help_topic_text"):
            if val.startswith("_"):
                val = val[1:].strip()
        if val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
        attrs[key] = val
    return attrs


def _parse_alias_list(raw: str) -> List[str]:
    """Parse an alias list value like ``-, _bas, Ft`` or ``Gt, Wst``
    into a list of tokens preserving PLUS/MINUS/BASE markers and
    bare terrain codes.

    Wesnoth's translation.cpp normalizes ``-`` to MINUS, ``+`` to
    PLUS, and ``_bas`` to BASE in the parsed ter_list. We preserve
    them verbatim as Python strings so the merge / value algorithms
    can treat them uniformly.
    """
    if not raw:
        return []
    out: List[str] = []
    for tok in raw.split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(t)
    return out


@dataclass
class TerrainEntry:
    """One [terrain_type] block, normalized.

    `mvt_type` and `def_type` are the Wesnoth-style alias lists used
    to RESOLVE movement / defense at runtime. They're computed per
    terrain.cpp:130-170:

        mvt_type_ = mvt_alias if present else aliasof if present else [string]
        def_type_ = def_alias if present else aliasof if present else [string]

    For terminal terrains (no aliasof, no mvt_alias), the list is
    `[string]` — `is_indivisible` returns true at runtime, the
    movement_costs / defense table for `id` is consulted directly.
    """
    string: str                              # e.g. "Wwf" or "^Vhs"
    id: str = ""                             # e.g. "ford"
    aliasof: List[str] = field(default_factory=list)
    mvt_alias: List[str] = field(default_factory=list)
    def_alias: List[str] = field(default_factory=list)
    default_base: str = ""
    hidden: bool = False
    is_overlay: bool = False                 # `string` starts with "^"
    # `heals=N` on the [terrain_type]: amount the hex heals a unit
    # parked on it per init_side. Villages all use 8; the oasis
    # overlay (^Do) is heals=8 and behaves like a village for
    # healing/poison-cure purposes (per heal.cpp::poison_progress
    # and heal_amount, both of which call map().gives_healing).
    # Critically, oasis is NOT a village (no capture, no income).
    # Default 0 = no terrain-based healing.
    heals: int = 0
    # `light=N` on the [terrain_type]: hex-level illumination bonus
    # added to the base ToD lawful_bonus. Per terrain.hpp:132
    # `light_bonus(base) = bounded_add(base, light_modification_,
    # max_light_, min_light_)`. The campfire overlay (^Ecf) has
    # light=25 with max_light/min_light defaulting to 25, which
    # CLAMPS lawful_bonus to exactly 25 on that hex regardless of
    # base ToD. Used in tod_manager::get_illuminated_time_of_day
    # before applying unit illumination effects.
    # Composite (base+overlay): light = base.light + overlay.light,
    # max_light = max(base.max_light, overlay.max_light),
    # min_light = min(base.min_light, overlay.min_light).
    # Default 0/0/0.
    light: int = 0
    max_light: int = 0
    min_light: int = 0
    # True when the terrain.cfg explicitly set max_light or min_light
    # (vs taking the default = light value). Composition needs to
    # know whether to use the inherited light or the explicit cap.
    has_max_light: bool = False
    has_min_light: bool = False

    @property
    def mvt_type(self) -> List[str]:
        """The list value() walks for MOVEMENT cost resolution."""
        if self.mvt_alias:
            return list(self.mvt_alias)
        if self.aliasof:
            return list(self.aliasof)
        return [self.string]

    @property
    def def_type(self) -> List[str]:
        """The list value() walks for DEFENSE resolution."""
        if self.def_alias:
            return list(self.def_alias)
        if self.aliasof:
            return list(self.aliasof)
        return [self.string]

    def to_dict(self) -> dict:
        return {
            "string":        self.string,
            "id":            self.id,
            "aliasof":       self.aliasof,
            "mvt_alias":     self.mvt_alias,
            "def_alias":     self.def_alias,
            "default_base":  self.default_base,
            "hidden":        self.hidden,
            "is_overlay":    self.is_overlay,
            "mvt_type":      self.mvt_type,
            "def_type":      self.def_type,
            "heals":         self.heals,
            "light":         self.light,
            "max_light":     self.max_light,
            "min_light":     self.min_light,
        }


def parse_terrain_cfg(path: Path = _TERRAIN_CFG) -> Dict[str, TerrainEntry]:
    """Walk every `[terrain_type]` block; return `{string: TerrainEntry}`.

    Multiple blocks can share a `string` (very rare, but defensively
    we keep the LAST occurrence — matches Wesnoth's load order
    where later definitions override earlier ones).
    """
    if not path.exists():
        raise FileNotFoundError(f"terrain.cfg not found: {path}")
    text = _strip_comments(path.read_text(encoding="utf-8", errors="replace"))

    entries: Dict[str, TerrainEntry] = {}
    for m in _BLOCK_RE.finditer(text):
        attrs = _parse_attrs(m.group(1))
        s = attrs.get("string", "").strip()
        if not s:
            continue   # malformed; skip
        try:
            heals_v = int(attrs.get("heals", "0").strip() or "0")
        except (TypeError, ValueError):
            heals_v = 0
        def _int_or(default):
            def f(s):
                try: return int(s.strip())
                except (TypeError, ValueError, AttributeError): return default
            return f
        light_v = _int_or(0)(attrs.get("light", "0"))
        # max_light / min_light default to the light value when absent.
        has_max = "max_light" in attrs
        has_min = "min_light" in attrs
        max_light_v = _int_or(light_v)(attrs.get("max_light", str(light_v))) if has_max else light_v
        min_light_v = _int_or(light_v)(attrs.get("min_light", str(light_v))) if has_min else light_v
        entry = TerrainEntry(
            string=s,
            id=attrs.get("id", "").strip(),
            aliasof=_parse_alias_list(attrs.get("aliasof", "")),
            mvt_alias=_parse_alias_list(attrs.get("mvt_alias", "")),
            def_alias=_parse_alias_list(attrs.get("def_alias", "")),
            default_base=attrs.get("default_base", "").strip(),
            hidden=attrs.get("hidden", "").strip().lower() in ("yes", "true", "1"),
            is_overlay=s.startswith("^"),
            heals=heals_v,
            light=light_v,
            max_light=max_light_v,
            min_light=min_light_v,
            has_max_light=has_max,
            has_min_light=has_min,
        )
        entries[s] = entry
    return entries


# ---------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------

# All ids that should appear as movement_costs keys (the union across
# the default-era movetypes). If the scraper finds a terrain whose `id`
# doesn't match anything in this set AND has no aliasof / mvt_alias,
# we'd never be able to resolve a cost for it at runtime — flag.
_KNOWN_MOVEMENT_IDS = {
    "castle", "cave", "deep_water", "flat", "forest", "frozen",
    "fungus", "hills", "impassable", "mountains", "reef", "sand",
    "shallow_water", "swamp_water", "unwalkable", "village",
}


def _audit(entries: Dict[str, TerrainEntry]) -> Dict[str, int]:
    """Quick QA on the parsed DB. Returns a counts dict for the
    summary log line and prints warnings."""
    counts = {
        "total":           len(entries),
        "overlays":        sum(1 for e in entries.values() if e.is_overlay),
        "terminal":        0,
        "aliased":         0,
        "unresolvable_id": 0,
    }
    for e in entries.values():
        if not e.aliasof and not e.mvt_alias:
            counts["terminal"] += 1
            if e.id and e.id not in _KNOWN_MOVEMENT_IDS:
                # Terminal terrain with an id that isn't in any known
                # movetype's movement_costs — likely fine (e.g. some
                # campaign-era terrains, or `_off` placeholder), but
                # log so we can audit later.
                counts["unresolvable_id"] += 1
                log.debug(f"terminal terrain {e.string!r} has id"
                          f"={e.id!r} not in known movement_costs keys")
        else:
            counts["aliased"] += 1
    return counts


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main(argv: List[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    entries = parse_terrain_cfg()
    counts = _audit(entries)
    out = {code: e.to_dict() for code, e in entries.items()}
    _OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log.info(
        f"wrote {_OUT_PATH.relative_to(_PROJECT_ROOT)}: "
        f"{counts['total']} entries "
        f"({counts['overlays']} overlays, "
        f"{counts['terminal']} terminal, "
        f"{counts['aliased']} aliased; "
        f"{counts['unresolvable_id']} terminal w/ unknown id)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
