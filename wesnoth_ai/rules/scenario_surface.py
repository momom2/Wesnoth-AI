#!/usr/bin/env python3
"""Every tag and attribute a pool scenario declares, classified.

Detector 2 of docs/scenario_build_plan_20260922.md. The expansion diff
(`expansion_diff.py`) answers "did we expand this scenario the way the
game does"; this answers the next question: **of everything the
expansion contains, what do we actually READ?**

A pair is one of:

  MODELLED    -- a reader exists, and is named `file.py:symbol`. The
                 test checks the symbol is really there, so deleting
                 the reader fails here rather than silently changing
                 what gets built.
  IGNORED     -- presentation; nothing downstream reads it, and the
                 entry says why.
  SUBSTITUTED -- real behaviour the sim produces some other way, or a
                 value generation deliberately chooses for itself.
  UNKNOWN     -- nobody has said. This is the bug state.

Why the reader binding rather than a list of known names: every defect
this plan was written for -- village gold, per-side fog, `[hides] id=`
-- was an ORDINARY, known attribute sitting in the tree with nothing
reading it. A classifier over names and values calls all three fine. A
classifier that demands a named reader calls all three UNKNOWN.

    python tools/analysis/scenario_surface.py
    python tools/analysis/scenario_surface.py --write-manifest
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import sys
from typing import Dict, List, Optional, Set, Tuple

from wesnoth_ai.paths import REPO_ROOT
from wesnoth_ai.rules.expansion_diff import POOL, _scenario_block
from wesnoth_ai.rules.scenario_cfg import load_scenario_wml

MANIFEST = REPO_ROOT / "tests" / "data" / "scenario_surface.json"
CLASSES = ("MODELLED", "IGNORED", "SUBSTITUTED")


# Every scenario whose expansion reaches a game we build or rebuild.
# Generation reads the pool; RECONSTRUCTION also loads, through the
# same expander, the corpus maps outside it -- Cynsaun Battlefield and
# Hornshark Island (582 games between them) and around_mini. The first
# version of this manifest covered the pool only, so it reported
# 0 UNKNOWN while Hornshark's preplaced units -- whose [heals] our
# expander had been emptying -- sat outside it entirely.
CORPUS_SCENARIOS = list(POOL) + ["multiplayer_Cynsaun_Battlefield",
                                 "multiplayer_Hornshark_Island",
                                 "around_mini"]

# Control flow is not a semantic position. `_apply_action` dispatches a
# [unit] inside [switch][case] or [if][then] with the same handler as a
# top-level one, so it must read the same manifest entry; keying on the
# raw nesting made every wrapped action a fresh, unclassified path. The
# wrappers' OWN attributes (`switch.variable`, `case.value`) are still
# recorded, under the wrapper's name.
CONTROL_FLOW = frozenset({"switch", "case", "else", "then", "do", "command",
                          "if", "foreach"})


def surface(scenarios=CORPUS_SCENARIOS) -> Dict[str, Dict[str, Set[str]]]:
    """{tag path: {attribute: {scenario ids}}} over the expansions
    generation and reconstruction actually read, control flow folded
    out of the path."""
    out: Dict[str, Dict[str, Set[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(set))

    def walk(node, prefix, scenario_id):
        for child in node.children:
            path = f"{prefix}/{child.tag}"
            out[path]                       # a tag with no attributes still counts
            for key in child.attrs:
                out[path][key].add(scenario_id)
            # A wrapper's children keep the wrapper's own position.
            walk(child, prefix if child.tag in CONTROL_FLOW else path,
                 scenario_id)

    for scenario_id in scenarios:
        block = _scenario_block(load_scenario_wml(scenario_id))
        if block is None:
            continue
        for key in block.attrs:
            out["scenario"][key].add(scenario_id)
        walk(block, "scenario", scenario_id)
    return {k: dict(v) for k, v in out.items()}


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def classify(path: str, attr: str, manifest: dict) -> dict:
    """The pair's entry: an exact pair wins, else the path's default,
    else UNKNOWN."""
    entry = manifest["pairs"].get(f"{path}.{attr}")
    if entry is not None:
        return entry
    entry = manifest["paths"].get(path)
    if entry is not None:
        return entry
    return {"classification": "UNKNOWN"}


def unknowns(found=None, manifest=None) -> List[Tuple[str, str, List[str]]]:
    found = surface() if found is None else found
    manifest = load_manifest() if manifest is None else manifest
    out = []
    for path in sorted(found):
        for attr, scenarios in sorted(found[path].items()):
            if classify(path, attr, manifest)["classification"] == "UNKNOWN":
                out.append((path, attr, sorted(scenarios)))
    return out


def stale(found=None, manifest=None) -> List[str]:
    """Manifest entries for pairs the pool no longer contains: dead
    classifications that would hide a real gap if the shape came back
    differently."""
    found = surface() if found is None else found
    manifest = load_manifest() if manifest is None else manifest
    live = {f"{p}.{a}" for p in found for a in found[p]}
    live_paths = set(found)
    return sorted([k for k in manifest["pairs"] if k not in live]
                  + [k for k in manifest["paths"] if k not in live_paths])


def masking_path_defaults(found=None, manifest=None) -> List[str]:
    """Path defaults that sit above a load-bearing attribute.

    A path default is a shortcut for "everything under here is
    presentation". The moment one attribute under it is MODELLED or
    SUBSTITUTED, the default stops being a shortcut and becomes a
    mask: delete the explicit entry and the load-bearing attribute
    silently inherits IGNORED. That is exactly how `[hides] id=` sat
    unread -- the tag looked like a display block and its id was the
    only thing that mattered.

    So the mixture is refused: enumerate the path instead.
    """
    found = surface() if found is None else found
    manifest = load_manifest() if manifest is None else manifest
    out = []
    for path, entry in sorted(manifest["paths"].items()):
        if entry.get("classification") != "IGNORED":
            continue
        for attr in sorted(found.get(path, {})):
            pair = manifest["pairs"].get(f"{path}.{attr}")
            if pair and pair.get("classification") != "IGNORED":
                out.append(
                    f"{path} defaults to IGNORED but {path}.{attr} is "
                    f"{pair['classification']}: enumerate the path")
    return out


def reader_source(rel: str, symbol: str) -> Optional[str]:
    """The source of a function, class or module-level assignment named
    `symbol` in `rel`, or None when there is no such definition."""
    path = REPO_ROOT / rel
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == symbol:
            return ast.get_source_segment(text, node)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == symbol for t in targets):
                return ast.get_source_segment(text, node)
    return None


def missing_readers(manifest=None) -> List[str]:
    """MODELLED entries whose named reader does not actually read the
    attribute.

    Checking only that the reader EXISTS proved nothing: the first
    version of this manifest named a real function for 22 of its 72
    MODELLED entries and that function never mentioned the attribute.
    `read_side` was bound to `[side] id`, which it does not read;
    `random_traits`, `affect_self` and `cumulative` were bound to
    readers although nothing in the project reads them at all. So the
    reader's own source must name the attribute as a quoted string --
    or the entry says in `generic` why the read happens elsewhere (a
    table the reader walks, a key the reader is called with)."""
    manifest = load_manifest() if manifest is None else manifest
    out = []
    for key, entry in sorted({**manifest["paths"], **manifest["pairs"]}.items()):
        if entry.get("classification") != "MODELLED":
            continue
        reader = entry.get("reader", "")
        if ":" not in reader:
            out.append(f"{key}: no reader named")
            continue
        rel, symbol = reader.split(":", 1)
        source = reader_source(rel, symbol)
        if source is None:
            out.append(f"{key}: {rel} defines no {symbol}")
            continue
        if entry.get("generic", "").strip():
            continue
        attr = key.rsplit(".", 1)[1] if "." in key else ""
        if attr and f'"{attr}"' not in source and f"'{attr}'" not in source:
            out.append(f"{key}: {symbol} never names {attr!r}")
    return out


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--write-manifest", action="store_true",
                    help="add every unclassified pair as UNKNOWN, so it "
                         "cannot be committed without someone deciding")
    args = ap.parse_args(argv)

    found = surface()
    manifest = load_manifest()
    n_pairs = sum(len(v) for v in found.values())
    print(f"{len(found)} tag paths, {n_pairs} path-attribute pairs "
          f"over {len(CORPUS_SCENARIOS)} scenarios")

    counts: collections.Counter = collections.Counter()
    for path in found:
        for attr in found[path]:
            counts[classify(path, attr, manifest)["classification"]] += 1
    print("  " + ", ".join(f"{k} {counts[k]}"
                           for k in ("MODELLED", "IGNORED", "SUBSTITUTED",
                                     "UNKNOWN") if counts[k]))

    bad = unknowns(found, manifest)
    for path, attr, scenarios in bad:
        print(f"  UNKNOWN {path}.{attr}  ({len(scenarios)} scenarios)")
    for line in missing_readers(manifest):
        print(f"  BROKEN READER {line}")
    for line in masking_path_defaults(found, manifest):
        print(f"  MASKING {line}")
    for key in stale(found, manifest):
        print(f"  STALE {key} (classified, not in the pool)")

    if args.write_manifest and bad:
        for path, attr, _ in bad:
            manifest["pairs"][f"{path}.{attr}"] = {
                "classification": "UNKNOWN", "why": ""}
        MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n",
                            encoding="utf-8")
        print(f"\nwrote {len(bad)} UNKNOWN entries to {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
