"""Lightweight filter pass over downloaded Wesnoth 1.18 replays.

Decompresses each .bz2 and walks it line-by-line looking for:
  - era_id / campaign_type / mp_game_title / version at file header,
  - [side] faction + controller + user_team_name,
  - [scenario] id / [multiplayer] scenario header,
  - active_mods (to spot nontrivial modifications we'd want to skip).

We DON'T parse the whole replay (that's Phase B1). This is just a
go/no-go count: how many Default-era 2p Knalgan-vs-Drake games exist in
the sample? If it's sizeable, Phase B1 is worth the engineering.

Usage:
    python tools/filter_replays.py replays_raw
"""
from __future__ import annotations

import bz2
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# Patterns to find at file header (before heavy [replay] content).
KEY_RE = re.compile(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"?([^"\r\n]*)"?\s*$')
TAG_OPEN_RE  = re.compile(r'^\s*\[([a-zA-Z_][a-zA-Z0-9_]*)\]\s*$')
TAG_CLOSE_RE = re.compile(r'^\s*\[/([a-zA-Z_][a-zA-Z0-9_]*)\]\s*$')


def parse_header(raw: bytes, max_lines: int = 200_000) -> dict:
    """Extract the fields we care about from the head of a replay.

    1.18 replay files wrap the starting snapshot in either a `[scenario]`
    or a `[replay_start]` block (the latter for mid-game replays
    uploaded from the public server). Both nest a `[side]` per player.
    We accept either container and only collect `[side]` blocks when
    the tag-stack bottom is one of those snapshot containers — this
    keeps us from picking up `[ai]`-nested or `[event]`-nested side
    references that aren't actual player sides.
    """
    out = {
        "top": {},
        "sides": [],
        "scenario_id": None,
        "scenario_name": None,
        "map_data_present": False,
        "truncated": False,
    }
    snapshot_containers = {"scenario", "replay_start", "snapshot"}
    stack: list[str] = []
    current_side: dict | None = None

    text = raw.decode("utf-8", errors="replace")
    for i, line in enumerate(text.splitlines()):
        if i >= max_lines:
            out["truncated"] = True
            break

        m = TAG_OPEN_RE.match(line)
        if m:
            tag = m.group(1)
            stack.append(tag)
            # A player side is a direct-ish descendant of a snapshot
            # container (may be nested one level deep, never more).
            if tag == "side" and any(s in snapshot_containers for s in stack[:-1]):
                current_side = {}
            continue

        m = TAG_CLOSE_RE.match(line)
        if m:
            tag = m.group(1)
            if stack and stack[-1] == tag:
                stack.pop()
            if tag == "side" and current_side is not None:
                # Only keep sides that carry a faction (i.e., a real
                # player side, not an `[ai][side]...` reference).
                if "faction" in current_side or "user_team_name" in current_side:
                    out["sides"].append(current_side)
                current_side = None
            continue

        m = KEY_RE.match(line)
        if not m:
            continue
        k, v = m.group(1), m.group(2)

        if not stack:
            out["top"][k] = v
        elif stack[-1] in ("scenario", "replay_start", "snapshot"):
            if k == "id":
                out["scenario_id"] = out["scenario_id"] or v
            elif k == "name" and out["scenario_name"] is None:
                out["scenario_name"] = v
            elif k == "map_data":
                out["map_data_present"] = True
        elif current_side is not None and stack[-1] == "side":
            if k in ("faction", "controller", "user_team_name",
                     "team_name", "side", "type", "recruit", "save_id"):
                current_side[k] = v

    return out


def classify(info: dict) -> dict:
    top = info["top"]
    sides = info["sides"]
    era  = (top.get("era_id") or "").strip().lower()
    mods = (top.get("active_mods") or "").strip()
    campaign_type = (top.get("campaign_type") or "").strip()
    version = top.get("version", "?")

    # Normalize era — 1.18 default uses era_id "default"; some games
    # label it "era_default" or "Default". Accept any of those as
    # "default". Rest is considered "other".
    if era in ("default", "era_default", "default era"):
        era_kind = "default"
    elif era == "":
        era_kind = "missing"
    else:
        era_kind = era  # keep raw for diagnostics

    n_sides = len(sides)
    factions = [s.get("faction", "") for s in sides]
    # Faction strings observed: "Knalgan Alliance", "Drakes", "Loyalists",
    # "Rebels", "Northerners", "Undead", or "Random" / empty.
    knalgan_drake = (n_sides == 2 and sorted(factions) == ["Drakes", "Knalgan Alliance"])

    return {
        "version": version,
        "era_kind": era_kind,
        "campaign_type": campaign_type,
        "mods_present": bool(mods),
        "n_sides": n_sides,
        "factions": tuple(factions),
        "knalgan_drake": knalgan_drake,
        "scenario_id": info.get("scenario_id"),
    }


def scan_dir(root: Path) -> None:
    files = sorted(root.glob("**/*.bz2"))
    print(f"Scanning {len(files)} .bz2 files under {root}\n")

    # Counters.
    versions      = Counter()
    eras          = Counter()
    mod_flag      = Counter()  # True/False
    n_sides_dist  = Counter()
    faction_dist  = Counter()
    kd_games      = []
    default_2p    = []
    default_any   = []
    scenarios_kd  = Counter()

    errors = 0
    for fi, path in enumerate(files, 1):
        try:
            with bz2.open(path, "rb") as f:
                # Read up to 4 MB uncompressed — enough to clear the
                # scenario header + all [side] blocks even when map_data
                # is a large inline tile matrix. Typical replay files
                # are 0.5-5 MB uncompressed, and [side] blocks appear
                # early; the big [replay] tail that follows is what
                # actually bloats file size.
                raw = f.read(4 * 1024 * 1024)
        except OSError:
            errors += 1; continue
        except Exception:
            errors += 1; continue

        try:
            info = parse_header(raw)
            clf = classify(info)
        except Exception:
            errors += 1; continue

        versions[clf["version"]] += 1
        eras[clf["era_kind"]] += 1
        mod_flag[clf["mods_present"]] += 1
        n_sides_dist[clf["n_sides"]] += 1
        for f_ in clf["factions"]:
            faction_dist[f_] += 1

        if (clf["era_kind"] == "default" and clf["n_sides"] == 2
                and clf["campaign_type"] == "multiplayer"):
            default_2p.append((str(path), clf))
            if clf["knalgan_drake"]:
                kd_games.append((str(path), clf))
                scenarios_kd[clf["scenario_id"] or "?"] += 1

        if clf["era_kind"] == "default":
            default_any.append(clf)

        if fi % 500 == 0:
            print(f"  [{fi}/{len(files)}] scanned so far", flush=True)

    total = len(files)
    print("\n=== Top-line counts ===")
    print(f"  Total .bz2 replays scanned: {total}")
    print(f"  Parse errors: {errors}")

    print("\n=== Wesnoth versions seen ===")
    for v, c in versions.most_common(8):
        print(f"  {v!r:20s} {c}")

    print("\n=== Era mix ===")
    for e, c in eras.most_common(10):
        print(f"  {e!r:30s} {c}")

    print("\n=== Campaign type ===")
    campaign_ct = Counter();
    # Gather in a re-walk would be wasteful; just skip — eras + n_sides already tell most.

    print("\n=== Side count distribution ===")
    for n, c in sorted(n_sides_dist.items()):
        print(f"  {n} sides: {c}")

    print("\n=== Faction appearances (across all [side] blocks) ===")
    for f, c in faction_dist.most_common(12):
        print(f"  {f!r:20s} {c}")

    print("\n=== Default-era 2p MP games ===")
    print(f"  Count: {len(default_2p)}")
    matchups = Counter()
    scenarios_2p = Counter()
    for _, clf in default_2p:
        a, b = sorted(clf["factions"])
        matchups[(a, b)] += 1
        scenarios_2p[clf["scenario_id"] or "?"] += 1
    print(f"  Matchup distribution (top 15):")
    for (a, b), c in matchups.most_common(15):
        print(f"    {a!r:25s} vs {b!r:25s} {c}")
    print(f"  Scenario distribution (top 15):")
    for s, c in scenarios_2p.most_common(15):
        print(f"    {s!r:45s} {c}")

    print("\n=== Default-era 2p KNALGAN vs DRAKES ===")
    print(f"  Count: {len(kd_games)}")
    print(f"  Scenario id distribution:")
    for s, c in scenarios_kd.most_common(10):
        print(f"    {s!r:40s} {c}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: filter_replays.py REPLAYS_DIR"); sys.exit(2)
    scan_dir(Path(sys.argv[1]))
