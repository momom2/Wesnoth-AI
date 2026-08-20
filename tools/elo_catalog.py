"""Catalog of Elo-verified checkpoints (user directive 2026-08-17).

One committed JSON (`training/metrics/elo_catalog.json`) holds every
pairwise Elo measurement this project has made, and derives each
checkpoint's rating by a single global fit over ALL recorded edges,
with one fixed reference at Elo 0. Ground truth is the per-pair
W-D-L records (PURE convention: decisive games only -- user ruling
2026-08-17: a capped game is NOT a draw, it is a truncated
observation recorded on the edge as `no_result` and ignored by the
fit); ratings are always recomputed from them, so a new checkpoint
chains onto the scale the moment it shares a match with any rated
one.

AUTO-UPDATE: `tools/elo_collect.py` calls `update_from_games()` after
every fit (opt out with --no-catalog), keyed by the games-dir name --
re-collecting the same directory REPLACES its edge rather than
double-counting. The catalog updates wherever elo_collect runs; the
committed copy in this repo is canonical, so box-side game dirs
should be pulled and collected here (the existing workflow).

Reference: 2291k, formerly ref_2p29M (seed_20260718.pt,
decision_step 2,290,529) = 0
-- the anchor of the 2026-07-30 preregistered triangle, from which
all current ratings chain.

Labels follow docs/checkpoint_naming.md (lineage-path names, e.g.
`2516k-b-294k-l4-430k`). `rename` migrates a label and records an
alias so old names in game dirs and docs still resolve.

CLI:
    python tools/elo_catalog.py show
    python tools/elo_catalog.py seed-july   (one-time bootstrap)
    python tools/elo_catalog.py add-meta LABEL key=value ...
    python tools/elo_catalog.py rename OLD NEW
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("elo_catalog")

CATALOG_PATH = Path("training/metrics/elo_catalog.json")
CATALOG_VERSION = 1
REFERENCE_LABEL = "2291k"  # docs/checkpoint_naming.md; ex ref_2p29M


def load_catalog(path: Path = CATALOG_PATH) -> Dict:
    if path.exists():
        cat = json.loads(path.read_text(encoding="utf-8"))
        if cat.get("version") != CATALOG_VERSION:
            raise ValueError(f"{path}: unknown catalog version "
                             f"{cat.get('version')}")
        return cat
    return {"version": CATALOG_VERSION,
            "reference": {"label": REFERENCE_LABEL, "elo": 0.0},
            "checkpoints": {}, "edges": {}, "aliases": {}}


def resolve_label(cat: Dict, label: str) -> str:
    """Follow the alias chain (old name -> current name)."""
    aliases = cat.get("aliases", {})
    seen = set()
    while label in aliases and label not in seen:
        seen.add(label)
        label = aliases[label]
    return label


def rename_label(cat: Dict, old: str, new: str) -> None:
    """Rename a checkpoint label everywhere (checkpoints, edges,
    reference) and record `aliases[old] = new`. Existing aliases
    pointing at `old` are re-pointed at `new` (chain compression),
    so every historical name stays one hop from current."""
    if old not in cat["checkpoints"]:
        raise KeyError(f"unknown label {old!r}")
    if new in cat["checkpoints"]:
        raise ValueError(f"label {new!r} already exists")
    cat["checkpoints"][new] = cat["checkpoints"].pop(old)
    for e in cat["edges"].values():
        for k in ("label_a", "label_b"):
            if e[k] == old:
                e[k] = new
    if cat["reference"]["label"] == old:
        cat["reference"]["label"] = new
    aliases = cat.setdefault("aliases", {})
    for k, v in aliases.items():
        if v == old:
            aliases[k] = new
    aliases[old] = new


def save_catalog(cat: Dict, path: Path = CATALOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cat, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")


def record_edge(cat: Dict, source_key: str, label_a: str,
                label_b: str, wins_a: int, draws: int, wins_b: int,
                protocol: Optional[Dict] = None,
                date: Optional[str] = None,
                no_result: int = 0) -> None:
    """Upsert one measured edge. `source_key` (games dir / session
    id) is the idempotency key: re-recording the same source
    replaces, never double-counts. `no_result` records ABSENCES
    (capped/stalled games -- user ruling 2026-08-17: not draws, zero
    rating information; kept for provenance, ignored by refit).
    Labels are alias-resolved so edges recorded under a renamed
    checkpoint's old name chain onto its current node."""
    label_a = resolve_label(cat, label_a)
    label_b = resolve_label(cat, label_b)
    cat["edges"][source_key] = {
        "label_a": label_a, "label_b": label_b,
        "wins_a": int(wins_a), "draws": int(draws),
        "wins_b": int(wins_b),
        "no_result": int(no_result),
        "protocol": protocol or {},
        "date": date or time.strftime("%F"),
    }
    for lab in (label_a, label_b):
        cat["checkpoints"].setdefault(lab, {})


def refit(cat: Dict) -> None:
    """Recompute every checkpoint's rating from ALL edges, reference
    pinned at its stored Elo. Ratings for labels disconnected from
    the reference component are marked unanchored (fit still runs --
    fit_elo handles the graph; their numbers float on the same call
    but only the reference component is meaningful, so we flag)."""
    from tools.elo_ladder import PairRecord, fit_elo

    labels = sorted(cat["checkpoints"])
    if not labels:
        return
    idx = {lab: i for i, lab in enumerate(labels)}
    pairs: Dict[Tuple[int, int], PairRecord] = {}
    adj: Dict[str, set] = {lab: set() for lab in labels}
    for e in cat["edges"].values():
        a, b = idx[e["label_a"]], idx[e["label_b"]]
        i, j = min(a, b), max(a, b)
        rec = pairs.setdefault((i, j), PairRecord())
        a_is_i = a == i
        rec.wins_i += e["wins_a"] if a_is_i else e["wins_b"]
        rec.wins_j += e["wins_b"] if a_is_i else e["wins_a"]
        rec.draws += e["draws"]
        adj[e["label_a"]].add(e["label_b"])
        adj[e["label_b"]].add(e["label_a"])
    ref = cat["reference"]["label"]
    anchor_idx = idx.get(ref, 0)
    elo, se = fit_elo(len(labels), pairs, anchor_idx,
                      anchor_elo=float(cat["reference"].get("elo", 0.0)),
                      prior_games=1.0, draw_weight=0.5)
    # Connected component of the reference (only these ratings chain).
    seen = set()
    stack = [ref] if ref in adj else []
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        stack.extend(adj[u] - seen)
    n_games_of = {lab: 0 for lab in labels}
    for e in cat["edges"].values():
        n = e["wins_a"] + e["draws"] + e["wins_b"]
        n_games_of[e["label_a"]] += n
        n_games_of[e["label_b"]] += n
    for lab in labels:
        meta = cat["checkpoints"][lab]
        meta["elo"] = round(float(elo[idx[lab]]), 1)
        meta["se"] = round(float(se[idx[lab]]), 1)
        meta["n_games"] = n_games_of[lab]
        meta["anchored"] = lab in seen
    cat["last_refit"] = time.strftime("%FT%TZ", time.gmtime())


def update_from_games(games_dir: Path, games: List[dict],
                      protocol: Optional[Dict] = None,
                      path: Path = CATALOG_PATH,
                      label_map: Optional[Dict[str, str]] = None) -> None:
    """The elo_collect hook: aggregate one games dir's PURE W-D-L
    per label pair, upsert (idempotent by dir name + pair), refit,
    save. Multi-pair dirs record one edge per pair. `label_map`
    renames run-local labels to canonical catalog labels (e.g.
    rated_anchor -> new_2p52M) so edges chain to existing nodes."""
    label_map = label_map or {}
    # Tally = [wins_a, genuine_draws, wins_b, no_result]. Under the
    # 2026-08-17 ruling every non-decisive outcome is a no-result
    # absence (there are no draws in real Wesnoth); the draws slot
    # stays for legacy edges and a hypothetical future genuine-draw
    # outcome, and is always 0 from this path.
    tallies: Dict[Tuple[str, str], List[int]] = {}
    for g in games:
        a = label_map.get(g["label_a"], g["label_a"])
        b = label_map.get(g["label_b"], g["label_b"])
        key = (a, b) if a <= b else (b, a)
        t = tallies.setdefault(key, [0, 0, 0, 0])
        out = g["outcome_a"]
        if out == "win":
            t[0 if a <= b else 2] += 1
        elif out == "loss":
            t[2 if a <= b else 0] += 1
        else:
            t[3] += 1
    cat = load_catalog(path)
    for (a, b), (wa, d, wb, nr) in sorted(tallies.items()):
        source_key = f"{Path(games_dir).name}:{a}~{b}"
        record_edge(cat, source_key, a, b, wa, d, wb,
                    protocol=protocol, no_result=nr)
    refit(cat)
    save_catalog(cat, path)
    log.info(f"elo catalog updated ({len(tallies)} edge(s) from "
             f"{Path(games_dir).name}) -> {path}")


def render(cat: Dict) -> str:
    lines = [f"Elo catalog -- reference {cat['reference']['label']} "
             f"= {cat['reference']['elo']:.0f}, last refit "
             f"{cat.get('last_refit', 'never')}"]
    rated = sorted(cat["checkpoints"].items(),
                   key=lambda kv: -(kv[1].get("elo") or 0))
    for lab, m in rated:
        anch = "" if m.get("anchored", True) else "  [UNANCHORED]"
        step = m.get("decision_step", "?")
        lines.append(
            f"  {lab:<16} {m.get('elo', float('nan')):>7.1f} "
            f"± {m.get('se', float('nan')):>5.1f}  "
            f"({m.get('n_games', 0)} games, step {step}){anch}")
    for k, e in sorted(cat["edges"].items()):
        nr = e.get("no_result", 0)
        tail = f" +{nr}nr" if nr else ""
        lines.append(f"    edge {k}: {e['label_a']} "
                     f"{e['wins_a']}-{e['draws']}-{e['wins_b']} "
                     f"{e['label_b']}{tail} ({e['date']})")
    return "\n".join(lines)


def seed_july(path: Path = CATALOG_PATH) -> None:
    """One-time bootstrap from the 2026-07-30 preregistered triangle
    (training/logs/elo_triangle_20260730/POOLED_TRIANGLE.json) and
    its reused edges. Idempotent (fixed source keys)."""
    cat = load_catalog(path)
    edges = [
        ("elo_q_transform_20260729:ref~old", "ref_2p29M", "old_2p40M",
         45, 0, 55, "2026-07-29"),
        ("elo_regress_20260730:old~new", "old_2p40M", "new_2p52M",
         36, 0, 64, "2026-07-30"),
        ("elo_triangle_20260730:ref~new", "ref_2p29M", "new_2p52M",
         30, 0, 70, "2026-07-30"),
        ("elo_triangle_20260730:ref~old", "ref_2p29M", "old_2p40M",
         20, 0, 20, "2026-07-30"),
    ]
    proto = {"mcts_sims": 32, "convention": "PURE",
             "note": "2026-07 tier-a chain (preregistered triangle "
                     "+ reused edges; see POOLED_TRIANGLE.json)"}
    for key, a, b, wa, d, wb, date in edges:
        record_edge(cat, key, a, b, wa, d, wb, protocol=proto,
                    date=date)
    cat["checkpoints"]["ref_2p29M"].update(
        {"file": "seed_20260718.pt", "decision_step": 2290529,
         "lineage": "tier-a", "arch": "5M"})
    cat["checkpoints"]["old_2p40M"].update(
        {"file": "campaign_live_20260729.pt", "decision_step": 2403615,
         "lineage": "tier-a", "arch": "5M"})
    cat["checkpoints"]["new_2p52M"].update(
        {"file": "campaign_live_20260730.pt", "decision_step": 2515896,
         "lineage": "tier-a", "arch": "5M"})
    refit(cat)
    save_catalog(cat, path)
    print(render(cat))


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=("show", "seed-july", "add-meta",
                                    "refit", "rename"))
    ap.add_argument("label", nargs="?")
    ap.add_argument("kv", nargs="*")
    ap.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=logging.INFO)
    if args.cmd == "seed-july":
        seed_july(args.catalog)
        return 0
    cat = load_catalog(args.catalog)
    if args.cmd == "show":
        print(render(cat))
        return 0
    if args.cmd == "refit":
        refit(cat)
        save_catalog(cat, args.catalog)
        print(render(cat))
        return 0
    if args.cmd == "rename":
        if not args.label or len(args.kv) != 1:
            print("usage: elo_catalog.py rename OLD NEW")
            return 2
        rename_label(cat, args.label, args.kv[0])
        save_catalog(cat, args.catalog)
        print(render(cat))
        return 0
    if args.cmd == "add-meta":
        if not args.label or args.label not in cat["checkpoints"]:
            print(f"unknown label {args.label!r}; have "
                  f"{sorted(cat['checkpoints'])}")
            return 2
        for pair in args.kv:
            k, _, v = pair.partition("=")
            try:
                v = int(v)
            except ValueError:
                pass
            cat["checkpoints"][args.label][k] = v
        save_catalog(cat, args.catalog)
        print(render(cat))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
