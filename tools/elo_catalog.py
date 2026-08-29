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


def rename_label(cat: Dict, old: str, new: str,
                 global_alias: bool = False) -> None:
    """Rename a checkpoint label everywhere (checkpoints, edges,
    reference, gauge, and the stored dir-scoped label maps).

    By DEFAULT the old name chains only DIR-SCOPED: each dir that
    recorded games under `old` gets `old -> new` persisted on its
    edges (the map update_from_games re-seeds from), so a plain
    re-collect of THAT dir still lands here, while an unrelated
    future dir reusing the same generic run label (pin, seed, ...)
    does NOT silently chain onto this checkpoint (round-30 C4 --
    the same cross-dir hazard that keeps dir maps out of the
    global alias table, round-20 C4).

    `global_alias=True` additionally records `aliases[old] = new`:
    use it when `old` is a catalog-wide identity of this checkpoint
    (a lineage rename like 2404k -> its lineage-path name), so
    every future write under `old`, from ANY dir, resolves here."""
    if old not in cat["checkpoints"]:
        raise KeyError(f"unknown label {old!r}")
    if new in cat["checkpoints"]:
        raise ValueError(f"label {new!r} already exists")
    _aliases = cat.get("aliases", {})
    if new in _aliases and resolve_label(cat, new) != old:
        # Renaming onto another node's ALIAS would silently make
        # every historical name of `old` resolve to that unrelated
        # node (round-29 C3: verified pooling two checkpoints ~520
        # Elo apart on the committed catalog).
        raise ValueError(
            f"label {new!r} is already an alias for "
            f"{resolve_label(cat, new)!r}; renaming onto it would "
            f"chain every historical name of {old!r} to that "
            f"node.")
    _dangling = [k for k, v in _aliases.items()
                 if v == new and resolve_label(cat, k) != old]
    if _dangling:
        # `new` is a dangling alias TARGET (its node was pruned by
        # a relabeling re-collect): reusing the name would pool the
        # pruned node's historical names onto this one -- the same
        # corruption as the key-side guard above, on the other side
        # of the arrow (round-30 C1).
        raise ValueError(
            f"label {new!r} is the target of alias(es) "
            f"{sorted(_dangling)} (historical names of a "
            f"since-pruned node); renaming {old!r} onto it would "
            f"pool those names onto this node. Pick another name "
            f"or clean the stale aliases first.")
    cat["checkpoints"][new] = cat["checkpoints"].pop(old)
    for e in cat["edges"].values():
        _was_old = old in (e["label_a"], e["label_b"])
        for k in ("label_a", "label_b"):
            if e[k] == old:
                e[k] = new
        lm = e.get("label_map")
        if _was_old and isinstance(lm, dict):
            # Value rewrite only on edges that REFERENCE the
            # renamed node (round-32 C0: rewriting every edge's
            # stray stale values re-attributed an unrelated dir's
            # games across checkpoints).
            for _rl, _canon in list(lm.items()):
                if _canon == old:
                    lm[_rl] = new
        if _was_old:
            # Dir-scoped chaining: the plain re-collect of this
            # dir re-seeds from here (round-30 C4).
            e.setdefault("label_map", {}).setdefault(old, new)
    if cat["reference"]["label"] == old:
        cat["reference"]["label"] = new
    _g = cat.get("gauge")
    if isinstance(_g, dict) and _g.get("label") == old:
        _g["label"] = new                        # round-30 C2
    aliases = cat.setdefault("aliases", {})
    for k, v in aliases.items():
        if v == old:
            aliases[k] = new
    # The re-pointing above just turned any pre-existing `new ->
    # old` chain into a self-alias through `new`. Drop it on EVERY
    # path (round-31 C0: gated on global_alias, the surviving
    # self-alias was re-pointed into a REAL global alias by the
    # next dir-scoped rename, defeating the round-30 default).
    aliases.pop(new, None)
    if global_alias:
        aliases[old] = new


def save_catalog(cat: Dict, path: Path = CATALOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cat, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")


def _edge_mass(e: Dict) -> int:
    """Decisive mass of an edge -- what the fit consumes (absences
    carry none). ONE definition shared by refit and the horizon
    guard so their skip rules cannot drift (round-28 C2)."""
    return e["wins_a"] + e["wins_b"] + e["draws"]


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
    if label_a == label_b:
        raise ValueError(
            f"self-edge refused: both labels resolve to "
            f"{label_a!r} (round-21 C0).")
    _new_proto = dict(protocol or {})
    # Untagged downgrades are REFUSED upstream by update_from_games'
    # polarity guard (round-17 C1), but a TAGGED overwrite that
    # omits fields must not erase provenance prose already on the
    # edge -- the committed censoring caveats ("29/40 completed",
    # "9 games wall-clock-censored") live in `note` (round-18 C1).
    _prev_e = cat["edges"].get(source_key) or {}
    # A recycled source_key (a rename shuffle can make a DIFFERENT
    # pair land on an existing key) is a different measurement: its
    # provenance must not stick (round-35 C2).
    _same_pair = bool(_prev_e) and (
        {resolve_label(cat, _prev_e.get("label_a", "")),
         resolve_label(cat, _prev_e.get("label_b", ""))}
        == {label_a, label_b})
    _prev_pr = _prev_e.get("protocol") if _same_pair else None
    if _new_proto.get("note") == "":
        # Explicit empty note CLEARS (round-19 C2: carry-forward
        # alone made notes write-once and unclearable).
        _new_proto.pop("note")
    elif isinstance(_prev_pr, dict) and _prev_pr.get("note") \
            and not _new_proto.get("note"):
        _new_proto["note"] = _prev_pr["note"]
    # max_turns is estimand provenance and equally sticky
    # (round-26 C0: a re-collect that omitted --catalog-max-turns
    # silently erased the catalog's only horizon witness, disarming
    # the round-25 cross-dir guard). A DIFFERING declared/measured
    # value wins -- both of its sources are deliberate -- but
    # loudly.
    if _new_proto.get("max_turns") == "":
        # Explicit clear hatch, mirroring the note's "" sentinel
        # (round-27 C0: stickiness without a hatch made a typo'd
        # horizon permanent).
        _new_proto.pop("max_turns")
    elif isinstance(_prev_pr, dict) \
            and _prev_pr.get("max_turns") is not None:
        if _new_proto.get("max_turns") is None:
            _new_proto["max_turns"] = _prev_pr["max_turns"]
        elif _new_proto["max_turns"] != _prev_pr["max_turns"]:
            print(f"WARNING: catalog: edge {source_key} horizon "
                  f"rewritten max_turns={_prev_pr['max_turns']} -> "
                  f"{_new_proto['max_turns']}; games measured at "
                  f"the old horizon are gone from this edge.")
    cat["edges"][source_key] = {
        "label_a": label_a, "label_b": label_b,
        "wins_a": int(wins_a), "draws": int(draws),
        "wins_b": int(wins_b),
        "no_result": int(no_result),
        "protocol": _new_proto,
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
    # Only DECISIVE mass rates (round-27 C1/C3): a fully-censored
    # edge is a pair key with zero information (it minted a rating
    # equal to the opponent's under the played prior), and a node
    # with zero decisive mass has a gauge-arbitrary gamma whose
    # singular Fisher row rendered as a confident se 0.0. Such
    # nodes report elo/se None and never chain the component walk.
    mass = {lab: 0 for lab in labels}
    n_games_of = {lab: 0 for lab in labels}
    for e in cat["edges"].values():
        m = _edge_mass(e)
        n = e["wins_a"] + e["draws"] + e["wins_b"]
        for lab in (e["label_a"], e["label_b"]):
            mass[lab] += m
            n_games_of[lab] += n
    rated = [lab for lab in labels if mass[lab] > 0]
    _h_mixed = {(e.get("protocol") or {}).get("max_turns")
                for e in cat["edges"].values()
                if isinstance(e.get("protocol"), dict)
                and _edge_mass(e) > 0} - {None, ""}
    if len(_h_mixed) > 1:
        # Belt for paths that bypass update_from_games (manual
        # JSON edits, direct record_edge callers) -- round-29 C4.
        print(f"WARNING: catalog: decisive edges span MIXED turn "
              f"horizons {sorted(_h_mixed)}; the fit pools "
              f"different estimands. Fix with elo_collect "
              f"--catalog-max-turns.")
    for lab in labels:
        meta = cat["checkpoints"][lab]
        meta["n_games"] = n_games_of[lab]
        if lab not in rated:
            meta["elo"] = None
            meta["se"] = None
            meta["anchored"] = False
    ref = cat["reference"]["label"]
    if (cat["reference"].get("auto") and REFERENCE_LABEL in rated
            and ref != REFERENCE_LABEL):
        # A PROVISIONAL auto-designation releases the moment the
        # canonical reference gains decisive mass (round-31 C3:
        # a permanent first-collect designation made a fresh
        # catalog's whole scale and every SE depend on collect
        # ORDER, not on the final edge set).
        ref = REFERENCE_LABEL
        cat["reference"] = {"label": ref, "elo": 0.0}
    def _auto_pick():
        # ONE deterministic rule for every provisional designation
        # (round-32 C2): max decisive mass, smallest label as the
        # tiebreak (max() keeps the first maximal element of the
        # sorted iteration). A pure function of the edge set, so
        # collect order cannot move the gauge.
        return max(sorted(rated), key=lambda lab: mass[lab])

    if (cat["reference"].get("auto") and rated
            and REFERENCE_LABEL not in rated):
        # A PROVISIONAL designation is recomputed from the CURRENT
        # edge set on every refit (round-32 C2: freezing the
        # first-collect pick left --catalog-path side catalogs --
        # where the canonical reference never plays -- order-
        # dependent).
        _best = _auto_pick()
        if _best != ref:
            cat["reference"] = {"label": _best, "elo": 0.0,
                                "auto": True}
            ref = _best
    _walk_root = ref
    _gauge_fallback = False
    _anchor_elo = float(cat["reference"].get("elo", 0.0))
    if ref not in rated and rated:
        if ref not in cat["checkpoints"]:
            # Absent stored reference (fresh --catalog-path,
            # round-18 C0; orphaned auto-reference, round-22 C0):
            # no node can ever carry the gauge, so re-designate
            # permanently.
            ref = _auto_pick()
            _walk_root = ref
            cat["reference"] = {"label": ref, "elo": 0.0,
                                "auto": True}
            _anchor_elo = 0.0
        else:
            # Present but transiently zero-decisive-mass (round-28
            # C0): the node re-rates as soon as a decisive edge
            # lands, so anchor the fit LOCALLY and leave the stored
            # gauge alone -- persisting here made refit history-
            # dependent (identical edges, different board depending
            # on transient collect order) and silently re-anchored
            # the committed gauge every doc figure is stated in.
            print(f"WARNING: catalog: reference {ref!r} has no "
                  f"decisive mass this refit; ratings are gauged "
                  f"on {rated[0]!r} = 0 until it re-rates.")
            ref = rated[0]
            _anchor_elo = 0.0
            _gauge_fallback = True
    if not rated:
        # Nothing is rated: no gauge claim can hold either
        # (round-30 C3: a stale fallback record named a node whose
        # own elo is None).
        cat.pop("gauge", None)
        cat["last_refit"] = time.strftime("%FT%TZ", time.gmtime())
        return
    idx = {lab: i for i, lab in enumerate(rated)}
    pairs: Dict[Tuple[int, int], PairRecord] = {}
    adj: Dict[str, set] = {lab: set() for lab in rated}
    for e in cat["edges"].values():
        if _edge_mass(e) == 0:
            continue          # censored edge: not a chain link
        a, b = idx[e["label_a"]], idx[e["label_b"]]
        i, j = min(a, b), max(a, b)
        rec = pairs.setdefault((i, j), PairRecord())
        a_is_i = a == i
        rec.wins_i += e["wins_a"] if a_is_i else e["wins_b"]
        rec.wins_j += e["wins_b"] if a_is_i else e["wins_a"]
        rec.draws += e["draws"]
        adj[e["label_a"]].add(e["label_b"])
        adj[e["label_b"]].add(e["label_a"])
    anchor_idx = idx.get(ref, 0)
    elo, se = fit_elo(len(rated), pairs, anchor_idx,
                      anchor_elo=_anchor_elo,
                      prior_games=1.0, draw_weight=0.5,
                      prior_scope="played")
    # Connected component of the STORED reference (only these
    # ratings chain). On a transient gauge fallback the stored
    # reference has no mass-bearing edge, so `seen` stays empty
    # and every node truthfully reports anchored=False -- walking
    # from the fallback anchor instead stamped anchored=True on
    # components that never chained to the reference (round-29
    # C1).
    seen = set()
    stack = [_walk_root] if _walk_root in adj else []
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        stack.extend(adj[u] - seen)
    for lab in rated:
        meta = cat["checkpoints"][lab]
        meta["elo"] = round(float(elo[idx[lab]]), 1)
        meta["se"] = round(float(se[idx[lab]]), 1)
        meta["anchored"] = lab in seen
    if _gauge_fallback:
        # Persist the gauge fact IN the JSON -- render's header is
        # stdout-only and consumers read the file (round-29 C1).
        cat["gauge"] = {"label": ref, "fallback": True}
    else:
        cat.pop("gauge", None)
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
    label_map = dict(label_map or {})
    # Dir-scoped label persistence (round-22 C2): the mapping used
    # to canonicalize THIS dir is stored on its edges, and a later
    # collect of the same dir with no caller mapping seeds from it
    # -- so the documented plain re-collect reproduces the same
    # canonical edge instead of resurrecting run-local phantoms.
    # Deliberately NOT written to cat["aliases"]: a generic run
    # label like 'seed' is reused across dirs under different
    # procedures and must not chain globally (round-20 C4).
    # The scan ALWAYS runs and merges under caller precedence
    # (round-23 C0: gating it on an empty caller map meant a
    # partial --catalog-alias re-collect discarded the persisted
    # entries and resurrected run-local phantoms -- the exact
    # corruption the persistence exists to prevent).
    _pfx = f"{Path(games_dir).name}:"
    _seed_cat = load_catalog(path)
    _caller = dict(label_map)
    # Collision detection compares RESOLVED targets -- the tally
    # aggregation resolves too, so a raw-string compare is blind to
    # any collision hidden behind one alias hop (round-25 C0:
    # verified pooling two checkpoints through the committed
    # l5_pin3060972 alias).
    _caller_res = {k: resolve_label(_seed_cat, v)
                   for k, v in _caller.items()}
    _run_labels = ({g["label_a"] for g in games}
                   | {g["label_b"] for g in games})
    _seeded = {}
    _all_tombs = {}
    for _ek, _e in _seed_cat.get("edges", {}).items():
        if not _ek.startswith(_pfx):
            continue
        for _kk, _vv in (_e.get("label_map") or {}).items():
            if _kk in label_map:
                continue
            if _kk not in _run_labels:
                # Not a run label of this collect: either an inert
                # tombstone from an old run (round-25 C1 -- never
                # refused) or a CANONICAL chain stamped by a
                # dir-scoped rename (round-30 C4). Collected for
                # PER-EDGE application below (round-34 C0: both a
                # dir-wide union and a live-key veto were order-
                # dependent -- each edge's stamps must redirect
                # only ITS OWN run labels).
                _all_tombs.setdefault(_kk, _vv)
                continue
            label_map[_kk] = _vv
            _seeded[_kk] = _vv
    if _seeded:
        print(f"catalog: seeded dir-scoped label map "
              f"{_seeded} from existing edges")
    # Per-RUN-LABEL redirect, applying each edge's OWN rename
    # stamps to the run labels THAT edge names (round-34 C0: the
    # read side mirrors the round-33 per-edge persist -- a freed-
    # and-reused name's stamp on one edge must never redirect a
    # different edge's run label, in either rename order). Caller
    # precedence is preserved: the stamp applies to the caller's
    # canonical, it never replaces it.
    _redirect = {}
    for _ek2 in sorted(_seed_cat.get("edges", {})):
        if not _ek2.startswith(_pfx):
            continue
        _m2 = _seed_cat["edges"][_ek2].get("label_map") or {}
        _tombs2 = {k: v for k, v in _m2.items()
                   if k not in _run_labels}
        for _rl2 in _m2:
            if _rl2 not in _run_labels:
                continue
            _canon = label_map.get(_rl2, _rl2)
            _fin = _tombs2.get(_canon, _canon)
            if _rl2 in _redirect and _redirect[_rl2] != _fin:
                print(f"WARNING: catalog: run label {_rl2!r} "
                      f"redirects differently across the dir's "
                      f"edges ({_redirect[_rl2]!r} vs {_fin!r}); "
                      f"keeping {_redirect[_rl2]!r}.")
                continue
            _redirect[_rl2] = _fin
    # Collision guard on POST-REDIRECT finals (round-35 C1: the raw
    # comparison spuriously refused the free-and-reuse rename's
    # documented partial re-collect, and its identity-alias escape
    # minted a phantom node). A SEEDED run label whose final lands
    # on a caller run label's final is the genuine pooling risk
    # (round-24 C0/C1) -- refuse before any write.
    def _final_of(lbl):
        _c = _redirect.get(lbl, label_map.get(lbl, lbl))
        return resolve_label(_seed_cat, _c)

    _caller_fin = {k: _final_of(k) for k in _caller}
    for _kk3 in sorted(_seeded):
        _fk3 = _final_of(_kk3)
        _hit = sorted(k for k, v in _caller_fin.items()
                      if v == _fk3)
        if _hit:
            raise ValueError(
                f"dir-scoped map collision: persisted alias "
                f"{_kk3!r} and this collect's {_hit[0]!r} both "
                f"resolve to {_fk3!r}. If both run labels really "
                f"are {_fk3!r}, pass --catalog-alias "
                f"{_kk3}={_fk3} explicitly too; otherwise correct "
                f"the --catalog-alias entries. The catalog is "
                f"untouched.")
    # Estimand guard (round-14 C1): the global refit pools every
    # edge into ONE Bradley-Terry fit, so an edge measured under a
    # different procedure than the catalog's existing notes would
    # silently shift every rating. Notes are compared when both
    # sides declare one; empty/legacy notes pass with a warning.
    def _proc_key(pr):
        """Structured estimand key of an edge's protocol: the
        'procedure' field (round-15 C0: comparing free-text NOTES
        refused every properly-tagged new edge while legacy prose
        passed -- inverted polarity), plus the pt_config for plan
        runs (round-15 C5: same procedure, different knobs is still
        a different estimand). Legacy shapes (str/None/no field)
        yield None = unconstrained."""
        if not isinstance(pr, dict):
            return None
        proc = pr.get("procedure")
        if not proc:
            return None
        ptc = pr.get("pt_config")
        tsc = pr.get("turn_config")
        return (proc,
                json.dumps(ptc, sort_keys=True)
                if ptc is not None else None,
                json.dumps(tsc, sort_keys=True)
                if tsc is not None else None)

    _cat = load_catalog(path)
    _existing_all = {_proc_key(e.get("protocol"))
                     for e in _cat.get("edges", {}).values()} - {None}
    # The MISMATCH compares against OTHER dirs' DECISIVE edges
    # only: this collect replaces its own dir's edges, so a
    # mistyped --catalog-procedure must not veto its own
    # correction (round-29 C2, the rule the horizon guard already
    # follows), and censored edges are outside the fit (round-28
    # C2). The POLARITY refusal below still scans ALL edges -- an
    # untagged write must not slip in just because the only tagged
    # edges are its own.
    _existing = {_proc_key(e.get("protocol"))
                 for _k2, e in _cat.get("edges", {}).items()
                 if not _k2.startswith(_pfx)
                 and _edge_mass(e) > 0} - {None}
    _new_key = _proc_key(protocol)
    # Incoming decisive mass: a fully-censored collect cannot pool
    # estimands, so BOTH symmetric guards (procedure here, horizon
    # below) gate on it -- round-30 C0 mirrored round-29 C0's
    # order-independence fix onto the procedure axis.
    _new_mass = sum(1 for g in games
                    if g.get("outcome_a") in ("win", "loss"))
    _own_prev = {_proc_key(e.get("protocol"))
                 for _k2, e in _cat.get("edges", {}).items()
                 if _k2.startswith(_pfx)} - {None}
    if _new_key and _own_prev and _new_key not in _own_prev:
        # record_edge overwrites the procedure silently; a same-dir
        # retag is a deliberate correction but must be AUDIBLE.
        print(f"WARNING: catalog: dir {Path(games_dir).name} "
              f"procedure retagged {sorted(_own_prev)} -> "
              f"{_new_key}; the dir's prior edges under the old "
              f"tag are replaced.")
    if (_new_key and _existing and _new_mass > 0
            and _new_key not in _existing):
        raise ValueError(
            f"catalog estimand mismatch: edge is {_new_key} but "
            f"the catalog holds {sorted(_existing)} -- estimands "
            f"don't mix in one fit. Pass --no-catalog, or start a "
            f"separate catalog for the new procedure.")
    if not _new_key and _existing_all:
        # Polarity fix (round-16 C2): a tagged catalog REFUSES
        # untagged writes -- the guard must not admit exactly the
        # edges it cannot verify. Declare the procedure with
        # elo_collect --catalog-procedure (legacy dirs), or skip
        # with --no-catalog.
        raise ValueError(
            "catalog holds structured procedures but this edge "
            "declares none: declare it (elo_collect "
            "--catalog-procedure, e.g. 'mcts:32' for pre-tag "
            "dirs) or pass --no-catalog.")
    if not _new_key:
        print("WARNING: catalog edge recorded without a structured "
              "procedure (all-legacy catalog); the refit pools it "
              "with the existing edges.")
    # Horizon guard (round-25 C5): the global refit must not pool
    # edges measured at different turn horizons -- the horizon
    # decides decisive-vs-absence, the quantity the fit is built
    # on. Component-wise with silence-unconstrained (a hard tuple
    # slot in _proc_key would refuse every future collect against
    # the committed horizon-silent edges).
    _mt_new = (protocol or {}).get("max_turns") \
        if isinstance(protocol, dict) else None
    if _mt_new == "":                      # explicit clear sentinel
        _mt_new = None
    # Own-dir edges are excluded: this collect REPLACES them, so
    # they must not veto their own correction (round-27 C0: one
    # typo'd --catalog-max-turns was permanently uncorrectable and
    # the record_edge rewrite warning was unreachable).
    # Censored edges are excluded from the fit, so they cannot
    # pool estimands -- a cap-heavy probe's horizon must not veto
    # the verdict match's (round-28 C2). Same skip rule as refit,
    # via the shared _edge_mass.
    _mt_existing = {e["protocol"].get("max_turns")
                    for _k2, e in _cat.get("edges", {}).items()
                    if isinstance(e.get("protocol"), dict)
                    and not _k2.startswith(_pfx)
                    and _edge_mass(e) > 0} - {None, ""}
    # Gated on _new_mass for order-independence (round-29 C0:
    # probe-then-verdict accepted, verdict-then-probe refused, on
    # the identical final edge set).
    if (_mt_new is not None and _mt_existing and _new_mass > 0
            and _mt_new not in _mt_existing):
        raise ValueError(
            f"catalog horizon mismatch: edge measured at "
            f"max_turns={_mt_new} but the catalog holds "
            f"{sorted(_mt_existing)} -- estimands don't mix in "
            f"one fit. Pass --no-catalog, or start a separate "
            f"catalog (--catalog-path) for the new horizon.")
    # Tally = [wins_a, genuine_draws, wins_b, no_result]. Under the
    # 2026-08-17 ruling every non-decisive outcome is a no-result
    # absence (there are no draws in real Wesnoth); the draws slot
    # stays for legacy edges and a hypothetical future genuine-draw
    # outcome, and is always 0 from this path.
    tallies: Dict[Tuple[str, str], List[int]] = {}
    _contrib: Dict[Tuple[str, str], set] = {}
    for g in games:
        a = _redirect.get(g["label_a"],
                          label_map.get(g["label_a"],
                                        g["label_a"]))
        b = _redirect.get(g["label_b"],
                          label_map.get(g["label_b"],
                                        g["label_b"]))
        key = (a, b) if a <= b else (b, a)
        t = tallies.setdefault(key, [0, 0, 0, 0])
        _contrib.setdefault(key, set()).update(
            (g["label_a"], g["label_b"]))
        out = g["outcome_a"]
        if out == "win":
            t[0 if a <= b else 2] += 1
        elif out == "loss":
            t[2 if a <= b else 0] += 1
        else:
            t[3] += 1
    cat = load_catalog(path)
    # Canonicalize through the catalog's alias chain, then AGGREGATE
    # on the resolved pair before writing (round-19 C0 + round-20
    # C0: keying on resolved labels without re-merging let two raw
    # pairs that resolve to one canonical pair clobber each other --
    # the later write silently dropped the earlier tally's games).
    resolved: dict = {}
    _rcontrib: Dict[Tuple[str, str], set] = {}
    for (a, b), (wa, d, wb, nr) in sorted(tallies.items()):
        ra, rb = resolve_label(cat, a), resolve_label(cat, b)
        if ra == rb:
            raise ValueError(
                f"self-edge refused: raw labels {a!r} and {b!r} "
                f"both resolve to {ra!r} (check --catalog-alias / "
                f"the alias chain). A checkpoint cannot be its own "
                f"opponent; recording it corrupts every rating in "
                f"the global fit (round-21 C0: verified NaN board).")
        if ra > rb:
            ra, rb = rb, ra
            wa, wb = wb, wa
        t = resolved.setdefault((ra, rb), [0, 0, 0, 0])
        t[0] += wa
        t[1] += d
        t[2] += wb
        t[3] += nr
        _rcontrib.setdefault((ra, rb), set()).update(
            _contrib.get((a, b), set()))
    _prefix = f"{Path(games_dir).name}:"
    # Snapshot BEFORE the record loop: a rename shuffle can make a
    # NEW pair land on an EXISTING key (displacement), and the old
    # pair's provenance must still migrate to its successor
    # (round-35 C2). record_edge assigns a fresh dict, so these
    # references stay the pre-write edges.
    _pre_edges = {k: e for k, e in cat["edges"].items()
                  if k.startswith(_prefix)}
    new_keys = set()
    for (ra, rb), (wa, d, wb, nr) in sorted(resolved.items()):
        source_key = f"{Path(games_dir).name}:{ra}~{rb}"
        new_keys.add(source_key)
        record_edge(cat, source_key, ra, rb, wa, d, wb,
                    protocol=protocol, no_result=nr)
        _edge_runs = _rcontrib.get((ra, rb), set())
        _persist = {**{k: v for k, v in _all_tombs.items()
                       if v in (ra, rb)},
                    **{k: _redirect.get(k, v)
                       for k, v in label_map.items()
                       if k in _edge_runs}}
        if _persist:
            # PER-EDGE map: only the run labels that fed THIS edge
            # and the tombstones targeting ITS nodes. A dir-wide
            # union left foreign stale entries on unrelated edges,
            # which a later rename rotted into cross-checkpoint
            # re-attribution (round-33 C0). Caller precedence and
            # RESOLVED targets kept (rounds 31 C1 / 32 C0).
            cat["edges"][source_key]["label_map"] = _persist
    # A dir's edge set is REPLACED, never accumulated: drop any
    # stale edge from the same dir whose key was not re-written.
    # The dropped edge's censoring note migrates to the dir's new
    # edge if that edge has none (round-20 C2: the round-18 note
    # carry-forward was keyed on source_key and lost the caveat
    # across a relabel).
    _note_cleared = (protocol or {}).get("note") == ""
    _migrated = set()
    _mt_cleared = (protocol or {}).get("max_turns") == "" \
        if isinstance(protocol, dict) else False
    _stale_mts = set()
    for _k in [k for k in _pre_edges
               if k not in new_keys
               or {_pre_edges[k]["label_a"], _pre_edges[k]["label_b"]}
               != {cat["edges"][k]["label_a"],
                   cat["edges"][k]["label_b"]}]:
        _stale = _pre_edges[_k]
        if _k not in new_keys:
            cat["edges"].pop(_k, None)
        _stale_pr = _stale.get("protocol") \
            if isinstance(_stale.get("protocol"), dict) else {}
        _note = (_stale_pr or {}).get("note")
        if _stale_pr.get("max_turns") is not None:
            _stale_mts.add(_stale_pr["max_turns"])
        if _note and not _note_cleared:
            # Only the stale edge's own PAIR successor inherits the
            # caveat (round-22 C1: stamping every noteless new edge
            # mis-attributed pair-specific censoring notes).
            # The stale edge's labels are CANONICAL after its own
            # relabel, while label_map keys run-local names -- so
            # invert the map the stale edge stored about itself to
            # recover the run-local name, THEN apply the current
            # map (round-23 C1: without the inversion a second
            # corrective relabel computed a successor that never
            # existed and silently dropped the censoring note).
            _prev_map = _stale.get("label_map") or {}

            _stale_tombs = {k: v for k, v in _prev_map.items()
                            if k not in _run_labels}

            def _canon2(x):
                # Canonicalization = the STALE EDGE'S OWN rename
                # stamps + the global alias chain (round-32 C1;
                # per-edge since round-34 C0 -- another edge's
                # stamp must not steer this pair's successor).
                x = _stale_tombs.get(x, x)
                return resolve_label(cat, x)

            def _succ_labels(lab):
                # Stored canonicals resolve before comparing
                # (round-24 C2: rename_label rewrites edge labels
                # in place but not the maps stored on them). The
                # inverted map can fan out when the stale edge
                # POOLED two run-local names (round-24 C4) --
                # return every successor.
                hits = {_canon2(label_map[_rl])
                        for _rl, _canon in _prev_map.items()
                        if _canon2(_canon) == lab
                        and _rl in label_map}
                return hits or {_canon2(label_map.get(lab, lab))}
            _succs = set()
            for _sa in _succ_labels(_stale["label_a"]):
                for _sb in _succ_labels(_stale["label_b"]):
                    _lo, _hi = ((_sa, _sb) if _sa <= _sb
                                else (_sb, _sa))
                    _succs.add(
                        f"{Path(games_dir).name}:{_lo}~{_hi}")
            _hits = sorted(_succs & new_keys)
            if len(_hits) > 1:
                print(f"WARNING: catalog: stale edge {_k}'s note "
                      f"describes a POOLED pair that split into "
                      f"{_hits}; the caveat is copied to all of "
                      f"them -- re-check it applies to each")
            for _h in _hits:
                np_ = cat["edges"][_h].setdefault("protocol", {})
                if not isinstance(np_, dict):
                    continue
                _cur = np_.get("note")
                if not _cur:
                    np_["note"] = _note
                    _migrated.add(_h)
                elif _h in _migrated and _note not in _cur:
                    # Two stale edges pooling into one successor
                    # keep BOTH caveats (round-24 C3: the second
                    # was silently discarded).
                    np_["note"] = f"{_cur} | {_note}"
                elif _note not in _cur:
                    print(f"WARNING: catalog: note on stale edge "
                          f"{_k} not migrated ({_h} already "
                          f"carries a note): {_note!r}")
            if not _hits:
                print(f"WARNING: catalog: note on stale edge {_k} "
                      f"not migrated (no successor among "
                      f"{sorted(new_keys)}): {_note!r}")
        print(f"catalog: replaced stale edge {_k} "
              f"({'relabeled' if _k not in new_keys else 'pair displaced'})")
    # The horizon is DIR-scoped (one collect = one horizon), unlike
    # the pair-scoped censoring note, so it migrates to EVERY new
    # edge of the dir that lacks one -- the pair-successor path
    # silently dropped it when the run labels changed entirely
    # (round-27 C2). A horizon the new collect declares itself wins
    # (the stamp only fills gaps); an explicit clear stops it.
    if _stale_mts and not _mt_cleared:
        if len(_stale_mts) > 1:
            print(f"WARNING: catalog: stale edges of "
                  f"{Path(games_dir).name} carried MIXED horizons "
                  f"{sorted(_stale_mts)}; none migrated.")
        else:
            _mt_dir = next(iter(_stale_mts))
            for _k2 in sorted(new_keys):
                np2 = cat["edges"][_k2].setdefault("protocol", {})
                if isinstance(np2, dict) \
                        and np2.get("max_turns") is None:
                    np2["max_turns"] = _mt_dir
    # Effective-horizon re-check AFTER stickiness and migration
    # settle (round-29 C4: a censored edge's declared horizon
    # becomes a live estimand the moment replacement games make it
    # decisive, and record_edge's carry-forward restores it while
    # the entry guard only saw the DECLARED None). Raising here
    # precedes save_catalog, so the on-disk catalog is untouched.
    _mt_eff = {(cat["edges"][_k2].get("protocol") or {}).get(
                   "max_turns")
               for _k2 in new_keys
               if isinstance(cat["edges"][_k2].get("protocol"),
                             dict)
               and _edge_mass(cat["edges"][_k2]) > 0} - {None, ""}
    _mt_bad = _mt_eff - _mt_existing if _mt_existing else set()
    if _mt_bad:
        raise ValueError(
            f"catalog horizon mismatch after carry-forward: this "
            f"dir's decisive edges land at max_turns="
            f"{sorted(_mt_bad)} but the catalog holds "
            f"{sorted(_mt_existing)}. Correct the dir's horizon "
            f"with --catalog-max-turns, or clear it with "
            f"--catalog-max-turns \"\".")
    # Prune checkpoint nodes no edge references (round-20 C1: a
    # relabeling re-collect left phantom nodes that refit rated via
    # the ghost-game prior, biasing the real board toward 50%).
    # Deliberate meta-only entries (a 'file' pin) and the reference
    # survive.
    _referenced = {lab for e in cat["edges"].values()
                   for lab in (e["label_a"], e["label_b"])}
    # No reference exemption (round-22 C0): a genuinely-referenced
    # reference is in _referenced anyway, and pruning an ORPHANED
    # auto-designated reference lets refit re-designate to a real
    # label instead of pinning the fit to a phantom anchor.
    for _lab in [lab for lab, m in cat["checkpoints"].items()
                 if lab not in _referenced
                 and not (m or {}).get("file")]:
        cat["checkpoints"].pop(_lab)
        print(f"catalog: pruned orphan checkpoint {_lab}")
    refit(cat)
    save_catalog(cat, path)
    log.info(f"elo catalog updated ({len(tallies)} edge(s) from "
             f"{Path(games_dir).name}) -> {path}")


def render(cat: Dict) -> str:
    _g = cat.get("gauge") or {}
    _gauge = ("" if not _g.get("fallback")
              else f"  [reference UNRATED this refit -- gauged "
                   f"locally on {_g.get('label')!r} = 0; nothing "
                   f"is on the reference scale]")
    lines = [f"Elo catalog -- reference {cat['reference']['label']} "
             f"= {cat['reference']['elo']:.0f}, last refit "
             f"{cat.get('last_refit', 'never')}{_gauge}"]
    rated = sorted(cat["checkpoints"].items(),
                   key=lambda kv: -(kv[1].get("elo") or 0))
    for lab, m in rated:
        anch = "" if m.get("anchored", True) else "  [UNANCHORED]"
        step = m.get("decision_step", "?")
        if m.get("elo") is None:
            lines.append(
                f"  {lab:<16} {'unrated':>15}  "
                f"({m.get('n_games', 0)} games, step {step}){anch}")
            continue
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
                     "+ reused edges; see POOLED_TRIANGLE.json)", "procedure": "mcts:32"}
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
    ap.add_argument("--global-alias", action="store_true",
                    help="rename only: also record a GLOBAL alias "
                         "OLD->NEW, so writes under OLD from ANY "
                         "future dir chain to the renamed node. "
                         "Default is dir-scoped (only the dirs "
                         "that recorded games under OLD chain) -- "
                         "use the flag for lineage renames of a "
                         "catalog-wide identity, never for "
                         "generic run labels like pin/seed "
                         "(round-30 C4).")
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
            print("usage: elo_catalog.py rename OLD NEW "
                  "[--global-alias]")
            return 2
        rename_label(cat, args.label, args.kv[0],
                     global_alias=args.global_alias)
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
