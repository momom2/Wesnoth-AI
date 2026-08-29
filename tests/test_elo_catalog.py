"""Elo-catalog tests: idempotent edge recording, global refit
chaining, reference anchoring, and the elo_collect auto-update hook
-- production code paths on synthetic edges.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.elo_catalog import (  # noqa: E402
    load_catalog, record_edge, refit, save_catalog, update_from_games,
)


def _fresh(tmp_path) -> Path:
    return tmp_path / "cat.json"


def test_edge_upsert_is_idempotent(tmp_path):
    p = _fresh(tmp_path)
    cat = load_catalog(p)
    record_edge(cat, "dirX:a~b", "A", "B", 30, 0, 10)
    record_edge(cat, "dirX:a~b", "A", "B", 25, 5, 10)   # replaces
    assert len(cat["edges"]) == 1
    assert cat["edges"]["dirX:a~b"]["wins_a"] == 25
    refit(cat)
    save_catalog(cat, p)
    cat2 = load_catalog(p)
    assert cat2["checkpoints"]["A"]["n_games"] == 40


def test_refit_chains_and_anchors(tmp_path):
    """A beats ref 70-30; C beats A 70-30 -> C chains ABOVE A above
    ref, all anchored; D (no path to ref) is flagged unanchored."""
    p = _fresh(tmp_path)
    cat = load_catalog(p)
    cat["reference"] = {"label": "ref", "elo": 0.0}
    record_edge(cat, "s1", "A", "ref", 70, 0, 30)
    record_edge(cat, "s2", "C", "A", 70, 0, 30)
    record_edge(cat, "s3", "D", "E", 10, 0, 10)
    refit(cat)
    ck = cat["checkpoints"]
    assert abs(ck["ref"]["elo"]) < 1e-6
    assert ck["A"]["elo"] > 80          # ~+147 with prior shrinkage
    assert ck["C"]["elo"] > ck["A"]["elo"] + 80
    assert ck["A"]["anchored"] and ck["C"]["anchored"]
    assert not ck["D"]["anchored"] and not ck["E"]["anchored"]


def test_update_from_games_hook(tmp_path):
    """The elo_collect hook: raw game records aggregate to a PURE
    edge, keyed by dir name, and the catalog file lands on disk.
    Non-decisive games are no-result absences (user ruling
    2026-08-17), recorded on the edge but outside the W-D-L."""
    p = _fresh(tmp_path)
    games = (
        [{"label_a": "new", "label_b": "old", "outcome_a": "win"}] * 26
        + [{"label_a": "new", "label_b": "old", "outcome_a": "loss"}] * 10
        + [{"label_a": "new", "label_b": "old", "outcome_a": "draw"}] * 4
    )
    update_from_games(Path("eval_games/run1"), games, path=p)
    cat = load_catalog(p)
    (key, edge), = cat["edges"].items()
    assert key == "run1:new~old"
    assert (edge["wins_a"], edge["draws"], edge["wins_b"]) == (26, 0, 10)
    assert edge["no_result"] == 4
    assert cat["checkpoints"]["new"]["elo"] > \
        cat["checkpoints"]["old"]["elo"]
    # Re-collecting the same dir must not double-count.
    update_from_games(Path("eval_games/run1"), games, path=p)
    cat = load_catalog(p)
    assert len(cat["edges"]) == 1
    assert cat["checkpoints"]["new"]["n_games"] == 36


def test_rename_migrates_edges_and_aliases_resolve(tmp_path):
    """docs/checkpoint_naming.md migration path: rename moves the
    node and rewrites edges; an edge later recorded under the OLD
    label (e.g. a stale box-side games dir) chains onto the renamed
    node instead of creating a phantom sibling."""
    from tools.elo_catalog import rename_label, resolve_label
    p = _fresh(tmp_path)
    cat = load_catalog(p)
    cat["reference"] = {"label": "ref", "elo": 0.0}
    record_edge(cat, "s1", "A", "ref", 70, 0, 30)
    rename_label(cat, "A", "2516k", global_alias=True)
    assert "A" not in cat["checkpoints"]
    assert cat["edges"]["s1"]["label_a"] == "2516k"
    # Stale-label edge resolves through the alias.
    record_edge(cat, "s2", "A", "ref", 60, 0, 40)
    assert cat["edges"]["s2"]["label_a"] == "2516k"
    refit(cat)
    assert cat["checkpoints"]["2516k"]["n_games"] == 200
    # Chained rename compresses: oldest alias points at the head.
    rename_label(cat, "2516k", "2810k", global_alias=True)
    assert resolve_label(cat, "A") == "2810k"


def test_edge_protocol_is_durable(tmp_path):
    """Round-16 C0/C2 (amended round-17 C1): (a) a tagged catalog
    refuses an untagged write instead of admitting-and-stripping --
    this refusal IS the anti-downgrade protection (the former
    record_edge merge branch was unreachable dead code); (b) a
    declared procedure is preserved on re-collect."""
    import json
    import pytest
    from tools.elo_catalog import update_from_games
    cat_path = tmp_path / "cat.json"
    games = [{"label_a": "a", "label_b": "b", "outcome_a": "win",
              "margin_a": 0.5}] * 4
    gd = tmp_path / "runX"
    update_from_games(gd, games, protocol={"procedure": "mcts:32"},
                      path=cat_path)
    # (a) untagged write into the now-tagged catalog refuses.
    with pytest.raises(ValueError):
        update_from_games(gd, games, protocol=None, path=cat_path)
    # (b) tagged re-collect keeps the structured field.
    cat = json.loads(cat_path.read_text(encoding="utf-8"))
    key = next(iter(cat["edges"]))
    assert cat["edges"][key]["protocol"]["procedure"] == "mcts:32"
    update_from_games(gd, games,
                      protocol={"procedure": "mcts:32",
                                "note": "rerun"}, path=cat_path)
    cat = json.loads(cat_path.read_text(encoding="utf-8"))
    assert cat["edges"][key]["protocol"]["procedure"] == "mcts:32"
    assert cat["edges"][key]["protocol"]["note"] == "rerun"


def test_same_procedure_different_pt_config_refused(tmp_path):
    """Round-17 C0: same procedure tag, different --pt-* knobs is a
    different estimand -- the catalog guard must compare the knob
    component, which requires collect to actually attach it."""
    import pytest
    from tools.elo_catalog import update_from_games
    games = [{"label_a": "a", "label_b": "b", "outcome_a": "win",
              "margin_a": 0.5}] * 4
    cat = tmp_path / "cat.json"
    update_from_games(tmp_path / "run1", games,
                      protocol={"procedure": "plan_tournament:32",
                                "pt_config": {"budget_forwards": 700}},
                      path=cat)
    with pytest.raises(ValueError):
        update_from_games(tmp_path / "run2", games,
                          protocol={"procedure": "plan_tournament:32",
                                    "pt_config":
                                        {"budget_forwards": 350}},
                          path=cat)


def test_fresh_catalog_first_refit_anchors(tmp_path):
    """Round-18 C0: on a fresh --catalog-path catalog, the FIRST
    refit must anchor the component to the re-designated reference
    (a stale local flagged everything UNANCHORED until a second
    refit healed it)."""
    import json
    from tools.elo_catalog import update_from_games
    cat_path = tmp_path / "cat.json"
    games = ([{"label_a": "A", "label_b": "B", "outcome_a": "win",
               "margin_a": 0.5}] * 7
             + [{"label_a": "A", "label_b": "B", "outcome_a": "loss",
                 "margin_a": -0.5}] * 3)
    update_from_games(tmp_path / "run1", games,
                      protocol={"procedure": "mcts:32"},
                      path=cat_path)
    cat = json.loads(cat_path.read_text(encoding="utf-8"))
    assert cat["reference"]["label"] in cat["checkpoints"]
    assert all(v[0] if isinstance(v, list) else v.get("anchored")
               for v in cat["checkpoints"].values()), \
        f"first refit left UNANCHORED flags: {cat['checkpoints']}"


def test_tagged_overwrite_preserves_note(tmp_path):
    """Round-18 C1: a tagged re-collect that omits the note must not
    erase censoring caveats already recorded on the edge."""
    import json
    from tools.elo_catalog import update_from_games
    cat_path = tmp_path / "cat.json"
    games = [{"label_a": "a", "label_b": "b", "outcome_a": "win",
              "margin_a": 0.5}] * 4
    update_from_games(tmp_path / "runN", games,
                      protocol={"procedure": "mcts:32",
                                "note": "9 games censored"},
                      path=cat_path)
    update_from_games(tmp_path / "runN", games,
                      protocol={"procedure": "mcts:32"},
                      path=cat_path)
    cat = json.loads(cat_path.read_text(encoding="utf-8"))
    key = next(iter(cat["edges"]))
    assert cat["edges"][key]["protocol"]["note"] == "9 games censored"


def test_recollect_with_alias_replaces_not_duplicates(tmp_path):
    """Round-19 C0: the idempotency key must be invariant under the
    relabeling --catalog-alias exists for -- a corrective re-collect
    replaces the dir's edge instead of double-counting its games."""
    import json
    from tools.elo_catalog import update_from_games
    cat_path = tmp_path / "cat.json"
    games = ([{"label_a": "pin", "label_b": "seed",
               "outcome_a": "win", "margin_a": 0.5}] * 3
             + [{"label_a": "pin", "label_b": "seed",
                 "outcome_a": "loss", "margin_a": -0.5}] * 7)
    gd = tmp_path / "match1"
    update_from_games(gd, games, protocol={"procedure": "mcts:32"},
                      path=cat_path)
    update_from_games(gd, games, protocol={"procedure": "mcts:32"},
                      path=cat_path,
                      label_map={"seed": "canonical_seed"})
    cat = json.loads(cat_path.read_text(encoding="utf-8"))
    assert len(cat["edges"]) == 1, \
        f"aliased re-collect duplicated the edge: {list(cat['edges'])}"
    total = sum(e["wins_a"] + e["wins_b"]
                for e in cat["edges"].values())
    assert total == 10, "games double-counted"


def test_colliding_resolved_pairs_aggregate(tmp_path):
    """Round-20 C0: two raw label pairs in one dir that resolve to
    the same canonical pair must SUM, not clobber."""
    import json
    from tools.elo_catalog import (load_catalog, save_catalog,
                                   update_from_games)
    cat_path = tmp_path / "cat.json"
    cat = load_catalog(cat_path)
    cat["aliases"] = {"pin": "PIN", "seed": "SEED"}
    save_catalog(cat, cat_path)
    games = ([{"label_a": "pin", "label_b": "seed",
               "outcome_a": "win", "margin_a": 0.5}] * 10
             + [{"label_a": "PIN", "label_b": "SEED",
                 "outcome_a": "win", "margin_a": 0.5}] * 10)
    update_from_games(tmp_path / "m1", games,
                      protocol={"procedure": "mcts:32"},
                      path=cat_path)
    cat = json.loads(cat_path.read_text(encoding="utf-8"))
    assert len(cat["edges"]) == 1
    e = next(iter(cat["edges"].values()))
    assert e["wins_a"] + e["wins_b"] == 20, "colliding tallies lost"


def test_relabel_recollect_prunes_phantom_nodes(tmp_path):
    """Round-20 C1: a corrective re-collect must not leave the old
    run-local labels as phantom checkpoint nodes (they bias every
    real rating through the ghost-game prior)."""
    import json
    from tools.elo_catalog import update_from_games
    cat_path = tmp_path / "cat.json"
    games = ([{"label_a": "pin", "label_b": "seed",
               "outcome_a": "win", "margin_a": 0.5}] * 3
             + [{"label_a": "pin", "label_b": "seed",
                 "outcome_a": "loss", "margin_a": -0.5}] * 7)
    gd = tmp_path / "m1"
    update_from_games(gd, games, protocol={"procedure": "mcts:32"},
                      path=cat_path)
    update_from_games(gd, games, protocol={"procedure": "mcts:32"},
                      path=cat_path,
                      label_map={"seed": "canonical_seed"})
    cat = json.loads(cat_path.read_text(encoding="utf-8"))
    assert "seed" not in cat["checkpoints"], \
        "phantom node survived the relabeling re-collect"
    assert set(cat["checkpoints"]) == {"pin", "canonical_seed"}


def test_self_edge_refused(tmp_path):
    """Round-21 C0: labels collapsing to one node must refuse -- a
    recorded self-edge NaN'd every rating on the board."""
    import pytest
    from tools.elo_catalog import (load_catalog, save_catalog,
                                   update_from_games)
    cat_path = tmp_path / "cat.json"
    cat = load_catalog(cat_path)
    cat["aliases"] = {"pin": "X", "seed": "X"}
    save_catalog(cat, cat_path)
    before = cat_path.read_text(encoding="utf-8")
    games = [{"label_a": "pin", "label_b": "seed",
              "outcome_a": "win", "margin_a": 0.5}] * 4
    with pytest.raises(ValueError):
        update_from_games(tmp_path / "m1", games,
                          protocol={"procedure": "mcts:32"},
                          path=cat_path)
    assert cat_path.read_text(encoding="utf-8") == before, \
        "refused write must leave the catalog untouched"


def test_recollect_without_alias_reuses_dir_scoped_map(tmp_path):
    """Round-22 C2: a plain re-collect of a dir originally collected
    with --catalog-alias must reproduce the SAME canonical edge (the
    dir-scoped map is seeded from the dir's own edges), not
    resurrect run-local phantom nodes."""
    import json
    from tools.elo_catalog import update_from_games
    cat_path = tmp_path / "cat.json"
    games = [{"label_a": "pin", "label_b": "seed",
              "outcome_a": "win", "margin_a": 0.5}] * 4
    gd = tmp_path / "m1"
    update_from_games(gd, games, protocol={"procedure": "mcts:32"},
                      path=cat_path,
                      label_map={"seed": "CANON_SEED",
                                 "pin": "CANON_PIN"})
    update_from_games(gd, games, protocol={"procedure": "mcts:32"},
                      path=cat_path)          # no label_map
    cat = json.loads(cat_path.read_text(encoding="utf-8"))
    assert len(cat["edges"]) == 1
    assert set(cat["checkpoints"]) == {"CANON_PIN", "CANON_SEED"}, \
        f"phantom run-local nodes resurrected: {set(cat['checkpoints'])}"


def test_orphaned_auto_reference_is_pruned_and_rebound(tmp_path):
    """Round-22 C0: when a relabeling re-collect orphans an
    auto-designated reference, the orphan is pruned and refit
    re-designates -- the board must not pin to a phantom anchor
    with every real node UNANCHORED."""
    import json
    from tools.elo_catalog import update_from_games
    cat_path = tmp_path / "cat.json"
    games = ([{"label_a": "A", "label_b": "B", "outcome_a": "win",
               "margin_a": 0.5}] * 7
             + [{"label_a": "A", "label_b": "B", "outcome_a": "loss",
                 "margin_a": -0.5}] * 3)
    gd = tmp_path / "run1"
    update_from_games(gd, games, protocol={"procedure": "mcts:32"},
                      path=cat_path)
    update_from_games(gd, games, protocol={"procedure": "mcts:32"},
                      path=cat_path, label_map={"A": "canonA"})
    cat = json.loads(cat_path.read_text(encoding="utf-8"))
    assert "A" not in cat["checkpoints"], "phantom anchor survived"
    assert all((v[0] if isinstance(v, list) else v.get("anchored"))
               for v in cat["checkpoints"].values()), \
        f"board unanchored: {cat['checkpoints']}"


def test_partial_alias_recollect_merges_persisted_map(tmp_path, capsys):
    """Round-23 C0: a re-collect that supplies only the NEWLY-needed
    alias must merge the persisted dir-scoped map under caller
    precedence, not discard it -- discarding resurrected run-local
    phantom nodes and moved the dir's games off the canonical node."""
    from tools.elo_catalog import load_catalog, update_from_games
    cat = tmp_path / "cat.json"
    games = [{"label_a": "seed", "label_b": "pin",
              "outcome_a": "win", "margin_a": 0.5}] * 4
    proto = {"procedure": "mcts:32"}
    update_from_games(tmp_path / "m1", games, protocol=proto,
                      path=cat, label_map={"seed": "CANON_SEED"})
    update_from_games(tmp_path / "m1", games, protocol=proto,
                      path=cat, label_map={"pin": "CANON_PIN"})
    c = load_catalog(cat)
    assert set(c["edges"]) == {"m1:CANON_PIN~CANON_SEED"}
    assert set(c["checkpoints"]) == {"CANON_PIN", "CANON_SEED"}
    assert c["edges"]["m1:CANON_PIN~CANON_SEED"]["label_map"] == \
        {"seed": "CANON_SEED", "pin": "CANON_PIN"}
    # Caller precedence still allows a genuine CORRECTION.
    update_from_games(tmp_path / "m2", games, protocol=proto,
                      path=cat, label_map={"seed": "WRONG"})
    update_from_games(tmp_path / "m2", games, protocol=proto,
                      path=cat, label_map={"seed": "RIGHT"})
    c = load_catalog(cat)
    assert "m2:RIGHT~pin" in c["edges"]
    assert "WRONG" not in c["checkpoints"]


def test_censoring_note_survives_second_relabel(tmp_path):
    """Round-23 C1: after a first relabel the stale edge's labels are
    already canonical, so the successor lookup must invert the map
    the edge stored about itself -- otherwise a corrective second
    relabel silently drops the censoring note and the edge reads as
    a complete uncensored match."""
    from tools.elo_catalog import load_catalog, update_from_games
    cat = tmp_path / "cat.json"
    games = [{"label_a": "seed", "label_b": "pin",
              "outcome_a": "win", "margin_a": 0.5}] * 4
    note = "9 games wall-clock-censored at 120min"
    proto = {"procedure": "mcts:32", "note": note}
    update_from_games(tmp_path / "m1", games, protocol=proto,
                      path=cat)
    update_from_games(tmp_path / "m1", games,
                      protocol={"procedure": "mcts:32"}, path=cat,
                      label_map={"seed": "S", "pin": "P"})
    c = load_catalog(cat)
    assert c["edges"]["m1:P~S"]["protocol"].get("note") == note
    update_from_games(tmp_path / "m1", games,
                      protocol={"procedure": "mcts:32"}, path=cat,
                      label_map={"seed": "S2", "pin": "P"})
    c = load_catalog(cat)
    assert set(c["edges"]) == {"m1:P~S2"}
    assert c["edges"]["m1:P~S2"]["protocol"].get("note") == note


def test_seeded_target_collision_refused(tmp_path):
    """Round-24 C0/C1: a persisted dir-scoped alias whose TARGET
    collides with a caller-supplied alias under a different key is a
    correction-in-progress -- merging would pool two distinct
    checkpoints onto one node (or deadlock on the self-edge guard).
    Refused, catalog untouched; the identity alias clears it."""
    import pytest
    from tools.elo_catalog import load_catalog, update_from_games
    cat = tmp_path / "cat.json"
    proto = {"procedure": "mcts:32"}
    g_ac = [{"label_a": "a", "label_b": "c", "outcome_a": "win",
             "margin_a": 0.5}] * 4
    g_bc = [{"label_a": "b", "label_b": "c", "outcome_a": "loss",
             "margin_a": -0.5}] * 4
    update_from_games(tmp_path / "m1", g_ac + g_bc, protocol=proto,
                      path=cat, label_map={"a": "X"})
    before = cat.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="collision"):
        update_from_games(tmp_path / "m1", g_ac + g_bc,
                          protocol=proto, path=cat,
                          label_map={"b": "X"})
    assert cat.read_text(encoding="utf-8") == before
    # The documented escape: identity alias clears the stale entry.
    update_from_games(tmp_path / "m1", g_ac + g_bc, protocol=proto,
                      path=cat, label_map={"b": "X", "a": "a"})
    c = load_catalog(cat)
    assert set(c["edges"]) == {"m1:X~c", "m1:a~c"}
    assert c["edges"]["m1:X~c"]["wins_a"] == 0     # X == b, 0-4
    assert c["edges"]["m1:a~c"]["wins_a"] == 4


def test_note_migration_follows_renames(tmp_path):
    """Round-24 C2: rename_label rewrites edge labels in place but
    not the label_map stored on them; the successor inversion must
    resolve stored canonicals through the alias chain or the note
    lands on the wrong pair after a rename."""
    from tools.elo_catalog import (load_catalog, rename_label,
                                   save_catalog, update_from_games)
    cat = tmp_path / "cat.json"
    proto = {"procedure": "mcts:32"}
    g_cs = [{"label_a": "cand", "label_b": "seed",
             "outcome_a": "win", "margin_a": 0.5}] * 4
    g_os = [{"label_a": "other", "label_b": "seed",
             "outcome_a": "loss", "margin_a": -0.5}] * 4
    update_from_games(
        tmp_path / "d", g_cs, path=cat,
        protocol=dict(proto, note="cand censoring caveat"),
        label_map={"cand": "C1", "seed": "S"})
    c = load_catalog(cat)
    rename_label(c, "C1", "C1x")
    save_catalog(c, cat)
    update_from_games(
        tmp_path / "d", g_cs + g_os, protocol=proto, path=cat,
        label_map={"cand": "C2", "other": "C1x", "seed": "S"})
    c = load_catalog(cat)
    assert c["edges"]["d:C2~S"]["protocol"].get("note") == \
        "cand censoring caveat"
    assert not (c["edges"]["d:C1x~S"].get("protocol") or {}).get(
        "note")


def test_pooled_note_split_fans_out_and_merges(tmp_path, capsys):
    """Round-24 C3/C4: a note on a POOLED stale edge fans out to
    every successor of the split (loudly), and two stale notes
    pooling INTO one successor are concatenated, not dropped."""
    from tools.elo_catalog import load_catalog, update_from_games
    cat = tmp_path / "cat.json"
    proto = {"procedure": "mcts:32"}
    g = lambda a: [{"label_a": a, "label_b": "opp",
                    "outcome_a": "win", "margin_a": 0.5}] * 4
    # (C4) pooled -> split: the caveat reaches BOTH successors.
    update_from_games(
        tmp_path / "d1", g("s1") + g("s2"), path=cat,
        protocol=dict(proto, note="pooled caveat"),
        label_map={"s1": "X", "s2": "X", "opp": "Y"})
    update_from_games(
        tmp_path / "d1", g("s1") + g("s2"), protocol=proto,
        path=cat, label_map={"s1": "A", "s2": "B", "opp": "Y"})
    c = load_catalog(cat)
    assert c["edges"]["d1:A~Y"]["protocol"].get("note") == \
        "pooled caveat"
    assert c["edges"]["d1:B~Y"]["protocol"].get("note") == \
        "pooled caveat"
    assert "POOLED" in capsys.readouterr().out
    # (C3) split -> pooled: BOTH caveats survive on the merged edge.
    # Two edges with DISTINCT notes are seeded directly (a single
    # collect stamps one dir-wide note, so it cannot construct
    # this state).
    from tools.elo_catalog import record_edge, save_catalog
    c = load_catalog(cat)
    record_edge(c, "d2:P~Y", "P", "Y", 4, 0, 0,
                protocol=dict(proto, note="caveat one"))
    c["edges"]["d2:P~Y"]["label_map"] = {"s1": "P", "opp": "Y"}
    record_edge(c, "d2:Q~Y", "Q", "Y", 4, 0, 0,
                protocol=dict(proto, note="caveat two"))
    c["edges"]["d2:Q~Y"]["label_map"] = {"s2": "Q", "opp": "Y"}
    save_catalog(c, cat)
    update_from_games(
        tmp_path / "d2", g("s1") + g("s2"), protocol=proto,
        path=cat, label_map={"s1": "M", "s2": "M", "opp": "Y"})
    c = load_catalog(cat)
    note = c["edges"]["d2:M~Y"]["protocol"].get("note")
    assert "caveat one" in note and "caveat two" in note


def test_collision_hidden_behind_alias_hop_refused(tmp_path):
    """Round-25 C0: collision detection resolves targets through
    the alias chain -- the raw-string compare pooled two distinct
    checkpoints whenever the persisted entry held the OLD name of
    a since-renamed node."""
    import pytest
    from tools.elo_catalog import (load_catalog, rename_label,
                                   save_catalog, update_from_games)
    cat = tmp_path / "cat.json"
    proto = {"procedure": "mcts:32"}
    g = lambda a, b: [{"label_a": a, "label_b": b,
                       "outcome_a": "win", "margin_a": 0.5}] * 4
    update_from_games(tmp_path / "d", g("a", "c"), protocol=proto,
                      path=cat, label_map={"a": "OLD", "c": "C"})
    c = load_catalog(cat)
    rename_label(c, "OLD", "NEW")
    save_catalog(c, cat)
    before = cat.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="resolve"):
        update_from_games(tmp_path / "d",
                          g("a", "c") + g("b", "c"),
                          protocol=proto, path=cat,
                          label_map={"b": "NEW", "c": "C"})
    assert cat.read_text(encoding="utf-8") == before


def test_inert_persisted_alias_neither_seeded_nor_refused(tmp_path):
    """Round-25 C1: a persisted key naming a run label absent from
    this collect's games is inert -- re-running a dir under a
    fresh step-count label must neither refuse nor accumulate
    identity tombstones."""
    from tools.elo_catalog import load_catalog, update_from_games
    cat = tmp_path / "cat.json"
    proto = {"procedure": "mcts:32"}
    g = lambda a: [{"label_a": a, "label_b": "c",
                    "outcome_a": "win", "margin_a": 0.5}] * 4
    update_from_games(tmp_path / "d", g("old_run"), protocol=proto,
                      path=cat,
                      label_map={"old_run": "X", "c": "C"})
    update_from_games(tmp_path / "d", g("new_run"), protocol=proto,
                      path=cat,
                      label_map={"new_run": "X", "c": "C"})
    c = load_catalog(cat)
    assert set(c["edges"]) == {"d:C~X"}
    # The caller entries are live; the old run's entry may persist
    # as an inert tombstone (round-31 C1 union-persist keeps rename
    # stamps alive, and a tombstone key never matches a canonical
    # label) -- what matters is that nothing REFUSED and no phantom
    # node appeared.
    lm = c["edges"]["d:C~X"]["label_map"]
    assert lm["new_run"] == "X" and lm["c"] == "C"
    assert set(c["checkpoints"]) == {"C", "X"}


def test_cross_dir_horizon_mismatch_refused(tmp_path):
    """Round-25 C5: the turn horizon rides the edge protocol and
    the catalog refuses pooling different horizons in one fit;
    horizon-silent edges stay unconstrained (the committed catalog
    predates the field)."""
    import pytest
    from tools.elo_catalog import update_from_games
    cat = tmp_path / "cat.json"
    g = [{"label_a": "a", "label_b": "b", "outcome_a": "win",
          "margin_a": 0.5}] * 4
    update_from_games(tmp_path / "d1", g, path=cat,
                      protocol={"procedure": "mcts:32"})
    update_from_games(tmp_path / "d2", g, path=cat,
                      protocol={"procedure": "mcts:32",
                                "max_turns": 200})
    with pytest.raises(ValueError, match="horizon"):
        update_from_games(tmp_path / "d3", g, path=cat,
                          protocol={"procedure": "mcts:32",
                                    "max_turns": 30})


def test_ghost_prior_confined_to_played_pairs(tmp_path):
    """Round-26 C3: on the catalog's sparse graph, the all-pairs
    ghost prior gave each node (n-1)*prior_games of ghost mass, so
    merely ADDING unrelated checkpoints moved every rating. With
    the played-pair scope, a disconnected edge moves nothing."""
    from tools.elo_catalog import load_catalog, update_from_games
    cat = tmp_path / "cat.json"
    proto = {"procedure": "mcts:32"}
    g = ([{"label_a": "A", "label_b": "ref", "outcome_a": "win",
           "margin_a": 0.5}] * 19
         + [{"label_a": "A", "label_b": "ref", "outcome_a": "loss",
             "margin_a": -0.5}] * 1)
    update_from_games(tmp_path / "d1", g, protocol=proto, path=cat)
    c = load_catalog(cat)
    before = (c["checkpoints"]["A"]["elo"]
              - c["checkpoints"]["ref"]["elo"])
    gxy = ([{"label_a": "X", "label_b": "Y", "outcome_a": "win",
             "margin_a": 0.5}] * 20
           + [{"label_a": "X", "label_b": "Y", "outcome_a": "loss",
               "margin_a": -0.5}] * 20)
    update_from_games(tmp_path / "d2", gxy, protocol=proto,
                      path=cat)
    c = load_catalog(cat)
    # Gauge-INVARIANT comparison (round-32 C2 made an auto gauge a
    # recomputed function of the edge set, so absolute numbers can
    # legitimately re-zero): ghost-prior leakage would move the
    # A-vs-ref DIFFERENCE.
    after = (c["checkpoints"]["A"]["elo"]
             - c["checkpoints"]["ref"]["elo"])
    # 0.15 = two stored-value quantizations (round(elo, 1) per
    # node); the round-26 defect moved ratings by 30-240 Elo.
    assert abs(after - before) < 0.15, (before, after)


def test_horizon_sticky_across_recollect_and_relabel(tmp_path):
    """Round-26 C0: max_turns carries forward across a re-collect
    that omits it (like note), follows a relabel to the successor
    edge, and the cross-dir guard stays armed."""
    import pytest
    from tools.elo_catalog import load_catalog, update_from_games
    cat = tmp_path / "cat.json"
    g = [{"label_a": "a", "label_b": "b", "outcome_a": "win",
          "margin_a": 0.5}] * 4
    update_from_games(tmp_path / "d", g, path=cat,
                      protocol={"procedure": "mcts:32",
                                "max_turns": 200})
    update_from_games(tmp_path / "d", g, path=cat,
                      protocol={"procedure": "mcts:32"})
    c = load_catalog(cat)
    assert c["edges"]["d:a~b"]["protocol"]["max_turns"] == 200
    # Relabel migrates the horizon to the successor edge.
    update_from_games(tmp_path / "d", g, path=cat,
                      protocol={"procedure": "mcts:32"},
                      label_map={"a": "A2"})
    c = load_catalog(cat)
    assert c["edges"]["d:A2~b"]["protocol"]["max_turns"] == 200
    # The guard is still armed: a 60-horizon dir refuses.
    with pytest.raises(ValueError, match="horizon"):
        update_from_games(tmp_path / "d60", g, path=cat,
                          protocol={"procedure": "mcts:32",
                                    "max_turns": 60})


def test_horizon_correctable_and_clearable(tmp_path, capsys):
    """Round-27 C0: a typo'd declared horizon must be correctable by
    a same-dir re-collect (own-dir edges do not veto their own
    replacement; the rewrite warns loudly), other-dir conflicts stay
    refused, and the "" sentinel clears the field like the note's
    hatch."""
    import pytest
    from tools.elo_catalog import load_catalog, update_from_games
    cat = tmp_path / "cat.json"
    g = [{"label_a": "a", "label_b": "b", "outcome_a": "win",
          "margin_a": 0.5}] * 4
    update_from_games(tmp_path / "d", g, path=cat,
                      protocol={"procedure": "mcts:32",
                                "max_turns": 60})       # the typo
    update_from_games(tmp_path / "d", g, path=cat,
                      protocol={"procedure": "mcts:32",
                                "max_turns": 100})      # correction
    c = load_catalog(cat)
    assert c["edges"]["d:a~b"]["protocol"]["max_turns"] == 100
    assert "rewritten" in capsys.readouterr().out
    with pytest.raises(ValueError, match="horizon"):
        update_from_games(tmp_path / "d2", g, path=cat,
                          protocol={"procedure": "mcts:32",
                                    "max_turns": 30})   # other dir
    update_from_games(tmp_path / "d", g, path=cat,
                      protocol={"procedure": "mcts:32",
                                "max_turns": ""})       # clear
    c = load_catalog(cat)
    assert "max_turns" not in c["edges"]["d:a~b"]["protocol"]


def test_horizon_survives_full_relabel(tmp_path):
    """Round-27 C2: the horizon is DIR-scoped, so it migrates to the
    dir's new edges even when the run labels change entirely and no
    pair successor exists."""
    from tools.elo_catalog import load_catalog, update_from_games
    cat = tmp_path / "cat.json"
    g_ab = [{"label_a": "a", "label_b": "b", "outcome_a": "win",
             "margin_a": 0.5}] * 4
    g_cd = [{"label_a": "c", "label_b": "d", "outcome_a": "win",
             "margin_a": 0.5}] * 4
    update_from_games(tmp_path / "dq", g_ab, path=cat,
                      protocol={"procedure": "mcts:32",
                                "max_turns": 60})
    update_from_games(tmp_path / "dq", g_cd, path=cat,
                      protocol={"procedure": "mcts:32"})
    c = load_catalog(cat)
    assert c["edges"]["dq:c~d"]["protocol"]["max_turns"] == 60


def test_fully_censored_match_rates_nothing(tmp_path):
    """Round-27 C3/C1: a 0-0-0 + 40-absence match carries zero
    rating information -- the challenger must come out UNRATED
    (elo/se None, anchored False), not tied to its opponent with a
    prior-minted rating, and the opponent's rating must not move."""
    from tools.elo_catalog import load_catalog, update_from_games
    cat = tmp_path / "cat.json"
    proto = {"procedure": "mcts:32"}
    g = ([{"label_a": "A", "label_b": "ref", "outcome_a": "win",
           "margin_a": 0.5}] * 7
         + [{"label_a": "A", "label_b": "ref", "outcome_a": "loss",
             "margin_a": -0.5}] * 3)
    update_from_games(tmp_path / "d1", g, protocol=proto, path=cat)
    before = load_catalog(cat)["checkpoints"]["A"]["elo"]
    gx = [{"label_a": "X", "label_b": "A", "outcome_a": "timeout",
           "margin_a": 0.0}] * 40
    update_from_games(tmp_path / "d2", gx, protocol=proto, path=cat)
    c = load_catalog(cat)
    x = c["checkpoints"]["X"]
    assert x["elo"] is None and x["se"] is None
    assert x["anchored"] is False
    assert c["checkpoints"]["A"]["elo"] == before


def test_reference_gauge_survives_transient_censoring(tmp_path):
    """Round-28 C0: refit must be a pure function of the edges. A
    re-collect that transiently zeroes the reference's decisive
    mass (partial sync: only capped games landed) must not persist
    a re-anchor -- after the full data returns, the board must be
    identical to before."""
    from tools.elo_catalog import load_catalog, update_from_games
    cat = tmp_path / "cat.json"
    proto = {"procedure": "mcts:32"}
    g_tri = ([{"label_a": "r2291k", "label_b": "c2516k",
               "outcome_a": "win", "margin_a": 0.5}] * 10
             + [{"label_a": "r2291k", "label_b": "c2516k",
                 "outcome_a": "loss", "margin_a": -0.5}] * 30)
    g_l5 = ([{"label_a": "c2516k", "label_b": "l5",
              "outcome_a": "win", "margin_a": 0.5}] * 12
            + [{"label_a": "c2516k", "label_b": "l5",
                "outcome_a": "loss", "margin_a": -0.5}] * 8)
    update_from_games(tmp_path / "tri", g_tri, protocol=proto,
                      path=cat)
    update_from_games(tmp_path / "l5m", g_l5, protocol=proto,
                      path=cat)
    board_a = load_catalog(cat)
    ref_label = board_a["reference"]["label"]
    # Pin the reference DELIBERATELY (drop the auto flag), like the
    # committed catalog's 2291k: the round-28 gauge-stability
    # property protects pinned references; an AUTO designation now
    # legitimately recomputes each refit (round-32 C2).
    from tools.elo_catalog import save_catalog
    board_a["reference"] = {"label": ref_label, "elo": 0.0}
    save_catalog(board_a, cat)
    elos_a = {k: m["elo"] for k, m in board_a["checkpoints"].items()}
    # Partial sync: the triangle dir re-collects as all-censored.
    g_cens = [{"label_a": "r2291k", "label_b": "c2516k",
               "outcome_a": "timeout", "margin_a": 0.0}] * 40
    update_from_games(tmp_path / "tri", g_cens, protocol=proto,
                      path=cat)
    assert load_catalog(cat)["reference"]["label"] == ref_label
    # Full data returns: the board must equal board A exactly.
    update_from_games(tmp_path / "tri", g_tri, protocol=proto,
                      path=cat)
    board_c = load_catalog(cat)
    assert board_c["reference"]["label"] == ref_label
    assert {k: m["elo"] for k, m in
            board_c["checkpoints"].items()} == elos_a


def test_censored_edge_does_not_veto_horizon(tmp_path):
    """Round-28 C2: a fully-censored edge is excluded from the fit,
    so its horizon must not veto a later decisive collect at a
    different horizon."""
    from tools.elo_catalog import load_catalog, update_from_games
    cat = tmp_path / "cat.json"
    proto40 = {"procedure": "mcts:32", "max_turns": 40}
    proto100 = {"procedure": "mcts:32", "max_turns": 100}
    g_cens = [{"label_a": "A", "label_b": "B",
               "outcome_a": "timeout", "margin_a": 0.0}] * 12
    g_dec = [{"label_a": "A", "label_b": "B", "outcome_a": "win",
              "margin_a": 0.5}] * 7
    update_from_games(tmp_path / "probe40", g_cens,
                      protocol=proto40, path=cat)
    update_from_games(tmp_path / "verdict", g_dec,
                      protocol=proto100, path=cat)
    c = load_catalog(cat)
    assert c["edges"]["verdict:A~B"]["protocol"]["max_turns"] == 100


def test_censored_probe_collect_is_order_independent(tmp_path):
    """Round-29 C0: whether a fully-censored collect is accepted
    must not depend on collect order -- both orders yield the same
    edge set."""
    from tools.elo_catalog import load_catalog, update_from_games
    g_cens = [{"label_a": "A", "label_b": "B",
               "outcome_a": "timeout", "margin_a": 0.0}] * 12
    g_dec = [{"label_a": "A", "label_b": "B", "outcome_a": "win",
              "margin_a": 0.5}] * 7
    p40 = {"procedure": "mcts:32", "max_turns": 40}
    p100 = {"procedure": "mcts:32", "max_turns": 100}
    edges = {}
    for name, order in (("ab", (("probe", g_cens, p40),
                                ("verdict", g_dec, p100))),
                        ("ba", (("verdict", g_dec, p100),
                                ("probe", g_cens, p40)))):
        cat = tmp_path / f"cat_{name}.json"
        for dirname, games, proto in order:
            update_from_games(tmp_path / dirname, games,
                              protocol=proto, path=cat)
        edges[name] = load_catalog(cat)["edges"]
    assert edges["ab"] == edges["ba"]


def test_transient_censoring_does_not_fake_anchoring(tmp_path):
    """Round-29 C1: during a transient gauge fallback nothing is on
    the reference scale, so disconnected components must NOT flip
    to anchored=True, and the gauge fact must be IN the JSON."""
    from tools.elo_catalog import load_catalog, update_from_games
    cat = tmp_path / "cat.json"
    proto = {"procedure": "mcts:32"}
    g_ref = ([{"label_a": "aref", "label_b": "zzz",
               "outcome_a": "win", "margin_a": 0.5}] * 12
             + [{"label_a": "aref", "label_b": "zzz",
                 "outcome_a": "loss", "margin_a": -0.5}] * 8)
    g_disc = [{"label_a": "ccc", "label_b": "ddd",
               "outcome_a": "win", "margin_a": 0.5}] * 15
    update_from_games(tmp_path / "dref", g_ref, protocol=proto,
                      path=cat)
    update_from_games(tmp_path / "ddisc", g_disc, protocol=proto,
                      path=cat)
    c = load_catalog(cat)
    ref_label = c["reference"]["label"]
    # Pin deliberately (see test above; round-32 C2).
    from tools.elo_catalog import save_catalog
    c["reference"] = {"label": ref_label, "elo": 0.0}
    save_catalog(c, cat)
    assert not c["checkpoints"]["ccc"]["anchored"]
    g_cens = [{"label_a": "aref", "label_b": "zzz",
               "outcome_a": "timeout", "margin_a": 0.0}] * 20
    update_from_games(tmp_path / "dref", g_cens, protocol=proto,
                      path=cat)
    c = load_catalog(cat)
    assert c["reference"]["label"] == ref_label
    assert c.get("gauge", {}).get("fallback") is True
    assert not c["checkpoints"]["ccc"]["anchored"]
    assert not c["checkpoints"]["ddd"]["anchored"]
    update_from_games(tmp_path / "dref", g_ref, protocol=proto,
                      path=cat)
    c = load_catalog(cat)
    assert "gauge" not in c
    assert c["checkpoints"][ref_label]["anchored"]


def test_procedure_correctable_same_dir(tmp_path, capsys):
    """Round-29 C2: a mistyped --catalog-procedure must be
    correctable by a same-dir re-collect (loudly); untagged writes
    stay refused, and OTHER dirs' mismatches stay refused."""
    import pytest
    from tools.elo_catalog import load_catalog, update_from_games
    cat = tmp_path / "cat.json"
    g = [{"label_a": "a", "label_b": "b", "outcome_a": "win",
          "margin_a": 0.5}] * 4
    update_from_games(tmp_path / "d", g, path=cat,
                      protocol={"procedure": "mcts:16"})
    update_from_games(tmp_path / "d", g, path=cat,
                      protocol={"procedure": "mcts:32"})
    c = load_catalog(cat)
    assert c["edges"]["d:a~b"]["protocol"]["procedure"] == "mcts:32"
    assert "retagged" in capsys.readouterr().out
    with pytest.raises(ValueError, match="estimand"):
        update_from_games(tmp_path / "d2", g, path=cat,
                          protocol={"procedure": "mcts:16"})
    with pytest.raises(ValueError):
        update_from_games(tmp_path / "d", g, path=cat,
                          protocol=None)


def test_rename_onto_alias_refused_and_undo_clean(tmp_path):
    """Round-29 C3: renaming a checkpoint onto another node's ALIAS
    silently pooled two checkpoints; refused now. An undo-rename
    must not leave a self-alias."""
    import pytest
    from tools.elo_catalog import (load_catalog, record_edge,
                                   rename_label, resolve_label)
    cat = load_catalog(tmp_path / "cat.json")
    record_edge(cat, "s1", "X", "ref", 7, 0, 3)
    record_edge(cat, "s2", "Y", "ref", 3, 0, 7)
    rename_label(cat, "Y", "Ynew", global_alias=True)
    with pytest.raises(ValueError, match="alias"):
        rename_label(cat, "X", "Y", global_alias=True)
    assert resolve_label(cat, "X") == "X"
    rename_label(cat, "X", "Z", global_alias=True)
    rename_label(cat, "Z", "X", global_alias=True)
    assert resolve_label(cat, "Z") == "X"
    assert resolve_label(cat, "X") == "X"
    assert cat["aliases"].get("X") is None


def test_sticky_horizon_recheck_on_decisive_upgrade(tmp_path):
    """Round-29 C4: when replacement games make a censored probe
    decisive, its sticky declared horizon becomes a live estimand
    -- the plain re-collect must refuse (before any save), not
    silently pool two horizons."""
    import pytest
    from tools.elo_catalog import update_from_games
    g_cens = [{"label_a": "A", "label_b": "B",
               "outcome_a": "timeout", "margin_a": 0.0}] * 12
    g_dec_p = [{"label_a": "A", "label_b": "B", "outcome_a": "win",
                "margin_a": 0.5}] * 5
    g_dec_v = [{"label_a": "C", "label_b": "D", "outcome_a": "win",
                "margin_a": 0.5}] * 7
    cat = tmp_path / "cat.json"
    update_from_games(tmp_path / "probe", g_cens,
                      protocol={"procedure": "mcts:32",
                                "max_turns": 100}, path=cat)
    update_from_games(tmp_path / "verdict", g_dec_v,
                      protocol={"procedure": "mcts:32",
                                "max_turns": 200}, path=cat)
    before = cat.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="carry-forward"):
        update_from_games(tmp_path / "probe", g_dec_p,
                          protocol={"procedure": "mcts:32"},
                          path=cat)
    assert cat.read_text(encoding="utf-8") == before


def test_censored_probe_order_independent_procedure_axis(tmp_path):
    """Round-30 C0: the round-29 order-independence fix covered the
    horizon guard only; the PROCEDURE guard needed the same gate on
    incoming decisive mass."""
    from tools.elo_catalog import load_catalog, update_from_games
    g_cens = [{"label_a": "P", "label_b": "Q",
               "outcome_a": "timeout", "margin_a": 0.0}] * 12
    g_dec = [{"label_a": "A", "label_b": "B", "outcome_a": "win",
              "margin_a": 0.5}] * 7
    edges = {}
    for name, order in (("ab", (("probe", g_cens,
                                 {"procedure": "tcs:plan"}),
                                ("verdict", g_dec,
                                 {"procedure": "mcts:32"}))),
                        ("ba", (("verdict", g_dec,
                                 {"procedure": "mcts:32"}),
                                ("probe", g_cens,
                                 {"procedure": "tcs:plan"})))):
        cat = tmp_path / f"cat_{name}.json"
        for dirname, games, proto in order:
            update_from_games(tmp_path / dirname, games,
                              protocol=proto, path=cat)
        edges[name] = load_catalog(cat)["edges"]
    assert edges["ab"] == edges["ba"]


def test_rename_onto_dangling_alias_target_refused(tmp_path):
    """Round-30 C1: a pruned node's alias TARGET must not be reused
    by an unrelated checkpoint -- the pruned node's historical
    names would silently resolve onto it."""
    import pytest
    from tools.elo_catalog import (load_catalog, rename_label,
                                   resolve_label, save_catalog,
                                   update_from_games)
    cat = tmp_path / "cat.json"
    proto = {"procedure": "mcts:32"}
    g = lambda a: [{"label_a": a, "label_b": "W",
                    "outcome_a": "win", "margin_a": 0.5}] * 4
    update_from_games(tmp_path / "d1", g("A"), protocol=proto,
                      path=cat)
    update_from_games(tmp_path / "d2", g("B"), protocol=proto,
                      path=cat)
    c = load_catalog(cat)
    rename_label(c, "A", "A_new", global_alias=True)
    save_catalog(c, cat)
    # Relabeling re-collect orphans A_new; alias A->A_new dangles.
    update_from_games(tmp_path / "d1", g("runlab"), protocol=proto,
                      path=cat, label_map={"runlab": "C"})
    c = load_catalog(cat)
    assert "A_new" not in c["checkpoints"]
    with pytest.raises(ValueError, match="target"):
        rename_label(c, "B", "A_new")
    assert resolve_label(c, "B") == "B"


def test_default_rename_is_dir_scoped(tmp_path):
    """Round-30 C4: renaming a generic run label (pin/seed) chains
    only the dirs that recorded games under it -- a FUTURE dir
    reusing the label must not silently land on the renamed node.
    A plain re-collect of the renamed dir still chains correctly."""
    from tools.elo_catalog import (load_catalog, rename_label,
                                   save_catalog, update_from_games)
    cat = tmp_path / "cat.json"
    proto = {"procedure": "mcts:32"}
    g = [{"label_a": "pin", "label_b": "seed", "outcome_a": "win",
          "margin_a": 0.5}] * 5
    update_from_games(tmp_path / "l6", g, protocol=proto, path=cat)
    c = load_catalog(cat)
    rename_label(c, "pin", "l6-canonical")
    save_catalog(c, cat)
    assert "pin" not in c.get("aliases", {})
    # Plain re-collect of the SAME dir chains onto the renamed node.
    update_from_games(tmp_path / "l6", g, protocol=proto, path=cat)
    c = load_catalog(cat)
    assert "l6:l6-canonical~seed" in c["edges"]
    assert "pin" not in c["checkpoints"]
    # A DIFFERENT leg reusing 'pin' stays its own (run-local) node.
    update_from_games(tmp_path / "l7", g, protocol=proto, path=cat)
    c = load_catalog(cat)
    assert "l7:pin~seed" in c["edges"]
    assert c["checkpoints"]["l6-canonical"]["n_games"] == 5


def test_gauge_follows_rename_and_clears_when_nothing_rated(tmp_path):
    """Round-30 C2/C3: the persisted gauge label follows a rename,
    and a refit where NOTHING is rated drops the gauge record."""
    from tools.elo_catalog import (load_catalog, rename_label,
                                   save_catalog, update_from_games)
    cat = tmp_path / "cat.json"
    proto = {"procedure": "mcts:32"}
    g_ref_cens = [{"label_a": "aref", "label_b": "zzz",
                   "outcome_a": "timeout", "margin_a": 0.0}] * 8
    g_dec = [{"label_a": "ccc", "label_b": "ddd",
              "outcome_a": "win", "margin_a": 0.5}] * 6
    update_from_games(tmp_path / "dref", g_ref_cens, protocol=proto,
                      path=cat)
    update_from_games(tmp_path / "ddec", g_dec, protocol=proto,
                      path=cat)
    c = load_catalog(cat)
    if c.get("gauge"):
        gl = c["gauge"]["label"]
        rename_label(c, gl, gl + "_ren")
        assert c["gauge"]["label"] == gl + "_ren"
        save_catalog(c, cat)
    # Censor the decisive dir too: nothing rated -> gauge dropped.
    g_dec_cens = [{"label_a": "ccc", "label_b": "ddd",
                   "outcome_a": "timeout", "margin_a": 0.0}] * 6
    update_from_games(tmp_path / "ddec", g_dec_cens, protocol=proto,
                      path=cat)
    c = load_catalog(cat)
    assert "gauge" not in c


def test_dir_scoped_rename_no_global_leak_via_self_alias(tmp_path):
    """Round-31 C0: an undo-rename's self-alias must be dropped on
    the DIR-SCOPED path too -- the next rename re-pointed it into a
    real global alias, silently chaining an unrelated future dir."""
    from tools.elo_catalog import (load_catalog, rename_label,
                                   resolve_label, save_catalog,
                                   update_from_games)
    cat = tmp_path / "cat.json"
    proto = {"procedure": "mcts:32"}
    g = lambda a, b: [{"label_a": a, "label_b": b,
                       "outcome_a": "win", "margin_a": 0.5}] * 5
    update_from_games(tmp_path / "d1", g("pin", "seed"),
                      protocol=proto, path=cat)
    c = load_catalog(cat)
    rename_label(c, "pin", "L5-100k", global_alias=True)
    rename_label(c, "L5-100k", "pin")            # undo, dir-scoped
    rename_label(c, "pin", "L5-120k")            # corrective
    assert "pin" not in c.get("aliases", {})
    assert resolve_label(c, "pin") == "pin"
    save_catalog(c, cat)
    # An unrelated later dir reusing 'pin' stays its own node.
    update_from_games(tmp_path / "d2", g("pin", "r2"),
                      protocol=proto, path=cat)
    c = load_catalog(cat)
    assert "d2:pin~r2" in c["edges"]


def test_documented_recollect_survives_dir_scoped_rename(tmp_path):
    """Round-31 C1: re-running the dir's documented collect command
    (--catalog-alias pin=OLD) after a dir-scoped rename OLD->NEW
    must land on NEW, not resurrect OLD -- and must keep doing so
    on the collect after that (the stamp survives re-persist)."""
    from tools.elo_catalog import (load_catalog, rename_label,
                                   save_catalog, update_from_games)
    cat = tmp_path / "cat.json"
    proto = {"procedure": "mcts:32"}
    g = [{"label_a": "pin", "label_b": "seed", "outcome_a": "win",
          "margin_a": 0.5}] * 5
    update_from_games(tmp_path / "d", g, protocol=proto, path=cat,
                      label_map={"pin": "OLD"})
    c = load_catalog(cat)
    rename_label(c, "OLD", "NEWNAME")
    save_catalog(c, cat)
    for _ in range(2):
        update_from_games(tmp_path / "d", g, protocol=proto,
                          path=cat, label_map={"pin": "OLD"})
        c = load_catalog(cat)
        assert set(c["edges"]) == {"d:NEWNAME~seed"}, c["edges"]
        assert "OLD" not in c["checkpoints"]


def test_fresh_catalog_gauge_is_order_independent(tmp_path):
    """Round-31 C3: on a fresh catalog the auto-designated
    reference is PROVISIONAL -- once the canonical reference gains
    decisive mass, both collect orders yield the same board."""
    from tools.elo_catalog import REFERENCE_LABEL, load_catalog, \
        update_from_games
    proto = {"procedure": "mcts:32"}
    g_ab = ([{"label_a": "A", "label_b": "B", "outcome_a": "win",
              "margin_a": 0.5}] * 7
            + [{"label_a": "A", "label_b": "B",
                "outcome_a": "loss", "margin_a": -0.5}] * 3)
    g_ref = ([{"label_a": REFERENCE_LABEL, "label_b": "A",
               "outcome_a": "win", "margin_a": 0.5}] * 6
             + [{"label_a": REFERENCE_LABEL, "label_b": "A",
                 "outcome_a": "loss", "margin_a": -0.5}] * 4)
    boards = {}
    for name, order in (("ab_first", ("dab", "dref")),
                        ("ref_first", ("dref", "dab"))):
        cat = tmp_path / f"cat_{name}.json"
        for dirname in order:
            update_from_games(
                tmp_path / dirname,
                g_ab if dirname == "dab" else g_ref,
                protocol=proto, path=cat)
        c = load_catalog(cat)
        boards[name] = (c["reference"]["label"],
                        {k: m["elo"]
                         for k, m in c["checkpoints"].items()})
    assert boards["ab_first"] == boards["ref_first"]


def test_stale_map_value_not_reattributed_across_renames(tmp_path):
    """Round-32 C0: the persisted map stores RESOLVED targets, and
    a rename rewrites map values only on edges that reference the
    renamed node -- so an unrelated later rename of a same-named
    node can no longer re-attribute a dir's games to a different
    checkpoint."""
    from tools.elo_catalog import (load_catalog, rename_label,
                                   save_catalog, update_from_games)
    cat = tmp_path / "cat.json"
    proto = {"procedure": "mcts:32"}
    gd = [{"label_a": "cand", "label_b": "ref", "outcome_a": "win",
           "margin_a": 0.5}] * 10
    ge = [{"label_a": "pin", "label_b": "ref", "outcome_a": "loss",
           "margin_a": -0.5}] * 8
    update_from_games(tmp_path / "dirD", gd, protocol=proto,
                      path=cat, label_map={"cand": "pin"})
    c = load_catalog(cat)
    rename_label(c, "pin", "leg5-pin")
    save_catalog(c, cat)
    # Documented re-collect with the ORIGINAL command.
    update_from_games(tmp_path / "dirD", gd, protocol=proto,
                      path=cat, label_map={"cand": "pin"})
    c = load_catalog(cat)
    assert c["edges"]["dirD:leg5-pin~ref"]["label_map"]["cand"] == \
        "leg5-pin"
    # Unrelated dir E reuses run label 'pin'; its own rename must
    # not touch dirD.
    update_from_games(tmp_path / "dirE", ge, protocol=proto,
                      path=cat)
    c = load_catalog(cat)
    rename_label(c, "pin", "leg6-pin")
    save_catalog(c, cat)
    update_from_games(tmp_path / "dirD", gd, protocol=proto,
                      path=cat)               # plain re-collect
    c = load_catalog(cat)
    assert "dirD:leg5-pin~ref" in c["edges"]
    assert "leg5-pin" in c["checkpoints"]
    assert "leg6-pin" in c["checkpoints"]


def test_reused_name_tombstone_cannot_pool_checkpoints(tmp_path,
                                                       capsys):
    """Round-33 C0: a dir-scoped rename tombstone whose key is a
    LIVE checkpoint (name freed and reused) is suppressed loudly,
    and persisted maps are PER-EDGE -- the six-step rename shuffle
    must end with two distinct edges and both checkpoints alive."""
    from tools.elo_catalog import (load_catalog, rename_label,
                                   save_catalog, update_from_games)
    cat = tmp_path / "cat.json"
    proto = {"procedure": "mcts:32"}
    gp = [{"label_a": "runP", "label_b": "seed",
           "outcome_a": "win", "margin_a": 0.5}] * 10
    gq = [{"label_a": "runQ", "label_b": "seed",
           "outcome_a": "loss", "margin_a": -0.5}] * 10
    update_from_games(tmp_path / "D", gp + gq, protocol=proto,
                      path=cat, label_map={"runP": "aaa",
                                           "runQ": "bbb"})
    c = load_catalog(cat)
    rename_label(c, "aaa", "kkk")            # operator error
    save_catalog(c, cat)
    update_from_games(tmp_path / "D", gp + gq, protocol=proto,
                      path=cat)              # plain heal
    c = load_catalog(cat)
    rename_label(c, "kkk", "zzz")            # correction
    save_catalog(c, cat)
    c = load_catalog(cat)
    rename_label(c, "bbb", "kkk")            # reuse the freed name
    save_catalog(c, cat)
    update_from_games(tmp_path / "D", gp + gq, protocol=proto,
                      path=cat)              # plain re-collect
    c = load_catalog(cat)
    assert "D:kkk~seed" in c["edges"], sorted(c["edges"])
    assert "D:seed~zzz" in c["edges"], sorted(c["edges"])
    assert "kkk" in c["checkpoints"] and "zzz" in c["checkpoints"]


def test_freed_name_reuse_order_variant(tmp_path):
    """Round-34 C0: the OTHER rename order (free a name, reuse it,
    then re-collect WITH the documented --catalog-alias map) must
    also yield two distinct edges -- the round-33 live-key veto
    fixed one order by breaking this one; the per-edge redirect
    fixes both."""
    from tools.elo_catalog import (load_catalog, rename_label,
                                   save_catalog, update_from_games)
    cat = tmp_path / "cat.json"
    proto = {"procedure": "mcts:32"}
    gp = [{"label_a": "runP", "label_b": "seed",
           "outcome_a": "win", "margin_a": 0.5}] * 10
    gq = [{"label_a": "runQ", "label_b": "seed",
           "outcome_a": "loss", "margin_a": -0.5}] * 10
    lmap = {"runP": "aaa", "runQ": "bbb"}
    update_from_games(tmp_path / "D", gp + gq, protocol=proto,
                      path=cat, label_map=lmap)
    c = load_catalog(cat)
    rename_label(c, "aaa", "zzz")
    rename_label(c, "bbb", "aaa")        # reuse the freed name
    save_catalog(c, cat)
    update_from_games(tmp_path / "D", gp + gq, protocol=proto,
                      path=cat, label_map=lmap)
    c = load_catalog(cat)
    assert "D:aaa~seed" in c["edges"], sorted(c["edges"])
    assert "D:seed~zzz" in c["edges"], sorted(c["edges"])
    assert c["edges"]["D:seed~zzz"]["wins_a"] + \
        c["edges"]["D:seed~zzz"]["wins_b"] == 10
    assert "zzz" in c["checkpoints"] and "aaa" in c["checkpoints"]


def test_partial_recollect_after_reuse_not_refused(tmp_path):
    """Round-35 C1: the collision guard compares POST-REDIRECT
    finals -- a partial --catalog-alias re-collect after a
    free-and-reuse rename lands correctly instead of being refused
    (and the genuine no-tombstone collision still refuses)."""
    import pytest
    from tools.elo_catalog import (load_catalog, rename_label,
                                   save_catalog, update_from_games)
    cat = tmp_path / "cat.json"
    proto = {"procedure": "mcts:32"}
    gp = [{"label_a": "runP", "label_b": "seed",
           "outcome_a": "win", "margin_a": 0.5}] * 10
    gq = [{"label_a": "runQ", "label_b": "seed",
           "outcome_a": "loss", "margin_a": -0.5}] * 10
    update_from_games(tmp_path / "D", gp + gq, protocol=proto,
                      path=cat, label_map={"runP": "aaa",
                                           "runQ": "bbb"})
    c = load_catalog(cat)
    rename_label(c, "aaa", "zzz")
    rename_label(c, "bbb", "aaa")
    save_catalog(c, cat)
    update_from_games(tmp_path / "D", gp + gq, protocol=proto,
                      path=cat, label_map={"runP": "aaa"})
    c = load_catalog(cat)
    assert "D:aaa~seed" in c["edges"]
    assert "D:seed~zzz" in c["edges"]
    assert "zzz" in c["checkpoints"]
    # The genuine collision (two run labels, one target, no
    # tombstones) still refuses.
    g2 = [{"label_a": "x1", "label_b": "w", "outcome_a": "win",
           "margin_a": 0.5}] * 4
    update_from_games(tmp_path / "E", g2, protocol=proto,
                      path=cat, label_map={"x1": "T1"})
    g3 = [{"label_a": "x2", "label_b": "w", "outcome_a": "win",
           "margin_a": 0.5}] * 4
    with pytest.raises(ValueError, match="resolve"):
        update_from_games(tmp_path / "E", g2 + g3, protocol=proto,
                          path=cat, label_map={"x2": "T1"})
