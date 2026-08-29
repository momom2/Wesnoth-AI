"""No-result contract tests (user ruling 2026-08-17: a capped game
is NOT a draw -- it is a truncated observation, excluded from the
PURE fit, recorded as an absence, and replaced by run_elo_batch up
to a hard guard). Production code paths on synthetic games/files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.elo_collect import build_pairs                # noqa: E402
from tools.elo_catalog import load_catalog, update_from_games  # noqa: E402
from tools.run_elo_batch import (                        # noqa: E402
    result_name, scan_slots, slot_for,
)


def _g(outcome, a="new", b="old", margin=0.0):
    return {"label_a": a, "label_b": b, "outcome_a": outcome,
            "margin_a": margin}


def test_build_pairs_excludes_no_result_from_pure():
    games = ([_g("win")] * 6 + [_g("loss")] * 2
             + [_g("draw", margin=0.4)] * 5      # capped, A ahead
             + [_g("timeout", margin=-0.4)])     # capped, B ahead
    labels, pure, mat, nores = build_pairs(games, eps=0.02)
    (rec,) = pure.values()
    # PURE: only the 8 decisive games; zero draws ever.
    assert (rec.wins_i, rec.draws, rec.wins_j) == (6, 0, 2)
    assert nores[(0, 1)] == 6
    # MATERIAL-SIGN diagnostic still adjudicates the absences.
    (mrec,) = mat.values()
    assert (mrec.wins_i, mrec.draws, mrec.wins_j) == (11, 0, 3)


def test_catalog_edge_records_absences_and_fit_ignores_them(tmp_path):
    p = tmp_path / "cat.json"
    games = ([_g("win")] * 20 + [_g("loss")] * 10 + [_g("draw")] * 30)
    update_from_games(Path("eval_games/capfest"), games, path=p)
    cat = load_catalog(p)
    (edge,) = cat["edges"].values()
    assert (edge["wins_a"], edge["draws"], edge["wins_b"]) == (20, 0, 10)
    assert edge["no_result"] == 30
    # Rating games = decisive only; absences widen nothing silently.
    assert cat["checkpoints"]["new"]["n_games"] == 30
    assert cat["checkpoints"]["new"]["elo"] > cat["checkpoints"]["old"]["elo"]


def _write(outdir, i, outcome, seed_base=10_000, a="A", b="B", gen=0):
    from tools.run_elo_batch import replacement_slot_for
    side, seed = replacement_slot_for(i, seed_base, gen)
    path = outdir / result_name(a, b, side, seed)
    path.write_text(json.dumps({"outcome_a": outcome}), encoding="utf-8")
    return path


def test_scan_slots_schedules_replacements_up_to_guard(tmp_path):
    # 6 base slots: 2 decisive, 3 no-result, 1 unplayed. Round-30
    # C5: a replacement is the capped slot's next same-side
    # GENERATION, not an appended index.
    from tools.run_elo_batch import replacement_slot_for
    for i, oc in enumerate(["win", "draw", "loss", "draw", "draw"]):
        _write(tmp_path, i, oc)
    n_res, n_nr, pending, extra = scan_slots(
        tmp_path, "A", "B", games=6, seed_base=10_000, max_extra=2)
    assert (n_res, n_nr) == (2, 3)
    assert extra == 2                      # guard binds below demand (3)
    # Pending = unplayed base slot 5 (classification pass), then
    # gen-1 replacements of capped slots 1 and 3 granted by the
    # budget pass (slot 4's would exceed the guard).
    assert [(s[0], s[4]) for s in pending] == [(5, 0), (1, 1), (3, 1)]
    # Same deterministic derivation, same SIDE as the slot replaced
    # (pending[1] is slot 1's gen-1 replacement).
    assert pending[1][1:3] == replacement_slot_for(1, 10_000, 1)
    assert pending[1][1] == slot_for(1, 10_000)[0]


def test_scan_slots_replacement_chain_is_bounded_and_resumable(tmp_path):
    # Every played game caps -- worst case. Base 4, guard 3.
    for i in range(4):
        _write(tmp_path, i, "draw")
    n_res, n_nr, pending, extra = scan_slots(
        tmp_path, "A", "B", games=4, seed_base=10_000, max_extra=3)
    assert (n_res, n_nr, extra) == (0, 4, 3)
    assert [(s[0], s[4]) for s in pending] == [(0, 1), (1, 1), (2, 1)]
    # A replacement that ALSO caps consumes the guard without growing
    # the chain past it: play two gen-1 replacements as caps, re-scan.
    _write(tmp_path, 0, "draw", gen=1)
    _write(tmp_path, 1, "draw", gen=1)
    n_res, n_nr, pending, extra = scan_slots(
        tmp_path, "A", "B", games=4, seed_base=10_000, max_extra=3)
    assert (n_res, n_nr, extra) == (0, 6, 3)
    # Guard spent chain-first: slot 0's gen-2 replacement is the
    # only pending slot (slots 1-3 stop at the exhausted guard).
    assert [(s[0], s[4]) for s in pending] == [(0, 2)]
