"""The labels of a self-play run (tools/pg_labels.py): b and beta recover a
planted baseline and luck slope, a game's two labels are opposite, capped
games are labelled at z = 0 and enter the fits, and no game is labelled by
a fit that saw it."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.match_luck import COLUMNS  # noqa: E402
from tools.pg_labels import fit_baseline, half_of, side_labels  # noqa: E402
from tools.selfplay_rows import cell_of  # noqa: E402

MAPS = ("m1", "m2")
FACTIONS = ("Drakes", "Undead", "Rebels")
BETA = np.array([0.3, -0.2, 0.0, 0.1, 0.05, 0.4, -0.1])


def planted_world(seed: int = 0):
    """The planted baseline of every side-1 cell; a side-2 cell's is its
    mirror's negative."""
    rng = np.random.default_rng(seed)
    a = {m: rng.normal(0, 0.3) for m in MAPS}
    f = {(x, y): rng.normal(0, 0.2) for x in FACTIONS for y in FACTIONS}
    f = {(x, y): (f[(x, y)] - f[(y, x)]) / 2 for x, y in f}              # antisymmetric pairs
    d = {(m, x, y): rng.normal(0, 0.15) for m in MAPS for x in FACTIONS for y in FACTIONS}
    return {cell_of(m, 1, x, y): a[m] + f[(x, y)] + d[(m, x, y)]
            for m in MAPS for x in FACTIONS for y in FACTIONS}


def synthetic_rows(n_games: int, world, seed: int = 1, cap_every: int = 0, noise: float = 0.5):
    """Side rows of `n_games` games: z(side 1) = mu(cell) + BETA . S + noise,
    side 2 its mirror; every `cap_every`-th game capped at z = 0."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_games):
        m = MAPS[rng.integers(len(MAPS))]
        f1, f2 = FACTIONS[rng.integers(3)], FACTIONS[rng.integers(3)]
        S = rng.normal(0, 1, len(COLUMNS))
        capped = bool(cap_every) and i % cap_every == 0
        z = 0.0 if capped else world[cell_of(m, 1, f1, f2)] + BETA @ S + rng.normal(0, noise)
        for side, sign, own, opp in ((1, 1, f1, f2), (2, -1, f2, f1)):
            rows.append({"game": f"g{i}", "index": i, "side": side, "z": sign * z, "capped": capped,
                         "cell": cell_of(m, side, own, opp), "S": list(sign * S)})
    return rows


def test_the_fit_recovers_the_planted_cells_and_slope():
    world = planted_world()
    fit = fit_baseline(synthetic_rows(8000, world))
    assert fit.beta == pytest.approx(BETA, abs=0.03)
    for cell, mu in world.items():
        assert fit.b(cell) == pytest.approx(mu, abs=0.1), cell
        mirror = cell.split("|")
        assert fit.b(f"{mirror[0]}|2|{mirror[3]}|{mirror[2]}") == pytest.approx(-mu, abs=0.1)


def test_labels_are_opposite_across_seats_and_capped_games_enter_at_zero():
    rows = synthetic_rows(1200, planted_world(), cap_every=7)
    labels, info = side_labels(rows, heldout_from=800)
    assert len(labels) == len(rows) and not info["left_out"]
    by_game = {}
    for lab in labels:
        by_game.setdefault(lab["game"], {})[lab["side"]] = lab
    for pair in by_game.values():
        assert pair[1]["R"] == pytest.approx(-pair[2]["R"], abs=1e-9)
        assert pair[1]["R"] == pytest.approx(pair[1]["z"] - pair[1]["b"] - pair[1]["beta_S"])
    capped = [lab for lab in labels if lab["capped"]]
    assert capped and all(lab["z"] == 0.0 for lab in capped)
    # A capped held-out game is fitted on: its luck sums move the fit.
    i = next(r["index"] for r in rows if r["capped"] and r["index"] >= 800)
    moved = [dict(r, S=[x * 50 for x in r["S"]]) if r["index"] == i else r for r in rows]
    assert side_labels(moved, 800)[1]["fits"]["train"]["beta"] != info["fits"]["train"]["beta"]


def test_split_half_and_heldout_columns():
    labels, _ = side_labels(synthetic_rows(40, planted_world()), heldout_from=24)
    for lab in labels:
        i = int(lab["game"][1:])
        assert lab["split"] == i % 8 and lab["heldout"] == (i >= 24)
        assert lab["half"] == (("A" if (i // 2) % 2 == 0 else "B") if i >= 24 else None)
    assert {lab["half"] for lab in labels if lab["heldout"]} == {"A", "B"}
    # Both seats of a game share its split, its held-out flag and its half:
    # a game's two labels share one result, so the halves stay independent.
    by_game = {}
    for lab in labels:
        by_game.setdefault(lab["game"], set()).add((lab["split"], lab["heldout"], lab["half"]))
    assert all(len(v) == 1 for v in by_game.values())
    assert all(sum(lab["game"] == g for lab in labels) == 2 for g in by_game)


def test_no_game_is_labelled_by_a_fit_that_saw_it():
    """Moving one game's result moves no label of that game but its z, and
    moves exactly the labels whose fits read it: a training game's moves
    none; a held-out game of half A moves the training games' and half
    B's."""
    rows = synthetic_rows(400, planted_world())
    base, _ = side_labels(rows, heldout_from=240)
    for i in (17, next(j for j in range(240, 400) if half_of(j) == "A")):
        moved_rows = [dict(r, z=r["z"] + (50.0 if r["side"] == 1 else -50.0)) if r["index"] == i else r
                      for r in rows]
        moved, _ = side_labels(moved_rows, heldout_from=240)
        changed = set()
        for old, new in zip(base, moved):
            if abs(old["b"] - new["b"]) > 1e-12 or abs(old["beta_S"] - new["beta_S"]) > 1e-12:
                changed.add((old["game"], old["side"]))
            if old["game"] == f"g{i}":
                assert new["z"] - old["z"] == pytest.approx(50.0 if old["side"] == 1 else -50.0)
        assert changed == {(lab["game"], lab["side"]) for lab in base if _reads(lab, i)}
        assert (f"g{i}", 1) not in changed and (f"g{i}", 2) not in changed


def _reads(label: dict, i: int) -> bool:
    """Whether the fit that labels `label` reads held-out game i."""
    if i < 240:
        return False
    return not label["heldout"] or label["half"] != half_of(i)


def sparse_world_rows(seed: int = 0, games: int = 5000, tau: float = 0.15, sigma: float = 0.8):
    """A held-out half at production sparsity: 21 maps, 6 factions, 5,000
    games, about 6.6 rows a side-cell; cell deviations of sd `tau` around an
    additive map and faction-pair baseline, result noise of sd `sigma`:
    (the planted b of every side-1 cell, the side rows)."""
    maps, factions = [f"m{i}" for i in range(21)], [f"f{i}" for i in range(6)]
    rng = np.random.default_rng(seed)
    a = {m: rng.normal(0, 0.1) for m in maps}
    f = {(x, y): rng.normal(0, 0.1) for x in factions for y in factions}
    f = {(x, y): (f[(x, y)] - f[(y, x)]) / 2 for x, y in f}
    world = {cell_of(m, 1, x, y): a[m] + f[(x, y)] + rng.normal(0, tau) for m in maps for x in factions
             for y in factions}
    rows = []
    for i in range(games):
        m, f1, f2 = maps[rng.integers(21)], factions[rng.integers(6)], factions[rng.integers(6)]
        S = rng.normal(0, 1, len(COLUMNS))
        z = world[cell_of(m, 1, f1, f2)] + BETA @ S + rng.normal(0, sigma)
        for side, sign, own, opp in ((1, 1, f1, f2), (2, -1, f2, f1)):
            rows.append({"game": f"g{i}", "index": i, "side": side, "z": sign * z, "capped": False,
                         "cell": cell_of(m, side, own, opp), "S": list(sign * S)})
    return world, rows


def test_the_shrinkage_is_near_the_oracle_at_production_sparsity():
    """At 6.6 rows a side-cell the cell means are mostly noise: the fitted k
    lands near the planted sigma^2 / tau^2, and b's error against the
    planted cells is within 6 % of the fit at the true k, which beats both
    raw cell means (k = 0) and the additive baseline alone (k infinite).
    A k ten times too large costs 13 % here."""
    world, rows = sparse_world_rows()
    true_k = 0.8 ** 2 / 0.15 ** 2

    def error(fit):
        return float(np.mean([(fit.b(c) - mu) ** 2 for c, mu in world.items()]))

    fit = fit_baseline(rows)
    oracle = error(fit_baseline(rows, k=true_k))
    assert true_k / 2.5 <= fit.k <= true_k * 2.5, fit.k
    assert error(fit) <= 1.06 * oracle, (error(fit), oracle)
    assert oracle <= 0.3 * error(fit_baseline(rows, k=0.0))
    assert oracle <= 0.95 * error(fit_baseline(rows, k=1e12))
