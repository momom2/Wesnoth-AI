"""The enemy's faction as a player can know it: one-hot when the opponent
chose it openly; under Random, uniform over the factions its draw could
give (the era's, less the side's own under "No Mirror") that can field
every informative enemy unit type the side has seen, its advancements
included; a seen set no faction can field keeps the prior and is counted,
and so is a posterior that leaves out the truth. The replay's record
carries what the prior needs to the posterior of every decision."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from wesnoth_ai.constants import DEFAULT_FACTIONS  # noqa: E402
from wesnoth_ai.faction_posterior import (faction_posterior, posterior_counts,  # noqa: E402
                                          reset_posterior_counts)

FACTION_IDS = {name: i for i, name in enumerate(DEFAULT_FACTIONS)}
FACTION_IDS["Dunefolk"] = len(FACTION_IDS)
DEFAULT_ERA = tuple(DEFAULT_FACTIONS[1:])


def _support(probs: np.ndarray) -> dict:
    names = {i: n for n, i in FACTION_IDS.items()}
    return {names[i]: round(float(p), 4) for i, p in enumerate(probs) if p > 0}


def test_an_open_choice_is_known():
    probs = faction_posterior("Undead", False, DEFAULT_ERA, [], FACTION_IDS)
    assert _support(probs) == {"Undead": 1.0}


def test_random_with_nothing_seen_is_uniform_over_the_era():
    assert _support(faction_posterior("Drakes", True, DEFAULT_ERA, [], FACTION_IDS)) == \
        {f: round(1 / 6, 4) for f in DEFAULT_ERA}
    dune = DEFAULT_ERA + ("Dunefolk",)
    assert len(_support(faction_posterior("Drakes", True, dune, [], FACTION_IDS))) == 7


def test_what_the_side_saw_narrows_it():
    mage = faction_posterior("Rebels", True, DEFAULT_ERA, ["Mage"], FACTION_IDS)
    assert _support(mage) == {"Loyalists": 0.5, "Rebels": 0.5}
    white = faction_posterior("Rebels", True, DEFAULT_ERA, ["White Mage"], FACTION_IDS)
    assert _support(white) == {"Loyalists": 0.5, "Rebels": 0.5}          # an advancement of the Mage
    whelp = faction_posterior("Northerners", True, DEFAULT_ERA, ["Troll Whelp"], FACTION_IDS)
    assert _support(whelp) == {"Northerners": 1.0}
    corpse = faction_posterior("Undead", True, DEFAULT_ERA, ["Walking Corpse"], FACTION_IDS)
    assert _support(corpse) == {"Undead": 1.0}


def test_an_impossible_sighting_keeps_the_prior_and_is_counted():
    reset_posterior_counts()
    probs = faction_posterior("Rebels", True, DEFAULT_ERA, ["Mage", "Troll Whelp"], FACTION_IDS)
    assert len(_support(probs)) == 6
    assert posterior_counts() == {"posteriors": 1, "inconsistent": 1, "excludes_truth": 0}


def test_under_no_mirror_a_random_side_cannot_hold_the_sides_own_faction():
    others = {f: round(1 / 5, 4) for f in DEFAULT_ERA if f != "Loyalists"}
    assert _support(faction_posterior("Undead", True, DEFAULT_ERA, [], FACTION_IDS, own_faction="Loyalists",
                                      random_faction_mode="No Mirror")) == others
    for mode in ("Independent", "No Ally Mirror"):             # a 1v1's two sides are not allies
        assert len(_support(faction_posterior("Undead", True, DEFAULT_ERA, [], FACTION_IDS,
                                              own_faction="Loyalists", random_faction_mode=mode))) == 6


def test_a_type_no_faction_of_the_era_fields_is_not_evidence():
    """Hornshark Island gives the Loyalists a Young Ogre and a Sergeant."""
    reset_posterior_counts()
    probs = faction_posterior("Loyalists", True, DEFAULT_ERA, ["Young Ogre", "Sergeant", "Bowman"], FACTION_IDS)
    assert _support(probs) == {"Loyalists": 1.0}
    assert posterior_counts() == {"posteriors": 1, "inconsistent": 0, "excludes_truth": 0}


def test_a_posterior_that_leaves_out_the_true_faction_is_counted():
    reset_posterior_counts()
    probs = faction_posterior("Undead", True, DEFAULT_ERA, ["Troll Whelp"], FACTION_IDS)
    assert _support(probs) == {"Northerners": 1.0}
    assert posterior_counts()["excludes_truth"] == 1


def test_a_posterior_whose_candidates_miss_the_truth_is_counted():
    """A mirror under No Mirror cannot happen in the engine; a posterior
    that nonetheless leaves the truth out is an error the barrier counts."""
    reset_posterior_counts()
    faction_posterior("Loyalists", True, DEFAULT_ERA, [], FACTION_IDS, own_faction="Loyalists",
                      random_faction_mode="No Mirror")
    assert posterior_counts()["excludes_truth"] == 1


def test_the_record_carries_the_prior_to_every_decision(tmp_path):
    """A replay of the Dunefolk era under "No Mirror", side 1 Undead by a
    Random choice, side 2 Loyalists chosen openly: before either side has
    seen the other, side 2's posterior spreads over the six factions its
    enemy could have drawn, and side 1's is one-hot."""
    from helpers.synthetic_replay import side_block, turn, write_replay
    from tools.replay_dataset import _build_initial_gamestate
    from tools.replay_extract import extract_replay
    from wesnoth_ai import game_core as gc
    if gc.game_core_class() is None:
        pytest.skip("wesnoth_core.GameCore not available")
    sides = [side_block(1, "alice", [("Dark Sorcerer", 2, 4, True)], chose_random=True, faction="Undead"),
             side_block(2, "bob", [("Lieutenant", 11, 4, True)], chose_random=False)]
    record = extract_replay(write_replay(tmp_path / "g.bz2", sides, [*turn(1), *turn(2)],
                                         multiplayer={"random_faction_mode": '"No Mirror"'},
                                         era_id="era_dunefolk"))
    assert (record["era_id"], record["random_faction_mode"]) == ("era_dunefolk", "No Mirror")
    ids = dict(FACTION_IDS)
    cs = gc.CoreState.from_state(_build_initial_gamestate(record))
    side2 = _support(cs._faction_probs(2, "Loyalists", "Undead", ids))
    assert side2 == {f: round(1 / 6, 4) for f in (*DEFAULT_ERA, "Dunefolk") if f != "Loyalists"}
    assert _support(cs._faction_probs(1, "Undead", "Loyalists", ids)) == {"Loyalists": 1.0}
