"""The enemy's faction as a player can know it: one-hot when the opponent
chose it openly; under Random, uniform over the era's factions that can
field every enemy unit type the side has seen, its advancements included;
a seen set no faction can field keeps the prior and is counted."""
from __future__ import annotations

import numpy as np

from wesnoth_ai.constants import DEFAULT_FACTIONS
from wesnoth_ai.faction_posterior import faction_posterior, posterior_counts, reset_posterior_counts

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
    assert posterior_counts() == {"posteriors": 1, "inconsistent": 1}
