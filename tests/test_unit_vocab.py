"""A fresh network's unit-type vocabulary (tools/unit_vocab.py): every
unit type that can take part in our games has its own embedding row,
and a vocabulary that would put two names on the overflow row is
refused at seeding and reported at loading."""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

from tools import unit_vocab  # noqa: E402
from wesnoth_ai.encoder import (MAX_UNIT_TYPES, GameStateEncoder,  # noqa: E402
                                encode_raw, names_on_overflow_row)


def test_every_recruit_of_every_faction_reads_its_own_row():
    """The recruits of a Northerners position, the faction whose recruits
    shared one row in the vocabularies seeded from all of unit_stats.json,
    encode to seven distinct ids, none on the overflow row."""
    from sim_test_helpers import require_scenario_data
    from tools.scenario_pool import build_scenario_gamestate, load_factions, random_setup
    require_scenario_data()
    enc = GameStateEncoder(d_model=8)
    unit_vocab.seed_vocab(enc)
    vocab = enc.unit_type_to_id
    assert names_on_overflow_row(vocab) == []
    factions = load_factions()
    for faction in (factions if isinstance(factions, list) else factions.values()):
        for name in list(faction.recruit) + list(faction.leader_pool):
            assert vocab[name] < MAX_UNIT_TYPES - 1, (faction.name, name)
    gs = build_scenario_gamestate(random_setup(random.Random(7), forced_faction="Northerners"))
    raw = encode_raw(gs, type_to_id=vocab, faction_to_id=enc.faction_to_id)
    assert len(set(raw.recruit_type_ids)) == len(raw.recruit_type_ids) == 7


def test_a_plague_corpses_variation_reads_its_base_row_on_every_path(caplog):
    """A Walking Corpse raised from a merman (`Walking Corpse:swimmer`)
    encodes to the Walking Corpse row both through the vocabulary the
    pre-encoder is handed and through the live encoder, which registers
    names as it meets them; a name the vocabulary lacks takes the
    overflow row on both paths instead of a row training never updates."""
    from sim_test_helpers import require_scenario_data
    from tools.scenario_pool import build_scenario_gamestate, random_setup
    require_scenario_data()
    enc = GameStateEncoder(d_model=8)
    unit_vocab.seed_vocab(enc)
    vocab = enc.unit_type_to_id
    assert vocab["Walking Corpse:swimmer"] == vocab["Walking Corpse"] < MAX_UNIT_TYPES - 1
    assert vocab["Soulless:bat"] == vocab["Soulless"]
    gs = build_scenario_gamestate(random_setup(random.Random(3)))
    mover = gs.global_info.current_side
    own = next(u for u in gs.map.units if u.side == mover)   # always visible to its side

    def encoded_id(name):
        own.name = name
        enc.register_names(gs)
        raw = encode_raw(gs, type_to_id=vocab, faction_to_id=enc.faction_to_id)
        return int(dict(zip(raw.unit_ids, raw.unit_type_ids))[own.id])

    assert encoded_id("Walking Corpse:swimmer") == vocab["Walking Corpse"]
    with caplog.at_level(logging.WARNING):
        assert encoded_id("Nobody Knows This Unit") == MAX_UNIT_TYPES - 1
    assert "Nobody Knows This Unit" not in vocab and "frozen" in caplog.text


def test_seeding_refuses_a_set_that_would_share_the_overflow_row(monkeypatch):
    names = [f"Type {i:03d}" for i in range(MAX_UNIT_TYPES)]
    monkeypatch.setattr(unit_vocab, "reachable_unit_types", lambda *a, **k: names)
    with pytest.raises(ValueError, match="overflow row"):
        unit_vocab.seed_vocab(GameStateEncoder(d_model=8))


def test_loading_a_checkpoint_whose_types_share_the_overflow_row_says_so(tmp_path, caplog):
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(0)
    policy = TransformerPolicy(device=torch.device("cpu"), d_model=32, num_layers=1,
                               num_heads=2, d_ff=64)
    vocab = policy._encoder.unit_type_to_id
    vocab.update({"Orcish Grunt": MAX_UNIT_TYPES - 1, "Troll Whelp": MAX_UNIT_TYPES + 3})
    path = tmp_path / "spilled.pt"
    policy.save_checkpoint(path)
    fresh = TransformerPolicy(device=torch.device("cpu"), d_model=32, num_layers=1,
                              num_heads=2, d_ff=64)
    with caplog.at_level(logging.WARNING):
        fresh.load_checkpoint(path)
    assert any("share the overflow row" in r.getMessage() for r in caplog.records)
