#!/usr/bin/env python3
"""Every-Nth-per-category validation exports (2026-07-15).

User spec: during training, every 100th game of each category
(mini / ladder / ladder_fogless / midgame) exports a strict-sync
Wesnoth replay; the HF uploader ships them; a local batch runner
verifies them in real Wesnoth. Both export shapes are
engine-verified (2026-07-15): fresh tentacle-map games play back
clean, and spliced midgame replays (human prefix + sim
continuation) played 272/272 clean on a 12-turn Den of Onis cut.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools.validation_exports import ValidationExporter, category_of

_DATASET = Path(__file__).parent.parent / "replays_dataset"


def _fake_sim(scenario_id="multiplayer_Hamlets", fog=True,
              midgame=False):
    gi = SimpleNamespace(_fog=fog)
    sim = SimpleNamespace(
        gs=SimpleNamespace(global_info=gi),
        scenario_id=scenario_id,
        _midgame_start=midgame,
    )
    return sim


def test_category_classification():
    assert category_of(_fake_sim()) == "ladder"
    assert category_of(_fake_sim(fog=False)) == "ladder_fogless"
    assert category_of(_fake_sim("enclave_micro_isar")) == "mini"
    # a fogless MINI stays "mini": the fogless split is ladder-only
    assert category_of(_fake_sim("enclave_micro_isar",
                                 fog=False)) == "mini"
    assert category_of(_fake_sim(midgame=True)) == "midgame"


def test_picker_takes_first_of_each_block_per_category(
        tmp_path, monkeypatch):
    exported = []
    monkeypatch.setattr(
        "tools.sim_to_replay.export_replay_from_scratch",
        lambda sim, out: exported.append(Path(out).name))
    ex = ValidationExporter(tmp_path, every=3)
    for i in range(7):
        ex.maybe_export(_fake_sim(), game_label=f"L{i}")      # ladder
    for i in range(4):
        ex.maybe_export(_fake_sim(fog=False), game_label=f"F{i}")
    # ladder: games 1,4,7 -> 3 exports; fogless: games 1,4 -> 2.
    ladder = [n for n in exported if n.startswith("ladder_n")]
    fogless = [n for n in exported if n.startswith("ladder_fogless_")]
    assert len(ladder) == 3 and len(fogless) == 2, exported
    assert ladder[0].startswith("ladder_n00001_")
    assert ladder[1].startswith("ladder_n00004_")


def test_export_failure_never_raises(tmp_path):
    # A sim with no template/scenario must log-and-continue, not
    # kill the training loop.
    ex = ValidationExporter(tmp_path, every=1)
    assert ex.maybe_export(_fake_sim("no_such_scenario")) is None


@pytest.mark.skipif(not (_DATASET / "value_corpus_index.jsonl").exists(),
                    reason="replays_dataset not present")
def test_midgame_splice_composes_and_parses(tmp_path):
    import bz2
    import re

    from tools.midgame_starts import sample_midgame_start
    from tools.validation_exports import export_midgame_replay
    from tools.wesnoth_sim import WesnothSim

    rng = random.Random(42)
    mg = None
    while mg is None:
        mg = sample_midgame_start(rng, _DATASET)
    gs, scen_id, cut_turn, begin_side, prov = mg
    assert prov["file"] and prov["boundary_idx"] > 0
    sim = WesnothSim(gs, scenario_id=scen_id, max_turns=cut_turn + 2,
                     apply_scenario_events=False, begin_side=begin_side)
    sim._midgame_start = True
    sim._midgame_provenance = prov
    while not sim.done:
        sim.step({"type": "end_turn"})
    out = tmp_path / "midgame.bz2"
    export_midgame_replay(sim, out)
    wml = bz2.open(out, "rt", encoding="utf-8").read()
    # the human prefix AND the sim continuation are both present:
    # total init_side commands exceed the continuation's alone.
    inits = re.findall(r"\[init_side\]", wml)
    cont_inits = sum(1 for rc in sim.command_history
                     if rc.kind == "init_side")
    assert len(inits) > cont_inits
    assert "[replay_start]" in wml or "[scenario]" in wml
    assert re.search(r"^\s*current_time=\d+", wml, re.MULTILINE)


def test_recruit_rng_predictor_named_races():
    """Name generation draws SYNCED RNG for named races
    (markov_generator.cpp pre-draws next_random(); see
    docs/wesnoth_rules.md). The Wose is the canonical trap: zero
    random traits, single gender, but race wose HAS names -> seed
    required (validation pipeline's first real catch, 2026-07-15).
    Nameless undead stay seedless (the 2026-07-06 Skeleton
    calibration)."""
    from tools.sim_to_replay import _recruit_consumes_synced_rng as f
    assert f("Wose") is True              # named race, 0 traits
    assert f("Skeleton") is False         # undead: nameless, musthave-only
    assert f("Ghoul") is False            # undead: 2 musthaves > num_traits
    assert f("Walking Corpse") is False
    assert f("Vampire Bat") is True       # bats nameless BUT draws a trait
    assert f("Dark Adept") is True        # multi-gender
    assert f("Elvish Fighter") is True    # named + trait draws


def test_midgame_export_carries_source_economy():
    """Third validation-pipeline catch (2026-07-15): midgame exports
    dropped the source game's economy, so playback under-accrued
    gold on base_income-3/4 or village_income variants until
    recruits bounced ("unit 'X' is too expensive to recruit").
    The [side] income attr is an OFFSET over
    game_config::base_income=2 (team.hpp:179). Engine-verified:
    a 14-turn, 32-recruit prefix on a base_income=3 / gold=150
    corpus game plays back clean 515/515."""
    from tools.validation_exports import side_economy_from_dataset
    econ = side_economy_from_dataset([
        {"side": 1, "gold": 150, "village_income": 1,
         "village_support": 2, "base_income": 3},
        {"side": 2, "gold": 200, "base_income": 4},
    ])
    assert econ[1] == {"gold": 150, "village_gold": 1,
                       "village_support": 2, "income_offset": 1}
    assert econ[2]["gold"] == 200 and econ[2]["income_offset"] == 2
    assert econ[2]["village_gold"] == 2      # dataset default
    # And build_save_wml emits them per side.
    from test_sim_to_replay_from_scratch import _build_sim_for
    from tools.sim_to_replay import build_save_wml
    import re
    sim = _build_sim_for("multiplayer_Hamlets")
    wml = build_save_wml(sim, side_economy=econ)
    sides = re.findall(r"\[side\](.*?)\[/side\]", wml, re.S)
    s1 = next(s for s in sides if 'side="1"' in s)
    s2 = next(s for s in sides if 'side="2"' in s)
    assert 'income="1"' in s1 and 'gold="150"' in s1 \
        and 'village_gold="1"' in s1
    assert 'income="2"' in s2 and 'gold="200"' in s2
