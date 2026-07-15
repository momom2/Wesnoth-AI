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
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from tools.validation_exports import ValidationExporter, category_of

_DATASET = Path(__file__).parent / "replays_dataset"


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
