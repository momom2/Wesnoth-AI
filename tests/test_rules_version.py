"""The rules version of pre-encoded records (wesnoth_ai/rules_version.py):
the adapter's phase has one; records another phase built under the same
rules load and train to the bit as the adapter's own phase's; records of
another rules version, of a phase the table does not hold, or naming no
phase are refused, the corpus's and the own game-sides' alike; and the
fingerprint follows the rules version, not the phase."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from test_sequence_launches import QUIET

from tools.preencode_sequences import MANIFEST_NAME, encoding_fingerprint
from wesnoth_ai import rules_version as rv
from wesnoth_ai.game_core import _CORE_PHASE

# A phase no wheel declares, registered under the adapter's rules version:
# "another phase that builds positions alike".
SIBLING = 10 ** 6


def test_the_adapter_phase_has_a_rules_version():
    assert _CORE_PHASE in rv.RULES_VERSION_OF_PHASE, (
        f"core phase {_CORE_PHASE} is not in wesnoth_ai/rules_version.py: add it, mapped to its predecessor's "
        f"rules version if every position is built as before, else to itself")


def test_each_rules_version_is_named_after_the_first_phase_that_built_it():
    table = rv.RULES_VERSION_OF_PHASE
    phases = sorted(table)
    assert [table[p] for p in phases] == sorted(table.values()), "a later phase under earlier rules"
    for phase in phases:
        version = table[phase]
        assert version <= phase and table.get(version) == version, (phase, version)
        assert all(table[p] != version for p in phases if p < version), (phase, version)


def test_the_fingerprint_follows_the_rules_version_not_the_phase(monkeypatch):
    vocab = ({"Spearman": 1, "Bowman": 2}, {"Loyalists": 1})
    monkeypatch.setitem(rv.RULES_VERSION_OF_PHASE, SIBLING, rv.current_rules_version())
    own = encoding_fingerprint(*vocab, 5, rv.rules_version_of(_CORE_PHASE))
    assert encoding_fingerprint(*vocab, 5, rv.rules_version_of(SIBLING)) == own
    monkeypatch.setitem(rv.RULES_VERSION_OF_PHASE, SIBLING, SIBLING)
    assert encoding_fingerprint(*vocab, 5, rv.rules_version_of(SIBLING)) != own


def test_a_phase_outside_the_table_has_no_rules_version():
    with pytest.raises(ValueError, match="core phase 30 has no rules version"):
        rv.rules_version_of(30)


def _restamped(pass_inputs, tmp_path: Path, phase):
    """pass_inputs with its sequences copied and their manifest restamped
    with `phase` (None: no core phase)."""
    i = pass_inputs.index("--sequences")
    copy = tmp_path / "sequences"
    shutil.copytree(pass_inputs[i + 1], copy)
    path = copy / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if phase is None:
        manifest.pop("core_phase")
    else:
        manifest["core_phase"] = phase
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return [*pass_inputs[:i + 1], str(copy), *pass_inputs[i + 2:]]


def _first_window(args, out: Path):
    from tools import sequence_train
    trainer = sequence_train.build_trainer(sequence_train.parse_args([*args, *QUIET, "--out", str(out)]))
    result = trainer.train_window()
    return result["loss"], [p.detach().clone() for p in trainer.params]


@pytest.mark.slow
def test_records_of_another_phase_under_the_same_rules_train_to_the_bit(tmp_path, pass_inputs, monkeypatch):
    monkeypatch.setitem(rv.RULES_VERSION_OF_PHASE, SIBLING, rv.current_rules_version())
    loss, params = _first_window(pass_inputs, tmp_path / "own_phase.pt")
    sibling = _restamped(pass_inputs, tmp_path / "sibling", SIBLING)
    loss2, params2 = _first_window(sibling, tmp_path / "sibling.pt")
    assert loss2 == loss and all(torch.equal(a, b) for a, b in zip(params, params2))


@pytest.mark.slow
@pytest.mark.parametrize("phase, refusal", [(SIBLING, "built under rules version"), (30, "has no rules version"),
                                            (None, "names no core phase")])
def test_records_of_other_rules_are_refused(tmp_path, pass_inputs, monkeypatch, phase, refusal):
    """A phase registered under a rules version of its own (a rules change),
    a phase before the table, a manifest without a phase."""
    from tools import sequence_train
    monkeypatch.setitem(rv.RULES_VERSION_OF_PHASE, SIBLING, SIBLING)
    args = _restamped(pass_inputs, tmp_path, phase)
    with pytest.raises(SystemExit, match=refusal):
        sequence_train.build_trainer(sequence_train.parse_args([*args, *QUIET, "--out", str(tmp_path / "a.pt")]))


def _own_directory(tmp_path: Path, corpus: dict, phase: int) -> Path:
    """An own-records manifest (no records: the streams' facts only) with
    the corpus's encoding, stamped with `phase`."""
    own = tmp_path / "own"
    own.mkdir()
    keys = ("encoding", "observation_epoch", "unit_type_to_id", "faction_to_id")
    manifest = {**{k: corpus[k] for k in keys}, "core_phase": phase, "source": "own", "slots": 64,
                "fingerprint": "own", "games": {"o.json.gz": {"positions_side1": 5, "positions_side2": 4,
                                                              "holdout": False, "n_commands": 9, "winner": 1}}}
    (own / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return own


@pytest.mark.parametrize("registered, loads", [(True, True), (False, False)],
                         ids=["same rules", "a rules change"])
def test_own_records_load_under_the_same_rules_whatever_their_phase(tmp_path, monkeypatch, registered, loads):
    """The own game-sides beside a corpus of the adapter's phase: own
    records of a sibling phase train beside it (the round-3 bench's case:
    own records of phase 31 beside a corpus of phase 32), own records whose
    phase starts another rules version are refused."""
    from tools.sequence_own import OwnStreams
    corpus = {"encoding": {"relevant_set": True}, "observation_epoch": 11, "core_phase": _CORE_PHASE,
              "unit_type_to_id": {"Spearman": 1}, "faction_to_id": {"Loyalists": 1}}
    monkeypatch.setitem(rv.RULES_VERSION_OF_PHASE, SIBLING,
                        rv.current_rules_version() if registered else SIBLING)
    own = _own_directory(tmp_path, corpus, SIBLING)
    if loads:
        streams = OwnStreams(own, corpus, seed=1, n_streams=2, slot_offset=2)
        assert streams.slots == 64 and streams.schedule.total_positions == 9
    else:
        with pytest.raises(SystemExit, match="built under rules version"):
            OwnStreams(own, corpus, seed=1, n_streams=2, slot_offset=2)
