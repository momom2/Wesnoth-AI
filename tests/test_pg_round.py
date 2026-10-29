"""One round's step (tools/pg_round.py) on the miniature stage 1
(helpers.pg_run.e2e_run): the ladder through the command line reads three
rungs and writes an adopted rung's network as theta_k + s d; the cosine
with the previous direction reads 1 on the same pass; round 1 taken from
stage C's decision writes its step without a ladder, and nothing when stage
C kept pi_k; the radius and the arm are read from either readout."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai import game_core as gc  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")


def _factory_args(run):
    args = []
    for key, value in (("checkpoint", run.checkpoint), ("games", run.games), ("labels", run.labels),
                       ("device", "cpu"), ("streams", 4)):
        args += ["--factory-arg", f"{key}={value}"]
    return args


def _same_network(path: Path, checkpoint: Path, d, s: float) -> None:
    """The written candidate is theta_k + s d, added by code of its own."""
    from helpers.pg_games import moved_network
    from tools.pg_pass import load_network
    from wesnoth_ai.param_groups import named_model_parameters
    written = dict(named_model_parameters(*reversed(load_network(path, torch.device("cpu")))))
    want = dict(named_model_parameters(*reversed(moved_network(checkpoint, d, s))))
    assert set(written) == set(want)
    assert all(torch.allclose(written[n], want[n], atol=1e-6) for n in want)


@needs_core
@pytest.mark.slow
def test_a_round_reads_its_ladder_and_writes_an_adopted_rung(tmp_path):
    from helpers.pg_run import e2e_run
    from tools.pg_eval import network_factory
    from tools.pg_round import main
    from wesnoth_ai.pg_step import LADDER_MULTIPLIERS, decisions_per_side
    run = e2e_run()
    factory, _, _ = network_factory(str(run.checkpoint), str(run.games), str(run.labels), device="cpu", streams="4")
    eps = 0.25 / decisions_per_side(factory({}).game_sides())
    common = ["--pass", str(run.pass_path), "--factory", "tools.pg_eval:network_factory", *_factory_args(run)]
    out, candidate, d_path = tmp_path / "round.json", tmp_path / "pi_1.pt", tmp_path / "direction.pt"
    assert main(common + ["--eps", repr(eps), "--out", str(out), "--candidate", str(candidate),
                          "--direction-out", str(d_path)]) == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    dec = r["decision"]
    assert r["arm"] == "reduced_adamw" and r["eps"] == eps and len(dec["rungs"]) == len(LADDER_MULTIPLIERS)
    assert dec["next_eps"] in (eps / 2.0, *(m * eps for m in LADDER_MULTIPLIERS))
    d = torch.load(d_path, weights_only=True)
    if dec["adopt"]:
        assert r["candidate"]["s"] == dec["s"] and candidate.exists()
        _same_network(candidate, run.checkpoint, d, dec["s"])
    else:
        assert r["candidate"] is None and not candidate.exists()
    again = tmp_path / "again.json"
    assert main(common + ["--eps-from", str(out), "--previous", str(d_path), "--out", str(again),
                          "--candidate", str(tmp_path / "pi_2.pt")]) == 0
    r2 = json.loads(again.read_text(encoding="utf-8"))
    assert r2["eps"] == dec["next_eps"] and r2["cosine_previous"] == pytest.approx(1.0, abs=1e-9)


@needs_core
@pytest.mark.slow
def test_round_1_is_stage_cs_decision_taken_as_read(tmp_path):
    from helpers.pg_run import e2e_run
    from tools.pg_round import main
    from wesnoth_ai.pg_step import direction
    run = e2e_run()
    common = ["--pass", str(run.pass_path), "--factory", "tools.pg_eval:network_factory", *_factory_args(run)]
    for adopt in (True, False):
        stage_c = tmp_path / f"stage_c_{adopt}.json"
        stage_c.write_text(json.dumps({"optimizer": {"winner": "sgd"},
                                       "ladder": {"arm": "sgd", "adopt": adopt, "s": 0.5 if adopt else 0.0,
                                                  "eps": 1e-3, "next_eps": 2e-3 if adopt else 5e-4, "chosen": 2,
                                                  "reason": "read on half B", "rungs": []}}), encoding="utf-8")
        out, candidate = tmp_path / f"round_{adopt}.json", tmp_path / f"pi_1_{adopt}.pt"
        assert main(common + ["--adopt-from", str(stage_c), "--out", str(out), "--candidate", str(candidate)]) == 0
        r = json.loads(out.read_text(encoding="utf-8"))
        assert r["arm"] == "sgd" and r["decision"]["adopt"] is adopt and "rungs" not in r["decision"]
        if adopt:
            pass_ = torch.load(run.pass_path, weights_only=True)
            _same_network(candidate, run.checkpoint, direction(pass_, "sgd"), 0.5)
        else:
            assert r["candidate"] is None and not candidate.exists()


def test_the_radius_and_the_arm_are_read_from_either_readout(tmp_path):
    from tools.pg_round import arm_of, eps_of
    round_json, stage_c = tmp_path / "round.json", tmp_path / "stage_c.json"
    round_json.write_text(json.dumps({"decision": {"next_eps": 0.004}}), encoding="utf-8")
    stage_c.write_text(json.dumps({"ladder": {"next_eps": 0.002}, "optimizer": {"winner": "muon"}}),
                       encoding="utf-8")
    assert eps_of(round_json) == 0.004 and eps_of(stage_c) == 0.002 and arm_of(stage_c) == "muon"
    (tmp_path / "bare.json").write_text("{}", encoding="utf-8")
    assert arm_of(tmp_path / "bare.json") == "reduced_adamw"
    with pytest.raises(ValueError, match="no next_eps"):
        eps_of(tmp_path / "bare.json")
