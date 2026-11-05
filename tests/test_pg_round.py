"""One round's step (tools/pg_round.py) on the miniature stage 1
(helpers.pg_run.e2e_run): the ladder through the command line reads three
rungs and writes an adopted rung's network as theta_k + s d; the cosine
with the previous direction reads 1 on the same pass; round 1 taken from
stage C's decision writes its step without a ladder, nothing when stage C
kept pi_k, and the chosen rung's network when a cross-play match must read
it; the radius and the arm are read from either readout; the
probe_span arm's span Fisher is stage C's in round 1 and the round's own
later."""
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
def test_a_round_averaged_with_earlier_directions_steps_along_the_one_half_a_selects(tmp_path):
    """Prereg departure 37 through the command line: with --average-with
    the round reads both ladders, records them and the averaged direction,
    saves its own direction (not the average) and writes the candidate
    along the chosen one (saved beside it when it is the average); an
    earlier direction off the heads, or an arm other than heads_only, is
    recorded and the round reads its own ladder alone."""
    from helpers.pg_run import e2e_run
    from tools.pg_eval import network_factory
    from tools.pg_round import main
    from wesnoth_ai.pg_step import AVERAGE_MULTIPLIERS, LADDER_MULTIPLIERS, averaged, decisions_per_side, head_of
    run = e2e_run()
    factory, _, _ = network_factory(str(run.checkpoint), str(run.games), str(run.labels), device="cpu", streams="4")
    eps = 0.25 / decisions_per_side(factory({}).game_sides())
    common = ["--pass", str(run.pass_path), "--factory", "tools.pg_eval:network_factory", *_factory_args(run),
              "--eps", repr(eps), "--arm", "heads_only"]
    own = tmp_path / "own.pt"
    assert main(common + ["--out", str(tmp_path / "own.json"), "--candidate", str(tmp_path / "pi_own.pt"),
                          "--direction-out", str(own)]) == 0
    d = torch.load(own, weights_only=True)
    g = torch.Generator().manual_seed(5)
    earlier = {n: (torch.randn(t.shape, generator=g) * float(t.abs().mean() + 1e-6) if head_of(n) else torch.zeros_like(t))
               for n, t in d.items()}
    earlier_path, saved, out, candidate = (tmp_path / "earlier.pt", tmp_path / "direction.pt", tmp_path / "round.json",
                                           tmp_path / "pi.pt")
    torch.save(earlier, earlier_path)
    assert main(common + ["--average-with", str(earlier_path), "--out", str(out), "--candidate", str(candidate),
                          "--direction-out", str(saved)]) == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    paired, dec = r["paired"], r["decision"]
    assert paired["chosen"] in ("plain", "average") and r["average"]["of"] == [str(earlier_path)]
    assert [x["multiplier"] for x in paired["rungs"]["plain"]] == list(LADDER_MULTIPLIERS)
    assert [x["multiplier"] for x in paired["rungs"]["average"]] == list(AVERAGE_MULTIPLIERS)
    assert dec["rungs"] == paired["rungs"][paired["chosen"]]
    assert paired["average_minus_plain_A"] is None or set(paired["average_minus_plain_A"]) == {"estimate", "se", "z"}
    assert all(torch.equal(torch.load(saved, weights_only=True)[n], d[n]) for n in d)
    averaged_path = tmp_path / "direction_average.pt"
    assert averaged_path.exists() is (paired["chosen"] == "average")
    if paired["chosen"] == "average":
        assert paired["direction"] == str(averaged_path)
        want = averaged([d, earlier])
        assert all(torch.equal(torch.load(averaged_path, weights_only=True)[n], want[n]) for n in want)
    if dec["adopt"]:
        along = averaged([d, earlier]) if paired["chosen"] == "average" else d
        _same_network(candidate, run.checkpoint, along, dec["s"])
    else:
        assert not candidate.exists()
    own_r = json.loads((tmp_path / "own.json").read_text(encoding="utf-8"))
    trunk = dict(earlier)
    name = next(n for n in d if head_of(n) is None)
    trunk[name] = torch.ones_like(d[name])
    torch.save(trunk, tmp_path / "trunk.pt")
    for tag, cmd, why in (("x", common + ["--average-with", str(tmp_path / "trunk.pt")], "more than the heads"),
                          ("y", [c if c != "heads_only" else "reduced_adamw" for c in common]
                           + ["--average-with", str(earlier_path)], "heads_only arm")):
        assert main(cmd + ["--out", str(tmp_path / f"{tag}.json"), "--candidate", str(tmp_path / f"{tag}.pt")]) == 0
        r = json.loads((tmp_path / f"{tag}.json").read_text(encoding="utf-8"))
        assert why in r["average"]["error"] and "paired" not in r
        if tag == "x":                                          # the plain round, as without --average-with
            assert r["decision"] == own_r["decision"]


@needs_core
@pytest.mark.slow
def test_round_1_is_stage_cs_decision_taken_as_read(tmp_path):
    from helpers.pg_run import e2e_run
    from tools.pg_round import main
    from wesnoth_ai.pg_step import direction
    run = e2e_run()
    common = ["--pass", str(run.pass_path), "--factory", "tools.pg_eval:network_factory", *_factory_args(run)]
    xp_rung = {"multiplier": 2.0, "s": 0.7, "halves": {"A": {"ess_ok": True}, "B": {"ess_ok": False}}}
    pass_ = torch.load(run.pass_path, weights_only=True)
    for name, adopt, chosen, rungs in (("adopt", True, 2, []), ("keep", False, 2, []), ("xp", False, 0, [xp_rung])):
        stage_c = tmp_path / f"stage_c_{name}.json"
        stage_c.write_text(json.dumps({"optimizer": {"winner": "sgd"},
                                       "ladder": {"arm": "sgd", "adopt": adopt, "s": 0.5 if adopt else 0.0,
                                                  "eps": 1e-3, "next_eps": 2e-3 if adopt else 5e-4, "chosen": chosen,
                                                  "reason": "read on half B", "rungs": rungs}}), encoding="utf-8")
        out, candidate, net = tmp_path / f"round_{name}.json", tmp_path / f"pi_1_{name}.pt", tmp_path / f"rung_{name}.pt"
        assert main(common + ["--adopt-from", str(stage_c), "--out", str(out), "--candidate", str(candidate),
                              "--chosen-network", str(net)]) == 0
        r = json.loads(out.read_text(encoding="utf-8"))
        assert r["arm"] == "sgd" and r["decision"]["adopt"] is adopt and "rungs" not in r["decision"]
        if adopt:
            _same_network(candidate, run.checkpoint, direction(pass_, "sgd"), 0.5)
        else:
            assert r["candidate"] is None and not candidate.exists()
        if name == "xp":                                 # the rung the match reads, not a candidate
            assert r["chosen_network"]["s"] == 0.7 and r["chosen_network"]["rung"] == 0
            _same_network(net, run.checkpoint, direction(pass_, "sgd"), 0.7)
        else:
            assert r["chosen_network"] is None and not net.exists()


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


def _heads_params():
    torch.manual_seed(0)
    return {f"{h}.weight": torch.randn(3, 2) for h in ("actor_head", "type_head", "target_q_proj", "weapon_head.2")}


def test_probe_span_takes_stage_cs_fisher_in_round_1_and_its_own_later(monkeypatch):
    """Round 1's probe_span steps by stage C's span Fisher (its heads must be
    this network's); a later round's comes from its pass, else the KL, the
    order stage C reads them in (tools/pg_stage_c.StageC.span)."""
    import numpy as np
    import tools.pg_head_floor as head_floor
    import tools.pg_round as pr
    params = _heads_params()
    heads = list(pr.head_directions(params))
    F = np.diag([1.0, 2.0, 3.0, 4.0])
    got, source = pr.round_span_fisher({}, params, None, {"span": {"heads": heads, "fisher": F.tolist(),
                                                                  "fisher_source": "pass"}})
    assert np.array_equal(got, F) and source == "stage C (pass)"
    with pytest.raises(ValueError, match="not this network's"):
        pr.round_span_fisher({}, params, None, {"span": {"heads": heads[:2], "fisher": F.tolist()}})
    with pytest.raises(ValueError, match="no span Fisher"):
        pr.round_span_fisher({}, params, None, {"ladder": {}})
    monkeypatch.setattr(head_floor, "fisher_from_pass", lambda pass_, names: 2 * F if names == heads else None)
    got, source = pr.round_span_fisher({}, params, None)
    assert np.array_equal(got, 2 * F) and source == "pass"
    monkeypatch.setattr(head_floor, "fisher_from_pass", lambda pass_, names: None)
    monkeypatch.setattr(pr, "span_fisher", lambda factory, dirs, s: (factory, len(dirs), s))
    got, source = pr.round_span_fisher({}, params, "factory")
    assert got == ("factory", 4, pytest.importorskip("tools.pg_stage_c").SPAN_STEP) and source == "kl"


def test_round_1_adopts_a_probe_span_step_along_stage_cs_fisher(monkeypatch):
    import numpy as np
    import tools.pg_round as pr
    params = _heads_params()
    torch.manual_seed(1)
    pass_ = {"total": {n: torch.randn_like(t) for n, t in params.items()}, "games": 4, "decisions": 40}
    monkeypatch.setattr(pr, "direction", lambda p, arm, prm, fisher: {"fisher": fisher})
    F = np.diag([1.0, 2.0, 3.0, 4.0])
    stage_c = {"span": {"heads": list(pr.head_directions(params)), "fisher": F.tolist(), "fisher_source": "kl"}}
    adopt = {"adopt": True, "s": 0.5, "eps": 1e-3, "next_eps": 2e-3, "chosen": 1, "reason": "read"}
    monkeypatch.setattr(pr, "norm", lambda d: 1.0)
    readout, d, s, _ = pr.run_round(pass_, params, None, "probe_span", 1e-3, adopt=adopt, stage_c=stage_c)
    assert np.array_equal(d["fisher"], F) and s == 0.5 and readout["span_fisher"]["source"] == "stage C (kl)"
