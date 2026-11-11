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


class _KLEvaluator:
    """KL q s^2 a decision along the direction it is bound to, q its squared
    norm; `decisions` decisions per game-side."""

    def __init__(self, d, decisions=10, games=4):
        import tools.pg_round as pr
        from wesnoth_ai.pg_protocol import GameSide
        self.d, self.q = d, pr.norm(d) ** 2
        self.sides = [GameSide(f"g{g}", side, 1.0, None, (2 * g + side - 1) * decisions, (2 * g + side) * decisions)
                      for g in range(games) for side in (1, 2)]

    def game_sides(self):
        return self.sides

    def mean_kl(self, s):
        return self.q * s * s


def test_a_lean_step_searches_the_averaged_direction_to_the_radius_and_adopts_it_unread():
    """Prereg departure 43: the step is the KL search alone, along the
    averaged direction, to eps or the range's edge if that is lower, and is
    adopted with one rung that no half read (so no cross-play rung); an
    earlier direction off the heads leaves it along d, the reason
    recorded."""
    import tools.pg_round as pr
    from wesnoth_ai.pg_cross_play import cross_play_case
    from wesnoth_ai.pg_step import KL_RTOL, SIDE_KL_RANGE, averaged
    params = _heads_params()
    torch.manual_seed(2)
    d = {n: torch.randn_like(t) for n, t in params.items()}
    earlier = {n: torch.randn_like(t) for n, t in params.items()}
    bound = []

    def factory(direction):
        bound.append(direction)
        return _KLEvaluator(direction)

    edge = SIDE_KL_RANGE / 10
    for eps, target in ((2 * edge, edge), (edge / 4, edge / 4)):
        bound.clear()
        readout = {}
        readout, d_out, s, step_d = pr._lean_step(readout, factory, "heads_only", eps, d, pr.norm(d), [earlier],
                                                  ["earlier.pt"])
        want = averaged([d, earlier])
        assert d_out is d and all(torch.equal(step_d[n], want[n]) for n in want)
        assert all(torch.equal(bound[0][n], want[n]) for n in want) and len(bound) == 1
        dec = readout["decision"]
        assert dec["adopt"] and dec["lean"] and dec["s"] == s and dec["eps"] == eps and dec["next_eps"] == target
        assert dec["chosen"] == 0 and len(dec["rungs"]) == 1 and dec["rungs"][0]["halves"] == {}
        assert abs(_KLEvaluator(want).mean_kl(s) - target) <= KL_RTOL * target
        assert readout["lean"]["along"] == "average" and readout["lean"]["edge"] == pytest.approx(edge)
        assert readout["average"]["of"] == ["earlier.pt"] and readout["average"]["norm"] == pytest.approx(pr.norm(want))
        assert cross_play_case(dec, dec["rungs"]) is None
    trunk = {**earlier, "encoder.weight": torch.ones(2, 2)}
    readout, _, s, step_d = pr._lean_step({}, factory, "heads_only", edge, d, pr.norm(d), [trunk], ["trunk.pt"])
    assert step_d is d and readout["lean"]["along"] == "plain" and "more than the heads" in readout["average"]["error"]
    readout, _, _, step_d = pr._lean_step({}, factory, "heads_only", edge, d, pr.norm(d), [], None)
    assert step_d is d and readout["lean"]["along"] == "plain" and "average" not in readout


def test_a_lean_step_takes_no_stage_c_decision():
    import tools.pg_round as pr
    with pytest.raises(ValueError, match="round 1's step is stage C's"):
        pr.run_round({}, _heads_params(), None, "heads_only", 1e-3, adopt={"adopt": True}, lean=True)


def test_a_lean_radius_scales_the_target_and_carries_into_the_next_eps():
    """Prereg departure 47: --lean-radius R searches to R times min(eps,
    edge), records R, and its next eps is that target, so the round after
    reads R times the edge again; R needs --lean and is a finite 1 or more
    (below 1 the radius would shrink every round: the code review's L1)."""
    import tools.pg_round as pr
    from wesnoth_ai.pg_step import KL_RTOL, SIDE_KL_RANGE
    params = _heads_params()
    torch.manual_seed(3)
    d = {n: torch.randn_like(t) for n, t in params.items()}
    edge = SIDE_KL_RANGE / 10
    eps = 0.98 * edge
    for radius, target in ((1.0, eps), (2.0, 2 * eps)):
        readout, _, s, step_d = pr._lean_step({}, _KLEvaluator, "heads_only", eps, d, pr.norm(d), [], None, radius)
        dec = readout["decision"]
        assert dec["next_eps"] == pytest.approx(target) and readout["lean"]["radius"] == radius
        assert abs(_KLEvaluator(step_d).mean_kl(s) - target) <= KL_RTOL * target
        assert dec["rungs"][0]["multiplier"] == radius and dec["rungs"][0]["in_range"] == (radius == 1.0)
        assert ("departure 47" in dec["reason"]) == (radius != 1.0)
    readout, *_ = pr._lean_step({}, _KLEvaluator, "heads_only", 2 * eps, d, pr.norm(d), [], None, 2.0)
    assert readout["decision"]["next_eps"] == pytest.approx(2 * edge)
    for kw in ({"lean": False, "radius": 2.0}, {"lean": True, "radius": -1.0}, {"lean": True, "radius": 0.5},
               {"lean": True, "radius": float("inf")}):
        with pytest.raises(ValueError, match="lean radius"):
            pr.run_round({}, params, None, "heads_only", 1e-3, **kw)


@needs_core
@pytest.mark.slow
def test_a_lean_round_through_the_command_line_writes_the_averaged_step(tmp_path):
    """Prereg departure 43 on the miniature stage 1: --lean with
    --average-with writes the candidate theta_k + s d_avg, saves d and the
    average apart, and no chosen network."""
    from helpers.pg_run import e2e_run
    from tools.pg_eval import network_factory
    from tools.pg_round import main
    from wesnoth_ai.pg_step import KL_RTOL, averaged, decisions_per_side, head_of
    run = e2e_run()
    factory, _, _ = network_factory(str(run.checkpoint), str(run.games), str(run.labels), device="cpu", streams="4")
    eps = 0.25 / decisions_per_side(factory({}).game_sides())
    common = ["--pass", str(run.pass_path), "--factory", "tools.pg_eval:network_factory", *_factory_args(run),
              "--eps", repr(eps), "--arm", "heads_only"]
    own = tmp_path / "own.pt"
    assert main(common + ["--lean", "--out", str(tmp_path / "own.json"), "--candidate", str(tmp_path / "own_pi.pt"),
                          "--direction-out", str(own)]) == 0
    d = torch.load(own, weights_only=True)
    g = torch.Generator().manual_seed(5)
    earlier = {n: (torch.randn(t.shape, generator=g) * float(t.abs().mean() + 1e-6) if head_of(n) else torch.zeros_like(t))
               for n, t in d.items()}
    torch.save(earlier, tmp_path / "earlier.pt")
    out, candidate, net = tmp_path / "round.json", tmp_path / "pi.pt", tmp_path / "rung.pt"
    assert main(common + ["--lean", "--average-with", str(tmp_path / "earlier.pt"), "--out", str(out), "--candidate",
                          str(candidate), "--direction-out", str(tmp_path / "direction.pt"), "--chosen-network",
                          str(net)]) == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    dec, want = r["decision"], averaged([d, earlier])
    assert dec["adopt"] and dec["lean"] and r["lean"]["along"] == "average" and "paired" not in r
    assert r["lean"]["direction"] == str(tmp_path / "direction_average.pt")
    assert abs(dec["rungs"][0]["kl"] - dec["next_eps"]) <= KL_RTOL * dec["next_eps"]
    assert r["candidate"]["s"] == dec["s"] and r["chosen_network"] is None and not net.exists()
    _same_network(candidate, run.checkpoint, want, dec["s"])
    assert all(torch.equal(torch.load(tmp_path / "direction.pt", weights_only=True)[n], d[n]) for n in d)


def test_a_lean_round_with_a_zero_direction_is_still_marked_lean(monkeypatch):
    """The review's L3: a lean round whose direction is zero keeps pi_k and
    is marked lean, so rounds_local measures it as a lean round."""
    import tools.pg_round as pr
    params = _heads_params()
    monkeypatch.setattr(pr, "direction", lambda p, arm, prm, fisher: {n: torch.zeros_like(t) for n, t in prm.items()})
    readout, _, s, _ = pr.run_round({}, params, None, "heads_only", 1e-3, lean=True)
    assert s is None and readout["decision"]["lean"] and not readout["decision"]["adopt"]
    readout, _, _, _ = pr.run_round({}, params, None, "heads_only", 1e-3)
    assert "lean" not in readout["decision"]


def _whole_params():
    """_heads_params' head maps and two parameters off them."""
    torch.manual_seed(4)
    return {**_heads_params(), "encoder.layers.0.linear1.weight": torch.randn(4, 3),
            "slot_memory.gate.weight": torch.randn(2, 5)}


def _group_norm(d, heads: bool) -> float:
    from wesnoth_ai.pg_step import head_of, norm
    return norm({n: t for n, t in d.items() if bool(head_of(n)) == heads})


def test_a_fixed_step_moves_each_group_its_length_and_reads_no_kl(monkeypatch):
    """Prereg departure 49: the head maps move HEADS in weight space along
    their direction averaged with the earlier rounds' head maps, every other
    parameter moves REST along its own, and the step is adopted at s = 1
    with no rung, marked lean and fixed, eps carried unchanged. No evaluator
    is bound, so no KL is read. An earlier direction that lacks a head map
    leaves the head maps along d, the reason recorded."""
    import tools.pg_round as pr
    from wesnoth_ai.pg_cross_play import cross_play_case
    from wesnoth_ai.pg_step import averaged, dot, head_of, norm
    params = _whole_params()
    torch.manual_seed(6)
    d = {n: torch.randn_like(t) for n, t in params.items()}
    earlier = {n: torch.randn_like(t) for n, t in params.items()}           # a full direction: its head maps are read
    heads = {n: t for n, t in d.items() if head_of(n)}
    rest = {n: t for n, t in d.items() if not head_of(n)}

    def factory(direction):
        raise AssertionError("a fixed step binds no evaluator")

    pass_ = {"total": d, "games": 4, "decisions": 40}
    monkeypatch.setattr(pr, "direction", lambda p, arm, prm, fisher: d)
    readout, d_out, s, step_d = pr.run_round(pass_, params, factory, "reduced_adamw", 1e-3, average=[earlier],
                                             average_of=["earlier.pt"], lean=True, fixed=(0.5, 0.019))
    want = averaged([heads, earlier])
    assert d_out is d and s == 1.0 and set(step_d) == set(d)
    assert _group_norm(step_d, True) == pytest.approx(0.5, rel=1e-5)
    assert _group_norm(step_d, False) == pytest.approx(0.019, rel=1e-5)
    step_heads = {n: step_d[n] for n in heads}
    assert dot(step_heads, want) / (norm(step_heads) * norm(want)) == pytest.approx(1.0, abs=1e-6)
    step_rest = {n: step_d[n] for n in rest}
    assert dot(step_rest, rest) / (norm(step_rest) * norm(rest)) == pytest.approx(1.0, abs=1e-6)
    dec = readout["decision"]
    assert dec["adopt"] and dec["lean"] and dec["fixed"] and dec["s"] == 1.0 and dec["rungs"] == []
    assert dec["eps"] == dec["next_eps"] == 1e-3 and "no KL read" in dec["reason"]
    assert cross_play_case(dec, dec["rungs"]) is None
    fixed = readout["lean"]["fixed"]
    assert readout["lean"]["along"] == "average" and fixed["moved"] == {"heads": 0.5, "rest": 0.019}
    assert fixed["heads_norm"] == pytest.approx(norm(heads)) and fixed["rest_norm"] == pytest.approx(norm(rest))
    assert readout["average"]["of"] == ["earlier.pt"] and readout["average"]["norm"] == pytest.approx(norm(want))
    short = {n: t for n, t in earlier.items() if not n.startswith("actor_head")}
    readout, _, _, step_d = pr._fixed_step({}, 1e-3, d, [short], ["short.pt"], (0.5, 0.019))
    assert readout["lean"]["along"] == "plain" and "lacks" in readout["average"]["error"]
    step_heads = {n: step_d[n] for n in heads}
    assert dot(step_heads, heads) / (norm(step_heads) * norm(heads)) == pytest.approx(1.0, abs=1e-6)
    readout, _, _, step_d = pr._fixed_step({}, 1e-3, d, [], None, (0.5, 0.0))
    assert _group_norm(step_d, True) == pytest.approx(0.5, rel=1e-5) and _group_norm(step_d, False) == 0.0
    assert "average" not in readout and readout["lean"]["fixed"]["moved"] == {"heads": 0.5, "rest": 0.0}


def test_a_fixed_step_with_no_direction_to_move_keeps_the_policy_and_bad_lengths_are_refused():
    """A group whose direction is zero stays; when no group with a length
    has a direction, pi_k stays and the round is still marked lean and
    fixed. A fixed step needs --lean at radius 1, two finite lengths that
    are not negative and not both 0, and an arm that reads every parameter
    when the rest moves."""
    import tools.pg_round as pr
    from wesnoth_ai.pg_step import head_of
    params = _whole_params()
    torch.manual_seed(7)
    d = {n: (torch.zeros_like(t) if head_of(n) else torch.randn_like(t)) for n, t in params.items()}
    readout, _, s, step_d = pr._fixed_step({}, 1e-3, d, [], None, (0.5, 0.019))
    assert s == 1.0 and _group_norm(step_d, True) == 0.0 and _group_norm(step_d, False) == pytest.approx(0.019, rel=1e-5)
    assert readout["lean"]["fixed"]["moved"] == {"heads": 0.0, "rest": 0.019}
    readout, d_out, s, step_d = pr._fixed_step({}, 1e-3, d, [], None, (0.5, 0.0))
    dec = readout["decision"]
    assert s is None and step_d is d_out and not dec["adopt"] and dec["lean"] and dec["fixed"] and dec["next_eps"] == 1e-3
    for arm, kw in (("reduced_adamw", {"lean": False, "fixed": (0.5, 0.019)}),
                    ("reduced_adamw", {"lean": True, "radius": 2.0, "fixed": (0.5, 0.019)}),
                    ("reduced_adamw", {"lean": True, "fixed": (0.0, 0.0)}),
                    ("reduced_adamw", {"lean": True, "fixed": (-0.5, 0.019)}),
                    ("reduced_adamw", {"lean": True, "fixed": (0.5, float("inf"))}),
                    ("reduced_adamw", {"lean": True, "fixed": (0.5,)}),
                    ("heads_only", {"lean": True, "fixed": (0.5, 0.019)})):
        with pytest.raises(ValueError, match="fixed step"):
            pr.run_round({}, params, None, arm, 1e-3, **kw)


@needs_core
@pytest.mark.slow
def test_a_fixed_round_through_the_command_line_writes_the_step_of_the_two_lengths(tmp_path):
    """Prereg departure 49 on the miniature stage 1: --lean --fixed-step on
    the reduced_adamw arm writes the candidate theta_k + step, the step
    saved as step.pt with the two lengths on its groups, d saved apart, and
    no chosen network; without --lean it is refused."""
    from helpers.pg_run import e2e_run
    from tools.pg_round import main
    run = e2e_run()
    common = ["--pass", str(run.pass_path), "--factory", "tools.pg_eval:network_factory", *_factory_args(run),
              "--eps", "0.001", "--arm", "reduced_adamw"]
    out, candidate, net = tmp_path / "round.json", tmp_path / "pi.pt", tmp_path / "rung.pt"
    assert main(common + ["--lean", "--fixed-step", "0.05", "0.01", "--out", str(out), "--candidate", str(candidate),
                          "--direction-out", str(tmp_path / "direction.pt"), "--chosen-network", str(net)]) == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    dec = r["decision"]
    assert dec["adopt"] and dec["lean"] and dec["fixed"] and dec["s"] == 1.0 and dec["next_eps"] == 0.001
    assert r["lean"]["direction"] == str(tmp_path / "step.pt") and r["candidate"]["s"] == 1.0
    assert r["chosen_network"] is None and not net.exists() and "paired" not in r
    step = torch.load(tmp_path / "step.pt", weights_only=True)
    d = torch.load(tmp_path / "direction.pt", weights_only=True)
    assert set(step) == set(d) and _group_norm(d, False) > 0
    assert _group_norm(step, True) == pytest.approx(0.05, rel=1e-5)
    assert _group_norm(step, False) == pytest.approx(0.01, rel=1e-5)
    _same_network(candidate, run.checkpoint, step, 1.0)
    with pytest.raises(SystemExit):
        main(common + ["--fixed-step", "0.05", "0.01", "--out", str(out), "--candidate", str(candidate)])
