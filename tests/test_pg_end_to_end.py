"""Stages 1 and C end to end on the CPU: a tiny network's temperature-1
self-play games logged by the production path, their rows and labels
(branch A), the gradient pass (branch B) and the stage C driver on B's
network evaluator (branch C), with the identities each seam promises:
- P1: the pass's probe gradients are the logs' (1/2N) sum R (1[c] - p(c)),
  R read from A's labels and p from A's log;
- the replay identity: the trainer's law at every logged decision, in the
  pass and in the evaluator, is the logged law (fp32 on both sides);
- the pass's head slopes are the stage C driver's slopes of the pass's
  total along the head directions;
- the first-order step: along the directions stage C builds from pass.pt,
  the evaluator's first-order gain on the held-out games moves at s times
  the held-out gradient's inner product with each, the held-out gradient
  read by a pass of its own; and each arm's step sits at its target KL;
- the driver's readout carries a verdict."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai import game_core as gc  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
CPU = torch.device("cpu")
PRICE_PER_GAME = 7.76e-5        # design §2: a generated and trained game
ROUND_FIXED = 1.23              # design §2: a round's fixed part, low end


def _log_sums(run, rows):
    """The logs' (1/2N) sum over the training games' decisions of
    R (1[c] - p(c)), by class (for attacks p(attack | the sampled actor),
    no term where that actor has no attack)."""
    from tools.pg_positions import RecordIndex, read_record
    from wesnoth_ai.pg_records import decisions_of
    from wesnoth_ai.server_priors import KIND_ATTACK, KIND_END_TURN, KIND_RECRUIT
    index = RecordIndex([run.games])
    R = {(r.game, r.side): r.R for r in rows if not r.heldout}
    games = {g for g, _ in R}
    sums = {"end_turn": 0.0, "attack": 0.0, "recruit": 0.0}
    for game in games:
        for d in decisions_of(read_record(index.path(game))):
            r = R[(game, d.side)]
            sums["end_turn"] += r * ((d.kind == KIND_END_TURN) - d.p_end)
            sums["recruit"] += r * ((d.kind == KIND_RECRUIT) - d.p_rec)
            if not math.isnan(d.p_att_actor):
                sums["attack"] += r * ((d.kind == KIND_ATTACK) - d.p_att_actor)
    return {c: v / (2 * len(games)) for c, v in sums.items()}


def _heldout_gradient(run, rows):
    """The held-out games' gradient (1/2N_H) sum R sum_t grad log pi, by a
    pass over them relabelled as training games, its window longer than
    the run so the memory's gradient is never cut."""
    import dataclasses
    from tools.pg_pass import GradientPass, PassConfig
    from tools.pg_positions import RecordIndex
    held = [dataclasses.replace(r, heldout=False, half=None) for r in rows if r.heldout]
    index = RecordIndex([run.games])
    records = {r.game: index.path(r.game) for r in held}
    return GradientPass(run.checkpoint, records, held, CPU, PassConfig(streams=2, window=4096)).run()["total"]


@needs_core
@pytest.mark.slow
def test_stage_1_to_stage_c_on_the_cpu(tmp_path):
    from helpers.pg_run import e2e_run
    from tools.pg_eval import network_factory
    from tools.pg_positions import RecordIndex, read_record
    from tools.pg_stage_c import main as stage_c
    from wesnoth_ai.pg_records import decisions_of, read_labels
    from wesnoth_ai.pg_step import DEFAULT_ARM, KL_RTOL, direction, dot, first_order_gain, norm
    run = e2e_run()
    rows = read_labels(run.labels)
    pass_ = torch.load(run.pass_path, map_location="cpu", weights_only=True)

    # P1 and the replay identity in the pass.
    want = _log_sums(run, rows)
    for c, v in want.items():
        assert pass_["probes"][c] == pytest.approx(v, abs=2e-5), c
    assert max(abs(v) for v in want.values()) > 1e-3
    assert abs(pass_["replay"]["kl"]) < 1e-6 and pass_["replay"]["max_abs"] < 1e-4
    assert pass_["games"] == 8 and pass_["split_games"] == [1] * 8

    # Stage C through the command line, on B's evaluator.
    out = tmp_path / "stage_c.json"
    args = ["--pass", str(run.pass_path), "--factory", "tools.pg_eval:network_factory",
            "--price-per-game", str(PRICE_PER_GAME), "--round-fixed-cost", str(ROUND_FIXED), "--out", str(out)]
    for key, value in (("checkpoint", run.checkpoint), ("games", run.games), ("labels", run.labels),
                       ("device", "cpu"), ("streams", 4)):
        args += ["--factory-arg", f"{key}={value}"]
    assert stage_c(args) == 0
    readout = json.loads(out.read_text(encoding="utf-8"))
    assert readout["errors"] == {} and readout["gate"]["verdict"] in ("resolved", "rounds_at_size", "neither")
    assert readout["pass"]["checkpoint_sha256"] == pass_["checkpoint_sha256"]
    eps = readout["heldout"]["eps"]
    for arm, row in readout["arms"].items():
        assert abs(row["kl"] - eps) <= KL_RTOL * eps, arm
    span = readout["span"]
    assert np.allclose(span["slopes"], span["pass_slopes"], rtol=1e-3, atol=1e-8)

    # The first-order step along stage C's directions from pass.pt. A central
    # difference at a thousandth of the direction's norm: along SGD's the
    # network is smooth to 3e-6 there; reduced AdamW's leans on parameters of
    # small second moment, and finite steps along it read 0.3-0.5 % below
    # the gradient in code of its own too (2026-10-22), hence its 1 %.
    factory, _, sha = network_factory(str(run.checkpoint), str(run.games), str(run.labels), device="cpu",
                                      streams="4")
    assert sha == pass_["checkpoint_sha256"]
    g_H = _heldout_gradient(run, rows)
    index = RecordIndex([run.games])
    for arm, rel in (("sgd", 1e-3), (DEFAULT_ARM, 1e-2)):
        d = direction(pass_, arm)
        ev = factory(d)
        sides = ev.game_sides()
        base = ev.base_logp()
        if arm == "sgd":
            logged = [x.logp for gs in sides for x in decisions_of(read_record(index.path(gs.game)))
                      if x.side == gs.side]
            assert np.max(np.abs(base - np.array(logged))) < 1e-4
        s = 1e-3 / norm(d)
        slope = (first_order_gain(ev.logp(s) - base, sides).estimate
                 - first_order_gain(ev.logp(-s) - base, sides).estimate) / (2.0 * s)
        forecast = dot(g_H, d)
        assert abs(forecast) > 1e-3
        assert slope == pytest.approx(forecast, rel=rel), arm
        ev.context.close()
