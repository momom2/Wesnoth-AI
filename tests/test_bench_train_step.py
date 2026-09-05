"""tools/bench_train_step.py on a tiny policy: two mini-map start
states with prior-sampled visit counts, N=2, fp32 and bf16 autocast on
cpu, batch 1 and 2. Every stage must record time, the parity block
must find the fp32 rerun identical to its reference (the stubbed
optimizer leaves the weights alone), the implied-iteration arithmetic
must land in (0, 1), and a compiled trunk that deviates must fail its
parity rows (they are scored against the eager reference)."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_test_helpers import require_scenario_data  # noqa: E402
from tools.bench_train_step import (  # noqa: E402
    PER_EXP_STAGES, LoopShape, experiences_from_states, markdown_report, run_benchmark,
)
from tools.scenario_pool import build_scenario_gamestate, random_setup  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402


def test_bench_train_step_times_every_stage_and_checks_parity():
    require_scenario_data()
    torch.manual_seed(0)
    device = torch.device("cpu")
    policy = TransformerPolicy(device=device, d_model=32, num_layers=1, num_heads=2, d_ff=64)
    states = [build_scenario_gamestate(random_setup(random.Random(i), mini_maps=True))
              for i in range(2)]
    exps = experiences_from_states(policy, states, sims=4, rng=random.Random(0))
    assert len(exps) == 2 and all(e.visit_counts for e in exps)
    assert all(sum(v[3] for v in e.visit_counts) == 4 for e in exps)

    res = run_benchmark(policy, exps, device=device, n_list=[2], batch_sizes=[1, 2],
                        precisions=["fp32", "bf16"], repeats=1, parity_n=2,
                        loop=LoopShape(games_per_iter=4, exps_per_game=2.0))

    rows = res["rows"]
    assert {(r["precision"], r["batch_size"]) for r in rows} == {
        ("fp32", 1), ("fp32", 2), ("bf16", 1), ("bf16", 2)}
    for r in rows:
        assert all(r["ms_per_exp"][s] > 0.0 for s in PER_EXP_STAGES), r
        assert r["step_wall_ms_per_exp"] > 0.0 and r["optimizer_ms"] > 0.0
        assert r["snapshot_ms"] > 0.0 and r["clip_ms"] > 0.0
        assert math.isfinite(r["grad_norm"]) and math.isfinite(r["total_loss"])

    parity = res["parity"]
    assert parity[0]["reference"] and parity[1]["config"].endswith("rerun")
    assert parity[1]["cosine"] > 0.9999 and parity[1]["grad_rel_l2_diff"] < 1e-6
    assert {p["config"] for p in parity} >= {"fp32 B=2", "bf16 B=1", "bf16 B=2"}
    assert all(math.isfinite(p["cosine"]) for p in parity)

    implied = res["implied"]
    assert len(implied) == len(rows)
    assert all(0.0 < i["fraction_saturated"] < 1.0 and i["train_path_s"] > 0.0 for i in implied)

    md = markdown_report(res)
    assert "encode_raw" in md and "fp32 B=1 rerun" in md and "Implied az_loop iteration" in md


def test_compiled_parity_is_scored_against_the_eager_reference(monkeypatch):
    """The compiled rows are scored against the eager fp32 B=1 result
    on the same weights, taken before the trunk is compiled -- so a
    compiled trunk that deviates fails, fp32 B=1 included. A forward
    that scales the trunk's output stands in for torch.compile."""
    import tools.bench_train_step as bts
    require_scenario_data()
    torch.manual_seed(0)
    device = torch.device("cpu")
    policy = TransformerPolicy(device=device, d_model=32, num_layers=1, num_heads=2, d_ff=64)
    states = [build_scenario_gamestate(random_setup(random.Random(i), mini_maps=True))
              for i in range(2)]
    exps = experiences_from_states(policy, states, sims=4, rng=random.Random(0))

    def deviating_compile(model):
        eager = model.encoder.forward
        model.encoder.forward = lambda *a, **k: 4.0 * eager(*a, **k)
        return {"cache_size_limit": 0}

    monkeypatch.setattr(bts, "enable_compiled_trunk", deviating_compile)
    res = bts.run_benchmark(policy, exps, device=device, n_list=[2], batch_sizes=[1],
                            precisions=["fp32"], repeats=1, parity_n=2, compile_trunk=True,
                            loop=LoopShape(games_per_iter=4, exps_per_game=2.0))
    by_name = {p["config"]: p for p in res["parity"]}
    assert by_name["fp32 B=1"]["reference"] and by_name["fp32 B=1 eager"]["reference"]
    compiled = by_name["fp32 B=1 compiled"]
    assert not compiled["reference"] and not compiled["ok"], compiled
    assert res["compile"]["active"] is True
    assert "forward" not in policy._trainer.model.encoder.__dict__, "compile left enabled"
    assert "eager fp32 B=1 on the same weights" in markdown_report(res)
