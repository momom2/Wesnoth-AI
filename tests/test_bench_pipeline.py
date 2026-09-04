"""Pipeline benchmark harness (tools/bench_pipeline.py, plan step 1.1)."""
from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

DATASET = Path(__file__).parent.parent / "replays_dataset_imitation"


def _tiny_policy():
    import torch
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(0)
    return TransformerPolicy(d_model=32, num_layers=1, num_heads=2,
                             d_ff=64, device=torch.device("cpu"))


def _scenario_states(n):
    from tools.scenario_pool import build_scenario_gamestate, random_setup
    out = []
    for i in range(n):
        setup = random_setup(random.Random(10 + i))
        out.append((build_scenario_gamestate(setup), setup.scenario_id))
    return out


def test_select_holdout_ladder_filters_on_both_flags():
    from tools.bench_pipeline import select_holdout_ladder
    rows = [{"file": "a", "holdout": True}, {"file": "b", "holdout": False},
            {"file": "c", "holdout": True}]
    scen = {"a": "multiplayer_Hamlets", "b": "multiplayer_Hamlets", "c": "2p_mini"}
    got = select_holdout_ladder(rows, lambda r: scen[r["file"]], ["multiplayer_Hamlets"])
    assert [r["file"] for r in got] == ["a"]


def test_bucket_edges_are_quantiles():
    from tools.bench_pipeline import bucket_edges
    assert bucket_edges(list(range(1, 101)), 4) == [25, 50, 75, 100]
    assert bucket_edges([5, 9, 7], 2) == [5, 9]


def test_component_costs_on_scenario_states():
    from tools.bench_pipeline import COMPONENTS, component_costs
    costs = component_costs(_scenario_states(2), _tiny_policy(), repeats=1)
    assert set(costs) == set(COMPONENTS)
    assert all(v >= 0.0 for v in costs.values())
    assert costs["encode_raw"] > 0.0 and costs["enumerate_priors"] > 0.0


def test_forward_costs_report_per_bucket_and_batch():
    from tools.bench_pipeline import forward_costs
    rows = forward_costs(_tiny_policy(), _scenario_states(3), batch_sizes=(1, 2),
                         samples_per_config=2, k_buckets=2)
    assert rows and {r["batch"] for r in rows} == {1, 2}
    assert all(r["samples_per_s"] > 0 and r["ms_per_sample"] > 0 for r in rows)


def test_markdown_report_lists_every_section():
    from tools.bench_pipeline import markdown_report
    res = {"label": "x", "components": {"fork": 0.5},
           "forwards": [{"tokens_max": 900, "batch": 4, "ms_per_sample": 1.25,
                         "samples_per_s": 800.0}],
           "games": {"raw_vs_raw": {"games": 20, "secs_per_game_median": 12.0,
                                    "forwards_a_mean": 110.0, "games_per_hour": 1200.0,
                                    "games_per_dollar": 3600.0}}}
    md = markdown_report(res)
    assert "| fork | 0.500 |" in md and "| 900 | 4 | 1.25 | 800 |" in md
    assert "| raw_vs_raw | 20 | 12 | 110 | 1200 | 3600 |" in md


@pytest.mark.skipif(not (DATASET / "manifest.jsonl").exists(),
                    reason="imitation dataset not present")
def test_reconstruct_boundary_begins_the_cut_side_turn():
    from tools.bench_pipeline import reconstruct_boundary
    rows = [json.loads(ln) for ln in (DATASET / "manifest.jsonl").open(encoding="utf-8")]
    row = next(r for r in rows if r.get("holdout") and r["n_turns"] >= 6)
    with gzip.open(DATASET / row["file"], "rt", encoding="utf-8") as f:
        data = json.load(f)
    gs, begin_side = reconstruct_boundary(data, 3)
    assert begin_side in (1, 2)
    assert gs.global_info.current_side == begin_side
    assert gs.global_info.turn_number >= 3
    assert {1, 2} <= {u.side for u in gs.map.units if u.is_leader}


def test_pack_states_copies_only_needed_files(tmp_path):
    from tools.bench_pipeline import pack_states
    ds = tmp_path / "ds"
    ds.mkdir()
    for name in ("g1.json.gz", "g2.json.gz", "g3.json.gz"):
        with gzip.open(ds / name, "wt", encoding="utf-8") as f:
            json.dump({"commands": []}, f)
    (ds / "manifest.jsonl").write_text(
        "\n".join(json.dumps({"file": n, "holdout": True}) for n in
                  ("g1.json.gz", "g2.json.gz", "g3.json.gz")) + "\n", encoding="utf-8")
    man = tmp_path / "bench_states.json"
    man.write_text(json.dumps({"states": [{"file": "g1.json.gz"}, {"file": "g3.json.gz"},
                                          {"file": "g1.json.gz"}]}), encoding="utf-8")
    out = tmp_path / "packed"
    assert pack_states(man, ds, out) == 2
    assert sorted(p.name for p in out.glob("*.json.gz")) == ["g1.json.gz", "g3.json.gz"]
    rows = [json.loads(ln) for ln in (out / "manifest.jsonl").open(encoding="utf-8")]
    assert sorted(r["file"] for r in rows) == ["g1.json.gz", "g3.json.gz"]


def test_summarize_games_counts_outcomes(tmp_path):
    from tools.bench_pipeline import _summarize_games
    for i, (o, secs) in enumerate([("win", 10.0), ("loss", 20.0), ("timeout", 30.0)]):
        (tmp_path / f"game_a_b_s1_{i}.json").write_text(json.dumps(
            {"outcome_a": o, "secs": secs, "forwards_a": 100, "forwards_b": 50,
             "turns": 10}), encoding="utf-8")
    s = _summarize_games(tmp_path, wall_s=60.0, dollars_per_hour=0.5)
    assert s["games"] == 3 and s["outcomes_a"] == {"win": 1, "loss": 1, "timeout": 1}
    assert abs(s["decisive_frac"] - 2 / 3) < 1e-9
    assert s["games_per_hour"] == 180.0 and s["games_per_dollar"] == 360.0


def test_seam_costs_report_both_protocols():
    from tools.bench_pipeline import seam_costs
    rows = seam_costs(_tiny_policy(), _scenario_states(3), batch_sizes=(2,), calls=2)
    assert {r["protocol"] for r in rows} == {"logits", "priors"}
    by = {r["protocol"]: r for r in rows}
    assert by["priors"]["wire_bytes_per_leaf"] < by["logits"]["wire_bytes_per_leaf"]
    assert all(r["leaves_per_s"] > 0 for r in rows)
