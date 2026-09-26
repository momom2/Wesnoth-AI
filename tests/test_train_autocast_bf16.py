"""TrainerConfig.train_autocast_bf16, the trainer's bf16 switch for
step_mcts (az_loop --train-bf16), driven through the benchmark's own
step wrapper (tools/bench_train_step.stubbed_step: optimizer stubbed,
no clipping) so the bench's bf16 rows and the loop share one path.

On cpu the switch is a no-op: no autocast context is opened and one
step's losses and gradient are the fp32 ones bit for bit. On cuda
(skipped without one) one step at batch 16 on the bench positions
must agree with fp32 batch 1 within the benchmark's parity band:
loss 1%, gradient norm 5%, cosine >= 0.99 (the numbers are printed).
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from sim_test_helpers import require_scenario_data  # noqa: E402
from tools.bench_train_step import (  # noqa: E402
    configure_trainer_like_az_loop, experiences_from_states, parity_row, stubbed_step,
)
from tools.scenario_pool import build_scenario_gamestate, random_setup  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402

MANIFEST = ROOT / "configs" / "bench_states.json"
DATASET = Path(os.environ.get("WESNOTH_BENCH_DATASET",
                              ROOT / "replays_dataset_imitation"))
SEED_CHECKPOINT = Path(os.environ.get(
    "WESNOTH_SEED_CHECKPOINT",
    ROOT / "training" / "checkpoints" / "seed_imit_tierb_start.pt"))


class _AutocastSpy(torch.autocast):
    """Every autocast context the code under test opens, as
    (args, kwargs)."""
    calls: list = []

    def __init__(self, *args, **kwargs):
        type(self).calls.append((args, kwargs))
        super().__init__(*args, **kwargs)


@pytest.fixture
def autocast_calls(monkeypatch):
    _AutocastSpy.calls = []
    monkeypatch.setattr(torch, "autocast", _AutocastSpy)
    return _AutocastSpy.calls


def _mini_map_experiences(policy, n_states: int = 3):
    require_scenario_data()
    states = [build_scenario_gamestate(random_setup(random.Random(i), mini_maps=True))
              for i in range(n_states)]
    return experiences_from_states(policy, states, sims=8, rng=random.Random(0))


def test_switch_is_a_no_op_on_cpu(autocast_calls):
    device = torch.device("cpu")
    torch.manual_seed(0)
    policy = TransformerPolicy(device=device, d_model=32, num_layers=1, num_heads=2, d_ff=64)
    configure_trainer_like_az_loop(policy._trainer)
    exps = _mini_map_experiences(policy)
    cfg = policy._trainer.config
    assert cfg.train_autocast_bf16 is False

    ref = stubbed_step(policy, exps, precision="fp32", batch_size=2, device=device)
    cand = stubbed_step(policy, exps, precision="bf16", batch_size=2, device=device)

    assert cfg.train_autocast_bf16 is False, "the bench wrapper must restore the switch"
    assert not autocast_calls, autocast_calls
    assert ref[0]["total_loss"] != 0.0 and ref[0]["grad_norm"] > 0.0
    assert cand[0] == ref[0], (cand[0], ref[0])
    assert torch.equal(cand[1], ref[1])


def _bench_states(n: int):
    if not MANIFEST.exists():
        pytest.skip(f"{MANIFEST} missing")
    first = json.loads(MANIFEST.read_text(encoding="utf-8"))["states"][0]["file"]
    if not (DATASET / first).exists():
        pytest.skip(f"bench dataset not at {DATASET} (WESNOTH_BENCH_DATASET)")
    from tools.bench_pipeline import load_states
    return [gs for gs, _scenario in load_states(MANIFEST, DATASET, n)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a cuda device")
def test_bf16_step_matches_fp32_within_the_parity_band_on_cuda(autocast_calls):
    """The seed checkpoint when it is on this machine (the benchmark's
    subject), else a random init at the production architecture."""
    from tools.eval_players import _load_policy
    device = torch.device("cuda")
    n_states, batch = 32, 16
    states = _bench_states(n_states)
    ckpt = SEED_CHECKPOINT if SEED_CHECKPOINT.exists() else None
    policy = _load_policy(ckpt, device, label="test_train_bf16")
    configure_trainer_like_az_loop(policy._trainer)
    exps = experiences_from_states(policy, states, sims=32, rng=random.Random(0))
    assert len(exps) == n_states

    autocast_calls.clear()      # count the steps' regions only
    ref = stubbed_step(policy, exps, precision="fp32", batch_size=1, device=device)
    assert not autocast_calls
    cand = stubbed_step(policy, exps, precision="bf16", batch_size=batch, device=device)
    assert policy._trainer.config.train_autocast_bf16 is False

    # One region per chunk, on cuda, in bf16.
    assert len(autocast_calls) == -(-n_states // batch), autocast_calls
    assert all(a[0] == "cuda" and k.get("dtype") is torch.bfloat16
               for a, k in autocast_calls), autocast_calls

    row = parity_row(f"bf16 B={batch}", ref, cand)
    print(f"{'seed' if ckpt else 'random init'}: total loss fp32 {ref[0]['total_loss']:.6f} "
          f"bf16 {cand[0]['total_loss']:.6f} rel {row['loss_rel_diff']:.2e} | "
          f"grad norm ratio {row['grad_norm_ratio']:.5f} | cosine {row['cosine']:.5f} | "
          f"grad rel L2 {row['grad_rel_l2_diff']:.2e}")
    assert row["ok"], row
