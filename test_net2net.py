"""Net2Net width/depth transfer (tools/net2net, plan §3.5).

Pins: the IDENTITY case is exact (same arch -> full copy -> bit-
identical function); widening/deepening transfers the overlapping
block (incl. the QKV-split for attention in_proj), keeps new capacity
fresh, and runs; and grow_checkpoint round-trips to a loadable, larger
checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from transformer_policy import TransformerPolicy             # noqa: E402
from tools.net2net import (                                  # noqa: E402
    transfer_state_dict, grow_checkpoint, _copy_leading_block,
)
from sim_test_helpers import fresh_scenario_sim              # noqa: E402


def _pol(d_model=48, num_layers=2, num_heads=4, d_ff=96, aux=False):
    return TransformerPolicy(device=torch.device("cpu"), d_model=d_model,
                             num_layers=num_layers, num_heads=num_heads,
                             d_ff=d_ff, aux_score=aux)


def _aligned_clone(src_pol, dst_pol):
    """Copy src's vocab onto dst so encode() produces identical indices,
    then encode the same state on both (registering no new types)."""
    dst_pol._inference_encoder.unit_type_to_id = dict(
        src_pol._inference_encoder.unit_type_to_id)
    dst_pol._inference_encoder.faction_to_id = dict(
        src_pol._inference_encoder.faction_to_id)


def test_identity_transfer_is_exact_function():
    """Same arch: transfer must reproduce the source's outputs exactly."""
    src = _pol()
    dst = _pol()  # different random init
    sim = fresh_scenario_sim(seed=21, max_turns=12, mini=True)
    # Grow src's vocab to cover the state, then mirror it onto dst.
    src._inference_encoder.encode(sim.gs)
    _aligned_clone(src, dst)

    transfer_state_dict(src._inference_model.state_dict(), dst._inference_model)
    transfer_state_dict(src._inference_encoder.state_dict(),
                        dst._inference_encoder)

    with torch.no_grad():
        o_src = src._inference_model(src._inference_encoder.encode(sim.gs))
        o_dst = dst._inference_model(dst._inference_encoder.encode(sim.gs))
    assert torch.allclose(o_src.value, o_dst.value, atol=1e-6)
    assert torch.allclose(o_src.actor_logits, o_dst.actor_logits, atol=1e-6)
    assert torch.allclose(o_src.value_logits, o_dst.value_logits, atol=1e-6)


def test_wider_transfer_preserves_block_and_runs():
    src = _pol(d_model=48)
    dst = _pol(d_model=96)
    sim = fresh_scenario_sim(seed=21, max_turns=12, mini=True)
    src._inference_encoder.encode(sim.gs)
    _aligned_clone(src, dst)

    transfer_state_dict(src._inference_model.state_dict(), dst._inference_model)
    transfer_state_dict(src._inference_encoder.state_dict(),
                        dst._inference_encoder)

    src_sd = src._inference_model.state_dict()
    dst_sd = dst._inference_model.state_dict()
    # Generic 2D param: the trained block must land in the top-left.
    w_name = "encoder.layers.0.linear1.weight"
    so, si = src_sd[w_name].shape
    assert torch.allclose(dst_sd[w_name][:so, :si], src_sd[w_name], atol=1e-6)

    # QKV-stacked in_proj: each of the 3 dst blocks' overlap matches the
    # corresponding src block (NOT a naive [:3E] copy).
    ip = "encoder.layers.0.self_attn.in_proj_weight"
    de, se = dst_sd[ip].shape[0] // 3, src_sd[ip].shape[0] // 3
    for i in range(3):
        d_blk = dst_sd[ip][i * de:(i + 1) * de]
        s_blk = src_sd[ip][i * se:(i + 1) * se]
        assert torch.allclose(d_blk[:se, :s_blk.shape[1]], s_blk, atol=1e-6)

    # The wider model runs and produces finite outputs.
    with torch.no_grad():
        out = dst._inference_model(dst._inference_encoder.encode(sim.gs))
    assert torch.isfinite(out.value).all()
    assert torch.isfinite(out.actor_logits).all()


def test_deeper_transfer_keeps_extra_layer_fresh_and_runs():
    src = _pol(num_layers=2)
    dst = _pol(num_layers=3)
    sim = fresh_scenario_sim(seed=21, max_turns=12, mini=True)
    src._inference_encoder.encode(sim.gs)
    _aligned_clone(src, dst)
    rep = transfer_state_dict(src._inference_model.state_dict(),
                              dst._inference_model)
    # Layer 2 has no source match -> stays fresh.
    assert any("layers.2." in n for n in rep["fresh"])
    # Layers 0/1 transferred exactly (same width).
    assert any("layers.0." in n for n in rep["exact"])
    transfer_state_dict(src._inference_encoder.state_dict(),
                        dst._inference_encoder)
    with torch.no_grad():
        out = dst._inference_model(dst._inference_encoder.encode(sim.gs))
    assert torch.isfinite(out.value).all()


def test_grow_checkpoint_roundtrip(tmp_path):
    src_pol = _pol(d_model=48)
    sim = fresh_scenario_sim(seed=21, max_turns=12, mini=True)
    src_pol._encoder.encode(sim.gs)        # grow vocab on the TRAIN encoder
    src_p = tmp_path / "narrow.pt"
    src_pol.save_checkpoint(src_p)

    out_p = tmp_path / "wide.pt"
    rep = grow_checkpoint(src_p, out_p, d_model=96, num_heads=4)
    assert rep["dst_arch"]["d_model"] == 96
    assert out_p.exists()

    # The grown checkpoint loads cleanly into a matching policy.
    wide = _pol(d_model=96)
    wide.load_checkpoint(out_p)
    raw = torch.load(out_p, map_location="cpu", weights_only=False)
    assert raw["arch"]["d_model"] == 96
    with torch.no_grad():
        out = wide._inference_model(wide._inference_encoder.encode(sim.gs))
    assert torch.isfinite(out.value).all()


def test_copy_leading_block_handles_shrink_and_grow():
    big = torch.zeros(4, 4)
    small = torch.arange(9.0).reshape(3, 3)
    _copy_leading_block(big, small)               # grow target
    assert torch.equal(big[:3, :3], small)
    assert big[3, 3] == 0.0                        # untouched new region
    out = torch.zeros(2, 2)
    _copy_leading_block(out, small)               # shrink target
    assert torch.equal(out, small[:2, :2])
