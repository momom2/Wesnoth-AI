#!/usr/bin/env python3
"""SL <-> MCTS checkpoint round-trip contract (2026-07-16).

The campaign flow is: latest MCTS checkpoint -> supervised
(behavior-cloning) pass on the human corpus -> back into MCTS
self-play, always building on the latest checkpoint (user policy).
That round-trip silently broke three ways before this test:

  1. supervised _save_checkpoint hardcoded the 471K-era arch header
     (128/3/4/256) -- the policy loader hard-errors on arch
     mismatch (or a 128 policy would truncate-load a 256 net).
  2. The SL trainer constructed WesnothModel() without the optional
     aux_score / moves_left heads, so strict=False resume DROPPED
     the campaign's trained aux head as "unexpected keys".
  3. decision_step (combat-oracle anneal position) wasn't carried,
     so the next MCTS resume restarted the anneal at full strength.

This test drives the REAL production path end to end: save a
policy checkpoint, run the SL trainer resumed from it for a few
pairs, load the SL output back into a fresh policy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

_DATASET = Path(__file__).parent / "replays_dataset"

_ARCH = dict(d_model=64, num_layers=2, num_heads=2, d_ff=128)


@pytest.mark.skipif(not _DATASET.exists(),
                    reason="replays_dataset not present")
def test_sl_pass_round_trips_mcts_checkpoint(tmp_path):
    from transformer_policy import TransformerPolicy
    from tools.supervised_train import train

    # 1. An "MCTS campaign" checkpoint: aux + moves_left heads ON,
    #    a nonzero decision_step, some vocab.
    pol = TransformerPolicy(device=torch.device("cpu"),
                            aux_score=True, moves_left=True, **_ARCH)
    pol._decision_step = 123_456
    src = tmp_path / "campaign.pt"
    pol.save_checkpoint(src)
    aux_w_before = {
        k: v.clone() for k, v in pol._model.state_dict().items()
        if k.startswith("aux_score_head.")}
    assert aux_w_before, "fixture: aux head must exist"

    # 2. SL pass resumed from it (tiny: 6 pairs, no holdout/eval).
    out = tmp_path / "supervised.pt"
    train(
        dataset_dir=_DATASET,
        checkpoint_out=out,
        epochs=1, batch_size=2, max_pairs=6, max_replays=2,
        holdout_games=0, eval_every=0,
        device_str="cpu", resume=src,
        **_ARCH,
    )
    assert out.exists()

    # 3. Load the SL output back into a fresh policy of the SAME
    #    shape -- must not raise (arch header), must carry
    #    decision_step, and the aux head must ride through UNCHANGED
    #    (no SL loss touches it; AdamW skips grad-less params).
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert ckpt["arch"] == _ARCH, f"arch header lies: {ckpt['arch']}"
    assert ckpt.get("decision_step") == 123_456
    assert ckpt.get("aux_score") is True
    assert ckpt.get("moves_left") is True

    pol2 = TransformerPolicy(device=torch.device("cpu"),
                             aux_score=True, moves_left=True, **_ARCH)
    pol2.load_checkpoint(out)
    assert pol2._decision_step == 123_456
    after = pol2._model.state_dict()
    for k, v in aux_w_before.items():
        assert k in after, f"aux head key lost in SL pass: {k}"
        assert torch.equal(v, after[k]), \
            f"aux head weights changed through the SL pass: {k}"
