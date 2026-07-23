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
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

_DATASET = Path(__file__).parent.parent / "replays_dataset"

_ARCH = dict(d_model=64, num_layers=2, num_heads=2, d_ff=128)


@pytest.mark.skipif(not _DATASET.exists(),
                    reason="replays_dataset not present")
def test_sl_pass_round_trips_mcts_checkpoint(tmp_path):
    from wesnoth_ai.transformer_policy import TransformerPolicy
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


def test_joint_value_loss_one_hot_edges():
    """Joint SL value loss (user 2026-07-16): corpus outcomes are
    decisive by construction (z in {-1,+1}), so the C51 target is a
    one-hot on the support's edge atom and CE = -log p(edge). The
    weighted term flows into total; the raw CE is reported
    unweighted; z=None keeps the legacy policy-only behavior."""
    import torch
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.model import WesnothModel
    from tools.supervised_train import (_loss_parts_for_output,
                                        _pair_stream_serial)
    from tools.replay_dataset import filter_competitive_2p
    from pathlib import Path as _P

    files = filter_competitive_2p(_P("replays_dataset"))[:1]
    enc = GameStateEncoder(d_model=128)
    mdl = WesnothModel(d_model=128, num_layers=2, num_heads=4, d_ff=128)
    item = next(i for i in _pair_stream_serial(files) if i[0] == "pair")
    _, state, ai, _name = item
    out = mdl(enc.encode(state))
    dev = torch.device("cpu")

    off = _loss_parts_for_output(out, ai, dev)
    assert not off.value_fired and float(off.value) == 0.0

    on_win = _loss_parts_for_output(out, ai, dev, value_z=1,
                                    value_weight=0.5)
    on_loss = _loss_parts_for_output(out, ai, dev, value_z=-1,
                                     value_weight=0.5)
    assert on_win.value_fired and on_loss.value_fired
    logp = torch.log_softmax(out.value_logits[0], dim=-1)
    K = logp.shape[0]
    assert abs(float(on_win.value) - float(-logp[K - 1])) < 1e-5
    assert abs(float(on_loss.value) - float(-logp[0])) < 1e-5
    # weighted contribution lands in total
    assert abs(float(on_win.total - off.total)
               - 0.5 * float(on_win.value)) < 1e-4
    assert on_win.total.requires_grad


def test_flush_batch_carries_value_loss():
    """The batched (GPU) flow re-sums head stacks instead of using
    p.total -- the value term must be explicitly stacked in, or
    value training silently vanishes on GPU only. Drives
    _flush_batch directly with per-sample (z, weight)."""
    from collections import deque
    import torch
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.model import WesnothModel
    from tools.supervised_train import (_encode_one, _flush_batch,
                                        _pair_stream_serial)
    from tools.replay_dataset import filter_competitive_2p
    from pathlib import Path as _P

    files = filter_competitive_2p(_P("replays_dataset"))[:1]
    enc = GameStateEncoder(d_model=128)
    mdl = WesnothModel(d_model=128, num_layers=2, num_heads=4, d_ff=128)
    dev = torch.device("cpu")
    opt = torch.optim.AdamW(list(mdl.parameters())
                            + list(enc.parameters()), lr=1e-4)
    pairs = []
    for item in _pair_stream_serial(files):
        if item[0] == "pair":
            pairs.append((item[1], item[2]))
        if len(pairs) >= 2:
            break
    batch = [_encode_one(enc, st, dev) for st, _ in pairs]
    ais = [ai for _, ai in pairs]
    dq = {k: deque(maxlen=20) for k in
          ("t", "a", "ty", "tg", "w", "v")}
    _flush_batch(mdl, enc, batch, ais, [(1, 0.5), (-1, 0.5)],
                 opt, list(mdl.parameters()), 2, dev,
                 dq["t"], dq["a"], dq["ty"], dq["tg"], dq["w"],
                 dq["v"])
    assert len(dq["v"]) == 2, "value CE must be logged per sample"
    assert all(0.0 < v < 20.0 for v in dq["v"])


def test_batched_training_loop_actually_steps(tmp_path):
    """Integration guard for the 2026-07-16 stall: a duplicated
    positional arg at the _flush_batch CALL SITE made every flush
    raise TypeError, silently swallowed by the resilience except --
    the trainer encoded the whole corpus and never stepped. Unit
    tests on _flush_batch itself cannot catch call-site bugs; this
    drives train() end-to-end in batched mode and requires actual
    optimizer steps."""
    import re
    from tools.supervised_train import train

    out = tmp_path / "sl_it.pt"
    train(dataset_dir=Path("replays_dataset"), checkpoint_out=out,
          epochs=1, batch_size=8, max_pairs=24, log_every=1,
          device_str="cpu", batched_forward=True,
          d_model=128, num_layers=2, num_heads=4, d_ff=128,
          holdout_games=20, eval_every=0, eval_pairs=5,
          value_loss_weight=0.5)
    import torch
    ck = torch.load(out, map_location="cpu", weights_only=False)
    assert int(ck.get("supervised_step", 0)) >= 3, \
        "batched train() must land optimizer steps"
