"""Relevant-set imitation arm (docs/model_cost_study_20260905.md 2.5, 7):
labels in the subset basis, the legality-masked target CE, and the
checkpoint flag the eval loader reads. Real replays, tiny model, CPU.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools.replay_dataset import (_action_indices, filter_competitive_2p,
                                  iter_replay_pairs)
from wesnoth_ai.visibility import (hexes_in_slot_order,
                                   relevant_hexes_in_slot_order)

_DATASET = Path(__file__).parent.parent / "replays_dataset"
_ARCH = dict(d_model=32, num_layers=1, num_heads=2, d_ff=64)
_needs_corpus = pytest.mark.skipif(not _DATASET.exists(),
                                   reason="replays_dataset not present")


@pytest.fixture(scope="module")
def replay() -> Path:
    files = filter_competitive_2p(_DATASET)
    assert files, "no competitive-2p replay in the local corpus"
    return files[0]


def _hex_at(hexes, idx):
    return (hexes[idx].position.x, hexes[idx].position.y)


@_needs_corpus
def test_relevant_set_labels_point_at_the_same_hex_as_full_board_labels(replay):
    """Lockstep over one replay in both bases: same pair stream, same
    actor / type / weapon labels, and the subset target slot resolves
    to the hex the full-board slot resolves to. No target falls
    outside the subset."""
    full = iter_replay_pairs(replay)
    rel = iter_replay_pairs(replay, relevant_set=True)
    n = 0
    for (gs_f, ai_f), (gs_r, ai_r) in itertools.islice(zip(full, rel), 40):
        n += 1
        assert (ai_f.action_type, ai_f.actor_idx, ai_f.weapon_idx,
                ai_f.type_idx) == (ai_r.action_type, ai_r.actor_idx,
                                   ai_r.weapon_idx, ai_r.type_idx)
        assert not ai_r.target_off_subset
        if ai_f.target_idx is None:
            assert ai_r.target_idx is None
            continue
        subset = relevant_hexes_in_slot_order(gs_r)
        assert ai_r.target_idx < len(subset) < len(gs_r.map.hexes)
        assert _hex_at(subset, ai_r.target_idx) == _hex_at(
            hexes_in_slot_order(gs_f), ai_f.target_idx)
    assert n >= 10


@_needs_corpus
def test_off_subset_target_keeps_the_pair_and_flags_it(replay, monkeypatch):
    gs, _ = next(iter_replay_pairs(replay))
    side = gs.global_info.current_side
    leader = next(u for u in gs.map.units if u.side == side and u.is_leader)
    lx, ly = leader.position.x, leader.position.y

    # Off-board target: dropped in both bases (same pair stream).
    off_board = ["move", [lx, 999], [ly, 999]]
    assert _action_indices(gs, off_board) is None
    assert _action_indices(gs, off_board, relevant_set=True) is None

    # On-board target with no subset slot: pair kept, flagged.
    import wesnoth_ai.visibility as vis
    monkeypatch.setattr(vis, "relevant_hexes_in_slot_order", lambda _gs: [])
    on_board = ["move", [lx, lx], [ly, ly]]
    ai = _action_indices(gs, on_board, relevant_set=True)
    assert ai is not None and ai.action_type == "move"
    assert ai.target_idx is None and ai.target_off_subset
    ai_full = _action_indices(gs, on_board)
    assert ai_full.target_idx is not None and not ai_full.target_off_subset


def test_masked_nll_equals_plain_on_full_mask_and_differs_on_subset():
    from tools.supervised_train import _masked_target_nll
    torch.manual_seed(0)
    row = torch.randn(50)
    t = 7
    plain = float(F.cross_entropy(row.unsqueeze(0), torch.tensor([t])))

    nll, _hit = _masked_target_nll(row, torch.ones(50, dtype=torch.bool), t)
    assert abs(nll - plain) < 1e-5

    half = torch.zeros(50, dtype=torch.bool)
    half[::2] = True
    half[t] = True
    nll_half, _ = _masked_target_nll(row, half, t)
    assert nll_half < plain - 1e-3

    only_t = torch.zeros(50, dtype=torch.bool)
    only_t[t] = True
    assert _masked_target_nll(row, only_t, t) == (0.0, True)

    without_t = half.clone()
    without_t[t] = False
    assert _masked_target_nll(row, without_t, t) is None


@_needs_corpus
@pytest.mark.parametrize("relevant_set", [False, True])
def test_evaluate_reports_masked_target_ce(replay, relevant_set):
    from tools.supervised_train import _evaluate
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.model import WesnothModel
    torch.manual_seed(0)
    enc = GameStateEncoder(d_model=_ARCH["d_model"],
                           relevant_set_hexes=relevant_set)
    model = WesnothModel(**_ARCH)
    stats = _evaluate(model, enc, [replay], torch.device("cpu"),
                      eval_pairs=8)
    assert stats["n"] == 8
    assert stats["mask_errors"] == 0
    assert stats["target_off_subset"] == 0
    assert stats["target_masked_n"] + stats["target_off_mask"] \
        == stats["target_n"] > 0
    assert stats["target_masked_ce"] == stats["target_masked_ce"]  # finite
    assert 0.0 <= stats["target_masked_top1"] <= 1.0


@_needs_corpus
def test_relevant_set_checkpoint_is_peeked_and_loaded_in_that_basis(tmp_path):
    """train(--init-from full-board seed, --relevant-set-hexes) writes a
    checkpoint that eval_sim's loader builds in the subset basis; the
    warm start copies the weights and resets the counters."""
    from tools.eval_sim import _load_policy, peek_checkpoint_arch
    from tools.supervised_train import _save_checkpoint, train
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.model import WesnothModel

    torch.manual_seed(1)
    enc = GameStateEncoder(d_model=_ARCH["d_model"])
    model = WesnothModel(**_ARCH)
    opt = torch.optim.AdamW(list(model.parameters()) + list(enc.parameters()))
    src = tmp_path / "seed.pt"
    _save_checkpoint(src, model, enc, opt, step=999, pairs=99_999,
                     epoch=1, arch=_ARCH)
    assert "relevant_set_hexes" not in peek_checkpoint_arch(src)

    out = tmp_path / "arm.pt"
    train(dataset_dir=_DATASET, checkpoint_out=out, epochs=1,
          batch_size=2, max_pairs=4, max_replays=2, lr=0.0,
          holdout_games=0, eval_every=0, device_str="cpu",
          init_from=src, relevant_set_hexes=True, seed=3, **_ARCH)

    peek = peek_checkpoint_arch(out)
    assert peek.get("relevant_set_hexes") is True
    assert {k: peek[k] for k in _ARCH} == _ARCH
    ck = torch.load(out, map_location="cpu", weights_only=False)
    assert 1 <= ck["supervised_step"] <= 2 and ck["supervised_epoch"] == 0
    assert ck["training_meta"]["init_from"] == str(src)
    # lr=0: every weight the warm start copied is still the seed's.
    src_state = torch.load(src, map_location="cpu",
                           weights_only=False)["model_state"]
    assert src_state and all(torch.equal(v, ck["model_state"][k])
                             for k, v in src_state.items())

    pol = _load_policy(out, torch.device("cpu"), "arm")
    assert pol._relevant_set_hexes
    assert pol._inference_encoder.relevant_set_hexes
