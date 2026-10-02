"""A checkpoint is played and saved in the hex basis it was trained in.

The eval path reads `relevant_set_hexes` with the other structural flags
(`tools.eval_players.peek_checkpoint_arch`), and since 2026-09-29
`TransformerPolicy.load_checkpoint` restores it as it restores the fog gate
and the terrain view: the demo built its policy from the arch alone, and the
value tools built theirs without the basis, so a relevant-set checkpoint
such as the reference player chose its hex targets in the full-board basis
and was saved back in it."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_ARCH = dict(device=torch.device("cpu"), d_model=32, num_layers=1, num_heads=2, d_ff=64)


def test_demo_player_keeps_the_checkpoint_basis(tmp_path):
    from tools.sim_demo_game import load_player
    from wesnoth_ai.transformer_policy import TransformerPolicy
    relset = tmp_path / "relset.pt"
    full = tmp_path / "full.pt"
    TransformerPolicy(relevant_set_hexes=True, **_ARCH).save_checkpoint(relset)
    TransformerPolicy(relevant_set_hexes=False, **_ARCH).save_checkpoint(full)

    for ckpt, want in ((relset, True), (full, False)):
        player = load_player(ckpt, torch.device("cpu"), mcts_sims=0,
                             temperature=0.0, end_turn_offset=-1.5)
        base = player._base
        assert base._encoder.relevant_set_hexes is want, ckpt.name
        assert base._inference_encoder.relevant_set_hexes is want, ckpt.name


def test_a_loaded_checkpoint_keeps_its_basis_and_saves_it(tmp_path):
    from wesnoth_ai.transformer_policy import TransformerPolicy
    relset = tmp_path / "relset.pt"
    TransformerPolicy(relevant_set_hexes=True, **_ARCH).save_checkpoint(relset)
    policy = TransformerPolicy(relevant_set_hexes=False, **_ARCH)
    policy.load_checkpoint(relset)
    assert policy._encoder.relevant_set_hexes and policy._inference_encoder.relevant_set_hexes
    resaved = tmp_path / "resaved.pt"
    policy.save_checkpoint(resaved)
    assert torch.load(resaved, map_location="cpu", weights_only=True)["relevant_set_hexes"] is True
