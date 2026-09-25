"""The demo plays a checkpoint in the hex basis it was trained in.

`relevant_set_hexes` is not restored by `TransformerPolicy.load_checkpoint`
(the policy is BUILT in a basis); the eval path reads it with the other
structural flags (`tools.eval_sim.peek_checkpoint_arch`). The demo built
its policy from the arch alone, so a relevant-set checkpoint such as the
reference player chose its hex targets in the full-board basis."""
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
