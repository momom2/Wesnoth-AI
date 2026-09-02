"""Arm VG3: checkpoint continuation metadata + pre-flip grounding.

  - training_meta round-trips through TransformerPolicy save/load;
  - MCTSPolicy applies it when the recipe fingerprint matches and
    REFUSES it (loudly, unapplied) when trust_delta differs;
  - grounding labels from a captured pre-flip state hand the turn
    over first.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from test_inference_snapshot import _gs  # noqa: E402
from tools.mcts import MCTSConfig  # noqa: E402
from tools.mcts_policy import MCTSPolicy  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402


def test_training_meta_round_trip_and_recipe_gate(tmp_path):
    src = MCTSPolicy(TransformerPolicy(), MCTSConfig(n_simulations=1))
    src._trust_lambda = 5.5
    src._consist_bias = -0.3
    src._consist_sigma2 = 0.12
    src._dv_history = [0.3, 0.1]
    ck = tmp_path / "meta.pt"
    src.save_checkpoint(ck)

    dst = MCTSPolicy(TransformerPolicy(), MCTSConfig(n_simulations=1))
    dst.load_checkpoint(ck)
    cfg = dst._base._trainer.config
    assert dst._trust_lambda == 5.5
    assert cfg.consist_bias == -0.3 and cfg.consist_sigma2 == 0.12
    assert cfg.trust_lambda == 5.5
    assert dst._dv_history == [0.3, 0.1]

    # Different delta = different recipe: metadata must NOT apply.
    other = MCTSPolicy(TransformerPolicy(), MCTSConfig(n_simulations=1))
    other._base._trainer.config.trust_delta = 0.04
    other.load_checkpoint(ck)
    assert getattr(other, "_trust_lambda", 1.0) == 1.0
    assert other._base._trainer.config.consist_bias == 0.0


def test_meta_applies_when_wrapper_is_built_after_base_load(tmp_path):
    """The launcher loads the checkpoint into the BASE policy and
    wraps it afterwards; the wrapper must pick the stash up."""
    from tools.turn_policy import TurnCommitPolicy
    from tools.turn_search_config import TurnSearchConfig
    from tools.value_grounding import GroundingConfig
    src = TurnCommitPolicy(TransformerPolicy(), MCTSConfig(n_simulations=1),
                           turn_config=TurnSearchConfig(),
                           grounding_config=GroundingConfig(enabled=True))
    src._trust_lambda = 12.1
    src._consist_bias = 0.149
    src._consist_sigma2 = 0.052
    ck = tmp_path / "cal.pt"
    src.save_checkpoint(ck)

    base = TransformerPolicy()
    base.load_checkpoint(ck)                     # launcher order
    wrapped = TurnCommitPolicy(base, MCTSConfig(n_simulations=1),
                               turn_config=TurnSearchConfig(),
                               grounding_config=GroundingConfig(enabled=True))
    assert wrapped._trust_lambda == 12.1
    assert base._trainer.config.trust_lambda == 12.1
    assert abs(base._trainer.config.consist_bias - 0.149) < 1e-9


def test_preflip_labels_hand_the_turn_over_first():
    from tools.value_grounding import GroundingConfig, rollout_outcome

    class _Sim:
        def __init__(self):
            self.gs = _gs()
            self.gs.global_info.current_side = 1
            self.done = False
            self.steps = []

        def fork(self):
            return self

        def step(self, action):
            self.steps.append(action["type"])
            if action["type"] == "end_turn" and len(self.steps) == 1:
                # Turn handed over; the game then ends at once.
                self.done = True
                self.winner = 2

    sim = _Sim()
    cfg = GroundingConfig(enabled=True, rollout_max_halfturns=3,
                          rollout_max_actions=2)
    z = rollout_outcome(None, sim, 1, 0, cfg, np.random.default_rng(0))
    assert sim.steps[0] == "end_turn"
    assert z == -1.0        # side 1 handed over, side 2 won
