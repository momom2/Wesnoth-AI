"""MCTSConfig.value_center: search reads the network value minus the
centering constant (docs/archive/az_leg_20260903.md: a mover-frame level b
enters the act-vs-end_turn choice as 2b, so the head's level decides
K; centering removes that lever from the level).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

import tools.mcts as mcts  # noqa: E402
from tools.mcts import MCTSNode, _expand  # noqa: E402
from wesnoth_ai.action_sampler import LegalActionPrior  # noqa: E402


def _node(side: int = 1) -> MCTSNode:
    gs = SimpleNamespace(global_info=SimpleNamespace(current_side=side))
    return MCTSNode(SimpleNamespace(gs=gs, done=False, winner=0))


class _Model:
    def __call__(self, encoded):
        return SimpleNamespace(value=torch.tensor([[0.6]]),
                               cliffness=torch.tensor([[0.1]]),
                               moves_left=None, aux_score=None)


class _Encoder:
    def encode(self, gs):
        return SimpleNamespace()


def _one_end_turn(*a, **k):
    return [LegalActionPrior(action={"type": "end_turn"}, prior=1.0,
                             actor_idx=0, target_idx=None,
                             weapon_idx=None, type_idx=None)]


def test_expand_subtracts_value_center(monkeypatch):
    monkeypatch.setattr(mcts, "enumerate_legal_actions_with_priors",
                        _one_end_turn)
    monkeypatch.setattr(mcts, "_leaf_to_cpu", lambda e, o: (e, o))
    plain = _expand(_node(), _Model(), _Encoder(), tiebreak=None)
    assert abs(plain - 0.6) < 1e-6
    node = _node()
    centered = _expand(node, _Model(), _Encoder(), tiebreak=None,
                       value_center=0.25)
    assert abs(centered - 0.35) < 1e-6
    assert abs(node.value - 0.35) < 1e-6
    assert len(node.edges) == 1 and node.expanded


def test_value_center_defaults_off():
    assert mcts.MCTSConfig().value_center == 0.0
