"""The self-play learner's signal telemetry (tools/az_signal.py): the
probe leaves training bit for bit as it was; the loss terms it splits the
gradient by add up to the loss step_mcts optimizes; a term weighed at
zero has no share. A tiny network on the CPU, positions of the
hand-built board of tests/helpers/tiny_state.py."""
import random

import pytest
import torch

from helpers.tiny_state import _gs
from tools.az_loop import COLUMNS
from tools.az_recipe import configure_az_trainer
from tools.az_signal import AZ_SIGNAL_COLUMNS, SelfPlaySignal
from tools.signal_telemetry import read_signal_rows
from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors
from wesnoth_ai.classes import Attack, DamageType
from wesnoth_ai.trainer import MCTS_LOSS_TERMS, MCTS_POLICY_TERMS, MCTSExperience
from wesnoth_ai.transformer_policy import TransformerPolicy


def _policy(seed: int, **heads) -> TransformerPolicy:
    """A tiny network whose trainer the az recipe configures."""
    torch.manual_seed(seed)
    policy = TransformerPolicy(device=torch.device("cpu"), d_model=32, num_layers=1,
                               num_heads=2, d_ff=64, **heads)
    configure_az_trainer(policy._trainer, lr=1e-3)
    return policy


def _experiences(policy: TransformerPolicy, n: int, seed: int):
    """`n` positions of the tiny board with the units' hit points drawn
    at random and a second, ranged weapon for every unit (one weapon
    leaves the weapon head no choice, hence no gradient), each with visit
    counts on every legal attack and six other legal actions (so every
    policy head has a term), an outcome, a game and a game weight."""
    rng = random.Random(seed)
    encoder, model = policy._inference_encoder, policy._inference_model
    experiences = []
    for i in range(n):
        gs = _gs()
        for unit in gs.map.units:
            unit.current_hp = rng.randint(8, 40)
            unit.attacks.append(Attack(type_id=DamageType.FIRE, number_strikes=2,
                                       damage_per_strike=6, is_ranged=True, weapon_specials=set()))
        encoder.register_names(gs)
        with torch.no_grad():
            encoded = encoder.encode(gs)
            legal = enumerate_legal_actions_with_priors(encoded, model(encoded), gs)
        attacks = [la for la in legal if la.action.get("type") == "attack"]
        others = rng.sample([la for la in legal if la.action.get("type") != "attack"], 6)
        assert attacks, "the tiny board offers no attack"
        visits = [(la.actor_idx, la.target_idx, la.weapon_idx, float(rng.randint(1, 8)), la.type_idx)
                  for la in attacks + others]
        experiences.append(MCTSExperience(
            game_state=gs, visit_counts=visits, z=rng.choice((-1.0, 1.0)),
            game_weight=1.0 / (1 + i % 3), game_id=f"g{i // 2}"))
    return experiences


def _params(trainer):
    return [p for p in list(trainer.model.parameters()) + list(trainer.encoder.parameters())
            if p.requires_grad]


def _probe_rows(path):
    return [r for r in read_signal_rows(path, progress="decision_step") if r["kind"] == "probe"]


def _training_state(trainer):
    """Everything a later step reads: the weights, the optimizer's state,
    the generators, the vocabulary and the modules' modes."""
    return {
        "params": [p.detach().clone() for p in _params(trainer)],
        "optimizer": [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in s.items()}
                      for s in (trainer.optimizer.state.get(p, {}) for p in _params(trainer))],
        "trainer_rng": trainer.rng.getstate(),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "vocabulary": dict(trainer.encoder.unit_type_to_id),
        "modes": (trainer.model.training, trainer.encoder.training),
    }


def _assert_same_state(a, b):
    assert all(torch.equal(x, y) for x, y in zip(a["params"], b["params"]))
    for sa, sb in zip(a["optimizer"], b["optimizer"]):
        assert sa.keys() == sb.keys()
        assert all(torch.equal(torch.as_tensor(sa[k]), torch.as_tensor(sb[k])) for k in sa)
    assert torch.equal(a["torch_rng"], b["torch_rng"])
    for key in ("trainer_rng", "python_rng", "vocabulary", "modes"):
        assert a[key] == b[key], key


def test_the_probe_leaves_training_bit_identical(tmp_path):
    """Three steps of the az recipe in chunks of three, with a probe of
    five of the seven states after each step and without: the weights,
    the optimizer's state, the trainer's, torch's and Python's generators,
    the vocabulary and the modules' modes come out the same, bit for bit.
    Between the steps the modules are put in train mode, which the probe
    must hand back."""
    def run(probe: bool):
        policy = _policy(seed=7)
        trainer = policy._trainer
        trainer.config.train_batch_size = 3
        experiences = _experiences(policy, 7, seed=8)
        signal = (SelfPlaySignal(tmp_path / "signal.jsonl", trainer, probe_states=5, seed=1)
                  if probe else None)
        torch.manual_seed(99)
        random.seed(99)
        trainer.rng.seed(99)
        for it in range(3):
            trainer.step_mcts(experiences[it:] + experiences[:it])
            trainer.model.train()
            trainer.encoder.train()
            if signal is not None:
                columns = signal.record(experiences, it=it, decision_step=10 * it)
                assert set(columns) == set(AZ_SIGNAL_COLUMNS) and set(columns) <= set(COLUMNS)
                assert columns["sig_probe_states"] == 5 and columns["sig_probe_failures"] == 0
                assert columns["sig_trunk_value_share_update"] is not None
        return _training_state(trainer), signal

    plain, _ = run(probe=False)
    probed, signal = run(probe=True)
    _assert_same_state(plain, probed)
    rows = _probe_rows(signal.path)
    assert [r["iter"] for r in rows] == [0, 1, 2] and signal.failures == 0
    assert all(r["probe_states"] == 5 and "probe_error" not in r for r in rows)


def test_the_terms_add_up_to_the_loss_the_step_optimizes():
    """Every term on, the value loss at a coefficient other than one, in
    three chunks: the terms' summed gradient is the gradient step_mcts
    accumulates before its clip, their summed value is the loss it
    reports, and each term has a gradient of its own."""
    policy = _policy(seed=11, aux_score=True, moves_left=True)
    trainer = policy._trainer
    config = trainer.config
    config.train_batch_size = 3
    config.value_coef, config.aux_coef, config.moves_left_coef = 0.7, 0.3, 0.2
    config.trust_lambda, config.consist_sigma2, config.consist_bias = 0.5, 0.4, 0.1
    rng = random.Random(3)
    experiences = _experiences(policy, 7, seed=12)
    for i, e in enumerate(experiences):
        e.aux_target = rng.uniform(-0.9, 0.9)
        e.moves_left_target = rng.uniform(0.0, 1.0)
        if i % 3 == 0:
            e.label_kind, e.z = "consist", rng.uniform(-0.8, 0.8)
        if i % 2:
            e.v_anchor = rng.uniform(-0.5, 0.5)
    params = _params(trainer)

    summed = {}

    def keep(terms):
        assert tuple(terms) == MCTS_LOSS_TERMS
        for name, term in terms.items():
            summed[name] = summed.get(name, 0.0) + term

    trainer.mcts_loss_terms(experiences, keep)
    assert not summed.pop("gbc").requires_grad          # no GBC head on this network
    for name, term in summed.items():
        own = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        assert sum(float(g.pow(2).sum()) for g in own if g is not None) > 0, name
    from_terms = torch.autograd.grad(sum(summed.values()), params, allow_unused=True)

    trainer.optimizer.step = lambda *a, **k: None       # the step's gradient, not its update
    config.grad_clip = 1e9
    stats = trainer.step_mcts(experiences)
    for p, g in zip(params, from_terms):
        step_grad = torch.zeros_like(p) if p.grad is None else p.grad
        g = torch.zeros_like(p) if g is None else g
        assert float((g - step_grad).norm()) <= 1e-5 * float(step_grad.norm()) + 1e-9
    assert sum(float(t.detach()) for t in summed.values()) == pytest.approx(stats.total_loss, rel=1e-5)
    assert sum(float(summed[head].detach()) for head in MCTS_POLICY_TERMS) == \
        pytest.approx(stats.policy_loss, rel=1e-5)


def test_a_term_weighed_at_zero_has_no_share(tmp_path):
    """The value coefficient at zero: the value term's norm and share are
    zero in every group of both spaces and the policy heads' shares add
    up to one; at the recipe's coefficient of one the value term has a
    share of the trunk's update."""
    shares = {}
    for coef in (0.0, 1.0):
        policy = _policy(seed=13)
        trainer = policy._trainer
        trainer.config.value_coef = coef
        experiences = _experiences(policy, 6, seed=14)
        trainer.step_mcts(experiences)        # the optimizer's state: the update space
        signal = SelfPlaySignal(tmp_path / f"coef_{coef}.jsonl", trainer, seed=2)
        columns = signal.record(experiences, it=0, decision_step=0)
        row = _probe_rows(signal.path)[-1]
        shares[coef] = columns["sig_trunk_value_share_update"]
        if coef:
            continue
        for space in ("gradient", "update"):
            for group in ("encoder", "trunk", "heads", "all"):
                node = row[space][group]
                assert node["value"]["norm"] == 0.0 and node["value"]["share"] == 0.0
                policy_share = sum(node[head]["share"] for head in MCTS_POLICY_TERMS)
                assert abs(policy_share - 1.0) < 1e-6, (space, group)
    assert shares[0.0] == 0.0 and abs(shares[1.0]) > 1e-6
