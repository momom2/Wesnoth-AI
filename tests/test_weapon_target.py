"""The weapon's score reads the target (wesnoth_ai/model.py, `weapon_target`;
docs/blindspot_audit_20261125.md, gap 3).

A network without the pointer scores a unit's weapon the same whichever
enemy it strikes. These tests pin: a network given the pointer at zero is
its source to the last bit; with the pointer away from zero the weapon's
law differs by target, and the inference server, the two enumerations on
the CPU and the gradient pass give one probability to each action; the
loop's tools play, pass and step such a network, Adam meeting the new
parameters as ones it has never stepped; and a reader of the per-actor
scores alone fails instead of leaving the target out.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wesnoth_ai.game_core import game_core_class  # noqa: E402

pytestmark = pytest.mark.skipif(game_core_class() is None, reason="wesnoth_core.GameCore not available")

CPU = torch.device("cpu")
RECIPE = dict(relevant_set=True, fog_hides_enemy_villages=True, terrain_multi_hot=True, observation_parity=True,
              relevant_set_version=2)


def _tiny_checkpoint(tmp_path):
    from helpers.pg_games import save_network, tiny_network
    vocab, encoder, model = tiny_network()
    return save_network(tmp_path / "pi.pt", vocab, encoder, model)


def _loud(checkpoint: Path, out: Path) -> Path:
    """`checkpoint` with the pointer's keys away from zero, so the target
    moves the weapon's score."""
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    torch.manual_seed(7)
    ck["model_state"]["weapon_k_proj.weight"] = 2.0 * torch.randn_like(ck["model_state"]["weapon_k_proj.weight"])
    torch.save(ck, out)
    return out


def _attack_positions(checkpoint: Path, games_dir: Path, seeds, want: int = 12):
    """Positions of games `checkpoint` plays where a unit has two enemies
    to attack: (raw encoding with its masks, packed masks)."""
    from helpers.pg_games import selfplay_games
    from tools.game_record import read_records, walk_core
    from tools.pg_pass import load_network
    from wesnoth_ai.server_priors import core_packed_masks
    encoder, _model = load_network(checkpoint, CPU)
    vocab = dict(type_to_id=encoder.unit_type_to_id, faction_to_id=encoder.faction_to_id)
    records = selfplay_games(checkpoint, games_dir, seeds, max_turns=10)
    out = []
    for path in records.values():
        [rec] = list(read_records(path))
        for _k, cs, cmd in walk_core(rec):
            if cmd[0] == "init_side" or int(cs.core.current_side) not in (1, 2) or len(out) >= want:
                continue
            raw = cs.encode_raw(masks=True, **vocab, **RECIPE)
            masks = core_packed_masks(raw)
            H = masks.n_hexes
            attacks = np.unpackbits(masks.attack_valid, axis=1)[:, :H].sum(axis=1)
            if (attacks >= 2).any():
                out.append((raw, masks))
    return records, out


def test_a_network_given_the_pointer_at_zero_is_its_source(tmp_path):
    """The source's weights are untouched, the keys are zero, and on
    positions with attacks every head gives the source's output to the last
    bit; the weapon's scores, read for each target, are the source's
    per-actor scores. The games the two play from the same seeds are the
    same games, command for command, with the same logged probabilities:
    the serving path adds nothing either."""
    from helpers.pg_games import SLOTS, initial_forward, selfplay_games
    from tools.add_weapon_pointer import NEW_WEIGHTS, PointerRefused, add_pointer
    from tools.eval_players import peek_checkpoint_arch
    from tools.eval_provenance import file_sha256
    from tools.game_record import read_records
    from tools.pg_pass import load_network
    from wesnoth_ai.constants import WEAPON_TARGET_RANK
    from wesnoth_ai.model_output import weapon_logits_for
    source = _tiny_checkpoint(tmp_path)
    pointed = tmp_path / "pi_pointer.pt"
    rec = add_pointer(source, pointed)
    assert rec["base_sha256"] == file_sha256(source) and rec["checkpoint_sha256"] == file_sha256(pointed)
    ck = torch.load(pointed, map_location="cpu", weights_only=True)
    src = torch.load(source, map_location="cpu", weights_only=True)
    assert ck["weapon_target"] == WEAPON_TARGET_RANK and ck["weapon_pointer_added"]["base_sha256"] == rec["base_sha256"]
    assert list(ck["model_state"])[-3:] == list(NEW_WEIGHTS)
    assert not ck["model_state"]["weapon_k_proj.weight"].any() and ck["model_state"]["weapon_q_proj.weight"].any()
    for name, weight in src["model_state"].items():
        assert torch.equal(ck["model_state"][name], weight), name
    with pytest.raises(PointerRefused, match="already"):
        add_pointer(pointed, tmp_path / "again.pt")
    assert peek_checkpoint_arch(pointed)["weapon_target"] == WEAPON_TARGET_RANK
    assert not peek_checkpoint_arch(source).get("weapon_target")

    enc1, model1 = load_network(source, CPU)
    enc2, model2 = load_network(pointed, CPU)
    assert (model1.weapon_target, model2.weapon_target) == (0, WEAPON_TARGET_RANK)
    _records, positions = _attack_positions(source, tmp_path / "games_probe", [(1, 13), (2, 7)])
    assert len(positions) >= 6, "too few positions with two enemies to attack"
    raws = [raw for raw, _masks in positions]
    with torch.no_grad():
        out1 = initial_forward(model1, enc1, raws, SLOTS)
        out2 = initial_forward(model2, enc2, raws, SLOTS)
    for name in ("actor_logits", "type_logits", "target_logits", "value", "value_logits"):
        assert torch.equal(getattr(out1, name), getattr(out2, name)), name
    assert out2.weapon_logits is None and torch.equal(out2.weapon_actor_logits, out1.weapon_logits)
    B, A, H = out1.target_logits.shape
    b, a, h = torch.meshgrid(torch.arange(B), torch.arange(A), torch.arange(H), indexing="ij")
    per_target = weapon_logits_for(out2, b, a, h)                              # [B, A, H, W]
    assert torch.equal(per_target, out1.weapon_logits.float().unsqueeze(2).expand_as(per_target))

    slots = [(1, 13), (2, 47)]
    games1 = selfplay_games(source, tmp_path / "games_source", slots)
    games2 = selfplay_games(pointed, tmp_path / "games_pointed", slots)
    for stem in games1:
        [r1], [r2] = list(read_records(games1[stem])), list(read_records(games2[stem]))
        assert r1["commands"] == r2["commands"], stem
        assert [d["logp"] for d in r1["decisions"]] == [d["logp"] for d in r2["decisions"]], stem


def test_with_the_pointer_the_weapon_depends_on_the_target_and_every_path_agrees(tmp_path):
    """With the keys away from zero: (1) a unit's weapon law differs between
    two enemies it can attack; (2) the inference server's compact priors,
    the vectorized enumeration and the per-actor reference enumeration give
    the same legal actions with the same probabilities; (3) the gradient
    pass reproduces the probabilities the players logged through the
    server (its exact check)."""
    import json
    from helpers.pg_games import SLOTS, initial_forward, selfplay_games
    import tools.pg_game_pass as gp
    from tools.add_weapon_pointer import add_pointer
    from tools.pg_pass import load_network
    from wesnoth_ai.action_sampler import _enumerate_legal_actions_reference, enumerate_legal_actions_with_priors
    from wesnoth_ai.model_output import weapon_logits_for
    from wesnoth_ai.server_priors import batched_priors
    source = _tiny_checkpoint(tmp_path)
    pointed = tmp_path / "pi_pointer.pt"
    add_pointer(source, pointed)
    loud = _loud(pointed, tmp_path / "pi_loud.pt")
    encoder, model = load_network(loud, CPU)

    records, positions = _attack_positions(loud, tmp_path / "games_probe", [(1, 13), (2, 7), (1, 47)])
    assert len(positions) >= 6, "too few positions with two enemies to attack"
    raws = [raw for raw, _masks in positions]
    packs = [masks for _raw, masks in positions]
    with torch.no_grad():
        padded = initial_forward(model, encoder, raws, SLOTS)
        compact = batched_priors(padded, packs)
    # (1) The weapon's law by target.
    spread = 0.0
    for i, masks in enumerate(packs):
        attacks = np.unpackbits(masks.attack_valid, axis=1)[:, :masks.n_hexes]
        for actor in np.flatnonzero(attacks.sum(axis=1) >= 2).tolist():
            n = int(masks.n_attacks[actor])
            if n < 2:
                continue
            hexes = torch.as_tensor(np.flatnonzero(attacks[actor]))
            laws = torch.softmax(weapon_logits_for(padded, torch.full_like(hexes, i),
                                                   torch.full_like(hexes, actor), hexes)[:, :n], dim=-1)
            spread = max(spread, float((laws - laws[0]).abs().max()))
    assert spread > 0.05, f"the weapon's law barely moves with the target ({spread})"
    assert sum(int((c.kind == 0).sum()) for c in compact) > 0, "the server lists attacks"
    # (2) One probability an action, by three routes: on a tiny policy with
    # the pointer, its keys away from zero, over played states with attacks
    # (the fixtures of tests/test_server_priors.py).
    from helpers.priors_parity import _same, _states
    from wesnoth_ai.constants import WEAPON_TARGET_RANK
    from wesnoth_ai.server_priors import pack_masks, unpack_compact
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(3)
    policy = TransformerPolicy(d_model=32, num_layers=1, num_heads=2, d_ff=64, device=CPU,
                               weapon_target=WEAPON_TARGET_RANK)
    enc_p, model_p = policy._inference_encoder, policy._inference_model
    with torch.no_grad():
        model_p.weapon_k_proj.weight.normal_(0.0, 2.0)
    states = _states()
    encs = [enc_p.encode(gs) for gs in states]
    packs_p = [pack_masks(e, gs) for e, gs in zip(encs, states)]
    with torch.no_grad():
        compact_p = batched_priors(model_p.forward_padded(encs), packs_p)
        for e, gs, out, c in zip(encs, states, model_p.forward_batch(encs), compact_p):
            assert out.weapon_logits is None and out.weapon_q is not None
            fast = enumerate_legal_actions_with_priors(e, out, gs)
            _same(fast, unpack_compact(c, e))
            _same(_enumerate_legal_actions_reference(e, out, gs), fast)
    assert 0 in {k for c in compact_p for k in c.kind.tolist()}, "attacks exercised"
    # (3) The pass against the served probabilities.
    games = tmp_path / "games_loud"
    selfplay_games(loud, games, [(1, 13), (2, 47)])
    grad = tmp_path / "grad.pt"
    assert gp.main(["--checkpoint", str(loud), "--games", str(games), "--out", str(grad), "--device", "cpu",
                    "--workers", "0"]) == 0
    replay = json.loads(grad.with_suffix(".json").read_text(encoding="utf-8"))["replay"]
    assert replay["half_mean_square"] < 1e-9, replay
    total = torch.load(grad, map_location="cpu", weights_only=True)["total"]
    assert float(total["model.weapon_q_proj.weight"].abs().max()) > 0, "the queries get a gradient"
    assert float(total["model.weapon_k_proj.weight"].abs().max()) > 0, "and the keys"
    assert records


def test_the_loops_tools_play_pass_and_step_a_network_given_the_pointer(tmp_path):
    """A round of the simple loop on each side of the change, on a tiny
    network: games played and passed and one Adam step without the pointer;
    the stepped checkpoint and its optimizer state given the pointer; then
    games played by the new checkpoint, whose pass reproduces the players'
    logged probabilities, and a step from the carried state. The optimizer
    meets the pointer's parameters as ones it has never stepped: the keys
    move by the learning rate at most (Adam's first step), the queries not
    at all (their gradient is zero while the keys are), and every other
    parameter is at its second step, the encoder's under their new
    indices."""
    from helpers.pg_games import selfplay_games
    import tools.pg_game_pass as gp
    from tools.add_weapon_pointer import PARAMETERS, add_pointer
    from tools.eval_provenance import file_sha256
    from tools.pg_adam_step import adam_step
    lr = 1e-5
    theta0 = _tiny_checkpoint(tmp_path)

    def round_of(theta, name, state=None):
        games = tmp_path / f"games_{name}"
        selfplay_games(theta, games, [(1, 13), (2, 7)])
        grad = tmp_path / f"grad_{name}.pt"
        assert gp.main(["--checkpoint", str(theta), "--games", str(games), "--out", str(grad), "--device", "cpu",
                        "--workers", "0"]) == 0
        out, state_out = tmp_path / f"pi_{name}.pt", tmp_path / f"adam_{name}.pt"
        return out, state_out, adam_step(theta, grad, state, lr, out, state_out)

    theta1, adam1, _step1 = round_of(theta0, "a")
    pointed, pointed_state = tmp_path / "pi_pointer.pt", tmp_path / "adam_pointer.pt"
    rec = add_pointer(theta1, pointed, adam1, pointed_state)
    state = torch.load(pointed_state, map_location="cpu", weights_only=True)
    source_state = torch.load(adam1, map_location="cpu", weights_only=True)
    assert state["checkpoint_sha256"] == file_sha256(pointed) == rec["checkpoint_sha256"]
    names, old = state["names"], list(source_state["names"])
    at = names.index(PARAMETERS[0])
    assert names[at:at + 3] == list(PARAMETERS) and names[:at] + names[at + 3:] == old
    assert all(n.startswith("model.") for n in names[:at]) and all(n.startswith("encoder.") for n in names[at + 3:])
    moments = state["optimizer"]["state"]
    assert not any(i in moments for i in range(at, at + 3)), "no moments yet: Adam builds them at its first step"
    for i, name in enumerate(old):                       # every other parameter keeps its own moments
        j = names.index(name)
        assert torch.equal(moments[j]["exp_avg"], source_state["optimizer"]["state"][i]["exp_avg"]), name

    theta2, adam2, step2 = round_of(pointed, "b", pointed_state)
    assert step2["step"] == 2
    before = torch.load(pointed, map_location="cpu", weights_only=True)["model_state"]
    stepped = torch.load(theta2, map_location="cpu", weights_only=True)
    keys = stepped["model_state"]["weapon_k_proj.weight"].abs()
    assert keys.max() > 0, "the keys train"
    assert keys.max() <= lr * (1 + 1e-6), float(keys.max())
    assert torch.equal(stepped["model_state"]["weapon_q_proj.weight"], before["weapon_q_proj.weight"])
    assert stepped["weapon_target"] == rec["rank"]
    after = torch.load(adam2, map_location="cpu", weights_only=True)["optimizer"]["state"]
    assert int(after[at + 2]["step"]) == 1 and int(after[0]["step"]) == 2 and int(after[len(names) - 1]["step"]) == 2


def test_a_reader_of_the_per_actor_scores_fails_and_structures_do_not_mix(tmp_path):
    """The network's `weapon_logits` is None, so code that scores a weapon
    without its target cannot run on it; a policy built without the pointer
    refuses the checkpoint, and one built with it refuses the source."""
    from helpers.pg_games import SLOTS, command_positions, initial_forward, selfplay_games
    from tools.add_weapon_pointer import add_pointer
    from tools.eval_players import peek_checkpoint_arch
    from tools.pg_pass import load_network
    from wesnoth_ai.action_sampler import _masked_weapon_logits
    from wesnoth_ai.pg_loss import factor_laws
    from wesnoth_ai.transformer_policy import TransformerPolicy
    source = _tiny_checkpoint(tmp_path)
    pointed = tmp_path / "pi_pointer.pt"
    add_pointer(source, pointed)
    encoder, model = load_network(pointed, CPU)
    records = selfplay_games(source, tmp_path / "games", [(1, 13)])
    vocab = (encoder.unit_type_to_id, encoder.faction_to_id)
    raws = [raw for raw, _masks in command_positions(next(iter(records.values())), vocab, limit=4)]
    with torch.no_grad():
        padded = initial_forward(model, encoder, raws, SLOTS)
    assert padded.weapon_logits is None
    with pytest.raises((AttributeError, TypeError)):
        factor_laws(padded, {}, torch.arange(len(raws)))
    with pytest.raises(ValueError, match="target"):
        _masked_weapon_logits(padded.sample(0), 0, 2)
    with pytest.raises(RuntimeError, match="weapon_target"):
        TransformerPolicy(**{**peek_checkpoint_arch(pointed), "weapon_target": 0}).load_checkpoint(pointed)
    with pytest.raises(RuntimeError, match="weapon_target"):
        TransformerPolicy(**{**peek_checkpoint_arch(source), "weapon_target": 64}).load_checkpoint(source)
