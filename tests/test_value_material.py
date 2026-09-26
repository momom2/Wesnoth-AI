"""Material as an input of the value head (docs/value_head_study_20260907.md).

The seed's recipe continued with the value head also reading
material (cost x HP fraction, mover minus visible enemies) is one arm
of the study; the other is the recipe unchanged. The arm must start
exactly at the seed (zero-initialized projection), the flag must ride
the checkpoint, and models without the flag must be byte-identical
to before."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from wesnoth_ai.material import MATERIAL_SCALE, material_of_units, material_score  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402


def _policy(value_material: bool, seed: int = 0) -> TransformerPolicy:
    torch.manual_seed(seed)
    return TransformerPolicy(d_model=32, num_layers=1, num_heads=2, d_ff=64,
                             device=torch.device("cpu"), value_material=value_material)


@pytest.fixture(scope="module")
def states():
    return [fresh_scenario_sim(seed=s, mini=True).gs for s in (1, 2, 3)]


def test_material_rides_the_encoding_and_matches_the_study_metric(states):
    pol = _policy(False)
    enc = pol._inference_encoder
    for gs in states:
        side = gs.global_info.current_side
        encoded = enc.encode(gs)
        assert encoded.material.shape == (1, 1)
        assert abs(float(encoded.material) - material_score(gs, side)) < 1e-6
    # The sign convention: the mover's units count positive.
    gs = states[0]
    side = gs.global_info.current_side
    ours = [u for u in gs.map.units if u.side == side]
    assert material_of_units(ours, side) > 0 > material_of_units(ours, 3 - side)


def test_flag_off_is_unchanged_and_flag_on_starts_at_the_same_outputs(states):
    off, on = _policy(False), _policy(True)
    assert "material_proj.weight" not in off._model.state_dict()
    assert torch.count_nonzero(on._model.material_proj.weight) == 0
    # Same trunk weights on both, then the flagged model must read the
    # same value: the projection is zero.
    on._model.load_state_dict(off._model.state_dict(), strict=False)
    on._inference_model.load_state_dict(off._inference_model.state_dict(), strict=False)
    enc = off._inference_encoder
    with torch.no_grad():
        for gs in states:
            e = enc.encode(gs)
            a, b = off._inference_model(e), on._inference_model(e)
            assert torch.allclose(a.value_logits, b.value_logits, atol=1e-6)
            assert torch.allclose(a.actor_logits, b.actor_logits, atol=1e-6)
        # And through the batched path the eval script and the trainer use.
        encs = [enc.encode(gs) for gs in states]
        outs_off = off._inference_model.forward_batch(encs)
        outs_on = on._inference_model.forward_batch(encs)
        for x, y in zip(outs_off, outs_on):
            assert torch.allclose(x.value, y.value, atol=1e-6)


def test_the_projection_learns_and_changes_the_value(states):
    """A gradient step on a value loss moves the projection off zero,
    and the value then depends on the material scalar."""
    on = _policy(True)
    enc = on._inference_encoder
    with torch.no_grad():                 # the graph starts at the model
        e = enc.encode(states[0])
    model = on._model
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    target = torch.zeros(1, dtype=torch.long)          # push mass to the lowest atom
    for _ in range(3):
        out = model(e)
        loss = torch.nn.functional.cross_entropy(out.value_logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.count_nonzero(model.material_proj.weight) > 0
    model.eval()
    with torch.no_grad():
        import dataclasses
        v0 = model(e).value
        e2 = dataclasses.replace(e, material=e.material + 5.0 * MATERIAL_SCALE)
        v1 = model(e2).value
    assert not torch.allclose(v0, v1)


def test_flag_rides_the_checkpoint_and_a_flagless_seed_warm_starts(tmp_path, states):
    from tools.eval_players import _load_policy, peek_checkpoint_arch
    on = _policy(True)
    on.save_checkpoint(tmp_path / "on.pt")
    assert peek_checkpoint_arch(tmp_path / "on.pt")["value_material"] is True
    loaded = _load_policy(tmp_path / "on.pt", torch.device("cpu"), label="t")
    assert loaded._inference_model.material_proj is not None
    off = _policy(False)
    off.save_checkpoint(tmp_path / "off.pt")
    assert not peek_checkpoint_arch(tmp_path / "off.pt").get("value_material", False)
    grafted = TransformerPolicy(d_model=32, num_layers=1, num_heads=2, d_ff=64,
                                device=torch.device("cpu"), value_material=True)
    grafted.load_checkpoint(tmp_path / "off.pt")     # material_proj is in the whitelist
    enc = off._inference_encoder
    with torch.no_grad():
        e = enc.encode(states[1])
        assert torch.allclose(off._inference_model(e).value_logits,
                              grafted._inference_model(e).value_logits, atol=1e-6)


def test_a_flagged_model_refuses_a_state_without_material(states):
    import dataclasses
    on = _policy(True)
    e = dataclasses.replace(on._inference_encoder.encode(states[0]), material=None)
    with pytest.raises(ValueError, match="material"):
        on._inference_model(e)
