"""Global feature 5 (the enemy's village count) under fog.

Wesnoth never shows a player an enemy side's village count under fog
or shroud (docs/wesnoth_rules.md, "Enemy side statistics under fog").
The encoder used the true count on every path; behind the checkpoint
flag `fog_hides_enemy_villages` it counts only the enemy villages
inside the mover's vision disc, and the flag rides the checkpoint so
the seed's encoding stays what it was trained with."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from wesnoth_ai.encoder import _static_hex_arrays, encode_raw  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402
from wesnoth_ai.visibility import enemy_villages_visible_to, visible_hexes_for  # noqa: E402

THEIR_VILLAGES_FEATURE = 5


def _state_with_enemy_villages():
    """A ladder start where the enemy owns villages on both sides of
    our vision disc (the owner map is what the sim keeps; the hexes
    need no village terrain for either count); returns how many lie
    inside and outside the disc."""
    gs = fresh_scenario_sim(seed=2, mini=False).gs
    side = gs.global_info.current_side
    enemy = 3 - side
    disc = visible_hexes_for(gs, side)
    static = _static_hex_arrays(gs)
    keys_in = [k for k in static.keys if k in disc][:3]
    keys_out = [k for k in static.keys if k not in disc][:5]
    assert keys_in and keys_out
    owner = {k: enemy for k in keys_in + keys_out}
    gs.global_info._village_owner = owner
    gs.sides[enemy - 1].nb_villages_controlled = len(owner)
    return gs, side, len(keys_in), len(keys_out)


def _feat5(raw) -> float:
    return float(raw.global_feats[THEIR_VILLAGES_FEATURE])


def test_visible_enemy_villages_follow_the_vision_disc_and_the_fog_switch():
    gs, side, inside, outside = _state_with_enemy_villages()
    assert outside > 0, "the fixture needs an enemy village outside the disc"
    assert enemy_villages_visible_to(gs, side) == inside
    gs.global_info._fog = False
    assert enemy_villages_visible_to(gs, side) == inside + outside


def test_encoder_gates_feature_5_only_behind_the_flag():
    gs, side, inside, outside = _state_with_enemy_villages()
    pol = TransformerPolicy(d_model=32, num_layers=1, num_heads=2, d_ff=64,
                            device=torch.device("cpu"))
    enc = pol._inference_encoder
    kw = dict(type_to_id=enc.unit_type_to_id, faction_to_id=enc.faction_to_id)
    legacy = encode_raw(gs, **kw)
    gated = encode_raw(gs, fog_hides_enemy_villages=True, **kw)
    assert _feat5(legacy) > 0
    assert abs(_feat5(gated) - _feat5(legacy) * inside / (inside + outside)) < 1e-6
    assert _feat5(gated) < _feat5(legacy)
    for i in range(len(legacy.global_feats)):        # nothing else moves
        if i != THEIR_VILLAGES_FEATURE:
            assert float(legacy.global_feats[i]) == float(gated.global_feats[i])
    gs.global_info._fog = False                     # fog off: the gate is inert
    assert _feat5(encode_raw(gs, fog_hides_enemy_villages=True, **kw)) == _feat5(legacy)


def test_flag_rides_the_checkpoint_the_encoder_and_the_remote_encoder(tmp_path):
    from tools.eval_sim import _load_policy, peek_checkpoint_arch
    from tools.inference_seam import RemoteEncoder
    on = TransformerPolicy(d_model=32, num_layers=1, num_heads=2, d_ff=64,
                           device=torch.device("cpu"), fog_hides_enemy_villages=True)
    assert on._inference_encoder.fog_hides_enemy_villages is True
    on.save_checkpoint(tmp_path / "on.pt")
    assert peek_checkpoint_arch(tmp_path / "on.pt")["fog_hides_enemy_villages"] is True
    loaded = _load_policy(tmp_path / "on.pt", torch.device("cpu"), label="t")
    assert loaded._inference_encoder.fog_hides_enemy_villages is True
    off = TransformerPolicy(d_model=32, num_layers=1, num_heads=2, d_ff=64,
                            device=torch.device("cpu"))
    off.save_checkpoint(tmp_path / "off.pt")
    assert not peek_checkpoint_arch(tmp_path / "off.pt").get("fog_hides_enemy_villages", False)
    gs, side, inside, outside = _state_with_enemy_villages()
    enc = on._inference_encoder
    kw = dict(type_to_id=enc.unit_type_to_id, faction_to_id=enc.faction_to_id)
    expected = _feat5(encode_raw(gs, fog_hides_enemy_villages=True, **kw))
    remote = RemoteEncoder(enc.unit_type_to_id, enc.faction_to_id, fog_hides_enemy_villages=True)
    assert _feat5(remote.encode(gs)._raw) == expected
    assert _feat5(RemoteEncoder(enc.unit_type_to_id, enc.faction_to_id).encode(gs)._raw) > expected
