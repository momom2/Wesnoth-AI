"""The hex's terrain as its full set from the engine's aliases.

Wesnoth resolves a hex's movement and defense through its terrain
aliases (a forested hill is `Ht` AND `Ft`), while the encoder carried
ONE class per hex picked by enum ordinal, which labelled 86% of the
Ladder pool's forest-overlay hexes as something other than forest
(tools/analysis/hide_cover_census.py, 2026-09-13). `Hex.terrain_mask`
carries the set; behind the checkpoint flag `terrain_multi_hot` the
encoder embeds it as a multi-hot over the same table, and the flag
rides the checkpoint so the reference player's observations stay what
it was trained on."""
import dataclasses
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from tools.eval_sim import peek_checkpoint_arch  # noqa: E402
from tools.inference_seam import RemoteEncoder, inference_blueprint  # noqa: E402
from tools.scenario_pool import LADDER_SCENARIO_IDS, build_scenario_gamestate, random_setup  # noqa: E402
from tools.terrain_resolver import strip_start_position, terrain_mask, terrain_members  # noqa: E402
from wesnoth_ai.classes import Terrain  # noqa: E402
from wesnoth_ai.encoder import NUM_TERRAINS, GameStateEncoder, _first_terrain_id, encode_raw  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402

FOREST_BIT = 1 << Terrain.FOREST.value


def _tiny_policy():
    return TransformerPolicy(d_model=32, num_layers=1, num_heads=2, d_ff=64,
                             device=torch.device("cpu"))


def test_terrain_members_follow_the_engine_aliases():
    assert terrain_members("Gs^Fp") == {Terrain.FLAT, Terrain.FOREST}
    assert terrain_members("Hh^Fp") == {Terrain.HILLS, Terrain.FOREST}
    assert terrain_members("Wwf") == {Terrain.FLAT, Terrain.SHALLOWWATER}      # a ford is both
    assert terrain_members("Ss^Vhs") == {Terrain.SWAMP, Terrain.VILLAGE}
    assert terrain_members("Xv") == {Terrain.IMPASSABLE}
    assert terrain_members("Kh") == {Terrain.CASTLE}
    assert terrain_mask("Gs^Fp") == (1 << Terrain.FLAT.value) | FOREST_BIT
    assert terrain_mask("") == 0


def test_every_forest_overlay_hex_of_a_ladder_map_carries_the_forest_bit():
    base = random_setup(random.Random(0))
    gs = build_scenario_gamestate(dataclasses.replace(base, scenario_id=LADDER_SCENARIO_IDS[0]))
    codes = gs.global_info._terrain_codes
    forest = [h for h in gs.map.hexes
              if strip_start_position(codes.get((h.position.x, h.position.y), "")).partition("^")[2].startswith("F")]
    assert forest, "the fixture map needs forest overlays"
    assert all(h.terrain_mask & FOREST_BIT for h in forest)
    legacy_forest = sum(_first_terrain_id(h.terrain_types) == Terrain.FOREST.value for h in forest)
    assert legacy_forest < len(forest), "the one-class view is what the mask corrects"


def test_encode_raw_carries_masks_only_behind_the_flag():
    gs = fresh_scenario_sim(seed=3, mini=False).gs
    enc = _tiny_policy()._inference_encoder
    kw = dict(type_to_id=enc.unit_type_to_id, faction_to_id=enc.faction_to_id)
    legacy = encode_raw(gs, **kw)
    masks = encode_raw(gs, terrain_multi_hot=True, **kw)
    assert legacy.hex_terrain_ids.max() < NUM_TERRAINS
    assert masks.hex_terrain_ids.min() > 0 and masks.hex_terrain_ids.max() < (1 << NUM_TERRAINS)
    assert np.array_equal(legacy.hex_xs, masks.hex_xs)
    assert np.array_equal(legacy.hex_modifier_flags, masks.hex_modifier_flags)
    assert not np.array_equal(legacy.hex_terrain_ids, masks.hex_terrain_ids)
    assert legacy.hex_terrain_ids.dtype == masks.hex_terrain_ids.dtype == np.int64


def test_terrain_tokens_sum_the_table_rows_of_the_set_bits():
    torch.manual_seed(0)
    enc = GameStateEncoder(d_model=16, terrain_multi_hot=True)
    w = enc.terrain_embed.weight.detach()
    ids = torch.tensor([Terrain.FLAT.value, Terrain.FOREST.value, Terrain.HILLS.value])
    single = enc.terrain_tokens(1 << ids)
    assert torch.allclose(single, w[ids])
    both = enc.terrain_tokens(torch.tensor([(1 << ids[0]) | (1 << ids[1])]))
    assert torch.allclose(both[0], w[ids[0]] + w[ids[1]])
    enc.terrain_multi_hot = False
    assert torch.allclose(enc.terrain_tokens(ids), w[ids])


def test_the_flag_rides_the_checkpoint_and_is_off_for_older_ones(tmp_path):
    pol = _tiny_policy()
    assert pol._terrain_multi_hot and pol._encoder.terrain_multi_hot \
        and pol._inference_encoder.terrain_multi_hot
    path = tmp_path / "fresh.pt"
    pol.save_checkpoint(path)
    assert peek_checkpoint_arch(path).get("terrain_multi_hot") is True
    again = _tiny_policy()
    again.load_checkpoint(path)
    assert again._inference_encoder.terrain_multi_hot
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    del ckpt["terrain_multi_hot"]
    older = tmp_path / "older.pt"
    torch.save(ckpt, older)
    assert "terrain_multi_hot" not in peek_checkpoint_arch(older)
    legacy = _tiny_policy()
    legacy.load_checkpoint(older)
    assert not legacy._terrain_multi_hot and not legacy._encoder.terrain_multi_hot \
        and not legacy._inference_encoder.terrain_multi_hot


def test_the_shared_inference_path_carries_the_flag():
    pol = _tiny_policy()
    bp = inference_blueprint(pol._inference_model, pol._inference_encoder)
    assert bp.encoder_kwargs["terrain_multi_hot"] is True
    gs = fresh_scenario_sim(seed=3, mini=False).gs
    enc = pol._inference_encoder
    local = encode_raw(gs, type_to_id=enc.unit_type_to_id, faction_to_id=enc.faction_to_id,
                       terrain_multi_hot=True)
    remote = RemoteEncoder(enc.unit_type_to_id, enc.faction_to_id, terrain_multi_hot=True).encode(gs)
    assert np.array_equal(remote._raw.hex_terrain_ids, local.hex_terrain_ids)
    plain = RemoteEncoder(enc.unit_type_to_id, enc.faction_to_id).encode(gs)
    assert plain._raw.hex_terrain_ids.max() < NUM_TERRAINS


def test_raw_of_carries_every_switch_of_its_encoder():
    """The one way to build a raw an encoder will read as its own
    `encode` does. The trainer, the anchor tools and the benches go
    through it: a bare encode_raw with the basis alone encoded another
    observation (a fresh network trained on one-class tokens and
    played on the set), which tests/test_value_loss_form.py caught as
    a nonzero no-miss loss."""
    gs = fresh_scenario_sim(seed=3, mini=False).gs
    enc = _tiny_policy()._inference_encoder
    enc.register_names(gs)
    own = enc.raw_of(gs)
    explicit = encode_raw(gs, type_to_id=enc.unit_type_to_id, faction_to_id=enc.faction_to_id,
                          relevant_set=enc.relevant_set_hexes,
                          fog_hides_enemy_villages=enc.fog_hides_enemy_villages,
                          terrain_multi_hot=True)
    for field in ("hex_terrain_ids", "global_feats", "hex_xs", "unit_type_ids"):
        assert np.array_equal(getattr(own, field), getattr(explicit, field)), field
    bare = encode_raw(gs, type_to_id=enc.unit_type_to_id, faction_to_id=enc.faction_to_id)
    assert not np.array_equal(own.hex_terrain_ids, bare.hex_terrain_ids)
    assert np.array_equal(enc.encode(gs).hex_tokens.detach(),
                          enc.encode_from_raw(own).hex_tokens.detach())


def test_the_pool_play_command_carries_the_flag():
    repo = Path(__file__).parent.parent
    pool_src = (repo / "tools/actor_pool.py").read_text(encoding="utf-8")
    actor_src = (repo / "tools/actor_worker.py").read_text(encoding="utf-8")
    assert "self._terrain_multi_hot())" in pool_src and "terrain_multi_hot=_tmh" in actor_src
