"""The raw player behind a shared inference server picks on the compact
legal-action arrays and materializes one action; the choice and the
rng draws equal the list path's (2026-09-11 worker profile: unpacking
every legal action was a quarter of the worker's Python per decision)."""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tests.test_server_priors import _policy, _states  # noqa: E402


def _seam(policy):
    from tools.inference_seam import InferenceServer, RemoteEncoder, RemoteModel
    enc, model = policy._inference_encoder, policy._inference_model
    renc = RemoteEncoder(enc.unit_type_to_id, enc.faction_to_id, server_priors=True,
                         fog_hides_enemy_villages=enc.fog_hides_enemy_villages)
    rmodel = RemoteModel(InferenceServer(model, enc))
    return SimpleNamespace(_inference_model=rmodel, _inference_encoder=renc,
                           _lock=threading.Lock(), _decision_step=0)


def test_compact_action_equals_the_unpacked_element():
    from wesnoth_ai.server_priors import compact_action, unpack_compact
    policy = _policy()
    base = _seam(policy)
    with torch.no_grad():
        for gs in _states():
            encoded = base._inference_encoder.encode(gs)
            out = base._inference_model(encoded)
            compact = out.legal_compact
            full = unpack_compact(compact, encoded)
            assert len(full) == len(compact.prior) > 1
            for i in range(len(full)):
                assert compact_action(compact, i, encoded) == full[i].action


def test_raw_player_compact_choice_matches_the_list_path():
    from tools.raw_player import RawPolicyPlayer
    policy = _policy()
    base = _seam(policy)
    states = _states()
    for temperature in (0.0, 0.7):
        for forbid in (False, True):
            fast = RawPolicyPlayer(base, temperature, seed=11, forbid_end_turn=forbid)
            slow = RawPolicyPlayer(base, temperature, seed=11, forbid_end_turn=forbid,
                                   compact_selection=False)
            for gs in states:
                assert fast.select_action(gs) == slow.select_action(gs)


def test_pick_index_argmax_takes_the_first_maximum():
    from tools.raw_player import pick_index
    priors = np.array([0.2, 0.5, 0.5, 0.1])
    assert pick_index(priors, 0.0, np.random.default_rng(0)) == 1
