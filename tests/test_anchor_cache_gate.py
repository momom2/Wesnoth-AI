"""Pre-encoded caches carry the fog gate of global feature 5 and the
sim's observation epoch; a consumer under a different one refuses them.

A cache without the fog mark was encoded with the true enemy village
count; a cache without the epoch mark is epoch 1. The epoch exists
because the gate and the vocab do not move when the SIM's own rules
about what a player sees do -- the 2026-09-13 hide-cover change made
every earlier pre-encoded observation stale while leaving the vocab,
the hex basis and the fog gate identical."""
import json
import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.build_human_anchor import anchor_gate, anchor_meta_path, check_anchor_gate  # noqa: E402
from tools.policy_anchor import CACHE_VERSION, load_policy_anchor  # noqa: E402
from tools.preencode_corpus import vocab_fingerprint  # noqa: E402
from wesnoth_ai.constants import OBSERVATION_EPOCH  # noqa: E402


def _policy_cache(path: Path, meta: dict) -> None:
    meta = {"observation_epoch": OBSERVATION_EPOCH, **meta}
    with path.open("wb") as f:
        pickle.dump({"version": CACHE_VERSION, "meta": meta, "games": [[("raw", "ai")]]}, f)


def test_policy_anchor_cache_gate_is_checked_against_the_consumer(tmp_path):
    legacy = tmp_path / "legacy.pkl"
    _policy_cache(legacy, {"games": 1})                    # pre-gate cache: no mark
    gated = tmp_path / "gated.pkl"
    _policy_cache(gated, {"games": 1, "fog_hides_enemy_villages": True})
    assert load_policy_anchor(legacy) == [[("raw", "ai")]]                 # no consumer gate given
    assert load_policy_anchor(legacy, fog_hides_enemy_villages=False)
    assert load_policy_anchor(gated, fog_hides_enemy_villages=True)
    with pytest.raises(ValueError, match="--fog-hides-enemy-villages"):
        load_policy_anchor(legacy, fog_hides_enemy_villages=True)
    with pytest.raises(ValueError, match="fog_hides_enemy_villages=True"):
        load_policy_anchor(gated, fog_hides_enemy_villages=False)


def test_human_anchor_sidecar_records_the_gate(tmp_path):
    anchor = tmp_path / "human_anchor.pkl"
    anchor.write_bytes(pickle.dumps([]))
    assert anchor_gate(anchor) is False                    # no sidecar: the old encoding
    check_anchor_gate(anchor, False)
    with pytest.raises(ValueError, match="Rebuild"):
        check_anchor_gate(anchor, True)
    anchor_meta_path(anchor).write_text(json.dumps({"fog_hides_enemy_villages": True}))
    assert anchor_gate(anchor) is True
    check_anchor_gate(anchor, True)
    with pytest.raises(ValueError):
        check_anchor_gate(anchor, False)



def test_a_cache_from_an_earlier_observation_epoch_is_refused(tmp_path):
    """The rules that decide what a player SEES are not part of the
    vocab, the hex basis or the fog gate, so only the epoch can catch a
    cache built before they changed."""
    stale = tmp_path / "stale.pkl"
    with stale.open("wb") as f:
        pickle.dump({"version": CACHE_VERSION,
                     "meta": {"games": 1, "observation_epoch": OBSERVATION_EPOCH - 1},
                     "games": [[("raw", "ai")]]}, f)
    with pytest.raises(ValueError, match=f"observation epoch {OBSERVATION_EPOCH - 1}"):
        load_policy_anchor(stale)

    unmarked = tmp_path / "unmarked.pkl"      # written before the mark existed: epoch 1
    with unmarked.open("wb") as f:
        pickle.dump({"version": CACHE_VERSION, "meta": {"games": 1},
                     "games": [[("raw", "ai")]]}, f)
    if OBSERVATION_EPOCH != 1:
        with pytest.raises(ValueError, match="observation epoch 1"):
            load_policy_anchor(unmarked)

    current = tmp_path / "current.pkl"
    _policy_cache(current, {"games": 1})
    assert load_policy_anchor(current) == [[("raw", "ai")]]


def test_the_preencoded_fingerprint_moves_with_the_observation_epoch(monkeypatch):
    """Same vocab, same basis, same gate -- a different epoch must still
    produce a different fingerprint, or a stale corpus loads silently."""
    import tools.preencode_corpus as pc
    types, factions = {"Wose": 0}, {"Rebels": 0}
    here = vocab_fingerprint(types, factions, True, False)
    monkeypatch.setattr(pc, "OBSERVATION_EPOCH", OBSERVATION_EPOCH + 1)
    assert vocab_fingerprint(types, factions, True, False) != here
