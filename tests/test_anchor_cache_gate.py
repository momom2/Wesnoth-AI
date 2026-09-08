"""Pre-encoded caches carry the fog gate of global feature 5 and a
consumer under the other gate refuses them (a cache without the mark
was encoded with the true enemy village count)."""
import json
import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.build_human_anchor import anchor_gate, anchor_meta_path, check_anchor_gate  # noqa: E402
from tools.policy_anchor import CACHE_VERSION, load_policy_anchor  # noqa: E402


def _policy_cache(path: Path, meta: dict) -> None:
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
