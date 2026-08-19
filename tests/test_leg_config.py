"""Typed leg-config contract (2026-08-19 architectural ruling).
This schema is what makes launch-decision omission UNREPRESENTABLE
-- the class that shipped leg 3 at cap [60,200] and leg 4 (attempt
2) without its policy anchor. Guards: structural requiredness,
unknown-key fatality, decline semantics, shell-export safety.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.leg_config import export_lines, load, required_fields  # noqa: E402

GOOD = {"campaign_file": "tier_b_l4.pt",
        "seed_hf_file": "tier-b/a3/seed_imit_tierb_start.pt",
        "probe_t0": "3.2070",
        "policy_anchor": "replays_dataset_imitation/policy_anchor.npz"}


def _write(tmp_path, obj):
    p = tmp_path / "leg.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def test_valid_config_exports_env_mapping(tmp_path):
    cfg = load(_write(tmp_path, GOOD))
    lines = export_lines(cfg)
    assert "export CAMPAIGN_FILE='tier_b_l4.pt'" in lines
    assert "export PROBE_T0='3.2070'" in lines


def test_every_required_decision_missing_is_listed_at_once(tmp_path):
    p = _write(tmp_path, {"campaign_file": "x.pt"})
    with pytest.raises(ValueError) as e:
        load(p)
    msg = str(e.value)
    for name in required_fields():
        if name != "campaign_file":
            assert name in msg, f"missing '{name}' not reported"


def test_unknown_key_is_fatal(tmp_path):
    bad = dict(GOOD, polcy_anchor="typo.npz")  # note the typo
    with pytest.raises(ValueError, match="unknown keys"):
        load(_write(tmp_path, bad))


def test_none_declines_to_empty_export(tmp_path):
    cfg = load(_write(tmp_path, dict(GOOD, policy_anchor="none",
                                     probe_t0="none")))
    lines = export_lines(cfg)
    assert "export HUMAN_ANCHOR_POLICY_FILE=''" in lines
    assert "export PROBE_T0=''" in lines


def test_export_shell_quoting_survives_quotes(tmp_path):
    cfg = load(_write(tmp_path, dict(
        GOOD, extra_env={"NOTE": "it's quoted"})))
    (line,) = [ln for ln in export_lines(cfg) if "NOTE" in ln]
    assert "'it'\\''s quoted'" in line
