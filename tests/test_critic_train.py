"""The step-1 critic (wesnoth_ai/critic.py) and its trainer
(tools/critic_train.py): what a full critic takes from the reference, what
a value-only step moves and leaves alone, and what a run records."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import critic_train  # noqa: E402
from wesnoth_ai import critic as cr  # noqa: E402
from wesnoth_ai import critic_data as cd  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
TINY = {"d_model": 16, "num_layers": 1, "num_heads": 2, "d_ff": 32}
POLICY_PREFIXES = ("actor_head", "type_head", "target_q_proj", "target_k_proj", "weapon_head", "belief_head")


def _vocab():
    from helpers.parity_games import FACTION_IDS, vocab_of
    return vocab_of(["Lieutenant", "Spearman"]), dict(FACTION_IDS)


def test_a_full_critic_takes_the_reference_less_its_memory_and_says_so(tmp_path):
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.model import WesnothModel
    types, factions = _vocab()
    torch.manual_seed(0)
    encoder = GameStateEncoder(d_model=16, relevant_set_hexes=True, fog_hides_enemy_villages=True,
                               terrain_multi_hot=True, observation_parity=True, relevant_set_version=2,
                               unit_type_to_id=types, faction_to_id=factions)
    reference = WesnothModel(observation_parity=True, memory_slots=4, **TINY)
    ckpt = {"arch": dict(TINY), "observation_parity": True, "memory_slots": 4, "relevant_set_version": 2,
            "model_state": reference.state_dict(), "encoder_state": encoder.state_dict(),
            "unit_type_to_id": types, "faction_to_id": factions}
    path = tmp_path / "ref.pt"
    torch.save(ckpt, path)
    c_encoder, c_model = cr.build_critic(TINY, types, factions, torch.device("cpu"))
    loaded = cr.load_reference(c_encoder, c_model, path)
    assert loaded["missing"] == sorted(cr.AUX_KEYS)
    assert loaded["unexpected"] and all(k.startswith("slot_memory.") for k in loaded["unexpected"])
    assert torch.equal(c_model.value_head[0].weight, reference.value_head[0].weight)
    assert torch.equal(c_model.encoder.layers[0].linear1.weight, reference.encoder.layers[0].linear1.weight)
    broken = dict(ckpt, model_state={k: v for k, v in ckpt["model_state"].items() if not k.startswith("value_head")})
    torch.save(broken, path)
    with pytest.raises(ValueError, match="unexpected partial load"):
        cr.load_reference(*cr.build_critic(TINY, types, factions, torch.device("cpu")), path)


def _items(n: int):
    """`n` encodings of a fogged two-sided board whose side-1 Spearman
    has from 4 to 38 hit points, labelled won when it has 20 or more."""
    from helpers.parity_games import core_of, record
    types, factions = _vocab()
    items = []
    for k in range(n):
        hp = 4 + (34 * k) // max(1, n - 1)
        cs = core_of(record([("Lieutenant", 1, 1, 3, True), ("Spearman", 1, 2, 3, False, {"hp": hp}),
                             ("Lieutenant", 2, 18, 3, True), ("Spearman", 2, 3, 3, False)], fog=True))
        cs.apply_command(["init_side", 1])
        z = 1.0 if hp >= 20 else -1.0
        items.append((cd.pack_raw(cd.encode_view(cs, "true", types, factions)), z, float(hp - 36)))
    return items


def _args(tmp_path, **over):
    args = dict(critic_train.DEFAULTS, lr=3e-3, warmup_steps=1, weight_decay=0.0, batch=4, max_epochs=3,
                evals_per_epoch=1, patience=10, max_minutes=5.0, signal_every=4, probe_positions=2, seed=1)
    args.update(view="true", source="M", fraction=1.0, arch="small", init=None, out=tmp_path / "critic.pt",
                fp32=True)
    args.update(over)
    return argparse.Namespace(**args)


def _trainer(tmp_path, monkeypatch, train, holdout, **over):
    monkeypatch.setattr(cr, "SMALL_ARCH", dict(TINY))
    types, factions = _vocab()
    torch.manual_seed(3)
    return critic_train.CriticTrainer(_args(tmp_path, **over), torch.device("cpu"), types, factions,
                                      train, holdout, {"test": True})


@needs_core
def test_a_value_only_step_fits_the_value_and_leaves_the_policy_heads_alone(tmp_path, monkeypatch):
    items = _items(8)
    trainer = _trainer(tmp_path, monkeypatch, items, items)
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
    first = trainer.holdout_loss()["value"]
    batch = critic_train.unpack_batch(items)
    for _ in range(25):
        trainer.step(batch)
    assert trainer.holdout_loss()["value"] < first - 0.2
    moved = {n for n, p in trainer.model.named_parameters() if not torch.equal(p, before[n])}
    assert not [n for n in moved if n.startswith(POLICY_PREFIXES)], "a policy head moved"
    assert any(n.startswith("encoder.layers") for n in moved) and any(n.startswith("value_head") for n in moved)
    assert any(n.startswith("aux_score_head") for n in moved)
    names = {n for n, _ in cr.trained_parameters(trainer.model, trainer.encoder)}
    assert not [n for n in names if n.startswith(tuple("model." + p for p in POLICY_PREFIXES))]


@needs_core
def test_the_aux_head_reaches_the_trunk(tmp_path, monkeypatch):
    items = _items(4)
    trainer = _trainer(tmp_path, monkeypatch, items, items)
    raws, z, aux = critic_train.unpack_batch(items)
    parts = trainer.losses(raws, z, aux)
    parts["aux"].backward()
    grads = [p.grad for n, p in trainer.model.named_parameters() if n.startswith("encoder.layers")]
    assert any(g is not None and float(g.abs().sum()) > 0 for g in grads)
    assert trainer.model.value_head[0].weight.grad is None, "the aux loss alone does not train the value head"


@needs_core
def test_a_run_keeps_the_best_holdout_checkpoint_and_records_as_it_goes(tmp_path, monkeypatch):
    items = _items(8)
    trainer = _trainer(tmp_path, monkeypatch, items, items[:4])
    ended = trainer.run()
    assert ended == "max_epochs"
    stem = tmp_path / "critic"
    reads = [json.loads(line) for line in Path(f"{stem}.holdout.jsonl").read_text().splitlines()]
    assert len(reads) == 3 and reads[0]["best"]
    best = min(r["holdout_value"] for r in reads)
    steps = [json.loads(line) for line in Path(f"{stem}.steps.jsonl").read_text().splitlines()]
    assert len(steps) == 6 and {"encoder", "trunk", "value_head", "aux_ml"} <= set(steps[0]["grad"])
    signal = [json.loads(line) for line in Path(f"{stem}.signal.jsonl").read_text().splitlines()]
    probes = [r for r in signal if r["kind"] == "probe"]
    assert signal[0]["kind"] == "start" and probes and "probe_error" not in probes[0]
    assert set(probes[0]["gradient"]["trunk"]) >= {"value", "aux", "total_norm"}
    summary = json.loads(Path(f"{stem}.summary.json").read_text())
    assert summary["DONE"] and summary["ended_by"] == "max_epochs"
    encoder, model, meta = cr.load_critic(tmp_path / "critic.pt", torch.device("cpu"))
    assert meta["training"]["best"] == pytest.approx(best)
    values = cr.critic_values(encoder, model, [cd.unpack_raw(b) for b, _, _ in items[:2]], torch.device("cpu"))
    assert len(values) == 2 and all(-1.0 <= v <= 1.0 for v in values)


def test_the_size_curve_nests_its_games_and_shares_one_holdout():
    rows = [{"key": f"g{k}", "source": "M", "status": "ok", "positions": 3, "split": split, "size_u": u}
            for k, (split, u) in enumerate([("train", 0.1), ("train", 0.3), ("train", 0.6), ("holdout", 0.2),
                                            ("holdout", 0.9)])]
    rows.append({"key": "capped", "source": "M", "status": "capped", "positions": 0, "split": "train", "size_u": 0.0})
    rows.append({"key": "h", "source": "H", "status": "ok", "positions": 3, "split": "train", "size_u": 0.0})
    t25, h25 = critic_train.select_games(rows, "M", 0.25)
    t100, h100 = critic_train.select_games(rows, "M", 1.0)
    assert [r["key"] for r in t25] == ["g0"] and [r["key"] for r in t100] == ["g0", "g1", "g2"]
    assert h25 == h100 and [r["key"] for r in h100] == ["g3", "g4"]
