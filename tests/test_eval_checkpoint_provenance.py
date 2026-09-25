"""Which checkpoint each side played: every result file records the
SHA-256 of each side's checkpoint file (checkpoint_sha256_a/_b; None for
'dummy' and 'random', which load none), and a resume whose checkpoint
differs is refused -- by the batch pre-scan, by the per-file guard, and
by elo_collect for a dir that holds two checkpoints under one label. A
label names a checkpoint only by convention: before this, a resume that
pointed --spec-a at other weights under the same label mixed two players
in one outdir without a word."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wesnoth_ai.constants import OBSERVATION_EPOCH  # noqa: E402


def _tiny_checkpoint(path: Path, seed: int) -> str:
    import torch
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(seed)
    TransformerPolicy(device=torch.device("cpu"), d_model=32, num_layers=1, num_heads=2,
                      d_ff=64).save_checkpoint(path)
    return str(path)


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _result(**over) -> dict:
    rec = {"label_a": "A", "label_b": "B", "outcome_a": "win", "margin_a": 0.5,
           "procedure_a": "raw:t0", "procedure_b": "raw", "max_turns": 200,
           "combat_stream": "per_game", "observation_epoch": int(OBSERVATION_EPOCH),
           "side_a": 1, "seed": 10000}
    rec.update(over)
    return rec


def test_a_result_records_the_checkpoint_each_side_played(tmp_path):
    """A real game: side A a checkpoint file, side B the scripted dummy."""
    from tools.elo_eval_game import main
    spec = _tiny_checkpoint(tmp_path / "net.pt", seed=1)
    out = tmp_path / "games"
    assert main(["x", "A", spec, "B", "dummy", "1", "7", str(out), "--mcts-sims", "0",
                 "--raw-temperature-a", "0", "--max-turns", "1", "--device", "cpu"]) == 0
    rec = json.loads((out / "game_A_B_s1_7.json").read_text(encoding="utf-8"))
    assert rec["checkpoint_sha256_a"] == _sha(spec)
    assert rec["checkpoint_sha256_b"] is None


def test_the_per_file_guard_refuses_another_checkpoint(tmp_path):
    from tools.elo_eval_game import main
    spec = _tiny_checkpoint(tmp_path / "net.pt", seed=1)
    out = tmp_path / "games"
    out.mkdir()
    argv = ["x", "A", spec, "B", "dummy", "1", "7", str(out), "--mcts-sims", "0",
            "--raw-temperature-a", "0", "--device", "cpu"]
    slot = out / "game_A_B_s1_7.json"
    # A fresh network plays the terrain-set view.
    slot.write_text(json.dumps(_result(terrain_a="set", checkpoint_sha256_a="0" * 64)),
                    encoding="utf-8")
    with pytest.raises(SystemExit, match="checkpoint"):
        main(argv)
    slot.write_text(json.dumps(_result(terrain_a="set")), encoding="utf-8")   # before the field
    with pytest.raises(SystemExit, match="checkpoint"):
        main(argv)
    slot.write_text(json.dumps(_result(terrain_a="set", checkpoint_sha256_a=_sha(spec))),
                    encoding="utf-8")
    assert main(argv) == 0                                           # same weights: skipped


def _batch_argv(out: Path, spec: str):
    return ["x", "--label-a", "A", "--spec-a", spec, "--label-b", "B", "--spec-b", "dummy",
            "--outdir", str(out), "--games", "1", "--mcts-sims", "0",
            "--raw-temperature-a", "0", "--device", "cpu", "--jobs", "1",
            "--max-extra-games", "0", "--time-budget-min", "5", "--min-free-mb", "0"]


def test_a_resume_under_another_checkpoint_is_refused(tmp_path, monkeypatch):
    """The slot's result was played by other weights under label A: the
    pre-scan refuses before any game; the same weights resume (the one
    slot is done, so the match is complete)."""
    from tools import run_elo_batch as rb
    spec = tmp_path / "net.pt"
    spec.write_bytes(b"new weights")
    # The basis and terrain view the checkpoint plays in, as the driver's
    # peek would read them (the bytes above are not a real checkpoint).
    monkeypatch.setitem(rb._FLAGS_MEMO, str(spec), ("full", "set"))
    out = tmp_path / "games"
    out.mkdir()
    slot = out / "game_A_B_s1_10000.json"
    for recorded in ({"checkpoint_sha256_a": hashlib.sha256(b"old weights").hexdigest()}, {}):
        slot.write_text(json.dumps(_result(terrain_a="set", **recorded)), encoding="utf-8")
        with pytest.raises(SystemExit, match="checkpoint"):
            rb.main(_batch_argv(out, str(spec)))
    slot.write_text(json.dumps(_result(terrain_a="set", checkpoint_sha256_a=_sha(spec),
                                       checkpoint_sha256_b=None)), encoding="utf-8")
    assert rb.main(_batch_argv(out, str(spec))) == rb.EXIT_COMPLETE


def test_a_timeout_artifact_carries_the_checkpoints(tmp_path, monkeypatch):
    """The driver writes the no-result artifact of a game it killed on
    timeout; without the checkpoint fields, every resume of the outdir
    would refuse it as a file that cannot say which weights played."""
    from tools import run_elo_batch as rb
    script = tmp_path / "sleeper.py"
    script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    monkeypatch.setattr(rb, "GAME_SCRIPT", script)
    monkeypatch.setattr(rb, "POLL_S", 0.05)
    spec = tmp_path / "net.pt"
    spec.write_bytes(b"weights")
    monkeypatch.setitem(rb._FLAGS_MEMO, str(spec), ("full", "set"))
    out = tmp_path / "games"
    argv = _batch_argv(out, str(spec))
    rb.main(argv + ["--per-game-timeout-min", "0.0001"])
    (art,) = [json.loads(p.read_text(encoding="utf-8")) for p in out.glob("game_*.json")]
    assert art["outcome_a"] == "timeout_kill"
    assert art["checkpoint_sha256_a"] == _sha(spec) and art["checkpoint_sha256_b"] is None
    assert rb.main(argv) == rb.EXIT_GUARD_SPENT


def test_a_server_that_loaded_other_bytes_is_refused(tmp_path, monkeypatch):
    """Under shared inference the games record the checkpoint the server
    loaded. When it differs from the file the driver hashed (the file
    changed on disk between the two reads), the driver refuses before
    any game."""
    from types import SimpleNamespace
    from tools import eval_inference_server
    from tools import run_elo_batch as rb
    spec = tmp_path / "net.pt"
    spec.write_bytes(b"weights")
    served = SimpleNamespace(
        address="unused", alive=lambda: True, shutdown=lambda timeout=30.0: None,
        info={"infer_bf16": False, "packed_trunk": False, "relevant_set": False,
              "terrain_multi_hot": True, "checkpoint_sha256": "0" * 64})
    monkeypatch.setattr(eval_inference_server, "launch_inference_server",
                        lambda *a, **k: served)
    out = tmp_path / "games"
    with pytest.raises(SystemExit, match="checkpoint"):
        rb.main(_batch_argv(out, str(spec)) + ["--persistent-workers", "--shared-inference"])
    assert not list(out.glob("game_*.json"))


def test_collect_refuses_two_checkpoints_under_one_label(tmp_path):
    from tools import elo_collect
    d = tmp_path / "games"
    d.mkdir()
    for seed, sha in ((10000, "1" * 64), (10001, "2" * 64)):
        (d / f"game_A_B_s1_{seed}.json").write_text(
            json.dumps(_result(seed=seed, checkpoint_sha256_a=sha)), encoding="utf-8")
    with pytest.raises(SystemExit, match="checkpoint"):
        elo_collect.main(["x", str(d), "--no-catalog"])
    (d / "game_A_B_s1_10001.json").write_text(
        json.dumps(_result(seed=10001, checkpoint_sha256_a="1" * 64)), encoding="utf-8")
    assert elo_collect.main(["x", str(d), "--no-catalog"]) == 0
