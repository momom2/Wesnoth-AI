"""Persistent evaluation workers (tools/eval_workers.py, plan 1.5):
one elo_eval_game --worker process plays many games; the batch loop
drives them through the Popen-like handle."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

SCRIPT = Path(__file__).parent.parent / "tools" / "elo_eval_game.py"


def _argv(out: Path, seed: int):
    return [str(SCRIPT), "A", "dummy", "B", "dummy", "1", str(seed), str(out),
            "--mcts-sims", "0", "--max-turns", "2", "--device", "cpu"]


def test_one_worker_plays_two_games(tmp_path):
    from tools.eval_workers import WorkerPool
    pool = WorkerPool([sys.executable, "-u", str(SCRIPT), "--worker"], 1, tmp_path)
    try:
        h1, errf = pool.submit(_argv(tmp_path, 7), game_tag="1_7")
        assert h1.wait(timeout=180) == 0
        h2, _ = pool.submit(_argv(tmp_path, 9), game_tag="2_9")
        assert h2.wait(timeout=180) == 0
        assert h1.pid == h2.pid, "the second game must reuse the worker"
        assert len(pool.workers) == 1 and pool.workers[0].games == 2
        for seed in (7, 9):
            r = json.loads((tmp_path / f"game_A_B_s1_{seed}.json").read_text(encoding="utf-8"))
            assert r["procedure_a"] == "raw" and r["max_turns"] == 2
        # the loop's stderr helpers see a shim whose name never exists
        assert not Path(errf.name).exists()
    finally:
        pool.shutdown()


def test_killed_worker_is_replaced(tmp_path):
    from tools.eval_workers import WorkerPool
    pool = WorkerPool([sys.executable, "-u", str(SCRIPT), "--worker"], 1, tmp_path)
    try:
        h1, _ = pool.submit(_argv(tmp_path, 11), game_tag="1_11")
        h1.kill()
        assert h1.poll() is not None and h1.returncode != 0
        h2, _ = pool.submit(_argv(tmp_path, 13), game_tag="2_13")
        assert h2.wait(timeout=180) == 0
        assert h2.pid != h1.pid
    finally:
        pool.shutdown()


def test_batch_driver_persistent_workers_end_to_end(tmp_path):
    from tools.run_elo_batch import main
    out = tmp_path / "games"
    rc = main(["x", "--label-a", "A", "--spec-a", "dummy", "--label-b", "B",
               "--spec-b", "dummy", "--outdir", str(out), "--games", "2",
               "--mcts-sims", "0", "--max-turns", "2", "--device", "cpu",
               "--jobs", "1", "--persistent-workers", "--max-extra-games", "0",
               "--time-budget-min", "5", "--min-free-mb", "0"])
    assert rc == 0
    files = sorted(out.glob("game_*.json"))
    assert len(files) == 2
    assert not list(out.glob(".stderr_*"))       # per-game shims never materialize


def test_worker_cache_is_per_side(tmp_path):
    """Same spec on both sides of a match: each side keeps its own
    policy object (own decision counter, pending queue and forward
    counter -- a shared object left side A's counter at 0), while a
    later game with the same label reuses the cached object."""
    import torch
    from tools import elo_eval_game as g
    from wesnoth_ai.transformer_policy import TransformerPolicy
    spec = str(tmp_path / "tiny.pt")
    TransformerPolicy(device=torch.device("cpu"), d_model=32, num_layers=1,
                      num_heads=2, d_ff=64).save_checkpoint(spec)
    g._POLICY_CACHE.clear()
    g._WORKER_MODE = True
    try:
        cpu = torch.device("cpu")
        pa, ca = g._build_player(spec, "A", 0, cpu, raw_temperature=0.0, raw_seed=1)
        pb, cb = g._build_player(spec, "B", 0, cpu, raw_temperature=0.0, raw_seed=1)
        assert pa._base is not pb._base
        assert ca is not cb and pa._base._inference_model is ca and pb._base._inference_model is cb
        pa2, ca2 = g._build_player(spec, "A", 0, cpu, raw_temperature=0.0, raw_seed=1)
        assert pa2._base is pa._base and ca2 is not ca            # cached object, fresh counter
        assert ca2._inner is ca._inner                            # the counter never nests
    finally:
        g._WORKER_MODE = False
        g._POLICY_CACHE.clear()


def test_random_reference_is_never_cached_and_refusals_keep_their_reason(capsys):
    """`random` draws a fresh net per game in worker mode too, and a
    refusal's reason reaches the worker's stderr."""
    import io
    import torch
    from tools import elo_eval_game as g
    g._POLICY_CACHE.clear()
    g._WORKER_MODE = True
    try:
        cpu = torch.device("cpu")
        a, _ = g._build_player("random", "A", 0, cpu, raw_temperature=0.0, raw_seed=1)
        b, _ = g._build_player("random", "A", 0, cpu, raw_temperature=0.0, raw_seed=1)
        assert a._base is not b._base and not g._POLICY_CACHE
    finally:
        g._WORKER_MODE = False
        g._POLICY_CACHE.clear()
    argv = ["x", "A", "dummy", "B", "dummy", "1", "3", "unused_dir",
            "--mcts-sims", "4", "--raw-temperature-a", "0"]        # refused: sims > 0
    old = sys.stdin
    sys.stdin = io.StringIO(json.dumps(argv) + "\n")
    try:
        assert g.worker_loop() == 0
    finally:
        sys.stdin = old
    captured = capsys.readouterr()
    assert "__DONE__ 1" in captured.out
    assert "raw-temperature" in captured.err.lower() or "sims" in captured.err.lower()


def test_timeout_artifact_passes_the_resume_guards(tmp_path):
    """A game killed by the per-game timeout leaves an artifact that
    the next chunk's pre-scan accepts (it once lacked mcts_batch and
    the precision fields, so every resume of a cuda outdir aborted)."""
    from tools.run_elo_batch import main
    out = tmp_path / "games"
    common = ["x", "--label-a", "A", "--spec-a", "dummy", "--label-b", "B",
              "--spec-b", "dummy", "--outdir", str(out), "--games", "1",
              "--mcts-sims", "0", "--max-turns", "2", "--device", "cpu",
              "--jobs", "1", "--max-extra-games", "0", "--time-budget-min", "5",
              "--min-free-mb", "0", "--mcts-batch-size", "4"]
    main(common + ["--per-game-timeout-min", "0.0001"])
    files = sorted(out.glob("game_*.json"))
    assert len(files) == 1
    art = json.loads(files[0].read_text(encoding="utf-8"))
    assert art["mcts_batch"] == 4 and art["infer_bf16"] is False and art["infer_compile"] is False
    rc = main(common)                    # the resume must not raise SystemExit
    assert rc == 0
