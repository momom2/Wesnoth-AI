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


def test_worker_cache_is_per_side():
    """Same spec on both sides of a match: each side keeps its own
    policy object (own decision counter, pending queue and forward
    counter -- a shared object left side A's counter at 0), while a
    later game with the same label reuses the cached object."""
    import torch
    from tools import elo_eval_game as g
    g._POLICY_CACHE.clear()
    g._WORKER_MODE = True
    try:
        cpu = torch.device("cpu")
        pa, ca = g._build_player("random", "A", 0, cpu, raw_temperature=0.0, raw_seed=1)
        pb, cb = g._build_player("random", "B", 0, cpu, raw_temperature=0.0, raw_seed=1)
        assert pa._base is not pb._base
        assert ca is not cb and pa._base._inference_model is ca and pb._base._inference_model is cb
        pa2, ca2 = g._build_player("random", "A", 0, cpu, raw_temperature=0.0, raw_seed=1)
        assert pa2._base is pa._base and ca2 is not ca            # cached object, fresh counter
        assert ca2._inner is ca._inner                            # the counter never nests
    finally:
        g._WORKER_MODE = False
        g._POLICY_CACHE.clear()
