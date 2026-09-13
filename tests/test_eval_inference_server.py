"""Shared batched inference for the eval workers
(tools/eval_inference_server.py, plan 1.5): the coalescing window,
the end-to-end path through the batch driver, and the refusal to mix
shared-inference games with per-process games."""
from __future__ import annotations

import json
import sys
import threading
from multiprocessing.connection import Client, Listener
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def _service(window_s: float, max_batch: int, poison=None):
    """An in-process service over a fake model: each payload is a
    one-element list, its reply the element doubled; the batch sizes
    it saw are the coalescer's record. A payload holding `poison`
    makes the forward raise."""
    from tools.eval_inference_server import InferenceService
    seen = []

    def infer(payloads, stats=None):
        seen.append(sum(len(p) for p in payloads))
        if poison is not None and any(poison in p for p in payloads):
            raise ValueError(f"poisoned position {poison}")
        return [[2 * p[0]] for p in payloads]

    listener = Listener()
    svc = InferenceService(listener, infer, {"hello": True},
                           window_s=window_s, max_batch=max_batch)
    svc.start()
    return svc, listener.address, seen


def _send_together(address: str, values):
    """One connection per value, all requests sent on a barrier;
    returns the replies by value."""
    conns = [Client(address) for _ in values]
    barrier = threading.Barrier(len(values))
    replies = {}

    def go(conn, v):
        barrier.wait()
        conn.send(("infer", v, [v]))
        replies[v] = conn.recv()

    threads = [threading.Thread(target=go, args=(c, v)) for c, v in zip(conns, values)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    for c in conns:
        c.close()
    return replies


def test_coalescer_window_and_max_batch():
    svc, address, seen = _service(window_s=0.25, max_batch=3)
    try:
        c = Client(address)
        c.send(("hello",))
        assert c.recv() == {"hello": True}
        c.send(("infer", 7, [5]))
        assert c.recv() == (7, [10])
        c.close()
        assert seen == [1]
        # Three at once inside the window: one batch of three.
        replies = _send_together(address, [1, 2, 3])
        assert replies == {1: (1, [2]), 2: (2, [4]), 3: (3, [6])}
        assert seen[1:] == [3]
        # Five at once against max_batch 3: no batch above the cap,
        # every request answered.
        replies = _send_together(address, [11, 12, 13, 14, 15])
        assert replies == {v: (v, [2 * v]) for v in (11, 12, 13, 14, 15)}
        assert max(seen[2:]) <= 3 and sum(seen[2:]) == 5
        st = svc.stats.as_dict()
        assert st["requests"] == 9 and st["leaves"] == 9
        assert st["batches"] == len(seen) and st["mean_batch"] == pytest.approx(9 / len(seen))
    finally:
        svc.stop()


def test_a_failing_request_does_not_fail_its_batch_mates():
    """A batch holds one decision from each of several games: when
    the forward raises, the requests are retried one at a time and
    only the offender's game sees the error."""
    svc, address, seen = _service(window_s=0.25, max_batch=8, poison=13)
    try:
        replies = _send_together(address, [11, 13, 14])
        assert replies[11] == (11, [22]) and replies[14] == (14, [28])
        assert replies[13][:2] == (13, None) and "poisoned position 13" in replies[13][2]
        assert 3 in seen and seen.count(1) >= 3, seen
        st = svc.stats.as_dict()
        assert st["failed_batches"] == 1 and st["failed_requests"] == 1
        assert st["requests"] == 3
    finally:
        svc.stop()


def test_startup_failure_reports_the_servers_stderr(tmp_path):
    """A server that dies before serving (here: a --spec that does not
    exist) surfaces its own stderr in the exception and names the log
    it was written to."""
    from tools.eval_inference_server import launch_inference_server
    with pytest.raises(RuntimeError) as excinfo:
        launch_inference_server(str(tmp_path / "missing.pt"), tmp_path, "t", device="cpu",
                                infer_bf16=None, window_ms=1.0, max_batch=2,
                                startup_timeout_s=120.0)
    msg = str(excinfo.value)
    assert "does not exist" in msg, msg
    assert ".inference_server_t.log" in msg


def _tiny_checkpoint(path: Path) -> str:
    """A random tiny net whose vocab holds the ladder unit types (a
    checkpoint's vocab is what the server hands the workers)."""
    import torch
    from sim_test_helpers import fresh_scenario_sim
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(1)
    pol = TransformerPolicy(device=torch.device("cpu"), d_model=32, num_layers=1,
                            num_heads=2, d_ff=64)
    for seed in range(3):
        pol._inference_encoder.register_names(fresh_scenario_sim(seed=seed).gs)
    pol.save_checkpoint(path)
    return str(path)


def test_shared_inference_end_to_end(tmp_path):
    """Two 2-turn games, both sides the same tiny checkpoint at argmax
    through ONE server, two workers: result files carry the shared
    provenance, and the server's stats account for every forward the
    workers counted."""
    from tools.run_elo_batch import main
    spec = _tiny_checkpoint(tmp_path / "tiny.pt")
    out = tmp_path / "games"
    rc = main(["x", "--label-a", "A", "--spec-a", spec, "--label-b", "B",
               "--spec-b", spec, "--outdir", str(out), "--games", "2",
               "--mcts-sims", "0", "--raw-temperature-a", "0",
               "--raw-temperature-b", "0", "--max-turns", "2", "--device", "cpu",
               "--jobs", "2", "--persistent-workers", "--shared-inference",
               "--max-extra-games", "0", "--time-budget-min", "5",
               "--min-free-mb", "0"])
    assert rc == 0
    files = sorted(out.glob("game_*.json"))
    assert len(files) == 2
    forwards = 0
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        assert r["outcome_a"] in ("win", "loss", "draw", "timeout")
        assert r["procedure_a"] == "raw:t0" and r["procedure_b"] == "raw:t0"
        assert r["shared_inference"] is True
        assert r["infer_bf16"] is False and r["infer_compile"] is False
        assert r["infer_packed_trunk"] is False
        assert r["forwards_a"] > 0 and r["forwards_b"] > 0
        forwards += r["forwards_a"] + r["forwards_b"]
    stats = json.loads((out / ".inference_server_0.json").read_text(encoding="utf-8"))
    assert stats["requests"] == stats["leaves"] == forwards
    assert stats["batches"] >= 1 and stats["connections"] == 2
    assert not list(out.glob(".inference_server_1.*")), "one server for one spec"


def _prev(shared: bool, procedure_b: str = "raw:t0") -> dict:
    return {"label_a": "A", "label_b": "B", "procedure_a": "raw:t0",
            "procedure_b": procedure_b, "max_turns": 2, "mcts_batch": 1,
            "infer_bf16": False, "infer_compile": False,
            "shared_inference": shared, "infer_packed_trunk": False,
            "outcome_a": "win", "side_a": 1, "seed": 10000,
            "combat_stream": "per_game"}


def test_refuses_to_mix_shared_and_per_process(tmp_path):
    """Batched forwards are different numerics: the batch pre-scan
    refuses an outdir in either direction before loading anything, and
    the per-game guard refuses a shared file on a per-process replay."""
    from tools.run_elo_batch import main as batch_main
    from tools.elo_eval_game import main as game_main
    spec = tmp_path / "unused.pt"
    spec.write_bytes(b"")                       # existence check only
    common = ["x", "--label-a", "A", "--spec-a", str(spec), "--label-b", "B",
              "--spec-b", "dummy", "--games", "1", "--mcts-sims", "0",
              "--raw-temperature-a", "0", "--max-turns", "2", "--device", "cpu",
              "--jobs", "1", "--min-free-mb", "0", "--max-extra-games", "0"]
    out = tmp_path / "shared"
    out.mkdir()
    (out / "game_A_B_s1_10000.json").write_text(json.dumps(_prev(True, "raw")),
                                                 encoding="utf-8")
    with pytest.raises(SystemExit, match="shared_inference"):
        batch_main(common + ["--outdir", str(out)])
    with pytest.raises(SystemExit, match="shared_inference"):
        game_main(["x", "A", str(spec), "B", "dummy", "1", "10000", str(out),
                   "--mcts-sims", "0", "--raw-temperature-a", "0",
                   "--max-turns", "2", "--device", "cpu"])
    out2 = tmp_path / "per_process"
    out2.mkdir()
    (out2 / "game_A_B_s1_10000.json").write_text(json.dumps(_prev(False, "raw")),
                                                  encoding="utf-8")
    with pytest.raises(SystemExit, match="shared_inference"):
        batch_main(common + ["--outdir", str(out2), "--persistent-workers",
                             "--shared-inference"])


def test_shared_inference_flag_needs_its_surface(tmp_path):
    """The driver refuses the combinations the server does not serve."""
    from tools.run_elo_batch import main
    spec = tmp_path / "unused.pt"
    spec.write_bytes(b"")
    base = ["x", "--label-a", "A", "--spec-a", str(spec), "--label-b", "B",
            "--spec-b", "dummy", "--outdir", str(tmp_path / "o"), "--device", "cpu"]
    for extra in (["--shared-inference", "--mcts-sims", "0", "--raw-temperature-a", "0"],
                  ["--shared-inference", "--persistent-workers", "--mcts-sims", "0"],
                  ["--shared-inference", "--persistent-workers", "--mcts-sims", "8"]):
        with pytest.raises(SystemExit):
            main(base + extra)
