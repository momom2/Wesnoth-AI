"""Shared batched inference for the eval workers
(tools/eval_inference_server.py, plan 1.5): the coalescing window,
the end-to-end path through the batch driver, and the refusal to mix
shared-inference games with per-process games."""
from __future__ import annotations

import json
import struct
import sys
import threading
import time
from multiprocessing.connection import Client, Listener
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from wesnoth_ai.constants import OBSERVATION_EPOCH  # noqa: E402


def _service(window_s: float, max_batch: int, poison=None, health=None, pipelined: bool = False,
             device_s: float = 0.0, poison_at: str = "launch", log=None):
    """An in-process service over a fake model: each request carries one
    int64, its reply the value doubled; the batch sizes it saw are the
    coalescer's record. A request holding `poison` makes the forward
    raise; `health` is the device probe.

    `pipelined`: the pipelined loop over a fake device that runs each
    launched batch for `device_s` seconds after the one before it (a
    queue); the poison raises at the launch or, with
    `poison_at="complete"`, at the completion. `log` gets ("launch" |
    "complete", batch number, the batch's connection numbers) in the
    serve thread's order."""
    from tools.serve_loop import REQUEST_HEAD_BYTES, InferenceService, ok_message
    from tools.serve_pipelined import PipelinedInferenceService
    seen = []

    def values_of(requests):
        return [struct.unpack_from("<q", r.message, REQUEST_HEAD_BYTES)[0] for r in requests]

    def answer(requests, values):
        if poison is not None and poison in values:
            raise ValueError(f"poisoned position {poison}")
        return [ok_message(r.rid, struct.pack("<q", 2 * v)) for r, v in zip(requests, values)]

    def infer(requests, stats=None):
        seen.append(sum(r.leaves for r in requests))
        return answer(requests, values_of(requests))

    listener = Listener()
    kw = dict(window_s=window_s, max_batch=max_batch, health_fn=health)
    if not pipelined:
        svc = InferenceService(listener, infer, {"hello": True}, **kw)
        svc.start()
        return svc, listener.address, seen
    device = {"free_at": 0.0, "n": 0}

    class Launched:
        def __init__(self, requests):
            self.requests, self.values = requests, values_of(requests)
            self.n = device["n"]
            device["n"] += 1
            self.done_at = device["free_at"] = max(time.monotonic(), device["free_at"]) + device_s

        def wait(self):
            time.sleep(max(0.0, self.done_at - time.monotonic()))

        def complete(self):
            self.wait()
            if log is not None:
                log.append(("complete", self.n, sorted(r.cid for r in self.requests)))
            if poison_at == "complete":
                return answer(self.requests, self.values)
            return [ok_message(r.rid, struct.pack("<q", 2 * v)) for r, v in zip(self.requests, self.values)]

    def launch(requests, stats=None):
        seen.append(sum(r.leaves for r in requests))
        launched = Launched(requests)
        if log is not None:
            log.append(("launch", launched.n, sorted(r.cid for r in requests)))
        if poison_at == "launch":
            answer(requests, launched.values)
        return launched

    svc = PipelinedInferenceService(listener, infer, launch, {"hello": True}, **kw)
    svc.start()
    return svc, listener.address, seen


# The loop tests marked so run on the serial loop and on the pipelined one.
BOTH_LOOPS = pytest.mark.parametrize("pipelined", [False, True], ids=["serial", "pipelined"])


def _request(v: int) -> bytes:
    """A one-leaf request of id `v` carrying `v`."""
    from tools.serve_loop import KIND_INFER
    return struct.pack("<qqqq", KIND_INFER, v, 1, v)


def _reply(message):
    """(rid, [value]) for an answer, (rid, None, error text) for a failure."""
    from tools.serve_loop import REPLY_HEAD_BYTES, STATUS_OK, error_text, reply_head
    rid, status = reply_head(message)
    if status == STATUS_OK:
        return rid, [struct.unpack_from("<q", message, REPLY_HEAD_BYTES)[0]]
    return rid, None, error_text(message)


def _send_together(address: str, values):
    """One connection per value, all requests sent on a barrier;
    returns the replies by value."""
    conns = [Client(address) for _ in values]
    barrier = threading.Barrier(len(values))
    replies = {}

    def go(conn, v):
        barrier.wait()
        conn.send_bytes(_request(v))
        replies[v] = _reply(conn.recv_bytes())

    threads = [threading.Thread(target=go, args=(c, v)) for c, v in zip(conns, values)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    for c in conns:
        c.close()
    return replies


@BOTH_LOOPS
def test_a_reply_is_counted_before_it_is_sent(monkeypatch, pipelined):
    """A client holding its reply reads counters that include it: the
    service counts each batch before answering it. The counter is read
    at every reply the serve thread sends, one request per batch."""
    from multiprocessing.connection import _ConnectionBase
    counted_at_send = []
    real_send = _ConnectionBase.send_bytes
    box = {}

    def send_bytes(self, buf, *a, **k):
        if threading.current_thread().name == "infer-serve":
            counted_at_send.append(box["svc"].stats.requests)
        return real_send(self, buf, *a, **k)

    monkeypatch.setattr(_ConnectionBase, "send_bytes", send_bytes)
    svc, address, _seen = _service(window_s=0.0, max_batch=1, pipelined=pipelined)
    box["svc"] = svc
    try:
        replies = _send_together(address, [1, 2, 3])
        assert replies == {1: (1, [2]), 2: (2, [4]), 3: (3, [6])}
        assert counted_at_send == [1, 2, 3]
    finally:
        svc.stop()


@BOTH_LOOPS
def test_coalescer_window_and_max_batch(pipelined):
    from tools.serve_loop import HELLO_MESSAGE
    svc, address, seen = _service(window_s=0.25, max_batch=3, pipelined=pipelined)
    try:
        c = Client(address)
        c.send_bytes(HELLO_MESSAGE)
        assert c.recv() == {"hello": True}
        c.send_bytes(_request(7))
        assert _reply(c.recv_bytes()) == (7, [14])
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
        assert st["connections"] == 9
    finally:
        svc.stop()


@BOTH_LOOPS
def test_a_closed_connection_leaves_the_others_served(pipelined):
    """A client that goes away (a worker killed mid-game) is dropped; the
    games on the other connections keep being answered."""
    svc, address, _seen = _service(window_s=0.0, max_batch=8, pipelined=pipelined)
    try:
        gone = Client(address)
        gone.send_bytes(_request(4))
        gone.close()
        assert _send_together(address, [5, 6]) == {5: (5, [10]), 6: (6, [12])}
    finally:
        svc.stop()


@pytest.mark.skipif(sys.platform != "win32", reason="the buffer is a Windows named pipe's")
def test_a_reply_larger_than_the_old_pipe_buffer_is_written_without_its_client():
    """The server's pipes hold a whole reply: a 64 kB write (an 8 kB pipe
    makes the writer wait for its reader) returns while the client is not
    reading, which it does only seconds later."""
    from tools.eval_inference_server import pipe_listener
    listener = pipe_listener()
    read, got = threading.Event(), []

    def client():
        c = Client(listener.address)
        read.wait(10)
        got.append(len(c.recv_bytes()))
        c.close()
    t = threading.Thread(target=client)
    t.start()
    conn = listener.accept()
    try:
        t0 = time.monotonic()
        conn.send_bytes(b"x" * 65536)
        took = time.monotonic() - t0
    finally:
        read.set()
        t.join(10)
        conn.close()
        listener.close()
    assert took < 1.0, took
    assert got == [65536]


FAILING_AT = pytest.mark.parametrize("pipelined,poison_at", [(False, "launch"), (True, "launch"),
                                                             (True, "complete")],
                                     ids=["serial", "pipelined-launch", "pipelined-complete"])


@FAILING_AT
def test_a_failing_request_does_not_fail_its_batch_mates(pipelined, poison_at):
    """A batch holds one decision from each of several games: when
    the forward raises, the requests are retried one at a time and
    only the offender's game sees the error. Pipelined, the same when the
    batch's launch raises and when its completion does."""
    svc, address, seen = _service(window_s=0.25, max_batch=8, poison=13, pipelined=pipelined,
                                  poison_at=poison_at, device_s=0.02)
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


@FAILING_AT
def test_a_device_that_fails_its_probe_stops_the_service(pipelined, poison_at):
    """A sticky device fault (an illegal address, a device-side assert)
    fails every later call on the device: the 2026-09-14 server answered
    every request after one with the same CUDA error and looked alive
    while 40 of 40 games failed. After a failed batch the service probes
    the device; when the probe fails too, it answers the batch with the
    fault and stops, and serve_until_closed returns the fault so the
    server process exits. Control: a passing probe keeps it serving."""
    def broken():
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    def service(health):
        return _service(window_s=0.0, max_batch=8, poison=13, health=health, pipelined=pipelined,
                        poison_at=poison_at, device_s=0.02)

    svc, address, _seen = service(broken)
    stdin_closed = threading.Event()             # the driver keeps our stdin open
    try:
        replies = _send_together(address, [13])
        assert replies[13][:2] == (13, None)
        assert "illegal memory access" in replies[13][2]
        fault = svc.serve_until_closed(stdin_closed.wait, poll_s=0.01)
        assert fault is not None and "illegal memory access" in fault
    finally:
        stdin_closed.set()
        svc.stop()

    svc, address, _seen = service(lambda: None)
    try:
        replies = _send_together(address, [13])
        assert replies[13][:2] == (13, None) and "poisoned position 13" in replies[13][2]
        assert _send_together(address, [5]) == {5: (5, [10])}
        assert svc.fatal is None
        assert svc.serve_until_closed(lambda: None, poll_s=0.01) is None   # stdin closed
    finally:
        svc.stop()


def test_the_pipelined_loop_answers_every_client_its_own_reply():
    """Sixteen games of fifteen decisions each through the pipelined loop
    over a fake device of 10 ms a batch: every reply is its own request's,
    the loop launched a batch while the one before it ran, and two batches
    on the device together never held one connection (a game's next
    request follows its reply)."""
    log = []
    svc, address, _seen = _service(window_s=0.001, max_batch=64, pipelined=True, device_s=0.01, log=log)
    games, decisions = 16, 15
    bad = []

    def game(k):
        conn = Client(address)
        try:
            for i in range(decisions):
                v = 1000 * k + i
                conn.send_bytes(_request(v))
                got = _reply(conn.recv_bytes())
                if got != (v, [2 * v]):
                    bad.append((v, got))
        finally:
            conn.close()

    threads = [threading.Thread(target=game, args=(k,)) for k in range(games)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert not bad, bad[:5]
        st = svc.stats.as_dict()
        assert st["requests"] == games * decisions
    finally:
        svc.stop()
    running, overlapped = {}, 0
    for event, n, cids in log:
        if event == "launch":
            for other in running.values():
                assert not set(other) & set(cids), (n, cids, running)
            overlapped += bool(running)
            running[n] = cids
        else:
            running.pop(n)
    assert overlapped > 0, log[:10]
    assert not running


def test_stop_answers_the_batch_in_flight():
    """stop() while a batch runs on the device: the serve thread answers it
    before it closes the connections."""
    log = []
    svc, address, _seen = _service(window_s=0.0, max_batch=8, pipelined=True, device_s=0.5, log=log)
    conn = Client(address)
    got = []
    try:
        conn.send_bytes(_request(21))
        reader = threading.Thread(target=lambda: got.append(_reply(conn.recv_bytes())))
        reader.start()
        deadline = time.monotonic() + 10
        while not log and time.monotonic() < deadline:
            time.sleep(0.005)
        assert log and log[0][0] == "launch"
        svc.stop()
        reader.join(10)
        assert got == [(21, [42])]
        assert [e for e, _n, _c in log] == ["launch", "complete"]
    finally:
        conn.close()
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


# The pipelined end-to-end game is in the slow tier: the serial one already
# takes about 20 s on the desktop, and CI runs both.
@pytest.mark.parametrize("pipelined", [False, pytest.param(True, marks=pytest.mark.slow)],
                         ids=["serial", "pipelined"])
def test_shared_inference_end_to_end(tmp_path, pipelined):
    """Two 2-turn games, both sides the same tiny checkpoint at argmax
    through ONE server, two workers: result files carry the shared
    provenance, including the checkpoint the server loaded, and the
    server's stats account for every forward the workers counted.
    Pipelined too (--inference-pipelined), which the server records."""
    import hashlib
    from tools.run_elo_batch import EXIT_GUARD_SPENT, main
    spec = _tiny_checkpoint(tmp_path / "tiny.pt")
    out = tmp_path / "games"
    rc = main(["x", "--label-a", "A", "--spec-a", spec, "--label-b", "B",
               "--spec-b", spec, "--outdir", str(out), "--games", "2",
               "--mcts-sims", "0", "--raw-temperature-a", "0",
               "--raw-temperature-b", "0", "--max-turns", "2", "--device", "cpu",
               "--jobs", "2", "--persistent-workers", "--shared-inference",
               "--max-extra-games", "0", "--time-budget-min", "5",
               "--min-free-mb", "0"] + (["--inference-pipelined"] if pipelined else []))
    # 2 turns decide nothing and no replacement is allowed.
    assert rc == EXIT_GUARD_SPENT
    sha = hashlib.sha256(Path(spec).read_bytes()).hexdigest()
    files = sorted(out.glob("game_*.json"))
    assert len(files) == 2
    forwards = 0
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        assert r["outcome_a"] in ("win", "loss", "draw", "timeout")
        assert r["procedure_a"] == "raw:t0" and r["procedure_b"] == "raw:t0"
        assert r["checkpoint_sha256_a"] == r["checkpoint_sha256_b"] == sha
        assert r["shared_inference"] is True
        assert r["infer_bf16"] is False and r["infer_compile"] is False
        assert r["infer_packed_trunk"] is False
        assert r["forwards_a"] > 0 and r["forwards_b"] > 0
        forwards += r["forwards_a"] + r["forwards_b"]
    stats = json.loads((out / ".inference_server_0.json").read_text(encoding="utf-8"))
    assert stats["pipelined"] is pipelined
    assert stats["requests"] == stats["leaves"] == forwards
    assert stats["batches"] >= 1 and stats["connections"] == 2
    # Every decision's stages, from both workers, summed per decision.
    stages = json.loads((out / ".worker_stages.json").read_text(encoding="utf-8"))
    assert stages["workers"] == 2 and stages["decisions"] == forwards
    per = stages["stage_ms_per_decision"]
    for name in ("snapshot", "decide", "encode_raw", "masks", "pack", "round_trip", "step"):
        assert per[name] > 0, name
    assert per["round_trip"] < per["decide"] < stages["decision_wall_ms"]
    assert not list(out.glob(".inference_server_1.*")), "one server for one spec"


def _prev(shared: bool, procedure_b: str = "raw:t0") -> dict:
    return {"label_a": "A", "label_b": "B", "procedure_a": "raw:t0",
            "procedure_b": procedure_b, "max_turns": 2, "mcts_batch": 1,
            "infer_bf16": False, "infer_compile": False,
            "shared_inference": shared, "infer_packed_trunk": False,
            "outcome_a": "win", "side_a": 1, "seed": 10000,
            "combat_stream": "per_game", "observation_epoch": OBSERVATION_EPOCH}


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


def test_the_resume_guard_reads_the_servers_compile_setting(tmp_path):
    """A shared server's games record its compile_packed as infer_compile;
    the driver expected False whatever the flag, so it could not tell a
    compiled run from an eager one (2026-09-29 audit)."""
    from tools.run_elo_batch import main as batch_main
    spec = tmp_path / "unused.pt"
    spec.write_bytes(b"")                       # existence check only
    common = ["x", "--label-a", "A", "--spec-a", str(spec), "--label-b", "B",
              "--spec-b", "dummy", "--games", "1", "--mcts-sims", "0",
              "--raw-temperature-a", "0", "--max-turns", "2", "--device", "cpu",
              "--jobs", "1", "--min-free-mb", "0", "--max-extra-games", "0",
              "--persistent-workers", "--shared-inference"]
    eager = tmp_path / "eager"
    eager.mkdir()
    (eager / "game_A_B_s1_10000.json").write_text(json.dumps(_prev(True, "raw")), encoding="utf-8")
    with pytest.raises(SystemExit, match="infer_compile"):
        batch_main(common + ["--outdir", str(eager), "--compile-packed"])
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    (compiled / "game_A_B_s1_10000.json").write_text(
        json.dumps({**_prev(True, "raw"), "infer_compile": True}), encoding="utf-8")
    with pytest.raises(SystemExit, match="infer_compile"):
        batch_main(common + ["--outdir", str(compiled)])


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
