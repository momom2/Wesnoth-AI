"""The serve loop's replies on Windows named pipes, on the CPU: the serve
thread writing each batch's replies (inline) against a writer thread
writing them while the next batch runs (async), interleaved runs.

A stand-in for stage 1's local self-play (docs/selfplay_t1_local_prereg_
20261028.md): one server process running tools/serve_loop.py's
InferenceService on a named pipe (multiprocessing.connection), and 12
client processes of 8 threads, each thread a game with its own connection
(96 in flight). A game sends a request of a real request's size, waits for
its reply, spends its step and sends the next. The step is 2 ms: a busy
loop holding its process's GIL for `--step-gil-ms`, then a sleep (the
Rust core's calls release the GIL). A reply larger than the pipe's 8 kB
buffer is written only as its client reads it, and the client reads only
once it holds its process's GIL, so the GIL-held part sets the cost of a
write: 0.3 ms gives 0.19 ms a write on the inline path, stage 1's being
0.174 ms (README, "Calibration").

The forward is a sleep (it releases the GIL, as a wait on the device
does) of `--fixed-ms + --per-leaf-ms x leaves`: 6.8 ms and 1.84 ms, the
least-squares fit of the server's infer seconds on its batches and leaves
over stage 1's first two matches' 30-second timelines (273 intervals,
README). The replies are the real wire (wesnoth_ai/byte_wire.py
`encode_replies`) of a leaf with n legal actions, n drawn from the
20-quantile midpoints of the decisions logged in stage 1's second match.

    python bench_replies.py --out <dir> [--reps 3]

Each run appends one line to <dir>/results.jsonl: the decisions the
clients completed in the measured window and the server's counters over
that window; the summary prints at the end.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# Legal actions per decision: the midpoints of 20 quantile bins of the
# `n` field (decision_log: the priors' length) over 75,464 decisions of
# the first 200 games of stage 1's second match (games_s01), mean 437.
LEGAL_ACTIONS = (7, 62, 94, 130, 170, 210, 249, 289, 326, 366, 407, 449, 493, 539, 590, 647, 710,
                 795, 923, 1191)
# A request's bytes: own1's encoding of 300 positions of random legal
# games with the memory kept on the server (tools/bench_serve_host.py
# harvest_leaves + requests_of(served=True)): mean 23,825, p10 16,152,
# p90 31,624.
REQUEST_BYTES = 23825
BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
ADDR = "__ADDR__ "
# The counters the server reads over the measured window.
COUNTERS = ("requests", "batches", "leaves", "idle_s", "window_s", "recv_s", "infer_s", "reply_s",
            "send_s", "gpu_ms", "send_failures")


# ---------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------

def reply_bodies():
    """One reply body (the message past its rid and status) per entry of
    LEGAL_ACTIONS: a leaf of 12 units and 6 recruits, the priors of n
    legal actions, the memory kept on the server."""
    import numpy as np
    from wesnoth_ai.byte_wire import encode_replies
    from wesnoth_ai.server_priors import CompactActions
    from tools.serve_loop import REPLY_HEAD_BYTES
    U, R = 12, 6
    bodies = []
    for n in LEGAL_ACTIONS:
        c = CompactActions(actor=np.zeros(n, np.int32), kind=np.zeros(n, np.int8),
                           target=np.zeros(n, np.int32), weapon=np.zeros(n, np.int8),
                           prior=np.full(n, 1.0 / n))
        wire = ("c", U, R, np.zeros((1, 1), np.float32), np.zeros((1, 51), np.float32),
                np.zeros((1, 1), np.float32), np.zeros((1, 1), np.float32),
                np.zeros((1, U + R + 1), np.int8), c, None)
        bodies.append(encode_replies([wire], prefix=(0, 0), versions=[1])[REPLY_HEAD_BYTES:])
    return bodies


def serve(args) -> int:
    """The service on a named pipe, the stand-in forward, its address on
    stdout. Stdin: a line "start" or "end" reads the counters at that
    moment (the measured window's bounds); its end stops the service."""
    from multiprocessing.connection import Listener
    from tools.serve_loop import InferenceService, ok_message
    bodies = reply_bodies()
    rng = random.Random(args.seed)
    fixed, per_leaf = args.fixed_ms / 1000.0, args.per_leaf_ms / 1000.0

    def infer(requests, stats=None):
        leaves = sum(r.leaves for r in requests)
        s = fixed + per_leaf * leaves
        time.sleep(s)
        if stats is not None:
            stats["gpu_ms"] = stats.get("gpu_ms", 0.0) + 1000.0 * s
        return [ok_message(r.rid, bodies[rng.randrange(len(bodies))]) for r in requests]

    listener = Listener()
    svc = InferenceService(listener, infer, {"bench": True}, window_s=args.window_ms / 1000.0,
                           max_batch=args.max_batch, async_replies=args.async_replies)
    svc.start()
    marks = {}

    def commands() -> None:
        for line in sys.stdin:
            name = line.strip()
            if name in ("start", "end"):
                marks[name] = {k: getattr(svc.stats, k) for k in COUNTERS}
                marks[name]["wall_s"] = time.monotonic()

    print(f"{ADDR}{listener.address}", flush=True)
    svc.serve_until_closed(commands)
    svc.stop()
    out = {"async_replies": svc.async_replies, "total": svc.stats.as_dict()}
    if "start" in marks and "end" in marks:
        out["window"] = {k: marks["end"][k] - marks["start"][k] for k in marks["end"]}
    Path(args.stats_out).write_text(json.dumps(out), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------
# The clients
# ---------------------------------------------------------------------

def clients(args) -> int:
    """`--threads` games, each its own connection. Prints READY once every
    game's hello is answered, reads the start time (epoch seconds) on
    stdin, plays from it until the measured window ends, and prints its
    counts as one JSON line."""
    from multiprocessing.connection import Client
    from tools.serve_loop import HELLO_MESSAGE, KIND_INFER, REQUEST_HEAD_BYTES, STATUS_OK, reply_head
    held = args.step_gil_ms / 1000.0
    free = max(0.0, args.step_ms - args.step_gil_ms) / 1000.0
    payload = bytes(args.request_bytes - REQUEST_HEAD_BYTES)
    counts = [0] * args.threads
    # Seconds over the measured decisions: sending the request, waiting for
    # the reply, the step (from the reply to the next send).
    spent = [[0.0, 0.0, 0.0] for _ in range(args.threads)]
    errors = []
    lock = threading.Lock()
    hellos = [0]
    go = threading.Event()
    when = {}

    def game(k: int) -> None:
        try:
            conn = Client(args.address)
            conn.send_bytes(HELLO_MESSAGE)
            conn.recv()
        except Exception as e:                        # noqa: BLE001 -- reported in the result
            errors.append(f"{k}: {e!r}")
            return
        finally:
            with lock:
                hellos[0] += 1
        try:
            go.wait()
            lo, hi = when["lo"], when["hi"]
            time.sleep(max(0.0, when["start"] - time.time()))
            rid = 0
            t_step = time.perf_counter()
            while time.time() < hi:
                t0 = time.perf_counter()
                conn.send_bytes(struct.pack("<qqq", KIND_INFER, rid, 1) + payload)
                t1 = time.perf_counter()
                got, status = reply_head(conn.recv_bytes())
                t2 = time.perf_counter()
                if got != rid or status != STATUS_OK:
                    raise RuntimeError(f"reply {got} status {status} for request {rid}")
                if lo <= time.time() < hi:
                    counts[k] += 1
                    sp = spent[k]
                    sp[0] += t1 - t0
                    sp[1] += t2 - t1
                    sp[2] += t0 - t_step
                rid += 1
                t_step = time.perf_counter()
                end = t_step + held                   # the step's Python: the GIL held
                while time.perf_counter() < end:
                    pass
                if free:
                    time.sleep(free)                  # the step's Rust: the GIL released
            conn.close()
        except Exception as e:                        # noqa: BLE001 -- reported in the result
            errors.append(f"{k}: {e!r}")

    threads = [threading.Thread(target=game, args=(k,)) for k in range(args.threads)]
    for t in threads:
        t.start()
    while hellos[0] < args.threads:
        time.sleep(0.01)
    print("READY", flush=True)
    start = float(sys.stdin.readline())
    when.update(start=start, lo=start + args.warmup_s, hi=start + args.warmup_s + args.measure_s)
    go.set()
    for t in threads:
        t.join()
    print(json.dumps({"decisions": sum(counts), "per_game": counts, "errors": errors,
                      "spent_s": [sum(sp[i] for sp in spent) for i in range(3)]}), flush=True)
    return 0


# ---------------------------------------------------------------------
# The runs
# ---------------------------------------------------------------------

def _sleep_until(t: float) -> None:
    time.sleep(max(0.0, t - time.time()))


def one_run(args, async_replies: bool, rep: int) -> dict:
    """One server, the client processes, every game connected before the
    start; the counts over the measured window."""
    py = sys.executable
    me = str(Path(__file__).resolve())
    mode = "async" if async_replies else "inline"
    stats_out = Path(args.out) / f"server_{rep}_{mode}.json"
    server = subprocess.Popen(
        [py, "-u", me, "--role", "server", "--stats-out", str(stats_out),
         "--async-replies" if async_replies else "--no-async-replies",
         "--fixed-ms", str(args.fixed_ms), "--per-leaf-ms", str(args.per_leaf_ms),
         "--window-ms", str(args.window_ms), "--max-batch", str(args.max_batch), "--seed", str(rep)],
        cwd=str(REPO), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        creationflags=BELOW_NORMAL)
    address = None
    for line in server.stdout:
        if line.startswith(ADDR):
            address = line[len(ADDR):].strip()
            break
    if address is None:
        raise SystemExit(f"the server exited before serving (rc {server.wait()})")
    procs = [subprocess.Popen(
        [py, "-u", me, "--role", "clients", "--address", address, "--threads", str(args.threads),
         "--warmup-s", str(args.warmup_s), "--measure-s", str(args.measure_s),
         "--step-ms", str(args.step_ms), "--step-gil-ms", str(args.step_gil_ms),
         "--request-bytes", str(args.request_bytes)],
        cwd=str(REPO), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        creationflags=BELOW_NORMAL)
        for _ in range(args.processes)]
    for p in procs:
        if p.stdout.readline().strip() != "READY":
            raise SystemExit(f"a client process failed to connect (rc {p.wait()})")
    start = time.time() + 0.5
    for p in procs:
        p.stdin.write(f"{start!r}\n")
        p.stdin.close()
    _sleep_until(start + args.warmup_s)
    server.stdin.write("start\n")
    server.stdin.flush()
    _sleep_until(start + args.warmup_s + args.measure_s)
    server.stdin.write("end\n")
    server.stdin.flush()
    outs = [json.loads(p.communicate()[0].strip().splitlines()[-1]) for p in procs]
    server.stdin.close()
    server.wait(timeout=60)
    stats = json.loads(stats_out.read_text(encoding="utf-8"))
    w = stats["window"]
    decisions = sum(o["decisions"] for o in outs)
    spent = [sum(o["spent_s"][i] for o in outs) for i in range(3)]
    games = sorted(n for o in outs for n in o["per_game"])
    return {
        "rep": rep, "mode": mode,
        "decisions_per_s": decisions / args.measure_s, "decisions": decisions,
        "errors": [e for o in outs for e in o["errors"]],
        "game_decisions_min_median_max": [games[0], games[len(games) // 2], games[-1]],
        "mean_batch": w["leaves"] / w["batches"] if w["batches"] else 0.0,
        # A game's milliseconds per decision: sending its request, waiting
        # for the reply, its step (the reply to the next send).
        "client_ms": {name: 1000.0 * x / max(1, decisions)
                      for name, x in zip(("send", "wait", "step"), spent)},
        "server_window": w, "server_total": stats["total"],
        "config": {k: getattr(args, k) for k in ("fixed_ms", "per_leaf_ms", "window_ms", "max_batch",
                                                 "processes", "threads", "step_ms", "step_gil_ms",
                                                 "request_bytes",
                                                 "warmup_s", "measure_s")},
    }


def stats_line(r: dict) -> str:
    w = r["server_window"]
    return (f"{w['requests']} requests in {w['batches']} batches (mean {r['mean_batch']:.1f}); idle "
            f"{w['idle_s']:.1f}s window {w['window_s']:.1f}s (reads {w['recv_s']:.1f}s) infer "
            f"{w['infer_s']:.1f}s reply {w['reply_s']:.2f}s (writer {w['send_s']:.1f}s, "
            f"{w['send_failures']} failed) of {w['wall_s']:.1f}s wall (stand-in gpu {w['gpu_ms'] / 1000:.1f}s)")


def summarize(rows) -> str:
    lines = []
    for mode in ("inline", "async"):
        rs = [r for r in rows if r["mode"] == mode]
        if not rs:
            continue
        v = [r["decisions_per_s"] for r in rs]
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        lines.append(f"{mode:6s}: {statistics.fmean(v):7.1f} decisions/s (sd {sd:.1f}, {len(v)} runs: "
                     + ", ".join(f"{x:.1f}" for x in v) + ")")
        for r in rs:
            lines.append(f"    rep {r['rep']}: {stats_line(r)}")
    reps = sorted({r["rep"] for r in rows})
    gains = []
    for rep in reps:
        by = {r["mode"]: r["decisions_per_s"] for r in rows if r["rep"] == rep}
        if "inline" in by and "async" in by:
            gains.append(by["async"] / by["inline"] - 1.0)
    if gains:
        sd = statistics.stdev(gains) if len(gains) > 1 else 0.0
        lines.append(f"async over inline, paired by rep: {100 * statistics.fmean(gains):+.1f}% "
                     f"(sd {100 * sd:.1f} points; " + ", ".join(f"{100 * g:+.1f}%" for g in gains) + ")")
    return "\n".join(lines)


def run(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    rows = []
    for rep in range(args.reps):
        order = (False, True) if rep % 2 == 0 else (True, False)
        if args.modes != "both":
            order = (args.modes == "async",)
        for async_replies in order:
            r = one_run(args, async_replies, rep)
            rows.append(r)
            with open(results, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(r) + "\n")
            c = r["client_ms"]
            print(f"rep {rep} {r['mode']}: {r['decisions_per_s']:.1f} decisions/s; {stats_line(r)}; "
                  f"a game's ms per decision: send {c['send']:.1f}, wait {c['wait']:.1f}, step {c['step']:.1f}"
                  + (f"; ERRORS {r['errors'][:3]}" if r["errors"] else ""), flush=True)
    print(summarize(rows), flush=True)
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--role", choices=("run", "server", "clients"), default="run")
    ap.add_argument("--out", default=None, help="run: the directory of results.jsonl and the server stats")
    ap.add_argument("--reps", type=int, default=3, help="run: pairs of runs, the order alternating")
    ap.add_argument("--processes", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8, help="games (connections) per client process")
    ap.add_argument("--step-ms", type=float, default=2.0, help="a game's step between its decisions")
    ap.add_argument("--step-gil-ms", type=float, default=0.3,
                    help="the part of the step that holds the process's GIL (a busy loop); the rest "
                         "sleeps, as the Rust core's calls release the GIL")
    ap.add_argument("--modes", choices=("both", "inline", "async"), default="both")
    ap.add_argument("--request-bytes", type=int, default=REQUEST_BYTES)
    ap.add_argument("--fixed-ms", type=float, default=6.8, help="the stand-in forward's cost per batch")
    ap.add_argument("--per-leaf-ms", type=float, default=1.84, help="... and per leaf")
    ap.add_argument("--window-ms", type=float, default=1.5, help="run_elo_batch's --inference-window-ms")
    ap.add_argument("--max-batch", type=int, default=96, help="run_elo_batch's default: --jobs")
    ap.add_argument("--warmup-s", type=float, default=10.0)
    ap.add_argument("--measure-s", type=float, default=40.0)
    ap.add_argument("--async-replies", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--address", default=None)
    args = ap.parse_args(argv[1:])
    if args.role == "server":
        return serve(args)
    if args.role == "clients":
        return clients(args)
    if args.out is None:
        ap.error("--out is required")
    return run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
