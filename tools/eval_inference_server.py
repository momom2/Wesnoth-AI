"""Shared batched inference server for the evaluation workers (plan
step 1.5).

Ten persistent eval workers (tools/eval_workers.py) each ran batch-1
forwards on one GPU: about 24 ms of forward per decision against
1.5 ms per sample batched (docs/box_specs.md, "Evaluation throughput:
persistent workers"). Here one server process per distinct checkpoint
owns the model on the GPU (bf16, the packed varlen trunk where the
flash kernel applies) and serves the workers' forwards in coalesced
batches. The workers keep the game loop, the encoder (RemoteEncoder
with server-side priors) and the raw player, so a game is the same
estimand as the per-process path up to bf16 batched numerics.

Transport: `multiprocessing.connection` Listener/Client (stdlib). The
workers are Popen'd processes, not multiprocessing children, so the
queues of tools/actor_pool.py cannot reach them. Listener/Client is a
Unix domain socket on Linux and a named pipe on Windows behind one
API: a request is one pickle and one length-framed write, a reply the
same back, tens of microseconds each way against ~1.5 ms of forward
per sample. `multiprocessing.managers` adds a proxy layer (a dispatch
pickle and a lock per call, one thread per client that cannot batch
across clients without this same queue underneath); a raw socket is
the same syscalls with the framing and the Windows path written by
hand. Requests are the one-buffer form of wesnoth_ai/leaf_wire.py;
replies the numpy wire dicts of tools/inference_seam.py.

Batching: the serve thread takes the first queued request, collects
what arrives within `--window-ms` up to `--max-batch` leaves, and
runs one forward. Every worker has exactly one decision in flight (the
game loop is sequential: the next state depends on the step's
outcome), so a batch never exceeds the number of workers; the
achieved sizes are recorded in the stats file (`batch_hist`,
`mean_batch`).

Protocol on a connection (client -> server -> client):
  ("hello",)                     -> dict: vocab, relevant_set, precision
  ("infer", rid, PackedRequest)  -> (rid, [wire, ...]) or (rid, None, error)
The server prints `__ADDR__ <address>` then `__INFO__ <json>` on
stdout once it serves, and exits when its stdin closes (the driver's
shutdown), writing `--stats-out` first.

Provenance: results played through a server record
`shared_inference: true`, `infer_bf16` and `infer_packed_trunk` (the
precision path); an outdir never mixes shared and per-process games
(tools/run_elo_batch.py, tools/elo_eval_game.py guards). The
reference procedure tag stays `raw:t0`. Before a shared-inference
gate is quoted, re-pin raw:t0 against itself once (the 20-game
determinism check of docs/box_specs.md): batch composition varies
from run to run, so bf16 batched numerics are not bit-identical to
the per-process path and argmax can flip on near-ties.

Usage (the driver does this; by hand for a smoke):
    python tools/eval_inference_server.py --spec CKPT.pt --device cuda \\
        --window-ms 1.5 --max-batch 10 --stats-out out/.inference_server_0.json
"""
from __future__ import annotations

import argparse
import json
import logging
import queue
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

log = logging.getLogger("eval_inference_server")

ADDR_PREFIX = "__ADDR__ "
INFO_PREFIX = "__INFO__ "
DEFAULT_WINDOW_MS = 1.5


# ---------------------------------------------------------------------
# Service: connections, coalescing, replies (torch-free; generic payloads)
# ---------------------------------------------------------------------

@dataclass
class _Pending:
    conn: Any
    send_lock: threading.Lock
    rid: int
    payload: Any            # len(payload) = leaves it carries


@dataclass
class ServeStats:
    requests: int = 0
    batches: int = 0
    leaves: int = 0
    connections: int = 0
    failed_batches: int = 0      # batches whose forward raised
    failed_requests: int = 0     # requests that still failed alone
    batch_hist: Counter = field(default_factory=Counter)
    idle_s: float = 0.0      # waiting for a first request
    window_s: float = 0.0    # collecting the rest of a batch
    infer_s: float = 0.0
    reply_s: float = 0.0
    gpu_ms: float = 0.0
    started: float = field(default_factory=time.monotonic)

    def as_dict(self) -> Dict:
        return {
            "requests": self.requests, "batches": self.batches,
            "leaves": self.leaves, "connections": self.connections,
            "failed_batches": self.failed_batches,
            "failed_requests": self.failed_requests,
            "mean_batch": (self.leaves / self.batches if self.batches else 0.0),
            "batch_hist": {str(k): v for k, v in sorted(self.batch_hist.items())},
            "idle_s": round(self.idle_s, 3), "window_s": round(self.window_s, 3),
            "infer_s": round(self.infer_s, 3), "reply_s": round(self.reply_s, 3),
            "gpu_ms": round(self.gpu_ms, 1),
            "wall_s": round(time.monotonic() - self.started, 3),
        }


class InferenceService:
    """Accepts connections on `listener`, answers "hello" with `hello`,
    coalesces "infer" requests and replies through `infer_fn`.

    `infer_fn(payloads) -> replies`: one reply per payload, in order;
    `stats` is passed as a keyword so the model path can add device
    time under "gpu_ms". Generic over the payload so the batching is
    testable without torch."""

    def __init__(self, listener: Listener, infer_fn: Callable, hello: Dict, *,
                 window_s: float, max_batch: int):
        self._listener = listener
        self._infer_fn = infer_fn
        self._hello = hello
        self._window = max(0.0, float(window_s))
        self._max_batch = max(1, int(max_batch))
        self._q: "queue.Queue[_Pending]" = queue.Queue()
        self._stop = threading.Event()
        self._serve_thread: Optional[threading.Thread] = None
        self.stats = ServeStats()

    def start(self) -> None:
        self._serve_thread = threading.Thread(target=self._serve, daemon=True,
                                              name="infer-serve")
        self._serve_thread.start()
        threading.Thread(target=self._accept, daemon=True, name="infer-accept").start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        try:
            self._listener.close()
        except OSError:
            pass
        if self._serve_thread is not None:
            self._serve_thread.join(timeout)

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                conn = self._listener.accept()
            except (OSError, EOFError):
                if self._stop.is_set():
                    return
                log.exception("accept failed")
                return
            self.stats.connections += 1
            threading.Thread(target=self._reader, args=(conn,), daemon=True,
                             name="infer-reader").start()

    def _reader(self, conn) -> None:
        send_lock = threading.Lock()
        try:
            while not self._stop.is_set():
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    return
                kind = msg[0]
                if kind == "hello":
                    with send_lock:
                        conn.send(self._hello)
                elif kind == "infer":
                    self._q.put(_Pending(conn, send_lock, msg[1], msg[2]))
                else:
                    with send_lock:
                        conn.send((None, None, f"unknown message kind {kind!r}"))
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _collect(self) -> List[_Pending]:
        """The next batch: the first request (blocking, with a short
        timeout so stop is noticed), then what arrives within the
        window up to max_batch leaves."""
        st = self.stats
        t0 = time.monotonic()
        try:
            first = self._q.get(timeout=0.05)
        except queue.Empty:
            st.idle_s += time.monotonic() - t0
            return []
        t1 = time.monotonic()
        st.idle_s += t1 - t0
        batch = [first]
        n = len(first.payload)
        deadline = t1 + self._window
        while n < self._max_batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = self._q.get(timeout=remaining)
            except queue.Empty:
                break
            batch.append(item)
            n += len(item.payload)
        st.window_s += time.monotonic() - t1
        return batch

    def _infer(self, batch: List[_Pending], gpu: Dict[str, float]) -> list:
        replies = self._infer_fn([p.payload for p in batch], stats=gpu)
        if len(replies) != len(batch):
            raise RuntimeError(f"infer_fn returned {len(replies)} replies "
                               f"for {len(batch)} requests")
        return list(replies)

    def _answer(self, batch: List[_Pending], gpu: Dict[str, float]) -> Tuple[list, list]:
        """(replies, errors) for the batch, one of each per request.
        One forward for the whole batch; when it raises, each request
        alone, so only the offenders fail: a batch holds one decision
        from each of up to `jobs` games, and a fault of one position
        must not cost the other games their play."""
        try:
            return self._infer(batch, gpu), [None] * len(batch)
        except Exception:                          # noqa: BLE001
            error = traceback.format_exc()
        self.stats.failed_batches += 1
        if len(batch) == 1:
            replies, errors = [None], [error]
        else:
            log.error("batch of %d requests failed; retrying them one at a time:\n%s",
                      len(batch), error)
            replies, errors = [], []
            for p in batch:
                try:
                    replies.extend(self._infer([p], gpu))
                    errors.append(None)
                except Exception:                  # noqa: BLE001
                    replies.append(None)
                    errors.append(traceback.format_exc())
        failed = [p.rid for p, e in zip(batch, errors) if e is not None]
        self.stats.failed_requests += len(failed)
        if failed:
            log.error("%d of %d request(s) failed (rids %s):\n%s", len(failed), len(batch),
                      failed, next(e for e in errors if e is not None))
        else:
            log.warning("every request of the failed batch passed alone (a batch-level "
                        "fault, not a position's)")
        return replies, errors

    def _serve(self) -> None:
        st = self.stats
        while not self._stop.is_set():
            batch = self._collect()
            if not batch:
                continue
            t0 = time.monotonic()
            gpu: Dict[str, float] = {}
            replies, errors = self._answer(batch, gpu)
            t1 = time.monotonic()
            for p, reply, error in zip(batch, replies, errors):
                msg = (p.rid, reply) if error is None else (p.rid, None, error)
                try:
                    with p.send_lock:
                        p.conn.send(msg)
                except (OSError, EOFError, ValueError):
                    pass                            # the client went away
            t2 = time.monotonic()
            leaves = sum(len(p.payload) for p in batch)
            st.requests += len(batch)
            st.batches += 1
            st.leaves += leaves
            st.batch_hist[leaves] += 1
            st.infer_s += t1 - t0
            st.reply_s += t2 - t1
            st.gpu_ms += float(gpu.get("gpu_ms", 0.0))


# ---------------------------------------------------------------------
# Client: a worker's transport (tools/inference_seam.InferenceTransport)
# ---------------------------------------------------------------------

class EvalInferenceClient:
    """One connection to a server. `hello` carries the server's vocab
    and precision; `infer_batch` takes the (RawEncoded, PackedMasks)
    pairs RemoteEncoder(server_priors=True) produces and returns
    ModelOutputs carrying `legal_compact`. Synchronous: one request in
    flight per connection."""

    def __init__(self, address: str):
        self.address = address
        self._conn = Client(address)
        self._conn.send(("hello",))
        self.hello: Dict = self._conn.recv()
        self._rid = 0
        # Set on a transport failure: a cached client must not be
        # reused for the next game (elo_eval_game reconnects instead).
        self.broken = False

    def infer(self, raw):
        raise NotImplementedError(
            "the eval inference server serves the priors protocol only: "
            "build the RemoteEncoder with server_priors=True")

    def infer_batch(self, items):
        if not items:
            return []
        if not all(isinstance(it, tuple) for it in items):
            raise TypeError("expected (RawEncoded, PackedMasks) pairs")
        from tools.inference_seam import output_from_wire
        from wesnoth_ai.leaf_wire import pack_request
        rid = self._rid
        self._rid += 1
        try:
            self._conn.send(("infer", rid, pack_request(items)))
            reply = self._conn.recv()
        except (OSError, EOFError) as e:
            self.broken = True
            raise RuntimeError(f"lost the inference server at {self.address}: "
                               f"{e!r}") from e
        if reply[0] != rid:
            raise RuntimeError(f"inference reply for request {reply[0]} while "
                               f"waiting for {rid}")
        if reply[1] is None:
            raise RuntimeError(f"inference server failed on this batch:\n{reply[2]}")
        return [output_from_wire(w) for w in reply[1]]

    def close(self) -> None:
        try:
            self._conn.close()
        except OSError:
            pass


# ---------------------------------------------------------------------
# Launcher for the driver (torch-free)
# ---------------------------------------------------------------------

class InferenceServerHandle:
    """A running server process: its address, the precision it
    reports, and a clean shutdown that returns the stats it wrote."""

    def __init__(self, proc, address: str, info: Dict, errf, stats_path: Path,
                 spec: str, log_path: Path):
        self.proc = proc
        self.address = address
        self.info = info
        self.errf = errf
        self.stats_path = stats_path
        self.spec = spec
        self.log_path = log_path

    def alive(self) -> bool:
        return self.proc.poll() is None

    def err_tail(self, n: int = 4096) -> str:
        """The last `n` bytes of the server's stderr, read from the log
        on disk: valid while the server runs and after shutdown closed
        the handle (2026-09-05 review: read through the closed handle,
        every startup failure reported an empty tail)."""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - n))
                return f.read().decode("utf-8", "replace")
        except OSError:
            return ""

    def shutdown(self, timeout: float = 30.0) -> Optional[Dict]:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            log.warning("inference server for %s did not exit in %.0fs; killing",
                        self.spec, timeout)
            self.proc.kill()
            self.proc.wait()
        try:
            self.errf.close()
        except OSError:
            pass
        if self.stats_path.exists():
            try:
                return json.loads(self.stats_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
        return None


def launch_inference_server(spec: str, outdir: Path, tag: str, *, device: str,
                            infer_bf16: Optional[bool], window_ms: float,
                            max_batch: int, torch_threads: int = 4,
                            startup_timeout_s: float = 600.0,
                            python: Optional[str] = None,
                            packed_embed: bool = True,
                            compile_packed: bool = False,
                            graphed: bool = False) -> InferenceServerHandle:
    """Popen one server for `spec` and wait for its address and info
    lines. Stderr goes to `outdir/.inference_server_<tag>.log`, stats
    to `.inference_server_<tag>.json` (dot-prefixed: the result globs
    never see them)."""
    outdir = Path(outdir)
    stats_path = outdir / f".inference_server_{tag}.json"
    cmd = [python or sys.executable, "-u", str(_THIS), "--spec", str(spec),
           "--device", device, "--window-ms", str(window_ms),
           "--max-batch", str(max_batch), "--torch-threads", str(torch_threads),
           "--stats-out", str(stats_path), "--label", tag]
    if infer_bf16 is not None:
        cmd.append("--infer-bf16" if infer_bf16 else "--no-infer-bf16")
    cmd.append("--packed-embed" if packed_embed else "--no-packed-embed")
    if compile_packed:
        cmd.append("--compile-packed")
    if graphed:
        cmd.append("--graphed")
    log_path = outdir / f".inference_server_{tag}.log"
    errf = open(log_path, "w+b")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=errf, text=True, bufsize=1)
    lines: "queue.Queue[Optional[str]]" = queue.Queue()

    def _pump():
        try:
            for line in proc.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    threading.Thread(target=_pump, daemon=True).start()
    handle = InferenceServerHandle(proc, "", {}, errf, stats_path, spec, log_path)

    def _startup_failure(what: str) -> RuntimeError:
        handle.shutdown(timeout=5.0)            # the child has flushed its stderr after this
        tail = handle.err_tail()[-800:].strip() or "(stderr empty)"
        return RuntimeError(f"inference server for {spec} {what} (rc={proc.returncode}); "
                            f"its stderr tail (full log: {log_path}):\n{tail}")

    deadline = time.monotonic() + startup_timeout_s
    address = info = None
    while address is None or info is None:
        try:
            line = lines.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            raise _startup_failure(f"gave no address in {startup_timeout_s:.0f}s")
        if line is None:
            raise _startup_failure("exited before serving")
        if line.startswith(ADDR_PREFIX):
            address = line[len(ADDR_PREFIX):].rstrip("\r\n")
        elif line.startswith(INFO_PREFIX):
            info = json.loads(line[len(INFO_PREFIX):])
    handle.address = address
    handle.info = info
    return handle


# ---------------------------------------------------------------------
# Server process
# ---------------------------------------------------------------------

def _resolve_device(name: str):
    import torch
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but no CUDA device is visible; "
                             "refusing to silently serve from the CPU")
        return torch.device("cuda")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _model_infer_fn(server) -> Callable:
    """The service's infer_fn over the real model: unpack the packed
    requests, one batched forward with server-side priors, wire the
    outputs back per request."""
    from tools.inference_seam import output_to_wire
    from wesnoth_ai.leaf_wire import unpack_request

    def infer(payloads, stats=None):
        flat = []
        counts = []
        for p in payloads:
            leaves = unpack_request(p)
            flat.extend(leaves)
            counts.append(len(leaves))
        outs = server.infer_batch(flat, stats=stats)
        wires = [output_to_wire(o) for o in outs]
        replies = []
        i = 0
        for k in counts:
            replies.append(wires[i:i + k])
            i += k
        return replies

    return infer


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", required=True, help="checkpoint .pt path")
    ap.add_argument("--label", default="server")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--infer-bf16", action=argparse.BooleanOptionalAction, default=None,
                    help="bf16 autocast for the served forwards. Default AUTO: on "
                         "with cuda, off on cpu; forcing it on with cpu is refused.")
    ap.add_argument("--packed-trunk", action=argparse.BooleanOptionalAction, default=True,
                    help="The packed varlen trunk (model.infer_packed_trunk) where the "
                         "flash kernel applies (cuda + bf16); the padded trunk elsewhere.")
    ap.add_argument("--packed-embed", action=argparse.BooleanOptionalAction, default=True,
                    help="Embed the batch as one packed sequence on the host (the pool's "
                         "default; -4.9 ms of host work per 16-leaf batch); with the "
                         "packed trunk only.")
    ap.add_argument("--compile-packed", action="store_true",
                    help="torch.compile the packed layer loop (model.configure_packed_compile, "
                         "inductor, no CUDA graphs) and warm it up before serving: the "
                         "per-batch launch overhead, which is most of a small batch's cost "
                         "(docs/box_specs.md 2026-09-11). With the packed trunk only.")
    ap.add_argument("--graphed", action=argparse.BooleanOptionalAction, default=False,
                    help="Serve priors batches from per-bucket CUDA graphs "
                         "(wesnoth_ai/graphed_serve.py; cuda + bf16 + packed trunk only): "
                         "the per-batch host launch cost collapses to one graph launch.")
    ap.add_argument("--window-ms", type=float, default=DEFAULT_WINDOW_MS)
    ap.add_argument("--max-batch", type=int, default=16, help="leaves per forward")
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--stats-out", type=Path, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    spec = Path(args.spec)
    if not spec.exists():
        raise SystemExit(f"--spec {spec} does not exist (a missing path would serve a "
                         f"random-init net under a checkpoint's label)")

    import torch
    from tools.eval_sim import _load_policy
    from tools.inference_seam import InferenceServer
    from wesnoth_ai.packed_trunk import check_packed_trunk_supported, flash_varlen_applies
    torch.set_num_threads(max(1, args.torch_threads))
    device = _resolve_device(args.device)
    cuda = device.type == "cuda"
    bf16 = cuda if args.infer_bf16 is None else bool(args.infer_bf16)
    if bf16 and not cuda:
        raise SystemExit("--infer-bf16 requires a cuda device: on cpu it would no-op "
                         "and the results would claim a precision that never ran")
    policy = _load_policy(spec, device, label=args.label, infer_bf16=bf16,
                          infer_compile=False)
    model = policy._inference_model
    encoder = policy._inference_encoder
    packed = False
    if args.packed_trunk and cuda and bf16 and flash_varlen_applies(device, torch.bfloat16):
        check_packed_trunk_supported(model.encoder)
        model.infer_packed_trunk = True
        packed = True
    compiled = False
    if args.compile_packed:
        if not packed:
            log.warning("--compile-packed needs the packed trunk (cuda + bf16); serving eager")
        else:
            model.configure_packed_compile()
            log.info("packed compile warmup: %s", model.warmup_packed_compile())
            compiled = True
    packed_embed = bool(args.packed_embed and packed)
    graphed = None
    if args.graphed:
        if not packed:
            log.warning("--graphed needs the packed trunk (cuda + bf16); serving eager")
        else:
            from wesnoth_ai.graphed_serve import Caps, GraphedServe
            # Eval batches run 5-8 leaves at max_batch 20 (docs/box_specs.md
            # "The eval path does not want more workers or more servers"),
            # so the segment axis is bucketed rather than padded to the
            # maximum every time.
            b_caps = tuple(b for b in (4, 8, 12, 16, 20, 24, 32) if b < args.max_batch)
            graphed = GraphedServe(model, encoder, device,
                                   caps=Caps(b_cap=args.max_batch, b_caps=b_caps + (args.max_batch,)))
    server = InferenceServer(model, encoder, device=device,
                             output_device=torch.device("cpu"), autocast_bf16=bf16,
                             packed_embed=packed_embed, graphed=graphed)
    hello = {
        "spec": str(spec), "device": device.type, "infer_bf16": bf16,
        "packed_trunk": packed, "packed_embed": packed_embed, "compile_packed": compiled,
        "graphed": graphed is not None,
        "relevant_set": bool(getattr(encoder, "relevant_set_hexes", False)),
        "fog_hides_enemy_villages": bool(getattr(encoder, "fog_hides_enemy_villages", False)),
        "type_to_id": dict(encoder.unit_type_to_id),
        "faction_to_id": dict(encoder.faction_to_id),
    }
    listener = Listener()
    service = InferenceService(listener, _model_infer_fn(server), hello,
                               window_s=args.window_ms / 1000.0, max_batch=args.max_batch)
    service.start()
    info = {k: v for k, v in hello.items() if k not in ("type_to_id", "faction_to_id")}
    info.update(window_ms=args.window_ms, max_batch=args.max_batch)
    print(f"{ADDR_PREFIX}{listener.address}", flush=True)
    print(f"{INFO_PREFIX}{json.dumps(info)}", flush=True)
    log.info("serving %s on %s (device=%s bf16=%s packed=%s packed_embed=%s compiled=%s "
             "window=%.1fms max_batch=%d)", spec.name, listener.address, device.type, bf16,
             packed, packed_embed, compiled, args.window_ms, args.max_batch)
    for _line in sys.stdin:            # until the driver closes our stdin
        pass
    service.stop()
    stats = dict(info, **service.stats.as_dict())
    if graphed is not None:
        stats["graphed_serve"] = graphed.summary()
        log.info("graphed serve: %s", stats["graphed_serve"])
    if args.stats_out is not None:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    log.info("served %d requests in %d batches (mean batch %.2f); exiting",
             stats["requests"], stats["batches"], stats["mean_batch"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
