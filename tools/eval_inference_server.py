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
API: a message is one length-framed write. `multiprocessing.managers`
adds a proxy layer (a dispatch pickle and a lock per call, one thread
per client that cannot batch across clients without this same queue
underneath); a raw socket is the same syscalls with the framing and the
Windows path written by hand. Requests and replies are raw bytes
(wesnoth_ai/byte_wire.py), no pickling.

Batching and the connections (tools/serve_loop.py): one serve thread
reads every connection, takes every ready request and what arrives
within `--window-ms` up to `--max-batch` leaves, and runs one forward.
Every worker game has exactly one decision in flight (the game loop is
sequential: the next state depends on the step's outcome), so a batch
never exceeds the games in flight; the achieved sizes are recorded in
the stats file (`batch_hist`, `mean_batch`), with the serve thread's
seconds by stage.

Host work. With --rust-stage (the default) the Rust core parses each
batch's requests in place and writes the staging buffers and the replies
(tools/serve_stage.py); --no-rust-stage decodes them into Python objects
and builds the buffers with numpy (`InferenceServer.serve_priors`). The
two give the same bytes.

Memory. With --server-memory (the default, for a model with a memory)
each game-side's memory state stays on the device between its decisions
(tools/serve_stage.py `ServerMemory`): a request names the state by the
key its client gave the game-side and the version it read, the reply
gives the version written, and the states of a connection go at its
release (the end of each game) or when it closes. --no-server-memory
sends each state over the wire both ways.

Protocol on a connection (client -> server -> client), int64 words:
  hello: (1, 0, 0)                        -> pickled dict: vocab, relevant_set, precision
  infer: (2, rid, leaves) + request bytes -> (rid, 0) + reply bytes, or (rid, 1) + error text
  release: (3, 0, 0)                      -> no reply; the connection's memory states go
The server prints `__ADDR__ <address>` then `__INFO__ <json>` on
stdout once it serves, and exits when its stdin closes (the driver's
shutdown), writing `--stats-out` first. It exits with status 3 when a
failed batch leaves the device failing its health probe (a sticky CUDA
fault): the stats carry the fault under "fatal", and the driver admits
no more games.

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
import os
import queue
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Callable, Dict, List, Optional

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from tools.serve_loop import (HELLO_MESSAGE, KIND_INFER, RELEASE_MESSAGE,  # noqa: E402
                              REPLY_HEAD_BYTES, REQUEST_HEAD_BYTES, STATUS_OK, InferenceService,
                              error_text, reply_head)
from wesnoth_ai.paths import TOOLS_DIR  # noqa: E402
from wesnoth_ai.stage_timing import stage  # noqa: E402

log = logging.getLogger("eval_inference_server")

# The command line a server process is launched through.
SERVER_SCRIPT = TOOLS_DIR / "eval_inference_server.py"

ADDR_PREFIX = "__ADDR__ "
INFO_PREFIX = "__INFO__ "
DEFAULT_WINDOW_MS = 1.5


# ---------------------------------------------------------------------
# Client: a worker's transport (tools/inference_seam.InferenceTransport)
# ---------------------------------------------------------------------

class EvalInferenceClient:
    """One connection to a server. `hello` carries the server's vocab
    and precision; `infer_batch` takes the (RawEncoded, PackedMasks)
    pairs RemoteEncoder(server_priors=True) produces and returns
    ModelOutputs carrying `legal_compact`. Synchronous: one request in
    flight per connection.

    When the server keeps the memory states (`hello["server_memory"]`),
    a player's state is a `byte_wire.ServedMemory`: a game-side's first
    decision gets a key never used before on this client, and each reply
    the version the server wrote. A state this client did not hand out
    is refused; `release_memories` gives up the states at a game's end."""

    def __init__(self, address: str):
        self.address = address
        self._conn = Client(address)
        self._conn.send_bytes(HELLO_MESSAGE)
        self.hello: Dict = self._conn.recv()
        self._rid = 0
        # Set on a transport failure: a cached client must not be
        # reused for the next game (elo_eval_game reconnects instead).
        self.broken = False
        self.server_memory = bool(self.hello.get("server_memory", False))
        # Which client a served state belongs to: a state handed to another
        # connection (a reconnect, another server) names keys of its own.
        self._token = int.from_bytes(os.urandom(7), "little")
        self._next_key = 0
        self._holds = False             # states held on the server since the last release

    def infer(self, raw):
        raise NotImplementedError(
            "the eval inference server serves the priors protocol only: "
            "build the RemoteEncoder with server_priors=True")

    def infer_batch(self, items):
        if not items:
            return []
        if not all(isinstance(it, tuple) for it in items):
            raise TypeError("expected (RawEncoded, PackedMasks) pairs, or triples with a MemoryState")
        from tools.inference_seam import output_from_wire
        from wesnoth_ai.byte_wire import decode_replies_with_versions, encode_request
        rid = self._rid
        self._rid += 1
        with stage("pack"):
            served = self._served_keys(items)
            request = encode_request(items, prefix=(KIND_INFER, rid, len(items)), served=served)
        try:
            with stage("round_trip"):
                self._conn.send_bytes(request)
                reply = self._conn.recv_bytes()
        except (OSError, EOFError) as e:
            self.broken = True
            raise RuntimeError(f"lost the inference server at {self.address}: "
                               f"{e!r}") from e
        got, status = reply_head(reply)
        if got != rid:
            raise RuntimeError(f"inference reply for request {got} while "
                               f"waiting for {rid}")
        if status != STATUS_OK:
            raise RuntimeError(f"inference server failed on this batch:\n{error_text(reply)}")
        with stage("unwire"):
            # A writable copy: the outputs' tensors are views into it.
            wires, versions = decode_replies_with_versions(bytearray(reply), REPLY_HEAD_BYTES)
            outs = [output_from_wire(w) for w in wires]
            if served is not None:
                self._attach_served(outs, served, versions)
            return outs

    def _served_keys(self, items):
        """Per leaf (key, version read) when the server keeps the memory
        states, None when it does not."""
        from wesnoth_ai.byte_wire import ServedMemory
        states = [it[2].state for it in items if len(it) > 2 and it[2] is not None]
        if not self.server_memory:
            if any(isinstance(st, ServedMemory) for st in states):
                raise ValueError("a memory state the server kept, sent to a server that keeps none")
            return None
        served = []
        for it in items:
            memory = it[2] if len(it) > 2 else None
            state = None if memory is None else memory.state
            if memory is None:
                served.append((-1, 0))
            elif state is None:
                served.append((self._next_key, 0))
                self._next_key += 1
            elif isinstance(state, ServedMemory):
                if state.client != self._token:
                    raise ValueError(f"a memory state of another connection ({state}): the server at "
                                     f"{self.address} keeps this client's states only")
                served.append((state.key, state.version))
            else:
                raise ValueError("the server keeps each game-side's memory state: a state on the wire "
                                 "is refused (a player carried over from a server that keeps none?)")
        self._holds = True
        return served

    def _attach_served(self, outs, served, versions) -> None:
        """Each output's memory: the state the server wrote, as a handle."""
        from wesnoth_ai.byte_wire import ServedMemory
        for out, (key, read), written in zip(outs, served, versions):
            if key < 0:
                continue
            if written != read + 1:
                raise RuntimeError(f"the server wrote memory version {written} for key {key}, which "
                                   f"read version {read}")
            out.memory = ServedMemory(self._token, key, written)

    def release_memories(self) -> None:
        """Every memory state the server keeps for this connection goes (a
        game ended, however). No reply; a lost connection holds none."""
        if not (self.server_memory and self._holds) or self.broken:
            return
        try:
            self._conn.send_bytes(RELEASE_MESSAGE)
        except (OSError, EOFError, ValueError):
            self.broken = True
        self._holds = False

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
                            graphed: bool = False, rust_stage: bool = True,
                            server_memory: bool = True) -> InferenceServerHandle:
    """Popen one server for `spec` and wait for its address and info
    lines. Stderr goes to `outdir/.inference_server_<tag>.log`, stats
    to `.inference_server_<tag>.json` (dot-prefixed: the result globs
    never see them)."""
    outdir = Path(outdir)
    stats_path = outdir / f".inference_server_{tag}.json"
    cmd = [python or sys.executable, "-u", str(SERVER_SCRIPT), "--spec", str(spec),
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
    cmd.append("--rust-stage" if rust_stage else "--no-rust-stage")
    cmd.append("--server-memory" if server_memory else "--no-server-memory")
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


def _read_to_eof(fd: int) -> None:
    """Block until the pipe on `fd` reaches its end. Raw reads hold no
    lock of sys.stdin's buffered reader, so the process can exit while a
    thread still waits here: a daemon thread blocked in that reader makes
    interpreter shutdown abort ("could not acquire lock ... possibly due
    to daemon threads")."""
    while os.read(fd, 1 << 16):
        pass


def _device_health_fn(device) -> Optional[Callable[[], None]]:
    """A probe that raises when `device` can no longer run work. A
    sticky CUDA fault (an illegal address, a device-side assert)
    corrupts the context, so every later call on it fails; a transient
    one, such as an out-of-memory, leaves it usable. None off CUDA."""
    if device.type != "cuda":
        return None
    import torch

    def probe() -> None:
        torch.cuda.synchronize(device)
        (torch.ones(1, device=device) + 1).item()

    return probe


def _model_infer_fn(server, *, rust_stage: bool = True, memory=None) -> Callable:
    """The service's infer_fn over the real model: one batched forward
    with server-side priors, one reply message per request. With
    `rust_stage` the Rust core does the host work (tools/serve_stage.py
    `serve_requests`); otherwise the requests are decoded into Python
    objects and the replies written from the batch's host arrays,
    recording the host seconds of decoding (t_decode) and of writing the
    replies (t_wire). `memory`: the `ServerMemory` that keeps the states."""
    if rust_stage:
        from tools.serve_stage import serve_requests

        def infer_rust(requests, stats=None):
            return serve_requests(server, requests, stats, memory=memory, cids=[r.cid for r in requests])

        return infer_rust
    from wesnoth_ai.byte_wire import ServedMemory, decode_request, encode_replies

    def infer(requests, stats=None):
        from tools.serve_stage import answer_refused, plan_handles
        t0 = time.perf_counter()
        items = []
        counts = []
        for r in requests:
            leaves = decode_request(r.message, REQUEST_HEAD_BYTES)
            items.extend(leaves)
            counts.append(len(leaves))
        memory_batch = versions = writes = None
        if memory is None and any(len(it) > 2 and isinstance(it[2].state, ServedMemory) for it in items):
            raise ValueError("a leaf names a memory state the server keeps, and this server keeps none")
        if memory is not None and items and len(items[0]) > 2:
            leaf_cids = [r.cid for r, k in zip(requests, counts) for _ in range(k)]
            plan = plan_handles(memory, items, leaf_cids)
            if plan.refused:
                return answer_refused(requests, counts, plan.why, memory,
                                      lambda kept: infer(kept, stats))
            memory_batch, writes, versions = plan.memory_batch, plan.writes, plan.versions
        t1 = time.perf_counter()
        try:
            batch = server.serve_priors(items, stats=stats, memory_batch=memory_batch)
            if memory_batch is not None:
                memory.scatter(writes, batch.device_memory)
        except BaseException:
            if memory_batch is not None:
                memory.abort()
            raise
        if memory_batch is not None:
            memory.commit()
        t2 = time.perf_counter()
        replies = []
        i = 0
        for r, k in zip(requests, counts):
            replies.append(encode_replies([batch.wire(b) for b in range(i, i + k)],
                                          prefix=(r.rid, STATUS_OK),
                                          versions=None if versions is None else versions[i:i + k]))
            i += k
        if stats is not None:
            stats["t_decode"] = stats.get("t_decode", 0.0) + (t1 - t0)
            stats["t_wire"] = stats.get("t_wire", 0.0) + (time.perf_counter() - t2)
        return replies

    return infer


def server_hello(spec, checkpoint_sha256: str, encoder, model, *, device: str, bf16: bool,
                 packed: bool, packed_embed: bool, compiled: bool, graphed: bool,
                 rust_stage: bool = True, server_memory: bool = False) -> Dict:
    """What a server tells each connection: the checkpoint it serves, its
    precision path, the structure a worker's encoder must build and the
    vocab."""
    return {
        "spec": str(spec), "checkpoint_sha256": checkpoint_sha256,
        "device": device, "infer_bf16": bf16,
        "packed_trunk": packed, "packed_embed": packed_embed, "compile_packed": compiled,
        "graphed": graphed,
        # The host work in Rust (tools/serve_stage.py), and whether the
        # server keeps each game-side's memory state between decisions.
        "rust_stage": bool(rust_stage), "server_memory": bool(server_memory),
        "relevant_set": bool(getattr(encoder, "relevant_set_hexes", False)),
        "fog_hides_enemy_villages": bool(getattr(encoder, "fog_hides_enemy_villages", False)),
        "terrain_multi_hot": bool(getattr(encoder, "terrain_multi_hot", False)),
        # The parity-memory recipe's structure: what a worker's encoder must
        # build and how many slots a player's memory may use.
        "observation_parity": bool(getattr(encoder, "observation_parity", False)),
        "relevant_set_version": int(getattr(encoder, "relevant_set_version", 1)),
        "memory_slots": int(getattr(model, "memory_slots", 0) or 0),
        "type_to_id": dict(encoder.unit_type_to_id),
        "faction_to_id": dict(encoder.faction_to_id),
    }


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
    ap.add_argument("--rust-stage", action=argparse.BooleanOptionalAction, default=True,
                    help="The batch's host work in Rust (tools/serve_stage.py): the same bytes "
                         "as the Python path, about 1/10 of its host time.")
    ap.add_argument("--server-memory", action=argparse.BooleanOptionalAction, default=True,
                    help="Keep each game-side's memory state on the device between its "
                         "decisions (a model with a memory only); off, the states cross the "
                         "wire both ways at every decision.")
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
    from tools.eval_players import _load_policy
    from tools.inference_seam import InferenceServer
    from tools.eval_provenance import file_sha256
    from wesnoth_ai.packed_trunk import check_packed_trunk_supported, flash_varlen_applies
    torch.set_num_threads(max(1, args.torch_threads))
    device = _resolve_device(args.device)
    cuda = device.type == "cuda"
    bf16 = cuda if args.infer_bf16 is None else bool(args.infer_bf16)
    if bf16 and not cuda:
        raise SystemExit("--infer-bf16 requires a cuda device: on cpu it would no-op "
                         "and the results would claim a precision that never ran")
    # The games record which checkpoint they were served (the hello).
    checkpoint_sha256 = file_sha256(spec)
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
    rust_stage = bool(args.rust_stage)
    if rust_stage and not cuda:
        # The Rust stage writes the embedded path's buffers (stage_raws's
        # layout); on the CPU the Python path serves the same way.
        packed_embed = True
    elif rust_stage and not packed_embed:
        log.warning("--rust-stage writes the packed embed's buffers; with --no-packed-embed the "
                    "Python path serves")
        rust_stage = False
    memory = None
    if args.server_memory and int(getattr(model, "memory_slots", 0) or 0):
        from tools.serve_stage import ServerMemory
        memory = ServerMemory(model, device)
    graphed = None
    if args.graphed:
        if not packed:
            log.warning("--graphed needs the packed trunk (cuda + bf16); serving eager")
        else:
            from wesnoth_ai.graphed_serve import GraphedServe, eval_caps
            # Eval batches run below max_batch, so the segment axis is
            # bucketed rather than padded to the maximum every time --
            # coarsely, since every bucket costs one capture (a 28-worker
            # arm spent its infer time capturing 72 of them,
            # docs/box_specs.md).
            graphed = GraphedServe(model, encoder, device, caps=eval_caps(args.max_batch),
                                   device_memory=memory is not None)
            # The largest bucket first, before any game: the graphs' shared
            # pool reaches its size now, or the server serves eager from the
            # start (wesnoth_ai/graphed_serve.py, "Memory").
            with torch.no_grad():
                log.info("graphed serve preflight: %s", graphed.preflight())
    server = InferenceServer(model, encoder, device=device,
                             output_device=torch.device("cpu"), autocast_bf16=bf16,
                             packed_embed=packed_embed, graphed=graphed)
    hello = server_hello(spec, checkpoint_sha256, encoder, model, device=device.type, bf16=bf16,
                         packed=packed, packed_embed=packed_embed, compiled=compiled,
                         graphed=graphed is not None and graphed.disabled is None,
                         rust_stage=rust_stage, server_memory=memory is not None)
    listener = Listener()
    service = InferenceService(listener, _model_infer_fn(server, rust_stage=rust_stage, memory=memory),
                               hello, window_s=args.window_ms / 1000.0, max_batch=args.max_batch,
                               health_fn=_device_health_fn(device),
                               release_fn=None if memory is None else memory.release_connection)
    service.start()
    info = {k: v for k, v in hello.items() if k not in ("type_to_id", "faction_to_id")}
    info.update(window_ms=args.window_ms, max_batch=args.max_batch)
    print(f"{ADDR_PREFIX}{listener.address}", flush=True)
    print(f"{INFO_PREFIX}{json.dumps(info)}", flush=True)
    log.info("serving %s on %s (device=%s bf16=%s packed=%s packed_embed=%s compiled=%s "
             "window=%.1fms max_batch=%d)", spec.name, listener.address, device.type, bf16,
             packed, packed_embed, compiled, args.window_ms, args.max_batch)
    # Until the driver closes our stdin, or a device fault: then this
    # process exits and the driver, seeing it dead, admits no more games.
    fatal = service.serve_until_closed(lambda: _read_to_eof(sys.stdin.fileno()))
    service.stop()
    stats = dict(info, **service.stats.as_dict(), fatal=fatal)
    if memory is not None:
        stats["memory_states"] = memory.summary()
    if graphed is not None:
        stats["graphed_serve"] = graphed.summary()
        log.info("graphed serve: %s", stats["graphed_serve"])
    if args.stats_out is not None:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    log.info("served %d requests in %d batches (mean batch %.2f); exiting",
             stats["requests"], stats["batches"], stats["mean_batch"])
    return 0 if fatal is None else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv))
