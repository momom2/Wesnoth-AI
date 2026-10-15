"""Self-play generation plays the same games however fast it runs: the
worker's decisions as the code of 2026-10-14 made them (the encodings,
the masks, the view, the step), a game served through the inference
server process (its host work in Rust) as the same game served
in-process, a game whose memory the server keeps as the same game with
the memory on the wire, and games played on threads of one worker as each
game played alone."""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools.raw_player import RawPolicyPlayer  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
FIXTURE = Path(__file__).parent / "data" / "trajectory_digests.json"
_PLAYER_FORWARD = RawPolicyPlayer._forward


@needs_core
def test_the_workers_decisions_are_the_recorded_ones():
    """Per decision of two random legal player games: the encoding and
    masks the worker ships, the view the player reads, the action and
    the core's state after the step, as recorded from main at 0.21.1."""
    from helpers.trajectory_digest import play_digests
    ref = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for seed, want in ref["games"].items():
        got = play_digests(int(seed), ref["decisions"])
        for k, (a, b) in enumerate(zip(got, want)):
            assert a == b, f"game {seed} decision {k}: {a} != {b}"
        assert len(got) == len(want), f"game {seed}: {len(got)} decisions, recorded {len(want)}"


def _memory_checkpoint(path: Path) -> str:
    """A random tiny net of own1's recipe (relevant set 2, terrain set,
    parity observation, a memory of 8 slots) whose vocab holds the ladder
    unit types."""
    from sim_test_helpers import fresh_scenario_sim
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(5)
    pol = TransformerPolicy(device=torch.device("cpu"), d_model=32, num_layers=1, num_heads=2, d_ff=64,
                            relevant_set_hexes=True, fog_hides_enemy_villages=True, terrain_multi_hot=True,
                            observation_parity=True, memory_slots=8, relevant_set_version=2)
    for seed in range(3):
        pol._inference_encoder.register_names(fresh_scenario_sim(seed=seed).gs)
    pol.save_checkpoint(path)
    return str(path)


class _InProcess:
    """The transport a game's players use, with the forward in this
    process: the server's hello, `InferenceServer.infer_batch` as the
    eval server builds it on the CPU."""

    def __init__(self, spec: str, hello: dict):
        from tools.eval_players import _load_policy
        from tools.inference_seam import InferenceServer
        cpu = torch.device("cpu")
        policy = _load_policy(Path(spec), cpu, label="inproc", infer_bf16=False, infer_compile=False)
        # The embedded path, as a CPU server serves under its Rust stage.
        self._server = InferenceServer(policy._inference_model, policy._inference_encoder, device=cpu,
                                       output_device=cpu, autocast_bf16=False, packed_embed=True)
        self.hello = hello
        self.broken = False
        self.address = "in-process"

    def infer_batch(self, items):
        return self._server.infer_batch(items)

    def close(self) -> None:
        pass


def _game_argv(spec: str, seed: int, out: Path, address: str):
    return ["x", "A", spec, "B", spec, str(1 + seed % 2), str(seed), str(out),
            "--mcts-sims", "0", "--raw-temperature-a", "0", "--raw-temperature-b", "0",
            "--raw-end-turn-offset-a", "-8", "--raw-end-turn-offset-b", "-8",
            "--memory-a", "8", "--memory-b", "8", "--max-turns", "4", "--device", "cpu",
            "--inference-address-a", address, "--inference-address-b", address,
            "--no-infer-bf16", "--no-infer-compile", "--no-infer-packed-trunk"]


def _commands(out: Path, seed: int):
    from tools.game_record import read_records
    [path] = sorted(out.glob(f"game_A_B_s*_{seed}.game.jsonl.gz"))
    return next(read_records(path))["commands"]


class _Outputs:
    """Every forward's output the raw players read, per game: the bytes
    of its legal actions' priors, its value and the memory it wrote (the
    server's handle when the server keeps it)."""

    def __init__(self, monkeypatch):
        import threading
        self.by_game: dict = {}
        lock = threading.Lock()
        forward = _PLAYER_FORWARD

        def spy(player, base, encoded, game_label, game_state):
            out = forward(player, base, encoded, game_label, game_state)
            c = out.legal_compact
            memory = out.memory.numpy().tobytes() if torch.is_tensor(out.memory) else out.memory
            row = (c.actor.tobytes(), c.kind.tobytes(), c.target.tobytes(), c.prior.tobytes(),
                   out.value.numpy().tobytes(), memory)
            with lock:
                self.by_game.setdefault(game_label, []).append(row)
            return out

        monkeypatch.setattr(RawPolicyPlayer, "_forward", spy)


@needs_core
@pytest.mark.slow
def test_a_served_game_plays_as_the_in_process_game(tmp_path, monkeypatch):
    """One game through the inference server process (the byte wire, the
    serve loop, the Rust stage, one leaf per batch) against the same game
    with the same forward in-process: the same commands, and the same
    bytes in every reply the players read (priors, value, memory)."""
    import tools.elo_eval_game as eg
    from tools.eval_inference_server import EvalInferenceClient, launch_inference_server
    spec = _memory_checkpoint(tmp_path / "memory.pt")
    served_out = _Outputs(monkeypatch)
    # The server's forwards on as many threads as a game sets for its own
    # (elo_eval_game.main): a sum split over another number of threads
    # may round differently.
    server = launch_inference_server(spec, tmp_path, "t", device="cpu", infer_bf16=None,
                                     window_ms=1.0, max_batch=4, torch_threads=2,
                                     startup_timeout_s=300.0, server_memory=False)
    try:
        served = tmp_path / "served"
        assert eg.main(_game_argv(spec, 21, served, server.address)) == 0
        hello = EvalInferenceClient(server.address).hello
    finally:
        stats = server.shutdown()
    assert stats["requests"] > 100 and stats["mean_batch"] == 1.0
    assert hello == _hello_of(spec)
    local = _InProcess(spec, hello)
    monkeypatch.setattr(eg, "_shared_client", lambda address: local)
    inproc_out = _Outputs(monkeypatch)
    inproc = tmp_path / "inproc"
    assert eg.main(_game_argv(spec, 21, inproc, "in-process")) == 0
    a, b = _commands(served, 21), _commands(inproc, 21)
    assert len(a) > 100 and a == b
    [(_, rows)] = served_out.by_game.items()
    assert len(rows) == stats["requests"]
    assert inproc_out.by_game == served_out.by_game


def _turn_digests(out: Path, seed: int):
    from tools.game_record import read_records
    [path] = sorted(out.glob(f"game_A_B_s*_{seed}.game.jsonl.gz"))
    return next(read_records(path))["turn_digests"]


@needs_core
@pytest.mark.slow
def test_a_game_with_the_memory_on_the_server_plays_as_with_it_on_the_wire(tmp_path, monkeypatch):
    """The same game through two server processes, one keeping each
    game-side's memory between its decisions and one sending it over the
    wire: the same commands and turn digests, the same bytes in every
    reply's priors and value. The game's release leaves the server
    holding no state."""
    import tools.elo_eval_game as eg
    from tools.eval_inference_server import launch_inference_server
    spec = _memory_checkpoint(tmp_path / "memory.pt")
    games, outputs, stats = {}, {}, {}
    for mode in ("wire", "server"):
        outputs[mode] = _Outputs(monkeypatch)
        server = launch_inference_server(spec, tmp_path, mode, device="cpu", infer_bf16=None,
                                         window_ms=1.0, max_batch=4, torch_threads=2,
                                         startup_timeout_s=300.0, server_memory=mode == "server")
        try:
            games[mode] = tmp_path / mode
            assert eg.main(_game_argv(spec, 22, games[mode], server.address)) == 0
        finally:
            stats[mode] = server.shutdown()
    assert _commands(games["wire"], 22) == _commands(games["server"], 22)
    assert len(_commands(games["wire"], 22)) > 100
    assert _turn_digests(games["wire"], 22) == _turn_digests(games["server"], 22)
    [(_, wire)] = outputs["wire"].by_game.items()
    [(_, kept)] = outputs["server"].by_game.items()
    assert len(wire) == len(kept) == stats["server"]["requests"]
    assert [row[:5] for row in wire] == [row[:5] for row in kept]
    assert all(isinstance(row[5], bytes) for row in wire) and not any(isinstance(row[5], bytes) for row in kept)
    memory = stats["server"]["memory_states"]
    assert memory["refused"] == 0 and memory["held"] == 0 and memory["released"] == 2
    assert stats["server"]["server_memory"] is True and stats["wire"]["server_memory"] is False


@needs_core
@pytest.mark.slow
def test_games_on_threads_play_as_each_game_alone(tmp_path, monkeypatch):
    """Three games at once on threads of one process (a worker's
    --worker-games), the forward in-process, against each game played
    alone: the same commands and outputs. The threads run the core's
    calls side by side, its long ones without the GIL."""
    import tools.elo_eval_game as eg
    spec = _memory_checkpoint(tmp_path / "memory.pt")
    hello = _hello_of(spec)
    monkeypatch.setattr(eg, "_shared_client", lambda address: _thread_transport(spec, hello))
    seeds = (31, 32, 33)
    alone_out = _Outputs(monkeypatch)
    alone = tmp_path / "alone"
    for seed in seeds:
        assert eg.main(_game_argv(spec, seed, alone, "in-process")) == 0
    together_out = _Outputs(monkeypatch)
    together = tmp_path / "together"
    with ThreadPoolExecutor(max_workers=len(seeds)) as pool:
        rcs = list(pool.map(lambda s: eg.main(_game_argv(spec, s, together, "in-process")), seeds))
    assert rcs == [0] * len(seeds)
    for seed in seeds:
        a, b = _commands(alone, seed), _commands(together, seed)
        assert len(a) > 50
        assert a == b, f"game {seed}"
    assert len(together_out.by_game) == len(seeds)
    assert together_out.by_game == alone_out.by_game


_TRANSPORTS: dict = {}


def _thread_transport(spec: str, hello: dict) -> _InProcess:
    """One in-process transport per thread, as each game thread of a
    worker holds its own connection."""
    import threading
    key = (spec, threading.get_ident())
    if key not in _TRANSPORTS:
        _TRANSPORTS[key] = _InProcess(spec, hello)
    return _TRANSPORTS[key]


def _hello_of(spec: str) -> dict:
    """The hello the eval server sends for `spec` on the CPU."""
    from tools.eval_inference_server import server_hello
    from tools.eval_players import _load_policy
    from tools.eval_provenance import file_sha256
    policy = _load_policy(Path(spec), torch.device("cpu"), label="hello", infer_bf16=False,
                          infer_compile=False)
    return server_hello(spec, file_sha256(Path(spec)), policy._inference_encoder, policy._inference_model,
                        device="cpu", bf16=False, packed=False, packed_embed=True, compiled=False,
                        graphed=False, rust_stage=True, server_memory=False)
