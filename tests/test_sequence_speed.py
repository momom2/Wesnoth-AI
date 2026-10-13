"""The sequence trainer's speed work (docs/trainer_speed_20261013.md): every
step's row counts the tokens its trunk ran; the staged belief targets are
the ones the positions define; a prebuilt trunk index gives the forward the
index built inside gives; kept activations train what recomputed ones
train; the packed trunk trains what the padded trunk trains without
dropout; a non-finite carried memory resets its slot only."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from helpers.model_streams import hand_raw
from tools.bench_sequence_step import set_dropout
from tools.sequence_timing import plan_kept_steps
from wesnoth_ai.sequence_loss import GameFacts, step_labels
from wesnoth_ai.sequence_streams import GameSide, Step

CPU = torch.device("cpu")
D = 32
QUIET = ["--probe-every", "1000000", "--barrier-positions", "1000000", "--signal-every", "1000000"]


def _trainer(pass_inputs, tmp_path, *extra):
    from tools import sequence_train
    tmp_path.mkdir(parents=True, exist_ok=True)
    args = sequence_train.parse_args([*pass_inputs, *QUIET, "--out", str(tmp_path / "arm.pt"), *extra])
    return sequence_train.build_trainer(args)


def _gradients_before_zeroing(trainer) -> list:
    """Each step's gradients, read as the optimizer zeroes them."""
    seen, zero_grad = [], trainer.opt.zero_grad

    def hooked(*a, **kw):
        seen.append([None if p.grad is None else p.grad.clone() for p in trainer.params])
        return zero_grad(*a, **kw)

    trainer.opt.zero_grad = hooked
    return seen


def _trunk_calls(trainer) -> list:
    """(padded tokens, real tokens) of every call of the padded trunk with
    gradients on: a window's forwards, then their recomputations."""
    calls = []

    def hook(module, args, kwargs):
        if torch.is_grad_enabled():
            src, mask = args[0], kwargs["src_key_padding_mask"]
            calls.append((src.shape[0] * src.shape[1], int((~mask).sum())))

    trainer.model.encoder.register_forward_pre_hook(hook, with_kwargs=True)
    return calls


# ---- the timing stream ---------------------------------------------------------

@pytest.mark.slow
def test_every_steps_row_counts_the_tokens_its_trunk_ran(tmp_path, pass_inputs):
    """tokens_padded is the padded batch's size and tokens_real its unmasked
    tokens, summed over the window's forwards (the trunk's own input and
    key-padding mask, read by a hook); the other timing fields are there."""
    trainer = _trainer(pass_inputs, tmp_path)
    calls = _trunk_calls(trainer)
    want = []
    while not trainer._exhausted():
        calls.clear()
        assert trainer.train_window()
        T = len(calls) // 2                      # each forward is recomputed once in the backward pass
        assert len(calls) == 2 * T and T > 0
        want.append((sum(p for p, _ in calls[:T]), sum(r for _, r in calls[:T])))
    rows = [json.loads(line) for line in (tmp_path / "arm.steps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["tokens_padded"], r["tokens_real"]) for r in rows] == want
    assert any(p > r for p, r in want), "some window pads its rows"
    for r in rows:
        assert r["seconds"] > 0 and r["kept_steps"] == 0 and r["peak_gb"] is None
        assert r["gc_seconds"] >= 0 and r["loader_wait_seconds"] >= 0 and r["decode_seconds"] >= 0


@pytest.mark.parametrize("columns", [False, True], ids=["pickles", "columns"])
def test_the_loader_clocks_its_reads_and_the_waits_for_them(tmp_path, monkeypatch, columns):
    """decode_seconds: the CPU time its threads spent reading records;
    wait_seconds: the time the caller waited for a read not yet done; for
    the pickled records' loader and the columns' alike."""
    from tools import sequence_setup
    from wesnoth_ai.sequence_columns import ColumnarSide

    def burn():
        # 50 ms of CPU in zlib, which releases the GIL as the loaders' inflate
        # does: a loop of Python bytecode would hold it, and the calling thread
        # could reach its read only after the read was done (no wait to clock).
        data, t0 = bytes(range(256)) * 4096, time.thread_time()
        while time.thread_time() - t0 < 0.05:
            zlib.compress(data, 6)

    def slow_read(path):
        burn()
        return SimpleNamespace(sides={1: [path.name]})

    def slow_side(path, side):
        burn()
        return ColumnarSide(file=path.name, side=side, n=0, arrays={})

    monkeypatch.setattr(sequence_setup, "read_record", slow_read)
    monkeypatch.setattr(sequence_setup, "read_side", slow_side)
    loader = (sequence_setup.ColumnLoader if columns else sequence_setup.SequenceLoader)(tmp_path)
    read = loader.side if columns else loader.sides
    loader.prefetch([GameSide("a", 1), GameSide("b", 1)])
    got = read(GameSide("b", 1))
    assert (got.file if columns else got) == ("b.seq.cols" if columns else ["b.seq.zpkl"])
    assert read(GameSide("a", 1)) is not None
    assert loader.wait_seconds > 0.01
    assert loader.decode_seconds >= 0.09
    waited = loader.wait_seconds
    assert read(GameSide("a", 1)) is not None and loader.wait_seconds == waited, "a record already read costs no wait"


def test_the_kept_steps_are_the_longest_suffix_that_fits():
    assert plan_kept_steps([3, 2, 5], 7) == 2
    assert plan_kept_steps([3, 2, 5], 10) == 3
    assert plan_kept_steps([3, 2, 5], 4.9) == 0
    assert plan_kept_steps([3, 9, 1], 5) == 1, "a suffix: the step past the one that does not fit stays out"
    assert plan_kept_steps([], 1e9) == 0


# ---- the staging --------------------------------------------------------------

def test_the_staged_belief_targets_are_the_ones_the_positions_define():
    """Rows of differing hex counts: the mask is each row's tokens with no
    visible unit, the target its hidden enemies' tokens, zero past the row."""
    rng = np.random.default_rng(3)
    positions, steps = [], []
    for b, H in enumerate((5, 9, 2, 7)):
        hidden = np.sort(rng.choice(H, size=min(H, b), replace=False)).astype(np.int64)
        positions.append(SimpleNamespace(
            label=SimpleNamespace(action_type="end_turn", actor_idx=0), hidden_tokens=hidden,
            no_visible_unit=rng.random(H) > 0.4))
        steps.append(Step(b, GameSide(f"g{b}", 1), 0, 8, True))
    sizes = [(1, 0, len(p.no_visible_unit)) for p in positions]
    games = {f"g{b}": GameFacts(winner=1, n_commands=10, policy_weight=1.0) for b in range(4)}
    labels = step_labels(positions, steps, sizes, games, seed=1, value_states_per_game=16, value_weight=1.0)
    target, mask = np.zeros((4, 9), np.float32), np.zeros((4, 9), np.float32)
    for b, p in enumerate(positions):
        mask[b, :len(p.no_visible_unit)] = p.no_visible_unit
        target[b, p.hidden_tokens] = 1.0
    on = labels.belief_on(CPU)
    for got in (labels.belief_target, on[0]):
        assert got.dtype == torch.float32 and np.array_equal(got.numpy(), target)
    for got in (labels.belief_mask, on[1]):
        assert got.dtype == torch.float32 and np.array_equal(got.numpy(), mask)


def _parity_pair(slots=8):
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.model import WesnothModel
    torch.manual_seed(0)
    encoder = GameStateEncoder(d_model=D, terrain_multi_hot=True, observation_parity=True, relevant_set_version=2)
    model = WesnothModel(d_model=D, num_layers=2, num_heads=2, d_ff=64, observation_parity=True,
                         memory_slots=slots)
    return encoder, model


@pytest.mark.parametrize("packed", [False, True], ids=["padded", "packed"])
def test_a_prebuilt_trunk_index_gives_the_forward_the_index_built_inside_gives(packed):
    """Rows of differing hexes, units, recruits, sightings and memory slots
    (0 among them), dropout drawing: the outputs, the new memories and every
    parameter's gradient are the same tensors."""
    encoder, model = _parity_pair()
    set_dropout(SimpleNamespace(model=model, encoder=encoder), 0.3)
    raws = [hand_raw(21, U=3, R=2, H=9, parity=True, S=2), hand_raw(22, U=1, R=0, H=5, parity=True, S=0),
            hand_raw(23, U=4, R=1, H=12, parity=True, S=3)]
    ks = [8, 0, 3]
    g = torch.Generator().manual_seed(5)
    states = [0.5 * torch.randn(k, D, generator=g) for k in ks]
    results = []
    for prebuilt in (False, True):
        encoder.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        streams = encoder.encode_from_raw_embedded(raws)
        index = model.trunk_index(streams.sizes, streams.sighting_counts, ks, CPU, packed) if prebuilt else None
        torch.manual_seed(9)
        out = model.forward_embedded(streams, packed=packed, memory=[s.clone() for s in states], index=index)
        (out.actor_logits.sum() + out.target_logits.sum() + out.value_logits.sum()
         + out.belief_logits.sum() + out.memory_padded.sum()).backward()
        grads = [p.grad.clone() if p.grad is not None else None
                 for p in list(model.parameters()) + list(encoder.parameters())]
        results.append((out, grads))
    (a, ga), (b, gb) = results
    for f in ("actor_logits", "type_logits", "target_logits", "weapon_logits", "value_logits", "belief_logits",
              "memory_padded"):
        assert torch.equal(getattr(a, f), getattr(b, f)), f
    assert sum(x is not None for x in ga) > 10
    for x, y in zip(ga, gb):
        assert (x is None and y is None) or torch.equal(x, y)


@pytest.mark.parametrize("sightings, ks", [([2, 0], [5, 3]), ([2, 0], [4, 1]), ([2, 0], [2, 4]), ([0, 2], [4, 2])],
                         ids=["wider memory", "same widest memory", "memory rows swapped", "sighting rows swapped"])
def test_a_trunk_index_for_another_batch_is_refused(sightings, ks):
    """The batch holds sightings [2, 0] and memories of 4 and 2 slots; an
    index built for other per-row counts is refused, also when the widest
    row and the totals are the same (its masks would cover other tokens)."""
    encoder, model = _parity_pair()
    raws = [hand_raw(21, U=3, R=2, H=9, parity=True, S=2), hand_raw(22, U=1, R=0, H=5, parity=True, S=0)]
    streams = encoder.encode_from_raw_embedded(raws)
    assert list(streams.sighting_counts) == [2, 0]
    states = [torch.zeros(4, D), torch.zeros(2, D)]
    with torch.no_grad():
        for packed in (False, True):
            other = model.trunk_index(streams.sizes, sightings, ks, CPU, packed)
            with pytest.raises(ValueError, match="another batch"):
                model.forward_embedded(streams, packed=packed, memory=states, index=other)
            with pytest.raises(ValueError, match="trunk index reached"):
                model.forward_embedded(streams, packed=not packed, memory=states, index=other)


def test_the_compiled_packed_trunk_refuses_training():
    from wesnoth_ai.model import WesnothModel, random_padded_streams
    model = WesnothModel(d_model=16, num_layers=1, num_heads=2, d_ff=32)
    model.infer_packed_trunk = model.infer_compile_packed = True
    with pytest.raises(ValueError, match="inference only"):
        model.forward_streams(*random_padded_streams([(2, 1, 5), (1, 0, 3)], 16, CPU), packed=True)


# ---- the activation budget -------------------------------------------------------

def _three_windows(pass_inputs, tmp_path, *extra):
    """Three windows of two time steps, every dropout at 0.3: per window its
    returned values, gradients, parameters after the step, carried memories,
    kept steps and each forward's padded tokens."""
    trainer = _trainer(pass_inputs, tmp_path, "--window", "2", *extra)
    set_dropout(trainer, 0.3)
    grads, calls = _gradients_before_zeroing(trainer), _trunk_calls(trainer)
    out = []
    for _ in range(3):
        calls.clear()
        result = trainer.train_window()
        assert result
        T = (len(calls) + trainer.last_timing["kept_steps"]) // 2   # the forwards, then the recomputed ones
        out.append({"result": result, "grads": grads[-1], "params": [p.detach().clone() for p in trainer.params],
                    "memories": {s: m.clone() for s, m in trainer.memories.items()},
                    "kept": trainer.last_timing["kept_steps"], "tokens": [p for p, _ in calls[:T]]})
    return out


@pytest.mark.slow
@pytest.mark.parametrize("own", [False, True], ids=["corpus", "own"])
def test_kept_activations_train_what_recomputed_ones_train(tmp_path, request, own):
    """Budget 0, a budget that keeps the last time step of every window, and
    one that keeps them all: the same losses, logged sums, gradients,
    parameters and carried memories, to the bit (the recomputation restores
    the random state its forward drew from, so the dropout masks agree); on
    the corpus alone, and with own game-sides on half the streams."""
    pass_inputs = request.getfixturevalue("own_pass_inputs" if own else "pass_inputs")
    none = _three_windows(pass_inputs, tmp_path / "none")
    assert [w["kept"] for w in none] == [0, 0, 0]
    assert all((w["result"].get("own_n", 0) > 0) == own for w in none)
    last = max(w["tokens"][-1] for w in none)
    assert all(len(w["tokens"]) == 2 for w in none)
    assert last < min(sum(w["tokens"]) for w in none), "no budget keeps exactly one step of every window"
    one_step = ["--activation-bytes-per-token", "1", "--activation-budget-gb", repr(last / 2 ** 30)]
    half = _three_windows(pass_inputs, tmp_path / "half", *one_step)
    every = _three_windows(pass_inputs, tmp_path / "all", "--activation-budget-gb", "1000")
    assert [w["kept"] for w in half] == [1, 1, 1] and [w["kept"] for w in every] == [2, 2, 2]
    for run in (half, every):
        for a, b in zip(none, run):
            assert a["result"] == b["result"]
            for x, y in zip(a["grads"], b["grads"]):
                assert (x is None and y is None) or torch.equal(x, y)
            assert all(torch.equal(x, y) for x, y in zip(a["params"], b["params"]))
            assert a["memories"].keys() == b["memories"].keys()
            assert all(torch.equal(a["memories"][s], b["memories"][s]) for s in a["memories"])


# ---- the packed trunk ------------------------------------------------------------

def _mixed_sizes(monkeypatch):
    """Every training game-side at a memory size of its own."""
    from wesnoth_ai import sequence_streams
    monkeypatch.setattr(sequence_streams, "memory_size",
                        lambda seed, g: 8 * (int(g.file[1]) + 1) - 5 * (g.side - 1))


@pytest.mark.slow
def test_the_packed_trunk_trains_what_the_padded_trunk_trains_without_dropout(tmp_path, pass_inputs, monkeypatch):
    """One window from the same weights, rows of differing hexes and memory
    sizes in a time step: the loss and every parameter's gradient agree
    within float32 rounding between the packed trunk (per-segment attention
    off CUDA) and the padded trunk with its dropout at 0."""
    import wesnoth_ai.model as model_module
    _mixed_sizes(monkeypatch)
    real_packed = model_module.packed_trunk
    packed_calls = []
    monkeypatch.setattr(model_module, "packed_trunk", lambda *a: packed_calls.append(1) or real_packed(*a))
    runs = {}
    for name, extra in (("padded", ()), ("packed", ("--packed-trunk",))):
        trainer = _trainer(pass_inputs, tmp_path / name, "--grad-clip", "1e9", *extra)
        set_dropout(trainer, 0.0)
        grads, seen = _gradients_before_zeroing(trainer), []
        index = trainer._trunk_index
        trainer._trunk_index = lambda steps, staged, _i=index: seen.append(
            ([h for _, _, h in staged.sizes], [s.k for s in steps])) or _i(steps, staged)
        runs[name] = (trainer.train_window(), grads[-1], seen)
    assert packed_calls, "the packed trunk ran"
    (a, ga, seen), (b, gb, _) = runs["padded"], runs["packed"]
    assert any(len(set(hs)) > 1 and len(set(ks)) > 1 for hs, ks in seen), "a time step mixes hexes and sizes"
    assert b["loss"] == pytest.approx(a["loss"], rel=1e-5, abs=1e-6)
    n = 0
    for x, y in zip(ga, gb):
        assert (x is None) == (y is None)
        if x is not None:
            assert torch.allclose(y, x, rtol=1e-5, atol=1e-6), (x - y).abs().max()
            n += 1
    assert n > 10


@pytest.mark.slow
def test_a_pass_resumes_on_its_own_trunk_and_a_new_pass_starts_from_either(tmp_path, pass_inputs):
    from tools import sequence_train
    common = [*pass_inputs, *QUIET]
    first = tmp_path / "first.pt"
    assert sequence_train.main([*common, "--out", str(first), "--max-positions", "4"]) == 0
    with pytest.raises(SystemExit, match="trunk"):
        sequence_train.main([*common, "--out", str(first), "--resume", "--packed-trunk"])
    second = tmp_path / "second.pt"
    assert sequence_train.main([*common, "--out", str(second), "--init-from", str(first), "--packed-trunk",
                                "--max-positions", "4"]) == 0
    assert torch.load(second, map_location="cpu", weights_only=True)["training_meta"]["trunk"] == "packed"
    older = torch.load(first, map_location="cpu", weights_only=True)
    del older["training_meta"]["trunk"]                      # written before the trunk was recorded
    torch.save(older, first)
    assert sequence_train.main([*common, "--out", str(first), "--resume"]) == 0


# ---- the benchmark ----------------------------------------------------------------------

def _bench(*argv) -> subprocess.CompletedProcess:
    """The bench run by path, as the box runs it (it rearranges its import path)."""
    root = Path(__file__).resolve().parent.parent
    return subprocess.run([sys.executable, str(root / "tools" / "bench_sequence_step.py"), *argv], cwd=root,
                          capture_output=True, text=True, timeout=600)


@pytest.mark.slow
def test_the_benchmark_times_two_runs_and_compares_them(tmp_path, pass_inputs):
    """On the CPU, 1 warm-up and 2 timed windows a run: the first record read
    from the live trainer, a row per window with the trainer's timing, the
    synchronized total; the comparison refuses runs that differ in a field
    it was not told of, and gives the speedup when told."""
    from tools import bench_sequence_step as bench
    train = [*pass_inputs, *QUIET, "--window", "2"]
    for name, extra in (("a", ()), ("b", ("--activation-budget-gb", "1000"))):
        done = _bench("run", "--out", str(tmp_path / f"{name}.jsonl"), "--warmup", "1", "--windows", "2",
                      "--profile-windows", "0", "--smi-ms", "0", "--not-production", "--", *train,
                      "--out", str(tmp_path / f"{name}.pt"), *extra)
        assert done.returncode == 0, done.stderr[-2000:]
    config, windows, total = bench.read_run(tmp_path / "a.jsonl")
    assert (config["streams"], config["window"], config["trunk"], config["autocast"]) == (2, 2, "padded", "None")
    assert config["dropout"] and config["fingerprints"]["corpus"] and config["code"]["version"]
    assert [w["i"] for w in windows] == [0, 1] and all(w["seconds"] > 0 and w["tokens_padded"] > 0 for w in windows)
    assert total["positions"] == sum(w["positions"] for w in windows) > 0
    assert bench.read_run(tmp_path / "b.jsonl")[1][0]["kept_steps"] == 2
    assert bench.production_problems(config) and not bench.production_problems(
        {**config, **bench.PRODUCTION})
    with pytest.raises(SystemExit, match="activation_budget_gb"):
        bench.main(["compare", str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")])
    assert bench.main(["compare", str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl"), "--allow",
                       "activation_budget_gb", "activation_bytes_per_token", "--out", str(tmp_path / "c.json")]) == 0
    result = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    assert result["windows"] == 2 and result["speedup_ci95"][0] <= result["speedup_median"] <= result["speedup_ci95"][1]


@pytest.mark.slow
def test_the_gradient_check_passes_the_packed_trunk(tmp_path, pass_inputs):
    """The packed trunk's gradients against the padded trunk's at dropout 0
    (per-segment attention on the CPU): cosines of 1 within rounding."""
    done = _bench("gradcheck", "--out", str(tmp_path / "check.json"), "--windows", "2", "--",
                  *pass_inputs, *QUIET, "--out", str(tmp_path / "arm.pt"))
    assert done.returncode == 0, done.stderr[-2000:]
    check = json.loads((tmp_path / "check.json").read_text(encoding="utf-8"))
    assert check["pass"] and len(check["windows"]) == 2
    cosines = [c for w in check["windows"] for c in w["cosine"].values() if c is not None]
    assert len(cosines) > 5 and min(cosines) > 0.9999


def _run_file(path: Path, tokens) -> Path:
    """A bench run's records: its windows hold 512 positions each, and the
    given (real, padded) trunk tokens (None: a tree that does not count them)."""
    rows = [{"kind": "config", "code": {}, "streams": 32}]
    for i, t in enumerate(tokens):
        row = {"kind": "window", "i": i, "seconds": 2.0 + 0.1 * i, "positions": 512}
        if t is not None:
            row.update(tokens_real=t[0], tokens_padded=t[1])
        rows.append(row)
    rows.append({"kind": "total", "windows": len(tokens), "positions": 512 * len(tokens), "seconds": 9.0,
                 "positions_per_second": 512 * len(tokens) / 9.0})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_the_comparison_refuses_runs_whose_windows_ran_other_tokens(tmp_path):
    """Every window holds 512 positions whatever it trains; the trunk's
    tokens tell the windows apart when both runs count them."""
    from tools import bench_sequence_step as bench
    same = [(9000, 15000), (9100, 15200), (8900, 14800)]
    a = _run_file(tmp_path / "a.jsonl", same)
    assert bench.main(["compare", str(a), str(_run_file(tmp_path / "b.jsonl", same))]) == 0
    for name, other in (("real", [(9000, 15000), (9101, 15200), (8900, 14800)]),
                        ("padded", [(9000, 15000), (9100, 15200), (8900, 14900)])):
        with pytest.raises(SystemExit, match="other tokens"):
            bench.main(["compare", str(a), str(_run_file(tmp_path / f"{name}.jsonl", other))])
    base = _run_file(tmp_path / "base.jsonl", [None, None, None])
    assert bench.main(["compare", str(base), str(a)]) == 0


def test_a_failed_profile_leaves_a_row_that_names_its_error(tmp_path, monkeypatch):
    """The profile comes after the timed windows' total: its failure is a
    row of the run, not the run's failure."""
    from tools import bench_sequence_step as bench

    def broken(trainer, n, workdir):
        raise RuntimeError("CUPTI_ERROR_NOT_INITIALIZED")
    monkeypatch.setattr(bench, "profile_windows", broken)
    row = bench.profile_row(SimpleNamespace(), 3, tmp_path)
    assert row["kind"] == "profile" and row["error"] == "RuntimeError" and "CUPTI" in row["message"]


def test_the_code_record_names_every_trainer_source_by_its_git_blob():
    """The sources the process loaded from the tree, each by the id git
    gives its bytes (`git ls-tree -r <commit>` lists a commit's)."""
    from tools import bench_sequence_step as bench
    from tools import sequence_train  # noqa: F401  (the trainer's modules loaded, as a run loads them)
    root = Path(__file__).resolve().parent.parent
    sources = bench.code_identity(root)["sources"]
    for path in ("tools/sequence_train.py", "tools/bench_sequence_step.py", "wesnoth_ai/model.py",
                 "wesnoth_ai/memory.py"):
        blob = subprocess.run(["git", "hash-object", path], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
        assert sources[path] == blob, path
    assert all(p.split("/")[0] in ("tools", "wesnoth_ai") for p in sources)


def test_the_kernel_classes_and_the_idle_fraction_read_a_trace(tmp_path):
    from tools.bench_sequence_step import trace_summary
    events = [{"ph": "X", "cat": "kernel", "name": "flash_fwd_kernel", "ts": 0, "dur": 100},
              {"ph": "X", "cat": "kernel", "name": "ampere_bf16_s16816gemm_bf16", "ts": 50, "dur": 100},
              {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy HtoD (Pinned -> Device)", "ts": 400, "dur": 100},
              {"ph": "X", "cat": "kernel", "name": "vectorized_elementwise_kernel", "ts": 600, "dur": 200},
              {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 0, "dur": 900}]
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    got = trace_summary(path, wall=1e-3, n_windows=1)
    assert got["ms_per_window"] == {"attention": 0.1, "elementwise_norm": 0.2, "gemm": 0.1, "memcpy": 0.1}
    assert got["idle_fraction"] == pytest.approx(1 - 450 / 1000)


# ---- the telemetry's cadence -------------------------------------------------------------

@pytest.mark.slow
def test_the_telemetry_rows_leave_the_weights_untouched(tmp_path, pass_inputs):
    """A pass with a telemetry row after every window ends with the weights
    of a pass without any (dropout drawing): the cadence is free to change."""
    weights = []
    for name, every in (("rows", "1"), ("none", "1000000")):
        trainer = _trainer(pass_inputs, tmp_path / name, "--window", "2", "--signal-every", every)
        set_dropout(trainer, 0.3)
        while trainer.train_window():
            pass
        weights.append([p.detach().clone() for p in trainer.params])
    rows = (tmp_path / "rows" / "arm.signal.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3 and not (tmp_path / "none" / "arm.signal.jsonl").exists()
    assert all(torch.equal(a, b) for a, b in zip(*weights))


# ---- the read-back -------------------------------------------------------------------

@pytest.mark.slow
def test_a_non_finite_carried_memory_resets_its_slot_only(tmp_path, pass_inputs, monkeypatch):
    from wesnoth_ai import sequence_streams
    monkeypatch.setattr(sequence_streams, "memory_size", lambda seed, g: 8)
    trainer = _trainer(pass_inputs, tmp_path, "--window", "1")
    assert trainer.train_window()
    going_on = [s for s, (i, offset) in enumerate(trainer.schedule.slots) if i >= 0 and offset > 0]
    assert len(going_on) == 2, "both slots carry their game-side into the next window"
    bad, good = going_on
    trainer.memories[bad] = torch.full_like(trainer.memories[bad], float("nan"))
    kept = trainer.memories[good].clone()
    assert trainer.train_window()
    assert trainer.state["memory_resets"] == 1 and trainer.state["nonfinite_steps"] == 1
    assert torch.equal(trainer.memories[bad], trainer.model.initial_memory(8).detach())
    assert torch.isfinite(trainer.memories[good]).all() and not torch.equal(trainer.memories[good], kept)
    assert not torch.equal(trainer.memories[good], trainer.model.initial_memory(8).detach())
