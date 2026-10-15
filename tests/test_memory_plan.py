"""The activation budget from the device's memory (`--activation-budget-gb
auto`, tools/sequence_memory_plan.py), the allocator's default, and the
box's calibration (tools/bench_memory.py), on the CPU with the device's
memory stood in for: the planner leaves the window's measured rest and the
memory outside PyTorch free, its figures only grow, the trainer plans each
window from the windows before it and records it; expandable segments are
the default only where nothing else is set; the calibration scans the pass,
runs its largest windows where the pass meets them, and an out-of-memory
window fails the calibration while the others run."""
from __future__ import annotations

import json

import pytest
import torch
from test_sequence_launches import QUIET

from tools import sequence_memory_plan as plan
from tools.sequence_memory_plan import GIB, DeviceMemory, MemoryPlanner


class FakeMemory:
    """A device of `capacity` GiB holding `outside` GiB outside PyTorch, whose
    window peaks are read off `peaks` (GiB), the last repeated."""

    def __init__(self, capacity: float, outside: float, peaks):
        self.capacity, self.outside_gb, self.peaks, self.reads = capacity, outside, list(peaks), 0

    def device(self) -> DeviceMemory:
        def peak():
            value = self.peaks[min(self.reads, len(self.peaks) - 1)]
            self.reads += 1
            return int(value * GIB)
        return DeviceMemory(capacity=lambda: int(self.capacity * GIB), outside=lambda: int(self.outside_gb * GIB),
                            peak_allocated=peak)


def test_the_budget_leaves_the_rest_and_the_outside_free_and_its_figures_only_grow():
    memory = FakeMemory(24.0, 0.5, [8.7, 7.5, 12.0])
    planner = MemoryPlanner(memory.device(), 1.0 * GIB, kept_bytes_per_token=102_000)
    free = 22.5 * GIB
    assert planner.budget(100_000) == pytest.approx(free - plan.PRIOR_REST_BYTES_PER_TOKEN * 100_000)
    # Window 1: 100,000 tokens, 80,000 kept in 7.2e9 bytes, a peak of 8.7 GiB.
    planner.observe(100_000, 80_000, int(7.2e9))
    rest = (8.7 * GIB - 7.2e9) / 100_000
    assert planner.kept_per_token == pytest.approx(90_000) and planner.rest_per_token == pytest.approx(rest)
    assert planner.budget(200_000) == pytest.approx(free - rest * 200_000)
    planner.observe(100_000, 80_000, int(6.4e9))          # a lighter window moves neither figure
    assert planner.kept_per_token == pytest.approx(90_000) and planner.rest_per_token == pytest.approx(rest)
    memory.outside_gb = 0.75
    planner.observe(100_000, 0, 0)                        # nothing kept: the kept figure stays
    assert planner.kept_per_token == pytest.approx(90_000)
    assert planner.rest_per_token == pytest.approx(12.0 * GIB / 100_000)
    assert planner.outside == pytest.approx(0.75 * GIB)
    assert planner.budget(10 ** 9) == 0.0
    assert planner.state()["plan_capacity_gb"] == 24.0


def test_expandable_segments_are_the_default_only_where_nothing_else_is_set(monkeypatch):
    cuda = torch.device("cuda")
    monkeypatch.delenv(plan.ALLOC_CONF_VARIABLE, raising=False)
    monkeypatch.setattr(plan, "_SET_BY_TRAINER", {})
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    assert plan.default_cuda_allocator(torch.device("cpu")) == {"alloc_conf": None, "source": "not CUDA"}
    assert plan.ALLOC_CONF_VARIABLE not in __import__("os").environ
    assert plan.default_cuda_allocator(cuda) == {"alloc_conf": "expandable_segments:True", "source": "trainer default"}
    assert plan.default_cuda_allocator(cuda)["source"] == "trainer default"      # a second trainer of the process
    monkeypatch.setenv(plan.ALLOC_CONF_VARIABLE, "garbage_collection_threshold:0.8")
    assert plan.default_cuda_allocator(cuda) == {"alloc_conf": "garbage_collection_threshold:0.8",
                                                 "source": "environment"}
    monkeypatch.delenv(plan.ALLOC_CONF_VARIABLE)
    monkeypatch.setattr(plan, "_SET_BY_TRAINER", {})
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    assert plan.default_cuda_allocator(cuda)["alloc_conf"] is None
    assert plan.ALLOC_CONF_VARIABLE not in __import__("os").environ


def _auto_trainer(pass_inputs, tmp_path, monkeypatch, memory: FakeMemory, *extra):
    from tools import sequence_train
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sequence_train, "device_memory", lambda device: memory.device())
    args = sequence_train.parse_args([*pass_inputs, *QUIET, "--out", str(tmp_path / "auto.pt"), "--packed-trunk",
                                      "--window", "3", "--activation-budget-gb", "auto", *extra])
    return sequence_train.build_trainer(args)


def _rows(tmp_path):
    return [json.loads(line) for line in (tmp_path / "auto.steps.jsonl").read_text().splitlines()]


@pytest.mark.slow
def test_the_trainer_plans_each_window_from_the_ones_before(tmp_path, pass_inputs, monkeypatch):
    from tools import sequence_train
    roomy = _auto_trainer(pass_inputs, tmp_path / "roomy", monkeypatch, FakeMemory(1e6, 0.5, [2.0]))
    roomy.train_window()
    roomy.train_window()
    first, second = _rows(tmp_path / "roomy")
    assert first["kept_steps"] == second["kept_steps"] == 3, "a device with room keeps every step"
    rest = 2.0 * GIB / first["tokens_real"]                 # the CPU measures no kept bytes: the rest is the peak
    assert first["plan_rest_bytes_per_token"] == round(rest)
    assert second["budget_gb"] == pytest.approx((1e6 - 1.5) - rest * second["tokens_real"] / GIB, abs=1e-3)
    meta = torch.load(_save(roomy), map_location="cpu", weights_only=True)["training_meta"]
    assert meta["activation_budget_gb"] == "auto" and meta["cuda_allocator"]["source"] == "not CUDA"
    full = _auto_trainer(pass_inputs, tmp_path / "full", monkeypatch, FakeMemory(1.4, 0.5, [2.0]))
    full.train_window()
    assert _rows(tmp_path / "full")[0]["kept_steps"] == 0 and _rows(tmp_path / "full")[0]["budget_gb"] == 0.0
    monkeypatch.setattr(sequence_train, "device_memory", plan.device_memory)
    with pytest.raises(SystemExit, match="CUDA device's memory"):
        sequence_train.build_trainer(sequence_train.parse_args(
            [*pass_inputs, *QUIET, "--out", str(tmp_path / "x.pt"), "--activation-budget-gb", "auto"]))


def _save(trainer):
    trainer.save()
    return trainer.args.out


@pytest.mark.slow
def test_the_calibration_runs_the_largest_windows_where_the_pass_meets_them(tmp_path, pass_inputs, monkeypatch):
    from tools import bench_memory
    from tools import bench_sequence_step as bench
    from tools import sequence_train
    memory = FakeMemory(1e6, 0.5, [2.0])
    monkeypatch.setattr(sequence_train, "device_memory", lambda device: memory.device())
    flags = "--packed-trunk --activation-budget-gb auto --memory-margin-gb 1"
    common = [*pass_inputs, *QUIET, "--window", "1"]
    out = tmp_path / "memory.json"
    assert bench.main(["memory", "--out", str(out), f"--flags={flags}", "--warmup", "1", "--windows", "1",
                       "--worst", "2", "--", *common, "--out", str(tmp_path / "arm.pt")]) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    fresh = sequence_train.build_trainer(sequence_train.parse_args([*common, "--out", str(tmp_path / "f.pt"),
                                                                    "--packed-trunk"]))
    tokens = []
    while True:
        window = fresh._window()
        if not window:
            break
        tokens.append(bench_memory.window_tokens(fresh, window))
    assert result["scan"]["windows"] == len(tokens) > 4 and result["scan"]["tokens_max"] == max(tokens)
    later = sorted(range(2, len(tokens)), key=lambda i: (-tokens[i], i))[:2]
    assert [r["i"] for r in result["worst"]] == sorted(later)
    assert all(r["tokens_real"] == r["scan_tokens"] == tokens[r["i"]] for r in result["worst"])
    assert result["fits"] and result["flags"] == flags and result["planner"]["plan_capacity_gb"] == 1e6

    real = sequence_train.Trainer.train_window
    calls = {"n": 0}

    def oom_once(self):
        calls["n"] += 1
        if calls["n"] == 3:                                   # the first of the worst windows
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 776.00 MiB.")
        return real(self)
    monkeypatch.setattr(sequence_train.Trainer, "train_window", oom_once)
    out2 = tmp_path / "memory2.json"
    assert bench.main(["memory", "--out", str(out2), f"--flags={flags}", "--warmup", "1", "--windows", "1",
                       "--worst", "2", "--", *common, "--out", str(tmp_path / "arm2.pt")]) == 0
    second = json.loads(out2.read_text(encoding="utf-8"))
    assert "oom" in second["worst"][0] and "tokens_real" in second["worst"][1]
    assert second["fits"] is False and second["flags"] is None and second["ooms"] == 1
