"""tools/box_probe.py's pure parts and the server timeline it reads
(tools/serve_loop.py Timeline): a steady window's rates from rows the
serving counters wrote, a row being written is left out, nvidia-smi's and
Vast's names of one GPU or CPU agree, a figure's passes give its mean and
spread, offers rank by the throughput a dollar their GPU's probes predict
(the same CPU's first), the probe's training command is one the trainer
bench parses, and the container's cores and CPU seconds come from its
cgroup. The timeline writes cumulative rows and a last one at stop, and a
write that fails ends the timeline, never the server's exit."""
from __future__ import annotations

import json
import time
from argparse import Namespace

import pytest

from tools import box_probe as bp
from tools import serve_loop


def _counters_at(monkeypatch, stats, t):
    """stats.counters() as the serving thread's clock reads `t`."""
    monkeypatch.setattr(serve_loop.time, "monotonic", lambda: t)
    return stats.counters()


def _serve(stats, requests, batches, leaves, gpu_ms, infer_s, idle_s):
    stats.requests += requests
    stats.batches += batches
    stats.leaves += leaves
    stats.gpu_ms += gpu_ms
    stats.infer_s += infer_s
    stats.idle_s += idle_s


def test_the_steady_window_reads_the_serving_counters_rates(monkeypatch):
    """Rows every 5 s: nothing served for 10 s (the workers start), then
    100 requests in 4 batches of 25 leaves a second, 40 device ms and 50 ms
    of infer a batch, 0.1 s idle a second. The window from 20 s after the
    first request, 30 s long, reads exactly those rates."""
    stats = serve_loop.ServeStats(started=0.0)
    rows = [_counters_at(monkeypatch, stats, t) for t in (5.0, 10.0)]
    for t in range(15, 80, 5):
        _serve(stats, 500, 20, 500, 800.0, 1.0, 0.5)
        rows.append(_counters_at(monkeypatch, stats, float(t)))
    got = bp.steady_rate(rows, warm_s=20.0, window_s=30.0)
    assert got["from_s"] == 35.0 and got["window_s"] == 30.0
    assert got["decisions_per_s"] == 100.0 and got["mean_batch"] == 25.0
    assert got["device_ms_per_batch"] == 40.0 and got["infer_ms_per_batch"] == 50.0
    assert got["device_busy"] == 0.16 and got["idle_fraction"] == 0.1
    assert bp.steady_rate(rows, warm_s=20.0, window_s=60.0) is None, "the timeline does not cover it yet"
    assert bp.steady_rate(rows[:2], warm_s=0.0, window_s=5.0) is None, "no request yet"


def test_a_row_being_written_is_left_out(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({"wall_s": 5.0, "requests": 1}) + "\n" + '{"wall_s": 10.0, "requ', encoding="utf-8")
    assert bp.read_timeline(path) == [{"wall_s": 5.0, "requests": 1}]
    assert bp.read_timeline(tmp_path / "absent.jsonl") == []


def test_the_timeline_writes_cumulative_rows_and_a_last_one_at_stop(tmp_path):
    stats = serve_loop.ServeStats()
    path = tmp_path / "timeline.jsonl"
    timeline = serve_loop.Timeline(stats, path, every_s=0.02).start()
    deadline = time.monotonic() + 10.0
    while len(bp.read_timeline(path)) < 3 and time.monotonic() < deadline:
        _serve(stats, 7, 1, 7, 1.0, 0.01, 0.0)
        time.sleep(0.01)
    _serve(stats, 7, 1, 7, 1.0, 0.01, 0.0)
    timeline.stop()
    rows = bp.read_timeline(path)
    assert len(rows) >= 4 and timeline.failed is None
    for key in ("wall_s", "requests", "batches", "leaves", "gpu_ms", "idle_s", "infer_s"):
        assert all(a[key] <= b[key] for a, b in zip(rows, rows[1:])), key
    assert rows[-1]["requests"] == stats.requests and rows[-1]["batches"] == stats.batches


def test_a_timeline_that_cannot_write_ends_without_failing_the_server(tmp_path):
    """The path's directory does not exist: the rows fail, the thread ends,
    and stop() returns (the server writes its stats after it)."""
    timeline = serve_loop.Timeline(serve_loop.ServeStats(), tmp_path / "absent" / "t.jsonl", every_s=0.01).start()
    time.sleep(0.1)
    timeline.stop()
    assert timeline.failed is not None and "FileNotFoundError" in timeline.failed
    stopped = serve_loop.Timeline(serve_loop.ServeStats(), tmp_path / "absent" / "u.jsonl", every_s=60.0).start()
    stopped.stop()
    assert stopped.failed is not None, "the last row's failure is recorded too"


@pytest.mark.parametrize("smi,vast", [("NVIDIA GeForce RTX 4090", "RTX 4090"),
                                      ("NVIDIA GeForce RTX 3090 Ti", "RTX_3090_Ti")])
def test_a_gpu_has_one_name_whoever_writes_it(smi, vast):
    assert bp.gpu_key(smi) == bp.gpu_key(vast)


def test_a_cpu_name_loses_its_trademarks_frequency_and_filler():
    assert bp.cpu_key("Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz") == "intel xeon gold 6248r"
    assert bp.cpu_key("AMD EPYC 7V13 64-Core Processor") == "amd epyc 7v13"
    assert bp.cpu_key("AMD EPYC 7V13 64-Core Processor") == bp.cpu_key("AMD EPYC 7V13")


def test_a_figure_is_the_mean_and_spread_of_the_passes_that_read_it():
    passes = [{"pass": 0, "decisions_per_s": 2400.0}, {"pass": 1, "failed": "no steady window"},
              {"pass": 2, "decisions_per_s": 2600.0}]
    fig = bp.figure(passes, "decisions_per_s")
    assert fig["decisions_per_s"] == 2500.0 and fig["spread"] == 0.08 and fig["passes"] == passes
    assert "decisions_per_s" not in bp.figure([{"failed": "x"}], "decisions_per_s")
    assert bp.figure([{"decisions_per_s": 10.0}], "decisions_per_s")["spread"] is None


def _probe(gpu, cpu, generation=None, training=None):
    rec = {"probe_version": bp.PROBE_VERSION,
           "facts": {"gpu_key": bp.gpu_key(gpu), "cpu_key": bp.cpu_key(cpu)}}
    if generation is not None:
        rec["generation"] = {"decisions_per_s": generation}
    if training is not None:
        rec["training"] = {"positions_per_second": training}
    return rec


def test_offers_rank_by_the_throughput_a_dollar_their_probes_predict():
    probes = [_probe("NVIDIA GeForce RTX 4090", "AMD EPYC 7V13 64-Core Processor", 2400.0, 1400.0),
              _probe("NVIDIA GeForce RTX 4090", "AMD EPYC 7V13 64-Core Processor", 2600.0),
              _probe("NVIDIA GeForce RTX 4090", "Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz", 1800.0),
              _probe("NVIDIA GeForce RTX 3090", "AMD EPYC 7V13 64-Core Processor", 1200.0)]
    offers = [{"id": 1, "gpu_name": "RTX 4090", "cpu_name": "AMD EPYC 7V13 64-Core Processor", "dph_total": 0.50},
              {"id": 2, "gpu_name": "RTX 4090", "cpu_name": "AMD EPYC 7702 64-Core Processor", "dph_total": 0.40},
              {"id": 3, "gpu_name": "RTX 3090", "cpu_name": "AMD EPYC 7V13 64-Core Processor", "dph_total": 0.15},
              {"id": 4, "gpu_name": "RTX 5090", "cpu_name": "AMD EPYC 7V13 64-Core Processor", "dph_total": 0.30},
              {"id": 5, "gpu_name": "RTX 4090", "cpu_name": "AMD EPYC 7V13 64-Core Processor", "dph_total": 0}]
    rows = bp.rank_offers(offers, probes)
    assert [r["id"] for r in rows] == [3, 2, 1], "no probe of the 5090, no price for offer 5"
    by_id = {r["id"]: r for r in rows}
    assert by_id[1]["expected"] == 2500.0 and by_id[1]["match"] == "gpu+cpu" and by_id[1]["probes"] == 2
    assert by_id[1]["per_dollar"] == round(2500.0 * 3600 / 0.5)
    assert by_id[2]["match"] == "gpu" and by_id[2]["expected"] == 2400.0 and by_id[2]["probes"] == 3
    assert [r["id"] for r in bp.rank_offers(offers, probes, "training")] == [2, 1]


def test_probes_are_found_under_their_directory_and_nothing_else_is(tmp_path):
    (tmp_path / "box_a").mkdir()
    (tmp_path / "box_a" / "probe.json").write_text(json.dumps(_probe("RTX 4090", "x", 1.0)), encoding="utf-8")
    (tmp_path / "speed.json").write_text(json.dumps({"arms": {}}), encoding="utf-8")
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    found = bp.load_probes(tmp_path)
    assert len(found) == 1 and found[0]["_path"].endswith("probe.json")
    assert bp.load_probes(tmp_path / "absent") == []


def test_the_training_pass_is_a_command_the_trainer_bench_parses(tmp_path):
    from tools.bench_sequence_step import parse
    opts = Namespace(train_warmup=3, train_windows=12)
    cmd = bp.training_command(tmp_path / "windows", tmp_path / "own1.pt", tmp_path / "rows.jsonl",
                              tmp_path / "out.pt", opts)
    bench, train = parse(cmd[2:])
    assert bench.command == "run" and bench.warmup == 3 and bench.windows == 12 and bench.out == tmp_path / "rows.jsonl"
    assert train[train.index("--init-from") + 1] == str(tmp_path / "own1.pt")
    for flag in bp.CONFIG_B:
        assert flag in train


def test_the_containers_cores_and_cpu_seconds_come_from_its_cgroup(monkeypatch):
    files = {"/sys/fs/cgroup/cpu.max": "250000 100000", "/sys/fs/cgroup/cpu.stat": "usage_usec 12500000\nuser_usec 1"}
    monkeypatch.setattr(bp, "_read", files.get)
    monkeypatch.setattr(bp.os, "sched_getaffinity", lambda pid: set(range(32)), raising=False)
    assert bp.usable_cores() == 3
    assert bp.cgroup_cpu_seconds() == 12.5
    files.clear()
    files.update({"/sys/fs/cgroup/cpu.max": "max 100000", "/sys/fs/cgroup/cpuacct/cpuacct.usage": "2000000000"})
    assert bp.usable_cores() == 32
    assert bp.cgroup_cpu_seconds() == 2.0
