"""The server's batch profile (tools/server_profile.py): a trace's kernels
by class and by the innermost region that launched them, the device's idle
time, the trunk's arithmetic against its classes' time; and the profiler
around the server's real batch, which serves the same replies."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.server_profile import BatchProfiler, kernel_class, summarize_trace  # noqa: E402


def _trace():
    """Two batches on one thread: regions forward (holding trunk) and
    priors, a launch outside any region, four kernels each, the device busy
    700 us of 1000."""
    ev = []

    def region(name, ts, dur):
        ev.append({"ph": "X", "cat": "user_annotation", "name": name, "ts": ts, "dur": dur, "tid": 1})

    def launch(corr, ts, name, kts, kdur):
        ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": ts, "dur": 1,
                   "tid": 1, "args": {"correlation": corr}})
        ev.append({"ph": "X", "cat": "kernel", "name": name, "ts": kts, "dur": kdur, "tid": 7,
                   "args": {"correlation": corr}})
    for b, t in enumerate((0, 500)):
        region("forward", t, 200)
        region("trunk", t + 10, 100)
        launch(10 * b + 1, t + 20, "ampere_bf16_s16816gemm_bf16_128x128", t + 30, 150)
        launch(10 * b + 2, t + 150, "vectorized_layer_norm_kernel", t + 180, 50)
        region("priors", t + 300, 100)
        launch(10 * b + 3, t + 310, "softmax_warp_forward", t + 330, 100)
        launch(10 * b + 4, t + 450, "Memcpy DtoH", t + 430, 50)
    return {"traceEvents": ev}


def test_a_trace_reads_by_class_by_region_and_idle_time():
    got = summarize_trace(_trace(), wall_s=1e-3, batches=2,
                          shapes=[{"leaves": 2, "tokens": 1000, "tokens_sq": 500_000, "a_max": 9, "h_max": 400,
                                   "k_max": 8}] * 2,
                          dims={"d_model": 64, "num_layers": 2, "d_ff": 128})
    assert got["ms_per_batch"] == {"gemm": 0.15, "softmax": 0.1, "layer_norm": 0.05, "memcpy": 0.05}
    # The layer norm's launch lies inside forward, past trunk's end.
    assert got["region_ms_per_batch"] == {"trunk": 0.15, "priors": 0.1, "forward": 0.05, "outside": 0.05}
    assert got["kernels_per_batch"] == 4 and got["idle_fraction"] == 0.3
    assert got["top_kernels"][0]["class"] == "gemm" and got["top_kernels"][0]["per_batch"] == 1.0
    arith = got["arithmetic"]
    assert arith["gemm_gflop_per_batch"] == round(2 * 1000 * 2 * (4 * 64 * 64 + 2 * 64 * 128) / 1e9, 2)
    assert arith["gemm_tflops"] == pytest.approx(2 * 1000 * 2 * (4 * 64 * 64 + 2 * 64 * 128) / 0.15e-3 / 1e12, abs=0.05)


@pytest.mark.parametrize("name, cls", [
    ("flash_fwd_kernel<Flash_fwd_kernel_traits<32, 128, 64, 4>>", "attention"),
    ("cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x4_tn_align8", "gemm"),
    ("void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float>", "layer_norm"),
    ("void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda>", "cast_copy"),
    ("void at::native::index_elementwise_kernel<128, 4>", "index"),
    ("void at_cuda_detail::cub::DeviceScanKernel", "scan_sort"),
    ("void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl>", "elementwise"),
])
def test_kernel_classes(name, cls):
    assert kernel_class(name, "kernel") == cls


def test_the_profiler_around_a_batch_serves_the_same_replies(tmp_path):
    """Batches 3 and 4 of five profiled on the CPU (no device kernels): the
    replies are those of the unprofiled function, the summary counts the
    profiled batches and their shapes."""
    from tools.bench_serve_host import harvest_leaves
    from tools.inference_seam import InferenceServer
    from tools.serve_loop import KIND_INFER
    from tools.serve_stage import serve_requests
    from wesnoth_ai.byte_wire import encode_request
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.memory import MemoryState
    from wesnoth_ai.model import WesnothModel
    torch.manual_seed(3)
    model = WesnothModel(d_model=32, num_layers=1, num_heads=2, d_ff=64, observation_parity=True,
                         memory_slots=8).eval()
    encoder = GameStateEncoder(d_model=32, relevant_set_hexes=True, fog_hides_enemy_villages=True,
                               terrain_multi_hot=True, observation_parity=True, relevant_set_version=2).eval()
    cpu = torch.device("cpu")
    server = InferenceServer(model, encoder, device=cpu, output_device=cpu, autocast_bf16=False,
                             packed_embed=True)

    class Req:
        def __init__(self, rid, message):
            self.rid, self.message, self.cid, self.leaves = rid, message, 1, 1
    leaves = harvest_leaves(10, seed=6)
    batches = [[Req(i, encode_request([(raw, m, MemoryState(8, None))], prefix=(KIND_INFER, i, 1)))
                for i, (raw, m) in enumerate(leaves[2 * k:2 * k + 2])] for k in range(5)]

    def infer(requests, stats=None):
        return serve_requests(server, requests, stats)
    profiled = BatchProfiler(infer, after=2, batches=2, workdir=tmp_path, device=cpu,
                             dims={"d_model": 32, "num_layers": 1, "d_ff": 64})
    for batch in batches:
        assert profiled(batch, stats={}) == infer(batch)
    got = profiled.summary()
    assert got["batches"] == 2 and "error" not in got, got
    assert got["batch_shape_mean"]["leaves"] == 2 and got["batch_shape_mean"]["tokens"] > 100
    assert not list(tmp_path.iterdir()), "the trace file is removed"
