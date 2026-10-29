"""own2's four fp16 GEMM shapes at M = 20480: rocBLAS's default pick
against PyTorch TunableOp's (argv[1]: default | tunable). fp16 only:
hipBLASLt has no fp32 path on gfx1032. Prints per-shape ms and the
max difference from the default pick's output (saved by the default run)."""
import os
import sys
import time

mode = sys.argv[1]
out_dir = sys.argv[2]
if mode == "tunable":
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
    os.environ["PYTORCH_TUNABLEOP_TUNING"] = "1"
    os.environ["PYTORCH_TUNABLEOP_FILENAME"] = os.path.join(out_dir, "tunableop_results.csv")
    os.environ["PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS"] = "500"
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

dev = torch.device("cuda")
M = 20480
shapes = {"qkv": (384, 1152), "out": (384, 384), "ff1": (384, 1536), "ff2": (1536, 384)}
g = torch.Generator(device=dev).manual_seed(0)
total = 0.0
for name, (k, n) in shapes.items():
    x = torch.randn(M, k, device=dev, dtype=torch.float16, generator=g)
    w = torch.randn(n, k, device=dev, dtype=torch.float16, generator=g) * 0.05
    b = torch.randn(n, device=dev, dtype=torch.float16, generator=g)
    y = F.linear(x, w, b)
    path = os.path.join(out_dir, f"ref_{name}.pt")
    if mode == "default":
        torch.save(y.cpu(), path)
        diff = 0.0
    else:
        diff = (y.float().cpu() - torch.load(path, weights_only=True).float()).abs().max().item()
    for _ in range(5):
        F.linear(x, w, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        F.linear(x, w, b)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 30 * 1e3
    total += ms
    print(f"{mode} {name}: {ms:.3f} ms, {2 * M * k * n / ms / 1e9:.2f} TFLOP/s, max diff from default {diff:.4f}")
print(f"{mode} sum {total:.3f} ms")
