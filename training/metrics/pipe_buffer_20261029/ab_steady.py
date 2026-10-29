"""Steady decisions a second of each A/B run: the slope of the server's
cumulative timeline between 30 s and 40 s before its last record (the
tail, where games run out, is left out), plus the run's GPU-busy share."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    tl = d / ".inference_server_0.timeline.jsonl"
    if not tl.exists():
        print(d.name, "no timeline")
        continue
    recs = [json.loads(line) for line in tl.read_text().splitlines() if line.strip()]
    end = recs[-1]["wall_s"]
    win = [r for r in recs if 30 <= r["wall_s"] <= end - 40]
    if len(win) < 2:
        print(d.name, "too short", end)
        continue
    a, b = win[0], win[-1]
    dt = b["wall_s"] - a["wall_s"]
    rate = (b["requests"] - a["requests"]) / dt
    gpu = (b["gpu_ms"] - a["gpu_ms"]) / 1000 / dt
    batch = (b["requests"] - a["requests"]) / max(1, b["batches"] - a["batches"])
    print(f"{d.name}: {rate:.1f} decisions/s steady over {a['wall_s']:.0f}-{b['wall_s']:.0f} s, "
          f"GPU busy {gpu:.3f}, mean batch {batch:.1f}, total wall {end:.0f} s")
