"""The certification's verdict (scripts/core_certify_box.sh, its summary
step): the replays summarised are counted against the file list, so a
shard that died without its summary line reads INCOMPLETE, never clean
(2026-09-29 audit: a killed shard summed to a clean sweep), and any
divergence reads DIVERGENT. The step's own Python is run on fake shard
logs."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _summary_code() -> str:
    text = (ROOT / "scripts" / "core_certify_box.sh").read_text(encoding="utf-8")
    m = re.search(r"<<'PYEOF'[^\n]*\n(.*?)\nPYEOF\n", text, re.S)
    assert m, "the summary heredoc is gone"
    return m.group(1)


def _verdict(tmp_path: Path, files: int, logs: dict) -> str:
    shards = tmp_path / "shards"
    shards.mkdir()
    (shards / "files.txt").write_text("".join(f"g{i}.json.gz\n" for i in range(files)), encoding="utf-8")
    for name, text in logs.items():
        (shards / name).write_text(text, encoding="utf-8")
    out = tmp_path / "summary.txt"
    subprocess.run([sys.executable, "-", str(shards), str(out)], input=_summary_code(),
                   text=True, check=True, capture_output=True)
    return out.read_text(encoding="utf-8").splitlines()[0]


def _line(replays: int, clean: int) -> str:
    return (f"diff_core: {replays} replays, {clean} clean, {replays - clean} with divergences\n"
            "state=0, encode=0\n")


def test_a_clean_sweep_reads_clean(tmp_path):
    first = _verdict(tmp_path, 3, {"shard_000.log": _line(2, 2), "shard_001.log": _line(1, 1)})
    assert first.startswith("core certification CLEAN: 3 of 3 replays")


def test_a_divergence_reads_divergent(tmp_path):
    first = _verdict(tmp_path, 3, {"shard_000.log": _line(2, 1), "shard_001.log": _line(1, 1)})
    assert "DIVERGENT" in first


def test_a_shard_that_died_reads_incomplete(tmp_path):
    first = _verdict(tmp_path, 3, {"shard_000.log": _line(2, 2), "shard_001.log": "Killed\n"})
    assert first.startswith("core certification INCOMPLETE: 2 of 3 replays")
