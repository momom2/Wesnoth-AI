"""A characterization pin on what the extractor reads out of real
replays, so the reconstruction half of the WML dedup cannot change a
game without saying so.

`tools/diff_replay.py` over the corpus is the project's proof that
reconstruction is faithful, but it needs a box and ~20 minutes. This
is the cheap local half: it fingerprints the record
`replay_extract.extract_replay` produces for a fixed sample of corpus
replays, which is the layer the shared WML reader touches. A change
there moves a fingerprint here long before it would move a replay
sweep.

The sample is fixed by an evenly strided walk of the manifest, so it
spans maps, eras and host settings without storing a list. The
fingerprints live in `tests/data/replay_extract_snapshot.json`;
regenerate deliberately with

    python tests/test_replay_extract_snapshot.py --update

and say why in the commit message. Skipped where the corpus is absent,
which is every machine that has not staged it.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent
DATASET = ROOT / "replays_dataset_imitation"
SNAPSHOT = Path(__file__).parent / "data" / "replay_extract_snapshot.json"
SAMPLE = 120            # every (corpus / SAMPLE)-th manifest row


def sample_sources():
    """(game file, raw replay path) for the fixed sample, in manifest
    order. Empty when the corpus is not staged."""
    manifest = DATASET / "manifest.jsonl"
    if not manifest.exists():
        return []
    rows = [json.loads(line) for line in open(manifest, encoding="utf-8")]
    stride = max(1, len(rows) // SAMPLE)
    out = []
    for row in rows[::stride][:SAMPLE]:
        src = ROOT / row["source"].replace("\\", "/")
        if src.exists():
            out.append((row["file"], src))
    return out


def fingerprint(path: Path) -> str:
    """A digest of the whole extracted record, commands included. Any
    field the extractor reads differently moves it."""
    from tools.replay_extract import extract_replay

    record = extract_replay(path)
    if record is None:
        return "none"
    return hashlib.sha1(
        json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _require_corpus():
    sources = sample_sources()
    if not sources:
        pytest.skip("the imitation corpus is not staged on this machine")
    if not SNAPSHOT.exists():
        pytest.skip("no snapshot recorded; run this file with --update")
    return sources


def test_the_extracted_records_match_the_snapshot():
    sources = _require_corpus()
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    changed = {}
    for name, path in sources:
        if name not in expected:
            continue
        got = fingerprint(path)
        if got != expected[name]:
            changed[name] = (expected[name], got)
    assert changed == {}, (
        f"{len(changed)} of {len(sources)} extracted records changed, e.g. "
        f"{list(changed)[:3]}. If that is the point of the change, "
        f"regenerate the snapshot and say why in the commit message.")


def test_the_sample_still_resolves():
    """A snapshot whose replays have moved would pass vacuously."""
    sources = _require_corpus()
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    covered = [name for name, _ in sources if name in expected]
    assert len(covered) >= 0.9 * len(expected), (
        f"only {len(covered)} of {len(expected)} snapshotted replays are in "
        f"the sample; the manifest or the stride changed")


def main() -> int:
    sources = sample_sources()
    if not sources:
        print("the imitation corpus is not staged; nothing to record")
        return 1
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    data = {name: fingerprint(path) for name, path in sources}
    SNAPSHOT.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {SNAPSHOT} ({len(data)} replays)")
    return 0


if __name__ == "__main__":
    sys.exit(main() if "--update" in sys.argv else pytest.main([__file__, "-q"]))
