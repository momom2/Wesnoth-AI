"""Annotate an existing imitation corpus with each side's fog and shroud
settings, without re-extracting the raw replays (they live only on the
laptop). The values come from a table parsed from the raw replays'
[side] blocks (training/metrics/value_head/corpus_fog_shroud.json,
2026-09-06); tools/build_imitation_dataset.py records the same fields
at extraction time for future builds.

Per game: `starting_sides[i]` gets `fog` and `shroud` (booleans), the
manifest row gets `fog` (the encoder's switch, tools/replay_dataset
.fog_on_for) and `shroud`; games the corpus rule quarantines (fog off
with shroud on) leave the manifest and the value index and are listed
in quarantined.jsonl. Files are rewritten atomically; a second run is
a no-op.

Usage:
    python tools/annotate_corpus_fog.py --dataset replays_dataset_imitation \\
        --table training/metrics/value_head/corpus_fog_shroud.json
"""
import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.replay_dataset import fog_on_for, quarantine_reason  # noqa: E402


def _flag(v, default: bool) -> bool:
    if v is None:
        return default
    return str(v).lower() in ("yes", "true", "1")


def side_flags(entry: dict):
    """{side number: (fog, shroud)} from one table row."""
    out = {}
    for s, d in (entry.get("sides") or {}).items():
        out[int(s)] = (_flag(d.get("fog"), True), _flag(d.get("shroud"), False))
    return out


def annotate_file(path: Path, flags) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    changed = False
    for s in data.get("starting_sides", []):
        fog, shroud = flags.get(int(s.get("side", 0)), (True, False))
        if s.get("fog") != fog or s.get("shroud") != shroud:
            s["fog"], s["shroud"] = fog, shroud
            changed = True
    if changed:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    return data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", type=Path, default=ROOT / "replays_dataset_imitation")
    ap.add_argument("--table", type=Path,
                    default=ROOT / "training/metrics/value_head/corpus_fog_shroud.json")
    args = ap.parse_args(argv)
    table = {r["file"]: r for r in json.loads(args.table.read_text(encoding="utf-8"))
             if "error" not in r}
    manifest_path = args.dataset / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    kept, quarantined, missing = [], [], 0
    t0 = time.time()
    for i, row in enumerate(rows):
        entry = table.get(row["file"])
        if entry is None:
            missing += 1
            kept.append(row)
            continue
        data = annotate_file(args.dataset / row["file"], side_flags(entry))
        sides = data.get("starting_sides", [])
        why = quarantine_reason(sides)
        if why is not None:
            quarantined.append(dict(row, quarantined=why))
            continue
        row["fog"] = fog_on_for(sides)
        row["shroud"] = any(bool(s.get("shroud", False)) for s in sides)
        kept.append(row)
        if (i + 1) % 2000 == 0:
            print(f"{i + 1}/{len(rows)} {time.time() - t0:.0f} s", flush=True)
    # The quarantined games leave the directory too (the trainer used
    # to scan it; it now keeps to the manifest, but a stale copy of
    # the corpus should not carry them either).
    qdir = args.dataset.parent / (args.dataset.name + "_quarantined")
    for r in quarantined:
        src = args.dataset / r["file"]
        if src.exists():
            qdir.mkdir(parents=True, exist_ok=True)
            os.replace(src, qdir / r["file"])
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
    os.replace(tmp, manifest_path)
    (args.dataset / "quarantined.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in quarantined), encoding="utf-8")
    index_path = args.dataset / "value_corpus_index.jsonl"
    if index_path.exists():
        gone = {r["file"] for r in quarantined}
        lines = [line for line in index_path.read_text(encoding="utf-8").splitlines()
                 if line.strip() and json.loads(line)["file"] not in gone]
        index_path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    n_off = sum(1 for r in kept if r.get("fog") is False)
    print(f"annotated {len(kept)} games (fog off {n_off}, no table entry {missing}), "
          f"quarantined {len(quarantined)} in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
