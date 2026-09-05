"""Wall time per playout of turn_gap runs over the same positions
(box chain32, 2026-09-05: per-process workers against the shared
inference server at 12 and 24 workers). Per-position seconds are
recorded in every result file; playouts per position come from the
candidates' outcome lists (terminal candidates play none).

Usage:
    python tools/analysis/turn_gap_timing.py REF.json OTHER.json [OTHER2.json ...]
"""
import json
import sys
from pathlib import Path


def per_position(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for r in data["positions"]:
        cands = [r["base"]] + r["alternatives"]
        played = sum(0 if c.get("terminal_in_turn") else len(c["outcomes"]) for c in cands)
        out[r["index"]] = (r["secs"], played)
    prov = data.get("provenance", {})
    wall = (data.get("summary") or {}).get("wall_secs")
    server = (data.get("summary") or {}).get("inference_server")
    return out, prov, wall, server


def main(argv):
    files = [Path(a) for a in argv[1:]]
    runs = [per_position(f) for f in files]
    common = set.intersection(*(set(r[0]) for r in runs))
    print(f"{len(common)} positions in common: {sorted(common)}")
    print("| file | jobs | shared | positions | sum secs | playouts | s/playout | "
          "wall s (whole run) | playouts/s (whole run) | server mean batch |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    ref = None
    for f, (pos, prov, wall, server) in zip(files, runs):
        secs = sum(pos[i][0] for i in common)
        played = sum(pos[i][1] for i in common)
        spp = secs / max(1, played)
        ref = ref or spp
        mb = f"{server['mean_batch']:.2f}" if server else "-"
        all_played = sum(v[1] for v in pos.values())
        rate = f"{all_played / wall:.2f}" if wall else "-"
        print(f"| {f.name} | {prov.get('jobs')} | {prov.get('shared_inference')} | "
              f"{len(common)} | {secs:.0f} | {played} | {spp:.2f} ({ref / spp:.2f}x) | "
              f"{wall if wall is None else round(wall)} | {rate} | {mb} |")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
