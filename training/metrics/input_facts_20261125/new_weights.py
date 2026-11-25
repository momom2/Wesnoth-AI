"""The size of the weights that read the unit row's new columns
(docs/input_facts_prereg_20261125.md, measurement 3).

Reads checkpoints, changes nothing. For each: the root mean square and the
largest absolute value of `unit_feat_proj2` (the 232 columns of unit row
version 2), by block of columns, beside the old projection's root mean
square; and, against the checkpoint named first, how far each weight moved
in units of `--lr` rounds (1.0 means every weight moved by the learning
rate every round, Adam's most for a gradient that keeps its sign).

    python -I -X utf8 new_weights.py --lr 1e-5 round_0544/pi.pt round_0560/pi.pt ...
"""
import argparse
import json
import re
from pathlib import Path

import torch

OLD = "unit_feat_proj.weight"
NEW = "unit_feat_proj2.weight"
# The added columns' blocks (wesnoth_ai/encoder.py, "The unit row, version 2").
BLOCKS = (("level", 0, 1), ("defence where it stands", 1, 2), ("defence by terrain", 2, 18),
          ("movement by terrain", 18, 34), ("fights", 34, 232))


def rms(t: torch.Tensor) -> float:
    return float(t.double().pow(2).mean().sqrt())


def step_of(path: Path, ck: dict) -> int:
    m = re.search(r"round_(\d+)", str(path))
    return int(m.group(1)) if m else int((ck.get("pg_step") or {}).get("step", -1))


def read(path: Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    state = ck["encoder_state"]
    out = {"file": str(path), "round": step_of(path, ck), "unit_row_version": int(ck.get("unit_row_version", 1)),
           "old_rms": rms(state[OLD])}
    if NEW in state:
        w = state[NEW]
        out.update(new_rms=rms(w), new_max=float(w.abs().max()), nonzero_share=float((w != 0).double().mean()),
                   blocks={name: {"rms": rms(w[:, a:b]), "max": float(w[:, a:b].abs().max())}
                           for name, a, b in BLOCKS})
        out["_w"] = w
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoints", type=Path, nargs="+")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = [read(p) for p in args.checkpoints]
    first = rows[0]
    for r in rows:
        if "_w" in r and "_w" in first and r["round"] > first["round"]:
            rounds = r["round"] - first["round"]
            moved = (r["_w"] - first["_w"]).abs().double()
            r["rounds_since_first"] = rounds
            r["mean_move_per_round_in_lr"] = float(moved.mean()) / (rounds * args.lr)
    for r in rows:
        r.pop("_w", None)
        line = f"round {r['round']}: old projection rms {r['old_rms']:.4f}"
        if "new_rms" in r:
            line += (f"; new projection rms {r['new_rms']:.3g}, largest {r['new_max']:.3g}, "
                     f"{100 * r['nonzero_share']:.0f}% of its weights moved")
            if "mean_move_per_round_in_lr" in r:
                line += f"; mean move a round {r['mean_move_per_round_in_lr']:.2f} of the learning rate"
        print(line)
        for name, b in (r.get("blocks") or {}).items():
            print(f"    {name}: rms {b['rms']:.3g}, largest {b['max']:.3g}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
