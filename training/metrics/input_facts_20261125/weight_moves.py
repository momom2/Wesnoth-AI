"""How far the network's weights moved between checkpoints
(docs/input_facts_prereg_20261125.md, measurement 3's yardstick).

Reads checkpoints, changes nothing. Between the checkpoint named first and
each later one, and between each and the next: the root mean square of the
change of every floating tensor the two share (the whole network), and of
a few named ones. A tensor one of the two lacks is left out of the whole
and reported alone, against zero: so the new projection of unit row
version 2 reads as its own size.

    python -I -X utf8 weight_moves.py start.pt round_0016/pi.pt ... round_0560/pi.pt
"""
import argparse
import json
from pathlib import Path

import torch

STATES = ("model_state", "encoder_state")
NAMED = ("encoder_state.unit_feat_proj.weight", "encoder_state.unit_feat_proj2.weight",
         "model_state.value_head.0.weight", "model_state.weapon_head.0.weight")


def tensors(path: Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return {f"{state}.{name}": t.double() for state in STATES for name, t in ck[state].items()
            if isinstance(t, torch.Tensor) and torch.is_floating_point(t)}


def rms(t: torch.Tensor) -> float:
    return float(t.pow(2).mean().sqrt())


def moves(a: dict, b: dict) -> dict:
    """The change from `a` to `b`: the shared tensors pooled, then the named ones."""
    count = 0
    squares = 0.0
    for name in a.keys() & b.keys():
        if a[name].shape == b[name].shape:
            d = b[name] - a[name]
            count += d.numel()
            squares += float(d.pow(2).sum())
    out = {"weights": count, "rms_move": (squares / count) ** 0.5}
    for name in NAMED:
        if name in b:
            d = b[name] - a[name] if name in a else b[name]
            out[name] = rms(d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoints", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    torch.set_num_threads(4)
    rows = []
    first = prev = None
    for path in args.checkpoints:
        cur = tensors(path)
        if first is None:
            first = cur
        else:
            row = {"file": str(path), "from_first": moves(first, cur), "from_previous": moves(prev, cur)}
            rows.append(row)
            for label in ("from_first", "from_previous"):
                m = row[label]
                named = ", ".join(f"{k.split('.', 1)[1]} {m[k]:.3g}" for k in NAMED if k in m)
                print(f"{path}: {label.replace('_', ' ')}: whole network {m['rms_move']:.3g} ({named})")
        prev = cur
    if args.out:
        args.out.write_text(json.dumps({"first": str(args.checkpoints[0]), "rows": rows}, indent=1) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
