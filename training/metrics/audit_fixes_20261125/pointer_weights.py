"""The weapon pointer's weights at a round (docs/audit_fixes_prereg_20261125.md,
measurement 3: "the keys' and the queries' root mean square").

The keys' map (`weapon_k_proj.weight`) started at zero at the switch, so its
root mean square is how far it has moved. The queries' map
(`weapon_q_proj`) started from nn.Linear's law, so what is read of it is its
move from the switch's checkpoint, the first one given.

    python -I -X utf8 pointer_weights.py --lr 1e-5 --out pointer.json <switch pi.pt> <round pi.pt> [...]
"""
import argparse
import json
from pathlib import Path

import torch

KEYS = "weapon_k_proj.weight"
QUERIES = ("weapon_q_proj.weight", "weapon_q_proj.bias")


def rms(t: torch.Tensor) -> float:
    return float(t.double().pow(2).mean().sqrt())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    base = torch.load(args.checkpoints[0], map_location="cpu", weights_only=True)["model_state"]
    rows = []
    for path in args.checkpoints:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        state = ck["model_state"]
        keys = state[KEYS]
        row = {
            "checkpoint": str(path),
            "weapon_target": int(ck.get("weapon_target", 0)),
            "keys_rms": rms(keys),
            "keys_rms_in_learning_rates": rms(keys) / args.lr,
            "keys_share_nonzero": float((keys != 0).double().mean()),
        }
        for name in QUERIES:
            moved = state[name] - base[name]
            row[name] = {"rms": rms(state[name]), "moved_rms": rms(moved),
                         "moved_rms_in_learning_rates": rms(moved) / args.lr}
        rows.append(row)
    text = json.dumps({"lr": args.lr, "switch_checkpoint": str(args.checkpoints[0]), "rounds": rows}, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
