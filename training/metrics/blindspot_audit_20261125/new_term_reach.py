"""Does the new projection's term reach the served policy?
(docs/blindspot_audit_20261125.md, "tensor to decision").

Games are served in fp16 on the GPU; the new projection's weights are
small enough that its term is about one fp16 rounding step. On recorded
game-sides, for each decision's chosen action, on the CPU in float32:
- `with`: its log probability under the checkpoint as it is;
- `without`: the same with `unit_feat_proj2` set to zero.
The difference e = with - without is the new columns' effect on the
policy. The log probability the player logged (fp16 serving) is then
regressed on it: (logged - without) = slope * e. A slope of 1 says the
term reaches play in full, 0 that serving does not carry it.

    python new_term_reach.py --checkpoint round_0576/pi.pt --sides 8 <game.jsonl.gz> ...
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for p in (REPO, REPO / "tools", REPO / "tests"):
    sys.path.insert(0, str(p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("games", type=Path, nargs="+")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--sides", type=int, default=8, help="game-sides to replay")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    from helpers.pg_games import side_forwards
    from tools.pg_pass import load_network
    from tools.pg_positions import positions_of, read_record
    from wesnoth_ai import pg_loss
    cpu = torch.device("cpu")
    encoder, model = load_network(args.checkpoint, cpu)
    plain_encoder = copy.deepcopy(encoder)
    with torch.no_grad():
        plain_encoder.unit_feat_proj2.weight.zero_()
    vocab = (dict(encoder.unit_type_to_id), dict(encoder.faction_to_id))
    logged, with_, without = [], [], []
    done = 0
    for path in args.games:
        rec = read_record(path)
        pos = positions_of(rec, *vocab, game=path.name, unit_row_version=2)
        for side in (1, 2):
            if done >= args.sides:
                break
            rows = [p for p in pos.sides[side]]
            if not rows or any(p.shadow for p in rows):
                continue                      # a refused decision rolls the memory back: keep to plain sides
            with torch.no_grad():
                a = [float(pg_loss.action_log_probs(o, b)[0]) for o, b in side_forwards(encoder, model, rows)]
                z = [float(pg_loss.action_log_probs(o, b)[0]) for o, b in side_forwards(plain_encoder, model, rows)]
            logged += [p.logp for p in rows]
            with_ += a
            without += z
            done += 1
            print(f"{path.name[:44]} side {side}: {len(rows)} decisions", flush=True)
        if done >= args.sides:
            break
    logged, with_, without = (np.array(v, dtype=np.float64) for v in (logged, with_, without))
    e = with_ - without
    y = logged - without
    slope = float((e * y).sum() / (e * e).sum())
    resid = y - slope * e
    se = float(np.sqrt((resid ** 2).sum() / (len(e) - 1) / (e * e).sum()))
    out = {"decisions": int(len(e)), "game_sides": done,
           "rms_effect": float(np.sqrt((e ** 2).mean())), "mean_abs_effect": float(np.abs(e).mean()),
           "largest_effect": float(np.abs(e).max()),
           "rms_logged_minus_with": float(np.sqrt(((logged - with_) ** 2).mean())),
           "rms_logged_minus_without": float(np.sqrt((y ** 2).mean())),
           "slope": slope, "slope_se": se}
    print(json.dumps(out, indent=1))
    if args.out:
        args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
