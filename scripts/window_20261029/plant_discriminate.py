"""Where does the plant's miss come from? On a subset of the readout's
game-sides, in float32 on the CPU (no float16 anywhere):

  measured  the evaluator's teacher-forced change of the mean over the
            subset's game-sides of sum_t p_t(attack), (M(+x) - M(-x)) / 2,
            along d (the planted pass's reduced-AdamW direction) at x = s,
            s/4, s/16, and along P'u at h (the readout's finite difference)
            and at delta s (the plant's own part of the step);
  windowed  the first order x <grad M, dir> with grad M cut at 16-decision
            windows, as tools/pg_plant_grad.py reads it for the forecast;
  full      the same with the memory's gradient carried through each whole
            game-side (one window).

Nonlinearity at s shows as measured(s)/s differing from measured(s/16)/(s/16);
a windowing error as windowed differing from full; the evaluator's agreement
with the full gradient at small x checks both readers.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\momom2\Desktop\perso\projects\wesnoth_ai-stage1")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import numpy as np
import torch

from tools.pg_eval import EvalConfig, EvalContext
from tools.pg_plant_grad import MassGradients
from tools.pg_positions import RecordIndex
from wesnoth_ai.pg_loss import CLASSES
from wesnoth_ai.pg_records import read_labels
from wesnoth_ai.pg_step import dot, reduced_adamw, second_moment

RUNS = Path("C:/Users/momom2/wesnoth_runs")
SC = RUNS / "stage_c_local"
CK = ROOT / "training" / "checkpoints" / "own2.pt"
S, H = 9.338408159542613e-05, 1.1018425011285517e-06


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--skip-grads", action="store_true", help="the evaluator's readings alone")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fp16", action="store_true", help="the evaluator as the readout ran it (a GPU only)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    pp = torch.load(SC / "pass_planted.pt", map_location="cpu", weights_only=True)
    gr = torch.load(SC / "plant_grads.pt", map_location="cpu", weights_only=True)
    names = list(pp["names"])
    delta = float(gr["delta"])
    v = {n: t.float() for n, t in second_moment(pp).items()}
    d = reduced_adamw({n: pp["total"][n].float() for n in names}, v)
    pu = reduced_adamw({n: sum(p[n].float() for p in gr["u"]) for n in names}, v)
    games = sorted(gr["readout_games"])[:: max(1, len(gr["readout_games"]) // a.games)][: a.games]
    keep = set(games)
    rows = [r for r in read_labels(SC / "labels_planted.jsonl") if r.heldout and r.game in keep]
    index = RecordIndex(sorted(p for p in (RUNS / "selfplay_t1_local").glob("games_s*") if p.is_dir()))
    records = {g: index.path(g) for g in games}
    col = CLASSES.index(gr["class"])
    result = {"games": len(games), "class": gr["class"], "s": S, "h": H, "delta": delta}

    context = EvalContext(CK, records, rows, torch.device(a.device),
                          EvalConfig(workers=2, slots=gr["config"]["slots"], fp16=a.fp16))
    result["device"], result["fp16"] = a.device, a.fp16
    dev = {n: t.to(device=context.device, dtype=torch.float32) for n, t in d.items()}      # as NetworkEvaluator does
    pu_dev = {n: t.to(device=context.device, dtype=torch.float32) for n, t in pu.items()}
    sides = list(context.sides)
    result["game_sides"] = len(sides)

    def mean_mass(reading) -> float:
        return float(np.mean([np.nansum(reading.masses[gs.start:gs.stop, col]) for gs in sides]))

    steps_d = [S, -S, S / 4, -S / 4, S / 16, -S / 16]
    read_d = context.read(dev, [0.0, *steps_d], sides)
    m = {x: mean_mass(read_d[x]) for x in read_d}
    result["base"] = m[0.0]
    result["measured_d"] = {f"s/{k}": (m[S / k] - m[-S / k]) / 2 * k for k in (1, 4, 16)}   # in units of x = s
    result["even_d"] = {f"s/{k}": (m[S / k] + m[-S / k]) / 2 - m[0.0] for k in (1, 4, 16)}
    print(json.dumps(result, indent=1), f"{time.time() - t0:.0f} s", flush=True)
    read_u = context.read(pu_dev, [H, -H, delta * S, -delta * S], sides)
    mu = {x: mean_mass(read_u[x]) for x in read_u}
    result["measured_pu"] = {"fd_h (as s delta D_pu M)": (mu[H] - mu[-H]) / (2 * H) * S * delta,
                             "at delta s": (mu[delta * S] - mu[-delta * S]) / 2}
    context.close()
    print(json.dumps(result, indent=1), f"{time.time() - t0:.0f} s", flush=True)

    seat = {"all": [(gs.game, gs.side) for gs in sides]}
    cfg = gr["config"]
    for label, window in (() if a.skip_grads else (("windowed", cfg["window"]), ("full", 10 ** 7))):
        if label == "full" and a.skip_full:
            continue
        mg = MassGradients(CK, records, torch.device("cpu"), streams=cfg["streams"], window=window, seed=cfg["seed"],
                           slots=cfg["slots"], workers=2)
        mg.recompute = True                      # the whole game-side's graph by recomputation, on the CPU too
        g = mg.seat_gradients(seat, gr["class"])["all"]
        mg.close()
        g = {n: g[n].float() for n in names}
        result[label] = {"along d at s": S * dot(g, d), "along P'u (as s delta D_pu M)": S * delta * dot(g, pu)}
        print(json.dumps(result, indent=1), f"{time.time() - t0:.0f} s", flush=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
