#!/usr/bin/env python3
"""T-B: teacher-advantage probe (docs/eval_box.md queue, 2026-08-04).

For sampled human-corpus states, computes BOTH the raw value head's
E[V] and the 32-sim search's root value (visit-weighted mean child Q)
on the IDENTICAL state, with the game's known outcome as label. The
paired rows let the analyzer compare AUCs: if search adds <= +0.02
AUC over the raw head, there is nothing worth distilling into the
value head at the campaign's search budget (kill the value channel).

Search runs under the CAMPAIGN config (32 sims, tiebreak cap 0.3,
advice per checkpoint) because the question is what the TRAINING
teacher would provide -- not the eval contract's pure-strength view.

Chunk-friendly: --skip-games/--games partition the corpus index and
--out appends one JSON line per state, so shards can run in parallel
and be pooled by the analyzer:

    python tools/probe_teacher_advantage.py CKPT --skip-games 0 \
        --games 50 --out tb_part0.jsonl
    # pool shards with a paired-AUC comparison over the jsonl rows
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--dataset-dir", type=Path,
                    default=Path("replays_dataset"))
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--skip-games", type=int, default=0)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv[1:])

    import torch
    torch.set_num_threads(2)
    from wesnoth_ai.transformer_policy import TransformerPolicy
    from tools.mcts import MCTSConfig, mcts_search
    from tools.draw_tiebreak import DrawTiebreakConfig
    from tools.replay_dataset import (_apply_command,
                                      _build_initial_gamestate,
                                      _setup_scenario_events)
    from tools.value_corpus import _DECISION_KINDS
    from tools.midgame_starts import _load_index as load_index
    from tools.wesnoth_sim import WesnothSim

    raw = torch.load(args.checkpoint, map_location="cpu",
                     weights_only=False)
    a = raw["arch"]
    pol = TransformerPolicy(
        device=torch.device("cpu"), d_model=a["d_model"],
        num_layers=a["num_layers"], num_heads=a["num_heads"],
        d_ff=a["d_ff"],
        aux_score=bool(raw.get("aux_score")),
        moves_left=bool(raw.get("moves_left")),
        advice=bool(raw.get("advice")),
        relevant_set_hexes=bool(raw.get("relevant_set_hexes")))
    pol.load_checkpoint(args.checkpoint)
    model, enc = pol._inference_model, pol._inference_encoder
    model.eval()
    atoms = pol._model._value_atoms
    cfg = MCTSConfig(n_simulations=args.sims,
                     advice=bool(raw.get("advice")),
                     draw_tiebreak=DrawTiebreakConfig(cap=0.3))

    index = load_index(args.dataset_dir)
    rng = random.Random(args.seed)
    rng.shuffle(index)
    sample = index[args.skip_games:args.skip_games + args.games]
    rng_np = __import__("numpy").random.default_rng(args.seed)

    n_states = 0
    with args.out.open("a", encoding="utf-8") as fout:
        for r in sample:
            try:
                with gzip.open(args.dataset_dir / r["file"], "rt",
                               encoding="utf-8") as f:
                    data = json.load(f)
                gs = _build_initial_gamestate(data)
                sid = data.get("scenario_id", "")
                _setup_scenario_events(gs, sid)
            except Exception:                    # noqa: BLE001
                continue
            offset = rng.randrange(max(1, args.stride))
            k = 0
            states = []
            for cmd in data.get("commands", []):
                kind = cmd[0] if cmd else "?"
                if kind in _DECISION_KINDS:
                    side = gs.global_info.current_side
                    if side in (1, 2) and k % args.stride == offset:
                        states.append((copy.deepcopy(gs), side,
                                       gs.global_info.turn_number))
                    k += 1
                try:
                    _apply_command(gs, cmd)
                except Exception:                # noqa: BLE001
                    break
            for st, side, turn in states:
                z = +1.0 if side == r["winner"] else -1.0
                try:
                    enc.register_names(st)
                    import torch as _t
                    with _t.no_grad():
                        out = model(enc.encode(st))
                        ev = float((_t.softmax(
                            out.value_logits.squeeze(), -1)
                            * atoms).sum())
                    sim = WesnothSim(copy.deepcopy(st),
                                     scenario_id=sid,
                                     max_turns=turn + 40)
                    root = mcts_search(sim, model, enc, cfg,
                                       rng=rng_np)
                    tv = sum(e.n_visits for e in root.edges)
                    sv = (sum(e.n_visits * e.q_value
                              for e in root.edges) / tv
                          if tv else float(root.value))
                except Exception as e:           # noqa: BLE001
                    fout.write(json.dumps(
                        {"file": r["file"], "turn": turn,
                         "err": type(e).__name__}) + "\n")
                    continue
                fout.write(json.dumps(
                    {"file": r["file"], "turn": turn, "side": side,
                     "z": z, "ev_raw": round(ev, 4),
                     "ev_search": round(sv, 4)}) + "\n")
                n_states += 1
            fout.flush()
    print(f"T-B probe: wrote {n_states} paired states -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
