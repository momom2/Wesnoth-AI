"""Warm-start divergence gate for a net2net grow (plan §3.5 / §10).

Answers ONE question before a campaign is launched on a grown net:
**do the transferred weights still compute the same function?** The
weights always load — that is not evidence. Net2Net is only
*approximately* function-preserving through LayerNorm + MHA, so the
grown net can load cleanly and evaluate the board completely
differently.

The project has two measured reference points for the headline number
(value MAE on the ±1 value scale):

  0.017  ACCEPTED — the 128->256 width grow that seeded Tier-a
                    (`docs/superhuman_training_plan.md` §10).
  0.217  REJECTED — the relevant-set encoder switch, T2-C
                    (`docs/autonomous_run.md` cycle 24). "The weights
                    load; the FUNCTION does not carry over."

So: <~0.02 is a drop-in warm start, ~0.2 means the grown net needs a
fine-tune leg before any cross-arch comparison means anything.

HEAD ALIGNMENT — the reason this tool takes several `--arch` candidates.
`nn.MultiheadAttention` packs head h at rows [h*d_head, (h+1)*d_head)
inside each of the stacked Q/K/V blocks, and `net2net._transfer_param`
copies a *leading block* per Q/K/V — it does not know about heads. That
copy is head-aligned only when **d_head is unchanged** (d_model /
num_heads constant). The accepted 0.017 precedent grew 128/4 -> 256/8,
i.e. d_head 32 -> 32, so it was head-aligned by luck of the numbers. A
grow that changes d_head shears each old head across two new ones.
This tool measures that rather than assuming it — run both variants.

Usage:
    python tools/measure_warm_start.py \\
        --source training/checkpoints/campaign_live_20260730.pt \\
        --arch 384,8,12,1536 --arch 384,8,8,1536 \\
        --states 200 --save-json warm_start.json

`--arch` is `d_model,num_layers,num_heads,d_ff` and repeats; every
candidate is scored on the SAME collected states, so the numbers are
directly comparable.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import torch

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

log = logging.getLogger("measure_warm_start")

# Acceptance reference points, both MEASURED on this project (see the
# module docstring for citations). Reported alongside every result so a
# number is never read without its scale.
MAE_ACCEPTED_PRECEDENT = 0.017
MAE_REJECTED_PRECEDENT = 0.217


# ---------------------------------------------------------------------
# State collection
# ---------------------------------------------------------------------

def collect_states(
    policy, *, n_states: int, seed: int, stride: int, max_turns: int,
    mini_every: int, max_decisions: int = 0,
) -> List:
    """Play games with `policy` and snapshot `sim.gs` every `stride`
    decisions until `n_states` states are held.

    States come from the policy's OWN play so they sit on the
    distribution the grown net will actually meet — a warm start that
    holds on random boards but not on the policy's own trajectory is
    not a warm start. Scenarios alternate mini / ladder (`mini_every`)
    so the sample is not one map's geometry.

    `max_decisions` > 0 abandons a game after that many decisions and
    starts a fresh one. Sampling shallowly across MANY games beats
    playing few games to the end: it bounds runtime (a ladder game to
    the turn cap is thousands of CPU-seconds here) and spreads the
    sample over more openings, factions and maps. It does bias the
    sample toward early/midgame states — stated because it matters for
    reading the result, not hidden.
    """
    from tools.scenario_pool import (
        build_scenario_gamestate, load_factions, random_setup,
    )
    from tools.wesnoth_sim import WesnothSim

    load_factions()
    states: List = []
    game_i = 0
    while len(states) < n_states:
        mini = (mini_every > 0) and (game_i % mini_every == 0)
        setup = random_setup(random.Random(seed + game_i), forced_faction=None,
                             mini_maps=mini)
        sim = WesnothSim(build_scenario_gamestate(setup),
                         scenario_id=setup.scenario_id, max_turns=max_turns)
        label = f"warmstart_g{game_i}"
        decisions = 0
        while (not sim.done and len(states) < n_states
               and (max_decisions <= 0 or decisions < max_decisions)):
            # Stable snapshot for select_action — the same load-bearing
            # deepcopy documented in sim_self_play.play_one_game.
            pre_state = copy.deepcopy(sim.gs)
            if decisions % stride == 0:
                states.append(copy.deepcopy(pre_state))
            action = policy.select_action(pre_state, game_label=label, sim=sim)
            sim.step(action)
            decisions += 1
        # Drop this game's pending transitions so the queue does not grow.
        try:
            policy.finalize_game(label, winner=sim.winner)
        except Exception:                      # pragma: no cover - telemetry
            log.debug("finalize_game failed for %s (ignored)", label)
        game_i += 1
        log.info("collected %d/%d states (%d games)",
                 len(states), n_states, game_i)
    return states


# ---------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------

def _kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(P || Q) in nats over the last axis, flattened and averaged."""
    p = torch.log_softmax(p_logits.reshape(-1, p_logits.shape[-1]), dim=-1)
    q = torch.log_softmax(q_logits.reshape(-1, q_logits.shape[-1]), dim=-1)
    return float((p.exp() * (p - q)).sum(dim=-1).mean())


def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def compare(src_pol, dst_pol, states: List) -> Dict:
    """Score `dst_pol` against `src_pol` on shared states.

    Both encoders are aligned to the SOURCE's vocab and frozen, so a
    divergence can only come from the weights — never from an id
    landing in a different embedding row.
    """
    dst_pol._inference_encoder.unit_type_to_id = dict(
        src_pol._inference_encoder.unit_type_to_id)
    dst_pol._inference_encoder.faction_to_id = dict(
        src_pol._inference_encoder.faction_to_id)
    src_pol._inference_encoder.freeze_vocab()
    dst_pol._inference_encoder.freeze_vocab()

    v_abs: List[float] = []
    aux_abs: List[float] = []
    c51_kl: List[float] = []
    actor_kl: List[float] = []

    with torch.no_grad():
        for gs in states:
            o_s = src_pol._inference_model(src_pol._inference_encoder.encode(gs))
            o_d = dst_pol._inference_model(dst_pol._inference_encoder.encode(gs))
            v_abs.append(float((o_s.value - o_d.value).abs().mean()))
            c51_kl.append(_kl(o_s.value_logits, o_d.value_logits))
            actor_kl.append(_kl(o_s.actor_logits, o_d.actor_logits))
            a_s = getattr(o_s, "aux_score", None)
            a_d = getattr(o_d, "aux_score", None)
            if a_s is not None and a_d is not None:
                aux_abs.append(float((a_s - a_d).abs().mean()))

    n = len(v_abs)
    out = {
        "n_states": n,
        "value_mae": sum(v_abs) / n if n else float("nan"),
        "value_p90": _quantile(v_abs, 0.90),
        "value_max": max(v_abs) if v_abs else float("nan"),
        "c51_kl_nats": sum(c51_kl) / n if n else float("nan"),
        "actor_kl_nats": sum(actor_kl) / n if n else float("nan"),
    }
    if aux_abs:
        out["aux_mae"] = sum(aux_abs) / len(aux_abs)
        out["aux_max"] = max(aux_abs)
    out["verdict"] = (
        "DROP-IN" if out["value_mae"] <= 2 * MAE_ACCEPTED_PRECEDENT else
        "NEEDS-FINETUNE" if out["value_mae"] < MAE_REJECTED_PRECEDENT else
        "NOT-A-WARM-START")
    return out


def _parse_arch(spec: str) -> Dict[str, int]:
    parts = spec.split(",")
    if len(parts) != 4:
        raise SystemExit(
            f"--arch must be 'd_model,num_layers,num_heads,d_ff'; got {spec!r}")
    try:
        d_model, num_layers, num_heads, d_ff = (int(p) for p in parts)
    except ValueError:
        raise SystemExit(f"--arch values must be integers; got {spec!r}")
    if d_model % num_heads:
        raise SystemExit(
            f"--arch {spec}: num_heads ({num_heads}) must divide "
            f"d_model ({d_model})")
    return {"d_model": d_model, "num_layers": num_layers,
            "num_heads": num_heads, "d_ff": d_ff}


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True,
                    help="Trained checkpoint to grow FROM.")
    ap.add_argument("--dest", type=Path, default=None,
                    help="Compare --source against THIS checkpoint "
                         "directly (arch+flags read from it) instead "
                         "of net2net-growing --arch candidates. The "
                         "T2 re-measure mode (2026-08-05): the 0.217 "
                         "record predates the cdf263a encode-path "
                         "fix and needs re-derivation on a clean "
                         "instrument.")
    ap.add_argument("--arch", action="append", required=False,
                    help="Candidate 'd_model,num_layers,num_heads,d_ff'. "
                         "Repeatable; all scored on the same states.")
    ap.add_argument("--states", type=int, default=200)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--stride", type=int, default=17,
                    help="Snapshot every Nth decision (prime by default "
                         "so snapshots don't phase-lock to turn structure).")
    ap.add_argument("--max-turns", type=int, default=24)
    ap.add_argument("--mini-every", type=int, default=3,
                    help="Every Nth collection game uses a mini map "
                         "(0 = ladder only).")
    ap.add_argument("--max-decisions", type=int, default=60,
                    help="Abandon a collection game after N decisions "
                         "(0 = play to the end). Spreads the sample over "
                         "more games and bounds runtime.")
    ap.add_argument("--save-json", type=Path, default=None)
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING"])
    args = ap.parse_args(argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S")

    if not args.source.exists():
        log.error("source checkpoint not found: %s", args.source)
        return 1

    from wesnoth_ai.transformer_policy import TransformerPolicy
    from tools.net2net import grow_checkpoint

    raw = torch.load(args.source, map_location="cpu", weights_only=False)
    src_arch = raw.get("arch", {}) or {}
    aux = bool(raw.get("aux_score", False))
    # Optional heads must be reproduced or the "source" being measured is
    # a stripped model -- load_checkpoint only WARNS about unexpected keys.
    flags = {k: bool(raw.get(k, False))
             for k in ("moves_left", "advice", "relevant_set_hexes")}
    log.info("source %s: arch=%s aux_score=%s %s decision_step=%s",
             args.source.name, src_arch, aux, flags, raw.get("decision_step"))

    src_pol = TransformerPolicy(device=torch.device("cpu"),
                                aux_score=aux, **flags, **src_arch)
    src_pol.load_checkpoint(args.source)

    states = collect_states(
        src_pol, n_states=args.states, seed=args.seed, stride=args.stride,
        max_turns=args.max_turns, mini_every=args.mini_every,
        max_decisions=args.max_decisions)

    if not args.dest and not args.arch:
        ap.error("pass --arch (grow mode) or --dest (two-checkpoint)")

    results = []
    if args.dest:
        # Two-checkpoint mode: no grow; flags/arch from the dest raw.
        draw = torch.load(args.dest, map_location="cpu",
                          weights_only=False)
        darch = draw.get("arch", {}) or {}
        dflags = {k: bool(draw.get(k, False))
                  for k in ("moves_left", "advice",
                            "relevant_set_hexes")}
        dst_pol = TransformerPolicy(
            device=torch.device("cpu"),
            aux_score=bool(draw.get("aux_score", False)),
            **dflags, **darch)
        dst_pol.load_checkpoint(args.dest)
        row = {"arch": darch,
               "params_m": round(sum(
                   p.numel() for p in dst_pol._model.parameters())
                   / 1e6, 2),
               "d_head_src": (src_arch.get("d_model", 0)
                              // max(1, src_arch.get("num_heads", 1))),
               "d_head_dst": (darch.get("d_model", 0)
                              // max(1, darch.get("num_heads", 1)))}
        row["head_aligned"] = row["d_head_src"] == row["d_head_dst"]
        row.update(compare(src_pol, dst_pol, states))
        results.append(row)
        log.info("%s vs %s | value MAE %.4f (p90 %.4f max %.4f) | "
                 "C51 KL %.4f | actor KL %.4f | %s",
                 args.source.name, args.dest.name, row["value_mae"],
                 row["value_p90"], row["value_max"],
                 row["c51_kl_nats"], row["actor_kl_nats"],
                 row["verdict"])

    with tempfile.TemporaryDirectory() as td:
        for spec in (args.arch or []):
            arch = _parse_arch(spec)
            d_head_src = (src_arch.get("d_model", 0)
                          // max(1, src_arch.get("num_heads", 1)))
            d_head_dst = arch["d_model"] // arch["num_heads"]
            grown = Path(td) / f"grown_{spec.replace(',', '_')}.pt"
            grow_checkpoint(args.source, grown, **arch)   # carries the flags
            dst_pol = TransformerPolicy(device=torch.device("cpu"),
                                        aux_score=aux, **flags, **arch)
            dst_pol.load_checkpoint(grown)
            n_params = sum(p.numel() for p in dst_pol._model.parameters())
            row = {
                "arch": arch,
                "params_m": round(n_params / 1e6, 2),
                "d_head_src": d_head_src,
                "d_head_dst": d_head_dst,
                "head_aligned": d_head_src == d_head_dst,
            }
            row.update(compare(src_pol, dst_pol, states))
            results.append(row)
            log.info(
                "%s -> %.2fM params | d_head %d->%d (%s) | value MAE %.4f "
                "(p90 %.4f max %.4f) | C51 KL %.4f | actor KL %.4f | %s",
                spec, row["params_m"], d_head_src, d_head_dst,
                "aligned" if row["head_aligned"] else "SHEARED",
                row["value_mae"], row["value_p90"], row["value_max"],
                row["c51_kl_nats"], row["actor_kl_nats"], row["verdict"])

    print()
    print(f"warm-start divergence vs {args.source.name} "
          f"({results[0]['n_states']} shared states)")
    print(f"  reference: {MAE_ACCEPTED_PRECEDENT} accepted (Tier-a grow), "
          f"{MAE_REJECTED_PRECEDENT} rejected (T2-C relevant-set)")
    print(f"  {'arch':<22} {'params':>8} {'d_head':>8} {'valMAE':>8} "
          f"{'p90':>7} {'C51KL':>7} {'actKL':>7}  verdict")
    for r in results:
        a = r["arch"]
        name = (f"{a['d_model']}/L{a['num_layers']}/"
                f"H{a['num_heads']}/ff{a['d_ff']}")
        head = (f"{r['d_head_src']}->{r['d_head_dst']}"
                + ("" if r["head_aligned"] else "!"))
        print(f"  {name:<22} {r['params_m']:>7.2f}M {head:>8} "
              f"{r['value_mae']:>8.4f} {r['value_p90']:>7.4f} "
              f"{r['c51_kl_nats']:>7.4f} {r['actor_kl_nats']:>7.4f}  "
              f"{r['verdict']}")

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(
            {"source": str(args.source), "src_arch": src_arch,
             "n_states": results[0]["n_states"], "results": results},
            indent=2))
        log.info("wrote %s", args.save_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
