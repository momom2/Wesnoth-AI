"""Does the detector advice signal produce a SIGNIFICANT amount of gradient?

The right success metric for a training SIGNAL is not Elo (far too noisy and
downstream) but whether the path it feeds actually receives learning signal.
Precedent in this repo: the moves-left head is kept as telemetry at "~0.03%
of the gradient" -- this tool reports the same idiom for the advice path.

Two numbers decide it:

  1. FIRE RATE -- on what fraction of real DECISION states does the
     prospective advisor produce >=1 opportunity? If advice is present on
     ~0% of decisions, per-sample gradient magnitude is irrelevant.
  2. GRADIENT SHARE -- ||g_advice|| / ||g_total|| under the REAL training
     loss (`_mcts_factored_policy_loss` + the categorical value loss), on
     states that actually carry advice.

It also tracks the BOOTSTRAP: at the zero-init graft only `advice_out` gets
a non-zero gradient (everything upstream flows through it and it is 0), so
the tool takes optimizer steps and reports whether ||advice_out|| grows off
zero and whether the rest of the path then starts receiving gradient.

Usage:
  python -m tools.measure_advice_gradient --checkpoint training/checkpoints/tier_a_campaign_final.pt
  python -m tools.measure_advice_gradient --fresh --states 400 --grad-states 12 --steps 3
"""
from __future__ import annotations

import argparse
import copy
import glob
import sys
from pathlib import Path

import torch

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from tools.detector_advisor import (                               # noqa: E402
    prospective_opportunities, opportunities_to_features,
)
from tools.swap_detector import load_side_turns                    # noqa: E402
from tools.replay_dataset import _apply_command                    # noqa: E402
from wesnoth_ai.action_sampler import (                            # noqa: E402
    enumerate_legal_actions_with_priors,
)
from wesnoth_ai.trainer import (                                   # noqa: E402
    _mcts_factored_policy_loss, _categorical_value_loss,
)
from wesnoth_ai.transformer_policy import TransformerPolicy        # noqa: E402


def decision_states(bundle_paths, limit_states, min_turn=4):
    """Real DECISION states: walk each recorded side-turn and yield the state
    BEFORE each command -- exactly the states the policy faces.

    Spread across the WHOLE game (a couple of states per side-turn, over
    every side-turn) rather than taking the first N sequentially. The
    opening turns are pre-contact -- armies are still marching from their
    keeps -- so any head-of-game sample reports a meaningless 0% fire rate
    for adjacency-based motifs. `min_turn` additionally skips the
    contactless opening."""
    out = []
    n_b = max(1, len(bundle_paths))
    per_bundle = max(2, limit_states // n_b + 1)
    for bp in bundle_paths:
        # Materialize the bundle's side-turns (load_side_turns already
        # deepcopies each pre_state, so this costs nothing extra), then
        # STRIDE over them: the per-bundle budget must spread across the
        # game's whole arc, not be eaten by its first few turns.
        sts = [st for st in load_side_turns(Path(bp))
               if st.turn >= min_turn and st.actions]
        if not sts:
            continue
        stride = max(1, len(sts) // max(1, per_bundle // 2))
        got = 0
        for st in sts[::stride]:
            # Two samples per side-turn: its first decision (all units
            # fresh) and a mid-turn decision (partially committed board).
            picks = {0, len(st.actions) // 2}
            gs = copy.deepcopy(st.pre_state)
            for i, cmd in enumerate(st.actions):
                if i in picks:
                    out.append((copy.deepcopy(gs), st.side))
                    got += 1
                    if len(out) >= limit_states:
                        return out
                _apply_command(gs, cmd)
            if got >= per_bundle:
                break
    return out


def _grad_norms(model):
    """(advice-path norm, total norm, per-group norms)."""
    tot_sq = 0.0
    adv_sq = 0.0
    groups = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        s = float(p.grad.detach().pow(2).sum())
        tot_sq += s
        if name.startswith("advice_"):
            adv_sq += s
            key = name.split(".")[0]
            groups[key] = groups.get(key, 0.0) + s
    return (adv_sq ** 0.5, tot_sq ** 0.5,
            {k: v ** 0.5 for k, v in sorted(groups.items())})


def visit_counts_from_priors(encoded, output, gs, top_k=8, total=32):
    """A realistic MCTS-shaped distillation target: visits ∝ the model's own
    legal-action priors over the top-k actions. Shape matters for gradient
    flow; exact values do not."""
    priors = enumerate_legal_actions_with_priors(encoded, output, gs)
    if not priors:
        return []
    top = sorted(priors, key=lambda p: -p.prior)[:top_k]
    z = sum(p.prior for p in top) or 1.0
    vc = []
    for p in top:
        n = max(1, int(round(total * p.prior / z)))
        # Pass the index fields through VERBATIM (including None) -- exactly
        # what mcts.extract_visit_counts emits. Coercing None->0 would turn a
        # recruit / end_turn actor into a UNIT ATTACK (type_idx 0) and the
        # loss would index unit_ids with a recruit slot.
        vc.append((p.actor_idx, p.target_idx, p.weapon_idx, n, p.type_idx))
    return vc


def loss_for_state(policy, gs, z_val, with_advice):
    """The REAL training loss on one state (policy distillation + value CE),
    optionally with prospective advice tokens attached exactly as the trainer
    reforward does. Returns (loss, n_opportunities)."""
    model, encoder = policy._model, policy._encoder
    encoded = encoder.encode(gs)
    n_opp = 0
    if with_advice and getattr(model, "has_advice", False):
        opps = prospective_opportunities(gs)
        n_opp = len(opps)
        mids, feats, mu, dh = opportunities_to_features(encoded, opps)
        encoded.advice_tokens = model.build_advice_tokens(
            encoded, mids, feats, mu, dh)
    output = model(encoded)
    vc = visit_counts_from_priors(encoded, output, gs)
    if not vc:
        return None, n_opp
    policy_loss, _tv, _nlp = _mcts_factored_policy_loss(
        encoded, output, gs, vc, vectorized=True, decision_step=0)
    z_t = torch.tensor([z_val], dtype=torch.float32)
    value_loss = _categorical_value_loss(
        output.value_logits, z_t, model._value_atoms)
    return policy_loss + 1.0 * value_loss, n_opp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Advice gradient measurement")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--fresh", action="store_true",
                    help="small fresh model instead of a checkpoint (fast)")
    ap.add_argument("--bundles",
                    default="training/validate_exports/hf_bundles")
    ap.add_argument("--states", type=int, default=400,
                    help="decision states for the FIRE RATE measurement")
    ap.add_argument("--grad-states", type=int, default=12,
                    help="advice-carrying states per gradient step")
    ap.add_argument("--steps", type=int, default=3,
                    help="optimizer steps (bootstrap tracking)")
    args = ap.parse_args(argv)

    pat = args.bundles
    if Path(pat).is_dir():
        pat = str(Path(pat) / "*.tar")
    paths = sorted(glob.glob(pat))
    print(f"collecting decision states from {len(paths)} bundle(s) ...",
          flush=True)
    states = decision_states(paths, args.states)
    print(f"  {len(states)} decision states", flush=True)

    # ---------------- 1. FIRE RATE ----------------
    fired, by_motif, n_opps_total = 0, {}, 0
    firing_states = []
    for i, (gs, side) in enumerate(states):
        gs.global_info.current_side = side
        opps = prospective_opportunities(gs, side=side)
        if opps:
            fired += 1
            n_opps_total += len(opps)
            firing_states.append((gs, side))       # reused for the grad batch
            for o in opps:
                by_motif[o.motif] = by_motif.get(o.motif, 0) + 1
        if (i + 1) % 100 == 0:
            print(f"    scanned {i+1}/{len(states)}  fired={fired}", flush=True)
    n = len(states) or 1
    print("\n=== 1. FIRE RATE (real decision states) ===")
    print(f"states with >=1 advice opportunity: {fired}/{n} "
          f"({100.0*fired/n:.1f}%)")
    print(f"mean opportunities per firing state: "
          f"{(n_opps_total/fired if fired else 0):.2f}")
    for m, c in sorted(by_motif.items(), key=lambda t: -t[1]):
        print(f"  {m:24s} {c:5d} opportunities")

    # ---------------- 2. GRADIENT SHARE ----------------
    print("\n=== 2. GRADIENT SHARE (real training loss) ===", flush=True)
    if args.fresh:
        policy = TransformerPolicy(d_model=64, num_layers=2, num_heads=4,
                                   d_ff=128, advice=True)
        print("  model: fresh d_model=64/2L (advice=True)")
    else:
        raw = torch.load(args.checkpoint, map_location="cpu",
                         weights_only=False)
        arch = raw.get("arch", {}) or {}
        kw = {k: int(arch[k]) for k in
              ("d_model", "num_layers", "num_heads", "d_ff") if k in arch}
        policy = TransformerPolicy(
            aux_score=bool(raw.get("aux_score", False)),
            moves_left=bool(raw.get("moves_left", False)),
            advice=True, **kw)
        policy.load_checkpoint(Path(args.checkpoint))
        print(f"  model: {args.checkpoint} {kw} + advice graft")

    model = policy._model
    model.train()
    advice_states = firing_states[:args.grad_states]
    print(f"  gradient batch: {len(advice_states)} advice-carrying states",
          flush=True)
    if not advice_states:
        print("  NO advice-carrying states -> signal is dead on arrival")
        return 0

    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        used = 0
        for gs, side in advice_states:
            gs = copy.deepcopy(gs)
            gs.global_info.current_side = side
            loss, n_opp = loss_for_state(policy, gs, 0.0, with_advice=True)
            if loss is None:
                continue
            loss.backward()
            used += 1
        g_adv, g_tot, groups = _grad_norms(model)
        ao = float(model.advice_out.weight.detach().abs().sum())
        share = (100.0 * g_adv / g_tot) if g_tot else 0.0
        print(f"  step {step}: states={used}  ||g_advice||={g_adv:.4e}  "
              f"||g_total||={g_tot:.4e}  share={share:.3f}%  "
              f"|advice_out|={ao:.4e}", flush=True)
        for k, v in groups.items():
            print(f"      {k:24s} ||g||={v:.4e}", flush=True)
        opt.step()

    print("\ninterpretation: share% is the fraction of the gradient flowing "
          "through the advice path on advice-carrying states; the moves-left "
          "head sits at ~0.03% as pure telemetry. Watch |advice_out| grow off "
          "zero -- that is the bootstrap that switches the rest of the path on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
