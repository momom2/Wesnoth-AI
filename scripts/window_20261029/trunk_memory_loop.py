"""How the memory's loop grows what a trunk step changes in it (CPU, float32;
Idle priority).

A full-network step's KL sits in a few game-sides, where the stepped network's
memory ends 40% to 60% off theta_k's (trunk_memory_lane.py), and up to a gap
of some 20% of the memory's norm the gap is in proportion to the step (its
--scales reading). In that range the gap g_t between the two memories follows

    g_{t+1} = J_t g_t + s_t

with J_t the Jacobian of theta_k's write in the memory it reads at decision t
(through the trunk) and s_t the source: what the step changes in one
decision's write from the same memory. This script walks game-sides decision
by decision and reads both by finite differences on theta_k's own memory M_t,
theta' = theta_k + s d being stage C's reduced_adamw step (--direction reads
the pass's noise or a random direction of d's norms in its place):

  the loop    N memories M_t + EPS |M_t| q_i, the q_i orthonormal, each
              written by theta_k, the differences re-orthonormalised (QR)
              after every decision (Benettin et al., Meccanica 1980): |R_ii|
              is the growth of the i-th most growing direction at that
              decision, and the mean of its log a Lyapunov exponent of the
              side.
  the gap     theta_k + (s / SMALL) d on its own memory, times SMALL: g_t
              where it is linear in the step. And theta' on its own memory,
              stage C's reading.
  the source  theta_k + (s / SMALL) d reading M_t, its write's distance from
              theta_k's, times SMALL: s_t.
  one step    theta_k on M_t + EPS |M_t| g^_t (the gap's direction) and on a
              random direction: |J v| for each. And on M_t + a |M_t| g^_t for
              the amplitudes a of AMPS: whether the growth holds at a gap of
              10% and of 30%, and what the policy reads of such a gap.
  the write   m' = (1 - z) m + z c, so a difference dm in the memory read
              writes (1 - z) dm + (c - m) dz + z dc: the carry, the gate's
              part (dz through the gate's own read of m, and through the
              trunk's output h) and the candidate's part (through h alone).
              Each part's projection on the written difference's direction,
              over |dm|: the parts sum to the growth.

    python scripts/window_20261029/trunk_memory_loop.py --sides game_own2a_own2b_s2_827637:2 game_own2a_own2b_s2_827637:1 --out training/metrics/trunk_memory_loop_20261110.json
"""
import argparse
import collections
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

EPS = 3e-3                       # a probe's size over |M_t|: the gap is linear to 0.2, float32's noise far below
SMALL = 64.0                     # the step's divisor for the linear gap (trunk_memory_lane.py --scales: linear at 1/64)
AMPS = (0.1, -0.1, 0.3, -0.3)    # the gap direction's finite sizes over |M_t|
PARTIAL_EVERY = 50               # decisions between two writes of a side's partial reading
PROBED = ("top", "gap", "random")
SEGMENTS = ((0, 10), (10, 40), (40, 80), (80, 160), (160, 10 ** 6))


class WriteTap:
    """The inputs and outputs of the write's two linear maps at the last
    forward of the network it is hooked on."""

    def __init__(self, slot_memory):
        self.seen = {}
        self.handles = [module.register_forward_hook(self._keep(name))
                        for name, module in (("gate", slot_memory.gate), ("candidate", slot_memory.candidate))]

    def _keep(self, name):
        def hook(module, inputs, output):
            self.seen[name] = (inputs[0].detach(), output.detach())
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def write_parts(tap, slot_memory, read, wrote, i: int) -> dict:
    """Row i's written difference from row 0's, split by the write's terms
    (module docstring): each term's projection on the written difference's
    direction over |dm|, and the trunk's part in it."""
    import torch
    d = read.size(-1)
    (_, z_logit), (h, c_logit) = tap.seen["gate"], tap.seen["candidate"]
    m0, z0, c0 = read[0], torch.sigmoid(z_logit[0]), torch.tanh(c_logit[0])
    dm, dh = read[i] - m0, h[i] - h[0]
    dz, dc = torch.sigmoid(z_logit[i]) - z0, torch.tanh(c_logit[i]) - c0
    out = wrote[i] - wrote[0]
    carry, gate, candidate = (1.0 - z0) * dm, (c0 - m0) * dz, z0 * dc
    direct = (c0 - m0) * z0 * (1.0 - z0) * (dm @ slot_memory.gate.weight[:, d:].T)
    size, to = dm.norm(), out / out.norm()

    def on(part) -> float:
        return float((part * to).sum() / size)

    return {"growth": float(out.norm() / size), "carry": on(carry), "gate_direct": on(direct),
            "gate_trunk": on(gate - direct), "candidate": on(candidate),
            "second_order": on(out - carry - gate - candidate),
            "kept": float((to * dm).sum() / size),                       # cosine of the written and the read difference
            "trunk_gain": float(dh.norm() / size), "trunk_cos": float((dh * dm).sum() / (dh.norm() * size)),
            "trunk_common": common_share(dh),                            # of the trunk's answer, the part every slot shares
            "candidate_gain": float(dc.norm() / dh.norm()),
            "z": float((z0 * dm * dm).sum() / (dm * dm).sum())}          # the gate's opening where the difference sits


def slot_shares(v):
    """[k]: each slot's share of v's squared norm."""
    e = (v * v).sum(-1)
    return e / e.sum()


def common_share(v) -> float:
    """The share of v's squared norm ([k, d]) in its mean over the slots: 1
    when every slot holds the same vector, about 1/k for a random v."""
    return float(v.size(0) * v.mean(0).pow(2).sum() / v.pow(2).sum())


def participation(shares) -> float:
    """The number of equal shares that would spread as these do."""
    return float(1.0 / (shares * shares).sum())


def walk(context, nets, gs, generator, partial, limit=None):
    """The readings along game-side `gs` (module docstring): per-decision
    columns, and per-decision arrays over the slots."""
    import torch
    from tools.pg_pass import stage_step
    from wesnoth_ai.constants import TURN_NORM
    from wesnoth_ai.memory import MemoryBatch
    from wesnoth_ai.pg_loss import LAW_KEYS, action_log_probs, factor_laws, legal_kl_laws

    positions = context.loader.side(gs.game, gs.side)
    if len(positions) != gs.stop - gs.start:
        raise ValueError(f"{gs.game} side {gs.side}: {len(positions)} positions for {gs.stop - gs.start} decisions")
    k, device, N = context.config.slots, context.device, nets["directions"]
    M, small, full = (nets[net][1].initial_memory(k).to(device) for net in ("k", "small", "step"))
    flat = M.numel()
    Q = torch.linalg.qr(torch.randn(flat, N, generator=generator))[0].T.reshape(N, *M.shape).to(device)
    tap = WriteTap(nets["k"][1].slot_memory)
    cols, slots = collections.defaultdict(list), collections.defaultdict(list)
    last_source, last_turn = None, None

    def forward(net, memories, p):
        encoder, model = nets[net]
        n = memories.size(0)
        staged = context.encoder.stage_raws([p.raw] * n, device=device)
        batch = stage_step([p] * n, staged, context.model, device)
        padded = model.forward_embedded(encoder.embed_staged(staged), packed=context.config.packed,
                                        memory=MemoryBatch(padded=memories, counts=[k] * n))
        return (padded.memory_padded[:, :k].clone(), factor_laws(padded, batch.views, batch.rows),
                action_log_probs(padded, batch))

    def kl_from(ref, laws):
        return legal_kl_laws(dataclasses.replace(laws, **{key: ref[key].expand_as(getattr(laws, key))
                                                          for key in LAW_KEYS}), laws)

    try:
        with torch.no_grad():
            for t, p in enumerate(positions[:limit]):
                norm = M.norm()
                g = (small - M) * SMALL
                gap, g_hat = g.norm(), g / g.norm()
                u = torch.randn(M.shape, generator=generator).to(device)
                u = u / u.norm()
                read = torch.cat([M[None], M[None] + EPS * norm * Q, (M + EPS * norm * g_hat)[None],
                                  (M + EPS * norm * u)[None], torch.stack([M + a * norm * g_hat for a in AMPS])])
                wrote, laws, logp = forward("k", read, p)
                ref = {key: getattr(laws, key)[:1] for key in LAW_KEYS}
                kl_k = kl_from(ref, laws)
                D = (wrote[1:1 + N] - wrote[0]) / (EPS * norm)                    # J q_i
                J_g, J_u = ((wrote[i] - wrote[0]) / (EPS * norm) for i in (1 + N, 2 + N))
                Q_new, R = torch.linalg.qr(D.reshape(N, flat).T)
                Q_new = Q_new.T.reshape(N, *M.shape)
                w_small, laws_small, _ = forward("small", torch.stack([small, M]), p)
                source = (w_small[1] - wrote[0]) * SMALL
                g_new = (w_small[0] - wrote[0]) * SMALL
                w_full, laws_full, logp_full = forward("step", full[None], p)
                kl_small, kl_full = kl_from(ref, laws_small), kl_from(ref, laws_full)
                z0 = torch.sigmoid(tap.seen["gate"][1][0])
                c0 = torch.tanh(tap.seen["candidate"][1][0])
                new_norm = wrote[0].norm()
                turn = float(p.raw.global_feats[0]) * TURN_NORM

                cols["shadow"].append(bool(p.shadow))
                cols["kind"].append(int(p.action[1]))
                cols["turn"].append(round(turn, 2))
                cols["new_turn"].append(last_turn is not None and round(turn) != round(last_turn))
                cols["units"].append(len(p.raw.unit_ids))
                cols["ours"].append(int(p.raw.unit_is_ours.sum()))
                cols["sightings"].append(0 if p.raw.sight_type_ids is None else len(p.raw.sight_type_ids))
                cols["norm"].append(float(norm))
                cols["write"].append(float((wrote[0] - M).norm() / norm))
                cols["growth"].append([float(x) for x in R.diagonal().abs()])
                cols["gap"].append(float(gap / norm))
                cols["gap_full"].append(float((full - M).norm() / norm))
                cols["J_gap"].append(float(J_g.norm()))
                cols["J_random"].append(float(J_u.norm()))
                cols["source"].append(float(source.norm() / new_norm))
                cols["source_on_gap"].append(float((source * J_g).sum() / (source.norm() * J_g.norm())))
                cols["source_on_last"].append(None if last_source is None else
                                              float((source * last_source).sum() / (source.norm() * last_source.norm())))
                cols["linear_error"].append(float((J_g * gap + source - g_new).norm() / g_new.norm()))
                cols["gap_in_span"].append(float(((Q.reshape(N, flat) @ g_hat.reshape(flat)) ** 2).sum()))
                cols["gap_on_top"].append(float((Q[0] * g_hat).sum() ** 2))
                cols["source_in_span"].append(float(((Q_new.reshape(N, flat) @ source.reshape(flat)) ** 2).sum()
                                                    / (source * source).sum()))
                for j, a in enumerate(AMPS):
                    diff = wrote[3 + N + j] - wrote[0]
                    cols[f"amp{a:+g}"].append(float(diff.norm() / (abs(a) * norm)))
                    cols[f"amp{a:+g}_on_linear"].append(float(math.copysign(1.0, a) * (diff * J_g).sum()
                                                              / (diff.norm() * J_g.norm())))
                    cols[f"amp{a:+g}_kl"].append(float(kl_k[3 + N + j]))
                for name, i in zip(PROBED, (1, 1 + N, 2 + N)):
                    for key, value in write_parts(tap, nets["k"][1].slot_memory, read, wrote, i).items():
                        cols[f"{name}_{key}"].append(value)
                cols["z_mean"].append(float(z0.mean()))
                for bound in (0.01, 0.03, 0.1):
                    cols[f"z_below_{bound:g}"].append(float((z0 < bound).float().mean()))
                cols["z_above_0.5"].append(float((z0 > 0.5).float().mean()))
                cols["c_saturated"].append(float((c0.abs() > 0.95).float().mean()))
                cols["m_saturated"].append(float((M.abs() > 0.9).float().mean()))
                cols["kl_full"].append(float(kl_full[0]))
                cols["delta_full"].append(float(logp[0] - logp_full[0]))
                cols["kl_linear"].append(float(kl_small[0]) * SMALL * SMALL)   # the small step's KL, were it quadratic
                cols["kl_source"].append(float(kl_small[1]) * SMALL * SMALL)   # the same with the memory held
                for name, v in (("gap", g_hat), ("top", Q[0]), ("source", source)):
                    shares = slot_shares(v)
                    cols[f"{name}_slots"].append(participation(shares))
                    cols[f"{name}_channels"].append(participation(slot_shares(v.T)))
                    slots[name].append(shares.cpu().numpy())
                slots["z"].append(z0.mean(-1).cpu().numpy())
                slots["m"].append(slot_shares(M).cpu().numpy())

                last_turn = turn
                if not p.shadow:                                              # a refused row writes nothing
                    M, small, full, Q, last_source = wrote[0], w_small[0], w_full[0], Q_new, source
                if (t + 1) % PARTIAL_EVERY == 0:
                    partial(cols)
    finally:
        tap.close()
    slots["final_gap"], slots["final_top"] = [((small - M) * SMALL).cpu().numpy()], [Q[0].cpu().numpy()]
    return dict(cols), dict(slots)


def summary(cols) -> dict:
    """The side's exponents, its gap at marks, and each probe's mean parts, by
    stretch of the side (SEGMENTS)."""
    import numpy as np
    live = ~np.array(cols["shadow"])
    T = len(live)
    growth = np.log(np.array(cols["growth"]))
    marks = [m for m in (1, 5, 10, 20, 40, 80, 160, 320) if m < T] + [T - 1]
    out = {"decisions": T, "refused": int((~live).sum()),
           "gap": {str(m): cols["gap"][m] for m in marks}, "gap_full": {str(m): cols["gap_full"][m] for m in marks},
           "norm": {str(m): cols["norm"][m] for m in marks}, "segments": {}}
    for a, b in SEGMENTS:
        seg = np.zeros(T, bool)
        seg[a:b] = True
        seg &= live
        if not seg.any():
            continue

        def mean(name, seg=seg):
            return float(np.mean([x for x, keep in zip(cols[name], seg) if keep and x is not None]))

        def log_mean(name, seg=seg):
            return float(np.mean(np.log([x for x, keep in zip(cols[name], seg) if keep])))

        r = {"decisions": int(seg.sum()), "exponents": [float(x) for x in growth[seg].mean(0)],
             "log_J_gap": log_mean("J_gap"), "log_J_random": log_mean("J_random"),
             **{f"log_amp{a:+g}": log_mean(f"amp{a:+g}") for a in AMPS},
             **{name: mean(name) for name in ("write", "source", "source_on_gap", "source_on_last", "linear_error",
                                              "gap_in_span", "gap_on_top", "source_in_span", "z_mean", "z_below_0.01",
                                              "z_below_0.03", "z_below_0.1", "z_above_0.5", "c_saturated",
                                              "m_saturated", "kl_full", "kl_linear", "kl_source", "gap_slots",
                                              "gap_channels", "top_slots", "source_slots",
                                              *(f"amp{a:+g}_kl" for a in AMPS))},
             "parts": {name: {key: mean(f"{name}_{key}") for key in (
                 "growth", "carry", "gate_direct", "gate_trunk", "candidate", "second_order", "kept", "trunk_gain",
                 "trunk_cos", "candidate_gain", "z")} for name in PROBED}}
        out["segments"][f"[{a},{min(b, T)})"] = r
    return out


def step_direction(p, which: str, generator):
    """Stage C's reduced_adamw direction ("pass"); the half-difference of the
    pass's first and last four splits, which holds its noise and no signal
    (trunk_direction_noise.py; "noise"); or a Gaussian direction with the
    pass's norm on every tensor ("random")."""
    import torch
    from wesnoth_ai.pg_step import direction, pooled_gradient
    d = direction(p, "reduced_adamw")
    if which == "pass":
        return d
    if which == "noise":
        halves = ((0, 1, 2, 3), (4, 5, 6, 7))
        games = [int(n) for n in p["split_games"]]
        n_a, n_b = (sum(games[i] for i in half) for half in halves)
        d_a, d_b = (direction(p, "reduced_adamw", gradient=pooled_gradient(p, half)) for half in halves)
        c = math.sqrt(n_a * n_b) / (n_a + n_b)
        return {n: c * (d_a[n] - d_b[n]) for n in d}
    out = {}
    for n, t in d.items():
        r = torch.randn(t.shape, generator=generator).to(t.dtype)
        out[n] = r * (t.norm() / r.norm()) if float(t.norm()) > 0 else torch.zeros_like(t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sides", nargs="+", required=True, help="held-out game-sides, each <game>:<side>")
    ap.add_argument("--direction", choices=("pass", "noise", "random"), default="pass")
    ap.add_argument("--scale", type=float, default=1.0, help="the step, in multiples of stage C's")
    ap.add_argument("--directions", type=int, default=12, help="the loop's probes N")
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--limit", type=int, help="read only the first decisions of each side")
    ap.add_argument("--arrays", type=Path, help="a directory for the per-slot arrays, one .npz a side")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception:                                   # a reading, not a run: priority is best effort
        pass
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    from tools.pg_stage_c import load_pass
    from trunk_kl_by_module import STAGE_C, context_on

    t0 = time.time()
    wanted = [(text.rsplit(":", 1)[0], int(text.rsplit(":", 1)[1])) for text in a.sides]
    s = a.scale * json.loads((STAGE_C / "stage_c.json").read_text(encoding="utf-8"))["arms"]["reduced_adamw"]["s"]
    d = step_direction(load_pass(STAGE_C / "pass.pt"), a.direction, torch.Generator().manual_seed(a.seed))
    context = context_on(list(dict.fromkeys(game for game, _ in wanted)))
    nets = {"k": context.network(d, 0.0), "small": context.network(d, s / SMALL), "step": context.network(d, s),
            "directions": a.directions}
    out = {"s": s, "direction": a.direction, "eps": EPS, "small": SMALL, "amps": list(AMPS),
           "directions": a.directions, "seed": a.seed, "limit": a.limit, "sides": {}}
    if a.out.exists():                                  # a stopped reading goes on from the sides it finished
        before = json.loads(a.out.read_text(encoding="utf-8"))
        if all(before.get(key) == value for key, value in out.items() if key != "sides"):
            out["sides"] = {name: side for name, side in before["sides"].items() if "summary" in side}

    def write():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out), encoding="utf-8")

    by_name = {(gs.game, gs.side): gs for gs in context.sides}
    for game, side in wanted:
        name = f"{game}:{side}"
        if name in out["sides"]:
            continue
        gs = by_name[(game, side)]

        def partial(cols, name=name, gs=gs):
            out["sides"][name] = {"R": gs.R, "reading": cols}
            write()
            print(f"{time.time() - t0:7.0f} s  {name} at decision {len(cols['shadow'])}", flush=True)

        cols, slots = walk(context, nets, gs, torch.Generator().manual_seed(a.seed), partial, a.limit)
        out["sides"][name] = {"R": gs.R, "summary": summary(cols), "reading": cols}
        write()
        if a.arrays:
            a.arrays.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(a.arrays / f"{name.replace(':', '_side')}.npz",
                                **{key: np.stack(rows) for key, rows in slots.items()})
        print(f"{time.time() - t0:7.0f} s  {name}", json.dumps(out["sides"][name]["summary"], indent=1), flush=True)
    context.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
