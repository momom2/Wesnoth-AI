"""The weight directions that push a write along the memory's pattern, and a
trunk step with them removed (CPU, float32; Idle priority, 4 threads beside a
run).

trunk_memory_states.py found that a trunk step's KL is the memory's, and that
the stepped weights feed the gap at every write along one pattern p
(trunk_memory_pattern.py). This reading takes that push at its source. At a
decision of a fit side, with theta_k's memory there given, the gradient at
theta_k of p . (the written memory's mean over the slots) is the weight
direction q that moves the write along p. q is summed over chunks of
consecutive writes of each fit side, and the reading holds:

  - the chunks' cosines with each other, within a side and across sides (the
    push's common weight direction against each chunk's own);
  - the pass's signal on the push: q . d_i for the direction of each of the
    pass's splits alone, as a mean with its SE over the splits;
  - the first-order push a write, s q . d over the chunk's writes, to set
    beside the measured one;
  - the step's part in the q's span, beside its noise's (the half-difference
    of the pass's splits), and that span's share of the step's signal and of
    its first-order gain, read between the pass's halves.

--save keeps the step with that part removed (rest.pt) and that part alone
(push.pt), for trunk_memory_states.py --direction-file.

    python scripts/window_20261029/trunk_memory_project.py --sides game_own2a_own2b_s2_827637:2 --pattern <npy> --save <dir> --out training/metrics/trunk_memory_project_20261110.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def reference_memories(context, gs, limit):
    """Game-side `gs`'s written decisions, each with theta_k's memory before it."""
    import torch
    from wesnoth_ai.memory import MemoryBatch

    k, device = context.config.slots, context.device
    M = context.model.initial_memory(k).detach().to(device)
    rows = []
    with torch.no_grad():
        for p in context.loader.side(gs.game, gs.side)[:limit]:
            if p.shadow:                                                  # a refused row writes nothing
                continue
            rows.append((p, M))
            staged = context.encoder.stage_raws([p.raw], device=device)
            padded = context.model.forward_embedded(context.encoder.embed_staged(staged), packed=context.config.packed,
                                                    memory=MemoryBatch(padded=M[None], counts=[k]))
            M = padded.memory_padded[0, :k].clone()
    return rows


def push(context, params, rows, pattern, batch):
    """The gradient at theta_k of the pattern's part of the write, summed over
    `rows` (reference_memories), as one vector over `params`."""
    import torch
    from wesnoth_ai.memory import MemoryBatch

    k, device = context.config.slots, context.device
    total = None
    for i in range(0, len(rows), batch):
        part = rows[i:i + batch]
        staged = context.encoder.stage_raws([p.raw for p, _ in part], device=device)
        padded = context.model.forward_embedded(context.encoder.embed_staged(staged), packed=context.config.packed,
                                                memory=MemoryBatch(padded=torch.stack([m for _, m in part]),
                                                                   counts=[k] * len(part)))
        on_p = (padded.memory_padded[:, :k].float().mean(1) @ pattern).sum()
        grads = torch.autograd.grad(on_p, params, allow_unused=True)
        flat = torch.cat([torch.zeros(p.numel()) if g is None else g.reshape(-1).float().cpu()
                          for g, p in zip(grads, params)])
        total = flat if total is None else total + flat
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sides", nargs="+", required=True, help="the fit game-sides, each <game>:<side>")
    ap.add_argument("--pattern", type=Path, required=True, help="a .npy unit vector of the memory's width")
    ap.add_argument("--chunk", type=int, default=60, help="the writes summed into one q")
    ap.add_argument("--batch", type=int, default=20, help="the writes a forward holds")
    ap.add_argument("--limit", type=int, default=240, help="read only the first decisions of each side")
    ap.add_argument("--save", type=Path, help="a directory for rest.pt and push.pt")
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
    from trunk_memory_groups import memory_group
    from trunk_memory_loop import step_direction
    from wesnoth_ai.param_groups import named_model_parameters
    from wesnoth_ai.pg_step import direction, head_of, pooled_gradient

    t0 = time.time()
    wanted = [(text.rsplit(":", 1)[0], int(text.rsplit(":", 1)[1])) for text in a.sides]
    s = json.loads((STAGE_C / "stage_c.json").read_text(encoding="utf-8"))["arms"]["reduced_adamw"]["s"]
    pattern = torch.from_numpy(np.load(a.pattern)).float()
    context = context_on(list(dict.fromkeys(game for game, _ in wanted)))
    named = named_model_parameters(context.model, context.encoder)
    params = [p for _, p in named]

    def flat(d):
        return torch.cat([(d[n] if n in d else torch.zeros_like(p)).detach().reshape(-1).float().cpu()
                          for n, p in named])

    def dots(v):
        return torch.tensor([float(q @ v) for q in qs], dtype=torch.float64)

    qs, chunks = [], []
    by_name = {(gs.game, gs.side): gs for gs in context.sides}
    for game, side in wanted:
        rows = reference_memories(context, by_name[(game, side)], a.limit)
        for first in range(0, len(rows), a.chunk):
            part = rows[first:first + a.chunk]
            if 2 * len(part) < a.chunk:                 # a side's short tail joins no q
                continue
            qs.append(push(context, params, part, pattern, a.batch))
            chunks.append({"side": f"{game}:{side}", "first": first, "writes": len(part),
                           "on_p": float(torch.stack([m.mean(0) @ pattern for _, m in part]).mean()),
                           "norm": float(qs[-1].norm())})
            print(f"{time.time() - t0:7.0f} s  {game}:{side} writes {first}-{first + len(part)}: "
                  f"memory on p {chunks[-1]['on_p']:+.2f}, |q| a write {chunks[-1]['norm'] / len(part):.4g}", flush=True)
    context.close()

    n = len(qs)
    norms = torch.tensor([c["norm"] for c in chunks], dtype=torch.float64)
    gram = torch.tensor([[float(qs[i] @ qs[j]) for j in range(n)] for i in range(n)], dtype=torch.float64)
    cosines = gram / torch.outer(norms, norms)
    same = torch.tensor([[x["side"] == y["side"] for y in chunks] for x in chunks])
    off = ~torch.eye(n, dtype=torch.bool)

    passed = load_pass(STAGE_C / "pass.pt")
    d = flat(direction(passed, "reduced_adamw"))
    noise = flat(step_direction(passed, "noise", None))
    on_d, on_noise = dots(d), dots(noise)
    splits = torch.stack([dots(flat(direction(passed, "reduced_adamw", gradient=pooled_gradient(passed, (i,)))))
                          for i in range(len(passed["splits"]))])       # [splits, chunks]
    half = len(passed["splits"]) // 2
    halves = [tuple(range(half)), tuple(range(half, len(passed["splits"])))]
    g_half = [flat(pooled_gradient(passed, h)) for h in halves]
    d_half = [flat(direction(passed, "reduced_adamw", gradient=pooled_gradient(passed, h))) for h in halves]
    g = flat(direction(passed, "sgd"))
    del passed

    def over_splits(column):
        mean, se = float(column.mean()), float(column.std(unbiased=True) / len(column) ** 0.5)
        return {"mean": mean, "se": se, "z": mean / se if se else None}

    for i, chunk in enumerate(chunks):
        chunk["step_push_a_write"] = s * float(on_d[i]) / chunk["writes"]
        chunk["noise_push_a_write"] = s * float(on_noise[i]) / chunk["writes"]
        chunk["cos_with_step"] = float(on_d[i] / (norms[i] * d.norm()))
        chunk["splits"] = over_splits(splits[:, i] / norms[i])

    # the step's part in the q's span: d_par = sum_i c_i q_i / |q_i|, the cosines' pseudo-inverse on (q_i . d) / |q_i|
    inverse = torch.linalg.pinv(cosines, rtol=1e-10, hermitian=True)

    def in_span(on):
        coefficients = inverse @ (on / norms)
        return coefficients, float(coefficients @ (on / norms))

    def span_dot(x, y):
        """(P x) . (P y) for the span's projector P, from x's and y's dots with the q's."""
        return float((x / norms) @ inverse @ (y / norms))

    coefficients, par_sq = in_span(on_d)
    _, noise_par_sq = in_span(on_noise)
    par = torch.zeros_like(d)
    for c, q, size in zip(coefficients.tolist(), qs, norms.tolist()):
        par += (c / size) * q
    rest = d - par
    left = dots(rest)

    # the pass's halves against each other: the step's signal (d_a . d_b holds no noise) and its first-order gain
    # (one half's gradient on the other's direction), each with its part in the span
    on_g_half, on_d_half = [dots(v) for v in g_half], [dots(v) for v in d_half]
    signal = float(d_half[0] @ d_half[1])
    gain = 0.5 * (float(g_half[1] @ d_half[0]) + float(g_half[0] @ d_half[1]))
    halves_out = {"step_signal_share": signal / float(d @ d),
                  "signal_share_in_span": span_dot(on_d_half[0], on_d_half[1]) / signal,
                  "gain": gain,
                  "gain_share_in_span": 0.5 * (span_dot(on_g_half[1], on_d_half[0])
                                               + span_dot(on_g_half[0], on_d_half[1])) / gain,
                  "gain_in_sample": float(g @ d),
                  "gain_in_sample_share_in_span": span_dot(dots(g), on_d) / float(g @ d)}
    groups = {}
    offset = 0
    for name, p in named:
        group = memory_group(name, head_of)
        piece = slice(offset, offset + p.numel())
        row = groups.setdefault(group, {"step": 0.0, "push": 0.0})
        row["step"] += float(d[piece] @ d[piece])
        row["push"] += float(par[piece] @ par[piece])
        offset += p.numel()
    out = {"s": s, "pattern": a.pattern.name, "chunk": a.chunk, "limit": a.limit, "sides": a.sides, "chunks": chunks,
           "cosines": {"within_side": float(cosines[same & off].mean()) if bool((same & off).any()) else None,
                       "across_sides": float(cosines[~same].mean()) if bool((~same).any()) else None,
                       "matrix": cosines.tolist(),
                       "spectrum": (torch.linalg.eigvalsh(cosines).flip(0) / n).tolist()},
           "pooled": {"splits": over_splits((splits / norms).sum(1)),            # the unit q's sum
                      "step_push_a_write": s * float(on_d.sum()) / sum(c["writes"] for c in chunks),
                      "noise_push_a_write": s * float(on_noise.sum()) / sum(c["writes"] for c in chunks)},
           "step": {"norm": float(d.norm()), "share_in_span": par_sq / float(d @ d),
                    "noise_share_in_span": noise_par_sq / float(noise @ noise),
                    "push_norm": float(par.norm()), "rest_norm": float(rest.norm()),
                    "rest_on_q_over_step": float(left.abs().max() / on_d.abs().max())},
           "halves": halves_out,
           "groups": {group: {"step": row["step"] ** 0.5, "push": row["push"] ** 0.5} for group, row in groups.items()}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    if a.save:
        a.save.mkdir(parents=True, exist_ok=True)
        for file, vector in (("rest.pt", rest), ("push.pt", par)):
            pieces, offset = {}, 0
            for name, p in named:
                pieces[name] = vector[offset:offset + p.numel()].reshape(p.shape).clone()
                offset += p.numel()
            torch.save(pieces, a.save / file)
    print(f"{time.time() - t0:7.0f} s  {n} q's: cosines within a side {out['cosines']['within_side']}, across sides "
          f"{out['cosines']['across_sides']:.4f}; the pass on their sum {json.dumps(out['pooled'])}; "
          f"the step in their span {json.dumps(out['step'])}; the pass's halves {json.dumps(halves_out)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
