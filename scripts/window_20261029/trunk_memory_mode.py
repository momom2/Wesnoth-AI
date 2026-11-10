"""What the direction a trunk step grows in the memory is (CPU, float32; Idle
priority).

trunk_memory_loop.py reads that the gap a trunk step opens in a diverging
game-side's memory sits in a direction theta_k's own write keeps (a growth of
about 1 a decision, where a random direction loses 10% to 14%), and that over
half of it is one channel vector written in every slot. theta_k's own memory
is such a vector too: 82% to 89% of its squared norm is its mean over the 64
slots. This script reads, along game-sides and on theta_k's own memory M_t,
with g^_t the gap's direction (theta_k + (s / SMALL) d on its own memory):

  one write   |J v| and the write's parts (trunk_memory_loop.write_parts) for
              v = g^_t, its common part (its mean over the slots, in every
              slot), the rest of it, a random channel vector in every slot and
              a random direction: whether the trunk answers any change the
              slots share as it answers the gap's, or that one alone.
  the policy  theta_k on M_t + a |M_t| v at the pairs of FINITE: the KL from
              its own law, and what moves (the value, the mass on end_turn,
              attacks and recruits).
  free lanes  at each decision of --seeds, copies of M_t displaced along g^_t
              and from there written by theta_k alone, with no step at all:
              one held at EPS |M| (its growth is the loop's own, the source
              left out), and four left free at +-AMP |M_t| and +-3 AMP |M_t|
              (whether the loop keeps, forgets or grows a difference of the
              size a step makes). The same, left free at +AMP, from a random
              direction and from a random common one. And the stepped
              network's own memory at that decision, handed to theta_k:
              whether the gap the step built stays once the step is gone.

    python scripts/window_20261029/trunk_memory_mode.py --sides game_own2a_own2b_s2_827637:2 game_own2a_own2b_s2_827637:1 --out training/metrics/trunk_memory_mode_20261110.json
"""
import argparse
import collections
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

AMP = 0.1                        # a free lane's displacement over |M_t|: a diverging side's gap passes it by decision 40
PROBES = ("gap", "gap_common", "gap_rest", "random_common", "random")
FINITE = (("gap", AMP), ("gap", -AMP), ("random", AMP), ("random_common", AMP))
FREE = (("gap", AMP), ("gap", -AMP), ("gap", 3 * AMP), ("gap", -3 * AMP), ("random", AMP), ("random_common", AMP))
PART_KEYS = ("growth", "carry", "gate_direct", "gate_trunk", "candidate", "kept", "trunk_gain", "trunk_cos",
             "trunk_common", "z")
LANE_MARKS = (0, 1, 2, 5, 10, 20, 40, 80, 160)   # decisions after a lane's seed


@dataclasses.dataclass
class Lane:
    """A copy of the memory displaced at decision t0 and written by theta_k
    since: `state` is its unit direction when it is held at EPS |M|
    (`linear`), its memory otherwise."""
    t0: int
    linear: bool
    state: object
    rows: dict = dataclasses.field(default_factory=lambda: collections.defaultdict(list))


def walk(context, nets, gs, generator, partial, seeds, limit=None):
    """The readings along game-side `gs` (module docstring): per-decision
    columns, the lanes' own columns from their seeds on, and arrays."""
    import torch
    from tools.pg_pass import stage_step
    from trunk_memory_loop import EPS, PARTIAL_EVERY, SMALL, WriteTap, common_share, write_parts
    from wesnoth_ai.constants import TURN_NORM
    from wesnoth_ai.memory import MemoryBatch
    from wesnoth_ai.pg_loss import LAW_KEYS, class_masses, factor_laws, legal_kl_laws

    positions = context.loader.side(gs.game, gs.side)
    if len(positions) != gs.stop - gs.start:
        raise ValueError(f"{gs.game} side {gs.side}: {len(positions)} positions for {gs.stop - gs.start} decisions")
    k, device = context.config.slots, context.device
    M, small, full = (nets[net][1].initial_memory(k).to(device) for net in ("k", "small", "step"))
    tap = WriteTap(nets["k"][1].slot_memory)
    cols, arrays, lanes = collections.defaultdict(list), collections.defaultdict(list), {}
    last_turn, last_g = None, None

    def forward(net, memories, p):
        encoder, model = nets[net]
        n = memories.size(0)
        staged = context.encoder.stage_raws([p.raw] * n, device=device)
        batch = stage_step([p] * n, staged, context.model, device)
        padded = model.forward_embedded(encoder.embed_staged(staged), packed=context.config.packed,
                                        memory=MemoryBatch(padded=memories, counts=[k] * n))
        return (padded.memory_padded[:, :k].clone(), factor_laws(padded, batch.views, batch.rows),
                padded.value.float()[:, 0], class_masses(padded, batch))

    def kl_from(ref, laws):
        return legal_kl_laws(dataclasses.replace(laws, **{key: ref[key].expand_as(getattr(laws, key))
                                                          for key in LAW_KEYS}), laws)

    def unit(v):
        return v / v.norm()

    def common(v):
        return v.mean(0, keepdim=True).expand_as(v)

    try:
        with torch.no_grad():
            for t, p in enumerate(positions[:limit]):
                norm = M.norm()
                g = (small - M) * SMALL
                gap = g.norm()
                u = unit(torch.randn(M.shape, generator=generator).to(device))
                g_hat = unit(g) if float(gap) > 0 else u
                probes = {"gap": g_hat, "gap_common": unit(common(g_hat)), "gap_rest": unit(g_hat - common(g_hat)),
                          "random_common": unit(common(torch.randn(M.shape, generator=generator).to(device))),
                          "random": u}
                if t in seeds:
                    lanes[f"gap@{t}"] = Lane(t, True, g_hat)
                    for name, amp in FREE:
                        lanes[f"{name}{amp:+g}@{t}"] = Lane(t, False, M + amp * norm * probes[name])
                    if float((full - M).norm()) > 0:
                        lanes[f"step@{t}"] = Lane(t, False, full.clone())
                    arrays["seed_gap"].append(g_hat.cpu().numpy())
                    arrays["seed_memory"].append(M.cpu().numpy())
                rows = [M] + [M + EPS * norm * v for v in probes.values()] \
                    + [M + a * norm * probes[name] for name, a in FINITE]
                lane_row = {}
                for name, lane in lanes.items():
                    lane_row[name] = len(rows)
                    rows.append(M + EPS * norm * lane.state if lane.linear else lane.state)
                read = torch.stack(rows)
                wrote, laws, value, masses = forward("k", read, p)
                ref = {key: getattr(laws, key)[:1] for key in LAW_KEYS}
                kl = kl_from(ref, laws)
                w_small, laws_small, _, _ = forward("small", small[None], p)
                w_full, laws_full, _, _ = forward("step", full[None], p)
                turn = float(p.raw.global_feats[0]) * TURN_NORM

                cols["shadow"].append(bool(p.shadow))
                cols["kind"].append(int(p.action[1]))
                cols["turn"].append(round(turn, 2))
                cols["new_turn"].append(last_turn is not None and round(turn) != round(last_turn))
                cols["units"].append(len(p.raw.unit_ids))
                cols["ours"].append(int(p.raw.unit_is_ours.sum()))
                cols["norm"].append(float(norm))
                cols["write"].append(float((wrote[0] - M).norm() / norm))
                cols["memory_common"].append(common_share(M))
                cols["gap"].append(float(gap / norm))
                cols["gap_full"].append(float((full - M).norm() / norm))
                stepped = float((full - M).norm()) > 0
                cols["full_on_gap"].append(float(((full - M) * g_hat).sum() / (full - M).norm()) if stepped else None)
                cols["full_common_share"].append(common_share(full - M) if stepped else None)
                cols["gap_common_share"].append(common_share(g_hat))
                cols["gap_on_last"].append(None if last_g is None else float((g_hat * last_g).sum()))
                cols["gap_on_memory"].append(float((g_hat * M).sum() / norm))
                cols["value"].append(float(value[0]))
                cols["mass"].append([float(x) for x in masses[0]])
                cols["kl_linear"].append(float(kl_from(ref, laws_small)[0]) * SMALL * SMALL)
                cols["kl_full"].append(float(kl_from(ref, laws_full)[0]))
                for i, name in enumerate(PROBES, start=1):
                    for key, x in write_parts(tap, nets["k"][1].slot_memory, read, wrote, i).items():
                        cols[f"{name}_{key}"].append(x)
                for i, (name, a) in enumerate(FINITE, start=1 + len(PROBES)):
                    tag = f"{name}{a:+g}"
                    cols[f"{tag}_growth"].append(float((wrote[i] - wrote[0]).norm() / (abs(a) * norm)))
                    cols[f"{tag}_kl"].append(float(kl[i]))
                    cols[f"{tag}_value"].append(float(value[i] - value[0]))
                    cols[f"{tag}_mass"].append([float(x) for x in masses[i] - masses[0]])
                for name, lane in lanes.items():
                    i = lane_row[name]
                    delta = lane.state if lane.linear else lane.state - M
                    size = EPS * norm if lane.linear else delta.norm()
                    lane.rows["growth"].append(float((wrote[i] - wrote[0]).norm() / size))
                    lane.rows["on_gap"].append(float((delta * g_hat).sum() / delta.norm()))
                    lane.rows["common"].append(common_share(delta))
                    if not lane.linear:
                        lane.rows["size"].append(float(size / norm))
                        lane.rows["kl"].append(float(kl[i]))
                        lane.rows["value"].append(float(value[i] - value[0]))
                        lane.rows["mass"].append([float(x) for x in masses[i] - masses[0]])
                arrays["gap_mean"].append(g.mean(0).cpu().numpy())
                arrays["memory_mean"].append(M.mean(0).cpu().numpy())

                last_turn, last_g = turn, g_hat
                if not p.shadow:                                              # a refused row writes nothing
                    for name, lane in lanes.items():
                        i = lane_row[name]
                        lane.state = unit(wrote[i] - wrote[0]) if lane.linear else wrote[i]
                    M, small, full = wrote[0], w_small[0], w_full[0]
                if (t + 1) % PARTIAL_EVERY == 0:
                    partial(cols)
    finally:
        tap.close()
    arrays["final_gap"], arrays["final_memory"] = [((small - M) * SMALL).cpu().numpy()], [M.cpu().numpy()]
    return dict(cols), {name: {"t0": lane.t0, **lane.rows} for name, lane in lanes.items()}, dict(arrays)


def summary(cols, lanes) -> dict:
    """Each probe's mean parts and each finite displacement's reading by
    stretch of the side (trunk_memory_loop.SEGMENTS), and each lane at
    LANE_MARKS decisions after its seed."""
    import numpy as np
    from trunk_memory_loop import SEGMENTS
    live = ~np.array(cols["shadow"])
    T = len(live)
    out = {"decisions": T, "refused": int((~live).sum()), "segments": {}, "lanes": {}}
    for a, b in SEGMENTS:
        seg = np.zeros(T, bool)
        seg[a:b] = True
        seg &= live
        if not seg.any():
            continue

        def mean(name, seg=seg):
            return np.mean(np.array([x for x, keep in zip(cols[name], seg) if keep and x is not None], float),
                           axis=0).tolist()

        def log_mean(name, seg=seg):
            return float(np.mean(np.log([x for x, keep in zip(cols[name], seg) if keep])))

        r = {"decisions": int(seg.sum()),
             **{name: mean(name) for name in ("write", "memory_common", "gap", "gap_full", "gap_common_share",
                                              "full_on_gap", "full_common_share", "gap_on_last", "gap_on_memory",
                                              "value", "mass", "kl_linear", "kl_full")},
             "probes": {name: {"log_growth": log_mean(f"{name}_growth"),
                               **{key: mean(f"{name}_{key}") for key in PART_KEYS}} for name in PROBES},
             "finite": {f"{name}{amp:+g}": {"log_growth": log_mean(f"{name}{amp:+g}_growth"),
                                            **{key: mean(f"{name}{amp:+g}_{key}") for key in ("kl", "value", "mass")}}
                        for name, amp in FINITE}}
        out["segments"][f"[{a},{min(b, T)})"] = r
    for name, lane in lanes.items():
        n = len(lane["growth"])
        kept = live[lane["t0"]:lane["t0"] + n]
        grown = np.concatenate([[0.0], np.cumsum(np.where(kept, np.log(lane["growth"]), 0.0))])
        marks = sorted({m for m in LANE_MARKS if m < n} | {n - 1})
        out["lanes"][name] = {"t0": lane["t0"], "marks": marks,
                              "log_growth_to": [float(grown[m]) for m in marks],
                              **{key: [lane[key][m] for m in marks]
                                 for key in ("on_gap", "common", "size", "kl", "value") if key in lane}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sides", nargs="+", required=True, help="held-out game-sides, each <game>:<side>")
    ap.add_argument("--direction", choices=("pass", "noise", "random"), default="pass")
    ap.add_argument("--scale", type=float, default=1.0, help="the step, in multiples of stage C's")
    ap.add_argument("--seeds", type=int, nargs="*", default=[10, 40, 80, 160], help="the decisions that seed lanes")
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--limit", type=int, help="read only the first decisions of each side")
    ap.add_argument("--arrays", type=Path, help="a directory for the arrays, one .npz a side")
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
    from trunk_memory_loop import EPS, SMALL, step_direction

    t0 = time.time()
    wanted = [(text.rsplit(":", 1)[0], int(text.rsplit(":", 1)[1])) for text in a.sides]
    s = a.scale * json.loads((STAGE_C / "stage_c.json").read_text(encoding="utf-8"))["arms"]["reduced_adamw"]["s"]
    d = step_direction(load_pass(STAGE_C / "pass.pt"), a.direction, torch.Generator().manual_seed(a.seed))
    context = context_on(list(dict.fromkeys(game for game, _ in wanted)))
    nets = {"k": context.network(d, 0.0), "small": context.network(d, s / SMALL), "step": context.network(d, s)}
    out = {"s": s, "direction": a.direction, "eps": EPS, "small": SMALL, "amp": AMP, "seeds": a.seeds,
           "seed": a.seed, "limit": a.limit, "sides": {}}
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

        cols, lanes, arrays = walk(context, nets, gs, torch.Generator().manual_seed(a.seed), partial, set(a.seeds),
                                   a.limit)
        out["sides"][name] = {"R": gs.R, "summary": summary(cols, lanes), "reading": cols, "lanes": lanes}
        write()
        if a.arrays:
            a.arrays.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(a.arrays / f"{name.replace(':', '_side')}.npz",
                                **{key: np.stack(rows) for key, rows in arrays.items()})
        print(f"{time.time() - t0:7.0f} s  {name}", json.dumps(out["sides"][name]["summary"]["lanes"]), flush=True)
    context.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
