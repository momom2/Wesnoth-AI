"""The inference server's host work per batch, timed on the CPU: the
Python path against the Rust stage (tools/serve_stage.py).

The leaves are positions a random legal player reaches on the Ladder maps,
encoded as own1 encodes them (relevant set 2, terrain set, parity
observation, the core's masks), each sent as its own request, as a
worker's game sends it, with a memory of 64 slots of width 384 (own1's):
the state on the wire (one leaf in ten at a game-side's first decision),
or with --server-memory kept on the server. Each stage is host work the
eval server's batch does (tools/eval_inference_server.py,
tools/inference_seam.py): reading the requests, the memory states, the
embedding stage, the packed trunk's index, the masks, reading the priors'
output buffer and writing the replies. The forward and the priors run on
the CPU between the stages and are left out of the stage times. Both
paths answer every batch, and their replies must be the same bytes.

    python tools/bench_serve_host.py --batch 40 --repeats 20 [--server-memory]
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

D_MODEL, SLOTS = 384, 64        # own1's width and memory
CID = 1                         # every request on one connection


def harvest_leaves(n: int, seed: int = 3) -> List[tuple]:
    """`n` (RawEncoded, PackedMasks) pairs from random legal games."""
    from tools.inference_seam import RemoteEncoder
    from tools.random_positions import random_legal_action
    from tools.selfplay_game import _would_recruit_bounce
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate, random_setup
    enc = RemoteEncoder({}, {}, relevant_set=True, fog_hides_enemy_villages=True, server_priors=True,
                        terrain_multi_hot=True, observation_parity=True, relevant_set_version=2)
    out: List[tuple] = []
    while len(out) < n:
        rng = random.Random(seed)
        setup = random_setup(random.Random(seed))
        sim = WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id, max_turns=30)
        for _ in range(150):
            if sim.done or len(out) >= n:
                break
            gs = sim.gs
            side = gs.global_info.current_side
            e = enc.encode(gs)
            out.append((e._raw, e._masks))
            leaders = {u.side: (u.position.x, u.position.y) for u in gs.map.units if u.is_leader}
            action = random_legal_action(e, rng, aim=leaders.get(3 - side), keep=leaders.get(side))
            if _would_recruit_bounce(action, sim.gs):
                tgt = action["target_hex"]
                sim.reject_recruit_hex(tgt.x, tgt.y)
                continue
            sim.step(action)
        seed += 1
    return out


def requests_of(leaves: List[tuple], served: bool, seed: int = 0) -> List[bytes]:
    """One request per leaf: its memory state on the wire (none for one
    leaf in ten), or with `served` the key of a state the server keeps
    (key = the leaf's index, version 0)."""
    from tools.serve_loop import KIND_INFER
    from wesnoth_ai.byte_wire import encode_request
    from wesnoth_ai.memory import MemoryState
    rng = np.random.default_rng(seed)
    out = []
    for i, (raw, masks) in enumerate(leaves):
        if served:
            out.append(encode_request([(raw, masks, MemoryState(SLOTS, None))], prefix=(KIND_INFER, i, 1),
                                      served=[(i, 0)]))
            continue
        state = None if i % 10 == 0 else rng.standard_normal((SLOTS, D_MODEL)).astype(np.float32)
        out.append(encode_request([(raw, masks, MemoryState(SLOTS, state))], prefix=(KIND_INFER, i, 1)))
    return out


class _Req:
    def __init__(self, rid, message):
        self.rid, self.message, self.leaves, self.cid = rid, message, 1, CID


def model_pair():
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.model import WesnothModel
    torch.manual_seed(0)
    model = WesnothModel(d_model=D_MODEL, num_layers=1, num_heads=4, d_ff=64, observation_parity=True,
                         memory_slots=SLOTS).eval()
    encoder = GameStateEncoder(d_model=D_MODEL, relevant_set_hexes=True, fog_hides_enemy_villages=True,
                               terrain_multi_hot=True, observation_parity=True,
                               relevant_set_version=2).eval()
    return model, encoder


def python_path(server, requests, T: Dict[str, list], memory=None) -> List[bytes]:
    """The eager server's host stages on one batch (tools/inference_seam.py
    `_serve_priors`, tools/eval_inference_server.py `_model_infer_fn`)."""
    from tools.inference_seam import ServedBatch
    from tools.serve_loop import REQUEST_HEAD_BYTES, STATUS_OK
    from tools.serve_stage import plan_handles
    from wesnoth_ai.byte_wire import decode_request, encode_replies
    from wesnoth_ai.server_priors import PendingPriors, _stage_masks, priors_outputs
    model, encoder = server._model, server._encoder
    cpu = torch.device("cpu")
    t = time.perf_counter
    t0 = t()
    items = []
    for r in requests:
        items.extend(decode_request(r.message, REQUEST_HEAD_BYTES))
    t1 = t()
    raws, packs = [it[0] for it in items], [it[1] for it in items]
    plan = None
    if memory is not None:
        plan = plan_handles(memory, items, [CID] * len(items))
        states = plan.memory_batch
    else:
        states = server._memory_batch(server._checked_memories([it[2] for it in items]))
    t2 = t()
    staged = encoder.stage_raws(raws, device=cpu)
    t3 = t()
    index = model.trunk_index(staged.sizes, staged.sighting_counts, states.counts, cpu, packed=True)
    t4 = t()
    with torch.no_grad():
        streams = encoder.embed_staged(staged)
        padded = model.forward_embedded(streams, memory=states, packed=False)
    A_max, T_ = padded.type_logits.shape[1], padded.type_logits.shape[2]
    H_max, W = padded.target_logits.shape[2], padded.weapon_logits.shape[2]
    t5 = t()
    layout, host, capacity = _stage_masks(packs, A_max, H_max, T_, W, False)
    t6 = t()
    with torch.no_grad():
        v = layout.torch_views(host)
        ride = [padded.value, padded.value_logits, padded.cliffness]
        if plan is None:
            ride.append(padded.memory_padded.float())
        out_layout, flat_out = priors_outputs(padded, v, capacity, ride)
    pending = PendingPriors(host=flat_out.clone(), layout=out_layout, n_samples=len(packs),
                            capacity=capacity, n_extras=len(ride), device=cpu)
    if plan is not None:
        memory.scatter(plan.writes, padded.memory_padded)
        memory.commit()
    t7 = t()
    compact, host_x = pending.finish()
    new_memory = None
    if plan is None:
        new_memory, host_x = host_x[-1], host_x[:-1]
    batch = ServedBatch(compact=compact, extras=dict(zip(["value", "value_logits", "cliffness"], host_x)),
                        actor_kind=padded.actor_kind.cpu().numpy(), sizes=[tuple(s) for s in padded.sizes],
                        memory=new_memory, memory_counts=None if plan else list(padded.memory_counts))
    t8 = t()
    replies = [encode_replies([batch.wire(b)], prefix=(r.rid, STATUS_OK),
                              versions=None if plan is None else plan.versions[b:b + 1])
               for b, r in enumerate(requests)]
    t9 = t()
    del index
    for k, dt in (("decode", t1 - t0), ("memory", t2 - t1), ("stage_embed", t3 - t2), ("trunk_index", t4 - t3),
                  ("masks", t6 - t5), ("finish", t8 - t7), ("replies", t9 - t8)):
        T[k].append(dt)
    return replies


def rust_path(server, requests, T: Dict[str, list], memory=None) -> List[bytes]:
    """The same batch through the Rust stage (tools/serve_stage.py); its
    host stages as `serve_requests` records them (on the CPU the forward
    and the priors fall in t_forward and t_priors, left out here; the
    memory gather, t_memory, is device work on CUDA, a few launches)."""
    from tools.serve_stage import serve_requests
    stats: Dict[str, float] = {}
    replies = serve_requests(server, requests, stats, memory=memory, cids=[r.cid for r in requests])
    for k in ("t_parse", "t_stage", "t_memory", "t_finish", "t_replies"):
        T[k].append(stats.get(k, 0.0))
    return replies


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--server-memory", action="store_true",
                    help="the memory states kept on the server (tools/serve_stage.py ServerMemory)")
    args = ap.parse_args(argv[1:])
    from tools.inference_seam import InferenceServer
    from tools.serve_stage import ServerMemory
    leaves = harvest_leaves(args.batch * 2)
    messages = requests_of(leaves, args.server_memory)
    model, encoder = model_pair()
    cpu = torch.device("cpu")
    server = InferenceServer(model, encoder, device=cpu, output_device=cpu, autocast_bf16=False,
                             packed_embed=True)
    memories = ((ServerMemory(model, cpu, max_keys=len(messages)), ServerMemory(model, cpu, max_keys=len(messages)))
                if args.server_memory else (None, None))
    rng = random.Random(1)
    T: Dict[str, list] = defaultdict(list)
    R: Dict[str, list] = defaultdict(list)
    for rep in range(args.repeats + 2):
        picked = rng.sample(range(len(messages)), args.batch)
        reqs = [_Req(i, messages[i]) for i in picked]
        if rep == 2:
            T.clear()
            R.clear()
        want = python_path(server, reqs, T, memories[0])
        got = rust_path(server, reqs, R, memories[1])
        if got != want:
            raise SystemExit(f"batch {rep}: the Rust stage's replies differ from the Python path's")
    mode = "server memory" if args.server_memory else "memory on the wire"
    for name, table in (("python", T), ("rust", R)):
        print(f"{name} host stages ({mode}), batch {args.batch}, median ms per batch of {args.repeats}:")
        medians = {k: 1e3 * float(np.median(v)) for k, v in table.items()}
        for k, v in medians.items():
            print(f"  {k:12s} {v:7.3f}")
        print(f"  {'sum':12s} {sum(medians.values()):7.3f}")
    print("the replies are the same bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
