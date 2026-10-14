"""A game's decisions as digests: what the worker encodes and masks at each
decision, the action a seeded random legal player picks from those masks,
and the core's state after the step. A change that alters any of them (the
encoding, the masks, the view the player reads, the simulator's step)
changes the digests; tests/data/trajectory_digests.json holds them as the
code of 2026-10-14 (main at 0.21.1, core phase 32) played them."""
from __future__ import annotations

import hashlib
import random
from typing import Dict, List

import numpy as np

FIXTURE_SEEDS = (3, 11)
FIXTURE_DECISIONS = 120


def _array_digest(h, a) -> None:
    a = np.ascontiguousarray(a)
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(a.tobytes())


def encoding_digest(enc) -> str:
    """The raw streams, the parity streams and the packed masks of one
    encoded decision, in field order."""
    from wesnoth_ai.leaf_wire import MASK_FIELDS, PARITY_FIELDS, RAW_FIELDS
    h = hashlib.sha256()
    for name, _dt in RAW_FIELDS + PARITY_FIELDS:
        a = getattr(enc._raw, name, None)
        h.update(name.encode())
        if a is not None:
            _array_digest(h, a)
    for name, _dt in MASK_FIELDS:
        a = getattr(enc._masks, name)
        h.update(name.encode())
        if a is not None:
            _array_digest(h, a)
    m = enc._masks
    h.update(repr((float(m.end_turn_bias), m.n_units, m.n_recruits, m.n_hexes)).encode())
    h.update(repr((enc._raw.our_faction_id, enc._raw.their_faction_id, list(enc._raw.unit_ids))).encode())
    return h.hexdigest()[:16]


def view_digest(gs) -> str:
    """The units of the view the player reads, and its sides."""
    units = sorted((u.id, u.name, u.side, u.position.x, u.position.y, u.current_hp, u.current_moves,
                    u.current_exp, bool(u.has_attacked)) for u in gs.map.units)
    sides = [(s.current_gold, s.nb_villages_controlled) for s in gs.sides]
    gi = gs.global_info
    return hashlib.sha256(repr((units, sides, gi.current_side, gi.turn_number)).encode()).hexdigest()[:16]


def _action_text(action: Dict) -> str:
    return repr(sorted((k, repr(v)) for k, v in action.items()))


def play_digests(seed: int, decisions: int = FIXTURE_DECISIONS) -> List[List[str]]:
    """Per decision [encoding, view, action, state after the step] of a
    game the random legal player plays on the scenario `seed` draws,
    encoded as own1's recipe encodes (relevant set 2, terrain set,
    parity observation, the core's masks)."""
    from tools.inference_seam import RemoteEncoder
    from tools.random_positions import random_legal_action
    from tools.selfplay_game import _would_recruit_bounce
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate, random_setup
    rng = random.Random(seed)
    setup = random_setup(random.Random(seed))
    sim = WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id, max_turns=30)
    enc = RemoteEncoder({}, {}, relevant_set=True, fog_hides_enemy_villages=True, server_priors=True,
                        terrain_multi_hot=True, observation_parity=True, relevant_set_version=2)
    out = []
    for _ in range(decisions):
        if sim.done:
            break
        gs = sim.gs
        side = gs.global_info.current_side
        leaders = {u.side: (u.position.x, u.position.y) for u in gs.map.units if u.is_leader}
        e = enc.encode(gs)
        row = [encoding_digest(e), view_digest(gs)]
        action = random_legal_action(e, rng, aim=leaders.get(3 - side), keep=leaders.get(side))
        row.append(_action_text(action))
        if _would_recruit_bounce(action, sim.gs):
            tgt = action["target_hex"]
            sim.reject_recruit_hex(tgt.x, tgt.y)
        else:
            sim.step(action)
        row.append(str(sim.core.state_key()))
        out.append(row)
    return out
