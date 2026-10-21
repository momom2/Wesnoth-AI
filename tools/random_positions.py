"""Positions bound to the Rust core, from games a random legal player
plays on the Ladder maps: scenario starts (recruits), midgame positions
with moves, attacks and sightings of enemy units gone back into the fog,
and one position per game with a recruit rejected on the castle. The
tests encode them (tests/helpers/core_positions.py), and
tools/graphed_check.py serves them through a checkpoint on a box.
"""
from __future__ import annotations

import random

import numpy as np


def random_legal_action(encoded, rng: random.Random, aim=None, keep=None, act: float = 0.95) -> dict:
    """An action drawn from the legal ones the encoding's packed masks
    admit: an attack half the time there is one, a recruit most of the
    time there is one, else a move, toward `aim` (x, y) most of the time,
    never by the unit standing on `keep`; end_turn with probability
    1 - `act` or when nothing else is legal."""
    from wesnoth_ai.model import UnitActionType
    from tools.abilities import hex_distance
    m = encoded._masks
    U, R, H = m.n_units, m.n_recruits, m.n_hexes
    att = np.unpackbits(m.attack_valid, axis=1)[:, :H]
    mv = np.unpackbits(m.move_valid, axis=1)[:, :H]
    un = np.unpackbits(m.union_valid, axis=1)[:, :H]
    attacks, moves, recruits = [], [], []
    for a in range(U):
        if not m.actor_mask[a]:
            continue
        start = encoded.unit_positions[a]
        if m.type_valid[a, UnitActionType.ATTACK]:
            for h in np.flatnonzero(att[a]).tolist():
                for w in range(int(m.n_attacks[a])):
                    attacks.append({"type": "attack", "start_hex": start,
                                    "target_hex": encoded.hex_positions[h], "attack_index": w})
        if m.type_valid[a, UnitActionType.MOVE] and (start.x, start.y) != keep:
            moves += [{"type": "move", "start_hex": start, "target_hex": encoded.hex_positions[h]}
                      for h in np.flatnonzero(mv[a]).tolist()]
    for r in range(R):
        if m.actor_mask[U + r]:
            recruits += [{"type": "recruit", "unit_type": encoded.recruit_types[r],
                          "target_hex": encoded.hex_positions[h]} for h in np.flatnonzero(un[U + r]).tolist()]
    if rng.random() >= act or not (attacks or moves or recruits):
        return {"type": "end_turn"}
    if attacks and rng.random() < 0.5:
        return rng.choice(attacks)
    if recruits and (not moves or rng.random() < 0.7):
        return rng.choice(recruits)
    if not moves:
        return rng.choice(attacks) if attacks else {"type": "end_turn"}
    if aim is not None and rng.random() < 0.85:
        return min(moves, key=lambda o: (hex_distance(o["target_hex"].x, o["target_hex"].y, *aim),
                                         rng.random()))
    return rng.choice(moves)


def harvest(seed: int, decisions: int, every: int, max_turns: int):
    """Snapshots (views of forks of the simulator's core) taken every
    `every` decisions of a game the random legal player plays."""
    from tools.inference_seam import RemoteEncoder
    from tools.selfplay_game import _would_recruit_bounce
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.game_core import core_of, snapshot_view
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate, random_setup
    rng = random.Random(seed)
    setup = random_setup(random.Random(seed))
    sim = WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id,
                     max_turns=max_turns)
    renc = RemoteEncoder({}, {}, server_priors=True)
    parity = RemoteEncoder({}, {}, relevant_set=True, fog_hides_enemy_villages=True,
                           terrain_multi_hot=True, observation_parity=True, relevant_set_version=2)
    out = []
    rejected = False
    sighted = 0
    for k in range(decisions):
        if sim.done:
            break
        sightings = parity.raw_of(sim.gs).sight_xs.shape[0]
        # The first decisions (the recruits), two per `every` decisions, and
        # up to three positions with a sighting.
        if k in (0, 2) or k % every in (every // 2, every - 1) or (sightings and sighted < 3 and k % 3 == 0):
            out.append(snapshot_view(sim.gs))
            sighted += bool(sightings)
        if not rejected and k >= every:
            # A recruit the side tried and the engine refused: the
            # rejection set the mask reads (principle 6).
            obs = core_of(sim.gs).observe(sim.gs.global_info.current_side, reach=True)
            castle = np.flatnonzero(obs.recruit_row)
            if len(castle):
                x, y = obs.geometry.keys[int(castle[0])]
                sim.reject_recruit_hex(x, y)
                out.append(snapshot_view(sim.gs))
                rejected = True
        side = sim.gs.global_info.current_side
        leaders = {u.side: (u.position.x, u.position.y) for u in sim.gs.map.units if u.is_leader}
        action = random_legal_action(renc.encode(sim.gs), rng, aim=leaders.get(3 - side),
                                     keep=leaders.get(side))
        if _would_recruit_bounce(action, sim.gs):
            tgt = action["target_hex"]
            sim.reject_recruit_hex(tgt.x, tgt.y)
            continue
        sim.step(action)
    return out
