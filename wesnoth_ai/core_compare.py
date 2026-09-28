"""Two game states compared over what the Rust core models: the units
field for field (their underscore attributes optionally), the sides,
the turn scalars, the stash the core keeps on `global_info`, the map
(hex set, terrain codes, time areas) and the scenario's event state
(latches, WML variables, stored locations). The certification of the
core against the Python applier runs on it (tools/diff_core.py,
tests/test_game_core.py)."""
from __future__ import annotations

from typing import List

from wesnoth_ai.classes import GameState, Unit
from wesnoth_ai.game_core import MODELED_GLOBALS, UNIT_STASH_KEYS


def units_equal(a: Unit, b: Unit) -> bool:
    """Field-level equality of two units (the dataclass compares ids only)."""
    if a.__dict__.keys() - {k for k in a.__dict__ if k.startswith("_")} != \
            b.__dict__.keys() - {k for k in b.__dict__ if k.startswith("_")}:
        return False
    for k, v in a.__dict__.items():
        if k.startswith("_"):
            continue
        w = getattr(b, k)
        if k == "attacks":
            if len(v) != len(w) or any(
                    (p.type_id, p.number_strikes, p.damage_per_strike, p.is_ranged, set(p.weapon_specials or ()))
                    != (q.type_id, q.number_strikes, q.damage_per_strike, q.is_ranged, set(q.weapon_specials or ()))
                    for p, q in zip(v, w)):
                return False
        elif isinstance(v, (set, frozenset)):
            if set(map(str, v)) != set(map(str, w)):
                return False
        elif v != w:
            return False
    return True


def _normalized_global(k: str, va, vb):
    """The two values of a modeled global in comparable form (the Python
    applier and the core spell an unset value differently)."""
    if k in ("_advance_uniform", "_did_first_init_side", "_fog"):
        return bool(va), bool(vb)
    if k in ("_tod_start_offset", "_rng_request_counter", "_advance_counter"):
        return int(va or 0), int(vb or 0)
    if k == "_advance_salt":
        return str(va or ""), str(vb or "")
    if k == "_experience_modifier":
        return int(va or 100), int(vb or 100)
    if k == "_next_uid_counter":
        return int(va or 1), int(vb or 1)
    if k in ("_uncovered_units", "_recruit_rejected_hexes"):
        return set(va or ()), set(vb or ())
    if k in ("_village_owner", "_pickadvance_game"):
        return ({kk: v for kk, v in (va or {}).items() if v}, {kk: v for kk, v in (vb or {}).items() if v})
    if k in ("_advance_choices", "_last_advance_events"):
        return ([tuple(x) if isinstance(x, (list, tuple)) else x for x in (va or [])],
                [tuple(x) if isinstance(x, (list, tuple)) else x for x in (vb or [])])
    if k == "_last_checkup_strikes":
        return va or None, vb or None
    if k == "_fog_cleared":
        return ({s: frozenset(h) for s, h in (va or {}).items()},
                {s: frozenset(h) for s, h in (vb or {}).items()})
    return va, vb


def _hex_facts(gs: GameState) -> dict:
    return {(h.position.x, h.position.y): (sorted(int(t) for t in h.terrain_types),
                                           sorted(int(m) for m in h.modifiers), int(h.terrain_mask))
            for h in gs.map.hexes}


def map_differences(a: GameState, b: GameState) -> List[str]:
    """The hex set's facts, the terrain codes and the time areas."""
    diffs: List[str] = []
    ha, hb = _hex_facts(a), _hex_facts(b)
    if ha != hb:
        bad = sorted(p for p in ha.keys() | hb.keys() if ha.get(p) != hb.get(p))
        diffs.append(f"hexes: {bad[:6]}")
    ca = getattr(a.global_info, "_terrain_codes", None) or {}
    cb = getattr(b.global_info, "_terrain_codes", None) or {}
    if ca != cb:
        bad = sorted(p for p in ca.keys() | cb.keys() if ca.get(p) != cb.get(p))
        diffs.append(f"terrain codes: {[(p, ca.get(p), cb.get(p)) for p in bad[:4]]}")
    ta = {p: list(c) for p, c in (getattr(a.global_info, "_time_areas", None) or {}).items()}
    tb = {p: list(c) for p, c in (getattr(b.global_info, "_time_areas", None) or {}).items()}
    if ta != tb:
        bad = sorted(p for p in ta.keys() | tb.keys() if ta.get(p) != tb.get(p))
        diffs.append(f"time areas: {[(p, ta.get(p), tb.get(p)) for p in bad[:4]]}")
    return diffs


def event_differences(a: GameState, b: GameState) -> List[str]:
    """The events' latches, the WML variables and the stored locations."""
    diffs: List[str] = []
    ga, gb = a.global_info, b.global_info
    fa = [(ev.name, bool(ev.fired)) for ev in (getattr(ga, "_scenario_events", None) or ())]
    fb = [(ev.name, bool(ev.fired)) for ev in (getattr(gb, "_scenario_events", None) or ())]
    if fa != fb:
        diffs.append(f"events fired: {fa} != {fb}")
    wa = dict(getattr(ga, "_wml_variables", None) or {})
    wb = dict(getattr(gb, "_wml_variables", None) or {})
    if wa != wb:
        diffs.append(f"wml variables: {wa} != {wb}")
    sa = {k: set(v) for k, v in (getattr(ga, "_scenario_vars", None) or {}).items()}
    sb = {k: set(v) for k, v in (getattr(gb, "_scenario_vars", None) or {}).items()}
    if sa != sb:
        diffs.append(f"stored locations: {sorted(sa.keys() ^ sb.keys()) or 'contents'}")
    return diffs


def observation_differences(a, b) -> List[str]:
    """The fields of two `observe.Observation` records that differ, the
    per-unit arrays compared by unit id (each side keeps its own unit
    order)."""
    import numpy as np
    from wesnoth_ai.observe import _ARRAY_FIELDS
    out: List[str] = []
    if (a.side, a.fog_on, a.leader_on_keep) != (b.side, b.fog_on, b.leader_on_keep):
        out.append("observation scalars")
    if sorted(a.unit_ids) != sorted(b.unit_ids):
        return out + ["observation unit ids"]
    perm = [a.unit_ids.index(uid) for uid in b.unit_ids]
    per_unit = ("unit_hex", "visible", "acting", "unit_can_move", "unit_can_attack", "landable")
    for k in _ARRAY_FIELDS:
        x, y = getattr(a, k), getattr(b, k)
        if x is None or y is None:
            if (x is None) != (y is None):
                out.append(f"observation {k}")
            continue
        if k in per_unit:
            x = x[perm]
        if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y):
            out.append(f"observation {k}")
    return out


def encoding_differences(a, b) -> List[str]:
    """The fields of two `encoder.RawEncoded` records that differ,
    arrays byte for byte."""
    import dataclasses
    import numpy as np
    from wesnoth_ai.encoder import RawEncoded
    out: List[str] = []
    for f in dataclasses.fields(RawEncoded):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if f.name == "observation":
            if (x is None) != (y is None):
                out.append("observation")
            elif x is not None:
                out += observation_differences(x, y)
        elif isinstance(x, np.ndarray):
            if not (isinstance(y, np.ndarray) and x.dtype == y.dtype and x.shape == y.shape
                    and x.tobytes() == y.tobytes()):
                out.append(f.name)
        elif x != y:
            out.append(f.name)
    return out


def state_differences(a: GameState, b: GameState, *, stash: bool = True,
                      map_and_events: bool = True) -> List[str]:
    """The differences between two states over the modeled content, as
    strings (empty when equal). `map_and_events` False leaves out the map
    and the event state, which only a setup, an init_side or an end_turn
    changes."""
    diffs: List[str] = []
    ua = {u.id: u for u in a.map.units}
    ub = {u.id: u for u in b.map.units}
    if ua.keys() != ub.keys():
        diffs.append(f"units: {sorted(ua.keys() ^ ub.keys())}")
    for uid in ua.keys() & ub.keys():
        if not units_equal(ua[uid], ub[uid]):
            diffs.append(f"unit {uid}: {ua[uid]} != {ub[uid]}")
        if stash:
            for k in UNIT_STASH_KEYS:
                if getattr(ua[uid], k, None) != getattr(ub[uid], k, None):
                    diffs.append(f"unit {uid} stash {k}")
    if len(a.sides) != len(b.sides):
        diffs.append("sides count")
    for i, (s, t) in enumerate(zip(a.sides, b.sides)):
        if (s.player, list(s.recruits), s.current_gold, s.base_income, s.nb_villages_controlled, s.faction) != \
                (t.player, list(t.recruits), t.current_gold, t.base_income, t.nb_villages_controlled, t.faction):
            diffs.append(f"side {i + 1}: {s} != {t}")
    ga, gb = a.global_info, b.global_info
    for k in ("current_side", "turn_number", "time_of_day", "village_gold", "village_upkeep", "base_income"):
        if getattr(ga, k) != getattr(gb, k):
            diffs.append(f"global {k}: {getattr(ga, k)} != {getattr(gb, k)}")
    for k in MODELED_GLOBALS:
        va, vb = _normalized_global(k, getattr(ga, k, None), getattr(gb, k, None))
        if va != vb:
            diffs.append(f"global {k}: {va!r} != {vb!r}")
    if (a.game_over, a.winner) != (b.game_over, b.winner):
        diffs.append("game over / winner")
    if not map_and_events:
        return diffs
    return diffs + map_differences(a, b) + event_differences(a, b)


__all__ = ["units_equal", "state_differences", "map_differences", "event_differences",
           "observation_differences", "encoding_differences"]
