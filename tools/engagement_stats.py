"""Per-game engagement telemetry (user spec 2026-07-12).

Instruments the causal chain of winning — scout -> expand -> build ->
engage -> convert -> finish — as per-game counters logged during
ONLINE training (no anatomy runs, no per-turn persistence).

Event flow: `replay_dataset`'s attack handler and healing loop emit
typed events through a THREAD-LOCAL sink. `WesnothSim` sets the sink
only around its own `_apply_command` calls and only when stats are
enabled on that instance (`sim.enable_engagement_stats()`), so:

  - MCTS search forks pay NOTHING (`fork()` never carries the
    accumulator, so no sink is ever installed for fork steps);
  - the in-process `--workers` THREAD mode cannot cross-talk
    (thread-local sink);
  - replay reconstruction / parity tooling is unaffected (no sink
    installed -> `emit_event` is a no-op attribute read).

Attribution rules (user 2026-07-12):
  - healing counts ACTUAL applied HP (post-cap), never theoretical;
  - rest heal is attributed first, the remainder goes to the main
    source (deterministic, no double count);
  - the main source is the first of village/oasis > regen > healer
    whose amount equals the applied max (village and oasis share the
    "village" bucket; regen and adjacent-healer share "ability").
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

_sink_tls = threading.local()


def set_event_sink(fn) -> None:
    _sink_tls.fn = fn


def clear_event_sink() -> None:
    _sink_tls.fn = None


def emit_event(kind: str, **payload) -> None:
    """Called from replay_dataset hot paths; must be near-free when
    no sink is installed (the overwhelmingly common case)."""
    fn = getattr(_sink_tls, "fn", None)
    if fn is not None:
        fn(kind, payload)


def _sides() -> Dict[int, int]:
    return {1: 0, 2: 0}


def _sidesf() -> Dict[int, float]:
    return {1: 0.0, 2: 0.0}


@dataclass
class EngagementStats:
    """Per-game accumulator. All dicts keyed by side (1/2)."""

    attacks_attempted:       Dict[int, int] = field(default_factory=_sides)
    # Tripwire: attacks that Wesnoth-as-played would refuse (target
    # missing / own-side / scenery-side / petrified — the UI gate at
    # mouse_events.cpp:753). Classified from RAW unit state at
    # execution time, independent of the legality mask. Expect 0.
    attacks_invalid_wesnoth: Dict[int, int] = field(default_factory=_sides)
    # Attacks the sim's own gate refused (defense in depth).
    attacks_rejected_sim:    Dict[int, int] = field(default_factory=_sides)
    damage_dealt:            Dict[int, int] = field(default_factory=_sides)
    # killer side -> victim unit-type -> count; value = summed cost.
    kills: Dict[int, Dict[str, int]] = field(
        default_factory=lambda: {1: {}, 2: {}})
    kills_value:             Dict[int, int] = field(default_factory=_sides)
    advancements:            Dict[int, int] = field(default_factory=_sides)
    heal_village:            Dict[int, int] = field(default_factory=_sides)
    heal_rest:               Dict[int, int] = field(default_factory=_sides)
    heal_ability:            Dict[int, int] = field(default_factory=_sides)
    # Anti-poison economy (2026-07-12): cures are invisible in the
    # healing buckets (curing REPLACES healing), so count them
    # explicitly. poison_damage_taken = NET actual HP lost on
    # poison-normal turns (rest folded in per the simultaneous-
    # application rule; the combined clamp makes any finer split
    # arbitrary).
    poison_cured:            Dict[int, int] = field(default_factory=_sides)
    poison_damage_taken:     Dict[int, int] = field(default_factory=_sides)
    first_contact_turn:      Optional[int] = None
    # Scouting: fraction of map visible RIGHT BEFORE each of the
    # side's end_turns (hexes revealed during a turn can re-hide
    # after it). Averaged per game. Fogged games only.
    _scout_sum:              Dict[int, float] = field(default_factory=_sidesf)
    _scout_n:                Dict[int, int] = field(default_factory=_sides)
    # Unused-MP fraction: unspent / total MP across the side's units,
    # sampled at each of its end_turns, averaged per game.
    _mp_unused:              Dict[int, int] = field(default_factory=_sides)
    _mp_total:               Dict[int, int] = field(default_factory=_sides)

    # ---- event sink (combat + heal, from replay_dataset) ----------
    def on_event(self, kind: str, p: dict) -> None:
        if kind == "combat":
            a, d = p["a_side"], p["d_side"]
            if a in (1, 2):
                self.damage_dealt[a] = (self.damage_dealt.get(a, 0)
                                        + p["dmg_to_defender"])
            if d in (1, 2):
                self.damage_dealt[d] = (self.damage_dealt.get(d, 0)
                                        + p["dmg_to_attacker"])
            if p["defender_died"] and a in (1, 2):
                bucket = self.kills.setdefault(a, {})
                bucket[p["defender_name"]] = \
                    bucket.get(p["defender_name"], 0) + 1
                self.kills_value[a] = (self.kills_value.get(a, 0)
                                       + int(p["defender_cost"]))
            if p["attacker_died"] and d in (1, 2):
                bucket = self.kills.setdefault(d, {})
                bucket[p["attacker_name"]] = \
                    bucket.get(p["attacker_name"], 0) + 1
                self.kills_value[d] = (self.kills_value.get(d, 0)
                                       + int(p["attacker_cost"]))
        elif kind == "heal":
            s = p["side"]
            if s in (1, 2):
                self.heal_village[s] += p.get("village", 0)
                self.heal_rest[s] += p.get("rest", 0)
                self.heal_ability[s] += p.get("ability", 0)
        elif kind == "poison":
            s = p["side"]
            if s in (1, 2):
                if p.get("cured"):
                    self.poison_cured[s] += 1
                self.poison_damage_taken[s] += p.get("damage", 0)

    # ---- sim-side hooks --------------------------------------------
    def note_attack_attempt(self, gs, action) -> None:
        side = gs.global_info.current_side
        if side not in (1, 2):
            return
        self.attacks_attempted[side] += 1
        if self.first_contact_turn is None:
            self.first_contact_turn = gs.global_info.turn_number
        # Wesnoth-as-played validity, from RAW unit state (never the
        # mask/encoder): target exists, is a hostile player-side
        # unit, and is not incapacitated (mouse_events.cpp:753).
        t = action.get("target_hex")
        dfd = None
        if t is not None:
            dfd = next((u for u in gs.map.units
                        if u.position.x == t.x and u.position.y == t.y),
                       None)
        invalid = (dfd is None
                   or dfd.side == side
                   or dfd.side not in (1, 2)
                   or "petrified" in (dfd.statuses or set()))
        if invalid:
            self.attacks_invalid_wesnoth[side] += 1

    def note_end_turn(self, gs, side: int) -> None:
        if side not in (1, 2):
            return
        units = [u for u in gs.map.units if u.side == side]
        if units:
            self._mp_unused[side] += sum(u.current_moves for u in units)
            self._mp_total[side] += sum(u.max_moves for u in units)
        if getattr(gs.global_info, "_fog", True):
            from visibility import visible_fraction_for
            self._scout_sum[side] += visible_fraction_for(gs, side)
            self._scout_n[side] += 1

    # ---- reduction ---------------------------------------------------
    def to_dict(self) -> dict:
        def frac(num, den):
            return {s: (num[s] / den[s] if den[s] else None)
                    for s in (1, 2)}
        return {
            "attacks_attempted": dict(self.attacks_attempted),
            "attacks_invalid_wesnoth": dict(self.attacks_invalid_wesnoth),
            "attacks_rejected_sim": dict(self.attacks_rejected_sim),
            "damage_dealt": dict(self.damage_dealt),
            "kills": {s: dict(v) for s, v in self.kills.items()},
            "kills_value": dict(self.kills_value),
            "advancements": dict(self.advancements),
            "heal_village": dict(self.heal_village),
            "heal_rest": dict(self.heal_rest),
            "heal_ability": dict(self.heal_ability),
            "poison_cured": dict(self.poison_cured),
            "poison_damage_taken": dict(self.poison_damage_taken),
            "first_contact_turn": self.first_contact_turn,
            "scouted_frac": frac(self._scout_sum, self._scout_n),
            "unused_mp_frac": frac(self._mp_unused, self._mp_total),
        }
