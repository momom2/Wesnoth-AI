"""Synthetic coin games whose decision effects are known, in the forms the
coin label code reads: a result file's fields, a game record's coins, and
the luck walk's row (tools/coin_labels.py, tools/coin_plant.py,
tools/match_luck.py)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from tools.match_luck import COLUMNS

P = 0.25
W = 0.35
# Ordered pair types (argmax kind, runner-up kind) with the runner-up's
# effect for the mover, c + b G, in result units.
EFFECTS = {("move", "move"): (0.08, 0.0), ("move", "attack"): (-0.03, 0.04), ("end_turn", "move"): (0.0, 0.06)}
TYPES = list(EFFECTS)
# A runner-up attack's coin landing 1 brings one more band decision of the
# same mover, of this type (the policy re-decides after the roll).
EXTRA = ("end_turn", "move")
# z's slope on side A's hp luck in these games (the dice control's).
DICE_SLOPE = 0.0042


def effect(pair, gap) -> float:
    c, b = EFFECTS[pair]
    return c + b * gap


# The extra decision's expected effect, its gap uniform on [0, W).
EXTRA_MEAN = EFFECTS[EXTRA][0] + EFFECTS[EXTRA][1] * W / 2


def entry(side, i, gap, d, pair) -> dict:
    return {"side": side, "i": i, "cmd": None, "gap": gap, "p": P, "d": d,
            "a0": {"kind": pair[0]}, "a1": {"kind": pair[1]}}


def _luck_columns(rng, big: np.ndarray, small: np.ndarray, extra_hp: float = 0.0) -> dict:
    noise = rng.normal(0, 1, len(COLUMNS))
    return {"hp": big.sum() + 3.0 * small.sum() + extra_hp, "material": 0.5 * big.sum() + small.sum() + noise[1],
            "material_nonterminal": noise[2] * 20, "terminal": 0.01 * big.sum(),
            "value_50": 0.004 * big.sum() + 0.1 * noise[4], "value_100": 0.002 * big.sum(), "value_200": noise[6]}


def _forms(match: str, game: str, seed: int, side_a: int, z: int, coins: dict, S: dict):
    outcome = {1: "win", -1: "loss", 0: "timeout"}[z]
    result = {"outcome_a": outcome, "seed": seed, "side_a": side_a, "ended_by": "leader_killed" if z else "max_turns"}
    record = {"game_label": game, "winner": 0 if z == 0 else (side_a if z == 1 else 3 - side_a),
              "players": {"a": {"side": side_a}, "b": {"side": 3 - side_a}}, "commands": [], "coins": coins}
    luck = {"match": match, "game": game, "z": z, "decisive": z != 0, "S": S,
            "counts": {"attacks": 10, "enumerated": 10, "outside_support": 0, "not_enumerable": 0,
                       "counter_weapon_differs": 0, "no_units": 0}}
    return result, record, luck


def coin_rows_of(games, match: str = "m") -> list:
    """The coin rows tools/coin_rows.py extracts from these games (results
    and records as `synthetic_game` and `attack_game` return them), its
    manifest last."""
    from tools.coin_rows import extract_game
    rows = [extract_game(match, g[1]["game_label"], g[0], g[1]) for g in games]
    return rows + [{"match": match, "manifest": {"results": len(games)}}]


def synthetic_game(rng, k: int, leak: float = 0.0, match: str = "m", game: Optional[str] = None,
                   side_a: Optional[int] = None):
    """(result, record, luck row, truths) of one synthetic game. The movers
    alternate every four decisions; each band decision's coin plays the
    runner-up with probability P, adding s * (c + b G) to side A's mean
    result; a runner-up attack played brings an extra band decision. The
    dice: two large fights and 8 + 2 x (coins landed 1) small ones, each of
    mean zero whatever the coins, read by the seven luck columns. z is +1
    with probability (1 + mean) / 2, and 1 game in 100 is capped. `leak`
    adds the coins' signed count to the hp column: a variate read after the
    coin. `truths` holds each decision's effect under the coin policy."""
    side_a = side_a or (1 if k % 2 == 0 else 2)
    entries, truths, mean, ones = [], [], 0.03 * (1 if side_a == 1 else -1), 0
    n_base, i_side = 16, {1: 0, 2: 0}
    kinds = rng.integers(len(TYPES), size=n_base)
    pending = [(1 + (j // 4) % 2, TYPES[int(kinds[j])]) for j in range(n_base)]
    gaps, coins = rng.uniform(0, W, 2 * n_base).tolist(), (rng.random(2 * n_base) < P).tolist()
    leaked = 0.0
    while pending:
        side, pair = pending.pop(0)
        gap, d = gaps[len(entries)], int(coins[len(entries)])
        s = 1 if side == side_a else -1
        truths.append(effect(pair, gap) + (P * EXTRA_MEAN if pair[1] == "attack" else 0.0))
        entries.append(entry(side, i_side[side], gap, d, pair))
        i_side[side] += 1
        mean += s * effect(pair, gap) * d
        ones += d
        leaked += s * d
        if d and pair[1] == "attack":
            pending.insert(0, (side, EXTRA))
    big = 50.0 * rng.uniform(-1, 1, 2)
    small = rng.uniform(-1, 1, 8 + 2 * ones)
    mean += DICE_SLOPE * big.sum() + 0.004 * small.sum()
    z = 1 if rng.random() < (1 + np.clip(mean, -1, 1)) / 2 else -1
    if rng.random() < 0.01:
        z = 0
    coins = {"bands": {"a": {"band": W, "p": P}, "b": {"band": W, "p": P}}, "decisions": entries}
    S = _luck_columns(rng, big, small, leak * leaked)
    return (*_forms(match, game or f"game_s{side_a}_{k}", k, side_a, z, coins, S), truths)


def attack_game(rng, k: int, hook_effect: float, match: str = "m", game: Optional[str] = None,
                side_a: Optional[int] = None):
    """(result, record, luck row) of one game whose attack coin hooks player
    A (d 1) or B (d 0), the hooked player's mean result lowered by
    `hook_effect`."""
    side_a = side_a or 1 + k % 2
    d = int(rng.random() < 0.5)
    big = 50.0 * rng.uniform(-1, 1, 2)
    small = rng.uniform(-1, 1, 8)
    mean = (-hook_effect if d else hook_effect) + 0.03 * (1 if side_a == 1 else -1) + DICE_SLOPE * big.sum()
    z = 1 if rng.random() < (1 + mean) / 2 else -1
    coins = {"bands": {}, "decisions": [], "attack": {"offset": 1.0, "p": 0.5, "d": d, "seat": "a" if d else "b"}}
    return _forms(match, game or f"g{k}", k, side_a, z, coins, _luck_columns(rng, big, small))
