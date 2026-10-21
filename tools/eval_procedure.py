"""Procedure provenance for eval results -- stdlib-only on purpose:
run_elo_batch is a long-lived torch-free driver, and importing the
tag helper from elo_eval_game pulled torch + the sim stack into it
(round-5 C9: +174 MB RSS in the batch process)."""
from __future__ import annotations

from typing import Optional


def end_turn_refusal(side: str, spec: str, sims: int, raw_temperature,
                     raw_end_turn: str, raw_end_turn_offset: float) -> Optional[str]:
    """Why side `side`'s end_turn decode flags cannot apply, or None.
    They are knobs of the joint-temperature raw player
    (tools/raw_player.py), which a side plays only at sims 0 with a raw
    temperature and a network. The search, the legacy sampler and the
    scripted 'dummy' never read them, while the result file recorded
    them as if they had played."""
    if raw_end_turn == "joint" and not raw_end_turn_offset:
        return None
    if sims > 0 or raw_temperature is None or spec == "dummy":
        return (f"--raw-end-turn-{side}/--raw-end-turn-offset-{side} decode the "
                f"joint-temperature raw player only: side {side} needs sims 0, "
                f"--raw-temperature-{side} and a network (the search, the legacy "
                f"sampler and 'dummy' never read them).")
    return None


COIN_P_DEFAULT = 0.25


def add_coin_args(ap) -> None:
    """The coin flags (tools/coin_decode.py) on an argparse parser, the same
    in the match driver and the per-game worker."""
    ap.add_argument("--attack-coin-offset", type=float, default=0.0,
                    help="Add this offset to every attack's logit for the one player the game's "
                         "attack coin chooses (P 1/2, keyed by the game seed; procedure tag "
                         "'+aoc<x>' on both sides); 0: off.")
    ap.add_argument("--coin-band-a", type=float, default=0.0,
                    help="Near-tie coins for player A: at a decision whose two best joint "
                         "log-priors differ by less than this many nats, a logged coin plays "
                         "the runner-up with probability --coin-p-a (raw player at argmax "
                         "only; procedure tag '+coin<w>p<p>'); 0: off.")
    ap.add_argument("--coin-p-a", type=float, default=COIN_P_DEFAULT,
                    help="The near-tie coin's propensity for player A.")
    ap.add_argument("--coin-band-b", type=float, default=0.0,
                    help="Player B (see --coin-band-a).")
    ap.add_argument("--coin-p-b", type=float, default=COIN_P_DEFAULT,
                    help="The near-tie coin's propensity for player B.")


def coin_argv(args) -> list:
    """The coin flags that hand the parsed `args`' coins to a child, the
    ones at their defaults left out."""
    out = []
    if args.attack_coin_offset:
        out += ["--attack-coin-offset", str(float(args.attack_coin_offset))]
    for side in ("a", "b"):
        band = getattr(args, f"coin_band_{side}")
        if band:
            out += [f"--coin-band-{side}", str(float(band)),
                    f"--coin-p-{side}", str(float(getattr(args, f"coin_p_{side}")))]
    return out


def coin_refusal(side: str, spec: str, sims: int, raw_temperature, raw_end_turn: str,
                 coin_band: float, coin_p: float, attack_coin_offset: float) -> Optional[str]:
    """Why side `side`'s coin flags cannot apply, or None. The near-tie
    coins (`--coin-band-<side>`, tools/coin_decode.py) decide at the joint
    argmax of the raw player (sims 0, temperature 0, rule 'joint', a
    network), with a propensity strictly between 0 and 1. The attack coin
    (`--attack-coin-offset`) may put its offset on either side, so each side
    must be a raw player at a temperature with a network."""
    if coin_band < 0:
        return f"--coin-band-{side} must be >= 0 (0: no coins)"
    raw = sims <= 0 and raw_temperature is not None and spec != "dummy"
    if coin_band:
        if not raw or float(raw_temperature) != 0.0 or raw_end_turn != "joint":
            return (f"--coin-band-{side}: the near-tie coins decide at the joint argmax of the raw "
                    f"player: side {side} needs sims 0, --raw-temperature-{side} 0, the joint "
                    f"end_turn rule and a network")
        if not 0.0 < coin_p < 1.0:
            return f"--coin-p-{side} must lie strictly between 0 and 1"
    if attack_coin_offset and not raw:
        return (f"--attack-coin-offset puts the attack offset on either side through the raw "
                f"player's decode: side {side} needs sims 0, --raw-temperature-{side} and a network")
    return None


def procedure_of(sims: int, plan: bool, no_turn_search: bool,
                 raw_temperature=None, gumbel_root: bool = True, *,
                 raw_end_turn: str = "joint", raw_end_turn_offset: float = 0.0,
                 attack_coin_offset: float = 0.0, coin_band: float = 0.0,
                 coin_p: float = 0.0) -> str:
    """Canonical procedure tag for result provenance. Carries the
    sims budget (round-13 C2: 'mcts' alone let an outdir silently
    mix --mcts-sims 16 and 32 games -- different estimands), the
    root procedure of a plain search ('mcts' = Gumbel root, 'puct' =
    plain PUCT root, what the az legs trained with) and, for the raw
    player, its joint sampling temperature (tools/raw_player.py;
    None = the legacy factored sampler), its end_turn decode
    ('+endm' = decided at the actor level, '+eo<x>' = the end_turn
    logit offset x; panel test 1) and its coins ('+aoc<x>' = the attack
    offset x on the side the game's attack coin chose, which both
    sides' tags carry; '+coin<w>p<p>' = the near-tie coins at band width
    w and propensity p, tools/coin_decode.py), so those estimands never
    share an outdir either."""
    if sims <= 0:
        if raw_temperature is None:
            return "raw"
        tag = f"raw:t{float(raw_temperature):g}"
        if raw_end_turn == "actor":
            tag += "+endm"
        if raw_end_turn_offset:
            tag += f"+eo{float(raw_end_turn_offset):g}"
        if attack_coin_offset:
            tag += f"+aoc{float(attack_coin_offset):g}"
        if coin_band:
            tag += f"+coin{float(coin_band):g}p{float(coin_p):g}"
        return tag
    if plan:
        name = "plan_tournament"
    elif no_turn_search:
        name = "mcts" if gumbel_root else "puct"
    else:
        name = "tcs"
    return f"{name}:{sims}"
