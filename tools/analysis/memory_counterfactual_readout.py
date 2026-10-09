#!/usr/bin/env python3
"""The readings of docs/memory_in_play_parity3_prereg_20261009.md.

From the rows of tools/analysis/memory_counterfactual.py, per source and
turn bucket: how often the memory player (m) and the 0-slot player (z)
choose alike, how their disagreements split (only m ends the turn, only z
does, both act differently), their mean end_turn and best-other priors, each
of these again by the mover's standing (behind, level, ahead) and by the
side's outcome (won, lost, capped); the reproduction rate of a match source;
on a human source the recorded label's prior under each and its log-prior
gain, by label kind and by the side's outcome. Means of a game-level
quantity carry the standard error between games.

From the fits of the three matches (tools/elo_collect.py --save-json; each
candidate against the 0-slot player): the pre-registered reading of T, L
and Q. From the rows: the reading of W.

    python tools/analysis/memory_counterfactual_readout.py own=cf_own.jsonl \\
        other=cf_other.jsonl human=cf_human.jsonl \\
        --fit A1=a1.fit.json --fit A2=a2.fit.json --fit A3=a3.fit.json \\
        --opponent parity3_slots0 [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

BUCKETS = ((1, 5), (6, 10), (11, 15), (16, 10 ** 6))
ALL = (1, 10 ** 6)
LATE = (6, 10 ** 6)                          # "turns 6 and later"
STANDINGS = ("behind", "level", "ahead")
OUTCOMES = ("won", "lost", "capped")
TOOL_CHECK_REPRODUCTION = 0.95               # below it on own, no row reading is read
W_AGREEMENT_GAP = 0.05                       # W: agreement ahead minus behind, on own, all turns
FULL_PLAY = "PURE (decisive only, primary)"


def bucket_name(lo: int, hi: int) -> str:
    if (lo, hi) == ALL:
        return "all"
    if (lo, hi) == LATE:
        return "late (6+)"
    return f"{lo}+" if hi >= 10 ** 6 else f"{lo}-{hi}"


def mean_se(xs: Iterable[float]) -> Tuple[Optional[float], Optional[float]]:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, None
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var / len(xs))


def rounded(pair: Tuple[Optional[float], Optional[float]]) -> List[Optional[float]]:
    return [None if v is None else round(v, 5) for v in pair]


def per_game(rows: List[Dict], value: Callable[[Dict], Optional[float]]) -> Dict[str, float]:
    """Per game, the mean of value(row) over the rows that define it."""
    acc: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        v = value(r)
        if v is not None:
            acc[r["game"]].append(float(v))
    return {g: sum(v) / len(v) for g, v in acc.items()}


def game_mean(rows: List[Dict], value) -> List[Optional[float]]:
    return rounded(mean_se(per_game(rows, value).values()))


def ends(choice: Optional[str]) -> bool:
    return choice == "end_turn"


def only_m_ends(r: Dict) -> bool:
    return ends(r["choice_m"]) and not ends(r["choice_z"])


def only_z_ends(r: Dict) -> bool:
    return ends(r["choice_z"]) and not ends(r["choice_m"])


def split(rows: List[Dict]) -> Dict:
    """The decisions' agreement and the split of their disagreements."""
    n = len(rows)
    dis = [r for r in rows if not r["agree"]]
    m_only = sum(only_m_ends(r) for r in dis)
    z_only = sum(only_z_ends(r) for r in dis)
    both_act = len(dis) - m_only - z_only
    share = (lambda k: round(k / len(dis), 4) if dis else None)
    return {"decisions": n, "agreement": round(1 - len(dis) / n, 4) if n else None,
            "disagreements": len(dis), "only_m_ends": m_only, "only_z_ends": z_only,
            "both_act": both_act, "share_only_m_ends": share(m_only), "share_only_z_ends": share(z_only),
            "share_both_act": share(both_act)}


PRIOR_READINGS = (
    ("agree_rate", lambda r: float(r["agree"])),
    ("p_end_m", lambda r: r["p_end_m"]), ("p_end_z", lambda r: r["p_end_z"]),
    ("p_end_diff", lambda r: r["p_end_m"] - r["p_end_z"]),
    ("max_act_m", lambda r: r["max_act_m"]), ("max_act_z", lambda r: r["max_act_z"]),
    ("only_m_ends_rate", lambda r: float(only_m_ends(r))),
    ("only_z_ends_rate", lambda r: float(only_z_ends(r))),
)


def priors(rows: List[Dict]) -> Dict:
    """Per-game means with their standard error between games."""
    return {name: game_mean(rows, value) for name, value in PRIOR_READINGS}


def reading(rows: List[Dict]) -> Dict:
    return {**split(rows), **priors(rows)}


def log_gain(r: Dict) -> Optional[float]:
    """log prior(m) - log prior(z) of the recorded action; None when a
    label is not among the legal actions (or has no prior)."""
    if r.get("rec_prior_m") and r.get("rec_prior_z"):
        return math.log(r["rec_prior_m"]) - math.log(r["rec_prior_z"])
    return None


def label_reading(rows: List[Dict]) -> Dict:
    """The recorded label's prior under m and under z and its log-prior
    gain (positive: the memory gives the label more mass), per game."""
    scored = [r for r in rows if log_gain(r) is not None]
    return {"decisions": len(rows), "unscored": len(rows) - len(scored),
            "prior_m": game_mean(scored, lambda r: r["rec_prior_m"]),
            "prior_z": game_mean(scored, lambda r: r["rec_prior_z"]),
            "log_prior_gain": game_mean(scored, log_gain)}


def label_gain(rows: List[Dict]) -> Dict:
    """label_reading over all label kinds and per kind."""
    out = {"all": label_reading(rows)}
    for kind in sorted({r["rec"] for r in rows}):
        out[kind] = label_reading([r for r in rows if r["rec"] == kind])
    return out


def paired_gain_won_minus_lost(rows: List[Dict]) -> List[Optional[float]]:
    """Per game with both sides scored: the winning side's mean log-prior
    gain minus the losing side's; mean and standard error over games."""
    won = per_game([r for r in rows if r["outcome"] == "won"], log_gain)
    lost = per_game([r for r in rows if r["outcome"] == "lost"], log_gain)
    return rounded(mean_se(won[g] - lost[g] for g in won.keys() & lost.keys()))


def read_source(rows: List[Dict]) -> Dict:
    out: Dict = {"games": len({r["game"] for r in rows}), "decisions": len(rows)}
    if rows and "repro" in rows[0]:
        out["reproduction"] = round(sum(bool(r["repro"]) for r in rows) / len(rows), 4)
    human = bool(rows) and rows[0]["src"] == "human"
    for lo, hi in (ALL,) + BUCKETS + (LATE,):
        sel = [r for r in rows if lo <= r["turn"] <= hi]
        entry = reading(sel)
        entry["by_standing"] = {s: reading([r for r in sel if r["standing"] == s]) for s in STANDINGS}
        entry["by_outcome"] = {o: reading([r for r in sel if r["outcome"] == o]) for o in OUTCOMES}
        if human:
            entry["label_gain"] = label_gain(sel)
            entry["label_by_outcome"] = {o: label_gain([r for r in sel if r["outcome"] == o])
                                         for o in ("won", "lost")}
            entry["label_gain_won_minus_lost"] = paired_gain_won_minus_lost(sel)
        out[bucket_name(lo, hi)] = entry
    return out


# ---------------------------------------------------------------------------
# The pre-registered verdicts
# ---------------------------------------------------------------------------

def resolved_above(value: List[Optional[float]], bar: float = 0.0) -> bool:
    """value = [mean, se]: the mean exceeds `bar` by two standard errors."""
    m, se = value
    return m is not None and se is not None and m - 2 * se > bar


def w_verdict(readings: Dict) -> Dict:
    """W: on own, turns 6 and later (in the first turns the HP margin moves
    with the order of recruitment), the agreement when ahead exceeds the
    agreement when behind by at least W_AGREEMENT_GAP and by two standard
    errors (the two per-game samples' errors combined); on human, all
    turns, the memory's log-prior gain on the recorded label is above 0 on
    winning sides by two standard errors, and the winning side's gain
    exceeds the losing side's by two standard errors (paired by game)."""
    out: Dict = {}
    own, human = readings.get("own"), readings.get("human")
    if own is not None:
        stand = own["late (6+)"]["by_standing"]
        (a, sa), (b, sb) = stand["ahead"]["agree_rate"], stand["behind"]["agree_rate"]
        if None not in (a, sa, b, sb):
            d, se = a - b, math.hypot(sa, sb)
            out["own_agreement_ahead_minus_behind"] = [round(d, 5), round(se, 5)]
            out["own_holds"] = d >= W_AGREEMENT_GAP and d > 2 * se
    if human is not None:
        alltime = human["all"]
        won = alltime["label_by_outcome"]["won"]["all"]["log_prior_gain"]
        diff = alltime["label_gain_won_minus_lost"]
        out["human_gain_won"] = won
        out["human_gain_won_minus_lost"] = diff
        out["human_holds"] = resolved_above(won) and resolved_above(diff)
    out["W_supported"] = bool(out.get("own_holds")) and bool(out.get("human_holds"))
    return out


def row_verdicts(readings: Dict) -> Dict:
    """The tool's check and the row readings the matches are read beside."""
    out: Dict = {}
    own = readings.get("own")
    if own is not None:
        out["tool_check"] = (own.get("reproduction") or 0) >= TOOL_CHECK_REPRODUCTION
        late = own["late (6+)"]
        out["own_late_share_only_m_ends"] = late["share_only_m_ends"]
        out["own_late_p_end_diff"] = late["p_end_diff"]
    if own is not None and "human" in readings:
        for name in ("p_end_diff", "only_m_ends_rate"):
            a, b = own["late (6+)"][name], readings["human"]["late (6+)"][name]
            if None not in (a[0], b[0], a[1], b[1]):
                d, se = a[0] - b[0], math.hypot(a[1], b[1])
                out[f"own_minus_human_{name}"] = [round(d, 5), round(se, 5),
                                                  "differs" if abs(d) >= 2 * se else "within 2 SE"]
    out["W"] = w_verdict(readings)
    return out


def candidate_elo(fit: Dict, opponent: str) -> Tuple[float, float]:
    """(Elo, standard error) of the fit's other label against `opponent`,
    from its decisive games; the anchor's error is 0, so the difference's
    error is the other label's."""
    table = fit["tables"][FULL_PLAY]
    if opponent not in table or len(table) != 2:
        raise SystemExit(f"a fit of {sorted(table)} is not a match against {opponent}")
    (cand,) = [k for k in table if k != opponent]
    c, o = table[cand], table[opponent]
    return c["elo"] - o["elo"], math.hypot(c["se"], o["se"])


def match_verdict(arms: Dict[str, Tuple[float, float]]) -> Dict:
    """The pre-registered reading of A1 (64 slots at -2.5), A2 (64 slots,
    per-turn reset, -1.5) and A3 (64 slots at -3.5), each against 0 slots
    at -1.5. An arm 'recovers' when its Elo lies within one standard error
    of 0 or above, 'nears' when within two but not one, and 'trails'
    otherwise. T: A1 or A3 recovers. L: A2 recovers, A1 and A3 do not. Q:
    no arm within two standard errors. In between: partial T when A1 or A3
    nears (and no arm recovers), partial L when only A2 nears."""
    def status(elo_se):
        elo, se = elo_se
        return "recovers" if elo + se >= 0 else "nears" if elo + 2 * se >= 0 else "trails"
    st = {name: status(arms[name]) for name in ("A1", "A2", "A3")}
    tempo = "recovers" in (st["A1"], st["A3"])
    if tempo:
        reading = "T" + (" (A2 recovers too)" if st["A2"] == "recovers" else "")
    elif st["A2"] == "recovers":
        reading = "L"
    elif all(s == "trails" for s in st.values()):
        reading = "Q"
    elif "nears" in (st["A1"], st["A3"]):
        reading = "partial T" + (" (A2 nears too)" if st["A2"] == "nears" else "")
    else:
        reading = "partial L"
    return {"arms": {k: [round(arms[k][0], 2), round(arms[k][1], 2), st[k]] for k in st},
            "reading": reading}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sources", nargs="*", help="NAME=ROWS.jsonl (own, other, human)")
    ap.add_argument("--fit", action="append", default=[], help="A1|A2|A3=FIT.json")
    ap.add_argument("--opponent", default="parity3_slots0", help="the 0-slot player's label in the fits")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    readings = {}
    for spec in args.sources:
        name, _, path = spec.partition("=")
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
        readings[name] = read_source(rows)
    result: Dict = {"readings": readings, "verdicts": row_verdicts(readings)}
    fits = dict(spec.partition("=")[::2] for spec in args.fit)
    if fits:
        if sorted(fits) != ["A1", "A2", "A3"]:
            ap.error("--fit takes A1, A2 and A3, once each")
        arms = {name: candidate_elo(json.loads(Path(p).read_text(encoding="utf-8")), args.opponent)
                for name, p in fits.items()}
        result["matches"] = match_verdict(arms)
    text = json.dumps(result, indent=1)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
