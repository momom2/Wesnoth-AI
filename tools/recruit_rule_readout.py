"""Match L's readout (docs/recruit_rule_l_prereg_20261021.md): the recruit
rule on the seat a per-game coin chose (tools/recruit_rule.py), read from
the games' result files and records against the rule's seal
(tools/recruit_rule_fit.py's readout). numpy and the standard library only.

The reading. Over the decisive games, the hooked seat's result z_h (side
A's z where the seat coin chose A, -z where it chose B): its Elo by
elo_collect's fit on the games relabelled "hooked" and "plain"
(tools/elo_collect.build_pairs and the fit it runs), and by the logit of its
mean score (match_luck_readout.elo_from_mean). The lines at the games read
(round10_synthesis_budget.py section 1): match SE s = 347.4 / sqrt(N), miss
half-width 2 sqrt(seal SE^2 + s^2), null line 2 s. The outcome, on the logit
reading x: "inside" when |x - seal| is within the half-width, "above" past
it, "null" when x is at or under the null line, "below" (a partial link,
the lower bound above 0) otherwise.

The checks (the design's I-cap, I-key, I-match and I-side):
  - every result names the rule's hash in both procedure tags and its seat,
    and the record's seat coin is the one its seed gives;
  - every rule decision is the hooked side's, its inputs (factions, types,
    earlier recruits) and its value f recomputed by R0's code from the
    record (tools/recruit_rule_fit.py: the setup's factions, the commands'
    recruits, the design matrix times the frozen rule's weights) equal the
    logged ones; the enemy faction is the setup's;
  - the switches recomputed from those values and the cap equal the logged
    ones; no side-game switches more than the cap; every switch played a1
    and every other rule decision a0 (tools/coin_rows.played_status);
  - the rule switched at least once on every faction it moved;
  - the seat coin against 1/2, and its seat against side A's side (2 x 2
    chi-square: the first mover's +9.3 Elo must not ride on the seat).
Fire rates (switches and rule decisions per hooked side-game) by the hooked
side's faction are reported.

    python tools/recruit_rule_readout.py --match l00=DIR [--match ...] \\
        --rule configs/recruit_rule_bd_<hash>.json [--plan matches.txt] [--save-json readout.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.coin_labels import CHECK_SE, chi2_upper_p, coin_frequency  # noqa: E402
from tools.coin_rows import parse_plan, played_status, slot_report  # noqa: E402
from tools.match_luck import match_items, parse_record, result_z  # noqa: E402
from tools.recruit_rule import SHORT_HASH, RULE_SEAT_DOMAIN, RULE_SEAT_P  # noqa: E402
from tools.recruit_rule_fit import ELO_SE_1, CommandContext, Rows, design, row_features, values  # noqa: E402

INPUTS = ("side", "fac", "enemy", "u0", "u1", "n0", "n1")


def seat_of_seed(seed: int) -> str:
    """The seat coin of a seed, recomputed here from its definition."""
    import hashlib
    digest = hashlib.sha256(":".join([RULE_SEAT_DOMAIN, str(int(seed))]).encode("utf-8")).digest()
    return "a" if int.from_bytes(digest[:8], "big") / 2.0 ** 64 < RULE_SEAT_P else "b"


def rule_vector(rule: dict, names: Sequence[str]) -> np.ndarray:
    """The frozen rule's weight of each design column (0 for a key it lacks)."""
    flat = {**rule["units"], **rule["matchups"], "<count n0 - n1>": rule["count"]}
    return np.array([float(flat.get(n, 0.0)) for n in names])


def recomputed_values(rule: dict, feats: List[dict]) -> np.ndarray:
    """f of each rule decision from R0's reading of its inputs off the
    record (`row_features`): `values` on the design of those rows with the
    frozen rule's weights."""
    if not feats:
        return np.zeros(0)
    n = len(feats)
    rows = Rows(match=np.zeros(n, int), gid=np.zeros(n, int), side=np.array([f["side"] for f in feats]),
                cmd=np.zeros(n, int), d=np.zeros(n, int), fac=np.array([f["fac"] for f in feats]),
                enemy=np.array([f["enemy"] for f in feats]),
                u0=np.array([f"{f['fac']}|{f['u0']}" for f in feats]), u1=np.array([f"{f['fac']}|{f['u1']}" for f in feats]),
                dd=np.array([f["n0"] - f["n1"] for f in feats], dtype=float), st=np.zeros(n, int), T=np.zeros(n),
                K=np.zeros(n, int), dev=np.zeros(n, int), side_games=0, games_by_match={})
    des = design(rows)
    return values(des, rule_vector(rule, des.names))


def chi2_2x2(table: np.ndarray) -> dict:
    """Pearson's chi-square of independence on a 2 x 2 table, 1 df."""
    n = table.sum()
    expect = np.outer(table.sum(1), table.sum(0)) / n if n else table * 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.nansum((table - expect) ** 2 / np.where(expect > 0, expect, np.nan)))
    return {"table": table.astype(int).tolist(), "chi2": chi2, "p_value": chi2_upper_p(chi2, 1) if n else 1.0}


def hooked_elo_fit(games: List[dict]) -> dict:
    """elo_collect's PURE fit on the games relabelled by seat."""
    from tools.elo_collect import build_pairs
    from tools.elo_ladder import fit_elo
    relabelled = [{"label_a": "hooked" if g["seat"] == "a" else "plain",
                   "label_b": "plain" if g["seat"] == "a" else "hooked",
                   "outcome_a": g["outcome_a"], "margin_a": g.get("margin_a")} for g in games]
    labels, pure, _mat, _nores = build_pairs(relabelled, 0.02)
    if len(labels) < 2 or not any(r.wins_i + r.wins_j for r in pure.values()):
        return {"elo": None, "se": None}
    elo, se = fit_elo(len(labels), pure, labels.index("plain"), anchor_elo=0.0, prior_games=1.0,
                      draw_weight=0.5, prior_scope="played")
    k = labels.index("hooked")
    return {"elo": float(elo[k]), "se": float(se[k])}


def outcome(x: float, seal: float, half: float, null: float) -> str:
    if abs(x - seal) <= half:
        return "inside"
    if x > seal + half:
        return "above"
    return "null" if x <= null else "below"


def read_games(sources: Sequence[Tuple[str, Path]], rule_file: dict) -> dict:
    """Every game of the matches with its seat, result and rule decisions,
    the per-game checks counted."""
    short = rule_file["hash"][:SHORT_HASH]
    games, entries, issues = [], [], defaultdict(list)
    accounts = {}
    for match, path in sources:
        items = match_items(match, path)
        acc = accounts[match] = {"results": len(items), "records": 0, "seeds": []}
        for item in items:
            res = item.result
            acc["seeds"].append((res.get("seed"), result_z(res.get("outcome_a")) != 0))
            if item.record is None:
                issues["no_record"].append(item.game)
                continue
            acc["records"] += 1
            rec = parse_record(item.record, item.game)
            rr = rec.get("recruit_rule") or {}
            coin = (rec.get("coins") or {}).get("rule") or {}
            seat = rr.get("seat")
            if not all(str(res.get(f"procedure_{s}", "")).endswith(f"+rr{short}") for s in ("a", "b")) \
                    or res.get("recruit_rule") != short or rr.get("hash") != rule_file["hash"] \
                    or coin.get("hash") != rule_file["hash"]:
                issues["rule_not_named"].append(item.game)
                continue
            if not (seat == coin.get("seat") == res.get("recruit_rule_seat") == seat_of_seed(res["seed"])) \
                    or int(coin.get("d", -1)) != int(seat == "a"):
                issues["seat_coin"].append(item.game)
            side_a = int(rec["players"]["a"]["side"])
            hooked_side = side_a if seat == "a" else 3 - side_a
            g = {"match": match, "game": item.game, "seed": res["seed"], "seat": seat, "side_a": side_a,
                 "hooked_side": hooked_side, "z": result_z(res.get("outcome_a")), "outcome_a": res.get("outcome_a"),
                 "margin_a": res.get("margin_a"), "hooked_faction": rec["setup"][f"faction{hooked_side}"],
                 "decisions": 0, "switches": 0}
            ctx = CommandContext(rec["commands"])
            rejected = {(int(k), int(x), int(y)) for k, x, y in rec.get("rejections", ())}
            for e in rr.get("decisions", []):
                if int(e["side"]) != hooked_side:
                    issues["decision_off_seat"].append(item.game)
                status = played_status({**e, "d": int(bool(e["switched"]))}, rec["commands"], rejected)
                if status == "mismatched":
                    issues["played_not_as_logged"].append(item.game)
                g["decisions"] += 1
                g["switches"] += bool(e["switched"])
                entries.append({**e, "_game": len(games),
                                "_x": row_features(rec, ctx, int(e["side"]), int(e["cmd"]), e["u0"], e["u1"])})
            games.append(g)
    return {"games": games, "entries": entries, "issues": issues, "accounts": accounts}


def rule_checks(rule_file: dict, data: dict) -> dict:
    """I-cap, I-key, I-match on the rule decisions read."""
    rule, entries = rule_file["rule"], data["entries"]
    f = recomputed_values(rule, [e["_x"] for e in entries])
    bad_inputs = sum(any(e["_x"][k] != e[k] for k in INPUTS) for e in entries)
    logged_f = np.array([float(e["f"]) for e in entries])
    f_diff = float(np.max(np.abs(f - logged_f))) if len(f) else 0.0
    # the switches the values and the cap give, in each side-game's order
    order = sorted(range(len(entries)), key=lambda k: (entries[k]["_game"], entries[k]["side"], entries[k]["cmd"],
                                                       entries[k]["i"]))
    fires: Dict[Tuple[int, int], int] = defaultdict(int)
    wrong_switch = 0
    for k in order:
        key = (entries[k]["_game"], int(entries[k]["side"]))
        want = bool(f[k] > 0.0 and fires[key] < int(rule["cap"]))
        fires[key] += want
        wrong_switch += want != bool(entries[k]["switched"])
    per_side_game = [g["switches"] for g in data["games"]]
    issues = data["issues"]
    out = {"decisions": len(entries), "switches": int(sum(bool(e["switched"]) for e in entries)),
           "inputs_differ": int(bad_inputs), "max_f_difference": f_diff, "switch_differs": int(wrong_switch),
           "max_switches_per_side_game": int(max(per_side_game, default=0)),
           "played_not_as_logged": len(issues["played_not_as_logged"]),
           "decision_off_seat": len(issues["decision_off_seat"]),
           "refused": int(sum(bool(e.get("refused")) for e in entries))}
    out["pass"] = bool(bad_inputs == 0 and f_diff <= 1e-12 and wrong_switch == 0
                       and out["max_switches_per_side_game"] <= int(rule["cap"])
                       and out["played_not_as_logged"] == 0 and out["decision_off_seat"] == 0)
    return out


def fire_rates(games: List[dict]) -> dict:
    by = defaultdict(lambda: [0, 0, 0])
    for g in games:
        b = by[g["hooked_faction"]]
        b[0] += 1
        b[1] += g["switches"]
        b[2] += g["decisions"]
    return {fac: {"hooked_side_games": n, "switches_per_side_game": s / n, "decisions_per_side_game": d / n}
            for fac, (n, s, d) in sorted(by.items())}


def readout(sources: Sequence[Tuple[str, Path]], rule_file: dict, seal_readout: dict,
            plan: Optional[Dict[str, Tuple[int, int]]] = None) -> dict:
    from tools.match_luck_readout import elo_from_mean
    data = read_games(sources, rule_file)
    games = data["games"]
    dec = [g for g in games if g["z"] != 0]
    seal, seal_se = float(seal_readout["seal"]["elo"]), float(seal_readout["seal"]["se"])
    out: dict = {"rule": rule_file["hash"], "seal": {"elo": seal, "se": seal_se},
                 "games": {"read": len(games), "decisive": len(dec), "capped": len(games) - len(dec)},
                 "matches": {}, "checks": {}}
    for m, acc in data["accounts"].items():
        out["matches"][m] = {"results": acc["results"], "records": acc["records"]}
        if plan and m in plan:
            out["matches"][m]["slots"] = slot_report(acc["seeds"], plan[m])
    issues = data["issues"]
    checks = out["checks"]
    checks["games_whole"] = {"no_record": len(issues["no_record"]), "rule_not_named": len(issues["rule_not_named"]),
                             "pass": not issues["no_record"] and not issues["rule_not_named"] and bool(games)}
    checks["seat_coin_logged"] = {"games": len(issues["seat_coin"]), "pass": not issues["seat_coin"]}
    checks["i_cap"] = rule_checks(rule_file, data)
    rates = fire_rates(games)
    out["fire_rates"] = rates
    checks["fires_every_faction"] = {"silent": [f for f, r in rates.items() if r["switches_per_side_game"] == 0],
                                     "pass": bool(rates) and all(r["switches_per_side_game"] > 0 for r in rates.values())}
    d = np.array([g["seat"] == "a" for g in games], dtype=float)
    coin = coin_frequency(d, np.full(len(d), RULE_SEAT_P))
    table = np.zeros((2, 2))
    for g in games:
        table[int(g["seat"] == "b"), int(g["side_a"] == 2)] += 1
    seat_side = chi2_2x2(table)
    out["seat_coin"] = {**coin, "hooked_on_side_1": float(np.mean([g["hooked_side"] == 1 for g in games])) if games else None,
                        "seat_by_side_a": seat_side}
    checks["i_side"] = {"pass": bool(coin["chi2"] <= CHECK_SE ** 2 and seat_side["chi2"] <= CHECK_SE ** 2)}
    if dec:
        zh = np.array([g["z"] if g["seat"] == "a" else -g["z"] for g in dec], dtype=float)
        m, se = float(zh.mean()), float(zh.std(ddof=1) / math.sqrt(len(zh))) if len(zh) > 1 else float("nan")
        x, x_se = elo_from_mean(m, se)
        s = ELO_SE_1 / math.sqrt(len(dec))
        half, null = 2 * math.hypot(seal_se, s), 2 * s
        out["hooked"] = {"mean_z": m, "mean_z_se": se, "logit": {"elo": x, "se": x_se},
                         "elo_collect": hooked_elo_fit(dec), "decisive_games": len(dec)}
        out["lines"] = {"match_se": s, "half_width": half, "null_line": null, "window": [seal - half, seal + half],
                        "ratio": x / seal if seal else None}
        out["outcome"] = outcome(x, seal, half, null)
    failed = sorted(k for k, c in checks.items() if not c["pass"])
    out["verdict"] = "PASS" if not failed else "FAIL: " + " ".join(failed)
    return out


def report_text(r: dict) -> str:
    g = r["games"]
    lines = [f"games {g['read']} read, {g['decisive']} decisive, {g['capped']} capped; rule {r['rule'][:SHORT_HASH]}, "
             f"seal {r['seal']['elo']:+.1f} +- {r['seal']['se']:.1f} Elo"]
    for m, a in r["matches"].items():
        line = f"  {m}: {a['results']} results, {a['records']} records"
        s = a.get("slots")
        if s:
            line += f"; {s['with_result']} of {s['planned']} slots from seed {s['seed_base']} ({s['decisive']} decisive)"
        lines.append(line)
    if "hooked" in r:
        h, L = r["hooked"], r["lines"]
        ec = h["elo_collect"]
        lines.append(f"hooked seat: mean z {h['mean_z']:+.4f} +- {h['mean_z_se']:.4f}; logit {h['logit']['elo']:+.1f} +- "
                     f"{h['logit']['se']:.1f} Elo; elo_collect "
                     + (f"{ec['elo']:+.1f} +- {ec['se']:.1f}" if ec["elo"] is not None else "n/a")
                     + f"; window [{L['window'][0]:+.1f}, {L['window'][1]:+.1f}], null line {L['null_line']:.2f}, "
                     f"ratio to the seal {L['ratio']:.2f}: {r['outcome'].upper()}")
    c = r["checks"]["i_cap"]
    lines.append(f"I-cap: {c['decisions']} rule decisions, {c['switches']} switches (max {c['max_switches_per_side_game']} "
                 f"a side-game); inputs differ {c['inputs_differ']}, max |f - R0's f| {c['max_f_difference']:.1e}, "
                 f"switches differ {c['switch_differs']}, played not as logged {c['played_not_as_logged']}, "
                 f"off the seat {c['decision_off_seat']}, refused {c['refused']}")
    lines.append("fire rate by hooked faction (switches, rule decisions per side-game): " + "; ".join(
        f"{k} {v['switches_per_side_game']:.3f}, {v['decisions_per_side_game']:.3f} ({v['hooked_side_games']})"
        for k, v in r["fire_rates"].items()))
    sc = r["seat_coin"]
    lines.append(f"seat coin: A {sc['ones']} of {sc['n']} (chi2 {sc['chi2']:.2f}); hooked on side 1 "
                 f"{sc['hooked_on_side_1']:.3f}; seat by side A's side chi2 {sc['seat_by_side_a']['chi2']:.2f}")
    lines.append(f"verdict: {r['verdict']}")
    return "\n".join(lines)


def _source(text: str) -> Tuple[str, Path]:
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"--match takes NAME=PATH, not {text!r}")
    return name, Path(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", type=_source, action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--rule", type=Path, required=True)
    ap.add_argument("--seal", type=Path, default=None, help="the rule's readout (default: beside the rule)")
    ap.add_argument("--plan", type=Path, default=None, help="a box script's matches.txt")
    ap.add_argument("--save-json", type=Path, default=None)
    args = ap.parse_args(argv)
    from tools.recruit_rule import RecruitRule
    rule_file = json.loads(args.rule.read_text(encoding="utf-8"))
    RecruitRule.from_data(rule_file)                     # its content against its hash
    seal_path = args.seal or args.rule.with_name(args.rule.stem + "_readout.json")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal["hash"] != rule_file["hash"]:
        raise SystemExit(f"{seal_path} seals rule {seal['hash']}, not {rule_file['hash']}")
    result = readout(args.match, rule_file, seal, parse_plan(args.plan) if args.plan else None)
    print(report_text(result))
    if args.save_json is not None:
        tmp = args.save_json.with_name(args.save_json.name + ".tmp")
        tmp.write_text(json.dumps(result, indent=1), encoding="utf-8")
        tmp.replace(args.save_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
