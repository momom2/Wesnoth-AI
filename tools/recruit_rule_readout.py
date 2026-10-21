"""Match L's readout (docs/recruit_rule_l_prereg_20261021.md): the recruit
rule on the seat a per-game coin chose (tools/recruit_rule.py), read from
the games' result files and records against the rule's seal
(tools/recruit_rule_fit.py's readout). numpy and the standard library only.

extract writes one row per game of a match (its result, seat and rule
decisions, each decision's inputs and value recomputed by R0's code), then
the match's manifest; readout pools the rows of the matches.

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
  - every match read whole: each result file has its record, and every
    result names the rule's hash in both procedure tags and its seat;
  - the record's seat coin is the one its seed gives (recomputed here);
  - every rule decision is the hooked side's, and its inputs (factions,
    types, earlier recruits) and its value f recomputed by R0's code from
    the record (tools/recruit_rule_fit.py: the setup's factions, the
    commands' recruits, the design matrix times the frozen rule's weights)
    equal the logged ones, so the enemy faction is the setup's;
  - the switches recomputed from those values and the cap equal the logged
    ones; no side-game switches more than the cap; every switch played a1
    and every other rule decision a0 (tools/coin_rows.played_status);
  - the rule switched at least once on every faction it moved for;
  - the seat coin against 1/2, and its seat against side A's side (2 x 2
    chi-square: the first mover's +9.3 Elo must not ride on the seat).
Fire rates (switches and rule decisions per hooked side-game) by the hooked
side's faction are reported.

    python tools/recruit_rule_readout.py extract --match l00=DIR --rule RULE.json --out rows_l00.jsonl
    python tools/recruit_rule_readout.py readout --rows rows_l00.jsonl [...] --rule RULE.json \\
        [--plan matches.txt] [--save-json readout.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.coin_labels import CHECK_SE, chi2_upper_p, coin_frequency, iter_jsonl  # noqa: E402
from tools.coin_rows import parse_plan, played_status, slot_report  # noqa: E402
from tools.match_luck import match_items, parse_record, result_z  # noqa: E402
from tools.recruit_rule import SHORT_HASH, RULE_SEAT_DOMAIN, RULE_SEAT_P  # noqa: E402
from tools.recruit_rule_fit import ELO_SE_1, CommandContext, Rows, design, row_features, values  # noqa: E402

INPUTS = ("side", "fac", "enemy", "u0", "u1", "n0", "n1")
F_TOLERANCE = 1e-12


def seat_of_seed(seed: int) -> str:
    """The seat coin of a seed, recomputed here from its definition."""
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
    zeros = np.zeros(n, int)
    rows = Rows(match=zeros, gid=zeros, side=np.array([f["side"] for f in feats]), cmd=zeros, d=zeros,
                fac=np.array([f["fac"] for f in feats]), enemy=np.array([f["enemy"] for f in feats]),
                u0=np.array([f"{f['fac']}|{f['u0']}" for f in feats]), u1=np.array([f"{f['fac']}|{f['u1']}" for f in feats]),
                dd=np.array([f["n0"] - f["n1"] for f in feats], dtype=float), st=zeros, T=np.zeros(n),
                K=zeros, dev=zeros, side_games=0, games_by_match={})
    des = design(rows)
    return values(des, rule_vector(rule, des.names))


# ---------------------------------------------------------------------
# extract: one row per game
# ---------------------------------------------------------------------
def _game_row(match: str, game: str, res: dict, rec: dict, rule_file: dict) -> Tuple[dict, List[dict]]:
    """A game's row and R0's reading of each of its rule decisions."""
    short = rule_file["hash"][:SHORT_HASH]
    rr = rec.get("recruit_rule") or {}
    coin = (rec.get("coins") or {}).get("rule") or {}
    seat = rr.get("seat")
    side_a = int(rec["players"]["a"]["side"])
    row = {"match": match, "game": game, "seed": res.get("seed"), "z": result_z(res.get("outcome_a")),
           "outcome_a": res.get("outcome_a"), "margin_a": res.get("margin_a"), "side_a": side_a, "seat": seat,
           "issues": []}
    named = (all(str(res.get(f"procedure_{s}", "")).endswith(f"+rr{short}") for s in ("a", "b"))
             and res.get("recruit_rule") == short and rr.get("hash") == rule_file["hash"]
             and coin.get("hash") == rule_file["hash"] and seat in ("a", "b"))
    if not named:
        row["issues"].append("rule_not_named")
        return row, []
    if not (seat == coin.get("seat") == res.get("recruit_rule_seat") == seat_of_seed(res["seed"])) \
            or int(coin.get("d", -1)) != int(seat == "a"):
        row["issues"].append("seat_coin")
    hooked = side_a if seat == "a" else 3 - side_a
    row.update(hooked_side=hooked, hooked_faction=rec["setup"][f"faction{hooked}"])
    ctx = CommandContext(rec["commands"])
    rejected = {(int(k), int(x), int(y)) for k, x, y in rec.get("rejections", ())}
    decisions, feats = [], []
    for e in rr.get("decisions", []):
        x = row_features(rec, ctx, int(e["side"]), int(e["cmd"]), e["u0"], e["u1"])
        status = played_status({**e, "d": int(bool(e["switched"]))}, rec["commands"], rejected)
        decisions.append({"side": int(e["side"]), "cmd": int(e["cmd"]), "i": int(e["i"]), "f": float(e["f"]),
                          "switched": bool(e["switched"]), "status": status, "refused": bool(e.get("refused")),
                          "inputs_ok": all(x[k] == e[k] for k in INPUTS), "on_seat": int(e["side"]) == hooked})
        feats.append(x)
    row["decisions"] = decisions
    return row, feats


def extract_rows(match: str, path: Path, rule_file: dict) -> Iterable[dict]:
    """The rows of one match's games directory or tarball, then its manifest."""
    items = match_items(match, path)
    pending, feats = [], []
    for item in items:
        if item.record is None:
            pending.append({"match": match, "game": item.game, "seed": item.result.get("seed"),
                            "z": result_z(item.result.get("outcome_a")), "issues": ["no_record"]})
            continue
        row, x = _game_row(match, item.game, item.result, parse_record(item.record, item.game), rule_file)
        pending.append(row)
        feats += x
    f = recomputed_values(rule_file["rule"], feats).tolist()
    k = 0
    for row in pending:
        for e in row.get("decisions", ()):
            e["f_r0"] = f[k]
            k += 1
        yield row
    yield {"match": match, "manifest": {"results": len(items)}}


# ---------------------------------------------------------------------
# readout
# ---------------------------------------------------------------------
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


def rule_checks(games: List[dict], cap: int) -> dict:
    """I-cap, I-key, I-match on the rule decisions of the games read."""
    out = defaultdict(int)
    worst = 0
    for g in games:
        fires: Dict[int, int] = defaultdict(int)
        for e in sorted(g.get("decisions", ()), key=lambda e: (e["side"], e["cmd"], e["i"])):
            out["decisions"] += 1
            out["switches"] += e["switched"]
            out["inputs_differ"] += not e["inputs_ok"]
            out["f_differs"] += abs(e["f"] - e["f_r0"]) > F_TOLERANCE
            want = e["f_r0"] > 0.0 and fires[e["side"]] < cap
            fires[e["side"]] += want
            out["switch_differs"] += want != e["switched"]
            out["played_not_as_logged"] += e["status"] == "mismatched"
            out["off_seat"] += not e["on_seat"]
            out["refused"] += e["refused"]
        worst = max(worst, sum(e["switched"] for e in g.get("decisions", ())))
    out = {k: int(out[k]) for k in ("decisions", "switches", "inputs_differ", "f_differs", "switch_differs",
                                    "played_not_as_logged", "off_seat", "refused")}
    out["max_switches_per_side_game"] = worst
    out["pass"] = bool(out["inputs_differ"] == out["f_differs"] == out["switch_differs"] == 0
                       and worst <= cap and out["played_not_as_logged"] == out["off_seat"] == 0)
    return out


def fire_rates(games: List[dict]) -> dict:
    by = defaultdict(lambda: [0, 0, 0])
    for g in games:
        b = by[g["hooked_faction"]]
        b[0] += 1
        b[1] += sum(e["switched"] for e in g["decisions"])
        b[2] += len(g["decisions"])
    return {fac: {"hooked_side_games": n, "switches_per_side_game": s / n, "decisions_per_side_game": d / n}
            for fac, (n, s, d) in sorted(by.items())}


def readout(rows: Iterable[dict], rule_file: dict, seal_readout: dict,
            plan: Optional[Dict[str, Tuple[int, int]]] = None) -> dict:
    """Every reading and check of the rows of matches read whole; a match
    whose manifest is missing or whose rows do not account for its result
    files is left out and fails the verdict."""
    from tools.match_luck_readout import elo_from_mean
    by_match: Dict[str, List[dict]] = defaultdict(list)
    manifests: Dict[str, int] = {}
    for r in rows:
        if "manifest" in r:
            manifests[r["match"]] = int(r["manifest"]["results"])
        else:
            by_match[r["match"]].append(r)
    out: dict = {"rule": rule_file["hash"], "matches": {}, "checks": {}}
    games, issues, left_out = [], defaultdict(int), []
    for m in sorted(set(by_match) | set(manifests)):
        rs = by_match.get(m, [])
        a = out["matches"][m] = {"results": manifests.get(m), "rows": len(rs)}
        if plan and m in plan:
            a["slots"] = slot_report([(r.get("seed"), r["z"] != 0) for r in rs], plan[m])
        if manifests.get(m) != len(rs):
            left_out.append(m)
            continue
        for r in rs:
            for why in r["issues"]:
                issues[why] += 1
        games += [r for r in rs if "decisions" in r]
    seal, seal_se = float(seal_readout["seal"]["elo"]), float(seal_readout["seal"]["se"])
    dec = [g for g in games if g["z"] != 0]
    out["seal"] = {"elo": seal, "se": seal_se}
    out["games"] = {"read": len(games), "decisive": len(dec), "capped": len(games) - len(dec)}
    checks = out["checks"]
    checks["matches_whole"] = {"left_out": left_out, "no_record": issues["no_record"],
                               "rule_not_named": issues["rule_not_named"],
                               "pass": not left_out and not issues["no_record"] and not issues["rule_not_named"]
                               and bool(games)}
    checks["seat_coin_logged"] = {"games": issues["seat_coin"], "pass": not issues["seat_coin"]}
    checks["i_cap"] = rule_checks(games, int(rule_file["rule"]["cap"]))
    rates = out["fire_rates"] = fire_rates(games)
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
    if len(dec) > 1:
        zh = np.array([g["z"] if g["seat"] == "a" else -g["z"] for g in dec], dtype=float)
        m, se = float(zh.mean()), float(zh.std(ddof=1) / math.sqrt(len(zh)))
        x, x_se = elo_from_mean(m, se)
        s = ELO_SE_1 / math.sqrt(len(dec))
        half, null = 2 * math.hypot(seal_se, s), 2 * s
        out["hooked"] = {"mean_z": m, "mean_z_se": se, "logit": {"elo": x, "se": x_se},
                         "elo_collect": hooked_elo_fit(dec)}
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
        line = f"  {m}: {a['results']} result files, {a['rows']} rows"
        s = a.get("slots")
        if s:
            line += f"; {s['with_result']} of {s['planned']} slots from seed {s['seed_base']} ({s['decisive']} decisive)"
        lines.append(line + ("; LEFT OUT" if m in r["checks"]["matches_whole"]["left_out"] else ""))
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
                 f"a side-game); inputs differ {c['inputs_differ']}, f differs {c['f_differs']}, switches differ "
                 f"{c['switch_differs']}, played not as logged {c['played_not_as_logged']}, off the seat {c['off_seat']}, "
                 f"refused {c['refused']}")
    lines.append("fire rate by hooked faction (switches, rule decisions per side-game): " + "; ".join(
        f"{k} {v['switches_per_side_game']:.3f}, {v['decisions_per_side_game']:.3f} ({v['hooked_side_games']})"
        for k, v in r["fire_rates"].items()))
    sc = r["seat_coin"]
    if sc["n"]:
        lines.append(f"seat coin: A {sc['ones']} of {sc['n']} (chi2 {sc['chi2']:.2f}); hooked on side 1 "
                     f"{sc['hooked_on_side_1']:.3f}; seat by side A's side chi2 {sc['seat_by_side_a']['chi2']:.2f}")
    lines.append(f"verdict: {r['verdict']}")
    return "\n".join(lines)


def _source(text: str) -> Tuple[str, Path]:
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"--match takes NAME=PATH, not {text!r}")
    return name, Path(path)


def load_rule(path: Path) -> dict:
    """The rule file, its content checked against its hash."""
    from tools.recruit_rule import RecruitRule
    data = json.loads(path.read_text(encoding="utf-8"))
    RecruitRule.from_data(data)
    return data


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract", help="one row per game of a match")
    e.add_argument("--match", type=_source, required=True, metavar="NAME=PATH")
    e.add_argument("--rule", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("readout", help="the reading and the checks of the matches' rows")
    r.add_argument("--rows", type=Path, nargs="+", required=True)
    r.add_argument("--rule", type=Path, required=True)
    r.add_argument("--seal", type=Path, default=None, help="the rule's readout (default: beside the rule)")
    r.add_argument("--plan", type=Path, default=None, help="a box script's matches.txt")
    r.add_argument("--save-json", type=Path, default=None)
    args = ap.parse_args(argv)
    rule_file = load_rule(args.rule)
    if args.command == "extract":
        tmp = args.out.with_name(args.out.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for row in extract_rows(args.match[0], args.match[1], rule_file):
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        tmp.replace(args.out)
        return 0
    seal_path = args.seal or args.rule.with_name(args.rule.stem + "_readout.json")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal["hash"] != rule_file["hash"]:
        raise SystemExit(f"{seal_path} seals rule {seal['hash']}, not {rule_file['hash']}")
    result = readout(iter_jsonl(args.rows), rule_file, seal, parse_plan(args.plan) if args.plan else None)
    print(report_text(result))
    if args.save_json is not None:
        tmp = args.save_json.with_name(args.save_json.name + ".tmp")
        tmp.write_text(json.dumps(result, indent=1), encoding="utf-8")
        tmp.replace(args.save_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
