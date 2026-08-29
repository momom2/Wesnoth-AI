"""Fit Elo from a directory of elo_eval_game.py result files, under
TWO conventions from the same games:

  PURE (primary): only decisive games carry rating information.
    Material advantage must not factor into evaluation (user decision
    2026-07-11, reversing the 2026-07-04 material-primary lock), and
    -- user revision 2026-08-17 -- **a capped game is NOT a draw**:
    there are no draws in real Wesnoth, so a game that hit the turn
    cap (or any other non-decisive stop) is a TRUNCATED OBSERVATION,
    recorded as an absence ("no result") and EXCLUDED from the fit.
    The confidence interval widens accordingly; if a fixed CI is
    required, run_elo_batch schedules replacement games (bounded by
    its --max-extra-games guard).
  MATERIAL-SIGN (diagnostic only): a no-result game whose final
    material margin from A exceeds +/-EPS counts as a win for the
    side ahead. More separating while ladder games are cap-heavy,
    useful for watching progress -- but never the headline number.

Usage:
    python tools/elo_collect.py GAMES_DIR [--anchor dummy]
        [--save-json PATH] [--eps 0.02]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.elo_ladder import PairRecord, fit_elo


def load_games(games_dir: Path) -> List[dict]:
    games = []
    for p in sorted(games_dir.glob("game_*.json")):
        try:
            games.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            print(f"skipping unreadable {p.name}", file=sys.stderr)
    return games


def build_pairs(
    games: List[dict], eps: float,
) -> Tuple[List[str], Dict[Tuple[int, int], PairRecord],
           Dict[Tuple[int, int], PairRecord],
           Dict[Tuple[int, int], int]]:
    labels = sorted({g["label_a"] for g in games}
                    | {g["label_b"] for g in games})
    idx = {lab: i for i, lab in enumerate(labels)}
    pure: Dict[Tuple[int, int], PairRecord] = {}
    mat:  Dict[Tuple[int, int], PairRecord] = {}
    nores: Dict[Tuple[int, int], int] = {}
    for g in games:
        a, b = idx[g["label_a"]], idx[g["label_b"]]
        i, j = min(a, b), max(a, b)
        a_is_i = (a == i)
        for d in (pure, mat):
            d.setdefault((i, j), PairRecord())
        nores.setdefault((i, j), 0)
        out = g["outcome_a"]
        if out == "win":
            win_i = a_is_i
        elif out == "loss":
            win_i = not a_is_i
        else:
            # NO RESULT (user ruling 2026-08-17): a capped/stalled
            # game is a truncated observation, not a draw. It is
            # excluded from the PURE fit and recorded as an absence.
            nores[(i, j)] += 1
            if g.get("margin_a") is None:
                # timeout_kill artifacts (round-32 C5) carry no
                # final state: absences under BOTH conventions.
                continue
            m = float(g.get("margin_a", 0.0))
            if abs(m) <= eps:
                mat[(i, j)].draws += 1
            else:
                ahead_is_a = m > 0
                win_i = ahead_is_a == a_is_i
                if win_i:
                    mat[(i, j)].wins_i += 1
                else:
                    mat[(i, j)].wins_j += 1
            continue
        for d in (pure, mat):
            if win_i:
                d[(i, j)].wins_i += 1
            else:
                d[(i, j)].wins_j += 1
    return labels, pure, mat, nores


def _print_table(title: str, labels, elo, se, pairs,
                 nores=None) -> None:
    print(f"\n=== {title} ===")
    order = sorted(range(len(labels)),
                   key=lambda k: (-elo[k] if elo[k] is not None
                                  else float("inf")))
    for k in order:
        if elo[k] is None:
            print(f"  {labels[k]:<10} n/a (no decisive games)")
        else:
            print(f"  {labels[k]:<10} {elo[k]:>8.1f} ± {se[k]:.0f}")
    for (i, j), rec in sorted(pairs.items()):
        nr = (nores or {}).get((i, j), 0)
        tail = f" + {nr} no-result (excluded)" if nr else ""
        print(f"    {labels[i]} vs {labels[j]}: "
              f"{rec.wins_i}-{rec.draws}-{rec.wins_j} (W-D-L){tail}")


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("games_dir", type=Path)
    ap.add_argument("--anchor", default="dummy")
    ap.add_argument("--eps", type=float, default=0.02,
                    help="material dead zone: |margin| <= eps stays "
                         "a draw under the MATERIAL convention.")
    ap.add_argument("--save-json", type=Path, default=None)
    ap.add_argument("--no-catalog", action="store_true",
                    help="Skip the automatic elo-catalog update "
                         "(tools/elo_catalog.py; on by default per "
                         "user directive 2026-08-17).")
    ap.add_argument("--catalog-protocol", default=None,
                    help="Optional protocol note recorded on the "
                         "catalog edge, e.g. 'mcts:32'.")
    ap.add_argument("--catalog-procedure", default=None,
                    help="Structured procedure tag for the catalog "
                         "edge (e.g. 'mcts:32'). Required when "
                         "collecting a LEGACY games dir (no "
                         "procedure fields in the files) into a "
                         "tagged catalog; ignored when the files "
                         "declare a procedure (round-16 C0).")
    ap.add_argument("--catalog-path", type=Path, default=None,
                    help="Alternate catalog file -- a new procedure "
                         "gets its own catalog rather than mixing "
                         "estimands in one fit (round-16 C3).")
    ap.add_argument("--catalog-max-turns", default=None,
                    help="Operator-declared turn horizon for a "
                         "LEGACY games dir (files without a "
                         "max_turns field); refused when it "
                         "contradicts a measured value (round-25 "
                         "C5). The empty string \"\" CLEARS a "
                         "stale declared horizon from the dir's "
                         "edge (round-27 C0).")
    ap.add_argument("--catalog-alias", action="append", default=[],
                    metavar="RUN_LABEL=CATALOG_LABEL",
                    help="Rename a run-local label to its canonical "
                         "catalog label so the edge chains to an "
                         "existing rated node. Repeatable. The "
                         "mapping persists on the dir's edges and "
                         "re-seeds on later collects of the same "
                         "dir; the identity form RUN_LABEL="
                         "RUN_LABEL clears a stale persisted "
                         "entry.")
    args = ap.parse_args(argv[1:])

    games = load_games(args.games_dir)
    if not games:
        print("no game files found")
        return 2
    labels, pure, mat, nores = build_pairs(games, args.eps)
    anchor_idx = (labels.index(args.anchor)
                  if args.anchor in labels else 0)
    n = len(labels)
    n_nores = sum(nores.values())
    results = {}
    # Estimand guard BEFORE any table is printed (round-12 C6: a
    # mixed fit was fully rendered above its own refusal), None
    # normalized to 'legacy' so the set sorts (round-12 C0), and
    # plan runs must also share one pt_config (round-12 C7).
    procs = {(g.get("procedure_a") or "legacy",
              g.get("procedure_b") or "legacy") for g in games}
    if len(procs) > 1:
        raise SystemExit(f"mixed procedures in one games dir: "
                         f"{sorted(procs)} -- estimands don't mix.")
    pt_cfgs = {json.dumps(g.get("pt_config"), sort_keys=True)
               for g in games if g.get("pt_config") is not None}
    if len(pt_cfgs) > 1:
        raise SystemExit("mixed --pt-* configs in one games dir -- "
                         "estimands don't mix.")
    ts_cfgs = {json.dumps(g.get("turn_config"), sort_keys=True)
               for g in games if g.get("turn_config") is not None}
    if len(ts_cfgs) > 1:
        raise SystemExit("mixed turn-search configs in one games "
                         "dir -- estimands don't mix (round-32 "
                         "C3).")
    _mts = {g.get("max_turns") for g in games}
    if len(_mts) > 1:
        raise SystemExit(
            f"mixed turn horizons in one games dir: "
            f"{sorted(str(m) for m in _mts)} -- the horizon "
            f"decides decisive-vs-absence, so estimands don't mix "
            f"(round-24 C9).")
    _mt = next(iter(_mts)) if _mts else None
    _proc_tag = next(iter(procs)) if procs else ("legacy", "legacy")

    for title, pairs, nr in (
            ("PURE (decisive only, primary)", pure, nores),
            ("MATERIAL-SIGN (diagnostic)", mat, None)):
        # Only labels with decisive mass UNDER THIS CONVENTION are
        # fitted (round-28 C1, mirroring refit): a zero-mass label
        # got a gauge-arbitrary elo printed with se 0 -- and an
        # all-capped match rendered as an exact 0.0 +- 0 tie.
        mass = [0.0] * n
        for (i, j), rec in pairs.items():
            m = rec.wins_i + rec.wins_j + rec.draws
            mass[i] += m
            mass[j] += m
        rated_idx = [k for k in range(n) if mass[k] > 0]
        elo_full = [None] * n
        se_full = [None] * n
        if rated_idx:
            remap = {k: r for r, k in enumerate(rated_idx)}
            rpairs = {(remap[i], remap[j]): rec
                      for (i, j), rec in pairs.items()
                      if i in remap and j in remap
                      and (rec.wins_i + rec.wins_j + rec.draws) > 0}
            if anchor_idx not in remap:
                print(f"NOTE: anchor {labels[anchor_idx]} has no "
                      f"decisive games under {title}; gauged on "
                      f"{labels[rated_idx[0]]} = 0")
            elo_r, se_r = fit_elo(
                len(rated_idx), rpairs, remap.get(anchor_idx, 0),
                anchor_elo=0.0, prior_games=1.0, draw_weight=0.5,
                prior_scope="played")
            for k, r in remap.items():
                elo_full[k] = float(elo_r[r])
                se_full[k] = float(se_r[r])
        _print_table(title, labels, elo_full, se_full, pairs,
                     nores=nr)
        results[title] = {lab: {"elo": e, "se": s}
                          for lab, e, s in zip(labels, elo_full,
                                               se_full)}
    print(f"\ngames: {len(games)} ({n_nores} no-result, excluded from "
          f"PURE) | anchor: {labels[anchor_idx]} = 0"
          f" | procedure: {_proc_tag[0]}/{_proc_tag[1]}")
    # Auto-update the committed Elo catalog (user directive
    # 2026-08-17): every collected games dir records its PURE
    # per-pair W-D-L as an edge (idempotent by dir name) and the
    # global ratings refit. See tools/elo_catalog.py.
    if not args.no_catalog and _proc_tag[0] != _proc_tag[1]:
        # A heterogeneous-procedure match (e.g. plan vs mcts) is not
        # representable as a catalog edge: the global refit assumes
        # one protocol per component (round-14 C1). The fit above
        # stands; the catalog is skipped loudly.
        print(f"catalog SKIPPED: heterogeneous procedures "
              f"{_proc_tag[0]} vs {_proc_tag[1]} do not form a "
              f"catalog edge. Use --no-catalog to silence.")
    elif not args.no_catalog:
        try:
            from tools.elo_catalog import update_from_games
            # Structured provenance (round-15 C0): the measured
            # procedure tag rides a dedicated field the catalog
            # guard compares; the free-text note stays prose. The
            # games' validated pt_config (unique per dir, guarded
            # above) rides along on EVERY tagged branch -- the
            # round-16 version attached it only on a dead legacy
            # branch, leaving the guard's knob component inert
            # (round-17 C0).
            proto = None
            if _proc_tag[0] != "legacy":
                proto = {"procedure": _proc_tag[0]}
                if (args.catalog_procedure
                        and args.catalog_procedure != _proc_tag[0]):
                    raise SystemExit(
                        f"--catalog-procedure "
                        f"{args.catalog_procedure!r} contradicts "
                        f"the measured tag {_proc_tag[0]!r}.")
            elif args.catalog_procedure:
                # Operator-declared procedure for a legacy dir
                # (round-16 C0).
                proto = {"procedure": args.catalog_procedure}
            if proto is not None and pt_cfgs:
                proto["pt_config"] = json.loads(next(iter(pt_cfgs)))
            if proto is not None and ts_cfgs:
                proto["turn_config"] = json.loads(
                    next(iter(ts_cfgs)))
            # The horizon rides the edge so the catalog-level
            # guard can compare it across dirs (round-25 C5). NOT
            # gated on proto existing: a legacy dir with only
            # --catalog-max-turns (no --catalog-procedure) must
            # still stamp -- proto {"max_turns": N} has no
            # "procedure", so the round-16 polarity guard is
            # untouched (round-26 C2: the nested version silently
            # ignored the flag exactly there).
            _cmt = args.catalog_max_turns
            if _cmt not in (None, ""):
                try:
                    _cmt = int(_cmt)
                except ValueError:
                    raise SystemExit(
                        f"--catalog-max-turns must be an integer "
                        f"or \"\" (clear); got {_cmt!r}")
            if _mt is not None and _cmt == "":
                raise SystemExit(
                    f"--catalog-max-turns \"\" cannot clear a "
                    f"MEASURED horizon (files say max_turns="
                    f"{_mt}).")
            if (_mt is not None and _cmt not in (None, "")
                    and _cmt != _mt):
                raise SystemExit(
                    f"--catalog-max-turns {_cmt} contradicts the "
                    f"measured max_turns={_mt}.")
            if _cmt == "":
                # Clear sentinel rides to record_edge (round-27
                # C0) and suppresses the stale-edge migration.
                proto = dict(proto or {})
                proto["max_turns"] = ""
            else:
                _mt_eff = _mt if _mt is not None else _cmt
                if _mt_eff is not None:
                    proto = dict(proto or {})
                    proto["max_turns"] = int(_mt_eff)
            if args.catalog_protocol is not None:
                # Empty string CLEARS the edge's carried-forward
                # note (round-20 C3: the clear hatch existed in
                # record_edge but no CLI path could reach it).
                proto = dict(proto or {})
                proto["note"] = args.catalog_protocol
            _bad = [p for p in args.catalog_alias
                    if "=" not in p
                    or not p.split("=", 1)[0]
                    or not p.split("=", 1)[1]]
            if _bad:
                raise SystemExit(
                    f"--catalog-alias needs RUN_LABEL="
                    f"CATALOG_LABEL; got {_bad} (round-24 C10: a "
                    f"silently dropped alias records the edge "
                    f"under a phantom run-local node).")
            lmap = dict(p.split("=", 1)
                        for p in args.catalog_alias)
            _unknown = sorted(k for k in set(lmap) - set(labels)
                              if lmap[k] != k)
            if _unknown:
                raise SystemExit(
                    f"--catalog-alias keys not among this dir's "
                    f"run labels {sorted(labels)}: {_unknown} "
                    f"(direction is RUN_LABEL=CATALOG_LABEL; an "
                    f"identity alias K=K is allowed to clear a "
                    f"persisted dir-scoped mapping).")
            kw = ({"path": args.catalog_path}
                  if args.catalog_path else {})
            update_from_games(args.games_dir, games, protocol=proto,
                              label_map=lmap, **kw)
            print("elo catalog updated (see tools/elo_catalog.py "
                  "show); use --no-catalog to skip")
        except ValueError as e:
            # Estimand mismatch is a REFUSAL, not a warning: a
            # swallowed refusal lets a verdict match print its Elo
            # while its edge silently never reaches the board
            # (round-15 C0).
            raise SystemExit(str(e))
        except Exception as e:                      # noqa: BLE001
            print(f"WARNING: elo catalog update failed: {e!r} -- "
                  f"the fit above is unaffected; update manually "
                  f"via tools/elo_catalog.py")
    if args.save_json:
        args.save_json.write_text(
            json.dumps({"n_games": len(games), "tables": results},
                       indent=2), encoding="utf-8")
        print(f"written: {args.save_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
