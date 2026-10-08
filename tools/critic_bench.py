#!/usr/bin/env python3
"""The step-1 critics on the turn-value benchmark (docs/selfplay_program_20261008.md,
"Step 1"; the benchmark is docs/turn_value_prereg_20260925.md "Validation" on
branch exp/turn-value, its records HF tier-b/turn_value_20260925/).

    python tools/critic_bench.py rebuild --records validation.json --dataset DIR \\
        --vocab-from parity3.pt --out OUT/bench [--workers N]
    python tools/critic_bench.py read --out OUT/bench --critic T25=OUT/T25.pt ... [--device cuda]
    python tools/critic_bench.py stats --out OUT/bench
    python tools/critic_bench.py extract --cache validation.pt --verdict verdict.json --to ARRAYS

rebuild  Each benchmark position again from its corpus game (the corpus the
         benchmark was played from, tools/bench_states.reconstruct_boundary),
         and per candidate turn the state before its end_turn (its recorded
         commands and recruit rejections applied, checked against the
         recorded digest) and the state right after it (the simulator's
         end_turn: the opponent's turn begun; read 0). Both are encoded in
         both views (wesnoth_ai/critic_data.py). A candidate whose digest
         differs, or whose rebuild fails, is counted and read by no grader; a
         position whose base turn does is left out whole. The HP margin right
         after the turn is compared with the recorded `hp_margin_post` and
         the differences counted, not dropped: on the maps with a third side
         (Caves of the Basilisk, Silverhead Crossing, Sullas Ruins) the
         benchmark's simulator gave that side a turn after side 2's (fixed in
         0.7.7), so its recorded read 0 is the third side's turn start, where
         the margin is still the one before the end_turn; the playouts went
         on from there to the turn start read here. One pickle per position
         is appended to states.pkl as positions finish, a row to
         rebuild.jsonl, and a resumed run skips the positions it holds.
reads    Each critic's value of each state in its own view, signed to the
         mover (the opponent moves at read 0): reads_<NAME>.json, one per
         critic, as each finishes.
stats    The pre-registered statistics at each read, against the
         luck-adjusted truth of playouts 9 to 28 (the verdict's luck
         coefficients) and the raw truth beside it: per critic, the
         corrected within-position correlation and the selection gain over
         the base turn, each paired against the static HP margin at the same
         state, and T100 against T25; then the reading
         (wesnoth_ai/turn_bench_stats.readings), printed with why it fires.
         readout.json and readout.md.
extract  The benchmark's truth as arrays (BENCH_ARRAYS, committed): per
         candidate its position, slot, game, outcomes, playout and turn luck
         and recorded HP margin after the turn, and the verdict's luck
         coefficients, from the benchmark's cache and verdict.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wesnoth_ai import critic_data as cd  # noqa: E402
from wesnoth_ai import turn_bench_stats as S  # noqa: E402

log = logging.getLogger("critic_bench")

BENCH_ARRAYS = Path(__file__).resolve().parent.parent / "training/metrics/turn_value_20260925/bench_truth.npz"
DIGEST_VERSION = 1          # the benchmark's digests predate the sighting record (state_digest version 2)
MARGIN = "hp_margin"
SIZE_PAIR = ("T100", "T25")


# ---------------------------------------------------------------------
# Rebuilding the benchmark's states
# ---------------------------------------------------------------------

def candidates(record: Dict) -> List[Dict]:
    """Slot 0 is the base turn, slot j the j-th alternative."""
    return [record["base"]] + list(record.get("alternatives", []))


def apply_turn(cs, commands: Sequence[list], rejections: Sequence[Sequence[int]]) -> None:
    """A candidate turn's commands on `cs`, each recruit rejection where it
    happened (its command index, as tools/game_record.walk_core applies a
    record's)."""
    by_index: Dict[int, List[Tuple[int, int]]] = {}
    for k, x, y in rejections:
        by_index.setdefault(int(k), []).append((int(x), int(y)))
    for k, cmd in enumerate(commands):
        for x, y in by_index.get(k, ()):
            cs.core.add_recruit_rejected(x, y)
        cs.apply_command(list(cmd))
    for x, y in by_index.get(len(commands), ()):
        cs.core.add_recruit_rejected(x, y)


def end_turn(cs, scenario_id: str, salt: str):
    """The core after the simulator's end_turn from `cs`'s position: the
    next player side's turn begun, as the benchmark's playouts started."""
    from tools.wesnoth_sim import WesnothSim
    gs = cs.to_state()
    sim = WesnothSim(gs, scenario_id, max_turns=int(gs.global_info.turn_number) + 30,
                     apply_scenario_events=False, begin_turn=False)
    sim._is_search_fork = True
    sim._seed_salt = salt
    sim.step({"type": "end_turn"})
    if sim.done:
        raise RuntimeError("the end_turn ended the game")
    return sim.core


def rebuild_position(record: Dict, dataset: Path, type_to_id: Dict[str, int],
                     faction_to_id: Dict[str, int]) -> Dict:
    """One position's candidates: per slot its checks and, when they hold,
    its encodings {read: {view: packed}} and the side to move at each read."""
    import gzip
    from tools.bench_states import reconstruct_boundary
    from wesnoth_ai.classes import state_digest
    from wesnoth_ai.game_core import core_of
    meta = record["meta"]
    out = {"index": record["index"], "file": meta["file"], "candidates": []}
    with gzip.open(Path(dataset) / meta["file"], "rt", encoding="utf-8") as f:
        data = json.load(f)
    res = reconstruct_boundary(data, int(meta["cut_turn"]))
    if res is None or res[1] != int(meta["begin_side"]):
        raise RuntimeError(f"{meta['file']}: the boundary does not reconstruct")
    boundary = core_of(res[0])
    mover = int(record["side"])
    for slot, cand in enumerate(candidates(record)):
        row: Dict = {"slot": slot}
        snap = cand.get("pre_end_turn")
        if cand.get("terminal_in_turn") or not snap:
            row["skipped"] = "terminal_in_turn" if cand.get("terminal_in_turn") else "no_snapshot"
            out["candidates"].append(row)
            continue
        try:
            pre = boundary.fork()
            apply_turn(pre, snap["commands"], snap.get("rejections", []))
            row["digest_ok"] = state_digest(pre.to_state(), version=DIGEST_VERSION) == snap["digest"]
            post = end_turn(pre.fork(), meta["scenario_id"], record.get("turn_salt") or "")
            row["hp_margin_pre"] = cd.hp_margin(pre.to_state(), mover)
            row["hp_margin_post"] = cd.hp_margin(post.to_state(), mover)
            row["hp_margin_post_recorded"] = cand.get("hp_margin_post")
            row["margin_ok"] = row["hp_margin_post"] == cand.get("hp_margin_post")
            row["to_move"] = {"pre": int(pre.core.current_side), "read0": int(post.core.current_side)}
            if row["digest_ok"]:
                row["raws"] = {read: {view: cd.pack_raw(cd.encode_view(cs, view, type_to_id, faction_to_id))
                                      for view in cd.VIEWS}
                               for read, cs in (("pre", pre), ("read0", post))}
        except Exception as e:  # noqa: BLE001 - counted per candidate
            row["error"] = f"{type(e).__name__}: {e}"[:300]
        out["candidates"].append(row)
    return out


_W: Dict = {}


def _init(dataset: str, type_to_id: Dict[str, int], faction_to_id: Dict[str, int]) -> None:
    _W.update(dataset=Path(dataset), type_to_id=type_to_id, faction_to_id=faction_to_id)


def _rebuild(record: Dict) -> Dict:
    try:
        return rebuild_position(record, _W["dataset"], _W["type_to_id"], _W["faction_to_id"])
    except Exception as e:  # noqa: BLE001 - a position that fails is counted, not fatal
        return {"index": record["index"], "file": record["meta"]["file"], "candidates": [],
                "error": f"{type(e).__name__}: {e}"[:300]}


def read_states(path: Path) -> Iterator[Dict]:
    """The positions a rebuild appended, a cut last one skipped."""
    from wesnoth_ai import unpickle
    if not path.exists():
        return
    with path.open("rb") as f:
        while True:
            try:
                yield unpickle.load(f)
            except EOFError:
                return
            except (pickle.UnpicklingError, ValueError, AttributeError):
                log.warning("%s: a cut record at the end is skipped", path)
                return


def cmd_rebuild(args) -> int:
    from wesnoth_ai.critic import reference_vocab
    type_to_id, faction_to_id = reference_vocab(args.vocab_from)
    records = json.loads(args.records.read_text(encoding="utf-8"))["positions"]
    args.out.mkdir(parents=True, exist_ok=True)
    states = args.out / "states.pkl"
    done = {p["index"] for p in read_states(states)}
    todo = [r for r in records if r["index"] not in done]
    log.info("%d positions, %d rebuilt before", len(records), len(done))
    t0 = time.time()
    with states.open("ab") as sink, (args.out / "rebuild.jsonl").open("a", encoding="utf-8") as rows, \
            mp.get_context("spawn").Pool(args.workers, initializer=_init,
                                         initargs=(str(args.dataset), type_to_id, faction_to_id)) as pool:
        for i, pos in enumerate(pool.imap_unordered(_rebuild, todo), 1):
            pickle.dump(pos, sink, protocol=pickle.HIGHEST_PROTOCOL)
            sink.flush()
            rows.write(json.dumps({"index": pos["index"], "error": pos.get("error"),
                                   "candidates": [{k: v for k, v in c.items() if k != "raws"}
                                                  for c in pos["candidates"]]}) + "\n")
            rows.flush()
            if i % 20 == 0 or i == len(todo):
                log.info("%d/%d positions, %.0f s", i, len(todo), time.time() - t0)
    counts = rebuild_counts(list(read_states(states)))
    (args.out / "rebuild_summary.json").write_text(json.dumps(counts, indent=1), encoding="utf-8")
    log.info("REBUILD_DONE %s", counts)
    return 0


def rebuild_counts(positions: Sequence[Dict]) -> Dict[str, int]:
    c = {"positions": len(positions), "position_errors": 0, "candidates": 0, "skipped": 0, "errors": 0,
         "digest_mismatch": 0, "post_margin_differs": 0, "post_margin_differs_recorded_is_pre": 0,
         "encoded": 0}
    for pos in positions:
        c["position_errors"] += int(bool(pos.get("error")))
        for cand in pos["candidates"]:
            c["candidates"] += 1
            c["skipped"] += int("skipped" in cand)
            c["errors"] += int("error" in cand)
            c["digest_mismatch"] += int(cand.get("digest_ok") is False)
            differs = cand.get("margin_ok") is False
            c["post_margin_differs"] += int(differs)
            c["post_margin_differs_recorded_is_pre"] += int(
                differs and cand.get("hp_margin_post_recorded") == cand.get("hp_margin_pre"))
            c["encoded"] += int("raws" in cand)
    return c


# ---------------------------------------------------------------------
# Reading the critics
# ---------------------------------------------------------------------

def cmd_read(args) -> int:
    import torch
    from wesnoth_ai.critic import critic_values, file_sha256, load_critic
    positions = list(read_states(args.out / "states.pkl"))
    device = torch.device(args.device)
    for spec in args.critic:
        name, path = spec.split("=", 1)
        encoder, model, meta = load_critic(Path(path), device)
        view = meta["critic"]["view"]
        keys, raws, signs = [], [], []
        for pos in positions:
            for cand in pos["candidates"]:
                for read, by_view in cand.get("raws", {}).items():
                    keys.append((pos["index"], cand["slot"], read))
                    raws.append(cd.unpack_raw(by_view[view]))
                    mover_moves = cand["to_move"][read] == cand["to_move"]["pre"]
                    signs.append(1.0 if mover_moves else -1.0)
        t0 = time.time()
        values = critic_values(encoder, model, raws, device, batch=args.batch,
                               autocast=torch.bfloat16 if device.type == "cuda" else None)
        reads = [{"index": i, "slot": s, "read": r, "value": sign * v}
                 for (i, s, r), sign, v in zip(keys, signs, values)]
        payload = {"critic": name, "checkpoint": str(path), "sha256": file_sha256(Path(path)),
                   "view": view, "meta": meta.get("critic"), "training": meta.get("training"), "reads": reads}
        tmp = args.out / f"reads_{name}.json.tmp"
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, args.out / f"reads_{name}.json")
        log.info("%s: %d reads in %.0f s", name, len(reads), time.time() - t0)
        del encoder, model
    return 0


# ---------------------------------------------------------------------
# Statistics and the reading
# ---------------------------------------------------------------------

def benchmark_arrays(cache_path: Path, verdict_path: Path) -> Dict:
    """Per candidate of the benchmark's cache (tools/turn_value.py on
    exp/turn-value): its position index, slot, cluster, outcomes, playout
    and turn luck, recorded HP margin after the turn, and `obs8`'s value
    head read offline before the end_turn (`value_reference`) and after it
    (`value_post`); and the verdict's luck coefficients."""
    import torch
    c = torch.load(cache_path, map_location="cpu", weights_only=True)
    verdict = json.loads(Path(verdict_path).read_text(encoding="utf-8"))

    def column(values):
        return np.asarray([math.nan if v is None else float(v) for v in values])
    return {"index": c["index"].numpy(), "slot": c["slot"].numpy(), "cluster": np.asarray(c["group"]),
            "outcomes": c["outcomes"].double().numpy(), "luck": c["luck"].double().numpy(),
            "turn_luck": c["turn_luck"].double().numpy(),
            "hp_margin_post": column(c["baselines"]["hp_margin_post"]),
            "value_reference": c["value_reference"].double().numpy(),
            "value_post": column(c["baselines"]["value_post"]),
            "beta": np.asarray(verdict["luck"]["beta"], dtype=float)}


def save_arrays(arrays: Dict, path: Path) -> None:
    """The arrays at the precision the cache holds them (float32 outcomes
    and lucks, which convert back exactly)."""
    np.savez_compressed(path, index=arrays["index"], slot=arrays["slot"], cluster=arrays["cluster"].astype(str),
                        outcomes=arrays["outcomes"].astype(np.float32), luck=arrays["luck"].astype(np.float32),
                        turn_luck=arrays["turn_luck"].astype(np.float32), hp_margin_post=arrays["hp_margin_post"],
                        value_reference=arrays["value_reference"].astype(np.float32),
                        value_post=arrays["value_post"].astype(np.float32), beta=arrays["beta"])


def load_arrays(path: Path = BENCH_ARRAYS) -> Dict:
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    for k in ("outcomes", "luck", "turn_luck", "value_reference", "value_post"):
        out[k] = out[k].astype(float)
    return out


def cmd_extract(args) -> int:
    save_arrays(benchmark_arrays(args.cache, args.verdict), args.to)
    log.info("wrote %s", args.to)
    return 0


def grades_at(arrays: Dict, positions: Sequence[Dict], critic_reads: Dict[str, Dict], read: str
              ) -> Dict[str, np.ndarray]:
    """Per grader, a grade per benchmark row at `read` (NaN where the
    candidate did not rebuild; every row of a position whose base turn did
    not rebuild)."""
    row_of = {(int(i), int(s)): k for k, (i, s) in enumerate(zip(arrays["index"], arrays["slot"]))}
    n = len(arrays["index"])
    margin = np.full(n, np.nan)
    ok = np.zeros(n, dtype=bool)
    for pos in positions:
        cands = {c["slot"]: c for c in pos["candidates"]}
        base_ok = "raws" in cands.get(0, {})
        for slot, cand in cands.items():
            k = row_of.get((int(pos["index"]), int(slot)))
            if k is None or "raws" not in cand or not base_ok:
                continue
            ok[k] = True
            margin[k] = cand["hp_margin_pre"] if read == "pre" else cand["hp_margin_post"]
    grades = {MARGIN: margin}
    for name, payload in critic_reads.items():
        g = np.full(n, np.nan)
        for r in payload["reads"]:
            k = row_of.get((int(r["index"]), int(r["slot"])))
            if r["read"] == read and k is not None and ok[k]:
                g[k] = r["value"]
        grades[name] = g
    return grades


def statistics(arrays: Dict, grades_by_read: Dict[str, Dict[str, np.ndarray]], critics: Sequence[str],
               correlation_resamples: int = S.CORRELATION_RESAMPLES, gain_resamples: int = S.GAIN_RESAMPLES
               ) -> Dict:
    """Per truth basis and read: `paired_correlations` and `paired_gains`
    of every critic against the HP margin, and of T100 against T25 when
    both are read."""
    positions = np.unique(arrays["index"], return_inverse=True)[1]
    truths = {"adjusted": S.adjusted_outcomes(arrays["outcomes"], arrays["luck"], arrays["turn_luck"],
                                              arrays["beta"])[:, S.TRUTH_FROM:],
              "raw": arrays["outcomes"][:, S.TRUTH_FROM:]}
    pairs = [(c, MARGIN) for c in critics]
    if all(name in critics for name in SIZE_PAIR):
        pairs.append(SIZE_PAIR)
    out: Dict = {}
    for basis, truth in truths.items():
        out[basis] = {}
        for read, grades in grades_by_read.items():
            out[basis][read] = {
                "correlations": S.paired_correlations(grades, truth, positions, arrays["cluster"], pairs,
                                                      resamples=correlation_resamples),
                "gains": S.paired_gains(grades, S.truth_mean(truth), positions, arrays["slot"], pairs,
                                        resamples=gain_resamples)}
    return out


def markdown(stats: Dict, reading: Dict, counts: Dict) -> str:
    lines = [f"# Step 1 readout: {reading['reading']}", "", reading["why"], "",
             f"rebuild: {json.dumps(counts)}", ""]
    for basis in ("adjusted", "raw"):
        for read, s in stats[basis].items():
            lines += [f"## {basis} truth, {read}", "",
                      "| grader | corrected r | minus HP margin | selection gain | minus HP margin |",
                      "|---|---|---|---|---|"]
            for name, c in s["correlations"]["graders"].items():
                g = s["gains"]["graders"][name]
                dc = s["correlations"]["differences"].get(f"{name}-{MARGIN}")
                dg = s["gains"]["differences"].get(f"{name}-{MARGIN}")
                lines.append(f"| {name} | {c['corrected']:.3f} +- {c['corrected_se']:.3f} | "
                             f"{_fmt(dc)} | {g['mean']:+.3f} +- {g['se']:.3f} | {_fmt(dg)} |")
            size = "-".join(SIZE_PAIR)
            if size in s["correlations"]["differences"]:
                lines.append(f"\n{size}: correlation {_fmt(s['correlations']['differences'][size])}, "
                             f"selection gain {_fmt(s['gains']['differences'][size])}")
            lines += ["", f"positions: correlation {s['correlations']['clusters']}, gains {s['gains']['positions']}", ""]
    return "\n".join(lines)


def _fmt(d: Optional[Dict]) -> str:
    return "-" if not d else f"{d['mean']:+.3f} +- {d['se']:.3f}"


def cmd_stats(args) -> int:
    from wesnoth_ai import __version__
    positions = list(read_states(args.out / "states.pkl"))
    critic_reads = {}
    for path in sorted(args.out.glob("reads_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        critic_reads[payload["critic"]] = payload
    critics = sorted(critic_reads)
    arrays = load_arrays(args.arrays)
    grades = {read: grades_at(arrays, positions, critic_reads, read) for read in S.READS}
    stats = statistics(arrays, grades, critics)
    reading = S.readings(stats["adjusted"], critics, margin=MARGIN, large=SIZE_PAIR[0], small=SIZE_PAIR[1])
    counts = rebuild_counts(positions)
    out = {"reading": reading, "statistics": stats, "rebuild": counts, "code_version": __version__,
           "critics": {k: {f: v.get(f) for f in ("checkpoint", "sha256", "view", "meta", "training")}
                       for k, v in critic_reads.items()},
           "truth": {"playouts": f"{S.TRUTH_FROM + 1}-28", "basis": "luck-adjusted (verdict beta)",
                     "beta": arrays["beta"].tolist()},
           "sources": {"arrays": str(args.arrays)}}
    (args.out / "readout.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    (args.out / "readout.md").write_text(markdown(stats, reading, counts), encoding="utf-8")
    print(f"READING {reading['reading']}: {reading['why']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("rebuild")
    r.add_argument("--records", type=Path, required=True, help="the benchmark's validation.json")
    r.add_argument("--dataset", type=Path, required=True, help="the corpus the benchmark was played from")
    r.add_argument("--vocab-from", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    d = sub.add_parser("read")
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--critic", action="append", required=True, help="NAME=CHECKPOINT")
    d.add_argument("--device", default="cpu")
    d.add_argument("--batch", type=int, default=64)
    s = sub.add_parser("stats")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--arrays", type=Path, default=BENCH_ARRAYS, help="the benchmark's truth (extract)")
    e = sub.add_parser("extract")
    e.add_argument("--cache", type=Path, required=True, help="the benchmark's validation.pt")
    e.add_argument("--verdict", type=Path, required=True, help="the benchmark's verdict.json")
    e.add_argument("--to", type=Path, default=BENCH_ARRAYS)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return {"rebuild": cmd_rebuild, "read": cmd_read, "stats": cmd_stats,
            "extract": cmd_extract}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
