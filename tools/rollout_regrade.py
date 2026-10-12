#!/usr/bin/env python3
"""Rollouts from the mover's own view, re-grading the turn benchmark
(docs/rollout_regrade_prereg_20261012.md; the readout is
tools/rollout_readout.py).

    python tools/rollout_regrade.py --records validation.json --dataset DIR \\
        --checkpoint own2.pt --step1 rebuild.jsonl --out OUT [--workers N] [--device cuda]

The states. Each candidate turn of the benchmark (HF
tier-b/turn_value_20260925/validation.json: 199 positions, 966 candidates
that do not end the game in their turn) is rebuilt as step 1 rebuilt it
(tools/critic_bench.py `rebuild_candidate`: the position from its corpus
game, the candidate's recorded commands and recruit rejections, the digest
checked, the simulator's end_turn), and its HP margins before and after the
end_turn are compared with step 1's rebuild (`--step1`, the rebuild.jsonl of
HF tier-b/critic_step1_20261009/bench.tar.gz, which holds that rebuild's
checks and encodings, not its states).

The world. From the state right after the end_turn (the opponent's turn
begun), the mover's observed world (wesnoth_ai/lookahead_world.py, the
"observed" determinization built for the mover): every unit of another side
the mover does not see removed, the opponent's gold and base income set to
the mover's.

The rollouts. ROLLOUTS per candidate, both sides the checkpoint at 0 memory
slots at raw:t0.5+eo-1.5, to the READS[-1]-th player turn start after the
candidate or the game's end. Read k is the k-th player turn start after the
candidate, read 0 the state right after it (as the turn-value records count
them: the mover's own turn starts are the odd reads). At each read of READS:
the mover's HP margin over the two player sides
(wesnoth_ai/critic_data.player_hp_margin) and the value head at 0 slots from
the side to move, signed to the mover (tools/turn_gap.value_read). A rollout
that ended before a read takes the result there: the value is the game's
result for the mover (+1, -1, 0 undecided), the margin the final
position's.

Seeds. Rollout r of candidate (position i, slot c) draws its dice from the
world salt "regrade:<seed>:p<i>:c<c>:r<r>"; side s samples with numpy seed
crc32(salt) + s - 1. Both are recorded with each rollout.

Parallel play. One inference server (tools/eval_inference_server.py, the
eval path's) holds the checkpoint; each of --workers spawned processes
plays one rollout at a time through it, one decision in flight, so the
server batches across workers. A task is a position (its corpus game
replayed once); each candidate's record goes to this process through a
queue as it finishes and is appended to rollouts.jsonl. A resumed run skips
the candidates whose last record is complete and redoes the others; the
readout reads the last record of each candidate. `--workers 0` plays in
this process on a local copy of the checkpoint (tests, timing checks).

Failures are counted, never dropped: a position whose game does not
rebuild, a candidate that fails its rebuild or whose digest or margins
differ from step 1's, and a rollout that raises are recorded with their
reason; regrade_summary.json counts them by reason, with each rollout's
decisions, forwards and seconds (the worker's own time per decision is the
wall minus the forwards' round trips).
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import queue
import statistics
import sys
import time
import zlib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wesnoth_ai.classes import PLAYER_SIDES  # noqa: E402

log = logging.getLogger("rollout_regrade")

READS = (2, 4)
ROLLOUTS = 8
TEMPERATURE = 0.5
END_TURN_OFFSET = -1.5
MEMORY_SLOTS = 0
DEFAULT_SEED = 20261012
EXIT_INCOMPLETE = 5          # some candidate has no record, or no complete one
QUEUE_GRACE_S = 60.0         # after every task returned, how long a record may still be in flight


@dataclass(frozen=True)
class RolloutConfig:
    seed: int = DEFAULT_SEED
    rollouts: int = ROLLOUTS
    reads: Tuple[int, ...] = READS
    temperature: float = TEMPERATURE
    end_turn_offset: float = END_TURN_OFFSET
    memory_slots: int = MEMORY_SLOTS


# ---------------------------------------------------------------------
# Seeds and players
# ---------------------------------------------------------------------

def rollout_salt(seed: int, index: int, slot: int, r: int) -> str:
    return f"regrade:{seed}:p{index}:c{slot}:r{r}"


def sampling_seed(salt: str, side: int) -> int:
    return zlib.crc32(salt.encode("utf-8")) + int(side) - 1


def rollout_pairs(base, cfg: RolloutConfig, salt: str) -> Dict[int, object]:
    """Both sides' players of one rollout: the checkpoint at cfg's decode
    and memory, each side seeded from the salt."""
    from tools.eval_players import _PolicyPair
    from tools.raw_player import RawPolicyPlayer
    return {side: _PolicyPair(policy=RawPolicyPlayer(base, cfg.temperature, seed=sampling_seed(salt, side),
                                                     end_turn_offset=cfg.end_turn_offset,
                                                     memory_slots=cfg.memory_slots),
                              label="regrade", side=side)
            for side in PLAYER_SIDES}


# ---------------------------------------------------------------------
# One rollout
# ---------------------------------------------------------------------

def result_for(sim, mover: int) -> float:
    """The finished game's result for the mover: +1 won, -1 lost, 0 neither."""
    if sim.winner == mover:
        return 1.0
    return -1.0 if sim.winner in PLAYER_SIDES else 0.0


def _turn_key(sim) -> Tuple[int, int]:
    return int(sim.turn_number), int(sim.current_side)


def play_rollout(sim, pairs: Dict[int, object], mover: int, reads: Sequence[int],
                 value_of: Callable[[object], float], game_label: str) -> Dict:
    """Play `sim` (the mover's world, the opponent to move at read 0) with
    `pairs` until the last read's player turn start or the game's end.

    Returns {"reads": {k: {"margin", "value", "ended"}}, "decisions",
    "player_turns" (the turn starts reached), "result" and "ended_by"
    (None while the game goes on)}. `value_of(sim)` is the value head at
    the turn start, signed to the mover."""
    from tools.eval_players import play_decision
    from wesnoth_ai.critic_data import player_hp_margin
    taken: Dict[int, Dict] = {}
    turn_starts, at, decisions = 0, _turn_key(sim), 0
    last = max(reads)
    while not sim.done:
        if turn_starts in reads and turn_starts not in taken:
            taken[turn_starts] = {"margin": player_hp_margin(sim.gs, mover), "value": float(value_of(sim)),
                                  "ended": False}
        if turn_starts >= last:
            break
        play_decision(sim, pairs[int(sim.current_side)], game_label)
        decisions += 1
        now = _turn_key(sim)
        if now != at:
            turn_starts, at = turn_starts + 1, now
    result = result_for(sim, mover) if sim.done else None
    for k in reads:
        if k not in taken:
            taken[k] = {"margin": player_hp_margin(sim.gs, mover), "value": result, "ended": True}
    return {"reads": {str(k): taken[k] for k in reads}, "decisions": decisions, "player_turns": turn_starts,
            "result": result, "ended_by": sim.ended_by if sim.done else None}


def mover_world(post_sim, mover: int, salt: str):
    """The mover's observed world of the state right after its end_turn,
    its dice drawn from `salt`."""
    from wesnoth_ai.lookahead_world import build_world
    return build_world(post_sim, mover, "observed", salt)


def value_reader(pairs: Dict[int, object], mover: int) -> Callable[[object], float]:
    """The value head from the side to move at a turn start, signed to the
    mover, read through that side's player (its memory at cfg's slots)."""
    from tools.turn_gap import value_read
    from wesnoth_ai.game_core import snapshot_view

    def value_of(sim) -> float:
        v = value_read(pairs[int(sim.current_side)].policy, snapshot_view(sim.gs), mover, {})
        if v is None:
            raise RuntimeError("the player has no value head to read")
        return v
    return value_of


def run_rollout(post_sim, mover: int, base, counter, cfg: RolloutConfig, index: int, slot: int,
                r: int) -> Dict:
    """Rollout r of candidate (index, slot): its record, or its error."""
    salt = rollout_salt(cfg.seed, index, slot, r)
    rec: Dict = {"r": r, "salt": salt, "seeds": [sampling_seed(salt, s) for s in PLAYER_SIDES]}
    label = f"rg{index}c{slot}r{r}"
    forwards, fwd_secs, t0 = counter.n_forwards, counter.fwd_secs, time.perf_counter()
    pairs = {}
    try:
        world = mover_world(post_sim, mover, salt)
        pairs = rollout_pairs(base, cfg, salt)
        rec.update(play_rollout(world.sim, pairs, mover, cfg.reads, value_reader(pairs, mover), label))
        rec["removed"] = len(world.removed_ids)
    except Exception as e:  # noqa: BLE001 - counted per rollout
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        for pair in pairs.values():
            pair.drop_pending(label)
    rec.update(seconds=round(time.perf_counter() - t0, 4), forwards=counter.n_forwards - forwards,
               forward_seconds=round(counter.fwd_secs - fwd_secs, 4))
    return rec


# ---------------------------------------------------------------------
# One candidate
# ---------------------------------------------------------------------

def step1_problem(row: Dict, step1_row: Optional[Dict]) -> Optional[str]:
    """Why a rebuilt candidate is not step 1's state (None when it is)."""
    if step1_row is None:
        return "not in step 1's rebuild"
    if not step1_row.get("digest_ok") or "error" in step1_row:
        return "step 1 did not rebuild it"
    for key in ("hp_margin_pre", "hp_margin_post"):
        if row.get(key) != step1_row.get(key):
            return f"{key} {row.get(key)} where step 1 has {step1_row.get(key)}"
    return None


def candidate_record(boundary, record: Dict, slot: int, base, counter, cfg: RolloutConfig,
                     step1_row: Optional[Dict]) -> Dict:
    """One candidate's record: its rebuild's checks, then its rollouts
    (none when it is skipped or its state is not step 1's)."""
    from tools.critic_bench import candidates, rebuild_candidate
    cand = candidates(record)[slot]
    index, mover = int(record["index"]), int(record["side"])
    row, _pre, post = rebuild_candidate(boundary, record, slot, cand)
    rec: Dict = {"index": index, "slot": slot, "mover": mover,
                 "proposer": "base" if slot == 0 else cand.get("proposer", "sample"),
                 "rebuild": row}
    if "skipped" in row:
        rec["skipped"] = row["skipped"]
        return rec
    if "error" in row:
        rec["error"] = f"rebuild: {row['error']}"
        return rec
    if not row["digest_ok"]:
        rec["error"] = "digest_mismatch"
        return rec
    problem = step1_problem(row, step1_row)
    if problem is not None:
        rec["error"] = f"step1_mismatch: {problem}"
        return rec
    rec["rollouts"] = [run_rollout(post, mover, base, counter, cfg, index, slot, r) for r in range(cfg.rollouts)]
    return rec


def is_complete(rec: Dict, cfg: RolloutConfig) -> bool:
    """A candidate needs no more work: skipped by the benchmark, or every
    rollout recorded without an error."""
    if "skipped" in rec:
        return True
    rollouts = rec.get("rollouts") or []
    return "error" not in rec and len(rollouts) == cfg.rollouts and not any("error" in x for x in rollouts)


# ---------------------------------------------------------------------
# Positions, in this process or on workers sharing the server
# ---------------------------------------------------------------------

_W: Dict = {}


def _init_worker(spec: Dict, sink) -> None:
    logging.basicConfig(level=getattr(logging, spec["log_level"]),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    import torch
    torch.set_num_threads(1)
    _W.update(spec, sink=sink, cfg=RolloutConfig(**spec["cfg"]))
    try:
        _W["base"], _W["counter"] = make_base(spec)
    except Exception as e:  # noqa: BLE001 - every task reports it
        _W["init_error"] = f"{type(e).__name__}: {e}"[:300]
        log.error("worker init failed: %s", _W["init_error"])


def make_base(spec: Dict):
    """(policy base, forward counter): over the shared server at
    spec["address"], or a local copy of spec["checkpoint"]."""
    if spec.get("address"):
        from tools.elo_eval_game import remote_base
        return remote_base(spec["address"], False, spec["infer_bf16"], spec["infer_packed_trunk"])
    import torch
    from tools.elo_eval_game import _CountingModel
    from tools.eval_players import _load_policy
    policy = _load_policy(Path(spec["checkpoint"]), torch.device(spec["device"]), label="regrade")
    counter = _CountingModel(policy._inference_model)
    policy._inference_model = counter
    return policy, counter


def position_records(record: Dict, slots: Sequence[int], base, counter, cfg: RolloutConfig, dataset: Path,
                     step1: Dict[Tuple[int, int], Dict]) -> Iterable[Dict]:
    """The records of `slots` of one position, as each finishes."""
    from tools.critic_bench import boundary_of
    index = int(record["index"])
    try:
        boundary = boundary_of(record, dataset)
    except Exception as e:  # noqa: BLE001 - every candidate of the position carries it
        for slot in slots:
            yield {"index": index, "slot": slot, "mover": int(record["side"]),
                   "error": f"position: {type(e).__name__}: {e}"[:300]}
        return
    for slot in slots:
        yield candidate_record(boundary, record, slot, base, counter, cfg, step1.get((index, slot)))


def _position_task(task: Tuple[Dict, List[int]]) -> int:
    record, slots = task
    if "init_error" in _W:
        for slot in slots:
            _W["sink"].put({"index": int(record["index"]), "slot": slot, "mover": int(record["side"]),
                            "error": f"worker: {_W['init_error']}"})
        return len(slots)
    n = 0
    for rec in position_records(record, slots, _W["base"], _W["counter"], _W["cfg"], Path(_W["dataset"]),
                                _W["step1"]):
        _W["sink"].put(rec)
        n += 1
    return n


def run_parallel(tasks: Sequence[Tuple[Dict, List[int]]], spec: Dict, workers: int,
                 write: Callable[[Dict], None], stall_s: float) -> int:
    """Play `tasks` on `workers` processes; `write` each record as it
    arrives. Stops when every record came, when every task returned and
    the rest did not come within QUEUE_GRACE_S, or when none came for
    `stall_s` (a worker that died takes its task with it). Returns the
    records received."""
    expected = sum(len(slots) for _, slots in tasks)
    ctx = mp.get_context("spawn")
    sink = ctx.Queue()
    got = 0
    with ctx.Pool(workers, initializer=_init_worker, initargs=(spec, sink)) as pool:
        pending = pool.map_async(_position_task, tasks, chunksize=1)
        last = time.monotonic()
        ready_at = None
        while got < expected:
            try:
                rec = sink.get(timeout=5.0)
            except queue.Empty:
                now = time.monotonic()
                if pending.ready():
                    ready_at = ready_at or now
                if (ready_at is not None and now - ready_at > QUEUE_GRACE_S) or now - last > stall_s:
                    log.error("%d of %d records never came (tasks returned: %s)", expected - got, expected,
                              pending.ready())
                    break
                continue
            write(rec)
            got += 1
            last = time.monotonic()
        if pending.ready():
            try:
                pending.get(timeout=0)
            except Exception as e:  # noqa: BLE001 - the missing records are counted
                log.error("a task failed outside its own handling: %s: %s", type(e).__name__, e)
    return got


def run_local(tasks: Sequence[Tuple[Dict, List[int]]], spec: Dict, write: Callable[[Dict], None]) -> int:
    cfg = RolloutConfig(**spec["cfg"])
    base, counter = make_base(spec)
    got = 0
    for record, slots in tasks:
        for rec in position_records(record, slots, base, counter, cfg, Path(spec["dataset"]), spec["step1"]):
            write(rec)
            got += 1
    return got


# ---------------------------------------------------------------------
# Records on disk
# ---------------------------------------------------------------------

KEEP = ("pre_end_turn", "terminal_in_turn", "hp_margin_post", "proposer")


def load_records(path: Path) -> List[Dict]:
    """The benchmark's positions with what the rebuild and the record
    read (the playouts' outcomes and reads left out)."""
    from tools.critic_bench import candidates
    out = []
    for p in json.loads(Path(path).read_text(encoding="utf-8"))["positions"]:
        slim = {k: p[k] for k in ("index", "side", "meta") if k in p}
        slim["turn_salt"] = p.get("turn_salt")
        cands = [{k: c[k] for k in KEEP if k in c} for c in candidates(p)]
        slim["base"], slim["alternatives"] = cands[0], cands[1:]
        out.append(slim)
    return out


def load_step1(path: Optional[Path]) -> Dict[Tuple[int, int], Dict]:
    """Step 1's rebuild rows by (position, slot)."""
    if path is None:
        return {}
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            pos = json.loads(line)
            for c in pos["candidates"]:
                row = dict(c)
                if pos.get("error") and "error" not in row:
                    row["error"] = pos["error"]
                out[(int(pos["index"]), int(c["slot"]))] = row
    return out


def read_rollouts(path: Path) -> List[Dict]:
    """The records of a rollouts file in order; a line a kill cut short is
    left out (and cut off by `truncate_partial`)."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            log.warning("%s: a cut record is skipped", path)
    return out


def last_records(records: Iterable[Dict]) -> Dict[Tuple[int, int], Dict]:
    out = {}
    for rec in records:
        out[(int(rec["index"]), int(rec["slot"]))] = rec
    return out


def truncate_partial(path: Path) -> None:
    """Cut a last line that does not end in a newline (a kill mid-write)."""
    if not path.exists():
        return
    data = path.read_bytes()
    end = data.rfind(b"\n") + 1
    if end < len(data):
        with path.open("r+b") as f:
            f.truncate(end)


def summarize(records: Dict[Tuple[int, int], Dict], expected: int, cfg: RolloutConfig) -> Dict:
    """Counts of the run's last records: candidates by state, failures by
    reason, and the rollouts' decisions and times."""
    states, reasons, rollout_reasons, ended = Counter(), Counter(), Counter(), Counter()
    decisions, per_decision, forwards = [], [], []
    for rec in records.values():
        if "skipped" in rec:
            states["skipped"] += 1
            continue
        if "error" in rec:
            states["failed"] += 1
            reasons[rec["error"].split(":")[0]] += 1
            continue
        states["complete" if is_complete(rec, cfg) else "rollout_failed"] += 1
        for ro in rec["rollouts"]:
            if "error" in ro:
                rollout_reasons[ro["error"].split(":")[0]] += 1
                continue
            ended[ro["ended_by"] or "running"] += 1
            decisions.append(ro["decisions"])
            forwards.append(ro["forwards"])
            if ro["decisions"]:
                per_decision.append((ro["seconds"] - ro["forward_seconds"]) / ro["decisions"])
    return {"candidates": {"expected": expected, "recorded": len(records), **states},
            "candidate_failures": dict(reasons), "rollout_failures": dict(rollout_reasons),
            "rollouts": {"played": len(decisions), "ended_by": dict(ended),
                         "decisions": sum(decisions), "forwards": sum(forwards),
                         "decisions_median": statistics.median(decisions) if decisions else None,
                         "worker_seconds_per_decision_median":
                             round(statistics.median(per_decision), 5) if per_decision else None}}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def cmd_run(args) -> int:
    from wesnoth_ai import __version__
    from tools.critic_bench import candidates
    cfg = RolloutConfig(seed=args.seed, rollouts=args.rollouts)
    records = load_records(args.records)
    step1 = load_step1(args.step1)
    args.out.mkdir(parents=True, exist_ok=True)
    sink_path = args.out / "rollouts.jsonl"
    truncate_partial(sink_path)
    done = {k for k, rec in last_records(read_rollouts(sink_path)).items() if is_complete(rec, cfg)}
    tasks = []
    for rec in records:
        slots = [s for s in range(len(candidates(rec))) if (int(rec["index"]), s) not in done]
        if slots:
            tasks.append((rec, slots))
    expected = sum(len(candidates(r)) for r in records)
    todo = sum(len(s) for _, s in tasks)
    log.info("%d positions, %d candidates, %d complete before, %d to play, %d workers",
             len(records), expected, len(done), todo, args.workers)
    spec = {"cfg": asdict(cfg), "dataset": str(args.dataset), "step1": step1, "checkpoint": str(args.checkpoint),
            "device": args.device, "log_level": args.log_level}
    header = {"config": asdict(cfg), "checkpoint": str(args.checkpoint), "records": str(args.records),
              "step1": None if args.step1 is None else str(args.step1), "workers": args.workers,
              "code_version": __version__, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    t0 = time.time()
    with sink_path.open("a", encoding="utf-8") as sink:
        n = [0]

        def write(rec: Dict) -> None:
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            n[0] += 1
            if n[0] % 20 == 0 or n[0] == todo:
                log.info("%d/%d candidates, %.0f s", n[0], todo, time.time() - t0)
        server_stats = None
        if args.workers <= 0:
            run_local(tasks, spec, write)
        elif tasks:
            server_stats = _with_server(args, spec, tasks, write)
    header["wall_s"] = round(time.time() - t0, 1)
    summary = summarize(last_records(read_rollouts(sink_path)), expected, cfg)
    payload = dict(header, **summary, inference_server=server_stats)
    (args.out / "regrade_summary.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    c = summary["candidates"]
    incomplete = c["recorded"] < expected or c.get("failed", 0) or c.get("rollout_failed", 0)
    log.info("REGRADE_%s %s; failures: candidates %s, rollouts %s", "INCOMPLETE" if incomplete else "DONE",
             json.dumps(c), summary["candidate_failures"], summary["rollout_failures"])
    return EXIT_INCOMPLETE if incomplete else 0


def _with_server(args, spec: Dict, tasks, write) -> Optional[Dict]:
    """Run the tasks on workers behind one inference server; its stats."""
    from tools.eval_inference_server import launch_inference_server
    server = launch_inference_server(str(args.checkpoint), args.out, tag="regrade", device=args.device,
                                     infer_bf16=None, window_ms=args.window_ms, max_batch=max(1, args.workers))
    log.info("inference server at %s: %s", server.address, server.info)
    spec = dict(spec, address=server.address, infer_bf16=bool(server.info["infer_bf16"]),
                infer_packed_trunk=bool(server.info["packed_trunk"]))
    try:
        run_parallel(tasks, spec, args.workers, write, args.stall_min * 60.0)
    finally:
        stats = server.shutdown()
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", type=Path, required=True, help="the benchmark's validation.json")
    ap.add_argument("--dataset", type=Path, required=True, help="the corpus the benchmark was played from")
    ap.add_argument("--checkpoint", type=Path, required=True, help="both sides' player and value head")
    ap.add_argument("--step1", type=Path, default=None, help="step 1's rebuild.jsonl, the states' check")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4),
                    help="processes sharing one inference server; 0 plays here on a local copy")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--window-ms", type=float, default=1.5)
    ap.add_argument("--stall-min", type=float, default=15.0, help="stop when no record came for this long")
    ap.add_argument("--rollouts", type=int, default=ROLLOUTS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if not args.checkpoint.is_file():
        raise SystemExit(f"--checkpoint {args.checkpoint} does not exist")
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
