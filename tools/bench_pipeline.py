#!/usr/bin/env python3
"""Pipeline benchmark (docs/plan_20260904.md step 1.1): the numbers the
1000x program is claimed from, measured on fixed inputs.

Three sections, one JSON, one markdown table:

  A. Per-state component costs (CPU, serial, median ms over the
     benchmark states): deepcopy, sim fork, encode_raw,
     encode_from_raw, legality masks, legal-action enumeration with
     priors, one sim step, state_key. These are the targets of the
     Rust port; their sum is the Python overhead per decision.
  B. Network forward cost by token-count bucket and batch size: ms
     per sample and samples per second, with the precision and
     compile flags the eval harness applies.
  C. End-to-end games through the real eval worker
     (tools/run_elo_batch.py): raw:t0 vs raw:t0 and mcts:32 vs raw:t0
     at a given concurrency; seconds per game, forwards per game,
     games per hour and games per dollar.

Benchmark states come from the imitation manifest's HOLDOUT games on
ladder maps (never trained on), reconstructed at a side-turn boundary
as the side to move sees them. The list is pinned in
configs/bench_states.json so every box measures the same positions;
--build-states writes it (needs replays_dataset_imitation/).

Usage (box):
  python tools/bench_pipeline.py --checkpoint training/checkpoints/seed.pt \\
      --device cuda --games 20 --jobs 10 --dollars-per-hour 0.334 \\
      --out bench.json
  python tools/bench_pipeline.py --build-states 200 --seed 1
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import logging
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

log = logging.getLogger("bench_pipeline")

DEFAULT_MANIFEST = ROOT / "configs" / "bench_states.json"
DEFAULT_DATASET = ROOT / "replays_dataset_imitation"
COMPONENTS = ("deepcopy", "fork", "encode_raw", "encode_from_raw",
              "legality_masks", "enumerate_priors", "sim_step", "state_key")


# ---------------------------------------------------------------------
# Benchmark states: holdout ladder games at side-turn boundaries
# ---------------------------------------------------------------------

def select_holdout_ladder(rows: Sequence[dict], scenario_of: Callable[[dict], str],
                          ladder_ids: Sequence[str]) -> List[dict]:
    """Manifest rows flagged holdout whose scenario is a ladder map."""
    ladder = set(ladder_ids)
    return [r for r in rows if r.get("holdout") and scenario_of(r) in ladder]


def reconstruct_boundary(data: dict, cut_turn: int):
    """Walk a replay's commands to the first init_side of a player
    side with turn_number >= cut_turn and return (state, begin_side)
    with the side's turn begun (income, healing applied), i.e. the
    position the side to move faces. None when the game ends first
    or a leader is dead."""
    from tools.replay_dataset import (_apply_command, _build_initial_gamestate,
                                      _setup_scenario_events)
    from tools.wesnoth_sim import WesnothSim
    gs = _build_initial_gamestate(data)
    scenario_id = data.get("scenario_id", "")
    _setup_scenario_events(gs, scenario_id)
    for cmd in data.get("commands", []):
        if (cmd and cmd[0] == "init_side"
                and gs.global_info.turn_number >= cut_turn
                and gs.global_info.current_side in (1, 2)
                and len(cmd) > 1 and cmd[1] in (1, 2)):
            if not {1, 2} <= {u.side for u in gs.map.units if u.is_leader}:
                return None
            for attr in ("_last_advance_events", "_last_checkup_strikes"):
                if hasattr(gs.global_info, attr):
                    setattr(gs.global_info, attr, [] if attr.endswith("events") else None)
            begin_side = int(cmd[1])
            sim = WesnothSim(gs, scenario_id, max_turns=200,
                             apply_scenario_events=False, begin_side=begin_side)
            return sim.gs, begin_side
        _apply_command(gs, cmd)
    return None


def build_state_manifest(dataset_dir: Path, n: int, seed: int) -> dict:
    """Pick n holdout ladder positions: uniform game, uniform cut turn
    in [2, n_turns - 1]; every entry is verified to reconstruct."""
    from tools.scenario_pool import LADDER_SCENARIO_IDS
    rows = [json.loads(ln) for ln in
            (dataset_dir / "manifest.jsonl").open(encoding="utf-8") if ln.strip()]
    hold = [r for r in rows if r.get("holdout") and r.get("n_turns", 0) >= 4]
    rng = random.Random(seed)
    rng.shuffle(hold)
    ladder = set(LADDER_SCENARIO_IDS)
    entries: List[dict] = []
    for r in hold:
        if len(entries) >= n:
            break
        with gzip.open(dataset_dir / r["file"], "rt", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("scenario_id") not in ladder:
            continue
        cut = rng.randint(2, r["n_turns"] - 1)
        res = reconstruct_boundary(data, cut)
        if res is None:
            continue
        gs, begin_side = res
        entries.append({"file": r["file"], "scenario_id": data["scenario_id"],
                        "cut_turn": cut, "begin_side": begin_side,
                        "turn_number": gs.global_info.turn_number,
                        "n_units": len(gs.map.units), "n_hexes": len(gs.map.hexes)})
        log.info("state %d: %s turn %d side %d units %d hexes %d",
                 len(entries), r["file"], cut, begin_side,
                 len(gs.map.units), len(gs.map.hexes))
    return {"dataset": dataset_dir.name, "seed": seed, "states": entries}


def pack_states(manifest_path: Path, dataset_dir: Path, out_dir: Path) -> int:
    """Copy just the game files the manifest needs (plus a manifest.jsonl
    subset) into out_dir, so a box can rebuild the benchmark states
    without the full dataset. Returns the number of files copied."""
    import shutil
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    files = sorted({e["file"] for e in man["states"]})
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(ln) for ln in
            (dataset_dir / "manifest.jsonl").open(encoding="utf-8") if ln.strip()]
    keep = [r for r in rows if r["file"] in set(files)]
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")
    for name in files:
        shutil.copy2(dataset_dir / name, out_dir / name)
    return len(files)


def load_states(manifest_path: Path, dataset_dir: Path,
                limit: Optional[int] = None) -> List[Tuple[object, str]]:
    """(GameState, scenario_id) for every manifest entry, rebuilt from
    the dataset (bit-exact reconstruction, so every box sees the same
    positions)."""
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    out = []
    for e in man["states"][:limit]:
        with gzip.open(dataset_dir / e["file"], "rt", encoding="utf-8") as f:
            data = json.load(f)
        res = reconstruct_boundary(data, e["cut_turn"])
        if res is None or res[1] != e["begin_side"]:
            raise RuntimeError(f"benchmark state failed to reconstruct: {e}")
        out.append((res[0], e["scenario_id"]))
    return out


# ---------------------------------------------------------------------
# Section A: component costs
# ---------------------------------------------------------------------

def _first_move_action(legal) -> dict:
    for la in legal:
        if la.action.get("type") == "move":
            return la.action
    return {"type": "end_turn"}


def component_costs(states: Sequence[Tuple[object, str]], policy,
                    repeats: int = 3) -> Dict[str, float]:
    """Median milliseconds per call of each per-decision component
    over the states (CPU-side Python work plus one forward for the
    enumeration's inputs)."""
    import torch
    from wesnoth_ai.action_sampler import (_build_legality_masks,
                                           enumerate_legal_actions_with_priors)
    from wesnoth_ai.classes import state_key
    from wesnoth_ai.encoder import encode_raw
    from tools.wesnoth_sim import WesnothSim
    enc = policy._inference_encoder
    model = policy._inference_model
    times: Dict[str, List[float]] = {k: [] for k in COMPONENTS}

    def clock(name, fn):
        best = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            r = fn()
            dt = (time.perf_counter() - t0) * 1000.0
            best = dt if best is None else min(best, dt)
        times[name].append(best)
        return r

    for gs, scenario_id in states:
        enc.register_names(gs)
        sim = WesnothSim(copy.deepcopy(gs), scenario_id, max_turns=200,
                         apply_scenario_events=False, begin_turn=False)
        clock("deepcopy", lambda: copy.deepcopy(gs))
        clock("fork", sim.fork)
        raw = clock("encode_raw", lambda: encode_raw(
            gs, type_to_id=enc.unit_type_to_id, faction_to_id=enc.faction_to_id,
            relevant_set=bool(getattr(enc, "relevant_set_hexes", False))))
        with torch.no_grad():
            encoded = clock("encode_from_raw", lambda: enc.encode_from_raw(raw))
            output = model(encoded)
            clock("legality_masks", lambda: _build_legality_masks(encoded, gs))
            legal = clock("enumerate_priors", lambda: enumerate_legal_actions_with_priors(
                encoded, output, gs))
        action = _first_move_action(legal)

        def _fork_step():
            f = sim.fork()
            f.step(action)
        fork_ms = times["fork"][-1]
        clock("sim_step", _fork_step)
        times["sim_step"][-1] = max(0.0, times["sim_step"][-1] - fork_ms)
        clock("state_key", lambda: state_key(gs))
    return {k: statistics.median(v) for k, v in times.items() if v}


# ---------------------------------------------------------------------
# Section B: forward cost by bucket and batch size
# ---------------------------------------------------------------------

def n_tokens(encoded) -> int:
    return (encoded.hex_tokens.size(1) + encoded.unit_tokens.size(1)
            + encoded.recruit_tokens.size(1) + 2)


def bucket_edges(counts: Sequence[int], k: int = 4) -> List[int]:
    """Upper edges of k quantile buckets over token counts."""
    s = sorted(counts)
    return [s[min(len(s) - 1, (len(s) * (i + 1)) // k - 1)] for i in range(k)]


def forward_costs(policy, states: Sequence[Tuple[object, str]],
                  batch_sizes: Sequence[int] = (1, 4, 16, 64),
                  samples_per_config: int = 64, k_buckets: int = 4) -> List[dict]:
    """Per (token bucket, batch size): median ms per forward call,
    ms per sample and samples per second. Batch 1 uses the production
    single-sample path; larger batches use forward_batch."""
    import torch
    enc = policy._inference_encoder
    model = policy._inference_model
    device = next(policy._model.parameters()).device
    sync = (torch.cuda.synchronize if device.type == "cuda" else (lambda: None))
    with torch.no_grad():
        encoded = [enc.encode(gs) for gs, _ in states]
    toks = [n_tokens(e) for e in encoded]
    edges = bucket_edges(toks, k_buckets)
    rows = []
    lo = 0
    for edge in edges:
        members = [e for e, t in zip(encoded, toks) if lo < t <= edge]
        lo = edge
        if not members:
            continue
        for B in batch_sizes:
            calls = max(2, -(-samples_per_config // B))
            batches = [[members[(c * B + j) % len(members)] for j in range(B)]
                       for c in range(calls)]
            with torch.no_grad():
                for b in batches[:2]:                     # warm-up / compile
                    (model(b[0]) if B == 1 else model.forward_batch(b))
                    sync()
                ts = []
                for b in batches:
                    t0 = time.perf_counter()
                    (model(b[0]) if B == 1 else model.forward_batch(b))
                    sync()
                    ts.append((time.perf_counter() - t0) * 1000.0)
            med = statistics.median(ts)
            rows.append({"tokens_max": edge, "tokens_mean": round(
                statistics.fmean(n_tokens(e) for e in members)),
                "n_states": len(members), "batch": B, "ms_per_call": med,
                "ms_per_sample": med / B, "samples_per_s": 1000.0 * B / med})
    return rows


# ---------------------------------------------------------------------
# Section C: end-to-end games through the eval worker
# ---------------------------------------------------------------------

def _summarize_games(outdir: Path, wall_s: float, dollars_per_hour: float) -> dict:
    secs, fwd_a, fwd_b, turns = [], [], [], []
    outcomes: Dict[str, int] = {}
    for f in sorted(outdir.glob("game_*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("secs") is None:
            continue
        secs.append(r["secs"])
        fwd_a.append(r.get("forwards_a") or 0)
        fwd_b.append(r.get("forwards_b") or 0)
        turns.append(r.get("turns") or 0)
        o = str(r.get("outcome_a"))
        outcomes[o] = outcomes.get(o, 0) + 1
    n = len(secs)
    if not n:
        return {"games": 0}
    per_hour = n / (wall_s / 3600.0)
    decisive = outcomes.get("win", 0) + outcomes.get("loss", 0)
    return {"games": n, "wall_s": wall_s, "secs_per_game_median": statistics.median(secs),
            "secs_per_game_max": max(secs), "turns_median": statistics.median(turns),
            "outcomes_a": outcomes, "decisive_frac": decisive / n,
            "forwards_a_mean": statistics.fmean(fwd_a),
            "forwards_b_mean": statistics.fmean(fwd_b),
            "games_per_hour": per_hour,
            "games_per_dollar": per_hour / dollars_per_hour if dollars_per_hour else None}


def end_to_end(checkpoint: Path, outdir: Path, games: int, jobs: int, device: str,
               dollars_per_hour: float, sims: int = 32) -> Dict[str, dict]:
    common = ["--label-b", "seed_t0", "--spec-b", str(checkpoint),
              "--raw-temperature-b", "0", "--games", str(games),
              "--device", device, "--jobs", str(jobs),
              "--time-budget-min", "120", "--max-extra-games", "0"]
    runs = {
        "raw_vs_raw": ["--label-a", "bench_t0", "--spec-a", str(checkpoint),
                       "--mcts-sims", "0", "--raw-temperature-a", "0"],
        f"mcts{sims}_vs_raw": ["--label-a", f"bench_mcts{sims}", "--spec-a",
                               str(checkpoint), "--mcts-sims-a", str(sims),
                               "--mcts-sims-b", "0", "--no-turn-search"],
    }
    out = {}
    for name, extra in runs.items():
        d = outdir / name
        cmd = [sys.executable, str(ROOT / "tools" / "run_elo_batch.py"),
               "--outdir", str(d)] + extra + common
        t0 = time.perf_counter()
        subprocess.run(cmd, cwd=str(ROOT), check=False)
        out[name] = _summarize_games(d, time.perf_counter() - t0, dollars_per_hour)
        log.info("%s: %s", name, out[name])
    return out


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def markdown_report(result: dict) -> str:
    lines = [f"### bench_pipeline {result.get('label', '')}", "",
             "| component | ms (median) |", "|---|---|"]
    for k, v in result.get("components", {}).items():
        lines.append(f"| {k} | {v:.3f} |")
    lines += ["", "| tokens (max) | batch | ms/sample | samples/s |", "|---|---|---|---|"]
    for r in result.get("forwards", []):
        lines.append(f"| {r['tokens_max']} | {r['batch']} | {r['ms_per_sample']:.2f} "
                     f"| {r['samples_per_s']:.0f} |")
    g = result.get("games", {})
    if g:
        lines += ["", "| match | games | s/game (median) | forwards A | games/h | games/$ |",
                  "|---|---|---|---|---|---|"]
        for name, s in g.items():
            if s.get("games"):
                gpd = s.get("games_per_dollar")
                lines.append(f"| {name} | {s['games']} | {s['secs_per_game_median']:.0f} | "
                             f"{s['forwards_a_mean']:.0f} | {s['games_per_hour']:.0f} | "
                             f"{gpd:.0f} |" if gpd else
                             f"| {name} | {s['games']} | {s['secs_per_game_median']:.0f} | "
                             f"{s['forwards_a_mean']:.0f} | {s['games_per_hour']:.0f} | - |")
    return "\n".join(lines)


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--states", type=int, default=None,
                    help="Use only the first N manifest states.")
    ap.add_argument("--build-states", type=int, default=0,
                    help="Write --manifest with N holdout ladder positions and exit.")
    ap.add_argument("--pack-states", type=Path, default=None,
                    help="Copy the manifest's game files into this directory "
                         "(a self-contained mini dataset for boxes) and exit.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch-sizes", default="1,4,16,64")
    ap.add_argument("--samples-per-config", type=int, default=64)
    ap.add_argument("--games", type=int, default=0,
                    help="End-to-end games per match (0 = skip section C).")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--dollars-per-hour", type=float, default=0.0)
    ap.add_argument("--infer-bf16", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--infer-compile", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--games-outdir", type=Path, default=ROOT / "eval_games" / "bench_pipeline")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.build_states:
        man = build_state_manifest(args.dataset, args.build_states, args.seed)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(man, indent=1), encoding="utf-8")
        print(f"wrote {len(man['states'])} states to {args.manifest}")
        return 0

    if args.pack_states is not None:
        n = pack_states(args.manifest, args.dataset, args.pack_states)
        print(f"packed {n} game files into {args.pack_states}")
        return 0

    import torch
    from tools.eval_sim import _load_policy
    cuda = args.device == "cuda"
    if cuda and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no CUDA device is visible")
    bf16 = cuda if args.infer_bf16 is None else args.infer_bf16
    comp = cuda if args.infer_compile is None else args.infer_compile
    policy = _load_policy(args.checkpoint, torch.device("cuda") if cuda else torch.device("cpu"),
                          label="bench", infer_bf16=bf16, infer_compile=comp)
    states = load_states(args.manifest, args.dataset, args.states)
    log.info("loaded %d benchmark states", len(states))
    result = {"label": args.label, "checkpoint": str(args.checkpoint), "device": args.device,
              "infer_bf16": bf16, "infer_compile": comp, "n_states": len(states),
              "torch": torch.__version__}
    result["components"] = component_costs(states, policy, repeats=args.repeats)
    result["forwards"] = forward_costs(
        policy, states, batch_sizes=[int(b) for b in args.batch_sizes.split(",")],
        samples_per_config=args.samples_per_config)
    if args.games > 0:
        result["games"] = end_to_end(args.checkpoint, args.games_outdir, args.games,
                                     args.jobs, args.device, args.dollars_per_hour,
                                     sims=args.sims)
    report = markdown_report(result)
    print(report)
    if args.out:
        args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        args.out.with_suffix(".md").write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
