#!/usr/bin/env python3
"""Turn-level value gap (docs/plan_20260904.md section 5, the phase-2
prerequisite; pre-registration in docs/turn_gap_prereg_20260904.md).

At N turn-boundary positions the side to move plays its BASE turn:
the whole sequence of atomic actions `raw:t0` (tools/raw_player.py at
temperature 0, the reference player) plays until it ends the turn.
K ALTERNATIVE whole turns come from the same weights at temperature T
with distinct sampling seeds. An alternative whose post-turn position
equals the base's or an earlier alternative's (wesnoth_ai.classes
.state_key) is dropped. Every remaining post-turn position is played
out P times to the end of the game with `raw:t0` on both sides;
playouts of one position differ by the simulator's combat seed only.
Outcome per playout, from the mover's side: +1 win, -1 loss, 0 draw
or undecided at the turn cap. The gap of a position is the best
alternative's mean outcome minus the base turn's.

Seeds. The candidate turns of one position share one combat salt
(common random numbers across candidates); playout r of candidate c
of position i uses the salt "turn_gap:<seed>:p<i>:c<c>:r<r>" (c = 0
is the base, c = k + 1 the k-th sampled alternative). Alternative k
samples with numpy seed sha256("turn_gap:<seed>:p<i>:alt<k>"). Every
salt and seed is recorded, so a run reproduces exactly.

Reading the report. The headline fraction (positions with gap >=
threshold) is biased upward by selection: the best of K noisy means
exceeds an independent noisy mean even when all candidates are equal.
`summarize` therefore also reports the permutation null of that
fraction (outcomes reshuffled across the candidates of a position)
and the split-half gap (alternative chosen on the even-numbered
playouts, gap measured on the odd-numbered ones), whose mean is an
unbiased estimate of the chosen alternative's true advantage.

Usage (box, docs/box_specs.md):
  python tools/turn_gap.py --checkpoint training/checkpoints/seed.pt \\
      --device cuda --jobs 10 --dollars-per-hour 0.33 --out turn_gap.json
  python tools/turn_gap.py --summarize turn_gap.partial.json   # mid-run reading
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import statistics
import sys
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.bench_pipeline import DEFAULT_DATASET, DEFAULT_MANIFEST, load_states
from tools.eval_sim import _PolicyPair, _play_one_eval_game
from tools.mcts import fork_guard
from tools.raw_player import RawPolicyPlayer
from tools.sim_self_play import _would_recruit_bounce
from tools.wesnoth_sim import WesnothSim
from wesnoth_ai.classes import GameState, state_key

log = logging.getLogger("turn_gap")

REFERENCE_PROCEDURE = "raw:t0"
# Gap = best alternative mean - base mean, both in [-1, 1].
GAP_HISTOGRAM_EDGES = [round(-2.0 + 0.25 * i, 2) for i in range(17)]


# ---------------------------------------------------------------------
# Configuration and inputs
# ---------------------------------------------------------------------

@dataclass
class GapConfig:
    k_alternatives: int = 4
    continue_edits: int = 0            # continue-edit alternatives: the base turn
                                       # minus its end_turn plus 1..k more argmax
                                       # actions (docs/turn_proposer_design_20260905.md:
                                       # two of the three confirmed gaps were early
                                       # end_turns)
    playouts: int = 40
    temperature: float = 1.0      # of the alternative turns
    cap_turns: int = 40           # playouts end undecided at the start of turn T0 + cap + 1
    seed: int = 1
    playout_offset: int = 0            # first playout index: fresh salts for a
                                       # confirmation run on selected positions
    playout_temperature: float = 0.0   # of both sides during the playouts (0 = raw:t0);
                                       # 2026-09-05: raw:t0 self-play stalls to the
                                       # 200-turn cap in 17 of 40 games, raw:t0.5
                                       # scores 22-18 against it with no stalls

    def __post_init__(self):
        if self.k_alternatives < 0 or self.playouts < 1 or self.cap_turns < 1:
            raise ValueError("k_alternatives >= 0, playouts >= 1, cap_turns >= 1")
        if self.temperature < 0.0 or self.playout_temperature < 0.0:
            raise ValueError("temperatures must be >= 0")
        if self.playout_offset < 0:
            raise ValueError("playout_offset must be >= 0")
        if self.continue_edits < 0:
            raise ValueError("continue_edits must be >= 0")


@dataclass
class PolicySpec:
    """What a worker needs to load the reference policy itself, or to
    reach the shared inference server that holds it."""
    checkpoint: Optional[str]     # path, or "random" for a random init
    device: str                   # "cpu" | "cuda"
    infer_bf16: bool
    infer_compile: bool
    # Shared inference (tools/eval_inference_server.py): the workers
    # keep the sim and the raw player, one server owns the model.
    inference_address: Optional[str] = None
    infer_packed_trunk: bool = False


@dataclass
class BoundaryPosition:
    """A side-turn boundary as the mover sees it: `gs` already carries
    the mover's init_side (income, healing, refreshed moves), as
    tools/bench_pipeline.reconstruct_boundary returns it."""
    index: int
    gs: GameState
    scenario_id: str
    meta: Dict = field(default_factory=dict)


def positions_from_manifest(manifest: Path, dataset: Path,
                            n: int) -> List[BoundaryPosition]:
    entries = json.loads(Path(manifest).read_text(encoding="utf-8"))["states"]
    if n > len(entries):
        log.warning("manifest holds %d states, %d requested", len(entries), n)
        n = len(entries)
    states = load_states(manifest, dataset, n)
    return [BoundaryPosition(index=i, gs=gs, scenario_id=sid, meta=dict(entry))
            for i, ((gs, sid), entry) in enumerate(zip(states, entries[:n]))]


def load_reference_policy(spec: PolicySpec):
    if spec.inference_address is not None:
        return remote_policy(spec.inference_address)
    import torch
    from tools.eval_sim import _load_policy
    ckpt = None if spec.checkpoint in (None, "random") else Path(spec.checkpoint)
    return _load_policy(ckpt, torch.device(spec.device), label="turn_gap",
                        infer_bf16=spec.infer_bf16, infer_compile=spec.infer_compile)


def remote_policy(address: str):
    """A policy base over the shared inference server: a RemoteEncoder
    on the server's vocab with server-side priors and a RemoteModel,
    the surface RawPolicyPlayer and _value_read consume (the same
    construction as tools/elo_eval_game's shared-inference player)."""
    import threading
    from types import SimpleNamespace
    import torch
    from tools.eval_inference_server import EvalInferenceClient
    from tools.inference_seam import RemoteEncoder, RemoteModel
    client = EvalInferenceClient(address)
    h = client.hello
    encoder = RemoteEncoder(h["type_to_id"], h["faction_to_id"],
                            device=torch.device("cpu"),
                            relevant_set=bool(h["relevant_set"]), server_priors=True)
    return SimpleNamespace(_inference_model=RemoteModel(client),
                           _inference_encoder=encoder,
                           _lock=threading.Lock(), _decision_step=0)


# ---------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------

def turn_salt(seed: int, index: int) -> str:
    return f"turn_gap:{seed}:p{index}:turn"


def playout_salt(seed: int, index: int, candidate: int, playout: int) -> str:
    return f"turn_gap:{seed}:p{index}:c{candidate}:r{playout}"


def alternative_seed(seed: int, index: int, k: int) -> int:
    digest = hashlib.sha256(f"turn_gap:{seed}:p{index}:alt{k}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & (2 ** 63 - 1)


# ---------------------------------------------------------------------
# Playing: one side-turn, one playout
# ---------------------------------------------------------------------

def sim_from_state(gs: GameState, scenario_id: str, max_turns: int,
                   salt: str) -> WesnothSim:
    """A simulator over a deep copy of `gs`. The side to move has its
    turn begun already (its init_side is in the state), so the
    constructor must not fire it again; scenario events were wired
    when the state was built. Combat and trait rolls come from `salt`
    (WesnothSim._next_seed), the mechanism search forks use."""
    sim = WesnothSim(copy.deepcopy(gs), scenario_id, max_turns=max_turns,
                     apply_scenario_events=False, begin_turn=False)
    sim._seed_salt = salt
    return sim


def _select(player, sim: WesnothSim, game_label: str) -> Dict:
    # The player sees a stable snapshot; the guard checks it did not
    # touch the live state (as in tools/eval_sim._play_one_eval_game).
    pre_state = copy.deepcopy(sim.gs)
    with fork_guard(sim):
        return player.select_action(pre_state, game_label=game_label, sim=sim)


def _decide(player, sim: WesnothSim, game_label: str) -> Dict:
    """One decision, the way the eval loop makes it: a recruit onto a
    hex the sim knows is occupied (fog) bounces, the hex joins the
    per-turn rejection set, and the player decides again."""
    action = _select(player, sim, game_label)
    while _would_recruit_bounce(action, sim.gs):
        tgt = action["target_hex"]
        rejected = (getattr(sim.gs.global_info, "_recruit_rejected_hexes", None)
                    or set())
        rejected.add((tgt.x, tgt.y))
        setattr(sim.gs.global_info, "_recruit_rejected_hexes", rejected)
        player.drop_last_pending(game_label)
        action = _select(player, sim, game_label)
    return action


def _action_to_json(action: Dict) -> Dict:
    """An action dict with positions as (x, y) lists: recorded so a
    confirmation run can REPLAY a candidate turn instead of resampling
    it (2026-09-05 audit: sampling from a seed does not reproduce
    across runs under bf16 kernels)."""
    out = {}
    for k, v in action.items():
        if hasattr(v, "x") and hasattr(v, "y"):
            out[k] = [int(v.x), int(v.y)]
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def play_side_turn(sim: WesnothSim, player, game_label: str,
                   actions_out: Optional[List[Dict]] = None) -> int:
    """Play the side to move until it has ended its turn or the game
    is over. Returns the number of decisions before the end_turn;
    appends every action (end_turn included) to `actions_out`."""
    side = sim.current_side
    decisions = 0
    while not sim.done and sim.current_side == side:
        action = _decide(player, sim, game_label)
        if action.get("type", "end_turn") != "end_turn":
            decisions += 1
        if actions_out is not None:
            actions_out.append(_action_to_json(action))
        sim.step(action)
    return decisions


def _action_from_json(action: Dict) -> Dict:
    """Inverse of _action_to_json for the keys the simulator reads."""
    from wesnoth_ai.classes import Position
    out = {}
    for k, v in action.items():
        if isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) for x in v) \
                and (k.endswith("_hex") or k.endswith("_pos") or k == "position"):
            out[k] = Position(v[0], v[1])
        else:
            out[k] = v
    return out


def _continue_candidate(position: BoundaryPosition, base_actions: List[Dict],
                        policy, extra: int, max_turns: int, salt: str,
                        game_label: str) -> Tuple[Dict, WesnothSim]:
    """The base turn without its end_turn, then `extra` more argmax
    decisions that may not be end_turn (while any other action is
    legal), then end_turn. Same combat salt as the base, so the
    replayed prefix realizes identically."""
    sim = sim_from_state(position.gs, position.scenario_id, max_turns, salt)
    side = sim.current_side
    actions: List[Dict] = []
    for a in base_actions:
        if a.get("type") == "end_turn" or sim.done or sim.current_side != side:
            break
        act = _action_from_json(a)
        actions.append(_action_to_json(act))
        sim.step(act)
    player = RawPolicyPlayer(policy, 0.0, forbid_end_turn=True)
    added = 0
    while added < extra and not sim.done and sim.current_side == side:
        act = _decide(player, sim, game_label)
        if act.get("type", "end_turn") == "end_turn":
            break
        actions.append(_action_to_json(act))
        sim.step(act)
        added += 1
    if not sim.done and sim.current_side == side:
        end = {"type": "end_turn"}
        actions.append(end)
        sim.step(end)
    mover = position.gs.global_info.current_side
    candidate = {
        "sample_seed": None, "proposer": "continue", "extra_decisions": added,
        "n_decisions": sum(1 for a in actions if a.get("type") != "end_turn"),
        "actions": actions,
        "value_post": (None if sim.done else _value_read(player, sim.gs, mover)),
        "hp_margin_post": _hp_margin(sim.gs, mover),
        "post_state_key": state_key(sim.gs),
        "terminal_in_turn": bool(sim.done),
    }
    return candidate, sim


def _hp_margin(gs: GameState, mover: int) -> int:
    """Mover's total unit HP minus the opponent's: the exact-material
    pre-grader of docs/turn_proposer_design_20260905.md."""
    ours = sum(int(u.current_hp) for u in gs.map.units if u.side == mover)
    theirs = sum(int(u.current_hp) for u in gs.map.units if u.side != mover)
    return ours - theirs


def _value_read(policy, gs: GameState, mover: int) -> Optional[float]:
    """The policy's value head on `gs` from the mover's side (the head
    scores the side to move; after the mover's end_turn that is the
    opponent, hence the sign). None when the player has no base policy
    (tests with scripted players)."""
    base = getattr(policy, "_base", policy)
    enc = getattr(base, "_inference_encoder", None)
    model = getattr(base, "_inference_model", None)
    if enc is None or model is None:
        return None
    import torch
    with torch.no_grad():
        out = model(enc.encode(gs))
    v = float(out.value.squeeze().item())
    return v if gs.global_info.current_side == mover else -v


def outcome_for(sim: WesnothSim, mover: int) -> Tuple[int, bool]:
    """(+1 / 0 / -1 from the mover's side, capped). Capped = undecided
    at the turn or action limit; a draw (mutual elimination) is 0 and
    not capped, as tools/eval_sim maps it."""
    if sim.winner == mover:
        return 1, False
    if sim.winner == 0:
        return 0, sim.ended_by in ("max_turns", "max_actions")
    return -1, False


def reference_pairs(policy, temperature: float = 0.0,
                    seed: Optional[int] = None) -> Dict[int, _PolicyPair]:
    """Both sides' players for a playout: the reference `raw:t0`, or
    the same weights at `temperature` with a per-playout sampling
    seed (side 2 gets seed + 1)."""
    label = REFERENCE_PROCEDURE if temperature == 0.0 else f"raw:t{temperature:g}"
    return {side: _PolicyPair(policy=RawPolicyPlayer(
                                  policy, temperature,
                                  seed=None if seed is None else seed + side - 1),
                              label=label, side=side)
            for side in (1, 2)}


def playout_pairs(policy, cfg: "GapConfig", salt: str) -> Dict[int, _PolicyPair]:
    """The players of one playout: shared argmax players at playout
    temperature 0, else fresh sampling players seeded from the
    playout's salt (reproducible, independent across playouts)."""
    if cfg.playout_temperature == 0.0:
        return reference_pairs(policy)
    return reference_pairs(policy, cfg.playout_temperature,
                           seed=zlib.crc32(salt.encode("utf-8")))


def play_out(post_gs: GameState, scenario_id: str, mover: int, max_turns: int,
             salt: str, pairs: Dict[int, _PolicyPair],
             game_label: str) -> Tuple[int, bool, int]:
    """One playout from a post-turn position with `raw:t0` on both
    sides. Returns (outcome, capped, final turn number)."""
    sim = sim_from_state(post_gs, scenario_id, max_turns, salt)
    _play_one_eval_game(sim, pairs[mover], pairs[3 - mover], game_label=game_label)
    outcome, capped = outcome_for(sim, mover)
    return outcome, capped, sim.gs.global_info.turn_number


# ---------------------------------------------------------------------
# One position
# ---------------------------------------------------------------------

def _candidate_turn(position: BoundaryPosition, player, max_turns: int,
                    salt: str, sample_seed: Optional[int],
                    game_label: str) -> Tuple[Dict, WesnothSim]:
    sim = sim_from_state(position.gs, position.scenario_id, max_turns, salt)
    actions: List[Dict] = []
    decisions = play_side_turn(sim, player, game_label, actions)
    mover = position.gs.global_info.current_side
    candidate = {
        "sample_seed": sample_seed,
        "n_decisions": decisions,
        "actions": actions,
        # Forward-only pre-graders (docs/turn_proposer_design_20260905.md):
        # the value head on the post-turn state and the HP margin, both
        # from the mover's side, to be compared with the playout mean.
        "value_post": (None if sim.done else _value_read(player, sim.gs, mover)),
        "hp_margin_post": _hp_margin(sim.gs, mover),
        # Process-local (Python hash of a tuple with strings): used to
        # drop duplicate turns within a run, not comparable across runs.
        "post_state_key": state_key(sim.gs),
        "terminal_in_turn": bool(sim.done),
    }
    return candidate, sim


def _run_playouts(candidate: Dict, sim: WesnothSim, position: BoundaryPosition,
                  mover: int, max_turns: int, cfg: GapConfig, c: int,
                  policy, game_label: str) -> None:
    outcomes: List[int] = []
    capped: List[bool] = []
    turns: List[int] = []
    seeds: List[str] = []
    for r in range(cfg.playout_offset, cfg.playout_offset + cfg.playouts):
        salt = playout_salt(cfg.seed, position.index, c, r)
        if sim.done:
            # The candidate turn ended the game: the one terminal
            # result stands for every playout (the gap arithmetic
            # reads P entries per candidate) and none is played.
            o, cp = outcome_for(sim, mover)
            t = sim.gs.global_info.turn_number
        else:
            o, cp, t = play_out(sim.gs, position.scenario_id, mover, max_turns,
                                salt, playout_pairs(policy, cfg, salt),
                                f"{game_label}c{c}r{r}")
        outcomes.append(o)
        capped.append(cp)
        turns.append(t)
        seeds.append(salt)
    candidate.update(outcomes=outcomes, capped=capped, turns=turns, seeds=seeds,
                     playouts_run=0 if sim.done else len(outcomes))


def playouts_run(candidate: Dict) -> int:
    """Playouts actually played for a candidate: none when its turn
    ended the game (the terminal result is repeated P times for the
    gap arithmetic). Records from before the field carry the flag."""
    if "playouts_run" in candidate:
        return int(candidate["playouts_run"])
    return 0 if candidate.get("terminal_in_turn") else len(candidate["outcomes"])


def finish_record(index: int, meta: Dict, base: Dict, alternatives: List[Dict],
                  dropped: List[Dict], cfg: GapConfig, secs: float) -> Dict:
    """Per-position record from candidate dicts carrying `outcomes`
    and `capped` lists (the gap arithmetic lives here, for the
    measurement and for synthetic records alike)."""
    for cand in [base] + alternatives:
        cand["mean"] = float(statistics.fmean(cand["outcomes"]))
        cand["n_capped"] = int(sum(bool(x) for x in cand["capped"]))
    best = max(alternatives, key=lambda a: a["mean"]) if alternatives else None
    gap = (best["mean"] - base["mean"]) if best is not None else 0.0
    record = {"index": index, "gap": float(gap),
              "best_alternative_mean": (None if best is None else best["mean"]),
              "base_is_best": bool(gap <= 0.0),
              "n_alternatives_sampled": cfg.k_alternatives,
              "n_alternatives": len(alternatives),
              "base": base, "alternatives": alternatives, "dropped": dropped,
              "secs": round(float(secs), 2)}
    record.update(meta)
    return record


def measure_position(policy, position: BoundaryPosition, cfg: GapConfig) -> Dict:
    t0 = time.perf_counter()
    gi = position.gs.global_info
    mover, turn0 = gi.current_side, gi.turn_number
    max_turns = turn0 + cfg.cap_turns
    label = f"tg{position.index}"
    salt = turn_salt(cfg.seed, position.index)
    pairs = reference_pairs(policy)

    base, base_sim = _candidate_turn(position, pairs[mover].policy, max_turns,
                                     salt, None, label + "base")
    seen: Dict[int, str] = {base["post_state_key"]: "base"}
    alternatives: List[Tuple[int, Dict, WesnothSim]] = []
    dropped: List[Dict] = []
    for k in range(cfg.k_alternatives):
        sample_seed = alternative_seed(cfg.seed, position.index, k)
        player = RawPolicyPlayer(policy, cfg.temperature, seed=sample_seed)
        alt, alt_sim = _candidate_turn(position, player, max_turns, salt,
                                       sample_seed, f"{label}alt{k}")
        same = seen.get(alt["post_state_key"])
        if same is not None:
            alt["identical_to"] = same
            dropped.append(alt)
            continue
        seen[alt["post_state_key"]] = f"alt{k}"
        alternatives.append((k, alt, alt_sim))
    for j in range(1, cfg.continue_edits + 1):
        k = cfg.k_alternatives + j - 1
        alt, alt_sim = _continue_candidate(position, base["actions"], policy, j,
                                           max_turns, salt, f"{label}cont{j}")
        same = seen.get(alt["post_state_key"])
        if same is not None:
            alt["identical_to"] = same
            dropped.append(alt)
            continue
        seen[alt["post_state_key"]] = f"cont{j}"
        alternatives.append((k, alt, alt_sim))

    _run_playouts(base, base_sim, position, mover, max_turns, cfg, 0, policy, label)
    for k, alt, alt_sim in alternatives:
        _run_playouts(alt, alt_sim, position, mover, max_turns, cfg, k + 1, policy, label)

    meta = {"scenario_id": position.scenario_id, "turn_number": turn0,
            "side": mover, "turn_salt": salt, "meta": dict(position.meta)}
    record = finish_record(position.index, meta, base, [a for _, a, _ in alternatives],
                           dropped, cfg, time.perf_counter() - t0)
    n_playouts = sum(len(c["outcomes"]) for c in [base] + record["alternatives"])
    n_capped = sum(c["n_capped"] for c in [base] + record["alternatives"])
    log.info("position %d %s turn %d side %d: base %+.2f best alt %s gap %+.2f, "
             "%d/%d alternatives distinct, capped %d/%d, %d decisions, %.0f s",
             position.index, position.scenario_id, turn0, mover, base["mean"],
             ("none" if record["best_alternative_mean"] is None
              else f"{record['best_alternative_mean']:+.2f}"),
             record["gap"], record["n_alternatives"], cfg.k_alternatives,
             n_capped, n_playouts, base["n_decisions"], record["secs"])
    return record


# ---------------------------------------------------------------------
# Many positions: in-process or one policy per worker process
# ---------------------------------------------------------------------

_WORKER_POLICY = None


def _worker_init(spec: PolicySpec, log_level: str) -> None:
    global _WORKER_POLICY
    logging.basicConfig(level=getattr(logging, log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    import torch
    torch.set_num_threads(2)
    _WORKER_POLICY = load_reference_policy(spec)


def _worker_task(args) -> Dict:
    position, cfg = args
    return measure_position(_WORKER_POLICY, position, cfg)


def _measure_parallel(positions: Sequence[BoundaryPosition], cfg: GapConfig,
                      spec: PolicySpec, jobs: int, log_level: str,
                      on_record) -> List[Dict]:
    ctx = mp.get_context("spawn")
    records: List[Dict] = []
    with ctx.Pool(jobs, initializer=_worker_init, initargs=(spec, log_level)) as pool:
        for rec in pool.imap_unordered(_worker_task, [(p, cfg) for p in positions]):
            records.append(rec)
            log.info("collected position %d (%d/%d)", rec["index"],
                     len(records), len(positions))
            on_record(rec)
    records.sort(key=lambda r: r["index"])
    return records


def measure_positions(positions: Sequence[BoundaryPosition], cfg: GapConfig, *,
                      policy=None, spec: Optional[PolicySpec] = None,
                      jobs: int = 1, log_level: str = "INFO",
                      on_record=None) -> List[Dict]:
    """Records for every position, in position order. `jobs` > 1 runs
    positions in separate processes, each loading the policy from
    `spec` once. `on_record` is called with each record as it
    completes (the CLI writes a partial file from it)."""
    on_record = on_record or (lambda rec: None)
    if jobs <= 1:
        if policy is None:
            policy = load_reference_policy(spec)
        records = []
        for p in positions:
            records.append(measure_position(policy, p, cfg))
            on_record(records[-1])
        return records
    if spec is None:
        raise ValueError("jobs > 1 needs a PolicySpec: each worker loads the policy")
    return _measure_parallel(positions, cfg, spec, jobs, log_level, on_record)


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def split_gap(record: Dict) -> Optional[float]:
    """Best alternative chosen on the even-numbered playouts, its gap
    to the base measured on the odd-numbered ones. None when a half
    is empty or the position has no alternative."""
    alts = [a["outcomes"] for a in record["alternatives"]]
    base = record["base"]["outcomes"]
    if not alts or len(base) < 2 or any(len(a) < 2 for a in alts):
        return None
    best = max(alts, key=lambda xs: statistics.fmean(xs[0::2]))
    return float(statistics.fmean(best[1::2]) - statistics.fmean(base[1::2]))


def null_gap_fraction(records: Sequence[Dict], threshold: float,
                      n_permutations: int, seed: int = 0) -> float:
    """Fraction of positions with gap >= threshold when the outcomes
    of a position are reshuffled across its candidates (all candidates
    equal): the selection-noise floor of the headline fraction."""
    rng = np.random.default_rng(seed)
    per_position = []
    for r in records:
        cands = [r["base"]] + list(r["alternatives"])
        if len(cands) < 2:
            per_position.append(0.0)
            continue
        pooled = np.concatenate([np.asarray(c["outcomes"], dtype=float) for c in cands])
        bounds = np.cumsum([0] + [len(c["outcomes"]) for c in cands])
        hits = 0
        for _ in range(n_permutations):
            perm = rng.permutation(pooled)
            means = [perm[bounds[i]:bounds[i + 1]].mean() for i in range(len(cands))]
            hits += (max(means[1:]) - means[0]) >= threshold
        per_position.append(hits / n_permutations)
    return float(np.mean(per_position))


def _mean_se(values: Sequence[float]) -> Tuple[float, float]:
    n = len(values)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return mean, se


def summarize(records: Sequence[Dict], *, threshold: float = 0.25,
              wall_secs: float = 0.0, dollars_per_hour: float = 0.0,
              null_permutations: int = 200, seed: int = 0) -> Dict:
    if not records:
        raise ValueError("no records")
    n = len(records)
    gaps = [r["gap"] for r in records]
    n_ge = sum(g >= threshold for g in gaps)
    frac = n_ge / n
    mean_gap, mean_gap_se = _mean_se(gaps)
    splits = [s for s in (split_gap(r) for r in records) if s is not None]
    split_mean, split_se = _mean_se(splits) if splits else (None, None)
    counts, _ = np.histogram(gaps, bins=GAP_HISTOGRAM_EDGES)
    cands = [([r["base"]] + list(r["alternatives"])) for r in records]
    # Playouts played (a terminal candidate turn plays none; its
    # repeated entries count in the means, not here).
    playouts_total = sum(playouts_run(c) for cs in cands for c in cs)
    playouts_capped = sum(sum(bool(x) for x in c["capped"])
                          for cs in cands for c in cs if playouts_run(c))
    alt_decisions = [a["n_decisions"] for r in records for a in r["alternatives"]]
    return {
        "n_positions": n,
        "gap_threshold": threshold,
        "n_gap_ge_threshold": int(n_ge),
        "frac_gap_ge_threshold": frac,
        "frac_gap_ge_threshold_se": math.sqrt(frac * (1.0 - frac) / n),
        "null_frac_gap_ge_threshold": null_gap_fraction(records, threshold,
                                                         null_permutations, seed),
        "null_permutations": null_permutations,
        "mean_gap": mean_gap,
        "mean_gap_se": mean_gap_se,
        "n_split": len(splits),
        "mean_gap_split": split_mean,
        "mean_gap_split_se": split_se,
        "frac_gap_split_ge_threshold": (sum(s >= threshold for s in splits) / len(splits)
                                        if splits else None),
        "n_base_best": int(sum(bool(r["base_is_best"]) for r in records)),
        "frac_base_best": sum(bool(r["base_is_best"]) for r in records) / n,
        "n_no_alternative": int(sum(r["n_alternatives"] == 0 for r in records)),
        "n_terminal_in_turn": int(sum(bool(r["base"].get("terminal_in_turn"))
                                      for r in records)),
        "alternatives_distinct_mean": float(np.mean([r["n_alternatives"] for r in records])),
        "alternatives_sampled": int(records[0]["n_alternatives_sampled"]),
        "histogram": [{"lo": GAP_HISTOGRAM_EDGES[i], "hi": GAP_HISTOGRAM_EDGES[i + 1],
                       "count": int(counts[i])} for i in range(len(counts))],
        "playouts_total": int(playouts_total),
        "playouts_capped": int(playouts_capped),
        "playouts_capped_frac": (playouts_capped / playouts_total if playouts_total else 0.0),
        "decisions_per_turn_base": float(np.mean([r["base"]["n_decisions"] for r in records])),
        "decisions_per_turn_alternatives": (float(np.mean(alt_decisions))
                                            if alt_decisions else None),
        "wall_secs": float(wall_secs),
        "dollars": float(wall_secs) / 3600.0 * float(dollars_per_hour),
    }


def markdown_summary(s: Dict) -> str:
    def opt(v, fmt):
        return "-" if v is None else format(v, fmt)
    thr = s["gap_threshold"]
    rows = [
        ("positions", f"{s['n_positions']}"),
        (f"gap >= {thr:g}", f"{s['n_gap_ge_threshold']}/{s['n_positions']} = "
                            f"{s['frac_gap_ge_threshold']:.3f} +- "
                            f"{s['frac_gap_ge_threshold_se']:.3f}"),
        (f"same, permutation null ({s['null_permutations']} shuffles)",
         f"{s['null_frac_gap_ge_threshold']:.3f}"),
        ("mean gap", f"{s['mean_gap']:+.3f} +- {s['mean_gap_se']:.3f}"),
        ("mean split-half gap", f"{opt(s['mean_gap_split'], '+.3f')} +- "
                                f"{opt(s['mean_gap_split_se'], '.3f')} (n={s['n_split']})"),
        (f"split-half gap >= {thr:g}", opt(s["frac_gap_split_ge_threshold"], ".3f")),
        ("base turn best or tied", f"{s['n_base_best']}/{s['n_positions']} = "
                                   f"{s['frac_base_best']:.3f}"),
        ("positions without a distinct alternative", f"{s['n_no_alternative']}"),
        ("positions whose base turn ended the game", f"{s['n_terminal_in_turn']}"),
        ("distinct alternatives per position", f"{s['alternatives_distinct_mean']:.2f} "
                                               f"of {s['alternatives_sampled']}"),
        ("playouts played", f"{s['playouts_total']}, capped {s['playouts_capped']} "
                     f"({s['playouts_capped_frac']:.3f})"),
        ("decisions per turn, base / alternatives",
         f"{s['decisions_per_turn_base']:.1f} / "
         f"{opt(s['decisions_per_turn_alternatives'], '.1f')}"),
        ("wall", f"{s['wall_secs'] / 3600.0:.2f} h, ${s['dollars']:.2f}"),
    ]
    lines = ["| quantity | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    lines += ["", "| gap bin | positions |", "|---|---|"]
    lines += [f"| [{b['lo']:+.2f}, {b['hi']:+.2f}) | {b['count']} |"
              for b in s["histogram"] if b["count"]]
    return "\n".join(lines)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _resolve_inference(args) -> PolicySpec:
    """Device and precision as tools/elo_eval_game resolves them:
    bf16 and compile default ON on cuda, OFF on cpu; forcing them on
    cpu is refused (they would no-op and mislabel the run)."""
    import torch
    cuda = args.device == "cuda"
    if cuda and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no CUDA device is visible")
    bf16 = cuda if args.infer_bf16 is None else args.infer_bf16
    comp = cuda if args.infer_compile is None else args.infer_compile
    if (bf16 or comp) and not cuda:
        raise SystemExit("--infer-bf16/--infer-compile need --device cuda")
    ckpt = args.checkpoint
    if ckpt != "random" and not Path(ckpt).exists():
        raise SystemExit(f"--checkpoint {ckpt!r} does not exist (pass the literal "
                         f"'random' for a deliberate random-init policy)")
    return PolicySpec(checkpoint=ckpt, device=args.device, infer_bf16=bf16,
                      infer_compile=comp)


def write_json(path: Path, payload: Dict) -> None:
    """Atomic: a kill mid-write never leaves a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def resummarize(path: Path, threshold: float, dollars_per_hour: float) -> Tuple[Dict, str]:
    """Summary and markdown from a result file, full or partial
    (`<out>.partial.json`, written after every completed position).
    Wall time is the file's own when it has one."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    wall = float((data.get("summary") or {}).get("wall_secs", 0.0))
    summary = summarize(data["positions"], threshold=threshold, wall_secs=wall,
                        dollars_per_hour=dollars_per_hour)
    return summary, markdown_summary(summary)


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None,
                    help="Reference weights (.pt), or the literal 'random'.")
    ap.add_argument("--summarize", type=Path, default=None,
                    help="Print the summary of an existing result file (full or "
                         ".partial.json) at --gap-threshold and exit.")
    ap.add_argument("--states-json", type=Path, default=DEFAULT_MANIFEST,
                    help="Boundary-position manifest (tools/bench_pipeline.py).")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help="Replay dataset the manifest's games are rebuilt from.")
    ap.add_argument("--n-states", type=int, default=60,
                    help="First N manifest positions.")
    ap.add_argument("--alternatives", type=int, default=4, help="K sampled turns.")
    ap.add_argument("--continue-edits", type=int, default=0,
                    help="Continue-edit alternatives: the base turn minus its "
                         "end_turn plus 1..k more argmax non-end actions.")
    ap.add_argument("--playouts", type=int, default=40, help="P per candidate.")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Sampling temperature of the alternative turns.")
    ap.add_argument("--positions", default=None,
                    help="Comma-separated manifest indices to measure instead of "
                         "the first --n-states (a confirmation run).")
    ap.add_argument("--playout-offset", type=int, default=0,
                    help="First playout index; the salts of a confirmation run "
                         "must not overlap the run it confirms.")
    ap.add_argument("--playout-temperature", type=float, default=0.0,
                    help="Temperature of both sides during the playouts "
                         "(0 = the reference raw:t0; 0.5 avoids the "
                         "deterministic stalls, docs/box_specs.md).")
    ap.add_argument("--cap-turns", type=int, default=40,
                    help="Playouts end undecided at the start of turn T0 + cap + 1.")
    ap.add_argument("--gap-threshold", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1,
                    help="Positions in parallel, one process (and policy) each.")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--infer-bf16", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--infer-compile", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--shared-inference", action="store_true",
                    help="One inference server (tools/eval_inference_server.py) "
                         "owns the model; the --jobs workers keep the sim and "
                         "the raw player and send their forwards to it in "
                         "coalesced batches. The server's precision path "
                         "(bf16 and the packed trunk on cuda) is recorded.")
    ap.add_argument("--inference-window-ms", type=float, default=1.5,
                    help="Shared inference: how long the server collects "
                         "requests after the first before one forward.")
    ap.add_argument("--dollars-per-hour", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=None,
                    help="JSON with every record and the summary; the markdown "
                         "summary goes next to it as .md.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.summarize is not None:
        _, report = resummarize(args.summarize, args.gap_threshold, args.dollars_per_hour)
        print(report)
        return 0
    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required (or --summarize FILE)")

    import torch
    torch.set_num_threads(2)
    spec = _resolve_inference(args)
    server = None
    if args.shared_inference:
        if spec.checkpoint == "random":
            raise SystemExit("--shared-inference needs a checkpoint file (the server "
                             "loads it), not 'random'")
        from tools.eval_inference_server import launch_inference_server
        server = launch_inference_server(
            spec.checkpoint, args.out.parent if args.out else Path("."),
            tag="turn_gap", device=spec.device, infer_bf16=spec.infer_bf16,
            window_ms=args.inference_window_ms, max_batch=max(1, args.jobs))
        spec = PolicySpec(checkpoint=spec.checkpoint, device=spec.device,
                          infer_bf16=bool(server.info["infer_bf16"]),
                          infer_compile=False, inference_address=server.address,
                          infer_packed_trunk=bool(server.info["packed_trunk"]))
        log.info("inference server at %s: %s", server.address, server.info)
    cfg = GapConfig(k_alternatives=args.alternatives, playouts=args.playouts,
                    temperature=args.temperature, cap_turns=args.cap_turns,
                    playout_temperature=args.playout_temperature,
                    playout_offset=args.playout_offset,
                    continue_edits=args.continue_edits,
                    seed=args.seed)
    if args.positions:
        wanted = sorted({int(x) for x in args.positions.split(",")})
        positions = [p for p in positions_from_manifest(args.states_json, args.dataset,
                                                        max(wanted) + 1)
                     if p.index in set(wanted)]
        if len(positions) != len(wanted):
            raise SystemExit(f"--positions: {sorted(set(wanted) - {p.index for p in positions})} "
                             f"not in the manifest")
    else:
        positions = positions_from_manifest(args.states_json, args.dataset, args.n_states)
    log.info("%d positions, K=%d P=%d T=%g cap=%d seed=%d, %s bf16=%s compile=%s "
             "packed=%s shared=%s jobs=%d",
             len(positions), cfg.k_alternatives, cfg.playouts, cfg.temperature,
             cfg.cap_turns, cfg.seed, spec.device, spec.infer_bf16,
             spec.infer_compile, spec.infer_packed_trunk,
             spec.inference_address is not None, args.jobs)

    t0 = time.time()
    header = {
        "config": asdict(cfg),
        "provenance": {
            "reference_procedure": REFERENCE_PROCEDURE,
            "alternative_procedure": f"raw:t{cfg.temperature:g}",
            "playout_procedure": f"raw:t{cfg.playout_temperature:g}",
            "policy": asdict(spec), "shared_inference": server is not None,
            "torch": torch.__version__,
            "states_json": str(args.states_json), "dataset": str(args.dataset),
            "n_states": len(positions), "jobs": args.jobs,
            "dollars_per_hour": args.dollars_per_hour,
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
        },
    }
    partial: List[Dict] = []

    def on_record(rec: Dict) -> None:
        # Completed positions survive a box death; `--summarize` reads them.
        if args.out is None:
            return
        partial.append(rec)
        write_json(args.out.with_suffix(".partial.json"),
                   dict(header, summary={"wall_secs": time.time() - t0},
                        positions=sorted(partial, key=lambda r: r["index"])))

    try:
        records = measure_positions(positions, cfg, spec=spec, jobs=args.jobs,
                                    log_level=args.log_level, on_record=on_record)
    finally:
        server_stats = server.shutdown() if server is not None else None
    wall = time.time() - t0
    summary = summarize(records, threshold=args.gap_threshold, wall_secs=wall,
                        dollars_per_hour=args.dollars_per_hour)
    if server is not None:
        summary["inference_server"] = server_stats
        log.info("inference server stats: %s", server_stats)
    report = markdown_summary(summary)
    print(report)
    if args.out:
        write_json(args.out, dict(header, summary=summary, positions=records))
        args.out.with_suffix(".md").write_text(report + "\n", encoding="utf-8")
        log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
