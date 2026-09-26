#!/usr/bin/env python3
"""Training-path cost of one az_loop update (plan 1.3, "training-path
bf16/compile validation"): ms per experience for every stage of
`Trainer.step_mcts` as the loop runs it, fp32 against bf16 autocast
through the trainer's own switch (TrainerConfig.train_autocast_bf16,
what az_loop --train-bf16 runs; fp32 master weights and the optimizer
untouched) and against a compiled trunk, the loss/gradient agreement
of each variant on one batch (the compiled variants against the eager
fp32 reference on the same weights), and the implied seconds per loop
iteration next to the pool's generation time.

Stages, as Trainer.step_mcts reports them through its `timings` hook
(stream-ordered on cuda: CUDA events, so a CPU stage that overlaps
queued GPU work is charged only its non-overlapped part; perf_counter
on cpu), plus the snapshot timed here:
  encode_raw   GameState -> RawEncoded, one per experience, all up front
  encode       encode_from_raw_batch (embedding lookups), per chunk
  forward      model.forward_padded, per chunk
  policy_loss  legality masks staged on the host + the batched
               factored CE of the chunk (trainer._batched_factored_policy_loss)
  value_loss   the value-side terms of the chunk
  backward     chunk_loss.backward()
  clip         clip_grad_norm_, once per step
  optimizer    AdamW step, once per step
  snapshot     TransformerPolicy._snapshot_inference_weights, once

Experiences:
  --source bench  the 200 holdout-ladder positions of
                  configs/bench_states.json (the positions every
                  phase-1 benchmark measures) with visit counts drawn
                  from the prior. Seconds to build, deterministic, and
                  the trainer's cost is set by the states (tokens,
                  legal-action count), not by which of the <= sims
                  legal actions carry the visits.
  --source pool   one short actor-pool iteration as az_loop runs it:
                  real search targets on real self-play states, minutes
                  of box time. --experiences-out caches the result.
  --source file   a cache written by --experiences-out.
N larger than the source cycles through it; the trainer keeps no
per-state cache, so repeats cost what fresh states cost.

Box:
  python tools/bench_train_step.py --checkpoint training/checkpoints/seed.pt \\
      --device cuda --dataset /workspace/bench_dataset --compile \\
      --out /workspace/bench/train_step.json --md /workspace/bench/train_step.md
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import pickle
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch  # noqa: E402

from tools.az_recipe import configure_az_trainer  # noqa: E402
from wesnoth_ai.trainer import STEP_MCTS_STAGES  # noqa: E402

log = logging.getLogger("bench_train_step")

PER_EXP_STAGES = ("encode_raw", "encode", "forward", "policy_loss", "value_loss", "backward")
PER_STEP_STAGES = ("clip", "optimizer", "snapshot")
STAGES = PER_EXP_STAGES + PER_STEP_STAGES
assert set(STAGES) - {"snapshot"} == set(STEP_MCTS_STAGES)

# Pass band for a variant against the fp32 B=1 reference on the same
# batch: bf16 forward/backward noise on this model measured ~1e-2 of
# scale (docs/box_specs.md, packed-trunk tests), so 1% on the loss and
# 5% on the gradient norm with the gradient direction within cosine
# 0.99. The fp32 rerun row is the noise floor of the harness itself.
PARITY_LOSS_REL = 1e-2
PARITY_GRAD_NORM_REL = 5e-2
PARITY_COSINE_MIN = 0.99


@dataclass
class LoopShape:
    """What one az_loop iteration asks of the training path, at the
    loop's defaults. Experiences per game: 17,231 over 32 games at max
    30 turns (training/metrics/bench_pipeline/pool_packedtrunk.json);
    the loop's max 60 turns makes games longer, so this is a floor.
    Leaves per second: the same record's saturated (833) and
    iteration-average (489) rates."""
    games_per_iter: int = 24
    holdout_frac: float = 0.2
    exps_per_game: float = 540.0
    step_cap: int = 4000          # TrainerConfig.max_transitions_per_step
    step_trials: int = 1          # backtracking trials (1 = full step accepted)
    sims: int = 32
    leaves_per_s: float = 833.0
    iteration_leaves_per_s: float = 489.0
    signal_states: int = 128      # signal_grad_norms subsample ...
    signal_surgeries: int = 4     # ... one fwd+bwd pass per surgery
    probe_forwards: int = 256 + 96 + 96   # eval_value_metrics, level, target_amplitude
    kl_states: int = 100          # policy_shift states, (1 + trials) passes


# ---------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------

class StageClock:
    """Accumulates seconds per stage. On cuda every stage is a pair of
    CUDA events on the current stream, read after one synchronize in
    `finish`; on cpu perf_counter. The trainer's own stages arrive
    through `add` (Trainer.step_mcts's `timings` hook resolves its
    events itself)."""

    def __init__(self, device: torch.device):
        self.cuda = device.type == "cuda"
        self.seconds: Dict[str, float] = {s: 0.0 for s in STAGES}
        self._events: List[Tuple[str, object, object]] = []

    def add(self, seconds: Dict[str, float]) -> None:
        for name, s in seconds.items():
            self.seconds[name] += s

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self.cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._events.append((name, start, end))
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self.seconds[name] += time.perf_counter() - t0

    def finish(self) -> None:
        if self.cuda:
            torch.cuda.synchronize()
            for name, start, end in self._events:
                self.seconds[name] += start.elapsed_time(end) / 1000.0
            self._events.clear()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@contextlib.contextmanager
def _train_precision(trainer, precision: str) -> Iterator[None]:
    """The trainer's own bf16 switch for one step, restored after, so
    a bf16 row measures what az_loop --train-bf16 runs (a no-op on
    cpu, where the trainer keeps fp32)."""
    if precision not in ("fp32", "bf16"):
        raise ValueError(f"unknown precision {precision!r}")
    cfg = trainer.config
    prev = cfg.train_autocast_bf16
    cfg.train_autocast_bf16 = precision == "bf16"
    try:
        yield
    finally:
        cfg.train_autocast_bf16 = prev


def timed_step(policy, exps: List, *, precision: str, device: torch.device,
               clock: Optional[StageClock] = None):
    """One loop update: step_mcts then the inference snapshot (what
    MCTSPolicy.train_step does with replay off). Returns (stats, wall
    seconds); stage seconds land in `clock`."""
    clock = clock or StageClock(device)
    _sync(device)
    t0 = time.perf_counter()
    step_seconds: Dict[str, float] = {}
    with _train_precision(policy._trainer, precision):
        stats = policy._trainer.step_mcts(exps, timings=step_seconds)
    clock.add(step_seconds)
    with clock.stage("snapshot"):
        policy._snapshot_inference_weights()
    _sync(device)
    wall = time.perf_counter() - t0
    clock.finish()
    return stats, wall


# ---------------------------------------------------------------------
# Experiences
# ---------------------------------------------------------------------

def experiences_from_states(policy, states: Sequence, *, sims: int, rng: random.Random,
                            states_per_game: int = 20, masks: bool = True) -> List:
    """One MCTSExperience per state, visit counts = `sims` draws from
    the prior over its legal actions (the inference model, as search
    would consult it), z a coin flip, states grouped into pseudo-games
    of `states_per_game` for game_id / game_weight. `masks`: the
    state's packed legality masks ride on the experience as the actors
    ship them (commit 5c877f8); without them the trainer rebuilds the
    masks on the host, which is what the 2026-09-05 record measured
    (35.4 against 2.2 ms per experience, docs/box_specs.md "Training
    path cost")."""
    from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors
    from wesnoth_ai.server_priors import pack_masks
    from wesnoth_ai.trainer import MCTSExperience
    encoder, model = policy._inference_encoder, policy._inference_model
    out = []
    for i, gs in enumerate(states):
        with torch.no_grad():
            enc = encoder.encode(gs)
            legal = enumerate_legal_actions_with_priors(enc, model(enc), gs)
        if not legal:
            raise RuntimeError(f"state {i}: no legal action")
        weights = [max(float(la.prior), 0.0) for la in legal]
        if sum(weights) <= 0.0:
            weights = None
        counts = Counter(rng.choices(range(len(legal)), weights=weights, k=sims))
        visits = [(legal[j].actor_idx, legal[j].target_idx, legal[j].weapon_idx,
                   float(c), legal[j].type_idx) for j, c in sorted(counts.items())]
        out.append(MCTSExperience(
            game_state=gs, visit_counts=visits, z=rng.choice((-1.0, 1.0)),
            game_weight=1.0 / states_per_game, game_id=f"bench{i // states_per_game}",
            masks=pack_masks(enc, gs) if masks else None))
    return out


def bench_state_experiences(policy, manifest: Path, dataset: Path, *, limit: Optional[int],
                            sims: int, rng: random.Random, masks: bool = True) -> Tuple[List, dict]:
    from tools.bench_pipeline import load_states
    if not (dataset / "manifest.jsonl").exists():
        raise SystemExit(f"--dataset {dataset} has no manifest.jsonl (pack it with "
                         f"tools/bench_pipeline.py --pack-states, or use --source pool)")
    states = [gs for gs, _scenario in load_states(manifest, dataset, limit)]
    exps = experiences_from_states(policy, states, sims=sims, rng=rng, masks=masks)
    return exps, {"kind": "bench", "manifest": str(manifest), "states": len(states),
                  "visits": f"{sims} draws from the prior", "masks_shipped": bool(masks)}


def pool_experiences(policy, *, actors: int, games: int, sims: int, max_turns: int,
                     device: torch.device, seed: int, timeout: float) -> Tuple[List, dict]:
    """One ActorPool iteration configured as az_loop configures it
    (plain PUCT, no Gumbel root, no tree reuse, ladder maps, server
    priors, bf16 + packed trunk on cuda)."""
    from tools.actor_pool import ActorPool
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy, ReplayConfig
    from tools.wesnoth_sim import PvPDefaults
    cuda = device.type == "cuda"
    if cuda:
        base = getattr(policy, "_inference_base", policy._inference_model)
        base.infer_autocast_bf16 = True
        base.infer_packed_trunk = True
    mcts_cfg = MCTSConfig(n_simulations=sims, gumbel_root=False, tree_reuse=False,
                          playout_cap_randomization=False, draw_tiebreak=None,
                          batch_size=16 if cuda else 1)
    mpolicy = MCTSPolicy(policy, mcts_cfg, replay_config=ReplayConfig(enabled=False),
                         holdout_size=0, gbc_labels=False)
    pool = ActorPool(mpolicy, actors, mcts_cfg, turn_cfg=None, pt_cfg=None,
                     gbc_labels=False, train_kwargs={},
                     scenario_opts=dict(forced_faction=None, mini_maps=None, mini_ratio=0.0,
                                        fogless_ratio=0.0, ladder_ratio=1.0,
                                        midgame_ratio=0.0, midgame_dataset=None),
                     max_turns=max_turns, max_turns_min=max_turns,
                     pvp_defaults=PvPDefaults(), device=device, max_batch=16,
                     log_level=logging.WARNING, iteration_timeout=timeout,
                     drain_grace=120.0, server_priors=True, infer_bf16=cuda)
    pool.start()
    try:
        outcomes, exps = pool.run_iteration(0, games, seed)
    finally:
        pool.shutdown()
    gen = getattr(pool, "last_iteration_seconds", None)
    forwards = getattr(pool, "last_served_forwards", 0) or 0
    return exps, {"kind": "pool", "actors": actors, "games": len(outcomes),
                  "decisive": sum(1 for o in outcomes if o.winner != 0),
                  "sims": sims, "max_turns": max_turns, "gen_seconds": gen,
                  "forwards": forwards,
                  "leaves_per_s": (forwards / gen) if gen else None,
                  "experiences_per_game": (len(exps) / len(outcomes)) if outcomes else None}


def cycle_to(exps: Sequence, n: int) -> List:
    return [exps[i % len(exps)] for i in range(n)]


# ---------------------------------------------------------------------
# Parity, compile
# ---------------------------------------------------------------------

def _params(trainer) -> List[torch.nn.Parameter]:
    return list(trainer.model.parameters()) + list(trainer.encoder.parameters())


def stubbed_step(policy, exps: List, *, precision: str, batch_size: int,
                 device: torch.device) -> Tuple[dict, torch.Tensor]:
    """step_mcts with the optimizer step stubbed and clipping off:
    the weights stay put, the gradient is the raw one. Returns the
    losses and the flattened gradient over model + encoder."""
    tr = policy._trainer
    real_step, real_clip, real_b = tr.optimizer.step, tr.config.grad_clip, tr.config.train_batch_size
    tr.optimizer.step = lambda *a, **k: None
    tr.config.grad_clip = 1e9
    tr.config.train_batch_size = batch_size
    try:
        with _train_precision(tr, precision):
            stats = tr.step_mcts(exps)
    finally:
        tr.optimizer.step = real_step
        tr.config.grad_clip = real_clip
        tr.config.train_batch_size = real_b
    grad = torch.cat([(p.grad.detach().float().flatten() if p.grad is not None
                       else torch.zeros(p.numel(), device=p.device))
                      for p in _params(tr)])
    tr.optimizer.zero_grad(set_to_none=True)
    losses = {"policy_loss": float(stats.policy_loss), "value_loss": float(stats.value_loss),
              "total_loss": float(stats.total_loss), "grad_norm": float(grad.norm().item())}
    return losses, grad


def parity_row(name: str, ref: Tuple[dict, torch.Tensor], cand: Tuple[dict, torch.Tensor],
               *, is_reference: bool = False) -> dict:
    ref_losses, ref_g = ref
    losses, g = cand
    ref_g, g = ref_g.double(), g.double()     # float32 dot products read cosine > 1
    ref_norm = float(ref_g.norm().item())
    rel_l2 = float((g - ref_g).norm().item()) / max(ref_norm, 1e-12)
    cosine = float((g @ ref_g).item()) / max(ref_norm * float(g.norm().item()), 1e-12)
    loss_rel = (abs(losses["total_loss"] - ref_losses["total_loss"])
                / max(abs(ref_losses["total_loss"]), 1e-9))
    norm_ratio = losses["grad_norm"] / max(ref_norm, 1e-12)
    ok = (loss_rel <= PARITY_LOSS_REL and abs(norm_ratio - 1.0) <= PARITY_GRAD_NORM_REL
          and cosine >= PARITY_COSINE_MIN)
    return {"config": name, "reference": is_reference, **losses,
            "loss_rel_diff": loss_rel, "grad_norm_ratio": norm_ratio,
            "grad_rel_l2_diff": rel_l2, "cosine": cosine, "ok": bool(ok)}


def parity_check(policy, exps: List, configs: Sequence[Tuple[str, int]], *,
                 device: torch.device, label: str = "",
                 reference: Optional[Tuple[dict, torch.Tensor]] = None) -> List[dict]:
    """Every (precision, batch) against fp32 B=1 on the same batch.
    Without `reference` the fp32 B=1 result is computed here and its
    rerun is the harness's own noise floor. With it (the compiled
    table passes the eager fp32 B=1 result computed before the trunk
    was compiled, on the weights the compiled rows see) every config
    is a candidate, fp32 B=1 included: a compiled trunk is scored
    against the eager one, never against itself."""
    if reference is None:
        ref = stubbed_step(policy, exps, precision="fp32", batch_size=1, device=device)
        rows = [parity_row("fp32 B=1" + label, ref, ref, is_reference=True),
                parity_row("fp32 B=1 rerun" + label, ref,
                           stubbed_step(policy, exps, precision="fp32", batch_size=1,
                                        device=device))]
        skip = ("fp32", 1)
    else:
        ref = reference
        rows = [parity_row("fp32 B=1 eager", ref, ref, is_reference=True)]
        skip = None
    for precision, b in configs:
        if (precision, b) == skip:
            continue
        cand = stubbed_step(policy, exps, precision=precision, batch_size=b, device=device)
        rows.append(parity_row(f"{precision} B={b}{label}", ref, cand))
    return rows


def enable_compiled_trunk(model) -> dict:
    """The one-line switch: compile the trunk's forward as an instance
    attribute of the trainer's nn.TransformerEncoder. Both the
    single-sample and the padded path call self.encoder(...), and the
    parameters stay where they are, so state_dict keys (the inference
    snapshot's strict load) are unchanged. Dynamic shapes: every chunk
    has its own token count."""
    from torch._dynamo.utils import counters
    counters.clear()
    model.encoder.forward = torch.compile(model.encoder.forward, dynamic=True)
    return {"cache_size_limit": int(torch._dynamo.config.cache_size_limit)}


def disable_compiled_trunk(model) -> None:
    if "forward" in model.encoder.__dict__:
        del model.encoder.forward


def compile_counters() -> dict:
    from torch._dynamo.utils import counters
    out = {}
    for key in ("frames", "stats"):
        if key in counters:
            out[key] = {k: int(v) for k, v in counters[key].items()}
    breaks = counters.get("graph_break", {})
    out["graph_breaks"] = {str(k)[:160]: int(v)
                           for k, v in sorted(breaks.items(), key=lambda kv: -kv[1])[:10]}
    return out


# ---------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------

def measure_config(policy, source: List, *, n: int, batch_size: int, precision: str,
                   device: torch.device, repeats: int, compiled: bool) -> dict:
    tr = policy._trainer
    tr.config.train_batch_size = batch_size
    batch = cycle_to(source, n)
    _sync(device)
    t0 = time.perf_counter()
    timed_step(policy, batch, precision=precision, device=device)   # warmup (compile lands here)
    warmup_s = time.perf_counter() - t0
    reps = []
    for _ in range(repeats):
        clock = StageClock(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        stats, wall = timed_step(policy, batch, precision=precision, device=device, clock=clock)
        reps.append({
            "seconds": dict(clock.seconds), "wall": wall,
            "peak_mb": (torch.cuda.max_memory_allocated() / 2 ** 20
                        if device.type == "cuda" else None),
            "grad_norm": float(stats.grad_norm), "total_loss": float(stats.total_loss)})
    med = lambda key: statistics.median(r[key] for r in reps)  # noqa: E731
    med_stage = lambda s: statistics.median(r["seconds"][s] for r in reps)  # noqa: E731
    ms_per_exp = {s: 1000.0 * med_stage(s) / n for s in PER_EXP_STAGES}
    per_step_ms = {s: 1000.0 * med_stage(s) for s in PER_STEP_STAGES}
    wall = med("wall")
    attributed = sum(med_stage(s) for s in STAGES)
    row = {
        "precision": precision, "batch_size": batch_size, "n": n, "compiled": compiled,
        "repeats": repeats, "warmup_s": warmup_s,
        "ms_per_exp": ms_per_exp,
        "fwd_bwd_ms_per_exp": sum(ms_per_exp.values()),
        "clip_ms": per_step_ms["clip"], "optimizer_ms": per_step_ms["optimizer"],
        "snapshot_ms": per_step_ms["snapshot"],
        "step_wall_s": wall, "step_wall_ms_per_exp": 1000.0 * wall / n,
        "unattributed_ms_per_exp": 1000.0 * (wall - attributed) / n,
        "gpu_peak_mb": med("peak_mb") if device.type == "cuda" else None,
        "grad_norm": med("grad_norm"), "total_loss": med("total_loss"),
    }
    if compiled:
        row["compile"] = compile_counters()
    return row


def implied_iteration(row: dict, loop: LoopShape) -> dict:
    """Seconds one az_loop iteration spends on the training path at
    this row's per-experience costs, next to the pool's generation
    time for the same iteration. Forward+backward passes: the step
    (capped), the held-out loss before the step and once per trial,
    the signal-telemetry surgeries. Forward-only passes (probes and
    policy_shift) are priced at the row's encode + forward stages."""
    n_held_games = max(1, int(round(loop.holdout_frac * loop.games_per_iter)))
    n_train_games = loop.games_per_iter - n_held_games
    step_exps = min(n_train_games * loop.exps_per_game, loop.step_cap)
    held_exps = n_held_games * loop.exps_per_game
    held_passes = 1 + loop.step_trials
    signal_exps = loop.signal_surgeries * loop.signal_states
    fwd_bwd_exps = step_exps + held_exps * held_passes + signal_exps
    fwd_only_exps = loop.probe_forwards + loop.kl_states * (1 + loop.step_trials)
    ms = row["ms_per_exp"]
    fwd_only_ms = ms["encode_raw"] + ms["encode"] + ms["forward"]
    n_step_calls = 1 + n_held_games * held_passes + loop.signal_surgeries
    # publish_weights = two load_state_dicts, once per trial plus the
    # full proposal; two state_dict clones; the step's own snapshot.
    n_snapshot_equiv = 1 + 2 + 2 * (1 + loop.step_trials)
    fixed_s = (row["optimizer_ms"] + row["clip_ms"] * n_step_calls
               + row["snapshot_ms"] * n_snapshot_equiv) / 1000.0
    train_s = (fwd_bwd_exps * row["fwd_bwd_ms_per_exp"]
               + fwd_only_exps * fwd_only_ms) / 1000.0 + fixed_s
    leaves = loop.games_per_iter * loop.exps_per_game * loop.sims
    gen_sat = leaves / loop.leaves_per_s
    gen_avg = leaves / loop.iteration_leaves_per_s
    return {
        "precision": row["precision"], "batch_size": row["batch_size"],
        "compiled": row["compiled"], "n_measured": row["n"],
        "fwd_bwd_experiences": fwd_bwd_exps, "fwd_only_experiences": fwd_only_exps,
        "step_experiences": step_exps, "held_experiences_per_pass": held_exps,
        "fixed_s": fixed_s, "train_path_s": train_s,
        "leaves_per_iteration": leaves,
        "generation_s_saturated": gen_sat,
        "fraction_saturated": train_s / (train_s + gen_sat),
        "generation_s_iteration_avg": gen_avg,
        "fraction_iteration_avg": train_s / (train_s + gen_avg),
    }


def run_benchmark(policy, exps: List, *, device: torch.device, n_list: Sequence[int],
                  batch_sizes: Sequence[int], precisions: Sequence[str], repeats: int,
                  parity_n: int, compile_trunk: bool = False,
                  loop: Optional[LoopShape] = None,
                  partial_out: Optional[Path] = None) -> dict:
    """`partial_out`: rewritten after every parity table and every timed
    row, so a run can be read (or cut) while it is going."""
    loop = loop or LoopShape()

    def _save_partial(parity, rows, compile_info):
        if partial_out is not None:
            partial_out.write_text(json.dumps(
                {"partial": True, "parity": parity, "rows": rows,
                 "compile": compile_info}, indent=1), encoding="utf-8")
    tr = policy._trainer
    configure_az_trainer(tr)
    # The largest timed step must fit in one trainer step.
    tr.config.max_transitions_per_step = int(max(loop.step_cap, max(n_list)))
    configs = [(p, b) for p in precisions for b in batch_sizes]
    parity_batch = cycle_to(exps, parity_n)
    # Parity first, on the checkpoint's weights (the timed steps move them).
    parity = parity_check(policy, parity_batch, configs, device=device)
    rows = []
    compile_info: dict = {"requested": bool(compile_trunk), "active": False}
    _save_partial(parity, rows, compile_info)
    for precision, b in configs:
        for n in n_list:
            log.info(f"timing {precision} B={b} N={n}")
            rows.append(measure_config(policy, exps, n=n, batch_size=b, precision=precision,
                                       device=device, repeats=repeats, compiled=False))
            log.info(f"  -> {rows[-1].get('step_wall_ms_per_exp', rows[-1])}")
            _save_partial(parity, rows, compile_info)
    if compile_trunk:
        # The eager fp32 B=1 result on the weights the compiled rows
        # will see (the timed rows above moved them), taken BEFORE the
        # trunk is compiled: the compiled rows are scored against it.
        eager_ref = stubbed_step(policy, parity_batch, precision="fp32", batch_size=1,
                                 device=device)
        try:
            compile_info.update(enable_compiled_trunk(tr.model))
            parity += parity_check(policy, parity_batch, configs, device=device,
                                   label=" compiled", reference=eager_ref)
            for precision, b in configs:
                for n in n_list:
                    log.info(f"timing {precision} B={b} N={n} compiled")
                    rows.append(measure_config(policy, exps, n=n, batch_size=b,
                                               precision=precision, device=device,
                                               repeats=repeats, compiled=True))
                    _save_partial(parity, rows, compile_info)
            compile_info["active"] = True
            compile_info["counters"] = compile_counters()
        except Exception as e:  # noqa: BLE001 -- the tool reports, the reader decides
            log.warning(f"compiled trunk unavailable: {e!r}")
            compile_info["error"] = repr(e)
        finally:
            disable_compiled_trunk(tr.model)
    largest = max(n_list)
    implied = [implied_iteration(r, loop) for r in rows if r["n"] == largest]
    return {"rows": rows, "parity": parity, "implied": implied, "compile": compile_info,
            "loop": asdict(loop), "unique_experiences": len(exps),
            "parity_n": parity_n, "n_list": list(n_list), "batch_sizes": list(batch_sizes),
            "precisions": list(precisions), "repeats": repeats,
            "parity_tolerance": {"loss_rel": PARITY_LOSS_REL,
                                 "grad_norm_rel": PARITY_GRAD_NORM_REL,
                                 "cosine_min": PARITY_COSINE_MIN}}


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def _f(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{nd}f}"


def markdown_report(res: dict) -> str:
    env = res.get("env", {})
    out = ["# Training-path cost (tools/bench_train_step.py)", ""]
    if env:
        out.append(f"{env.get('checkpoint', '?')} ({env.get('params_m', '?')} M params) on "
                   f"{env.get('device_name', '?')}, torch {env.get('torch', '?')}; "
                   f"source: {env.get('source', {})}")
        out.append("")
    stage_cols = " | ".join(PER_EXP_STAGES)
    out += [f"## Stage costs, ms per experience (median of {res['repeats']} steps; "
            f"{res['unique_experiences']} unique experiences, cycled)", "",
            f"| precision | B | N | compiled | {stage_cols} | fwd+bwd | clip ms | "
            "optimizer ms | snapshot ms | step wall ms/exp | unattributed ms/exp | "
            "GPU peak MB | warmup s |",
            "|---" * (12 + len(PER_EXP_STAGES)) + "|"]
    for r in res["rows"]:
        ms = r["ms_per_exp"]
        out.append(
            f"| {r['precision']} | {r['batch_size']} | {r['n']} | {'yes' if r['compiled'] else 'no'} | "
            + " | ".join(_f(ms[s]) for s in PER_EXP_STAGES)
            + f" | {_f(r['fwd_bwd_ms_per_exp'])} | {_f(r['clip_ms'])} | {_f(r['optimizer_ms'])} | "
            f"{_f(r['snapshot_ms'])} | {_f(r['step_wall_ms_per_exp'])} | "
            f"{_f(r['unattributed_ms_per_exp'])} | {_f(r['gpu_peak_mb'], 0)} | {_f(r['warmup_s'], 1)} |")
    tol = res["parity_tolerance"]
    out += ["", f"## Parity on one batch of {res['parity_n']} (optimizer stubbed, no clipping; "
            f"reference fp32 B=1 -- for the compiled rows the eager fp32 B=1 on the same "
            f"weights; pass = loss within {tol['loss_rel']:.0%}, gradient norm within "
            f"{tol['grad_norm_rel']:.0%}, cosine >= {tol['cosine_min']})", "",
            "| config | total loss | loss rel diff | grad norm | norm ratio | grad rel L2 diff | cosine | ok |",
            "|---|---|---|---|---|---|---|---|"]
    for p in res["parity"]:
        out.append(f"| {p['config']} | {_f(p['total_loss'], 5)} | {p['loss_rel_diff']:.2e} | "
                   f"{_f(p['grad_norm'], 4)} | {_f(p['grad_norm_ratio'], 4)} | "
                   f"{p['grad_rel_l2_diff']:.2e} | {_f(p['cosine'], 5)} | "
                   f"{'ref' if p['reference'] else ('yes' if p['ok'] else 'NO')} |")
    lp = res["loop"]
    out += ["", f"## Implied az_loop iteration ({lp['games_per_iter']} games, holdout "
            f"{lp['holdout_frac']}, {lp['exps_per_game']:.0f} experiences per game, step cap "
            f"{lp['step_cap']}, {lp['step_trials']} trial(s), {lp['sims']} sims)", "",
            "| precision | B | compiled | train-path s | generation s at "
            f"{lp['leaves_per_s']:.0f} leaves/s | fraction | generation s at "
            f"{lp['iteration_leaves_per_s']:.0f} leaves/s | fraction |",
            "|---|---|---|---|---|---|---|---|"]
    for i in res["implied"]:
        out.append(f"| {i['precision']} | {i['batch_size']} | {'yes' if i['compiled'] else 'no'} | "
                   f"{_f(i['train_path_s'], 1)} | {_f(i['generation_s_saturated'], 0)} | "
                   f"{_f(i['fraction_saturated'], 3)} | {_f(i['generation_s_iteration_avg'], 0)} | "
                   f"{_f(i['fraction_iteration_avg'], 3)} |")
    if res["implied"]:
        i0 = res["implied"][0]
        out += ["", f"Training-path passes per iteration: {i0['fwd_bwd_experiences']:.0f} "
                f"forward+backward experiences (step {i0['step_experiences']:.0f}, held-out "
                f"{i0['held_experiences_per_pass']:.0f} x {1 + lp['step_trials']}, signal "
                f"{lp['signal_surgeries'] * lp['signal_states']}) and "
                f"{i0['fwd_only_experiences']:.0f} forward-only."]
    c = res.get("compile", {})
    if c.get("requested"):
        out += ["", "## Compiled trunk", "",
                f"active: {c.get('active')}; " + (f"error: {c['error']}" if "error" in c else
                                                  f"counters: {json.dumps(c.get('counters', {}))}")]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    ap.add_argument("--source", default="bench", choices=("bench", "pool", "file"))
    ap.add_argument("--manifest", type=Path, default=ROOT / "configs" / "bench_states.json")
    ap.add_argument("--dataset", type=Path, default=ROOT / "replays_dataset_imitation",
                    help="directory holding the manifest's game files "
                         "(tools/bench_pipeline.py --pack-states output on a box)")
    ap.add_argument("--states", type=int, default=None, help="bench: cap on positions loaded")
    ap.add_argument("--masks", action=argparse.BooleanOptionalAction, default=True,
                    help="bench: the experiences carry their packed legality masks as the "
                         "actors ship them (production); --no-masks makes the trainer rebuild "
                         "them on the host, the path the 2026-09-05 record measured.")
    ap.add_argument("--experiences-in", type=Path, default=None, help="--source file input")
    ap.add_argument("--experiences-out", type=Path, default=None,
                    help="pickle the experiences used (rerun with --source file)")
    ap.add_argument("--pool-actors", type=int, default=8)
    ap.add_argument("--pool-games", type=int, default=8)
    ap.add_argument("--pool-max-turns", type=int, default=12)
    ap.add_argument("--pool-timeout", type=float, default=900.0)
    ap.add_argument("--sims", type=int, default=32,
                    help="search budget: pool simulations, or prior draws per bench state")
    ap.add_argument("--n-list", type=_int_list, default=[512, 1024, 2048],
                    help="experiences per timed step; 512 is about one game's states "
                         "(one held-out per-game loss), the loop's step is capped at 4000")
    ap.add_argument("--batch-sizes", type=_int_list, default=[1, 16],
                    help="TrainerConfig.train_batch_size; the loop runs 16")
    ap.add_argument("--precisions", default=None,
                    help="comma list of fp32,bf16 (default: both on cuda, fp32 on cpu)")
    ap.add_argument("--compile", action="store_true",
                    help="also time every config with the trunk forward compiled (dynamic shapes)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--parity-n", type=int, default=64)
    ap.add_argument("--games-per-iter", type=int, default=LoopShape.games_per_iter)
    ap.add_argument("--holdout-frac", type=float, default=LoopShape.holdout_frac)
    ap.add_argument("--exps-per-game", type=float, default=LoopShape.exps_per_game)
    ap.add_argument("--step-trials", type=int, default=LoopShape.step_trials)
    ap.add_argument("--loop-sims", type=int, default=LoopShape.sims,
                    help="the loop's search budget, for the generation-time estimate "
                         "(independent of --sims, which sizes this tool's experiences)")
    ap.add_argument("--leaves-per-s", type=float, default=LoopShape.leaves_per_s)
    ap.add_argument("--iteration-leaves-per-s", type=float,
                    default=LoopShape.iteration_leaves_per_s)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--out", type=Path, default=None, help="JSON record")
    ap.add_argument("--md", type=Path, default=None, help="markdown table (always printed)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from tools.eval_players import _load_policy
    device = (torch.device("cuda") if args.device == "cuda" and torch.cuda.is_available()
              else torch.device("cpu"))
    if args.device == "cuda" and device.type != "cuda":
        raise SystemExit("--device cuda but no cuda device")
    if device.type == "cuda":
        torch.set_num_threads(4)    # the learner process's setting in az_loop
    precisions = (args.precisions.split(",") if args.precisions
                  else (["fp32", "bf16"] if device.type == "cuda" else ["fp32"]))
    if "bf16" in precisions and device.type != "cuda":
        log.warning("the trainer's bf16 switch is a no-op on cpu: bf16 rows repeat fp32")
    policy = _load_policy(args.checkpoint, device, label="bench_train_step")
    rng = random.Random(args.seed)

    t0 = time.perf_counter()
    if args.source == "bench":
        exps, source = bench_state_experiences(policy, args.manifest, args.dataset,
                                               limit=args.states, sims=args.sims, rng=rng,
                                               masks=bool(args.masks))
    elif args.source == "pool":
        exps, source = pool_experiences(policy, actors=args.pool_actors, games=args.pool_games,
                                        sims=args.sims, max_turns=args.pool_max_turns,
                                        device=device, seed=args.seed, timeout=args.pool_timeout)
    else:
        if not args.experiences_in:
            raise SystemExit("--source file needs --experiences-in")
        from wesnoth_ai import unpickle
        with open(args.experiences_in, "rb") as f:
            exps = unpickle.load(f)
        source = {"kind": "file", "path": str(args.experiences_in)}
    source["experiences"] = len(exps)
    source["build_seconds"] = time.perf_counter() - t0
    if not exps:
        raise SystemExit("no experiences")
    if args.experiences_out:
        with open(args.experiences_out, "wb") as f:
            pickle.dump(exps, f)
    log.info(f"experiences: {source}")

    loop = LoopShape(games_per_iter=args.games_per_iter, holdout_frac=args.holdout_frac,
                     exps_per_game=args.exps_per_game, step_trials=args.step_trials,
                     sims=args.loop_sims, leaves_per_s=args.leaves_per_s,
                     iteration_leaves_per_s=args.iteration_leaves_per_s)
    res = run_benchmark(policy, exps, device=device, n_list=args.n_list,
                        batch_sizes=args.batch_sizes, precisions=precisions,
                        repeats=args.repeats, parity_n=args.parity_n,
                        compile_trunk=args.compile, loop=loop,
                        partial_out=(args.out.with_suffix(".partial.json")
                                     if args.out else None))
    n_params = sum(p.numel() for p in _params(policy._trainer))
    res["env"] = {
        "checkpoint": args.checkpoint.name, "arch": policy._arch,
        "params_m": round(n_params / 1e6, 2), "torch": torch.__version__,
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"),
        "source": source,
        "loop_train_batch_size": 16,
        "note": "az_loop sets TrainerConfig.train_batch_size 16 (2026-09-05); "
                "value loss mse_mean, coef 1, lr 1e-4, clip 1, no auxiliary terms",
    }
    md = markdown_report(res)
    print(md)
    if args.md:
        args.md.write_text(md, encoding="utf-8")
    if args.out:
        args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
