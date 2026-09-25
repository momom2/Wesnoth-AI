"""Light in-training signal telemetry (user ruling 2026-09-01:
"record the signal profiling telemetry along the timing telemetry
and all the usual signals at all times"), the trend view of the
offline signal_profiler's gradient tree.

The self-play learner (`trainer.step_mcts`, az_loop): per iteration,
on a SUBSAMPLE of the incoming batch, the gradient norm of each
signal source (policy distill / game-outcome value / rollout-grounded
value / consistency value), unclipped, optimizer stubbed. Plus
dv_consult: how far the just-applied update moved the value head's
predictions on THIS iteration's search-consulted states (the erosion
gauge; search accepts plans on ~2-atom ≈ 0.08 differences). Cost: 4
extra backward passes on <=128 states + <=64*2 value forwards ≈ a
few percent of an iteration.

The imitation trainer (tools/supervised_train.py): `ImitationSignal`,
one row every IMITATION_SIGNAL_EVERY trained pairs in
<checkpoint stem>_signal.jsonl, the gradient of a probe of the batch
just trained split by loss term over the encoder, the trunk and the
heads, and the real steps' gradient norms (its docstring). Cost: about
two training steps per row, under 1% of a pass, recorded in each row.

The full update-space tree stays offline (signal_profiler)."""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch

log = logging.getLogger("signal_telemetry")

SIG_SUBSAMPLE = 128
SIG_CONSULT_CAP = 64

_KILL = {"policy_weight": 0.0, "aux_target": None,
         "moves_left_target": None, "gbc_labels": None}


class _NoStep:
    def __init__(self, inner):
        self._inner = inner

    def zero_grad(self, *a, **k):
        return self._inner.zero_grad(*a, **k)

    def step(self, *a, **k):
        return None

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _surgeries():
    from tools.value_grounding import is_grounding_experience as ig

    def value_where(pred):
        def fn(e):
            kw = dict(_KILL)
            if not pred(e):
                kw["value_weight"] = 0.0
            return kw
        return fn

    def kind(e):
        k = getattr(e, "label_kind", None)
        if k is not None:
            return k
        if not ig(e):
            return "game"
        return "roll" if e.value_weight >= 0.9 else "consist"

    return {
        "sig_policy_norm": lambda e: {
            "value_weight": 0.0, "aux_target": None,
            "moves_left_target": None, "gbc_labels": None},
        "sig_value_game_norm": value_where(lambda e: kind(e) == "game"),
        "sig_value_ground_norm": value_where(lambda e: kind(e) == "roll"),
        "sig_value_consist_norm": value_where(
            lambda e: kind(e) == "consist"),
    }


def _grad_norm(trainer, exps: List) -> float:
    real_opt = trainer.optimizer
    real_clip = trainer.config.grad_clip
    trainer.optimizer = _NoStep(real_opt)
    trainer.config.grad_clip = 1e9
    try:
        trainer.step_mcts(list(exps))
        # One device-to-host read for the whole norm. Reading each of
        # the ~175 parameters' gradients back with its own .item() is
        # ~175 syncs per call and this runs four times an iteration.
        enc = getattr(trainer, "encoder", None)
        params = list(trainer.model.parameters())
        if enc is not None:
            params += list(enc.parameters())
        squares = [p.grad.detach().pow(2).sum() for p in params if p.grad is not None]
        if not squares:
            return 0.0
        return float(torch.stack(squares).sum().sqrt().item())
    finally:
        trainer.optimizer = real_opt
        trainer.config.grad_clip = real_clip


def signal_grad_norms(trainer, batch: List, rng) -> Dict[str, float]:
    """Per-source gradient norms on a fixed-size subsample. Runs
    AFTER the real updates (grads are scratch; the next real step
    zero_grads first). Returns {} on any failure — telemetry must
    never kill training."""
    if not batch:
        return {}
    try:
        sub = (batch if len(batch) <= SIG_SUBSAMPLE
               else rng.sample(batch, SIG_SUBSAMPLE))
        out = {}
        for name, surgery in _surgeries().items():
            exps = [dataclasses.replace(e, **surgery(e)) for e in sub]
            out[name] = _grad_norm(trainer, exps)
        return out
    except Exception as e:  # noqa: BLE001
        log.warning(f"signal telemetry failed: {e!r}")
        return {}


def consult_values(base_policy, batch: List,
                   cap: int = SIG_CONSULT_CAP) -> Optional[List]:
    """(states, values) of up to `cap` grounding-captured states
    under the CURRENT trainer-side weights; None when none exist."""
    import torch
    from tools.value_grounding import is_grounding_experience
    states = [e.game_state for e in batch
              if is_grounding_experience(e)][:cap]
    if not states:
        return None
    model, enc = base_policy._model, base_policy._encoder
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            vals = [float(model(enc.encode(gs)).value.squeeze().item())
                    for gs in states]
    finally:
        if was:
            model.train()
    return [states, vals]


def dv_stats(pre, base_policy) -> Dict[str, float]:
    """Mean |dv| on the pre-captured consult states after updates."""
    if not pre:
        return {}
    try:
        states, v0 = pre
        import torch
        model, enc = base_policy._model, base_policy._encoder
        was = model.training
        model.eval()
        try:
            with torch.no_grad():
                v1 = [float(model(enc.encode(gs)).value
                            .squeeze().item()) for gs in states]
        finally:
            if was:
                model.train()
        d = [abs(a - b) for a, b in zip(v0, v1)]
        return {"sig_dv_consult_mean": sum(d) / len(d),
                "sig_dv_consult_n": float(len(d))}
    except Exception as e:  # noqa: BLE001
        log.warning(f"dv_consult telemetry failed: {e!r}")
        return {}


# ---- The imitation trainer (tools/supervised_train.py) ----------------------

IMITATION_SOURCES = ("actor", "type", "target", "weapon", "value")
POLICY_SOURCES = ("actor", "type", "target", "weapon")
SIGNAL_GROUPS = ("encoder", "trunk", "heads")
# Trained pairs between rows: about 113 rows over obs8's 2.83M-pair pass.
# A probe costs about two training steps, so at batch 64 (390 steps per
# row) the telemetry takes about 0.5% of the pass.
IMITATION_SIGNAL_EVERY = 25_000
IMITATION_PROBE_PAIRS = 32


def signal_group(name: str) -> str:
    """The telemetry's group of a parameter named the way the profiler
    names them ("model.<name>" or "encoder.<name>"): encoder, trunk or
    heads."""
    from signal_profiler.gradient_tree import group_of
    group = group_of(name)
    return group if group in ("encoder", "trunk") else "heads"


def summarize_gram(gram: Sequence[Sequence[Sequence[float]]]) -> Dict[str, Dict]:
    """Readings of the per-group Gram matrices of the loss terms'
    gradients: `gram[k][i][j]` is <g_i, g_j> over the parameters of
    SIGNAL_GROUPS[k], terms in IMITATION_SOURCES order.

    Per group, and for "all" (their sum): the norm of the summed
    gradient and, per term, its norm, its signed share of the summed
    gradient (<g_i, g> / |g|^2; the shares add up to 1, a negative one
    pulls against the update) and its cosine with it; and the cosine
    between the four policy terms' sum and the value term."""
    n = len(IMITATION_SOURCES)
    groups = dict(zip(SIGNAL_GROUPS, gram))
    groups["all"] = [[sum(g[i][j] for g in gram) for j in range(n)] for i in range(n)]
    policy = [IMITATION_SOURCES.index(s) for s in POLICY_SOURCES]
    value = IMITATION_SOURCES.index("value")
    out = {}
    for name, g in groups.items():
        total_sq = sum(sum(row) for row in g)
        total_norm = math.sqrt(max(total_sq, 0.0))
        node: Dict = {"total_norm": total_norm}
        for i, source in enumerate(IMITATION_SOURCES):
            norm = math.sqrt(max(g[i][i], 0.0))
            dot = sum(g[i])
            node[source] = {
                "norm": norm,
                "share": dot / total_sq if total_sq > 0 else None,
                "cos": dot / (norm * total_norm) if norm > 0 and total_norm > 0 else None}
        policy_sq = sum(g[a][b] for a in policy for b in policy)
        policy_value = sum(g[a][value] for a in policy)
        value_sq = g[value][value]
        node["policy_value_cos"] = (policy_value / math.sqrt(policy_sq * value_sq)
                                    if policy_sq > 0 and value_sq > 0 else None)
        out[name] = node
    return out


class ImitationSignal:
    """The imitation trainer's always-on signal telemetry: one JSONL row
    each time the trained-pair count crosses a multiple of `every`.

    A row holds:
      - `steps`: the real optimizer steps since the previous row, their
        pre-clip gradient norms (mean, max, the share above the clip)
        and how many were not finite;
      - `groups`: the gradient of a probe (up to `probe_pairs` pairs of
        the batch just trained, at the weights that batch produced),
        split by loss term (`summarize_gram`) over the encoder, the
        trunk, the heads and all parameters. The terms are the ones the
        trainer sums (`ImitationLossParts.source_losses`), each divided
        by the probe's size as the trainer divides by the batch's;
      - `gram`: the per-group Gram matrices those readings come from,
        so any other combination of terms can be derived later;
      - `probe_ms`: what the probe cost.

    The probe changes nothing the training reads: it picks its pairs
    with its own generator, seeded by the step so a resumed run picks
    the same ones; it forks torch's generators around its forward (the
    model's dropout draws); and it takes gradients with
    `torch.autograd.grad`, which leaves every parameter's `.grad` and
    the optimizer alone. A probe that fails warns, counts into
    `failures` and the epoch accounting, and training goes on."""

    def __init__(self, path: Path, batch_loss: Callable, model: torch.nn.Module,
                 encoder: torch.nn.Module, *, pairs: int = 0,
                 every: int = IMITATION_SIGNAL_EVERY,
                 probe_pairs: int = IMITATION_PROBE_PAIRS,
                 clip: float = 1.0, seed: int = 0):
        """`batch_loss(raws, ais, zw)` returns the trainer's
        (ImitationLossParts, targets) for those pairs; `pairs` is the
        trained-pair count the run starts from (a resume's)."""
        if every < 1 or probe_pairs < 1:
            raise ValueError(f"signal telemetry needs every >= 1 and probe_pairs >= 1, "
                             f"got {every} and {probe_pairs}")
        self.path = path
        self._batch_loss = batch_loss
        named = [("model." + n, p) for n, p in model.named_parameters()]
        named += [("encoder." + n, p) for n, p in encoder.named_parameters()]
        named = [(n, p) for n, p in named if p.requires_grad]
        self._params = [p for _, p in named]
        self._group_of_param = [SIGNAL_GROUPS.index(signal_group(n)) for n, _ in named]
        self._every = every
        self._probe_pairs = probe_pairs
        self._clip = clip
        self._seed = seed
        self._last_pairs = pairs
        # Pre-clip norms of the real steps since the last row, left on
        # the device: one transfer per row, none per step.
        self.step_norms: List[torch.Tensor] = []
        self.rows = 0
        self.failures = 0
        self.probe_seconds = 0.0

    def due(self, pairs: int) -> bool:
        """Whether the trained-pair count crossed a multiple of `every`
        since the previous call."""
        crossed = pairs // self._every != self._last_pairs // self._every
        self._last_pairs = pairs
        return crossed

    def record(self, batch: Optional[tuple], *, epoch: int, step: int, pairs: int) -> Dict:
        """Write one row and return it. `batch` is the (raws, ais, zw)
        the last step trained on, or None where there is none to probe
        (the per-pair flow keeps no batch): that row carries the step
        norms alone."""
        row: Dict = {"epoch": epoch, "step": step, "pairs": pairs,
                     "ts": time.strftime("%FT%T"), "steps": self._drain_step_norms()}
        started = time.perf_counter()
        if batch is None:
            row["probe"] = "none: the per-pair flow keeps no batch"
        else:
            try:
                row.update(self._probe(*batch, step=step))
            except Exception as e:  # noqa: BLE001 -- telemetry never stops training
                self.failures += 1
                row["probe_error"] = repr(e)[:300]
                if self.failures <= 5:
                    log.warning(f"signal probe failed at step {step} "
                                f"({self.failures} so far): {e!r}"[:400])
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        elapsed = time.perf_counter() - started
        self.probe_seconds += elapsed
        row["probe_ms"] = round(1000.0 * elapsed, 1)
        self._write(row)
        return row

    def summary(self, wall_seconds: float) -> str:
        """The epoch accounting's fields: rows, failures, share of the wall."""
        share = self.probe_seconds / wall_seconds if wall_seconds > 0 else 0.0
        return (f"signal_rows={self.rows} signal_failures={self.failures} "
                f"signal_wall={share:.1%}")

    def _probe(self, raws: Sequence, ais: Sequence, zw: Sequence, *, step: int) -> Dict:
        rng = random.Random(self._seed * 1_000_003 + step)
        pick = sorted(rng.sample(range(len(raws)), min(len(raws), self._probe_pairs)))
        batch = ([raws[i] for i in pick], [ais[i] for i in pick], [zw[i] for i in pick])
        on_cuda = self._params[0].is_cuda
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if on_cuda else []):
            parts, _ = self._batch_loss(*batch)
            losses = parts.source_losses()
            grads = {}
            for k, source in enumerate(IMITATION_SOURCES):
                loss = losses[source]
                if not loss.requires_grad:
                    grads[source] = [None] * len(self._params)
                    continue
                grads[source] = torch.autograd.grad(
                    loss / len(pick), self._params,
                    retain_graph=k + 1 < len(IMITATION_SOURCES), allow_unused=True)
            del parts, losses
        gram = self._grams(grads).cpu().tolist()
        return {"probe_pairs": len(pick), "groups": summarize_gram(gram),
                "gram": dict(zip(SIGNAL_GROUPS, gram))}

    def _grams(self, grads: Dict[str, Sequence]) -> torch.Tensor:
        """[groups, terms, terms]: the terms' gradient inner products
        over each group's parameters."""
        n = len(IMITATION_SOURCES)
        device = self._params[0].device
        gram = torch.zeros(len(SIGNAL_GROUPS), n, n, dtype=torch.float64, device=device)
        for j, (param, group) in enumerate(zip(self._params, self._group_of_param)):
            rows = [grads[source][j] for source in IMITATION_SOURCES]
            if all(r is None for r in rows):
                continue
            flat = torch.stack([torch.zeros(param.numel(), device=device) if r is None
                                else r.reshape(-1).float() for r in rows])
            gram[group] += (flat @ flat.T).double()
        return gram

    def _drain_step_norms(self) -> Dict:
        if not self.step_norms:
            return {"n": 0}
        norms = torch.stack([n.float() for n in self.step_norms]).cpu()
        self.step_norms.clear()
        finite = norms[torch.isfinite(norms)]
        out: Dict = {"n": int(norms.numel()), "nonfinite": int(norms.numel() - finite.numel())}
        if finite.numel():
            out.update(mean=float(finite.mean()), max=float(finite.max()),
                       clipped=float((finite > self._clip).float().mean()))
        return out

    def _write(self, row: Dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            self.rows += 1
        except OSError as e:
            self.failures += 1
            log.warning(f"signal row write failed ({self.path}): {e!r}")
