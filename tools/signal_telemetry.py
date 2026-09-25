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
a row every IMITATION_SIGNAL_EVERY trained pairs in
<checkpoint stem>_signal.jsonl: a probe of the batch just trained
split by loss term over the encoder, the trunk and the heads, in
gradient space and in the optimizer's update space (`GradientProbe`,
which any trainer can use), and the real steps' gradient norms (its
docstring). Cost: two to three training steps per row, recorded in
each row.

The full update-space tree stays offline (signal_profiler)."""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy
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


# ---- Parameter groups -----------------------------------------------------------

# Predicates over `named_parameters()` names, the model's and the encoder's
# namespaced as "model." and "encoder."; a parameter no predicate claims is
# the trunk's. The offline profiler (signal_profiler/) groups the same way.
PARAM_GROUPS = (
    ("encoder",     lambda n: n.startswith("encoder.")),
    ("value_head",  lambda n: n.startswith(("model.value_head", "model.material_proj"))),
    ("actor_head",  lambda n: n.startswith("model.actor_head")),
    ("type_head",   lambda n: n.startswith("model.type_head")),
    ("target_proj", lambda n: n.startswith("model.target_")),
    ("weapon_head", lambda n: n.startswith("model.weapon_head")),
    ("gbc_heads",   lambda n: n.startswith("model.gbc_heads")),
    ("aux_ml",      lambda n: n.startswith(("model.aux_score_head", "model.moves_left"))),
)


def group_of(name: str) -> str:
    """The parameter group of a namespaced parameter name."""
    for group, claims in PARAM_GROUPS:
        if claims(name):
            return group
    return "trunk"


def named_model_parameters(model: torch.nn.Module,
                           encoder: Optional[torch.nn.Module]) -> List[Tuple[str, torch.nn.Parameter]]:
    """The trained parameters, namespaced as PARAM_GROUPS expects."""
    named = [("model." + n, p) for n, p in model.named_parameters()]
    if encoder is not None:
        named += [("encoder." + n, p) for n, p in encoder.named_parameters()]
    return [(n, p) for n, p in named if p.requires_grad]


# ---- A gradient probe any trainer can use ---------------------------------------

def summarize_gram(gram: Sequence[Sequence[Sequence[float]]], terms: Sequence[str],
                   groups: Sequence[str], *, policy_terms: Sequence[str] = (),
                   shared_groups: Sequence[str] = ()) -> Dict[str, Dict]:
    """Readings of per-group Gram matrices of loss terms' vectors:
    `gram[k][i][j]` is <u_i, u_j> over the parameters of groups[k].

    Per group, and for "all" (their sum): the norm of the summed vector
    and, per term, its norm, its signed share of the summed vector
    (<u_i, u> / |u|^2; the shares add up to 1, and a negative one pulls
    against the rest) and its cosine with it. In each of `shared_groups`
    (and "all"), the cosine between the sum of `policy_terms` and the
    "value" term, when both are present and non-zero."""
    n = len(terms)
    by_group = dict(zip(groups, gram))
    by_group["all"] = [[sum(g[i][j] for g in gram) for j in range(n)] for i in range(n)]
    policy = [terms.index(t) for t in policy_terms]
    value = terms.index("value") if "value" in terms else None
    out = {}
    for name, g in by_group.items():
        total_sq = sum(sum(row) for row in g)
        total_norm = math.sqrt(max(total_sq, 0.0))
        node: Dict = {"total_norm": total_norm}
        for i, term in enumerate(terms):
            norm = math.sqrt(max(g[i][i], 0.0))
            dot = sum(g[i])
            node[term] = {
                "norm": norm,
                "share": dot / total_sq if total_sq > 0 else None,
                "cos": dot / (norm * total_norm) if norm > 0 and total_norm > 0 else None}
        if policy and value is not None and (name == "all" or name in shared_groups):
            policy_sq = sum(g[a][b] for a in policy for b in policy)
            policy_value = sum(g[a][value] for a in policy)
            value_sq = g[value][value]
            node["policy_value_cos"] = (policy_value / math.sqrt(policy_sq * value_sq)
                                        if policy_sq > 0 and value_sq > 0 else None)
        out[name] = node
    return out


class GradientProbe:
    """The Gram matrices of loss terms' gradients over parameter groups,
    in two spaces:

      - gradient: each term's gradient as the loss defines it;
      - update: each term's gradient times lr / (sqrt(v_hat) + eps),
        coordinate by coordinate, with v_hat Adam's bias-corrected
        second moment (its running maximum under amsgrad) and lr the
        parameter's group rate: the step the optimizer takes for that
        gradient, momentum and weight decay aside
        (signal_profiler/update_tree.py, "exp_avg zeroed, exp_avg_sq
        kept"). A parameter with a gradient but no optimizer state yet
        (it has never been stepped) is left out of the update space and
        counted per group, so a partial group total is visible.

    The gradients come from `torch.autograd.grad`, which leaves every
    parameter's `.grad` and the optimizer untouched; the caller runs the
    forward inside `fork_rng()` so that dropout draws do not move the
    training's generators."""

    def __init__(self, named_params: Sequence[Tuple[str, torch.nn.Parameter]],
                 group_of_name: Callable[[str], str], groups: Sequence[str],
                 optimizer: Optional[torch.optim.Optimizer] = None):
        named = [(n, p) for n, p in named_params if p.requires_grad]
        self.groups = tuple(groups)
        self._params = [p for _, p in named]
        self._group_index = [self.groups.index(group_of_name(n)) for n, _ in named]
        self._optimizer = optimizer
        self._hyper = {}
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                for p in param_group["params"]:
                    self._hyper[id(p)] = param_group

    def fork_rng(self):
        """torch's generators forked for the probe: the CPU one and every
        CUDA device the parameters live on (not merely the current one)."""
        devices = sorted({p.device.index for p in self._params if p.is_cuda})
        return torch.random.fork_rng(devices=devices)

    def grams(self, losses: Dict[str, torch.Tensor], scale: float
              ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[int]]:
        """[groups, terms, terms] Gram matrices of the terms' gradients
        (each loss times `scale`), in gradient space and in update space
        (None before the optimizer has stepped), and per group the count
        of parameters with a gradient but no optimizer state."""
        terms = list(losses)
        grads = {}
        for k, term in enumerate(terms):
            loss = losses[term]
            if not loss.requires_grad:
                grads[term] = [None] * len(self._params)
                continue
            grads[term] = torch.autograd.grad(loss * scale, self._params,
                                              retain_graph=k + 1 < len(terms), allow_unused=True)
        n = len(terms)
        device = self._params[0].device
        gradient = torch.zeros(len(self.groups), n, n, dtype=torch.float64, device=device)
        update = torch.zeros_like(gradient)
        stateless = [0] * len(self.groups)
        stepped = False
        for j, (param, group) in enumerate(zip(self._params, self._group_index)):
            rows = [grads[term][j] for term in terms]
            if all(r is None for r in rows):
                continue
            flat = torch.stack([torch.zeros(param.numel(), device=device) if r is None
                                else r.reshape(-1).float() for r in rows])
            gradient[group] += (flat @ flat.T).double()
            step_per_gradient = self._update_scale(param)
            if step_per_gradient is None:
                if self._optimizer is not None:
                    stateless[group] += 1
                continue
            scaled = flat * step_per_gradient.reshape(-1)
            update[group] += (scaled @ scaled.T).double()
            stepped = True
        return gradient, (update if stepped else None), stateless

    def _update_scale(self, param: torch.nn.Parameter) -> Optional[torch.Tensor]:
        """lr / (sqrt(v_hat) + eps) per coordinate, None without state."""
        state = self._optimizer.state.get(param) if self._optimizer is not None else None
        if not state or "exp_avg_sq" not in state:
            return None
        hyper = self._hyper[id(param)]
        beta2, eps = hyper["betas"][1], hyper["eps"]
        second = state["max_exp_avg_sq"] if hyper.get("amsgrad") else state["exp_avg_sq"]
        v_hat = second.float() / (1.0 - beta2 ** float(state["step"]))
        return float(hyper["lr"]) / v_hat.sqrt().add_(eps)


class StepNorms:
    """The real optimizer steps' pre-clip gradient norms, left on the
    device until read: one transfer per read, none per step."""

    def __init__(self, clip: float):
        self.clip = clip
        self._norms: List[torch.Tensor] = []

    def append(self, norm) -> None:
        """A step's pre-clip norm: a tensor (left on its device) or a
        float (a trainer that already read it back)."""
        self._norms.append(norm.detach() if torch.is_tensor(norm) else torch.tensor(float(norm)))

    def __len__(self) -> int:
        return len(self._norms)

    def drain(self) -> Dict:
        """Count, mean, max and share above the clip since the last read;
        non-finite norms are counted apart."""
        if not self._norms:
            return {"n": 0}
        device = self._norms[0].device
        norms = torch.stack([n.float().to(device) for n in self._norms]).cpu()
        self._norms.clear()
        finite = norms[torch.isfinite(norms)]
        out: Dict = {"n": int(norms.numel()), "nonfinite": int(norms.numel() - finite.numel())}
        if finite.numel():
            out.update(mean=float(finite.mean()), max=float(finite.max()),
                       clipped=float((finite > self.clip).float().mean()))
        return out


# ---- The imitation trainer (tools/supervised_train.py) ----------------------

IMITATION_SOURCES = ("actor", "type", "target", "weapon", "value")
POLICY_SOURCES = ("actor", "type", "target", "weapon")
SIGNAL_GROUPS = ("encoder", "trunk", "heads")
# Trained pairs between rows: about 113 rows over obs8's 2.83M-pair pass.
# A probe costs about two to three training steps (one forward and five
# backwards over 32 pairs), so at batch 64 (390 steps per row) the
# telemetry takes about 0.5-0.7% of the pass; each row records its cost.
IMITATION_SIGNAL_EVERY = 25_000
IMITATION_PROBE_PAIRS = 32
MIN_PROBE_PAIRS = 4       # the smallest probe an out-of-memory retry goes down to


def signal_group(name: str) -> str:
    """The telemetry's coarse group of a namespaced parameter: encoder,
    trunk or heads."""
    group = group_of(name)
    return group if group in ("encoder", "trunk") else "heads"


class ImitationSignal:
    """The imitation trainer's signal telemetry, always on: JSONL rows in
    <checkpoint stem>_signal.jsonl.

    Rows, by "kind":
      - "start": written when a run (or a resumed run) starts: the terms,
        the groups, the cadence, the clip. Rows written earlier with a
        larger pair count belong to a run that was cut and resumed from
        an earlier checkpoint; `read_signal_rows` drops them.
      - "probe": written at the first trained batch after the trained-pair
        count crosses a multiple of `every`. `step` is the trainer's step
        counter, which does not count an epoch's residual flush. `steps`:
        the real optimizer steps since the previous row (count, pre-clip
        gradient norm mean and max, share above the clip, non-finite
        count). `gradient` and `update`: a probe of up to `probe_pairs`
        pairs of the batch just trained, at the weights that step
        produced, split by loss term (`ImitationLossParts.source_losses`,
        each divided by the probe's size as the trainer divides by the
        batch's) over the encoder, the trunk, the heads and all
        parameters (`summarize_gram`), in gradient space and in the
        optimizer's update space (`GradientProbe`); `*_gram` the matrices
        they come from, terms in the start row's order;
        `update_stateless` the groups whose update total leaves out
        never-stepped parameters; `fired`: how many probe pairs each
        term covers (a policy term only where the pair's policy weight is
        not zero). A probe that runs out of memory is retried at half its
        size, down to MIN_PROBE_PAIRS.
      - "steps": at an epoch's end, a cut, or a periodic checkpoint, the
        step norms since the previous row, so that a resume from that
        checkpoint loses none.

    The probe changes nothing the training reads: it picks its pairs with
    its own generator seeded by the pair count (a resumed run picks the
    same ones), forks torch's generators around its forward, and takes
    gradients with `torch.autograd.grad`. Any failure warns, counts into
    `failures` and the epoch accounting, and training goes on."""

    def __init__(self, path: Path, batch_loss: Callable, model: torch.nn.Module,
                 encoder: torch.nn.Module, *, optimizer: Optional[torch.optim.Optimizer] = None,
                 pairs: int = 0, last_row_pairs: Optional[int] = None,
                 every: int = IMITATION_SIGNAL_EVERY, probe_pairs: int = IMITATION_PROBE_PAIRS,
                 clip: float = 1.0, seed: int = 0):
        """`batch_loss(raws, ais, zw)` returns the trainer's
        (ImitationLossParts, ImitationTargets) for those pairs; `pairs` is
        the trained-pair count the run starts from and `last_row_pairs`
        the count at its last probe row (`state()`, saved with a
        checkpoint)."""
        check_signal_cadence(every, probe_pairs)
        self.path = path
        self._batch_loss = batch_loss
        self._probe = GradientProbe(named_model_parameters(model, encoder), signal_group,
                                    SIGNAL_GROUPS, optimizer)
        self.step_norms = StepNorms(clip)
        self._every = every
        self._probe_pairs = probe_pairs
        self._seed = seed
        self._last_row_pairs = pairs if last_row_pairs is None else last_row_pairs
        self.rows = 0
        self.probes = 0
        self.failures = 0
        self.probe_seconds = 0.0
        self._write({"kind": "start", "pairs": pairs, "ts": time.strftime("%FT%T"),
                     "terms": list(IMITATION_SOURCES), "groups": list(SIGNAL_GROUPS),
                     "every": every, "probe_pairs": probe_pairs, "clip": clip,
                     "update": "gradient / (sqrt(v_hat) + eps) of the optimizer's second moment"})

    def state(self) -> Dict[str, int]:
        """What a checkpoint keeps so that a resumed run writes its rows
        where the uncut run does."""
        return {"last_row_pairs": self._last_row_pairs}

    def due(self, pairs: int) -> bool:
        """Whether the trained-pair count has crossed a multiple of `every`
        since the last probe row."""
        return pairs // self._every > self._last_row_pairs // self._every

    def record(self, batch: Optional[tuple], *, epoch: int, step: int, pairs: int,
               max_probe_pairs: Optional[int] = None) -> Dict:
        """Write a probe row and return it. `batch` is the (raws, ais,
        zw) the last step trained on, or None where there is none (the
        per-pair flow keeps no batch; that row carries the step norms
        alone). `max_probe_pairs` caps the probe, e.g. at the chunk size
        the step itself had to split its batch into."""
        self._last_row_pairs = pairs
        self.probes += 1
        return self._row("probe", batch, epoch=epoch, step=step, pairs=pairs,
                         max_probe_pairs=max_probe_pairs)

    def close(self, *, epoch: int, step: int, pairs: int) -> Optional[Dict]:
        """The step norms since the last row, at an epoch's end, a cut or
        a periodic checkpoint; nothing when no step was taken since."""
        if not len(self.step_norms):
            return None
        return self._row("steps", None, epoch=epoch, step=step, pairs=pairs)

    def summary(self, wall_seconds: float) -> str:
        """The epoch accounting's fields: probe rows, failures, share of
        the wall."""
        share = self.probe_seconds / wall_seconds if wall_seconds > 0 else 0.0
        return (f"signal_probes={self.probes} signal_failures={self.failures} "
                f"signal_wall={share:.1%}")

    def _row(self, kind: str, batch: Optional[tuple], *, epoch: int, step: int, pairs: int,
             max_probe_pairs: Optional[int] = None) -> Dict:
        started = time.perf_counter()
        row: Dict = {"kind": kind, "epoch": epoch, "step": step, "pairs": pairs,
                     "ts": time.strftime("%FT%T")}
        try:
            row["steps"] = self.step_norms.drain()
            if kind == "probe":
                if batch is None:
                    row["probe"] = "none: the per-pair flow keeps no batch"
                else:
                    row.update(self._probe_batch(*batch, pairs=pairs, cap=max_probe_pairs))
        except Exception as e:  # noqa: BLE001 -- telemetry never stops training
            self.failures += 1
            row["probe_error"] = repr(e)[:300]
            if self.failures <= 5:
                log.warning(f"signal telemetry failed at step {step} "
                            f"({self.failures} so far): {e!r}"[:400])
        elapsed = time.perf_counter() - started
        self.probe_seconds += elapsed
        row["probe_ms"] = round(1000.0 * elapsed, 1)
        self._write(row)
        return row

    def _probe_batch(self, raws: Sequence, ais: Sequence, zw: Sequence, *, pairs: int,
                     cap: Optional[int]) -> Dict:
        order = random.Random(self._seed * 1_000_003 + pairs).sample(range(len(raws)), len(raws))
        n = min(len(raws), self._probe_pairs, cap or len(raws))
        while True:
            try:
                return self._probe_once(raws, ais, zw, sorted(order[:n]))
            except torch.cuda.OutOfMemoryError:
                if n <= MIN_PROBE_PAIRS:
                    raise
                n = max(MIN_PROBE_PAIRS, n // 2)
            # Past the except clause, the failed attempt's frames and graph
            # are gone, so the allocator can hand their memory to the retry.
            torch.cuda.empty_cache()

    def _probe_once(self, raws: Sequence, ais: Sequence, zw: Sequence, pick: List[int]) -> Dict:
        with self._probe.fork_rng():
            parts, targets = self._batch_loss([raws[i] for i in pick], [ais[i] for i in pick],
                                              [zw[i] for i in pick])
            gradient, update, stateless = self._probe.grams(parts.source_losses(), 1.0 / len(pick))
            del parts
        readings = dict(terms=IMITATION_SOURCES, groups=SIGNAL_GROUPS,
                        policy_terms=POLICY_SOURCES, shared_groups=("encoder", "trunk"))
        gradient_gram = gradient.cpu().tolist()
        weighted = numpy.asarray(targets.policy_w) > 0
        fired = {head: int(numpy.sum(numpy.asarray(flags) & (weighted if head in POLICY_SOURCES
                                                               else True)))
                 for head, flags in targets.ok.items()}
        row = {"probe_pairs": len(pick), "fired": fired,
               "gradient": summarize_gram(gradient_gram, **readings),
               "gradient_gram": dict(zip(SIGNAL_GROUPS, gradient_gram))}
        if update is not None:
            update_gram = update.cpu().tolist()
            row["update"] = summarize_gram(update_gram, **readings)
            row["update_gram"] = dict(zip(SIGNAL_GROUPS, update_gram))
            partial = {g: n for g, n in zip(SIGNAL_GROUPS, stateless) if n}
            if partial:
                row["update_stateless"] = partial
        return row

    def _write(self, row: Dict) -> None:
        if write_signal_row(self.path, row):
            self.rows += 1
        else:
            self.failures += 1


def write_signal_row(path: Path, row: Dict) -> bool:
    """Append one JSONL row; False (and a warning) when it cannot."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return True
    except (OSError, TypeError, ValueError) as e:
        log.warning(f"signal row write failed ({path}): {e!r}")
        return False


def check_signal_cadence(every: int, probe_pairs: int = IMITATION_PROBE_PAIRS) -> None:
    """Refuse a cadence the telemetry cannot run at (it is always on)."""
    if every < 1 or probe_pairs < 1:
        raise ValueError(f"the signal telemetry is always on: it needs every >= 1 and "
                         f"probe_pairs >= 1, got {every} and {probe_pairs}")


def read_signal_rows(path: Path) -> List[Dict]:
    """The rows of a signal file that describe the training as it went:
    after a "start" row at pair count P, earlier rows past P are dropped
    (their run was cut and resumed from an earlier checkpoint). A last
    line cut short (the trainer killed mid-write) is skipped."""
    rows: List[Dict] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for k, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if k == len(lines) - 1:
                break
            raise
        if row.get("kind") == "start":
            rows = [r for r in rows if r["pairs"] <= row["pairs"]]
        rows.append(row)
    return rows


# ---- A value head fitted on cached features (tools/value_head_fit.py) --------

def outcome_terms(loss_per_state: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
    """A value loss split by the outcome each state is labelled with:
    the won, lost and drawn states' summed losses, over all states
    (so the terms add up to the mean loss). Drawn only when present."""
    n = max(1, int(z.numel()))
    terms = {"won": loss_per_state[z > 0].sum() / n, "lost": loss_per_state[z < 0].sum() / n}
    if bool((z == 0).any()):
        terms["drawn"] = loss_per_state[z == 0].sum() / n
    return terms


def value_head_signal(head: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer],
                      loss_per_state: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                      feats: torch.Tensor, z: torch.Tensor) -> Dict:
    """The head's gradient on a probe of cached states split by outcome
    (`outcome_terms`), in gradient and update space: whether wins or
    losses drive the head's update, and whether they pull against each
    other (a negative share). Leaves training untouched: forked
    generators, gradients by `autograd.grad`."""
    probe = GradientProbe([("head." + n, p) for n, p in head.named_parameters()],
                          lambda _name: "head", ("head",), optimizer)
    with probe.fork_rng():
        terms = outcome_terms(loss_per_state(head(feats), z), z)
        gradient, update, stateless = probe.grams(terms, 1.0)
    names = list(terms)
    row = {"probe_states": int(z.numel()), "terms": names,
           "gradient": summarize_gram(gradient.cpu().tolist(), terms=names, groups=("head",))["head"],
           "gradient_gram": gradient.cpu().tolist()[0]}
    if update is not None:
        row["update"] = summarize_gram(update.cpu().tolist(), terms=names, groups=("head",))["head"]
        row["update_gram"] = update.cpu().tolist()[0]
    return row
