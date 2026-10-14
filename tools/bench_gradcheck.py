"""Candidate trainer configurations' gradients against one reference's
(tools/bench_sequence_step.py gradcheck; docs/trainer_speed_20261013.md).

    python tools/bench_sequence_step.py gradcheck --out check.json [--windows 3] [--reference=FLAGS]
        [--candidate=FLAGS ...] -- TRAIN_ARGS

Every trainer is built from the training arguments plus its flags, every
dropout at 0, all from the same weights at a learning rate of 0 (AdamW then
leaves the weights as they are). Window by window the reference trains
first, then each candidate: per window and candidate the loss and, per
parameter group, the cosine of its gradient with the reference's. A
candidate passes when every cosine is at least 0.99 and every loss within
0.5% of the reference's.

Each candidate is checked on its own: one that fails to build, raises in a
window (a compile error, out of memory, a refusal) or runs out of positions
is recorded as failed with its reason and its windows so far, and the others
go on. The reference failing fails the command (nothing to compare against).
The command exits 0 when every candidate passes, else 8; its JSON gives each
candidate's verdict: a crash barrier before a pass on the candidate, not a
verdict on it.
"""
from __future__ import annotations

import gc
import json
import logging
import shlex
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from tools.bench_sequence_step import build, quiet, set_dropout, use_repo

log = logging.getLogger(__name__)

NOT_PASSED = 8
GRADCHECK_COSINE = 0.99
GRADCHECK_LOSS = 0.005
DEFAULT_CANDIDATE = "--packed-trunk"


def _group_gradients(trainer) -> Dict[str, torch.Tensor]:
    from wesnoth_ai.param_groups import group_of, named_as
    flat: Dict[str, List[torch.Tensor]] = {}
    for name, p in named_as(trainer.params, trainer.model, trainer.encoder):
        if p.grad is not None:
            flat.setdefault(group_of(name), []).append(p.grad.detach().float().reshape(-1))
    return {g: torch.cat(ts) for g, ts in flat.items()}


def gradient_cosine(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    """The cosine of two gradients; None when both are zero (a head no
    label of the window trains), 0 when only one is."""
    na, nb = float(a.norm()), float(b.norm())
    if na == 0.0 and nb == 0.0:
        return None
    if na == 0.0 or nb == 0.0:
        return 0.0
    return round(float(torch.dot(a.double(), b.double()) / (na * nb)), 6)


def _rate_zero(t) -> None:
    """No probe, rate 0 (AdamW then leaves every weight as it is), no dropout."""
    quiet(t)
    t._lr_now = lambda: 0.0
    set_dropout(t, 0.0)


def _capture_gradients(t, slot: int, captured: List[Dict[str, torch.Tensor]]) -> None:
    """Each window's gradients by group into captured[slot], read as the
    optimizer zeroes them."""
    zero_grad = t.opt.zero_grad

    def hooked(*a, **kw):
        captured[slot] = _group_gradients(t)
        return zero_grad(*a, **kw)
    t.opt.zero_grad = hooked


def compare_window(ra: Dict, rb: Dict, ga: Dict[str, torch.Tensor], gb: Dict[str, torch.Tensor], i: int) -> Dict:
    """One window's reading of a candidate against the reference."""
    if set(ga) != set(gb):
        raise SystemExit(f"the trunks trained other groups: {sorted(set(ga) ^ set(gb))}")
    cos = {g: gradient_cosine(ga[g], gb[g]) for g in sorted(ga)}
    rel = abs(rb["loss"] - ra["loss"]) / max(abs(ra["loss"]), 1e-12)
    good = rel <= GRADCHECK_LOSS and all(c is None or c >= GRADCHECK_COSINE for c in cos.values())
    return {"i": i, "loss_reference": ra["loss"], "loss_candidate": rb["loss"], "loss_rel": round(rel, 6),
            "cosine": cos, "pass": good}


def failure(e: BaseException) -> str:
    """An exception as one line: its type and its message's first line."""
    lines = str(e).strip().splitlines()
    return f"{type(e).__name__}: {lines[0] if lines else ''}"[:500]


def _release() -> None:
    """The memory of a dropped trainer back to the device."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _candidate(train_argv: Sequence[str], flags: str, n: int, alone: bool):
    """Candidate n's trainer at rate 0, writing its own steps file."""
    _, _, t = build(list(train_argv) + shlex.split(flags))
    _rate_zero(t)
    suffix = "_candidate" if alone else f"_candidate{n}"
    t.args.out = t.args.out.with_name(t.args.out.stem + suffix + ".pt")
    return t


def _window(t, i: int) -> Dict:
    result = t.train_window()
    if not result:
        raise SystemExit(f"the pass ran out after {i} windows")
    return result


def gradcheck(opts, train_argv) -> int:
    """Every candidate against the one reference, window by window, from the
    same weights at rate 0, each candidate on its own (module docstring)."""
    use_repo(None)
    candidates = opts.candidate or [DEFAULT_CANDIDATE]
    _, _, reference = build(list(train_argv) + shlex.split(opts.reference))
    _rate_zero(reference)
    captured: List[Dict[str, torch.Tensor]] = [{} for _ in range(len(candidates) + 1)]
    _capture_gradients(reference, 0, captured)
    trainers, errors = {}, {}
    for n, flags in enumerate(candidates):
        try:
            trainers[n] = _candidate(train_argv, flags, n, len(candidates) == 1)
            _capture_gradients(trainers[n], n + 1, captured)
        except (Exception, SystemExit) as e:        # one candidate's failure leaves the others checked
            errors[n] = "build: " + failure(e)
            log.exception("candidate %d (%s) failed to build", n, flags)
            _release()
    rows: List[List[Dict]] = [[] for _ in candidates]
    for i in range(opts.windows):
        ra = _window(reference, i)
        for n in sorted(trainers):
            try:
                rb = _window(trainers[n], i)
                rows[n].append(compare_window(ra, rb, captured[0], captured[n + 1], i))
            except (Exception, SystemExit) as e:    # one candidate's failure leaves the others checked
                errors[n] = f"window {i}: " + failure(e)
                log.exception("candidate %d (%s) failed in window %d", n, candidates[n], i)
                del trainers[n]
                _release()
    results = []
    for n, flags in enumerate(candidates):
        entry = {"candidate": flags, "pass": n not in errors and all(w["pass"] for w in rows[n]),
                 "windows": rows[n]}
        if n in errors:
            entry["error"] = errors[n]
        results.append(entry)
    ok = all(r["pass"] for r in results)
    result = {"pass": ok, "reference": opts.reference, "min_cosine": GRADCHECK_COSINE,
              "max_loss_rel": GRADCHECK_LOSS, "candidates": results}
    text = json.dumps(result, indent=1)
    print(text)
    Path(opts.out).write_text(text + "\n", encoding="utf-8")
    return 0 if ok else NOT_PASSED
