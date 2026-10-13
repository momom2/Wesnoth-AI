"""The sequence trainer's always-on telemetry (tools/sequence_train.py,
user ruling 2026-09-01): the gradient split by loss term over the parameter
groups (<out>.signal.jsonl, every `--signal-every` positions), a row per
step with its timing (<out>.steps.jsonl) and the log line."""
from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Optional

import torch

from tools.signal_telemetry import (POLICY_SOURCES, GradientProbe, named_model_parameters, signal_group,
                                    summarize_gram)
from wesnoth_ai.imitation_loss import imitation_loss_parts
from wesnoth_ai.own_loss import own_policy_loss
from wesnoth_ai.param_groups import MEMORY_GROUPS
from wesnoth_ai.sequence_loss import belief_loss

log = logging.getLogger("sequence_train")

# The telemetry's slots, terms and parameter groups.
SIGNAL_STREAMS = 4
SIGNAL_TERMS = POLICY_SOURCES + ("value", "belief")
SIGNAL_GROUPS = ("encoder", "trunk", "heads", "memory")


def _squares(norms: Dict[str, float]) -> Dict[str, float]:
    """A step's squared gradient norm and the memory's part of it,
    summable over the steps a log line covers."""
    return {"grad_sq": sum(v * v for v in norms.values()),
            "grad_memory_sq": sum(norms.get(g, 0.0) ** 2 for g in MEMORY_GROUPS)}


def sequence_signal_group(name: str) -> str:
    """The telemetry group of a namespaced parameter: the learned memory
    (its initial state, slot embedding and write), else the encoder, the
    trunk or the heads."""
    return "memory" if name.startswith("model.slot_memory.") else signal_group(name)


class TrainerTelemetry:
    """The trainer's telemetry (a part of `sequence_train.Trainer`)."""

    def _signal_probe(self) -> GradientProbe:
        """The telemetry over the optimizer's current param groups (built
        again whenever its state is loaded, which replaces them)."""
        return GradientProbe(named_model_parameters(self.model, self.encoder),
                             sequence_signal_group, SIGNAL_GROUPS, self.opt)

    def _step_row(self, lr: float, loss: float, norm: float, grads: Dict[str, float], applied: bool,
                  sums: Dict, timing: Optional[Dict] = None) -> None:
        """The step's row in <out>.steps.jsonl (module docstring), with the
        window's timing (tools/sequence_timing.py)."""
        row = {"step": self.state["steps"], "positions": self.state["positions"], "lr": lr, "loss": loss,
               "policy": sums["policy"] / max(1, sums["policy_n"]),
               "value": sums["value"] / max(1, sums["value_n"]),
               "belief": sums["belief"] / max(1, sums["belief_n"]),
               "grad_norm": norm, "grad": grads, "applied": applied}
        if "own" in sums:
            row["own"] = sums["own"] / max(1, sums["own_n"])
        row.update(timing or {})
        with open(self.args.out.with_suffix(".steps.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def signal_row(self, window, start_memories: Dict[int, torch.Tensor]) -> None:
        """The telemetry row: the window just trained, on its first
        SIGNAL_STREAMS slots from the memories they started it with, its loss
        split by term in gradient and update space (the optimizer's state
        read, nothing written, the training's random stream untouched)."""
        t0 = time.time()
        memories = dict(start_memories)
        terms = {t: torch.zeros((), device=self.device) for t in self.signal_terms}
        n = 0
        with self.signal.fork_rng():
            for steps in window:
                steps = [s for s in steps if s.slot in self.signal_slots]
                if not steps:
                    continue
                staged, mems, targets, b_target, b_mask = self._prepare(steps, memories)
                out = self._forward(staged, mems, self._trunk_index(steps, staged))
                for term, loss in imitation_loss_parts(out, targets).source_losses().items():
                    terms[term] = terms[term] + loss
                terms["belief"] = terms["belief"] + self.args.belief_weight * \
                    belief_loss(out.belief_logits, b_target, b_mask).sum()
                own = self._own_batch(steps, staged)
                if own is not None:
                    terms["own"] = terms["own"] + self.args.own_policy_weight * own_policy_loss(out, own).sum()
                for b, s in enumerate(steps):
                    memories[s.slot] = out.memory_padded[b, :s.k]
                n += len(steps)
            if not n:
                return
            gradient, update, stateless = self.signal.grams(terms, 1.0 / n)
        kw = dict(policy_terms=POLICY_SOURCES, shared_groups=("encoder", "trunk", "memory"))
        row = {"positions": self.state["positions"], "steps": self.state["steps"], "probe_positions": n,
               "gradient": summarize_gram(gradient.tolist(), self.signal_terms, SIGNAL_GROUPS, **kw),
               "update": None if update is None else summarize_gram(update.tolist(), self.signal_terms,
                                                                    SIGNAL_GROUPS, **kw),
               "stateless": stateless, "seconds": round(time.time() - t0, 2)}
        with open(self.args.out.with_suffix(".signal.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _log(self, rows: List[Dict], t0: float) -> None:
        rows = [r for r in rows if r]
        if not rows:
            return
        tot = {k: sum(r[k] for r in rows) for k in rows[0] if k not in ("loss", "grad_norm", "memory_write_grad_norm")}
        el = time.time() - t0
        log.info("positions %d/%d steps %d | loss %.4f grad %.2f memory write grad %.3f | memory share of "
                 "gradient %.3f | policy %.4f value %.4f belief %.4f | lr %.2e | %.1f positions/s",
                 self.state["positions"], self.pass_total, self.state["steps"],
                 sum(r["loss"] for r in rows) / len(rows),
                 max(r["grad_norm"] for r in rows), max(r["memory_write_grad_norm"] for r in rows),
                 tot["grad_memory_sq"] / max(tot["grad_sq"], 1e-30),
                 tot["policy"] / max(1, tot["policy_n"]), tot["value"] / max(1, tot["value_n"]),
                 tot["belief"] / max(1, tot["belief_n"]), self._lr_now(),
                 (self.state["positions"] - self.start_positions) / max(el, 1e-9))
        if "own" in tot:
            log.info("own positions: soft-target cross-entropy %.4f over %d", tot["own"] / max(1, tot["own_n"]),
                     tot["own_n"])
