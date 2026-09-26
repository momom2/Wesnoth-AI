"""The self-play learner's signal telemetry (tools/az_loop.py), always on.

After each iteration's step, a probe of up to AZ_PROBE_STATES of the
iteration's kept experiences at the weights the step left: the loss the
trainer optimizes split by term (`Trainer.mcts_loss_terms`: the four
policy heads, the value loss and the optional terms at their
coefficients), over the encoder, the trunk and the heads, in gradient
space and in the optimizer's update space (`GradientProbe`,
tools/signal_telemetry.py). The imitation trainer's rows
(`ImitationSignal`) read the same way.

One JSONL row per iteration in <workdir>/az_signal.jsonl, read back with
`read_signal_rows(path, progress="decision_step")`, and AZ_SIGNAL_COLUMNS
in the history row. Rows, by "kind":
  - "start": written when a run (or a resumed run) starts: the terms,
    the groups, the probe size, the loss settings, and the decision step
    of the weights it starts from.
  - "probe": `iter` and `decision_step`; `composition`, what the probe
    states can train (their games, the states with visit counts at a
    policy weight above zero, the states with a value weight above
    zero); `gradient` and `update` with their `*_gram` matrices and
    `update_stateless` (`probe_readings`); `probe_ms`, its cost;
    `failures`, the probes that failed so far in this run, and
    `probe_error` when this one did.

The probe changes nothing training reads: it draws its states with its
own generator (seeded by the run's seed and the iteration), forks
torch's generators around its forward, takes gradients with
`torch.autograd.grad` (no `.grad`, no optimizer state), and leaves the
model and the encoder in the modes it found them in
(tests/test_az_signal.py). The states it probes were encoded by the
step or by the held-out loss before it, so their names are already in
the encoder's vocabulary. Any failure warns, is counted in the row and
the history, and training goes on.

It runs every iteration: one forward, and one backward per term the
batch exercises, over AZ_PROBE_STATES states, where the step it follows
runs a forward and a backward over all of the iteration's training
experiences; each row records its own cost.
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from tools.signal_telemetry import (
    SIGNAL_GROUPS, GradientProbe, add_gradients, named_model_parameters, probe_readings,
    signal_group, write_signal_row,
)
from wesnoth_ai.trainer import MCTS_LOSS_TERMS, MCTS_POLICY_TERMS

log = logging.getLogger("az_signal")

# The size of the gradient-norm telemetry's subsample
# (signal_telemetry.SIG_SUBSAMPLE), so the two readings carry the same
# sampling noise.
AZ_PROBE_STATES = 128
AZ_SIGNAL_FILE = "az_signal.jsonl"

# The history row's readings of the probe.
AZ_SIGNAL_COLUMNS = (
    # Cosine between the policy heads' summed gradient and the value
    # term's, over the trunk's parameters; then the same in update space.
    "sig_trunk_policy_value_cos_gradient",
    "sig_trunk_policy_value_cos_update",
    # The value term's share of the trunk's update, <u_value, u> / |u|^2.
    "sig_trunk_value_share_update",
    # States the probe ran on (empty when it did not run), and the
    # probes that failed so far in this run.
    "sig_probe_states",
    "sig_probe_failures",
)

# The trainer settings the start row records.
_LOSS_SETTINGS = ("value_loss_form", "value_coef", "aux_coef", "moves_left_coef", "gbc_coef",
                  "trust_lambda", "value_label_smoothing", "grad_clip", "learning_rate",
                  "train_batch_size", "train_autocast_bf16")


class SelfPlaySignal:
    """The az loop's probe rows (module docstring) for one trainer."""

    def __init__(self, path: Path, trainer, *, start_iter: int = 0, decision_step: int = 0,
                 probe_states: int = AZ_PROBE_STATES, seed: int = 0):
        if probe_states < 1:
            raise ValueError(f"the signal probe needs probe_states >= 1, got {probe_states}")
        self.path = path
        self._trainer = trainer
        self._probe_states = probe_states
        self._seed = seed
        self.rows = 0
        self.probes = 0
        self.failures = 0
        self.probe_seconds = 0.0
        config = trainer.config
        self._write({"kind": "start", "iter": start_iter, "decision_step": decision_step,
                     "ts": time.strftime("%FT%T"), "terms": list(MCTS_LOSS_TERMS),
                     "policy_terms": list(MCTS_POLICY_TERMS), "groups": list(SIGNAL_GROUPS),
                     "probe_states": probe_states, "seed": seed,
                     "loss": {name: getattr(config, name) for name in _LOSS_SETTINGS},
                     "update": "gradient x lr / (sqrt(v_hat) + eps) of the optimizer's "
                               "second moment"})

    def record(self, experiences: Sequence, *, it: int, decision_step: int) -> Dict:
        """Probe the current weights on up to `probe_states` of
        `experiences`, write the row, and return AZ_SIGNAL_COLUMNS for
        the history row."""
        started = time.perf_counter()
        row: Dict = {"kind": "probe", "iter": it, "decision_step": decision_step,
                     "ts": time.strftime("%FT%T")}
        try:
            if experiences:
                row.update(self._probe(experiences, it))
                self.probes += 1
            else:
                row["probe"] = "none: the iteration kept no experiences"
        except Exception as e:  # noqa: BLE001 -- telemetry never stops training
            self.failures += 1
            row["probe_error"] = repr(e)[:300]
            log.warning(f"signal probe failed at iteration {it} "
                        f"({self.failures} so far): {e!r}"[:400])
        elapsed = time.perf_counter() - started
        self.probe_seconds += elapsed
        row["probe_ms"] = round(1000.0 * elapsed, 1)
        row["failures"] = self.failures
        self._write(row)
        return self._history_columns(row)

    def _probe(self, experiences: Sequence, it: int) -> Dict:
        n = min(len(experiences), self._probe_states)
        pick = random.Random(self._seed * 1_000_003 + it).sample(range(len(experiences)), n)
        sample = [experiences[i] for i in sorted(pick)]
        trainer = self._trainer
        probe = GradientProbe(named_model_parameters(trainer.model, trainer.encoder),
                              signal_group, SIGNAL_GROUPS, trainer.optimizer)
        summed: Optional[Dict[str, List[Optional[torch.Tensor]]]] = None

        def take_gradients(terms: Dict[str, torch.Tensor]) -> None:
            nonlocal summed
            summed = add_gradients(summed, probe.gradients(terms, 1.0))

        with probe.fork_rng(), torch.enable_grad():
            trainer.mcts_loss_terms(sample, take_gradients)
        gradient, update, stateless = probe.grams_of(summed)
        return {"probe_states": n, "composition": _composition(sample),
                **probe_readings(gradient, update, stateless,
                                 terms=MCTS_LOSS_TERMS, policy_terms=MCTS_POLICY_TERMS)}

    def _history_columns(self, row: Dict) -> Dict:
        gradient = row.get("gradient", {}).get("trunk", {})
        update = row.get("update", {}).get("trunk", {})
        return {"sig_trunk_policy_value_cos_gradient": gradient.get("policy_value_cos"),
                "sig_trunk_policy_value_cos_update": update.get("policy_value_cos"),
                "sig_trunk_value_share_update": update.get("value", {}).get("share"),
                "sig_probe_states": row.get("probe_states"),
                "sig_probe_failures": self.failures}

    def _write(self, row: Dict) -> None:
        if write_signal_row(self.path, row):
            self.rows += 1
        else:
            self.failures += 1


def _composition(sample: Sequence) -> Dict[str, int]:
    """What the probe's states can train: the games behind them, the
    states with visit counts at a policy weight above zero, and the
    states with a value weight above zero."""
    return {"games": len({getattr(e, "game_id", "") for e in sample}),
            "policy_states": sum(1 for e in sample
                                 if e.visit_counts and getattr(e, "policy_weight", 1.0) > 0),
            "value_states": sum(1 for e in sample if getattr(e, "value_weight", 1.0) > 0)}
