"""One gradient pass over a round's logged games
(docs/training_design_20261019/current_design.md §1 step 3;
build_contracts.md §3): the sum

    g = (1 / 2N) sum_games sum_sides R_{g,s} sum_{t in s} grad log pi(a_t | h_t)

at the checkpoint's weights theta_k, over the N training games of the label
rows (not held out), with no value, belief or imitation loss and no
optimizer step. log pi is the masked joint law the player sampled from
(wesnoth_ai/pg_loss.py), at every logged decision, refused ones included as
shadow rows (tools/pg_positions.py). The gradient flows through the
memory across windows of `--window` decisions, as the sequence trainer's
does, and the memory is carried into the next window detached.

Each split runs alone (tools/pg_schedule.py), so its windows' gradients
accumulate into its own sum. Per window w the gradient g_w of
sum_{rows} R log pi is kept as follows:

    total        sum over every window of g_w / 2N
    splits[i]    the same over split i's windows (sum(splits) == total)
    sq[i]        the mean over split i's windows of g_w ** 2, elementwise
    probes       d/de of the pass's sum at the three probe scalars
                 (pg_loss.probe_offsets), divided by 2N: probe P1

What pass.pt holds beyond the contract's keys:

    head_slopes    per head of pg_loss.HEADS: (1/2N) sum R * score
    head_fisher    4 x 4: (1/2N) sum over decisions of score score^T
                   (the head-temperature floor's slopes and Fisher, §3.4)
    replay         the trainer's log pi of each logged action against the
                   logged log p: `kl`, the mean of (logged - trainer), an
                   unbiased estimate of the per-decision KL from the logged
                   law to the trainer's (the actions were drawn from the
                   logged law); `half_mean_square`, its second-order
                   reading; the mean and largest absolute difference; the
                   `limit` it was held to
    shadow         shadow rows trained
    split_windows  windows per split
    position_counts  the replay's counts (pg_positions)
    seconds        the pass's wall time
    config         the pass's settings

The labels must name the network's own games: every labelled game-side was
played by the checkpoint (its match result's hash) at the pass's memory
slots under the temperature-1 procedure with no offset, or the pass refuses
to start (`pg_positions.refuse_foreign`). It fails after its sum when the
replay's KL estimate exceeds `REPLAY_KL_LIMIT`, and the command line then
writes its result as `<out>.replay_failed.pt` and exits 3.

The network runs in eval mode (no dropout: the law the player sampled), in
float32 unless `--bf16` (the device's bfloat16 autocast: its rounding,
about 0.016 nats on a log probability near -5, is a KL of the order of the
limit, so a bf16 pass is held to a limit of its own, `--replay-kl-limit`).

    python tools/pg_pass.py --checkpoint own2.pt --games DIR [DIR ...] --labels labels.jsonl \\
        --out pass.pt --device cuda --workers 30
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from tools.pg_positions import (SELF_PLAY_PROCEDURE, PositionLoader, RecordIndex, record_facts,  # noqa: E402
                                refuse_foreign, side_lengths)
from tools.pg_schedule import PGMemoryRouter, split_schedules  # noqa: E402
from tools.sequence_setup import ARCH, build_modules  # noqa: E402
from wesnoth_ai.param_groups import named_model_parameters  # noqa: E402
from wesnoth_ai.pg_loss import (CLASSES, HEADS, action_log_probs, head_scores, probe_offsets,  # noqa: E402
                                stage_actions)
from wesnoth_ai.pg_records import N_SPLITS, LabelRow, labelled_games  # noqa: E402

log = logging.getLogger("pg_pass")


def load_network(path: Path, device: torch.device):
    """(encoder, model) of a checkpoint, in eval mode, on `device`."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    arch = {k: int(ck["arch"][k]) for k in ARCH}
    encoder, model = build_modules(ck["unit_type_to_id"], ck["faction_to_id"], device, arch)
    model.load_state_dict(ck["model_state"])
    encoder.load_state_dict(ck["encoder_state"])
    return encoder.eval(), model.eval()


def head_shapes(model):
    """(action types, weapon slots): the policy heads' widths."""
    return model.type_head.out_features, model.weapon_head[-1].out_features


def stage_step(positions, staged, model, device):
    """The step's sampled actions under their masks (`pg_loss.stage_actions`)."""
    sizes = staged.sizes
    A_max = max(u for u, _, _ in sizes) + max(r for _, r, _ in sizes) + 1
    H_max = max(h for _, _, h in sizes)
    return stage_actions(list(range(len(positions))), [p.masks for p in positions], [p.action for p in positions],
                         A_max, H_max, *head_shapes(model), device)


# The replay identity's limit: a twentieth of the smallest radius the step
# search reads, eps = 0.0021 nats per decision (current_design.md §1.2, §5:
# "the mean KL is below eps/20"), on the unbiased per-decision KL estimate.
REPLAY_KL_LIMIT = 1e-4
EXIT_REPLAY = 3


class ReplayMismatch(RuntimeError):
    """The trainer's law at the logged decisions is not the logged law."""

    def __init__(self, result: Dict):
        super().__init__(f"the replay's KL estimate {result['replay']['kl']:.3g} nats per decision exceeds its "
                         f"limit {result['replay']['limit']:.3g} ({json.dumps(result['replay'])})")
        self.result = result


@dataclass
class PassConfig:
    streams: int = 32
    window: int = 16
    seed: int = 20261021
    slots: int = 64                         # the memory size the player played at
    workers: int = 0                        # replay processes (0: on the calling thread)
    prefetch: int = 64                      # game-sides replayed ahead of the slots
    bf16: bool = False                      # the device's bfloat16 autocast (CUDA only)
    packed: Optional[bool] = None           # None: the trunk the forward picks
    recompute: Optional[bool] = None        # recompute each step in the backward pass (default: on CUDA)
    verify: bool = True                     # check the records' fingerprints while replaying
    procedure: str = SELF_PLAY_PROCEDURE    # the procedure every labelled game-side was played under
    replay_kl_limit: float = REPLAY_KL_LIMIT


class GradientPass:
    """The pass of module docstring over the training games of `labels`."""

    def __init__(self, checkpoint_path: Path, records: Dict[str, Path], labels: Sequence[LabelRow],
                 device: torch.device, config: PassConfig = PassConfig()):
        from tools.eval_provenance import file_sha256
        self.config, self.device = config, device
        self.checkpoint_sha256 = file_sha256(checkpoint_path)
        self.encoder, self.model = load_network(checkpoint_path, device)
        self.named = named_model_parameters(self.model, self.encoder)
        self.games = labelled_games(labels, heldout=False)
        if not self.games:
            raise ValueError("no training game in the labels")
        self.R = {(g, s): row.R for g, sides in self.games.items() for s, row in sides.items()}
        paths = {g: records[g] for g in self.games}
        facts = record_facts(paths, config.workers)
        refuse_foreign(facts, self.checkpoint_sha256, config.slots, config.procedure)
        self.lengths = side_lengths(facts)
        self.loader = PositionLoader(paths, dict(self.encoder.unit_type_to_id), dict(self.encoder.faction_to_id),
                                     workers=config.workers, verify=config.verify)
        splits = {g: sides[1].split for g, sides in self.games.items()}
        self.split_counts = Counter(splits.values())
        self.schedules = split_schedules(splits, self.lengths, config.streams, config.seed, config.slots)
        self.autocast = torch.bfloat16 if (device.type == "cuda" and config.bf16) else None
        self.recompute = device.type == "cuda" if config.recompute is None else bool(config.recompute)
        self.probes = torch.zeros(len(CLASSES), device=device, requires_grad=True)
        self._reset_sums()

    def _reset_sums(self) -> None:
        zeros = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in self.named}
        self.splits = [{n: z.clone() for n, z in zeros.items()} for _ in range(N_SPLITS)]
        self.sq = [{n: z.clone() for n, z in zeros.items()} for _ in range(N_SPLITS)]
        self.split_windows = [0] * N_SPLITS
        self.head_sums = torch.zeros(len(HEADS), dtype=torch.float64, device=self.device)
        self.head_fisher = torch.zeros(len(HEADS), len(HEADS), dtype=torch.float64, device=self.device)
        # The replay's sums of d = logged - trainer log p: |d|, max |d|, d, d^2.
        self.replay = torch.zeros(4, dtype=torch.float64, device=self.device)
        self.rows = 0
        self.shadow = 0
        self.memories: Dict[int, torch.Tensor] = {}

    # ---- one time step --------------------------------------------------
    def _step(self, staged, mems, batch, w, logged):
        """The step's signed loss, its new memories and its readings."""
        with torch.autocast(self.device.type, dtype=self.autocast, enabled=self.autocast is not None):
            streams = self.encoder.embed_staged(staged)
            out = self.model.forward_embedded(streams, memory=mems, packed=self.config.packed)
        out = out.float32()
        lp = action_log_probs(probe_offsets(out, batch, self.probes), batch)
        with torch.no_grad():
            scores = head_scores(out, batch).double()
            diff = logged - lp.detach().double()
        return -(w * lp).sum(), out.memory_padded, scores, diff

    def _forward_window(self, window) -> tuple:
        """The window's loss and the memories it carries out."""
        router = PGMemoryRouter(self.model, self.device, self.memories)
        total = torch.zeros((), device=self.device)
        for steps in window:
            positions = [self.loader.side(s.game_side.file, s.game_side.side)[s.offset] for s in steps]
            mems = router.batch(steps)
            staged = self.encoder.stage_raws([p.raw for p in positions], device=self.device)
            batch = stage_step(positions, staged, self.model, self.device)
            w = torch.tensor([self.R[(s.game_side.file, s.game_side.side)] for s in steps], device=self.device)
            logged = torch.tensor([p.logp for p in positions], dtype=torch.float64, device=self.device)
            args = (staged, mems, batch, w, logged)
            loss, mem_padded, scores, diff = (checkpoint(self._step, *args, use_reentrant=False) if self.recompute
                                              else self._step(*args))
            total = total + loss
            router.wrote(steps, mem_padded, [p.shadow for p in positions])
            self.head_sums += (w.double().unsqueeze(1) * scores).sum(dim=0)
            self.head_fisher += scores.T @ scores
            self.replay[0] += diff.abs().sum()
            self.replay[1] = torch.maximum(self.replay[1], diff.abs().max())
            self.replay[2] += diff.sum()
            self.replay[3] += (diff * diff).sum()
            self.rows += len(steps)
            self.shadow += sum(p.shadow for p in positions)
        return total, router.memories()

    # ---- the pass -------------------------------------------------------
    def _run_split(self, i: int, schedule) -> None:
        self.memories: Dict[int, torch.Tensor] = {}
        while not schedule.exhausted():
            window = schedule.window(self.config.window)
            if not window:
                break
            self.loader.prefetch(g.file for g in schedule.upcoming(self.config.prefetch))
            loss, memories = self._forward_window(window)
            loss.backward()
            for n, p in self.named:
                if p.grad is not None:
                    g = -p.grad.float()                   # the ascent direction: grad of sum R log pi
                    self.splits[i][n] += g
                    self.sq[i][n] += g * g
                    p.grad = None
            self.memories = {s: m.detach() for s, m in memories.items()}
            self.split_windows[i] += 1
            self.loader.keep_only(g.file for g in schedule.upcoming(self.config.prefetch))
        self.loader.keep_only(())

    def run(self) -> Dict:
        """The pass; the pass.pt dict (module docstring). Raises
        `ReplayMismatch`, which carries the dict, when the replay's KL
        estimate exceeds the limit."""
        t0 = time.time()
        self._reset_sums()
        self.probes.grad = None
        for _, p in self.named:
            p.grad = None
        for i, schedule in enumerate(self.schedules):
            log.info("split %d: %d game-sides, %d decisions", i, len(schedule.order), schedule.total_positions)
            self._run_split(i, schedule)
        self.loader.close()
        expected = sum(self.lengths.values())
        if self.rows != expected:
            raise RuntimeError(f"the pass trained {self.rows} decisions of the logs' {expected} (I-count)")
        result = self._result(time.time() - t0)
        if not result["replay"]["kl"] <= result["replay"]["limit"]:
            raise ReplayMismatch(result)
        return result

    def _result(self, seconds: float) -> Dict:
        N = len(self.games)
        scale = 1.0 / (2.0 * N)
        splits = [{n: (t * scale).cpu() for n, t in s.items()} for s in self.splits]
        total = {n: sum(s[n] for s in splits) for n, _ in self.named}
        sq = [{n: (t / max(1, k)).cpu() for n, t in s.items()} for s, k in zip(self.sq, self.split_windows)]
        probe_grad = torch.zeros(len(CLASSES)) if self.probes.grad is None else -self.probes.grad.detach().cpu()
        rows = max(1, self.rows)
        return {
            "names": [n for n, _ in self.named], "total": total, "splits": splits, "sq": sq,
            "windows": int(sum(self.split_windows)), "games": N, "decisions": int(self.rows),
            "split_games": [self.split_counts.get(i, 0) for i in range(N_SPLITS)],
            "probes": {c: float(probe_grad[k]) * scale for k, c in enumerate(CLASSES)},
            "checkpoint_sha256": self.checkpoint_sha256,
            "head_slopes": {h: float(self.head_sums[k]) * scale for k, h in enumerate(HEADS)},
            "head_fisher": (self.head_fisher * scale).cpu().tolist(),
            "replay": {"kl": float(self.replay[2]) / rows, "half_mean_square": 0.5 * float(self.replay[3]) / rows,
                       "mean_abs": float(self.replay[0]) / rows, "max_abs": float(self.replay[1]),
                       "limit": float(self.config.replay_kl_limit)},
            "shadow": int(self.shadow), "split_windows": list(self.split_windows),
            "position_counts": dict(self.loader.counts), "seconds": float(seconds),
            "config": asdict(self.config),
        }


def save_pass(result: Dict, out: Path) -> None:
    """Atomic: a temporary file, then a rename."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(result, tmp)
    os.replace(tmp, out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True, help="theta_k, the network that played the games")
    ap.add_argument("--games", type=Path, nargs="+", required=True, help="directories of the game records")
    ap.add_argument("--labels", type=Path, required=True, help="labels.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    defaults = PassConfig()
    for name in ("streams", "window", "seed", "slots", "workers", "prefetch"):
        ap.add_argument("--" + name, type=int, default=getattr(defaults, name))
    ap.add_argument("--bf16", action="store_true", help="the device's bfloat16 autocast (module docstring)")
    ap.add_argument("--replay-kl-limit", type=float, default=REPLAY_KL_LIMIT)
    ap.add_argument("--procedure", default=SELF_PLAY_PROCEDURE, help="the labelled game-sides' procedure tag")
    ap.add_argument("--no-verify", action="store_true", help="skip the records' fingerprints")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from wesnoth_ai.pg_records import read_labels
    labels = read_labels(args.labels)
    index = RecordIndex(args.games)
    records = {r.game: index.path(r.game) for r in labels if not r.heldout}
    config = PassConfig(streams=args.streams, window=args.window, seed=args.seed, slots=args.slots,
                        workers=args.workers, prefetch=args.prefetch, bf16=args.bf16, verify=not args.no_verify,
                        procedure=args.procedure, replay_kl_limit=args.replay_kl_limit)
    try:
        result = GradientPass(args.checkpoint, records, labels, torch.device(args.device), config).run()
    except ReplayMismatch as e:
        failed = args.out.with_name(args.out.stem + ".replay_failed.pt")
        save_pass(e.result, failed)
        log.error("PG_PASS_REPLAY_FAILED %s; the sums are in %s", e, failed)
        return EXIT_REPLAY
    save_pass(result, args.out)
    log.info("PG_PASS_DONE %d games, %d decisions, %d windows in %.0f s; replay %s; probes %s", result["games"],
             result["decisions"], result["windows"], result["seconds"], json.dumps(result["replay"]),
             json.dumps(result["probes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
