"""The memory player's own game-sides in the sequence trainer
(docs/memory_own_trajectories_prereg_20261011.md; the records come from
tools/own_trajectories.py and tools/own_targets.py).

Own game-sides run on slots of their own beside the corpus's: of the
trainer's `--streams` slots, `--own-streams` walk the own records and the
rest the corpus, so every time step holds that many own positions and the
rest human ones. An own game-side runs at the memory size its player used
(the records' `slots`), its memory carried along its decisions as a corpus
game-side's is. Its policy loss is the soft-target cross-entropy of
`wesnoth_ai.own_loss` at `--own-policy-weight`, in place of the hard label's;
its value loss reads its game's outcome (none for a game without a winner)
and its belief loss its true state, as for the corpus. Own games marked
`holdout` are left out of training (the agreement barrier reads them,
tools/own_barriers.py).

The own records must be encoded as the corpus's are: the same encoding,
observation epoch, core phase and vocabularies, or the trainer refuses
them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from tools.preencode_sequences import load_manifest
from wesnoth_ai.own_loss import OwnBatch, stage_own_batch
from wesnoth_ai.sequence_loss import GameFacts
from wesnoth_ai.sequence_streams import GameSide, Step, StreamSchedule, epoch_order

# What the own records must share with the corpus's to train beside them.
SHARED_ENCODING_KEYS = ("encoding", "observation_epoch", "core_phase", "unit_type_to_id", "faction_to_id")


class OwnStreams:
    """The own records' facts, train and holdout game-sides, and their
    schedule on the trainer's slots `slot_offset` to `slot_offset + n_streams - 1`."""

    def __init__(self, directory: Path, human_manifest: Dict, seed: int, n_streams: int, slot_offset: int):
        man = load_manifest(directory)
        if man is None or man.get("source") != "own":
            raise SystemExit(f"{directory}: no own-trajectory manifest (tools/own_targets.py)")
        for key in SHARED_ENCODING_KEYS:
            if man.get(key) != human_manifest.get(key):
                raise SystemExit(f"{directory} was encoded with another {key} than the corpus's sequences: "
                                 f"they cannot train side by side")
        self.directory = Path(directory)
        self.manifest = man
        self.slots = int(man["slots"])
        rows = man["games"]
        self.games: Dict[str, GameFacts] = {
            f: GameFacts(winner=int(r["winner"]), n_commands=int(r["n_commands"]), policy_weight=0.0)
            for f, r in rows.items()}
        train = sorted(f for f, r in rows.items() if not r["holdout"])
        lengths = {GameSide(f, s): int(rows[f].get(f"positions_side{s}", 0)) for f in train for s in (1, 2)}
        self.holdout: List[GameSide] = [GameSide(f, s) for f in sorted(rows) if rows[f]["holdout"]
                                        for s in (1, 2) if int(rows[f].get(f"positions_side{s}", 0)) > 0]
        self.schedule = StreamSchedule(epoch_order(list(lengths), seed), lengths, n_streams, seed,
                                       k=self.slots, slot_offset=slot_offset, name="own")
        self.files = frozenset(rows)

    def meta(self, policy_weight: float) -> Dict:
        """What a checkpoint records of the own streams (a resume must match it)."""
        return {"sequences": str(self.directory), "fingerprint": self.manifest["fingerprint"],
                "teacher_sha256": self.manifest.get("teacher_sha256"), "streams": self.schedule.n_streams,
                "slots": self.slots, "policy_weight": float(policy_weight)}

    def batch(self, steps: Sequence[Step], positions: Sequence, sizes, n_types: int, n_weapons: int,
              device: torch.device) -> Optional[OwnBatch]:
        """The own rows of one time step's batch staged for the soft-target
        loss; None when the step holds none. `sizes` are the staged batch's
        (U, R, H) per row, which set the padded output's widths."""
        rows = [b for b, s in enumerate(steps) if s.game_side.file in self.files]
        if not rows:
            return None
        A_max = max(u for u, _, _ in sizes) + max(r for _, r, _ in sizes) + 1
        H_max = max(h for _, _, h in sizes)
        return stage_own_batch(rows, [positions[b].masks for b in rows], [positions[b].own_target for b in rows],
                               A_max, H_max, n_types, n_weapons, device)
