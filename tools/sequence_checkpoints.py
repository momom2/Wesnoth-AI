"""Where a sequence trainer's pass starts (tools/sequence_train.py): its own
checkpoint (`--resume`) or another pass's (`--init-from`)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import torch

from tools.lr_law import Areas
from wesnoth_ai.checkpoint_structure import checkpoint_structure
from wesnoth_ai.constants import OBSERVATION_EPOCH

log = logging.getLogger("sequence_train")

# What a resume must share with the pass it continues, and the value a key
# holds in a checkpoint written before it was recorded.
RESUME_KEYS = ("seed", "streams", "window", "fingerprint", "arch", "decay_from", "pass_positions",
               "anneal_rule", "own", "trunk")
RECORDED_LATER = {"trunk": "padded"}


class TrainerCheckpoints:
    """The trainer's starting points (a part of `sequence_train.Trainer`)."""

    def resume(self, path: Path) -> None:
        """Continue `path`'s pass. A new pass may start from another trunk's
        checkpoint (`init_from`); a resume continues on the trunk its pass
        trained on."""
        ck = torch.load(path, map_location="cpu", weights_only=True)
        meta = ck.get("training_meta", {})
        for key in RESUME_KEYS:
            saved = meta.get(key, RECORDED_LATER.get(key))
            if saved != self.meta[key]:
                raise SystemExit(f"{path} was trained with {key}={saved!r}; this run has "
                                 f"{self.meta[key]!r}")
        self.meta["init_from"] = meta.get("init_from")
        self.model.load_state_dict(ck["model_state"])
        self.encoder.load_state_dict(ck["encoder_state"])
        self.opt.load_state_dict(ck["optimizer_state"])
        self.signal = self._signal_probe()
        saved = ck["sequence_resume"]
        self.state.update(saved["state"])
        self.schedule.load_state_dict(saved["schedule"])
        if self.own is not None:
            self.own.schedule.load_state_dict(saved["own_schedule"])
        self.memories = {int(s): m.to(self.device) for s, m in saved["memories"].items()}
        torch.set_rng_state(saved["rng"]["cpu"])
        if saved["rng"]["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
        log.info("resumed at %d positions, %d steps", self.state["positions"], self.state["steps"])

    def init_from(self, path: Path) -> None:
        """A new pass from `path`'s weights and optimizer state: the same
        encoding (fingerprint), streams, window and architecture; this run's
        schedule, carried memories and learning rate. A checkpoint whose
        memory barrier failed starts nothing."""
        ck = torch.load(path, map_location="cpu", weights_only=True)
        meta = ck.get("training_meta", {})
        same = ("streams", "window", "arch") if self.args.init_other_records else \
            ("streams", "window", "fingerprint", "arch")
        for key in same:
            if meta.get(key) != self.meta[key]:
                raise SystemExit(f"{path} was trained with {key}={meta.get(key)!r}; this run has "
                                 f"{self.meta[key]!r}")
        if self.args.init_other_records:
            self._refuse_other_encoding(ck, path)
        state = ck.get("sequence_resume", {}).get("state", {})
        if state.get("barrier_failed"):
            raise SystemExit(f"{path} failed its memory barrier: no pass starts from it")
        self.model.load_state_dict(ck["model_state"])
        self.encoder.load_state_dict(ck["encoder_state"])
        self.opt.load_state_dict(ck["optimizer_state"])
        self.signal = self._signal_probe()
        areas = state.get("areas")
        if areas is not None and self.args.initial_areas is not None:
            raise SystemExit(f"{path} carries its rate history's areas; --initial-areas would replace them")
        if areas is None and self.args.initial_areas is not None:
            areas = json.loads(self.args.initial_areas)
        self.state["areas"] = None if areas is None else Areas.from_dict(areas).to_dict()
        self.meta["init_from"] = {"path": str(path), "seed": meta.get("seed"),
                                  "positions": int(state.get("positions", 0)), "steps": int(state.get("steps", 0)),
                                  "init_from": meta.get("init_from"), "areas": self.state["areas"],
                                  "fingerprint": meta.get("fingerprint")}
        log.info("a new pass from %s, which trained %d positions in %d steps", path,
                 self.meta["init_from"]["positions"], self.meta["init_from"]["steps"])

    def _refuse_other_encoding(self, ck: Dict, path: Path) -> None:
        """`--init-other-records`: the checkpoint's weights must mean what
        this run's records mean: the same vocabularies, observation epoch
        and encoding, whatever core phase or corpus version built its
        records."""
        mine = {"unit_type_to_id": dict(self.manifest["unit_type_to_id"]),
                "faction_to_id": dict(self.manifest["faction_to_id"]),
                "observation_epoch": int(OBSERVATION_EPOCH), "relevant_set_hexes": True,
                "fog_hides_enemy_villages": True, "terrain_multi_hot": True,
                **checkpoint_structure(self.model, self.encoder)}
        for key, want in mine.items():
            got = ck.get(key)
            if (dict(got) if isinstance(got, dict) else got) != want:
                raise SystemExit(f"{path} has {key}={got!r} where this run has {want!r}: its weights do not "
                                 f"read this run's records")
