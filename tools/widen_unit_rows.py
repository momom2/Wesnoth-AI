"""Give a checkpoint the unit rows of version 2, once
(docs/input_facts_design_20261124.md, "What changes in the code").

A network of unit row version 1 reads 109 columns of each unit and recruit
row; one of version 2 reads 341 (wesnoth_ai/encoder.py, "The unit row,
version 2"). The 232 added columns are read by a projection of their own,
the encoder's `unit_feat_proj2`, [d_model, 232], with no bias. This tool
adds it at zero. The widened network therefore computes exactly what the
source computed: each unit token gains a sum of zeros, until training
moves the new weights. It writes the source's fields with that weight
added, `unit_row_version` 2, the running code's observation epoch, and
`unit_rows_widened` naming the source.

With `--state`, the simple loop's optimizer state (tools/pg_adam_step.py)
of the source is carried to the widened checkpoint: the new parameter's
name is added to it, with no moments. Adam then meets it as a parameter it
has never stepped, and gives it its ordinary update at the loop's one
learning rate. Every other parameter's moments and step counts are the
source's. (Zero columns appended to the old projection and to its
moments would not do that: Adam corrects its averages by a parameter's
step count, so new entries in a parameter stepped hundreds of times move
two to four times the learning rate for some hundred steps.)

Refused (exit 3, nothing written): a checkpoint that is not of version 1,
or not of the parity observation, or a state that does not continue from
the checkpoint.

    python tools/widen_unit_rows.py --checkpoint pi.pt --out pi_rows2.pt \\
        [--state adam.pt --state-out adam_rows2.pt]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from wesnoth_ai.checkpoint_structure import saved_structure  # noqa: E402
from wesnoth_ai.constants import OBSERVATION_EPOCH  # noqa: E402
from wesnoth_ai.encoder import UNIT_FEAT_DIM_PARITY, UNIT_FEAT_DIM_ROWS2  # noqa: E402

log = logging.getLogger("widen_unit_rows")

EXIT_REFUSED = 3
BASE_WEIGHT = "unit_feat_proj.weight"            # in the checkpoint's encoder_state
NEW_WEIGHT = "unit_feat_proj2.weight"
PARAMETER = "encoder." + NEW_WEIGHT              # in the optimizer state's names
ADDED_COLUMNS = UNIT_FEAT_DIM_ROWS2 - UNIT_FEAT_DIM_PARITY


class WidenRefused(ValueError):
    """The checkpoint or the state is not what this tool widens."""


def _atomic_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def widen_checkpoint(ck: Dict, base_sha256: str) -> Dict:
    """`ck` as a checkpoint of unit row version 2 (a new dict; the source's
    weights are shared with `ck`)."""
    structure = saved_structure(ck)
    if not structure["observation_parity"]:
        raise WidenRefused("the checkpoint does not read the parity observation, whose rows version 2 widens")
    if structure["unit_row_version"] != 1:
        raise WidenRefused(f"the checkpoint is of unit row version {structure['unit_row_version']}, not 1")
    base = ck["encoder_state"][BASE_WEIGHT]
    if base.shape[1] != UNIT_FEAT_DIM_PARITY or NEW_WEIGHT in ck["encoder_state"]:
        raise WidenRefused(f"{BASE_WEIGHT} has {base.shape[1]} columns, not version 1's {UNIT_FEAT_DIM_PARITY}, "
                           f"or the checkpoint already holds {NEW_WEIGHT}")
    out = dict(ck)
    # The new weight goes last, where a version-2 encoder registers it.
    out["encoder_state"] = {**ck["encoder_state"],
                            NEW_WEIGHT: torch.zeros(base.shape[0], ADDED_COLUMNS, dtype=base.dtype)}
    out["unit_row_version"] = 2
    out["observation_epoch"] = int(OBSERVATION_EPOCH)
    out["unit_rows_widened"] = {"base_sha256": base_sha256, "from_columns": int(UNIT_FEAT_DIM_PARITY),
                                "to_columns": int(UNIT_FEAT_DIM_ROWS2)}
    return out


def widen_state(state: Dict, base_sha256: str, out_sha256: str) -> Dict:
    """The optimizer state of the source checkpoint as the widened one's:
    the new parameter named last, as the widened network lists it, and
    without moments (Adam builds them at its first step)."""
    if state.get("checkpoint_sha256") != base_sha256:
        raise WidenRefused("the optimizer's state does not continue from the checkpoint")
    names = list(state["names"])
    if PARAMETER in names:
        raise WidenRefused(f"the optimizer's state already holds {PARAMETER}")
    optimizer = dict(state["optimizer"])
    groups = [dict(g) for g in optimizer["param_groups"]]
    if len(groups) != 1 or list(groups[0]["params"]) != list(range(len(names))):
        raise WidenRefused("the optimizer's state is not one group over the named parameters in order")
    groups[0]["params"] = list(groups[0]["params"]) + [len(names)]
    optimizer["param_groups"] = groups
    out = dict(state)
    out["names"] = names + [PARAMETER]
    out["optimizer"] = optimizer
    out["checkpoint_sha256"] = out_sha256
    return out


def widen(checkpoint: Path, out: Path, state: Optional[Path] = None, state_out: Optional[Path] = None) -> Dict:
    """The tool (module docstring); its record. Raises `WidenRefused`
    before anything is written."""
    from tools.eval_provenance import file_sha256
    if (state is None) != (state_out is None):
        raise WidenRefused("--state and --state-out go together")
    base_sha = file_sha256(checkpoint)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    wide = widen_checkpoint(ck, base_sha)
    saved_state = None
    if state is not None:
        saved_state = torch.load(state, map_location="cpu", weights_only=True)
        widen_state(saved_state, base_sha, "")              # refuse before the checkpoint is written
    _atomic_save(wide, out)
    out_sha = file_sha256(out)
    record = {"base_sha256": base_sha, "checkpoint_sha256": out_sha,
              "from_columns": int(UNIT_FEAT_DIM_PARITY), "to_columns": int(UNIT_FEAT_DIM_ROWS2),
              "parameter": PARAMETER, "observation_epoch": int(OBSERVATION_EPOCH), "state": None}
    if saved_state is not None:
        _atomic_save(widen_state(saved_state, base_sha, out_sha), state_out)
        record["state"] = {"step": int(saved_state["step"]), "sha256": file_sha256(state_out)}
    summary = out.with_name(out.stem + ".widen.json")
    tmp = summary.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, summary)
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--state", type=Path, default=None)
    ap.add_argument("--state-out", type=Path, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    try:
        record = widen(args.checkpoint, args.out, args.state, args.state_out)
    except WidenRefused as e:
        log.error("refused: %s", e)
        return EXIT_REFUSED
    with_state = "" if record["state"] is None else f", optimizer state at step {record['state']['step']}"
    log.info("widened %s -> %s (%d -> %d columns)%s", args.checkpoint, args.out, record["from_columns"],
             record["to_columns"], with_state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
