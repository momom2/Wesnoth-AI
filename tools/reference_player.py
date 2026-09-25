#!/usr/bin/env python3
"""The reference player, from configs/reference_player.json.

Every strength claim is a PURE match against one checkpoint under one
decode (docs/plan_20260904.md 3, rule 1). Both live in the config, not
in scripts, so a ruling that moves the reference is one edit:

    python tools/reference_player.py                # print the record
    python tools/reference_player.py --flags b      # run_elo_batch flags for side B
    python tools/reference_player.py --ensure       # fetch the checkpoint if missing
    python tools/reference_player.py --path         # the local checkpoint path

`--flags` prints the label, the spec and the decode flags of one side
in run_elo_batch's vocabulary, so a match script reads
    python tools/run_elo_batch.py --label-a arm --spec-a arm.pt \\
        $(python tools/reference_player.py --flags b) ...
and never spells the reference out. The decode is the raw player's
(mcts_sims 0): a searched reference would carry its own keys.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wesnoth_ai.paths import CONFIGS_DIR, REPO_ROOT  # noqa: E402

CONFIG = CONFIGS_DIR / "reference_player.json"


def load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def local_path(ref: dict | None = None) -> Path:
    ref = ref or load()
    return REPO_ROOT / ref["checkpoint_local"]


def ensure_checkpoint(ref: dict | None = None) -> Path:
    """The local checkpoint, fetched from the model host when missing."""
    ref = ref or load()
    path = local_path(ref)
    if path.exists():
        return path
    import shutil
    from huggingface_hub import hf_hub_download
    src = hf_hub_download(ref["hf_repo"], ref["checkpoint_hf"])
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, path)
    return path


def batch_flags(side: str, ref: dict | None = None) -> list[str]:
    """run_elo_batch flags for `side` ('a' or 'b'): label, spec and the
    decode. The temperature and the search budget are per-match flags
    the caller passes for both sides; only the reference's own decode
    options travel here."""
    ref = ref or load()
    side = side.lower()
    if side not in ("a", "b"):
        raise ValueError("side must be 'a' or 'b'")
    d = ref["decode"]
    flags = [f"--label-{side}", ref["label"], f"--spec-{side}", ref["checkpoint_local"]]
    if d.get("raw_end_turn", "joint") != "joint":
        flags += [f"--raw-end-turn-{side}", str(d["raw_end_turn"])]
    if d.get("raw_end_turn_offset"):
        flags += [f"--raw-end-turn-offset-{side}", str(d["raw_end_turn_offset"])]
    return flags


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--flags", choices=("a", "b"), default=None,
                    help="print the run_elo_batch flags that put the reference on this side")
    ap.add_argument("--ensure", action="store_true", help="fetch the checkpoint if it is missing")
    ap.add_argument("--path", action="store_true", help="print the local checkpoint path")
    args = ap.parse_args(argv)
    ref = load()
    if args.ensure:
        print(ensure_checkpoint(ref))
        return 0
    if args.flags:
        print(" ".join(batch_flags(args.flags, ref)))
        return 0
    if args.path:
        print(local_path(ref))
        return 0
    print(json.dumps({k: v for k, v in ref.items() if not k.startswith("_")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
