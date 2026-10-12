#!/usr/bin/env python3
"""The reference player, from configs/reference_player.json.

Every strength claim is a PURE match against one checkpoint under one
decode (docs/plan_20260904.md 3, rule 1). Both live in the config, not
in scripts, so a ruling that moves the reference is one edit:

    python tools/reference_player.py                # print the record
    python tools/reference_player.py --flags b      # run_elo_batch flags for side B
    python tools/reference_player.py --flags a --decode v.json   # ...under a knob vector
    python tools/reference_player.py --ensure       # fetch the checkpoint if missing
    python tools/reference_player.py --path         # the local checkpoint path

`--flags` prints the label, the spec, the memory and the decode flags of
one side in run_elo_batch's vocabulary, so a match script reads
    python tools/run_elo_batch.py --label-a arm --spec-a arm.pt \\
        $(python tools/reference_player.py --flags b) ...
and never spells the reference out. The decode is the raw player's
(mcts_sims 0): a searched reference would carry its own keys.

The decode block's knob vector (tools/eval_procedure.KNOB_VECTOR_KEYS: the
end_turn offset, the attack and recruit offsets, the attack time-of-day
slope; an absent key is 0) is adopted by editing that block, and
`procedure_tag` must be the tag the block plays under: `load` refuses a
config where they differ. `--decode FILE` replaces the knob vector by a
JSON file of those keys (a search's vector or its winner), keeping the
checkpoint, its memory and the temperature.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.eval_procedure import KNOB_VECTOR_KEYS, decode_flags, decode_procedure  # noqa: E402
from wesnoth_ai.paths import CONFIGS_DIR, REPO_ROOT  # noqa: E402

CONFIG = CONFIGS_DIR / "reference_player.json"


def load(path: Path = CONFIG) -> dict:
    """The reference record, its procedure tag checked against its decode."""
    ref = json.loads(Path(path).read_text(encoding="utf-8"))
    want = decode_procedure(ref["decode"])
    if ref.get("procedure_tag", want) != want:
        raise SystemExit(f"{path}: procedure_tag {ref['procedure_tag']!r} is not the tag its "
                         f"decode block plays under, {want!r}: edit the two together")
    return ref


def with_decode(ref: dict, vector: dict) -> dict:
    """`ref` playing the knob vector `vector` (KNOB_VECTOR_KEYS, absent =
    0) in place of its own, the rest of its decode kept."""
    unknown = sorted(set(vector) - set(KNOB_VECTOR_KEYS))
    if unknown:
        raise ValueError(f"not knob-vector keys: {unknown} (known: {list(KNOB_VECTOR_KEYS)})")
    decode = {k: v for k, v in ref["decode"].items() if k not in KNOB_VECTOR_KEYS}
    decode.update({k: float(vector.get(k, 0.0)) for k in KNOB_VECTOR_KEYS})
    return dict(ref, decode=decode, procedure_tag=decode_procedure(decode))


def local_path(ref: dict | None = None) -> Path:
    ref = ref or load()
    return REPO_ROOT / ref["checkpoint_local"]


def verify_checkpoint(path: Path, ref: dict | None = None) -> Path:
    """`path` when its SHA-256 is the adopted reference's; raises otherwise
    (a changed file would silently measure another reference)."""
    ref = ref or load()
    want = ref.get("checkpoint_sha256")
    if want:
        from tools.eval_provenance import file_sha256
        got = file_sha256(path)
        if got != want:
            raise SystemExit(f"{path} is not the adopted reference {ref['label']}: "
                             f"SHA-256 {got}, expected {want}")
    return path


def ensure_checkpoint(ref: dict | None = None) -> Path:
    """The local checkpoint, fetched from the model host when missing,
    checked against the adopted reference's SHA-256 either way."""
    ref = ref or load()
    path = local_path(ref)
    if not path.exists():
        import shutil
        from huggingface_hub import hf_hub_download
        src = hf_hub_download(ref["hf_repo"], ref["checkpoint_hf"])
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, path)
    return verify_checkpoint(path, ref)


def batch_flags(side: str, ref: dict | None = None) -> list[str]:
    """run_elo_batch flags for `side` ('a' or 'b'): label, spec, the
    memory slots it plays with (`memory_slots`, when it has a memory) and
    the decode. The temperature and the search budget are per-match flags
    the caller passes for both sides; only the reference's own options
    travel here."""
    ref = ref or load()
    side = side.lower()
    if side not in ("a", "b"):
        raise ValueError("side must be 'a' or 'b'")
    flags = [f"--label-{side}", ref["label"], f"--spec-{side}", ref["checkpoint_local"]]
    if ref.get("memory_slots"):
        flags += [f"--memory-{side}", str(int(ref["memory_slots"]))]
    return flags + decode_flags(side, ref["decode"])


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--flags", choices=("a", "b"), default=None,
                    help="print the run_elo_batch flags that put the reference on this side")
    ap.add_argument("--decode", type=Path, default=None,
                    help="with --flags: play this knob vector (a JSON of the knob-vector keys)")
    ap.add_argument("--ensure", action="store_true", help="fetch the checkpoint if it is missing")
    ap.add_argument("--path", action="store_true", help="print the local checkpoint path")
    args = ap.parse_args(argv)
    ref = load()
    if args.decode is not None:
        ref = with_decode(ref, json.loads(args.decode.read_text(encoding="utf-8")))
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
