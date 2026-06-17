"""Net2Net-style width/depth transfer (plan §3.5, progressive growth).

Warm-start a WIDER and/or DEEPER WesnothModel + encoder from a narrower
checkpoint instead of fresh init, so scaling the net (the §3.2
`--d-model/...` arch flags) doesn't throw away learned weights. Each
parameter's overlapping leading block is copied: a trained
[out_old, in_old] weight lands in the top-left of the wider
[out_new, in_new] matrix; the new rows/cols keep the wider model's
FRESH init (symmetry already broken, so the optimizer fine-tunes from a
warm subspace).

HONEST SCOPE: this is an APPROXIMATE warm start, NOT an exactly
function-preserving Net2WiderNet. Exact preservation through LayerNorm
(which renormalizes over the widened dim) + multi-head attention is
substantially more involved and rarely done exactly; the transplant
keeps the trained subspace (which beats random init) and the first
training steps absorb the new capacity. The IDENTITY case (same arch)
IS exact -- every block is a full copy, so the function is reproduced
bit-for-bit (test_net2net asserts it).

Attention `in_proj_weight`/`in_proj_bias` are [3E, .] stacked as
(Q; K; V); a d_model change is applied to each of the three blocks
SEPARATELY so widening doesn't shear Q's rows into K. All other params
(embeddings, out_proj, FFN, LayerNorm, heads, value atoms, the optional
aux head) use a single leading-block copy. Params with no source match
(e.g. extra transformer layers when growing DEPTH, or the aux head when
the source lacked it) keep their fresh init.

CLI:
    python tools/net2net.py --in narrow.pt --out wide.pt \\
        --d-model 384 --num-layers 6 --num-heads 8 --d-ff 1536
Then train from the grown checkpoint:
    python tools/sim_self_play.py --mcts --checkpoint-in wide.pt ...
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

log = logging.getLogger("net2net")


# ---------------------------------------------------------------------
# Core transfer
# ---------------------------------------------------------------------

def _copy_leading_block(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy the overlapping leading hyper-rectangle of `src` into
    `dst` (min size along every dim). In-place on `dst`."""
    sl = tuple(slice(0, min(d, s)) for d, s in zip(dst.shape, src.shape))
    dst[sl].copy_(src[sl])


def _transfer_param(name: str, dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy `src` into `dst` for one parameter, honoring the QKV-stacked
    layout of attention `in_proj_*` (split into 3 equal blocks along the
    leading dim and copied separately)."""
    if name.endswith(("in_proj_weight", "in_proj_bias")):
        de = dst.shape[0] // 3
        se = src.shape[0] // 3
        for i in range(3):
            _copy_leading_block(dst[i * de:(i + 1) * de],
                                src[i * se:(i + 1) * se])
    else:
        _copy_leading_block(dst, src)


def transfer_state_dict(
    src_sd: Dict[str, torch.Tensor], dst_model: torch.nn.Module,
) -> Dict[str, List[str]]:
    """Block-transfer `src_sd` into `dst_model` (in place). Returns a
    report: which params were copied exactly / partially (grown) /
    skipped (shape-incompatible) / left fresh (no source match)."""
    report = {"exact": [], "grown": [], "skipped": [], "fresh": []}
    new_sd = {k: v.clone() for k, v in dst_model.state_dict().items()}
    with torch.no_grad():
        for name, dst in new_sd.items():
            src = src_sd.get(name)
            if src is None:
                report["fresh"].append(name)
                continue
            if src.ndim != dst.ndim:
                report["skipped"].append(name)
                continue
            if tuple(src.shape) == tuple(dst.shape):
                dst.copy_(src)
                report["exact"].append(name)
            else:
                _transfer_param(name, dst, src)
                report["grown"].append(name)
    dst_model.load_state_dict(new_sd)
    return report


# ---------------------------------------------------------------------
# Checkpoint-level grow
# ---------------------------------------------------------------------

def grow_checkpoint(
    src_path: Path, out_path: Path, *,
    d_model: Optional[int] = None, num_layers: Optional[int] = None,
    num_heads: Optional[int] = None, d_ff: Optional[int] = None,
    aux_score: Optional[bool] = None, device=None,
) -> Dict:
    """Load a checkpoint, build a TransformerPolicy at the (possibly
    larger) target arch, block-transfer the trained weights into it,
    carry the vocab + decision_step, and save a new checkpoint ready to
    `--checkpoint-in`. Unspecified arch dims default to the source's."""
    from transformer_policy import TransformerPolicy

    dev = device or torch.device("cpu")
    raw = torch.load(src_path, map_location="cpu", weights_only=False)
    src_arch = raw.get("arch", {}) or {}
    src_aux = bool(raw.get("aux_score", False))

    arch = {
        "d_model":    d_model    if d_model    is not None else src_arch.get("d_model", 512),
        "num_layers": num_layers if num_layers is not None else src_arch.get("num_layers", 8),
        "num_heads":  num_heads  if num_heads  is not None else src_arch.get("num_heads", 8),
        "d_ff":       d_ff       if d_ff       is not None else src_arch.get("d_ff", 2048),
    }
    if arch["d_model"] % arch["num_heads"] != 0:
        raise SystemExit(
            f"--num-heads ({arch['num_heads']}) must divide "
            f"--d-model ({arch['d_model']}).")
    aux = src_aux if aux_score is None else bool(aux_score)

    pol = TransformerPolicy(device=dev, aux_score=aux, **arch)
    model_rep = transfer_state_dict(raw["model_state"], pol._model)
    enc_rep = transfer_state_dict(raw["encoder_state"], pol._encoder)

    # Carry the vocab so transferred embedding ROWS stay aligned with
    # their unit-type / faction ids; keep the inference encoder sharing
    # the same dict objects (as TransformerPolicy.__init__ set up).
    pol._encoder.unit_type_to_id = dict(raw.get("unit_type_to_id", {}))
    pol._encoder.faction_to_id = dict(raw.get("faction_to_id", {}))
    pol._inference_encoder.unit_type_to_id = pol._encoder.unit_type_to_id
    pol._inference_encoder.faction_to_id = pol._encoder.faction_to_id
    # Keep the inference snapshot consistent with the transferred trunk.
    pol._inference_model.load_state_dict(pol._model.state_dict())
    pol._inference_encoder.load_state_dict(pol._encoder.state_dict())
    pol._decision_step = int(raw.get("decision_step", 0))

    pol.save_checkpoint(out_path)
    report = {
        "src_arch": src_arch, "dst_arch": arch,
        "aux_score": aux, "decision_step": pol._decision_step,
        "model": {k: len(v) for k, v in model_rep.items()},
        "encoder": {k: len(v) for k, v in enc_rep.items()},
    }
    log.info(
        f"grew {Path(src_path).name} {src_arch} -> {arch} "
        f"(aux_score={aux}); model copied/grown/fresh/skipped="
        f"{report['model']}; encoder={report['encoder']}")
    return report


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", dest="out", type=Path, required=True)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--num-heads", type=int, default=None)
    ap.add_argument("--d-ff", type=int, default=None)
    aux = ap.add_mutually_exclusive_group()
    aux.add_argument("--aux-score", dest="aux_score", action="store_true",
                     default=None, help="Force the aux head ON in the "
                     "grown net (default: keep the source's setting).")
    aux.add_argument("--no-aux-score", dest="aux_score",
                     action="store_false",
                     help="Force the aux head OFF in the grown net.")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING"])
    args = ap.parse_args(argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S")
    if not args.src.exists():
        log.error(f"source checkpoint not found: {args.src}")
        return 1
    grow_checkpoint(
        args.src, args.out, d_model=args.d_model,
        num_layers=args.num_layers, num_heads=args.num_heads,
        d_ff=args.d_ff, aux_score=args.aux_score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
