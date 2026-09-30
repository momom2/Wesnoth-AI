"""The holdout policy cross-entropy of a checkpoint without a memory, over
every decision of the holdout games, as the sequence probe computes it for
the parity-memory arm (tools/sequence_probe.py): the action-type weighted
actor term plus the type, target and weapon terms, label smoothing
included, per decision of both sides, and over the winners' decisions.

The pre-registration's barrier compares the arm at 0 slots with `obs8` on
the same holdout decisions (docs/parity_memory_prereg_20260929.md "Bars");
this reads `obs8`'s side. The positions are rebuilt from the corpus and
encoded in the checkpoint's own observation and hex basis, so a target hex
outside `obs8`'s relevant set but inside the arm's (version 2 holds version
1) scores the arm's target head and not `obs8`'s.

    python tools/holdout_ce.py training/checkpoints/obs8.pt \\
        --dataset replays_dataset_imitation --out obs8_holdout_ce.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

log = logging.getLogger("holdout_ce")
BATCH = 64


def _mean_se(xs: List[float]):
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, None
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, (v / len(xs)) ** 0.5


def holdout_ce(checkpoint: Path, dataset: Path, device: torch.device) -> Dict:
    from tools.eval_players import _load_policy
    from tools.replay_dataset import iter_record_pairs
    from tools.supervised_train import _DEFAULT_ACTION_TYPE_LOSS_WEIGHT
    from wesnoth_ai.encoder import encode_raw
    from wesnoth_ai.imitation_loss import build_imitation_targets, imitation_loss_parts
    policy = _load_policy(checkpoint, device, label="holdout")
    encoder, model = policy._encoder.eval(), policy._model.eval()
    if int(getattr(model, "memory_slots", 0) or 0):
        raise SystemExit(f"{checkpoint} has a memory: its probe is tools/sequence_probe.py")
    rows = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    holdout = [r for r in rows if r.get("holdout")]
    kw = dict(type_to_id=encoder.unit_type_to_id, faction_to_id=encoder.faction_to_id,
              relevant_set=bool(encoder.relevant_set_hexes),
              fog_hides_enemy_villages=bool(getattr(encoder, "fog_hides_enemy_villages", False)),
              terrain_multi_hot=bool(getattr(encoder, "terrain_multi_hot", False)))
    per_game: Dict[str, List[float]] = {}
    winners: List[float] = []
    skipped = 0
    t0 = time.time()
    for r in holdout:
        try:
            with gzip.open(dataset / r["file"], "rt", encoding="utf-8") as f:
                data = json.load(f)
            items = [(encode_raw(gs, **kw), ai, int(gs.global_info.current_side))
                     for gs, ai in iter_record_pairs(data, relevant_set=kw["relevant_set"])]
        except Exception as e:                   # noqa: BLE001 - counted, reported
            skipped += 1
            log.warning("%s not rebuilt: %r", r["file"], e)
            continue
        ces: List[float] = []
        for start in range(0, len(items), BATCH):
            chunk = items[start:start + BATCH]
            with torch.no_grad():
                streams = encoder.encode_from_raw_embedded([raw for raw, _, _ in chunk], device=device)
                padded = model.forward_embedded(streams).float32()
            targets = build_imitation_targets(
                [ai for _, ai, _ in chunk], [(None, 0.0, 1.0)] * len(chunk), streams.sizes,
                n_types=padded.type_logits.shape[2], n_weapons=padded.weapon_logits.shape[2],
                n_atoms=padded.value_logits.shape[1],
                type_loss_weights=dict(_DEFAULT_ACTION_TYPE_LOSS_WEIGHT), device=device)
            parts = imitation_loss_parts(padded, targets)
            vals = (parts.actor_weighted + parts.type + parts.target + parts.weapon).cpu().tolist()
            for (_, _, side), ce, ok in zip(chunk, vals, targets.ok["actor"]):
                if ok:
                    ces.append(ce)
                    if side == int(r["winner_side"]):
                        winners.append(ce)
        per_game[r["file"]] = ces
    everything = [c for ces in per_game.values() for c in ces]
    game_means = [sum(c) / len(c) for c in per_game.values() if c]
    return {"checkpoint": str(checkpoint), "n_games": len(per_game), "skipped_games": skipped,
            "n_decisions": len(everything), "ce_all": _mean_se(everything)[0],
            "ce_all_se": _mean_se(game_means)[1], "ce_winners": _mean_se(winners)[0],
            "seconds": round(time.time() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    result = holdout_ce(args.checkpoint, args.dataset, torch.device(args.device))
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    log.info("HOLDOUT_CE %s", json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
