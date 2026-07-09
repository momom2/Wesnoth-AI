"""Load the human-replay value corpus as trainer-ready experiences.

Consumes the index written by tools/build_value_corpus.py
(replays_dataset/value_corpus_index.jsonl) and yields
`MCTSExperience`s with:

  - game_state: reconstructed state at a human decision point
    (bit-exact replay_dataset machinery, same states the encoder
    sees in self-play),
  - z: +-1 from the perspective of the side to move (same negamax
    convention as MCTSPolicy.finalize_game),
  - visit_counts=[]: trainer.step_mcts treats zero total visits as
    "no policy target" (early-return, loss 0) and still trains the
    value / aux / moves-left heads -- so these samples are directly
    batchable through the EXISTING training step, alone or mixed
    with self-play experiences,
  - moves_left_target: from the game's actual end turn (same
    MOVES_LEFT_NORM_TURNS normalization as self-play),
  - aux_target: signed material margin at game end (same
    material_margin the self-play path uses).

Sampling: every `stride`-th decision per game (deepcopy per sampled
state is the cost driver; stride=4 keeps ~60-150 states/game), with
an optional per-game cap.

CLI dry-run: reports pair counts, z balance, and (with --eval-ckpt)
the current checkpoint's value CE / prediction entropy / marginal
floor on a corpus sample -- the human-data analogue of the
fresh_value_ce probe.

    python tools/value_corpus.py replays_dataset [--stride 4]
        [--limit-games 50] [--eval-ckpt training/checkpoints/x.pt]
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trainer import MCTSExperience

log = logging.getLogger("value_corpus")

# Command kinds that correspond to a player decision at the current
# state (replay_dataset._apply_command dispatch). init_side is
# bookkeeping, not a decision.
_DECISION_KINDS = ("move", "attack", "recruit", "recall", "end_turn")


def game_experiences(gz_path: Path, winner: int, *,
                     stride: int = 4,
                     per_game_cap: Optional[int] = None,
                     rng: Optional[random.Random] = None,
                     moves_left_norm: Optional[float] = None,
                     ) -> List[MCTSExperience]:
    """Reconstruct one indexed game and return sampled experiences."""
    from tools.mcts_policy import MOVES_LEFT_NORM_TURNS
    from tools.replay_dataset import (_apply_command,
                                      _build_initial_gamestate,
                                      _setup_scenario_events)

    norm = moves_left_norm or MOVES_LEFT_NORM_TURNS
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    gs = _build_initial_gamestate(data)
    _setup_scenario_events(gs, data.get("scenario_id", ""))

    # Phase-align the stride across games (rng picks the offset) so
    # the corpus isn't biased toward turn starts.
    offset = (rng.randrange(stride) if rng and stride > 1 else 0)
    sampled = []          # (state_copy, side, turn)
    k = 0
    for cmd in data.get("commands", []):
        kind = cmd[0] if cmd else "?"
        if kind in _DECISION_KINDS:
            side = gs.global_info.current_side
            if side in (1, 2) and k % stride == offset:
                sampled.append((copy.deepcopy(gs), side,
                                gs.global_info.turn_number))
            k += 1
        _apply_command(gs, cmd)

    end_turn = gs.global_info.turn_number
    out: List[MCTSExperience] = []
    for st, side, turn in sampled:
        remaining = max(0, end_turn - turn)
        out.append(MCTSExperience(
            game_state=st,
            visit_counts=[],          # value/aux/ml-only sample
            z=(+1.0 if side == winner else -1.0),
            moves_left_target=min(1.0, remaining / norm),
        ))
    if per_game_cap is not None and len(out) > per_game_cap:
        out = (rng or random).sample(out, per_game_cap)
    return out


def game_raw_experiences(gz_path: Path, winner: int, *,
                         type_to_id: dict, faction_to_id: dict,
                         stride: int = 8,
                         rng: Optional[random.Random] = None,
                         moves_left_norm: Optional[float] = None):
    """Like game_experiences, but encode_raw each sampled state at
    sample time (no deepcopy) and return picklable
    (RawEncoded, z, moves_left) tuples — the worker-side producer for
    the parallel value fine-tune. `type_to_id`/`faction_to_id` MUST be
    the model's frozen vocab (encode_raw is read-only; out-of-vocab ->
    overflow bucket)."""
    from encoder import encode_raw
    from tools.mcts_policy import MOVES_LEFT_NORM_TURNS
    from tools.replay_dataset import (_apply_command,
                                      _build_initial_gamestate,
                                      _setup_scenario_events)
    norm = moves_left_norm or MOVES_LEFT_NORM_TURNS
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    gs = _build_initial_gamestate(data)
    _setup_scenario_events(gs, data.get("scenario_id", ""))
    offset = (rng.randrange(stride) if rng and stride > 1 else 0)
    sampled = []
    k = 0
    for cmd in data.get("commands", []):
        kind = cmd[0] if cmd else "?"
        if kind in _DECISION_KINDS:
            side = gs.global_info.current_side
            if side in (1, 2) and k % stride == offset:
                raw = encode_raw(gs, type_to_id=type_to_id,
                                 faction_to_id=faction_to_id)
                sampled.append((raw, side, gs.global_info.turn_number))
            k += 1
        _apply_command(gs, cmd)
    end_turn = gs.global_info.turn_number
    out = []
    for raw, side, turn in sampled:
        z = +1.0 if side == winner else -1.0
        ml = min(1.0, max(0, end_turn - turn) / norm)
        out.append((raw, z, ml))
    return out


def iter_corpus(index_path: Path, *,
                stride: int = 4,
                per_game_cap: Optional[int] = None,
                limit_games: Optional[int] = None,
                seed: int = 0,
                shuffle_games: bool = True,
                ) -> Iterator[List[MCTSExperience]]:
    """Yield one game's experiences at a time (game granularity keeps
    train/holdout splits leak-free, mirroring the self-play rule)."""
    rows = []
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:                   # noqa: BLE001
                continue
    rng = random.Random(seed)
    if shuffle_games:
        rng.shuffle(rows)
    if limit_games:
        rows = rows[:limit_games]
    base = index_path.parent
    for row in rows:
        try:
            exps = game_experiences(base / row["file"], row["winner"],
                                    stride=stride,
                                    per_game_cap=per_game_cap, rng=rng)
        except Exception as e:                  # noqa: BLE001
            log.warning(f"skip {row['file']}: {e}")
            continue
        if exps:
            yield exps


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_dir", type=Path)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--per-game-cap", type=int, default=None)
    ap.add_argument("--limit-games", type=int, default=None)
    ap.add_argument("--eval-ckpt", type=Path, default=None,
                    help="Also report this checkpoint's value CE / "
                         "prediction entropy / marginal floor on the "
                         "sampled corpus states.")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    index = args.dataset_dir / "value_corpus_index.jsonl"
    stats: Counter = Counter()
    all_exps: List[MCTSExperience] = []
    for exps in iter_corpus(index, stride=args.stride,
                            per_game_cap=args.per_game_cap,
                            limit_games=args.limit_games):
        stats["games"] += 1
        stats["pairs"] += len(exps)
        stats["z_pos"] += sum(1 for e in exps if e.z > 0)
        stats["z_neg"] += sum(1 for e in exps if e.z < 0)
        all_exps.extend(exps)
    log.info(f"corpus: {stats['games']} games, {stats['pairs']} pairs "
             f"(z +{stats['z_pos']} / -{stats['z_neg']})")

    if args.eval_ckpt and all_exps:
        import torch
        from transformer_policy import TransformerPolicy
        raw = torch.load(args.eval_ckpt, map_location="cpu",
                         weights_only=False)
        a = raw["arch"]
        policy = TransformerPolicy(
            d_model=a["d_model"], num_layers=a["num_layers"],
            num_heads=a["num_heads"], d_ff=a["d_ff"],
            aux_score=bool(raw.get("aux_score")),
            moves_left=bool(raw.get("moves_left")))
        policy.load_checkpoint(args.eval_ckpt)
        probe = (all_exps if len(all_exps) <= 512
                 else random.Random(0).sample(all_exps, 512))
        m = policy._trainer.eval_value_metrics(probe)
        log.info(f"checkpoint on human corpus (n={len(probe)}): "
                 f"value_ce={m['ce']:.4f} "
                 f"pred_entropy={m['pred_entropy']:.4f} "
                 f"marginal_floor={m['marginal_ce_floor']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
