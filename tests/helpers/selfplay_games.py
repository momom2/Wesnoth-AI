"""Short self-play games of a tiny random network on the core, written as a
run_elo_batch outdir holds them (result files beside their records)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple


def tiny_checkpoint(path: Path) -> str:
    """A random tiny network saved at `path`."""
    import torch
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(1)
    TransformerPolicy(device=torch.device("cpu"), d_model=32, num_layers=1, num_heads=2,
                      d_ff=64).save_checkpoint(path)
    return str(path)


def selfplay_game(spec: str, out: Path, seed: int, side_a: int = 1, *extra: str,
                  max_turns: int = 2) -> Tuple[dict, dict]:
    """One temperature-1 game of `spec` on both seats (labels A and B)
    into `out`: (its result, its record)."""
    from tools.elo_eval_game import main
    from tools.game_record import read_records
    assert main(["x", "A", spec, "B", spec, str(side_a), str(seed), str(out), "--mcts-sims", "0",
                 "--raw-temperature-a", "1", "--raw-temperature-b", "1", "--max-turns", str(max_turns),
                 "--device", "cpu", *extra]) == 0
    stem = f"game_A_B_s{side_a}_{seed}"
    result = json.loads((out / f"{stem}.json").read_text(encoding="utf-8"))
    (record,) = list(read_records(out / f"{stem}.game.jsonl.gz"))
    return result, record
