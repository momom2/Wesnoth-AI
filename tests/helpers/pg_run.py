"""A stage 1 run and the inputs of stage C in miniature, for the
end-to-end tests of the policy-gradient path: a tiny network's
temperature-1 self-play games through the production path
(helpers.pg_games.selfplay_games), each game's index its seed less the seed
base and its seat alternating as run_elo_batch alternates them; the luck
walk and the side rows (tools/selfplay_rows.py), the labels with their b and
beta fits (tools/pg_labels.py), and the gradient pass (tools/pg_pass.py),
each through its command line.

Games 0-7 are trained on (one game a split), 8-11 held out (8 and 9 in half
A, 10 and 11 in half B). Eight turns on the mini maps give decisive games in
both sets; the run checks it, since a capped game's z is 0 and a set of
capped games has R = 0 throughout.
"""
from __future__ import annotations

import atexit
import contextlib
import functools
import io
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

SEED_BASE = 108
GAMES = 12
HELDOUT_FROM = 8
MAX_TURNS = 8
PASS_STREAMS = 4


def play(root: Path):
    """The network and its games: (checkpoint, games directory)."""
    from helpers.pg_games import save_network, selfplay_games, tiny_network
    from tools.run_elo_batch import slot_for
    vocab, encoder, model = tiny_network()
    checkpoint = save_network(root / "net.pt", vocab, encoder, model)
    games = root / "games"
    with contextlib.redirect_stdout(io.StringIO()):            # elo_eval_game prints each result
        selfplay_games(checkpoint, games, [slot_for(i, SEED_BASE) for i in range(GAMES)], max_turns=MAX_TURNS)
    return checkpoint, games


def label(root: Path, games: Path, walk: Path) -> Path:
    """The side rows of `walk` (walked from `games` when absent) and their
    labels: labels.jsonl beside fits.json in `root`."""
    from tools import pg_labels, selfplay_rows
    rows = root / (walk.stem + ".rows.jsonl")
    assert selfplay_rows.main(["--games", str(games), "--walk", str(walk), "--out", str(rows),
                               "--seed-base", str(SEED_BASE)]) == 0
    labels = root / (walk.stem + ".labels.jsonl")
    with contextlib.redirect_stdout(io.StringIO()):
        assert pg_labels.main([str(rows), "--heldout-from", str(HELDOUT_FROM), "--out", str(labels),
                               "--fits", str(root / (walk.stem + ".fits.json"))]) == 0
    return labels


def gradient_pass(checkpoint: Path, games: Path, labels: Path, out: Path) -> Path:
    from tools.pg_pass import main
    assert main(["--checkpoint", str(checkpoint), "--games", str(games), "--labels", str(labels), "--out", str(out),
                 "--device", "cpu", "--streams", str(PASS_STREAMS), "--log-level", "WARNING"]) == 0
    return out


def decisive(labels: Path) -> dict:
    """Decisive games by set: {"train", "A", "B"}."""
    out = {"train": set(), "A": set(), "B": set()}
    for line in labels.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not row["capped"]:
            out[row["half"] if row["heldout"] else "train"].add(row["game"])
    return out


@functools.lru_cache(maxsize=None)
def selfplay_run() -> SimpleNamespace:
    """The run, built once per process in a temporary directory."""
    root = Path(tempfile.mkdtemp(prefix="pg_run_"))
    atexit.register(shutil.rmtree, root, True)
    checkpoint, games = play(root)
    walk = root / "walk.jsonl"
    labels = label(root, games, walk)
    found = decisive(labels)
    if len(found["train"]) < 2 or not found["A"] or not found["B"]:
        raise AssertionError(f"the run needs decisive games in training and in both halves: {found}")
    pass_path = gradient_pass(checkpoint, games, labels, root / "pass.pt")
    return SimpleNamespace(root=root, checkpoint=checkpoint, games=games, walk=walk, labels=labels,
                           pass_path=pass_path, decisive=found)
