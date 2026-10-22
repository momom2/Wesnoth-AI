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


def play(root: Path, n_games: int = GAMES):
    """The network and its games: (checkpoint, games directory)."""
    from helpers.pg_games import save_network, selfplay_games, tiny_network
    from tools.run_elo_batch import slot_for
    vocab, encoder, model = tiny_network()
    checkpoint = save_network(root / "net.pt", vocab, encoder, model)
    games = root / "games"
    with contextlib.redirect_stdout(io.StringIO()):            # elo_eval_game prints each result
        selfplay_games(checkpoint, games, [slot_for(i, SEED_BASE) for i in range(n_games)], max_turns=MAX_TURNS)
    return checkpoint, games


def label(root: Path, games: Path, walk: Path, heldout_from: int = HELDOUT_FROM, walked: bool = False,
          zero_baseline: bool = False) -> Path:
    """The side rows of `walk` (walked from `games` first, unless `walked`:
    a planted copy is converted as it is) and their labels: labels.jsonl
    beside fits.json in `root`."""
    from tools import pg_labels, selfplay_rows
    rows = root / (walk.stem + ".rows.jsonl")
    source = ["--no-walk"] if walked else ["--games", str(games)]
    assert selfplay_rows.main([*source, "--walk", str(walk), "--out", str(rows), "--seed-base", str(SEED_BASE)]) == 0
    labels = root / (walk.stem + ".labels.jsonl")
    zero = ["--zero-baseline"] if zero_baseline else []
    with contextlib.redirect_stdout(io.StringIO()):
        assert pg_labels.main([str(rows), "--heldout-from", str(heldout_from), "--out", str(labels),
                               "--fits", str(root / (walk.stem + ".fits.json")), *zero]) == 0
    return labels


def gradient_pass(checkpoint: Path, games: Path, labels: Path, out: Path) -> Path:
    from tools.pg_pass import main
    assert main(["--checkpoint", str(checkpoint), "--games", str(games), "--labels", str(labels), "--out", str(out),
                 "--device", "cpu", "--streams", str(PASS_STREAMS), "--log-level", "WARNING"]) == 0
    return out


def plant_grads(run, out: Path, cls: str, delta=None) -> Path:
    """pg_plant.py delta on the run: u and grad M, delta given or by the rule."""
    from tools.pg_plant import main
    args = ["delta", "--checkpoint", str(run.checkpoint), "--games", str(run.games), "--labels", str(run.labels),
            "--walk", str(run.walk), "--seed-base", str(run.seed_base), "--class", cls, "--device", "cpu",
            "--streams", str(PASS_STREAMS), "--readout-fraction", "1.0", "--out", str(out)]
    args += ["--pass", str(run.pass_path)] if delta is None else ["--delta", str(delta)]
    assert main(args) == 0
    return out


def planted_readout(run, grads: Path, tag: str, zero_baseline: bool = False) -> dict:
    """The plant of `grads` through the production chain (the walk's copy,
    its rows and labels, the pass) and pg_plant.py's readout of it."""
    from tools.pg_plant import main
    walk = run.root / f"walk_{tag}.jsonl"
    assert main(["plant", "--walk", str(run.walk), "--games", str(run.games), "--grads", str(grads),
                 "--out", str(walk)]) == 0
    labels = label(run.root, run.games, walk, run.heldout_from, walked=True, zero_baseline=zero_baseline)
    pass_path = gradient_pass(run.checkpoint, run.games, labels, run.root / f"pass_{tag}.pt")
    out = run.root / f"plant_{tag}.json"
    assert main(["readout", "--checkpoint", str(run.checkpoint), "--games", str(run.games), "--labels", str(labels),
                 "--pass", str(pass_path), "--grads", str(grads), "--device", "cpu", "--streams", "4",
                 "--kl-fraction", "1.0", "--out", str(out)]) == 0
    return json.loads(out.read_text(encoding="utf-8"))


def decisive(labels: Path) -> dict:
    """Decisive games by set: {"train", "A", "B"}."""
    out = {"train": set(), "A": set(), "B": set()}
    for line in labels.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not row["capped"]:
            out[row["half"] if row["heldout"] else "train"].add(row["game"])
    return out


@functools.lru_cache(maxsize=None)
def selfplay_run(n_games: int = GAMES, heldout_from: int = HELDOUT_FROM, decisive_halves: bool = True,
                 zero_baseline: bool = False) -> SimpleNamespace:
    """The run of `n_games` games, held out from `heldout_from`, built once
    per process in a temporary directory; with `decisive_halves`, both
    held-out halves must hold a decisive game (stage C reads R there);
    `zero_baseline` labels it with b = beta = 0."""
    root = Path(tempfile.mkdtemp(prefix="pg_run_"))
    atexit.register(shutil.rmtree, root, True)
    checkpoint, games = play(root, n_games)
    walk = root / "walk.jsonl"
    labels = label(root, games, walk, heldout_from, zero_baseline=zero_baseline)
    found = decisive(labels)
    if len(found["train"]) < 2 or (decisive_halves and not (found["A"] and found["B"])):
        raise AssertionError(f"the run needs decisive games in training and in both halves: {found}")
    pass_path = gradient_pass(checkpoint, games, labels, root / "pass.pt")
    return SimpleNamespace(root=root, checkpoint=checkpoint, games=games, walk=walk, labels=labels,
                           pass_path=pass_path, decisive=found, seed_base=SEED_BASE, heldout_from=heldout_from)
