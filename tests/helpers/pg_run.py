"""A stage 1 run and the inputs of stage C in miniature, for the
end-to-end tests of the policy-gradient path: a tiny network's
temperature-1 self-play games through the production path
(helpers.pg_games.selfplay_games), each game's index its seed less the seed
base and its seat alternating as run_elo_batch alternates them, played and
walked once per process (`played`); the side rows (tools/selfplay_rows.py),
the labels (tools/pg_labels.py) and the gradient pass (tools/pg_pass.py),
each through its command line.

Two runs read the set:
- `e2e_run`: its first 12 games, 0-7 trained on (one game a split) and 8-11
  held out (8 and 9 in half A, 10 and 11 in half B), labelled with their b
  and beta fits, and its pass. Eight turns on the mini maps give decisive
  games in both sets; the run checks it, since a capped game's z is 0 and a
  set of capped games has R = 0 throughout.
- `plant_run`: all 68, 0-63 trained on and 64-67 held out, b and beta at
  zero (four held-out games fit seven luck slopes exactly, and the training
  labels would carry that fit's noise), no pass (the planted readout's tests
  give delta).
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
GAMES = 68
E2E_GAMES, E2E_HELDOUT_FROM = 12, 8
PLANT_HELDOUT_FROM = 64
MAX_TURNS = 8
PASS_STREAMS = 32


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


def label(root: Path, games: Path, walk: Path, heldout_from: int, walked: bool = True,
          zero_baseline: bool = False) -> Path:
    """The side rows of `walk` (converted as it is, or walked from `games`
    first when not `walked`) and their labels: labels.jsonl beside fits.json
    in `root`."""
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
    """The pass through its command line; no floor reading (its own test,
    tests/test_pg_pass.py, reads it; on this CPU's bf16 it is 7 % of a pass)."""
    from tools.pg_pass import main
    assert main(["--checkpoint", str(checkpoint), "--games", str(games), "--labels", str(labels), "--out", str(out),
                 "--device", "cpu", "--streams", str(PASS_STREAMS), "--floor-dtype", "off",
                 "--log-level", "WARNING"]) == 0
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
    labels = label(run.root, run.games, walk, run.heldout_from, zero_baseline=zero_baseline)
    pass_path = gradient_pass(run.checkpoint, run.games, labels, run.root / f"pass_{tag}.pt")
    out = run.root / f"plant_{tag}.json"
    assert main(["readout", "--checkpoint", str(run.checkpoint), "--games", str(run.games), "--labels", str(labels),
                 "--pass", str(pass_path), "--grads", str(grads), "--device", "cpu", "--streams", "8",
                 "--out", str(out)]) == 0
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
def played() -> SimpleNamespace:
    """The games, played and walked once per process in a temporary
    directory."""
    from tools.match_luck import walk_matches
    from tools.selfplay_rows import MATCH_NAME
    root = Path(tempfile.mkdtemp(prefix="pg_run_"))
    atexit.register(shutil.rmtree, root, True)
    checkpoint, games = play(root)
    walk = root / "walk.jsonl"
    walk_matches([(MATCH_NAME, games)], walk)
    return SimpleNamespace(root=root, checkpoint=checkpoint, games=games, walk=walk, seed_base=SEED_BASE)


def _first_games(walk: Path, out: Path, n: int) -> Path:
    """The walk's rows of the games of index below n."""
    rows = [line for line in walk.read_text(encoding="utf-8").splitlines()
            if line.strip() and int(json.loads(line)["seed"]) - SEED_BASE < n]
    out.write_text("".join(r + "\n" for r in rows), encoding="utf-8")
    return out


@functools.lru_cache(maxsize=None)
def e2e_run() -> SimpleNamespace:
    """The set's first 12 games, labelled with their fits, and their pass."""
    p = played()
    walk = _first_games(p.walk, p.root / "walk_e2e.jsonl", E2E_GAMES)
    labels = label(p.root, p.games, walk, E2E_HELDOUT_FROM)
    found = decisive(labels)
    if len(found["train"]) < 2 or not (found["A"] and found["B"]):
        raise AssertionError(f"the run needs decisive games in training and in both halves: {found}")
    pass_path = gradient_pass(p.checkpoint, p.games, labels, p.root / "pass_e2e.pt")
    return SimpleNamespace(root=p.root, checkpoint=p.checkpoint, games=p.games, walk=walk, labels=labels,
                           pass_path=pass_path, decisive=found, seed_base=SEED_BASE, heldout_from=E2E_HELDOUT_FROM)


@functools.lru_cache(maxsize=None)
def plant_run() -> SimpleNamespace:
    """All 68 games, labelled with b = beta = 0; no pass."""
    p = played()
    labels = label(p.root, p.games, p.walk, PLANT_HELDOUT_FROM, zero_baseline=True)
    return SimpleNamespace(root=p.root, checkpoint=p.checkpoint, games=p.games, walk=p.walk, labels=labels,
                           pass_path=None, seed_base=SEED_BASE, heldout_from=PLANT_HELDOUT_FROM)
