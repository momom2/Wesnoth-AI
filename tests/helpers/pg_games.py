"""Self-play games with decision logs, for the policy-gradient tests
(docs/training_design_20261019/build_contracts.md §1-2).

The games are played the way stage 1 plays them: `elo_eval_game.main` with
both seats on one tiny network at temperature 1, no end_turn offset, 64
memory slots, `--log-decisions` (the raw player's own log,
tools/decision_log.py), behind the shared-inference transport (server-side
priors), its forward served in this process as the CPU inference server
serves it. Each game writes its record and its match result as a match
does. Three things differ from a box match, each by a patch for the game's
duration only:
- the transport is in-process (`InProcessTransport`, no server process);
- the scenario is drawn from the mini maps, so three turns hold contact;
- a harness may refuse chosen recruit decisions: the match loop's bounce
  check (`eval_players._timed_bounce`) answers yes for them, so the loop
  rejects the ordered hex and the player decides again, as for a recruit
  the engine bounces.

`played_games()` plays the suite's games once per process; `selfplay_games`
plays any list of match slots.
"""
from __future__ import annotations

import atexit
import functools
import json
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from unittest import mock

import torch

SLOTS = 64
TINY_ARCH = {"d_model": 32, "num_layers": 1, "num_heads": 2, "d_ff": 64}
MAX_TURNS = 3
LABELS = ("A", "B")
# The suite's games: name -> (seed, side A's side, the recruit decisions the
# harness refuses, by their running index over the game). Seeds 13 and 7 hold
# attacks in their first twelve decisions and a recruit followed by a
# decision that may still recruit elsewhere; seed 5's side 1 recruits three
# times in its first turn.
SUITE = {"g0": (13, 1, ()), "g1": (5, 1, (0, 2)), "g2": (7, 1, ())}


def tiny_network(seed: int = 4):
    """(vocabularies, encoder, model) of a tiny network of the recipe's
    structure (tools/sequence_setup.build_modules), in eval mode."""
    from tools.preencode_sequences import fresh_vocab
    from tools.sequence_setup import build_modules
    torch.manual_seed(seed)
    vocab = fresh_vocab()
    encoder, model = build_modules(*vocab, torch.device("cpu"), TINY_ARCH)
    return vocab, encoder.eval(), model.eval()


def save_network(path: Path, vocab, encoder, model) -> Path:
    """The network as a checkpoint the policy loader plays
    (eval_players._load_policy) and the pass and the evaluator load: the
    keys of `sequence_setup.save_checkpoint` without the trainer's state."""
    from wesnoth_ai.checkpoint_structure import checkpoint_structure
    from wesnoth_ai.constants import OBSERVATION_EPOCH
    torch.save({"observation_epoch": int(OBSERVATION_EPOCH), "arch": dict(TINY_ARCH),
                "relevant_set_hexes": True, "fog_hides_enemy_villages": True, "terrain_multi_hot": True,
                **checkpoint_structure(model, encoder),
                "model_state": model.state_dict(), "encoder_state": encoder.state_dict(),
                "unit_type_to_id": dict(vocab[0]), "faction_to_id": dict(vocab[1])}, path)
    return path


class InProcessTransport:
    """The shared-inference transport with the forward in this process:
    the CPU eval server's hello and `InferenceServer.infer_batch` as that
    server builds it (tests/test_generation_identity.py serves the same way)."""

    def __init__(self, spec: Path):
        from tools.eval_inference_server import server_hello
        from tools.eval_players import _load_policy
        from tools.eval_provenance import file_sha256
        from tools.inference_seam import InferenceServer
        cpu = torch.device("cpu")
        policy = _load_policy(Path(spec), cpu, label="inproc", infer_bf16=False, infer_compile=False)
        self._server = InferenceServer(policy._inference_model, policy._inference_encoder, device=cpu,
                                       output_device=cpu, autocast_bf16=False, packed_embed=True)
        self.hello = server_hello(str(spec), file_sha256(Path(spec)), policy._inference_encoder,
                                  policy._inference_model, device="cpu", bf16=False, packed=False,
                                  packed_embed=True, compiled=False, graphed=False, rust_stage=True,
                                  server_memory=False)
        self.broken = False
        self.address = "in-process"

    def infer_batch(self, items):
        return self._server.infer_batch(items)

    def close(self) -> None:
        pass


def _refusing_bounce(refuse: Sequence[int]):
    """The match loop's bounce check, answering yes for the recruit
    decisions whose running index (over the game, refused ones included)
    is in `refuse`."""
    import tools.eval_players as ep
    original = ep._timed_bounce
    count = {"recruits": 0}

    def bounce(action, gs) -> bool:
        if action.get("type") == "recruit":
            count["recruits"] += 1
            if count["recruits"] - 1 in refuse:
                return True
        return original(action, gs)
    return bounce


def game_stem(side_a: int, seed: int) -> str:
    """A match game's file stem (run_elo_batch.result_name)."""
    return f"game_{LABELS[0]}_{LABELS[1]}_s{side_a}_{seed}"


def selfplay_games(checkpoint: Path, out_dir: Path, slots: Sequence[Tuple[int, int]],
                   refuse: Optional[Mapping[int, Sequence[int]]] = None,
                   max_turns: int = MAX_TURNS) -> Dict[str, Path]:
    """Each (side A's side, seed) of `slots` played into `out_dir` as a
    stage 1 match plays it (module docstring), the recruit decisions of
    `refuse[seed]` refused; {stem: record path}."""
    import tools.elo_eval_game as eg
    from wesnoth_ai.rules.scenario_pool import random_setup
    refuse = refuse or {}
    transport = InProcessTransport(checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for side_a, seed in slots:
        argv = ["x", LABELS[0], str(checkpoint), LABELS[1], str(checkpoint), str(side_a), str(seed), str(out_dir),
                "--mcts-sims", "0", "--raw-temperature-a", "1", "--raw-temperature-b", "1",
                "--memory-a", str(SLOTS), "--memory-b", str(SLOTS), "--max-turns", str(max_turns),
                "--device", "cpu", "--inference-address-a", transport.address,
                "--inference-address-b", transport.address, "--no-infer-bf16", "--no-infer-compile",
                "--no-infer-packed-trunk", "--log-decisions"]
        with ExitStack() as patches:
            patches.enter_context(mock.patch.object(eg, "_shared_client", lambda address: transport))
            patches.enter_context(mock.patch.object(eg, "random_setup",
                                                    functools.partial(random_setup, mini_maps=True)))
            patches.enter_context(mock.patch("tools.eval_players._timed_bounce",
                                             _refusing_bounce(tuple(refuse.get(seed, ())))))
            if eg.main(argv) != 0:
                raise RuntimeError(f"elo_eval_game failed on slot {(side_a, seed)}")
        stem = game_stem(side_a, seed)
        records[stem] = out_dir / f"{stem}.game.jsonl.gz"
    return records


def named(games, table: Mapping) -> Dict:
    """A table keyed by the suite's names (`SUITE`: "g0", or ("g0", side))
    keyed by the games' stems instead."""
    def key(k):
        return (games.names[k[0]], *k[1:]) if isinstance(k, tuple) else games.names[k]
    return {key(k): v for k, v in table.items()}


def write_labels(path: Path, games: Dict[str, int], R: Dict[tuple, float], heldout: Sequence[str] = (),
                 halves: Optional[Dict[str, str]] = None) -> Path:
    """labels.jsonl rows (build_contracts.md §2) for `games` (name -> split)
    with R per (game, side); z, S, b and beta_S are placeholders that sum
    to R."""
    halves = halves or {}
    with open(path, "w", encoding="utf-8") as f:
        for game, split in games.items():
            for side in (1, 2):
                r = float(R[(game, side)])
                f.write(json.dumps({"game": game, "side": side, "split": int(split), "heldout": game in heldout,
                                    "half": halves.get(game) if game in heldout else None, "z": r,
                                    "capped": False, "cell": "test", "S": [0.0] * 7, "b": 0.0, "beta_S": 0.0,
                                    "R": r}) + "\n")
    return path


def command_positions(record: Path, vocab, limit: int = 12) -> List[tuple]:
    """(raw, masks) before the first `limit` player commands of a record,
    encoded as the player encodes (the recipe's encoding, the core's masks)."""
    from tools.game_record import read_records, walk_core
    from tools.preencode_sequences import ENCODING
    from wesnoth_ai.server_priors import core_packed_masks
    [rec] = list(read_records(record))
    out = []
    for _k, cs, cmd in walk_core(rec):
        player = int(cs.core.current_side) in (1, 2)
        if player and cmd[0] in ("move", "attack", "recruit", "end_turn") and len(out) < limit:
            raw = cs.encode_raw(type_to_id=vocab[0], faction_to_id=vocab[1], masks=True, **ENCODING)
            out.append((raw, core_packed_masks(raw)))
    return out


def initial_forward(model, encoder, raws, slots: int = SLOTS):
    """The network's padded output on `raws`, each at the learned initial
    memory."""
    staged = encoder.stage_raws(list(raws), device=torch.device("cpu"))
    memory = [model.initial_memory(slots) for _ in raws]
    return model.forward_embedded(encoder.embed_staged(staged), memory=memory)


def side_forwards(encoder, model, rows) -> List[tuple]:
    """Each decision of one game-side's positions (`pg_positions`) forwarded
    alone, in log order: (its padded output, its staged action batch), the
    memory carried as the player carried it (a shadow row reads it and
    writes nothing). Code of its own, beside the pass's and the evaluator's
    schedules."""
    from tools.pg_pass import stage_step
    cpu = torch.device("cpu")
    memory = model.initial_memory(SLOTS)
    out = []
    for p in rows:
        staged = encoder.stage_raws([p.raw], device=cpu)
        padded = model.forward_embedded(encoder.embed_staged(staged), memory=[memory])
        out.append((padded, stage_step([p], staged, model, cpu)))
        if not p.shadow:
            memory = padded.memory_padded[0, :SLOTS]
    return out


def moved_network(checkpoint: Path, direction: Dict[str, torch.Tensor], s: float):
    """(encoder, model) of `checkpoint` with s times `direction` added to
    the parameters it names ("model.<name>", "encoder.<name>")."""
    from tools.pg_pass import load_network
    encoder, model = load_network(checkpoint, torch.device("cpu"))
    modules = {"model": model, "encoder": encoder}
    with torch.no_grad():
        for name, d in direction.items():
            owner, _, rest = name.partition(".")
            modules[owner].get_parameter(rest).add_(s * d)
    return encoder, model


@functools.lru_cache(maxsize=None)
def played_games() -> SimpleNamespace:
    """The suite's games (`SUITE`), played once per process into a
    temporary directory: three games of one tiny network, the second with
    two of its recruits refused; `names` maps the suite's names to the
    games' stems, the names the records and labels go by."""
    from tools.eval_provenance import file_sha256
    root = Path(tempfile.mkdtemp(prefix="pg_games_"))
    atexit.register(shutil.rmtree, root, True)
    vocab, encoder, model = tiny_network()
    checkpoint = save_network(root / "net.pt", vocab, encoder, model)
    games = root / "games"
    slots = [(side_a, seed) for seed, side_a, _ in SUITE.values()]
    records = selfplay_games(checkpoint, games, slots, {seed: refuse for seed, _, refuse in SUITE.values()})
    names = {name: game_stem(side_a, seed) for name, (seed, side_a, _) in SUITE.items()}
    return SimpleNamespace(root=root, games=games, records=records, names=names, checkpoint=checkpoint,
                           sha256=file_sha256(checkpoint), vocab=vocab, encoder=encoder, model=model)
