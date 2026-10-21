"""Self-play games with decision logs, for the policy-gradient pass's tests
(docs/training_design_20261019/build_contracts.md §1-2).

Branch A's logger is not merged, so this helper writes the contract's
format itself: a tiny network plays mini games against itself at
temperature 1, no end_turn offset, 64 memory slots, through the production
play path (the remote encoder with server-side priors and the inference
server), and a subclass of the raw player logs every decision, refused ones
included. A harness may refuse chosen recruit decisions: it rejects the
ordered hex and the player decides again, as the match loop does for a
recruit the engine bounces (`eval_players._play_one_eval_game`).

`played_games()` plays the suite's games once per process and returns where
they are.
"""
from __future__ import annotations

import atexit
import functools
import json
import math
import shutil
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

SLOTS = 64
TINY_ARCH = {"d_model": 32, "num_layers": 1, "num_heads": 2, "d_ff": 64}
MAX_TURNS = 3
PROCEDURE = "raw:t1"                 # the raw player at temperature 1, no end_turn offset


def tiny_network(seed: int = 4):
    """(vocabularies, encoder, model) of a tiny network in eval mode."""
    from tools.preencode_sequences import fresh_vocab
    from tools.sequence_setup import build_modules
    torch.manual_seed(seed)
    vocab = fresh_vocab()
    encoder, model = build_modules(*vocab, torch.device("cpu"), TINY_ARCH)
    return vocab, encoder.eval(), model.eval()


def save_network(path: Path, vocab, encoder, model) -> Path:
    """The network as a checkpoint the pass and the evaluator load."""
    torch.save({"arch": dict(TINY_ARCH), "model_state": model.state_dict(), "encoder_state": encoder.state_dict(),
                "unit_type_to_id": dict(vocab[0]), "faction_to_id": dict(vocab[1])}, path)
    return path


def decision_entry(compact, pick: int, value: float) -> dict:
    """The contract's log entry of choosing `pick` among the compact legal
    actions, its side, command and refusal left for the caller."""
    from wesnoth_ai.server_priors import KIND_ATTACK, KIND_END_TURN, KIND_RECRUIT
    prior = np.asarray(compact.prior, dtype=np.float64)
    kind = np.asarray(compact.kind)
    actor = np.asarray(compact.actor)
    mass = float(prior.sum())
    law = prior / mass
    own = actor == actor[pick]
    attacks = own & (kind == KIND_ATTACK)
    return {
        "action": [int(actor[pick]), int(kind[pick]), int(compact.target[pick]), int(compact.weapon[pick])],
        "n": int(len(prior)), "mass": mass, "logp": float(np.log(law[pick])),
        "H": float(-(law * np.log(law)).sum()),
        "p_end": float(law[kind == KIND_END_TURN].sum()), "p_att": float(law[kind == KIND_ATTACK].sum()),
        "p_rec": float(law[kind == KIND_RECRUIT].sum()),
        "p_att_actor": float(law[attacks].sum() / law[own].sum()) if attacks.any() else math.nan,
        "cdf": [float(law[:pick].sum()), float(law[:pick + 1].sum())], "v": float(value)}


def logging_player(base, seed: int):
    """The raw player at temperature 1 with 64 slots, logging each decision
    into `player.log[game_label]`."""
    from tools.raw_player import RawPolicyPlayer
    from wesnoth_ai.server_priors import KIND_END_TURN, compact_action

    class LoggingPlayer(RawPolicyPlayer):
        def __init__(self):
            super().__init__(base, 1.0, seed=seed, memory_slots=SLOTS)
            self.log: Dict[str, List[dict]] = {}
            self._entry: Optional[dict] = None
            self._value = math.nan

        def _forward(self, base_, encoded, game_label, game_state):
            out = super()._forward(base_, encoded, game_label, game_state)
            self._value = float(out.value.reshape(-1)[0])
            return out

        def _select_compact(self, compact, encoded):
            priors = np.asarray(compact.prior, dtype=np.float64)
            pick = self._choose(priors, np.asarray(compact.actor), np.asarray(compact.kind) == KIND_END_TURN)
            self._entry = decision_entry(compact, pick, self._value)
            return compact_action(compact, pick, encoded)

        def select_action(self, game_state, *, game_label="default", sim=None):
            side, slot = int(game_state.global_info.current_side), len(sim.command_history)
            action = super().select_action(game_state, game_label=game_label, sim=sim)
            self.log.setdefault(game_label, []).append(
                {"side": side, "cmd": slot, "refused": False, **self._entry})
            return action

        def drop_last_pending(self, game_label):
            entry = self.log[game_label][-1]
            entry["refused"], entry["cmd"] = True, -1
            return super().drop_last_pending(game_label)

    return LoggingPlayer()


def play_logged_game(model, encoder, vocab, out_dir: Path, stem: str, scenario_seed: int, checkpoint_sha256: str,
                     refuse_recruits: Sequence[int] = (), max_turns: int = MAX_TURNS) -> Path:
    """One mini game of the network against itself, its record (with the
    `decisions` field) written to `out_dir/<stem>.game.jsonl.gz` and its
    match result, naming `checkpoint_sha256` on both seats, beside it. The
    recruit decisions whose running index is in `refuse_recruits` are
    refused by the harness."""
    from sim_test_helpers import scenario_setup
    from tools.game_record import GameRecordLog, game_record
    from tools.inference_seam import InferenceServer, RemoteEncoder, RemoteModel
    from tools.preencode_sequences import ENCODING
    from tools.selfplay_game import _would_recruit_bounce
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate
    setup = scenario_setup(scenario_seed, mini=True)
    sim = WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id, max_turns=max_turns)
    sim._seed_salt = stem
    remote = RemoteEncoder(*vocab, device=torch.device("cpu"), server_priors=True, **ENCODING)
    base = SimpleNamespace(_inference_model=RemoteModel(InferenceServer(model, encoder, device=torch.device("cpu"))),
                           _inference_encoder=remote, _lock=threading.Lock(), _decision_step=0)
    player = logging_player(base, seed=scenario_seed)
    recruits = 0

    def refused(action) -> bool:
        nonlocal recruits
        if action.get("type") != "recruit":
            return False
        recruits += 1
        return recruits - 1 in refuse_recruits or _would_recruit_bounce(action, sim.gs)

    while not sim.done:
        action = player.select_action(sim.gs, game_label=stem, sim=sim)
        while refused(action):
            sim.reject_recruit_hex(action["target_hex"].x, action["target_hex"].y)
            player.drop_last_pending(stem)
            action = player.select_action(sim.gs, game_label=stem, sim=sim)
        sim.step(action)
        if sim.last_step_rejected:
            raise RuntimeError(f"{stem}: the simulator refused {action!r}")
    path = out_dir / f"{stem}.game.jsonl.gz"
    players = {x: {"label": "p", "side": s, "memory": SLOTS, "procedure": PROCEDURE} for x, s in (("a", 1), ("b", 2))}
    GameRecordLog(path).write(game_record(sim, setup, game_label=stem, players=players,
                                          extra={"decisions": player.log[stem]}))
    result = {"side_a": 1, "outcome_a": {1: "win", 2: "loss"}.get(sim.winner, "timeout")}
    for x in ("a", "b"):
        result.update({f"checkpoint_sha256_{x}": checkpoint_sha256, f"memory_{x}": SLOTS, f"procedure_{x}": PROCEDURE})
    (out_dir / f"{stem}.json").write_text(json.dumps(result), encoding="utf-8")
    return path


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
    """The suite's games, played once per process into a temporary
    directory: three games of one tiny network, the second with two of its
    recruits refused, and its checkpoint."""
    from tools.eval_provenance import file_sha256
    root = Path(tempfile.mkdtemp(prefix="pg_games_"))
    atexit.register(shutil.rmtree, root, True)
    vocab, encoder, model = tiny_network()
    checkpoint = save_network(root / "net.pt", vocab, encoder, model)
    games = root / "games"
    games.mkdir()
    plan = {"g0": (3, ()), "g1": (5, (0, 2)), "g2": (11, ())}
    records = {stem: play_logged_game(model, encoder, vocab, games, stem, seed, file_sha256(checkpoint), refuse)
               for stem, (seed, refuse) in plan.items()}
    return SimpleNamespace(root=root, games=games, records=records, checkpoint=checkpoint, vocab=vocab,
                           encoder=encoder, model=model)
