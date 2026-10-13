"""The player's own game-sides beside the test pass's corpus
(tests/helpers/sequence_pass.py), built as round 2 built them in
miniature: a mini game a tiny network played against itself at 64 slots
through the match loop, selected, made candidates and labelled by the same
network at 0 slots (tools/own_trajectories.py, tools/own_targets.py), with
the corpus's vocabularies. With them the trainer runs its own path, as
production does on half its streams."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import List

SLOTS = 64
SHA = "d" * 64
OFFSET = -1.5


def _counting_player(base):
    """The raw player at the reference decode, counting its forwards as a
    match's result does (`forwards_<x>`)."""
    from tools.raw_player import RawPolicyPlayer

    class Counting(RawPolicyPlayer):
        forwards = 0

        def _forward(self, *a, **kw):
            self.forwards += 1
            return super()._forward(*a, **kw)
    return Counting(base, 0.0, memory_slots=SLOTS, end_turn_offset=OFFSET)


def _play(model, encoder, out_dir: Path) -> None:
    """One mini game of the network against itself at 64 slots, recorded
    as a match records it."""
    from sim_test_helpers import scenario_setup
    from tools.eval_players import _play_one_eval_game, _PolicyPair
    from tools.game_record import GameRecordLog, game_record
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate
    setup = scenario_setup(3, mini=True)
    sim = WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id, max_turns=3)
    sim._seed_salt = "own-pass"
    players = {}
    for side in (1, 2):
        base = SimpleNamespace(_inference_model=model, _inference_encoder=encoder, _lock=threading.Lock(),
                               _decision_step=0)
        players[side] = _counting_player(base)
    stem = "game_p_p_s1_7"
    _play_one_eval_game(sim, _PolicyPair(players[1], "p", 1), _PolicyPair(players[2], "p", 2), game_label=stem)
    GameRecordLog(out_dir / f"{stem}.game.jsonl.gz").write(game_record(
        sim, setup, game_label=stem, players={"a": {"label": "p", "side": 1, "memory": SLOTS},
                                              "b": {"label": "p", "side": 2, "memory": SLOTS}}))
    res = {"side_a": 1, "outcome_a": {1: "win", 2: "loss"}.get(sim.winner, "timeout")}
    for x, side in (("a", 1), ("b", 2)):
        res.update({f"checkpoint_sha256_{x}": SHA, f"memory_{x}": SLOTS, f"forwards_{x}": players[side].forwards,
                    f"raw_end_turn_offset_{x}": OFFSET})
    (out_dir / f"{stem}.json").write_text(json.dumps(res), encoding="utf-8")


def own_arguments(tmp_path: Path, pass_args: List[str]) -> List[str]:
    """The pass's arguments with `--own-sequences` naming the own
    game-sides built in `tmp_path`."""
    import torch
    from tools.inference_seam import InferenceServer
    from tools.own_targets import label_games, own_manifest
    from tools.own_trajectories import candidates_of, read_game_record, select_own_sides
    from tools.preencode_sequences import MANIFEST_NAME
    from tools.sequence_setup import build_modules
    corpus = json.loads((Path(pass_args[pass_args.index("--sequences") + 1]) / MANIFEST_NAME)
                        .read_text(encoding="utf-8"))
    vocab = (corpus["unit_type_to_id"], corpus["faction_to_id"])
    torch.manual_seed(4)
    encoder, model = build_modules(*vocab, torch.device("cpu"),
                                   {"d_model": 32, "num_layers": 1, "num_heads": 2, "d_ff": 64})
    encoder.eval()
    model.eval()
    games_dir, own_dir = tmp_path / "own_games", tmp_path / "own"
    games_dir.mkdir()
    own_dir.mkdir()
    _play(model, encoder, games_dir)
    games, selection = select_own_sides([games_dir], SHA, SLOTS)
    [(name, sides)] = games.items()
    game = candidates_of(read_game_record(Path(sides[0].record)), sides, *vocab)
    with torch.no_grad():
        counts = label_games(InferenceServer(model, encoder, device=torch.device("cpu")), SLOTS, [game], own_dir,
                             reference_offset=OFFSET)
    if counts["game_sides_kept"] != 2:
        raise RuntimeError(f"the own game kept {counts['game_sides_kept']} game-sides, not 2: {dict(counts)}")
    cand = {**{k: corpus[k] for k in ("fingerprint", "encoding", "observation_epoch", "core_phase",
                                      "unit_type_to_id", "faction_to_id")},
            "teacher": "tiny", "teacher_sha256": SHA, "slots": SLOTS, "selection": dict(selection),
            "errors": {}, "games": {name: {"holdout": False}}}
    (own_dir / MANIFEST_NAME).write_text(json.dumps(own_manifest(cand, {name: counts}, 16, OFFSET)),
                                         encoding="utf-8")
    return [*pass_args, "--own-sequences", str(own_dir)]
