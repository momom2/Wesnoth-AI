"""tools/analysis/memory_counterfactual.py reads a memory checkpoint's
decisions with its memory carried and at 0 slots: on a match record it
must reproduce every choice of the player that played, and on a corpus game
it must find every label among the legal actions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.analysis import memory_counterfactual as mc  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
D = 32


def _memory_policy(tmp_path, slots=8):
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(1)
    spec = tmp_path / "memory.pt"
    TransformerPolicy(device=torch.device("cpu"), d_model=D, num_layers=1, num_heads=2, d_ff=64,
                      relevant_set_hexes=True, observation_parity=True, memory_slots=slots,
                      relevant_set_version=2).save_checkpoint(spec)
    return spec


@needs_core
def test_a_corpus_game_is_read_decision_by_decision(tmp_path):
    from helpers.parity_games import record
    from tools.abilities import hex_neighbors
    enemy = (9, 2)
    near = next(h for h in hex_neighbors(*enemy) if h[0] == 8)
    start = next(h for h in hex_neighbors(*near) if h != enemy and h not in hex_neighbors(*enemy))
    data = record([("Lieutenant", 1, 1, 3, True), ("Spearman", 1, *start, False),
                   ("Lieutenant", 2, 18, 3, True), ("Spearman", 2, *enemy, False)], fog=False)
    data["commands"] = [
        ["init_side", 1], ["move", [start[0], near[0]], [start[1], near[1]], 1],
        ["attack", near[0], near[1], 9, 2, 0, -1, "1"], ["end_turn"],
        ["init_side", 2], ["move", [18, 17], [3, 3], 2], ["end_turn"],
        ["init_side", 1], ["end_turn"]]
    spec = _memory_policy(tmp_path)
    policy, slots = mc.load_policy(spec, torch.device("cpu"), bf16=False)
    rows = list(mc.human_rows(policy, data, "g", winner=1, slots=slots, offset=0.0))
    assert [r["rec"] for r in rows] == ["move", "attack", "end_turn", "move", "end_turn", "end_turn"]
    assert all(r["rec_prior_m"] is not None and r["rec_prior_z"] is not None for r in rows), \
        "every label is one of the legal actions"
    assert [(r["side"], r["turn"], r["t"]) for r in rows] == [(1, 1, 0), (1, 1, 1), (1, 1, 2),
                                                             (2, 1, 0), (2, 1, 1), (1, 2, 0)]
    assert [r["player"] for r in rows] == ["winner"] * 3 + ["loser"] * 2 + ["winner"]
    assert all(r["mem_abs"] > 0 for r in rows)


def test_a_move_then_an_attack_on_the_chosen_target_is_one_decision():
    from wesnoth_ai.classes import Position
    commands = [["move", [3, 4, 5], [2, 2, 3], 1], ["attack", 5, 3, 6, 3, 1, -1, "s"], ["end_turn"]]
    attack = {"type": "attack", "start_hex": Position(3, 2), "target_hex": Position(6, 3), "attack_index": 1}
    skip = set()
    assert mc._recorded(commands, 0, attack, skip) == (("attack", 3, 2, 6, 3, 1), "attack")
    assert skip == {1}
    move = {"type": "move", "start_hex": Position(3, 2), "target_hex": Position(5, 3)}
    skip = set()
    assert mc._recorded(commands, 0, move, skip) == (("move", 3, 2, 5, 3), "move") and not skip
    elsewhere = dict(attack, target_hex=Position(7, 4))
    assert mc._recorded(commands, 0, elsewhere, set()) == (("move", 3, 2, 5, 3), "move"), \
        "an attack on another target was cut short: the move is all that was recorded"


@needs_core
@pytest.mark.slow
def test_a_match_record_reproduces_the_player_that_played(tmp_path, monkeypatch):
    """A real game between the checkpoint at 8 slots and at 0, both deep in
    the turn (end_turn logit offset -6); read for each player, the reading
    that played reproduces every recorded decision, one row per forward the
    player made, and the memory player's state after each decision is the
    one it wrote in play."""
    from tools import raw_player
    from tools.elo_eval_game import main
    from tools.game_record import read_records
    written = []
    real_forward = raw_player.RawPolicyPlayer._forward

    def forward(self, base, encoded, game_label, game_state):
        output = real_forward(self, base, encoded, game_label, game_state)
        if self.memory_slots:
            written.append(float(output.memory.abs().mean()))
        return output

    monkeypatch.setattr(raw_player.RawPolicyPlayer, "_forward", forward)
    spec = _memory_policy(tmp_path)
    out = tmp_path / "games"
    assert main(["x", "A", str(spec), "B", str(spec), "1", "7", str(out), "--mcts-sims", "0",
                 "--raw-temperature-a", "0", "--raw-temperature-b", "0", "--memory-a", "8",
                 "--memory-b", "0", "--raw-end-turn-offset-a", "-6",
                 "--raw-end-turn-offset-b", "-6", "--max-turns", "3", "--device", "cpu"]) == 0
    result = json.loads((out / "game_A_B_s1_7.json").read_text(encoding="utf-8"))
    for player, forwards in (("A", result["forwards_a"]), ("B", result["forwards_b"])):
        rows_path = tmp_path / f"{player}.jsonl"
        assert mc.main(["--checkpoint", str(spec), "--games", str(out), "--player", player,
                        "--offset", "-6", "--device", "cpu", "--out", str(rows_path)]) == 0
        rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
        rec = next(read_records(out / "game_A_B_s1_7.game.jsonl.gz"))
        side = rec["players"]["a" if player == "A" else "b"]["side"]
        refused = sum(1 for k, x, y in rec.get("rejections", ()) if _side_at(rec, int(k)) == side)
        assert len(rows) == forwards - refused >= 20
        assert all(r["repro"] for r in rows), [r for r in rows if not r["repro"]][:3]
        assert {r["played"] for r in rows} == {8 if player == "A" else 0}
        if player == "A":
            assert [r["mem_abs"] for r in rows] == pytest.approx(written, rel=1e-6)


def _side_at(rec, index: int) -> int:
    side = 0
    for cmd in rec["commands"][:index]:
        if cmd[0] == "init_side":
            side = int(cmd[1])
    return side


@needs_core
@pytest.mark.slow
def test_shards_run_as_processes_give_the_rows_of_one_process(tmp_path):
    import gzip
    from helpers.parity_games import record
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    entries = []
    for g, (y1, y2) in enumerate(((2, 3), (4, 1), (3, 3))):
        data = record([("Lieutenant", 1, 1, y1, True), ("Lieutenant", 2, 18, y2, True)], fog=False)
        data["commands"] = [["init_side", 1], ["move", [1, 2], [y1, y1], 1], ["end_turn"],
                            ["init_side", 2], ["end_turn"]]
        name = f"g{g}.json.gz"
        (corpus / name).write_bytes(gzip.compress(json.dumps(data).encode("utf-8")))
        entries.append({"file": name, "winner_side": 1 + g % 2, "holdout": True})
    (corpus / "manifest.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    spec = _memory_policy(tmp_path)
    common = ["--checkpoint", str(spec), "--corpus", str(corpus), "--offset", "-1.5", "--device", "cpu"]
    assert mc.main([*common, "--out", str(tmp_path / "one.jsonl")]) == 0
    assert mc.main([*common, "--procs", "2", "--out", str(tmp_path / "two.jsonl")]) == 0
    one, two = (_rows_by_decision(tmp_path / name) for name in ("one.jsonl", "two.jsonl"))
    assert len(one) == 9 and one.keys() == two.keys()
    for key, row in one.items():
        # Floats to a relative 1e-5: a process's torch thread count orders its sums
        # (tools/elo_eval_game sets 2 in this one when a match test ran first).
        assert two[key] == {k: pytest.approx(v, rel=1e-5) if isinstance(v, float) else v
                            for k, v in row.items()}, key
    assert not list(tmp_path.glob("two.jsonl.part*"))


def _rows_by_decision(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {(r["game"], r["side"], r["turn"], r["t"]): r for r in rows}


def test_the_readout_splits_disagreements_and_applies_the_rules():
    from tools.analysis import memory_counterfactual_readout as ro

    def row(game, turn, m, z, p_end_m=0.2, p_end_z=0.1):
        return {"src": "match", "game": game, "turn": turn, "choice_m": m, "choice_z": z,
                "agree": m == z, "p_end_m": p_end_m, "p_end_z": p_end_z, "max_act_m": 0.3,
                "max_act_z": 0.3, "repro": True, "rec": "move"}
    rows = [row("a", 7, "end_turn", "move"), row("a", 7, "end_turn", "attack"), row("b", 8, "move", "end_turn"),
            row("b", 9, "move", "attack"), row("b", 9, "move", "move"), row("a", 2, "end_turn", "move")]
    late = ro.read_source(rows)["late (6+)"]
    assert (late["decisions"], late["disagreements"]) == (5, 4)
    assert (late["only_m_ends"], late["only_z_ends"], late["both_act"]) == (2, 1, 1)
    assert late["agreement"] == 0.2
    verdict = ro.verdicts({"own": ro.read_source(rows)})
    assert verdict["tool_check"] and verdict["T_rows"] and not verdict["Q_rows"]
    assert late["p_end_diff"][0] == pytest.approx(0.1)
