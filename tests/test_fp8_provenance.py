"""The fp8 inference switch's provenance (run_elo_batch --fp8-inference-a /
-b, the server's --fp8-inference): each side's switch rides its game
records, a slot played under the other value is refused, the driver refuses
the switch off the servers or off the bf16 stream, the two sides take one
server group unless one side alone plays fp8, and the collector reads a
record without the fields as played without fp8."""
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wesnoth_ai.constants import OBSERVATION_EPOCH  # noqa: E402
from tests.helpers.eval_records import current_forced_faction  # noqa: E402


def test_a_slot_played_with_fp8_refuses_a_replay_without_it(tmp_path):
    from tools.elo_eval_game import main
    out = tmp_path / "out"
    out.mkdir()
    prev = {"procedure_a": "mcts:32", "procedure_b": "mcts:32", "max_turns": 200,
            "combat_stream": "per_game", "observation_epoch": OBSERVATION_EPOCH,
            "forced_faction": current_forced_faction()}
    (out / "game_A_B_s1_7.json").write_text(json.dumps(prev), encoding="utf-8")
    assert main(["x", "A", "dummy", "B", "dummy", "1", "7", str(out), "--no-turn-search", "--device", "cpu"]) == 0
    (out / "game_A_B_s1_8.json").write_text(json.dumps({**prev, "infer_fp8_b": True}), encoding="utf-8")
    with pytest.raises(SystemExit, match="infer_fp8_b"):
        main(["x", "A", "dummy", "B", "dummy", "1", "8", str(out), "--no-turn-search", "--device", "cpu"])


def test_the_driver_refuses_an_outdir_of_fp8_games_for_a_run_without_it(tmp_path):
    from tools.run_elo_batch import main
    out = tmp_path / "out"
    out.mkdir()
    (out / "game_A_B_s1_1000.json").write_text(json.dumps(
        {"procedure_a": "mcts:0", "procedure_b": "mcts:0", "max_turns": 200, "infer_fp8_a": True,
         "combat_stream": "per_game", "observation_epoch": OBSERVATION_EPOCH,
         "forced_faction": current_forced_faction()}), encoding="utf-8")
    with pytest.raises(SystemExit, match="infer_fp8_a"):
        main(["x", "--label-a", "A", "--spec-a", "dummy", "--label-b", "B", "--spec-b", "dummy",
              "--outdir", str(out), "--games", "2", "--seed-base", "1000", "--device", "cpu"])


@pytest.mark.parametrize("extra,why", [([], "shared-inference"), (["--shared-inference", "--persistent-workers",
                                                                     "--raw-temperature-a", "0", "--mcts-sims", "0"],
                                                                    "bf16-residual")])
def test_the_driver_refuses_fp8_off_the_servers_or_off_the_bf16_stream(tmp_path, capsys, extra, why):
    from tools.run_elo_batch import main
    spec = tmp_path / "a.pt"
    spec.write_bytes(b"not read before the arguments are checked")
    with pytest.raises(SystemExit):
        main(["x", "--label-a", "A", "--spec-a", str(spec), "--label-b", "B", "--spec-b", "dummy",
              "--outdir", str(tmp_path / "out"), "--games", "2", "--fp8-inference-a", *extra])
    assert why in capsys.readouterr().err


def test_the_sides_share_a_server_group_unless_one_alone_plays_fp8():
    from tools.run_elo_batch import server_keys, server_label
    both = server_keys(Namespace(spec_a="m.pt", spec_b="m.pt", fp8_inference_a=True, fp8_inference_b=True))
    assert len(set(both.values())) == 1
    one = server_keys(Namespace(spec_a="m.pt", spec_b="m.pt", fp8_inference_a=True, fp8_inference_b=False))
    assert one == {"a": ("m.pt", True), "b": ("m.pt", False)}
    assert server_label(one["a"]) == "m.pt#fp8" and server_label(one["b"]) == "m.pt"
    assert server_keys(Namespace(spec_a="m.pt", spec_b="dummy", fp8_inference_a=False,
                                 fp8_inference_b=False)) == {"a": ("m.pt", False)}


def test_a_record_without_the_fields_reads_as_played_without_fp8():
    from tools.elo_collect import ESTIMAND_DEFAULTS
    assert ESTIMAND_DEFAULTS["infer_fp8_a"] is False and ESTIMAND_DEFAULTS["infer_fp8_b"] is False
