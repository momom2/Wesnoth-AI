"""The precision floor's serving precision (tools/pg_floor.py,
tools/pg_positions.py): a match result names the precision its game was
served in, and the floor reads that one. A bfloat16 floor on games served in
float16 (the local RX 6600) would loosen the replay limit by a rounding the
logs do not carry, a float16 floor on a box's bfloat16 games would tighten it
below theirs: both are refused, as is one pass over games of both. CPU only:
no network, no core."""
from __future__ import annotations

import pytest

from tools.pg_floor import FLOOR_DTYPES, FloorDtypeError, resolve_floor_dtype
from tools.pg_positions import serving_precision


def test_a_result_names_the_precision_its_game_was_served_in():
    """elo_eval_game's fields: infer_fp16 under --infer-fp16 (infer_bf16 then
    False), infer_bf16 under the box's autocast, neither for fp32 serving or
    a result older than the fields; no result, no precision."""
    assert serving_precision({"infer_fp16": True, "infer_bf16": False}) == "float16"
    assert serving_precision({"infer_bf16": True, "infer_fp16": False}) == "bfloat16"
    assert serving_precision({"infer_bf16": True}) == "bfloat16"
    assert serving_precision({"infer_bf16": False, "infer_fp16": False}) == "float32"
    assert serving_precision({}) == "float32"
    assert serving_precision(None) is None


@pytest.mark.parametrize("requested, servings, read", [
    ("auto", ["float16"] * 3, "float16"),
    ("auto", ["bfloat16"] * 3, "bfloat16"),
    ("auto", ["float32", "float32"], "off"),
    ("auto", [], "off"),
    ("auto", ["float16", None], "float16"),          # a game without a result: refuse_foreign's to refuse
    ("float16", ["float16"], "float16"),
    ("bfloat16", ["bfloat16"], "bfloat16"),
    ("bfloat16", ["float32"], "bfloat16"),           # the suite's CPU games: the caller's choice
    ("float16", ["float32"], "float16"),
    ("off", ["float16", "bfloat16"], "off"),
])
def test_the_floor_reads_the_games_serving_precision(requested, servings, read):
    assert resolve_floor_dtype(requested, servings) == read


@pytest.mark.parametrize("requested, servings, match", [
    ("bfloat16", ["float16", "float16"], "served in float16"),
    ("float16", ["bfloat16"], "served in bfloat16"),
    ("auto", ["float16", "bfloat16"], "bfloat16 and float16"),
    ("float16", ["float16", "bfloat16"], "served in bfloat16"),
    ("float32", ["float32"], "is none of"),
])
def test_a_floor_of_another_precision_than_the_games_is_refused(requested, servings, match):
    with pytest.raises(FloorDtypeError, match=match):
        resolve_floor_dtype(requested, servings)


def test_the_command_line_takes_the_games_precision_by_default():
    """pg_pass's --floor-dtype offers every choice and defaults to auto; the
    API's default stays bfloat16 (a named precision, refused on fp16 games)."""
    import argparse
    from tools import pg_pass
    from tools.pg_pass import PassConfig
    assert set(FLOOR_DTYPES) == {"bfloat16", "float16", "auto", "off"}
    assert PassConfig().floor_dtype == "bfloat16"
    seen = {}
    real = argparse.ArgumentParser.parse_args

    def capture(self, argv=None, namespace=None):
        args = real(self, argv, namespace)
        seen.update(vars(args))
        raise SystemExit(0)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(argparse.ArgumentParser, "parse_args", capture)
        with pytest.raises(SystemExit):
            pg_pass.main(["--checkpoint", "x.pt", "--games", "d", "--labels", "l.jsonl", "--out", "p.pt"])
    assert seen["floor_dtype"] == "auto"
