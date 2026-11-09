"""The precision floor's serving precision (tools/pg_floor.py,
tools/pg_positions.py): a match result names the precision its game was
served in, and the floor reads that one. A bfloat16 floor on games served in
float16 (the local RX 6600) would loosen the replay limit by a rounding the
logs do not carry, a float16 floor on a box's bfloat16 games would tighten it
below theirs: both are refused, as is one pass over games of two serving
precisions. Games served in float32 read the path alone. CPU only: no
network, no core."""
from __future__ import annotations

import pytest
import torch

from tools.pg_floor import (FLOOR_DTYPES, FLOOR_LIMIT_CAP, PATH_KL_LIMIT, PATH_ONLY, FloorDtypeError, FloorReader,
                            refusal, replay_limit, resolve_floor_dtype)
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


def test_a_result_served_on_other_numerics_names_them():
    """The server's numerics paths beside its autocast (elo_eval_game's
    infer_bf16_residual, infer_fused_relu, infer_fp8_a/b) are other
    roundings than the plain autocast the floor reads: the precision names
    them, and the floor refuses them under every choice but off."""
    assert serving_precision({"infer_bf16": True, "infer_bf16_residual": True}) == "bfloat16+infer_bf16_residual"
    assert serving_precision({"infer_bf16": True, "infer_fused_relu": True}) == "bfloat16+infer_fused_relu"
    assert serving_precision({"infer_bf16": True, "infer_bf16_residual": True, "infer_fp8_b": True}) == \
        "bfloat16+infer_bf16_residual+infer_fp8_b"
    assert serving_precision({"infer_fp16": True, "infer_bf16_residual": False, "infer_fused_relu": False,
                              "infer_fp8_a": False, "infer_fp8_b": False}) == "float16"
    for requested in ("auto", "bfloat16", "float16"):
        with pytest.raises(FloorDtypeError, match="numerics the floor does not read"):
            resolve_floor_dtype(requested, ["bfloat16", "bfloat16+infer_fp8_a"])
    assert resolve_floor_dtype("off", ["bfloat16+infer_fused_relu"]) == "off"


@pytest.mark.parametrize("requested, servings, read", [
    ("auto", ["float16"] * 3, "float16"),
    ("auto", ["bfloat16"] * 3, "bfloat16"),
    ("auto", ["float32", "float32"], PATH_ONLY),     # no rounding to read, the path still read
    ("auto", [], PATH_ONLY),
    ("auto", ["float16", None], "float16"),          # a game without a result: refuse_foreign's to refuse
    ("float16", ["float16"], "float16"),
    ("bfloat16", ["bfloat16"], "bfloat16"),
    ("bfloat16", ["float32"], "bfloat16"),           # the suite's CPU games: the caller's choice
    ("float16", ["float32"], "float16"),
    ("float16", ["float16", "float32"], "float16"),  # named: the caller's choice, as for float32 alone
    ("off", ["float16", "bfloat16"], "off"),
])
def test_the_floor_reads_the_games_serving_precision(requested, servings, read):
    assert resolve_floor_dtype(requested, servings) == read


@pytest.mark.parametrize("requested, servings, match", [
    ("bfloat16", ["float16", "float16"], "served in float16"),
    ("float16", ["bfloat16"], "served in bfloat16"),
    ("auto", ["float16", "bfloat16"], "bfloat16 and float16"),
    ("auto", ["float16", "float32", None], "float16 and float32"),
    ("auto", ["float32", "bfloat16"], "bfloat16 and float32"),
    ("float16", ["float16", "bfloat16"], "served in bfloat16"),
    ("float32", ["float32"], "is none of"),
])
def test_a_floor_of_another_precision_than_the_games_is_refused(requested, servings, match):
    with pytest.raises(FloorDtypeError, match=match):
        resolve_floor_dtype(requested, servings)


def test_games_served_in_float32_keep_the_path_check():
    """The floor of games served in float32 reads no rounding (they carry
    none, so the base limit holds and nothing is lifted past the cap), and
    still reads the path: the packed and padded trunks' fp32 agreement,
    which has nothing to do with the serving precision."""
    reader = FloorReader(None, None, torch.device("cpu"), set(), PATH_ONLY)
    assert reader.serving is None                    # no serving forward is run
    reader.merge({("g0", 1): [0.0, 2e-10, 50, 1e-11], ("g1", 2): [0.0, 4e-10, 30, 3e-11]})
    stats = reader.stats()
    assert stats["dtype"] == PATH_ONLY and stats["kl"] is None and stats["se"] is None and stats["upper"] is None
    assert stats["path_kl"] == pytest.approx(6e-10 / 80) and stats["path_max"] == 3e-11
    assert stats["decisions"] == 80 and stats["game_sides"] == 2 and stats["games"] == 2
    assert replay_limit(1e-4, stats) == 1e-4
    assert refusal(stats, 1e-4) is None
    assert refusal(stats, 2 * FLOOR_LIMIT_CAP) is None   # a base above the cap is the caller's, as with no floor
    departed = dict(stats, path_kl=2 * PATH_KL_LIMIT, path_max=5 * PATH_KL_LIMIT)
    assert refusal(departed, 1e-4).startswith("path: the packed and padded trunks disagree")


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
