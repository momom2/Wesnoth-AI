"""The decode-knob search driver (tools/knob_search.py,
docs/decode_knobs_prereg_20261012.md): the Latin hypercube is reproducible,
stratified and in range; successive halving picks the survivors the
pre-registration names from synthetic results, on the seed bases it names;
a search cut mid-stage resumes without replaying a finished vector; the null
arm and a changed directory are refused."""
from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import knob_search as ks  # noqa: E402
from tools import reference_player  # noqa: E402
from tools.eval_procedure import KNOB_VECTOR_KEYS, knob_vector  # noqa: E402


def test_the_hypercube_is_reproducible_stratified_and_in_range():
    n = ks.N_VECTORS - 1
    raw = ks.latin_hypercube(n, ks.RANGES, ks.LHS_SEED, decimals=None)
    assert raw == ks.latin_hypercube(n, ks.RANGES, ks.LHS_SEED, decimals=None)
    assert raw != ks.latin_hypercube(n, ks.RANGES, ks.LHS_SEED + 1, decimals=None)
    for key, (lo, hi) in ks.RANGES.items():
        strata = sorted(math.floor((p[key] - lo) / (hi - lo) * n) for p in raw)
        assert strata == list(range(n)), key                 # one point per stratum
    rounded = ks.latin_hypercube(n, ks.RANGES, ks.LHS_SEED)
    for point, exact in zip(rounded, raw):
        assert list(point) == list(KNOB_VECTOR_KEYS)
        for key, (lo, hi) in ks.RANGES.items():
            assert lo <= point[key] <= hi and point[key] == round(exact[key], 2)
    # The vectors the search plays (recorded in its search.jsonl before any game).
    assert rounded[0] == {"raw_end_turn_offset": -2.32, "raw_attack_offset": 1.04,
                          "raw_recruit_offset": -0.37, "raw_attack_tod_slope": 2.36}
    vectors = ks.search_vectors(knob_vector(reference_player.load()["decode"]))
    assert len(vectors) == ks.N_VECTORS and vectors[1:] == rounded
    assert vectors[0] == {"raw_end_turn_offset": -1.5, "raw_attack_offset": 0.0,
                          "raw_recruit_offset": 0.0, "raw_attack_tod_slope": 0.0}


class FakeBatch:
    """Stands in for run_elo_batch: writes the result files of a match
    (slots S..S+games-1; a capped slot is replaced at S+i+1e6, side kept),
    skipping the ones on disk, the outcome of slot i decided by
    `wins_of(label, seed_base)` (the first that many slots are wins).
    `stop_after` raises after that many matches, half of the last one
    written: a box cut mid-match."""

    def __init__(self, wins_of, stop_after=None):
        self.wins_of, self.stop_after, self.calls = wins_of, stop_after, []

    def __call__(self, argv, log_path):
        opt = {argv[k]: argv[k + 1] for k in range(len(argv) - 1) if argv[k].startswith("--")}
        label, outdir = opt["--label-a"], Path(opt["--outdir"])
        games, base = int(opt["--games"]), int(opt["--seed-base"])
        self.calls.append((label, base))
        outdir.mkdir(parents=True, exist_ok=True)
        wins = self.wins_of(label, base)
        cut = self.stop_after is not None and len(self.calls) > self.stop_after
        for i in range(games // 2 if cut else games):
            seeds = [base + i] if i % 7 else [base + i, base + i + ks.REPL_SEED_OFFSET]
            for k, seed in enumerate(seeds):
                path = outdir / f"game_{label}_{opt['--label-b']}_s{1 + i % 2}_{seed}.json"
                outcome = "timeout" if k + 1 < len(seeds) else ("win" if i < wins else "loss")
                if not path.exists():
                    path.write_text(json.dumps({"label_a": label, "seed": seed, "outcome_a": outcome}),
                                    encoding="utf-8")
        if cut:
            raise KeyboardInterrupt("the box was cut")
        return 0


# The pre-registration's stages, seed bases and survivor counts with fewer
# games, so the synthetic results stay small.
STAGES = tuple(replace(s, games=g) for s, g in zip(ks.STAGES, (20, 30, 40)))
CONFIRM = 40
# Stage-1 wins of kv00..kv23 out of 20: kv00 at 10 (the null arm), kv05
# and kv09 tied at 16 (kv05 first), kv11 at 18; the 8th place goes to kv03
# over kv04 at 11 apiece (the lower index).
STAGE1 = {0: 10, 1: 8, 2: 12, 3: 11, 4: 11, 5: 16, 6: 6, 7: 15, 8: 4, 9: 16, 10: 2, 11: 18,
          12: 9, 13: 14, 14: 10, 15: 9, 16: 7, 17: 13, 18: 9, 19: 8, 20: 8, 21: 7, 22: 7, 23: 7}
STAGE1_TOP8 = [11, 5, 9, 7, 13, 17, 2, 3]
# Stage 2 (slot j = the stage-1 rank): over all their games kv13 (14+25)
# and kv11 (18+18) go on, though kv03 won more in stage 2 alone (24). Stage
# 3: kv13 wins over all its games (39+20 against 36+22), kv11 in stage 3
# alone.
STAGE2 = {11: 18, 5: 15, 9: 14, 7: 14, 13: 25, 17: 10, 2: 15, 3: 24}
STAGE3 = {13: 20, 11: 22}


def _wins(stage1=STAGE1):
    def wins_of(label, base):
        i = int(label[2:])
        if base == ks.CONFIRM_SEED_BASE:
            return 22
        stage = next(s for s in reversed(STAGES) if base >= s.seed_base)
        return {1: stage1, 2: STAGE2, 3: STAGE3}[stage.number][i]
    return wins_of


def _search(root, runner, **kw):
    return ks.Search(root, runner=runner, stages=STAGES, confirm_games=CONFIRM, **kw)


def _fit(games_dir, out, anchor):
    out.write_text(json.dumps({"tables": {ks.PURE: {"kv13": {"elo": 31.0, "se": 12.5},
                                                    anchor: {"elo": 0.0, "se": 0.0}}}}),
                   encoding="utf-8")
    return 0


def test_halving_picks_the_survivors_on_the_preregistered_seed_bases(tmp_path):
    runner = FakeBatch(_wins())
    search = _search(tmp_path, runner, fit=_fit)
    assert search.run_stage(1) == ks.EXIT_DONE
    assert [c[1] for c in runner.calls] == [200_000 + 1000 * i for i in range(24)]
    assert search.survivors(1) == STAGE1_TOP8
    ranking = search.log.rows("ranking")[-1]
    assert ranking["null_arm"]["within_two_se"] and ranking["ranking"][0]["wins"] == 18
    assert all(r["decisive"] == 20 for r in ranking["ranking"])        # capped slots replaced
    assert search.run_stage(2) == ks.EXIT_DONE
    assert runner.calls[24:] == [(ks.label_of(i), 230_000 + 1000 * j) for j, i in enumerate(STAGE1_TOP8)]
    assert search.survivors(2) == [13, 11]
    assert search.run_stage(3) == ks.EXIT_DONE
    assert runner.calls[32:] == [("kv13", 240_000), ("kv11", 241_000)]
    assert search.survivors(3) == [13]
    assert search.confirm() == ks.EXIT_DONE
    assert runner.calls[34:] == [("kv13", 250_000)]
    row = search.log.rows("confirmation")[-1]
    assert (row["label"], row["decisive"], row["wins"], row["reading"]) == ("kv13", CONFIRM, 22, "gain")
    winner = json.loads((tmp_path / "winner_decode.json").read_text(encoding="utf-8"))
    assert winner == search.vectors[13] and list(winner) == list(KNOB_VECTOR_KEYS)
    # Every stage's rows went down as the matches finished: 24 + 8 + 2.
    assert len(search.log.rows("match")) == 34 and len(search.log.rows("vectors")) == 1


def test_a_search_cut_mid_stage_resumes_without_replaying_a_finished_vector(tmp_path):
    first = FakeBatch(_wins(), stop_after=5)
    with pytest.raises(KeyboardInterrupt):
        _search(tmp_path, first).run_stage(1)
    assert len(first.calls) == 6                          # kv05 cut halfway through
    again = FakeBatch(_wins())
    search = _search(tmp_path, again)
    assert search.run_stage(1, one=True) == ks.EXIT_MORE
    assert again.calls == [("kv05", 205_000)]             # resumed in its directory
    while search.run_stage(1, one=True) == ks.EXIT_MORE:
        pass
    assert [c[0] for c in again.calls] == [ks.label_of(i) for i in range(5, 24)]
    assert search.survivors(1) == STAGE1_TOP8
    assert search.run_stage(1) == ks.EXIT_DONE and len(again.calls) == 19     # nothing left to play


def test_the_null_arm_outside_two_standard_errors_stops_stage_two(tmp_path):
    off = {**STAGE1, 0: 5}                                # 5 of 20: 2.2 standard errors below 50%
    runner = FakeBatch(_wins(off))
    assert _search(tmp_path, runner).run_stage(1) == ks.EXIT_DONE
    assert _search(tmp_path, runner).run_stage(2) == ks.EXIT_NULL_ARM
    assert len(runner.calls) == 24
    assert ks.null_arm_check(40, 100)["within_two_se"] and not ks.null_arm_check(61, 100)["within_two_se"]
    assert _search(tmp_path, runner, ignore_null_arm=True).run_stage(2) == ks.EXIT_DONE


def test_a_changed_directory_or_other_vectors_are_refused(tmp_path):
    runner = FakeBatch(_wins())
    _search(tmp_path, runner).run_stage(1)
    # A game of a survivor turned from a win into a loss: the recorded
    # ranking no longer matches the games.
    wins = [p for p in sorted((tmp_path / "games_kv03").glob("game_*.json"))
            if json.loads(p.read_text(encoding="utf-8"))["outcome_a"] == "win"]
    for path in wins[:3]:                                   # kv03 at 8, below kv04's 11
        g = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(dict(g, outcome_a="loss")), encoding="utf-8")
    with pytest.raises(SystemExit, match="directories changed"):
        _search(tmp_path, runner).survivors(1)
    other = dict(reference_player.load(), label="someone_else")
    with pytest.raises(SystemExit, match="other vectors or another reference"):
        _search(tmp_path, runner, ref=other).run_stage(1)


def test_the_match_command_plays_the_vector_against_the_reference(tmp_path):
    ref = reference_player.load()
    vec = {"raw_end_turn_offset": -2.32, "raw_attack_offset": 1.04, "raw_recruit_offset": -0.37,
           "raw_attack_tod_slope": 2.36}
    argv = ks.batch_argv(ref, vec, "kv01", tmp_path / "g", 100, 201_000, ks.MatchOptions())
    opt = {argv[k]: argv[k + 1] for k in range(len(argv) - 1) if argv[k].startswith("--")}
    assert opt["--spec-a"] == opt["--spec-b"] == ref["checkpoint_local"]
    assert opt["--memory-a"] == opt["--memory-b"] == str(ref["memory_slots"])
    assert (opt["--label-a"], opt["--label-b"]) == ("kv01", ref["label"])
    assert (opt["--raw-end-turn-offset-a"], opt["--raw-attack-offset-a"], opt["--raw-recruit-offset-a"],
            opt["--raw-attack-tod-slope-a"]) == ("-2.32", "1.04", "-0.37", "2.36")
    assert opt["--raw-end-turn-offset-b"] == "-1.5" and "--raw-attack-offset-b" not in opt
    assert (opt["--games"], opt["--seed-base"], opt["--mcts-sims"], opt["--jobs"]) == ("100", "201000", "0", "20")
    assert {"--persistent-workers", "--shared-inference"} <= set(argv)
