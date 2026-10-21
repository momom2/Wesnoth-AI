"""Stage 1's offline readouts (tools/selfplay_readout.py) on planted data:
I-sample passes on the logged sampler and fails on a kept offset or a
colder temperature, the corrections enter only when they explain the
label, the sharpening slope recovers a planted slope, I-V catches a
correction read after its event, dice control catches a flipped
orientation, and I-count accounts for a real record's commands."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools.decision_log import decision_entry  # noqa: E402
from tools.raw_player import pick_index, shift_end_turn_prior  # noqa: E402
from tools.selfplay_readout import (  # noqa: E402
    SampleCheck, Table, decision_arrays, dice_control, i_v, icount_problems, label_noise,
    sharpening_slope, side_games,
)
from wesnoth_ai.server_priors import KIND_ATTACK, KIND_END_TURN, KIND_MOVE, KIND_RECRUIT  # noqa: E402


def sampled_entries(n: int, seed: int = 0, offset: float = 0.0, temperature: float = 1.0):
    """`n` decisions whose logged law is the network's and whose action is
    drawn by the raw player's choice at `offset` and `temperature`."""
    rng = np.random.default_rng(seed)
    entries = []
    for i in range(n):
        k = int(rng.integers(4, 30))
        kind = rng.choice([KIND_ATTACK, KIND_MOVE, KIND_RECRUIT], size=k, p=[0.3, 0.5, 0.2])
        kind[-1] = KIND_END_TURN
        actor = np.arange(k) // 3
        prior = rng.dirichlet(np.full(k, 0.7))
        played = shift_end_turn_prior(prior, kind == KIND_END_TURN, offset)
        pick = pick_index(played, temperature, rng)
        entries.append(decision_entry(side=1 + i % 2, at=i, actor=actor, kind=kind, target=np.zeros(k, int),
                                      weapon=np.zeros(k, int), prior=prior, pick=pick, value=0.0))
    return entries


def i_sample(entries):
    check = SampleCheck(seed=3)
    check.add(decision_arrays(entries))
    return check.i_sample()


def test_i_sample_passes_on_the_logged_law_and_fails_on_another_decode():
    assert i_sample(sampled_entries(4000))["pass"]
    kept_offset = i_sample(sampled_entries(4000, offset=-1.5))
    assert not kept_offset["pass"] and kept_offset["end_turn"]["p"] < 1e-6
    colder = i_sample(sampled_entries(4000, temperature=0.8))
    assert not colder["pass"] and colder["temperature_score"]["z"] > 5


def _table(R, n, sum_v, sum_v2, score, D=None, DV=None, beta_s=None, b=None):
    rows = len(R)
    zeros = np.zeros(rows)
    return Table(game=np.repeat(np.arange(rows // 2), 2), fold=np.repeat(np.arange(rows // 2) // 2 % 2, 2),
                 R=R, z=R, b=zeros if b is None else b, beta_s=zeros if beta_s is None else beta_s,
                 n=n, sum_v=sum_v, sum_v2=sum_v2, score=score, opp_score=score.reshape(-1, 2)[:, ::-1].reshape(-1),
                 D=np.zeros((rows, 3)) if D is None else D, DV=np.zeros((rows, 3)) if DV is None else DV)


def planted_value_table(games: int, alpha: float, seed: int = 0):
    """Side-games whose label is alpha times the mean value reading plus
    noise, mirrored between the two sides; the opponent scores are
    unrelated noise."""
    rng = np.random.default_rng(seed)
    n = rng.integers(50, 150, size=games)
    vbar = rng.normal(0, 0.5, size=games)
    R1 = alpha * vbar + rng.normal(0, 0.5, size=games)
    rows = {k: np.zeros(2 * games) for k in ("R", "n", "sum_v", "sum_v2", "score")}
    for side, sign in ((0, 1), (1, -1)):
        v = [sign * vbar[g] + rng.normal(0, 0.3, n[g]) for g in range(games)]
        rows["R"][side::2] = sign * R1
        rows["n"][side::2] = n
        rows["sum_v"][side::2] = [x.sum() for x in v]
        rows["sum_v2"][side::2] = [(x * x).sum() for x in v]
        rows["score"][side::2] = rng.normal(0, 3, size=games)
    return _table(rows["R"], rows["n"], rows["sum_v"], rows["sum_v2"], rows["score"])


def test_a_correction_enters_only_when_it_explains_the_label():
    out = label_noise(planted_value_table(3000, alpha=0.8))
    assert out["v"]["enters"] and not out["U"]["enters"]
    # Per decision, v_t carries noise of variance 0.09 around a mean of
    # variance 0.25: the fitted alpha is the planted 0.8 attenuated.
    for alpha, _kappa in out["v"]["coefficients_by_fold"]:
        assert alpha == pytest.approx(0.8 * 0.25 / (0.25 + 0.09), abs=0.06)
    # Var R = 0.64 * 0.25 + 0.25, less Cov(R, v)^2 / Var v = 0.2^2 / 0.34.
    assert out["luck"]["E"] == pytest.approx(0.41, abs=0.03)
    assert out["v"]["E"] == pytest.approx(0.41 - 0.04 / 0.34, abs=0.03)
    none = label_noise(planted_value_table(3000, alpha=0.0, seed=1))
    assert not none["v"]["enters"] and not none["v_and_U"]["enters"]


def test_the_sharpening_slope_recovers_a_planted_slope():
    rng = np.random.default_rng(5)
    G, sigma2 = 20000, 4.0
    s = rng.normal(0, np.sqrt(sigma2), size=(G, 2))
    for delta in (0.05, 0.0):
        R1 = delta * (s[:, 0] - s[:, 1]) + rng.normal(0, 0.8, size=G)
        R = np.stack([R1, -R1], axis=1).reshape(-1)
        out = sharpening_slope(R, s.reshape(-1))
        assert abs(out["slope"] - delta * sigma2) <= 3 * out["se"]
        assert out["resolved"] == (delta != 0.0)


def test_i_v_catches_a_correction_read_after_its_event():
    entries = sampled_entries(3000, seed=2)
    # Each "game" holds 60 decisions, 30 per side.
    games = [entries[i:i + 60] for i in range(0, len(entries), 60)]

    def table_with(value_of):
        rows = {k: [] for k in ("n", "sum_v", "sum_v2", "score", "D", "DV")}
        for g in games:
            for e in g:
                e["v"] = value_of(e)
            for side, sg in sorted(side_games(decision_arrays(g)).items()):
                for k in rows:
                    rows[k].append(getattr(sg, k))
        arr = {k: np.array(v, dtype=float) for k, v in rows.items()}
        R = np.zeros(len(arr["n"]))
        return _table(R, arr["n"], arr["sum_v"], arr["sum_v2"], arr["score"], arr["D"], arr["DV"])

    rng = np.random.default_rng(9)
    assert i_v(table_with(lambda e: float(rng.normal())))["pass"]
    peek = i_v(table_with(lambda e: float(e["action"][1] == KIND_END_TURN)))
    assert not peek["pass"] and not peek["v"]["end_turn"]["pass"]


def test_dice_control_reproduces_the_slope_and_catches_a_flipped_orientation():
    from tools.pg_labels import side_labels
    from tools.match_luck import COLUMNS
    from tests.test_pg_labels import planted_world, synthetic_rows
    # Results are wins and losses: the planted z's sign.
    rows = [dict(r, z=float(np.sign(r["z"]))) for r in synthetic_rows(3000, planted_world(), seed=4, noise=0.3)]
    labels, info = side_labels(rows, heldout_from=1500)
    walk = [{"game": r["game"], "outcome_a": "win" if r["z"] > 0 else "loss", "S": dict(zip(COLUMNS, r["S"]))}
            for r in rows if r["side"] == 1]
    good = dice_control(labels, walk, info["fits"]["train"]["beta"])
    assert good["beta_pass"] and good["slope_pass"], good
    flipped = [dict(lab, beta_S=-lab["beta_S"]) if not lab["heldout"] else lab for lab in labels]
    bad = dice_control(flipped, walk, info["fits"]["train"]["beta"])
    assert not bad["slope_pass"] and not bad["pass"]
    moved = dice_control(labels, walk, [b + 0.01 for b in info["fits"]["train"]["beta"]])
    assert not moved["beta_pass"]


def test_i_count_accounts_for_a_real_records_commands(tmp_path):
    from tests.helpers.selfplay_games import selfplay_game, tiny_checkpoint
    spec = tiny_checkpoint(tmp_path / "net.pt")
    _, rec = selfplay_game(spec, tmp_path / "g", 31, 1, "--log-decisions", max_turns=3)
    assert icount_problems(rec) == []
    dropped = copy.deepcopy(rec)
    live = [i for i, e in enumerate(dropped["decisions"]) if not e["refused"]]
    del dropped["decisions"][live[len(live) // 2]]
    assert icount_problems(dropped)
    false_refusal = copy.deepcopy(rec)
    false_refusal["decisions"][live[3]].update(refused=True, cmd=-1)
    assert icount_problems(false_refusal)
    assert icount_problems({k: v for k, v in rec.items() if k != "decisions"}) == ["no decision log"]


@pytest.mark.slow
def test_a_logged_run_goes_through_rows_labels_and_readout(tmp_path):
    """Six short logged games, walked into rows, labelled and read: every
    command is accounted for, the masses are 1, and every game-side is
    joined to its label."""
    import json
    from tests.helpers.selfplay_games import selfplay_game, tiny_checkpoint
    from tools import pg_labels, selfplay_readout, selfplay_rows
    spec = tiny_checkpoint(tmp_path / "net.pt")
    games = tmp_path / "games"
    for i in range(6):
        selfplay_game(spec, games, 500 + i, 1 + i % 2, "--log-decisions")
    assert selfplay_rows.main(["--games", str(games), "--walk", str(tmp_path / "walk.jsonl"),
                               "--out", str(tmp_path / "rows.jsonl"), "--seed-base", "500"]) == 0
    assert pg_labels.main([str(tmp_path / "rows.jsonl"), "--heldout-from", "2", "--out",
                           str(tmp_path / "labels.jsonl"), "--fits", str(tmp_path / "fits.json")]) == 0
    selfplay_readout.main(["--games", str(games), "--labels", str(tmp_path / "labels.jsonl"),
                           "--fits", str(tmp_path / "fits.json"), "--walk", str(tmp_path / "walk.jsonl"),
                           "--out", str(tmp_path / "readout.json")])
    out = json.loads((tmp_path / "readout.json").read_text(encoding="utf-8"))
    assert out["census"]["games"] == 6 and out["labelled_games"] == 6
    assert out["i_count"]["pass"] and out["legal_mass"]["pass"]
    assert out["i_sample"]["decisions"] == round(out["census"]["decisions_per_game"] * 6)
    assert set(out["p1_targets"]) == {"end_turn", "attack", "recruit"}
