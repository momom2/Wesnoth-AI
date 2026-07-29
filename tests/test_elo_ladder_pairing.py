"""Pairing contract of the Elo ladder (2026-07-29).

The cycle-26 strength read (Elo -137 +-260 over 8 games) exposed the
ladder's sampling as unpaired: every game drew a FRESH random setup,
so no setup was ever played from both sides by both models and
map/faction/side luck went straight into the noise. `_play_pair` now
plays mirrored setup pairs. These tests pin that contract on the
PRODUCTION `_play_pair` (the game itself is stubbed at the
`_play_single_game` boundary -- game play is exercised by the slow
smoke below and by the sim's own suite; the pairing logic is what
lives here and it must not be re-implemented in the test).
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import tools.elo_ladder as elo_ladder                      # noqa: E402
from tools.elo_ladder import Player, _play_pair            # noqa: E402


class _Setup:
    """Sentinel setup object; identity is what the pairing contract
    is about (the SAME object must drive both games of a mirror)."""
    _n = 0

    def __init__(self):
        _Setup._n += 1
        self.uid = _Setup._n

    def label(self):
        return f"setup{self.uid}"


def _run_play_pair(monkeypatch, games, outcomes):
    """Drive the production _play_pair with random_setup and
    _play_single_game stubbed at their boundaries. Returns
    (PairRecord, MirrorStats, calls) where calls is a list of
    (setup, pi_side, game_label)."""
    calls = []
    it = iter(outcomes)

    monkeypatch.setattr(elo_ladder, "random_setup",
                        lambda rng, forced_faction=None: _Setup())

    def fake_game(pi, pj, pi_side, setup, max_turns, game_label):
        calls.append((setup, pi_side, game_label))
        return next(it)

    monkeypatch.setattr(elo_ladder, "_play_single_game", fake_game)
    pi = Player(label="A", spec="dummy")
    pj = Player(label="B", spec="dummy")
    rec, mir = _play_pair(pi, pj, games, random.Random(0),
                          max_turns=100, forced_faction=None,
                          progress_prefix="[test]")
    return rec, mir, calls


def test_even_games_are_mirrored_setup_pairs(monkeypatch):
    rec, mir, calls = _run_play_pair(
        monkeypatch, games=8, outcomes=["win"] * 8)
    assert len(calls) == 8
    # Consecutive calls form mirror pairs: SAME setup object, sides
    # (1, 2). Setups differ across pairs.
    seen_setups = []
    for k in range(0, 8, 2):
        s1, side1, _ = calls[k]
        s2, side2, _ = calls[k + 1]
        assert s1 is s2, f"pair {k//2}: setups not shared"
        assert (side1, side2) == (1, 2)
        seen_setups.append(s1)
    assert len({s.uid for s in seen_setups}) == 4, "setups reused across pairs"
    # Game labels are unique (per-game policy state scoping).
    labels = [c[2] for c in calls]
    assert len(set(labels)) == len(labels)


def test_odd_game_plays_fresh_setup_side2(monkeypatch):
    rec, mir, calls = _run_play_pair(
        monkeypatch, games=5, outcomes=["win"] * 5)
    assert len(calls) == 5
    # Two mirror pairs + one unpaired leftover on side 2 with a setup
    # not shared with any pair.
    s_last, side_last, _ = calls[4]
    assert side_last == 2
    assert all(s_last is not c[0] for c in calls[:4])


def test_outcome_bookkeeping_and_mirror_classification(monkeypatch):
    # Pairs: (win,win)=sweep_i, (win,loss)=split, (loss,loss)=sweep_j,
    # (draw,win)=mixed.
    outcomes = ["win", "win", "win", "loss", "loss", "loss",
                "draw", "win"]
    rec, mir, calls = _run_play_pair(monkeypatch, games=8,
                                     outcomes=outcomes)
    assert (rec.wins_i, rec.draws, rec.wins_j) == (4, 1, 3)
    assert (mir.sweeps_i, mir.splits, mir.sweeps_j, mir.mixed) \
        == (1, 1, 1, 1)


def test_skipped_games_not_counted_and_pair_is_mixed(monkeypatch):
    # None = game could not be built/completed; it must not enter the
    # W-D-L record, and its pair cannot be classified as a sweep.
    outcomes = [None, "win", "loss", None]
    rec, mir, calls = _run_play_pair(monkeypatch, games=4,
                                     outcomes=outcomes)
    assert (rec.wins_i, rec.draws, rec.wins_j) == (1, 0, 1)
    assert (mir.sweeps_i, mir.splits, mir.sweeps_j, mir.mixed) \
        == (0, 0, 0, 2)


def test_two_player_ci_closed_form_cycle26():
    """Pin the Elo + CI math to the closed form on the exact cycle-26
    record (2-1-5, draws dropped, 1 ghost game): with the anchor
    pinned, the non-anchor SE must be ELO_PER_LN / sqrt(N p (1-p)).
    This is the computation behind the '-137 +-260' headline; a
    refactor that silently changes it would corrupt every strength
    comparison."""
    from tools.elo_ladder import PairRecord, fit_elo, _ELO_PER_LN
    pairs = {(0, 1): PairRecord(wins_i=2, draws=1, wins_j=5)}
    elo, se = fit_elo(2, pairs, anchor_idx=1, anchor_elo=0.0,
                      prior_games=1.0, draw_weight=0.0)
    # Effective: W_i = 2 + 0.5 ghost, W_j = 5 + 0.5 ghost, N = 8.
    p = 2.5 / 8.0
    expected_elo = _ELO_PER_LN * math.log(p / (1 - p))       # ~ -137
    expected_se = _ELO_PER_LN / math.sqrt(8.0 * p * (1 - p))  # ~ 132.5
    assert abs(elo[0] - expected_elo) < 0.5, (elo[0], expected_elo)
    assert abs(se[0] - expected_se) < 0.5, (se[0], expected_se)
    assert abs(expected_elo - (-137.0)) < 1.0
    assert abs(1.96 * expected_se - 260.0) < 1.0


def test_load_policy_builds_checkpoint_structural_flags(tmp_path):
    """An advice/aux-trained checkpoint must be loaded by eval into a
    policy BUILT with those paths, so no weight is silently dropped as
    an unexpected key. Found live (2026-07-29): the campaign checkpoint
    loaded into the ladder with its advice tensors dropped -- inert for
    raw-policy eval, but an `mcts:` eval would have measured a
    different model than the one that trained. Full state_dict
    equality is the assert: if anything was dropped, some key differs."""
    import torch
    from wesnoth_ai.transformer_policy import TransformerPolicy
    from tools.eval_sim import _load_policy

    src = TransformerPolicy(d_model=32, num_layers=1, num_heads=2,
                            d_ff=32, aux_score=True, advice=True)
    ckpt = tmp_path / "advice_ckpt.pt"
    src.save_checkpoint(ckpt)

    loaded = _load_policy(ckpt, None, label="test")
    assert loaded._advice is True
    assert loaded._aux_score is True
    src_sd = src._model.state_dict()
    got_sd = loaded._model.state_dict()
    assert set(src_sd) == set(got_sd)
    for k in src_sd:
        assert torch.equal(src_sd[k], got_sd[k]), f"weight differs: {k}"


def test_mcts_spec_honors_checkpoint_advice_and_contract(tmp_path):
    """An `mcts:` player built from an advice-trained checkpoint must
    search with root advice ON (the checkpoint's learned conditioning
    -- how it plays in production), while the eval-contract crutches
    (aux_value_bonus, draw_tiebreak) stay OFF regardless. Covers both
    builders: elo_ladder.Player.build and elo_eval_game._build_player."""
    from wesnoth_ai.transformer_policy import TransformerPolicy
    src = TransformerPolicy(d_model=32, num_layers=1, num_heads=2,
                            d_ff=32, advice=True)
    ckpt = tmp_path / "advice_ckpt.pt"
    src.save_checkpoint(ckpt)

    p = Player(label="m", spec=f"mcts:1:{ckpt}")
    p.build(None)
    cfg = p.policy._mcts_config
    assert cfg.advice is True
    assert cfg.aux_value_bonus == 0.0
    assert cfg.draw_tiebreak is None

    from tools.elo_eval_game import _build_player
    mp = _build_player(str(ckpt), "m2", sims=1, device=None)
    cfg2 = mp._mcts_config
    assert cfg2.advice is True
    assert cfg2.aux_value_bonus == 0.0
    assert cfg2.draw_tiebreak is None

    # A checkpoint WITHOUT the advice path stays advice-OFF.
    src2 = TransformerPolicy(d_model=32, num_layers=1, num_heads=2,
                             d_ff=32)
    ckpt2 = tmp_path / "plain_ckpt.pt"
    src2.save_checkpoint(ckpt2)
    p2 = Player(label="n", spec=f"mcts:1:{ckpt2}")
    p2.build(None)
    assert p2.policy._mcts_config.advice is False


@pytest.mark.slow
def test_run_ladder_smoke_real_sim():
    """Production-path smoke: run_ladder end-to-end through the real
    sim (dummy vs random-init policy, 1 mirror pair, 2-turn horizon).
    Guards the plumbing the stub tests deliberately bypass."""
    from tools.scenario_events import SCENARIO_DIR
    if not SCENARIO_DIR.exists():
        pytest.skip("wesnoth_src scenario dir not present")
    from tools.device_select import select_inference_device
    from tools.elo_ladder import run_ladder
    device = select_inference_device("cpu")
    players = [Player(label="dummy", spec="dummy"),
               Player(label="rand", spec="random")]
    for p in players:
        p.build(device)
    res = run_ladder(players, games_per_pair=2, max_turns=2, seed=3,
                     forced_faction=..., anchor_label="dummy",
                     anchor_elo=0.0, prior_games=1.0, draw_weight=0.0)
    assert res.n_games == 2
    assert res.max_turns == 2
    key = "dummy__vs__rand"
    assert key in res.mirror
    m = res.mirror[key]
    assert m["sweeps_i"] + m["splits"] + m["sweeps_j"] + m["mixed"] == 1
    # W-D-L totals mirror each other between the two players.
    ra, rb = res.record
    assert ra["win"] == rb["loss"] and ra["loss"] == rb["win"]
    assert ra["draw"] == rb["draw"]
    assert ra["win"] + ra["loss"] + ra["draw"] == 2
