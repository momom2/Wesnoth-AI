#!/usr/bin/env python3
"""Held-out value probe (--holdout-size) + decisive-rate abort
tripwire (--abort-decisive-rate).

Holdout: MCTSPolicy.finalize_game must divert WHOLE games into the
frozen holdout set until it reaches target size, then send every
later game to training; holdout_metrics() evaluates the current net's
value CE on the frozen set WITHOUT touching weights. Rationale: the
logged train value loss is measured on replay-buffer samples the net
has already fit, so it cannot distinguish learning from memorization
(see BACKLOG §2026-07-02).

Tripwire: sim_self_play.main() must exit with code 4 -- after saving
a checkpoint and flushing the trainer-history CSV -- when the
trailing decisive-game rate drops below --abort-decisive-rate.
Exercised through the real main() on tiny mini-map games (2-turn
games always draw), per the tests-drive-real-code rule.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import glob
import pytest
import torch

from transformer_policy import TransformerPolicy
from tools.mcts_policy import MCTSPolicy, _PendingMCTSState

# Minimal synthetic GameState builder shared with the snapshot tests.
from test_inference_snapshot import _gs


def _pend(n):
    return [_PendingMCTSState(gs=_gs(), visit_counts=[], side=1)
            for _ in range(n)]


# ---------------------------------------------------------------------
# Holdout diversion + freeze + metrics
# ---------------------------------------------------------------------

def test_holdout_diverts_whole_games_then_freezes():
    policy = TransformerPolicy()
    mp = MCTSPolicy(policy, holdout_size=4)

    # Game 1 (3 states): below target -> whole game diverted.
    mp._pending["g1"] = _pend(3)
    mp.finalize_game("g1", winner=1)
    assert len(mp._holdout) == 3
    assert not mp._queue, "held-out states must not reach training"
    assert mp.holdout_metrics() is None, "still collecting -> no metric"

    # Game 2 (3 states): still below target when the game ends ->
    # diverted as a whole (6 > 4 is fine; granularity is per game).
    mp._pending["g2"] = _pend(3)
    mp.finalize_game("g2", winner=0)
    assert len(mp._holdout) == 6
    assert not mp._queue

    # Game 3: target reached -> trains normally.
    mp._pending["g3"] = _pend(2)
    mp.finalize_game("g3", winner=2)
    assert len(mp._queue) == 2, "post-freeze games must train"
    assert len(mp._holdout) == 6, "holdout is frozen once full"

    # Metrics: finite, sized, and gradient-free.
    before = [p.detach().clone() for p in policy._model.parameters()]
    loss, n = mp.holdout_metrics()
    assert n == 6
    assert math.isfinite(loss) and loss > 0.0
    for a, b in zip(before, policy._model.parameters()):
        assert torch.equal(a, b), "holdout eval must not update weights"


def test_offer_holdout_game_pool_contract():
    """The actor-pool drain calls offer_holdout_game per _R_EXPS
    payload (= one game). Contract: divert whole games until full,
    then refuse; empty payloads and probe-off always refuse."""
    from trainer import MCTSExperience
    policy = TransformerPolicy()
    mp = MCTSPolicy(policy, holdout_size=3)
    game = [MCTSExperience(game_state=_gs(), visit_counts=[], z=0.0)
            for _ in range(2)]
    assert mp.offer_holdout_game([]) is False, "empty game -> refuse"
    assert mp.offer_holdout_game(list(game)) is True     # 2 < 3
    assert mp.offer_holdout_game(list(game)) is True     # whole game in
    assert len(mp._holdout) == 4, "freeze is per-game, may overshoot"
    assert mp.offer_holdout_game(list(game)) is False, "full -> train"
    # Probe off -> always refuse.
    mp_off = MCTSPolicy(policy)
    assert mp_off.offer_holdout_game(list(game)) is False


def test_holdout_off_by_default():
    policy = TransformerPolicy()
    mp = MCTSPolicy(policy)  # holdout_size defaults to 0
    mp._pending["g1"] = _pend(2)
    mp.finalize_game("g1", winner=1)
    assert len(mp._queue) == 2, "holdout off -> everything trains"
    assert mp.holdout_metrics() is None


def test_holdout_per_game_cap_spreads_across_games():
    """A 512-state holdout built from ~2 whole games measured those
    games' quirks, not generalization (2026-07-07 diagnosis). The cap
    forces the probe to span many games."""
    from trainer import MCTSExperience
    policy = TransformerPolicy()
    mp = MCTSPolicy(policy, holdout_size=8, holdout_per_game_cap=3)
    big_game = [MCTSExperience(game_state=_gs(), visit_counts=[], z=1.0)
                for _ in range(50)]
    assert mp.offer_holdout_game(list(big_game)) is True
    assert len(mp._holdout) == 3, "one game contributes at most cap"
    for _ in range(2):
        assert mp.offer_holdout_game(list(big_game)) is True
    assert len(mp._holdout) == 9, "filled across 3 games (cap 3)"
    assert mp._holdout_games == 3
    assert mp.offer_holdout_game(list(big_game)) is False, "frozen"


def test_value_label_smoothing_train_only():
    """Smoothing must change the TRAIN loss (bounded away from the
    hard-target CE) while eval_value_loss stays unsmoothed, so holdout
    curves remain comparable across runs with different smoothing."""
    from trainer import (MCTSExperience, _categorical_value_loss,
                         _project_returns_to_atoms)
    policy = TransformerPolicy()
    atoms = policy._model._value_atoms
    logits = torch.randn(4, atoms.numel())
    zs = torch.tensor([1.0, -1.0, 0.2, 0.0])
    hard = _categorical_value_loss(logits, zs, atoms)
    smooth = _categorical_value_loss(logits, zs, atoms,
                                     label_smoothing=0.05)
    assert not torch.isclose(hard, smooth), "smoothing must move loss"
    # Mixture identity: CE against (1-eps)*target + eps*uniform ==
    # (1-eps)*CE_hard + eps*CE_uniform.
    K = atoms.numel()
    logp = torch.log_softmax(logits, dim=-1)
    ce_uniform = -(logp.sum() / K)
    expected = 0.95 * hard + 0.05 * ce_uniform
    assert torch.isclose(smooth, expected, rtol=1e-5)

    # eval path ignores TrainerConfig.value_label_smoothing entirely.
    policy._trainer.config.value_label_smoothing = 0.5
    exps = [MCTSExperience(game_state=_gs(), visit_counts=[], z=1.0)
            for _ in range(2)]
    e1 = policy._trainer.eval_value_loss(exps)
    policy._trainer.config.value_label_smoothing = 0.0
    e2 = policy._trainer.eval_value_loss(exps)
    assert math.isclose(e1, e2, rel_tol=1e-9), \
        "eval CE must be independent of train smoothing"


def test_fresh_value_ce_reported_pre_update():
    """The replay-path train_step must report value CE on the incoming
    batch measured BEFORE the gradient steps (distribution-matched
    generalization signal, no data lost)."""
    from tools.mcts_policy import ReplayConfig
    policy = TransformerPolicy()
    mp = MCTSPolicy(policy, replay_config=ReplayConfig(
        enabled=True, capacity=64, updates_per_iter=2,
        minibatch=4, min_size=1))
    mp._pending["g"] = _pend(6)
    mp.finalize_game("g", winner=1)
    stats = mp.train_step()
    assert math.isfinite(stats.fresh_value_ce), \
        "replay-path stats must carry the pre-update CE"
    # Pre-update CE on never-seen states == eval CE of the PRE-step
    # net; after updates the net moved, so recomputing now on the
    # same states generally differs. We can't re-check equality, but
    # it must be a plausible CE (0, log K + margin).
    k_atoms = policy._model._value_atoms.numel()
    assert 0.0 < stats.fresh_value_ce < math.log(k_atoms) + 1.0
    # Companion probe metrics (2026-07-07 log enrichment):
    # prediction entropy in (0, ln K]; the marginal floor for a
    # single-outcome batch (all z=+1 here) is the projection's own
    # entropy ~ 0 (point mass on the +1 atom).
    assert 0.0 < stats.fresh_pred_entropy <= math.log(k_atoms) + 1e-6
    assert 0.0 <= stats.fresh_ce_floor < 0.1, \
        "all-same-outcome batch => near-zero marginal floor"


def test_eval_value_metrics_floor_reflects_outcome_mix():
    """The marginal floor must be ~0 for a single-outcome batch and
    ~ln 2 for a 50/50 win/loss batch (state-blind predictability cap)."""
    from trainer import MCTSExperience
    policy = TransformerPolicy()
    wins = [MCTSExperience(game_state=_gs(), visit_counts=[], z=1.0)
            for _ in range(4)]
    mixed = wins[:2] + [
        MCTSExperience(game_state=_gs(), visit_counts=[], z=-1.0)
        for _ in range(2)]
    m_pure = policy._trainer.eval_value_metrics(wins)
    m_mixed = policy._trainer.eval_value_metrics(mixed)
    assert m_pure["marginal_ce_floor"] < 0.05
    assert abs(m_mixed["marginal_ce_floor"] - math.log(2)) < 0.05
    # ce must equal the back-compat scalar
    assert math.isclose(m_pure["ce"],
                        policy._trainer.eval_value_loss(wins),
                        rel_tol=1e-9)


def test_eval_value_loss_matches_step_mcts_scale():
    """Same experiences, same net: eval_value_loss must land in the
    same ballpark as step_mcts's value term (identical normalization:
    mean CE per experience). Guards against a silent /N drift that
    would make holdout and train curves incomparable."""
    policy = TransformerPolicy()
    mp = MCTSPolicy(policy, holdout_size=2)
    mp._pending["g"] = _pend(2)
    mp.finalize_game("g", winner=1)
    loss, n = mp.holdout_metrics()
    assert n == 2
    # C51 CE against a projected point mass is bounded by log(K);
    # a normalization bug (e.g. missing /N) would double it.
    k_atoms = policy._model._value_atoms.numel()
    assert 0.0 < loss < math.log(k_atoms) + 1.0
    # ce_std (2026-07-22: fresh_value_ce is the default success
    # metric; the std qualifies its moves): finite, non-negative,
    # and its weighted mean path must agree with "ce".
    m = policy._trainer.eval_value_metrics(list(mp._holdout))
    assert m["ce_std"] >= 0.0 and m["ce_std"] == m["ce_std"]
    assert abs(m["ce"] - loss) < 1e-6


# ---------------------------------------------------------------------
# Abort tripwire, through the real main()
# ---------------------------------------------------------------------

def test_outcome_carries_map_class():
    """Every GameOutcome must say which map class produced it — the
    aggregate decisive rate over a mixed curriculum is misleading
    (2026-07-03: ~50% aggregate while ladder maps were 0/8 decisive),
    so the per-class split is the metric that matters."""
    from sim_test_helpers import fresh_scenario_sim
    from tools.scenario_pool import classify_scenario
    from tools.sim_self_play import _recruit_cost_lookup, play_one_game

    assert classify_scenario("multiplayer_Den_of_Onis") == "ladder"
    assert classify_scenario("multiplayer_elensefar_courtyard") == "ladder"
    assert classify_scenario("drill_duel") == "drill"
    assert classify_scenario("2p_mini_edited") == "mini"
    assert classify_scenario("") == ""

    class _EndTurn:
        uses_step_rewards = False
        def select_action(self, gs, *, game_label="d", sim=None):
            return {"type": "end_turn"}
        def observe(self, *a, **kw): pass
        def drop_pending(self, *a, **kw): pass
        def finalize_game(self, *a, **kw): pass

    sim = fresh_scenario_sim(3, mini=True, max_turns=2)
    out = play_one_game(sim, _EndTurn(), lambda *a, **kw: 0.0,
                        game_label="mc", cost_lookup=_recruit_cost_lookup())
    assert out.map_class == "mini"
    # Actions-per-side-turn tally: an end_turn-only policy makes
    # exactly ONE action per side-turn.
    assert out.turn_action_counts, "per-side-turn tally must be populated"
    assert all(c == 1 for c in out.turn_action_counts), (
        f"end_turn-only game must tally 1 action per side-turn, "
        f"got {out.turn_action_counts}")


def test_abort_tripwire_exits_4_and_saves(tmp_path):
    """2-turn mini-map games always draw -> decisive rate 0 -> the
    tripwire must fire once the window fills, after saving the
    checkpoint and the (line-buffered) history CSV."""
    from sim_self_play import main

    ckpt = tmp_path / "trip.pt"
    csv = tmp_path / "history.csv"
    rc = main([
        "sim_self_play.py",
        "--iterations", "6", "--games-per-iter", "1",
        "--max-turns", "2", "--mini-ratio", "1.0",
        "--ladder-ratio", "0.0",
        "--d-model", "32", "--num-layers", "1",
        "--num-heads", "2", "--d-ff", "64",
        "--abort-decisive-rate", "0.5", "--abort-window", "2",
        "--checkpoint-out", str(ckpt),
        "--trainer-history-csv", str(csv),
        "--save-every", "100",  # only the tripwire save should fire
        "--seed", "7", "--log-level", "WARNING",
    ])
    assert rc == 4, f"expected tripwire exit code 4, got {rc}"
    assert ckpt.exists(), "tripwire must save a final checkpoint"
    rows = csv.read_text(encoding="utf-8").strip().splitlines()
    # Header + one row per completed iteration (window=2 -> 2 iters).
    assert len(rows) >= 3, f"CSV must hold the aborted run's rows: {rows}"


@pytest.mark.slow          # ~16s: see pytest.ini two-tier note
def test_spool_workers_end_to_end(tmp_path):
    """--spool-workers through the real main(): one independent
    worker process plays a mini game, spools it, and the learner
    consumes + trains from it. Pins the whole seam: worker boot from
    the saved checkpoint, atomic spool files, collect() routing
    (decision-step advance + holdout offer + queue), train_step."""
    from sim_self_play import main

    ckpt = tmp_path / "spool.pt"
    rc = main([
        "sim_self_play.py",
        "--mcts", "--mcts-sims", "2",
        "--spool-workers", "1", "--spool-dir", str(tmp_path / "spool"),
        "--iterations", "1", "--games-per-iter", "1",
        "--max-turns", "2", "--mini-ratio", "1.0",
        "--ladder-ratio", "0.0",
        "--d-model", "32", "--num-layers", "1",
        "--num-heads", "2", "--d-ff", "64",
        "--replay-min-size", "1",
        "--checkpoint-out", str(ckpt),
        "--trainer-history-csv", str(tmp_path / "h.csv"),
        "--save-every", "1",
        "--seed", "11", "--log-level", "WARNING",
    ])
    assert rc == 0, f"spool e2e run failed rc={rc}"
    assert ckpt.exists()
    rows = (tmp_path / "h.csv").read_text(encoding="utf-8") \
        .strip().splitlines()
    assert len(rows) >= 2, "iteration row must be logged"
    # The consumed game must have carried transitions into training.
    assert ",1," in rows[1] or rows[1].split(",")[2] == "1", (
        f"expected 1 game consumed in the iteration row: {rows[1]}")


def test_holdout_stall_tripwire_exits_5(tmp_path):
    """Memorization tripwire through the real main(): with
    --abort-holdout-min-delta forced impossibly large, no iteration
    can register a holdout improvement, so the stall counter must hit
    the limit deterministically once the holdout probe activates ->
    exit code 5 with a saved checkpoint."""
    from sim_self_play import main

    ckpt = tmp_path / "stall.pt"
    rc = main([
        "sim_self_play.py",
        "--mcts", "--mcts-sims", "2",
        "--iterations", "8", "--games-per-iter", "1",
        "--max-turns", "2", "--mini-ratio", "1.0",
        "--ladder-ratio", "0.0",
        "--d-model", "32", "--num-layers", "1",
        "--num-heads", "2", "--d-ff", "64",
        "--holdout-size", "2",
        "--abort-holdout-stall", "2",
        "--abort-holdout-min-delta", "100",
        "--checkpoint-out", str(ckpt),
        "--trainer-history-csv", str(tmp_path / "h.csv"),
        "--save-every", "100",
        "--seed", "3", "--log-level", "WARNING",
    ])
    assert rc == 5, f"expected holdout-stall exit code 5, got {rc}"
    assert ckpt.exists(), "tripwire must save a final checkpoint"
