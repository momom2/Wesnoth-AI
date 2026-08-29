"""Plan-tournament behavioral guards (proposition 1, 2026-08-26;
review rounds 1-2 same day). Production-path tests. Pinned
regressions are each a leg-killer: noise distilled from abstained
turns, certification without re-validation (winner's curse), serving
plans whose pre_keys track a counterfactual dice stream, budget-
starved schedules, unbounded telemetry.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from tools.mcts import MCTSConfig  # noqa: E402
from tools.plan_tournament import (  # noqa: E402
    PlanTournamentPolicy, TournamentConfig, _odd_depths, _salted,
    beta_from_margin, launch_echo_schedule, predicted_demand,
    run_tournament, size_schedule,
)
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402


def _tiny():
    torch.manual_seed(0)
    return TransformerPolicy(device=torch.device("cpu"), d_model=32,
                             num_layers=1, num_heads=4, d_ff=64)


def _cfg(**over):
    base = dict(n_challengers=2, depths=(1,), redraws=1,
                cert_depth=1, cert_redraws=2,
                budget_forwards=300, margin_band=0.08)
    base.update(over)
    return TournamentConfig(**base)


def test_beta_from_margin_shape():
    cfg = TournamentConfig(margin_band=0.08, beta_max=0.25,
                           margin_ref=0.32)
    assert beta_from_margin(0.0, cfg) == 0.0
    assert beta_from_margin(0.08, cfg) == 0.0          # at band: abstain
    assert 0.0 < beta_from_margin(0.10, cfg) < cfg.beta_max
    assert beta_from_margin(0.32, cfg) == cfg.beta_max  # saturates


def test_depth_validation_enforces_own_frame():
    """Review C4: even/zero depths read the boundary in the
    opponent's fogged frame (the leg-4 blindness)."""
    assert _odd_depths((2, 4, 6), (1, 3), "t") == (1, 3)
    assert _odd_depths((1, 2, 3), (1,), "t") == (1, 3)
    assert _odd_depths((0, -1), (5,), "t") == (5,)


def test_schedule_sizes_itself_to_the_budget_with_cert_reserve():
    """Round-3 C0/C13: predicted demand at the ESTIMATED half-turn
    cost (not the incumbent's own length) must fit the budget, and
    an unaffordable floor returns (0, ()) -- honest abstention, not
    a schedule the budget cannot fund."""
    cfg = TournamentConfig()          # shipped defaults
    for spine_len in (2, 4, 8, 12, 17, 30):
        for per_half in (6, 12, 20):
            n, deps, reps = size_schedule(spine_len, per_half,
                                          cfg.budget_forwards, cfg)
            if n == 0:
                assert deps == ()
                continue
            d = predicted_demand(spine_len, per_half, n, deps, cfg,
                                 cert_ph=per_half,
                                 cert_redraws=reps)
            assert d <= cfg.budget_forwards, \
                f"unaffordable schedule at K={spine_len} " \
                f"ph={per_half}: n={n} depths={deps} demand={d}"
            # Round-8 C0: degraded evidence is never planned -- a
            # funded schedule always carries the FULL replicate set.
            assert deps and reps == cfg.cert_redraws
    # Warm-EMA floor schedules stay fundable through max_spine at
    # typical half-turn estimates (round-7 C0: the hard-bound
    # reserve abstained fundable turns at L>=30); at high estimates
    # the long tail abstains honestly rather than certifying on a
    # weakened test (round-8 C0).
    for per_half in (4, 6):
        for spine_len in (24, 30, 36, 40):
            n, deps, reps = size_schedule(spine_len, per_half,
                                          cfg.budget_forwards, cfg)
            assert n >= cfg.min_challengers, \
                f"warm-EMA floor unaffordable at K={spine_len} " \
                f"ph={per_half}"
            assert reps == cfg.cert_redraws
    for spine_len in (12, 17, 24):
        n, deps, reps = size_schedule(spine_len, 12,
                                      cfg.budget_forwards, cfg)
        assert n >= cfg.min_challengers and reps == cfg.cert_redraws


def test_cold_start_funds_a_real_tournament():
    """Round-4 C1/C4: at shipped defaults the COLD-START schedule
    (no EMA yet) must fund a real tournament in the target regime --
    the round-4 conservative fallback made the certification reserve
    exceed the whole budget at every spine length, a self-sustaining
    abstain deadlock (the EMA that corrects the estimate is fed by
    the selection loop the deadlock disabled)."""
    cfg = TournamentConfig()
    for k in (4, 8, 12, 17):
        n, deps, demand, per_half = launch_echo_schedule(cfg, k=k)
        assert n >= cfg.min_challengers and deps, \
            f"cold start cannot fund a tournament at K={k} " \
            f"(per_half={per_half}, demand={demand})"
        assert demand <= cfg.budget_forwards


def test_launch_echo_executes():
    """Round-4 C0: the launch echo rotted through a signature change
    because nothing executed it. launch_echo_schedule IS the echo's
    arithmetic; executing it here pins the callsite contract."""
    n, deps, demand, per_half = launch_echo_schedule(
        TournamentConfig())
    assert isinstance(n, int) and isinstance(deps, tuple)
    assert per_half == 13          # cold start at K=12: min(20, 13)


def test_salted_forks_leave_live_stream():
    sim = fresh_scenario_sim()
    f = _salted(sim, "pt:test:1")
    assert f._seed_salt == "pt:test:1"
    assert getattr(sim, "_seed_salt", "") != "pt:test:1"
    assert _salted(sim, "pt:test:2")._seed_salt != f._seed_salt


def test_serving_plan_tracks_live_dice_stream(monkeypatch):
    """Round-2 C0/C11: the SERVING spine is recorded unsalted, so
    pre_keys match the states the live game will actually visit --
    stepping the plan's commands on the live sim must reproduce
    every pre_key exactly (the round-2 bug diverged at the first
    stochastic action and dropped the certified targets).
    Certification statistics are pinned elsewhere; here `certify` is
    patched to isolate the respine property on a certified plan."""
    import tools.plan_tournament as pt
    from wesnoth_ai.classes import state_key
    monkeypatch.setattr(pt, "certify",
                        lambda *a, **k: (0.2, 0.2, 2))
    net = _tiny()
    sim = fresh_scenario_sim()
    side = sim.gs.global_info.current_side
    plan, stats = pt.run_tournament(net, sim, side, 0, _cfg(),
                                    np.random.default_rng(7),
                                    salt_ns="t:live")
    n_targets = sum(1 for t in plan.targets if t)
    if stats["n_challengers"] > 0:
        assert plan.certified and plan.beta == 0.2
        assert n_targets >= 1
    # THE property: replaying the plan on the LIVE sim visits
    # exactly the recorded pre-states, dice included.
    for i, cmd in enumerate(plan.commands):
        if sim.done or sim.gs.global_info.current_side != side:
            break
        assert state_key(sim.gs) == plan.pre_keys[i], \
            f"pre_key mismatch at slot {i}: serving spine is not " \
            f"on the live dice stream"
        sim.step(cmd)


def test_budget_abstain_survives_the_serving_path():
    """Round-5 C0: the early-return stats dict lacked a key
    _t_record reads, so every budget-abstained tournament raised
    KeyError out of select_action and the whole game was silently
    discarded -- the pre-registered abstain path could never run."""
    net = _tiny()
    pol = PlanTournamentPolicy(
        net, MCTSConfig(n_simulations=2),
        tournament_config=_cfg(budget_forwards=5))   # starves all
    sim = fresh_scenario_sim()
    import copy
    for _ in range(3):
        if sim.done:
            break
        snap = copy.deepcopy(sim.gs)
        sim.step(pol.select_action(snap, game_label="g", sim=sim))
    stats = pol.drain_tournament_stats()
    assert stats.get("pt_abstain_events_per_turn", 0) > 0


def test_exhausted_budget_cannot_certify_and_is_counted():
    net = _tiny()
    sim = fresh_scenario_sim()
    side = sim.gs.global_info.current_side
    plan, stats = run_tournament(
        net, sim, side, 0,
        _cfg(budget_forwards=5, margin_band=-10.0),
        np.random.default_rng(2), salt_ns="t:2")
    assert plan.beta == 0.0 and not plan.certified
    assert stats["abstained_budget"] == 1


def test_certification_requires_two_stage_acceptance():
    """Round-2 C1/C6: an argmax over single selection draws must
    never certify by itself -- certification is the stage-2
    two_stage_accept verdict over cert_redraws fresh paired grades.
    With an astronomically high band, even a 'winning' selection
    must abstain; stats must show the attempt."""
    net = _tiny()
    sim = fresh_scenario_sim()
    side = sim.gs.global_info.current_side
    plan, stats = run_tournament(
        net, sim, side, 0, _cfg(margin_band=10.0),
        np.random.default_rng(3), salt_ns="t:3")
    assert plan.beta == 0.0 and not plan.certified
    if stats["n_challengers"] > 0 and not stats["abstained_budget"]:
        assert stats["cert_attempts"] == 1


def test_side_turn_ledger_survives_bounce():
    """Round-2 C2/C8/C12: the per-side-turn spend ledger lives in
    the POLICY, so a bounce (drop_last_pending kills the plan) does
    not refresh the allowance."""
    net = _tiny()
    pol = PlanTournamentPolicy(net, MCTSConfig(n_simulations=2),
                               tournament_config=_cfg())
    sim = fresh_scenario_sim()
    import copy
    snap = copy.deepcopy(sim.gs)
    pol.select_action(snap, game_label="g", sim=sim)
    side = sim.gs.global_info.current_side
    turn_no = int(sim.gs.global_info.turn_number)
    spent1 = pol._turn_spent("g", side, turn_no)   # self-locking
    assert spent1 > 0
    assert pol.drop_last_pending("g")
    with pol._lock:
        assert "g" not in pol._t_plans, "bounce must drop the plan"
    assert pol._turn_spent("g", side, turn_no) == spent1, \
        "the ledger must survive the bounce"
    pol.finalize_game("g", winner=1, final_gs=sim.gs)
    assert pol._turn_spent("g", side, turn_no) == 0, \
        "finalize must clear the ledger"


def test_abstain_band_yields_value_only():
    net = _tiny()
    pol = PlanTournamentPolicy(
        net, MCTSConfig(n_simulations=2),
        tournament_config=_cfg(margin_band=10.0))   # abstain always
    sim = fresh_scenario_sim()
    import copy
    for _ in range(4):
        if sim.done:
            break
        snap = copy.deepcopy(sim.gs)
        sim.step(pol.select_action(snap, game_label="g", sim=sim))
    pol.finalize_game("g", winner=2, final_gs=sim.gs)
    with pol._lock:
        exps = list(pol._queue)
    assert exps and all(not e.visit_counts for e in exps)
    stats = pol.drain_tournament_stats()
    assert stats["pt_cert_rate"] == 0.0
    assert stats["pt_beta_mean"] == 0.0


def test_certification_replicates_share_a_spine_salt():
    """Round-6 C0: both arms of a certification replicate must
    re-execute on the SAME dice stream so a shared command prefix
    cancels out of the paired delta (the per-arm ':c'/':i' salts
    made the test unpaired). Pin the salt construction."""
    src = (Path(__file__).parent.parent
           / "tools/plan_tournament.py").read_text(encoding="utf-8")
    assert 'f"{s}:spine"' in src,         "certification spine salts must be shared per replicate"
    assert 'f"{s}:{tag}"' not in src.split("def certify")[1]         .split("def run_tournament")[0].replace(
            'f"{s}:p{tag}"', ""),         "per-arm spine salts reintroduced in certify()"


def test_budget_cut_preserves_measured_margins(monkeypatch):
    """Round-10 C0: a budget death inside a candidate's deeper
    projection must not erase its earlier-round measured margin --
    with a sole survivor that erasure skipped certification with
    the reserve fully intact. Kill _project_counted on a chosen
    unreserved call and assert certification is still attempted."""
    import tools.plan_tournament as pt
    real = pt._project_counted
    calls = {"n": 0}

    def dying(policy, sim, side, ds, ht, ma, rng, budget, salt,
              reserved=False, half_obs=None):
        if not reserved:
            calls["n"] += 1
            if calls["n"] >= 5:      # die from the 5th selection call
                budget.used = budget.cap - budget.reserve + 1
                return None
        return real(policy, sim, side, ds, ht, ma, rng, budget,
                    salt, reserved=reserved, half_obs=half_obs)

    monkeypatch.setattr(pt, "_project_counted", dying)
    net = _tiny()
    sim = fresh_scenario_sim()
    side = sim.gs.global_info.current_side
    plan, stats = pt.run_tournament(
        net, sim, side, 0,
        _cfg(n_challengers=2, depths=(1, 3), budget_forwards=4000),
        np.random.default_rng(22), salt_ns="t:cut")
    if stats["n_challengers"] > 0 and stats["graded"] > 0:
        assert stats["cert_attempts"] == 1, \
            "a cut erased measured evidence and skipped certification"


def test_accept_rule_is_n_aware():
    """Round-8/9 C0: the acceptance threshold must hold alpha across
    replicate counts -- a flat factor 2 at n=2 inflates the
    false-accept rate ~1.6x. Pin the table values through the pure
    rule, and the knob clamp that keeps every reachable n tabled."""
    from types import SimpleNamespace
    from tools.plan_tournament import (_T_CRIT, accept_rule,
                                       config_from_args)
    assert _T_CRIT == {2: 3.37, 3: 2.0, 4: 1.72, 5: 1.61}
    d2 = [0.30, 0.10]                  # sd = 0.1414..., n = 2
    ok2, mean2, thr2 = accept_rule(d2, band=0.0)
    import numpy as np
    sd2 = float(np.std(d2, ddof=1))
    assert abs(thr2 - 3.37 * sd2 / np.sqrt(2)) < 1e-9
    d3 = [0.30, 0.10, 0.20]
    _ok3, _m3, thr3 = accept_rule(d3, band=0.0)
    sd3 = float(np.std(d3, ddof=1))
    assert abs(thr3 - 2.0 * sd3 / np.sqrt(3)) < 1e-9
    # Band floor still binds.
    assert accept_rule([0.05, 0.05, 0.05], band=0.08)[2] == 0.08
    # Knob clamp: cert_redraws above the table is clipped.
    cfg = config_from_args(SimpleNamespace(plan_tournament=True,
                                           pt_cert_redraws=9))
    assert cfg.cert_redraws == max(_T_CRIT)


def test_eval_forward_counter_counts_every_procedure():
    """Round-13 C0/C1: the step-1 equal-measured-forwards check is
    only as good as the counter. The proxy must wrap BEFORE search
    wrappers cache the model reference (MCTS recorded a fabricated
    zero) and must count batched forwards per sample (TCS silently
    omitted them). Play two decisions per procedure and assert a
    nonzero count."""
    import copy
    from tools.elo_eval_game import _build_player
    import tools.elo_eval_game as eeg

    def fake_load(path, device, label="", **_kw):
        return _tiny()
    orig = eeg._load_policy
    eeg._load_policy = fake_load
    try:
        for kwargs in (dict(turn_search=False),          # mcts
                       dict(turn_search=True),           # tcs
                       dict(plan_tournament=True)):      # plan
            player, cnt = _build_player("x.pt", "t", 2, None,
                                        **kwargs)
            assert cnt is not None
            sim = fresh_scenario_sim()
            for _ in range(2):
                if sim.done:
                    break
                snap = copy.deepcopy(sim.gs)
                sim.step(player.select_action(snap, game_label="g",
                                              sim=sim))
            assert cnt.n_forwards > 0, \
                f"counter recorded zero forwards for {kwargs}"
    finally:
        eeg._load_policy = orig


def test_drain_returns_none_when_empty():
    net = _tiny()
    pol = PlanTournamentPolicy(net, MCTSConfig(n_simulations=2),
                               tournament_config=_cfg())
    assert pol.drain_distill_stats() is None
    assert pol.drain_tournament_stats() == {}


def test_drain_keys_all_reach_the_csv():
    """Round-3 C11/C17: a drain key missing from the CSV column
    list is dropped by DictWriter(extrasaction='ignore') -- dark
    telemetry, the leg-3 failure class. Every pt_* key the drain
    can emit must have a column."""
    from tools.plan_tournament import PT_DRAIN_KEYS
    csv_src = (Path(__file__).parent.parent
               / "tools/sim_self_play.py").read_text(encoding="utf-8")
    missing = {k for k in PT_DRAIN_KEYS if f'"{k}"' not in csv_src}
    assert not missing, f"drain keys with no CSV column: {missing}"


def test_pt_flag_symmetry_across_generation_paths():
    """Round-3 C18 (the leg-3 half-carried-config class): every
    --pt-* knob defined in sim_self_play must (a) be forwarded in
    the spool-worker command tail and (b) exist in the worker's own
    parser; and argparse defaults must not drift from
    TournamentConfig."""
    import re
    root = Path(__file__).parent.parent
    ssp = (root / "tools/sim_self_play.py").read_text(encoding="utf-8")
    wrk = (root / "tools/selfplay_worker.py").read_text(encoding="utf-8")
    knobs = set(re.findall(r'add_argument\("(--pt-[a-z-]+)"', ssp))
    assert knobs, "no pt knobs found in sim_self_play"
    for k in knobs:
        assert ssp.count(f'"{k}"') >= 2, \
            f"{k} defined but not forwarded to spool workers"
        assert f'"{k}"' in wrk, f"{k} missing from worker parser"
    # EVERY knob default must match TournamentConfig in both
    # parsers (round-14 C2 / round-15 C1: pinning only two knobs
    # left the other seven free to drift -- and the round-14 patch
    # for this silently failed to apply when an earlier section of
    # the same patch script aborted).
    from tools.plan_tournament import PT_KNOB_KEYS
    cfg = TournamentConfig()
    attr_of = {"challengers": "n_challengers"}
    for key in PT_KNOB_KEYS:
        flag = "--pt-" + key.replace("_", "-")
        want = getattr(cfg, attr_of.get(key, key))
        pat = (r'default="([\d,]+)"' if key == "depths"
               else r'default=([0-9.]+)')
        for src, name in ((ssp, "sim_self_play"), (wrk, "worker")):
            m = re.search(re.escape(f'"{flag}"')
                          + r'[\s\S]{0,120}?' + pat, src)
            assert m, f"{name}: no default found for {flag}"
            got = m.group(1)
            if key == "depths":
                got_v = tuple(int(x) for x in got.split(","))
                assert got_v == tuple(want), \
                    f"{name} {flag} default drifted: {got_v} vs " \
                    f"{tuple(want)}"
            else:
                assert float(got) == float(want), \
                    f"{name} {flag} default drifted: {got} vs {want}"


def test_reserve_covers_long_challengers():
    """Round-23 C2: cert_reserve prices both certification arms at
    the incumbent's spine length; the excess term re-prices the
    challenger arm (per replicate, plus the serving respine) at the
    pool's own max command-list length, capped where record_spine
    caps. Without it a longer challenger overran the reserve,
    dropped replicates, and the n-aware accept_rule taxed exactly
    the turn-lengthening plans."""
    from tools.plan_tournament import (
        cert_ph, cert_reserve, cert_reserve_excess, grade_cost)
    cfg = TournamentConfig()
    L, reps = 12, cfg.cert_redraws
    # No excess for a challenger no longer than the priced arm.
    assert cert_reserve_excess(L + 1, L, cfg, reps) == 0
    assert cert_reserve_excess(L - 3, L, cfg, reps) == 0
    # Identity: reserve + excess == the reserve with the challenger
    # arm (and the serve respine) priced at lc instead of L+1.
    for lc in (20, 40, 60):
        lc_cap = min(lc, cfg.max_spine)
        ph = cert_ph(cfg, 12)
        want = reps * ((L + 1) + lc_cap
                       + grade_cost(cfg.cert_depth, ph)) + lc_cap
        got = (cert_reserve(L, cfg, per_half=12, cert_redraws=reps)
               + cert_reserve_excess(lc, L, cfg, reps))
        assert got == want, (lc, got, want)


def test_reserve_bump_callsite_executes(monkeypatch):
    """The bump must run between pool construction and the selection
    loop (a rotted callsite is invisible to the arithmetic test --
    the round-4 launch-echo lesson)."""
    import tools.plan_tournament as pt
    calls = []
    orig = pt.cert_reserve_excess
    monkeypatch.setattr(
        pt, "cert_reserve_excess",
        lambda *a: (calls.append(a), orig(*a))[1])
    net = _tiny()
    sim = fresh_scenario_sim()
    side = sim.gs.global_info.current_side
    _plan, stats = pt.run_tournament(
        net, sim, side, 0, _cfg(), np.random.default_rng(11),
        salt_ns="t:rb")
    if stats["n_challengers"] > 0:
        assert calls and calls[0][0] >= 1


def test_reserve_bump_never_starves_selection(monkeypatch):
    """Round-24 C5 hardened in round-25 C2/C3: the excess bump is
    clamped AND the floor is priced at the same sel_ph cap the
    selection projections actually charge, so one complete graded
    comparison survives whenever challengers exist. The original
    test's challenger guard let it pass vacuously on the unclamped
    code, and the per_half-priced floor underfunded projections
    running to project_max_actions."""
    import tools.plan_tournament as pt
    monkeypatch.setattr(pt, "cert_reserve_excess",
                        lambda *a: 10 ** 6)
    net = _tiny()
    graded_any = False
    for seed in (11, 2, 3, 7):
        sim = fresh_scenario_sim()
        side = sim.gs.global_info.current_side
        _plan, stats = pt.run_tournament(
            net, sim, side, 0, _cfg(), np.random.default_rng(seed),
            salt_ns=f"t:clamp{seed}")
        if stats["n_challengers"] > 0:
            assert stats["graded"] > 0, \
                f"seed {seed}: challengers exist but selection " \
                f"was starved by the reserve bump"
            graded_any = True
    assert graded_any, ("no seed produced challengers -- the test "
                        "measured nothing; re-pick the seeds")


def test_censored_half_obs_cannot_ratchet_ema_down():
    """Round-26 C1: selection/cert projections are capped at a
    number derived from the EMA their observations feed, so an
    uncensored-mean update was a self-reinforcing ratchet (measured
    absorbing below the true mean). The censoring-aware estimate
    must RISE out of a shrunken state when the cap binds."""
    pol = PlanTournamentPolicy(
        _tiny(), MCTSConfig(n_simulations=2),
        tournament_config=_cfg())
    pol._t_half_ema = 1.0
    base = {"forwards": 0, "n_challengers": 0, "certified": 0,
            "abstained_budget": 0, "margin": None, "beta": 0.0,
            "graded": 0, "cert_attempts": 0, "arm_prefix": 0,
            "cert_replicates": 0, "cert_starved": 0,
            "fight_bucket": None, "cert_len_delta": None}
    for _ in range(20):
        pol._t_record(dict(base, half_obs=[(2, True)] * 10))
    assert pol._t_half_ema > 2.0, pol._t_half_ema
    # Mixed censoring: the right-censored mean exceeds the raw
    # truncated mean, so partial capping also pushes up.
    pol._t_half_ema = 1.0
    for _ in range(20):
        pol._t_record(dict(base,
                           half_obs=[(2, True), (2, True),
                                     (1, False)] * 3))
    assert pol._t_half_ema > 2.0, pol._t_half_ema
    st = pol.drain_tournament_stats()
    assert 0.0 < st["pt_half_cap_hit_rate"] <= 1.0


def test_project_counted_marks_cap_censoring():
    """The censoring flag comes from the production projection path
    (a rotted flag would silently re-open the ratchet)."""
    from tools.plan_tournament import _Budget, _project_counted
    net = _tiny()
    sim = fresh_scenario_sim()
    side = sim.gs.global_info.current_side
    obs = []
    v = _project_counted(net, sim, side, 0, 1, 1,
                         np.random.default_rng(0), _Budget(500),
                         salt="t:cens", half_obs=obs)
    assert v is not None
    assert obs and obs[0] == (1, True), obs


def test_certified_mass_matches_midgame_floor(monkeypatch):
    """Round-27 C4: the renorm factor's numerator must be the SAME
    divisor game_weight uses (midgame floor included) -- the raw
    state count shrank certified mass to a fraction of the
    pre-registered beta/2 on short midgame stubs."""
    from tools.mcts_policy import MCTSPolicy, side_weight_divisor
    monkeypatch.setattr(MCTSPolicy, "finalize_game",
                        lambda *a, **k: None)
    for midgame in (True, False):
        pol = PlanTournamentPolicy(
            _tiny(), MCTSConfig(n_simulations=2),
            tournament_config=_cfg())
        beta = 0.25

        class _S:
            def __init__(self, tgt):
                self.side = 1
                self.visit_counts = [1] if tgt else None
                self.policy_weight = beta if tgt else 1.0

        states = [_S(True), _S(False)]
        pol._pending["g"] = states
        pol.finalize_game("g", winner=1, midgame=midgame)
        gw = 1.0 / (2.0 * side_weight_divisor(2, midgame))
        assert abs(states[0].policy_weight * gw - beta / 2) < 1e-9, \
            (midgame, states[0].policy_weight, gw)
