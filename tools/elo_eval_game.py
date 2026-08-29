"""One Elo-ladder evaluation game between two player specs; result to
a JSON file. Designed to be launched N-way parallel (each game is an
independent process — the pattern that saturated a 4090 where the
central-server pool could not; see BACKLOG 2026-07-03).

Usage:
    python tools/elo_eval_game.py LABEL_A SPEC_A LABEL_B SPEC_B \
        SIDE_A SEED OUTDIR [--max-turns 200] [--mcts-sims 32]

SPEC is a checkpoint .pt path or the literal 'dummy' (scripted
baseline). Checkpoint players play through MCTS at --mcts-sims
(training-matched, 32) unless 0 (raw policy). Maps come from the
LADDER-ONLY default `random_setup` (pinned by test_elo_ladder_maps).

The result file records BOTH the outcome and the final material
margin from A's perspective, so the collector can fit Elo under the
PURE (primary -- decisive games only; a capped game is a no-result
absence, not a draw, user 2026-08-17; material advantage is a
training crutch and does not factor into evaluation, user
2026-07-11) and material-sign (diagnostic) conventions from one set
of games.
Eval search likewise runs WITHOUT the material shapers
(draw_tiebreak, aux_value_bonus) regardless of training config.
"""

from __future__ import annotations

import argparse
import json
import os
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.draw_tiebreak import DrawTiebreakConfig, material_margin
from tools.elo_ladder import _ScriptedAdapter
from tools.eval_sim import (_PolicyPair, _load_policy,
                            _play_one_eval_game)
from tools.scenario_pool import build_scenario_gamestate, random_setup
from tools.wesnoth_sim import WesnothSim

log = logging.getLogger("elo_eval_game")


def _search_policy_cls(turn_search: bool, plan_tournament: bool = False):
    """Deployment sampling matches the training default (user ruling
    2026-08-26). --plan-a/--plan-b selects the plan-tournament
    procedure (proposition 1); otherwise TCS unless --no-turn-search
    opts down to per-decision Gumbel MCTS."""
    if plan_tournament:
        from tools.plan_tournament import PlanTournamentPolicy
        return PlanTournamentPolicy
    if turn_search:
        from tools.turn_policy import TurnCommitPolicy
        return TurnCommitPolicy
    from tools.mcts_policy import MCTSPolicy
    return MCTSPolicy


from tools.eval_procedure import procedure_of as _procedure_of  # noqa: E402


# The knob machinery lives in tools/turn_search_config (torch-free,
# round-37 C3) so the batch driver can read it without pulling the
# sim stack.
from tools.turn_search_config import (  # noqa: E402
    TS_CHOICES, ts_config_from_args as _ts_config,
)


def _pt_config(args):
    """TournamentConfig for eval: explicit --pt-* knobs override the
    code defaults so a match can play the SAME config the leg
    trained with (review C16)."""
    from tools.plan_tournament import PT_KNOB_KEYS, config_from_args
    from types import SimpleNamespace
    ns = SimpleNamespace(plan_tournament=True)
    for key in PT_KNOB_KEYS:          # single source (round-11 C2)
        k = "pt_" + key
        v = getattr(args, k, None)
        if v is not None:
            setattr(ns, k, v)
    return config_from_args(ns)


class _CountingModel:
    """Transparent proxy counting net forwards for a player -- the
    step-1 mandate is EQUAL MEASURED FORWARDS (round-12 C5).
    Installed BEFORE any search wrapper is constructed, because
    MCTSPolicy caches base._inference_model at __init__ (round-13
    C0: attach-after-construction left the MCTS arm counting a
    fabricated zero). forward_batch counts per SAMPLE (round-13 C1:
    __getattr__ passthrough silently omitted every batched boundary
    forward from the TCS arm's count)."""

    def __init__(self, inner):
        self._inner = inner
        self.n_forwards = 0
        # Cumulative wall seconds inside the model -- recorded per
        # result so ms/forward stays measured under every precision/
        # compile config (user 2026-08-28: the compile+bf16 default
        # must remain reviewable from the result files alone).
        self.fwd_secs = 0.0

    def __call__(self, *a, **k):
        self.n_forwards += 1
        t0 = time.perf_counter()
        try:
            return self._inner(*a, **k)
        finally:
            self.fwd_secs += time.perf_counter() - t0

    def forward_batch(self, encs, *a, **k):
        self.n_forwards += len(encs)
        t0 = time.perf_counter()
        try:
            return self._inner.forward_batch(encs, *a, **k)
        finally:
            self.fwd_secs += time.perf_counter() - t0

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _build_player(spec: str, label: str, sims: int, device,
                  turn_search: bool = True,
                  plan_tournament: bool = False, pt_cfg=None,
                  ts_cfg=None, batch_size: int = 1,
                  infer_bf16: bool = False,
                  infer_compile: bool = False):
    if spec == "random":
        # Deliberate random-init reference (round-24 C8: reaching
        # random init through a nonexistent PATH is how a typo
        # produced a catalog edge against noise; the literal is the
        # only sanctioned route now).
        spec = None
    if spec == "dummy":
        from wesnoth_ai.dummy_policy import DummyPolicy
        return _ScriptedAdapter(DummyPolicy()), None
    policy = _load_policy(Path(spec) if spec else None, device,
                          label=label, infer_bf16=infer_bf16,
                          infer_compile=infer_compile)
    counter = _CountingModel(policy._inference_model)
    policy._inference_model = counter
    if sims > 0:
        from tools.mcts import MCTSConfig
        import os
        # EVALUATION CONTRACT (user, 2026-07-11): valuing material
        # advantage is a TRAINING crutch, not part of what policy
        # performance means -- so the material-based search shapers
        # (draw_tiebreak, aux_value_bonus) are OFF here regardless of
        # what the checkpoint trained with. Eval search sees the real
        # game: win +1, loss -1, draw 0. moves_left_utility (time
        # preference among equal outcomes, no material content) stays
        # env-configurable.
        cls = _search_policy_cls(turn_search, plan_tournament)
        mc = MCTSConfig(
            n_simulations=sims,
            batch_size=max(1, int(batch_size)),
            moves_left_utility=float(
                os.environ.get("ELO_MOVES_LEFT_UTILITY", "0") or 0))
        if plan_tournament:
            return cls(policy, mc, tournament_config=pt_cfg), counter
        if turn_search:
            # No config = dataclass defaults = a DIFFERENT estimand
            # than the leg trained (round-32 C3: boundary_frame
            # defaults to "opponent" while leg 5+ trains "mover").
            return cls(policy, mc, turn_config=ts_cfg), counter
        return cls(policy, mc), counter
    return policy, counter


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("label_a")
    ap.add_argument("spec_a")
    ap.add_argument("label_b")
    ap.add_argument("spec_b")
    ap.add_argument("side_a", type=int, choices=(1, 2))
    ap.add_argument("seed", type=int)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--mcts-sims", type=int, default=32)
    ap.add_argument("--mcts-sims-a", type=int, default=None,
                    help="Player A's sims budget (default: --mcts-sims). "
                         "0 = raw policy, so one match can play "
                         "search-vs-no-search on the SAME weights (the "
                         "does-search-help-at-all engine test).")
    ap.add_argument("--mcts-sims-b", type=int, default=None,
                    help="Player B's sims budget (see --mcts-sims-a).")
    ap.add_argument("--mcts-batch-size", type=int, default=1,
                    help="Leaf-evaluation batch (virtual-loss batching, "
                         "both players). 1 = sequential, the canonical "
                         "protocol and the CPU optimum (2026-04 "
                         "measurement, mcts.py header). On GPU, 8-32 "
                         "amortizes launch overhead (5-10x per the same "
                         "header). Recorded in the result file: batched "
                         "search explores differently, so B must never "
                         "mix within an outdir.")
    ap.add_argument("--infer-bf16", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="bfloat16 inference, both players. Default "
                         "AUTO: ON on cuda, OFF on cpu (user ruling "
                         "2026-08-28: compile+bf16 is the default -- "
                         "bench_infer measured 2.0x together, ~1x "
                         "each alone). Different logits = a "
                         "different measured object: the EFFECTIVE "
                         "value is recorded per result and never "
                         "mixes within an outdir. Explicitly forcing "
                         "it ON with a cpu device is refused (it "
                         "would silently no-op and mislabel).")
    ap.add_argument("--infer-compile", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="torch.compile the inference model. Default "
                         "AUTO: ON on cuda, OFF on cpu. See "
                         "--infer-bf16 for the ruling/provenance "
                         "contract; ~10-14s compile per shape bucket "
                         "per process, amortized via the shared "
                         "TORCHINDUCTOR_CACHE_DIR kernel cache.")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                    help="'auto' (default) uses CUDA when visible. PREFER "
                         "cuda when a GPU exists: profiled 2026-08-28, "
                         "86-90% of a CPU game is the model forward, and "
                         "the same game ran 10x faster on a 3060 at "
                         "~420MB VRAM per game process (6 concurrent = "
                         "2.5/12GB -- a 12GB card fits ~20 games). 'cpu' "
                         "remains for GPU-less boxes.")
    ap.add_argument("--no-turn-search", action="store_true",
                    help="BOTH players use per-decision Gumbel MCTS "
                         "instead of TCS. Default is TCS -- deployment "
                         "sampling matches the training default (user "
                         "ruling 2026-08-26). Pre-2026-08-26 catalog "
                         "numbers were measured with this flag's "
                         "behavior.")
    ap.add_argument("--no-turn-search-a", action="store_true",
                    help="Player A only plays MCTS (per-checkpoint "
                         "deployment: each side plays the sampling it "
                         "was trained for -- e.g. an imitation seed is "
                         "an MCTS-native checkpoint).")
    ap.add_argument("--no-turn-search-b", action="store_true",
                    help="Player B only plays MCTS (see "
                         "--no-turn-search-a).")
    ap.add_argument("--plan-a", action="store_true",
                    help="Player A plays the plan-tournament "
                         "procedure (proposition 1, 2026-08-26).")
    ap.add_argument("--plan-b", action="store_true",
                    help="Player B plays the plan-tournament "
                         "procedure.")
    # Plan-tournament knobs: the eval must be able to play the SAME
    # config the leg trained with (review C16: a knob-less eval
    # silently measures code defaults, a different estimand).
    ap.add_argument("--pt-challengers", type=int, default=None)
    ap.add_argument("--pt-depths", type=str, default=None)
    ap.add_argument("--pt-redraws", type=int, default=None)
    ap.add_argument("--pt-cert-depth", type=int, default=None)
    ap.add_argument("--pt-cert-redraws", type=int, default=None)
    ap.add_argument("--pt-budget-forwards", type=int, default=None)
    ap.add_argument("--pt-margin-band", type=float, default=None)
    ap.add_argument("--pt-beta-max", type=float, default=None)
    ap.add_argument("--pt-margin-ref", type=float, default=None)
    # TCS knobs (round-32 C3): the eval must be able to play the
    # SAME turn-search config the leg trained with; None = the
    # TurnSearchConfig dataclass default.
    ap.add_argument("--turn-alt", type=int, default=None)
    ap.add_argument("--turn-rounds", type=int, default=None)
    ap.add_argument("--turn-fast-rounds", type=int, default=None)
    ap.add_argument("--turn-reval-salts", type=int, default=None)
    ap.add_argument("--turn-min-delta", type=float, default=None)
    ap.add_argument("--turn-max-spine", type=int, default=None)
    ap.add_argument("--turn-full-prob", type=float, default=None)
    ap.add_argument("--turn-project", default=None,
                    choices=(None,) + TS_CHOICES["--turn-project"])
    ap.add_argument("--turn-project-halfturns", type=int,
                    default=None)
    ap.add_argument("--turn-project-max-actions", type=int,
                    default=None)
    ap.add_argument("--turn-target-link", default=None,
                    choices=(None,)
                    + TS_CHOICES["--turn-target-link"])
    ap.add_argument("--turn-target-beta", type=float, default=None)
    ap.add_argument("--turn-boundary-frame", default=None,
                    choices=(None,)
                    + TS_CHOICES["--turn-boundary-frame"])
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args(argv[1:])
    sims_a = (args.mcts_sims if args.mcts_sims_a is None
              else args.mcts_sims_a)
    sims_b = (args.mcts_sims if args.mcts_sims_b is None
              else args.mcts_sims_b)
    if (args.plan_a and sims_a <= 0) or (args.plan_b and sims_b <= 0):
        raise SystemExit(
            "--plan-a/--plan-b require that side's sims > 0 (sims 0 "
            "is the raw-policy player; a silently ignored procedure "
            "flag would mislabel the measured object).")
    for _n, _spec in (("spec_a", args.spec_a),
                      ("spec_b", args.spec_b)):
        if _spec not in ("dummy", "random") \
                and not Path(_spec).exists():
            raise SystemExit(
                f"{_n}={_spec!r} does not exist. A missing path "
                f"would silently play a RANDOM-INIT net under a "
                f"checkpoint's label and record a full Elo edge "
                f"against noise (round-24 C8). Pass the literal "
                f"'random' for a deliberate random-init player.")
    logging.basicConfig(level=getattr(logging, args.log_level))

    import torch
    torch.set_num_threads(2)
    if args.device == "cpu":
        device = None
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but no CUDA device is "
                             "visible; refusing to silently fall back to CPU "
                             "(an eval that quietly changes device is an "
                             "eval whose timings mean nothing).")
        device = torch.device("cuda")
    else:
        device = (torch.device("cuda") if torch.cuda.is_available() else None)

    # Precision/compile resolution (user ruling 2026-08-28:
    # compile+bf16 is the DEFAULT on cuda -- measured 2.0x together
    # on the real shape stream, ~1x each alone). On cpu both
    # default OFF; forcing them ON there is refused because they
    # would silently no-op and the result would be mislabeled.
    _cuda = device is not None and device.type == "cuda"
    inf_bf16 = _cuda if args.infer_bf16 is None else args.infer_bf16
    inf_compile = (_cuda if args.infer_compile is None
                   else args.infer_compile)
    if (inf_bf16 or inf_compile) and not _cuda:
        raise SystemExit(
            "--infer-bf16/--infer-compile require a cuda device: on "
            "cpu they no-op silently, so the result file would claim "
            "a precision that never ran.")
    logging.getLogger("elo_eval_game").warning(
        "inference config: bf16=%s compile=%s device=%s",
        inf_bf16, inf_compile, "cuda" if _cuda else "cpu")

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / (
        f"game_{args.label_a}_{args.label_b}_s{args.side_a}"
        f"_{args.seed}.json")
    if out_path.exists():
        # Procedure guard (review C14 round 3): a result produced
        # under a DIFFERENT decision procedure must never be
        # silently reused -- estimands don't mix.
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 -- unreadable = replay it
            prev = None
        if prev is not None:
            want_a = _procedure_of(
                sims_a, args.plan_a,
                args.no_turn_search or args.no_turn_search_a)
            want_b = _procedure_of(
                sims_b, args.plan_b,
                args.no_turn_search or args.no_turn_search_b)
            got_a = prev.get("procedure_a")
            got_b = prev.get("procedure_b")
            got_mt = prev.get("max_turns")
            # Absent field = 1: every pre-flag result was B=1.
            if prev.get("mcts_batch", 1) != args.mcts_batch_size:
                raise SystemExit(
                    f"{out_path.name} was played at leaf-batch "
                    f"B={prev.get('mcts_batch', 1)} but this run "
                    f"uses B={args.mcts_batch_size}: batched search "
                    f"explores differently, refusing to mix. Use a "
                    f"fresh outdir.")
            # Absent fields = False: pre-flag results were fp32 eager.
            if bool(prev.get("infer_bf16", False)) != inf_bf16:
                raise SystemExit(
                    f"{out_path.name} was played with infer_bf16="
                    f"{bool(prev.get('infer_bf16', False))} but this "
                    f"run uses {inf_bf16}: precision changes "
                    f"the logits, refusing to mix. Use a fresh "
                    f"outdir.")
            if bool(prev.get("infer_compile", False)) != inf_compile:
                raise SystemExit(
                    f"{out_path.name} was played with infer_compile="
                    f"{bool(prev.get('infer_compile', False))} but "
                    f"this run uses {inf_compile}: compiled kernels "
                    f"may reorder float ops, refusing to mix. Use a "
                    f"fresh outdir.")
            if (got_a, got_b, got_mt) != (want_a, want_b,
                                          args.max_turns):
                raise SystemExit(
                    f"{out_path.name} exists with procedure/horizon "
                    f"({got_a},{got_b},max_turns={got_mt}) but this "
                    f"run wants ({want_a},{want_b},max_turns="
                    f"{args.max_turns}): refusing to mix estimands "
                    f"in one outdir. Use a fresh outdir.")
            if "pt_config" in prev and (args.plan_a or args.plan_b):
                from tools.plan_tournament import pt_knobs_dict
                cur = _pt_config(args)
                cur_knobs = None if cur is None else pt_knobs_dict(cur)
                if prev.get("pt_config") != cur_knobs:
                    raise SystemExit(
                        f"{out_path.name} was played under a "
                        f"different --pt-* config: refusing to mix "
                        f"(round-4 C12). Use a fresh outdir.")
            _any_tcs = (
                (sims_a > 0 and not args.plan_a
                 and not (args.no_turn_search
                          or args.no_turn_search_a))
                or (sims_b > 0 and not args.plan_b
                    and not (args.no_turn_search
                             or args.no_turn_search_b)))
            if _any_tcs or "turn_config" in prev:
                from tools.turn_search import turn_knobs_dict
                _want_tc = (turn_knobs_dict(_ts_config(args))
                            if _any_tcs else None)
                if prev.get("turn_config") != _want_tc:
                    raise SystemExit(
                        f"{out_path.name} was played under a "
                        f"different turn-search config "
                        f"({prev.get('turn_config')} vs {_want_tc})"
                        f": refusing to mix estimands (round-32 "
                        f"C3). Use a fresh outdir.")
            print(f"exists, skipping: {out_path.name}")
            return 0

    pt_cfg = _pt_config(args) if (args.plan_a or args.plan_b) else None
    ts_cfg = _ts_config(args)
    pa, cnt_a = _build_player(
        args.spec_a, args.label_a, sims_a, device,
        turn_search=not (args.no_turn_search or args.no_turn_search_a),
        plan_tournament=args.plan_a, pt_cfg=pt_cfg, ts_cfg=ts_cfg,
        batch_size=args.mcts_batch_size, infer_bf16=inf_bf16,
        infer_compile=inf_compile)
    pb, cnt_b = _build_player(
        args.spec_b, args.label_b, sims_b, device,
        turn_search=not (args.no_turn_search or args.no_turn_search_b),
        plan_tournament=args.plan_b, pt_cfg=pt_cfg, ts_cfg=ts_cfg,
        batch_size=args.mcts_batch_size, infer_bf16=inf_bf16,
        infer_compile=inf_compile)


    rng = random.Random(args.seed)
    setup = random_setup(rng)
    gs = build_scenario_gamestate(setup)
    sim = WesnothSim(gs, scenario_id=setup.scenario_id,
                     max_turns=args.max_turns)
    game_label = out_path.stem
    t0 = time.time()
    r = _play_one_eval_game(
        sim,
        _PolicyPair(policy=pa, label=args.label_a, side=args.side_a),
        _PolicyPair(policy=pb, label=args.label_b, side=3 - args.side_a),
        game_label=game_label)
    margin_a = material_margin(sim.gs, args.side_a,
                               DrawTiebreakConfig(cap=0.3))
    if pt_cfg is None:
        pt_knobs = None
    else:
        from tools.plan_tournament import pt_knobs_dict
        pt_knobs = pt_knobs_dict(pt_cfg)
    _tcs_played = (
        (sims_a > 0 and not args.plan_a
         and not (args.no_turn_search or args.no_turn_search_a))
        or (sims_b > 0 and not args.plan_b
            and not (args.no_turn_search or args.no_turn_search_b)))
    if _tcs_played:
        from tools.turn_search import turn_knobs_dict
        ts_knobs = turn_knobs_dict(ts_cfg)
    else:
        ts_knobs = None
    result = {
        "label_a": args.label_a, "label_b": args.label_b,
        "pt_config": pt_knobs,
        "turn_config": ts_knobs,
        # Procedure provenance (review C19): result files from
        # different estimands must never be silently mergeable.
        "procedure_a": _procedure_of(
            sims_a, args.plan_a,
            args.no_turn_search or args.no_turn_search_a),
        "procedure_b": _procedure_of(
            sims_b, args.plan_b,
            args.no_turn_search or args.no_turn_search_b),
        # The horizon decides decisive-vs-absence, the quantity
        # the PURE fit is built on (round-24 C9).
        "max_turns": args.max_turns,
        # Leaf-batch provenance: batched (virtual-loss) search is a
        # slightly different explorer than sequential B=1.
        "mcts_batch": args.mcts_batch_size,
        # Precision/compile provenance (EFFECTIVE values): bf16
        # logits differ from fp32's; compiled kernels may reorder
        # float ops.
        "infer_bf16": inf_bf16,
        "infer_compile": inf_compile,
        "side_a": args.side_a, "seed": args.seed,
        "scenario_id": setup.scenario_id,
        "outcome_a": r.outcome,          # win/loss/draw/timeout from A
        "margin_a": float(margin_a),     # final material, A's view
        "turns": sim.gs.global_info.turn_number,
        # Measured forward counts (round-12 C5): the step-1 equal-
        # compute mandate is verified from these, per side. Per-
        # side-turn = forwards / turns (each side moves once/turn).
        "forwards_a": (cnt_a.n_forwards if cnt_a else None),
        "forwards_b": (cnt_b.n_forwards if cnt_b else None),
        # Wall seconds spent inside the model per side, so
        # ms/forward stays measured under whatever precision/
        # compile/device config -- the standing review record for
        # the compile+bf16 default (user 2026-08-28).
        "fwd_secs_a": (round(cnt_a.fwd_secs, 2) if cnt_a else None),
        "fwd_secs_b": (round(cnt_b.fwd_secs, 2) if cnt_b else None),
        "ended_by": sim.ended_by,
        "secs": round(time.time() - t0, 1),
    }
    # Atomic publish (round-24 C11): a kill mid-write must never
    # leave a truncated file occupying the slot.
    _tmp = out_path.with_suffix(".json.tmp")
    _tmp.write_text(json.dumps(result), encoding="utf-8")
    os.replace(_tmp, out_path)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
