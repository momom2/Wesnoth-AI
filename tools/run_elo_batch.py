"""Resumable, memory-guarded driver for a paired Elo run.

`elo_eval_game.py` is one game per process and already skips a result
file that exists, so a long ladder is really "loop over (side, seed)
until done". This driver is that loop plus the two things that made a
long run impossible to babysit here:

  * **A wall-clock budget.** It stops cleanly at `--time-budget-min`, so
    it can be run in short chunks that accumulate into one games dir.
    Re-running continues where it left off; nothing is recomputed.
  * **A memory guard.** Measured 2026-08-03: this laptop has 7.6 GB
    total, and with a browser open only ~0.6 GB was free. A torch
    process under that pressure page-thrashes rather than computes -- one
    eval game got ~1 s of CPU in 9 min of wall clock and produced
    nothing. Starting a game with no memory does not just run slowly, it
    wastes the whole slot and can take the machine down. So refuse.

Side assignment alternates so the pair is balanced: an odd game index
puts A on side 2. Seeds are derived from the index, so the same command
always schedules the same games and two chunks never collide.

Usage (raw-policy A/B -- `--mcts-sims 0` is what makes it RAW):
    python tools/run_elo_batch.py \\
        --label-a best  --spec-a training/checkpoints/campaign_live_20260730.pt \\
        --label-b anchor --spec-b training/checkpoints/selfplay_seed_20260718.pt \\
        --games 400 --outdir eval_games/tc_raw --mcts-sims 0 \\
        --time-budget-min 55

Then fit (decisive games only -- capped games are no-result absences,
user ruling 2026-08-17; see elo_collect.py):
    python tools/elo_collect.py eval_games/tc_raw
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

log = logging.getLogger("run_elo_batch")

# Below this, a torch process thrashes instead of running (see module
# docstring). Generous on purpose: the cost of pausing is one idle slot,
# the cost of proceeding is a wasted slot or a hung machine.
DEFAULT_MIN_FREE_MB = 1800


def free_mb() -> Optional[float]:
    """Memory THIS process tree can still use, in MB, or None if it
    cannot be determined (guard skipped rather than guessed at).
    Cgroup-aware: on a rented container the host's psutil reading
    is a fiction -- the host may show 100GB free while our slice is
    2GB from the OOM-killer. host_resources takes the binding
    minimum of the two."""
    from tools.host_resources import available_mb       # noqa: PLC0415
    return available_mb()


def result_name(label_a: str, label_b: str, side_a: int, seed: int) -> str:
    """Mirror elo_eval_game.py's output name so we can tell, without
    launching anything, whether this game is already done."""
    return f"game_{label_a}_{label_b}_s{side_a}_{seed}.json"


def slot_for(i: int, seed_base: int) -> Tuple[int, int]:
    """(side_a, seed) for BASE slot index i."""
    return (1 if i % 2 == 0 else 2), seed_base + i


# > any --games range, so replacement seeds stay disjoint from base
# seeds and from each other across generations.
REPL_SEED_OFFSET = 1_000_000


def replacement_slot_for(i: int, seed_base: int, gen: int,
                         ) -> Tuple[int, int]:
    """(side_a, seed) of base slot i's generation-`gen` game (gen 0
    = the base slot itself). A replacement KEEPS the side of the
    slot it replaces -- deriving it from an append index broke the
    side balance exactly when turn-cap no-results correlate with
    side (round-30 C5: 12 side-2 caps became 6/6, biasing the fit
    by ~0.3x the side advantage) -- and the per-slot chain is
    deterministic on resume regardless of completion order."""
    side_a, seed = slot_for(i, seed_base)
    return side_a, seed + gen * REPL_SEED_OFFSET


_UNREADABLE = "unreadable"


def _close_err(errf) -> None:
    """Close and remove a child's stderr file; tolerate every OS
    hiccup (the log is diagnostic, never load-bearing)."""
    try:
        name = errf.name
        errf.close()
        Path(name).unlink(missing_ok=True)
    except OSError:
        pass


def _err_tail(errf, n: int = 4096) -> str:
    """Last n bytes of a child's stderr file, then close+remove."""
    try:
        # The CHILD wrote through the inherited handle; the
        # parent's position is still 0, so seek from the file's
        # real size, not tell().
        size = os.fstat(errf.fileno()).st_size
        errf.seek(max(0, size - n))
        out = errf.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        out = ""
    _close_err(errf)
    return out


def outcome_of(path: Path) -> Optional[str]:
    """outcome_a of a finished game file; the _UNREADABLE sentinel
    for a file that exists but does not parse (truncated by a
    kill). Distinct from a no-result absence (round-24 C11)."""
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("outcome_a")
    except Exception:                                   # noqa: BLE001
        return _UNREADABLE


def scan_slots(outdir: Path, label_a: str, label_b: str, games: int,
               seed_base: int, max_extra: int,
               ) -> Tuple[int, int, List[Tuple[int, int, int, Path]],
                          int]:
    """Walk each base slot's replacement CHAIN (gen 0 = the base
    slot; each no-result earns the next same-side generation, up to
    `max_extra` replacements across all slots -- the hard guard that
    bounds worst-case run time even if every game caps).
    Classification per user ruling 2026-08-17: a capped game is not
    a draw; it is a no-result absence with zero rating information.
    Deterministic on resume: chains depend only on the files, never
    on completion order.

    Returns (n_results, n_no_result, pending_slots, extra_used)."""
    # Pass 1 CLASSIFIES every file on disk (order-free); pass 2
    # BUDGETS new replacements. Interleaving them spent the guard
    # chain-first, so a resume under-counted decisive replacement
    # games already on disk and re-allocated the guard differently
    # than the live loop had (round-31 C2).
    n_results = n_no_result = 0
    spent = 0
    pending: List[Tuple[int, int, int, Path, int]] = []
    want_repl: List[Tuple[int, int]] = []
    for i in range(games):
        gen = 0
        while True:
            side_a, seed = replacement_slot_for(i, seed_base, gen)
            out = outdir / result_name(label_a, label_b, side_a,
                                       seed)
            if not out.exists():
                if gen == 0:
                    pending.append((i, side_a, seed, out, 0))
                else:
                    want_repl.append((i, gen))
                break
            if gen >= 1:
                spent += 1        # an existing replacement file
            oc = outcome_of(out)
            if oc in ("win", "loss"):
                n_results += 1
                break
            if oc == _UNREADABLE:
                # Truncated leftover of a killed child: REPLAY the
                # slot (elo_eval_game overwrites an unreadable
                # file) instead of burning a replacement on a
                # phantom absence (round-24 C11).
                pending.append((i, side_a, seed, out, gen))
                break
            n_no_result += 1
            gen += 1
    for i, gen in want_repl:
        if spent >= max_extra:
            break
        spent += 1
        side_a, seed = replacement_slot_for(i, seed_base, gen)
        out = outdir / result_name(label_a, label_b, side_a, seed)
        pending.append((i, side_a, seed, out, gen))
    return n_results, n_no_result, pending, spent


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label-a", required=True)
    ap.add_argument("--spec-a", required=True)
    ap.add_argument("--label-b", required=True)
    ap.add_argument("--spec-b", required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--games", type=int, default=400,
                    help="Decisive RESULTS wanted in outdir (not per "
                         "chunk). A capped/stalled game is a no-result "
                         "absence (user ruling 2026-08-17) and earns a "
                         "replacement slot, bounded by "
                         "--max-extra-games.")
    ap.add_argument("--max-extra-games", type=int, default=None,
                    help="Hard guard on replacement slots for "
                         "no-result games (default: games // 2). "
                         "Bounds worst-case run time even if every "
                         "game caps; past the guard, absences are "
                         "recorded and the CI simply widens.")
    ap.add_argument("--mcts-sims", type=int, default=0,
                    help="0 = RAW policy (no search). 32 = training-matched.")
    ap.add_argument("--mcts-sims-a", type=int, default=None,
                    help="Player A's sims budget (default: --mcts-sims). "
                         "0 = raw policy, so one match can play "
                         "search-vs-no-search on the SAME weights.")
    ap.add_argument("--mcts-sims-b", type=int, default=None,
                    help="Player B's sims budget (see --mcts-sims-a).")
    ap.add_argument("--mcts-batch-size", type=int, default=1,
                    help="Leaf-evaluation batch for search, both "
                         "players. 1 = sequential (canonical, CPU "
                         "optimum); 8-32 on GPU amortizes launch "
                         "overhead. Never mixes within an outdir.")
    ap.add_argument("--infer-bf16", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="bfloat16 inference, both players. Default "
                         "AUTO: each game turns it ON iff its device "
                         "is cuda (compile+bf16 default, user ruling "
                         "2026-08-28). Effective value recorded per "
                         "result; never mixes within an outdir.")
    ap.add_argument("--infer-compile", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="torch.compile inference. Default AUTO (ON "
                         "iff cuda); see --infer-bf16.")
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=10_000)
    ap.add_argument("--time-budget-min", type=float, default=55.0,
                    help="Stop cleanly after this long. Re-run to continue.")
    ap.add_argument("--min-free-mb", type=float, default=DEFAULT_MIN_FREE_MB)
    ap.add_argument("--device", default="auto",
                    choices=("auto", "cpu", "cuda"),
                    help="Passed to each game. On a GPU box use 'cuda' "
                         "even with --jobs > 1: profiled 2026-08-28, "
                         "eval games are 86-90% model forward, cuda ran "
                         "10x faster, and a game process holds only "
                         "~420MB VRAM (a 12GB card fits ~20 concurrent "
                         "games). 'cpu' is for GPU-less boxes.")
    ap.add_argument("--jobs", type=int, default=None,
                    help="Concurrent games. Default: AUTO-SIZED to this "
                         "box -- min over cgroup CPU quota (2 threads/"
                         "game), cgroup-aware memory headroom "
                         "(--per-job-mb each), and free VRAM when the "
                         "device is not cpu (--per-job-vram-mb each) -- "
                         "because boxes change per leg and hand-tuned "
                         "numbers do not transfer. Pass an integer to "
                         "override. Each game is a separate process (the "
                         "pattern that saturated a 4090 where a central "
                         "pool could not).")
    ap.add_argument("--per-job-mb", type=float, default=2000.0,
                    help="Assumed RAM per game process for auto --jobs. "
                         "Conservative estimate pending a recorded "
                         "measurement; completed-game RSS is logged so "
                         "future runs can tighten it.")
    ap.add_argument("--per-job-vram-mb", type=float, default=600.0,
                    help="Assumed VRAM per game process for auto --jobs "
                         "on a GPU (measured ~420MB on a 3060, "
                         "2026-08-28; headroom included).")
    ap.add_argument("--no-turn-search", action="store_true",
                    help="BOTH players: per-decision Gumbel MCTS instead "
                         "of TCS (the pre-2026-08-26 catalog protocol). "
                         "Default is TCS: deployment sampling matches "
                         "the training default (user ruling 2026-08-26).")
    ap.add_argument("--no-turn-search-a", action="store_true",
                    help="Player A only plays MCTS (per-checkpoint "
                         "deployment; e.g. an imitation seed is "
                         "MCTS-native).")
    ap.add_argument("--no-turn-search-b", action="store_true",
                    help="Player B only plays MCTS.")
    ap.add_argument("--plan-a", action="store_true",
                    help="Player A plays the plan-tournament procedure.")
    ap.add_argument("--plan-b", action="store_true",
                    help="Player B plays the plan-tournament procedure.")
    ap.add_argument("--pt-args", type=str, default=None,
                    help="Extra --pt-* knobs forwarded verbatim to "
                         "every elo_eval_game (space-separated), so "
                         "a match plays the leg's training config.")
    ap.add_argument("--ts-args", type=str, default=None,
                    help="Extra --turn-* knobs forwarded to every "
                         "elo_eval_game (space-separated), so a TCS "
                         "match plays the leg's turn-search config "
                         "-- e.g. '--turn-boundary-frame mover' for "
                         "leg 5+ (round-32 C3).")
    ap.add_argument("--per-game-timeout-min", type=float, default=20.0,
                    help="Kill a single game that overruns; its slot is "
                         "skipped and the run continues.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    sims_a = (args.mcts_sims if args.mcts_sims_a is None
              else args.mcts_sims_a)
    sims_b = (args.mcts_sims if args.mcts_sims_b is None
              else args.mcts_sims_b)
    if (args.plan_a and sims_a <= 0) or (args.plan_b and sims_b <= 0):
        # Refuse up front (review C15): elo_eval_game rejects this
        # combination per game, so a default-sims batch would fail
        # every child yet exit 0.
        ap.error("--plan-a/--plan-b require that side's sims > 0")
    if args.pt_args and not (args.plan_a or args.plan_b):
        ap.error("--pt-args is inert without --plan-a/--plan-b")
    if args.device == "cpu" and (args.infer_bf16 or args.infer_compile):
        # Refuse up front: every child would refuse per game.
        ap.error("--infer-bf16/--infer-compile require a cuda device")
    _any_tcs = (
        (sims_a > 0 and not args.plan_a
         and not (args.no_turn_search or args.no_turn_search_a))
        or (sims_b > 0 and not args.plan_b
            and not (args.no_turn_search or args.no_turn_search_b)))
    if args.ts_args and not _any_tcs:
        ap.error("--ts-args is inert without a TCS arm")
    if args.ts_args:
        _ts_known = {"--turn-alt": int, "--turn-rounds": int,
                     "--turn-fast-rounds": int,
                     "--turn-reval-salts": int,
                     "--turn-min-delta": float,
                     "--turn-max-spine": int,
                     "--turn-full-prob": float,
                     "--turn-project": str,
                     "--turn-project-halfturns": int,
                     "--turn-project-max-actions": int,
                     "--turn-target-link": str,
                     "--turn-target-beta": float,
                     "--turn-boundary-frame": str}
        from tools.turn_search_config import TS_CHOICES
        _ts_toks = args.ts_args.split()
        _p = None
        for t in _ts_toks:
            if _p is not None:
                if _p in TS_CHOICES and t not in TS_CHOICES[_p]:
                    ap.error(f"--ts-args: bad value {t!r} for "
                             f"{_p} (choices: {TS_CHOICES[_p]})")
                try:
                    _ts_known[_p](t)
                except ValueError:
                    ap.error(f"--ts-args: bad value {t!r} for {_p}")
                _p = None
                continue
            if t not in _ts_known:
                ap.error(f"--ts-args: unknown knob {t!r} "
                         f"(known: {sorted(_ts_known)})")
            _p = t
        if _p is not None:
            ap.error("--ts-args: trailing knob without a value")
    if args.pt_args:
        # Validate forwarded knobs up front (review C16 round 3): a
        # typo would make every child exit 2 while the batch still
        # returned 0.
        from tools.plan_tournament import PT_KNOB_KEYS
        _known = {"--pt-" + k.replace("_", "-") for k in PT_KNOB_KEYS}
        _types = {"--pt-challengers": int, "--pt-redraws": int,
                  "--pt-cert-depth": int, "--pt-cert-redraws": int,
                  "--pt-budget-forwards": int,
                  "--pt-margin-band": float, "--pt-beta-max": float,
                  "--pt-margin-ref": float, "--pt-depths": str}
        toks = args.pt_args.split()
        pending = None
        for t in toks:
            if pending is not None:
                try:
                    if pending == "--pt-depths":
                        [int(x) for x in t.split(",")]
                    else:
                        _types[pending](t)
                except ValueError:
                    ap.error(f"--pt-args: bad value {t!r} for "
                             f"{pending} (round-6 C6: a mistyped "
                             f"value would kill every child)")
                pending = None
                continue
            if t not in _known:
                ap.error(f"--pt-args: unknown knob {t!r} "
                         f"(known: {sorted(_known)})")
            pending = t
        if pending is not None:
            ap.error("--pt-args: trailing knob without a value")
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    for _n, _spec in (("--spec-a", args.spec_a),
                      ("--spec-b", args.spec_b)):
        if _spec not in ("dummy", "random") \
                and not Path(_spec).exists():
            raise SystemExit(
                f"{_n}={_spec!r} does not exist -- every child "
                f"would play a RANDOM-INIT net under this label "
                f"(round-24 C8). Pass the literal 'random' for a "
                f"deliberate random-init player.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    deadline = time.perf_counter() + args.time_budget_min * 60.0
    # The concurrency decision is always logged WITH its derivation
    # (user 2026-08-28): retro-judging a box's throughput needs the
    # inputs of the choice, not just the number. Peak-RSS lines at
    # game completion are the other half of that audit.
    from tools.host_resources import auto_jobs
    _auto, how = auto_jobs(
        per_job_mb=args.per_job_mb,
        per_job_vram_mb=(None if args.device == "cpu"
                         else args.per_job_vram_mb))
    if args.jobs is None:
        jobs = _auto
        log.info("auto-sized --jobs: %s", how)
    else:
        jobs = max(1, args.jobs)
        log.info("explicit --jobs %d (auto would pick: %s)",
                 jobs, how)
    # Every concurrent game needs its own headroom, so the floor scales.
    floor = args.min_free_mb * jobs
    _peak_rss: dict = {}   # pid -> max sampled RSS (MB), best effort
    played = failed = 0
    max_extra = (args.games // 2 if args.max_extra_games is None
                 else args.max_extra_games)

    # TCS knob dict this run plays (None when no TCS arm) --
    # computed ONCE from the torch-free config module (round-37
    # C3: importing it via elo_eval_game/turn_search pulled torch
    # + the sim stack into the driver the memory guard sizes, and
    # the raw-mode resume imported it for nothing).
    _want_tc = None
    if _any_tcs:
        from types import SimpleNamespace
        from tools.turn_search_config import (ts_config_from_args,
                                              turn_knobs_dict)
        _tn0 = SimpleNamespace()
        _tt0 = (args.ts_args or "").split()
        for k_, v_ in zip(_tt0[0::2], _tt0[1::2]):
            setattr(_tn0, k_.lstrip("-").replace("-", "_"), v_)
        _want_tc = turn_knobs_dict(ts_config_from_args(_tn0))
    # Procedure pre-scan (review C14 round 3): scan_slots counts
    # existing files as results WITHOUT launching a child, so the
    # per-game procedure guard never fires for them -- a stale-
    # estimand outdir would be silently reused. Refuse here.
    from tools.eval_procedure import procedure_of
    want = (procedure_of(sims_a, args.plan_a,
                          args.no_turn_search or args.no_turn_search_a),
            procedure_of(sims_b, args.plan_b,
                          args.no_turn_search or args.no_turn_search_b))
    for f in sorted(args.outdir.glob("game_*.json")):
        try:
            prev = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 -- unreadable = replayed
            continue
        got = (prev.get("procedure_a"), prev.get("procedure_b"))
        if prev.get("max_turns") != args.max_turns:
            raise SystemExit(
                f"{f.name} was played at max_turns="
                f"{prev.get('max_turns')} but this run uses "
                f"{args.max_turns}: the horizon decides decisive-"
                f"vs-absence, so estimands don't mix -- fresh "
                f"outdir (round-24 C9).")
        # Absent field = 1: every pre-flag result was B=1.
        if prev.get("mcts_batch", 1) != args.mcts_batch_size:
            raise SystemExit(
                f"{f.name} was played at leaf-batch B="
                f"{prev.get('mcts_batch', 1)} but this run uses "
                f"B={args.mcts_batch_size}: batched search explores "
                f"differently, refusing to mix. Use a fresh outdir.")
        # Precision/compile pre-scan: the effective value is known
        # here only when the flag is explicit or the device is
        # forced; otherwise the per-game guard still protects.
        for _fld, _flag in (("infer_bf16", args.infer_bf16),
                            ("infer_compile", args.infer_compile)):
            _want = (_flag if _flag is not None
                     else {"cuda": True, "cpu": False}.get(args.device))
            if _want is not None \
                    and bool(prev.get(_fld, False)) != _want:
                raise SystemExit(
                    f"{f.name} was played with {_fld}="
                    f"{bool(prev.get(_fld, False))} but this run "
                    f"uses {_want}: numerics differ, refusing to "
                    f"mix. Use a fresh outdir.")
        if got == want and (args.plan_a or args.plan_b):
            # Same procedure but possibly different --pt-* knobs: a
            # chunked resume must not mix plan-tournament configs in
            # one outdir (round-5 C8; the per-game guard only fires
            # on slots it is about to SKIP, not on new slots writing
            # a different config alongside). The lazy import pulls
            # torch only in plan mode, where children pay it anyway.
            from types import SimpleNamespace
            from tools.elo_eval_game import _pt_config
            ns = SimpleNamespace()
            toks = (args.pt_args or "").split()
            for k_, v_ in zip(toks[0::2], toks[1::2]):
                setattr(ns, k_.lstrip("-").replace("-", "_"), v_)
            from tools.plan_tournament import pt_knobs_dict
            cur = _pt_config(ns)
            cur_knobs = None if cur is None else pt_knobs_dict(cur)
            if "pt_config" in prev and prev["pt_config"] != cur_knobs:
                raise SystemExit(
                    f"{f.name} was played under a different --pt-* "
                    f"config: estimands don't mix -- fresh outdir.")
        if got != want:
            # Legacy files without procedure fields refuse too
            # (round-4 C11: the eval-side guard treats (None,None)
            # as a mismatch; the batch must not be laxer).
            raise SystemExit(
                f"{f.name} holds procedure {got} but this run is "
                f"{want}: estimands don't mix -- use a fresh outdir.")
        if _any_tcs or "turn_config" in prev:
            # TCS knob parity (round-32 C3): the frame the leg
            # trains with is part of the estimand. _want_tc is
            # computed ONCE, torch-free (round-37 C3).
            if prev.get("turn_config") != _want_tc:
                raise SystemExit(
                    f"{f.name} was played under a different "
                    f"turn-search config: estimands don't mix -- "
                    f"use a fresh outdir (round-32 C3).")

    n_results, n_nores, pending, extra = scan_slots(
        args.outdir, args.label_a, args.label_b, args.games,
        args.seed_base, max_extra)
    log.info("%d results done, %d no-result (replacements used "
             "%d/%d), %d pending, %d concurrent, device=%s",
             n_results, n_nores, extra, max_extra, len(pending),
             jobs, args.device)

    def launch(slot):
        i, side_a, seed, _out, _gen = slot
        cmd = [sys.executable, "-u", str(_THIS.parent / "elo_eval_game.py"),
               args.label_a, args.spec_a, args.label_b, args.spec_b,
               str(side_a), str(seed), str(args.outdir),
               "--mcts-sims", str(args.mcts_sims),
               "--max-turns", str(args.max_turns),
               "--device", args.device]
        if args.mcts_sims_a is not None:
            cmd += ["--mcts-sims-a", str(args.mcts_sims_a)]
        if args.mcts_sims_b is not None:
            cmd += ["--mcts-sims-b", str(args.mcts_sims_b)]
        if args.mcts_batch_size != 1:
            cmd += ["--mcts-batch-size", str(args.mcts_batch_size)]
        if args.infer_bf16 is not None:
            cmd.append("--infer-bf16" if args.infer_bf16
                       else "--no-infer-bf16")
        if args.infer_compile is not None:
            cmd.append("--infer-compile" if args.infer_compile
                       else "--no-infer-compile")
        if args.no_turn_search:
            cmd.append("--no-turn-search")
        if args.no_turn_search_a:
            cmd.append("--no-turn-search-a")
        if args.no_turn_search_b:
            cmd.append("--no-turn-search-b")
        if args.plan_a:
            cmd.append("--plan-a")
        if args.plan_b:
            cmd.append("--plan-b")
        if args.pt_args:
            # =-joined so a value with a leading '-' (e.g. a
            # negative --pt-margin-band) survives the child's
            # argparse (round-27 C5: space-separated, every child
            # exited 2 and the batch still returned 0). The strict
            # knob/value alternation is validated at parse time.
            _toks = args.pt_args.split()
            cmd.extend(f"{k}={v}"
                       for k, v in zip(_toks[0::2], _toks[1::2]))
        if args.ts_args:
            _toks = args.ts_args.split()
            cmd.extend(f"{k}={v}"
                       for k, v in zip(_toks[0::2], _toks[1::2]))
        # Per-child stderr FILE, never an undrained PIPE (round-28
        # C3: a chatty child filled the 64KB pipe buffer, blocked
        # on write() forever, and was killed by the per-game
        # timeout -- every game "timed out" while finishing in
        # seconds standalone). Dot-prefixed so the game_*.json
        # globs never see it.
        errf = open(args.outdir / f".stderr_{i}_{seed}.log", "w+b")
        return (subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                 stderr=errf),
                time.perf_counter(), slot, errf)

    # Provenance for timeout artifacts (round-32 C5): the file a
    # kill leaves behind must satisfy the per-file and pre-scan
    # guards on later chunks, or the slot is re-launched with no
    # bound on every resume.
    _prov = {"label_a": args.label_a, "label_b": args.label_b,
             "procedure_a": want[0], "procedure_b": want[1],
             "max_turns": args.max_turns, "pt_config": None,
             "turn_config": None}
    if args.plan_a or args.plan_b:
        from types import SimpleNamespace
        from tools.elo_eval_game import _pt_config
        from tools.plan_tournament import pt_knobs_dict
        _pn = SimpleNamespace()
        _ptt = (args.pt_args or "").split()
        for k_, v_ in zip(_ptt[0::2], _ptt[1::2]):
            setattr(_pn, k_.lstrip("-").replace("-", "_"), v_)
        _pc = _pt_config(_pn)
        _prov["pt_config"] = (None if _pc is None
                              else pt_knobs_dict(_pc))
    if _any_tcs:
        _prov["turn_config"] = _want_tc

    def schedule_replacement(base_i, cur_gen):
        nonlocal extra
        if extra >= max_extra:
            return False
        extra += 1
        rs, rseed = replacement_slot_for(base_i, args.seed_base,
                                         cur_gen + 1)
        rout = args.outdir / result_name(args.label_a, args.label_b,
                                         rs, rseed)
        if not rout.exists():
            pending.append((base_i, rs, rseed, rout, cur_gen + 1))
        return True

    running = []
    stop = False
    try:
        # `stop` gates only ADMISSION; the loop keeps polling until
        # every in-flight child exits (bounded by the per-game
        # timeout kills below) -- returning with live children let
        # a resume run the same slots concurrently with orphans
        # (round-24 C7).
        while running or (pending and not stop):
            while pending and len(running) < jobs and not stop:
                if time.perf_counter() > deadline:
                    log.info("time budget reached — no new games; "
                             "draining %d in flight", len(running))
                    stop = True
                    break
                fm = free_mb()
                if fm is not None and fm < floor and not running:
                    # Only hard-stop when nothing is in flight; otherwise let
                    # the running games finish and free their memory first.
                    log.error(
                        "only %.0f MB free (need %.0f for %d job(s)). A torch "
                        "process below this thrashes instead of running. Close "
                        "applications or lower --jobs, then re-run; finished "
                        "games are kept.", fm, floor, jobs)
                    stop = True
                    break
                if fm is not None and fm < floor:
                    break                       # wait for a slot to free memory
                running.append(launch(pending.pop(0)))

            if not running:
                break
            time.sleep(2.0)
            # Peak-RSS sampling (best effort): --per-job-mb is an
            # assumption until a box has logged real numbers; the
            # "peak_rss" lines below are that record.
            try:
                import psutil                           # noqa: PLC0415
                for _p, _t, _slot, _e in running:
                    _rss = (psutil.Process(_p.pid).memory_info().rss
                            / (1024 ** 2))
                    _peak_rss[_p.pid] = max(
                        _peak_rss.get(_p.pid, 0.0), _rss)
            except Exception:                           # noqa: BLE001
                pass
            for entry in list(running):
                proc, t0, (i, side_a, seed, out, gen), errf = entry
                elapsed = time.perf_counter() - t0
                if proc.poll() is None:
                    if elapsed > args.per_game_timeout_min * 60.0:
                        proc.kill()
                        proc.wait()
                        _close_err(errf)
                        running.remove(entry)
                        if (out.exists()
                                and outcome_of(out) != _UNREADABLE):
                            # The child PUBLISHED before the kill
                            # landed (interpreter teardown takes
                            # ~0.3-0.6s after os.replace; round-33
                            # C1): keep the real result instead of
                            # clobbering a decisive game into an
                            # absence.
                            played += 1
                            if outcome_of(out) in ("win", "loss"):
                                n_results += 1
                            else:
                                n_nores += 1
                                schedule_replacement(i, gen)
                            log.info(
                                "game %d finished inside the kill "
                                "window (%s); result kept", i,
                                outcome_of(out))
                            continue
                        failed += 1
                        # Persist the kill as a no-result artifact
                        # (round-32 C5: an empty slot was re-
                        # launched identically on EVERY resume --
                        # observed 27/40 timeouts in the leg-5
                        # verdict -- with no bound and no
                        # replacement). Atomic, so a Ctrl-C here
                        # cannot leave a truncated file.
                        _art = dict(_prov, side_a=side_a, seed=seed,
                                    outcome_a="timeout_kill",
                                    margin_a=None,
                                    timeout_min=(
                                        args.per_game_timeout_min))
                        _tmpf = out.with_suffix(".json.tmp")
                        _tmpf.write_text(json.dumps(_art),
                                         encoding="utf-8")
                        os.replace(_tmpf, out)
                        n_nores += 1
                        _sched = schedule_replacement(i, gen)
                        log.warning(
                            "game %d (gen %d) timed out after %.0f "
                            "min; recorded as no-result, "
                            "replacement %s (guard %d/%d)", i, gen,
                            args.per_game_timeout_min,
                            "scheduled" if _sched else "guard spent",
                            extra, max_extra)
                    continue
                running.remove(entry)
                if proc.returncode == 0 and out.exists():
                    _close_err(errf)   # failure branch tails it instead
                    played += 1
                    if outcome_of(out) in ("win", "loss"):
                        n_results += 1
                    else:
                        # No-result absence: schedule ONE replacement
                        # slot past the base range, unless the guard is
                        # spent (bounded worst-case, user 2026-08-17).
                        n_nores += 1
                        if schedule_replacement(i, gen):
                            log.info("game %d (gen %d) was "
                                     "no-result (%s); same-side "
                                     "replacement gen %d scheduled "
                                     "(guard %d/%d)", i, gen,
                                     outcome_of(out), gen + 1,
                                     extra, max_extra)
                        else:
                            log.warning("game %d was no-result; guard "
                                        "exhausted (%d/%d) -- absence "
                                        "recorded, CI will widen", i,
                                        extra, max_extra)
                else:
                    failed += 1
                    err = _err_tail(errf)
                    log.warning("game %d (side %d, seed %d) failed rc=%s: %s",
                                i, side_a, seed, proc.returncode,
                                err.strip()[-200:])
                _pk = _peak_rss.pop(proc.pid, None)
                log.info("game %d done in %.1f min%s (results=%d/%d "
                         "no_result=%d failed=%d, %d pending, %d in "
                         "flight)", i, elapsed / 60.0,
                         (f", peak_rss {_pk:.0f}MB"
                          if _pk else ""), n_results,
                         args.games, n_nores, failed, len(pending),
                         len(running))

    finally:
        for _proc, _t0, (_i, _sa, _sd, _out, _g), _errf in running:
            if _proc.poll() is None:
                _proc.kill()
                _proc.wait()
                log.warning("killed in-flight game %d at exit", _i)
            _close_err(_errf)
    total = len(list(args.outdir.glob("game_*.json")))
    # Report as a fraction, never a percentage or an extrapolation.
    log.info("chunk end: %d/%d RESULTS (%d no-result absences, "
             "replacements %d/%d; %d files) in %s (this chunk: %d "
             "played, %d failed)",
             n_results, args.games, n_nores, extra, max_extra, total,
             args.outdir, played, failed)
    if n_results < args.games and (pending or extra < max_extra):
        log.info("re-run the same command to continue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
