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

A game that ends without a result file leaves failed_<game>.json (the
slot, the return code, the tail of its stderr) and stays unplayed, so a
re-run replays it. The exit status says whether the match is done
(EXIT_MEANING): 0 complete, 1 failed, 3 games left to play, 4 the
replacement guard is spent short of --games.
"""

from __future__ import annotations

import argparse
import hashlib
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

from wesnoth_ai.constants import OBSERVATION_EPOCH  # noqa: E402
from wesnoth_ai.paths import REPO_ROOT, TOOLS_DIR  # noqa: E402

log = logging.getLogger("run_elo_batch")

# The script that plays one game (or, with --worker, many).
GAME_SCRIPT = TOOLS_DIR / "elo_eval_game.py"
# Seconds between two polls of the games in flight.
POLL_S = 2.0

# Exit status of a chunk: whether the match is done, so a script acts
# on it without counting files. A time-budget cut, a memory stop and
# failed games all used to exit 0 with a short outdir.
EXIT_COMPLETE = 0
EXIT_FAILED = 1
EXIT_RESUMABLE = 3
EXIT_GUARD_SPENT = 4
EXIT_MEANING = {
    EXIT_COMPLETE: "the match is complete: --games decisive results",
    EXIT_FAILED: "games failed or an inference server died (see the failed_*.json "
                 "records); a re-run replays the failed games once the cause is fixed",
    EXIT_RESUMABLE: "the time budget or the memory guard left games to play; re-run "
                    "the same command to continue",
    EXIT_GUARD_SPENT: "every slot is played with fewer decisive results than --games: "
                      "the replacement guard is spent, a re-run adds nothing, and the "
                      "fit's interval widens",
}

# Below this, a torch process thrashes instead of running (see module
# docstring). Generous on purpose: the cost of pausing is one idle slot,
# the cost of proceeding is a wasted slot or a hung machine.
DEFAULT_MIN_FREE_MB = 1800
# A worker under --shared-inference holds no model: it ships the
# encoded leaf to the server and waits. Measured peak RSS per game
# process on that path is about 400 MB (the per-game telemetry in
# eval_games/), so the full-fat floor refuses jobs a box can easily
# run -- on a 31 GB box it capped a 20-job match at 13. The shared
# floor keeps a 1.7x margin over the measurement.
SHARED_INFERENCE_MIN_FREE_MB = 700


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


# Hex-basis provenance (2026-09-05 review). A side plays in the relevant
# hex subset ("relset") when its --relevant-set flag is set, when its
# checkpoint carries relevant_set_hexes, or when the inference server
# serving it does; otherwise on the full board ("full"). The basis
# changes the tokens the model attends over, so it is an estimand
# field: recorded per side in every result file and never mixed within
# an outdir. Shared by elo_eval_game (writer) and elo_collect (reader);
# this module stays torch-free, so the import is cheap for both.
BASES = ("full", "relset")


def bases_of(record: dict) -> Tuple[str, str]:
    """(basis_a, basis_b) of a result file. Files from before the field
    existed played the full board."""
    return (record.get("basis_a") or "full", record.get("basis_b") or "full")


def basis_refusal(name: str, record: dict, want: Tuple[str, str]) -> Optional[str]:
    """The refusal to keep `record` in an outdir whose games play in
    the `want` bases; None when they agree."""
    got = bases_of(record)
    if got == tuple(want):
        return None
    return (f"{name} was played in hex bases (a={got[0]}, b={got[1]}) but this "
            f"run plays (a={want[0]}, b={want[1]}): the basis changes the tokens "
            f"the model attends over, refusing to mix. Use a fresh outdir.")


# The terrain view (2026-09-19): a side's encoder carries each hex's
# terrain as its full set from the engine's aliases ("set", the
# checkpoint flag terrain_multi_hot: on for every fresh network) or as
# one class per hex ("class", every checkpoint before that day). It
# changes the hex tokens, so it is an estimand field like the basis:
# recorded per side, never mixed within an outdir, compared between
# dirs by the catalog. There is no CLI override: a checkpoint plays in
# its own view, a served side in its server's.
TERRAIN_VIEWS = ("class", "set")


def terrain_views_of(record: dict) -> Tuple[str, str]:
    """(terrain_a, terrain_b) of a result file. Files from before the
    field existed played the one-class view."""
    return (record.get("terrain_a") or "class", record.get("terrain_b") or "class")


def terrain_refusal(name: str, record: dict, want: Tuple[str, str]) -> Optional[str]:
    """The refusal to keep `record` in an outdir whose games play in
    the `want` terrain views; None when they agree."""
    got = terrain_views_of(record)
    if got == tuple(want):
        return None
    return (f"{name} was played in terrain views (a={got[0]}, b={got[1]}) but this "
            f"run plays (a={want[0]}, b={want[1]}): the view changes the hex tokens, "
            f"refusing to mix. Use a fresh outdir.")


# Which checkpoint each side played (2026-09-25): the SHA-256 of the
# file's bytes, recorded per side in every result file as
# checkpoint_sha256_a/_b (None for 'dummy' and 'random', which load no
# file). A label names a checkpoint only by convention, so a resume that
# pointed --spec-a at other weights under the same label mixed two
# players in one outdir without a word.
def file_sha256(path: Path) -> str:
    """The SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spec_sha256(spec: str) -> Optional[str]:
    """The SHA-256 of a player spec's checkpoint file; None for the
    literals 'dummy' and 'random', which load none."""
    if spec in ("dummy", "random"):
        return None
    return file_sha256(Path(spec))


def checkpoint_refusal(name: str, record: dict,
                       want: Tuple[Optional[str], Optional[str]]) -> Optional[str]:
    """The refusal to keep `record` in an outdir whose sides play the
    checkpoints `want` (SHA-256 per side); None when they agree. A file
    written before the field existed cannot say which checkpoint played
    a checkpoint side, so it is refused too."""
    def named(sha: Optional[str]) -> str:
        return f"checkpoint {sha}" if sha else "no checkpoint ('dummy' or 'random')"

    for side, sha in zip("ab", want):
        field = f"checkpoint_sha256_{side}"
        got = record.get(field)
        if got == sha:
            continue
        if field not in record:
            return (f"{name} does not record which checkpoint played side {side} (it "
                    f"predates the field) and this run plays {named(sha)}: refusing to "
                    f"mix what cannot be compared. Use a fresh outdir.")
        return (f"{name} was played by {named(got)} on side {side} but this run plays "
                f"{named(sha)} under the same label: refusing to mix two players. Use a "
                f"fresh outdir or another label.")
    return None


# The faction forced onto one side of every eval game (elo_eval_game
# passes scenario_pool.FORCED_FACTION to random_setup). It changes the
# games, so it is an estimand field: recorded in every result as the
# faction's name, or "none" when no faction is forced. Every result file
# from before the field was played with the Knalgan Alliance forced:
# the constant's value since 2026-04-30, and the eval has drawn its
# setups through random_setup's default since 2026-07-04.
LEGACY_FORCED_FACTION = "Knalgan Alliance"


def forced_faction_tag(faction: Optional[str]) -> str:
    """How a result records the forced faction: its name, or "none"."""
    return faction or "none"


def forced_faction_of(record: dict) -> str:
    return record.get("forced_faction", LEGACY_FORCED_FACTION)


def faction_refusal(name: str, record: dict, want: str) -> Optional[str]:
    """The refusal to keep `record` in an outdir whose games force the
    faction `want`; None when they agree."""
    got = forced_faction_of(record)
    if got == want:
        return None
    return (f"{name} was played with the forced faction {got!r} but this run forces "
            f"{want!r}: the faction changes the games, refusing to mix. Use a fresh "
            f"outdir.")


def failure_name(label_a: str, label_b: str, side_a: int, seed: int) -> str:
    """The failure record of a game slot, beside its result file's name
    (never matched by the result globs, game_*.json)."""
    return f"failed_{label_a}_{label_b}_s{side_a}_{seed}.json"


def record_failure(outdir: Path, label_a: str, label_b: str, side_a: int, seed: int, *,
                   slot: int, gen: int, returncode, reason: str) -> Path:
    """Write the record of a game that ended without a result file. The
    slot stays unplayed, so a re-run replays it; the record stays too,
    counting the slot's failures."""
    path = outdir / failure_name(label_a, label_b, side_a, seed)
    try:
        failures = int(json.loads(path.read_text(encoding="utf-8"))["failures"])
    except (OSError, ValueError, KeyError, TypeError):
        failures = 0
    rec = {"label_a": label_a, "label_b": label_b, "side_a": side_a, "seed": seed,
           "slot": slot, "gen": gen, "returncode": returncode, "reason": reason,
           "failures": failures + 1, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return path


def dead_servers(servers: dict) -> List[str]:
    """One line per inference server process that has exited: its spec,
    return code and the tail of its log."""
    return [f"{spec} (rc={h.proc.returncode}): {h.err_tail()[-800:].strip()}"
            for spec, handles in servers.items() for h in handles if not h.alive()]


def match_status(outdir: Path, label_a: str, label_b: str, games: int, seed_base: int,
                 max_extra: int, failed: bool) -> int:
    """The chunk's exit status (EXIT_*), read from the files on disk the
    way a resume reads them."""
    n_results, _n_nores, pending, _extra = scan_slots(
        outdir, label_a, label_b, games, seed_base, max_extra)
    if failed:
        return EXIT_FAILED
    if n_results >= games:
        return EXIT_COMPLETE
    return EXIT_RESUMABLE if pending else EXIT_GUARD_SPENT


_PEEK_FLAGS = (
    "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[2]); "
    "from tools.eval_sim import peek_checkpoint_arch; "
    "f = peek_checkpoint_arch(Path(sys.argv[1]), sys.argv[1]); "
    "print(('relset' if f.get('relevant_set_hexes') else 'full') + ' ' "
    "+ ('set' if f.get('terrain_multi_hot') else 'class'))")
_FLAGS_MEMO: dict = {}


def _checkpoint_flags(spec: str) -> Tuple[str, str]:
    """(basis, terrain view) a checkpoint spec plays in on its own,
    read ONCE per spec in a child interpreter so the driver stays
    torch-free. 'random' is a fresh net: the full board, and the set
    view every fresh network carries; 'dummy' has no encoder."""
    if spec == "dummy":
        return ("full", "class")
    if spec == "random":
        return ("full", "set")
    if spec in _FLAGS_MEMO:
        return _FLAGS_MEMO[spec]
    proc = subprocess.run(
        [sys.executable, "-c", _PEEK_FLAGS, spec, str(REPO_ROOT)],
        capture_output=True, text=True, timeout=600)
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    parts = lines[-1].split() if lines else []
    if (proc.returncode != 0 or len(parts) != 2 or parts[0] not in BASES
            or parts[1] not in TERRAIN_VIEWS):
        raise SystemExit(f"could not read the hex basis and terrain view of {spec!r}: "
                         f"{proc.stderr.strip()[-500:]}")
    _FLAGS_MEMO[spec] = (parts[0], parts[1])
    return _FLAGS_MEMO[spec]


def _checkpoint_basis(spec: str) -> str:
    """The basis a checkpoint spec plays in on its own: 'relset' when it
    carries relevant_set_hexes (see `_checkpoint_flags`)."""
    return _checkpoint_flags(spec)[0]


def _checkpoint_terrain(spec: str) -> str:
    """The terrain view a checkpoint spec plays in on its own: 'set'
    when it carries terrain_multi_hot (see `_checkpoint_flags`)."""
    return _checkpoint_flags(spec)[1]


def _want_terrains(args, terrain_of_spec) -> Tuple[str, str]:
    """The terrain view each side of this batch plays in: 'dummy' has
    no encoder; otherwise `terrain_of_spec(spec)` (the checkpoint's
    flag, or the server's)."""
    return tuple("class" if spec == "dummy" else terrain_of_spec(spec)
                 for spec in (args.spec_a, args.spec_b))


def _want_bases(args, basis_of_spec) -> Tuple[str, str]:
    """The basis each side of this batch plays in: 'dummy' has no
    encoder; a --relevant-set flag forces the subset; otherwise
    `basis_of_spec(spec)` (the checkpoint's flag, or the server's)."""
    out = []
    for spec, flag in ((args.spec_a, args.relevant_set_a),
                       (args.spec_b, args.relevant_set_b)):
        if spec == "dummy":
            out.append("full")
        elif flag:
            out.append("relset")
        else:
            out.append(basis_of_spec(spec))
    return out[0], out[1]


def _effective_precision(args, field: str) -> bool:
    """What elo_eval_game will record for `field` under this batch's
    flags: the explicit flag, else the device default (cuda on, cpu
    off), resolving --device auto the way the child does."""
    flag = getattr(args, field)
    if flag is not None:
        return bool(flag)
    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device == "cuda"


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


def _check_shared_inference_args(ap, args, sims_a: int, sims_b: int) -> None:
    """--shared-inference serves the certified surface only: raw
    players (sims 0, a raw temperature) on checkpoint specs, through
    persistent workers, eager kernels."""
    if not args.persistent_workers:
        ap.error("--shared-inference requires --persistent-workers")
    if args.infer_compile:
        ap.error("--shared-inference serves eager kernels; drop --infer-compile")
    ckpt_sides = [(s, spec, sims, temp) for s, spec, sims, temp in (
        ("a", args.spec_a, sims_a, args.raw_temperature_a),
        ("b", args.spec_b, sims_b, args.raw_temperature_b)) if spec != "dummy"]
    if not ckpt_sides:
        ap.error("--shared-inference is inert with no checkpoint side")
    for side, spec, sims, temp in ckpt_sides:
        if spec == "random":
            ap.error(f"--shared-inference cannot serve 'random' for side {side}: "
                     f"one fixed random-init net for the whole match is a "
                     f"different estimand than a fresh draw per game")
        if sims > 0 or temp is None:
            ap.error(f"--shared-inference serves the raw player only: side {side} "
                     f"needs sims 0 and --raw-temperature-{side}")


def _shutdown_servers(servers: dict) -> dict:
    """Stop every inference server; the stats each wrote, by spec. A
    spec may be served by several processes (--inference-servers), in
    which case its entry is the list of their stats."""
    stats = {}
    for spec, handles in servers.items():
        per = [h.shutdown() for h in handles]
        stats[spec] = per[0] if len(per) == 1 else per
    servers.clear()
    return stats


def _log_server_stats(stats: dict) -> None:
    for spec, st in list(stats.items()):
        if isinstance(st, list):                 # several servers for one spec
            for k, one in enumerate(st):
                _log_server_stats({f"{spec}#{k}": one})
            continue
        if not st:
            log.warning("inference server for %s left no stats", spec)
            continue
        if st.get("fatal"):
            log.error("inference server %s stopped serving on a device fault:\n%s",
                      Path(spec).name, st["fatal"][-2000:])
        log.info("inference server %s: %d requests in %d batches, mean batch "
                 "%.2f, hist %s; idle %.1fs window %.1fs infer %.1fs reply %.1fs "
                 "of %.1fs wall (gpu %.1fs)", Path(spec).name, st["requests"],
                 st["batches"], st["mean_batch"], st["batch_hist"], st["idle_s"],
                 st["window_s"], st["infer_s"], st["reply_s"], st["wall_s"],
                 st["gpu_ms"] / 1000.0)


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
    ap.add_argument("--value-center-a", type=float, default=0.0,
                    help="Search value centering for player A "
                         "(elo_eval_game --value-center-a).")
    ap.add_argument("--value-center-b", type=float, default=0.0)
    ap.add_argument("--shared-combat-stream", action="store_true",
                    help="Pass --shared-combat-stream to every game: "
                         "the pre-2026-09-13 luck stream every eval "
                         "game had in common. Reproduce old numbers "
                         "only.")
    ap.add_argument("--relevant-set-a", action="store_true",
                    help="elo_eval_game --relevant-set-a (relevant hex subset "
                         "for side A's encoder; fresh outdir).")
    ap.add_argument("--relevant-set-b", action="store_true")
    ap.add_argument("--gumbel-root-a", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Side A's plain-search root: Gumbel (default, "
                         "'mcts:<sims>') or --no-gumbel-root-a for plain "
                         "PUCT ('puct:<sims>', the az legs' training "
                         "search). elo_eval_game --gumbel-root-a.")
    ap.add_argument("--gumbel-root-b", action=argparse.BooleanOptionalAction,
                    default=True, help="Side B (see --gumbel-root-a).")
    ap.add_argument("--raw-temperature-a", type=float, default=None,
                    help="Player A at sims 0 plays the joint-temperature "
                         "raw player (elo_eval_game --raw-temperature-a; "
                         "0 = argmax). Default None = legacy sampler.")
    ap.add_argument("--raw-temperature-b", type=float, default=None,
                    help="Player B (see --raw-temperature-a).")
    ap.add_argument("--raw-end-turn-a", choices=("joint", "actor"), default="joint",
                    help="Player A's end_turn decode (elo_eval_game "
                         "--raw-end-turn-a): 'actor' = end_turn only when its "
                         "actor mass leads every actor marginal. Procedure "
                         "tag '+endm'.")
    ap.add_argument("--raw-end-turn-b", choices=("joint", "actor"), default="joint",
                    help="Player B (see --raw-end-turn-a).")
    ap.add_argument("--raw-end-turn-offset-a", type=float, default=0.0,
                    help="Offset on player A's end_turn actor logit (procedure "
                         "tag '+eo<x>').")
    ap.add_argument("--raw-end-turn-offset-b", type=float, default=0.0,
                    help="Player B (see --raw-end-turn-offset-a).")
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
    ap.add_argument("--min-free-mb", type=float, default=None,
                    help=f"Free system memory to require per concurrent game before "
                         f"starting a chunk. Default {DEFAULT_MIN_FREE_MB:.0f} MB, or "
                         f"{SHARED_INFERENCE_MIN_FREE_MB:.0f} MB under --shared-inference, "
                         f"where the worker holds no model (measured peak about 400 MB "
                         f"per game).")
    ap.add_argument("--device", default="auto",
                    choices=("auto", "cpu", "cuda"),
                    help="Passed to each game. On a GPU box use 'cuda' "
                         "even with --jobs > 1: profiled 2026-08-28, "
                         "eval games are 86-90%% model forward, cuda ran "
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
    ap.add_argument("--persistent-workers", action="store_true",
                    help="Play games through --jobs long-lived "
                         "elo_eval_game --worker processes that keep "
                         "the loaded policies across games "
                         "(tools/eval_workers.py) instead of one "
                         "process per game. Same result files, same "
                         "timeouts; a timed-out worker is replaced.")
    ap.add_argument("--shared-inference", action="store_true",
                    help="With --persistent-workers: one inference server "
                         "process per distinct checkpoint owns the model "
                         "on the GPU and serves the workers' forwards in "
                         "coalesced batches (tools/eval_inference_server.py). "
                         "Raw players (sims 0, a raw temperature) only. "
                         "Results record shared_inference and the "
                         "precision path; never mixes with per-process "
                         "games in one outdir. Re-pin raw:t0 against "
                         "itself once before quoting a gate through it.")
    ap.add_argument("--packed-embed", action=argparse.BooleanOptionalAction, default=True,
                    help="Shared inference: the server embeds each batch as one packed "
                         "sequence on the host (the pool's default).")
    ap.add_argument("--compile-packed", action="store_true",
                    help="Shared inference: the server serves the compiled packed layer "
                         "loop (docs/box_specs.md 2026-09-11: the per-batch launch "
                         "overhead is most of a small batch's cost). bf16 numerics differ "
                         "slightly from the eager loop; the game records carry the switch "
                         "and an outdir never mixes them.")
    ap.add_argument("--graphed-serve", action="store_true",
                    help="Shared inference: the server replays each priors batch from a "
                         "per-bucket CUDA graph (wesnoth_ai/graphed_serve.py; cuda + bf16 "
                         "+ packed trunk), one launch per batch instead of ~150.")
    ap.add_argument("--inference-servers", type=int, default=1,
                    help="Inference server PROCESSES per distinct checkpoint "
                         "(default 1). The eval path is bound by one server's "
                         "per-batch cycle, most of which is a fixed GPU launch "
                         "cost, so a second process on the same GPU raises the "
                         "ceiling; games are handed to servers round-robin. "
                         "Costs one model copy of VRAM per extra process.")
    ap.add_argument("--inference-window-ms", type=float, default=1.5,
                    help="Shared inference: after a first request, how long "
                         "the server collects more before one forward.")
    ap.add_argument("--inference-max-batch", type=int, default=None,
                    help="Shared inference: leaves per forward (default: "
                         "--jobs; with one decision in flight per worker a "
                         "larger value is inert).")
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
    if ((args.raw_temperature_a is not None and sims_a > 0)
            or (args.raw_temperature_b is not None and sims_b > 0)):
        ap.error("--raw-temperature-a/-b apply to a side at sims 0 only")
    from tools.eval_procedure import end_turn_refusal
    for _side, _spec, _sims, _temp in (("a", args.spec_a, sims_a, args.raw_temperature_a),
                                       ("b", args.spec_b, sims_b, args.raw_temperature_b)):
        _why = end_turn_refusal(_side, _spec, _sims, _temp,
                                getattr(args, f"raw_end_turn_{_side}"),
                                getattr(args, f"raw_end_turn_offset_{_side}"))
        if _why is not None:
            ap.error(_why)
    if args.label_a == args.label_b:
        ap.error("--label-a and --label-b must differ (result files and "
                 "the workers' per-side policy cache are keyed by label)")
    if args.shared_inference:
        _check_shared_inference_args(ap, args, sims_a, sims_b)
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
        # Shared inference: the workers hold no model, so no VRAM, and
        # they wait on the server two thirds of the time, so 1.25 per
        # core keeps the server's batches full (docs/box_specs.md
        # 2026-09-11: 20 workers on 16 cores 94 s, 32 workers 125 s).
        threads_per_job=(0.8 if args.shared_inference else 2),
        per_job_vram_mb=(None if args.device == "cpu" or args.shared_inference
                         else args.per_job_vram_mb))
    if args.jobs is None:
        jobs = _auto
        log.info("auto-sized --jobs: %s", how)
    else:
        jobs = max(1, args.jobs)
        log.info("explicit --jobs %d (auto would pick: %s)",
                 jobs, how)
    # Every concurrent game needs its own headroom, so the floor scales.
    min_free = args.min_free_mb
    if min_free is None:
        min_free = (SHARED_INFERENCE_MIN_FREE_MB if args.shared_inference
                    else DEFAULT_MIN_FREE_MB)
    floor = min_free * jobs
    _peak_rss: dict = {}   # pid -> max sampled RSS (MB), best effort
    # failed: games that ended without a result file (each leaves a
    # failure record); timed_out: games killed on the per-game timeout
    # (each leaves a no-result artifact, like a capped game).
    played = failed = timed_out = 0
    server_died = False
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
                          args.no_turn_search or args.no_turn_search_a,
                          args.raw_temperature_a, args.gumbel_root_a,
                          raw_end_turn=args.raw_end_turn_a,
                          raw_end_turn_offset=args.raw_end_turn_offset_a),
            procedure_of(sims_b, args.plan_b,
                          args.no_turn_search or args.no_turn_search_b,
                          args.raw_temperature_b, args.gumbel_root_b,
                          raw_end_turn=args.raw_end_turn_b,
                          raw_end_turn_offset=args.raw_end_turn_offset_b))
    # Hex-basis pre-scan (see BASES): per-process bases are read from
    # the checkpoints here; under shared inference the servers report
    # theirs once launched (below), and the scan repeats there.
    want_bases = want_terrains = None
    if not args.shared_inference:
        want_bases = _want_bases(args, _checkpoint_basis)
        want_terrains = _want_terrains(args, _checkpoint_terrain)
    # The checkpoint each side plays and the faction every game forces
    # (see checkpoint_refusal, faction_refusal).
    _sha_of = {spec: spec_sha256(spec) for spec in {args.spec_a, args.spec_b}}
    want_ckpts = (_sha_of[args.spec_a], _sha_of[args.spec_b])
    from tools import scenario_pool
    want_faction = forced_faction_tag(scenario_pool.FORCED_FACTION)
    for f in sorted(args.outdir.glob("game_*.json")):
        try:
            prev = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 -- unreadable = replayed
            continue
        if want_bases is not None:
            _why = basis_refusal(f.name, prev, want_bases)
            if _why is not None:
                raise SystemExit(_why)
            _why = terrain_refusal(f.name, prev, want_terrains)
            if _why is not None:
                raise SystemExit(_why)
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
        # Combat-luck regime (absent = the pre-2026-09-13 shared
        # stream). The per-game guard in elo_eval_game only fires on
        # slots it is about to SKIP, so a resume into an old outdir
        # would otherwise append per-game-stream games beside shared-
        # stream ones without a word.
        _prev_epoch = int(prev.get("observation_epoch", 1))
        if _prev_epoch != int(OBSERVATION_EPOCH):
            raise SystemExit(
                f"{f.name} was played under observation epoch {_prev_epoch} but this "
                f"sim is {OBSERVATION_EPOCH}: the players saw different games, refusing "
                f"to mix (constants.OBSERVATION_EPOCH). Use a fresh outdir.")
        _want_cs = ("shared" if args.shared_combat_stream
                    else "per_game")
        if prev.get("combat_stream", "shared") != _want_cs:
            raise SystemExit(
                f"{f.name} was played on combat_stream="
                f"{prev.get('combat_stream', 'shared')} but this run "
                f"uses {_want_cs}: the shared stream gives every game "
                f"the same luck vector, so the two are different "
                f"estimands. Use a fresh outdir.")
        # Precision/compile pre-scan: the effective value is known
        # here only when the flag is explicit or the device is
        # forced; otherwise the per-game guard still protects.
        for _fld, _flag in (("infer_bf16", args.infer_bf16),
                            ("infer_compile", args.infer_compile)):
            _want = (_flag if _flag is not None
                     else {"cuda": True, "cpu": False}.get(args.device))
            if _fld == "infer_compile" and args.shared_inference:
                _want = False            # the server runs eager kernels
            if _want is not None \
                    and bool(prev.get(_fld, False)) != _want:
                raise SystemExit(
                    f"{f.name} was played with {_fld}="
                    f"{bool(prev.get(_fld, False))} but this run "
                    f"uses {_want}: numerics differ, refusing to "
                    f"mix. Use a fresh outdir.")
        # Shared-inference forwards are batched (and packed on cuda):
        # different numerics from the per-process single-sample
        # forward. Absent = per-process. The packed-trunk field is
        # checked once the servers report it (below).
        if bool(prev.get("shared_inference", False)) != bool(args.shared_inference):
            raise SystemExit(
                f"{f.name} was played with shared_inference="
                f"{bool(prev.get('shared_inference', False))} but this "
                f"run uses {bool(args.shared_inference)}: batched "
                f"forwards have different numerics, refusing to mix. "
                f"Use a fresh outdir.")
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
        for _why in (checkpoint_refusal(f.name, prev, want_ckpts),
                     faction_refusal(f.name, prev, want_faction)):
            if _why is not None:
                raise SystemExit(_why)

    n_results, n_nores, pending, extra = scan_slots(
        args.outdir, args.label_a, args.label_b, args.games,
        args.seed_base, max_extra)
    log.info("%d results done, %d no-result (replacements used "
             "%d/%d), %d pending, %d concurrent, device=%s",
             n_results, n_nores, extra, max_extra, len(pending),
             jobs, args.device)

    worker_pool = None
    if args.persistent_workers:
        from tools.eval_workers import WorkerPool
        worker_pool = WorkerPool(
            [sys.executable, "-u", str(GAME_SCRIPT),
             "--worker"], jobs, args.outdir)
        log.info("persistent workers: up to %d elo_eval_game --worker "
                 "processes, policies cached across games", jobs)

    # Shared inference: one server per distinct checkpoint (both sides
    # of a same-spec match share one). The server reports its
    # effective precision path; every game records it, and existing
    # files must agree with it.
    servers: dict = {}
    shared_bf16 = shared_packed = False
    if args.shared_inference:
        from tools.eval_inference_server import launch_inference_server
        max_batch = args.inference_max_batch or jobs
        try:
            n_per_spec = max(1, int(args.inference_servers))
            for k, spec in enumerate(dict.fromkeys(
                    s for s in (args.spec_a, args.spec_b) if s != "dummy")):
                servers[spec] = []
                for j in range(n_per_spec):
                    handle = launch_inference_server(
                        spec, args.outdir, tag=f"{k}_{j}" if n_per_spec > 1 else str(k),
                        device=args.device,
                        infer_bf16=args.infer_bf16,
                        window_ms=args.inference_window_ms, max_batch=max_batch,
                        packed_embed=bool(args.packed_embed),
                        compile_packed=bool(args.compile_packed),
                        graphed=bool(args.graphed_serve))
                    servers[spec].append(handle)
                    log.info("inference server %d/%d for %s at %s: %s", k, j, spec,
                             handle.address, handle.info)
            infos = {(bool(h.info["infer_bf16"]), bool(h.info["packed_trunk"]))
                     for hs in servers.values() for h in hs}
            if len(infos) != 1:
                raise SystemExit(f"the inference servers disagree on precision "
                                 f"{sorted(infos)}; one match, one numerics path")
            shared_bf16, shared_packed = infos.pop()
            # The games record the checkpoint their server loaded; the
            # pre-scan and the timeout artifacts use the driver's read.
            for spec, hs in servers.items():
                for h in hs:
                    if h.info.get("checkpoint_sha256") != _sha_of[spec]:
                        raise SystemExit(
                            f"the inference server for {spec} loaded checkpoint "
                            f"{h.info.get('checkpoint_sha256')} but the driver read "
                            f"{_sha_of[spec]}: the file changed on disk between the two "
                            f"reads. Re-run once it no longer changes.")
            # A served side plays in the server's basis (its
            # checkpoint's flag) unless the CLI flag forces the subset.
            want_bases = _want_bases(
                args, lambda spec: ("relset" if servers[spec][0].info.get("relevant_set")
                                    else "full"))
            want_terrains = _want_terrains(
                args, lambda spec: ("set" if servers[spec][0].info.get("terrain_multi_hot")
                                    else "class"))
            for f in sorted(args.outdir.glob("game_*.json")):
                try:
                    prev = json.loads(f.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001 -- unreadable = replayed
                    continue
                for _fld, _want in (("infer_bf16", shared_bf16),
                                    ("infer_packed_trunk", shared_packed)):
                    if bool(prev.get(_fld, False)) != _want:
                        raise SystemExit(
                            f"{f.name} was played with {_fld}="
                            f"{bool(prev.get(_fld, False))} but the servers "
                            f"run {_want}: numerics differ, refusing to mix. "
                            f"Use a fresh outdir.")
                _why = basis_refusal(f.name, prev, want_bases)
                if _why is not None:
                    raise SystemExit(_why)
                _why = terrain_refusal(f.name, prev, want_terrains)
                if _why is not None:
                    raise SystemExit(_why)
        except BaseException:
            _shutdown_servers(servers)
            raise

    def launch(slot):
        i, side_a, seed, _out, _gen = slot
        cmd = [sys.executable, "-u", str(GAME_SCRIPT),
               args.label_a, args.spec_a, args.label_b, args.spec_b,
               str(side_a), str(seed), str(args.outdir),
               "--mcts-sims", str(args.mcts_sims),
               "--max-turns", str(args.max_turns),
               "--device", args.device]
        if args.mcts_sims_a is not None:
            cmd += ["--mcts-sims-a", str(args.mcts_sims_a)]
        if args.mcts_sims_b is not None:
            cmd += ["--mcts-sims-b", str(args.mcts_sims_b)]
        if args.value_center_a:
            cmd += ["--value-center-a", str(args.value_center_a)]
        if args.value_center_b:
            cmd += ["--value-center-b", str(args.value_center_b)]
        if args.shared_combat_stream:
            cmd.append("--shared-combat-stream")
        if args.relevant_set_a:
            cmd.append("--relevant-set-a")
        if args.relevant_set_b:
            cmd.append("--relevant-set-b")
        if not args.gumbel_root_a:
            cmd.append("--no-gumbel-root-a")
        if not args.gumbel_root_b:
            cmd.append("--no-gumbel-root-b")
        if args.raw_temperature_a is not None:
            cmd += ["--raw-temperature-a", str(args.raw_temperature_a)]
        if args.raw_temperature_b is not None:
            cmd += ["--raw-temperature-b", str(args.raw_temperature_b)]
        for side in ("a", "b"):
            rule = getattr(args, f"raw_end_turn_{side}")
            offset = getattr(args, f"raw_end_turn_offset_{side}")
            if rule != "joint":
                cmd += [f"--raw-end-turn-{side}", rule]
            if offset:
                cmd += [f"--raw-end-turn-offset-{side}", str(offset)]
        if args.mcts_batch_size != 1:
            cmd += ["--mcts-batch-size", str(args.mcts_batch_size)]
        if servers:
            # The servers' effective precision, stated explicitly so
            # the game records what ran (elo_eval_game checks it
            # against the server's hello).
            for side, spec in (("a", args.spec_a), ("b", args.spec_b)):
                if spec in servers:
                    # Spread the games over the servers WITHOUT lining
                    # the assignment up with the side swap: slots
                    # alternate side_a, so `i % 2` would send every
                    # A-plays-side-1 game to one server and every
                    # A-plays-side-2 game to another, confounding any
                    # difference between the servers with the side.
                    handles = servers[spec]
                    cmd += [f"--inference-address-{side}",
                            handles[(i // 2) % len(handles)].address]
            cmd += ["--infer-bf16" if shared_bf16 else "--no-infer-bf16",
                    "--no-infer-compile",
                    "--infer-packed-trunk" if shared_packed
                    else "--no-infer-packed-trunk"]
        else:
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
        if worker_pool is not None:
            handle, errf = worker_pool.submit(
                [str(GAME_SCRIPT)] + cmd[3:],
                game_tag=f"{i}_{seed}")
            return handle, time.perf_counter(), slot, errf
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
             "turn_config": None,
             # The pre-scan and per-game guards read these three with
             # absent = B 1 / fp32 / eager; an artifact without them
             # aborted every resume of a cuda outdir (2026-09-04 review).
             "mcts_batch": args.mcts_batch_size,
             "infer_bf16": (shared_bf16 if servers
                            else _effective_precision(args, "infer_bf16")),
             "infer_compile": (False if servers
                               else _effective_precision(args, "infer_compile")),
             "shared_inference": bool(servers),
             "infer_packed_trunk": shared_packed,
             "combat_stream": ("shared" if args.shared_combat_stream
                               else "per_game"),
             # The sim's observation epoch, as elo_eval_game records it:
             # an artifact without it reads as epoch 1 to elo_collect
             # and blocks the whole dir as a MIXED estimand.
             "observation_epoch": int(OBSERVATION_EPOCH),
             # Search knobs that change the player; None when no side
             # searches, matching what elo_eval_game writes -- an
             # artifact that omitted them would read as a MIXED
             # estimand to elo_collect and block the whole dir.
             "value_center_a": (args.value_center_a if sims_a > 0
                                else None),
             "value_center_b": (args.value_center_b if sims_b > 0
                                else None),
             "moves_left_utility": (
                 float(os.environ.get("ELO_MOVES_LEFT_UTILITY", "0") or 0)
                 if (sims_a > 0 or sims_b > 0) else None),
             # The effective hex basis and terrain view per side (see
             # BASES, TERRAIN_VIEWS).
             "basis_a": want_bases[0], "basis_b": want_bases[1],
             "terrain_a": want_terrains[0], "terrain_b": want_terrains[1],
             # The checkpoint per side and the forced faction (see
             # checkpoint_refusal, faction_refusal).
             "checkpoint_sha256_a": want_ckpts[0], "checkpoint_sha256_b": want_ckpts[1],
             "forced_faction": want_faction}
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
                dead = dead_servers(servers)
                if dead:
                    # Every game through a dead server fails: admit no
                    # more, drain the games in flight.
                    log.error("an inference server has exited; no new games, "
                              "draining %d in flight: %s", len(running), " | ".join(dead))
                    server_died = stop = True
                    break
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
            time.sleep(POLL_S)
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
                        timed_out += 1
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
                    rec = record_failure(args.outdir, args.label_a, args.label_b, side_a,
                                         seed, slot=i, gen=gen,
                                         returncode=proc.returncode, reason=err)
                    log.warning("game %d (side %d, seed %d) failed rc=%s, recorded in "
                                "%s: %s", i, side_a, seed, proc.returncode, rec.name,
                                err.strip()[-200:])
                _pk = _peak_rss.pop(proc.pid, None)
                log.info("game %d done in %.1f min%s (results=%d/%d "
                         "no_result=%d failed=%d timed_out=%d, %d pending, "
                         "%d in flight)", i, elapsed / 60.0,
                         (f", peak_rss {_pk:.0f}MB"
                          if _pk else ""), n_results,
                         args.games, n_nores, failed, timed_out, len(pending),
                         len(running))

    finally:
        for _proc, _t0, (_i, _sa, _sd, _out, _g), _errf in running:
            if _proc.poll() is None:
                _proc.kill()
                _proc.wait()
                log.warning("killed in-flight game %d at exit", _i)
                _close_err(_errf)
        if worker_pool is not None:
            worker_pool.shutdown()
        if servers:
            _log_server_stats(_shutdown_servers(servers))
    total = len(list(args.outdir.glob("game_*.json")))
    # Report as a fraction, never a percentage or an extrapolation.
    log.info("chunk end: %d/%d RESULTS (%d no-result absences, "
             "replacements %d/%d; %d files) in %s (this chunk: %d "
             "played, %d failed, %d timed out)",
             n_results, args.games, n_nores, extra, max_extra, total,
             args.outdir, played, failed, timed_out)
    status = match_status(args.outdir, args.label_a, args.label_b, args.games,
                          args.seed_base, max_extra, failed=bool(failed or server_died))
    n_records = len(list(args.outdir.glob("failed_*.json")))
    if n_records:
        log.warning("%d game slot(s) of this outdir have failed at least once "
                    "(failed_*.json)", n_records)
    log.info("exit %d: %s", status, EXIT_MEANING[status])
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
