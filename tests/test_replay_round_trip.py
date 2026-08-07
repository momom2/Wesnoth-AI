"""Replay round-trip test: extract -> re-emit -> re-extract.

The pipeline `replays_raw/*.bz2` -> `replay_extract` -> compact
`*.json.gz` is the production training-data path. The reverse path
`sim.command_history` -> `sim_to_replay.export_replay` -> `*.bz2` is
the demo / audit path. We want to be sure both directions are
loss-less for the bits that matter (player actions; not cosmetic
metadata).

Test strategy:
  1. Extract the strict-sync fixture via `replay_extract` -> compact
     commands list A.
  2. Build a `WesnothSim` whose `command_history` matches A
     (synthesized from the compact stream so we don't have to run a
     policy).
  3. Re-emit via `sim_to_replay.export_replay`.
  4. Re-extract the emitted bz2 via `replay_extract` -> commands list
     B.
  5. Assert A == B (in the bits that survive the round-trip: kind +
     coords + targets; seeds and bookkeeping vary).

What we tolerate: seeds change on re-emit (the exporter generates
fresh seeds; the BACKLOG documents this as a known compromise).
What we don't tolerate: a missing command, a coordinate drift, a
wrong target/side, an extra spurious init_side or end_turn.

Dependencies: tools.replay_extract, tools.sim_to_replay,
              tools.wesnoth_sim.
Dependents: regression CI.
"""
from __future__ import annotations

import gzip
import json
import tempfile
from pathlib import Path


from tools.replay_extract import extract_replay
from tools.sim_to_replay import export_replay
from tools.wesnoth_sim import RecordedCommand, WesnothSim

FIXTURE = Path(__file__).parent / "fixtures" / "strict_sync_hamlets_t9.bz2"


def _build_sim_from_extracted(extracted: dict) -> WesnothSim:
    """Synthesize a WesnothSim whose `command_history` matches the
    compact extracted commands. We don't run policy decisions; we
    just shove RecordedCommands into the history field. The sim's
    GameState is built from the extracted starting_units (via
    `from_replay` on a temp file) so any export consumers that read
    `sim.gs` see the right initial setup -- but the export itself
    only consumes `sim.command_history`.
    """
    # Persist to a temp json.gz so we can use from_replay -- it's
    # the canonical builder of the initial GameState.
    with tempfile.NamedTemporaryFile(
            suffix=".json.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
            json.dump(extracted, f)
        sim = WesnothSim.from_replay(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    # `WesnothSim.__init__` auto-records an `init_side(1)` for the
    # start-of-game side init. The extracted commands also start
    # with their own `init_side(1)`. Clear the auto-record so the
    # synthesized history matches the extracted stream 1:1.
    sim.command_history = []

    # Now translate each compact command into a RecordedCommand. The
    # mapping is direct: `cmd[0]` is the kind; the side comes from
    # the command's `from_side` slot (the 4th element of move /
    # attack / etc. or the 2nd for init_side). For recruits we also
    # need `extras["leader_pos"]` -- the leader's keep position at
    # recruit time -- which we recover by tracking the leader as
    # commands are applied.
    side1_leader = next(
        (u for u in sim.gs.map.units if u.side == 1 and u.is_leader),
        None,
    )
    side2_leader = next(
        (u for u in sim.gs.map.units if u.side == 2 and u.is_leader),
        None,
    )
    leader_pos = {
        1: (side1_leader.position.x, side1_leader.position.y) if side1_leader else (0, 0),
        2: (side2_leader.position.x, side2_leader.position.y) if side2_leader else (0, 0),
    }

    for cmd in extracted["commands"]:
        kind = cmd[0]
        if kind == "init_side":
            side = int(cmd[1])
            sim.command_history.append(RecordedCommand(
                kind="init_side", side=side, cmd=cmd))
        elif kind == "end_turn":
            sim.command_history.append(RecordedCommand(
                kind="end_turn", side=0, cmd=cmd))
        elif kind == "move":
            side = int(cmd[3]) if len(cmd) > 3 else 0
            sim.command_history.append(RecordedCommand(
                kind="move", side=side, cmd=cmd))
        elif kind == "attack":
            # ["attack", ax, ay, dx, dy, weap, dweap, seed, choose?]
            ax, ay = cmd[1], cmd[2]
            # Side: the attacker's side. Use the sim's GS if we can
            # find a unit at (ax, ay), else default to 1.
            side = 0
            for u in sim.gs.map.units:
                if u.position.x == ax and u.position.y == ay:
                    side = u.side
                    break
            sim.command_history.append(RecordedCommand(
                kind="attack", side=side, cmd=cmd))
        elif kind in ("recruit", "recall"):
            # ["recruit", unit_type, x, y, seed]
            side = int(cmd[4]) if len(cmd) > 4 and isinstance(cmd[4], int) else 1
            # Actual side: the side whose leader we're using.
            # Heuristic: cmd[5] if present, else infer from current
            # side -- but the compact format doesn't carry the side
            # for recruits, so we use the closest-keep heuristic.
            # The exporter needs leader_pos.
            sim.command_history.append(RecordedCommand(
                kind=kind, side=side, cmd=cmd,
                extras={"leader_pos": leader_pos.get(side, (0, 0))},
            ))
        else:
            # Unknown / skip.
            pass
    return sim


def _compare_commands(a: list, b: list, *, msg: str) -> None:
    """Compare two compact command streams modulo cosmetic
    differences (seeds, exact attack choose payloads). Returns
    silently on match, raises AssertionError with detail on diff."""
    if len(a) != len(b):
        raise AssertionError(
            f"{msg}: length differs ({len(a)} vs {len(b)})"
        )
    for i, (ca, cb) in enumerate(zip(a, b)):
        if not ca or not cb:
            assert ca == cb, f"{msg}: cmd[{i}] empty mismatch"
            continue
        if ca[0] != cb[0]:
            raise AssertionError(
                f"{msg}: cmd[{i}] kind differs "
                f"({ca[0]!r} vs {cb[0]!r})"
            )
        kind = ca[0]
        # Compare the kind-relevant fields, ignoring seeds.
        if kind == "init_side":
            assert ca[1] == cb[1], f"{msg}: init_side mismatch at {i}"
        elif kind == "end_turn":
            pass  # no payload
        elif kind == "move":
            # ["move", xs, ys, side]
            assert ca[1] == cb[1], (
                f"{msg}: cmd[{i}] move xs differ "
                f"({ca[1]} vs {cb[1]})"
            )
            assert ca[2] == cb[2], (
                f"{msg}: cmd[{i}] move ys differ "
                f"({ca[2]} vs {cb[2]})"
            )
            assert ca[3] == cb[3], (
                f"{msg}: cmd[{i}] move side differs"
            )
        elif kind == "attack":
            # ["attack", ax, ay, dx, dy, w, dw, seed, ...]
            assert ca[1:7] == cb[1:7], (
                f"{msg}: cmd[{i}] attack header differs "
                f"({ca[1:7]} vs {cb[1:7]})"
            )
            # Seed ([7]) and choose ([8:]) intentionally not compared.
        elif kind in ("recruit", "recall"):
            # ["recruit", type, x, y, seed]
            assert ca[1] == cb[1], (
                f"{msg}: cmd[{i}] recruit type differs "
                f"({ca[1]!r} vs {cb[1]!r})"
            )
            assert ca[2] == cb[2] and ca[3] == cb[3], (
                f"{msg}: cmd[{i}] recruit coords differ"
            )
        else:
            assert ca == cb, (
                f"{msg}: cmd[{i}] unknown-kind mismatch"
            )


def test_extract_idempotent():
    """Sanity: extracting the same file twice produces identical
    compact streams. If this fails, the extractor itself is non-
    deterministic and the round-trip test below is meaningless."""
    a = extract_replay(FIXTURE)
    b = extract_replay(FIXTURE)
    assert a is not None and b is not None
    _compare_commands(a["commands"], b["commands"],
                      msg="extract idempotence")


def test_round_trip_preserves_commands():
    """Extract -> emit -> re-extract preserves the compact command
    stream modulo seeds. This is the real round-trip property."""
    extracted_a = extract_replay(FIXTURE)
    assert extracted_a is not None

    sim = _build_sim_from_extracted(extracted_a)
    # Sanity: every player-action in the source became a
    # RecordedCommand. (init_side / end_turn included.)
    assert len(sim.command_history) == len(extracted_a["commands"])

    out_path = Path(tempfile.gettempdir()) / "round_trip_out.bz2"
    try:
        export_replay(sim, source_bz2=FIXTURE, out_path=out_path)
        extracted_b = extract_replay(out_path)
        assert extracted_b is not None, (
            "re-extraction returned None -- emitted bz2 is malformed"
        )
        _compare_commands(
            extracted_a["commands"], extracted_b["commands"],
            msg="round-trip",
        )
    finally:
        out_path.unlink(missing_ok=True)


def _fixture_with_first_recruit_mutated(chatter: str) -> bytes:
    """Return the fixture text with (a) an empty [checkup] added inside
    the first [recruit]'s [command] (so the undone-recruit heuristic
    engages) and (b) `chatter` inserted between that command and its
    [random_seed] reply."""
    import bz2 as _bz2
    text = _bz2.decompress(FIXTURE.read_bytes()).decode(
        "utf-8", errors="replace")
    i = text.find("[recruit]")
    end = text.find("[/command]", i)
    mutated = (text[:end] + "[checkup]\n[/checkup]\n" + text[end:])
    seed_at = mutated.find('new_seed="3ca69f8d"')
    cmd_start = mutated.rfind("[command]", 0, seed_at)
    mutated = mutated[:cmd_start] + chatter + mutated[cmd_start:]
    return _bz2.compress(mutated.encode("utf-8"))


_SPEAK_CHATTER = "".join(
    '[command]\nundo="no"\n[speak]\nid="server"\n'
    f'message="chatter {i}"\n[/speak]\n[/command]\n'
    for i in range(4)
)


def test_recruit_seed_found_past_chatter_window():
    """A finalized recruit whose [random_seed] reply is pushed several
    commands away by server chatter ([speak] control-swap traffic)
    must still be emitted with its seed. Real case: Micro Isar 2836
    (the user's own game) — 'becomes an observer'/'takes control'
    speaks between recruit and seed; the old fixed 3-command window
    dropped the recruit as undone and the whole file diverged
    src_missing (engine playback is clean, user-verified 2026-08-07)."""
    blob = _fixture_with_first_recruit_mutated(_SPEAK_CHATTER)
    p = Path(tempfile.gettempdir()) / "chatter_fixture.bz2"
    p.write_bytes(blob)
    try:
        base = extract_replay(FIXTURE)
        data = extract_replay(p)
        assert data is not None
        # Chatter must be fully transparent: identical compact stream.
        assert data["commands"] == base["commands"], (
            "chatter between recruit and seed changed the extraction"
        )
        first = next(c for c in data["commands"] if c[0] == "recruit")
        assert first[1] == "Gryphon Rider" and first[4] == "3ca69f8d"
    finally:
        p.unlink(missing_ok=True)


def test_undone_recruit_still_dropped():
    """Control: an empty-checkup recruit followed by a player ACTION
    before any seed is genuinely undone and stays dropped — the
    chatter-skip must not weaken the undone heuristic."""
    action_chatter = (
        "[command]\nfrom_side=1\n[end_turn]\n[/end_turn]\n[/command]\n"
    )
    blob = _fixture_with_first_recruit_mutated(action_chatter)
    p = Path(tempfile.gettempdir()) / "undone_fixture.bz2"
    p.write_bytes(blob)
    try:
        base = extract_replay(FIXTURE)
        n_base = sum(1 for c in base["commands"] if c[0] == "recruit")
        data = extract_replay(p)
        assert data is not None
        recruits = [c for c in data["commands"] if c[0] == "recruit"]
        # The mutated first recruit must be gone; its orphaned seed
        # must not have been back-filled anywhere.
        assert len(recruits) == n_base - 1, recruits[:3]
        assert all(c[4] != "3ca69f8d" for c in recruits), (
            "undone recruit's seed leaked into another slot"
        )
    finally:
        p.unlink(missing_ok=True)


def test_full_checkup_recruit_survives_block_boundary():
    """A recruit whose raw command carries a FULL pattern-A [checkup]
    ([result] with checksum/next_unit_id) is engine-committed and must
    survive the block-boundary trailer-drop even when its seed lands
    in the next block. Real case: Silverhead 8566 — a seedless (Ghost)
    recruit ending its block on the keep hex was dropped by the
    "next block recruits at the same hex" check, because later turns
    recruit at the same keep; engine playback is clean (2026-08-07)."""
    import bz2 as _bz2
    text = _bz2.decompress(FIXTURE.read_bytes()).decode(
        "utf-8", errors="replace")
    # Cut the first [replay] block right after the Gryphon recruit's
    # command (its checkup is pattern-A full), pushing the seed and
    # everything else into a second block.
    i = text.find("[recruit]")
    end = text.find("[/command]", i) + len("[/command]")
    mutated = text[:end] + "\n[/replay]\n[replay]\n" + text[end:]
    p = Path(tempfile.gettempdir()) / "boundary_fixture.bz2"
    p.write_bytes(_bz2.compress(mutated.encode("utf-8")))
    try:
        base = extract_replay(FIXTURE)
        data = extract_replay(p)
        assert data is not None
        first = next(c for c in data["commands"] if c[0] == "recruit")
        assert first[1] == "Gryphon Rider", (
            "committed full-checkup recruit was trailer-dropped"
        )
        n_base = sum(1 for c in base["commands"] if c[0] == "recruit")
        n_mut = sum(1 for c in data["commands"] if c[0] == "recruit")
        assert n_mut == n_base, (n_mut, n_base)
    finally:
        p.unlink(missing_ok=True)
