"""Provenance of an eval game: the estimand fields a result records, and
the guards that refuse to mix them.

Every result file of a match records per side the hex basis (`BASES`),
the terrain view (`TERRAIN_VIEWS`) and the SHA-256 of the checkpoint
played, and the faction forced onto one side; an outdir never mixes two
values of any of them. The `*_refusal` functions are those guards: the
per-game runner (tools/elo_eval_game.py) applies them to a result it is
about to reuse, the match driver (tools/run_elo_batch.py) to every
result in the outdir before it launches a game, and the collector
(tools/elo_collect.py) reads the same fields. `_pt_config` is the
plan-tournament config a match plays and records.

Torch-free: the match driver imports it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Tuple


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
