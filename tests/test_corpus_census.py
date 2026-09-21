"""tools/analysis/corpus_census.py on synthetic replay headers: the
census must read the era, the layout and the host's rule settings from
the replay rather than assume them, and it must split the corpus into
the packs the evaluation pool distinguishes."""
from __future__ import annotations

import bz2
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.analysis import corpus_census as cc  # noqa: E402

HEADER = """[replay_start]
	era_id="{era}"
	experience_modifier="{xp}"
	random_start_time="{tod}"
	mp_village_gold="{vgold}"
	mp_use_map_settings="yes"
	map_data="{map_data}"
	[side]
		side=1
		faction="{f1}"
	[/side]
	[side]
		side=2
		faction="{f2}"
	[/side]
	[modification]
		id="plan_unit_advance"
	[/modification]
[/replay_start]
"""
GRID_A = "Gg, Gg, Gg\nGg, Ke, Gg\nGg, Gg, Gg"
GRID_B = "Gg, Ww, Gg\nGg, Ke, Gg\nGg, Gg, Gg"


_RUN = itertools.count()


def _corpus(tmp_path: Path, games) -> Path:
    """A dataset directory with a manifest and one bz2 replay per game.
    One per call: a test that builds two corpora must not mix their
    manifests."""
    ds = tmp_path / f"corpus{next(_RUN)}"
    (ds / "raw").mkdir(parents=True)
    rows = []
    for i, (map_name, opts) in enumerate(games):
        name = f"2024-03-2{i % 9}_{map_name}_Turn_11_({100 + i}).json.gz"
        src = ds / "raw" / f"{i}.bz2"
        body = HEADER.format(era=opts.get("era", "era_default"),
                             xp=opts.get("xp", "70"),
                             tod=opts.get("tod", "no"),
                             vgold=opts.get("vgold", "2"),
                             map_data=opts.get("grid", GRID_A),
                             f1=opts.get("f1", "Rebels"),
                             f2=opts.get("f2", "Undead"))
        src.write_bytes(bz2.compress(body.encode()))
        rows.append({"file": name, "source": str(src.relative_to(cc.ROOT))
                     if str(src).startswith(str(cc.ROOT)) else str(src)})
    (ds / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return ds


def _run(tmp_path, games, out=None):
    ds = _corpus(tmp_path, games)
    # The scanner resolves sources under the repo root; these are absolute.
    orig = cc.ROOT
    cc.ROOT = Path("/")
    try:
        argv = ["--dataset", str(ds), "--workers", "1"]
        if out:
            argv += ["--out", str(out), "--rows", str(out.with_suffix(".rows.json.gz"))]
        assert cc.main(argv) == 0
    finally:
        cc.ROOT = orig
    if not out:
        return None
    summary = json.loads(out.read_text(encoding="utf-8"))
    import gzip
    with gzip.open(out.with_suffix(".rows.json.gz"), "rt", encoding="utf-8") as fh:
        summary["games"] = json.load(fh)["games"]
    return summary


def test_the_census_reads_era_factions_and_settings_from_the_replay(tmp_path):
    out = tmp_path / "census.json"
    data = _run(tmp_path, [
        ("2p__Hamlets", {}),
        ("2p__Hamlets", {"era": "era_dunefolk"}),
        ("2p_mini_edited", {"tod": "yes", "vgold": "3", "grid": GRID_B}),
        ("2p__Cynsaun_Battlefield", {"xp": "50"}),
    ], out=out)
    s = data["summary"]
    assert s["games"] == 4 and s["errors"] == 0
    assert s["settings"]["era_id"] == {"era_default": 3, "era_dunefolk": 1}
    assert s["factions"] == {"Rebels": 4, "Undead": 4}
    # The packs the evaluation pool distinguishes.
    assert s["packs"] == {"ladder": 2, "mini": 1, "mainline": 1}
    # Non-default rules are attributed to the pack that carries them.
    assert s["non_default_by_pack"]["mini"] == {"random_start_time": 1,
                                                "mp_village_gold": 1, "any": 1}
    assert s["non_default_by_pack"]["mainline"] == {"experience_modifier": 1, "any": 1}
    assert "ladder" not in s["non_default_by_pack"]
    assert data["games"][0]["mods"] == ["plan_unit_advance"]


def test_one_name_with_two_layouts_is_reported(tmp_path):
    """A scenario name that covers several boards -- a map picker, or a
    ladder variant of a mainline map -- must not pass silently, because
    every per-map statistic downstream assumes one board per name."""
    out = tmp_path / "census.json"
    data = _run(tmp_path, [("2p__Hamlets", {}),
                           ("2p__Hamlets", {"grid": GRID_B})], out=out)
    assert data["summary"]["multi_layout_maps"] == {"2p__Hamlets": 2}
    same = _run(tmp_path, [("2p__Hamlets", {}), ("2p__Hamlets", {})],
                out=tmp_path / "b.json")
    assert same["summary"]["multi_layout_maps"] == {}


def test_layout_hash_ignores_whitespace_but_not_terrain():
    a, rows, cols = cc.layout_hash(GRID_A)
    spaced, _, _ = cc.layout_hash(GRID_A.replace(", ", ",  "))
    other, _, _ = cc.layout_hash(GRID_B)
    assert (rows, cols) == (3, 3)
    assert a == spaced and a != other


def test_pack_assignment_follows_the_whitelist():
    cc._CORPUS_NAMES = ("2p__Hamlets", "2p__Caves_of_the_Basilisk",
                        "2p__Cynsaun_Battlefield", "2p_mini_edited", "2p_-_Troll_Toll")
    cc._LADDER_NAMES = None
    assert cc.pack_of("2p__Hamlets") == "ladder"
    # The whitelist id is `multiplayer_Basilisk`; the corpus name is longer.
    assert cc.pack_of("2p__Caves_of_the_Basilisk") == "ladder"
    # A mainline map the whitelist leaves out.
    assert cc.pack_of("2p__Cynsaun_Battlefield") == "mainline"
    assert cc.pack_of("2p_mini_edited") == "mini"
    assert cc.pack_of("2p_-_Troll_Toll") == "other"
