"""The box inputs' cache (tools/input_cache.py) with the model host stood in
by a local directory: a stored entry restores whole, its facts with it,
and only its input's files; an absent entry is a miss and a corrupted,
foreign or wrong-content entry is refused, the directory left as it was;
a directory that fails its check is not sent, a present entry is not sent
again unless replaced, and a store names the entries it supersedes without
deleting them; a legacy tarball is adopted only when its records hold the
key, sources included; an archive member outside its directory is refused;
the raw tarball must be the one its key names; each key changes with every
input it hashes."""
from __future__ import annotations

import gzip
import io
import json
import tarfile
from pathlib import Path

import pytest

from tools import input_cache as ic
from tools.build_imitation_dataset import CORPUS_VERSION
from tools.input_cache_kinds import (corpus_components, file_sha256, own_components, selection_of,
                                     sequences_components, sources_sha256)

ROOT = Path(__file__).resolve().parent.parent
SELECTION = selection_of({"outcome_classes": ["leader_death", "surrender"], "holdout_fraction": 0.02})
TYPES, FACTIONS = {"Cavalryman": 0, "Spearman": 1}, {"Loyalists": 0}
PLAYER = "p" * 64


def _corpus(d: Path, version: int = CORPUS_VERSION) -> Path:
    d.mkdir(parents=True)
    rows = []
    for i in range(3):
        with gzip.open(d / f"g{i}.json.gz", "wt", encoding="utf-8") as f:
            json.dump({"commands": [i]}, f)
        rows.append({"file": f"g{i}.json.gz", "winner_side": 1, "corpus_version": version})
    (d / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (d / "errors.jsonl").write_text("", encoding="utf-8")
    (d / "CORPUS_DONE").write_text("stage=x\n", encoding="utf-8")       # a box marker: not an input file
    return d


def _facts(tmp_path: Path, **over) -> Path:
    path = tmp_path / "corpus_summary.json"
    path.write_text(json.dumps({"games": 3, "problems": [], **over}), encoding="utf-8")
    return path


def _corpus_key(raw_sha: str = "a" * 64) -> dict:
    return ic.key_file("corpus", corpus_components(raw_sha, CORPUS_VERSION, 31, SELECTION), {"raw_path": "raw.tar"})


def _host(tmp_path: Path) -> ic.DirHost:
    return ic.DirHost(tmp_path / "host")


def _records(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _entry_files(host: ic.DirHost, key: dict):
    return sorted(p for p in (host.root / ic.entry_dir(key)).iterdir())


def test_a_stored_entry_restores_whole_with_its_facts_and_only_its_input_files(tmp_path):
    host, key, record = _host(tmp_path), _corpus_key(), tmp_path / "cache.txt"
    src = _corpus(tmp_path / "built")
    assert ic.store(host, key, src, facts=_facts(tmp_path), record=record) == 0
    assert "CACHE_STORED corpus" in _records(record)
    into, facts = tmp_path / "box2" / "corpus", tmp_path / "box2_summary.json"
    into.parent.mkdir()
    assert ic.restore(host, key, into, facts=facts, record=record) == 0
    names = sorted(p.name for p in into.iterdir())
    assert names == ["errors.jsonl", "g0.json.gz", "g1.json.gz", "g2.json.gz", "manifest.jsonl"]
    assert all((into / n).read_bytes() == (src / n).read_bytes() for n in names)
    assert json.loads(facts.read_text(encoding="utf-8")) == {"games": 3, "problems": []}
    assert "CACHE_RESTORED corpus" in _records(record)
    assert not list(tmp_path.joinpath("box2").glob(".*"))          # the restore's work directory is gone


def test_a_missing_entry_is_a_miss_and_the_directory_stays(tmp_path):
    into = _corpus(tmp_path / "corpus")
    assert ic.restore(_host(tmp_path), _corpus_key(), into) == ic.EXIT_ABSENT
    assert (into / "CORPUS_DONE").exists() and (into / "g0.json.gz").exists()


def test_a_corrupted_archive_is_refused_and_the_directory_stays(tmp_path):
    host, key, record = _host(tmp_path), _corpus_key(), tmp_path / "cache.txt"
    ic.store(host, key, _corpus(tmp_path / "built"), facts=_facts(tmp_path))
    [archive] = [p for p in _entry_files(host, key) if p.suffix == ".tar"]
    data = bytearray(archive.read_bytes())
    data[700] ^= 0xFF
    archive.write_bytes(bytes(data))
    into = _corpus(tmp_path / "old")
    assert ic.restore(host, key, into, record=record) == ic.EXIT_REFUSED
    assert "CACHE_REFUSED corpus" in _records(record) and "SHA-256" in _records(record)
    assert (into / "CORPUS_DONE").exists()                         # what was there stays


def test_an_entry_of_another_key_is_refused(tmp_path):
    host, theirs, mine = _host(tmp_path), _corpus_key("a" * 64), _corpus_key("b" * 64)
    ic.store(host, theirs, _corpus(tmp_path / "built"), facts=_facts(tmp_path))
    (host.root / ic.entry_dir(mine)).parent.mkdir(parents=True, exist_ok=True)
    (host.root / ic.entry_dir(theirs)).rename(host.root / ic.entry_dir(mine))
    record = tmp_path / "cache.txt"
    assert ic.restore(host, mine, tmp_path / "corpus", record=record) == ic.EXIT_REFUSED
    assert "not this key's" in _records(record) and not (tmp_path / "corpus").exists()


def test_an_entry_whose_files_do_not_hold_its_key_is_refused(tmp_path):
    """An entry sent by a faulty sender: the archive and its manifest agree,
    the corpus inside is of another version than the key's."""
    host, key, record = _host(tmp_path), _corpus_key(), tmp_path / "cache.txt"
    old = _corpus(tmp_path / "old_version", version=CORPUS_VERSION - 1)
    files = ic.entry_files("corpus", old)
    tar = tmp_path / "entry.tar"
    sha, size = ic.write_archive(files, tar)
    host.put(tar, f"{ic.entry_dir(key)}/{sha[:16]}.tar")
    manifest = {"format": ic.ENTRY_FORMAT, "kind": "corpus", "key": key["key"], "components": key["components"],
                "inputs": key["inputs"], "archive": f"{sha[:16]}.tar", "archive_sha256": sha, "archive_bytes": size,
                "files": [[n, p.stat().st_size] for n, p in files], "facts": None, "created": "x", "stage": ""}
    (tmp_path / "m.json").write_text(json.dumps(manifest), encoding="utf-8")
    host.put(tmp_path / "m.json", f"{ic.entry_dir(key)}/manifest.json")
    assert ic.restore(host, key, tmp_path / "corpus", record=record) == ic.EXIT_REFUSED
    assert "corpus versions" in _records(record) and not (tmp_path / "corpus").exists()


def test_a_directory_that_fails_its_check_is_not_sent(tmp_path):
    host, record = _host(tmp_path), tmp_path / "cache.txt"
    src = _corpus(tmp_path / "built")
    assert ic.store(host, _corpus_key(), src, facts=_facts(tmp_path, problems=["1 of 3 failed"]),
                    record=record) == ic.EXIT_REFUSED
    old = _corpus(tmp_path / "old", version=CORPUS_VERSION - 1)
    assert ic.store(host, _corpus_key(), old, facts=_facts(tmp_path), record=record) == ic.EXIT_REFUSED
    assert _records(record).count("CACHE_NOT_SENT") == 2 and not host.files(ic.PREFIX)


def test_a_present_entry_is_not_sent_again_unless_replaced(tmp_path):
    host, key, record = _host(tmp_path), _corpus_key(), tmp_path / "cache.txt"
    src = _corpus(tmp_path / "built")
    ic.store(host, key, src, facts=_facts(tmp_path))
    assert ic.store(host, key, src, facts=_facts(tmp_path), record=record) == 0
    assert "CACHE_PRESENT" in _records(record) and "CACHE_STORED" not in _records(record)
    assert ic.store(host, key, src, facts=_facts(tmp_path), record=record, replace=True) == 0
    assert "CACHE_STORED" in _records(record)


def test_a_store_names_the_entries_it_supersedes_and_deletes_none(tmp_path):
    host, record = _host(tmp_path), tmp_path / "cache.txt"
    first, second = _corpus_key("a" * 64), _corpus_key("b" * 64)
    src = _corpus(tmp_path / "built")
    ic.store(host, first, src, facts=_facts(tmp_path))
    ic.store(host, second, src, facts=_facts(tmp_path), record=record)
    assert f"CACHE_SUPERSEDED {ic.entry_dir(first)}" in _records(record)
    assert len(_entry_files(host, first)) == 2 and len(_entry_files(host, second)) == 2
    listed = ic.listing(host)
    assert len(listed) == 2 and all("created" in line for line in listed)


def _results(d: Path, games) -> None:
    """Match results as tools/run_elo_batch.py writes them, each with its record."""
    d.mkdir(parents=True)
    for stem, sha in games:
        res = {"side_a": 1, "outcome_a": "win"}
        for x in ("a", "b"):
            res.update({f"checkpoint_sha256_{x}": sha, f"memory_{x}": 64})
        (d / f"{stem}.json").write_text(json.dumps(res), encoding="utf-8")
        (d / f"{stem}.game.jsonl.gz").write_bytes(gzip.compress(stem.encode()))


def _own_key(game_dirs, teacher: str = "t" * 64) -> dict:
    sources, _ = sources_sha256(game_dirs)
    comps = own_components(TYPES, FACTIONS, 31, teacher, PLAYER, 64, 16, -1.5, 0.05, sources)
    return ic.key_file("own", comps, {"games": [str(d) for d in game_dirs]})


def _legacy_tarball(tmp_path: Path, key: dict, games, name: str, **over) -> Path:
    """An own-sequence directory as a box sent it, records without columns,
    a box marker among them, packed under its directory's name."""
    d = tmp_path / name / "own_sequences_r2"
    d.mkdir(parents=True)
    comps = key["components"]
    man = {"source": "own", "fingerprint": comps["fingerprint"], "teacher_sha256": comps["teacher_sha256"],
           "player_sha256": PLAYER, "slots": 64, "top": 16, "reference_offset": -1.5, "errors": {},
           "games": {g: {"positions_side1": 2, "holdout": False} for g in games}, **over}
    (d / "sequence_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    for g in games:
        (d / f"{g}.seq.zpkl").write_bytes(b"records")
    (d / "OWN_DONE").write_text("stage=old\n", encoding="utf-8")
    tar = tmp_path / name / "own_sequences_r2.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        tf.add(d, arcname="own_sequences_r2")
    return tar


def test_a_legacy_tarball_is_adopted_only_when_its_records_hold_the_key(tmp_path):
    games_dir = tmp_path / "games_m1"
    _results(games_dir, [("game_1", PLAYER), ("game_2", PLAYER), ("game_3", "q" * 64)])
    key, host, record = _own_key([games_dir]), _host(tmp_path), tmp_path / "cache.txt"
    own = ["games_m1__game_1", "games_m1__game_2"]
    cases = {"good": ({}, own, 4), "teacher": ({"teacher_sha256": "u" * 64}, own, 2),
             "sources": ({}, own[:1], 2)}
    for name, (over, games, want) in cases.items():
        host.put(_legacy_tarball(tmp_path, key, games, name, **over), f"legacy/{name}.tar.gz")
        into = tmp_path / f"own_{name}"
        assert ic.restore(host, key, into, record=record, legacy=f"legacy/{name}.tar.gz",
                          legacy_top="own_sequences_r2") == want, name
        assert into.exists() == (want == ic.EXIT_ADOPTED)
    assert sorted(p.name for p in (tmp_path / "own_good").iterdir()) == \
        ["games_m1__game_1.seq.zpkl", "games_m1__game_2.seq.zpkl", "sequence_manifest.json"]
    text = _records(record)
    assert "CACHE_ADOPTED own" in text and "teacher_sha256" in text and "the sources' own game-sides" in text
    # Without columns the adopted records are no entry yet: a store refuses them.
    assert ic.store(host, key, tmp_path / "own_good", record=record) == ic.EXIT_REFUSED


def test_an_archive_member_outside_its_directory_is_refused(tmp_path):
    games_dir = tmp_path / "games_m1"
    _results(games_dir, [("game_1", PLAYER)])
    key, host = _own_key([games_dir]), _host(tmp_path)
    evil = tmp_path / "evil.tar"
    with tarfile.open(evil, "w") as tf:
        info = tarfile.TarInfo("own_sequences_r2/../../escaped.txt")
        info.size = 3
        tf.addfile(info, io.BytesIO(b"bad"))
    host.put(evil, "legacy/evil.tar")
    assert ic.restore(host, key, tmp_path / "box" / "own", legacy="legacy/evil.tar",
                      legacy_top="own_sequences_r2") == ic.EXIT_REFUSED
    assert not list(tmp_path.rglob("escaped.txt"))


def test_the_raw_tarball_must_be_the_one_its_key_names(tmp_path, monkeypatch):
    monkeypatch.setattr(ic, "running_rules_version", lambda: 31)
    host = _host(tmp_path)
    raw = tmp_path / "raw.tar"
    with tarfile.open(raw, "w") as tf:
        info = tarfile.TarInfo("replays_raw/d/x.bz2")
        info.size = 4
        tf.addfile(info, io.BytesIO(b"bz2!"))
    host.put(raw, "tier-b/raw.tar")
    key = ic.corpus_key(host, "tier-b/raw.tar", ROOT / "configs" / "imitation.json")
    assert key["components"]["raw_sha256"] == file_sha256(raw)
    assert ic.fetch_raw(host, key, tmp_path / "raw") == 0
    assert (tmp_path / "raw" / "replays_raw" / "d" / "x.bz2").read_bytes() == b"bz2!"
    (host.root / "tier-b" / "raw.tar").write_bytes(b"another tarball")
    assert ic.fetch_raw(host, key, tmp_path / "raw2") == ic.EXIT_REFUSED


def _distinct(keys) -> bool:
    keys = list(keys)
    return len(set(keys)) == len(keys)


def test_each_key_changes_with_every_input_it_hashes(tmp_path, monkeypatch):
    import tools.preencode_sequences as ps
    corpus = [corpus_components("a" * 64, 5, 31, SELECTION), corpus_components("b" * 64, 5, 31, SELECTION),
              corpus_components("a" * 64, 6, 31, SELECTION), corpus_components("a" * 64, 5, 32, SELECTION),
              *[corpus_components("a" * 64, 5, 31, {**SELECTION, k: v})
                for k, v in (("outcome_classes", ["leader_death"]), ("quarantine_ai_player_sides", False),
                             ("holdout_fraction", 0.03))]]
    assert _distinct(ic.key_of("corpus", c) for c in corpus)

    def seq(corpus_key="c" * 64, types=TYPES, factions=FACTIONS, version=5, rules=31):
        return ic.key_of("sequences", sequences_components(corpus_key, types, factions, version, rules))
    base = seq()
    assert seq() == base
    variants = [seq(corpus_key="d" * 64), seq(types={**TYPES, "Spearman": 2}), seq(types={**TYPES, "Mage": 2}),
                seq(factions={"Loyalists": 0, "Rebels": 1}), seq(version=6), seq(rules=32)]
    monkeypatch.setattr(ps, "ENCODING", {**ps.ENCODING, "relevant_set_version": 3})
    variants.append(seq())
    monkeypatch.undo()
    monkeypatch.setattr(ps, "OBSERVATION_EPOCH", ps.OBSERVATION_EPOCH + 1)
    variants.append(seq())
    monkeypatch.undo()
    assert _distinct([base, *variants]) and seq() == base

    def own(**over):
        args = {"type_to_id": TYPES, "faction_to_id": FACTIONS, "rules_version": 31, "teacher_sha256": "t" * 64,
                "player_sha256": PLAYER, "slots": 64, "top": 16, "reference_offset": -1.5, "own_holdout": 0.05,
                "sources_sha256": "s" * 64, **over}
        return ic.key_of("own", own_components(**args))
    assert _distinct([own(), own(type_to_id={"Mage": 0}), own(rules_version=32), own(teacher_sha256="u" * 64),
                      own(player_sha256="q" * 64), own(slots=16), own(top=8), own(reference_offset=-2.5),
                      own(own_holdout=0.1), own(sources_sha256="r" * 64)])


def test_the_sources_hash_follows_the_games_and_nothing_else(tmp_path):
    d = tmp_path / "games_m1"
    _results(d, [("game_1", PLAYER), ("game_2", PLAYER)])
    first, n = sources_sha256([d])
    assert n == 2
    (d / "match.log").write_text("not a source", encoding="utf-8")
    assert sources_sha256([d])[0] == first
    (d / "game_2.game.jsonl.gz").write_bytes(gzip.compress(b"another record"))
    second = sources_sha256([d])[0]
    _results(tmp_path / "games_m2", [("game_1", PLAYER)])
    assert _distinct([first, second, sources_sha256([d, tmp_path / "games_m2"])[0]])


@pytest.mark.parametrize("missing", ["columns", "pickle"])
def test_sequences_are_sent_only_with_every_record_and_its_columns(tmp_path, missing):
    from wesnoth_ai.sequence_columns import COLUMNS_FORMAT
    comps = sequences_components("c" * 64, TYPES, FACTIONS, 5, 31)
    key, host = ic.key_file("sequences", comps), _host(tmp_path)
    d = tmp_path / "sequences"
    d.mkdir()
    games = {"g0.json.gz": {"positions_side1": 1}, "g1.json.gz": {"positions_side1": 1}}
    (d / "sequence_manifest.json").write_text(json.dumps(
        {"fingerprint": comps["fingerprint"], "corpus_version": 5, "n_games": 2, "n_manifest_games": 2,
         "errors": {}, "games": games}), encoding="utf-8")
    (d / "columns_manifest.json").write_text(json.dumps(
        {"format": COLUMNS_FORMAT, "fingerprint": comps["fingerprint"], "games": {g: 1 for g in games}}),
        encoding="utf-8")
    for g in games:
        (d / f"{g}.seq.zpkl").write_bytes(b"r")
        (d / f"{g}.seq.cols").write_bytes(b"c")
    assert ic.store(host, key, d) == 0
    into = tmp_path / "restored"
    assert ic.restore(host, key, into) == 0 and len(list(into.iterdir())) == 6
    (d / ("g1.json.gz.seq.cols" if missing == "columns" else "g1.json.gz.seq.zpkl")).unlink()
    record = tmp_path / "cache.txt"
    assert ic.store(host, key, d, record=record, replace=True) == ic.EXIT_REFUSED
    assert "CACHE_NOT_SENT" in _records(record)
