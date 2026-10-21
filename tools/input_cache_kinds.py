"""What makes each box input valid, for the input cache (tools/input_cache.py):
the inputs its key hashes, the files its entry holds, and the check that a
directory holds that input.

  corpus     the built corpus (tools/build_imitation_dataset.py). Its key:
             the raw replays' tarball by its SHA-256 (the tarball holds the
             dispositions ledger the build reads), the corpus version, the
             rules version (the outcome labeller reads the core) and the
             selection knobs of configs/imitation.json.
  sequences  the corpus's pre-encoded sequences and their columns
             (tools/preencode_sequences.py, tools/columnize_sequences.py).
             Its key: the corpus entry's key and the encoding's fingerprint
             (the vocabularies, the encoding's switches, the observation
             epoch, the corpus version, the rules version), and the columns'
             format.
  own        the memory player's own sequences and their columns
             (tools/own_trajectories.py, tools/own_targets.py). Its key: the
             own encoding's fingerprint, the teacher's and the player's
             SHA-256, the memory size, the targets' top actions and decode
             offset, the share of own games held out, the SHA-256 of every
             source game's result and record, and the columns' format.

Both sequence kinds hold the pickled records and the columns: the trainer
trains from the columns (`--columns`), and its probe reads the pickles of
the holdout and of the game-sides its last-seen baseline is fitted on
(tools/sequence_train.py `run_probe`), as do the agreement barrier
(tools/own_barriers.py) and a trainer without `--columns`.

KIND_VERSION bumps a kind when its builder changes what it writes in a way
none of its key's inputs names.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

KIND_VERSION = {"corpus": 1, "sequences": 1, "own": 2}
KINDS = tuple(KIND_VERSION)
# What an entry of each kind holds, by file name (fnmatch patterns); the
# box's step markers and anything else in the directory stay out.
ENTRY_FILES = {
    "corpus": ("*.json.gz", "*.jsonl"),
    "sequences": ("*.seq.zpkl", "*.seq.cols", "sequence_manifest.json", "columns_manifest.json"),
    "own": ("*.seq.zpkl", "*.seq.cols", "sequence_manifest.json", "columns_manifest.json"),
}
# The configs/imitation.json knobs the corpus build reads (the others shape the loss).
SELECTION_KNOBS = ("outcome_classes", "quarantine_ai_player_sides", "holdout_fraction")


def key_of(kind: str, components: Dict) -> str:
    """The SHA-256 of the kind, its version and the components, as canonical JSON."""
    blob = json.dumps({"kind": kind, "kind_version": KIND_VERSION[kind], **components}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---- the components of each kind's key ------------------------------------

def selection_of(config: Dict) -> Dict:
    """The selection knobs of an imitation config, with the build's defaults."""
    return {"outcome_classes": list(config["outcome_classes"]),
            "quarantine_ai_player_sides": bool(config.get("quarantine_ai_player_sides", True)),
            "holdout_fraction": float(config["holdout_fraction"])}


def corpus_components(raw_sha256: str, corpus_version: int, rules_version: int, selection: Dict) -> Dict:
    return {"raw_sha256": raw_sha256, "corpus_version": int(corpus_version), "rules_version": int(rules_version),
            "selection": dict(selection)}


def sequences_components(corpus_key: str, type_to_id: Dict[str, int], faction_to_id: Dict[str, int],
                         corpus_version: int, rules_version: int) -> Dict:
    from tools.preencode_sequences import encoding_fingerprint
    from wesnoth_ai.sequence_columns import COLUMNS_FORMAT
    return {"corpus_key": corpus_key,
            "fingerprint": encoding_fingerprint(type_to_id, faction_to_id, corpus_version, rules_version),
            "corpus_version": int(corpus_version), "rules_version": int(rules_version),
            "columns_format": COLUMNS_FORMAT}


def own_components(type_to_id: Dict[str, int], faction_to_id: Dict[str, int], rules_version: int,
                   teacher_sha256: str, player_sha256: str, slots: int, top: int, reference_offset: float,
                   own_holdout: float, sources_sha256: str) -> Dict:
    """The own records' fingerprint is the encoding's at corpus version 0
    (tools/own_trajectories.py)."""
    from tools.preencode_sequences import encoding_fingerprint
    from wesnoth_ai.sequence_columns import COLUMNS_FORMAT
    return {"fingerprint": encoding_fingerprint(type_to_id, faction_to_id, 0, rules_version),
            "rules_version": int(rules_version), "teacher_sha256": teacher_sha256, "player_sha256": player_sha256,
            "slots": int(slots), "top": int(top), "reference_offset": float(reference_offset),
            "own_holdout": float(own_holdout), "sources_sha256": sources_sha256, "columns_format": COLUMNS_FORMAT}


def source_files(game_dirs: Iterable[Path]) -> List[Tuple[str, Path]]:
    """(name, path) of what tools/own_trajectories.py reads in the match
    directories: each game's result and, when present, its record."""
    found = []
    for d in sorted((Path(d) for d in game_dirs), key=lambda d: d.name):
        for result in sorted(d.glob("game_*.json")):
            found.append((f"{d.name}/{result.name}", result))
            record = result.with_name(result.stem + ".game.jsonl.gz")
            if record.is_file():
                found.append((f"{d.name}/{record.name}", record))
    return found


def sources_sha256(game_dirs: Iterable[Path]) -> Tuple[str, int]:
    """The SHA-256 over every source file's name and SHA-256, and the count of results."""
    digest = hashlib.sha256()
    results = 0
    for name, path in source_files(game_dirs):
        digest.update(f"{name} {file_sha256(path)}\n".encode("utf-8"))
        results += name.endswith(".json")
    return digest.hexdigest(), results


# ---- what a directory must hold ------------------------------------------

def content_problem(kind: str, directory: Path, components: Dict, facts: Optional[Dict] = None,
                    game_dirs: Optional[List[Path]] = None, need_columns: bool = True) -> Optional[str]:
    """Why `directory` does not hold the input `components` name (None when
    it does). `facts`: the corpus build's summary, which must name no
    problem; `game_dirs`: the own records' sources, whose own game-sides
    must be the records' games; `need_columns`: the sequences' columns must
    stand for every record."""
    directory = Path(directory)
    if kind == "corpus":
        return _corpus_problem(directory, components, facts)
    if kind == "sequences":
        return _sequences_problem(directory, components, need_columns)
    if kind == "own":
        return _own_problem(directory, components, game_dirs, need_columns)
    raise ValueError(f"no input kind {kind!r}: one of {KINDS}")


def _corpus_problem(directory: Path, components: Dict, facts: Optional[Dict]) -> Optional[str]:
    path = directory / "manifest.jsonl"
    if not path.is_file():
        return "no manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return "no game in manifest.jsonl"
    versions = {int(r.get("corpus_version", 1)) for r in rows}
    if versions != {components["corpus_version"]}:
        return f"rows of corpus versions {sorted(versions)}, the key's is {components['corpus_version']}"
    missing = [r["file"] for r in rows if not (directory / r["file"]).is_file()]
    if missing:
        return f"{len(missing)} manifest games have no file (first {missing[0]})"
    if facts is not None:
        if facts.get("problems"):
            return f"the build's barrier names problems: {facts['problems']}"
        if int(facts.get("games", -1)) != len(rows):
            return f"the build's summary counts {facts.get('games')} games, the manifest {len(rows)}"
    return None


def _records_problem(directory: Path, man: Dict, need_columns: bool) -> Optional[str]:
    """Every game of the manifest has its pickle, and its columns when needed."""
    from tools.columnize_sequences import columns_problem
    from tools.preencode_sequences import record_path
    from wesnoth_ai.sequence_columns import columns_path
    missing = [g for g in man["games"] if not record_path(directory, g).is_file()]
    if missing:
        return f"{len(missing)} games have no record (first {missing[0]})"
    if need_columns:
        why = columns_problem(directory, man["games"])
        if why is not None:
            return why
        missing = [g for g in man["games"] if not columns_path(directory, g).is_file()]
        if missing:
            return f"{len(missing)} games have no columns file (first {missing[0]})"
    return None


def _sequences_problem(directory: Path, components: Dict, need_columns: bool) -> Optional[str]:
    from tools.preencode_sequences import load_manifest
    man = load_manifest(directory)
    if man is None:
        return "no sequence manifest"
    if man.get("fingerprint") != components["fingerprint"]:
        return f"records of fingerprint {man.get('fingerprint')}, the key's is {components['fingerprint']}"
    if int(man.get("corpus_version", -1)) != components["corpus_version"]:
        return f"records of corpus version {man.get('corpus_version')}, the key's is {components['corpus_version']}"
    if int(man["n_games"]) + len(man.get("errors", {})) != int(man["n_manifest_games"]):
        return f"{man['n_games']} games and {len(man.get('errors', {}))} errors of {man['n_manifest_games']}"
    return _records_problem(directory, man, need_columns)


def _own_problem(directory: Path, components: Dict, game_dirs: Optional[List[Path]],
                 need_columns: bool) -> Optional[str]:
    from tools.preencode_sequences import load_manifest
    man = load_manifest(directory)
    if man is None or man.get("source") != "own":
        return "no own-trajectory manifest"
    for field in ("fingerprint", "teacher_sha256", "player_sha256", "slots", "top", "reference_offset"):
        if man.get(field) != components[field]:
            return f"records of {field} {man.get(field)!r}, the key's is {components[field]!r}"
    if game_dirs is not None:
        from tools.own_trajectories import select_own_sides
        selected = set(select_own_sides(list(game_dirs), components["player_sha256"], components["slots"])[0])
        held = set(man["games"]) | set(man.get("errors", {}))
        if held != selected:
            return (f"the records hold {len(held)} games, the sources' own game-sides are of {len(selected)} "
                    f"({len(held - selected)} not in the sources, {len(selected - held)} absent)")
    return _records_problem(directory, man, need_columns)
