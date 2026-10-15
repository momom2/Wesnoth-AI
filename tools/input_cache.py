"""The box inputs' content-addressed cache on the model host
(docs/box_runbook.md, "The input cache"). A box run's inputs -- the corpus,
its sequences, the memory player's own sequences -- are each an entry keyed
by what makes them valid (tools/input_cache_kinds.py): a box that finds its
key's entry restores it, one that does not builds the input and sends it,
so the next box restores it.

On the host, under PREFIX (tier-b/input_cache):
    <kind>/<key16>/manifest.json   the entry: its key and the inputs the key
                                   hashes, its archive's SHA-256 and size,
                                   every file with its size, the facts
    <kind>/<key16>/<sha16>.tar     the files, uncompressed (each is a zlib or
                                   gzip stream already)
The manifest goes up after its archive: an entry without one is absent.

A restore verifies the archive's SHA-256 and size, its files against the
manifest's list, and the files' content against the key (the kind's check)
before the directory takes their place; a mismatch is refused, loudly, and
the directory is left as it was. A store checks the content first and sends
nothing that fails. Nothing here deletes from the host: a store prints the
entries of its kind it supersedes, and `list` names every entry, for the
lead to delete.

    python tools/input_cache.py key corpus --raw HOST_PATH --out KEY.json
    python tools/input_cache.py key sequences --corpus-key KEY.json (--vocab-from CKPT | --fresh-vocab) --out KEY.json
    python tools/input_cache.py key own --teacher CKPT --player-sha256 SHA --slots N --games DIR... --out KEY.json
    python tools/input_cache.py fetch-raw --key KEY.json --into DIR
    python tools/input_cache.py unpack --into DIR HOST_PATH...
    python tools/input_cache.py restore --key KEY.json --into DIR [--facts FILE] [--legacy HOST_PATH --legacy-top NAME]
    python tools/input_cache.py store --key KEY.json --from DIR [--facts FILE] [--replace]
    python tools/input_cache.py list

The host is the model repository (HF_TOKEN in the environment), or a
directory standing in for it (INPUT_CACHE_HOST=DIR). `--record FILE`
appends each outcome to FILE (the box's cache.txt). Exit status: 0 done;
1 the host failed; 2 refused (an entry or a directory that does not hold
its key); 3 no entry; 4 restored from the legacy tarball, which the box then
sends as an entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.input_cache_kinds import (ENTRY_FILES, KINDS, SELECTION_KNOBS, content_problem,  # noqa: E402,F401
                                     corpus_components, file_sha256, key_of, own_components,
                                     selection_of, sequences_components, sources_sha256)

REPO = "momom2/wesnoth-model-checkpoints"
PREFIX = os.environ.get("INPUT_CACHE_PREFIX", "tier-b/input_cache")
ENTRY_FORMAT = 1
EXIT_HOST, EXIT_REFUSED, EXIT_ABSENT, EXIT_ADOPTED = 1, 2, 3, 4
ABSENT_ERRORS = {"EntryNotFoundError", "RemoteEntryNotFoundError"}
ATTEMPTS, RETRY_PAUSE_S = 3, 20.0


class Absent(Exception):
    """The host has no file at that path."""


class Refused(Exception):
    """An entry, or a directory, that does not hold its key's input."""


# ---- the host ------------------------------------------------------------

class DirHost:
    """A directory standing in for the model host."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def info(self, path: str) -> Optional[Tuple[int, Optional[str]]]:
        p = self.root / path
        return (p.stat().st_size, file_sha256(p)) if p.is_file() else None

    def fetch(self, path: str, into: Path) -> Path:
        src = self.root / path
        if not src.is_file():
            raise Absent(path)
        dst = Path(into) / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        return dst

    def put(self, local: Path, path: str) -> None:
        dst = self.root / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copyfile(local, tmp)
        os.replace(tmp, dst)

    def files(self, prefix: str) -> List[Tuple[str, int]]:
        base = self.root / prefix
        if not base.is_dir():
            return []
        return sorted((p.relative_to(self.root).as_posix(), p.stat().st_size)
                      for p in base.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))


class HubHost:
    """The model repository on HF. A failure is named by its type only:
    messages can quote URLs and headers."""

    def __init__(self, repo: str = REPO):
        from huggingface_hub import HfApi
        self.repo = repo
        self.api = HfApi(token=os.environ.get("HF_TOKEN") or None)

    def info(self, path: str) -> Optional[Tuple[int, Optional[str]]]:
        found = _retry(lambda: self.api.get_paths_info(self.repo, [path]))
        if not found or getattr(found[0], "size", None) is None:
            return None
        lfs = getattr(found[0], "lfs", None)
        sha = None
        if lfs is not None:
            sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        return int(found[0].size), sha

    def fetch(self, path: str, into: Path) -> Path:
        from huggingface_hub import hf_hub_download
        return Path(_retry(lambda: hf_hub_download(self.repo, path, local_dir=str(into))))

    def put(self, local: Path, path: str) -> None:
        _retry(lambda: self.api.upload_file(path_or_fileobj=str(local), path_in_repo=path, repo_id=self.repo))

    def files(self, prefix: str) -> List[Tuple[str, int]]:
        try:
            tree = _retry(lambda: list(self.api.list_repo_tree(self.repo, path_in_repo=prefix, recursive=True)))
        except Absent:
            return []
        return sorted((f.path, int(f.size)) for f in tree if getattr(f, "size", None) is not None)


def _retry(call):
    """`call()`, with ATTEMPTS attempts on a host failure; Absent at once."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 -- the type name only
            if type(exc).__name__ in ABSENT_ERRORS:
                raise Absent() from None
            if attempt == ATTEMPTS:
                raise RuntimeError(f"the host failed {ATTEMPTS} times ({type(exc).__name__})") from None
            time.sleep(RETRY_PAUSE_S)


def make_host(spec: Optional[str] = None):
    spec = spec if spec is not None else os.environ.get("INPUT_CACHE_HOST", "")
    return HubHost() if spec in ("", "hub") else DirHost(Path(spec))


# ---- keys ------------------------------------------------------------------

def key_file(kind: str, components: Dict, inputs: Optional[Dict] = None) -> Dict:
    return {"kind": kind, "key": key_of(kind, components), "components": components, "inputs": inputs or {}}


def read_key(path: Path) -> Dict:
    key = json.loads(Path(path).read_text(encoding="utf-8"))
    if key.get("kind") not in KINDS or key.get("key") != key_of(key["kind"], key["components"]):
        raise SystemExit(f"{path} is not a key this code computes")
    return key


def entry_dir(key: Dict) -> str:
    return f"{PREFIX}/{key['kind']}/{key['key'][:16]}"


def running_rules_version() -> int:
    """The rules version of the wheel a build here runs (the builders read `wesnoth_core.__phase__`)."""
    import wesnoth_core
    from wesnoth_ai.rules_version import rules_version_of
    return rules_version_of(int(wesnoth_core.__phase__))


def corpus_key(host, raw_path: str, config_path: Path) -> Dict:
    from tools.build_imitation_dataset import CORPUS_VERSION
    found = host.info(raw_path)
    if found is None:
        raise SystemExit(f"no raw tarball at {raw_path} on the host")
    sha = found[1]
    if sha is None:                      # a file the host does not hash: fetched and hashed here
        with tempfile.TemporaryDirectory() as scratch:
            sha = file_sha256(host.fetch(raw_path, Path(scratch)))
    selection = selection_of(json.loads(Path(config_path).read_text(encoding="utf-8")))
    comps = corpus_components(sha, CORPUS_VERSION, running_rules_version(), selection)
    return key_file("corpus", comps, {"raw_path": raw_path})


def sequences_key(corpus: Dict, vocab_from: Optional[Path]) -> Dict:
    if vocab_from is None:
        from tools.preencode_sequences import fresh_vocab
        types, factions = fresh_vocab()
    else:
        from tools.preencode_corpus import vocab_from_checkpoint
        types, factions = vocab_from_checkpoint(vocab_from)
    comps = sequences_components(corpus["key"], types, factions, corpus["components"]["corpus_version"],
                                 running_rules_version())
    return key_file("sequences", comps)


def own_key(teacher: Path, player_sha256: str, slots: int, game_dirs: List[Path]) -> Dict:
    """The decode offset is the one tools/own_targets.py labels under by
    default, configs/reference_player.json's (read from the working directory)."""
    from tools.own_trajectories import OWN_HOLDOUT
    from tools.preencode_corpus import vocab_from_checkpoint
    from wesnoth_ai.own_loss import TOP_ACTIONS
    types, factions = vocab_from_checkpoint(teacher)
    offset = float(json.loads(Path("configs/reference_player.json").read_text(encoding="utf-8"))
                   ["decode"]["raw_end_turn_offset"])
    sources, results = sources_sha256(game_dirs)
    if results == 0:
        raise SystemExit(f"no game result in {[str(d) for d in game_dirs]}")
    comps = own_components(types, factions, running_rules_version(), file_sha256(teacher), player_sha256,
                           slots, TOP_ACTIONS, offset, OWN_HOLDOUT, sources)
    return key_file("own", comps, {"games": [str(d) for d in game_dirs], "results": results})


# ---- archives ----------------------------------------------------------------

class _HashingWriter:
    def __init__(self, fh):
        self.fh, self.digest, self.size = fh, hashlib.sha256(), 0

    def write(self, data: bytes) -> int:
        self.digest.update(data)
        self.size += len(data)
        return self.fh.write(data)


def entry_files(kind: str, directory: Path) -> List[Tuple[str, Path]]:
    """(name, path) of the files an entry of `kind` holds, sorted."""
    patterns = ENTRY_FILES[kind]
    return sorted((p.relative_to(directory).as_posix(), p) for p in Path(directory).rglob("*")
                  if p.is_file() and any(fnmatch(p.name, pat) for pat in patterns))


def write_archive(files: List[Tuple[str, Path]], dst: Path) -> Tuple[str, int]:
    """The files as an uncompressed tar with fixed metadata; its SHA-256 and size."""
    with open(dst, "wb") as fh:
        out = _HashingWriter(fh)
        with tarfile.open(fileobj=out, mode="w|", format=tarfile.PAX_FORMAT) as tf:
            for name, path in files:
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.mode = path.stat().st_size, 0, 0o644
                with open(path, "rb") as src:
                    tf.addfile(info, src)
    return out.digest.hexdigest(), out.size


def _safe_name(name: str, top: Optional[str]) -> Optional[str]:
    """The member's path under the entry's directory; None for one outside `top`."""
    p = PurePosixPath(name)
    if top is not None:
        if not p.parts or p.parts[0] != top:
            return None
        p = PurePosixPath(*p.parts[1:])
    if not p.parts or p.is_absolute() or ".." in p.parts or "\\" in name:
        raise Refused(f"the archive holds a member outside its directory: {name!r}")
    return p.as_posix()


def extract_archive(archive: Path, into: Path, top: Optional[str] = None,
                    keep: Optional[Tuple[str, ...]] = None) -> Dict[str, int]:
    """Every regular file of `archive` (a tar, compressed or not) under
    `into`, its directory member `top` stripped when given, and only the
    files whose names match a pattern of `keep` when given; name -> size."""
    sizes: Dict[str, int] = {}
    with tarfile.open(archive, "r:*") as tf:
        for member in tf:
            if member.isdir():
                continue
            if not member.isfile():
                raise Refused(f"the archive holds a {member.type!r} member: {member.name!r}")
            name = _safe_name(member.name, top)
            if name is None or (keep is not None and not any(fnmatch(PurePosixPath(name).name, pat)
                                                             for pat in keep)):
                continue
            dst = into / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(member) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
            sizes[name] = member.size
    return sizes


def _swap_in(staged: Path, into: Path) -> None:
    """`staged` becomes `into`, whatever `into` held."""
    if into.exists():
        shutil.rmtree(into)
    os.replace(staged, into)


# ---- restore, store, list ---------------------------------------------------------

def _record(path: Optional[Path], line: str) -> None:
    print(line, flush=True)
    if path is not None:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {line}\n")


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def restore(host, key: Dict, into: Path, facts: Optional[Path] = None, record: Optional[Path] = None,
            legacy: Optional[str] = None, legacy_top: Optional[str] = None) -> int:
    """`into` from the key's entry (0), else from the legacy tarball when
    one is named and holds the key (EXIT_ADOPTED); EXIT_ABSENT when there
    is neither; EXIT_REFUSED when what was found does not hold the key."""
    kind, tag = key["kind"], f"{key['kind']} {key['key'][:16]}"
    into = Path(into)
    work = into.parent / f".{into.name}.restore"
    shutil.rmtree(work, ignore_errors=True)
    game_dirs = [Path(d) for d in key["inputs"].get("games", [])] or None
    t0 = time.time()
    try:
        try:
            manifest = json.loads(host.fetch(f"{entry_dir(key)}/manifest.json", work / "download")
                                  .read_text(encoding="utf-8"))
        except Absent:
            manifest = None
        if manifest is not None:
            _check_manifest(manifest, key)
            archive = host.fetch(f"{entry_dir(key)}/{manifest['archive']}", work / "download")
            fetched = time.time() - t0
            size = archive.stat().st_size
            if size != manifest["archive_bytes"] or file_sha256(archive) != manifest["archive_sha256"]:
                raise Refused(f"the archive is not the one the manifest names ({size} bytes, its SHA-256 differs)")
            sizes = extract_archive(archive, work / "files")
            archive.unlink()
            if sizes != {name: n for name, n in manifest["files"]}:
                raise Refused(f"the archive holds {len(sizes)} files, the manifest lists {len(manifest['files'])}")
            why = content_problem(kind, work / "files", key["components"], game_dirs=game_dirs)
            if why is not None:
                raise Refused(why)
            _swap_in(work / "files", into)
            if facts is not None and manifest.get("facts") is not None:
                Path(facts).write_text(json.dumps(manifest["facts"], indent=1), encoding="utf-8")
            _record(record, f"CACHE_RESTORED {tag}: {len(sizes)} files, {_gb(size)} in {time.time() - t0:.0f} s "
                            f"({size / max(fetched, 1e-3) / 1e6:.1f} MB/s fetched; entry of {manifest.get('created')})")
            return 0
        if legacy is None:
            _record(record, f"CACHE_ABSENT {tag}")
            return EXIT_ABSENT
        archive = host.fetch(legacy, work / "download")
        legacy_bytes = archive.stat().st_size
        extract_archive(archive, work / "files", top=legacy_top, keep=ENTRY_FILES[kind])
        archive.unlink()
        why = content_problem(kind, work / "files", key["components"], game_dirs=game_dirs, need_columns=False)
        if why is not None:
            raise Refused(f"the legacy tarball {legacy}: {why}")
        _swap_in(work / "files", into)
        _record(record, f"CACHE_ADOPTED {tag} from {legacy}: {_gb(legacy_bytes)} in "
                        f"{time.time() - t0:.0f} s; it goes up as an entry, and {legacy} can go")
        return EXIT_ADOPTED
    except Refused as exc:
        _record(record, f"CACHE_REFUSED {tag}: {exc}")
        return EXIT_REFUSED
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _check_manifest(manifest: Dict, key: Dict) -> None:
    if manifest.get("format") != ENTRY_FORMAT:
        raise Refused(f"an entry of format {manifest.get('format')}, this code reads {ENTRY_FORMAT}")
    if manifest.get("key") != key["key"] or manifest.get("components") != key["components"]:
        raise Refused("the entry's key or its inputs are not this key's")


def _fetch_json(host, path: str) -> Dict:
    with tempfile.TemporaryDirectory() as scratch:
        return json.loads(host.fetch(path, Path(scratch)).read_text(encoding="utf-8"))


def present(host, key: Dict) -> bool:
    """The key's entry is on the host, its archive of the size its manifest names."""
    try:
        manifest = _fetch_json(host, f"{entry_dir(key)}/manifest.json")
    except Absent:
        return False
    found = host.info(f"{entry_dir(key)}/{manifest.get('archive')}")
    return manifest.get("key") == key["key"] and found is not None and found[0] == manifest.get("archive_bytes")


def store(host, key: Dict, source: Path, facts: Optional[Path] = None, record: Optional[Path] = None,
          replace: bool = False) -> int:
    """`source` as the key's entry: its content checked, its archive sent,
    then its manifest. An entry already present stays, unless `replace`."""
    kind, tag = key["kind"], f"{key['kind']} {key['key'][:16]}"
    source = Path(source)
    facts_data = json.loads(Path(facts).read_text(encoding="utf-8")) if facts is not None else None
    game_dirs = [Path(d) for d in key["inputs"].get("games", [])] or None
    why = content_problem(kind, source, key["components"], facts=facts_data, game_dirs=game_dirs)
    if why is not None:
        _record(record, f"CACHE_NOT_SENT {tag}: {source} does not hold its key's input: {why}")
        return EXIT_REFUSED
    if not replace and present(host, key):
        _record(record, f"CACHE_PRESENT {tag}: nothing sent")
        return 0
    files = entry_files(kind, source)
    tmp = source.parent / f".{source.name}.cache.tar"
    t0 = time.time()
    try:
        sha, size = write_archive(files, tmp)
        name = f"{sha[:16]}.tar"
        host.put(tmp, f"{entry_dir(key)}/{name}")
        manifest = {"format": ENTRY_FORMAT, "kind": kind, "key": key["key"], "components": key["components"],
                    "inputs": key["inputs"], "archive": name, "archive_sha256": sha, "archive_bytes": size,
                    "files": [[n, p.stat().st_size] for n, p in files], "facts": facts_data,
                    "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "stage": os.environ.get("STAGE", "")}
        man_path = source.parent / f".{source.name}.cache_manifest.json"
        man_path.write_text(json.dumps(manifest), encoding="utf-8")
        host.put(man_path, f"{entry_dir(key)}/manifest.json")
        man_path.unlink()
    finally:
        tmp.unlink(missing_ok=True)
    _record(record, f"CACHE_STORED {tag}: {len(files)} files, {_gb(size)} in {time.time() - t0:.0f} s")
    for path, n in superseded(host, key):
        _record(record, f"CACHE_SUPERSEDED {path} ({_gb(n)}): another {kind} entry; the lead deletes it "
                        f"when nothing uses it")
    return 0


def superseded(host, key: Dict) -> List[Tuple[str, int]]:
    """(entry directory, bytes) of the other entries of the key's kind."""
    mine = entry_dir(key)
    sizes: Dict[str, int] = {}
    for path, n in host.files(f"{PREFIX}/{key['kind']}"):
        d = path.rsplit("/", 1)[0]
        if d != mine:
            sizes[d] = sizes.get(d, 0) + n
    return sorted(sizes.items())


def listing(host) -> List[str]:
    """One line per entry directory on the host: its kind, its bytes, and
    what its manifest says (or that it has none)."""
    by_dir: Dict[str, List[Tuple[str, int]]] = {}
    for path, n in host.files(PREFIX):
        by_dir.setdefault(path.rsplit("/", 1)[0], []).append((path, n))
    lines = []
    for d, files in sorted(by_dir.items()):
        total = sum(n for _, n in files)
        try:
            m = _fetch_json(host, f"{d}/manifest.json")
            comps = {k: (v[:16] if isinstance(v, str) and len(v) > 16 else v) for k, v in m["components"].items()}
            lines.append(f"{d} {_gb(total)} {len(m['files'])} files, created {m['created']} stage {m['stage']}: "
                         f"{json.dumps(comps, sort_keys=True)}")
        except Absent:
            lines.append(f"{d} {_gb(total)}: NO MANIFEST (a torn upload or a superseded archive)")
    return lines


def fetch_raw(host, key: Dict, into: Path) -> int:
    """The raw replays' tarball the corpus key names, its SHA-256 checked, unpacked into `into`."""
    path, want = key["inputs"]["raw_path"], key["components"]["raw_sha256"]
    work = Path(into).parent / f".{Path(into).name}.download"
    shutil.rmtree(work, ignore_errors=True)
    try:
        archive = host.fetch(path, work)
        if file_sha256(archive) != want:
            print(f"RAW_CHANGED: {path} is no longer the tarball the key names", flush=True)
            return EXIT_REFUSED
        sizes = extract_archive(archive, Path(into))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("raw replays", sum(1 for name in sizes if name.endswith(".bz2")), flush=True)
    return 0


def unpack(host, paths: List[str], into: Path) -> int:
    """Each tarball at `paths` on the host (match directories) unpacked into `into`."""
    work = Path(into).parent / f".{Path(into).name}.download"
    shutil.rmtree(work, ignore_errors=True)
    try:
        for path in paths:
            archive = host.fetch(path, work)
            sizes = extract_archive(archive, Path(into))
            archive.unlink()
            print(path, sum(1 for name in sizes if name.endswith(".game.jsonl.gz")), "game records", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


# ---- the command line ----------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default=None, help="a directory standing in for the model host")
    ap.add_argument("--record", type=Path, default=None, help="a file each outcome is appended to")
    sub = ap.add_subparsers(dest="command", required=True)
    k = sub.add_parser("key")
    k.add_argument("kind", choices=KINDS)
    k.add_argument("--out", type=Path, required=True)
    k.add_argument("--raw", help="corpus: the raw replays' tarball on the host")
    k.add_argument("--config", type=Path, default=Path("configs/imitation.json"))
    k.add_argument("--corpus-key", type=Path, help="sequences: the corpus's key file")
    k.add_argument("--vocab-from", type=Path, help="sequences: the checkpoint whose vocabularies encode")
    k.add_argument("--fresh-vocab", action="store_true", help="sequences: a fresh network's vocabularies")
    k.add_argument("--teacher", type=Path, help="own: the labelling's checkpoint")
    k.add_argument("--player-sha256", help="own: the player whose game-sides are selected")
    k.add_argument("--slots", type=int, help="own: the memory size those sides played at")
    k.add_argument("--games", type=Path, nargs="+", help="own: the match directories")
    f = sub.add_parser("fetch-raw")
    f.add_argument("--key", type=Path, required=True)
    f.add_argument("--into", type=Path, required=True)
    u = sub.add_parser("unpack")
    u.add_argument("--into", type=Path, required=True)
    u.add_argument("paths", nargs="+", help="tarballs on the host")
    r = sub.add_parser("restore")
    r.add_argument("--key", type=Path, required=True)
    r.add_argument("--into", type=Path, required=True)
    r.add_argument("--facts", type=Path, default=None, help="where the entry's facts go")
    r.add_argument("--legacy", default=None, help="a tarball on the host to adopt when the entry is absent")
    r.add_argument("--legacy-top", default=None, help="the legacy tarball's directory member")
    s = sub.add_parser("store")
    s.add_argument("--key", type=Path, required=True)
    s.add_argument("--from", dest="source", type=Path, required=True)
    s.add_argument("--facts", type=Path, default=None, help="the facts the entry carries (the corpus's summary)")
    s.add_argument("--replace", action="store_true", help="send even when an entry is present")
    sub.add_parser("list")
    args = ap.parse_args(argv)
    try:
        host = make_host(args.host)
        if args.command == "key":
            key = _key_command(host, args)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(key, indent=1), encoding="utf-8")
            print(f"key {key['kind']} {key['key'][:16]}: {json.dumps(key['components'], sort_keys=True)}", flush=True)
            return 0
        if args.command == "fetch-raw":
            return fetch_raw(host, read_key(args.key), args.into)
        if args.command == "unpack":
            return unpack(host, args.paths, args.into)
        if args.command == "restore":
            return restore(host, read_key(args.key), args.into, args.facts, args.record, args.legacy, args.legacy_top)
        if args.command == "store":
            return store(host, read_key(args.key), args.source, args.facts, args.record, args.replace)
        for line in listing(host):
            print(line)
        return 0
    except (RuntimeError, Absent) as exc:
        _record(args.record, f"CACHE_HOST_FAILED {args.command}: {type(exc).__name__} {exc}")
        return EXIT_HOST
    except Refused as exc:
        _record(args.record, f"CACHE_REFUSED {args.command}: {exc}")
        return EXIT_REFUSED


def _key_command(host, args) -> Dict:
    if args.kind == "corpus":
        if not args.raw:
            raise SystemExit("key corpus: --raw")
        return corpus_key(host, args.raw, args.config)
    if args.kind == "sequences":
        if args.corpus_key is None or (args.vocab_from is None) == (not args.fresh_vocab):
            raise SystemExit("key sequences: --corpus-key, and --vocab-from or --fresh-vocab")
        return sequences_key(read_key(args.corpus_key), args.vocab_from)
    if args.teacher is None or not args.player_sha256 or args.slots is None or not args.games:
        raise SystemExit("key own: --teacher, --player-sha256, --slots and --games")
    return own_key(args.teacher, args.player_sha256, args.slots, list(args.games))


if __name__ == "__main__":
    raise SystemExit(main())
