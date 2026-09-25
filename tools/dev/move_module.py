#!/usr/bin/env python3
"""Move one Python module and rewrite everything that imports it.

    python tools/dev/move_module.py tools.replay_dataset wesnoth_ai.sim.replay_dataset
    python tools/dev/move_module.py tools.replay_dataset wesnoth_ai.sim.replay_dataset --apply

OLD and NEW are dotted module names under one import root: the repo root
(`tools.x`, `wesnoth_ai.x`) or tests/ (`imitation_helpers`,
`helpers.imitation`). A module has a name under every import root that
holds it (`tests/x.py` is `x` and `tests.x`); each is rewritten to the
new file's name under the same root.

A dry run (the default) changes nothing and prints:
- the rewrites: `import a.b [as c]`, `from a import b`, `from a.b import
  x`, function-level imports, relative imports that reach the module
  (made absolute, as are the moved module's own), the `a.b.x` chains a
  plain `import a.b` is used through, string patch targets
  (`monkeypatch.setattr("a.b.x", v)`, `mock.patch("a.b.x")`) and literal
  `importlib.import_module("a.b")`;
- the refusals, sites whose origin is ambiguous: a bare import through
  tools/ on sys.path, a rebound package name, the module reached as an
  attribute of a package it was not imported from, a statement it cannot
  rewrite in place. With any refusal nothing is applied;
- the patches through the module object (`monkeypatch.setattr(mod, "x",
  v)`), which follow the rewritten import, flagged when `x` is not bound
  at the module's top level, and imports by computed name;
- the lines of the moved module that use `__file__`, `__name__` or
  `sys.path`;
- every other mention of the module in a text file (Python strings and
  comments, docs, scripts, JSON such as tests/data/scenario_surface.json),
  for a human to decide.

`--apply` performs the rewrites, moves the file (`git mv` when it is
tracked) and adds the move to MOVED_MODULES in wesnoth_ai/unpickle.py,
so pickles written under the old path still load. Then run `ruff check
.` and the tests.
"""
from __future__ import annotations

import argparse
import ast
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dev.import_rewrite import (FilePlan, ModuleNames, Source,  # noqa: E402
                                      own_location_notes, plan_file, top_level_names)

log = logging.getLogger("move_module")

# The directories on sys.path in practice, relative to the repo root.
# Names under BARE_ROOTS resolve too, and must never be used.
IMPORT_ROOTS = (".", "tests")
BARE_ROOTS = ("tools",)
# Trees a walk leaves out when the root is not a git work tree.
SKIPPED_DIRS = {".git", ".claude", "__pycache__", "wesnoth_src", "kaggle", ".venv", "venv",
                "node_modules"}
UNPICKLE_MODULE = Path("wesnoth_ai") / "unpickle.py"
TEXT_SUFFIXES = {".py", ".md", ".txt", ".sh", ".bash", ".sbatch", ".ps1", ".json", ".jsonl",
                 ".toml", ".ini", ".cfg", ".yml", ".yaml", ".rs", ".lua", ".csv", ".wml"}
RECORD_PREFIXES = ("docs/archive/", "quarantine/")


class MoveError(Exception):
    """The move cannot be planned or applied as asked."""


@dataclass
class MovePlan:
    names: ModuleNames
    files: List[FilePlan]                 # the Python files with something to report
    mentions: Dict[str, List[str]]        # category -> "path:line: text"
    own_notes: List[str]                  # the moved module's __file__ / __name__ / sys.path lines


# ---------------------------------------------------------------------
# The module's names
# ---------------------------------------------------------------------
def module_file(root: Path, import_root: str, name: str) -> Optional[Path]:
    base = (root / import_root).resolve().joinpath(*name.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def name_under(root: Path, import_root: str, path: Path) -> Optional[str]:
    """The dotted name `path` imports as with `import_root` on sys.path."""
    try:
        rel = path.resolve().relative_to((root / import_root).resolve())
    except ValueError:
        return None
    parts = rel.with_suffix("").parts
    return ".".join(parts) if parts and all(p.isidentifier() for p in parts) else None


def resolve_move(root: Path, old: str, new: str) -> ModuleNames:
    found = [(r, f) for r in IMPORT_ROOTS if (f := module_file(root, r, old))]
    if len(found) != 1:
        raise MoveError(f"{old} resolves to {len(found)} files under the import roots "
                        f"{IMPORT_ROOTS}: {[str(f) for _, f in found]}")
    import_root, old_file = found[0]
    if old_file.name == "__init__.py":
        raise MoveError(f"{old} is a package; this tool moves one module file")
    new_file = (root / import_root).resolve().joinpath(*new.split(".")).with_suffix(".py")
    check_destination(root, import_root, new, new_file)
    renames: Dict[str, str] = {}
    forbidden: Set[str] = set()
    for r in IMPORT_ROOTS:
        before, after = name_under(root, r, old_file), name_under(root, r, new_file)
        if before and after:
            renames[before] = after
        elif before:
            forbidden.add(before)
    forbidden.update(n for r in BARE_ROOTS if (n := name_under(root, r, old_file)))
    return ModuleNames(old=old, new=new, old_file=old_file, new_file=new_file,
                       renames=renames, forbidden=forbidden)


def check_destination(root: Path, import_root: str, new: str, new_file: Path) -> None:
    if new_file.exists() or new_file.with_suffix("").exists():
        raise MoveError(f"{new} exists already at {new_file}")
    for r in IMPORT_ROOTS:
        clash = module_file(root, r, new)
        if clash:
            raise MoveError(f"{new} already resolves to {clash}")
    if not new_file.parent.is_dir():
        raise MoveError(f"no directory {new_file.parent}: create the package first")
    parts = new.split(".")
    package = (root / import_root).resolve() / parts[0]
    if (package / "__init__.py").is_file():
        for part in parts[1:-1]:
            package = package / part
            if not (package / "__init__.py").is_file():
                raise MoveError(f"{package} has no __init__.py: create the package first")


# ---------------------------------------------------------------------
# The files
# ---------------------------------------------------------------------
def project_files(root: Path) -> List[Path]:
    """The files git tracks or would add; a walk when `root` is not a git
    work tree."""
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                             cwd=root, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return sorted(p for p in root.rglob("*")
                      if p.is_file() and not SKIPPED_DIRS & set(p.relative_to(root).parts))
    names = {n for n in out.stdout.decode("utf-8").split("\0") if n}
    return sorted(p for p in (root / n for n in names) if p.is_file())


def git_tracked(root: Path, path: Path) -> bool:
    try:
        subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=root,
                       capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def read_text(path: Path) -> Optional[str]:
    try:
        return path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def package_of(root: Path, path: Path) -> str:
    """What a relative import in `path` resolves against: its directory's
    dotted name under the repo root."""
    return ".".join(path.parent.relative_to(root).parts)


# ---------------------------------------------------------------------
# Mentions left for a human
# ---------------------------------------------------------------------
def mention_patterns(root: Path, names: ModuleNames) -> List[re.Pattern]:
    """The dotted names, the repo path (either slash, with or without
    .py) and the bare file name."""
    patterns = [re.compile(r"(?<![\w.])" + re.escape(n) + r"(?!\w)")
                for n in sorted(set(names.renames) | names.forbidden) if "." in n]
    rel = names.old_file.relative_to(root).with_suffix("").as_posix()
    patterns.append(re.compile(r"(?<![\w.])" + re.escape(rel).replace("/", r"[/\\]")
                               + r"(?:\.py)?(?!\w)"))
    patterns.append(re.compile(r"(?<![\w/\\.])" + re.escape(names.old_file.name) + r"(?!\w)"))
    return patterns


def mention_category(rel: str) -> str:
    if rel.startswith(RECORD_PREFIXES):
        return "records (docs/archive, quarantine)"
    suffix = Path(rel).suffix
    if suffix == ".py":
        return "python strings and comments"
    if suffix in (".md", ".txt"):
        return "docs"
    if suffix in (".sh", ".bash", ".sbatch", ".ps1", ".yml", ".yaml"):
        return "scripts and CI"
    return f"other {suffix or '(no suffix)'}"


def mentions(root: Path, files: List[Path], names: ModuleNames,
             covered: Dict[Path, List[Tuple[int, int]]]) -> Dict[str, List[str]]:
    """Mentions of the module in text files, outside the rewritten spans."""
    patterns = mention_patterns(root, names)
    stem = names.old_file.stem
    out: Dict[str, List[str]] = {}
    for path in files:
        text = read_text(path) if path.suffix in TEXT_SUFFIXES else None
        if text is None or stem not in text:
            continue
        spans = covered.get(path, [])
        lines_hit = set()
        for pattern in patterns:
            for m in pattern.finditer(text):
                if not any(a <= m.start() < b for a, b in spans):
                    lines_hit.add(text.count("\n", 0, m.start()) + 1)
        rel = path.relative_to(root).as_posix()
        lines = text.splitlines()
        for line in sorted(lines_hit):
            out.setdefault(mention_category(rel), []).append(
                f"{rel}:{line}: {lines[line - 1].strip()[:140]}")
    return out


# ---------------------------------------------------------------------
# Plan, report, apply
# ---------------------------------------------------------------------
def plan(root: Path, old: str, new: str) -> MovePlan:
    root = root.resolve()
    names = resolve_move(root, old, new)
    files = project_files(root)
    old_text = read_text(names.old_file) or ""
    exported = top_level_names(old_text)
    stem = names.old_file.stem
    plans: List[FilePlan] = []
    for path in files:
        text = read_text(path) if path.suffix == ".py" else None
        moved = path == names.old_file
        # Every name, path or relative import of the module spells its stem.
        if text is None or (stem not in text and not moved):
            continue
        fp = plan_file(path, text, names, package_of(root, path), exported, moved=moved)
        if fp.edits or fp.refusals or fp.notes:
            plans.append(fp)
    covered = {fp.path: [(e.start, e.end) for e in fp.edits] for fp in plans}
    return MovePlan(names, plans, mentions(root, files, names, covered),
                    own_location_notes(old_text))


def report(root: Path, move: MovePlan) -> str:
    names = move.names
    out = [f"move {names.old} -> {names.new}",
           f"  file: {names.old_file.relative_to(root).as_posix()} -> "
           f"{names.new_file.relative_to(root).as_posix()}",
           "  names rewritten: " + ", ".join(f"{a} -> {b}" for a, b in sorted(names.renames.items())),
           "  names refused: " + (", ".join(sorted(names.forbidden)) or "none")]
    sections = (("rewrites", "edits"), ("refusals", "refusals"),
                ("patches through the module object, imports by computed name", "notes"))
    for title, attr in sections:
        rows = [f"  {fp.path.relative_to(root).as_posix()}:{item.line}: {item.what}"
                for fp in move.files for item in sorted(getattr(fp, attr), key=lambda i: i.line)]
        out.append(f"\n{title} ({len(rows)}):")
        out.extend(rows)
    out.append(f"\nthe moved module's own location and name ({len(move.own_notes)}):")
    out.extend(f"  {note}" for note in move.own_notes)
    out.append(f"\nother mentions ({sum(len(v) for v in move.mentions.values())}), "
               f"for a human to decide:")
    for category in sorted(move.mentions):
        out.append(f"  {category} ({len(move.mentions[category])}):")
        out.extend(f"    {row}" for row in move.mentions[category])
    return "\n".join(out)


def apply(root: Path, move: MovePlan) -> None:
    """Every new text is computed and parsed before anything is written."""
    names = move.names
    rewritten: Dict[Path, str] = {}
    for fp in move.files:
        if fp.edits:
            text = fp.rewritten()
            try:
                ast.parse(text)
            except SyntaxError as e:
                raise MoveError(f"{fp.path}: the rewrite does not parse ({e}); nothing changed") from e
            rewritten[fp.path] = text
    table = pickle_table_update(root, names)
    moved_text = rewritten.pop(names.old_file, None)
    for path, text in rewritten.items():
        path.write_bytes(text.encode("utf-8"))
    if git_tracked(root, names.old_file):
        subprocess.run(["git", "mv", str(names.old_file), str(names.new_file)], cwd=root, check=True)
    else:
        names.old_file.rename(names.new_file)
    if moved_text is not None:
        names.new_file.write_bytes(moved_text.encode("utf-8"))
    if table:
        table[0].write_bytes(table[1].encode("utf-8"))


def pickle_table_update(root: Path, names: ModuleNames) -> Optional[Tuple[Path, str]]:
    """The unpickler's module with old -> new (repo-root names) added to
    MOVED_MODULES, or None: no table in this tree, or a test module
    (nothing kept on disk pickles one)."""
    old, new = name_under(root, ".", names.old_file), name_under(root, ".", names.new_file)
    table = root / UNPICKLE_MODULE
    if not (old and new and table.is_file()) or old.startswith("tests."):
        return None
    text = table.read_bytes().decode("utf-8")
    for node in ast.parse(text).body:
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) else None)
        if isinstance(target, ast.Name) and target.id == "MOVED_MODULES" \
                and isinstance(node.value, ast.Dict):
            entries = {ast.literal_eval(k): ast.literal_eval(v)
                       for k, v in zip(node.value.keys, node.value.values)}
            entries[old] = new
            literal = "{\n" + "".join(f'    "{k}": "{v}",\n' for k, v in sorted(entries.items())) + "}"
            start, end = Source(text).span(node.value)
            return table, text[:start] + literal + text[end:]
    raise MoveError(f"{table}: no MOVED_MODULES dict to add {old} -> {new} to")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("old", help="the module's dotted name now, e.g. tools.replay_dataset")
    ap.add_argument("new", help="its dotted name after the move, e.g. wesnoth_ai.sim.replay_dataset")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite and move; without it, report and change nothing")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2],
                    help="the repo root (default: the repo holding this script)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        # Mentions quote lines of any text; a console that cannot show a
        # character gets its escape instead of an error.
        sys.stdout.reconfigure(errors="backslashreplace")
    root = args.root.resolve()
    try:
        move = plan(root, args.old, args.new)
    except MoveError as e:
        log.error(f"refused: {e}")
        return 2
    print(report(root, move))
    refusals = sum(len(fp.refusals) for fp in move.files)
    if refusals:
        print(f"\n{refusals} refusal(s): resolve them by hand, then run again. Nothing changed.")
        return 2
    if not args.apply:
        print("\ndry run: nothing changed; --apply performs it.")
        return 0
    try:
        apply(root, move)
    except MoveError as e:
        log.error(f"refused: {e}")
        return 2
    print("\nmoved. Next: ruff check . and the tests, then the mentions above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
