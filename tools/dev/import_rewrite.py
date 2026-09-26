"""The per-file half of tools/dev/move_module.py: where one Python file
names a module that moves, what to rewrite, and what to refuse.

Decisions are taken on the AST and applied as text splices, so a file
keeps its formatting and comments outside the rewritten spans. Scopes
are not analysed: a plain `import a.b` and the `a.b.x` chains that use
it are rewritten together wherever they sit in the file.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Calls whose first argument, as a string, names a patch target:
# monkeypatch.setattr / delattr, mock.patch, patch.multiple.
PATCH_BY_STRING = {"setattr", "delattr", "patch", "multiple"}
# Calls that patch an attribute of an object: monkeypatch.setattr(obj,
# "x", v), the builtin setattr, patch.object(obj, "x").
PATCH_BY_OBJECT = {"setattr", "delattr", "object"}
DYNAMIC_IMPORTS = {"import_module", "__import__"}


@dataclass
class ModuleNames:
    """A move: the module's names before and after it."""
    old: str
    new: str
    old_file: Path
    new_file: Path
    # Old dotted name -> new dotted name, one per import root holding both files.
    renames: Dict[str, str]
    # Names that reach the old file but have no counterpart after the move:
    # a bare name through tools/ on sys.path, or a root that loses the module.
    forbidden: Set[str]


@dataclass
class Edit:
    """Replace text[start:end] with `text`."""
    start: int
    end: int
    text: str
    line: int
    what: str


@dataclass
class Finding:
    line: int
    what: str


@dataclass
class FilePlan:
    path: Path
    text: str
    edits: List[Edit] = field(default_factory=list)
    refusals: List[Finding] = field(default_factory=list)
    notes: List[Finding] = field(default_factory=list)

    def rewritten(self) -> str:
        out = self.text
        floor = len(out) + 1
        for e in sorted(self.edits, key=lambda e: e.start, reverse=True):
            if e.end > floor:
                raise ValueError(f"{self.path}: overlapping rewrites at line {e.line}")
            out = out[:e.start] + e.text + out[e.end:]
            floor = e.start
        return out


class Source:
    """Character offsets from the AST's (line, UTF-8 byte column) pairs."""

    def __init__(self, text: str):
        self.text = text
        self.newline = "\r\n" if "\r\n" in text else "\n"
        self.lines = text.splitlines(keepends=True)
        self.starts = [0]
        for line in self.lines:
            self.starts.append(self.starts[-1] + len(line))

    def offset(self, lineno: int, byte_col: int) -> int:
        line = self.lines[lineno - 1] if lineno <= len(self.lines) else ""
        return self.starts[lineno - 1] + len(line.encode("utf-8")[:byte_col].decode("utf-8"))

    def span(self, node: ast.AST) -> Tuple[int, int]:
        return (self.offset(node.lineno, node.col_offset),
                self.offset(node.end_lineno, node.end_col_offset))

    def line_end(self, lineno: int) -> int:
        """The offset of the line's last character before its newline."""
        return self.starts[lineno - 1] + len(self.lines[lineno - 1].rstrip("\r\n"))


def dotted(node: ast.AST) -> Optional[str]:
    """`a.b.c` for an attribute chain on a bare name, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def chain_root(node: ast.AST) -> ast.AST:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node


def call_name(node: ast.Call) -> str:
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else "")


def string_arg(node: ast.Call, index: int) -> Optional[ast.Constant]:
    if len(node.args) > index and isinstance(node.args[index], ast.Constant) \
            and isinstance(node.args[index].value, str):
        return node.args[index]
    return None


# ---------------------------------------------------------------------
# What a module binds
# ---------------------------------------------------------------------
def non_import_bindings(tree: ast.AST) -> Dict[str, int]:
    """Name -> first line, for the names the file binds other than by an
    import, in any scope: a conservative test for a rebound package name."""
    names: Dict[str, int] = {}

    def add(name: str, line: int) -> None:
        names.setdefault(name, line)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            add(node.id, node.lineno)
        elif isinstance(node, ast.arg):
            add(node.arg, node.lineno)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add(node.name, node.lineno)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            add(node.name, node.lineno)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                add(name, node.lineno)
    return names


def top_level_names(text: str) -> Set[str]:
    """Names a module binds at its top level, inside top-level compound
    statements too, and through `global` in its functions."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    names: Set[str] = set()

    def visit(statements: List[ast.stmt]) -> None:
        for node in statements:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(a.asname or a.name.split(".")[0] for a in node.names)
                continue
            for child in ast.iter_child_nodes(node):
                if not isinstance(child, (ast.stmt, ast.ExceptHandler)):
                    names.update(n.id for n in ast.walk(child)
                                 if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
            for block in ("body", "orelse", "finalbody"):
                visit(getattr(node, block, []))
            for handler in getattr(node, "handlers", []):
                if handler.name:
                    names.add(handler.name)
                visit(handler.body)
    visit(tree.body)
    names.update(n for node in ast.walk(tree) if isinstance(node, ast.Global) for n in node.names)
    return names


def own_location_notes(text: str) -> List[str]:
    """Lines of the moved module that depend on where it lives or what it
    is called: __file__, __name__, sys.path."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ["the module does not parse"]
    lines = text.splitlines()
    notes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in ("__file__", "__name__"):
            notes.add((node.lineno, node.id))
        elif isinstance(node, ast.Attribute) and dotted(node) == "sys.path":
            notes.add((node.lineno, "sys.path"))
    return [f"line {line}: {what}: {lines[line - 1].strip()}" for line, what in sorted(notes)]


# ---------------------------------------------------------------------
# `from` imports
# ---------------------------------------------------------------------
def absolute_module(node: ast.ImportFrom, package: str) -> Optional[str]:
    """The absolute module a `from` import names, or None when a relative
    one climbs out of the file's package."""
    if not node.level:
        return node.module
    parts = package.split(".") if package else []
    if node.level - 1 > len(parts):
        return None
    base = parts[:len(parts) - (node.level - 1)]
    if node.module:
        base += node.module.split(".")
    return ".".join(base) or None


def from_module_span(src: Source, node: ast.ImportFrom) -> Optional[Tuple[int, int]]:
    """The span of `.x.y` in `from .x.y import ...`; None when it is not
    written plainly after `from` on the statement's first line."""
    i = src.offset(node.lineno, node.col_offset) + len("from")
    while i < len(src.text) and src.text[i] in " \t":
        i += 1
    expected = "." * node.level + (node.module or "")
    end = i + len(expected)
    if src.text.startswith(expected, i) and end < len(src.text) and src.text[end] in " \t\\(":
        return i, end
    return None


def realigned(tail: str, paren_column: int, shift: int) -> str:
    """`tail`, the rest of a `from` import after its module name, with its
    continuation lines moved `shift` columns: the names stay under the
    opening parenthesis at `paren_column` of the first line when the
    module name before it changes length. Unchanged unless every
    continuation line starts right after that parenthesis's column (a
    hanging indent, or any other layout, is left as written)."""
    first, newline, rest = tail.partition("\n")
    opened = first.find("(")
    if not shift or not newline or opened < 0:
        return tail
    after = first[opened + 1:].strip()
    if not after or after.startswith("#"):
        return tail
    column = paren_column + 1
    lines = rest.split("\n")
    if any(len(line) - len(line.lstrip(" ")) != column for line in lines):
        return tail
    return first + newline + "\n".join(" " * (column + shift) + line[column:] for line in lines)


# ---------------------------------------------------------------------
# One file
# ---------------------------------------------------------------------
class FilePlanner:
    def __init__(self, path: Path, text: str, names: ModuleNames, package: str,
                 exported: Set[str], moved: bool):
        self.plan = FilePlan(path=path, text=text)
        self.src = Source(text)
        self.names = names
        self.package = package
        self.exported = exported
        self.moved = moved
        self.module_aliases: Set[str] = set()   # local names bound to the moving module
        self.plain_imported: Set[str] = set()   # its dotted names imported without `as`
        self.other_roots: Set[str] = set()      # top-level names other plain imports bind

    def edit(self, start: int, end: int, text: str, line: int, what: str) -> None:
        self.plan.edits.append(Edit(start, end, text, line, what))

    def refuse(self, line: int, what: str) -> None:
        self.plan.refusals.append(Finding(line, what))

    def note(self, line: int, what: str) -> None:
        self.plan.notes.append(Finding(line, what))

    def renamed(self, name: str) -> Optional[str]:
        """The new spelling of `name` if it is the module or reaches into it."""
        for old, new in self.names.renames.items():
            if name == old or name.startswith(old + "."):
                return new + name[len(old):]
        return None

    def forbidden(self, name: str) -> Optional[str]:
        for bad in self.names.forbidden:
            if name == bad or name.startswith(bad + "."):
                return bad
        return None

    # -- import statements -----------------------------------------------
    def visit_import(self, node: ast.Import) -> None:
        for alias in node.names:
            if self.forbidden(alias.name):
                self.refuse(node.lineno, f"`import {alias.name}` reaches the module by a name "
                                         f"that has no counterpart after the move (a bare import "
                                         f"through tools/ on sys.path?): import it by its project "
                                         f"name first")
                continue
            new = self.renamed(alias.name)
            if new is None:
                if alias.asname is None:
                    self.other_roots.add(alias.name.split(".")[0])
                continue
            start = self.src.offset(alias.lineno, alias.col_offset)
            end = start + len(alias.name)
            if self.src.text[start:end] != alias.name:
                self.refuse(node.lineno, f"cannot locate `{alias.name}` in the statement")
                continue
            if alias.asname:
                self.module_aliases.add(alias.asname)
                self.edit(start, end, new, node.lineno, f"import {alias.name} -> import {new}")
            elif "." not in alias.name:
                # `import mod` binds `mod`; keep that binding.
                self.module_aliases.add(alias.name)
                self.edit(start, end, f"{new} as {alias.name}", node.lineno,
                          f"import {alias.name} -> import {new} as {alias.name}")
            else:
                self.plain_imported.add(alias.name)
                self.edit(start, end, new, node.lineno, f"import {alias.name} -> import {new}")

    def visit_import_from(self, node: ast.ImportFrom) -> None:
        module = absolute_module(node, self.package)
        if module is None:
            if self.moved and node.level:
                self.refuse(node.lineno, "a relative import that climbs out of the package")
            return
        if self.forbidden(module) or any(self.forbidden(f"{module}.{a.name}") for a in node.names):
            self.refuse(node.lineno, f"`from {module} import ...` reaches the module by a name "
                                     f"that has no counterpart after the move")
            return
        new_module = self.renamed(module)
        if new_module is not None:
            self.replace_from_module(node, new_module)
            return
        taken = [a for a in node.names if self.renamed(f"{module}.{a.name}")]
        if taken:
            self.split_from(node, module, taken)
        elif node.level and self.moved:
            # The moved module's own relative import: its package changes.
            self.replace_from_module(node, module)

    def replace_from_module(self, node: ast.ImportFrom, new_module: str) -> None:
        span = from_module_span(self.src, node)
        if span is None:
            self.refuse(node.lineno, "the module of this `from` import is not written plainly "
                                     "after `from`: rewrite it by hand")
            return
        old_text = self.src.text[span[0]:span[1]]
        _, end = self.src.span(node)
        tail = self.src.text[span[1]:end]
        paren_column = span[1] - self.src.starts[node.lineno - 1] + tail.find("(")
        tail = realigned(tail, paren_column, len(new_module) - len(old_text))
        self.edit(span[0], end, new_module + tail, node.lineno,
                  f"from {old_text} import ... -> from {new_module} import ...")

    def split_from(self, node: ast.ImportFrom, module: str, taken: List[ast.alias]) -> None:
        """`from pkg import mod, other` where `pkg.mod` moves: the moving
        name is imported from its new package, under the same local name."""
        start, end = self.src.span(node)
        statement = self.src.text[start:end]
        if "#" in statement:
            self.refuse(node.lineno, "a multi-line import with comments inside: split it by hand")
            return
        kept = [a for a in node.names if a not in taken]
        statements = []
        if kept:
            head = module
            if node.level and not self.moved:
                span = from_module_span(self.src, node)
                head = self.src.text[span[0]:span[1]] if span else module
            statements.append(f"from {head} import " + ", ".join(
                a.name + (f" as {a.asname}" if a.asname else "") for a in kept))
        for alias in taken:
            local = alias.asname or alias.name
            new = self.renamed(f"{module}.{alias.name}")
            parent, _, last = new.rpartition(".")
            if parent:
                statements.append(f"from {parent} import {last}" + (f" as {local}" if local != last else ""))
            else:
                statements.append(f"import {new}" + (f" as {local}" if local != new else ""))
            self.module_aliases.add(local)
        indent = self.src.text[self.src.starts[node.lineno - 1]:start]
        tail = self.src.text[end:self.src.line_end(node.end_lineno)].strip()
        if not indent.strip() and (not tail or tail.startswith("#")):
            text = ((f"  {tail}" if tail else "") + self.src.newline + indent).join(statements)
        else:
            text = "; ".join(statements)
        self.edit(start, end, text, node.lineno, f"{' '.join(statement.split())} -> "
                  + " / ".join(statements))

    # -- attribute chains --------------------------------------------------
    def visit_chains(self, attributes: List[ast.Attribute]) -> None:
        """`a.b.x` where a plain `import a.b` moves: rewrite the `a.b`.
        `a.b` reached as an attribute of a package imported otherwise:
        refuse, since it works only when something else loaded `a.b`."""
        for node in attributes:
            chain = dotted(node)
            if chain is None or (chain not in self.names.renames and chain not in self.names.forbidden):
                continue
            if chain in self.plain_imported:
                start, _ = self.src.span(chain_root(node))
                _, end = self.src.span(node)
                new = self.names.renames[chain]
                self.edit(start, end, new, node.lineno, f"{chain} -> {new}")
            else:
                self.refuse(node.lineno, f"`{chain}` is reached as an attribute of its package, "
                                         f"not through an import of the module: import it by name")

    def check_rebinding(self, tree: ast.AST, names: List[ast.Name]) -> None:
        """A plain `import a.b` binds `a`: refuse when `a` (or the new
        top-level name) is rebound in the file, or used for something
        other than `a.b` with no other import binding it."""
        if not self.plain_imported:
            return
        bound = non_import_bindings(tree)
        chain_starts = {e.start for e in self.plan.edits}
        for old in sorted(self.plain_imported):
            root_old, root_new = old.split(".")[0], self.names.renames[old].split(".")[0]
            for name in sorted({root_old, root_new} & set(bound)):
                self.refuse(bound[name], f"`{name}` is rebound in this file, so a chain on it may "
                                         f"not be the module: rewrite `import {old}` by hand")
            if root_old == root_new or root_old in self.other_roots:
                continue
            for node in names:
                if (node.id == root_old
                        and self.src.offset(node.lineno, node.col_offset) not in chain_starts):
                    self.refuse(node.lineno, f"`{root_old}` is used here through the binding "
                                             f"`import {old}` made, which the move removes")

    # -- calls -------------------------------------------------------------
    def visit_call(self, node: ast.Call) -> None:
        name = call_name(node)
        first = string_arg(node, 0)
        if name in DYNAMIC_IMPORTS:
            self.visit_dynamic_import(node, first)
        elif name in PATCH_BY_STRING and first is not None:
            self.rewrite_string(first, "patch target")
        if name in PATCH_BY_OBJECT and first is None:
            self.visit_patch_object(node, name)

    def visit_dynamic_import(self, node: ast.Call, first: Optional[ast.Constant]) -> None:
        if first is None or first.value.startswith("."):
            self.note(node.lineno, "an import by computed or relative name: check it by hand")
            return
        self.rewrite_string(first, "import by name")

    def rewrite_string(self, arg: ast.Constant, kind: str) -> None:
        literal = arg.value
        if self.forbidden(literal):
            self.refuse(arg.lineno, f"{kind} {literal!r} has no counterpart after the move")
            return
        new = self.renamed(literal)
        if new is None:
            return
        start, end = self.src.span(arg)
        raw = self.src.text[start:end]
        if literal not in raw:
            self.refuse(arg.lineno, f"cannot rewrite the {kind} {literal!r} in place")
            return
        self.edit(start, end, raw.replace(literal, new, 1), arg.lineno, f"{kind} {literal!r} -> {new!r}")

    def visit_patch_object(self, node: ast.Call, name: str) -> None:
        if name == "object" and dotted(node.func.value if isinstance(node.func, ast.Attribute)
                                       else node.func) not in ("patch", "mock.patch",
                                                               "unittest.mock.patch", "mocker.patch"):
            return
        attr = string_arg(node, 1)
        if not node.args or attr is None:
            return
        target = node.args[0]
        chain = dotted(target)
        if not ((isinstance(target, ast.Name) and target.id in self.module_aliases)
                or chain in self.plain_imported):
            return
        missing = "" if attr.value in self.exported else (
            f": `{attr.value}` is NOT bound at the module's top level")
        self.note(node.lineno, f"patches `{attr.value}` through the module object{missing}")

    def run(self) -> FilePlan:
        try:
            tree = ast.parse(self.src.text)
        except SyntaxError:
            stems = {n.split(".")[-1] for n in self.names.renames} | self.names.forbidden
            if any(stem in self.src.text for stem in stems):
                self.refuse(1, "the file does not parse and mentions the module")
            return self.plan
        by_type: Dict[type, list] = {ast.Import: [], ast.ImportFrom: [], ast.Attribute: [],
                                     ast.Call: [], ast.Name: []}
        for node in ast.walk(tree):
            bucket = by_type.get(type(node))
            if bucket is not None:
                bucket.append(node)
        for node in by_type[ast.Import]:
            self.visit_import(node)
        for node in by_type[ast.ImportFrom]:
            self.visit_import_from(node)
        self.visit_chains(by_type[ast.Attribute])
        self.check_rebinding(tree, by_type[ast.Name])
        for node in by_type[ast.Call]:
            self.visit_call(node)
        return self.plan


def plan_file(path: Path, text: str, names: ModuleNames, package: str, exported: Set[str],
              moved: bool = False) -> FilePlan:
    return FilePlanner(path, text, names, package, exported, moved).run()
