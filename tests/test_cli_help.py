"""Every argparse help string must survive being formatted.

argparse runs `help % params` when it renders `--help`, so a literal
percent sign in a help string raises at render time, not at import: the
entry point works and only its `--help` crashes, which is exactly how
`tools/az_loop.py --help` stayed broken (a "within 0.3% on one batch"
in a flag's help, found 2026-09-13). A percent must be written `%%`,
unless it is one of argparse's own `%(name)s` substitutions.

This is a source scan (the AST, so a help string built by
concatenation, parenthesised, or as an f-string is read too), so it
costs milliseconds and covers every tool at once; running each `--help`
as a subprocess would import torch dozens of times. A `help=` whose
value is a name or a call cannot be read here and is reported as such
rather than passed silently. The scan walks tools/, scripts/ and the
package recursively, so it still reads a parser that moves into a
subpackage.
"""
from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

from helpers.source_tree import source_files
from wesnoth_ai.paths import REPO_ROOT as ROOT

# A percent that argparse will try to interpret: not `%%` and not the
# start of a `%(name)s` substitution.
_BAD_PERCENT = re.compile(r"(?<!%)%(?![%(])")


def _literal_text(node: ast.AST) -> Optional[str]:
    """The text argparse will format, for a literal, an implicit or
    explicit concatenation of literals, or an f-string (its literal
    parts only); None when the value cannot be read statically."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value for v in node.values
                       if isinstance(v, ast.Constant) and isinstance(v.value, str))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _literal_text(node.left), _literal_text(node.right)
        return None if left is None or right is None else left + right
    return None


def _scan(text: str) -> Tuple[List[str], List[str]]:
    """(offenders, unreadable): one line per `help=` argument whose
    rendered text carries a bare percent, and one per `help=` whose
    value this scan cannot read (a name, a call). `description=` is
    left alone: argparse formats it only when it contains `%(prog)`
    (HelpFormatter._format_text), so a bare percent there is safe."""
    offenders: List[str] = []
    unreadable: List[str] = []
    tree = ast.parse(textwrap.dedent(text))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "help":
                continue
            blob = _literal_text(kw.value)
            if blob is None:
                unreadable.append(f"line {kw.value.lineno}")
            elif _BAD_PERCENT.search(blob):
                offenders.append(f"line {kw.value.lineno}: {blob[:90]!r}")
    return offenders, unreadable


def _offenders(text: str) -> List[str]:
    return _scan(text)[0]


# `help=` values built at runtime (a constant, a call) that the scan
# cannot read; each was checked by hand on 2026-09-14. A new one lands
# here only after the same check, or by becoming a literal.
_UNREADABLE_HELP = {
    "tools/cleanup.py", "tools/eval_vs_builtin.py", "tools/net2net.py",
    "tools/run_elo_batch.py",
}


def _python_files() -> List[Path]:
    return source_files("tools", "scripts", "wesnoth_ai") + [ROOT / "main.py"]


def test_no_help_string_contains_a_bare_percent():
    bad: List[Tuple[str, List[str]]] = []
    blind: List[Tuple[str, List[str]]] = []
    scanned = 0
    for f in _python_files():
        text = f.read_text(encoding="utf-8", errors="replace")
        if "add_argument" not in text and "ArgumentParser" not in text:
            continue
        scanned += 1
        hits, unreadable = _scan(text)
        name = str(f.relative_to(ROOT)).replace("\\", "/")
        if hits:
            bad.append((name, hits))
        if unreadable and name not in _UNREADABLE_HELP:
            blind.append((name, unreadable))
    assert scanned >= 15, f"the scan only reached {scanned} files with a parser"
    assert not bad, "argparse renders help with %-formatting; write %% for a literal percent:\n" + \
        "\n".join(f"  {name}: {', '.join(h)}" for name, h in bad)
    assert not blind, "a help= value this scan cannot read; check it by hand and list it:\n" + \
        "\n".join(f"  {name}: {', '.join(h)}" for name, h in blind)


def test_the_scan_would_catch_the_bug_it_was_written_for():
    """The az_loop regression, verbatim, must be flagged."""
    sample = '''
    ap.add_argument("--train-bf16", help="gradient "
                    "cosine 0.9994, norm within 0.3% on one batch of "
                    "64. Default on.")
    '''
    assert _offenders(sample), "the scan must catch a bare percent"
    fixed = sample.replace("0.3%", "0.3%%")
    assert not _offenders(fixed), "a doubled percent must pass"
    assert not _offenders('ap.add_argument("--x", help="default: %(default)s")'), \
        "argparse's own substitutions must pass"
    assert _offenders('ap.add_argument("--x", help=("within 0.3% of " + "the batch"))'), \
        "a parenthesised concatenation must be read"
    assert _offenders('ap.add_argument("--x", help=f"within 0.3% of {n}")'), \
        "an f-string's literal parts must be read"
    assert _scan('ap.add_argument("--x", help=HELP_TEXT)')[1], \
        "a value the scan cannot read must be reported, not passed"
    assert not _offenders('ap = argparse.ArgumentParser(description="50% of the time")'), \
        "a description is only formatted when it names %(prog)"
