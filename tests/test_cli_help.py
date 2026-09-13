"""Every argparse help string must survive being formatted.

argparse runs `help % params` when it renders `--help`, so a literal
percent sign in a help string raises at render time, not at import: the
entry point works and only its `--help` crashes, which is exactly how
`tools/az_loop.py --help` stayed broken (a "within 0.3% on one batch"
in a flag's help, found 2026-09-13). A percent must be written `%%`,
unless it is one of argparse's own `%(name)s` substitutions.

This is a text scan, so it costs milliseconds and covers every tool at
once; running each `--help` as a subprocess would import torch dozens
of times.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).parent.parent

# A percent that argparse will try to interpret: not `%%` and not the
# start of a `%(name)s` substitution.
_BAD_PERCENT = re.compile(r"(?<!%)%(?![%(])")
# `help=` / `description=` strings, including implicit concatenation of
# adjacent literals across lines.
_HELP_ARG = re.compile(
    r"""(?:help|description)\s*=\s*((?:\s*(?:'[^']*'|"[^"]*"|'''.*?'''|\"\"\".*?\"\"\"))+)""",
    re.DOTALL)
_LITERAL = re.compile(r"'''.*?'''|\"\"\".*?\"\"\"|'[^']*'|\"[^\"]*\"", re.DOTALL)


def _offenders(text: str) -> List[str]:
    out = []
    for m in _HELP_ARG.finditer(text):
        blob = "".join(lit[1:-1] if not lit.startswith(("'''", '"""')) else lit[3:-3]
                       for lit in _LITERAL.findall(m.group(1)))
        # `\%` cannot appear; a doubled percent is already safe.
        if _BAD_PERCENT.search(blob):
            line = text[:m.start()].count("\n") + 1
            out.append(f"line {line}: {blob[:90]!r}")
    return out


def _python_files() -> List[Path]:
    files: List[Path] = []
    for sub in ("tools", "scripts", "wesnoth_ai"):
        files.extend(sorted((ROOT / sub).glob("*.py")))
    files.append(ROOT / "main.py")
    return [f for f in files if f.exists()]


def test_no_help_string_contains_a_bare_percent():
    bad: List[Tuple[str, List[str]]] = []
    scanned = 0
    for f in _python_files():
        text = f.read_text(encoding="utf-8", errors="replace")
        if "add_argument" not in text and "ArgumentParser" not in text:
            continue
        scanned += 1
        hits = _offenders(text)
        if hits:
            bad.append((str(f.relative_to(ROOT)), hits))
    assert scanned >= 15, f"the scan only reached {scanned} files with a parser"
    assert not bad, "argparse renders help with %-formatting; write %% for a literal percent:\n" + \
        "\n".join(f"  {name}: {', '.join(h)}" for name, h in bad)


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
