#!/usr/bin/env python3
"""Our macro expansion of each pool scenario against the game's own.

    python tools/analysis/expansion_diff.py [--out FILE] [--write-expected] [--scenario ID]...

The comparison, and what it leaves out, is in
wesnoth_ai/rules/expansion_diff.py; tests/test_expansion_diff.py fails on
a divergence tests/data/expansion_diff_expected.json does not record.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wesnoth_ai.rules.expansion_diff import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
