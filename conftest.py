"""Project-local pytest configuration.

`pytest` from the project root discovers test files recursively. The
`wesnoth_src/` subtree is a vendored Wesnoth source checkout (see
CLAUDE.md -- pinned to the 1.18.4 tag) and contains Python 2 utility
scripts (e.g. `wesnoth_src/utils/ai_test/ai_test.py` imports
`time.clock`, removed in Python 3) that pytest fails to collect.

`collect_ignore` lists directory names to skip during collection.
Add to this list when other vendored subtrees show up.
"""

collect_ignore_glob = [
    "wesnoth_src/*",
    "add-ons/*",
    "logs/*",
    "replays_raw/*",
    "replays_dataset/*",
    "replays_dataset.old/*",
]
