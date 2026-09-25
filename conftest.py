"""Project-local pytest configuration.

`pytest` from the project root discovers test files recursively.
`collect_ignore_glob` keeps collection out of trees that hold data,
not tests: `wesnoth_src/` (a WML-only copy of the Wesnoth 1.18.7 data
tree; CLAUDE.md, "Wesnoth data provenance"), the Lua add-on, logs and
the replay corpora. Add a glob here when another such tree appears.
"""

collect_ignore_glob = [
    "wesnoth_src/*",
    "add-ons/*",
    "logs/*",
    "replays_raw/*",
    "replays_dataset/*",
    "replays_dataset.old/*",
]
