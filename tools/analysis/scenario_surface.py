#!/usr/bin/env python3
"""Every tag and attribute a scenario we build or rebuild declares,
classified by what reads it.

    python tools/analysis/scenario_surface.py
    python tools/analysis/scenario_surface.py --write-manifest

The classes and the checks are in wesnoth_ai/rules/scenario_surface.py;
the manifest is tests/data/scenario_surface.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wesnoth_ai.rules.scenario_surface import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
