"""Build the per-scenario templates under tools/templates/scenarios/ from
the game's own .cfg files, expanded by the game's own preprocessor.

    python tools/build_scenario_templates.py            # all
    python tools/build_scenario_templates.py --only multiplayer_Hamlets 2p_mini
    python tools/build_scenario_templates.py --out-dir /tmp/x  # then diff

The code, and what it does to each scenario, is in
wesnoth_ai/rules/build_scenario_templates.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wesnoth_ai.rules.build_scenario_templates import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv))
