"""The lobby era (tools/lobby_era.py): the committed era is what its
generator writes from the simulator's unit table, the live tool starts a
game in it only when the scenario declares no experience modifier, and
the first-frame check names every unit type whose experience is not a
hosted game's."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools import lobby_era  # noqa: E402
from tools.live_vs_rca import launch_args  # noqa: E402
from tools.replay_dataset import _scaled_max_exp  # noqa: E402
from tools.scenario_init_oracle import Declared  # noqa: E402
from wesnoth_ai.rules.wml_state import MP_EXPERIENCE_MODIFIER  # noqa: E402

PROBE = "Dwarvish Berserker"     # base 100: its units need the modifier itself


def test_the_committed_era_is_what_the_generator_writes():
    assert lobby_era.ERA_PATH.read_bytes().decode("utf-8") == lobby_era.render(), (
        "add-ons/wesnoth_ai/eras/lobby_era.cfg is stale: run python tools/lobby_era.py")


def test_a_scenario_declaring_its_modifier_is_started_in_the_default_era():
    def era_of(decl):
        return [a for a in launch_args("multiplayer_Hamlets", ("Drakes", "Undead"), 1, decl)
                if a.startswith("--era=")]
    assert era_of(Declared({}, {})) == [f"--era={lobby_era.ERA_ID}"]
    assert era_of(Declared({"experience_modifier": "90"}, {})) == ["--era=era_default"]


def _engine_report(modifier_of_base):
    """What lua/board_report.lua reports for every type of the table, the
    engine's base being the table's, plus one add-on type."""
    report = [{"type": name, "base": base, "applied": modifier_of_base(base)}
              for name, base in lobby_era.base_experiences().items()]
    return report + [{"type": "Some Add-on Unit", "base": 40, "applied": 40}]


def test_the_check_passes_a_hosted_games_experience_and_names_each_difference():
    hosted = _engine_report(lambda base: _scaled_max_exp(base, MP_EXPERIENCE_MODIFIER))
    defects, unknown = lobby_era.experience_defects(hosted, MP_EXPERIENCE_MODIFIER)
    assert (defects, unknown) == ([], ["Some Add-on Unit"])

    # The command line's own 100%, as if the era were not loaded.
    defects, _ = lobby_era.experience_defects(_engine_report(lambda base: base), MP_EXPERIENCE_MODIFIER)
    assert any(d.startswith(f"{PROBE}:") for d in defects)

    # The engine's base drifted from the table's for one type.
    drifted = [dict(e, base=e["base"] + 20) if e["type"] == PROBE else e for e in hosted]
    defects, _ = lobby_era.experience_defects(drifted, MP_EXPERIENCE_MODIFIER)
    assert len(defects) == 1 and defects[0].startswith(f"{PROBE}:")

    no_base = [dict(e, base=None) if e["type"] == PROBE else e for e in hosted]
    defects, _ = lobby_era.experience_defects(no_base, MP_EXPERIENCE_MODIFIER)
    assert len(defects) == 1 and "no base" in defects[0]
