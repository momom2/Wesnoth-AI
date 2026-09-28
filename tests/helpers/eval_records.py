"""Fields a synthetic eval result needs to pass the provenance guards it
is not testing."""
from tools.eval_provenance import forced_faction_tag


def current_forced_faction() -> str:
    """The faction regime a run started now would record
    (`scenario_pool.FORCED_FACTION`, read at call time so a test's
    monkeypatch is seen)."""
    from wesnoth_ai.rules import scenario_pool
    return forced_faction_tag(scenario_pool.FORCED_FACTION)
