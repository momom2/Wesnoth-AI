"""Guard against dual-imported modules (same file, two module objects).

Because tests/conftest.py and every entry script put BOTH the repo root
and tools/ on sys.path, `import wesnoth_sim` and `import
tools.wesnoth_sim` both resolve -- to TWO DISTINCT module objects for
the same file. Any module-level mutable state (the unit-stats DBs,
movement-cost memos, the opener registry, VALIDATION_EXPORTER) is then
duplicated, and writers on one side are invisible to readers on the
other. This silently cost a probe a zero-capture run and, pre-audit
(2026-07-30), the production worker process carried FOUR live dual
pairs (wesnoth_sim, replay_dataset, combat_outcomes, draw_tiebreak).

The audit normalized every import to the `tools.`-prefixed flavour.
These tests keep it that way:

1. test_no_bare_imports_of_tools_modules -- static AST lint: no file in
   the project may import a tools/ module by its bare name. Catches
   reintroduction at ANY import site, including function-level imports
   that a runtime check would only see if that code path ran.
2. test_no_same_file_duals_at_runtime -- imports the production entry
   modules, then asserts sys.modules holds no two names for one project
   file. Catches dynamic importers (importlib, pickle) that the static
   lint cannot see.

If the lint fails you almost certainly wrote `from wesnoth_sim import
...` (or similar); spell it `from tools.wesnoth_sim import ...`.
"""
import ast
import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
_ROOT = _TESTS.parent
for _p in (str(_ROOT), str(_ROOT / "tools"), str(_TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TOOLS_DIR = _ROOT / "tools"
# Directories whose sources must never bare-import a tools module.
# tmp_scratch/ is deliberately excluded (throwaway probes), as is
# wesnoth_src/ (vendored engine data, not ours).
LINTED_DIRS = ("tools", "tests", "wesnoth_ai", "benchmarks")


def _tools_module_names() -> set:
    return {p.stem for p in TOOLS_DIR.glob("*.py")}


def _linted_files():
    yield from _ROOT.glob("*.py")
    for d in LINTED_DIRS:
        base = _ROOT / d
        if base.is_dir():
            yield from base.glob("**/*.py")


def _bare_tool_imports(path: Path, tool_names: set):
    """Yield (lineno, text) for every bare import of a tools module."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                if top in tool_names:
                    yield node.lineno, f"import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.level:          # relative import -- not our concern
                continue
            mod = node.module or ""
            top = mod.split(".", 1)[0]
            if top in tool_names:
                yield node.lineno, f"from {mod} import ..."


def test_no_bare_imports_of_tools_modules():
    tool_names = _tools_module_names()
    assert tool_names, f"no modules found under {TOOLS_DIR}?"
    offenders = []
    for path in _linted_files():
        for lineno, text in _bare_tool_imports(path, tool_names):
            rel = path.relative_to(_ROOT)
            offenders.append(f"  {rel}:{lineno}: {text}")
    assert not offenders, (
        "Bare imports of tools/ modules create a SECOND module object "
        "next to the tools.-prefixed one (duplicated module state, "
        "divergent caches -- see this file's docstring). Use the "
        "tools.-prefixed form instead:\n" + "\n".join(sorted(offenders)))


def test_no_same_file_duals_at_runtime():
    """After importing the production entry modules, no project file may
    be present in sys.modules under two names."""
    # The production self-play graph plus the export/eval chains. These
    # are all import-safe (defs + sys.path bootstraps only).
    import tools.sim_self_play         # noqa: F401  learner entry
    import tools.selfplay_worker       # noqa: F401  worker entry
    import tools.mcts_policy           # noqa: F401  search stack
    import tools.validation_exports    # noqa: F401  export chain
    import tools.supervised_train      # noqa: F401  SL chain
    import tools.sim_demo_game         # noqa: F401  demo/export chain
    import tools.eval_vs_builtin       # noqa: F401  live-eval chain

    def _project_file(mod) -> str:
        f = getattr(mod, "__file__", None)
        if not f:
            return ""
        try:
            p = Path(f).resolve()
        except (OSError, ValueError):
            return ""
        for d in ("tools", "wesnoth_ai"):
            if (_ROOT / d) in p.parents:
                return str(p)
        return ""

    by_file = {}
    for name, mod in list(sys.modules.items()):
        if name == "__main__" or mod is None:
            continue
        f = _project_file(mod)
        if f:
            by_file.setdefault(f, set()).add(name)
    # A file is dual-imported iff two DISTINCT module objects share it;
    # deliberate aliases (sys.modules entries pointing at the SAME
    # object, e.g. sim_self_play's __main__ pin) are fine.
    dups = []
    for f, names in sorted(by_file.items()):
        objs = {id(sys.modules[n]) for n in names}
        if len(objs) > 1:
            dups.append(f"  {Path(f).relative_to(_ROOT)}: "
                        f"{sorted(names)}")
    assert not dups, (
        "One file, multiple live module objects -- module state is "
        "silently duplicated:\n" + "\n".join(dups))
