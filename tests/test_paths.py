"""wesnoth_ai/paths.py is the one place that locates the repo root.

A package module that computes a path from its own `__file__` points
somewhere else the day it moves one directory deeper, and nothing fails
at the move; these tests fail instead.
"""
import ast
import inspect
import subprocess
import sys
from pathlib import Path

from wesnoth_ai import paths

# Named locations that a clone does not hold (the corpora are not in git).
NOT_IN_GIT = {"IMITATION_DATASET_DIR"}

# The drivers that run without torch (docs/refactor_plan_20260925.md,
# "What a move must not break"); each imports paths.
TORCH_FREE_DRIVERS = ("tools.run_elo_batch", "tools.eval_procedure",
                      "tools.turn_search_config", "tools.host_resources")


def test_the_root_holds_the_package_and_every_named_location():
    here = Path(inspect.getsourcefile(paths)).resolve()
    assert paths.REPO_ROOT / "wesnoth_ai" / "paths.py" == here
    named = {name: value for name, value in vars(paths).items()
             if isinstance(value, Path) and name != "REPO_ROOT"}
    assert "UNIT_STATS_PATH" in named and "WESNOTH_SRC_DIR" in named
    missing = sorted(name for name, path in named.items()
                     if name not in NOT_IN_GIT and not path.exists())
    assert not missing, f"paths.py names locations that do not exist: {missing}"
    outside = sorted(name for name, path in named.items() if paths.REPO_ROOT not in path.parents)
    assert not outside, f"paths.py names locations outside the repo: {outside}"


def test_no_package_module_but_paths_reads_its_own_location():
    package = Path(inspect.getsourcefile(paths)).resolve().parent
    modules = sorted(p for p in package.rglob("*.py") if "__pycache__" not in p.parts)
    readers = []
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(isinstance(node, ast.Name) and node.id == "__file__" for node in ast.walk(tree)):
            readers.append(path.relative_to(package).as_posix())
    assert readers == ["paths.py"], (
        "a wesnoth_ai module computes a path from its own __file__; take the location "
        f"from wesnoth_ai.paths (add it there if it is new): {readers}")


def test_the_torch_free_drivers_import_without_torch():
    code = ("import sys\n"
            f"sys.path.insert(0, {str(paths.REPO_ROOT)!r})\n"
            "import wesnoth_ai.paths\n"
            + "".join(f"import {name}\n" for name in TORCH_FREE_DRIVERS)
            + "print('torch' in sys.modules)\n")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=120, cwd=str(paths.REPO_ROOT))
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip() == "False", f"a torch-free driver imports torch: {TORCH_FREE_DRIVERS}"
