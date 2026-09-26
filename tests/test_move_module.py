"""tools/dev/move_module.py on a small synthetic tree: a dry run changes
nothing and reports every site, --apply moves the module and every
importer still runs and computes what it did, an ambiguous origin is
refused with nothing changed."""
import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tools.dev.move_module import main

TREE = {
    "wesnoth_ai/__init__.py": "",
    "wesnoth_ai/sim/__init__.py": "",
    "wesnoth_ai/unpickle.py": "from typing import Dict\n\nMOVED_MODULES: Dict[str, str] = {}\n",
    "wesnoth_ai/helper.py": 'H = "helper"\n',
    "wesnoth_ai/rel_mod.py": """
        from .helper import H
        from . import helper

        VALUE = H + helper.H
        """,
    "wesnoth_ai/rel_user.py": """
        from .rel_mod import VALUE
        from . import rel_mod, helper

        RESULT = (VALUE, rel_mod.VALUE, helper.H)
        """,
    "tools/old_mod.py": """
        import logging
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        log = logging.getLogger(__name__)
        THING = 41


        def bump(x):
            return x + 1
        """,
    "tools/old_mod_extra.py": 'EXTRA = "a module whose name starts like the moving one"\n',
    "tools/other_mod.py": "V = 2\n",
    "tools/user_from.py": """
        from tools.old_mod import THING, bump  # noqa: E402
        from tools.old_mod_extra import EXTRA

        RESULT = (bump(THING), EXTRA)
        """,
    "tools/user_plain.py": """
        import tools.old_mod
        import tools.other_mod

        RESULT = tools.old_mod.bump(tools.other_mod.V)
        """,
    "tools/user_split.py": """
        from tools import old_mod, other_mod as om  # noqa: E402

        RESULT = old_mod.THING + om.V
        """,
    "tools/user_func.py": """
        def result():
            import tools.old_mod as m
            return m.THING


        RESULT = result()
        """,
    "tools/user_dynamic.py": """
        import importlib

        RESULT = importlib.import_module("tools.old_mod").THING
        """,
    "tools/user_lines.py": """
        from tools.old_mod import (THING,  # noqa: E402
                                   bump)
        from tools.old_mod import (
            THING as T,
        )

        RESULT = bump(THING) + T
        """,
    "tests/test_user.py": """
        import tools.old_mod as om


        def test_patches(monkeypatch):
            monkeypatch.setattr("tools.old_mod.THING", 1)
            monkeypatch.setattr(om, "THING", 2)
            monkeypatch.setattr(om, "NOT_THERE", 3, raising=False)
        """,
    "docs/notes.md": "See tools/old_mod.py and `tools.old_mod.bump`.\n",
    "scripts/run.sh": "python tools/old_mod.py\n",
    "tests/data/manifest.json": '{"reader": "tools/old_mod.py:bump"}\n',
}
USERS = ("tools.user_from", "tools.user_plain", "tools.user_split", "tools.user_func",
         "tools.user_dynamic", "tools.user_lines", "wesnoth_ai.rel_user")


def _write(root: Path, files: dict) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(textwrap.dedent(text).lstrip("\n").encode("utf-8"))


def _digest(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return h.hexdigest()


def _results(root: Path) -> dict:
    """Each user module's RESULT, imported in a fresh interpreter."""
    code = ("import importlib, json, sys\n"
            f"sys.path.insert(0, {str(root)!r})\n"
            f"print(json.dumps({{m: importlib.import_module(m).RESULT for m in {list(USERS)!r}}}))\n")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                         cwd=str(root))
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def _run(root: Path, capsys, *args: str):
    rc = main([*args, "--root", str(root)])
    return rc, capsys.readouterr().out


@pytest.fixture
def tree(tmp_path):
    _write(tmp_path, TREE)
    return tmp_path


def test_a_dry_run_changes_nothing_and_reports_every_site(tree, capsys):
    before = _digest(tree)
    rc, out = _run(tree, capsys, "tools.old_mod", "wesnoth_ai.sim.old_mod")
    assert rc == 0, out
    assert _digest(tree) == before, "a dry run must not touch the tree"
    for site in ("tools/user_from.py:1:", "tools/user_plain.py:1:", "tools/user_plain.py:4:",
                 "tools/user_split.py:1:", "tools/user_func.py:2:", "tools/user_dynamic.py:3:",
                 "tests/test_user.py:1:", "tests/test_user.py:5:"):
        assert site in out, f"no rewrite reported at {site}"
    assert "old_mod_extra" not in out.split("refusals")[0], "a longer module name was taken for it"
    assert "patches `THING` through the module object" in out
    assert "`NOT_THERE` is NOT bound at the module's top level" in out
    for own in ("__file__", "__name__", "sys.path"):
        assert f": {own}: " in out, f"the moved module's {own} line is not reported"
    for mention in ("docs/notes.md:1:", "scripts/run.sh:1:", "tests/data/manifest.json:1:"):
        assert mention in out, f"the mention at {mention} is not reported"


def test_apply_moves_the_module_and_every_importer_computes_what_it_did(tree, capsys):
    before = _results(tree)
    rc, out = _run(tree, capsys, "tools.old_mod", "wesnoth_ai.sim.old_mod", "--apply")
    assert rc == 0, out
    assert not (tree / "tools/old_mod.py").exists() and (tree / "wesnoth_ai/sim/old_mod.py").is_file()
    assert _results(tree) == before
    plain = (tree / "tools/user_plain.py").read_text(encoding="utf-8")
    assert "import wesnoth_ai.sim.old_mod\n" in plain
    assert "wesnoth_ai.sim.old_mod.bump(tools.other_mod.V)" in plain
    split = (tree / "tools/user_split.py").read_text(encoding="utf-8")
    assert split.startswith("from tools import other_mod as om  # noqa: E402\n"
                            "from wesnoth_ai.sim import old_mod  # noqa: E402\n")
    assert "from tools.old_mod_extra import EXTRA" in (tree / "tools/user_from.py").read_text(encoding="utf-8")
    lines = (tree / "tools/user_lines.py").read_text(encoding="utf-8")
    assert lines.startswith("from wesnoth_ai.sim.old_mod import (THING,  # noqa: E402\n"
                            "                                    bump)\n"
                            "from wesnoth_ai.sim.old_mod import (\n"
                            "    THING as T,\n"
                            ")\n"), "names aligned under the parenthesis stay under it; a hanging indent stays"
    patched = (tree / "tests/test_user.py").read_text(encoding="utf-8")
    assert 'monkeypatch.setattr("wesnoth_ai.sim.old_mod.THING", 1)' in patched
    table = (tree / "wesnoth_ai/unpickle.py").read_text(encoding="utf-8")
    assert '"tools.old_mod": "wesnoth_ai.sim.old_mod",' in table
    assert "tools/old_mod.py" in (tree / "docs/notes.md").read_text(encoding="utf-8"), \
        "mentions outside code are reported, never rewritten"


def test_a_module_s_relative_imports_are_made_absolute_and_its_relative_importers_follow(tree, capsys):
    before = _results(tree)
    rc, out = _run(tree, capsys, "wesnoth_ai.rel_mod", "wesnoth_ai.sim.rel_mod", "--apply")
    assert rc == 0, out
    moved = (tree / "wesnoth_ai/sim/rel_mod.py").read_text(encoding="utf-8")
    assert "from wesnoth_ai.helper import H\nfrom wesnoth_ai import helper\n" in moved
    user = (tree / "wesnoth_ai/rel_user.py").read_text(encoding="utf-8")
    assert "from wesnoth_ai.sim.rel_mod import VALUE\n" in user
    assert "from . import helper\nfrom wesnoth_ai.sim import rel_mod\n" in user
    assert _results(tree) == before


def test_both_names_of_a_tests_module_follow_it(tmp_path, capsys):
    _write(tmp_path, {
        "tests/kit/__init__.py": "",
        "tests/old_helper.py": "def f():\n    return 5\n",
        "tests/test_a.py": "from old_helper import f\n",
        "tests/test_b.py": "from tests.old_helper import f\n",
        "tests/test_c.py": "import old_helper\n\nX = old_helper.f\n",
        "wesnoth_ai/unpickle.py": "MOVED_MODULES = {}\n",
    })
    rc, out = _run(tmp_path, capsys, "old_helper", "kit.new_helper", "--apply")
    assert rc == 0, out
    assert (tmp_path / "tests/test_a.py").read_text(encoding="utf-8") == "from kit.new_helper import f\n"
    assert (tmp_path / "tests/test_b.py").read_text(encoding="utf-8") == "from tests.kit.new_helper import f\n"
    assert (tmp_path / "tests/test_c.py").read_text(encoding="utf-8").startswith(
        "import kit.new_helper as old_helper\n")
    assert (tmp_path / "wesnoth_ai/unpickle.py").read_text(encoding="utf-8") == "MOVED_MODULES = {}\n", \
        "nothing kept on disk pickles a test helper"


@pytest.mark.parametrize("name,text", [
    ("bare", "import old_mod\n"),
    ("rebound", "import tools.old_mod\n\ntools = None\n"),
    ("attribute", "import tools.other_mod\n\nX = tools.old_mod.THING\n"),
    ("binding", "import tools.old_mod\n\nY = tools.other_mod\n"),
])
def test_an_ambiguous_origin_is_refused_and_nothing_changes(tree, capsys, name, text):
    _write(tree, {f"tools/bad_{name}.py": text})
    before = _digest(tree)
    rc, out = _run(tree, capsys, "tools.old_mod", "wesnoth_ai.sim.old_mod", "--apply")
    assert rc == 2, out
    assert f"tools/bad_{name}.py:" in out.split("refusals")[1].split("patches through")[0]
    assert _digest(tree) == before
