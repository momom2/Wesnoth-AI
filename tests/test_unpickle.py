"""wesnoth_ai/unpickle.py: a record pickled under a module's old path
loads after the module moved, through every loader of project pickles."""
import ast
import inspect
import pickle
import sys
import types
from pathlib import Path

import pytest

from helpers.source_tree import source_files
from wesnoth_ai import unpickle


def _home(name: str, monkeypatch) -> type:
    """A module `name` in sys.modules holding a class `Record`, the way
    a project module holds a dataclass: pickled by reference."""
    module = types.ModuleType(name)

    def __init__(self, value):
        self.value = value

    record = type("Record", (), {"__init__": __init__, "__module__": name})
    module.Record = record
    monkeypatch.setitem(sys.modules, name, module)
    return record


def _written_before_the_move(monkeypatch) -> bytes:
    """A pickle of a Record from zz_old_home, which then moves to
    zz_new_home: the old path no longer imports."""
    old = _home("zz_old_home", monkeypatch)
    blob = pickle.dumps([old(7), {"nested": old("x")}], protocol=pickle.HIGHEST_PROTOCOL)
    monkeypatch.delitem(sys.modules, "zz_old_home")
    _home("zz_new_home", monkeypatch)
    return blob


def test_a_record_written_under_an_old_module_path_loads_from_the_new_one(monkeypatch):
    blob = _written_before_the_move(monkeypatch)
    with pytest.raises(ModuleNotFoundError):
        unpickle.loads(blob)
    monkeypatch.setitem(unpickle.MOVED_MODULES, "zz_old_home", "zz_new_home")
    first, second = unpickle.loads(blob)
    new = sys.modules["zz_new_home"].Record
    assert type(first) is new and first.value == 7
    assert type(second["nested"]) is new and second["nested"].value == "x"


def test_a_module_moved_twice_loads_from_its_last_home(monkeypatch):
    blob = _written_before_the_move(monkeypatch)
    monkeypatch.setitem(unpickle.MOVED_MODULES, "zz_old_home", "zz_middle_home")
    monkeypatch.setitem(unpickle.MOVED_MODULES, "zz_middle_home", "zz_new_home")
    first, _ = unpickle.loads(blob)
    assert type(first) is sys.modules["zz_new_home"].Record
    monkeypatch.setitem(unpickle.MOVED_MODULES, "zz_new_home", "zz_old_home")
    with pytest.raises(ValueError, match="cycle"):
        unpickle.loads(blob)


def test_the_pre_encoded_corpus_reader_maps_old_paths(monkeypatch, tmp_path):
    from tools.preencode_corpus import read_record, write_record
    old = _home("zz_old_home", monkeypatch)
    path = tmp_path / "game.json.gz.pairs.zpkl"
    write_record(path, [(old(1), old(2))])
    monkeypatch.delitem(sys.modules, "zz_old_home")
    _home("zz_new_home", monkeypatch)
    monkeypatch.setitem(unpickle.MOVED_MODULES, "zz_old_home", "zz_new_home")
    (raw, labels), = read_record(path)
    assert (raw.value, labels.value) == (1, 2)
    assert type(raw) is sys.modules["zz_new_home"].Record


def _pickle_loads(path: Path):
    """(line, call) for every read of a pickle that bypasses the
    unpickler: pickle.load / loads / Unpickler, or those names imported
    from pickle or _pickle."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    readers = {"load", "loads", "Unpickler"}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and node.attr in readers
                and isinstance(node.value, ast.Name) and node.value.id in ("pickle", "_pickle")):
            yield node.lineno, ast.unparse(node)
        elif isinstance(node, ast.ImportFrom) and node.module in ("pickle", "_pickle"):
            for alias in node.names:
                if alias.name in readers:
                    yield node.lineno, f"from {node.module} import {alias.name}"


def test_every_project_pickle_is_read_through_the_unpickler():
    """The production trees hold no other reader of pickles, so none
    skips the table of moved modules."""
    home = Path(inspect.getsourcefile(unpickle)).resolve()
    files = source_files("wesnoth_ai", "tools", "scripts", "signal_profiler", "benchmarks")
    assert home in files, "the scan must reach the unpickler itself"
    assert list(_pickle_loads(home)), "the scan must see the unpickler's own pickle.Unpickler"
    bypass = [f"{path}:{line}: {text}" for path in files if path != home
              for line, text in _pickle_loads(path)]
    assert not bypass, ("read project pickles with wesnoth_ai.unpickle.load / loads, so a "
                        "record written before a module moved still loads:\n" + "\n".join(bypass))
