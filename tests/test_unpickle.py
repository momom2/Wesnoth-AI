"""wesnoth_ai/unpickle.py: a record pickled under a module's old path
loads after the module moved."""
import pickle
import sys
import types

import pytest

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
