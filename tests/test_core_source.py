"""The hash of the Rust core's source (wesnoth_ai/core_source.py), which a
wheel carries as `__source_hash__` (rust/wesnoth_core/build.rs computes it;
CI checks that the two agree on the wheel it builds): every file the build
reads changes it, nothing else does, and the adapter warns about a wheel
built from another source without refusing it."""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai import core_source  # noqa: E402
from wesnoth_ai.core_source import RUST_CORE_DIR, source_hash  # noqa: E402


def _copy_crate(dst: Path) -> Path:
    """The crate's Cargo.toml, build.rs and src/ copied into `dst`."""
    dst.mkdir()
    for name in ("Cargo.toml", "build.rs"):
        shutil.copy2(RUST_CORE_DIR / name, dst / name)
    shutil.copytree(RUST_CORE_DIR / "src", dst / "src")
    return dst


def test_every_file_the_build_reads_changes_the_hash(tmp_path):
    crate = _copy_crate(tmp_path / "a")
    base = source_hash(crate)
    assert base is not None and base == source_hash(_copy_crate(tmp_path / "b")), "a copy elsewhere hashes alike"
    for rel in ("src/lib.rs", "src/core_label.rs", "Cargo.toml", "build.rs"):
        path = crate / rel
        text = path.read_bytes()
        path.write_bytes(text + b"\n// a change\n")
        assert source_hash(crate) != base, rel
        path.write_bytes(text)
        assert source_hash(crate) == base, rel
    (crate / "src" / "more").mkdir()
    (crate / "src" / "more" / "x.rs").write_text("// a new module\n")
    assert source_hash(crate) != base, "a new source file"


def test_line_ends_and_files_the_build_does_not_read_leave_it(tmp_path):
    crate = _copy_crate(tmp_path / "a")
    base = source_hash(crate)
    lib = crate / "src" / "lib.rs"
    lib.write_bytes(lib.read_bytes().replace(b"\n", b"\r\n"))
    (crate / "src" / "notes.md").write_text("not a source file\n")
    (crate / "Cargo.lock").write_text("# not hashed\n")
    assert source_hash(crate) == base
    assert source_hash(tmp_path / "nowhere") is None


def test_the_adapter_warns_about_a_wheel_built_from_another_source(monkeypatch, caplog):
    from wesnoth_ai import game_core
    monkeypatch.setattr(core_source, "source_hash", lambda: "a" * 64)
    caplog.set_level(logging.WARNING, logger="game_core")
    for wheel, warned in (("b" * 64, True), (None, True), ("a" * 64, False)):
        caplog.clear()
        monkeypatch.setattr(core_source, "wheel_source_hash", lambda w=wheel: w)
        game_core._warn_if_built_from_other_source()
        assert any("another source" in r.getMessage() for r in caplog.records) == warned, wheel
    caplog.clear()
    monkeypatch.setattr(core_source, "source_hash", lambda: None)       # an installed wheel alone
    game_core._warn_if_built_from_other_source()
    assert not caplog.records
