"""Give the CI-only coverage build of wesnoth_core a per-test dump.

Appends a function that writes the LLVM profile counters gathered since
its last call to a file and zeroes them, and registers it in the module.
The counters exist only in a build with `-C instrument-coverage`, which
links the profiler runtime that defines these symbols.
"""
from pathlib import Path

LIB = Path(__file__).resolve().parents[2] / "rust" / "wesnoth_core" / "src" / "lib.rs"

MODULE = '''
mod trim_coverage {
    use pyo3::prelude::*;
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int};

    extern "C" {
        fn __llvm_profile_set_filename(name: *const c_char);
        fn __llvm_profile_write_file() -> c_int;
        fn __llvm_profile_reset_counters();
    }

    /// Write the counters gathered since the last call to `path`, zero
    /// them, and point the write at process exit to `after`.
    #[pyfunction]
    pub fn trim_coverage_dump(path: &str, after: &str) -> PyResult<i32> {
        let path = CString::new(path)?;
        let after = CString::new(after)?;
        let written = unsafe {
            __llvm_profile_set_filename(path.as_ptr());
            let written = __llvm_profile_write_file();
            __llvm_profile_reset_counters();
            __llvm_profile_set_filename(after.as_ptr());
            written
        };
        Ok(written)
    }
}
'''

REGISTER = "    m.add_function(wrap_pyfunction!(trim_coverage::trim_coverage_dump, m)?)?;\n"
ANCHOR = '    m.add("__phase__"'


def main() -> None:
    text = LIB.read_text(encoding="utf-8")
    assert text.count(ANCHOR) == 1, "the module init's __phase__ line moved"
    text = text.replace(ANCHOR, REGISTER + ANCHOR) + MODULE
    LIB.write_text(text, encoding="utf-8")
    print(f"patched {LIB}")


if __name__ == "__main__":
    main()
