//! Embeds the hash of the source the module is built from as
//! `WESNOTH_CORE_SOURCE_HASH` (lib.rs exports it as `__source_hash__`),
//! so a wheel can be told from the source in front of it even when both
//! declare one phase (tools/kernel_status.py, `game_core.source_hash`).
//!
//! The hash: SHA-256 over Cargo.toml, build.rs and every .rs file under
//! src/, in the order of their paths relative to this directory ('/'
//! separated, compared bytewise); each contributes its path, a newline,
//! its length in bytes, a newline and its content, with CRLF line ends
//! read as LF. `wesnoth_ai/game_core.py` computes the same hash in Python.

use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

/// The .rs files under `dir`, as paths relative to the crate's root.
fn rust_sources(root: &Path, dir: &Path, out: &mut Vec<String>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("a directory entry").path();
        if path.is_dir() {
            rust_sources(root, &path, out);
        } else if path.extension().is_some_and(|x| x == "rs") {
            let rel = path.strip_prefix(root).expect("a path under the crate's root");
            let parts: Vec<String> = rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
            out.push(parts.join("/"));
        }
    }
}

/// `bytes` with each CRLF read as LF.
fn with_lf(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    for (i, &b) in bytes.iter().enumerate() {
        if !(b == b'\r' && bytes.get(i + 1) == Some(&b'\n')) {
            out.push(b);
        }
    }
    out
}

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = vec!["Cargo.toml".to_string(), "build.rs".to_string()];
    rust_sources(root, &root.join("src"), &mut files);
    files.sort();
    let mut hasher = Sha256::new();
    for rel in &files {
        let bytes = fs::read(root.join(rel)).unwrap_or_else(|e| panic!("cannot read {rel}: {e}"));
        let text = with_lf(&bytes);
        hasher.update(rel.as_bytes());
        hasher.update(b"\n");
        hasher.update(text.len().to_string().as_bytes());
        hasher.update(b"\n");
        hasher.update(&text);
    }
    let hex: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
    println!("cargo:rustc-env=WESNOTH_CORE_SOURCE_HASH={hex}");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=build.rs");
}
