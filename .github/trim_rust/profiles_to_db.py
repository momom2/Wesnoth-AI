"""Turn the per-test LLVM profiles of the Rust core into a database shaped
like coverage.py's (tables file, context, line_bits with numbits), so the
trim's surface table reads Rust lines the way it reads Python lines.

Usage: python profiles_to_db.py --dir trim_out/rust --object <the .so> --out db
"""
import argparse
import concurrent.futures
import os
import sqlite3
import subprocess
from collections import defaultdict
from pathlib import Path

CRATE = "rust/wesnoth_core/"
TOOLS: dict[str, str] = {}


def llvm_tool(name: str) -> str:
    sysroot = subprocess.check_output(["rustc", "--print", "sysroot"], text=True).strip()
    found = sorted(Path(sysroot, "lib", "rustlib").glob(f"*/bin/{name}"))
    if not found:
        raise SystemExit(f"{name} not found under {sysroot}; install llvm-tools-preview")
    return str(found[0])


def nums_to_numbits(nums) -> bytes:
    numbits = bytearray(max(nums) // 8 + 1)
    for num in nums:
        numbits[num // 8] |= 1 << (num % 8)
    return bytes(numbits)


def executed_lines(profraws: list[Path], obj: str, scratch: Path) -> dict[str, set[int]]:
    """Lines of the crate with a nonzero count in the merged profiles."""
    profdata = scratch / (profraws[0].stem + ".profdata")
    merged = subprocess.run(
        [TOOLS["llvm-profdata"], "merge", "-sparse", "-failure-mode=all",
         *map(str, profraws), "-o", str(profdata)],
        capture_output=True, text=True)
    if merged.returncode != 0:
        print(f"merge failed for {profraws[0].name}: {merged.stderr[-300:]}")
        return {}
    exported = subprocess.run(
        [TOOLS["llvm-cov"], "export", "-format=lcov", f"-instr-profile={profdata}", obj],
        capture_output=True, text=True)
    profdata.unlink(missing_ok=True)
    if exported.returncode != 0:
        print(f"export failed for {profraws[0].name}: {exported.stderr[-300:]}")
        return {}
    lines: dict[str, set[int]] = defaultdict(set)
    current = None
    for row in exported.stdout.splitlines():
        if row.startswith("SF:"):
            current = crate_path(row[3:])
        elif row.startswith("DA:") and current is not None:
            line, count = row[3:].split(",")[:2]
            if int(count) > 0:
                lines[current].add(int(line))
    return lines


def crate_path(path: str) -> str | None:
    """The crate's own source files, relative to the repository; None for
    dependencies and the standard library."""
    at = path.find(CRATE + "src/")
    if at >= 0:
        return path[at:]
    if path.startswith("src/"):
        return CRATE + path
    return None


def group_profiles(directory: Path) -> dict[str, list[Path]]:
    """Context name -> its profile files. Collection and anything outside a
    test share the empty context."""
    names = {}
    tsv = directory / "tests.tsv"
    for row in tsv.read_text(encoding="utf-8").splitlines():
        stem, nodeid = row.split("\t", 1)
        names[stem] = nodeid
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(directory.glob("*.profraw")):
        stem = path.name.split("-", 1)[0].removesuffix(".profraw")
        groups[names.get(stem, "")].append(path)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dir", required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    for name in ("llvm-profdata", "llvm-cov"):
        TOOLS[name] = llvm_tool(name)

    directory = Path(args.dir)
    scratch = directory / "merged"
    scratch.mkdir(exist_ok=True)
    groups = group_profiles(directory)
    print(f"{len(groups)} contexts, {sum(len(v) for v in groups.values())} profiles")

    con = sqlite3.connect(args.out)
    con.executescript(
        "create table file (id integer primary key, path text unique);"
        "create table context (id integer primary key, context text unique);"
        "create table line_bits (file_id integer, context_id integer, numbits blob);")
    file_ids: dict[str, int] = {}
    workers = os.cpu_count() or 2
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(executed_lines, files, args.object, scratch): context
                   for context, files in groups.items()}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            context = futures[future]
            lines = future.result()
            context_id = con.execute(
                "insert into context (context) values (?)", (context,)).lastrowid
            for path, nums in lines.items():
                if path not in file_ids:
                    file_ids[path] = con.execute(
                        "insert into file (path) values (?)", (path,)).lastrowid
                con.execute("insert into line_bits values (?, ?, ?)",
                            (file_ids[path], context_id, nums_to_numbits(nums)))
            if done % 200 == 0:
                print(f"{done} of {len(futures)}", flush=True)
    con.commit()
    con.close()


if __name__ == "__main__":
    main()
