"""Per-test surface, time and trim candidates (docs/test_trim_20261006.md).

Reads the coverage database and the timing files the trim-measure workflow
uploads, and writes one row per test that ran:

  surface  the sum, over the function-body lines of project code the test
           executes (in its own process or in a child it starts), of
           1 / (number of tests executing that line). Lines executed
           outside every test (collection, imports) weigh 0: no deletion
           can lose them.
  time     setup + call + teardown, the mean over the timing samples
           (with --min-time, the fastest sample).
  ratio    surface / time.

The threshold is a quarter of the 20th percentile of the ratio over the
tests that ran. Elimination then removes the test with the lowest ratio
below the threshold, recomputes every remaining test's surface against the
suite that is left, and repeats until no remaining test is below it.
Tests listed in --keep (one node id per line, '#' starts a comment) are
never removed.

Usage:
  python .github/trim/surface_table.py --db data.sqlite [rust.sqlite] \
      --timing timing_1.json timing_2.json --out table.csv \
      [--keep keep.txt] [--lost lost.txt]
"""
import argparse
import ast
import csv
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

REPO = Path(__file__).resolve().parents[2]
CI_WORKSPACE = "/home/runner/work/Wesnoth-AI/Wesnoth-AI/"


def numbits_to_lines(blob: bytes) -> list[int]:
    """coverage.py's numbits: bit j of byte i stands for line 8 * i + j."""
    lines = []
    for index, byte in enumerate(blob):
        if byte:
            for bit in range(8):
                if byte & (1 << bit):
                    lines.append(8 * index + bit)
    return lines


class FunctionBodies:
    """Which lines of a source file lie in a function body, and in which
    function. Module and class bodies run once at import, so their lines
    say nothing about what a test exercises."""

    def __init__(self, source: str):
        self.owner: dict[int, str] = {}
        self._walk(ast.parse(source), prefix="")

    def _walk(self, node, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + child.name
                for stmt in child.body:
                    for line in range(stmt.lineno, stmt.end_lineno + 1):
                        self.owner[line] = name
                self._walk(child, prefix=name + ".")
            elif isinstance(child, ast.ClassDef):
                self._walk(child, prefix=prefix + child.name + ".")
            else:
                self._walk(child, prefix=prefix)


RUST_FN = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?(?:extern\s+\"C\"\s+)?fn\s+(\w+)")


def rust_owners(source: str) -> dict[int, str]:
    """Every line of a Rust file, named by the function it follows. The
    profile counts only executed code, so no line needs filtering out."""
    owner, current = {}, "<module>"
    for number, text in enumerate(source.splitlines(), 1):
        match = RUST_FN.match(text)
        if match:
            current = match.group(1)
        owner[number] = current
    return owner


def is_test_code(rel_path: str) -> bool:
    parts = rel_path.split("/")
    return ("tests" in parts[:-1] or parts[-1].startswith("test_")
            or parts[-1] == "conftest.py")


def load_owners(rel_path: str) -> dict[int, str] | None:
    """Line -> owning function for the lines that count as surface; an
    empty map for test code, which is not surface."""
    if is_test_code(rel_path):
        return {}
    path = REPO / rel_path
    try:
        source = path.read_text(encoding="utf-8")
        if rel_path.endswith(".rs"):
            return rust_owners(source)
        return FunctionBodies(source).owner
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None


def load_timing(paths: list[str], use_min: bool = False) -> dict[str, dict]:
    """Node id -> record with the mean (or the least) duration over the
    samples that ran it."""
    samples = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    tests: dict[str, dict] = {}
    for nodeid in samples[0]:
        runs = [s[nodeid] for s in samples if nodeid in s]
        totals = [r["setup"] + r["call"] + r["teardown"] for r in runs]
        first = runs[0]
        tests[nodeid] = {
            "slow": first["slow"],
            "outcomes": sorted({r["outcome"] for r in runs}),
            "time": min(totals) if use_min else sum(totals) / len(totals),
            "setup": sum(r["setup"] for r in runs) / len(runs),
            "call": sum(r["call"] for r in runs) / len(runs),
            "teardown": sum(r["teardown"] for r in runs) / len(runs),
            "samples": [round(t, 3) for t in totals],
            "shared_fixtures": sorted(
                n for n in first["shared_fixtures"]
                if n not in ("_game_records_in_tmp", "tmp_path_factory")),
        }
    return tests


def load_coverage(db_paths: list[str]):
    """Return (line keys, {context: set of line indices}, unreadable files)
    over every database, Python's and Rust's."""
    owners: dict[str, dict[int, str] | None] = {}
    line_index: dict[tuple[str, int], int] = {}
    keys: list[tuple[str, int, str]] = []
    covered: dict[str, set[int]] = defaultdict(set)
    for db_path in db_paths:
        con = sqlite3.connect(db_path)
        files = dict(con.execute("select id, path from file"))
        contexts = dict(con.execute("select id, context from context"))
        for file_id, context_id, blob in con.execute(
                "select file_id, context_id, numbits from line_bits"):
            path = files[file_id]
            rel = path[len(CI_WORKSPACE):] if path.startswith(CI_WORKSPACE) else path
            if rel not in owners:
                owners[rel] = load_owners(rel)
            owner_of = owners[rel]
            if owner_of is None:
                continue
            target = covered[contexts[context_id]]
            for line in numbits_to_lines(blob):
                owner = owner_of.get(line)
                if owner is None:
                    continue
                key = (rel, line)
                index = line_index.get(key)
                if index is None:
                    index = line_index[key] = len(keys)
                    keys.append((rel, line, owner))
                target.add(index)
        con.close()
    unreadable = sorted(rel for rel, owner_of in owners.items() if owner_of is None)
    return keys, covered, unreadable


def build_matrix(nodeids: list[str], covered: dict[str, set[int]], n_lines: int):
    rows, cols = [], []
    for row, nodeid in enumerate(nodeids):
        lines = covered.get(nodeid, ())
        rows.extend([row] * len(lines))
        cols.extend(lines)
    data = np.ones(len(rows), dtype=np.float64)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(nodeids), n_lines))


def surfaces(matrix, alive: np.ndarray, free: np.ndarray):
    """Each test's surface against the tests still alive, and each line's
    count of live tests covering it."""
    counts = matrix.T @ alive.astype(np.float64)
    weights = np.zeros_like(counts)
    paid = (counts > 0) & ~free
    weights[paid] = 1.0 / counts[paid]
    return matrix @ weights, counts


def eliminate(matrix, times, free, threshold, keep):
    alive = np.ones(matrix.shape[0], dtype=bool)
    removed = []
    while True:
        surface, counts = surfaces(matrix, alive, free)
        ratio = surface / times
        eligible = alive & ~keep & (ratio < threshold)
        if not eligible.any():
            return alive, removed
        index = int(np.flatnonzero(eligible)[np.argmin(ratio[eligible])])
        row = matrix.getrow(index).indices
        sole = [int(c) for c in row if counts[c] == 1 and not free[c]]
        removed.append((index, float(surface[index]), float(ratio[index]), sole))
        alive[index] = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", nargs="+", required=True,
                        help="coverage databases: Python's, and Rust's if measured")
    parser.add_argument("--timing", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--keep")
    parser.add_argument("--min-time", action="store_true",
                        help="a test's time is its fastest sample, not the mean")
    parser.add_argument("--lost", help="write the lines each removal loses")
    args = parser.parse_args()

    tests = load_timing(args.timing, use_min=args.min_time)
    keys, covered, unreadable = load_coverage(args.db)
    ran = [n for n, t in tests.items()
           if "skipped" not in t["outcomes"] and t["time"] > 0]
    times = np.array([tests[n]["time"] for n in ran])
    free = np.zeros(len(keys), dtype=bool)
    free[list(covered.get("", ()))] = True
    matrix = build_matrix(ran, covered, len(keys))

    keep_ids = set()
    if args.keep:
        for line in Path(args.keep).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                keep_ids.add(line)
    unknown_keeps = sorted(keep_ids - set(ran))
    keep = np.array([n in keep_ids for n in ran])

    all_alive = np.ones(len(ran), dtype=bool)
    surface0, counts0 = surfaces(matrix, all_alive, free)
    ratio0 = surface0 / times
    p20 = float(np.percentile(ratio0, 20))
    threshold = p20 / 4
    alive, removed = eliminate(matrix, times, free, threshold, keep)
    rank = {index: k for k, (index, _, _, _) in enumerate(removed)}
    at_removal = {index: (s, r, sole) for index, s, r, sole in removed}

    unique0 = np.asarray(matrix[:, (counts0 == 1) & ~free].sum(axis=1)).ravel()
    raw = np.asarray(matrix[:, ~free].sum(axis=1)).ravel()
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "nodeid", "slow", "outcomes", "time", "setup", "call", "teardown",
            "samples", "shared_fixtures", "lines", "unique_lines", "surface",
            "ratio", "one_shot_candidate", "removed_rank", "surface_at_removal",
            "ratio_at_removal", "lines_lost_at_removal", "kept_by_hand"])
        for i, nodeid in enumerate(ran):
            t = tests[nodeid]
            s, r, sole = at_removal.get(i, ("", "", ()))
            writer.writerow([
                nodeid, int(t["slow"]), "/".join(t["outcomes"]),
                f"{t['time']:.4f}", f"{t['setup']:.4f}", f"{t['call']:.4f}",
                f"{t['teardown']:.4f}", " ".join(map(str, t["samples"])),
                " ".join(t["shared_fixtures"]), int(raw[i]), int(unique0[i]),
                f"{surface0[i]:.4f}", f"{ratio0[i]:.6f}",
                int(ratio0[i] < threshold), rank.get(i, ""),
                f"{s:.4f}" if s != "" else "", f"{r:.6f}" if r != "" else "",
                len(sole) if i in at_removal else "", int(keep[i])])

    if args.lost:
        with open(args.lost, "w", encoding="utf-8") as fh:
            for index, s, r, sole in removed:
                fh.write(f"{ran[index]}  time {times[index]:.2f}s  "
                         f"surface {s:.2f}  lost {len(sole)} lines\n")
                by_function = defaultdict(list)
                for c in sole:
                    rel, line, owner = keys[c]
                    by_function[(rel, owner)].append(line)
                for (rel, owner), lines in sorted(by_function.items()):
                    fh.write(f"    {rel}::{owner}  {len(lines)} lines "
                             f"({min(lines)}-{max(lines)})\n")

    removed_time = float(times[~alive].sum())
    summary = {
        "tests_in_timing": len(tests),
        "tests_ran": len(ran),
        "tests_without_coverage_context": sum(1 for n in ran if n not in covered),
        "unreadable_files": unreadable,
        "function_body_lines_executed": len(keys),
        "lines_free_at_collection": int(free.sum()),
        "time_total_s": round(float(times.sum()), 1),
        "ratio_p20": p20,
        "threshold": threshold,
        "one_shot_candidates": int((ratio0 < threshold).sum()),
        "one_shot_candidate_time_s": round(float(times[ratio0 < threshold].sum()), 1),
        "eliminated": len(removed),
        "eliminated_time_s": round(removed_time, 1),
        "lines_lost": sum(len(sole) for _, _, _, sole in removed),
        "keeps": int(keep.sum()),
        "unknown_keeps": unknown_keeps,
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
