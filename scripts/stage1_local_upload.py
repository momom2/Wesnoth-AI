"""Stage 1's local records to the model host, in the layout stage C restores
(scripts/stage_c_box.sh `restore_stage1`; docs/stage1_local_handoff_20261029.md).

    python scripts/stage1_local_upload.py --run C:/Users/<you>/wesnoth_runs/selfplay_t1_local
    python scripts/stage1_local_upload.py --run ... --upload

Without --upload it is a dry run (--dry-run, the default): it lists every
file it would send, with its size, its source and its destination, the
census of the games' match results and the checks that would stop an
upload, and it writes and sends nothing. --upload sends them through the
box library's uploader
(scripts/box/box_upload.py `Round`: three attempts a file, each under a
timeout, a file that landed unchanged never sent twice, every outcome a
line in upload.log) under huggingface_hub's own login on this machine: the
script never reads, prints or passes a token.

What goes up, each under its own name in --hf-dir of the model repository
(the names the box's stage 1 sends, scripts/selfplay_t1_box.sh):
- every regular file directly in the run directory: the walks, rows,
  labels, fits and readout, the D0 and memory fits, matches.txt, the logs,
  the done markers; but STOP, files being written (.tmp) and upload.log
  (the uploader's own);
- each games_<name> directory as games_<name>.tar.gz, its files under
  games_<name>/, hidden ones in and .tmp ones out (box_upload.py's tarball);
- <name>.inference_server_<k>.json and <name>.worker_stages.json, the
  server's and the workers' stats copied out of each games directory, as
  the box's play() copies them;
- plan.txt, which stage C reads first and the local runner does not
  write: "<games> <held out from> unknown", the box's size_plan line for a
  run that plays every planned game. Its third field, the first match's
  games an hour, is unread by stage C; the local first match's walls miss
  its interrupted attempts (docs/selfplay_t1_local_prereg_20261028.md,
  Incidents), so it is "unknown", as the box writes it when unread;
- local_provenance.json: where and how the games were served and the
  census of their results, for the reviewer (stage C does not read it);
- the done markers of the matches (chunk_sNN.done, measure_<name>.done)
  once the tarball each vouches for has landed, then upload.log, and
  ALL_DONE last, in a second round sent only when the first landed
  everything.

An upload is refused, nothing sent, unless the run is whole: the runner
ended (ALL_DONE, no STOP), every planned self-play match is done with its
games, walk and rows, the labels, fits and readout are there with the
readout's five gates passed (stage C refuses failed gates), the labels'
held-out split is the plan's, and every self-play result names `own2` at
64 slots on both seats under raw:t1, served in float16 alike. A folder
that already holds files on HF and that this staging directory did not
fill is refused too; the box stage 1's folder always is.

The staging directory (--staging, by default <run>_upload beside the run)
must be new, empty, or one this script filled: its record,
tmp/stage1_local_upload.json, names the files the script wrote there, and
only those are ever deleted (the ones a later upload no longer writes).
Any other existing directory is refused, and so is a staging directory
holding a file its record does not name, which the uploader would send.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import scripts.selfplay_t1_local as runner  # noqa: E402
from scripts.box import box_upload  # noqa: E402

DEFAULT_HF_DIR = "tier-b/selfplay_t1_local_20261028"
HF_DIR_PATTERN = re.compile(r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$")      # boxlib.sh's check of HF_DIR
NOT_SENT = {"STOP", box_upload.ALL_DONE, box_upload.LOG_NAME}
GENERATED = ("plan.txt", "local_provenance.json")
# The staging directory's record of the names this script wrote there
# (`stage`), under tmp/ (the uploader sends every regular file directly in
# the directory). Only the names it lists are ever deleted; an existing
# directory without it, or with a file it does not list, is refused.
STAGING_RECORD = Path("tmp") / "stage1_local_upload.json"
RECORD_SCRIPT = "scripts/stage1_local_upload.py"
# The self-play results' facts every game must share (scripts/selfplay_t1_local.py's sides).
SELFPLAY_PROCEDURE, SELFPLAY_SLOTS, SERVED = "raw:t1", 64, "float16"


@dataclass(frozen=True)
class Plan:
    """The self-play plan the run played (scripts/selfplay_t1_local.py)."""
    n_selfplay: int = runner.N_SELFPLAY
    chunk_games: int = runner.CHUNK_GAMES
    heldout_from: int = runner.HELDOUT_FROM
    seed_base: int = runner.SP_SEED_BASE

    def chunks(self) -> List[str]:
        return [f"s{k:02d}" for k in range((self.n_selfplay + self.chunk_games - 1) // self.chunk_games)]


# The runner's --smoke plan (scripts/selfplay_t1_local.py main): two matches of 24, the second held out.
SMOKE = Plan(n_selfplay=48, chunk_games=24, heldout_from=24, seed_base=873000)


@dataclass
class Entry:
    """One name in the HF folder. kind: file (sent from the run directory),
    copy (copied into the staging directory, then sent), text (written
    there), dir (a tarball of `source`); `deps` hold it until they land."""
    name: str
    kind: str
    source: Optional[Path] = None
    text: Optional[str] = None
    size: int = 0
    files: int = 1
    deps: List[str] = field(default_factory=list)


@dataclass
class Upload:
    entries: List[Entry]
    census: Dict
    blockers: List[str]
    skipped: List[str]


# ---------------------------------------------------------------------------
# The census of the match results
# ---------------------------------------------------------------------------

def serving(result: dict) -> str:
    """The precision the result's game was served in (tools/pg_positions.serving_precision)."""
    from tools.pg_positions import serving_precision
    return serving_precision(result)


def census_of(games: Path) -> Dict:
    """What the match results of one games directory say: their count, and
    the count of each value of the fields stage C and the reviewer read."""
    keys = Counter()
    count = unreadable = 0
    for p in sorted(games.glob("game_*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            unreadable += 1
            continue
        count += 1
        keys[json.dumps({"procedure": [r.get("procedure_a"), r.get("procedure_b")],
                         "memory": [r.get("memory_a"), r.get("memory_b")],
                         "checkpoint": sorted({str(r.get("checkpoint_sha256_a"))[:12],
                                               str(r.get("checkpoint_sha256_b"))[:12]}),
                         "served": serving(r), "packed": r.get("infer_packed_trunk"),
                         "decisions_logged": r.get("decisions_logged"),
                         "forced_faction": r.get("forced_faction"),
                         "code_version": r.get("code_version")}, sort_keys=True)] += 1
    return {"results": count, "unreadable": unreadable,
            "kinds": [{"games": n, **json.loads(k)} for k, n in keys.most_common()]}


def selfplay_problems(name: str, census: Dict, sha: str) -> List[str]:
    """Why a self-play match's results are not `own2`'s raw:t1 games at 64
    slots served in float16, all alike (stage C's refuse_foreign and the
    pass's floor read these)."""
    out = []
    if census["unreadable"]:
        out.append(f"games_{name}: {census['unreadable']} unreadable results")
    for kind in census["kinds"]:
        wrong = []
        if kind["procedure"] != [SELFPLAY_PROCEDURE] * 2:
            wrong.append(f"procedure {kind['procedure']}")
        if kind["memory"] != [SELFPLAY_SLOTS] * 2:
            wrong.append(f"memory {kind['memory']}")
        if kind["checkpoint"] != [sha[:12]]:
            wrong.append(f"checkpoint {kind['checkpoint']}")
        if kind["served"] != SERVED:
            wrong.append(f"served in {kind['served']}")
        if kind["decisions_logged"] is not True:
            wrong.append("no decision log")
        if wrong:
            out.append(f"games_{name}: {kind['games']} results with {', '.join(wrong)}")
    return out


def reference_sha() -> str:
    return json.loads((REPO / "configs" / "reference_player.json").read_text(encoding="utf-8"))["checkpoint_sha256"]


def box_stage1_hf_dir() -> Optional[str]:
    """The box stage 1's HF folder (scripts/selfplay_t1_box.sh's HF_DIR default)."""
    m = re.search(r'HF_DIR="\$\{HF_DIR:-([^}]+)\}"', (REPO / "scripts" / "selfplay_t1_box.sh").read_text(encoding="utf-8"))
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# What goes up
# ---------------------------------------------------------------------------

def _labels_split_problems(run: Path, plan: Plan) -> List[str]:
    """The labels' held-out flag against the plan's: a game is held out iff
    its index (rows.jsonl) is at least `heldout_from`."""
    index = {}
    for line in (run / "rows.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            index[r["game"]] = int(r["index"])
    wrong = 0
    for line in (run / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            wrong += (r["game"] not in index) or (bool(r["heldout"]) != (index[r["game"]] >= plan.heldout_from))
    return [f"labels.jsonl: {wrong} rows whose held-out flag is not the plan's (from {plan.heldout_from})"] if wrong \
        else []


def plan_upload(run: Path, plan: Plan = Plan(), sha: Optional[str] = None) -> Upload:
    """Every name the upload sends, the census, and what blocks an upload."""
    sha = sha or reference_sha()
    entries: List[Entry] = []
    skipped, blockers = [], []
    held = {}
    for name in plan.chunks():
        held[f"chunk_{name}.done"] = f"games_{name}.tar.gz"
    for name in ("d0", "memory"):
        held[f"measure_{name}.done"] = f"games_{name}.tar.gz"
    dirs = sorted(p for p in run.iterdir() if p.is_dir() and p.name.startswith("games_"))
    census = {}
    for d in dirs:
        files = [(full, rel) for full, rel in box_upload.dir_files(str(d))]
        entries.append(Entry(f"{d.name}.tar.gz", "dir", source=d, files=len(files),
                             size=sum(os.path.getsize(full) for full, _ in files)))
        stem = d.name[len("games_"):]
        for f in sorted(d.glob(".inference_server_*.json")) + sorted(d.glob(".worker_stages.json")):
            entries.append(Entry(f"{stem}.{f.name[1:]}", "copy", source=f, size=f.stat().st_size))
        census[d.name] = census_of(d)
    for p in sorted(run.iterdir()):
        if p.is_dir():
            if not p.name.startswith("games_"):
                skipped.append(f"{p.name}/ (a directory the box does not send)")
            continue
        if p.name in GENERATED:
            skipped.append(f"{p.name} (the one written at upload replaces it)")
            continue
        if p.name in NOT_SENT or p.name.endswith(".tmp"):
            if p.name != box_upload.ALL_DONE:          # sent last, below
                skipped.append(p.name)
            continue
        if p.name in held:
            entries.append(Entry(p.name, "copy", source=p, size=p.stat().st_size, deps=[held[p.name]]))
        else:
            entries.append(Entry(p.name, "file", source=p, size=p.stat().st_size))
    plan_line = f"{plan.n_selfplay} {plan.heldout_from} unknown\n"
    entries.append(Entry("plan.txt", "text", text=plan_line, size=len(plan_line)))
    provenance = json.dumps({
        "what": "stage 1 of docs/training_design_20261019/current_design.md, played on the local machine",
        "runner": "scripts/selfplay_t1_local.py (docs/selfplay_t1_local_prereg_20261028.md)",
        "served": "float16: run_elo_batch --infer-fp16, the packed trunk on the HIP kernels of "
                  "wesnoth_ai/hip_kernels.py, on the RX 6600; the games record infer_fp16 true",
        "plan": {"games": plan.n_selfplay, "chunk_games": plan.chunk_games, "heldout_from": plan.heldout_from,
                 "seed_base": plan.seed_base},
        "plan_txt": plan_line.strip(), "reference_sha256": sha, "census": census,
        "stage_c": {"STAGE1_HF_DIR": "this folder", "SP_SEED_BASE": plan.seed_base,
                    "CHUNK_GAMES": plan.chunk_games}}, indent=1) + "\n"
    entries.append(Entry("local_provenance.json", "text", text=provenance, size=len(provenance.encode())))
    if (run / box_upload.ALL_DONE).is_file():
        entries.append(Entry(box_upload.ALL_DONE, "copy", source=run / box_upload.ALL_DONE,
                             size=(run / box_upload.ALL_DONE).stat().st_size))

    # What stops an upload: a run that is not whole.
    if not (run / box_upload.ALL_DONE).is_file():
        blockers.append("ALL_DONE: the runner has not ended")
    if (run / "STOP").exists():
        blockers.append("STOP: the run was asked to stop")
    for name in plan.chunks():
        for need in (f"chunk_{name}.done", f"games_{name}", f"walk_{name}.jsonl", f"rows_{name}.jsonl"):
            if not (run / need).exists():
                blockers.append(f"{need}: missing")
        if (run / f"chunk_{name}.failed").exists():
            blockers.append(f"chunk_{name}.failed: the match failed")
        c = census.get(f"games_{name}")
        if c is not None:
            blockers.extend(selfplay_problems(name, c, sha))
    for need in ("rows.jsonl", "labels.jsonl", "fits.json", "readout.json", "readout.done"):
        if not (run / need).is_file():
            blockers.append(f"{need}: missing")
    if (run / "readout.json").is_file():
        words = runner.gates_of(run / "readout.json")
        if not runner.gates_passed(words):
            blockers.append(f"readout.json: the gates read '{words}' (stage C runs only on five passed gates)")
    if (run / "labels.jsonl").is_file() and (run / "rows.jsonl").is_file():
        blockers.extend(_labels_split_problems(run, plan))
    return Upload(entries, census, blockers, skipped)


# ---------------------------------------------------------------------------
# The dry run's listing, and the upload
# ---------------------------------------------------------------------------

def _size(n: int) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} B"


def listing(up: Upload, hf_dir: str, staging: Path, plan: Plan) -> str:
    dest = f"{box_upload.REPO}:{hf_dir}/"
    lines = [f"destination: {dest}", f"staging: {staging}", ""]
    width = max(len(e.name) for e in up.entries)
    order = sorted(up.entries, key=lambda e: (e.name == box_upload.ALL_DONE, bool(e.deps), e.kind == "dir", e.name))
    for e in order:
        if e.kind == "dir":
            what = f"{_size(e.size)} in {e.files} files, packed at upload (records already gzip), from {e.source}"
        elif e.kind == "text":
            what = f"{_size(e.size)}, written at upload" + (f": {e.text.strip()}" if e.name == "plan.txt" else "")
        else:
            what = f"{_size(e.size)} from {e.source}"
        if e.deps:
            what += f"; held until {', '.join(e.deps)} lands"
        if e.name == box_upload.ALL_DONE:
            what += "; last, once everything else landed"
        lines.append(f"  {e.name:<{width}}  {what}")
    lines.append(f"  {box_upload.LOG_NAME:<{width}}  the upload's own log, after the files")
    total = sum(e.size for e in up.entries)
    lines += ["", f"{len(up.entries) + 1} names, {_size(total)} before compression, to {dest}"]
    if up.skipped:
        lines.append(f"not sent: {', '.join(up.skipped)}")
    lines += ["", "census of the match results:"]
    for d, c in up.census.items():
        lines.append(f"  {d}: {c['results']} results" + (f", {c['unreadable']} unreadable" if c["unreadable"] else ""))
        for kind in c["kinds"]:
            lines.append(f"    {kind['games']}: {json.dumps({k: v for k, v in kind.items() if k != 'games'})}")
    lines.append("")
    if up.blockers:
        lines.append(f"NOT READY: an upload would be refused ({len(up.blockers)}):")
        lines += [f"  - {b}" for b in up.blockers]
    else:
        lines.append("READY: every check passed")
    lines += ["", "stage C on these records (docs/stage1_local_handoff_20261029.md):",
              f"  STAGE1_HF_DIR={hf_dir} SP_SEED_BASE={plan.seed_base} CHUNK_GAMES={plan.chunk_games}"]
    return "\n".join(lines)


class StagingError(RuntimeError):
    """The staging directory holds what this script did not write."""


def read_staging_record(staging: Path) -> Optional[Dict]:
    """The staging directory's record (STAGING_RECORD), or None when it has
    none or one this script did not write."""
    try:
        record = json.loads((staging / STAGING_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("script") != RECORD_SCRIPT \
            or not isinstance(record.get("written"), list):
        return None
    return record


def staging_problems(staging: Path) -> List[str]:
    """Why `staging` cannot take the upload: it exists, is not empty, and
    holds no record of this script; or it holds regular files the record
    does not name, which the uploader would send (it sends every regular
    file directly in its directory) and which this script never deletes.
    A new or empty directory takes it."""
    if not staging.exists():
        return []
    if not staging.is_dir():
        return [f"--staging {staging} is not a directory"]
    record = read_staging_record(staging)
    if record is None:
        if any(staging.iterdir()):
            return [f"--staging {staging} is not empty and holds no {STAGING_RECORD.as_posix()}: this script did "
                    f"not fill it, and nothing in it is touched; name a new or empty directory"]
        return []
    ours = set(record["written"]) | {box_upload.LOG_NAME}
    foreign = sorted(p.name for p in staging.iterdir() if p.is_file() and p.name not in ours)
    if foreign:
        return [f"--staging {staging} holds files this script did not write ({', '.join(foreign[:5])}"
                f"{', ...' if len(foreign) > 5 else ''}): the uploader would send them; move them out"]
    return []


def _write_record(staging: Path, record: Dict) -> None:
    path = staging / STAGING_RECORD
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def stage(up: Upload, staging: Path) -> Path:
    """The staging directory's files and the uploader's spec; its path. The
    run directory is only read: what is sent from it goes by path (`extra`
    and `dir` entries), and the copies, generated files and markers are
    written here. Refused (`StagingError`) on a directory `staging_problems`
    names; of the files there, only those the record lists and this upload
    no longer writes are deleted, before the record names what it writes."""
    problems = staging_problems(staging)
    if problems:
        raise StagingError("; ".join(problems))
    record = read_staging_record(staging) or {"script": RECORD_SCRIPT, "written": []}
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "tmp").mkdir(exist_ok=True)
    written = sorted({e.name for e in up.entries if e.kind in ("copy", "text")})
    for name in sorted(set(record["written"]) - set(written)):     # a former staging's; tmp/ keeps the landed
        p = staging / name
        if Path(name).name == name and p.is_file():                # a bare name directly here, nothing else
            p.unlink()
    _write_record(staging, {**record, "written": written})
    lines = []
    for e in up.entries:
        dst = staging / e.name
        if e.kind == "file":
            lines.append(f"extra\t{e.source}")
        elif e.kind == "dir":
            lines.append(f"dir\t{e.name[:-len('.tar.gz')]}\t{e.source}")
        elif e.kind == "copy":
            shutil.copy2(e.source, dst)  # its modification time too: an unchanged file is not sent again
        elif not dst.is_file() or dst.read_text(encoding="utf-8") != e.text:
            dst.write_text(e.text, encoding="utf-8")
        if e.deps:
            lines.append("\t".join(["hold", e.name, *e.deps]))
    spec = staging / "tmp" / "upload_spec.tsv"
    spec.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return spec


def foreign_files(api, hf_dir: str, staging: Path) -> List[str]:
    """The files of the HF folder this staging directory did not send: all
    of them when it never sent there, else those it has no record of (the
    uploader records no landing of upload.log and ALL_DONE)."""
    prefix = hf_dir.rstrip("/") + "/"
    on_hf = [f[len(prefix):] for f in api.list_repo_files(box_upload.REPO) if f.startswith(prefix)]
    landed = box_upload.load_state(staging, hf_dir)
    if not landed:
        return sorted(on_hf)
    return sorted(set(on_hf) - set(landed) - {box_upload.LOG_NAME, box_upload.ALL_DONE})


def upload(up: Upload, hf_dir: str, staging: Path, api, budget_s: float, min_rate: float) -> bool:
    """Two rounds of box_upload.Round: every file and tarball, the markers
    once their data landed and upload.log; then, only when all of it
    landed, ALL_DONE. True when everything landed."""
    spec = box_upload.read_spec(str(stage(up, staging)))
    timing = box_upload.Timing(min_rate=min_rate)
    deadline = time.monotonic() + budget_s
    first = box_upload.Round(staging, hf_dir, api, spec=spec, budget_s=budget_s, timing=timing)
    first.echo = True
    if not first.run(final=False):
        return False
    final = box_upload.Round(staging, hf_dir, api, spec=spec, budget_s=max(600.0, deadline - time.monotonic()),
                             timing=timing)
    final.echo = True
    return final.run(final=True)


def default_api():
    """huggingface_hub's client under its own stored login: no token here."""
    from huggingface_hub import HfApi
    return HfApi()


def main(argv: Optional[List[str]] = None, api_factory: Callable[[], object] = default_api,
         plan: Plan = Plan()) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="the local stage 1's output directory (only read)")
    ap.add_argument("--hf-dir", default=DEFAULT_HF_DIR, help="the folder in the model repository")
    ap.add_argument("--staging", type=Path, default=None,
                    help="where the copies, generated files, tarballs and upload state go "
                         "(default: <run>_upload beside the run directory)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="list, write and send nothing (the default)")
    mode.add_argument("--upload", action="store_true", help="send")
    ap.add_argument("--budget-h", type=float, default=8.0, help="the upload's time")
    ap.add_argument("--min-rate-kbs", type=float, default=100.0,
                    help="the slowest uplink an attempt waits for, kB/s (a file's timeout grows with its size)")
    ap.add_argument("--smoke", action="store_true", help="a run of the runner's --smoke plan (48 games from 873000)")
    args = ap.parse_args(argv)
    plan = SMOKE if args.smoke else plan
    run = args.run.resolve()
    staging = (args.staging or run.with_name(run.name + "_upload")).resolve()
    hf_dir = args.hf_dir.strip("/")
    if not run.is_dir():
        ap.error(f"--run {run} is not a directory")
    if not HF_DIR_PATTERN.match(hf_dir):
        ap.error(f"--hf-dir {args.hf_dir!r} is not a folder path")
    if hf_dir == box_stage1_hf_dir():
        ap.error(f"--hf-dir {hf_dir} is the box stage 1's folder: the local records go to a folder of their own")
    if staging == run or run in staging.parents:
        ap.error(f"--staging {staging} lies inside the run directory, which is only read")
    up = plan_upload(run, plan)
    print(listing(up, hf_dir, staging, plan))
    problems = staging_problems(staging)
    if problems:
        print("\nthe staging directory would be refused:")
        print("\n".join(f"  - {p}" for p in problems))
    if not args.upload:
        print("\ndry run: nothing written, nothing sent (--upload sends)")
        return 0
    if up.blockers or problems:
        print("\nREFUSED: nothing sent")
        return 2
    api = api_factory()
    foreign = foreign_files(api, hf_dir, staging)
    if foreign:
        print(f"\nREFUSED: {box_upload.REPO}:{hf_dir}/ already holds {len(foreign)} files this staging "
              f"directory did not send ({', '.join(foreign[:5])}): pick another --hf-dir")
        return 2
    ok = upload(up, hf_dir, staging, api, args.budget_h * 3600.0, args.min_rate_kbs * 1000.0)
    print(f"\n{'UPLOADED' if ok else 'INCOMPLETE: run it again to send what did not land'} "
          f"({staging / box_upload.LOG_NAME})")
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)                   # an upload thread hung on a socket must not hold the exit (box_upload.py)
