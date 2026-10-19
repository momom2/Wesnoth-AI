"""A box's throughput, measured at its bring-up, so runs and benches pick
boxes by throughput per dollar.

    python tools/box_probe.py run --out probe.json --reference CKPT [--windows DIR] [--passes 2]
    python tools/box_probe.py cut-windows --sequences SEQ --dataset CORPUS --own-sequences OWN --out DIR
    python tools/box_probe.py rank --probes DIR OFFERS.json
    python scripts/rent_box.py rank [--by generation|training]  (the market's offers, searched and ranked)

run: the box's facts (CPU model and usable cores, memory, GPU, its PCIe
link, clocks and power limit, the torch build, the code and the wheel), then
`--passes` passes of each figure, probe.json rewritten after every pass:

  - generation: decisions a second of a self-play match of the reference
    against itself at the production generation configuration
    (GENERATION_FLAGS: 120 games in flight, six a worker, the compiled
    loop on the bf16 residual stream), read from the inference server's
    timeline (tools/serve_loop.py Timeline) over a steady window: from
    `--warm-s` after the first request, for `--window-s`; the match is
    stopped at the window's end. Every pass plays the same seeds. The
    server's device busy fraction and mean batch over the window, and the
    container's CPU cores used (cgroup cpu.stat), come with it;
  - training: positions a second of configuration B (CONFIG_B) over the
    fixed window set `--windows` (cut once by `cut-windows`), from
    tools/bench_sequence_step.py's synchronized total over `--train-windows`
    windows after `--train-warmup`; absent, with its reason, when no
    window set is given.

Each figure records its passes, their mean and their spread ((max - min) /
mean). A pass that fails records its reason and the next one runs.

cut-windows: the fixed window set from a trainer box's inputs: the first
`--corpus-games` training games of the corpus manifest and its first
`--holdout-games` holdout games, the first `--own-games` own games, each
game's record and columns copied and every manifest cut to them; the
trainer reads the directory as it reads the full inputs (DIR/sequences,
DIR/corpus/manifest.jsonl, DIR/own).

rank: candidate offers (Vast offer records, JSON) ranked by the expected
figure a dollar: the median of the recorded probes of boxes of the same GPU
and CPU model (names normalized: `gpu_key`, `cpu_key`), else of the same
GPU alone, marked.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("box_probe")

PROBE_VERSION = 1
# Round 5's fastest generation configuration (training/metrics/
# generation_speed_r5_20261015/: compile_bf16res_p120x6).
GENERATION_JOBS = 120
GENERATION_FLAGS = ["--worker-games", "6", "--min-free-mb", "300", "--compile-packed", "--bf16-residual"]
MATCH_FLAGS = ["--mcts-sims", "0", "--raw-temperature-a", "0", "--raw-temperature-b", "0",
               "--persistent-workers", "--shared-inference", "--no-infer-compile", "--device", "cuda"]
# No committed result carries a seed from 131000 to 131999 but the probes'.
SEED_BASE = 131000
TIMELINE_S = 5.0
# Configuration B (training/metrics/trainer_calibration_20261015/).
CONFIG_B = ["--packed-trunk", "--columns", "--window-batched", "--compiled-trunk", "--activation-budget-gb", "auto"]
NEVER = str(10 ** 12)                      # a position no probe reaches


# ---- the box's facts ------------------------------------------------------------

def _read(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _cpu_model() -> Optional[str]:
    text = _read("/proc/cpuinfo") or ""
    m = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else (platform.processor() or None)


def usable_cores() -> int:
    """The cores this process may run on, within the cgroup's quota."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    quota = (_read("/sys/fs/cgroup/cpu.max") or "").split()
    if len(quota) == 2 and quota[0].isdigit() and quota[1].isdigit() and int(quota[1]) > 0:
        n = min(n, -(-int(quota[0]) // int(quota[1])))
    return n


def _meminfo_mb(field: str) -> Optional[int]:
    m = re.search(rf"^{field}:\s*(\d+) kB", _read("/proc/meminfo") or "", re.MULTILINE)
    return int(m.group(1)) // 1024 if m else None


GPU_QUERY = ("name,memory.total,driver_version,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,"
             "pcie.link.width.current,clocks.max.sm,power.limit,power.max_limit")


def gpu_facts() -> Dict:
    """nvidia-smi's answers for the first GPU (GPU_QUERY), and torch's."""
    out: Dict = {}
    try:
        line = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
        if line:
            out.update(zip(GPU_QUERY.split(","), [v.strip() for v in line[0].split(",")]))
    except (OSError, subprocess.SubprocessError) as e:
        out["nvidia_smi_error"] = type(e).__name__
    try:
        import torch
        out["torch"] = torch.__version__
        out["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            out["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            out["torch_name"] = torch.cuda.get_device_name(0)
    except Exception as e:                      # noqa: BLE001 -- a fact we could not read stays absent
        out["torch_error"] = f"{type(e).__name__}: {e}"[:200]
    return out


def box_facts() -> Dict:
    facts = {"host": socket.gethostname(), "instance": os.environ.get("CONTAINER_ID"),
             "cpu_model": _cpu_model(), "cpus": os.cpu_count(), "usable_cores": usable_cores(),
             "mem_total_mb": _meminfo_mb("MemTotal"), "mem_available_mb": _meminfo_mb("MemAvailable"),
             "cgroup_memory_max": _read("/sys/fs/cgroup/memory.max"), "gpu": gpu_facts()}
    try:
        import wesnoth_ai
        facts["code"] = wesnoth_ai.__version__
    except Exception:                           # noqa: BLE001
        facts["code"] = None
    try:
        import wesnoth_core
        facts["wheel_phase"] = int(wesnoth_core.__phase__)
    except Exception:                           # noqa: BLE001
        facts["wheel_phase"] = None
    facts["gpu_key"] = gpu_key(facts["gpu"].get("name") or facts["gpu"].get("torch_name") or "")
    facts["cpu_key"] = cpu_key(facts["cpu_model"] or "")
    return facts


def gpu_key(name: str) -> str:
    """A GPU's name as nvidia-smi and Vast both write it: "NVIDIA GeForce RTX
    4090" and "RTX 4090" are "rtx 4090"."""
    s = re.sub(r"\b(nvidia|geforce)\b", " ", name.lower())
    return re.sub(r"[\s_]+", " ", s).strip()


def cpu_key(name: str) -> str:
    """A CPU's model name without trademarks, frequency or filler words."""
    s = re.sub(r"\((r|tm)\)", " ", name.lower())
    s = re.sub(r"@\s*[\d.]+\s*ghz", " ", s)
    s = re.sub(r"\b(processor|cpu|\d+-core)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---- the generation pass -----------------------------------------------------------

def steady_rate(rows: Sequence[Dict], warm_s: float, window_s: float) -> Optional[Dict]:
    """The server's figures over the window from `warm_s` after its first
    row with a request, for at least `window_s`, from its timeline rows
    (cumulative counters); None until the timeline covers the window."""
    first = next((r for r in rows if r["requests"] > 0), None)
    if first is None:
        return None
    start = next((r for r in rows if r["wall_s"] >= first["wall_s"] + warm_s), None)
    if start is None:
        return None
    end = next((r for r in rows if r["wall_s"] >= start["wall_s"] + window_s), None)
    if end is None:
        return None
    dt = end["wall_s"] - start["wall_s"]
    batches = end["batches"] - start["batches"]

    def per_batch(key, scale=1.0):
        return round(scale * (end[key] - start[key]) / batches, 3) if batches else None
    return {"decisions_per_s": round((end["requests"] - start["requests"]) / dt, 2),
            "window_s": round(dt, 2), "from_s": start["wall_s"],
            "device_busy": round((end["gpu_ms"] - start["gpu_ms"]) / 1e3 / dt, 4),
            "mean_batch": round((end["leaves"] - start["leaves"]) / batches, 2) if batches else None,
            "device_ms_per_batch": per_batch("gpu_ms"), "infer_ms_per_batch": per_batch("infer_s", 1e3),
            "idle_fraction": round((end["idle_s"] - start["idle_s"]) / dt, 4)}


def read_timeline(path: Path) -> List[Dict]:
    """The timeline's complete rows (a row being written is left out)."""
    rows = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            break
    return rows


def cgroup_cpu_seconds() -> Optional[float]:
    """The container's CPU seconds so far (cgroup v2 cpu.stat, else v1 cpuacct)."""
    m = re.search(r"^usage_usec (\d+)$", _read("/sys/fs/cgroup/cpu.stat") or "", re.MULTILINE)
    if m:
        return int(m.group(1)) / 1e6
    v1 = _read("/sys/fs/cgroup/cpuacct/cpuacct.usage")
    return int(v1) / 1e9 if v1 and v1.isdigit() else None


def side_flags(side: str, label: str, spec: Path) -> List[str]:
    """The reference's run_elo_batch flags for `side` under `label`, its
    checkpoint at `spec`."""
    from tools.reference_player import batch_flags
    flags = batch_flags(side)
    for i in range(len(flags) - 1):
        if flags[i] == f"--label-{side}":
            flags[i + 1] = label
        elif flags[i] == f"--spec-{side}":
            flags[i + 1] = str(spec)
    return flags


def generation_command(reference: Path, outdir: Path, jobs: int, budget_min: float) -> List[str]:
    return [sys.executable, str(ROOT / "tools" / "run_elo_batch.py"),
            *side_flags("a", "probe_a", reference), *side_flags("b", "probe_b", reference),
            "--outdir", str(outdir), "--games", "100000", "--max-extra-games", "0",
            "--seed-base", str(SEED_BASE), *MATCH_FLAGS, "--jobs", str(jobs), *GENERATION_FLAGS,
            "--inference-timeline-s", str(TIMELINE_S), "--time-budget-min", f"{budget_min:.2f}"]


def _stop_group(proc: subprocess.Popen, grace_s: float = 30.0) -> None:
    """TERM to the match's process group (the driver, its workers and its
    server), KILL after `grace_s`."""
    if proc.poll() is not None:
        return
    if hasattr(os, "killpg"):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=grace_s)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    else:
        proc.kill()
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass


def generation_pass(reference: Path, scratch: Path, k: int, opts) -> Dict:
    """One generation pass (module docstring)."""
    outdir = scratch / f"generation_{k}"
    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True)
    budget = (opts.warm_s + opts.window_s) / 60.0 + 2.0
    cmd = generation_command(reference, outdir, opts.jobs, budget)
    timeline = outdir / ".inference_server_0.timeline.jsonl"
    t0 = time.monotonic()
    with open(scratch / f"generation_{k}.log", "wb") as logf:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT,
                                start_new_session=True)
        result: Dict = {"pass": k}
        cpu = []
        try:
            deadline = t0 + opts.start_timeout_s + opts.warm_s + opts.window_s
            while True:
                rows = read_timeline(timeline)
                got = steady_rate(rows, opts.warm_s, opts.window_s)
                first = next((r for r in rows if r["requests"] > 0), None)
                if first is not None and len(cpu) == 0 and rows[-1]["wall_s"] >= first["wall_s"] + opts.warm_s:
                    cpu.append((time.monotonic(), cgroup_cpu_seconds()))
                if got is not None:
                    cpu.append((time.monotonic(), cgroup_cpu_seconds()))
                    result.update(got)
                    break
                if proc.poll() is not None:
                    result["failed"] = f"the match ended (rc {proc.returncode}) before the window closed"
                    break
                if time.monotonic() > deadline:
                    result["failed"] = "no steady window within the deadline"
                    break
                time.sleep(1.0)
        finally:
            _stop_group(proc)
    if len(cpu) == 2 and cpu[0][1] is not None and cpu[1][1] is not None and cpu[1][0] > cpu[0][0]:
        result["cpu_cores_used"] = round((cpu[1][1] - cpu[0][1]) / (cpu[1][0] - cpu[0][0]), 2)
    result["wall_s"] = round(time.monotonic() - t0, 1)
    return result


# ---- the training pass --------------------------------------------------------------

def training_command(windows: Path, reference: Path, rows: Path, out_pt: Path, opts,
                     train_flags: Sequence[str] = CONFIG_B, extra: Sequence[str] = ()) -> List[str]:
    return [sys.executable, str(ROOT / "tools" / "bench_sequence_step.py"), "run", "--out", str(rows),
            "--warmup", str(opts.train_warmup), "--windows", str(opts.train_windows),
            "--profile-windows", "0", "--smi-ms", "0", *extra, "--",
            "--sequences", str(windows / "sequences"), "--dataset", str(windows / "corpus"),
            "--own-sequences", str(windows / "own"), "--init-from", str(reference), "--init-other-records",
            "--out", str(out_pt), "--warmup-steps", "0", "--probe-every", NEVER, "--barrier-positions", NEVER,
            "--checkpoint-every", NEVER, "--signal-every", NEVER, *train_flags]


def training_pass(windows: Path, reference: Path, scratch: Path, k: int, opts) -> Dict:
    rows_path = scratch / f"training_{k}.jsonl"
    rows_path.unlink(missing_ok=True)
    cmd = training_command(windows, reference, rows_path, scratch / f"training_{k}.pt", opts)
    t0 = time.monotonic()
    result: Dict = {"pass": k}
    with open(scratch / f"training_{k}.log", "wb") as logf:
        try:
            rc = subprocess.run(cmd, cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT,
                                timeout=opts.train_timeout_s).returncode
        except subprocess.TimeoutExpired:
            rc = None
    rows = []
    if rows_path.exists():
        rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = next((r for r in rows if r.get("kind") == "total"), None)
    windows_s = [r["seconds"] for r in rows if r.get("kind") == "window"]
    if total is None:
        result["failed"] = "timed out" if rc is None else f"no total row (rc {rc})"
    else:
        result.update(positions_per_second=total["positions_per_second"], positions=total["positions"],
                      window_seconds_median=round(statistics.median(windows_s), 4) if windows_s else None)
    result["wall_s"] = round(time.monotonic() - t0, 1)
    return result


# ---- the record -------------------------------------------------------------------

def figure(passes: List[Dict], key: str) -> Dict:
    """A figure's passes, their mean and spread over the passes that read it."""
    values = [p[key] for p in passes if p.get(key)]
    out: Dict = {"passes": passes}
    if values:
        mean = statistics.fmean(values)
        out[key] = round(mean, 2)
        out["spread"] = round((max(values) - min(values)) / mean, 4) if len(values) > 1 else None
    return out


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def run(opts) -> int:
    out = Path(opts.out)
    scratch = Path(opts.scratch or out.parent / "probe_scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    record: Dict = {"probe_version": PROBE_VERSION, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "facts": box_facts(),
                    "config": {"generation": {"jobs": opts.jobs, "flags": GENERATION_FLAGS, "seed_base": SEED_BASE,
                                              "warm_s": opts.warm_s, "window_s": opts.window_s},
                               "training": {"flags": CONFIG_B, "windows": str(opts.windows) if opts.windows else None,
                                            "train_warmup": opts.train_warmup,
                                            "train_windows": opts.train_windows}}}
    write_json(out, record)
    gen: List[Dict] = []
    if not opts.skip_generation:
        for k in range(opts.passes):
            gen.append(generation_pass(Path(opts.reference), scratch, k, opts))
            log.info("generation pass %d: %s", k, gen[-1])
            record["generation"] = figure(gen, "decisions_per_s")
            write_json(out, record)
    train: List[Dict] = []
    if opts.windows is None:
        record["training"] = {"absent": "no window set (--windows; cut one with cut-windows)"}
    else:
        for k in range(opts.passes):
            train.append(training_pass(Path(opts.windows), Path(opts.reference), scratch, k, opts))
            log.info("training pass %d: %s", k, train[-1])
            record["training"] = figure(train, "positions_per_second")
            write_json(out, record)
    record["wall_s"] = round(time.monotonic() - t0, 1)
    write_json(out, record)
    print(json.dumps({"generation": (record.get("generation") or {}).get("decisions_per_s"),
                      "generation_spread": (record.get("generation") or {}).get("spread"),
                      "training": record["training"].get("positions_per_second"),
                      "training_spread": record["training"].get("spread"),
                      "gpu": record["facts"]["gpu_key"], "cpu": record["facts"]["cpu_key"],
                      "wall_s": record["wall_s"]}), flush=True)
    return 0


# ---- the fixed window set -------------------------------------------------------------

def _cut_sequences(src: Path, dst: Path, games: List[str]) -> None:
    """`games`' records and columns copied from `src` to `dst`, its manifests
    cut to them (their encoding fingerprints unchanged)."""
    from tools.columnize_sequences import COLUMNS_MANIFEST
    from tools.preencode_sequences import MANIFEST_NAME, record_path
    from wesnoth_ai.sequence_columns import columns_path
    dst.mkdir(parents=True, exist_ok=True)
    man = json.loads((src / MANIFEST_NAME).read_text(encoding="utf-8"))
    cols = json.loads((src / COLUMNS_MANIFEST).read_text(encoding="utf-8"))
    for g in games:
        for path in (record_path(src, g), columns_path(src, g)):
            shutil.copyfile(path, dst / path.name)
    keep = set(games)
    man["games"] = {g: c for g, c in man["games"].items() if g in keep}
    for key in ("n_games", "n_done"):
        if key in man:
            man[key] = len(man["games"])
    cols["games"] = ({g: c for g, c in cols["games"].items() if g in keep} if isinstance(cols["games"], dict)
                     else [g for g in cols["games"] if g in keep])
    if isinstance(cols["games"], dict) and "positions" in cols:
        cols["positions"] = sum(cols["games"].values())
    write_json(dst / MANIFEST_NAME, man)
    write_json(dst / COLUMNS_MANIFEST, cols)


def cut_windows(opts) -> int:
    """The fixed window set (module docstring)."""
    from tools.preencode_sequences import load_manifest
    out = Path(opts.out)
    seq_man, own_man = load_manifest(Path(opts.sequences)), load_manifest(Path(opts.own_sequences))
    rows = [json.loads(line) for line in (Path(opts.dataset) / "manifest.jsonl").read_text(encoding="utf-8")
            .splitlines() if line.strip()]
    rows = [r for r in rows if r["file"] in seq_man["games"]]
    train = [r for r in rows if not r.get("holdout")][:opts.corpus_games]
    held = [r for r in rows if r.get("holdout")][:opts.holdout_games]
    own = [g for g in sorted(own_man["games"]) if not own_man["games"][g].get("holdout")][:opts.own_games]
    own += [g for g in sorted(own_man["games"]) if own_man["games"][g].get("holdout")][:opts.holdout_games]
    (out / "corpus").mkdir(parents=True, exist_ok=True)
    (out / "corpus" / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train + held),
                                                   encoding="utf-8")
    _cut_sequences(Path(opts.sequences), out / "sequences", [r["file"] for r in train + held])
    _cut_sequences(Path(opts.own_sequences), out / "own", own)
    print(json.dumps({"corpus_games": len(train), "holdout_games": len(held), "own_games": len(own)}))
    return 0


# ---- the ranking ------------------------------------------------------------------

def load_probes(directory: Path) -> List[Dict]:
    """Every probe record under `directory` (recursively) that read a figure."""
    found = []
    for p in sorted(Path(directory).rglob("*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(rec, dict) and rec.get("probe_version") and "facts" in rec:
            rec["_path"] = str(p)
            found.append(rec)
    return found


def expected(probes: List[Dict], gpu: str, cpu: str, key: str) -> Optional[Dict]:
    """The median recorded `key` ("generation" or "training") of the probes
    of boxes of this GPU and CPU, else of this GPU alone (`match` says
    which), with how many probes it rests on."""
    field = {"generation": "decisions_per_s", "training": "positions_per_second"}[key]

    def values(match_cpu):
        return [p[key][field] for p in probes
                if (p.get(key) or {}).get(field) and p["facts"].get("gpu_key") == gpu
                and (not match_cpu or p["facts"].get("cpu_key") == cpu)]
    for match, vs in (("gpu+cpu", values(True)), ("gpu", values(False))):
        if vs:
            return {"value": statistics.median(vs), "probes": len(vs), "match": match}
    return None


def rank_offers(offers: List[Dict], probes: List[Dict], key: str = "generation") -> List[Dict]:
    """The offers with an expected figure, best figure a dollar first; each
    row the offer's id, price, GPU and CPU, the expected figure and its
    units per dollar spent (the figure an hour over the price an hour)."""
    rows = []
    for o in offers:
        gpu, cpu = gpu_key(o.get("gpu_name") or ""), cpu_key(o.get("cpu_name") or "")
        exp = expected(probes, gpu, cpu, key)
        dph = float(o.get("dph_total") or 0.0)
        if exp is None or dph <= 0:
            continue
        rows.append({"id": o.get("id"), "dph": round(dph, 3), "gpu": gpu, "cpu": cpu,
                     "cores": o.get("cpu_cores_effective"), "expected": round(exp["value"], 1),
                     "probes": exp["probes"], "match": exp["match"],
                     "per_dollar": round(exp["value"] * 3600.0 / dph)})
    return sorted(rows, key=lambda r: -r["per_dollar"])


def rank(opts) -> int:
    offers = json.loads(Path(opts.offers).read_text(encoding="utf-8"))
    for row in rank_offers(offers, load_probes(Path(opts.probes)), opts.by):
        print(json.dumps(row))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--reference", required=True, help="the reference checkpoint (tools/reference_player.py --path)")
    r.add_argument("--windows", default=None, help="the fixed window set (cut-windows); absent: no training figure")
    r.add_argument("--scratch", default=None, help="the passes' match directories and logs (default beside --out)")
    r.add_argument("--passes", type=int, default=2)
    r.add_argument("--jobs", type=int, default=GENERATION_JOBS)
    r.add_argument("--warm-s", type=float, default=20.0, help="the window starts this long after the first request")
    r.add_argument("--window-s", type=float, default=50.0)
    r.add_argument("--start-timeout-s", type=float, default=420.0,
                   help="the time a match may take to start serving (the compile included)")
    r.add_argument("--train-warmup", type=int, default=3)
    r.add_argument("--train-windows", type=int, default=12)
    r.add_argument("--train-timeout-s", type=float, default=600.0)
    r.add_argument("--skip-generation", action="store_true")
    c = sub.add_parser("cut-windows")
    c.add_argument("--sequences", required=True)
    c.add_argument("--dataset", required=True)
    c.add_argument("--own-sequences", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--corpus-games", type=int, default=48)
    c.add_argument("--own-games", type=int, default=32)
    c.add_argument("--holdout-games", type=int, default=2)
    k = sub.add_parser("rank")
    k.add_argument("offers", help="a JSON list of offer records")
    k.add_argument("--probes", default=str(ROOT / "training" / "metrics" / "box_probes"))
    k.add_argument("--by", choices=("generation", "training"), default="generation")
    ap.add_argument("--log-level", default="INFO")
    opts = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, opts.log_level), format="%(asctime)s %(levelname)s %(message)s")
    return {"run": run, "cut-windows": cut_windows, "rank": rank}[opts.cmd](opts)


if __name__ == "__main__":
    sys.exit(main())
