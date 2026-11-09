"""Prereg departure 42's retry rule, automated (departure 43; the 2026-11-10 pre-launch review's M3).

Waits for round 19's fresh readout (round_19/play/readout.json, written after this watcher started and
after the failed attempt moved to play_failed42), reads I-V's v against recruit on the fresh games
alone and pooled with play_failed42, and writes rounds_local/STOP before round 19's step when the fresh
games alone read z below -2. rounds_local checks STOP before the pass and before the step. Exits once it
has read the readout, when round 19's step is done, or after 16 h. Log: wesnoth_runs/retry19_watch.log.
"""
import json
import math
import time
from pathlib import Path

RL = Path(r"C:\Users\momom2\wesnoth_runs\rounds_local")
LOG = Path(r"C:\Users\momom2\wesnoth_runs\retry19_watch.log")
FLOOR = -2.0
POLL_S = 30
LIFE_S = 16 * 3600


def log(line: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")


def v_recruit(path: Path):
    t = json.loads(path.read_text(encoding="utf-8"))["i_v"]["v"]["recruit"]
    return float(t["mean"]), float(t["se"])


def main() -> int:
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception:
        pass
    t0 = time.time()
    fresh, old = RL / "round_19" / "play" / "readout.json", RL / "round_19" / "play_failed42" / "readout.json"
    log(f"RETRY19_WATCH start: waiting for {fresh} (written after now), floor z {FLOOR}")
    while time.time() - t0 < LIFE_S:
        if (RL / "r19_step.done").exists():
            log("RETRY19_WATCH end: round 19's step is done before a fresh readout was read")
            return 1
        if old.exists() and fresh.exists() and fresh.stat().st_mtime > t0:
            try:
                m1, s1 = v_recruit(fresh)
                gates = json.loads(fresh.read_text(encoding="utf-8")).get("gates_pass")
            except (ValueError, KeyError, OSError):
                time.sleep(POLL_S)              # being written: read it again
                continue
            m0, s0 = v_recruit(old)
            w0, w1 = 1 / s0 ** 2, 1 / s1 ** 2
            pm, ps = (m0 * w0 + m1 * w1) / (w0 + w1), 1 / math.sqrt(w0 + w1)
            z1, zp = m1 / s1, pm / ps
            read = (f"v against recruit: fresh z {z1:+.2f} (mean {m1:.3g}, SE {s1:.3g}); pooled with play_failed42 "
                    f"z {zp:+.2f}; the fresh readout's gates pass: {gates}")
            if z1 < FLOOR:
                why = (f"STOP before round 19's step: departure 42's retry rule, {read}: below {FLOOR}, so the line "
                       f"stays stopped and the sampler is investigated")
                (RL / "STOP").write_text(why + "\n", encoding="utf-8")
                log(f"RETRY19_STOP {why}")
            else:
                log(f"RETRY19_PASS {read}: not below {FLOOR}")
            return 0
        time.sleep(POLL_S)
    log("RETRY19_WATCH end: no fresh readout within 16 h")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
