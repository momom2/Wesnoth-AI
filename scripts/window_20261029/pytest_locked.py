"""Run pytest under a machine-wide lock, so parallel agents never run two
pytest processes at once (CLAUDE.md, Testing: "Never run more than one
pytest at a time").

    <python> C:/Users/momom2/wesnoth_runs/tools/pytest_locked.py [pytest args...]

Waits for the lock (printing once that it waits), runs `<python> -m pytest
<args>` in the current directory, releases the lock when pytest exits, and
exits with pytest's code. The lock is an exclusive byte-range lock on
C:/Users/momom2/wesnoth_runs/tools/pytest.lock, released by the OS if the
holder dies.
"""
import msvcrt
import os
import subprocess
import sys
import time

LOCK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pytest.lock")


def main() -> int:
    f = open(LOCK, "a+b")
    waited = False
    while True:
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            break
        except OSError:
            if not waited:
                print("pytest_locked: another pytest holds the lock; waiting", flush=True)
                waited = True
            time.sleep(2)
    try:
        return subprocess.call([sys.executable, "-m", "pytest"] + sys.argv[1:])
    finally:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        f.close()


if __name__ == "__main__":
    sys.exit(main())
