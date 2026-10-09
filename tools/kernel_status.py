#!/usr/bin/env python3
"""Whether the installed Rust wheel serves this source tree.

Every rule of the simulator runs in the Rust core (`wesnoth_core`,
adapter `wesnoth_ai/game_core.py`), so a wheel that is absent, or older
than the adapter's phase, stops the simulator at its first state. A
wheel that imports is not necessarily current: measured on this laptop
2026-09-13, a phase-3 wheel imported while the source declared phase 9.
This reports the installed phase against the source's and the adapter's,
asked through the same gate production uses (`game_core_class`).

Quickstart
----------
    python tools/kernel_status.py          # one line per check
    python tools/kernel_status.py --json

    from tools.kernel_status import banner
    log.info(banner())

Dependencies: stdlib; wesnoth_ai.game_core, imported lazily.
Dependents:   tools/sim_self_play.py's startup banner, the box scripts'
              setup records.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wesnoth_ai.paths import RUST_CORE_SRC_DIR  # noqa: E402


def wheel_phase() -> Optional[int]:
    """`__phase__` of the installed wheel, or None when absent."""
    try:
        import wesnoth_core
    except ImportError:
        return None
    return int(getattr(wesnoth_core, "__phase__", 0))


def source_phase() -> Optional[int]:
    """`__phase__` the Rust source declares, or None if unreadable."""
    import re
    src = RUST_CORE_SRC_DIR / "lib.rs"
    try:
        text = src.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r'__phase__[^0-9]{0,40}?(\d+)', text)
    return int(m.group(1)) if m else None


def adapter_phase() -> int:
    """The wheel phase `wesnoth_ai/game_core.py` requires."""
    from wesnoth_ai.game_core import _CORE_PHASE
    return int(_CORE_PHASE)


def _game_core() -> bool:
    # `game_core_class()` is the gate every CoreState passes through:
    # the wheel imports and is of the phase the adapter reads.
    from wesnoth_ai.game_core import game_core_class
    return game_core_class() is not None


def _current() -> bool:
    have, want = wheel_phase(), source_phase()
    return have is not None and want is not None and have >= want


# name -> the check. A check that raises counts as failed.
_GATES = {
    "GameCore": _game_core,
    "wheel_current": _current,
}


def kernel_status() -> Dict[str, bool]:
    """{check name: passed}. A check that raises counts as failed --
    that is what production would see."""
    out: Dict[str, bool] = {}
    for name, gate in _GATES.items():
        try:
            out[name] = bool(gate())
        except Exception:            # noqa: BLE001 -- absence is the answer
            out[name] = False
    return out


def banner() -> str:
    """One line naming the installed phase, the source's and whether the
    simulator can run. Safe to log at startup."""
    st = kernel_status()
    have, want = wheel_phase(), source_phase()
    if have is None:
        return "kernels | wesnoth_core NOT INSTALLED: the simulator cannot run (pip install ./rust/wesnoth_core)"
    parts = [f"wesnoth_core phase {have}", f"source {want}", f"adapter {adapter_phase()}"]
    if not st["GameCore"]:
        parts.append("TOO OLD for the adapter: the simulator cannot run -- REBUILD")
    elif not st["wheel_current"]:
        parts.append("behind the source -- REBUILD to test the source's Rust")
    else:
        parts.append("current")
    return "kernels | " + " | ".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("Quickstart")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    st = kernel_status()
    if args.json:
        print(json.dumps({"checks": st, "wheel_phase": wheel_phase(),
                          "source_phase": source_phase(), "adapter_phase": adapter_phase()},
                         indent=2))
        return 0
    for name, ok in st.items():
        print(f"{name:20s} {'yes' if ok else 'NO'}")
    print(banner())
    return 0


if __name__ == "__main__":
    sys.exit(main())
