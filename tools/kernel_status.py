#!/usr/bin/env python3
"""Which Rust kernels are ACTUALLY live, not merely importable.

A wheel that imports is not a wheel that serves every kernel: each
kernel needs the wheel phase that gave it its current contract, and a
wheel several phases behind the source imports cleanly while the
kernels it is too old for fall back to Python. Measured on this laptop
2026-09-13: the wheel was phase 3 and imported, and four of the five
kernels then gated were Python.

Each kernel here is asked through the SAME gate production uses, so
this cannot drift from what actually runs.

Quickstart
----------
    python tools/kernel_status.py          # one line per kernel
    python tools/kernel_status.py --json

    from tools.kernel_status import banner
    log.info(banner())

Dependencies: stdlib; each gate's own module, imported lazily.
Dependents:   tools/sim_self_play.py's startup banner, and any launcher
              that wants the honest answer.
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


def _reach() -> bool:
    from tools.pathfind_sim import reach_kernel
    return reach_kernel() is not None


def _enumeration() -> bool:
    from tools.pathfind_sim import enumerate_kernel
    return enumerate_kernel() is not None


def _observe() -> bool:
    from wesnoth_ai import observe
    return observe.kernel() is not None


def _rows_from_reach() -> bool:
    from wesnoth_ai import observe
    return observe.kernel_rows_from_reach() is not None


def _combat() -> bool:
    from wesnoth_ai.combat import rust_combat_kernel
    return rust_combat_kernel() is not None


def _encode_streams() -> bool:
    from wesnoth_ai.encoder import _rust_encode_kernel
    return _rust_encode_kernel() is not None


def _game_core() -> bool:
    # `core_enabled()` is what WesnothSim actually consults, and it
    # honours WESNOTH_RUST_CORE as well as the phase gate -- unset is
    # the default and means OFF. Asking `game_core_class()` instead
    # (phase only) reports RUST for a kernel production is not using,
    # which is the exact failure this file exists to correct.
    from tools.wesnoth_sim import core_enabled
    return bool(core_enabled())


# name -> the production gate. Add a kernel here when you add a gate,
# or this file starts lying the way the banner did.
_GATES = {
    "reach": _reach,
    "enumeration": _enumeration,
    "observation": _observe,
    "rows_from_reach": _rows_from_reach,
    "combat": _combat,
    "encode_streams": _encode_streams,
    "GameCore": _game_core,
}


def kernel_status() -> Dict[str, bool]:
    """{kernel name: is the Rust one live}. A gate that raises counts
    as NOT live -- that is what production would see."""
    out: Dict[str, bool] = {}
    for name, gate in _GATES.items():
        try:
            out[name] = bool(gate())
        except Exception:            # noqa: BLE001 -- absence is the answer
            out[name] = False
    return out


def banner() -> str:
    """One line naming what is live and what is not, plus the phases
    when they disagree. Safe to log at startup."""
    st = kernel_status()
    live = [k for k, v in st.items() if v]
    py = [k for k, v in st.items() if not v]
    have, want = wheel_phase(), source_phase()
    parts = []
    parts.append("RUST: " + (", ".join(live) if live else "none"))
    parts.append("PYTHON: " + (", ".join(py) if py else "none"))
    if have is None:
        parts.append("wheel not installed")
    elif want and have < want:
        parts.append(f"wheel phase {have} < source {want} -- REBUILD to use "
                     f"the rest")
    elif have is not None:
        parts.append(f"wheel phase {have}")
    return "kernels | " + " | ".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("Quickstart")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    st = kernel_status()
    if args.json:
        print(json.dumps({"kernels": st, "wheel_phase": wheel_phase(),
                          "source_phase": source_phase()}, indent=2))
        return 0
    for name, on in st.items():
        print(f"{name:20s} {'RUST' if on else 'python'}")
    print(banner())
    return 0


if __name__ == "__main__":
    sys.exit(main())
