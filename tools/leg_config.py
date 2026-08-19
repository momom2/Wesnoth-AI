"""Typed leg-launch configuration (user ruling 2026-08-19: guards
must be architectural -- error classes unrepresentable, not checked
by hand-maintained lists).

One JSON file (`$WORKDIR/leg.json`) declares a leg. Requiredness is
STRUCTURAL: a field of the schema without a default MUST appear in
every leg.json everywhere, enforced by this parser -- adding a new
per-leg decision means adding a schema field, and every future
launch on every box then refuses until the decision is declared.
Unknown keys are a hard error (the typo class is dead, not
detected). Declining an arm is the explicit literal "none", never
absence.

This supersedes the 2026-08-19 env-var preflight (which was itself
a hand-maintained list -- the guarded failure mode, one level up).
The launcher uses this file when present; the env-var path remains
only as the legacy fallback for boxes mid-leg at rollout.

CLI:
    python tools/leg_config.py validate leg.json
    python tools/leg_config.py export leg.json   # shell export lines
"""
from __future__ import annotations

import json
import sys
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class LegConfig:
    """The schema. Fields WITHOUT defaults are per-leg decisions:
    structurally required, decline with "none"."""
    # --- required decisions (no defaults, by design) ---------------
    campaign_file: str          # rolling checkpoint = leg identity
    seed_hf_file: str           # HF object seeding a fresh box | "none"
    probe_t0: str               # human-CE tripwire baseline | "none"
    policy_anchor: str          # F1 rehearsal cache path | "none"
    # --- optional ----------------------------------------------------
    # Escape hatch for the remaining env-driven knobs. Every entry is
    # echoed loudly at launch; a knob that stops being exceptional
    # should graduate to a schema field or a code default.
    extra_env: Optional[Dict[str, str]] = None

    # Mapping to the launcher's environment variables.
    _ENV_MAP = {
        "campaign_file": "CAMPAIGN_FILE",
        "seed_hf_file": "HF_SEED_FILE",
        "probe_t0": "PROBE_T0",
        "policy_anchor": "HUMAN_ANCHOR_POLICY_FILE",
    }


def required_fields() -> list:
    return [f.name for f in fields(LegConfig)
            if f.default is MISSING and f.default_factory is MISSING]


def load(path: Path) -> LegConfig:
    """Parse + validate. Raises ValueError with EVERY problem listed
    (not just the first) so one refusal round-trip fixes the file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("leg config must be a JSON object")
    known = {f.name for f in fields(LegConfig)}
    problems = []
    unknown = sorted(set(raw) - known)
    if unknown:
        problems.append(f"unknown keys (typo?): {unknown}; known: "
                        f"{sorted(known)}")
    for name in required_fields():
        if name not in raw:
            problems.append(
                f"required decision '{name}' is MISSING -- set a "
                f"value, or decline explicitly with \"none\"")
        elif not isinstance(raw.get(name), str) or not raw[name].strip():
            problems.append(f"'{name}' must be a non-empty string "
                            f"(the literal \"none\" declines)")
    ee = raw.get("extra_env")
    if ee is not None and (
            not isinstance(ee, dict)
            or not all(isinstance(k, str) and isinstance(v, str)
                       for k, v in ee.items())):
        problems.append("extra_env must be a {str: str} object")
    if problems:
        raise ValueError("leg config INVALID:\n  - "
                         + "\n  - ".join(problems))
    return LegConfig(**{k: raw[k] for k in raw})


def _sq(v: str) -> str:
    """Single-quote for shell, escaping embedded quotes."""
    return "'" + v.replace("'", "'\\''") + "'"


def export_lines(cfg: LegConfig) -> list:
    """Shell export lines. "none" -> empty string (the launcher's
    decline convention)."""
    out = []
    for field_name, env_name in LegConfig._ENV_MAP.items():
        v = getattr(cfg, field_name)
        v = "" if v == "none" else v
        out.append(f"export {env_name}={_sq(v)}")
    for k, v in (cfg.extra_env or {}).items():
        out.append(f"export {k}={_sq(v)}  # extra_env override")
    return out


def main(argv) -> int:
    if len(argv) != 3 or argv[1] not in ("validate", "export"):
        print(__doc__)
        return 2
    try:
        cfg = load(Path(argv[2]))
    except (ValueError, OSError, json.JSONDecodeError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    if argv[1] == "validate":
        print(f"leg config OK: campaign={cfg.campaign_file} "
              f"(required decisions: {', '.join(required_fields())})")
        return 0
    print("\n".join(export_lines(cfg)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
