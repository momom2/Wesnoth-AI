#!/usr/bin/env python3
"""Does the corpus sweep see the hide-cover rule at all?

The 2026-09-13 root fix decides a hider's cover by the engine's
`[hides]` terrain globs (`terrain_resolver.hides_cover`) instead of a
hand-rolled defense-key table. `tools/diff_replay.py` over the corpus
came back clean after it -- and would have before it too, because the
replay format records no post-state: every check asks whether the next
recorded command's preconditions hold, and nothing reads a move's stop
reason or the uncovered-unit set, which is the only state this rule
moves. This tool measures that blindness instead of asserting it.

For a seeded sample of replays it reconstructs each one twice, once
under the engine rule and once under the OLD rule (put back here as
`old_defense_key_rule`, the same predicate over `_defense_keys_for_code`
the sim used before), and records per move where the walk stopped, why,
and which hiders it revealed. It reports how many sampled replays field
a hider at all, how many of those reconstruct DIFFERENTLY between the
rules, and how many divergences `diff_replay` reports under each -- the
three numbers the 2026-09-13 review quoted with no record.

    python tools/analysis/hider_rule_sample.py --dataset replays_dataset_imitation \\
        --sample 300 --seed 0 --workers 8 --json OUT.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from wesnoth_ai.rules import terrain_resolver  # noqa: E402

HIDE_ABILITIES = ("ambush", "concealment", "submerge", "nightstalk")
_OLD_KEY = {"ambush": "forest", "concealment": "village", "submerge": "deep_water"}
# The engine rule, captured at import so every spawned worker has it
# before anything is installed over `terrain_resolver.hides_cover`.
_ENGINE_RULE = terrain_resolver.hides_cover


def old_defense_key_rule(code: str, ability: str) -> bool:
    """The pre-2026-09-13 predicate: cover when the hex's DEFENSE keys
    (`replay_dataset._defense_keys_for_code`) name the terrain class."""
    from tools.replay_dataset import _defense_keys_for_code
    key = _OLD_KEY.get(ability)
    if key is None:
        return False
    keys = _defense_keys_for_code(code) if code else ["flat"]
    return key in keys


def _install(rule: str) -> None:
    """Point `terrain_resolver.hides_cover` at the rule under test. The
    predicate (`visibility._hide_cover_active`) and the core's map
    flags (`game_core.map_static`) both import it at call time."""
    terrain_resolver.hides_cover = old_defense_key_rule if rule == "old" else _ENGINE_RULE


def _trace(gz: Path) -> Dict[str, object]:
    """Reconstruct one replay under the rule currently installed: the
    per-move (index, stop reason, landing, uncovered set) trace, whether
    a hider ever stood on the board, and diff_replay's divergence count."""
    from tools.diff_replay import diff_replay
    from tools.replay_dataset import _apply_command, _build_initial_gamestate, _setup_scenario_events
    with gzip.open(gz, "rt", encoding="utf-8") as f:
        data = json.load(f)
    gs = _build_initial_gamestate(data)
    _setup_scenario_events(gs, data.get("scenario_id", ""))
    trace: List[tuple] = []
    hider = False
    for idx, cmd in enumerate(data.get("commands", [])):
        _apply_command(gs, list(cmd))
        if cmd[0] == "move":
            walk = getattr(gs.global_info, "_last_move_walk", None) or {}
            uncovered = tuple(sorted(getattr(gs.global_info, "_uncovered_units", None) or ()))
            trace.append((idx, walk.get("stop_reason"), tuple(walk.get("landed", ())), uncovered))
        if not hider and cmd[0] in ("init_side", "recruit"):
            hider = any(set(u.abilities or ()) & set(HIDE_ABILITIES) for u in gs.map.units)
    divergences = len(diff_replay(gz, stop_on_first=True))
    return {"trace": trace, "hider": hider, "divergences": divergences}


def _both(gz_str: str) -> Dict[str, object]:
    gz = Path(gz_str)
    out: Dict[str, object] = {"file": gz.name}
    try:
        _install("engine")
        new = _trace(gz)
        _install("old")
        old = _trace(gz)
    except Exception as e:  # noqa: BLE001 -- one bad file must not end the sample
        out["error"] = repr(e)
        return out
    out["hider"] = bool(new["hider"] or old["hider"])
    out["divergences_engine"] = new["divergences"]
    out["divergences_old"] = old["divergences"]
    first = None
    for a, b in zip(new["trace"], old["trace"]):
        if a != b:
            first = {"cmd": a[0], "engine": list(a[1:]), "old": list(b[1:])}
            break
    if first is None and len(new["trace"]) != len(old["trace"]):
        first = {"cmd": min(len(new["trace"]), len(old["trace"])), "engine": "length differs", "old": ""}
    out["differs"] = first is not None
    out["first_difference"] = first
    return out


def sample(dataset: Path, n: int, seed: int, workers: int) -> Dict[str, object]:
    files = sorted(dataset.glob("*.json.gz"))
    picked = random.Random(seed).sample(files, min(n, len(files)))
    with mp.Pool(workers) as pool:
        rows = pool.map(_both, [str(p) for p in picked], chunksize=4)
    ok = [r for r in rows if "error" not in r]
    hiders = [r for r in ok if r["hider"]]
    differ = [r for r in hiders if r["differs"]]
    summary = {
        "dataset": str(dataset), "seed": seed, "sampled": len(picked), "errors": len(rows) - len(ok),
        "with_a_hider": len(hiders),
        "hider_replays_reconstructing_differently": len(differ),
        "divergences_engine_rule": sum(r["divergences_engine"] for r in ok),
        "divergences_old_rule": sum(r["divergences_old"] for r in ok),
        "divergences_engine_rule_among_hiders": sum(r["divergences_engine"] for r in hiders),
        "divergences_old_rule_among_hiders": sum(r["divergences_old"] for r in hiders),
        "differing": [{"file": r["file"], **(r["first_difference"] or {})} for r in differ],
        "errors_detail": [r for r in rows if "error" in r][:10],
    }
    return summary


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("replays_dataset_imitation"))
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    out = sample(args.dataset, args.sample, args.seed, args.workers)
    for k in ("sampled", "errors", "with_a_hider", "hider_replays_reconstructing_differently",
              "divergences_engine_rule", "divergences_old_rule"):
        print(f"{k}: {out[k]}")
    for d in out["differing"][:20]:
        print("  ", d)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main(sys.argv[1:]))
