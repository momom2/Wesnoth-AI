"""Q8 CRN kill test (credit-assignment review 2026-08-17, $0 gate).

Event-keyed common-random-numbers for TCS would re-key each combat's
synced-RNG seed by the FIGHT'S IDENTITY instead of the global
request counter, so that when the search compares an incumbent turn
against a one-coordinate edit under the same salt, the UNEDITED
downstream fights reuse identical rolls (variance reduction in the
paired comparison). This probe measures whether there is anything
to key: across real incumbent/candidate pairs, how many downstream
RNG-consuming events even SHARE an identity after the edit?

PRE-REGISTERED DECISION RULE (design doc Q8): median count of
truly-shared downstream RNG events across pairs >= 1 keeps the CRN
family alive; median 0 kills it without writing any keying code.
Two identity notions reported:
  strict -- (attacker id, defender id, from-hex, to-hex, weapon):
            what honest event keying would use;
  loose  -- (attacker id, defender id) only: an upper bound.

Instrumentation: wraps WesnothSim.step (top-level command index +
pending fight identity) and WesnothSim._next_seed (the single
synced-RNG allocation point) -- no production code changes.
Materializations run skip_value=True: zero net forwards; the only
forwards are the spine recordings (~12/state).

Usage:
    python tools/crn_kill_probe.py [--states 25] [--pairs-per-state 3]
        [--checkpoint training/checkpoints/tier_b_tcs2_leg3_end.pt]
"""
from __future__ import annotations

import argparse
import logging
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("crn_kill_probe")

# Instrumentation state (single-threaded probe; plain globals).
_LOG = []            # [(cmd_idx, identity_or_None)]
_DEPTH = [0]
_CMD_IDX = [-1]
_PENDING = [None]


def _install():
    import tools.wesnoth_sim as ws
    orig_step = ws.WesnothSim.step
    orig_seed = ws.WesnothSim._next_seed

    def patched_step(self, action):
        top = _DEPTH[0] == 0
        if top:
            _CMD_IDX[0] += 1
            _PENDING[0] = None
            if isinstance(action, dict) \
                    and action.get("type") == "attack":
                sh = action.get("start_hex")
                th = action.get("target_hex")
                att = dfd = None
                if sh is not None and th is not None:
                    for u in self.gs.map.units:
                        if u.position.x == sh.x and u.position.y == sh.y:
                            att = u
                        elif (u.position.x == th.x
                              and u.position.y == th.y):
                            dfd = u
                _PENDING[0] = (
                    getattr(att, "id", "?"), getattr(dfd, "id", "?"),
                    (sh.x, sh.y) if sh is not None else None,
                    (th.x, th.y) if th is not None else None,
                    action.get("attack_index"))
        _DEPTH[0] += 1
        try:
            return orig_step(self, action)
        finally:
            _DEPTH[0] -= 1

    def patched_seed(self):
        _LOG.append((_CMD_IDX[0], _PENDING[0]))
        return orig_seed(self)

    ws.WesnothSim.step = patched_step
    ws.WesnothSim._next_seed = patched_seed
    return lambda: (setattr(ws.WesnothSim, "step", orig_step),
                    setattr(ws.WesnothSim, "_next_seed", orig_seed))


def _events_for(policy, sim, side, cmds, salt):
    from tools.turn_search import materialize
    _LOG.clear()
    _CMD_IDX[0] = -1
    _PENDING[0] = None
    m = materialize(policy, sim, side, cmds, salt, 0, skip_value=True)
    return None if m.invalid else list(_LOG)


def _matched(ev_inc, ev_cand, j_edit, loose: bool) -> int:
    def keys(evs):
        out = []
        for idx, ident in evs:
            if idx <= j_edit or ident is None:
                continue
            out.append((ident[0], ident[1]) if loose else ident)
        return Counter(out)
    a, b = keys(ev_inc), keys(ev_cand)
    return sum(min(a[k], b[k]) for k in a)


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=Path(
        "training/checkpoints/tier_b_tcs2_leg3_end.pt"))
    ap.add_argument("--dataset", type=Path,
                    default=Path("replays_dataset"))
    ap.add_argument("--states", type=int, default=25)
    ap.add_argument("--pairs-per-state", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    import torch
    from tools.midgame_starts import sample_midgame_start
    from tools.turn_search import (gumbel_top_k_alternatives,
                                   record_spine)
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.transformer_policy import TransformerPolicy

    raw = torch.load(args.checkpoint, map_location="cpu",
                     weights_only=False)
    a = raw["arch"]
    policy = TransformerPolicy(
        d_model=a["d_model"], num_layers=a["num_layers"],
        num_heads=a["num_heads"], d_ff=a["d_ff"],
        aux_score=bool(raw.get("aux_score")),
        moves_left=bool(raw.get("moves_left")))
    policy.load_checkpoint(args.checkpoint)

    rng_py = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)
    restore = _install()
    strict_counts, loose_counts = [], []
    n_down_inc = []
    tried = 0
    try:
        while len(strict_counts) < (args.states
                                    * args.pairs_per_state) \
                and tried < args.states * 8:
            tried += 1
            mg = sample_midgame_start(rng_py, args.dataset)
            if mg is None:
                continue
            gs, scen_id, cut_turn, begin_side, _prov = mg
            if cut_turn < 5:
                continue
            try:
                sim = WesnothSim(gs, scenario_id=scen_id,
                                 apply_scenario_events=False,
                                 begin_side=begin_side)
            except Exception:                           # noqa: BLE001
                continue
            side = sim.gs.global_info.current_side
            steps, _ = record_spine(policy, sim, side, 0, rng_np,
                                    max_spine=12)
            if len(steps) < 4:
                continue
            commands = [st.action for st in steps]
            # Edit coordinates spread over the early/mid turn (late
            # edits have no downstream by construction).
            js = sorted(rng_np.choice(
                max(1, len(steps) - 2),
                size=min(args.pairs_per_state, len(steps) - 2),
                replace=False).tolist())
            for j in js:
                st = steps[j]
                priors = np.array([x.prior for x in st.legal])
                et = next((i for i, x in enumerate(st.legal)
                           if x.action.get("type") == "end_turn"),
                          None)
                picks = gumbel_top_k_alternatives(
                    priors, st.action_idx, et, 2, rng_np)
                # Prefer a non-end_turn edit: an end_turn edit HAS
                # no downstream, and the CRN question is about
                # downstream fights.
                alt = next((i for i in picks
                            if st.legal[i].action.get("type")
                            != "end_turn"), None)
                if alt is None:
                    continue
                cand = (commands[:j] + [st.legal[alt].action]
                        + commands[j + 1:])
                salt = f"crn:{tried}:{j}"
                ev_i = _events_for(policy, sim, side, commands, salt)
                ev_c = _events_for(policy, sim, side, cand, salt)
                if ev_i is None or ev_c is None:
                    continue
                strict_counts.append(_matched(ev_i, ev_c, j, False))
                loose_counts.append(_matched(ev_i, ev_c, j, True))
                n_down_inc.append(sum(
                    1 for idx, k in ev_i
                    if idx > j and k is not None))
            if tried % 10 == 0:
                log.info(f"  {len(strict_counts)} pairs from "
                         f"{tried} sampled states")
    finally:
        restore()

    n = len(strict_counts)
    if n < 10:
        print(f"only {n} pairs -- inconclusive, need more states")
        return 2
    med_s = statistics.median(strict_counts)
    med_l = statistics.median(loose_counts)
    print(f"\npairs: {n}")
    print(f"downstream RNG events (incumbent): median "
          f"{statistics.median(n_down_inc)}, mean "
          f"{statistics.mean(n_down_inc):.2f}")
    print(f"matched downstream, STRICT key (ids+hexes+weapon): "
          f"median {med_s}, mean {statistics.mean(strict_counts):.2f},"
          f" frac>=1 {sum(1 for c in strict_counts if c >= 1) / n:.2f}")
    print(f"matched downstream, LOOSE key (ids only):          "
          f"median {med_l}, mean {statistics.mean(loose_counts):.2f},"
          f" frac>=1 {sum(1 for c in loose_counts if c >= 1) / n:.2f}")
    verdict = ("ALIVE (median strict matches >= 1)" if med_s >= 1
               else "DEAD (median strict matches = 0 -- edits change "
                    "which fights happen; event keying has nothing "
                    "to reuse)")
    print(f"\nPRE-REGISTERED VERDICT: CRN family {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
