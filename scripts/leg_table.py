#!/usr/bin/env python3
"""Per-iteration leg table from a trainer-history CSV (arm VG3+).

    python scripts/leg_table.py path/to/trainer_history_local.csv

Rows with n_games != 24 (smoke / slow-tier test rows) are skipped.
Columns: the value trust-region controller (dv vs delta, lambda),
the paired-label estimates, the TCS gate passivity gauges, and
the real-state value health -- one line per iteration, ordered by
decision_step so relaunches (whose iter counters restart) read as
one series.
"""
from __future__ import annotations

import csv
import sys


def _f(r, k, w=6):
    v = r.get(k, "")
    try:
        return f"{float(v):.4g}"[:w].ljust(w)
    except (TypeError, ValueError):
        return str(v)[:w].ljust(w)


def main(argv) -> int:
    path = argv[1]
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if r.get("n_games") == "24"]
    rows.sort(key=lambda r: int(float(r.get("decision_step") or 0)))
    hdr = ("step      K   dv     lam    head-tr bias   s2     acc/pl "
           "short  et_tgt frCE   floor  auc21  W  L  D  atk%  et%")
    print(hdr)
    for r in rows:
        print(f"{_f(r,'decision_step',9)} {_f(r,'actions_per_turn_median',3)} "
              f"{_f(r,'sig_dv_consult_mean')} {_f(r,'trust_lambda')} "
              f"{_f(r,'consist_head_minus_truth',7)} {_f(r,'consist_bias_hat')} "
              f"{_f(r,'consist_sigma2_hat')} {_f(r,'tcs_accepts_per_plan')} "
              f"{_f(r,'tcs_gate_shorten_per_plan')} {_f(r,'distill_et_target')} "
              f"{_f(r,'fresh_value_ce')} {_f(r,'fresh_ce_floor')} "
              f"{_f(r,'fresh_auc_d21_30',5)} {_f(r,'s1_wins',2)} "
              f"{_f(r,'s2_wins',2)} {_f(r,'draws',2)} "
              f"{_f(r,'action_attack_pct',5)} {_f(r,'action_end_turn_pct',5)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
