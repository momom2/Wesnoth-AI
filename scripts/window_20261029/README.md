# The autonomous window's helpers (2026-10-29 to 2026-11-08)

Copies of the one-off helpers that ran from `C:/Users/momom2/wesnoth_runs/tools/`
during the window (docs/autonomy_20261029.md), committed for the record
(the 2026-11-02 21:45 review). The run uses the copies there; these are
snapshots as of 2026-11-02 23:12.

- `pytest_locked.py`: one pytest at a time on this machine.
- `relaunch_after_stop.py`: relaunches stage C's driver after a stop.
- `arm_round1.ps1`: round 1 alone (prereg departure 30).
- `arm_chain.ps1 -Head <commit>`: the chain with `--skip-p` (departure 31).
- `after_round1.py`: the waiter. After round 1 it applies departure 31's kill,
  then departure 32's GPU check, then the plant's discriminator on the GPU,
  then re-arms the chain.
- `shared_trunk_gpu_check.py`: departure 32's check. The heads-only direction
  is read at theta_k and two rungs on one trunk, against each read alone.
- `plant_discriminate.py`: the plant's miss (H1). The evaluator's change along
  d and along P'u, against the windowed and the full-memory gradients, on a
  subset of the readout.
- `window_check.py`: the pass's full-network gradient windowed at 16 decisions
  against one window per game-side, on the CPU (the 21:45 review, item 1).
  `--resplit` deals the games over the pass's 8 splits, and `--pool` adds earlier runs'
  saved splits to the cross-split reading (updated 2026-11-03 07:13).
- `after_stop.py` and `arm_after_stop.ps1`: prereg departure 33, amended (33a, 2026-11-03 10:27). If a
  design stop ends the rounds, round K0's end is stamped and the rounds go on heads_only
  (`chain_local --after-stop`); the paired arm reading was withdrawn.
- `after_stop_line.py` (with `after_stop_line_stubtest.py`, its stub test): prereg departure 35
  (2026-11-03 22:42). It watches the after-stop line. A REDEPLOY stop that names "after
  round K0" re-arms the line at --head. A later design stop at a claim round K, its final
  match done, stamps round K's end and re-arms the line rebased on K, its final match into
  `final_match_after_stop_kKK`. `arm_after_stop.ps1` gained `-FinalOut` and appends its
  console log.
- `window_end.py` (with `window_end_stubtest.py`, its stub test): docs/window_end_prereg_20261104.md
  (2026-11-04 10:37). Once the after-stop line has ended cleanly (LINE_DONE, its last
  final match returned 0), it confirms the best final match's candidate on fresh seeds
  (12,800 games from 980,000, `confirm_match/`), then plays that candidate's T=1 tail on
  round 3's tail seeds (`tail_match_window/`) unless the candidate is round 3.
