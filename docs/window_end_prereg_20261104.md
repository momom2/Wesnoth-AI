# The window's end: a fresh-seed confirmation and a T=1 tail

Pre-registered 2026-11-04 10:37, during round 6's games, on the 2026-11-04
09:43 review's M1. The watcher is `C:/Users/momom2/wesnoth_runs/tools/window_end.py`,
with a tracked copy in `scripts/window_20261029/`.

## Why

- **The GPU would idle.** The after-stop line (docs/stage_c_rounds_local_prereg_20261030.md,
  departures 33-36) ends under `--end-by 2026-11-08 02:00`, after about
  round 17 at |H| 2,500 (about Nov 7 21:00). Its final match on the last
  adopted round follows (about 3.7 h). Nothing plays after that:
  `--after-stop` takes no tail or continuation (chain_local.py), and the line
  watcher exits on LINE_DONE. The GPU would idle from about Nov 8 00:40
  until the user returns.
- **Adoption needs an unselected reading.** Every final match plays the same
  seeds (base 900,000): round 3's read +31.8 +- 3.1. Each later stop's final
  match reads its round on those seeds too. The pairing makes them good for
  comparing candidates. But the best of several noisy readings overstates
  its candidate, and the user decides adoption. So the chosen candidate is
  read again on seeds no record has used.

## What runs

1. **Wait** until neither `chain_local.py` nor `after_stop_line.py` runs.
2. **Go on only on a clean end.** after_stop.log's last LINE line must be
   LINE_DONE, and chain.log's last CHAIN_DONE must be followed by
   `CHAIN_FINAL_DONE: match_local returned 0`. Otherwise WINDOW_END_HELD,
   and the heartbeat looks.
3. **The candidate.** Of the complete final matches at seed base 900,000
   with the procedure raw:t0+eo-1.5 on both sides (`final_match/`, and
   `final_match_after_stop*/` for the line), take the highest Elo against
   `own2`. Its checkpoint's SHA-256 must equal the one its final match
   recorded.
4. **The confirmation.**
   - `scripts/match_local.py` on the candidate, from the run tree at
     d045e17c, into `confirm_match/` (label `confirm_roundK`).
   - 12,800 decisive games from seed base 980,000. Both sides are raw
     players at argmax with the end_turn offset -1.5 and 64 memory slots;
     the reference is `own2` (configs/reference_player.json).
   - The standard local pool, 96 games in flight, served in fp16.
5. **The tail**, if the confirmation returned 0 and the candidate is not
   round 3 (round 3's tail is `tail_match/`).
   - `match_local --t1`, 12,800 games from seed base 950,000, into
     `tail_match_window/` (label `window_roundKt1`).
   - It shares round 3's tail's seeds, so the two pair.

## Seeds

- **Taken by this pre-registration:** 980,000-992,799. A capped game's
  replacements take seed + g x 1,000,000 (g = 1-9), so also 1,980,000-1,992,799
  through 9,980,000-9,992,799.
- **Taken elsewhere:**

  | Use | Seeds |
  |---|---|
  | Stage 1 | 800,000-848,199, and the same +g x 1M |
  | The final matches | 900,000-912,799, +g x 1M |
  | The tail | 950,000-962,799, +g x 1M |
  | The rounds | from 10M, in blocks of 1M |
  | P | 50M |
  | The link | 60M |
  | The cross-play matches | 70M |
  | The claims | 100M |
  | match_local's docstring | 600,000-899,999 and the pilots below 900,000 |

  No document or script on the stage1 branch names 980,000 (grepped
  2026-11-04).

## Serving: fp16, a caveat for the user

- Every match of the window serves in fp16 on the HIP kernels, the claims
  included, since D0 was read that way.
- Each round's pass measures fp16 against fp32 on its floor sample, at a KL
  of about 4e-6 a decision.
- An fp32 server was never run at match scale on this card, and an
  unattended night is the wrong place for a first. So the confirmation is
  fp16, and adopting from it carries that caveat. The box's standard
  serving (bf16/fp32) would be the user's check.

## Timeline (at departure 36's rounds)

| Step | Ends about |
|---|---|
| The line, ROUNDS_DONE after about round 17 | Nov 7 21:00 |
| The line's final match (3.7 h) | Nov 8 00:40 |
| The confirmation (12,800 games at argmax, 3.7 h) | Nov 8 04:25 |
| The tail (12,800 games at T=1, 2.7 h) | Nov 8 07:10 |

A design stop near the end, with under 6 h left, ends the line with
LINE_DONE after its final match: the same path.

## Readings and what they decide

- **The confirmation:** the candidate's PURE Elo +- SE against `own2`
  (tools/elo_collect.py, as match_local fits). It is reported beside the
  candidate's final-match reading; their difference is the selection's
  bias, plus noise.
- **The tail:** the candidate's T=1 Elo against `own2` at T=1, paired with
  round 3's +134.1 +- 3.3.
- **Nothing is decided automatically.** Adopting a new reference is the
  user's decision (configs/reference_player.json is not touched).

## Kill and holds

- **No clean end:** the watcher starts nothing (WINDOW_END_HELD).
- **A failed confirmation** (match_local not 0): no tail.
- **A display reset** (System event 4101) during a match: match_local
  writes STOP and exits 5. The match resumes from its games once the lead
  deletes STOP after finding the cause.
- **A stale checkpoint:** a SHA-256 that differs from the final's holds.

## Tests

The watcher was stub-tested with the processes, the matches and the
hashing stubbed, on seven cases:

- a later round reads best: its confirmation, then its tail;
- round 3 reads best: the confirmation alone;
- the line ends LINE_HELD;
- the last final match failed;
- the checkpoint changed;
- an incomplete final, and a final at other seeds, are skipped;
- a failed confirmation plays no tail.

On the real runs directory (read-only), it reads `final_match/` as round 3
at +31.8, the checkpoint's SHA-256 matches, and the line is not yet done.
