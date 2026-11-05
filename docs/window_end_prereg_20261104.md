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

Corrected by amendment 1 (below): the line ends after round 15 or 16, and the tail about Nov 8 04:00-08:00.

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

## Amendment 1 (2026-11-04 22:31, the 2026-11-04 21:43 review; before any step it changes has run)

*(Time corrected 2026-11-05 by the 09:43 review's L4: the heading read 22:35, but the commit is 22:31:33 and the watcher's restart 22:31:42, so it was written before the restart, not after.)*

The watcher was still waiting (the line runs round 7). It is restarted on the
amended script; nothing it started is lost.

- **The no-time end (the review's H1).**
  - The problem:
    - `after_stop_line.py` re-arms the line after a later design stop whenever at least 6 h remain (`LEFT_H`).
    - rounds_local's `no_time` needs 1.2 x the longest wall of the current |H|: about 7 h, and 11-13 h once round 9's claim wall counts.
    - Between the two, the re-armed rounds return ROUNDS_DONE at once (rc 0), and chain_local writes `CHAIN_NO_FINAL` (chain_local.py:258-261) and returns 0. The watcher then logs LINE_DONE.
    - Step 2 held on that end, since no CHAIN_FINAL_DONE follows the last CHAIN_DONE. The GPU would idle from the stop's final match on (about 5-15%: a PAUSE or a transfer stop at round 15).
  - The amendment: step 2 also goes on when the last CHAIN_DONE reads `the rounds returned 0` and is followed by CHAIN_NO_FINAL. The design stop's own final match, in the chain before, is then the last one, and step 3 chooses among the complete finals as before.
- **The extras (the review's L3: the GPU idles after the tail until the user returns).**
  - Step 6 runs after step 5, or after step 4 when the candidate is round 3, one match at a time.
  - Each extra runs only after the last match returned 0, and while no `window_end.STOP` file exists in the runs directory.
    - The file lets the user free the GPU without killing anything mid-match: the current match finishes, and no further one starts.
    - A match that returns non-zero ends the extras (WINDOW_END_HELD).
  - The extras, in order:
    1. **Round 3 on the confirmation's seeds,** if the candidate is not round 3: 12,800 decisive games from 980,000, into `confirm_match_round03/` (label `confirm_round3`). The candidate's gain over round 3 then reads paired on unselected seeds, beside the finals' paired +31.8 at 900,000.
    2. **Each adopted claim round after round 3 that no final match read** (6, 9, 12, ...; adopted as chain_local's `last_adopted` reads round.json), ascending. Each plays at the final matches' seeds: 12,800 decisive games from 900,000, into `final_match_roundKK/` (label `final_roundK`), paired with every final. They give the argmax learning curve that the T=1 claims cannot.
  - Each checkpoint's SHA-256 must equal its round.json's `candidate.sha256`, or the extras hold.
  - Each extra takes about 3.9 h. Today only round 6 would qualify; rounds 9, 12 and 15 join as they are adopted.
- **Seeds:** no new block. 980,000 is this pre-registration's; 900,000 is the final matches' (paired by design).
- **The kill:** as above, plus the STOP file.
- **Tests:** the stub test, now 14 cases (scripts/window_20261029/window_end_stubtest.py), all pass. The new cases:
  - the no-time end confirms;
  - CHAIN_NO_FINAL after a non-zero rounds code holds;
  - the extras' order after a tail;
  - the extras when the candidate is round 3;
  - the STOP file;
  - an extra's changed SHA holds;
  - a failed extra ends the extras.

  Read-only on the real files: the line is not yet done, the only final is round 3's at +31.8, rounds 1-6 are adopted, and round 6's SHA matches its round.json. The extras would play round 6 alone.

## Amendment 2 (2026-11-05 10:45, the 2026-11-05 09:43 review; the watcher is still waiting, so no step it changes has run)

The watcher is restarted on the amended script. Nothing it started is lost.

- **The no-time test, narrowed (the review's L1).**
  - The problem: amendment 1 accepted any `CHAIN_NO_FINAL` after `the rounds returned 0`. But chain_local writes two other CHAIN_NO_FINAL lines (chain_local.py:285-297, through `candidate`): the last round's candidate cannot be read, or its SHA-256 differs from its round.json's. The chain then returns 1, and the watcher would have confirmed the older finals and left the line's last rounds unread.
  - The amendment: step 2 now matches the text `CHAIN_NO_FINAL: no round after round` (chain_local.py:260), the no-time end alone. Any other CHAIN_NO_FINAL holds (WINDOW_END_HELD).
- **The extras' order (the review's L6).**
  - The adopted claim rounds that no final read now come first, latest first (15, 12, 9, 6), and round 3 on the confirmation's seeds last.
  - The reason: the argmax learning curve decides which round to adopt. The rounds nearest the candidate are its nearest rivals, so they matter most if the user returns before the queue empties. The 900,000 finals already pair round 3 with the line's last round.
  - Seeds, directories and labels are unchanged.
- **The wait, widened (the review's L2).** Step 1 also waits while `rounds_local.py`, `match_local.py` or `run_elo_batch.py` runs. A part of the line orphaned by a kill of chain_local alone would otherwise share the GPU with the confirmation.
- **Tests:** the stub test, now 16 cases, all pass.
  - Cases 10, 11, 13 and 14 take the new order.
  - New case 15: the two other CHAIN_NO_FINAL lines after `returned 0` hold.
  - New case 16: the wait covers each named script, an orphan included.
  - The test now imports the window_end.py beside it (the tracked copy). The deployed copy must be byte-equal (cmp).
- **Kill and holds:** as before.
