# Project review — bugs and improvements

Generated 2026-04-28 from a deep review of every major component.
Refreshed 2026-04-29 / 2026-05-02 / 2026-05-03 / 2026-05-04 /
2026-05-10. Items are graded by impact on the project's stated
goals (superhuman play via MCTS+self-play; readable/customizable
strategies; cluster economy).

## Current state at a glance (2026-05-11)

- **Simulator is the production training path.** `tools/wesnoth_sim.py`
  is bit-exact for combat (731/731 strikes verified vs `[mp_checkup]`
  oracle). Full-replay `diff_replay --filter-2p` clean rate
  **100% (5,484/5,484)** on the freshly extracted competitive-2p
  corpus (2026-05-11 sweep). The 16 → 0 reduction since 2026-05-10
  came from chasing specific cases one by one — multi-tier
  advancement, surrender-drop heuristic refinement to handle
  takeover-controller `skip_sighted="all"` false-positives,
  suffix-redo dedup for save-mid-move replays (Fallenstar Lake),
  partial-checkup-attack drop for save-mid-attack replays (Ruined
  Passage), and seven quarantines for replays a human cannot
  verifiably reconstruct from (3 corrupted including Sablestone
  Delta + Hamlets t67, 1 debug-tool-using, 3 Dunefolk-player-faction
  replays the classifier missed). No residual divergences.
- **Self-play training pipeline ready.** `tools/sim_self_play.py`
  drives REINFORCE+baseline by default, AlphaZero-style MCTS via
  `--mcts`. Cluster job + GUI controls in place.
- **Distributional value head + cliffness landed 2026-05-10.**
  K=51 atom C51 head over [-1, +1]. Cliffness signal published.
  Bayesian-precision bootstrap weighting + adaptive sim budget
  framework wired but default OFF pending calibration.
- **Live Wesnoth IPC** retained for `--display` and
  `tools/eval_vs_builtin.py`; not used for training.

> **2026-05-10 finding: dataset was stale (now refreshed).** The
> May-4 Stages 1–21 sweep brought combat/scenario fidelity to
> bit-exact; `replay_extract.py` was further fixed on 2026-05-08
> to concatenate all `[replay]` blocks for snapshot saves. The
> dataset's `.json.gz` files were last regenerated 2026-05-02 and
> were stale relative to both fixes. Investigation flow:
>
> 1. Ran `diff_replay --filter-2p` on stale `replays_dataset/`
>    (3941 replays) → 44 divergences across 34 unique failing
>    replays. ALL 34 were mid-game snapshot saves.
> 2. Surgical re-extract of 28 of those 34 (6 source bz2s
>    missing — moved to set-aside / mod purge) → 28/28 clean.
>    Confirmed extraction bug, not sim bug.
> 3. Built `tools/sort_replays.py` (parallel extract + classify
>    + bucket) and ran it on FULL `replays_raw/` (45,971) +
>    `replays_raw_set_aside/` (28,477) = 74,448 inputs. Sort:
>      - `replays_dataset/` (competitive_2p): **6,224**
>      - `replays_dataset_quarantine/modded/<mod>/`: 24,533
>          - plan_unit_advance: 24,327 (UI mod, set-aside)
>          - Color_Modification: 172 (cosmetic)
>          - Rav_Color_Mod: 34 (cosmetic)
>      - `replays_dataset_quarantine/non_2p/`: 37,110
>      - `replays_dataset_quarantine/non_vanilla_map/`: 887
>      - extract_failed (no payload written): 5,694
> 4. Re-ran `diff_replay --filter-2p` on the new
>    `replays_dataset/`: **5,467/5,490 clean = 99.58%**. Down
>    from 44 divergences to 23 — and the residuals are now real
>    sim signal rather than extraction noise.
>
> Old `replays_dataset/` archived to
> `replays_dataset.old.pre-2026-05-10/` for one-version rollback.

- [x] 🟠 **Re-extract `replays_dataset/` against current
  `replay_extract.py`** (DONE 2026-05-10). See block above.

- [x] 🟡 **Investigated 3× recruit:target_occupied at (11,5)** (DONE
  2026-05-10, commits 71fb35c + e8c677a). Three Hamlets replays
  all failing the same way at cmd[6] turn=1 traced to an intra-
  turn undo+redo: the player issued `recruit Horseman (12,6)`,
  undid it, re-issued `recruit Heavy Infantryman (12,6)` in the
  same [replay] block. Cross-block trailer-drop logic didn't apply
  (no save in between). Fix: in `replay_extract.py`, when
  emitting a recruit, peek at its [checkup]; if empty AND no
  [random_seed] follows in 1-3 commands BEFORE another player
  action, treat as undone and skip. First-attempt fix dropped
  Sablestone-style legitimate recruits; second-attempt fix had
  a no-RNG short-circuit that dropped Wose recruits incorrectly.
  Final universal rule: trust seed-or-checkup-result regardless
  of unit's RNG classification. New clean rate: 99.65% (was
  99.58%), with the 3 (11,5) cases gone and 1 attacker_missing
  also resolved.

- [x] 🟡 **Investigated the 3× path_non_adjacent residuals** (DONE
  2026-05-10, commit d3e85ab). Two distinct root causes:
    1. **Teleport-ability handling.** `tools/diff_replay.py`
       short-circuited only `len(xs) == 2` pure teleport moves;
       Wesnoth records walk + teleport + walk as a single move
       with the teleport at any step. New rule: each non-adjacent
       step accepted iff unit has `teleport` ability AND both
       endpoints are own-side villages. Cleared 2 cases (Howling
       Ghost Badlands turn 21, Hamlets turn 12 cmds 257+258).
    2. **Cross-block lookahead bug.** When a move at end of
       [replay] block i had empty [checkup], the extractor's
       3-cmd lookahead found block i+1's first action's
       [mp_checkup] and adopted ITS final_hex as block i's move
       destination, producing a fake non-adjacent step. Added
       block-boundary guard to the lookahead. Cleared 1 case
       (Hamlets turn 25).

- [x] 🟡 **Multi-advancement (HI→ST→IM in one combat)** (DONE
  2026-05-11, commit 7f5d3bf). `_maybe_advance_unit` was a single
  advancement step; Wesnoth's `advance_unit_at` LOOPS while
  carryover XP still meets the next-tier threshold. A low-level
  unit killing a high-level enemy can advance multiple tiers in
  one combat, especially at low experience_modifier (30 in this
  scenario) and with the intelligent trait (-20% threshold).
  Pre-fix, our sim stopped at intermediate tier, mismatching
  weapon damage / HP / MP for the rest of the game. Found via
  2p__Den_of_Onis_Turn_37_(114612) where Heavy Infantryman u67
  killed Dwarvish Dragonguard u46 at turn 27 s2 -> should chain
  HI→ST→IM but stopped at ST. Cleared 8 residual divergences
  (6× move:final_occupied cascades + 2× attack:weapon_oob + 1×
  move:mp_insufficient).

- [x] 🟡 **Investigate the 8 residual `diff_replay` divergences
  on the fresh corpus** (DONE 2026-05-11). Sweep through all
  residuals one-by-one with manual replay-viewer verification:
    - **Fixed in code:**
      - Suffix-redo dedup for save-mid-move (commit b64922f) —
        Fallenstar Lake t6.
      - Partial-checkup-attack drop for save-mid-attack (commit
        abe7834) — Ruined Passage t9.
      - Surrender-drop heuristic checkup-aware refinement
        (commit 9715b72) — Silverhead Crossing takeover case.
      - Multi-advancement loop (commit 7f5d3bf, above) —
        Den of Onis HI→ST→IM chain.
    - **Quarantined as unverifiable from corrupted source**
      (7 total this sweep, moved to
      `replays_dataset_quarantine/`): 3 corrupted (Hamlets t67,
      Hamlets t70, Sablestone Delta t16), 1 debug-tool-using
      (Howling Ghost Badlands), 3 Dunefolk-player-faction the
      classifier missed.
  Final: **100% diff_replay clean (5,484/5,484)** on the
  competitive-2p corpus.

See the **MCTS readiness scorecard** (refreshed 2026-05-10) further
down for a per-capability checklist.

**Severity:**
- 🔴 **CRIT** — silent corruption / wrong gradients / blocks a stated goal
- 🟠 **HIGH** — surfaces on the next nontrivial milestone (model attacks
  meaningfully / recalls / MCTS lands)
- 🟡 **MED** — correctness drift, polish, nice-to-have
- 🟢 **LOW** — cleanup, performance below noise floor

---

## NEW 2026-05-03 (post mod-purge + Hornshark sweep)

> **NOTE 2026-05-10:** The 91.05% headline below is HISTORICAL —
> it's the snapshot at end of 2026-05-03 work. Stages 1–21
> (entries dated 2026-05-04, see further down) brought the clean
> rate to **98.57%** on the 4,841-replay competitive-2p corpus.
> Use that as the current authoritative number.

**diff_replay clean rate: 91.05% on vanilla 2k sample (1821/2000).**
Up from 87% at session start. Mod purge accounted for the bulk;
village_gold extraction, AMLA queue pop, Hornshark Island
[switch]/[unit] events, and move-path ambush truncation closed
several smaller gaps. Remaining work:

- [x] 🟠 **Mid-game replay extractor anomaly: FIXED 2026-05-03
  in TWO stages.** The bug had two distinct flavors of the same
  underlying issue (mishandling Wesnoth's save file format).

  **Stage 1: cmd-order anomaly via wrong [replay] block selection.**
  Continuation saves (Wesnoth save file taken AT turn N, not
  from-scratch replays) carry TWO top-level `[replay]` blocks.
  The first is the from-turn-1 command stream (cmd[0]=[start],
  cmd[1]=[init_side side_number=1 ...]); the second is the
  post-save-point stream (starts directly with [move] mid-turn).
  The previous `max(replays, key=len)` heuristic in
  `replay_extract.py` picked only the LARGER of the two — for a
  Turn-5 save with sizes (62, 67), that's the post-save block.
  We then ran those Turn-5+ commands against `[replay_start]`
  (the canonical Turn-1 state with only leaders + scenery), and
  cmd[0] would src_missing on a hex held by a unit recruited
  during turns 1-4.
  
  Concrete case nailing it down:
  `2p__Caves_of_the_Basilisk_Turn_5_(120166).bz2`. Two `[replay]`
  blocks: 62 cmds (turn 1-4 history) + 67 cmds (turn 5 onwards).
  Old `max()` → second block, cmd[0]=move from (23,7), [replay_start]
  state had no unit there → src_missing.
  
  Fix: concatenate ALL `[replay]` blocks in document order. The
  empty-shell `(0, N)` case still works (0 contribution); the
  continuation `(62, 67)` case now produces a 129-cmd from-turn-1
  trajectory aligned with `[replay_start]`. Applied to
  `extract_replay` (compact-format path used by training) and
  `extract_pairs` (debug-format).
  
  **Stage 2: save-mid-action duplicate.** After Stage 1
  concatenation, a NEW failure mode surfaced: `recruit:target_
  occupied` at low cmd indices (avg cmd[79], 4 cases in
  competitive whitelist) plus residual cascade. Root cause:
  when Wesnoth saves DURING a player's turn (after the player
  issued an RNG-consuming action like recruit-with-traits or
  attack, but before the synced engine processed the
  [random_seed] follow-up), the save flushes the unfinished
  action to the END of [replay][i] WITHOUT its seed. On load,
  Wesnoth re-emits the same action (or a replacement, if the
  player undid + re-issued) at the START of [replay][i+1] WITH
  proper seeds. Naive concatenation thus emits both — the
  unfinished one (no seed) plus the redo (with seed) — and
  later commands that target the now-double-occupied hex break.

  Concrete examples:
    - 2p__Caves_of_the_Basilisk_Turn_16_(7251).bz2:
      [0] last cmd: [recruit] Fencer (18,5), no seed
      [1] cmd[0]: [recruit] Fencer (18,5)
      [1] cmd[1]: [random_seed] for the redo
    - 2p__Weldyn_Channel_Turn_14_(157907).bz2:
      [0] last cmd: [recruit] Dwarvish Thunderer (17,4), no seed
      [1] cmd[1]: [recruit] Dwarvish Fighter (17,4) -- player
                  undid + re-recruited a different unit type

  Fix: at each block-boundary cmd_idx, after processing the last
  command of block i, if `last_action_slot` is still set (meaning
  the trailing recruit/attack didn't get its seed within block i),
  drop that compact entry. Block i+1 will redo it properly.

  **Stage 3: REVERTED 2026-05-03.** I had removed `diff_replay`'s
  `gold < cost` recruit pre-check on the (incorrect) theory that
  Wesnoth's synced engine doesn't gate recruits on gold. User
  clarified this was wrong: Wesnoth's UI gate
  (`menu_events.cpp:327`) blocks insufficient-gold recruits at
  issue time, so a recorded recruit ALWAYS had gold >= cost in
  Wesnoth's reality. Negative gold only arises from
  upkeep/income, NOT from recruits. The 21 cases I flagged as
  "false positives" were actually real gold-tracker drift bugs
  (1-2g off between our sim and Wesnoth). Revert restores the
  check as a hard divergence flag. The +12 clean rate I
  attributed to Stage 3 was illusory; those 12 replays still
  have the same underlying gold-drift bug, the check is just
  visible again.

  **Stage 2 heuristic tightened (2026-05-03 audit).** The first
  iteration of the trailer-drop unconditionally dropped any
  recruit/attack at end of block i with no `[random_seed]` in
  block i. That was unsafe for musthave-only-trait recruits
  (75 default-era unit types: all undead, mechanical, elemental
  -- Ghoul, Skeleton, Walking Corpse, etc.) which legitimately
  COMPLETE without emitting a seed.
  Audit (`tools/verify_trailer_drop.py`) on a 500-replay sample
  found 0 false drops in practice (the 3 dropper-fires were all
  unfinished attacks), but the risk class was real. Tightened:
  ATTACKS still drop unconditionally (always emit seed when
  complete); RECRUITS only drop if the next block's first
  player action redoes the same recruit (same type+coords or
  same hex with different type = undo+different-recruit).
  Verified again on the same 500-sample: still 0 false drops,
  same 3 legitimate attack drops.

  **Stage 5: plague reanimation on counter-attack kill (NEW
  2026-05-03).** Built `tools/diff_unit_counter.py` which
  parses `[checkup] [result] next_unit_id` per command and
  compares against our sim's monotonic counter (instrumented
  in replay_dataset / scenario_events). Sweep across 347
  failing competitive-2p replays surfaced 40 cases of
  delta=-1 (sim missed a unit creation Wesnoth made) plus 1
  case of delta=-2.
  Concrete repro: `2p__Sablestone_Delta_Turn_16_(121180).bz2`
  cmd[220] -- Heavy Infantryman (HP=6) attacks Walking Corpse
  (HP=4, plague-impact); WC counter-strikes kill the
  Infantryman; Wesnoth spawns a side-2 Walking Corpse at the
  attacker's hex (15,12). Our sim was leaving the hex empty.
  Root cause: `combat.py`'s plague-spawn flag only set when
  defender died (a_stats.plague check), never when attacker
  died from defender's plague counter. `replay_dataset.py`'s
  attacker-died branch (`gs.map.units.discard(att)`) had no
  plague-spawn handling at all.
  Fix: added `plague_spawned_attacker_died` field to
  CombatResult; combat.py sets it when attacker dies AND
  defender has plague special; replay_dataset.py spawns the
  WC at att's hex on dfd's side via the existing
  `_spawn_plague_corpse` helper. The helper already enforces
  plague eligibility (not unplagueable, not on village), so
  the same restrictions apply as the offensive direction.
  Feeding ability is already symmetric in our sim
  (att_feed_bump and dfd_feed_bump both at
  replay_dataset.py:1530-1534) -- no fix needed there.
  Verified on the Sablestone case: unit-counter divergence
  resolved (sim now matches Wesnoth post-cmd[220]).

  **Stage 3 REVERTED (per user correction 2026-05-03).** I had
  removed `diff_replay`'s `gold < cost` recruit pre-check on
  the (incorrect) theory that Wesnoth's synced engine doesn't
  gate recruits on gold. User clarified: Wesnoth's UI gate
  (`menu_events.cpp:327`) blocks insufficient-gold recruits at
  issue time, so a recorded recruit ALWAYS had gold >= cost in
  Wesnoth's reality. Negative gold only arises from
  upkeep/income, NOT from recruits. The 21 cases I'd flagged
  as "false positives" were real gold-tracker drift bugs (1-2g
  off between our sim and Wesnoth). Reverted; check restored
  as a hard divergence flag.

  **Stage 4: level-0 trait HP off-by-one (NEW 2026-05-03).**
  After Stages 1-3 the residual was dominated by combat-state
  drift. Built `tools/diff_move_final_hex.py` (parses the older
  `[checkup] [result] final_hex_x/y` data that 99.5% of replays
  carry — gives Wesnoth's authoritative move endpoint per
  command). On the first concrete failing case
  (`2p__Caves_of_the_Basilisk_Turn_12_(20917).bz2`), traced
  to a move that wouldn't fit because a unit our sim hadn't
  killed was still in the destination hex. Sim said the unit
  had 1 HP after the attack; Wesnoth said dead. Combat math
  is bit-exact, so the divergence had to be in starting HP.

  Root cause: `tools/traits.py:386` had
  `max_hp += eff.hp_per_level * max(1, level)`. Wesnoth's
  resilient and healthy traits use `[effect] times=per level
  increase_total=1` which multiplies by RAW level (level 0 →
  +0). Our `max(1, level)` was giving level-0 units (Vampire
  Bat, Walking Corpse, Mudcrawler, Skeleton Footprint, etc.)
  an extra +1 HP from those traits. A level-0 Vampire Bat
  with feral+resilient should be 16+4=20 HP; we were giving
  21 HP. That extra 1 HP was the difference between dying
  and surviving the kind of 4×5=20-damage attacks that hit
  bats often. Fix: drop the `max(1, ...)`. Verified vs
  `wesnoth_src/data/core/macros/traits.cfg` TRAIT_RESILIENT
  / TRAIT_HEALTHY.

  Verification (competitive-2p subset of 23-map Ladder Era
  whitelist, ~4,863 replays):

  | Metric              | Pre-fix | S1      | S1+2    | S1-3    | All 4   |
  |---------------------|--------:|--------:|--------:|--------:|--------:|
  | Clean rate          | 89.98%  | 91.34%  | 91.80%  | 92.04%  |**92.86%**|
  | first_cmd_anomaly   |     88  |      0  |      0  |      0  |      0  |
  | near_start_anomaly  |     20  |      0  |      0  |      0  |      0  |
  | mid_replay_src_miss |    108  |    116  |    115  |    116  |    108  |
  | other               |    271  |    305  |    284  |    271  |    239  |

  Net: +141 fully-clean training trajectories per pass
  (+2.88pp in clean rate). The fixes also recovered ~1,200
  extractable replays vs the stale pre-fix dataset (15,067
  vs 13,866 .json.gz).

  **Stage 6: drain heals from undrainable targets (NEW
  2026-05-03, found via user GUI playback).** User loaded
  2p__Caves_of_the_Basilisk_Turn_11_(93641).bz2 in Wesnoth's
  replay viewer and reported Ghost u35's actual per-turn HP.
  Their trace had the Ghost at 7 HP entering its final combat
  (so a Skeleton's 12-max-damage WOULD kill it). Our sim had
  the Ghost at 18 HP -- healed to full by drain attacks against
  undead targets that should NOT heal.
  Bug: `combat.py:_perform_hit` checked only
  `striker_stats.drains and damage_done > 0` — never gated on
  the target being drainable. Wesnoth's
  `wesnoth_src/data/core/macros/traits.cfg` (verified by grep)
  has exactly 3 musthave traits adding `undrainable`:
  TRAIT_UNDEAD, TRAIT_MECHANICAL, TRAIT_ELEMENTAL — each via
  `[effect] apply_to=status add=undrainable`. Wesnoth's
  `attack.cpp` checks `opp.get_state("undrainable")` before
  healing.
  Fix: added `is_undrainable` field to `combat.CombatUnit`,
  populated in `tools/replay_dataset.py:_to_combat_unit` from
  the unit's statuses + `_UNPLAGUEABLE_TRAITS` set (same trait
  set applies to drain/poison/plague — Wesnoth's three
  musthaves add all three statuses together). Drain heal now
  skipped when `target.is_undrainable`. Verified: post-fix,
  Ghost u35's HP trajectory in our sim matches user's Wesnoth
  observation exactly (18 → 10 → 7).
  Note: feeding ability uses the same eligibility filter
  (already in place in our sim).

  **Stage 7: poison applied to unpoisonable targets (NEW
  2026-05-03).** After Stage 6 the Basilisk Ghoul case still
  failed at a different cmd. Tracing u39 Ghoul's HP history
  showed it taking 8 HP/init_side from the `poisoned` status
  starting at cmd[155] — but Ghouls are undead (unpoisonable)
  and Wesnoth's reality didn't poison them. Bug: `combat.py`'s
  poison-application at line 747 (`if striker_stats.poisons
  and not target.is_poisoned: target.is_poisoned = True`)
  never checked the target's `unpoisonable` status. Wesnoth's
  `attack.cpp:1057` gates poison via
  `opp.get_state("unpoisonable")`. The trait macros for
  TRAIT_UNDEAD / TRAIT_MECHANICAL / TRAIT_ELEMENTAL all add
  `unpoisonable` (verified via grep in
  `wesnoth_src/data/core/macros/traits.cfg` — exactly 3 traits
  in 1.18.4, same set as undrainable / unplagueable). Fix:
  added `is_unpoisonable` field to `combat.CombatUnit`,
  populated in `_to_combat_unit` from the unit's statuses +
  `_UNPLAGUEABLE_TRAITS` set, and gated the poison-status
  apply at combat.py:747 on `not target.is_unpoisonable`.
  Sub-corpus result: 86/88 → 87/88 (+1 clean replay).

  **Stage 8: scraper attack-specials name collision (NEW
  2026-05-03, devious!).** With Stages 1-7 in place, the
  500-replay sub-corpus was at 87/88 (one residual on
  2p__Cynsaun_Battlefield_Turn_33_(94735).bz2). User loaded
  the replay in Wesnoth GUI and reported the actual damage
  trace: Drake Arbiter dealt 24 dmg in cmd[1105] (= 2 hits),
  but our sim only got 1 hit. With the same RNG seed and
  CTH=60% on both sides, the only way the sequence diverges
  is if the strike ORDER differs.
  Tracing: Wesnoth's Drake Arbiter has two `[attack]` blocks
  both named `halberd` -- a blade one (no firststrike) and
  a pierce one (with firststrike). Our scraper's
  `_scan_attack_special_macros` matched attacks by `name=`
  only, so the pierce halberd's firststrike macro leaked
  into the blade halberd's specials list. At cmd[1105]:
    - Wesnoth: only defender (Spearman) has firststrike on
      its melee weapon -> Spearman strikes first, Drake
      strikes 2nd-and-4th-and-6th (rolls 5, 35, 92 with
      seed 7d3e8dc8) -> 2 hits.
    - Our sim: BOTH have firststrike (buggy LUT) -> attacker
      first per Wesnoth's rule, Drake strikes 1st-3rd-5th
      (rolls 72, 24, 82) -> 1 hit.
  Same RNG sequence, different strike order, different
  hits.
  Fix: rewrote the macro scanner as
  `_scan_attack_special_macros_by_index`, walking `[attack]`
  blocks in source order and returning one specials list per
  block, indexed positionally instead of by name. The old
  `_scan_attack_special_macros` is preserved for any external
  callers but no longer used by `extract_attacks`.
  Re-scrape `unit_stats.json`. Sub-corpus result: 87/88 →
  **88/88 = 100% clean**.

  **Stage 9: leadership bonus formula wrong (NEW 2026-05-03,
  found via user GUI trace).** Leadership formula was
  `25 * (leader.level - opponent.level)`, doubling the bonus.
  Should be `25 * (leader.level - buffed_unit.level)`. The
  Wesnoth macro string `value="(25 * (level - other.level))"`
  is ambiguous, but the English description on the same macro
  is explicit: "All adjacent lower-level units from the same
  side deal 25% more damage for each difference in level."
  The "difference in level" is between the LEADER and the
  BUFFED UNIT — the opponent's level is irrelevant.
  Concrete repro:
  2p__Hornshark_Island_Turn_12_(112807).bz2 cmd[96]: Mage
  (lvl 1) adjacent to Lieutenant (lvl 2) attacking Vampire
  Bat (lvl 0). User GUI replay showed Mage's per-strike
  damage = 7 (= 7 base × 0.75 ToD-night-lawful × 1.25
  leadership = 6.5625 → 7 round-half-toward-base). Our sim
  was computing 7 × 0.75 × 1.50 = 7.875 → 8 — wait actually
  9 (`7 × 1.0 + 25%-50% = 9` after rounding via integer
  `+50% −25% = +25%` then `7 × 1.25 = 8.75 → 9`). With 3
  strikes our sim killed the Bat (max 27 dmg vs hp=15) when
  Wesnoth left it at hp=1 (2 hits × 7 dmg = 14, 15-14=1).
  The Bat surviving at hp=1 in Wesnoth lets the next-cmd's
  Bowman attack succeed; in our sim the Bat was already
  dead, triggering attack:defender_missing. Fix: corrected
  the formula to `25 * (ally_level - unit_level)`.
  Verified: Mage's per-strike damage in our sim now 7 (was 9),
  leadership bonus 25% (was 50%). Full-corpus impact:
  95.43% → 95.97% (+0.54pp, +26 clean replays).

  **Stage 10: trailer-drop missed undone-recruit-then-end-turn
  (NEW 2026-05-03).** While investigating
  2p__Silverhead_Crossing_Turn_31_(127303).bz2 cmd[45]
  recruit:target_occupied at WML (4,12), found that block 0 of
  the raw replay ended with: ..., move, move, recruit Elvish
  Fighter (4,12) NO_SEED, end_turn, init_side. The recruit was
  issued, undone, then the player ended the turn. Wesnoth's
  reality has the recruit cancelled (no unit at (4,12)). Our
  Stage 2 trailer-drop heuristic only fired when the unfinished
  recruit was the LITERAL LAST entry in compact_commands; here
  the recruit is followed by end_turn + init_side so my pop
  didn't fire. The duplicate recruit at (4,12) (block 0 zombie
  + block 1 redo) blocked the legitimate later recruit.
  Fix: relaxed the "must be tail" check. Drop the trailer at
  `compact_commands[last_action_slot]` (NOT the literal tail);
  for recruits, additionally allow drop when an end_turn /
  init_side follows the unfinished recruit within the same block
  (signals an undone-and-ended-turn pattern). Verified: the
  Silverhead duplicate Elvish Fighter at (3,11) is gone in
  re-extracted output.

  **Corrupt-gold purge (NEW 2026-05-03).** User identified that
  some recruit:insufficient_gold cases are caused by player-edited
  or different-patch replays which Wesnoth itself desyncs on
  (Hornshark Turn 2 and Den_of_Onis Turn 2 verified). Built
  `tools/purge_corrupt_gold_replays.py` to move flagged replays to
  `replays_raw_review_gold/` for human review (NOT auto-delete).
  Default mode is dry-run. 105 raw replays moved with `--apply`
  (18 were competitive-2p).

  **Full-corpus result after Stages 1-11 + corrupt-gold purge**
  (competitive-2p, 4,845 replays after purge):

  | Metric              | Pre-S1 | After S5 | After S8 | After S10 | **All 11** |
  |---------------------|------------:|---------:|---------:|----------:|---------:|
  | Clean rate          |     89.98%  |  92.93%  |  95.43%  |   96.22%  | **97.50%**|
  | first_cmd anomaly   |         88  |       0  |        0 |        0  |        0 |
  | near_start anomaly  |         20  |       0  |        0 |        0  |        0 |
  | mid_replay_src_miss |        108  |     115  |       58 |       59  |       38 |
  | other               |        271  |     239  |      164 |      124  |       83 |

  **Stage 12: trailer-drop for save-mid-recruit when redo isn't
  block i+1's action #0 (NEW 2026-05-04).** Stage 2's heuristic
  only checked block i+1's FIRST player action for a same-hex/
  same-type recruit. But save-mid-recruit can manifest as
  "block[i] tail recruit (no seed) → block[i+1] starts with a
  move/move/move, THEN re-recruits at the same hex". The
  intervening moves were the player relocating units before
  re-issuing the recruit on load.
  Concrete repro: 2p__Weldyn_Channel_Turn_14_(157907).bz2.
  Block[0] tail = recruit Dwarvish Thunderer (16,3) no-seed.
  Block[1] cmd[0] = move, cmd[1] = recruit Dwarvish Fighter
  (16,3) (undo+replace at same hex, different type). The first-
  action sig pointed at the move, conditions (a)/(b) checked
  only the first action -> trailer kept, src_missing cascade
  at compact[83].
  Fix: extend the boundary metadata to include all recruit
  hexes used in block i+1 BEFORE its first end_turn/init_side
  (= recruits the same player issues during the resumed turn).
  Add condition (d): trailer hex matches any of those = drop.
  Cutoff matters -- without it, the heuristic dropped legitimate
  musthave-only-race recruit completions (no seed needed for
  Ghoul / Walking Corpse / etc.) when later turns happened to
  recruit at the same vacated hex. Concrete near-regression:
  2p__Aethermaw_Turn_18_(100985).bz2 -- block[0] tail = Ghoul
  (27,16) legitimately completed (musthave-only); block[1]
  starts with end_turn, then on later turns recruits Troll
  Whelp/Ghoul/Ghost at (27,16) after the original Ghoul moved.
  The end_turn cutoff filters those out.

  **Stage 13: musthave-only-race guard on trailer-drop condition
  (c) (NEW 2026-05-04, large fix).** Stage 10's condition (c)
  drops the trailer recruit if it's followed by `end_turn` /
  `init_side` within block i (= recruit issued, then turn ended,
  no [random_seed] follow-up = "must have been undone").
  But musthave-only races (Undead, Mechanical, Elemental — 75
  default-era unit types: Skeleton Archer, Ghoul, Walking Corpse,
  Vampire Bat, Dark Adept, etc.) NEVER emit a [random_seed] for
  recruits, regardless of completion. So a legitimately-completed
  Skeleton Archer recruit at end-of-turn looks identical to an
  undone one and got falsely dropped.
  Concrete: 2p__Caves_of_the_Basilisk_Turn_17_(10379).bz2.
  Side 2's turn 1 recruits FOUR Undead at the end of block[0]:
  Ghoul, Skeleton Archer x3. The first three survived because
  `last_action_slot` got overwritten by each successive recruit.
  The FOURTH (Skeleton Archer at (24,20)) was the trailing slot
  at the boundary; condition (c) saw end_turn after it and
  dropped it. Then cmd[27]'s side-2 move from (23,19) found no
  unit -> src_missing, breaking the entire replay.
  Affects every Undead-side-2 (or Mechanical/Elemental) replay
  where the player ended their turn with a recruit as the last
  RNG-tracked action -- the most common pattern in Caves of the
  Basilisk and other Undead-popular maps.
  Fix: gate condition (c) on `_recruit_consumes_rng(trailer_type)`.
  For musthave-only types, no-seed is normal -> don't drop.

  **Final full-corpus result after Stages 1-13** (competitive-2p,
  4,843 replays after one more OOS-corrupt purge):

  | Metric              | After S10 | After S11 | After S12 | **All 13** |
  |---------------------|----------:|----------:|----------:|---------:|
  | Clean rate          |    96.22% |    97.50% |    97.46% | **97.71%**|
  | mid_replay_src_miss |        59 |        38 |        39 |       30 |
  | other               |       124 |        83 |        84 |       81 |

  Stage 13 net: +11 replays clean (4721 -> 4732). The dropped
  fix space is mostly Caves-of-the-Basilisk-style cases where
  undead recruits got false-dropped at block boundaries.

  **Stage 14: defense-cap signs stripped at scrape time (NEW
  2026-05-04, devastating).** Wesnoth's `[defense]` blocks use
  NEGATIVE values to indicate CAPS (floors on to-be-hit, =
  ceilings on defense). E.g. `mounted` movetype has
  `forest=-70` meaning "Horseman has at MOST 30% defense on any
  forest-aliased terrain". The cap matters for composite hexes
  (`Gg^Fet`, `Gg^Fp`, etc.) where naive min-over-aliases would
  pick the better-of-base-and-overlay (60 flat vs |70| forest)
  and bypass the forest cap entirely.
  `tools/scrape_unit_stats.py` was applying `abs(int(v))` at
  the movetype layer (line 531) AND at the unit-variation layer
  (line 863), stripping the negative sign and losing the cap.
  Variants: terrain_resolver had `_collect_neg_caps` logic to
  handle preserved caps, but the caps weren't preserved at
  scrape time. Affected EVERY unit with a movetype-level cap
  (mounted forest, fly fungus, etc.) on any composite terrain
  in the corpus.
  Concrete repro: 2p__Weldyn_Channel_Turn_23_(40663).bz2 cmd[66]
  Horseman vs Ghost on Gg^Fet. Without cap: 60% to-be-hit ->
  Ghost CTH 60 -> all retaliations miss in our sim. With cap:
  70% to-be-hit -> Ghost CTH 70 -> Ghost's 3rd strike (roll 63)
  hits, drains 3 HP, Ghost survives at 11 HP (matches user's
  GUI trace). Without the fix, our sim killed Ghost ~3 turns
  early and broke the entire downstream replay.
  Fix: preserve negative-cap signs in scrape_unit_stats at
  both the movetype-level and unit-variation-level. Re-scrape
  unit_stats.json. Combat reads at runtime, no re-extraction
  needed.

  **Final full-corpus result after Stages 1-14** (competitive-2p,
  4,843 replays):

  | Metric              | After S11 | After S13 | **All 14** |
  |---------------------|----------:|----------:|---------:|
  | Clean rate          |    97.50% |    97.71% | **98.39%**|
  | mid_replay_src_miss |        38 |        30 |       20 |
  | other               |        83 |        81 |       58 |

  Stage 14 net: +33 replays clean (4732 -> 4765). The largest
  single-stage improvement of the session, second only to the
  Stage 1+2 trailer-drop work.

  Total session improvement: **+8.41pp clean rate** (89.98% ->
  98.39%), ~410 additional clean training trajectories. Plus
  105 corrupt-gold replays + 2 OOS-corrupt replays + 1 non-vanilla-
  RNG replay quarantined for review.

  **Stage 16: scraper variation movement_costs override (NEW
  2026-05-04).** `tools/scrape_unit_stats.py:extract_variations`
  was applying `[defense]` and `[resistance]` overrides for
  unit variations but missed `[movement_costs]`. The most
  consequential miss is the Walking Corpse:bat variation, which
  overrides cave/fungus/deep_water from the base `fly` movetype
  (3/3/1) to (1/1/1). Without the override, our sim treated
  bat-corpses on mushroom groves as cost-3 entries, rejecting
  legitimate moves through Hornshark Island's central mushroom
  patches as `mp_insufficient`.
  Concrete repro: 2p__Hornshark_Island_Turn_7_(135535) cmd[188]
  -- Walking Corpse:bat path from (24,18) through Tb^Tf
  (mushroom grove) to (25,13). Pre-fix path cost 7 > moves 5.
  Post-fix path cost 5 = moves 5, valid.
  Fix: scrape `[movement_costs]` from variation children, layer
  on top of the variation's chosen movetype defaults.

  **Final full-corpus result after Stages 1-16** (competitive-2p,
  4,843 replays):

  | Metric              | After S14 | **All 16** |
  |---------------------|----------:|---------:|
  | Clean rate          |    98.39% | **98.41%**|
  | mid_replay_src_miss |        20 |       21 |
  | other               |        58 |       56 |

  Stage 16 net: +1 replay clean. Modest but real fidelity fix
  affecting any future Walking Corpse:bat / Soulless:bat /
  similar variation corpse-bat encounter on cave / fungus
  hexes. (Stage 15 [+time] still deferred.)

  Total session improvement: **+8.43pp clean rate** (89.98% ->
  98.41%), ~411 additional clean training trajectories.

  **Stage 17: WML `[+tag]` semantics in parser (NEW 2026-05-04).**
  Wesnoth's `[+tag]` WML form has CONTEXT-DEPENDENT semantics:
  - For sequence-children (`[+time]` inside `[time_area]`): APPEND
    a new entry to the cycle. Tombs of Kesorak's illuminated area
    has 6 macro-expanded `[time]` blocks plus 4 `[+time]` overrides,
    forming a 10-entry cycle. On illuminated hexes, turns 6-9 stay
    at `lawful_bonus=25` (DAY) while the rest of the map cycles
    through Second Watch / First Watch.
  - For singleton-children (`[+unit]` inside `[side]`): MERGE
    attrs+children into the latest preceding sibling of the same
    tag. Caves of the Basilisk's `{UNIT_PETRIFY ... } [+unit]
    description=... [/unit]` adds the inscription text to the
    just-placed petrified statue rather than spawning a phantom.

  Implementation: `parse_wml` now recognizes `[+tag]` separately
  and tags the resulting `WMLNode` with `_is_plus_form=True`. A
  post-parse pass (`_resolve_plus_forms`) walks each node's
  children and applies APPEND for tags in `_PLUS_APPEND_TAGS`
  ({"time"}) and MERGE otherwise.

  Verified: Tombs of Kesorak time_area now parses 10 entries
  (was 6 before); Caves of the Basilisk has 15 [unit] children
  in side 3 (was 30 with naive append); Sullas Ruins 5 [unit]
  children. Both originally-targeted Tombs cmd[184] cases
  (0c7c71f64449, 3403628eaedc) now clean.

  Trade-off: 4 different Tombs replays became non-clean as the
  fidelity-correct ToD shift exposed deeper bugs that were
  previously masked. Net: 4766 -> 4764 clean (-2). The fix is
  fidelity-correct -- the new residuals are real bugs we'll
  surface and fix in a follow-up pass.

  **Stage 18: save-mid-move duplicate trailer-drop (NEW
  2026-05-04).** Analog of Stage 12 but for moves instead of
  recruits. When Wesnoth saves IMMEDIATELY after a move
  completes, the move's [checkup] result IS recorded in
  block[i] (move was committed). On load, the engine still
  re-emits the SAME move at block[i+1]'s start with another
  completed [checkup]. Concatenating both made our sim try to
  move the same unit twice -- the second attempt fails
  src_missing because the unit is already at the destination.
  Concrete: 2p__The_Freelands_Turn_18_(170491).bz2 -- block[0]
  tail = move (12,15)->(14,18), checkup with result; block[1]
  cmd[0] = same move, checkup with result. Compact[187] and
  compact[188] were both the same move; cmd[188] failed.
  Fix: track `last_move_slot` in addition to `last_action_slot`
  during compact emission. At each block boundary, if the
  trailer move's path matches block[i+1]'s first action sig,
  drop the trailer.

  **Final full-corpus result after Stages 1-18** (competitive-2p,
  4,841 replays after re-extraction):

  | Metric              | After S16 | After S17 | **All 18** |
  |---------------------|----------:|----------:|---------:|
  | Clean rate          |    98.41% |    98.37% | **98.39%**|
  | mid_replay_src_miss |        21 |        21 |       18 |
  | other               |        56 |        58 |       60 |

  Stages 17+18 net effect on the *headline* number is roughly
  flat (small Tombs/Hornshark trade-offs balance out), but the
  underlying simulator is now genuinely correct for [+time]
  illuminated zones AND save-mid-move duplicates -- two real
  Wesnoth WML/replay quirks we were silently tolerating before.

  **Stage 19: scenario `[object]` handler + combat weapon-specials
  union (NEW 2026-05-04).** Two-part fix surfaced by Hornshark
  Island Turn 11 (140989):
  (a) Hornshark's `MODIFY_BOWMAN` macro emits a top-level
  `[object]` inside a prestart event that grants `firststrike`
  to the bow attack of the side-1 Bowman at (0,0) and side-2
  Bowman at (27,23). We weren't dispatching `[object]` actions.
  Added `_object_action` to scenario_events that walks
  [filter]/[effect]/[set_specials] and merges new weapon
  specials onto the targeted unit's matching attack.
  (b) `tools/replay_dataset.py:_to_combat_unit` was reading
  weapon specials from BASE unit-stats only, losing any
  scenario-applied modifications. Fixed to union base specials
  with the unit's per-attack `weapon_specials`.
  Without this, Hornshark Bowman fired without firststrike,
  alternating combat order rolled 2 of 3 retaliation strikes
  hitting Elvish Archer at cmd[73] (rolls 25, 28 < CTH 30) for
  12 dmg. With firststrike, defender-first reorders RNG: only
  1 of (rolls 42, 12, 93) hits at CTH 30 → 6 dmg, matching
  user's GUI trace.

  **Final full-corpus result after Stages 1-19** (competitive-2p,
  4,841 replays):

  | Metric              | After S18 | **All 19** |
  |---------------------|----------:|---------:|
  | Clean rate          |    98.39% | **98.49%**|
  | mid_replay_src_miss |        18 |       19 |
  | other               |        60 |       54 |

  Stage 19 net: +5 replays clean. Mostly Hornshark Island
  cases where a Bowman retaliation dictates downstream
  combat outcomes.

  Total session improvement: **+8.51pp clean rate** (89.98% ->
  98.49%), ~417 additional clean training trajectories.

  **Stage 20: full `[effect]` apply_to vocabulary + custom-trait
  effect dispatch (NEW 2026-05-04).** Extended Stage 19's
  `_object_action` to handle every `apply_to` form used in 2p
  ladder scenarios via a shared `_apply_effect_to_unit` helper:
    - `apply_to=attack` with `increase_attacks=N`,
      `increase_damage=N`, `[set_specials]`, `range=` filter;
      mirrors Wesnoth's `apply_modifier` for percent-or-int
      increase strings.
    - `apply_to=new_attack` (Silverhead Crossing's "evil eye"
      ranged arcane on side-3 boss).
    - `apply_to=remove_attacks` (Thousand Stings statues).
    - `apply_to=hitpoints` `increase_total=` / `set=` /
      `heal_full=yes`.
    - `apply_to=movement` `set=` / `increase=`.
    - `apply_to=status add=`/`remove=`.
    - Cosmetic forms (ellipse, image_mod, overlay, profile,
      new_animation, halo, zoc) silently no-op.
  `_unit_action` also walks `[modifications]/[trait]/[effect]`
  for CUSTOM traits (id NOT in our named TRAITS registry --
  loyal/quick/resilient/strong/intelligent are already
  fully handled by `apply_traits_to_unit`; double-applying
  their [effect]s would stack +HP / +movement bonuses). The
  guard caught a regression where Hornshark Sergeants and
  Drake Fighters lost 1 movement to double-applied resilient.
  Custom traits like Caves of the Basilisk's `id=remove_hp`
  (drops max_hp -100% to make statues 1 HP) are now applied.
  And `[status] petrified=yes` on placed units routes to the
  petrified status flag, with movement/attacks zeroed (so
  scenario-event-spawned statues, if any, behave like
  [replay_start]-extracted ones).

  **Final full-corpus result after Stages 1-20** (competitive-2p,
  4,841 replays):

  | Metric              | After S19 | **All 20** |
  |---------------------|----------:|---------:|
  | Clean rate          |    98.49% | **98.53%**|
  | mid_replay_src_miss |        19 |       20 |
  | other               |        54 |       51 |

  Stage 20 net: +2 replays clean. The headline gain is small
  because the affected scenarios (Basilisk/Sullas/Thousand
  Stings statues; Silverhead evil-eye boss) place their
  modified units via `[replay_start]` -- already extracted
  with the final stats baked in -- so the effects are mostly
  redundant. The new dispatch matters for replays where
  scenario events SPAWN modified units mid-game (Hornshark's
  faction-specific heroes via `[switch] -> [unit]` are the
  current canonical example, but they only use named traits).

  Total session improvement: **+8.55pp clean rate** (89.98% ->
  98.53%), ~419 additional clean training trajectories.

  **Stage 21: scenario_events level-0 trait HP fix (NEW
  2026-05-04).** `_unit_action` was passing
  `level=int(stats.get("level", 1) or 1)` to
  `apply_traits_to_unit`. For level-0 units (Walking Corpse,
  Vampire Bat, Mudcrawler, statue side-3 units), the inner
  `stats.get("level", 1)` correctly returns 0 -- but `0 or 1`
  evaluates to 1 because 0 is falsy in Python. So our trait
  application thought the unit was level-1 and applied
  resilient's `hp_per_level=1` extra HP, giving level-0
  Walking Corpses 22 HP instead of 21.
  Defeats Stage 4's earlier fix on the same axis (which was
  for the recruit path; this is the scenario-event placement
  path, separate code).
  Concrete: 2p_Hornshark_Island.cfg places Walking
  Corpse:saurian (uu8) with TRAIT_STRONG + TRAIT_RESILIENT.
  Base hp 16 + 4 (resilient) + 1*level (with bug, +1; correct,
  0) + 1 (strong) = 22 (buggy) or 21 (correct). Young Ogre
  (uu3) with strong trait deals 6*1.10 = 7 dmg per hit; max
  3 hits = 21 dmg. With buggy 22 HP, uu8 always survives at
  1 HP and blocks subsequent moves to (21,13). With correct
  21 HP, 3 hits kill uu8 outright -- matching Wesnoth.
  Fix: separate the int conversion from the falsy-check.

  **Final full-corpus result after Stages 1-21** (competitive-2p,
  4,841 replays):

  | Metric              | After S20 | **All 21** |
  |---------------------|----------:|---------:|
  | Clean rate          |    98.53% | **98.57%**|
  | mid_replay_src_miss |        20 |       19 |
  | other               |        51 |       50 |

  Stage 21 net: +2 replays clean (Hornshark Walking Corpse
  saurian survival cases). The fix is fidelity-correct and
  matches Wesnoth's level-0-no-per-level-bonus rule.

  Total session improvement: **+8.59pp clean rate** (89.98% ->
  98.57%), ~421 additional clean training trajectories. Plus
  108 corrupt/OOS/non-vanilla-RNG replays quarantined for
  human review.

  **Stage 22: leadership level-0 falsy-coercion bug (NEW
  2026-05-04).** Same Python falsy-coercion antipattern as
  Stage 21 but in `tools/abilities.py:leadership_bonus`:
  `int(_stats_for(unit.name).get("level", 1) or 1)` coerced
  level-0 units to level-1, undercounting the leadership
  bonus for level-0 buffed units.
  Wesnoth's leadership formula:
    bonus = 25 * (leader.level - buffed_unit.level)
  For a Lieutenant (level 2) adjacent to a Woodsman (level 0),
  the bonus is 25 * (2 - 0) = +50%. Our bug computed
  25 * (2 - 1) = +25%, halving the boost.
  Concrete: 2p_Hornshark_Island Turn 11 (113429) cmd[147] --
  side-1 Lieutenant (uu1) is adjacent to Woodsman (uu7); when
  uu7 retaliates against uu19 Thief's attack, leadership
  should boost retaliation damage by +50% (4 base * 1.50 *
  pierce_resist = 6 dmg/hit), but our buggy +25% gave 5/hit.
  uu19 ended at hp 1 in our sim instead of dying outright,
  blocking subsequent side-1 moves to (4,11).
  Fix: replace `or 1` with a try/except float-to-int that
  preserves 0.

  **Final full-corpus result after Stages 1-22** (competitive-2p,
  4,841 replays):

  | Metric              | After S21 | **All 22** |
  |---------------------|----------:|---------:|
  | Clean rate          |    98.57% | **98.78%**|
  | mid_replay_src_miss |        19 |       15 |
  | other               |        50 |       44 |

  Stage 22 net: **+10 replays clean** (4772 -> 4782). Mostly
  Hornshark cases where level-0 hero retaliations were
  under-buffed by leadership.

  Total session improvement: **+8.80pp clean rate** (89.98% ->
  98.78%), ~431 additional clean training trajectories.

  **Stage 23: revert [+time] APPEND to MERGE — match Wesnoth's
  actual `[+element]` semantics (NEW 2026-05-04).** Stage 17
  defaulted [+time] to APPEND in `_PLUS_APPEND_TAGS = {"time"}`.
  But re-reading Wesnoth source `parser.cpp:217-242`, `[+element]`
  unconditionally re-opens the LAST same-named element in the
  parent for field/child merging -- it does NOT append a new
  element. Tombs of Kesorak's illuminated `[time_area]` is thus
  a 6-entry cycle (not 10), with each `[+time]` overriding the
  preceding macro-expanded `[time]`'s image+lawful_bonus.
  Concrete: Tombs Turn 16 (32287) cmd[136] -- side-2 Mage on
  illuminated WML(22,23) at turn 7. With MERGE 6-cycle: idx 0 =
  bright_dawn lawful=25, Mage 7 base * 1.25 = 9 dmg/hit -> uu13
  Spearman dies in cmd[136], reducing side-1 upkeep by 1 at
  the next init_side, and the cmd[154] Spearman recruit (cost
  14) succeeds with 14g (vs 13g in our pre-fix sim).
  Trade-off: +4 (Tombs gold drift cases) -2 (Tombs Turn 22
  cmd[184] cases that benefited from APPEND's stretched cycle).
  Net: +2 clean replays.

  **Final full-corpus result after Stages 1-23** (competitive-2p,
  4,841 replays):

  | Metric              | After S22 | **All 23** |
  |---------------------|----------:|---------:|
  | Clean rate          |    98.78% | **98.82%**|
  | mid_replay_src_miss |        15 |       15 |
  | other               |        44 |       42 |

  Total session improvement: **+8.84pp clean rate** (89.98% ->
  98.82%), ~433 additional clean training trajectories.

  **Stage 24: scenario_events `_unit_action` honors
  experience_modifier (NEW 2026-05-04, MAJOR).** For
  scenario-event-placed heroes, `_unit_action` was calling
  `_build_unit(udict, apply_leader_traits=False)` without
  passing exp_modifier, so max_exp defaulted to base. Wesnoth
  applies the scenario's `experience_modifier` (typically 70
  for ladder) to ALL units' xp-to-advance, so a Skeleton's
  base xp=39 becomes 27. Without the fix, Hornshark's Sorrek
  hero stayed Skeleton (max_exp=39, never reached) when
  Wesnoth had it advance to Deathblade at xp=27. Deathblade's
  +1 movement (6 vs 5) made downstream multi-hex moves valid
  in Wesnoth that we rejected as `mp_insufficient`.
  Concrete repro: 2p_Hornshark_Island_Turn_12_(103721)
  cmd[311] -- Sorrek 6-cost path move. In Wesnoth: Deathblade
  (6 MP), fits. In our sim: Skeleton (5 MP), `path cost=6 >
  current_moves=5`.
  Fix: read `gs.global_info._experience_modifier` and forward
  to `_build_unit`.

  **Final full-corpus result after Stages 1-24** (competitive-2p,
  4,841 replays):

  | Metric              | After S23 | **All 24** |
  |---------------------|----------:|---------:|
  | Clean rate          |    98.82% | **99.13%**|
  | mid_replay_src_miss |        15 |       11 |
  | other               |        42 |       31 |

  Stage 24 net: **+17 replays clean** in a single one-line fix.
  Mostly Hornshark Island heroes that should have advanced.

  Total session improvement: **+9.15pp clean rate** (89.98% ->
  99.13%), ~450 additional clean training trajectories.

  **Remaining residuals (~59 cases at 98.82%):** All are deep
  combat-state cascades where one earlier combat had the wrong
  outcome and propagated forward (e.g. uu19 Thief at hp 1 vs hp
  0 → blocks a downstream move; uu28 not advancing because of
  a missed kill XP; Skeleton mp_insufficient with cost==max+1).
  These are not addressable by systemic fixes — each requires
  a per-replay GUI trace from the user to identify the exact
  diverging combat. The tool support (`dump_unit_states.py`,
  `diff_unit_counter.py`, `diff_move_final_hex.py`) is in
  place; the path forward is human-in-the-loop per-residual
  triage.

  Per-scenario breakdown:
  - Hornshark Island: 22 (mostly hero combat cascades)
  - Ruphus Isle: 15 (mostly OOS-corrupt at cmd[84] hex (8,14);
    1 already confirmed/purged)
  - Other 11 scenarios: 16 single-replay cascades.

  Per-kind breakdown:
  - attack:defender_missing: 23
  - move:final_occupied: 14
  - move:src_missing: 10
  - attack:attacker_missing: 4
  - attack:friendly_fire: 3
  - move:mp_insufficient: 2 (potential pathfinder edge case)
  - recruit:insufficient_gold: 2 (gold drift cascades)
  - attack:weapon_oob: 1 (advancement-XP cascade — unit didn't
    advance to Sorceress in our sim)

  **Stage 11: advancement should clear poisoned/slowed/petrified/
  stunned (NEW 2026-05-04, found via user GUI trace).** Wesnoth's
  `actions/advancement.cpp:319-326` (`get_advanced_unit`) calls
  `heal_fully()` then explicitly:
    new_unit->set_state(unit::STATE_POISONED, false);
    new_unit->set_state(unit::STATE_SLOWED, false);
    new_unit->set_state(unit::STATE_PETRIFIED, false);
  Our `tools/replay_dataset.py:_maybe_advance_unit` was building
  the advanced unit with `_replace_unit(...)` and NOT passing a
  fresh `statuses=...` set, so the old unit's statuses persisted
  across advancement. Concrete repro:
  2p__Ruphus_Isle_Turn_18_(213496).bz2 -- a Gryphon Rider was
  poisoned by a Ghoul on turn 9, kept the poison status when
  advancing to Gryphon Master mid-fight at cmd[275], and then
  took -8 HP/turn from poison-tick across multiple init_sides,
  dropping the Master's HP from full 48 to ~12 by turn 16. In
  the broken state, the Master's hp was so low that the Ghoul's
  counter-attack at cmd[353] killed it (+16 kill XP), advancing
  the Ghoul to Necrophage; Wesnoth's Master was at full HP, the
  Ghoul didn't kill it, and the Ghoul stayed a Ghoul (XP 14 → 16,
  cap 21 = 30 × 70% experience_modifier).
  Fix: explicitly drop the four statuses from the new unit's
  status set during advancement. User clarification 2026-05-04:
  "All negative conditions are cleared on levelup (poisoned,
  slowed, stunned, etc.)" -- added `stunned` per user, even
  though Wesnoth's source enumerates only the three above.
  Verified on Ruphus Isle: u20 stays a Ghoul (hp=3/33,
  xp=16/21 entering cmd[355]) -- exactly matches user's GUI
  HP/XP trace.

  **Full-corpus validation after Stages 6-8** (competitive-2p,
  4,863 replays):

  | Metric              | Pre-session | After S5 | After S8 |
  |---------------------|------------:|---------:|---------:|
  | Clean rate          |     89.98%  |  92.93%  |**95.43%**|
  | first_cmd anomaly   |         88  |       0  |        0 |
  | near_start anomaly  |         20  |       0  |        0 |
  | mid_replay_src_miss |        108  |     115  |       58 |
  | other               |        271  |     239  |      164 |

  Total +5.45pp clean rate from session start. The 222 residual
  cases on the full corpus didn't appear in the random 500-replay
  sub-corpus -- next iteration would re-seed the sub-corpus to
  surface them.

  **Stage 5 finding (no fix banked): residual is mid-game
  cumulative drift, not early-turn bugs.** Searched the full
  competitive-2p corpus for ANY move divergence at turn ≤ 3
  using the new `tools/diff_move_final_hex.py` oracle.
  **Zero hits.** Every diverging move is at turn ≥ 4,
  confirming:
    - Initial-state extraction is correct (turn 1 plays match)
    - Move-truncation logic (path occupancy / friendly
      passthrough / fog ambush) is correct (turn 2-3 multi-hex
      moves match)
    - Trait HP / per-recruit RNG is correct (Stage 4 fix held)
    - Combat math is bit-exact (verified directly via 29/29
      strict-sync sample)
  
  What's left must be **per-turn effects accumulating drift**:
  healing/poison/regeneration at init_side, advancement /
  AMLA edge cases, or specific weapon-special interactions
  (drain over-heal cap, swarm strike-count, slow MP doubling)
  that combine to cause one HP off across many turns.

  Concrete drilldown attempted: Aethermaw Turn 11 (db34f98ea361)
  has a Ghost u5 attacked 3x at (35,22) over 4 commands, surviving
  in our sim but dying in Wesnoth's reality. Ghost is level-1,
  so the level-0 trait fix doesn't apply. No strict-sync
  `[mp_checkup]` data for this replay, so per-strike combat
  comparison is impossible — root cause requires either
  hand-tracing combat or strict-sync replay generation.

  **Path forward (when prioritized — Stage 6+):**
    1. Generate strict-sync replays via `tools/make_strict_replay.py`
       (already exists but needs a Wesnoth runtime to validate).
    2. Build a parser for the older `[checkup] [result]
       next_unit_id / random_calls` counters — drift in either
       is a precise root-cause signal across all 99.5% non-strict
       replays.
    3. Audit healing/poison/advancement per-turn effects against
       Wesnoth source.

  **Residual 7.96% non-clean breakdown** (387 cases):
    - 133 attack:defender_missing (avg cmd[328])
    - 126 move:final_occupied (avg cmd[363])
    - 116 mid_replay_src_missing (avg cmd[312])
    - 37 attack:attacker_missing (avg cmd[446])
    - 6 attack:friendly_fire (plague-side bug, BACKLOG-tracked)
    - <10 each: misc

  These are NOT combat-math drift: ran `tools/diff_combat_strike`
  on the only competitive-2p strict-sync replay we have
  (`2p__Arcanclave_Citadel_Turn_11_(93570).bz2`), got
  29/29 attacks bit-exact match, zero divergences. Combined
  with prior 731/731 verification this is strong evidence
  combat math is correct. Residual divergences must come from:
    (a) Movement edge cases (ZoC, ambush, multi-hex paths,
        terrain alias bugs)
    (b) Healing/poison/regenerate/cures interactions
    (c) Advancement / AMLA edge cases
    (d) Scenario-event handlers (prestart unit placements
        beyond Hornshark's [switch]/[case], turn-N events)
    (e) Plague side tracking
    (f) Cumulative state drift from any of the above

  **Limit of this iteration:** Per-case investigation needs
  oracle data we don't have. Of 1,001 sampled raw replays:
  5 (0.5%) have strict-sync `[mp_checkup]` data; 993 have
  only the older `[checkup]` format which carries `next_unit_id`
  and `random_calls` counters but NOT per-strike combat data.
  Individual cases require ~30-60min manual tracing each.

  **Path forward (when prioritized):**
    1. Build a parser for the older `[checkup]` format —
       compare per-command `next_unit_id` and `random_calls`
       against our sim's counters. A drift in either is a
       precise root-cause indicator.
    2. Audit Hamlets, Silverhead, The_Freelands, Weldyn_Channel
       scenario.cfg files for prestart events not currently
       modeled by `tools/scenario_events.py`.
    3. Hand-trace 3-5 attack:defender_missing cases to
       categorize (combat? movement? healing?).

  Tooling added: `tools/check_first_cmd_anomaly.py` — classifies
  first-divergence per replay by cmd_index and kind, with
  `--filter-competitive-2p` to scope to the training whitelist.
  Useful for any future regression in extractor behavior.

  Re-extracted output lives in `replays_dataset_reextract3/`
  pending operator decision to swap it in for `replays_dataset/`
  (the active training input). Old `replays_dataset/` reflects
  pre-fix extraction; one `mv` away from picking up the fix.

- [x] 🟡 **Recruit:insufficient_gold drift (18 / 2k → 0).** SUPERSEDED
  by the 2026-05-11 fidelity sweep. The 2 final residual gold-drift
  cases on the fresh competitive-2p corpus (Hamlets t34, Sablestone
  Delta t8) both verified as **corrupted replays** in Wesnoth's
  replay viewer; quarantined rather than fixed in code. No gold-
  drift cases remain on the current 5,484-replay corpus.

- [x] 🟡 **Friendly_fire plague edge cases (4 / 2k → 0).** SUPERSEDED
  by the 2026-05-11 fidelity sweep. No friendly-fire plague residuals
  on the current 5,484-replay corpus (the Stages 1–21 plague-event
  fixes + the multi-advancement / surrender-drop / suffix-redo /
  partial-checkup-attack fixes resolved everything else).

- [x] 🟢 **Cascade-class failures (~136 / 2k → 0).** SUPERSEDED by
  the 2026-05-11 fidelity sweep — all root causes closed.

- [x] 🟡 **`build_trait_info`: race-additional-traits skips
  fearless for neutral units** (DONE 2026-05-03). `build_trait_info`
  now accepts `alignment=` and the race-additionals loop passes
  `skip_fearless=is_neutral`. The skip is keyed on trait `id`
  (matching the C++ `t["id"] != "fearless"` check at
  `wesnoth_src/src/units/types.cpp:350`), not macro name, so it
  catches both `TRAIT_FEARLESS` and any race that introduces
  fearless under a different macro alias. Default-era trolls are
  chaotic so dormant for ladder play; activates for hypothetical
  neutral-troll eras and any future race that adds fearless.
  Re-run `python tools/scrape_unit_stats.py wesnoth_src
  unit_stats.json` to refresh the snapshot.

- [ ] 🟡 **Per-gender unit-type traits not handled.** Black Horse
  (Horse_Black.cfg) has `[male] {TRAIT_STRONG}` / `[female]
  {TRAIT_FEARLESS}` — gender-specific pool entries. Our scraper
  merges both genders' traits into a single pool. Not a
  default-era PvP issue (Black Horse isn't recruitable), but
  fidelity-relevant for any future Black-Horse-mod era.

- [ ] 🟢 **`diff_replay` could surface combat HP per-strike**
  to make cascade-class diagnosis easier. Currently it stops at
  the first divergence; combat divergence should ideally compare
  per-strike outcomes against the replay's `[mp_checkup]/[result]`
  blocks rather than waiting for a state-level cascade.

- [ ] 🟠 **Encoder runs at batch_size=1** (`encoder.py`,
  Phase 3.2 deferred). Trainer re-forwards every transition
  individually; on GPU this leaves kernels under-amortized.
  Padding states to fixed max-hex-count and batching 4-8
  transitions per forward would unlock 2-4x training throughput
  on the cluster. Largest single performance lever currently
  visible. Estimated 1-2 weeks (needs padding logic + batch dim
  handling through encoder + model). Now-unblocked since the
  re-forward / value-head split landed.

- [ ] 🟡 **`action_sampler.py:1113` swallows
  `expected_attack_net_damage` exceptions silently**, returning
  `net=0.0` for any failure. A subtle combat-LUT bug would
  manifest as the policy losing all attack bias rather than
  raising. At minimum log the unit-pair on first occurrence per
  process; better, narrow the except to the specific exception
  classes the helper actually raises (KeyError on missing
  resistance / IndexError on weapon idx). Same pattern at
  `encoder.py:851` for `_recruit_features_for` -- silent
  fallback to default stats hides a stale unit_stats.json.

- [ ] 🟡 **End-to-end replay round-trip test missing.** Take a
  real replay, run the simulator's reconstruction, re-export
  the command stream via `sim_to_replay`, then diff WML
  token-by-token against source. This is the strongest possible
  regression net for the bit-exactness work and isn't yet in
  pytest. Currently we rely on the offline `tools/diff_replay.py`
  sweep which the operator runs ad-hoc. Blocked on
  `sim_to_replay` rebuilding the bz2 from scratch (already a
  tracked item below) -- once that lands, wire as a single
  pytest case parameterized over a small fixture corpus.

- [ ] 🟢 **Optional `[checkup]` debug emission in `sim_to_replay`**
  (already tracked under exporter section). Promote: cheapest
  diagnostic possible for OOS / cascade-class debugging since
  Wesnoth would name the exact diverging strike. Worth doing
  before next OOS hunt.

References are `path/file.py:line`. Each item is actionable on its own.

---

## CLOSED 2026-05-03

- [x] 🔴 **Combat damage divergence: cascade-class diff_replay
  failures.** Was the headline problem at session start (~67/100
  diverging). Root cause was **77.8% of `replays_raw/` carried
  gameplay-affecting mods** — most prominently Biased RNG, which
  smooths combat hit/miss to the expected value (so vanilla MTRng
  with the same seed gives different per-strike hits). After
  `tools/purge_mod_replays.py` deleted 132,838 mod-using replays
  (keeping vanilla + cosmetic-only) and re-running on the
  remainder, the post-87% gap mostly closed. Combat math itself
  was already bit-exact (verified via [mp_checkup] from
  strict-sync replays, 731/731 strikes).

- [x] 🟠 **Re-extraction needs to complete on full
  `replays_raw/`.** Done implicitly via the mod purge — only
  vanilla replays remain, all already extracted to
  `replays_dataset/`. 13,866 vanilla .json.gz survive.

- [x] 🟠 **Hornshark Island pre-placed units.** Was failing
  every Hornshark replay with src_missing on cmd[0]. Implemented
  `[switch] / [case] / [fire_event] / [set_variable] / [lua] /
  [unit]` action handlers in `scenario_events.py`. Recovered
  ~120 Hornshark replays.

- [x] 🟠 **village_gold extraction wired through.** Builder was
  hardcoding 2; now reads from per-side `village_income`
  (extracted from `[side] village_gold=` attribute). Also added
  `village_support` extraction. Default Era uses 5/1; Hornshark
  often 3/1; varies by host.

- [x] 🟠 **AMLA emits [choose] value=N too.** Our
  `_maybe_advance_unit` AMLA branch wasn't popping from
  `_advance_choices`, so a stale value left in the queue would
  be consumed by the NEXT unit's REAL advancement. Fixed; one
  weapon_oob case resolved.

- [x] 🟠 **Move-path ambush truncation.** Added in
  `_apply_command` for "move": walk path, stop at first enemy
  hex (with overlap-backoff for friendly path-passthrough),
  zero remaining MP. Diff_replay's path_enemy_blocking check
  removed (over-counted legitimate fog ambushes as sim bugs).

---

## NEW since 2026-04-30 (post scenario-pivot)

- [ ] 🔴 **`sim_to_replay` should build the bz2 from scratch, not
  splice onto a source replay's bz2.** Today it inherits the
  source replay's `[scenario]` (with that source's leaders +
  factions) and just swaps in our `[replay]` commands. After
  the scenario-pool pivot (2026-04-30), self-play games have
  random factions/leaders that don't match the source bz2 --
  the user observed exporting an "Undead vs Loyalists" sim run
  and having Wesnoth load it as "Red Mage vs Deathblade" because
  that's what the source Arcanclave bz2 had. Every command then
  diverges from turn 0 onward.

  Proper fix: emit the full Wesnoth save WML from
  `tools.scenario_pool.build_scenario_gamestate`'s output +
  `sim.command_history`, sourcing the `[scenario]` block from
  `wesnoth_src/data/multiplayer/scenarios/2p_*.cfg`. No
  `replays_raw/*.bz2` involvement.

  Scope: needs `[savegame]` / `[snapshot]` outer wrappers,
  `[multiplayer]` / `[era]` blocks, full per-side `[side]`
  attrs, scenario `[unit]` blocks for any pre-placed units (CoB
  petrified, etc.), `[time]` / `[music]` sections, then the
  `[replay]` block with our commands. Multi-hour rewrite.

  Until then, `tools/sim_demo_game.py` exports are unreliable
  for validation. Self-play TRAINING is unaffected (no bz2
  export in the training path; only the demo + audit paths
  use sim_to_replay).

## NEW since 2026-04-29 review

Items surfaced during the supervised-resume + self-play infra
sessions that aren't in the original review.

- [x] 🟠 **Sim invariant bug: hex (0,0) duplicate Unit instances**
  (DONE 2026-04-30, commit 2bb7f00). Fixed by giving `Unit` an
  explicit `__eq__` matching `__hash__` on `(id, side)` only
  (classes.py:179-182, with `@dataclass(eq=False)` to suppress
  auto-eq). Now `discard(old) + add(new)` correctly hits the same
  set bucket whichever order the caller uses, and we can't have
  both old and new resident simultaneously.

- [ ] 🟠 **MCTS tree-search proper** -- now unblocked by the
  predict_priors / value-head / soft-target / recruit-enrichment
  quartet. `tools/mcts.py` already exists; PUCT selection,
  Dirichlet root noise, batched leaf eval are the remaining
  engineering. Estimated 1-2 weeks per project_state roadmap.

- [ ] 🟢 **Diagnostic in supervised resume**: list missing-key
  set in a structured way so a regression that adds a SECOND
  unexpected missing key (beyond `type_head.*` and
  `dynamic_flag_proj.weight`) is loud, not just a warning the
  operator might miss. Today the warning lists keys but no
  whitelist of "expected missing"; one-line addition: assert
  that `set(missing) ⊆ EXPECTED_NEW_KEYS` else hard-error.

## TOP PRIORITIES (rank-ordered)

These are the items where time spent has the highest payoff for the
stated goals.

- [x] 🔴 **Distribution-output API + soft-target loss** (DONE
  2026-04-28 + 2026-04-29). Two complementary surfaces:
  `enumerate_legal_actions_with_priors(encoded, output, gs)` returns
  the FLAT per-action prior list with multiplied joint probabilities
  (commit 5ee3b5d, MCTS-blockers) -- handy for "expand the root /
  what are all my options". `predict_priors(output, encoded, gs, *,
  decision_step, masks=None)` returns the FACTORED tensors as
  `ActionPriors(actor, type_, target_attack, target_move,
  target_recruit, weapon, value, masks)` with each conditional summed
  to 1 over its legal support and 0 elsewhere (commit 2314ec3, this
  session) -- the format the soft-target trainer + PUCT incremental
  expansion want. Soft-target loss is `Trainer.step_mcts` /
  `_mcts_factored_policy_loss` (commit 151392b, C.4) with 4-tuple
  legacy + 5-tuple type-aware visit-count schemas both supported.

- [x] 🔴 **Value head over global token + tanh + return clamp**
  (DONE 2026-04-28, commit 5ee3b5d). Value head reads only
  `global_ctx.squeeze(1)`; tanh-bounded to [-1,+1]. Trainer's
  `value_clip=1.0` matches.

- [x] 🔴 **Recruit token enrichment** (DONE 2026-04-28, commit
  5ee3b5d). Recruits get `type_emb + side_emb + pos_x_emb +
  pos_y_emb + unit_feat_proj(phantom)` where the phantom is full-HP
  / 0-MP / 0-XP / non-leader / leader's-keep. Was the diagnosed root
  cause of "no recruits".

- [x] 🔴 **Record + emit `[choose]` for advance-on-mid-attack-XP-cross
  in the simulator** (DONE 2026-04-28). `WesnothSim.step()` now
  snapshots attacker/defender (id, name, side) before each attack and
  detects post-`_apply_command` whether either's name changed
  (= advanced); `RecordedCommand.extras["advance_choices"]` carries the
  per-side index list (always 0 -- sim picks `targets[0]`).
  `_build_replay_wml` emits one `[command] dependent="yes" [choose]`
  block per advancing unit immediately after the attack's
  `[random_seed]` follow-up, in attacker-first / defender-second order
  per `attack_unit_and_advance` (attack.cpp:1556-1573). Covers BOTH
  kill-based AND damage-based threshold-crossings (the detection looks
  at id+name, not death). Tested in `test_sim_advance.py` (6 cases).

- [x] 🔴 **`cluster/gui.pyw` Train + Display buttons re-routed
  through the simulator** (DONE 2026-04-28). Train calls
  `tools/sim_self_play.py`; Display calls a new
  `tools/sim_demo_game.py` (one-game inference + bz2 replay export
  via `sim_to_replay.export_replay`, watchable in Wesnoth's replay
  viewer). Both auto-pick the freshest `supervised*.pt` from
  `training/checkpoints/`. Display warns + bails early if no
  checkpoint is present.

- [x] 🔴 **`cluster/job_selfplay.sbatch`** (DONE 2026-04-29, commit
  a86e11b). Mirrors supervised's chain pattern (ENSTA-l40s, 1 GPU,
  03:55h walltime, auto-resubmit). Latest-checkpoint pick: existing
  self-play ckpt → highest `supervised_epoch*.pt` → `supervised.pt`
  → random init. No automatic done; chain runs until user drops
  `training/checkpoints/selfplay_done.flag`. `--workers 6` rollout
  threads, `--device cuda`, `--reward-config
  cluster/configs/reward_selfplay.json`. Companion: `cluster/run.sh`
  is now polymorphic (`run.sh start [supervised|selfplay]`); GUI has
  Start buttons for both modes.

- [x] 🟠 **`pull_checkpoint.ps1` made atomic** (DONE 2026-04-28). scp
  now writes to `<LOCAL>.tmp`; on success, `Move-Item -Force` swaps it
  over `<LOCAL>` (same-volume NTFS Move-Item =
  MoveFileEx(MOVEFILE_REPLACE_EXISTING) which is atomic at the FS
  level). On scp failure the .tmp is cleaned up. Concurrent self-play
  readers now see either the old file in full or the new file in full,
  never a torn half-written buffer.

- [x] 🟠 **`select_action` state-snapshot contract documented + asserted**
  (DONE 2026-04-28). Big block-comment in `select_action`'s docstring
  explains the two failure modes (trainer re-forward divergence /
  zero-delta rewards). Debug-mode tripwire: per-(game_label, side)
  `_last_state_id` records id() of the most recent state; if the
  next call passes the same id, RuntimeError. Cleared in
  `drop_pending`. Fires under default Python (skipped under `python
  -O`). Test in `test_sim_advance.py`.

---

## Open: late-turn Arcanclave OOS, root cause unknown

(2026-04-28) After fixes for `oos_debug`, override-overlays
(`Chw^Xo`), recruit gold-check, lazy-RNG-seed gating, leader
upkeep, and dummy determinism — the Arcanclave dummy export still
OOSes around turn 22 ("found corrupt movement in replay").

User-visible: "Last valid: turn 22 side 2 Deathblade WML(44,1)→(45,1).
Then probably tries WML(46,1) which is occupied by a Cavalryman."

Verified facts about the export at the moment of the failure:
- Deathblade is at WML(44,1) at end of side-1's turn 22 (correct).
- Side-1's u4 Cavalryman is at WML(46,1) (correct).
- Our cmd 400 (the one that fails per user) is `x="45,45"
  y="1,2"` — internal (44,0)→(44,1), WML (45,1)→(45,2). Target is
  (45,2) which is `Gs` (grass) and unoccupied per our state.
- Wesnoth-faithful audit (terrain costs, MP tracking, target
  occupancy, with seeds correctly fed back from `[random_seed]`
  follow-ups so trait rolls match the sim) reports **0 issues
  across 463 commands**.

Hypotheses to investigate (each requires Wesnoth-side debug data
the user-side GUI doesn't surface):
1. Wesnoth's view of u4 Cavalryman's position differs from ours
   — some move earlier than turn 22 was rejected in Wesnoth's
   view but accepted by our sim. Cumulative drift puts u4 at
   (45,2) by turn 22 rather than (46,1).
2. Fog-of-war / `cache_hidden_units` finds an obstruction that
   our state doesn't have. Arcanclave has `mp_fog=yes`. Our sim
   doesn't model fog visibility per-side.
3. Wesnoth's `plot_turn` does something for `(45,1)→(45,2)` we
   don't (ZoC at start? skirmisher edge case? terrain alias
   chain we still mishandle?).

Next debug step: add to `tools/sim_to_replay.py` an option to
emit `[checkup]` blocks containing post-attack `attacker_hp /
defender_hp` and post-move `final_hex_x/y` per real-replay format.
With strict-sync `oos_debug=yes` set in our export AND `[checkup]`
emission, Wesnoth would tell us which command's checkup
mismatched — pointing directly at the divergence. Today we strip
oos_debug to no, which papers over divergence into "corrupt
movement" without naming it.

**Workaround for now**: filter Arcanclave-source replays from
the self-play pool. Dummy-Fallenstar and dummy-Aethermaw both
load and run cleanly. We have enough map coverage to proceed
with self-play training; revisit Arcanclave when we can get
Wesnoth-side debug data.

---

## Open: Arcanclave Citadel dummy-game OOS (earlier diagnoses)

After the `oos_debug="yes"` rewrite (commit cd35979) and the
override-overlay fix, Arcanclave Citadel is in a partially-working
state:

- **`SIM_arcanclave_minimal.bz2`** (hand-built, init_side+end_turn
  only) — loads cleanly. ✓
- **`SIM_dummy_arcanclave.bz2`** (DummyPolicy full game) — OOSes at
  turn 1 side 2 with **"found dependent command in replay while
  is_synced=false"** after the first Skeleton recruit at WML(3,25).

The error fires at `wesnoth_src/src/replay.cpp:849`: our
`[random_seed]` follow-up arrives while Wesnoth's `is_unsynced` flag
is still set, meaning the previous `[recruit]` command was rejected
before entering synced context. So Wesnoth's `find_recruit_location`
is rejecting our `[recruit]` for some reason we haven't pinned down.

Verified facts:
- Both recruit hexes (3,25) and (4,24) are part of the leader's
  92-hex castle network from keep WML(2,24) (BFS confirmed).
- Side 2 has 100g, Skeleton costs 15g — gold is fine.
- Same recruit format works on Fallenstar Lake and Aethermaw.
- Recruit emission, [from] block, [random_seed] format identical
  to a verified-working real Arcanclave replay (Turn_1_94211).

Possible causes (not yet investigated):
1. Wesnoth's recruit event handler for Arcanclave fires synced RNG
   we're not accounting for (no `recruit` events in the source's
   [scenario] / [era] blocks though).
2. The 92-hex castle network includes a SECOND keep at WML(6,24);
   `find_recruit_location` may not handle the "multiple keeps in
   one network" case the way our analysis assumes.
3. Some [side] / [scenario] attribute we're inheriting differs
   subtly from Fallenstar/Aethermaw and triggers a recruit-time
   user_choice we haven't seen.

**Workaround**: filter Arcanclave out of self-play initial-state
pools for now; 2/3 maps working is sufficient to proceed. Add
back when root cause is found.

To debug next: launch Wesnoth via subprocess on
SIM_dummy_arcanclave.bz2 with `--log-debug=replay` and capture which
exact hex the recruit fails on (my CLI attempts hit a load-side
"corrupt file" snag — the user's GUI-load test showed only the
high-level error message, not the per-line trace). Or instrument
the simulator to ALSO run the same recruit through Wesnoth
source's check logic via Lua harness.

---

## Local simulator (`tools/wesnoth_sim.py`, `tools/sim_to_replay.py`, `tools/replay_dataset.py`, `tools/traits.py`, `combat.py`, `tools/scenario_events.py`)

The faithfulness of this layer matters most: every divergence from
Wesnoth becomes either a wrong gradient in self-play or an OOS in
replay export.

### Bit-exactness

- [x] 🔴 **Advance-on-mid-attack-XP-cross export** — DONE 2026-04-28; see
  top-priority list. Covers kill-based AND damage-based threshold
  crossings (detection compares pre/post by unit id+name).

- [x] 🟠 **Recall actions guarded at sim entry** (DONE 2026-04-28).
  `WesnothSim._action_to_command` now rejects recall actions outright
  (returns None -> step() falls back to end_turn). Eliminates the
  whole class of broken `[recall]` WML the exporter would otherwise
  emit. Today's action_sampler doesn't produce recall actions; the
  guard is dormant insurance.
  Long-term: model a recall list in `gs.global_info`, populate on
  level-up of dying-side units, expose recall slots to the encoder
  / sampler / exporter.

- [x] 🟠 **Plague spawn type fixed** (DONE 2026-04-28; corrected
  course-mid-commit). Original commit used attacker's `parent_id`,
  which would have spawned `Soulless:mounted` from a Soulless plague
  kill. Wrong: WEAPON_SPECIAL_PLAGUE
  (wesnoth_src/data/core/macros/weapon_specials.cfg:47-57) hardcodes
  `type=Walking Corpse`, so EVERY default-era plague kill spawns a
  Walking Corpse regardless of attacker (WC, Soulless, Necromancer,
  any variation). attack.cpp:159-164 only falls back to parent_id
  when the special's `type=` is empty, which never happens for the
  canned macro. Fixed: `base_type = "Walking Corpse"` hardcoded;
  variation still comes from dead's `undead_variation`. Custom
  plague (Ant Queen -> Giant Ant Egg via PLAGUE_TYPE macro) is not
  modeled -- mainline 2p doesn't use it.

- [ ] 🟠 **Combat damage seed alignment is correct today** but verify with
  a smoke test once the model attacks meaningfully. Each attack:
  `_action_to_command` (`tools/wesnoth_sim.py:586`) allocates one
  `request_seed(N)`; the exporter emits the same hex via `[random_seed]
  request_id=N`. Wesnoth's `mt_rng::seed_random(seed_str, 0)` then
  produces the same stream as `combat.MTRng(seed_hex)`. Confirmed
  correct in code; needs an end-to-end smoke against a real Wesnoth
  replay where damage rolls differ between branches.

- [x] 🟢 **VERIFIED NON-ISSUE: trait RNG is NOT off-by-one for facing
  draw.** A reviewer flagged that `tools/traits.py:274` only consumes the
  gender draw before traits and missed `unit::init`'s `facing_` draw at
  `wesnoth_src/src/units/unit.cpp:721`. Verified: line 721 uses
  `randomness::rng::default_instance()` which is a separate
  non-synced `mt19937` (`wesnoth_src/src/random.cpp:35-54`); only line
  720 (gender) and `generate_traits` use the synced `generator`. Our
  trait code is correct as-is.

- [ ] 🟡 **Move command always emits 2-hex paths**
  (`tools/wesnoth_sim.py:570-573`, `tools/sim_to_replay.py:108-118`).
  Today the policy can only emit adjacent moves so this works, but
  Wesnoth replay stores full multi-hex paths and fires `enter_hex`
  events on intermediate hexes. Document the invariant; if we ever
  expose multi-hex moves to the policy, the recon engine will follow
  the straight-line interpretation (sometimes wrong) and miss
  enter_hex events on custom maps.

- [ ] 🟡 **Fog and shroud are not updated post-move/attack**
  (`tools/replay_dataset.py`). Default 2p has fog/shroud disabled so
  this is moot for the current eval set, but exporting on a fogged
  scenario will OOS on the first move that "should have" cleared fog.
  Audit: grep a real `replays_raw/*.bz2` for `[clear_shroud_uncovers_unit]`
  to confirm whether 2p replays carry these dependents.

- [x] 🟡 **Fog hexes retained in encoder** (DONE 2026-04-28). The
  encoder previously dropped `gs.map.fog` hexes from the hex token
  stream, silently making them ineligible for the recruit mask
  (BFS over the leader's castle network would skip fog castles).
  Per the legality-mask contract in CLAUDE.md, fog hexes ARE
  attemptable -- the rejection-history feature handles the
  bounce case. Token sequence length bumps from ~180 to ~256 on
  default 2p mid-game; negligible cost.

### Mutation / consistency hazards

- [x] 🟠 **`_action_to_command` no longer mutates caller's action**
  (DONE 2026-04-28). Returns `(cmd, terrain_cost)` tuple. step()
  reads terrain_cost from the return value, not from
  `action["_terrain_cost"]`. Eliminates the side-effect class of
  bug.

- [x] 🟡 **`_rebuild_unit` helper consolidates the pattern** (DONE
  2026-04-28). New `_rebuild_unit(unit, **changes)` returns a NEW
  Unit copy with `**changes` applied to dataclass fields, preserving
  the `_`-prefixed setattr stash. `_replace_unit` (in-place modify)
  delegates to it. All five open-coded sites in `replay_dataset.py`
  (healing, end_turn slow-clear, recruit spawn, plague spawn) +
  `_deduct_extra_mp` in `wesnoth_sim.py` now route through it.
  Future refactors can't silently drop `_defense_table` (feral
  override) or any other stashed attribute.

### Performance / scaling for MCTS

- [x] 🟡 **`_deduct_extra_mp` uses discard/add** (DONE 2026-04-28).
  Find target -> early-return when extra<=0 or already-clamped ->
  build replacement Unit -> discard old + add new. O(1) on average
  vs O(N) per call; ~30× speedup on a 30-unit mid-game state. Same
  logic preserves the `_`-stash (e.g. `_defense_table`).

- [x] 🟡 **Map.__deepcopy__ aliases immutable fields** (DONE
  pre-2026-04-29, classes.py:176). `mask`, `fog`, `hexes` aliased
  as immutable in self-play; `units` rebuilt as a fresh set so
  add/remove on one copy doesn't leak. ~10× speedup. Slow-path
  `Map.deep_clone()` available for terrain-mutating scenarios
  (Aethermaw morph, etc.).

- [x] 🟡 **Sim post-step invariants** (DONE 2026-04-28).
  `WesnothSim._assert_invariants` runs under `__debug__` after every
  `step`: HP in [0, max_hp], MP in [0, max_moves], no two units on
  the same hex, ≤1 leader per side. Each violation raises with
  unit id, name, side, command, turn -- enough context to debug.
  Already caught a sloppy unit setup in our own test fixtures
  (Spearman with current_hp=80 against max_hp=42). Skipped under
  `python -O`.

### Exporter

- [x] 🟡 **WML escaping for unit names** (DONE 2026-04-28).
  `_wml_quote` helper escapes `\` → `\\\\` and `"` → `""`.
  Applied to recruit + recall emission. Default era doesn't have
  units with special chars; this is dormant insurance for custom
  eras.

- [x] 🟡 **`_find_final_replay_block` regex-anchored** (DONE
  2026-04-28). Two `re.compile(r"^\[/?replay\]\s*$", MULTILINE)`
  patterns + the LAST open match + first close after it. A literal
  `[replay]` inside a quoted `[message]` body can't fool the splice
  anymore.

- [x] 🟡 **Encoding errors='replace' + warning** (DONE 2026-04-28).
  Decode to text via `errors="replace"` (preserves byte offsets
  via U+FFFD substitution) and log a warning if any FFFD landed.
  Old `errors="ignore"` could shift offsets by dropping bytes,
  breaking the [replay]-block splice silently.

- [ ] 🟢 **Optional `[checkup]` debug emission** — let the user flip a
  flag and emit `attacker_hp/defender_hp` checkup blocks so OOS
  divergences point at the exact diverging strike. Cheap diagnostic.

### Tests

- [x] 🟠 **Sim determinism test** (DONE 2026-04-28). New
  `test_sim_determinism.py` (4 cases): two DummyPolicy runs from the
  same `replays_dataset/*.json.gz` produce equal `state_key`s,
  byte-equal `command_history`, equal `_rng_requests` counters, and
  `WesnothSim.fork()` doesn't perturb the parent state. Necessary
  precondition for MCTS branching: rolling out from the same
  position must give the same result every time.

- [x] 🟠 **Reward unit tests** (DONE 2026-04-28). New
  `test_rewards.py` (31 cases): each `StepDelta` field exercised via
  hand-built (prev, new) GameState pairs (enemy/our HP & gold, village
  ±1, recruit-success / recruit-fail / zero-cost-fallback, leader
  moved / static / dying, invalid-action gate via static state +
  side-swap clear), plus `WeightedReward` summation paths (terminal
  outcome contributions, signed gold-killed-delta, leader-move
  penalty, invalid-action penalty, distance penalty), plus
  `_action_had_visible_effect` coverage. Locks in the comments about
  "0.22 flat-return plateau" / "400+ invalid recruits" so future
  refactors can't silently regress shaping math.

- [x] 🟡 **`hex_distance` test** (DONE 2026-04-28; folded into
  `test_rewards.py`): identity, symmetry, six-neighbour adjacency
  for both even and odd columns (covers the odd-q parity edge case
  from the docstring), long-range row/column.

- [ ] 🟡 **End-to-end round-trip test**: take a real replay, run our
  recon, re-export from `command_history`, diff WML token-stream
  against source. Hard regression net for the bit-exactness work.

---

## Model architecture (`model.py`, `encoder.py`, `action_sampler.py`, `transformer_policy.py`, `trainer.py`)

### Bugs (silent corruption)

- [x] 🟠 **`reforward_logprob_entropy` actor_idx bounds-check** (DONE
  2026-04-28). Explicit `if actor_idx >= len(encoded.unit_ids)`
  early-return with a debug log. Was previously masked off by
  the `weapon_idx is None` short-circuit; now guarded explicitly so
  a future change that gives recruits a weapon idx can't IndexError
  silently.

- [x] 🟡 **Trainer pass-1/pass-2 eval mode consistency** (DONE
  2026-04-28). Both passes now run inside the same eval() block
  (try/finally restores caller's mode); the value-for-advantage
  baseline and the value-for-MSE target see identical activations.
  Old code only set eval() for Pass 1; Pass 2 inherited train(),
  introducing dropout=1e-4 noise between the two value forwards.

- [x] 🟡 **Trainer random subsampling** (DONE 2026-04-28). Replaced
  uniform-stride subsampling with `random.sample(range(N), cap)`
  (sorted to preserve trajectory order for any sequential logic).
  Stride preferentially kept near-terminal transitions (which have
  ~7× the gamma-discounted return weight at gamma=0.99 over a
  200-step trajectory); random sampling keeps the sub-batch's return
  distribution unbiased relative to the full batch in expectation.

- [x] 🟡 **Fog castle hexes legal for recruit** (DONE 2026-04-28).
  Encoder retains fog hexes; `_recruit_hex_mask` no longer
  silently drops them. Per the legality-mask contract, fog castle
  hexes ARE legal recruit targets -- the model can attempt; the
  sim's god-view occupancy check + harness retry loop handle the
  bounce case via the per-turn rejection set. Recruit affordability
  also gated at the mask level (unaffordable unit-type slots get
  actor_valid=0).

### Architecture for MCTS / superhuman play

- [x] 🔴 **Distribution-output API + soft-target loss** — see top
  section.

- [x] 🔴 **Value head over global token + tanh** — see top section.

- [x] 🔴 **Recruit token enrichment** — see top section.

- [x] 🟠 **Per-actor action-type head (`ATTACK` vs `MOVE`)** (DONE
  2026-04-28 in C.1, commit 635691f). 2-way (`UnitActionType.ATTACK
  / MOVE`) head per actor, masked by `type_valid` and biased by
  `type_bias`. Sampler chain is actor → type → target → weapon for
  unit actors. RECRUIT / END_TURN actors skip the type sample (their
  type is implicit). HOLD intentionally not modeled (see
  CLAUDE.md).

- [ ] 🟡 **Cliffness adaptive sim budget — calibration pending**
  (2026-05-10, updated 2026-05-11). The framework is wired in
  `tools/mcts.py` (`MCTSConfig.adaptive_sim_budget`,
  `n_simulations_min`, `n_simulations_max`, `cliffness_max=0.577`
  ≈ 1/√3, the std of the continuous uniform on [-1, +1]; the
  discrete uniform on K=51 atoms matches it to 3 decimal places
  — see
  `test_distributional_value.test_cliffness_high_when_distribution_spread`).
  Defaults `n_simulations_min=100, n_simulations_max=400` are
  uncalibrated and shipped OFF (`adaptive_sim_budget=False`).
  Collection tool landed 2026-05-11: `tools/collect_cliffness.py`
  walks a sample of real replays, computes cliffness per decision
  step, and writes a percentile/histogram report to
  `docs/cliffness_calibration.md` plus calibration recommendations.
  **Blocked on a C51-trained checkpoint** -- all existing
  checkpoints (`supervised_epoch3.pt`, `sim_selfplay.pt`,
  `checkpoint_*.pt`) predate the C51 head landing 2026-05-10 and
  have a 1-scalar value head; the tool falls back to a random
  C51-head init in that case, producing only the uniform-prior
  baseline (clusters tight around 0.59, exactly the predicted
  uniform std). Re-run after the next supervised or self-play
  cycle finishes with the C51 head trained. Until calibrated,
  callers should leave `adaptive_sim_budget=False`.

- [ ] 🟡 **Cliffness similarity-hashing TT (lossy / soft TT)**
  (2026-05-10, deferred from the cliffness consumer pass). The
  exact-match TT in `tools/mcts.py` only fires when `state_key`
  collides exactly. In Wesnoth this hits ~0.4% per the
  `bench_mcts_tt.py` measurement: HP, MP, gold, village
  ownership all change every action, so true collisions are
  rare beyond intra-turn move-reorderings. A lossy hash
  (e.g. drop exact HP and bucket; ignore MP for end-of-turn
  states; ignore gold deltas under a threshold) would let
  "approximately equal" states share value estimates — the
  similarity error introduced by the lossy hash is exactly what
  cliffness should gate (high cliffness = state value is
  sensitive to small differences = don't transfer; low
  cliffness = transfer is safe). Was tried first as
  cliffness-gated EXACT TT, which was incoherent (exact
  matches ARE the same state, no error to gate); reverted
  2026-05-10. Re-attempt will need: (a) a real similarity hash
  (which fields drop, how to bucket continuous values like
  HP/MP/gold), (b) what to share when fingerprints collide
  (Q values? visit counts? prior-blended subtree?), (c) how to
  blend cached vs fresh leaf-v under cliffness. Each is its
  own design question; the whole feature is a separate project
  rather than a flag.

- [ ] 🟡 **`register_names` mutates encoder vocab during rollout**
  (`encoder.py:289`). On checkpoint resume with a never-before-seen unit
  type, the new id may collide with an embedding row. Pre-seed from
  `tools/scrape_unit_stats.py` output and freeze post-pretrain (warn if
  a new type shows up). PARTIAL (2026-04-28): vocab overflow now
  warns once per (unit-type / faction) on registration, surfacing
  the silent encode-time clamp at the source. Pre-seeding from
  scrape + freeze still TODO.

- [x] 🟡 **MAX_UNIT_TYPES overflow documented** (DONE 2026-04-28).
  Encoder.py docstring spells out: 200 covers default era + typical
  expansions, overflow clamps to id 199 (silent alias), watch the
  encoder's overflow log on first epoch -- if it fires, retrain
  with a larger MAX_UNIT_TYPES (changing it requires a fresh
  model). Linked to the `register_names` warn-once path.

- [x] 🟡 **Encoder NORM constants moved to constants.py** (DONE
  2026-04-28). `HP_NORM`, `MOVES_NORM`, `EXP_NORM`, `COST_NORM`,
  `GOLD_NORM`, `INCOME_NORM`, `VILLAGES_NORM`, `TURN_NORM` now
  live in `constants.py` with a comment block explaining the
  scale rationale. `encoder.py` re-exports them for backwards
  compatibility. Era mods can override in one place.

### Performance

- [x] 🟡 **`pos_to_hex` cached on EncodedState** (DONE 2026-04-28).
  `EncodedState.pos_to_hex: Dict[(x,y), int]` populated once during
  encode(); `_build_legality_masks` reads from there. Saves the
  per-call rebuild (~1ms on 250-hex states; matters for MCTS).

- [x] 🟡 **Trainer Pass-2 raw-encoded cache** (DONE 2026-04-29,
  commit 5cad46b). Both REINFORCE `step` and `_trainer_step_mcts`
  build a `RawEncoded` list once at the top of train_step, then
  call only `encode_from_raw` per chunk per pass. Caching the RAW
  (numpy) and re-running encode_from_raw matters: encode_from_raw's
  output is autograd-bound to the encoder's CURRENT embedding
  parameters, and Pass 1 (no_grad) produces tensors with no graph
  -- reusing them in Pass 2 would zero the gradient. Suite went
  ~36s → ~33s.

- [ ] 🟢 **`recruit_is_ours.detach().cpu().numpy()[0]`** every decision
  (`action_sampler.py:499`). Crosses device boundary for a tiny tensor.
  Add the numpy version to `EncodedState` and reuse.

### Stability / safety

- [x] 🟠 **`train_step` snapshot-for-inference** (DONE 2026-04-28).
  TransformerPolicy now holds TWO model instances: `_model` /
  `_encoder` (mutated by train_step) and `_inference_model` /
  `_inference_encoder` (read by select_action). After each
  train_step's gradient compute, `_snapshot_inference_weights`
  atomically copies `state_dict` into the inference copies under
  `_snapshot_lock`. Rollouts hold the same lock during their
  forward — short enough (~30 ms) that the trainer's snapshot
  waits a tiny moment but never reads torn parameters. Vocab
  dicts (`unit_type_to_id`, `faction_to_id`) shared by reference;
  they're append-only so no race. load_checkpoint also syncs the
  inference snapshot. 7 tests in `test_inference_snapshot.py`
  cover structural separation, snapshot semantics, and a
  multi-thread stress (rollouts + train_steps in parallel: no
  NaN, no exceptions).

- [x] 🟡 **Checkpoint partial loads (`strict=False` default)** (DONE
  2026-04-28). `load_checkpoint(strict=False)` tolerates mismatched
  submodules and logs missing/unexpected keys. Top-level `arch`
  fields (d_model, num_layers, ...) still hard-error -- those
  dimensions can't be partially loaded. `strict=True` restores the
  prior behavior.

### Customizability gaps

- [x] 🟡 **`COMBAT_LOGIT_ALPHA` moved to constants.py** (DONE
  2026-04-28). action_sampler now imports the constant; "pacifist"
  / "berserker" tuning is a config flip in one place.

- [ ] 🟢 **Combat-oracle bias only on attacks, not on moves**
  (`action_sampler.py:467`). Adding move-toward-good-attack-position
  bias would help unorthodox-strategy training. Infrastructure already
  there.

- [x] 🟢 **`DEFAULT_FACTIONS` lives in constants.py** (DONE
  2026-04-28). Encoder re-exports `_DEFAULT_FACTIONS` from
  `constants.DEFAULT_FACTIONS`. Era mods extend in one place.
  Order documented as load-bearing for checkpoint compatibility.

---

## Reward shaping & customization (`rewards.py`, `tools/sim_self_play.py`)

The user explicitly wants behavior gated by config/data/reward shaping
rather than weights. Currently `WeightedReward` has fixed scalar
weights; there is no opener gating, no per-unit-type bonus, no
curriculum hook.

- [x] 🟠 **Per-unit-type recruit bonuses** (DONE 2026-04-28).
  `UnitTypeBonus(unit_type, weight)` dataclass, list field on
  `WeightedReward.unit_type_bonuses`. Stackable. `StepDelta.units_recruited:
  Tuple[str, ...]` populated by `compute_delta` on successful
  recruit. Tested with stacking + non-match cases.

- [x] 🟠 **Turn-conditional bonuses** (DONE 2026-04-28).
  `TurnConditionalBonus(name, turn_range, predicate, weight, once)`
  dataclass; `WeightedReward.turn_conditional_bonuses` list. Predicate
  gets `(post_state, side)`; bonus fires if predicate True and
  `delta.turn` in range. `once=True` (default) gates on
  `(game_label, side, name)` -- `WeightedReward.reset_game_state(label)`
  clears for game restart. Defensive: predicate exceptions caught,
  silent if `post_state` not attached. `compute_delta` opt-in via
  `attach_post_state=True` to avoid retention overhead in the common
  case. `game_label` propagation through `compute_delta` lets
  multi-game runs scope `once` correctly.

- [x] 🟠 **Opener-gating policy wrapper** (DONE 2026-04-28).
  `tools/openers.py` -- `Opener(name, moves, sides)` + `OpenerPolicy`
  wraps a base policy. Per-(game_label, side) cursor advances on each
  fired opener move; moves returning None delegate to base WITHOUT
  advancing (gate-retry semantics). Forwards `observe`,
  `train_step`, `save_checkpoint`, `load_checkpoint`,
  `drop_pending` duck-typed to base so trainable wrappers keep
  working. Built-in helpers: `recruit_type(name)` and `end_turn()`.
  TODO: register openers + expose `--opener-spec` in
  `sim_self_play.py` so cluster jobs can flip openers via CLI.

- [x] 🟠 **`sim_self_play.py --reward-config path` + `--opener-spec
  NAME`** (DONE 2026-04-28). JSON / YAML reward config via
  `rewards.load_reward_config`: scalar weights override defaults,
  `unit_type_bonuses` and `turn_conditional_bonuses` populate via
  list-of-dict, predicates resolve through `_PREDICATE_REGISTRY`.
  Built-in predicates: `leader_on_village`, `leader_on_keep`,
  `controls_majority_villages`, `no_units_lost`. Validates unknown
  scalar keys (catches typos at load time, not silently). Opener
  registry in `tools/openers.py` -- built-ins self-register
  (`just_end_turn`, `drake_rush`, `knalgan_thunder`). `--help` lists
  available openers. Per-game state resets (`reset_game`,
  `reset_game_state`) wired into `run_iteration`. `attach_post_state`
  conditional on whether the reward function has turn-conditional
  bonuses configured (avoid retention overhead in the common case).

- [ ] 🟡 **Replay-corpus filter: no rating / faction / scenario triplet**
  (`tools/replay_dataset.py:1673`, `tools/supervised_train.py:781-790`).
  Add `--min-rating`, `--factions`, `--scenarios` flags to
  `supervised_train.py`. `index.jsonl` would need a rating field —
  extend the indexer.

- [ ] 🟡 **`WesnothSim.from_replay_at_turn(target_turn)` for mid-game
  starts** (`tools/wesnoth_sim.py:351`). Today `from_replay` discards
  the command stream entirely and starts from turn 1. For curriculum
  training (sample mid-game positions), wrap `iter_replay_pairs` and
  stop applying commands when `turn_number == target_turn`.

- [ ] 🟡 **Mini-scenarios for training specific capabilities**. Build a
  small library of hand-crafted starting positions that isolate ONE
  skill each, so training can curriculum-mix them with full self-play
  and we can verify capability gain test-by-test. Each scenario is
  a constructed `GameState` (or a `[scenario]` WML for export) with
  a clear win condition, e.g.:

    - **kite-1**: 1 archer vs 1 melee on flat — learn to retreat
      after attacking, never let the melee close.
    - **village-grab**: leader + 1 grunt with 2 villages 4 hexes
      apart, no enemies — learn that capturing villages > standing on
      keep when no recruits are immediately useful.
    - **focus-fire**: 3 fighters vs 1 wounded enemy at HP=8 — learn
      to commit all three to finishing kills before they're healed.
    - **flank-zoc**: skirmisher vs ZoC chain — learn to use
      skirmisher's ZoC immunity to break a wall.
    - **ToD-time**: lawful unit vs chaotic unit, choice of attack at
      day or wait until night — learn ToD timing.
    - **leader-safety**: enemy fast unit 5 hexes from our leader —
      learn to retreat the leader when threatened.

  Sketch: `tools/mini_scenarios.py` exports a `SCENARIOS: Dict[str,
  Callable[[], GameState]]` registry, plus a CLI to evaluate a
  checkpoint per-scenario and print a capability score per skill.
  Plumbing into `sim_self_play` via `--curriculum-mix kite=0.1
  village-grab=0.1 ...` lets the trainer over-sample weak skills
  without giving up full-game training. Locks in unit-test-style
  capability checks (vs only end-to-end self-play scoring).
  Companion to the eval harness — eval measures vs the built-in AI;
  mini-scenarios measure vs a fixed pinpoint task.

- [x] 🟡 **Recruit cost fallback to 14 + warning** (DONE 2026-04-28).
  `cost_lookup.get(unit_type)` returns None on miss, falls back to
  14 (smallfoot/orcishfoot baseline), logs a warning ONCE per
  unknown unit type. Prevents the policy from learning to spam an
  unknown unit type that scored zero gold-cost shaping.

- [ ] 🟢 **`leader_move_penalty` is unconditional** (`rewards.py:149`).
  Promote it to a `TurnConditionalBonus` so it can be turn-bounded
  (currently penalizes ALL leader moves, even necessary ones in
  endgame).

---

## Subprocess Wesnoth / dead code

With the simulator landed, much of the subprocess pipeline is dead
weight. Eval still needs real Wesnoth, so be cautious.

### DEFINITELY DELETABLE

- [x] 🟢 **Deleted `add-ons/wesnoth_ai/lua/headless_plugin.lua`**
  (2026-04-29, commit d01f671).
- [x] 🟢 **Deleted `add-ons/wesnoth_ai/lua/headless_probe.lua`**
  (2026-04-29, commit d01f671).
- [x] 🟢 **Deleted `add-ons/wesnoth_ai/scenarios/training_scenario_mp.cfg`**
  (2026-04-29, commit d01f671). `_main.cfg`'s pointer comment
  also dropped.

- [ ] 🟢 **Auto-clean `add-ons/wesnoth_ai/games/g*/` dirs**. 1983 stale
  dirs locally (gitignored, but inode pressure). `turn_stage.lua` can't
  delete (Lua sandbox excludes `os.remove`). Add a Python cleanup pass
  in `main.py:check_setup` or at `WesnothGame` finalization.

### KEEP (still load-bearing)

- `wesnoth_interface.py`, `state_converter.py` — used by
  `tools/eval_runner.py`, `tools/eval_vs_builtin.py`. Eval still uses
  real Wesnoth.
- `add-ons/wesnoth_ai/lua/state_collector.lua`, `action_executor.lua`,
  `turn_stage.lua`, `json_encoder.lua` — eval pipeline.
- `_main.cfg`, `ai_config.cfg`, `training_scenario.cfg`, `eval/*.cfg`.

### CONDITIONAL

- [ ] 🟡 **`game_manager.py`** — only entered via `main.py`'s training
  /display path. If `main.py --display` is no longer wanted (we have
  bz2 replay export now), DELETABLE.

- [ ] 🟡 **`main.py --display` mode + `training_scenario_display.cfg`** —
  only useful for watching a trained model in real Wesnoth. Replaced by
  `tools/sim_to_replay.py` + Wesnoth's GUI replay viewer. KEEP if you
  value live-watch; DELETABLE otherwise.

---

## Cluster (`cluster/*`)

- [x] 🔴 **`cluster/job_selfplay.sbatch`** — see top section.

- [x] 🟠 **`pull_checkpoint.ps1` atomic move** — see top section
  (DONE 2026-04-28).

- [x] 🟡 **`job.sbatch` sentinel computed from `EPOCHS`** (DONE
  2026-04-28). `EPOCHS=10` + `EPOCHS_LAST_SENTINEL=...epoch$((EPOCHS-1)).pt`
  block at the top; one place to bump both atomically. Bumping
  `--epochs` without updating the sentinel can no longer cause an
  infinite resubmit chain.

- [x] 🟡 **`sync.ps1` SHA-256 round-trip** (DONE 2026-04-29, commit
  d01f671). Local SHA-256 of every manifest file → packed in the
  tarball under `.wai_sync.sha256` → remote `sha256sum -c` after
  extract → fail loudly with the offending file's path on
  mismatch. Catches partial extraction, stream corruption, and
  tar header weirdness that the exit-code check missed.

- [x] 🟡 **GUI startup purges stale askpass dirs** (DONE 2026-04-28).
  `_purge_stale_askpass_dirs()` runs at the top of `main()` and
  recursively removes `%TEMP%/wai_gui_pw_*`. Idempotent +
  ignore-errors so concurrent GUI processes can't conflict.

---

## Eval harness (`tools/eval_*.py`, `add-ons/wesnoth_ai/scenarios/eval/*`)

- [x] 🟠 **GUI eval no_swap default flipped to False** (DONE
  2026-04-28). The GUI dialog now matches CLI semantics: side-
  swapping ON by default, eliminating first-mover bias from every
  eval run through the GUI.

- [x] 🟡 **Wilson 95% CI on eval win-rates** (DONE 2026-04-28).
  `_wilson_interval(wins, n)` returns the score-interval bounds;
  `_wr` formats `XX.X% (W/N, 95% CI lo%-hi%)`. Stays inside [0, 1]
  for small-n / extreme-p (textbook formula gives nonsensical
  bounds at 9/10 wins). 8 unit tests cover boundary cases (0/0,
  full sweeps, top-/bottom-clamp).

- [x] 🟢 **Per-faction × per-faction win-rate heatmap** (DONE
  2026-04-29, commit d01f671). 2D matrix view above the existing
  per-matchup list: rows = our faction, columns = opp faction,
  cells = compact `WW.W% (W/N)` via `_wr_short` helper. 3 new
  tests exercise the formatter + capsys end-to-end render.

---

## Tests (`test_integration.py`, `test_lua_actions.py`)

The current tests exercise the dead Wesnoth-IPC path, not the live sim
pipeline (per `CLAUDE.md`). Many high-value test gaps surfaced above;
collected here:

- [x] 🟠 **Sim determinism (sim)** (DONE pre-2026-05-11,
  `test_sim_determinism.py`, 4 cases): two runs from the same
  replay produce identical state_key trajectories; command
  history is byte-identical; RNG-counter advances consistently;
  `WesnothSim.fork()` doesn't perturb the parent's RNG counter.
- [x] 🟠 **Reward unit tests (`compute_delta` over hand-built
  pairs)** (DONE pre-2026-05-11, `test_rewards.py`, 58 cases):
  every `StepDelta` field exercised in isolation via hand-built
  (prev, new) GameState pairs; WeightedReward outcome terms,
  gold-killed signed-diff, leader-move-penalty charged once,
  invalid-action penalty, distance-scaling; `hex_distance`
  identity / symmetry / adjacency / long-range; visible-effect
  predicate on position / hp / has_attacked flips.
- [x] 🟠 **Replay round-trip (recon → re-emit → diff)** (DONE
  2026-05-11, `test_replay_round_trip.py`, see entry below).
- [x] 🟡 **`hex_distance` parity edges** (DONE pre-2026-05-11,
  4 cases in `test_rewards.py`: identity / symmetry / adjacency /
  long-range — covers the odd-row parity offset).
- [x] 🟡 **state_key fuzz** (DONE 2026-04-28). New
  `test_state_key.py` (16 cases): discrimination on every
  meaningful field (HP/MP/XP/position/has_attacked/status/gold/
  villages/turn/side/village_owner/RNG counter); order-invariance
  on unit-set iteration (2-unit and 6-unit); deterministic across
  calls. Necessary precondition for MCTS transposition correctness.
- [x] 🟡 **sim_self_play reward-flow smoke** (DONE 2026-04-28). New
  `test_sim_self_play_smoke.py` (3 cases): observe called per step
  + per terminal; per-unit-type bonuses propagate end-to-end; once-
  per-game gating resets between games via the harness's
  reset_game_state hook.

---

## MCTS readiness scorecard (refreshed 2026-05-10)

| Capability | Status | Notes |
|---|---|---|
| Clone GameState | ✅ Map.__deepcopy__ ~0.05ms | classes.py:176 |
| Hash GameState | ✅ `state_key(gs)` covered | test_state_key.py 16 cases |
| Forward inference | ✅ predict_priors API + LegalActionPrior enumeration | action_sampler.py |
| Visit-count target | ✅ Trainer.step_mcts + factored CE loss | trainer.py |
| Determinism | ✅ via request_seed(N) counter + tests | test_sim_determinism.py |
| Branching cost | ✅ batched leaf eval via virtual loss | tools/mcts.py |
| Bounded value | ✅ distributional C51 head (K=51 atoms, [-1,+1]) | model.py + test_distributional_value.py |
| Recruit token signal | ✅ phantom-unit features | encoder.py |
| Tree-search loop | ✅ implemented + tested | tools/mcts.py + test_mcts.py |
| Turns-vs-actions sign-flip | ✅ `parent.side == leaf_side` per edge | test_mcts.py 5 cases |
| Wired into self-play loop | ✅ MCTSPolicy + `--mcts` flag + GUI toggle | tools/sim_self_play.py + cluster/gui.pyw |
| Transposition table | ✅ exact-match per-search TT | tools/mcts.py + test_mcts.py + bench |
| Cliffness signal | ✅ `output.cliffness = std(Z(s))` published | test_distributional_value.py |
| Cliffness bootstrap weighting | ✅ Bayesian-precision shrink in _backup (off by default, alpha=0) | test_mcts_cliffness.py |
| Cliffness adaptive sim budget | ✅ framework wired (off by default, n_min/n_max uncalibrated) | test_mcts_cliffness.py |
| Cliffness similarity-hashing TT | ❌ deferred (own design project; needs lossy hash) | this BACKLOG entry |
| Cliffness empirical calibration | ⚠️ collection tool landed (`tools/collect_cliffness.py`, `docs/cliffness_calibration.md`); first real data needs a C51-trained checkpoint | blocks on training cycle |
| MCTS-vs-REINFORCE comparative eval | ⚠️ harness landed (`tools/eval_mcts_vs_reinforce.py`, `docs/mcts_vs_reinforce_eval.md`); pre-C51 + passive-supervised checkpoint produces 10/10 draws (caveat in report); needs trained C51 checkpoint | blocks on training cycle |
| MCTS-vs-REINFORCE comparative eval | ⚠️ moved up (see above) | --- |

---

## Out-of-scope / observed but not ranked

- **Model is too passive at supervised_epoch3**: 51-turn self-play game
  produced 10 recruits and **0 movement**, only end_turns. The
  recruit-token enrichment + per-actor action-type head should help; if
  not, more cluster training cycles will.
- **Both sides share one trainer queue**: probably fine, but per-side
  advantage normalization may help under asymmetric reward shaping.
