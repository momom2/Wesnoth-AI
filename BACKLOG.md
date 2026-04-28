# Project review — bugs and improvements

Generated 2026-04-28 from a deep review of every major component. Items
are graded by impact on the project's stated goals (superhuman play via
MCTS+self-play; readable/customizable strategies; cluster economy).

**Severity:**
- 🔴 **CRIT** — silent corruption / wrong gradients / blocks a stated goal
- 🟠 **HIGH** — surfaces on the next nontrivial milestone (model attacks
  meaningfully / recalls / MCTS lands)
- 🟡 **MED** — correctness drift, polish, nice-to-have
- 🟢 **LOW** — cleanup, performance below noise floor

References are `path/file.py:line`. Each item is actionable on its own.

---

## TOP PRIORITIES (rank-ordered)

These are the items where time spent has the highest payoff for the
stated goals.

- [ ] 🔴 **Add a distribution-output API to `WesnothModel`**
  (`model.py:122,194`). Today `actor_head` is a per-slot scalar that the
  sampler softmaxes to sample one action; AlphaZero PUCT requires
  exposed normalized priors `P(a|s)` and AlphaZero training needs a
  KL-against-visit-counts loss target. Add `model.predict_priors(encoded)
  -> (actor_logits, target_logits, weapon_logits)` (essentially what
  `forward` already returns) AND a soft-target trainer path (`trainer.py`
  currently only trains against single-action targets). **Blocks MCTS
  entirely.**

- [ ] 🔴 **Pool the value head over the global `[CLS]` token, not over a
  mean of all tokens** (`model.py:194`). Today `value = mean(x, dim=1)`
  dilutes any global signal by ~1700×. The global token (`x[:, H+U+R, :]`)
  is built but never read for value or end_turn. Wrap with `.tanh()` and
  normalize returns to [−1,1] (`model.py:130`). High-leverage stability
  improvement that helps even today's REINFORCE training.

- [ ] 🔴 **Enrich recruit tokens with full unit features**
  (`encoder.py:385-387`). Confirmed: recruit tokens have only
  `unit_type_emb + side_emb`; they collide with on-board units of the
  same type and the actor head can't distinguish recruit options. Fix
  by synthesizing a phantom Unit per recruit (full HP, 0 XP, position =
  leader's keep) and feeding it through the existing `_unit_features`
  projection. The user already diagnosed this as the root cause of "no
  recruits"; the supervised loss reweighting we landed is a partial
  workaround — this is the structural fix.

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

- [ ] 🔴 **Add `cluster/job_selfplay.sbatch`** before the next cluster
  window (`cluster/job.sbatch`). The existing sbatch only runs
  `tools/supervised_train.py`; with self-play training landed locally
  we need a sibling job that runs `tools/sim_self_play.py` against a
  pulled checkpoint. Mirror `job.sbatch` but with a different `done.flag`
  signal (e.g. iteration count instead of epoch9 detection).

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

- [ ] 🟡 **Map.hexes is immutable per game; `deepcopy(GameState)` is
  ~0.5 ms** (current measured). Add `__deepcopy__` to `Map` that aliases
  immutable fields (`hexes`, `mask`) and only deep-copies units + fog.
  Drops to ~0.05 ms. Negligible for current self-play; matters for
  MCTS. Estimated MCTS bottleneck is model forward (~30ms), not
  deepcopy — but every ms helps.

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

- [ ] 🟡 **Fogged hexes silently disappear from the legality mask**
  (`encoder.py:471`, `action_sampler.py:404,556`). The encoder drops
  fogged hexes; `_recruit_hex_mask` walks `mods_by_pos` (full map) but
  the BFS results map back through `pos_to_hex` (visible-only), so a
  recruit hex hidden by fog is silently illegal. Fix: build `pos_to_hex`
  from full map, not encoded set; or stop dropping fogged hexes for
  the side that owns them.

### Architecture for MCTS / superhuman play

- [ ] 🔴 **Distribution-output API + soft-target loss** — top priority,
  see top section.

- [ ] 🔴 **Value head over global token + tanh** — top priority, see top
  section.

- [ ] 🔴 **Recruit token enrichment** — top priority, see top section.

- [ ] 🟠 **Per-actor action-type head (`attack` vs `move` vs `hold`)**
  (`action_sampler.py:179`). Today the actor decides "act?" then the
  target decides where, with attack/move discrimination implicit in
  whether the target hex contains an enemy. PUCT priors should
  distinguish "attack any enemy with this unit" from "fortify by
  retreating". Add a 3-way head per actor that gates the target
  distribution; existing masks already enforce legality.

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

- [ ] 🟡 **Trainer Pass-2 re-encodes every transition** (`trainer.py:211`).
  Pass-1 just encoded all of them. Cache encoded chunk between passes.
  ~30-50% trainer-step speedup.

- [ ] 🟢 **`recruit_is_ours.detach().cpu().numpy()[0]`** every decision
  (`action_sampler.py:499`). Crosses device boundary for a tiny tensor.
  Add the numpy version to `EncodedState` and reuse.

### Stability / safety

- [ ] 🟠 **`train_step` mutates weights without lock during rollout**
  (`transformer_policy.py:280`). The "GIL-protected tensor ops" comment
  is misleading: PyTorch's `optimizer.step()` is not a single
  GIL-atomic operation. A rollout reading weights mid-step can produce
  garbage outputs. Take a `state_dict` snapshot for inference, or
  switch to a queue-based architecture where workers reload weights
  only between train_steps. **Probably already manifesting as occasional
  noise in self-play.**

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

- [ ] 🟢 **Delete `add-ons/wesnoth_ai/lua/headless_plugin.lua`**.
  Dormant artifact from the abandoned `--nogui --plugin` experiment.
  Not on `_main.cfg`'s load path, no code imports it.

- [ ] 🟢 **Delete `add-ons/wesnoth_ai/lua/headless_probe.lua`**. Same
  reason.

- [ ] 🟢 **Delete `add-ons/wesnoth_ai/scenarios/training_scenario_mp.cfg`**.
  Marked "NOT loaded by _main.cfg by default" in its own header
  (`training_scenario_mp.cfg:1-15`).

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

- [ ] 🔴 **Add `cluster/job_selfplay.sbatch`** — top priority, see top
  section.

- [ ] 🟠 **`pull_checkpoint.ps1` non-atomic** — top priority, see top
  section.

- [x] 🟡 **`job.sbatch` sentinel computed from `EPOCHS`** (DONE
  2026-04-28). `EPOCHS=10` + `EPOCHS_LAST_SENTINEL=...epoch$((EPOCHS-1)).pt`
  block at the top; one place to bump both atomically. Bumping
  `--epochs` without updating the sentinel can no longer cause an
  infinite resubmit chain.

- [ ] 🟡 **`sync.ps1`: no checksum verification of extracted files on
  remote** (`cluster/sync.ps1:122-147`). On non-zero return code we
  warn "files may be partially synced" but `tar -xf -` is not atomic
  — partial extraction overwrites in place. Add a `sha256sum` round-trip
  for files in the manifest.

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

- [ ] 🟢 **Eval has no per-faction strength heatmap**. We track per-side,
  per-map. Adding a faction × faction matrix would let the user see
  "model loses to Drakes 80% of the time but beats Knalgan 60%".

---

## Tests (`test_integration.py`, `test_lua_actions.py`)

The current tests exercise the dead Wesnoth-IPC path, not the live sim
pipeline (per `CLAUDE.md`). Many high-value test gaps surfaced above;
collected here:

- [ ] 🟠 Sim determinism (sim).
- [ ] 🟠 Reward unit tests (`compute_delta` over hand-built pairs).
- [ ] 🟠 Replay round-trip (recon → re-emit → diff against source WML).
- [ ] 🟡 `hex_distance` parity edges.
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

## MCTS readiness scorecard

| Capability | Status | Action |
|---|---|---|
| Clone GameState | ✅ deepcopy works (~0.5ms) | optimize Map __deepcopy__ for ~10× |
| Hash GameState | ❌ no canonical content hash | add `state_key(gs)` in `classes.py` |
| Forward inference | ⚠️ usable but no public masked-distribution API | factor out from `action_sampler` |
| Visit-count target | ❌ trainer only knows single-action targets | add soft-target path |
| Determinism | ✅ via `request_seed(N)` counter | add explicit determinism test |
| Branching cost | ⚠️ ~30 evals/sec single-thread | batched leaf eval mandatory before scale |
| Bounded value | ❌ unbounded scalar | tanh + return normalization |

---

## Out-of-scope / observed but not ranked

- **Model is too passive at supervised_epoch3**: 51-turn self-play game
  produced 10 recruits and **0 movement**, only end_turns. The
  recruit-token enrichment + per-actor action-type head should help; if
  not, more cluster training cycles will.
- **Both sides share one trainer queue**: probably fine, but per-side
  advantage normalization may help under asymmetric reward shaping.
