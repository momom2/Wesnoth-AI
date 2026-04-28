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

- [ ] 🟠 **Plague spawns wrong unit type when attacker is a variation**
  (`tools/replay_dataset.py:1401-1408`). Uses `attacker.name` but
  Wesnoth uses the attacker's `parent_id` field. `Walking Corpse:mounted`
  killing a Wose should raise `Walking Corpse` with `undead_variation=wose`,
  not `Walking Corpse:mounted:wose`. Dormant — only triggers when Undead
  attack non-Undead units. Add `parent_id` extraction to
  `tools/scrape_unit_stats.py` and use it.

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

- [ ] 🟠 **`_action_to_command` mutates the caller's action dict**
  (`tools/wesnoth_sim.py:564`). `action["_terrain_cost"] = cost` is
  side-effecting. If a caller hashes or reuses the dict, behavior
  breaks. Also: any code path that constructs the cmd directly
  (bypassing `_action_to_command`) silently underdeducts MP. Refactor:
  thread `terrain_cost` through a separate variable in `step()`.

- [ ] 🟡 **`_replace_unit` pattern not used everywhere; risk of dropping
  `_defense_table`** (`tools/replay_dataset.py:790-792` vs. various
  open-coded patterns). The `_`-prefixed stash is preserved in healing
  (line 1051-1064), end_turn (1106-1120), and recruit (1310-1316), but
  there's no central helper. A future refactor of any of these will
  silently drop e.g. `feral` Bat's defense overrides. Factor into a
  single `_replace_unit_safe(u, **changes)`.

### Performance / scaling for MCTS

- [ ] 🟡 **`_deduct_extra_mp` rebuilds the entire units set per call**
  (`tools/wesnoth_sim.py:461-474`). Should use the discard/add pattern
  from `_replace_unit`. ~30× speedup on this path. Negligible for
  REINFORCE; matters for MCTS rollouts.

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

- [ ] 🟡 **`sim_to_replay` doesn't escape `"` or `\` in unit names**
  (`tools/sim_to_replay.py:108-149`). For default era it doesn't matter;
  for custom-era unit types with special chars in their names the file
  becomes invalid WML. Wesnoth WML escapes `"` as `""`. Or assert.

- [ ] 🟡 **`_find_final_replay_block` is positional and brittle**
  (`tools/sim_to_replay.py:269-284`). `text.rfind("[replay]")` works for
  current source files; a literal `[/replay]` inside a `[message]`
  body would fool it. Anchor the regex to start-of-line.

- [ ] 🟡 **Encoding: `errors="ignore"` silently drops non-UTF8 bytes**
  (`tools/sim_to_replay.py:305`). Fine for the current corpus. Use
  `errors="replace"` and log a warning so we notice.

- [ ] 🟢 **Optional `[checkup]` debug emission** — let the user flip a
  flag and emit `attacker_hp/defender_hp` checkup blocks so OOS
  divergences point at the exact diverging strike. Cheap diagnostic.

### Tests

- [ ] 🟠 **Sim determinism test**: same seed → same outcome over N
  steps, both for combat and trait rolls. Currently no such test.
  Critical before MCTS — branching depends on this.

- [ ] 🟠 **Reward unit tests**: hand-built `(prev, new)` pairs covering
  each `StepDelta` field (kill, village swap, recruit-success,
  recruit-fail-invalid, leader-move). The reward function has accreted
  comments about "0.22 flat-return plateau" and other anti-patterns
  that nobody caught before overnight runs — unit tests would.

- [ ] 🟡 **`hex_distance` test**: identity at zero, symmetry, the
  odd-q parity edge case the docstring describes
  (`rewards.py:204-211`).

- [ ] 🟡 **End-to-end round-trip test**: take a real replay, run our
  recon, re-export from `command_history`, diff WML token-stream
  against source. Hard regression net for the bit-exactness work.

---

## Model architecture (`model.py`, `encoder.py`, `action_sampler.py`, `transformer_policy.py`, `trainer.py`)

### Bugs (silent corruption)

- [ ] 🟠 **`reforward_logprob_entropy` indexes `unit_ids` for any actor**
  (`action_sampler.py:255`). `encoded.unit_ids[actor_idx]` IndexErrors
  when `actor_idx` is a recruit slot (`actor_idx >= num_units`). Today
  it's masked off by `if weapon_idx is None: return ...` (line 249), but
  any future change that gives recruits a weapon idx breaks silently.
  Guard: `if actor_idx >= len(encoded.unit_ids): return log_prob, entropy`.

- [ ] 🟡 **Trainer pass-1/pass-2 mode inconsistency**
  (`trainer.py:184-190`). Pass 1 runs in `eval()` mode, Pass 2 inherits
  `train()` from `transformer_policy.train_step` (line 278). Even with
  `dropout=1e-4` this introduces tiny mismatch between value-for-advantage
  and value-for-MSE. Run both passes in eval mode (no_grad isn't
  involved).

- [ ] 🟡 **Trainer subsamples by uniform stride, correlated with episode
  position** (`trainer.py:151`). With `gamma=0.99`, this preferentially
  trains on near-terminal returns. Use random shuffle + slice.

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
  a new type shows up).

- [ ] 🟡 **`MAX_UNIT_TYPES=200` overflow buckets all unknowns to id 199**.
  Aliases all unknown types together. Probably fine for scope but
  document.

- [ ] 🟡 **Module-level constants `HP_NORM=80, COST_NORM=80` etc.**
  (`encoder.py:95-100`). These should live in `constants.py` so an
  era mod with cost-200 units doesn't silently saturate features.

### Performance

- [ ] 🟡 **`_build_legality_masks` rebuilds `pos_to_hex` and walks
  `gs.map.hexes` linearly per decision** (`action_sampler.py:404,485-490,556`).
  Cache once at the top; pull `hex_xs/hex_ys` numpy arrays into
  `EncodedState` instead of recomputing. Saves several ms per decision.

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

- [ ] 🟡 **Checkpoint compat refuses on any arch mismatch**
  (`transformer_policy.py:325-331`). Brittle for an RL system that gets
  tweaked. Allow partial loads (load matching submodules, warn on
  mismatch); store optimizer state per-module.

### Customizability gaps

- [ ] 🟡 **`_COMBAT_LOGIT_ALPHA = 0.1` baked in** (`action_sampler.py:71`).
  Strategic lever ("how aggressive is the attack bias"). Move to
  `constants.py` so a "pacifist" or "berserker" variant is a config
  flip.

- [ ] 🟢 **Combat-oracle bias only on attacks, not on moves**
  (`action_sampler.py:467`). Adding move-toward-good-attack-position
  bias would help unorthodox-strategy training. Infrastructure already
  there.

- [ ] 🟢 **`_DEFAULT_FACTIONS` is a Python literal** (`encoder.py:60-63`).
  Era mods would require a code edit. Pull from config file.

---

## Reward shaping & customization (`rewards.py`, `tools/sim_self_play.py`)

The user explicitly wants behavior gated by config/data/reward shaping
rather than weights. Currently `WeightedReward` has fixed scalar
weights; there is no opener gating, no per-unit-type bonus, no
curriculum hook.

- [ ] 🟠 **Add per-unit-type recruit bonuses** to `WeightedReward`. Today
  there's no way to express "+0.5 reward per Wose recruited". Sketch:
  ```python
  @dataclass
  class UnitTypeBonus:
      unit_type: str
      weight:    float
  # Add to WeightedReward:
  unit_type_bonuses: List[UnitTypeBonus] = field(default_factory=list)
  ```
  Trainer needs to pass `units_recruited: Tuple[str, ...]` in `StepDelta`.

- [ ] 🟠 **Add turn-conditional bonuses** for opener incentives. E.g.
  "+1.0 if leader on village by turn 3":
  ```python
  @dataclass
  class TurnConditionalBonus:
      name:       str
      turn_range: Tuple[int, int]
      predicate:  Callable[[GameState, int], bool]   # state, side
      weight:     float
      once:       bool = True
  ```
  Trainer needs `compute_delta` to receive post-state for the predicate.

- [ ] 🟠 **Add an opener-gating policy wrapper**. Currently no code path
  forces moves on early turns. Sketch:
  ```python
  # tools/openers.py (new)
  @dataclass
  class Opener:
      name: str
      moves: List[Callable[[GameState], Optional[Dict]]]
      sides: Tuple[int, ...] = (1, 2)
  class OpenerPolicy:
      """Wraps base; for turn ≤ len(opener.moves), tries opener
      action; falls through to base if None."""
  ```
  Wire via `--opener-spec` in `sim_self_play.py`.

- [ ] 🟠 **`sim_self_play.py` exposes NO reward weight as CLI arg**
  (`tools/sim_self_play.py:366`). Currently uses `WeightedReward()`
  defaults. Add `--reward-config path/to/yaml` so cluster jobs can
  experiment with reward shapes without code edits.

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

- [ ] 🟡 **Recruit cost lookup falls back to 0 silently**
  (`tools/sim_self_play.py:79`). For unit types not in `unit_stats.json`,
  `cost_lookup.get(unit_type, 0)` gives 0 reward credit on successful
  recruit. Use 14 (default) and log a warning at lookup time.

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

- [ ] 🟡 **`cluster/job.sbatch:102` hardcodes `supervised_epoch9.pt`** as
  done-detection sentinel, paired with `--epochs 10` (line 87). Change
  one without the other and the auto-resubmit chain runs forever.
  Compute the sentinel from `--epochs` or use a `--max-epoch=N`
  argument that emits the right `done.flag` itself.

- [ ] 🟡 **`sync.ps1`: no checksum verification of extracted files on
  remote** (`cluster/sync.ps1:122-147`). On non-zero return code we
  warn "files may be partially synced" but `tar -xf -` is not atomic
  — partial extraction overwrites in place. Add a `sha256sum` round-trip
  for files in the manifest.

- [ ] 🟡 **GUI: per-op `pw.txt` lingers in `%TEMP%` if GUI killed
  externally** (`cluster/gui.pyw:95,286-288,537-539`). Add an `atexit`
  hook that walks `%TEMP%/wai_gui_pw_*` on every GUI startup.

---

## Eval harness (`tools/eval_*.py`, `add-ons/wesnoth_ai/scenarios/eval/*`)

- [ ] 🟠 **GUI eval defaults `no_swap_var = True`** (`cluster/gui.pyw:481`).
  Halves game count by skipping side-swaps but introduces first-mover
  bias. The CLI default is `False`; the GUI silently regresses every
  eval the user runs through it. Flip to `False`.

- [ ] 🟡 **No confidence interval on eval win-rates**
  (`tools/eval_vs_builtin.py:139-217`). At ~30 games per matchup, a
  50% win-rate has a ±18% 95% CI. Add Wilson interval or at least
  `n` per cell (already there as `(W/decisive)`, just promote
  visibility).

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
- [ ] 🟡 State-key collision fuzz for MCTS transposition table.
- [ ] 🟡 `sim_self_play` reward-flow smoke (one game, terminal+shaping
  attached to right transitions).

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
