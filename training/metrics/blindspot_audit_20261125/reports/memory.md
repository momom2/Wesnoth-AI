# Blind-spot audit, part "time": what reaches the network from before the decision

Auditor's scope: the network's memory, the watched enemy turn, units last seen,
knowledge within the side's own turn, and knowledge that accumulates over a game.
Method: reading only (repository at `main`, 0.36.0, and Wesnoth 1.18.4 C++ on
GitHub). Nothing was run.

## How information from the past reaches the network (the map this audit checked)

There are exactly four paths. Each is verified by reading.

1. **The snapshot itself**, encoded at every decision of the side
   (`rust/wesnoth_core/src/core_encode.rs:282-475`). It carries within-turn
   history only as state: each visible unit's current moves and `has_attacked`
   (`core_encode.rs:398-400`), HP, XP and statuses. Fog in your own turn is
   accumulated, not refogged (`docs/wesnoth_rules.md:3344-3346`).
2. **The sighting stream.** For each side, the core records every unit of
   another side that the side has seen since its own last end_turn: its type, HP,
   max HP and the last hex it was seen on (`core_sight.rs:31-39`,
   `:57-65`). The record is updated after every command, from what the side sees
   at that moment (`core_sight.rs:70-94`). The callers are moves
   (`core_move.rs:279`), attacks (`core_attack.rs:311-368`, plus `:455`
   for the defender before its refog), init_side (`core_step.rs:294`), end_turn
   (`core_step.rs:315-316`), shroud commits (`core_shroud.rs:146-161`) and unit
   placement (`core_units.rs:245`). The record is cleared at the side's own end_turn
   (`core_step.rs:315`, `core_sight.rs:112-118`). At a decision, every entry whose
   unit the side does not see now becomes a token with side code 3
   (`core_parity.rs:458-475`, `encoder.py:268`, `:1248-1253`). The token's
   features are type id, x, y, hp/max hp and max hp/norm (`SIGHT_FEAT_DIM` 2,
   `core_parity.rs:55-56`).
3. **The seen-types set.** Every (side, unit type) the side has seen this game,
   never cleared (`core_sight.rs:12-13`, `:61-63`). It feeds the enemy-faction
   posterior in the global token (`game_core.py:956-965`,
   `faction_posterior.py:97-108`).
4. **The learned memory.** It has k float32 rows of model width
   (`wesnoth_ai/memory.py:4-22`). At each of the side's own decisions, the trunk
   reads them as MEMORY tokens and they are rewritten through a GTrXL gate
   (`memory.py:117-126`; `model.py:596`, `:621-623`, `:799-803`).
   - **When it is written:** only at the side's own decisions. The memory is
     keyed by (game, current side) and stepped inside `_forward`
     (`tools/raw_player.py:215-227`). A step at enemy decisions was explicitly
     rejected (`docs/parity_memory_design_20260929.md:418-419`).
   - **Lifetime:** it starts from the learned initial rows
     (`memory.py:77-85`). It is carried across every decision and turn of the
     game-side and dropped at game end (`raw_player.py:255-260`). It is never
     reset within a game. A "memory reset each turn" play mode exists, and the
     pass refuses game-sides played that way (`tools/pg_positions.py:371`,
     `:415-416`).
   - **Refused recruit:** the memory is rolled back to its state before the
     refused decision (`raw_player.py:262-279`). The pass mirrors this: a shadow
     row's slot keeps what it read (`tools/pg_schedule.py:108-112`).
   - **Same in play, the gradient pass and matches:** k = 64 in self-play
     (`scripts/simple_loop_local.py:87`), in the pass (`pg_game_pass.py:127`) and
     for the reference (`configs/reference_player.json:8`). The pass rebuilds each
     position from the record and checks that it reproduces the logged log
     probabilities (`pg_game_pass.py:51-59`, `:416-422`). So play and training
     see the same inputs and memory.
   - **Gradient span:** the simple loop takes the whole game-side as one graph,
     with no truncation (`pg_game_pass.py:28-35`, `:275-316`). The weights it
     started from were trained in imitation with 16-decision truncated
     back-propagation (`docs/parity_memory_design_20260929.md:240-246`), and in
     the rounds with 16-decision windows (`tools/pg_pass.py:12-13`, `:229`).
   - **How far it can reach:** the turn cap is 200 (`tools/run_elo_batch.py:595`)
     and the cap is 2,000 actions per side (`tools/wesnoth_sim.py:351`). A typical
     game is about 490 decisions, about 245 per side (`docs/status.md:683`). So
     the memory can in principle span the whole game: up to 200 turns, and about
     245 decisions per side in practice.
   - **Not trained in the simple loop:** the belief head, which taught the memory
     to track hidden units in imitation (`docs/parity_memory_design_20260929.md:199-209`).
     The loop's loss is the policy term plus the value cross-entropy only
     (`pg_game_pass.py:8-12`, `:252-273`). Nothing now asks the memory to keep
     hidden units, other than their use to the policy.

Two facts frame every "only through the memory" verdict below. Both are measured
on `own2`, not on the current network (`docs/status.md`):
- within `own2`, 64 slots against 0 reads +13.0 ± 12.3 Elo (`status.md:639`);
- `own2`'s memory "holds one learned pattern, the same in every slot", whose level
  follows the game's phase (`status.md:413-428`).

So "recoverable through the memory" is a statement about what the architecture
allows. The memory in fact carried little in `own2`. Whether the current network
uses it more is a run to make (below).

---

## 1. Findings, most serious first

### F1. Anything seen before the enemy's last turn lives only in the memory, and the measured memory holds almost nothing
- **What a player has:** their own memory of the whole game. Examples: the enemy
  leader's position two turns ago; a group of enemy units seen moving toward a
  flank; a hider (ambush or nightstalk) that was revealed and has since hidden
  again; an enemy unit at 3 HP that slipped into fog last turn. The interface
  has no history panel either, so this is the player's own memory (see F5 for
  the one panel that does exist).
- **What the network gets:**
  - The sighting record is emptied at the side's own end_turn
    (`core_step.rs:301-318`; `core_sight.rs:112-118`). At the next own turn it
    holds only what was seen since, which is the enemy's turn plus the current
    own turn.
  - A hider whose side starts its turn is hidden again
    (`core_step.rs:266-271`). Unless it is in view or adjacent again, it does
    not re-enter the record.
  - Anything older reaches the network only through the 64 memory rows.
  - Verdict: **not in the snapshot, recoverable only through the memory.**
- **Why it matters:**
  - A Ghost that retreated wounded into fog two turns ago and is now healing
    in a village is exactly what a player hunts.
  - An enemy leader last seen leaving its keep tells a player where recruits
    will come from.
  - The memory is the only path for either. On `own2` it measured as a phase
    clock (`status.md:413-428`) and worth +13 ± 12 Elo (`status.md:639`). The
    belief loss that taught it to track hidden units is not in the current loss
    (`pg_game_pass.py:252-273`).
- **Confidence:** the path is verified by reading. Whether the current network
  keeps such units is inferred (from `own2`'s measurements).
- **Confirm by a run:** checks R1 and R2 below.

### F2. The network never learns which action it chose; recruits redirected or refused by a hidden unit are lost entirely
- **What a player has:**
  - They know what they ordered.
  - When they order a recruit onto a fogged castle hex that a hidden enemy
    holds, the recruit appears on another hex. The engine treats the order as
    "no location" and takes the vacant castle hex nearest the leader
    (`src/actions/create.cpp:419-421` and `:392`, quoted in
    `docs/wesnoth_rules.md:3209-3238`).
  - When no castle hex is vacant, the recruit is refused with a message
    (`src/menu_events.cpp:350-370`, `docs/wesnoth_rules.md:3733-3745`).
  - Either way the player learns that an enemy stands on that castle hex.
- **What the network gets:**
  - The memory is written by the forward pass, before the action is sampled
    (`raw_player.py:170-181` and `:215-227`). Nothing feeds the chosen action
    back: neither an input nor a memory update carries it. The next snapshot
    shows only the outcome.
  - A redirected recruit therefore looks like a recruit ordered on the hex where
    it landed. The ordered hex is marked in `recruit_rejected`
    (`core_sim.rs:39-41`), but the encoder writes that column as 0 under the
    parity observation (`core_encode.rs:370-381`). Only the legality mask reads
    it.
  - For a refused recruit, the memory is also rolled back to its state before
    the decision (`raw_player.py:262-279`; `pg_schedule.py:108-112`). The next
    forward reads the same memory and the same snapshot. The only difference is
    the mask, which the network does not see.
  - Verdict: **lost entirely.** This breaks Principle 6 (the mask reads state
    the encoder does not show). `docs/status.md:576-577` already lists "rejected
    recruit hexes that the mask reads and the network cannot see" as against
    Principle 6. What this audit adds is the redirected case and the memory
    rollback: together they leave no trace at all.
- **Why it matters:** an enemy Elvish Ranger hidden in fog on your castle is a
  real threat to your leader. A player whose recruit "jumps" to another hex knows
  it is there. The network goes on as if the castle were empty.
- **Other cases of "ordered X, got Y":**
  - An ambushed move: the unit stops adjacent to the revealed hider with MP 0.
  - A blocked move: the unit stops before the hider and keeps its MP.
  - The sim does both (`core_move.rs:186-240`, `:316-329`; rules at
    `docs/wesnoth_rules.md:621-643`).
  - These are recoverable from the snapshot: the uncovered hider is visible, and
    the own unit's position and moves are in its row. They are not lost.
- **Confidence:** verified by reading.
- **Confirm by a run:** R3 below, which counts how often this happens.

### F3. A sighting token is thinner than what the player saw: no leader flag, XP, traits, statuses, side or age
- **What a player saw:**
  - the enemy unit itself, with its leader crown;
  - its sidebar while it was visible: XP, traits, poisoned or slowed status;
  - when it was seen.
- **What the network gets:**
  - The record keeps type, HP, max HP and the last hex
    (`core_sight.rs:31-39`).
  - The token adds only hp/max hp and max hp/norm (`core_parity.rs:55-56`,
    `:471-475`).
  - Side code 3 is shared by every non-own side (`encoder.py:268`), and the
    record covers every other side's non-scenery units (`core_sight.rs:86-93`).
  - A visible unit's row has `is_leader` (`core_encode.rs:398`). A sighting does
    not.
  - Verdict: **the fields are lost from the snapshot.** They are recoverable
    through the memory only if the memory wrote them while the unit was visible.
- **Why it matters:**
  - When the enemy leader steps out of view, the token shows a level-2 type.
    The enemy can also field that type as an advanced unit, so the network cannot
    tell its leader-kill target from a veteran.
  - A seen unit one fight from levelling up, or one that is poisoned, is also
    lost the moment it enters fog.
  - Whether a token is from this turn or last turn is not shown. Within the
    record's one-turn window that is a small gap.
- **Confidence:** verified by reading. The impact is inferred.
- **Confirm by a run:** R4 below.

### F4. The enemy's turn is one snapshot plus last-seen positions: who attacked whom, with what, and the dice are lost
- **What a player watches:**
  - Each enemy fight that hits their units, with its animation and damage
    numbers (`src/units/udisplay.cpp:599-733`; `docs/wesnoth_rules.md:3582-3598`).
  - Their own units dying.
  - Healing at their own turn start, shown as separate numbers.
- **What the network gets:**
  - The memory is not stepped during the enemy's turn (`raw_player.py:215-227`;
    design doc `:418-419`).
  - The sighting record keeps only each unit's last position and HP
    (`core_sight.rs:31-39`).
  - The fight's facts (`att_id`, `dfd_id`, damage, `core_attack.rs:401-424`)
    are telemetry only. A grep of `core_encode.rs` finds no use of fight, heal,
    kill or death history.
  - The next snapshot is taken after the own side's init_side, so healing,
    poison and rest are already applied (`core_step.rs:154-295`). The damage
    taken and the healing are merged.
- **What survives where:**
  - Which own units died: not in the snapshot (they are gone). Recoverable
    through the memory, which saw them at the last own decision.
  - Which enemy unit attacked: usually still adjacent and visible, because an
    attacker's MP is 0 after a fight (`core_attack.rs:427-431`). If refog hid
    it, it appears as a sighting (`core_attack.rs:454-456`). So it is partly in
    the snapshot.
  - Weapon used, strike-by-strike results, and damage before healing: **lost
    entirely.**
- **Why it matters (inferred, moderate):** a player who saw a Thunderer kill
  their Spearman, or saw the enemy roll well, adjusts. The network sees the
  Spearman missing, and other own units' HP net of healing. The enemy's weapons
  are in its unit rows, so the loss is mostly attribution and luck, not
  capability.
- **Confidence:** verified by reading.
- **Confirm by a run:** none needed for the fact itself. Whether it matters
  would need an input change, which is the owner's call.

### F5. The in-game Statistics dialog: the side's game-long tallies are not in the input
- **What a player can look up:** the Statistics hotkey opens the dialog for the
  viewing side: `menu_handler_.show_statistics(gui()->viewing_team()+1)`
  (`src/hotkey/hotkey_handler.cpp` about line 88, 1.18.4). It shows:
  - recruits, recalls, advancements, losses and kills by unit type;
  - damage inflicted and taken, actual against expected;
  - hits, with a percentile, over the game and "this turn".

  Source: `src/gui/dialogs/statistics_dialog.cpp`, `update_lists()`
  (`add_stat_row(_("Losses"), stats.deaths); add_stat_row(_("Kills"), stats.killed);`
  and the `add_damage_row` calls).
- **What the network gets:**
  - The global row holds turn, side, gold, income, the two village counts and
    two time-of-day bonuses (`core_encode.rs:441-446`), plus the parity extras
    (economy and fog flag, `core_parity.rs:52-54`).
  - A grep of `core_encode.rs` for kills, deaths or fight history: none.
  - Verdict: **not in the snapshot, recoverable only through the memory.**
- **Why it matters (inferred, low to moderate):** these tallies tell a player
  how the game has gone. Kills by type reveal how much of the enemy's army is
  spent, and damage against expected shows the luck so far. A player reads
  them when judging whether to press or turtle.
- **Confidence:** verified by reading (the C++ was fetched).

### F6. Moves are never interrupted when an enemy comes into view (low)
- **What a player has:** in 1.18 a move that brings a new enemy into view
  is interrupted, and the player decides again from the new view. Replays carry
  `skip_sighted` for this (`src/synced_commands.cpp:305-314`, quoted at
  `docs/wesnoth_rules.md:645-656`).
- **What the network gets:** the sim does not model sighting interrupts
  (`docs/wesnoth_rules.md:654-656`), so a move completes and the new enemies
  show up only at the next decision.
- **Why it is low:** the network can order shorter moves (any reachable hex is a
  target), which gets the same information at the same cost. This is an order
  convenience more than a blind spot.
- **Confidence:** verified by reading the rules document. I did not read the
  sim's move loop for interrupts beyond the rules document's statement.

---

## 2. Coverage (checked and found present and right)

**Memory: structure, lifetime and training**
- The memory's structure and gated write: `wesnoth_ai/memory.py:60-126`.
- A memory token enters the trunk; the new state is written after it:
  `model.py:596`, `:621-623`, `:799-803`, `:825-833`.
- The memory is carried from each decision to the next of the same side, within
  and across turns, keyed (game, side): `raw_player.py:215-227`, `:240-253`.
- It is dropped at game end, and each game starts from the learned initial rows:
  `raw_player.py:255-260`; `memory.py:77-85`.
- A refused recruit's memory is rolled back, in play and in the pass alike:
  `raw_player.py:262-279`; `pg_schedule.py:108-112`; `pg_game_pass.py:284-286`.
- k is 64 in self-play, the pass, matches and the reference: `simple_loop_local.py:87`;
  `pg_game_pass.py:127`, `:199-221`; `configs/reference_player.json:8`.
- No truncation of back-propagation in the current loop:
  `pg_game_pass.py:28-35`, `:275-316`.
- Play and pass see identical inputs, held by the replay-identity check:
  `pg_game_pass.py:51-59`, `:416-422`.
- Turn cap 200 and 2,000 actions per side: `run_elo_batch.py:595`;
  `wesnoth_sim.py:343-351`.

**Sighting record and the watched turn**
- The record follows every command type: the `note_sightings` callers listed in
  the map above.
- The defender's side records the fight before its refog, and before either
  unit advances: `core_attack.rs:450-464`; engine order at
  `docs/wesnoth_rules.md:3582-3630`.
- An enemy recruit seen and then lost to fog stays as a token, with its type,
  until the own end_turn: `core_sight.rs:57-65`, `:120-130`.
- An enemy unit that walks out of view keeps its last hex; one that crosses the
  view mid-move is not seen. This is the user's accepted ruling 8:
  `core_sight.rs:15-20`; design doc `:32-34`, `:156-160`. It is accepted, not
  reported.
- A unit that died where the side saw it leaves the record. One that died unseen
  stays, marked gone, as a player would still believe it there:
  `core_sight.rs:70-78`, `:96-110`; `core.rs:665`.
- At end_turn the record is cleared, then refilled with what is in view after
  the refog, so units in view at turn end are kept if they later walk into fog:
  `core_step.rs:314-316`.

**Knowledge that accumulates**
- Seen types accumulate for the whole game and feed the faction posterior:
  `core_sight.rs:12-13`, `:61-63`; `game_core.py:956-965`.
- Placed scenario units are not counted as seen types: `core_sight.rs:61`.

**Within the side's own turn**
- Units that already moved or attacked, through moves left and `has_attacked`:
  `core_encode.rs:398-400`.
- A fight's result is visible as the new HP and XP, and a death as a missing
  unit, set by `core_attack.rs:426-449`.
- An ambush reveals the hider (persistent `uncovered`) and zeroes MP; a block
  reveals the hider and keeps MP: `core_move.rs:186-240`, `:316-329`; rules at
  `docs/wesnoth_rules.md:576-643`.
- Fog accumulates over the own turn and is not refogged:
  `docs/wesnoth_rules.md:3344-3346`.

**No leak in the other direction (checked)**
- Sightings come only from `visible_to(side)`: `core_sight.rs:85-93`.
- Material is summed over visible units only: `core_encode.rs:389-409`.
- The belief truth is a training target only (design doc `:199-209`), and the
  current loop does not train it.
- Each side's memory is its own: `raw_player.py:220`; `memory.py:147-165`.
- No memory survives a game: `raw_player.py:255-260`.

## 3. To check by a run (for the main session)

- **R1. What the memory is worth now.** Play a match of the current loop network
  at k = 64 against the same network at k = 0, on fixed seeds, both at the
  reference decode, 800 decisive games. This repeats `status.md:639` (+13 ± 12
  for `own2`).
- **R2. Whether the memory carries anything across turns.** Same network: play
  "memory reset each turn" against the normal memory. The play mode exists
  (`memory_reset` in `tools/pg_positions.py:371`, `tests/test_own_trajectories.py:57-66`);
  find which flag in `run_elo_batch` / `elo_eval_game` sets it.

  A cheaper alternative is a probe on logged games: at each side's first
  decision of a turn, compare the policy's KL between its true memory and the
  memory it held at its previous turn's first decision. Only the memory differs
  between the two.
- **R3. How often recruits are redirected or refused.** On one round's self-play
  records, count:
  - decisions logged `refused` (`raw_player.py:269-271`);
  - recruits whose landing hex differs from the ordered hex.

  The second needs the decision log's target against the record's recruit hex.
- **R4. How often the enemy leader is out of view.** On the same records, count
  decisions where the enemy leader is out of view but in the sighting record,
  and decisions where it is in neither. This measures F3's leader-flag gap and
  F1's reliance on memory.

## 4. Could not determine

- Whether the current network's memory holds anything beyond the phase pattern
  measured on `own2`. Only R1 and R2 can say.
- Edge case: a hidden enemy in the record that dies on a hex the side has
  cleared but where the unit itself was invisible. The record drops it
  (`core_sight.rs:105`, which tests the hex, not the unit), so the network
  "learns" of a death the player did not see. It needs a neutral side or another
  enemy to kill it, so it is probably absent on the Ladder maps. I did not check
  the pool for neutral sides.
- Whether the "interrupt move when enemy sighted" preference defaults to on in
  1.18. I did not fetch the preferences code (F6 rests on the rules document).

## 5. For other parts

- **Enemy faction:** sim games never set `chose_random`
  (`classes.py:397` default False; `sim_to_replay.py:956` writes `"no"`; no
  setter in `tools/` or `wesnoth_ai/`). So the faction posterior is one-hot on
  the true faction from decision 1, and the seen-types path is inert in
  self-play and matches. It is a leak only if the modelled lobby is "Random"
  rather than factions chosen openly.
- **Turn limit:** the global row has the turn number but not the turn limit
  (`core_encode.rs:441-446`). The status bar shows turn/limit. The limit is a
  constant 200 here.
- **Village flag under fog:** an own village captured in fog shows as unowned to
  the network (`core_encode.rs:359-369`, `visible = ours || !fog_on || seen`).
  Check this against `display::get_flag` (present-state part).
