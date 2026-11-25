# Blind-spot audit, part "the outputs" (2026-11-25)

Scope: what a player can order in a two-player MP game of Wesnoth 1.18
against what the network can order; whether the legality masks match the
game's legality; what each score reads. Read only. Wesnoth C++ was read at
the 1.18.4 tag (raw.githubusercontent.com, copies in the scratchpad folder
`wsrc_actions\`). Repository at `main` (36b3f5f1).

The path that plays the simple loop's games was traced first, because it is
what trains: `scripts/simple_loop_local.py:131-142` (play through
`tools/run_elo_batch.py`, both sides `--raw-temperature 1`, no offset, the
POOL flags of `scripts/match_local.py:114-115`: shared inference, fp16) ->
`tools/elo_eval_game.py:1147-1166` (`WesnothSim`, `_play_one_eval_game`) ->
`tools/elo_eval_game.py:381-417` (`_remote_player`: `RawPolicyPlayer` over
server-side priors) -> masks from the Rust core
(`rust/wesnoth_core/src/core_masks.rs`, `server_priors.core_packed_masks`)
-> priors on the server (`wesnoth_ai/server_priors.py:311-393`) ->
`tools/raw_player.py:141-161` picks one entry.

---

## 1. Findings, most serious first

### F1. The advancement choice is never the network's, and in the loop's games it is always the first type listed

- **What the player orders.** In a networked MP game, when a unit with two
  or more advancement options levels on its own side's turn (as the
  attacker, typically), its owner picks the type in a dialog:
  `src/actions/advancement.cpp:205-207` (1.18.4):
  `if(!video::headless() && (force_dialog_ || (t.is_local_human() && ... && (is_current_side || !is_mp)))) { res = advance_unit_dialog(loc_); }`.
  On the enemy's turn (the unit levels while defending) the owner is NOT
  asked in networked MP: the `else` branch keeps `res = 0`, "default to the
  first unit listed in the unit's advancements" (`advancement.cpp:232-237`).
  With the mainline Plan Unit Advance modification active, the owner can
  also pre-choose by a menu item (a synced `[fire_event]`,
  `src/synced_commands.cpp:326`; `docs/wesnoth_rules.md:1771-1838`), which
  narrows the list so the defensive default becomes the planned type.
- **What the network gets.** No head for it: the actor/type/target/weapon
  chain (`wesnoth_ai/action_sampler.py:126-134`) has no advancement factor,
  and no action type names it (`tools/wesnoth_sim.py:1269-1528`). The core
  decides in `advance_once` (`rust/wesnoth_core/src/core_units.rs:90-114`):
  the replay's `[choose]` queue if any, else a uniform draw **only if**
  `advance_uniform` is set, else index 0. `advance_uniform` is set only by
  `WesnothSim.enable_uniform_advancement` (`tools/wesnoth_sim.py:736-753`),
  which `tools/selfplay_game.py:531,578` calls and
  `tools/elo_eval_game.py` (the path the simple loop plays through) never
  does (grep `enable_uniform_advancement|advance_uniform` over
  `tools/elo_eval_game.py`, `tools/run_elo_batch.py`, `tools/eval_players.py`,
  `scripts/simple_loop_local.py`, `scripts/match_local.py`: no hit). The
  attack commands the simulator emits carry no choices
  (`tools/wesnoth_sim.py:1411-1414`, 8 fields). Confirmed on the run's own
  records: all 102 self-play game records of round 583
  (`C:\Users\momom2\wesnoth_runs\simple_loop\round_0583\games_self\*.game.jsonl.gz`)
  carry `"uniform_advancement": false` (written at `tools/game_record.py:134`).
- **So, in every training game and every match**, every multi-option
  advancement takes `advances_to[0]` (from `unit_stats.json`): Spearman ->
  Swordsman (never Pikeman or Javelineer), Elvish Fighter -> Elvish Captain
  (never Hero), Elvish Archer -> Ranger (never Marksman), Elvish Shaman ->
  Druid (never Sorceress), Mage -> White Mage (never Red Mage), Troll Whelp
  -> Troll (never Rocklobber), Wolf Rider -> Goblin Knight, Goblin Spearman
  -> Impaler, Ghost -> Wraith, Skeleton -> Revenant, Drake Burner -> Fire
  Drake, Drake Clasher -> Thrasher, Saurian Augur -> Oracle, Merman Hunter
  -> Merman Spearman.
- **Against the human.** On defence the simulator does what a human without
  the modification gets (index 0). On attack the human chooses and the
  network cannot; with Plan Unit Advance the human chooses on defence too.
- **Why it matters.** A Loyalist player levelling a Mage on an attack picks
  Red Mage or White Mage by need (damage or healing/arcane against Undead);
  the network always gets the White Mage, and the Spearman line never yields
  a Javelineer's ranged retaliation. The network is also trained only on the
  states these forced choices produce.
- **Partly known.** `docs/status.md:574-577` lists "the advancement" among
  things decided beside the network. New here: that it is deterministic
  (index 0, not the uniform draw), that `tools/wesnoth_sim.py:29-31` and
  `:737-740` ("self-play draws uniformly") do not hold for the simple loop's
  path, and that the defensive half matches Wesnoth's networked-MP rule.
- **Confidence:** verified by reading (code and the run's records).
- **Run to confirm:** "To check by a run", item 1.

### F2. A move into fog does not stop when an enemy comes into sight, where the player's move does

- **What the player orders.** A move ordered by a click in a fogged game is
  executed with `continued_move = false` (`src/mouse_events.cpp:1225`,
  `actions::move_unit_and_record(steps, &pc_.get_undo_stack(), false, true, &interrupted)`),
  so sighting is checked: `src/actions/move.cpp:560-566`
  (`if ( !skip_sighting_ ) { sighted_ = enemy_count_ != 0 ; }`, allies too
  unless the preference skips them, `move.cpp:1247`), and the walk stops
  at the first reasonable hex after a sighting: `move.cpp:955-957`
  (`if ( sighted_ && is_reasonable_stop(*step_from) ) { sighted_stop_ = true; break; }`).
  The unit keeps its remaining movement and the player re-decides
  ("Units sighted!", `move.cpp:1118-1135`; reselection for "continue move",
  `mouse_events.cpp:1233-1238`). Without fog, sighting is skipped
  (`move.cpp:1241-1246`).
- **What the network gets.** The simulator does not model sighting
  interrupts: its move truncates only for "blocked" and "ambush"
  (`rust/wesnoth_core/src/core_move.rs:136-137, 195, 215`), and the export
  writes `skip_sighted="all"` so Wesnoth does not re-check
  (`docs/wesnoth_rules.md:652-656`). A move the mask offers to a far fogged
  hex runs to its end even when the first step reveals an enemy army.
- **Why it matters.** A scout sent 8 hexes into fog that reveals the enemy's
  main body after 2 hexes: the player's scout stops there and can retreat;
  the network's scout walks into the enemy's reach and is lost. The network
  can only get the player's behaviour by ordering one short move at a time,
  which nothing in training rewards it for discovering. It is also a
  fidelity gap under Principle 4: the same order gives a different result
  in Wesnoth (exports hide it by `skip_sighted="all"`).
- **Confidence:** verified by reading (engine and core); the frequency is not
  measured.
- **Run to confirm:** "To check by a run", item 3.

### F3. Orders the network cannot give that change little in these games

Each verified absent from the network's action set (no action type for it in
`tools/wesnoth_sim.py:1269-1528`, no actor slot for it in
`wesnoth_ai/model.py:393-401`).
- **Delay fog updates** (`auto_shroud`, `update_shroud`,
  `src/synced_commands.cpp:367, 383`; menu `menu_events.cpp:505-525`). The
  core can apply them (`docs/wesnoth_rules.md:3571-3577`) but the policy
  cannot order them. With updates delayed, moves reveal nothing until the
  update, so they stay undoable and raise no sighting stop. Low impact;
  inferred.
- **Undo** of a move that revealed nothing (the `use_undo` path of the move
  and recruit handlers, `synced_commands.cpp:70, 260`). Undo restores an
  earlier state, so the player gains little beyond looking at a move's
  result first. Low; inferred (undo's exact conditions were not read).
- **Plan Unit Advance** (above, F1).
- **Recall and disband** (`synced_commands.cpp:134, 233`): the network
  cannot recall (`tools/wesnoth_sim.py:1501-1526`, no recall slot). The
  recall list is empty in these games: no ladder scenario mentions recall
  (grep `recall` over `wesnoth_src/data/multiplayer/scenarios` and
  `wesnoth_src/data/add-ons`: only `4p_King_of_the_Hill.cfg`), and nothing
  in a 2p MP game puts a unit there. No loss; the "nothing puts a unit
  there" part is inferred.
- **Multi-turn goto** (`menu_events.cpp:903-993`, run at the start of a human
  turn, `playsingle_controller.cpp:561-567`) and **continue move**
  (`menu_events.cpp:862-872`): both stand for ordinary moves the network
  can order each turn. No loss.
- **Choosing the route** beside the destination: the network names only the
  destination, and the simulator routes by the core's Dijkstra
  (`tools/wesnoth_sim.py:1314-1346`, `rust/wesnoth_core/src/lib.rs:102-169`,
  terrain cost then a defense/ally subcost), as Wesnoth's own pathfinder
  routes a click. Any other route can be ordered as a chain of partial
  moves. Under fog the route decides which hexes get revealed, so this
  matters only together with F2. Inferred no loss on its own.
- **Resign, chat, labels, hold position, end-unit-turn** (`menu_events.cpp:1007-1041`:
  local flags; labels and chat change nothing in the game). Not orders of
  play.
- **Debug commands** (`synced_commands.cpp:450-765`): not available in a
  ladder game. Not a gap.

### F4. The decision to attack or move is made before, and without, the target scores

- `type_logits = self.type_head(actor_ctx)` (`wesnoth_ai/model.py:409`,
  `:1173`): P(ATTACK | unit) vs P(MOVE | unit) is a linear read of the
  acting unit's token alone. The target distribution is drawn afterwards
  from one shared logit row `q(actor)·k(hex)` (`model.py:419-421`), masked
  by type (`server_priors.py:333-334`). So "should this unit attack or
  move" cannot read how good the best attack or best move is, except
  through what the trunk has already put into the actor's own token. A
  player makes the two choices together (they click the hex).
- This is a factorization, not a lost order: every (type, hex) pair stays
  reachable and the product is a full joint law (`server_priors.py:385`).
  The cost is in learning: the model has to put an estimate of its own
  target scores into the actor token. Compare a pointer over all of the
  unit's legal hexes at once (attack and move together), where the type
  follows from the hex.
- **Confidence:** verified by reading (structure); the effect on play is
  inferred.
- **Run to confirm:** "To check by a run", item 5.

### F5. Stale statements in the code about the action space (documentation)

Each is wrong against the code it sits in. People and future audits rely on
them.
- `wesnoth_ai/action_sampler.py:41`: "End-turn: ... gated while
  moves/recruits". End turn is always legal: `action_sampler.py:1053-1054`,
  `core_masks.rs:77`. (That is right for Wesnoth: `menu_handler::end_turn`,
  `menu_events.cpp:571-622`, only asks for confirmation.)
- `wesnoth_ai/action_sampler.py:36-37`: "enemies the unit is adjacent to or
  can land next to this turn". Since 2026-11-24 only adjacent enemies:
  `rust/wesnoth_core/src/lib.rs:403-413`, `action_sampler.py:1115-1117`.
- `tools/wesnoth_sim.py:1276-1278`: "The policy's mask restricts moves to
  adjacent hexes". The mask offers the whole single-turn reach
  (`core_observe.rs:136-137`, `lib.rs:317-321`).
- `tools/wesnoth_sim.py:29-31` and `:737-740`: "self-play draws uniformly";
  not on the simple loop's path (F1).
- Confidence: verified by reading.

No leak of the god view into the masks was found (see Coverage, "Principle 6").

---

## 2. Coverage (checked and found present and right)

### The action space against Wesnoth's synced commands

`src/synced_commands.cpp` (1.18.4) registers: recruit `:70`, recall `:134`,
attack `:152`, disband `:233`, move `:260`, fire_event `:326`,
custom_command `:353`, auto_shroud `:367`, update_shroud `:383`, and
debug_* `:450-765`. End turn and init side are not synced commands.

- **Move to any single-turn reachable hex, including partial moves.** The
  move row is the landable row: every hex the Dijkstra reaches within
  `current_moves`, minus the start hex and visibly occupied hexes
  (`lib.rs:317-321`), over the side's own view (`core_observe.rs:136-137`).
  Any hex inside the reach is offered, not only the far edge.
  `core_masks.rs:108-131`, `action_sampler.py:1118-1131`.
- **Move into fog.** Fogged hexes are offered: the Dijkstra reads terrain
  costs and the visible units only (`lib.rs:126-167`, `observe.rs:144-166`;
  `occupied` is set for visible units only, `observe.rs:145-149`). A hex
  under a hidden unit stays offered, and the core resolves it as Wesnoth
  does (blocked/ambush, `core_move.rs:136-137`).
- **ZoC and skirmisher.** A visible enemy's ZoC (level >= 1, not petrified,
  `core_observe.rs:67`, `observe.rs:158-165`) ends movement on entry
  (`lib.rs:131-133, 150-154`, `pathfind.cpp:806` cited). Skirmisher ignores
  it (`lib.rs:131, 150`). Hidden units exert no ZoC (`docs/wesnoth_rules.md:641-643`).
- **Slowed movement** uses the slowed movement class
  (`core_observe.rs:123`).
- **Continuing an interrupted move.** The unit keeps its moves after a
  blocked stop (`docs/wesnoth_rules.md:640`), so the mask offers its new
  reach. (Sighting stops: F2.)
- **Attack with a chosen weapon.** Targets: seen, non-allied, non-inert
  enemies on the six hexes touching the unit (`lib.rs:403-413`,
  `observe.rs:72-74`, `core_masks.rs:85-89`), for a unit that has not
  attacked (`core_observe.rs:118`). Weapons: every slot below the unit's
  count (`core_masks.rs:78-80`, `server_priors.py:330-331`), as in the
  player's dialog (`mouse_events.cpp:1272-1296`, which lists every weapon
  with `attack_weight() > 0` and no active `disable` special; see "Could not
  determine"). `MAX_ATTACKS = 4` (`core_masks.rs:33`) covers the most weapons
  any type has in `unit_stats.json` (3, Drake Enforcer).
- **Weapons whose range the defender cannot answer** are offered like any
  other: the weapon mask depends on the weapon count only.
- **Defender's weapon** is the engine's choice for a human too; the
  simulator resolves it with the core's port of the engine's rating
  (`tools/wesnoth_sim.py:1407-1414`).
- **Petrified or scenery targets** are excluded (`observe.rs:150-152`,
  `observe.rs:72-74`), as in `mouse_events.cpp:753` (cited at
  `action_sampler.py:1081-1084`).
- **No moves after an attack.** The core sets `current_moves = 0` and
  `has_attacked` (`core_attack.rs:427-431`), matching
  `actions/attack.cpp:1371-1372` (`movement_used`, default everything). A
  unit with 0 moves that has not attacked still attacks an adjacent enemy
  (`core_observe.rs:112-118`).
- **Recruit a chosen type on a chosen castle hex.** One actor slot per
  recruit type (`model.py:393-401`), then a hex from the castle network of
  the leader's keep (`observe.rs:171-200`). Matches
  `game_state::can_recruit_on` (`src/game_state.cpp:320-351`): the
  recruit hex is castle (keeps included, `castle_or_keep`), the leader is on
  a keep, no **visible** unit on the hex, and a path of castle hexes from
  the keep, which units do not block (`castle_cost_calculator`,
  `game_state.cpp:255-278`; the BFS goes through occupied hexes,
  `observe.rs:188-196`). Gold: `cost > gold` refuses (`core_masks.rs:139-144`).
- **A leader that moved to another keep** recruits there: `leader_on_keep`
  and the network are computed from the leader's current hex
  (`observe.rs:174-177`). A leader recruits whatever its moves or attack
  (no movement test in `observe.rs:171-200`).
- **A castle partly held**: the visibly occupied hexes are dropped, the rest
  offered (`observe.rs:194`).
- **A recruit onto a hex a hidden unit holds** goes to the nearest vacant
  castle hex, gold spent (`tools/wesnoth_sim.py:1471-1492, 216-250`;
  `actions/create.cpp:397` checks `can_recruit_on`, which reads visible
  units only, `game_state.cpp:339`). With no vacant hex it is refused and
  decided again (`tools/eval_players.py:153-168`,
  `tools/selfplay_game.py:195-213`). The hex joins the turn's rejection set
  (`wesnoth_sim.py:1484`, `observe.rs:194`).
- **End the turn at any moment**: always legal (`core_masks.rs:77`,
  `action_sampler.py:1053-1054`), one actor slot competing with every unit
  and recruit (`server_priors.py:325-326`).
- **Mask and simulator agree**: the simulator refuses a move outside the
  planner's reach and an attack on a non-adjacent unit
  (`wesnoth_sim.py:1320-1340, 1364-1371`) and a match stops on any such
  refusal of a mask-offered order (`eval_players.py:179-187`), so the two
  agree in every match game that finishes.
- **The relevant hex subset** keeps every hex an order can name: villages,
  castles, the network, occupied hexes, the leader's hex and every landable
  hex of a unit that can move (`core_observe.rs:138-159`).

### Principle 6 (no god view in the masks)

- Unit visibility: own and scenery always, a hider only when an enemy of it
  stands adjacent, others only on a seen hex under fog (`observe.rs:111-131`).
- Occupancy, enemy, ally and ZoC flags come from the visible units only
  (`observe.rs:144-166`); the landable rows read only those
  (`core_observe.rs:136-137`); the attack targets only the seen ones
  (`core_masks.rs:85-89`); the recruit row only visible occupancy and the
  turn's rejection set (`observe.rs:194`).
- Gold and weapon counts are the side's own (`core_masks.rs:78-80, 139-141`).
- Scenery is visible under fog (`observe.rs:113-119`), a known exception by
  the user's ruling of 2026-09-29.

### What each score reads (question 3)

The trunk is full self-attention over every token with only a padding mask
(`model.py:382-383, 616`): hex tokens, unit tokens, recruit tokens, the
sighting and memory streams when the model has them (`model.py:590-597`),
the global token and the end-turn token. So every contextualized token can
depend on the whole visible board and the memory. The heads
(`model.py:1155-1196`, the same as the single path `model.py:403-438`):

| Head | Score computed from | Cannot depend on (beyond what the trunk put into that token) |
|---|---|---|
| Actor (which unit, which recruit type, or end turn) | `actor_head(actor_ctx)`, a linear read of each actor token alone; one softmax over all own legal actors and end turn (`model.py:1172`, `server_priors.py:325-326`) | the other actors' scores |
| Type (attack or move) | `type_head(actor_ctx)` (`model.py:1173`) | the target scores (F4) |
| Target hex (move and attack) | `q(actor_ctx)·k(hex_ctx)/sqrt(d)` (`model.py:1178-1180`): reads which unit acts | the type beyond the mask (one row serves both; the supports are disjoint, so nothing is lost) |
| Recruit hex | the same pointer, with the recruit type's token as query (`model.py:393, 1178-1180`): reads which type is recruited | nothing missing |
| Weapon | `weapon_head(actor_ctx)` (`model.py:1174`) | the target (known) |
| End turn | `actor_head` on the end-turn token, a learned constant input (`encoder.py:758`) contextualized by the trunk (`model.py:390, 393`): reads the whole board through attention | nothing missing |
| Value | `value_head(global_ctx)` plus a material projection when the checkpoint has `value_material` (`model.py:312-321, 1185-1192`) | everything outside the global token's context |

- The joint probability of an order is the product
  actor x type x target x weapon (`server_priors.py:374-385`;
  `action_sampler.py:577-583`), sampled or argmaxed as one joint law
  (`raw_player.py:66-76`). Pairs chosen independently of each other given
  the actor: the target and the weapon only (known). The type is chosen
  before the target (F4). A recruit's type and hex are a proper chain.

### Decode (stated only)

- **The player being trained**: `raw:t1`, a draw from the joint prior at
  temperature 1 (`scripts/simple_loop_local.py:139`, `raw_player.py:66-76`),
  no end-turn offset (`raw_player.py:44-63`, offset 0), 64 memory slots,
  fp16 serving on the shared server (`scripts/match_local.py:114-115`).
- **The reference** (`configs/reference_player.json:9-14`): `own2`, 64
  slots, `raw:t0+eo-1.5`: argmax of the joint prior after the end-turn
  actor logit is shifted by -1.5 (`raw_player.py:44-63, 70-71`).
- **Nothing beside the network alters the trained player's scores at decode
  time** apart from the legality mask: the server path has no gate, bias or
  offset (grep `gate|offset|nudge|bias|temperature|logit` over
  `tools/eval_inference_server.py`, `tools/serve_stage.py`,
  `tools/inference_seam.py`: only transport and the weight-publish gate);
  `forbid_end_turn` is False (`raw_player.py:100, 116`); mcts-sims 0. Parts
  of play are still decided beside the network: the advancement (F1), the
  route to a chosen hex (F3; Wesnoth's own default), the defender's weapon
  (the engine's rule) and a recruit's landing hex when the ordered hex is
  taken (the engine's rule).

---

## 3. To check by a run

1. **The advancements the loop played** (F1). Over one round's game records
   (`round_0583\games_self` and `games_past`), rebuild each game
   (`tools/game_record.py`, `start_core`), and after each attack read
   `core.advance_state_export()[2]` (`last_advance_events`, (side, index)).
   Count the advancements of units whose type has two or more
   `advances_to`, and confirm the index is 0 in all of them. Report how many
   advancements per game had a real choice.
2. **The training checkpoint's structure**: `peek_checkpoint_arch` on
   `round_0583\pi.pt` (`tools/eval_players.py:253-288`). Read
   `value_material`, `observation_parity` and `memory_slots`, which settle
   the value head's exact input (Coverage table).
3. **How often a simulated move would have stopped on a sighting** (F2).
   For each move command of a round's games, apply it step by step on a
   copy of the core and count the moves in which a previously unseen enemy
   becomes visible before the last hex (equivalently, export the games with
   `skip_sighted` left out and strict-sync them in Wesnoth; that is an
   engine oracle and needs the user's agreement).
4. **Default-era weapons with `attack_weight=0`, a `disable` special or a
   non-default `movement_used`/`attacks_used`**: scrape the 1.18.4 tag's
   `data/core/units/` (not in the repository) for those keys. None is
   expected; one would make the weapon mask offer a weapon the player
   cannot choose, or remove a move a player keeps after an attack.
5. **Whether the type head's choice is the bottleneck** (F4): on logged
   decisions, compare P(ATTACK | unit) with the share of the unit's best
   attack in a target softmax over its attack and move hexes together.
   Diagnostic only.

## 4. Could not determine

- Whether any default-era weapon has `attack_weight=0` or a `disable`
  special: `unit_stats.json` does not record the keys and
  `wesnoth_src/data/core/units/` is not in the repository (item 4 above).
- Whether the Plan Unit Advance modification is on by default in ladder
  games, or how often players use it; the corpus has it
  (`docs/wesnoth_rules.md:3526-3531`), the simulator's scenario games do
  not set it (no `plan_unit_advance` in `wesnoth_ai/rules/` or
  `tools/elo_eval_game.py`).
- Undo's exact conditions in 1.18.4 (`src/actions/undo*.cpp` not read).

## 5. For other parts

- `docs/status.md:576-577` says the network cannot see the rejected recruit
  hexes; CLAUDE.md Principle 6 says the encoder has a `recruit_rejected`
  bit. The input part should check which is current.
- A player sees an order's destination route and the multi-turn goto
  markers of their own units on screen (`menu_events.cpp:903-993`); the
  memory part may want to check that a planned goto needs no input.
- `tools/elo_eval_game.py:637` defaults `--max-turns 200`; the settings part
  should compare that with each ladder scenario's own turn limit.
- When a player clicks an enemy from a distance, the attack dialog shows the
  fight's numbers from the hex the unit would attack from, before it moves.
  That is ruled out of the input by the owner (no worked-out fight outcome);
  noted only.
