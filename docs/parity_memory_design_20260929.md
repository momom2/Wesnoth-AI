# The parity-memory retrain: design (2026-09-29)

One retrain that gives the network what a player has: everything the
interface shows in the present state, the enemy's faction only as far as a
player can know it, what the player watched during the enemy's turn, and a
learned memory for everything older. The pre-registration of its match is
docs/parity_memory_prereg_20260929.md.

## Rulings (user, 2026-09-29)

1. For each item of the present state that a player sees, the network sees
   it too (docs/observation_parity_20260926.md lists the gaps).
2. For previously seen enemies the network has a memory, and it is
   learned: a list of all earlier observations would grow without bound,
   and a learned memory can also cover gaps in the observation nobody has
   found yet.
3. The enemy's faction is known only as a player knows it. The input is
   the exact posterior over factions; there is no faction head.
4. Statues on fogged hexes stay visible. This breaks principle 6 and is
   accepted explicitly: a player who knows the map knows where they stand.
   The code says so where it shows them.
5. The terrain classes that share a row today (mushroom grove, reef) get
   their own, and each hex carries its own time of day.
6. The memory has 64 slots, trained at varying sizes so that one checkpoint
   can be measured at several.
7. Everything goes into one retrain, which runs only if the audit finds
   nothing that would force another (docs/parity_memory_prereg_20260929.md
   "Audit gate").

## What the network observes

Behind one checkpoint flag, `observation_parity`, with
`OBSERVATION_EPOCH` 11. `obs8` keeps its encoding (flag absent).

**Where it is built.** Only in the Rust core's `GameCore.encode_streams`.
Every consumer that encodes under the flag goes through a core: the
pre-encoding already does; the eval player, the pool's root decisions and
the holdout probe encode a fork of the simulator's core instead of a deep
copy of its view (a deep copy is unbound, and today takes the Python
builders). The Python builders and the kernel `encode_raw_streams` refuse
the flag; they serve `obs8` until the retirement deletes them.

**Units** (visible units of every side), added to today's 13 columns:

| columns | values |
|---|---|
| 3 weapon slots, in the unit's attack order (the weapon head's index) | present; damage after traits and objects / 40; strikes / 6; ranged; damage type (6, one-hot); specials (10, multi-hot: magical, poison, slow, marksman, firststrike, drains, backstab, charge, plague, berserk) |
| resistances (6) | (100 - damage percent) / 100, in blade, pierce, impact, fire, cold, arcane order |
| traits (13, multi-hot) | strong, quick, intelligent, resilient, healthy, dextrous, weak, slow, dim, fearless, undead, feral, elemental |
| abilities (13, multi-hot, from the unit's record, so scenario objects count) | leadership, skirmisher, regenerate, submerge, cures, ambush, heals +4, steadfast, feeding, illuminates, nightstalk, concealment, teleport |
| statuses (2) | poisoned, slowed (petrified stays side code 2) |
| time of day at the unit's hex | the engine's combat modifier for the unit's alignment and fearless trait / 25 |
| leadership received | the bonus percent the unit fights with now / 25 |

`UNIT_FEAT_DIM` 13 to 13 + 60 + 6 + 13 + 13 + 2 + 1 + 1 = 109. The ranges
come from the 190 reachable unit types (largest damage 40, Dwarvish
Dragonguard; most strikes 6, Inferno Drake; resistances 0 to 200 percent).
The time-of-day and leadership columns are the two factors the interface
folds into a weapon's displayed damage; with them, the displayed number is
a function of the unit's own columns.

**Recruit options** take the same columns from the type's base values (a
recruit's traits are rolled when it is recruited), and code the alignment
as board units do: today lawful and neutral are swapped between the two.

**Hexes:**
- a dynamic flag: the side sees this hex now (the fog overlay);
- a dynamic column: the hex's lawful bonus minus the board's, / 25 (time
  areas, lit terrain, illumination);
- two terrain classes, FUNGUS (`Tt`, mushroom grove) and REEF (`Wrt`), so
  `NUM_TERRAINS` is 16;
- the static village bit on every village hex, whether or not its owner is
  visible (a water village under fog reads as shallow water today).

`NUM_HEX_DYNAMIC_FLAGS` 3 to 5.

**Globals**, `GLOBAL_FEAT_DIM` 8 to 15: village gold / 8; village support
/ 4; own net income as the status table shows it / 50; fog on; and, with
fog off only (zero under fog), the enemy's gold / 500, net income / 50 and
upkeep / 50.

**The enemy's faction.** The global token adds, in place of the enemy
faction's embedding row, the posterior-weighted sum of the rows. The prior
is one-hot when the opponent chose a faction openly and uniform over the
era's factions when it chose Random (6 in the default era, 7 with
Dunefolk). The likelihood of a faction is 1 when it can field every enemy
unit type the side has seen, its leader included, and 0 otherwise. A
faction can field its recruits, its leaders and random leaders, and every
type they advance to (plague corpses are Walking Corpse variations, which
the Undead recruit). With the faction chosen openly the vector is one-hot,
which is `obs8`'s input; in eval the harness assigns factions openly.
A seen set that no faction can field leaves the prior in place and counts
as an error in the pre-encoding manifest.

**The watched turn.** A player watches the enemy's turn and sees every
enemy unit that crosses a hex the player can see, including units that end
their move in fog. The core keeps, per side, the enemy units that side saw
since its last end_turn: after every command, the visible enemy units;
during a move, each hex of the path the side can see, the unit's hiding
rules applied. At the side's decisions, each of these units that it cannot
see now becomes a sighting token at the last hex it was seen, with its type,
hit points and maximum hit points. The record is cleared at the side's
end_turn, so it never holds more than one enemy turn; anything older is the
memory's to keep. Sighting tokens are a new stream with their own token
kind and side code 3; they are never an actor or a target. The same
sightings feed the faction posterior's seen set, which is kept per side for
the whole game.

**The relevant set, version 2.** Today's set plus the six neighbours of
every own unit (the hexes an enemy must stand on to attack it; 30,648 of
142,848 such hexes lacked a token in the parity census) and every hex within
6 hexes of an own unit that the side does not see now (about 17 hexes a
decision in the census). About 9% more hex tokens.

**Game records.** `state_digest` covers the sighting record and the seen
sets; records written before this version verify under the old digest.

## The memory

- 64 slots of the model's width. At each of its decisions a side's network
  reads the memory it wrote at its previous decision and writes a new one;
  each side of each game has its own, starting from a learned initial
  memory.
- The active slots enter the trunk as tokens of a MEMORY kind, each with a
  learned slot embedding added, beside the other tokens; full attention.
- The write is gated per slot and channel, as in GTrXL (Parisotto et al.,
  "Stabilizing Transformers for Reinforcement Learning", ICML 2020):
  `z = sigmoid(W_z [h; m] + b_z)`, `m' = (1 - z) * m + z * tanh(W_c h)`,
  with `h` the trunk's output at the slot and `b_z` initialized to -2, so
  a fresh network keeps most of its memory at each step.
- The state and the write are float32 everywhere. The trunk may run in
  bfloat16, which reads a cast of the state; a bfloat16 state would lose a
  small update to rounding at every step, over 100 to 300 steps a game.
- **Sizes.** Only the first `k` slots are active, `k` fixed for a whole
  game-side. Training draws `k` per game-side from {0, 8, 16, 32, 64} with
  probabilities {1/8, 1/8, 1/8, 1/8, 1/2} (nested dropout: Rippel,
  Gelbart and Adams, "Learning Ordered Representations with Nested
  Dropout", ICML 2014; Kusupati et al., "Matryoshka Representation
  Learning", NeurIPS 2022). Inactive slots are left out of the sequence,
  not masked. `k` = 0 is the same network without memory.
- The checkpoint records `memory_slots` 64; a player is built with its
  `k`, recorded as an estimand on every eval edge (`memory_a`, `memory_b`).

## The belief head

For each hex token, the probability that an enemy unit the side cannot see
stands there, trained with binary cross-entropy against the true state
(the pre-encoded observation holds every unit's hex), averaged over the
hex tokens with no visible unit, weight 1 beside the policy and value
losses. It teaches the memory to track hidden units, which the policy loss
alone does only weakly, and it is the belief model a determinized search
root will sample from (docs/hidden_information_20260926.md). The truth is a
training target only: it never enters an input. The manifest counts the
hidden enemy units whose hex has no token.

## Training

- **Data:** the corpus rebuilt at `CORPUS_VERSION` 3 (version 2's
  corrections, docs/corpus_v2_20260926.md, plus each side's `chose_random`
  and the era, which the faction prior needs), the fresh vocabulary of
  190 unit types (docs/unit_vocab_retrain_prereg_20260925.md), the
  player-side and plague corrections already in the code. Version 3 also
  applies without pairing the moves the engine makes for standing orders
  at a turn start, and leaves out every game with shroud (user ruling
  2026-09-30: our games do without it).
- **Pre-encoding:** every decision of both player sides, in order, per
  game, with the flag on; beside each pair its side, whether it is a value
  state, and the belief targets (the hex-token indices of hidden enemy
  units) in a structure of their own, never in `RawEncoded`. A turn that
  ran out of time keeps its last position with the TIMEOUT label (user
  decision 2026-09-30): the memory, value and belief losses see it, the
  policy has no target there. Training and eval games have no turn timer
  (the simulator has none; clock play is not part of what the policy
  learns, user ruling 2026-09-30).
- **Streams:** each game gives two streams, one per side. 32 streams run
  side by side; each optimizer step unrolls 16 decisions of each (512
  positions), back-propagates through the memory across them, and carries
  the memory into the next window without gradient (truncated
  back-propagation through time; Williams and Peng, 1990). A stream that
  ends is replaced by the next game-side of the epoch's shuffled order.
  Each step's trunk is checkpointed and recomputed in the backward pass, to
  fit 512 positions on a 24 GB card.
- **Losses:** `obs8`'s policy loss (winners' decisions only, per-game
  weight, label smoothing 0.05); its value loss on its value states
  (each state kept with probability min(1, 16 / commands in the game)); the
  belief loss on every position.
- **Optimizer:** AdamW, weight decay 1e-4, as `obs8`; lr 2.8e-4, `obs8`'s
  1e-4 at batch 64 scaled by the square root of the batch ratio (Malladi et
  al., "On the SDEs and Scaling Rules for Adaptive Gradient Algorithms",
  NeurIPS 2022), with a 300-step linear warm-up; constant for the pass, as
  `obs8`'s first epoch was. One pass. Gradient clip 1.0.
- **Holdout probe:** holdout game-sides run whole, in order, at `k` = 0,
  16 and 64: policy CE on winners' decisions, per-game value AUC, the belief
  loss, and the belief loss of a last-seen baseline (the probability that a
  hidden unit stands where the side last saw it, the rest spread evenly;
  both rates fitted on training streams). Every 500,000 positions and at
  the end.
- **Resume** continues the pass exactly: the checkpoint holds each stream's
  game-side, offset, `k` and carried memory.
- **Telemetry:** the signal probe of `supervised_train` (each loss's share
  of the gradient per block), plus the norms of the memory write's gradient.

## Serving and play

- The eval player keeps each side's memory for the game and sends it with
  every request; the inference server returns the new memory with the
  outputs. Requests carry float32 state; the server batches memory tokens
  like any other stream.
- A recruit that bounces and is decided again is one more decision, one
  more memory step, as in the corpus.
- Search, the self-play pool, the graphed server and the live bridge refuse
  a memory checkpoint until they carry the state (a fork copies both
  sides' memories with the simulator); this retrain's verdict needs only
  raw players.

## Rejected

- A harness-kept list of every enemy seen, with its last hex and turn:
  the user's ruling for a learned memory (ruling 2).
- A faction head trained on the true faction: the exact posterior is
  cheap to compute, so the head would only learn to copy it (ruling 3).
- A memory step at every enemy decision, to watch the enemy's turn: it
  doubles every forward pass in play and training and still misses a unit
  that crosses the side's view within one move; the sighting record sees
  both.
- Losers' streams forward-only, to save about a quarter of the box: the
  belief head and the memory's writes would learn from half the games.
