# From the tensor to the decision: audit report (2026-11-25)

Part: whether what the encoder emits can reach the network's decision.
Read on `main` at 36b3f5f1 (0.36.0). The live run executes from the
worktree `feature/input-facts` at 2a523b56, which `main` merged; I read
`main` only. Nothing was run except two stdlib-only Python reads: the
pool's map sizes from the `.map` files, and the vocab and structure keys of
`C:\Users\momom2\wesnoth_runs\simple_loop\round_0576\pi.pt`, unpickled with
a stub class loader, without torch.

The live network, from that checkpoint: d_model 384, 8 layers, 12 heads
(head_dim 32), d_ff 1536, observation_parity, relevant_set_hexes with
relevant_set_version 2, terrain_multi_hot, fog_hides_enemy_villages, 64
memory slots, unit_row_version 2. It has 237 vocab names on ids up to 189,
and no name reaches the overflow row 199.

The production paths:
- **Serving** (self-play and matches): `tools/eval_inference_server.py`
  with `--infer-fp16`, the packed trunk and the Rust stage, via
  `tools/serve_stage.py`.
- **The pass:** `tools/pg_game_pass.py --fp16 --packed-trunk`
  (`scripts/simple_loop_local.py:295-298`).

---

## 1. Findings, most serious first

### F1. The new projection's term is about one fp16 rounding step of the token it joins. Serving keeps it exactly only in the float32 residual stream; every matrix product reads it rounded.

**What the network gets (verified by reading):**
- **Serving builds the unit tokens in float32.** `serve_stage.py:497-498`
  calls `encoder.embed_staged(staged)` outside any autocast. The fp16
  autocast wraps only `model.forward_embedded` (`serve_stage.py:500-502`).
  So `unit_feat_proj(feats[:, :109]) + unit_feat_proj2(feats[:, 109:])`
  (`encoder.py:817-824`) is summed in float32 with the type, side and
  position embeddings (`encoder.py:1240-1246`).
- **The trunk is the fused fp16 HIP loop** (`model.py:996-1045`, chosen
  at `model.py:856-864` because `_hip_fp16_inference` holds: eval mode,
  no grad, fp16 autocast, head_dim 32 in `HEAD_DIMS`):
  - `h32 = x.float()` is the float32 residual stream.
  - `h16 = h32.half()` (`model.py:1021-1022`) is what every linear
    reads: QKV, out-projection, FF1, FF2.
  - `hip_add_layer_norm` (`hip_elementwise.py:55-66, 259-282`) adds the
    fp16 sublayer output to the float32 residual in float32, normalises
    in float32, and writes both a float32 copy and an fp16 copy.
  - The attention kernel takes fp16 q, k, v and rounds its probabilities
    to fp16 before the PV product (`hip_kernels.py:200-203`).
- **The heads run under the fp16 autocast** (`model.py:1155-1180`). The
  actor, type and weapon linears and the target `bmm` produce fp16
  logits. These are cast to float32 only afterwards
  (`serve_stage.py:502`, `.float32()`), so the logits are quantised at
  fp16 resolution.

**The arithmetic.** fp16 has a 10-bit mantissa:
- The rounding step (ulp) is 2.4e-4 on [0.25, 0.5), 4.9e-4 on [0.5, 1),
  9.8e-4 on [1, 2), 1.95e-3 on [2, 4) and 3.9e-3 on [4, 8).
- The smallest normal number is 6.1e-5; subnormals are spaced 6.0e-8.

The sizes at round 576
(`training/metrics/input_facts_20261125/new_weights_round_0576.json`):

| Quantity | Value |
|---|---|
| `unit_feat_proj2` weights, root mean square | 6.19e-5 |
| `unit_feat_proj2` weights, largest | 2.36e-4 |
| `unit_feat_proj` weights, root mean square | 0.104 |

A unit row has about 34 non-zero added columns (level, where it stands,
16 defences, up to 16 movement columns), each roughly 0.3 to 1. A unit
next to an enemy has some 20 to 40 more fight columns.

So the added term per token component is c ≈ 6.2e-5 × sqrt(Σ x²):
- about 2e-4 if the weight signs are unrelated to the columns' signs;
- up to about 1e-3 if they line up.

The old projection's term is about 0.104 × sqrt(Σ x²) ≈ 0.3 to 0.4. The new
term is therefore about 6e-4 of the old term's size, which is fp16's
relative rounding step (2^-11 = 4.9e-4 is half an ulp, relative).

The token components themselves are sums of five or six embeddings.
`nn.Embedding` starts each at N(0, 1), and their trained sizes I could not
read without torch. At a component of size 1 to 4, c is 0.1 to 0.4 ulp.
After the first layer norm, the components are of order 1 and c shrinks by
the norm's 1/std, so it is again roughly 0.05 to 0.2 ulp.

**What follows (inferred from the arithmetic):**
- **It is not lost outright.** It is carried exactly in the float32
  residual stream: in the token sum, in `h32` through every add and norm,
  and in the memory write, which runs in float32 (`memory.py:117-126`).
- **At each cast to fp16 it survives only as dithering.** Adding c
  changes the fp16 value of a component with probability about |c|/ulp
  (10 to 40%). The expected effect is c, and the deterministic error
  around it is of order sqrt(c·ulp), larger than c itself.
- **The output logits are fp16.** Where a logit's size is 4 to 16, its
  step is 0.004 to 0.016, so a change of the new columns smaller than
  that moves a decision only through the same rounding.
- **The gradient is not affected the same way.** In the pass the autocast
  casts pass the gradient straight through, so the pass's gradient for
  `unit_feat_proj2` is not cut by these roundings.
  - The loss scale stayed at 1024 with no overflow in round 576
    (`wesnoth_runs/simple_loop/round_0576/grad.json`).
  - So the weights train at the ordinary pace, as recorded.
  - The open question is only how much of their present effect on play
    survives fp16 serving. As the weights grow (about 0.2 lr a round),
    c moves past one ulp in some tens to hundreds of rounds, and the
    question goes away.

**Why it matters for play:**
- At present weights, a fight column's value can change the attacker's
  logits by less than one fp16 logit step. Two attackers whose columns
  differ (70% to hit on a village against 40% in the open) may then get
  logits that differ only by rounding noise.
- The network may be slower to use the new numbers in play than its
  float32 weights would suggest.

**Confidence:** verified by reading for the code paths and dtypes;
inferred for the magnitudes (the token sizes are not measured).

**How a run would confirm it:** checks R1 to R3 below.

### F2. The pass builds the input tokens in fp16; serving builds them in float32. The precision differs before the trunk.

**What differs (verified by reading):**
- **The pass embeds inside the autocast.** In `pg_game_pass.py:257-258`,
  `self.encoder.embed_staged(staged)` runs inside
  `torch.autocast(..., dtype=float16)`, so every `nn.Linear` of the
  embedding runs in fp16, with fp16 weights, fp16 inputs and fp16 output:
  - `unit_feat_proj`;
  - `unit_feat_proj2`, whose weights below 6.1e-5 (most of them) become
    fp16 subnormals;
  - `modifier_proj`, `dynamic_flag_proj`, `global_proj` and
    `sight_feat_proj`;
  - the terrain multi-hot product `bits @ terrain_embed.weight`
    (`encoder.py:833-834`).
- **The embeddings stay float32 in both paths** (`nn.Embedding` is not
  on autocast's lists), so the sum is float32 in both.
- **Serving runs all of these in float32** (`serve_stage.py:497-498`;
  likewise `inference_seam.py:585` and `inference_seam.py:595-597`).

**The size of the difference:**
- The old unit projection's output (about 0.36 a component) is rounded
  to about 1.2e-4 in the pass and not in serving. This error is of the
  same size as F1's c, the whole new-projection term.
- The new projection itself is computed to about 0.1% in fp16, which is
  negligible.

**What it does to the run:**
- It is one cause of the replay identity's residual: half mean square
  7.3e-6, max |d| 0.078 nats, at round 576 (`round_0576/grad.json`).
- It is not a blind spot, and it does not bias the gradient's direction
  measurably. But it means the pass differentiates a function whose
  inputs differ from the served ones by as much as the new term's whole
  size.
- If rocBLAS flushed fp16 subnormals to zero (see "Could not
  determine"), the pass's forward would drop most of `unit_feat_proj2`'s
  weights while serving kept them. The backward would still train them,
  and the replay check could not see it at today's sizes (F3).

**Confidence:** verified by reading (dtypes); the share of the residual is
inferred.

**How a run would confirm it:** R4 (the pass with the embedding taken out
of the autocast; compare `half_mean_square` on the same round), and R3
for the subnormals.

### F3. The "exact check" cannot tell, at present weight sizes, whether serving reads the new columns.

**What the check is (verified by reading `pg_game_pass.py:48-59, 271,
419-422`):**
- d = (logged log p) − (the pass's log p) at every trained decision.
- The pass is refused if ½·mean(d²) > 1e-4, that is an rms d above 0.014
  nats.
- Round 576 read 7.3e-6, an rms d of 3.8e-3.

**What it would catch:** any per-decision difference between the served
law and the pass's law larger than about 0.014 nats rms, from any cause:
- an encoding drift between play and replay;
- a mask that differs;
- a memory that diverges, including the memory of refused (shadow)
  decisions;
- a projection or column misfed in ONE of the two paths, once that
  projection's effect on log p exceeds about 0.014 rms.

**What it would not catch:**
1. **A path that leaves `unit_feat_proj2` out today.** Its whole effect
   on log p is far below the fp16 noise floor (inferred from F1's
   sizes). Such an omission would add in quadrature to 3.8e-3 and stay
   under the limit.
2. **A fault common to both paths.** Both read the same Rust encoding
   (`pg_positions.py:244-245` re-encodes with the same core and
   `unit_row_version`) and the same `GameStateEncoder.embed_staged`. So a
   wrong column, a wrong order, or a projection left out of
   `unit_row_tokens` would be invisible to it.
3. **Precision losses common to both** (F1).

**What does cover the omission today (verified by reading):**
- **Every path builds unit and recruit tokens through one function,**
  `_unit_embedding`, which calls `unit_row_tokens`:
  - `encoder.py:1066-1072` (`encode_from_raw`);
  - `encoder.py:1095-1101` (recruits);
  - `encoder.py:1240-1246`;
  - `encoder.py:1385-1389` (`embed_streams`, used by `serve_stage`,
    `pg_game_pass`, `pg_pass`, `pg_eval`, `pg_floor`, `pg_plant_grad`,
    `sequence_train` and `bench_serve_host`);
  - `encoder.py:1498-1501` (`_embed_streams`);
  - `graphed_serve.py:536-538`.

  A grep for `unit_feat_proj` and `_unit_embedding` over every `.py`
  found no other builder.
- **`tests/test_unit_rows2.py:584-629`** pins the pass against logged
  probabilities with the new weights at 0.3·randn. It runs on the CPU, so
  the Rust-stage fp16 HIP serving path is not in it.
- **A GPU check with loud weights was done once by hand** (the session's
  memory records 0.0015 against 0.22). It is not a test.

**Why it matters:** the check is the run's guard on serving/pass identity,
and it is blind to the very change just made until the new weights are
about 100 times larger.

**Confidence:** verified by reading the check; inferred for the size of
the new term's effect on log p.

**How a run would confirm it:** R2.

### F4. The recruit-rejected column is written 0 under the parity observation: the mask reads the rejection, the network never sees it.

**What a player sees and can deduce:**
- When a recruit ordered onto a fogged castle hex that a hidden enemy
  holds is relocated, the player sees the recruit appear on another hex
  (`docs/wesnoth_rules.md`, "A recruit onto an occupied hex").
- The player can deduce that an enemy stands on the ordered hex.
- CLAUDE.md, Principle 6, states that "the encoder shows a
  `recruit_rejected` bit on it. The mask and the model read the same
  state."

**What the network gets (verified by reading):**
- `rust/wesnoth_core/src/core_encode.rs:370-381`: under the parity flag,
  `rejected` is an empty vector, so hex dynamic column 0 is always 0. The
  comment's reason: "a replay records only where a recruit landed, so
  corpus reconstruction never sets it and its weights would never train."
- `encoder.py:308-309` says the same ("the recruit-rejected column is
  written 0").
- **The legality mask does read the rejections.**
  `core_observe.rs:74` passes `self.recruit_rejected` to the side view.
  The mask is not an input to the network: it only removes logits after
  the heads (`server_priors.py:323-335`).
- So the live network (observation_parity) is told nothing.
  - On a refused recruit, the side's memory is rolled back
    (`raw_player.py:262-279`; the pass's shadow rows,
    `pg_schedule.py:108-112`). The next decision then reads exactly the
    same tokens and memory as the refused one.
  - On a relocated recruit, the fact "a hidden unit stands on hex X" is
    in neither tokens nor memory, except through whatever the network
    infers from the recruit landing elsewhere.
- **The comment's reason no longer holds for self-play.** The loop's game
  records carry the rejections (`pg_positions.py:180-197, 194-196`
  applies them to the replayed core), so the column could train now.

**Why it matters:** an enemy unit hiding on a fogged castle hex next to
the keep (an ambusher, or a unit in fog at night) stays unknown to the
network after the engine has as good as revealed it to a human. The
network keeps planning moves through that hex.

**Confidence:** verified by reading. CLAUDE.md's statement is false for
every parity network; this is a "known statement is wrong" finding.

**How a run would confirm it:** R5 (how often it happens in the loop's
games).

### F5. Nothing the network is given relates the fight columns' six directions to hex tokens, or a unit token to "its" hex token, except learned absolute position embeddings.

**What a player has:** the attack window opens on a specific enemy, so
the player sees which numbers belong to which enemy hex.

**What the network gets (verified by reading):**
- **Hex tokens** are `pos_x_embed[x] + pos_y_embed[y] + terrain +
  modifiers + dynamic flags` (`encoder.py:1233-1238`). They carry no
  occupant.
- **Unit tokens** are `type + side + pos_x_embed[x] + pos_y_embed[y] +
  row` (`encoder.py:1240-1246`).
- **The trunk is plain `nn.TransformerEncoderLayer`** (`model.py:151-164`;
  the packed re-implementation is `packed_trunk.py:561-590`). It has no
  relative-position bias and no adjacency mask, and attention is full
  over each sample's segment (`hip_kernels.py:134-189`).
- **The target head** scores (actor, hex) by q(actor)·k(hex)/sqrt(d)
  (`model.py:1178-1180`), and an attack's target is the enemy's HEX token.
- **The fight columns are laid out by direction** in the map's neighbour
  order (`core_parity.rs:295-316`, `m.nbrs[h*6+k]`). On an enemy's row
  they are zero (`encoder.py:356-358`).

**What the network must therefore learn (inferred):**
1. **"The unit on this hex":** an attention match on equal position
   embeddings. One head can do it: the score's x·x′ and y·y′ bilinear
   terms can peak at equality.
2. **"The hex in direction d of me":** a shift (x+1, y+δ). Under
   Wesnoth's offset columns, δ depends on the parity of x.
   - With additive embeddings, one attention score is
     F(x, x′) + G(x, y′) + H(y, x′) + K(y, y′).
   - That can express a parity-dependent neighbour preference only
     approximately. Exact selection of the six neighbours, excluding the
     centre and the two far corners of the 3×3 box, needs several
     heads, layers or the FF.
3. **Joining direction-d's numbers in the attacker's row to the hex
   token the target head scores:** a multiplicative interaction (column
   value × own position → preferred hex), possible only through the
   trunk's non-linear layers.

**Why it matters:**
- A unit on an even column with enemies to its north-east and south-east
  gets two fight lines (say 70% to hit against 40%). To prefer the
  north-east hex in the target softmax, the network must have learned
  which hex token is "north-east of me" for its column parity.
- Until it does, the new fight columns can inform whether to attack (the
  type head) but not which enemy.
- The legality mask limits the choice to at most six hexes, which helps.

**Confidence:** verified by reading for the absence; inferred for the
difficulty. The network attacked competently before version 2 (the
lineage's strength), so the position matching itself is learned. Whether
the direction-to-hex join is learned is open.

**How a run would confirm it:** R6.

---

## 2. Coverage: checked and found present and right

**Token limits**
- Position embeddings: `MAX_MAP_SIZE` = 128 rows (`encoder.py:48,
  731-732`). Positions are clamped to 0..127 in Rust: `clamp_pos`
  (`core_encode.rs:33-34, 340-341`), `encode.rs:73-74, 318-319, 337-338`
  and `core_parity.rs:466` for sightings. The pool's largest map, border
  included, is 62 columns (2p_Fantastic_World) and 48 rows
  (2p_Arcanclave_Citadel), from a scan of
  `wesnoth_src/data/multiplayer/maps/2p_*.map` and
  `add-ons/*/maps/*.map`. The border is stripped before indexing
  (`rules/wml_state.py:512-526`). No map reaches the clamp.
- Unit-type embedding: 200 rows with overflow row 199
  (`encoder.py:58, 1604-1641`). The live checkpoint has 237 names on ids
  up to 189 and none on 199. Every default-era type reachable from the
  six factions' recruits, leaders and advancements is in the vocab: the
  only missing ones were Dunefolk's, which are not in the pool. Walking
  Corpse and Soulless variations share their base type's row by design.
- Faction tables: 32 rows; 7 factions in the vocab.
- Weapon slots: 3 in the row (`PARITY_WEAPON_SLOTS`,
  `encoder.py:242`), 4 in the weapon head (`MAX_ATTACKS`), with
  `core_parity.rs:303` taking `min(WEAPON_SLOTS)`. The most attacks of
  any of the 356 unit types in `unit_stats.json` is 3, so nothing is
  truncated.
- Units, hexes, recruits and sightings have no per-sample cap anywhere:
  - padded path: `padded_streams.py:178-214`;
  - packed path: per-segment `cu_seqlens`;
  - HIP kernel: grid of ceil(max_len/96) query blocks, every key looped
    to the segment's `n` (`hip_kernels.py:97-99, 134-189, 414`);
  - priors compaction: exact capacity from the mask bits
    (`server_priors.py:140-181`, refusal at 265-269).
- A long game adds no tokens: the memory is a fixed k = 64 slots
  (`memory.py`), and the turn feature is a plain ratio.

**Padding and attention masks**
- Padded trunk: the key-padding mask covers exactly the pads of each
  stream (hex, unit, recruit, sighting, memory). The global and end_turn
  tokens are never masked. Actor gather: units, then recruits, then
  end_turn, with pads pointing at end_turn (`padded_streams.py:198-214`).
- Packed trunk (serving and pass): segments hold real tokens only
  (`model.py:744-773`; `serve_stage.py:437-451, 482-483`).
- Real tokens are not masked out of attention. Fogged hexes that are in
  the relevant set are tokens (`core_observe.rs:138-159`). Enemy units
  seen are tokens with side code 1 (`core_encode.rs:397`). Recruit rows
  are tokens.
- Head candidates:
  - actor mask = ownership × legal (`core_masks.rs:110-131`);
  - type, target and weapon masks are each a softmax masked to its own
    rows (`server_priors.py:323-335`, `pg_loss.py:106-124`);
  - weapons are masked by `n_attacks` = min(attacks, 4)
    (`core_masks.rs:78-80`).
- Every legal target has a token. `rows_from_landable` and the recruit
  row drop a hex whose token is −1 (`lib.rs:396-411`,
  `core_masks.rs:134-137`), but the relevant set includes every landable
  hex, every occupied (seen) hex, the castles, the castle network, the
  villages and the leader's hex (`core_observe.rs:138-159`). So no legal
  move, attack or recruit target is dropped under version 1's set, and
  version 2 only widens it.

**Unit and hex identity and order**
- Unit tokens are in (y, x, id) order (`core_encode.rs:315-319`). Unit
  positions come from the raw x and y (`game_core.py:935`). Hex
  positions come from `full_slots` (`game_core.py:926-930`). The actions
  index `hex_positions[h]` and `unit_positions[a]`
  (`server_priors.py:437-454`), so no ±1 enters here: all 0-indexed,
  with the border stripped.
- Recruit tokens sit at the mover's leader's hex (`core_encode.rs:410-415`)
  and follow the side's recruit-list order
  (`core_encode.rs:449-455`, `game_core.py:901, 938`).

**Unit projections (every path)**
- `unit_row_tokens` sends columns 0-108 through `unit_feat_proj` and
  109-340 through `unit_feat_proj2` (`encoder.py:738-740, 775-778,
  817-824`). Every unit and recruit token builder calls it (F3 lists
  them).
- Widths are refused on mismatch: `_check_raw` (`encoder.py:1156-1181`),
  `_stage` (`encoder.py:1433-1437`), the Rust stage's `fill_concat`
  exact byte check (`serve_stage.rs:395-427`) and `_require_widths`
  (`game_core.py:1112-1125`).
- Version 1 against version 2: the hello carries `unit_row_version`
  (`eval_inference_server.py:567`), and each worker's encoder builds raws
  at its own server's version (`elo_eval_game.py:404-412`). So the
  past-self opponent, a version 1 checkpoint, gets 109-column rows and
  its own projection. `TransformerPolicy.load_checkpoint` refuses a
  version mismatch (`tests/test_unit_rows2.py:578-581`).
- The pass encodes at the checkpoint's own version
  (`pg_game_pass.py:243-245`, `pg_positions.py:244-245`).
- The wire carries rows as float32 (`leaf_wire.py:40-43`), not fp16.

**Input scaling**
- Feature columns arrive as float32 from Rust; no normalisation layer
  sits before the projections.
- The terrain multi-hot reads bits 0-15 (`encoder.py:721-722, 833-834`),
  and the parity masks use 16 classes (`terrain.rs:303-337`).
- The faction posterior is a float32 weighted sum, never autocast
  (`encoder.py:1261-1268`).

**Heads**
- Actor, type, target (q·k/sqrt(d)) and weapon are computed on the same
  actor gather and the same hex rows in every trunk (`model.py:1155-1206`;
  `window_heads` at 872-883).
- The served law (`server_priors.py:311-393`) and the pass's law
  (`pg_loss.py:69-124`, via `own_loss.legal_log_probs`) factor alike:
  actor, then type, then target under the attack, move or union mask,
  then weapon.
- Tests that pin this:
  - `tests/test_core_masks.py`: Rust masks against the Python builder;
  - `tests/test_server_priors.py` and `test_server_priors_cuda.py`:
    compact priors against the reference enumeration;
  - `tests/test_forward_batch_padded.py`, `test_packed_trunk.py` and
    `test_packed_embed.py`: padded against packed against single-sample;
  - `tests/test_encoder_transfer.py`: the coalesced embed against
    per-field embedding;
  - `tests/test_unit_rows2.py:248` (a line exists exactly for the masked
    enemies) and `:483` (adjacency of attacks);
  - `tests/test_pg_game_pass.py`;
  - `tests/test_hip_kernels.py` (a HIP self-test against an fp32
    reference, tolerance 2e-2).

**Memory**
- Play: the initial memory is `initial_memory(k)`, and each decision
  writes once. A refused decision restores the previous state
  (`raw_player.py:215-227, 262-279`).
- The pass: the initial memory at `starts`; shadow rows keep what they
  read and write nothing (`pg_schedule.py:71-112`); k = 64 is checked
  against the records (`pg_game_pass.py:199-221`).
- The write is float32 in both (`memory.py:117-126`).
- The pass rebuilds the memory with its own forward rather than reading
  the logged one. The divergence is bounded by the replay check.

**Trunk numerics, serving against the pass**
- Serving: the fused HIP loop, unfolded biases (`model.fold_biases` is
  False by default, `model.py:180`; the server never sets it).
- The pass: `packed_trunk` under the fp16 autocast, with
  `hip_varlen_attention_trainable`, whose forward is the same kernel
  arithmetic plus the log-sum-exp (`hip_attention_grad.py:556-591`).
- Both use fp16 linears, a float32 residual and float32 layer norms,
  fp16 attention, no dropout and eval mode (`pg_pass.py:156-169`). The
  only precision difference is the embedding (F2).

---

## 3. To check by a run

R1 to R4 need the RX 6600 or a CPU job and must wait for a gap in the
live run. R1 can run on the CPU at idle priority.

- **R1. Token sizes against the new term.** On the CPU, load
  `round_0576\pi.pt` and replay about 2,000 positions of round 576's
  games (`PositionLoader`). For the unit tokens (embedding plus the
  UNIT token kind), report:
  - the rms and the distribution of |component|;
  - the rms and the max of c = `unit_feat_proj2(feats[:, 109:])`;
  - the share of components where `T.half() != (T - c).half()`;
  - the same three after layer 1's first layer norm (run the
    `packed_trunk` loop in float32 and capture `x` after `norm1`).

  If the share is above 50%, F1 is moot. If it is below 10%, the term
  reaches the matrix products mostly as rounding.
- **R2. The new term's effect on the served law.** On the same
  positions:
  - log p of the logged actions with `unit_feat_proj2` as trained, and
    with it zeroed, in float32 on the CPU (the true effect, e_true);
  - the same pair through the production fp16 path on the GPU
    (`serve_stage.serve_requests`, or `encoder.embed_staged` outside
    the autocast followed by `forward_embedded` under the fp16 autocast
    with the packed index), giving e_fp16;
  - the fp16 serving against the float32 forward with the same weights
    (the noise floor).

  Report rms e_true, rms e_fp16 and the floor. F3's blindness is
  confirmed if rms e_true is far below 0.014. F1's practical loss is
  measured by how far e_fp16 departs from e_true relative to the floor.
- **R3. fp16 subnormals on the RX 6600** (ROCm venv,
  `HIP_VISIBLE_DEVICES=1`, `AMD_SERIALIZE_KERNEL=3`):
  - `torch.tensor([1e-6, 3e-5], device="cuda").half().float()`
    against the CPU values;
  - `F.linear(X.half(), W2.half())` against `F.linear(X, W2)` with the
    round-576 W2 and real rows.

  A relative error far above 1e-3, or zeros where W2 is non-zero, means
  fp16 subnormals are flushed, so the pass's forward drops most of W2
  (F2).
- **R4. The embedding's share of the replay residual.** Run round 576's
  pass once with `embed_staged` moved out of the autocast in
  `GamePass._step`, on a scratch copy and not in the run. Compare
  `replay.half_mean_square` and `max_abs` with the logged 7.3e-6 and
  0.078.
- **R5. How often a recruit bounces.** Count the `rejections` entries in
  the loop's game records (all rounds since 545), and the decisions
  marked `refused`. Separate relocations (an extra rejection at a
  command index whose decision emitted a recruit) from refusals. This
  sizes F4.
- **R6. Whether the target head uses the fight columns' direction.** On
  float32 CPU forwards of positions where an own unit that can attack has
  at least 2 legal attack targets: swap the fight blocks (all 3 weapons ×
  11 columns) of two occupied directions in that unit's row, and measure
  the change in the target softmax between those two hexes. Control:
  swap one occupied direction with an empty one.

  At today's W2 the effect will be tiny. Scale W2 ×1000 in a scratch
  copy to test the wiring (an effect must appear), and read the trained
  network's answer later in the run.

## 4. Could not determine

- **The trained sizes of the embedding tables, and so the token
  components' typical size.** It needs torch to read the tensors (R1).
- **Whether ROCm's fp16 GEMM and cast flush subnormals on gfx1032 (R3).**
  HIP's default keeps fp16 denormals, but I did not verify it for the
  kernels torch selects.
- **The server-side memory books on a refused decision.** I did not
  trace `plan_handles`, the versions and `ServerMemory.commit/abort` in
  `tools/serve_stage.py`, which handle the client's rollback to the state
  before the decision when the server keeps the states
  (`raw_player.py:272-278` restores a handle). A mismatch would show in
  the replay check as rising d after refused decisions.
- **Whether the per-direction order of `m.nbrs` is the documented N, NE,
  SE, S, SW, NW.** That is the columns'-truth part; it matters to R6's
  interpretation only.

## 5. For other parts

- **Map:** under `relevant_set_hexes` (version 2 here), hexes outside the
  relevant set have no token, so their terrain and village state are not
  in the input (`core_encode.rs:301-305`, `core_observe.rs:138-159`).
  The player sees the whole map.
- **Map and units:** `terrain_mask_parity` ORs the movement-alias and
  defence-alias lists into one mask (`terrain.rs:316-335`). A hex token
  cannot say which alias list each class came from, and a mixed hex's
  best-of or worst-of rule is left for the network to infer from the
  unit's per-class tables.
- **Units:** Walking Corpse and Soulless variations share one type
  embedding row (vocab aliasing in the checkpoint). Their rows differ in
  movement and defence columns, which version 2 now carries.
- **Game state:** only the mover's recruit list becomes tokens
  (`core_encode.rs:449-455`). The enemy's recruit list is not tokenised;
  the faction posterior stands in for it.
- **Actions or state:** the recruit-rejected bit (F4) also belongs to the
  actions and state parts.
- **Columns' truth:** check that `m.nbrs` order matches the documented
  direction order of the fight columns (`encoder.py:341-345`).
