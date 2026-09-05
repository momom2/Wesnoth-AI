# Model cost study (plan step 1.4): scoping, 2026-09-05

Read-only analysis of the code and the recorded benchmarks. Token counts
are MEASURED on the 200 benchmark states (`configs/bench_states.json`),
reconstructed torch-free on the laptop (20 s single core; script and
per-state JSON in this directory: `relset_measure.py`, `relset_200.json`).
Forward costs are SCALED from the recorded 4090 rows
(`training/metrics/bench_pipeline/fwdbatch_bf16.json`); the proposed
`bench_model_cost.py` measures them on a box. Model: the 15M seed,
`d_model=384, layers=8, heads=12, d_ff=1536` (wesnoth_ai/model.py:262-275;
docs/archive/tier_b_runbook.md:29).

## 0. Summary

1. Hex tokens are 97% of the sequence (866 of 893 mean on the bench
   states; 1,150-1,300 tokens per production leaf). 65% of them (586 of
   866 mean; ~900 of ~1,230 at production sizes) cannot be the target of
   any legal action this decision.
2. At 1,270 tokens the linears are 64% of the FLOPs and attention 36%, so
   attention-pattern work (hex-local) caps at 1.25x in practice; only the
   token count moves the ceiling.
3. Relevant-set mode (already built, default off) gives 334 tokens mean
   on the bench states (p90 473, max 612), ~415 at production sizes: 2.7x
   to 3.1x fewer tokens, 4x fewer FLOPs, ceiling 5,000 leaves/s at the
   design doc's 70 TFLOPS, 11,900 at peak. Every legal action keeps a
   token by construction (asserted). The seed must be retrained in that
   basis; the label builder, the trainer and the encode worker lack the
   flag (~80 lines).
4. 2x2 hex pooling with a per-hex upsampling head reaches the same token
   count (~350) with the action space untouched, the whole board visible
   at half resolution and a warm-startable trunk; ~200 lines of new model
   code, unmeasured.
5. A hex-conv trunk at d=128 is 4-8 GFLOP per leaf (7-13x fewer) and the
   only design whose ceiling clears 3,000 leaves/s after the CPU floor is
   gone; ~500 lines, trained from scratch (2-3 epochs, $6-9), the largest
   imitation-CE risk (random-init precedent: 3.45 vs 3.11 after one epoch).
6. Hex-local attention through the SDPA mask degrades to a dense masked
   kernel (0 FLOPs saved, +620 MB of mask per batch). FlexAttention in
   torch 2.5.1 (block size 128, prototype, compile-only) saves 12-23% of
   FLOPs at r=8..3. Not a route to the target.
7. Ranking by leaves/s gained per experiment dollar: relevant set (code
   exists, $4 to a number), 2x2 pooling ($5), hex-local ($3 for 1.2x),
   conv trunk ($10-12).
8. On today's bench path all token-cutting options land on the same 6.6
   ms CPU floor per 16-leaf batch (~2,400 leaves/s); their differences
   only appear once the design doc's options 1-3 remove that floor.
9. Imitation CE is not comparable across hex bases (the target CE is over
   all H hex logits, supervised_train.py:608-613, :667); compare the
   legality-masked CE and top-1 accuracies, then the match.
10. First experiment: two arms from the seed weights, 0.5 epoch each
    (1.26M pairs, 4.6 h, $1.5 per arm on a 4090 box): full-board control
    vs relevant-set; masked holdout CE at equal pairs; then an 800-game
    PURE match of each arm against `raw:t0` ($0.36 each). Total about $4
    and one box-day; a full-epoch version is $7.5.
11. The plan's $2 for row 1.4 covers the cost table ($0.15 for
    `bench_model_cost.py`) but not a retrain; say so before renting.
12. Kill criteria: relevant-set arm loses the 800-game match to the
    control arm by more than 2 SE, or its masked CE at equal pairs is
    worse than the control by more than 0.05 nat; then run pooling.

## 1. Token budget today

Measured on the 200 bench states (side to move at a turn boundary):

| stream | mean | p50 | p90 | max | source |
|---|---|---|---|---|---|
| hex | 866 | 810 | 1,050 | 2,162 | full board (encoder.py:1041-1044) |
| unit (fog-visible) | 17.7 | 15.5 | 33 | 51 | visibility.py:470-479 |
| recruit | 7.3 | 7 | 8 | 8 | own faction list (encoder.py:1140) |
| global + end_turn | 2 | | | | |
| total | 893 | 832 | 1,083 | 2,200 | |

The manifest's `n_units` (21.5 mean) counts every unit; the encoder emits
the visible ones (17.7). Production leaves are longer, 1,150-1,300
tokens (docs/box_specs.md:261): big maps produce more decisions per game,
so they weight the leaf stream (docs/gpu_forward_design_20260904.md
section 10).

Where the hexes go, same 200 states, computed with the primitives the
legality mask uses (`unit_reach(...).landable`, tools/pathfind_sim.py:360-484;
`units_visible_to`; `leader_castle_network`):

| hex subset | mean hexes | fraction of board | p90 | max |
|---|---|---|---|---|
| own-unit single-turn reach (landable), all own units with moves | 264 | 0.326 | 0.521 | 0.704 |
| strict: reach + attackable enemy hexes (6.7) + castle network + own positions | 280 | 0.345 | 0.533 | 0.746 |
| relevant set (T2-B definition, section 2.1) | 307 | 0.375 | 0.572 | 0.751 |
| within 3 hexes of any visible unit | 282 | 0.339 | 0.537 | 0.630 |
| within 5 hexes | 436 | 0.524 | 0.774 | 0.851 |
| within 8 hexes | 603 | 0.721 | 0.963 | 0.994 |

So 586 of 866 hex tokens (65%) are hexes no legal action can target this
decision. On the 22 states with at least 1,050 hexes the relevant
fraction is 0.264, i.e. about 900 of the ~1,230 hex tokens of a
production leaf. The T2-A figure (0.30 mean over 1,840 mid-turn
decisions, docs/archive/autonomous_run.md:2217-2226) is lower than the
boundary figure because a boundary state has every unit at full moves;
search leaves are mid-turn, so production sits between 0.26 and 0.375.
Per own unit the landable set is 75 hexes mean.

FLOPs per leaf (design doc section 0.3 model: 28.3 MFLOP per token in
the linears, 8 x 4 L^2 d in attention): at L = 1,270, linears 35.9 GFLOP
(64%), attention 19.8 GFLOP (36%); at L = 893, 25.3 + 9.8. A perfect
attention kernel that computed nothing would leave 64% of the work, so
the ceiling from attention sparsity alone is 1.56x; the token count is
the lever.

## 2. Relevant-set mode (`relevant_set_hexes`)

### 2.1 What it keeps

`relevant_hex_positions` (wesnoth_ai/visibility.py:497-548), the union of:
(a) every own unit's single-turn landable reach (units with
`current_moves > 0`, not petrified; `unit_reach(u, state, ctx).landable`,
:538-546); (b) the hexes of all fog-visible units, own, enemy and scenery
(:529); (c) the leader's castle network including fog castle hexes,
plus the leader hex (:530-535); (d) every village hex; (e) every castle
or keep hex (:521-527). Order: `relevant_hexes_in_slot_order` (:550-561)
FILTERS `hexes_in_slot_order` (row-major (y, x), :491-494), so slot
indices stay a deterministic function of the state. `encode_raw` selects
it with `relevant_set=True` (encoder.py:1036-1038) and marks the result
`hex_subset=True` (:1178; EncodedState :284-289, RawEncoded :381-386).

### 2.2 Token count

Measured on the bench states: relevant-set tokens 334 mean, p50 328,
p90 473, max 612 (hex 307 + unit 17.7 + recruit 7.3 + 2). Ratio to the
full encoding: 2.67x on the mean, 3.1x on the 22 large-map states
(391 tokens against ~1,270). Length-sorted batches of 16 pad 1.06.

Two caveats for production: (i) inside one search request H is no
longer constant (the root has all units fresh; deep leaves have fewer
units with moves), so intra-request padding rises above today's 1.003
(design doc section 7.1) and the packed varlen trunk (design doc option
1) becomes the layout for it; (ii) the relevant set grows with the army
(reach is 85% of it), so a stronger policy fielding more units pays more.

### 2.3 What it costs the policy and the pipeline

Action space: nothing, by construction. Move targets are the landable
hexes (action_sampler.py:1572-1586), attack targets index the ENEMY's hex
(:1600-1616, `attack_row[_j]` on the enemy slot), recruit targets the
castle network (`_recruit_hex_mask`, :1724-1766); (a), (b), (c) contain
them, and a miss is an assertion in subset mode (:1580-1583, :1761-1763)
rather than a silent shrink. T2-A measured 0 violations in 1,840
decisions (autonomous_run.md:2221-2224).

Observation: 62% of the board's terrain is absent from the sequence on
average (up to 84% on the largest maps): the ground between the armies
beyond one move, chokepoints and rivers two turns away, the enemy's
approach lanes. Enemy units stay visible (unit tokens carry absolute
positions, encoder.py:436-437 position embeddings), villages and castles
stay visible everywhere. What the trunk can no longer compute is the
enemy's reach over terrain it does not see. Whether that matters is the
experiment's question; there is no prior measurement of it.

Pipeline: the Rust batch enumeration is bypassed for subset streams
(action_sampler.py:1227-1231: "None = Python path (no wheel /
relevant-set)"), so the legality masks fall back from 0.76 ms to the 2.2
ms Python path (docs/box_specs.md:213); `relevant_hex_positions` computes
every own unit's reach and `_build_legality_masks` computes it again
(share the ReachContext: ~40 lines); the static hex arrays are rebuilt
per encode (`_build_static_hex_arrays`, encoder.py:1038) instead of the
per-map cache (:1041-1044). Together about +2 ms of actor Python per leaf
until fixed, against an actor budget of ~5 ms today (box_specs.md:234-236).

### 2.4 The seed in this mode without retraining

Mechanically it runs: `encode_from_raw` honours `raw.hex_subset`
(encoder.py:768), the trunk and heads are shape-agnostic, the sampler
resolves targets through `pos_to_hex`. Numerically it is a distribution
shift on a trunk trained with ~870 hex tokens. The only measurement of
a same-weights switch is on the 5M net: warm-start value MAE 0.217, then
0.351 through the fixed instrument (docs/archive/backlog_20260904.md:
696-706, :820-826), i.e. "not a warm start"; no policy CE and no match
were ever run. A retrain in the new basis, from the seed weights, is
required before any number means anything.

### 2.5 "Equal imitation cross-entropy on the holdout": what it takes

Code, about 80 lines:
- `tools/supervised_train.py`: a `--relevant-set-hexes` flag into
  `GameStateEncoder(d_model=..., relevant_set_hexes=...)` (:1202) and
  into the checkpoint's `arch_record` (:1209), which `peek_checkpoint_arch`
  already reads (tools/eval_sim.py:253-288), so eval and self-play build
  the right encoder.
- `tools/replay_dataset.py`: the label builder `_action_indices`
  (:2624) resolves `target_idx` through `hexes_in_slot_order` (:2642-2646);
  under the flag it must use `relevant_hexes_in_slot_order`, and the flag
  has to reach it from `iter_replay_pairs` (call sites :2789, :2806) and
  from the worker (tools/encode_worker.py:106-110 calls `encode_raw`
  without `relevant_set`).
- `_evaluate` (supervised_train.py:897-1040): add a legality-masked
  target CE. The current target CE is a softmax over all H hex logits
  (:608-613, :667); under the subset the denominator shrinks by 2.7x, so
  equal raw CE would favour the subset arm by construction. The masked CE
  (softmax restricted to the mask-legal hexes, which is the distribution
  that plays) is the same quantity in both bases; report it with the
  per-head top-1 accuracies the function already computes.

Wall time and dollars, from the seed's own imitation run
(training/metrics/imitation_15m/imit_tierb_eval.jsonl, rows 1 and 47:
50,048 pairs at 22:59:21, 2,352,256 pairs at 07:24:24 next day): 2.30M
pairs in 8.42 h on a 4090 box, 76 pairs/s at batch 64 (step 782 at
50,048 pairs). One epoch of the 2.515M winner-side pairs
(docs/archive/claude_status_history.md:248-252) is 9.2 h, about $3.0 at
$0.33/h; the holdout CE plateaued at 3.10 from 1.2M pairs on (row 24:
3.1008; row 47: 3.1022). That run was CPU-bound on per-pair encoding,
not on the forward, so the subset arm is not cheaper per pair (its
encode is slower, section 2.3). Per arm: 0.5 epoch (1.26M pairs, 4.6 h,
$1.5) to 1 epoch ($3.0).

## 3. Hex-local attention

### 3.1 What model.py would need

The trunk is `nn.TransformerEncoder` (model.py:262-275) called with only
`src_key_padding_mask` (:591). A hex window needs a per-sample [L, L]
mask; the layer accepts `src_mask` as [L, L] or [B*heads, L, L], merges
it with the padding mask into a float bias in
`multi_head_attention_forward`, and SDPA then picks the cutlass
memory-efficient kernel with a dense additive bias (design doc section
5.1 items 2-4 and section 11: flash rejects any mask). Every (query, key)
pair is still computed, plus one bias read per pair, and the merged mask
is 16 x 12 x 1,270^2 bf16 = 620 MB per batch, re-materialised per layer
(`_canonical_mask`). So windowed attention through the SDPA mask saves 0
FLOPs and costs memory: a dead end.

The only block-skipping path in torch 2.5.1 is
`torch.nn.attention.flex_attention`, pinned from the v2.5.1 source:
`create_block_mask(mask_mod, B, H, Q_LEN, KV_LEN, device="cuda",
BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE, _compile=False)` with
`_DEFAULT_SPARSE_BLOCK_SIZE = 128`; `flex_attention(query, key, value,
score_mod=None, block_mask=None, scale=None, enable_gqa=False,
return_lse=False, kernel_options=None)`; the docstring calls it a
prototype feature and it raises unless dynamo is supported, i.e. it is
built to run under `torch.compile` (the uncompiled call is the slow
reference path). Requirements: the hand-written layer loop over each
layer's own parameters that the packed trunk needs anyway (design doc
section 5.2, ~40 lines), a `mask_mod` closing over per-sample hex (x, y)
device tensors (tokens sorted row-major, so the window is a band of
(2r+1) x cols tokens plus the unit/recruit/global/end_turn tokens, which
attend and are attended everywhere), one block mask per batch (cache by
map signature; H is fixed per map), and the compile hazards of design
doc section 3 (dynamic L, recompiles). About 250 lines including the
parity test. Triton compiles natively for sm_89, so the missing sm_89
SASS in the box wheel (box_specs.md:154-156) does not apply here.

### 3.2 FLOP reduction at L = 1,270 (cols = 38, the production map width)

| radius | hex window | exact-sparse pairs | GFLOP/leaf | 128-block band pairs | GFLOP/leaf |
|---|---|---|---|---|---|
| dense | all | 1.00 | 55.8 | 1.00 | 55.8 |
| 3 | 37 hexes | 0.076 | 37.5 (-33%) | 0.36 | 43.0 (-23%) |
| 5 | 91 | 0.119 | 38.3 (-31%) | 0.48 | 45.4 (-19%) |
| 8 | 217 | 0.218 | 40.3 (-28%) | 0.66 | 49.0 (-12%) |

Pairs include ~30 global tokens both ways. The block-band column is what
FlexAttention at block 128 actually computes on a row-major sequence
(a band of ±r map rows, ~(2r+1) x 38 tokens, plus one block of slack);
a block size of 64 tightens it by ~10 points. The flex kernel at
head_dim 32 is not faster per FLOP than the cutlass kernel, so the wall
gain is at most the FLOP gain: 1.1-1.25x.

### 3.3 Strength risk

Eight layers of radius-r hops give the hex stream an effective radius of
8r (24 hexes at r = 3, the width of most maps), and the unit, recruit and
global tokens attend to everything, so reinforcements and far villages
remain reachable through the unit tokens and through hops; the risk is
low at r >= 5, unmeasured at r = 3. The function changes, so the seed
needs a fine-tune (0.5 epoch, $1.5) before any number. Verdict: cheap,
small, and not a route to 3,000 leaves/s.

## 4. Convolutional trunk on the hex lattice

### 4.1 Design

Input: the map as a dense tensor [B, C, rows, cols] built from the same
per-hex embedding the encoder already produces (terrain, modifiers,
dynamic flags, positions; `_embed_streams`, encoder.py:843-849), with
visible units scattered onto their hex as added channels (the same
`unit_type_embed`, `side_embed`, `unit_feat_proj` sum, :850-859). The
board is the full border-stripped rectangle (replay_dataset.py:331-347
strips the 1-hex border; `n_hexes = (rows-2) x (cols-2)` holds for all 19
bench maps from the `.map` files), so the row-major flatten of the map
interior IS `hexes_in_slot_order` (visibility.py:491-494): the target
head's hex axis, the sampler and `server_priors` are untouched.

Geometry: pointy-top odd-q offset (tools/abilities.py:33-59): even
columns take neighbours (x, y-1), (x+1, y-1), (x+1, y), (x, y+1),
(x-1, y), (x-1, y-1); odd columns shift the diagonals down by one. A
7-tap hex conv is a 3x3 conv with two parity-dependent tap masks (two
weight sets, selected by a column-parity mask), or a standard 3x3 on
axial coordinates with two taps zeroed.

Trunk: N blocks of hex conv at width d, ConvNeXt-style (7-tap depthwise
+ 1x1 d -> 4d -> d) or ResNet-style (two dense 7-tap convs), with a
KataGo-style global-pooling bias every third block so the receptive
field (N hexes for N blocks) is not the limit on 23-51-wide maps. Units:
gather each visible unit's trunk feature at its hex, add the unit
embedding, run a small transformer over units + recruit phantoms +
global + end_turn (30-75 tokens, 4 layers at d = 256), then one
cross-attention from unit tokens to the hex map for long-range reads.
Heads: actor, type, weapon from the unit tokens as today
(model.py:596-598); target = q(unit token) . k(1x1 conv of the map,
flattened) as today (:600-604, the same [B, A, H] bmm); value from the
pooled map concatenated with the global token (:607-613). Batching: pad
maps to 3-4 rectangle buckets (max 48 x 51) with a validity plane; fixed
shapes per bucket make CUDA graphs natural (design doc section 4 costs
15-25% padding on the transformer; here a bucket pads the interior
rectangle only).

### 4.2 FLOPs per leaf (H = 1,230 hexes, production; heads and unit transformer +0.4 GFLOP)

| trunk | d = 128 | d = 256 |
|---|---|---|
| ConvNeXt-style, 12 blocks (16 d^2 per hex per block + depthwise) | 4.3 | 15.9 |
| ResNet-style, 10 blocks x two 3x3 convs (36 d^2 per hex per block) | 7.7 | 30.0 |

At d = 128 that is 7-13x fewer FLOPs than the transformer's 55.8; at
d = 256 ConvNeXt-style is 3.5x fewer. The GEMMs are narrow (N = 128) so
expect 30-50 TFLOPS on the 4090, and the block glue (LayerNorm, GELU,
depthwise) is memory-bound, ~0.05 GB per block per 16-leaf batch: 1.5-3
ms of GPU per batch at d = 128, under the CPU launch floor until compiled
or graphed.

### 4.3 Imitation-CE risk and cost

No trunk weights transfer from the seed (embeddings, unit projections
and heads do). The precedent for a fresh 15M trunk on this corpus is the
imitation A/B (claude_status_history.md:256-258): random init reached
holdout CE 3.449 after one epoch against 3.107 warm-started; a conv
trunk must close that 0.34-nat gap and the hex-conv prior may or may not
be worth it here (KataGo's is for Go; this policy reads unit stats,
gold, recruit lists and fog, which the unit transformer carries).
Budget 2-3 epochs ($6-9) before a fair comparison. Size: ~500 lines
(hex conv module, map rasteriser in the encoder, trunk, unit
transformer, head adapters, checkpoint arch flags, bucket padding) plus
tests. This is the largest option and the only one whose ceiling clears
3,000 leaves/s with margin once the CPU floor is gone; it is a
phase-2-scale investment, not a phase-1 row.

## 5. Half-resolution hex tokens (2x2 pooling) with an upsampling target head

### 5.1 Design

Map stream at half resolution: one token per 2x2 block of the odd-q
rectangle (x in {2i, 2i+1}, y in {2j, 2j+1}: a compact 4-hex
parallelogram), `W_pool` over the concatenated four hex embeddings
(missing hexes zero) plus a block position embedding; unit, recruit,
global and end_turn tokens unchanged at full resolution. Trunk: the
same 8 x 384 transformer, weights transferable from the seed (initialise
`W_pool` as the mean of the four so the trunk first sees "average hex"
tokens). Target head: per-hex key `k_h = W_k MLP([x_block(h) ; e_h ;
sub-position(h)])` with `e_h` the pre-trunk hex embedding the encoder
already computes; the bmm [B, A, H] and everything downstream unchanged.
The action space, slot order, label builder, Rust enumeration and
`server_priors` are untouched; H per map stays fixed, so intra-request
padding stays at 1.003. A 7-hex "flower" pooling (H/7 tokens) is the
geometrically cleaner variant at ~50 more lines of lattice indexing.

Head complexity: inside one block the head must separate four hexes with
the static hex embedding (terrain, village, castle: fine) and the block
context; adjacency to a specific enemy is only known at block level. Fix
at the encoder: six neighbour-occupancy bits per hex as dynamic flags
(`hex_dynamic_flags`, encoder.py:1178 path), ~20 lines, computed from
the same visible-unit list.

### 5.2 FLOPs per leaf

| | tokens | linears + attention | head MLP | total GFLOP |
|---|---|---|---|---|
| 2x2, production (H 1,230 -> 308 + 40) | 348 | 11.3 | 1.1 | 12.4 |
| 2x2, bench mean (866 -> 217 + 27) | 244 | 7.6 | 0.75 | 8.4 |
| 7-hex, production (176 + 40) | 216 | 6.7 | 1.1 | 7.8 |

Size: ~200 lines (pooling in `encode_from_raw`/`_embed_streams`, block
index per hex, the upsampling head, checkpoint flag) + ~20 for the
adjacency bits. Training: retrain from the seed weights, same cost as
the relevant-set arm ($1.5-3). Risk: medium-low; the whole board is
visible; local precision rests on the head. Unmeasured.

## 6. Cost table and ranking

Scaling rule, fitted to the four recorded batch-16 rows (bf16, eager
`forward_batch`, effective padded lengths 714 / 815 / 1,016 / 1,984
tokens at 10.1 / 11.2 / 13.7 / 33.4 ms): wall ms per 16-leaf batch =
max(6.6, 2.3 + 16 x GFLOP_per_leaf / 54 TFLOPS), within 7% on every row;
6.6 ms is the CPU launch floor (the batch-4 rows are flat at 6.7 ms at
every length, fwdbatch_bf16.md), 54 TFLOPS is 33% of peak. Ceilings
follow the design doc's method: 16 / GPU-ms per batch, at 70 TFLOPS
(its combined options 1-4, 42% of peak, section 8) and at the 165 TFLOPS
peak. The server path adds ~5 ms of priors work per batch on top
(design doc section 1.3).

| option | tokens/leaf (production) | GFLOP/leaf | ms per 16 (bench path today) | leaves/s (bench path) | ceiling @70T | ceiling @165T | new code | training | strength risk |
|---|---|---|---|---|---|---|---|---|---|
| full board (today) | 1,270 | 55.8 | 18.8 | 850 | 1,255 | 2,960 | 0 | none | reference |
| hex-local, flex r=5 (r=3 / r=8) | 1,270 | 45.4 (43.0 / 49.0) | 15.8 | 1,020 (1,060 / 950) | 1,540 (1,630 / 1,430) | 3,630 | ~250 lines, compile | fine-tune 0.5 epoch, $1.5 | low |
| relevant set | ~415 (334 bench mean) | 13.9 | 6.6 (floor) | 2,420 | 5,050 | 11,900 | ~80 lines (retrain plumbing) + ~100 (reach sharing, Rust subset path) | retrain 0.5-1 epoch, $1.5-3 | medium (62% of terrain unseen) |
| 2x2 pooling + upsampling head | ~350 | 12.4 | 6.6 (floor) | 2,420 | 5,630 | 13,300 | ~220 lines | retrain 0.5-1 epoch, $1.5-3 | medium-low |
| hex conv d=128 (d=256) | 1,230 map cells + ~40 | 4.3 (15.9) | 6.6 (floor) | 2,420 | 16,300 (4,400) | 38,000 | ~500 lines | from scratch 2-3 epochs, $6-9 | high on CE, medium on play |

Synthetic truncation rows (what `bench_model_cost.py --hex-tokens 300
600 900` should show, from the same rule): 340 tokens 6.6 ms; 640
tokens 9.2 ms, 1,750 leaves/s; 940 tokens 13.4 ms, 1,190 leaves/s.

Every token-cutting option is CPU-floor-bound on today's path at ~2,400
leaves/s; the 3,000 target is reached only together with the design
doc's floor removal (packed trunk, compiled tensor-only forward, one
staging buffer), after which the relevant set and pooling sit at
~5,000 and the conv trunk higher.

Ranking by leaves/s gained per dollar of experiment (experiment =
retrain arms + two 800-game matches at $0.36 each, box time only):
1. Relevant set: 2.8x on the bench path, 4x at the GPU ceiling, for $4
   (the encoder, the config gate and the superset assertions exist; the
   cost is the retrain).
2. 2x2 pooling: the same gain for $5 (model code first), with the action
   space untouched and no per-leaf H variance; the better long-term
   design if the relevant-set arm shows a strength cost.
3. Hex-local attention: 1.2x for $3; only worth doing as a by-product of
   the packed layer loop.
4. Conv trunk: the highest ceiling for $10-12 and the most code; a
   phase-2 decision, revisited if pooling and the relevant set both lose
   strength.

rejected: SDPA windowed masks, because the masked kernel is dense
(design doc section 5.1) and the mask is 620 MB per batch.
rejected: judging the relevant set with the same weights switched
(T2-C's 0.217 MAE), because that measures warm-start damage, not the
encoding (backlog_20260904.md:709-711).
rejected: unmasked imitation CE as the comparison across hex bases,
because the target softmax's support shrinks with the basis.

## 7. First experiment (two arms), pre-registration draft

What to train: from `seed_imit_tierb_start.pt` (step 2,809,659), two
arms on the same 1.26M-pair stream (half an epoch, same file order and
seed, `--reinit-value-head` off):
- control: full-board basis, unchanged code;
- relevant-set: `--relevant-set-hexes` with the section 2.5 plumbing.

What to compare, in order:
1. Masked holdout CE and per-head top-1 at equal pairs (every 50k pairs,
   `--eval-every`), both arms; the control arm's curve also tells whether
   half an epoch from the seed moves anything at all (the seed already
   sits at 3.10 raw CE).
2. An 800-game PURE match of each arm against `raw:t0` (sides
   alternated, ladder maps, `--raw-temperature-a/-b 0`,
   `--persistent-workers`, plan section 3 rule 1), SE stated; the
   relevant-set arm also plays the control arm 400 games.
3. Forward cost of both arms through `bench_model_cost.py` and
   `bench_pipeline.seam_costs` on the same box (the number row 1.4 owes).

Predictions to pre-register (mine): masked CE within 0.03 nat of the
control at equal pairs; the control arm within ±30 Elo of `raw:t0`
(half an epoch more imitation does not move it); the relevant-set arm
between -60 and +20 Elo against `raw:t0`. Kill: the relevant-set arm
below the control arm by more than 2 SE at 800 games, or masked CE
worse by more than 0.05 nat; then the pooling arm replaces it.

Cost: two arms x 4.6 h at $0.33/h = $3.0 (one box, sequential; the
imitation run is CPU-bound so two arms in parallel on a 24-core box
would need a throughput check first), two 800-game matches $0.72, one
400-game match $0.18, `bench_model_cost.py` $0.05, bring-up and pulls
~$0.3: about $4.3 and one box-day. Full-epoch arms: $7.5. Row 1.4's $2
covers the cost table, not the retrain; the retrain is the first item
that needs a rental proposal with this cost attached.

### 7b. Learning-rate control (pre-registered 2026-09-05 night, before the run)

Interim reading of the control arm's match (box 49875606, 340 of 800
results in): the seed leads 111-51 in the first 162 decisive games,
about -135 Elo for the arm, with games ending by leader death at
turns 7-13 (the normal argmax regime: the seed's decisive games
against its sampling self ran 17 turns median, 5 minimum). On the
same 1,200 holdout pairs (`--eval-only`, sample seed 0) the seed reads
CE 2.838 +- 0.079 and masked target CE 1.341 +- 0.051; the control arm
at the end of its half epoch 2.786 and 1.264. The pre-registered
prediction (control within +-30 Elo of the seed) is refuted: half an
epoch of the imitation recipe from the seed costs more than 100 Elo
while the holdout CE does not move against it.

Final number (2026-09-05, box 49875606): 276-524 over 800 decisive
games, 271 stalled games excluded, the control arm at -111 +- 13 Elo
against the seed, losing on both sides (109-291 as side 1, 167-233 as
side 2). Record: `training/metrics/elo/relset_arms/control_vs_seed.fit.json`.

What the arm is: the seed (`imit_tierb_rescued_2368k`, one epoch of
this same recipe at lr 1e-4 flat, docs/archive/claude_status_history.md
2026-08-10) continued for half an epoch over the same games at the
same rate. Not a recipe change and not a re-heating. Mechanisms left:
(a) under lr 1e-4 the policy wanders among minima of equal holdout CE
whose argmax strength differs by ~100 Elo, and the reference is a
selected good draw (it was the best of several checkpoints measured);
(b) a second pass over the same 17k games overfits them in a way the
human-state holdout does not show but self-play states do.

Arm: the control recipe unchanged except lr 1e-5, same stream, file
order and seed (`chain33` on the box, after the queue), then 800 games
against the seed at argmax (seed base 30000, sides alternated).
Cost: 3.9 h + 1.2 h at $0.33/h, about $1.7.

Prediction (operator): the lr 1e-5 arm lands within +-40 Elo of the
seed (small steps stay near the seed's minimum under either
mechanism). Readings: within +-40 Elo (2 SE at 800 decisive is about
+-26): continuation at 1e-5 keeps the seed's strength, and the
relevant-set retrain is rerun at 1e-5 with an argmax match as its
acceptance; a loss beyond -60 Elo: continued imitation itself costs
strength whatever the rate, and the basis change is done by
distillation from the seed's own priors on human states (the new
basis learns the seed's policy, not the human labels) instead of by
imitation; in between: both arms are run. Whatever the reading, a
holdout CE within noise of the seed's says nothing about strength,
and no retrained checkpoint replaces the seed without its own
800-game match.

## 8. Method notes

- Token counts: `relset_measure.py` reconstructs the 200 states with
  `tools.bench_pipeline.load_states` (torch never imported; asserted),
  then computes reach with `ReachContext.for_side` + `unit_reach`,
  attackable enemies with `tools.abilities.hex_neighbors`, the relevant
  set with `relevant_hex_positions`, and hex-distance discs in cube
  coordinates converted from the odd-q offset. Rows are in manifest
  order in `relset_200.json`.
- Effective padded lengths of the recorded bucket rows were reproduced
  from the manifest with `forward_costs`'s batching (bench_pipeline.py:
  282-291): the 2,200-token bucket's batches pad to 1,984 tokens, not
  their 1,245 mean, which is why that row costs 2.4x the 1,018 row.
- FLOP model: `gflop_per_leaf` in `bench_model_cost.py`, the same
  expression as design doc section 0.3.
- The 76 pairs/s imitation rate is from timestamps in
  `imit_tierb_eval.jsonl`; the batch size (64) is inferred from
  step 782 at 50,048 pairs.

## 9. Run plan (2026-09-05)

`scripts/relset_arms_box.sh` runs section 7 on a 4090 box after
`scripts/eval_box_setup.sh tier-b/a3/seed_imit_tierb_start.pt=seed.pt`,
writing under `/workspace/relset/`. Both arms:
`tools/supervised_train.py replays_dataset_imitation --init-from seed.pt
--imitation-config configs/imitation.json --epochs 1 --max-pairs 1260000
--seed 20260905 --bs 64 --lr 1e-4 --workers 20 --eval-every 50000
--eval-pairs 1200 --eval-pairs-per-game 8 --eval-sample-seed 0` at the
15M arch; the relevant-set arm adds `--relevant-set-hexes`. `--init-from`
loads weights and vocab only (fresh optimizer and counters, so a
completed-epoch seed does not skip the epoch); `--seed` fixes the file
order and the value-state draws, so the two arms train on the same
pair stream and stop at the same pair count. Labels are built in the
encoder's basis (`replay_dataset._action_indices(relevant_set=True)`,
also in the encode workers); a target with no subset slot keeps the
pair with the target head silent and is counted
(`target_off_subset` in the log and in `arm_eval.jsonl`; expected 0).
Each eval row carries `target_masked_ce` (the softmax over the
mask-legal hexes of the labelled actor and action type, the quantity
compared across bases), `target_masked_top1`, `target_masked_n`,
`target_off_mask` (holdout targets the legality mask does not offer;
identical for both arms) and the basis-dependent `target_ce`. The
checkpoint records `relevant_set_hexes`, which `eval_sim.peek_checkpoint_arch`
reads, so the matches (`run_elo_batch --mcts-sims 0 --raw-temperature-a 0
--raw-temperature-b 0 --persistent-workers --device cuda --jobs 10`, 800
games per arm against the seed, 400 arm against arm) build the eval
encoders in the trained basis without extra flags. Section 7 item 3
(`bench_model_cost.py`) is not in the script. Cost as in section 7,
about $4.3; the relevant-set arm's evals replay 150 holdout games in
the subset basis each, up to ~1 h more.

## Caveats recorded before the arms' matches (2026-09-05 evening)

Two findings of the second adversarial review apply to the run in
progress. (1) `--seed` does not fix the pair sequence under
`--workers > 0`: the encode workers emit replays in completion order,
so the two arms trained on 1.26M pairs drawn from the same shuffled
file order but not on the same pair set. The comparison stays a
same-distribution comparison at equal pair count; the pre-registration's
"same pairs" is not met and the stream is being made deterministic for
future arms. (2) The box script marks a match done without checking
its game count; the operator verifies each match's result count
against the pre-registered 800 / 800 / 400 before the fit is quoted.
