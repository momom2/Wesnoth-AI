# The sequence trainer's speed (2026-10-13)

The goal: more positions a second from the memory sequence trainer
(`tools/sequence_train.py`), the trainer a policy-gradient self-play run would
use, without changing what it trains unless a flag asks for it.

Production (RTX 4090, torch 2.5.1+cu124, 31 usable cores; round 2's records
in `training/metrics/memory_own_round2_20261011/`): 512 positions a step (32
streams of 16 decisions, 16 of the streams the player's own game-sides at 64
slots), bfloat16 autocast, every time step's forward recomputed in the
backward pass. Round 1 trained at about 217 positions a second, round 2 at
183 (11,599 s for 2 million positions).

## What binds

The trunk's GPU work: each position runs the 8-layer trunk about four
forward-equivalents (the forward, its recomputation, the backward). Every time
step pads its rows to the widest one, so pad tokens take about 45% of the
per-token work and 67% of the attention work, and the memory block pads to 64
slots in every batch because the own streams run at 64. The step time follows
the content (2.13 s a step corpus-only, 2.35-2.40 s in round 1, 2.68-2.74 s in
round 2) and the holdout probe's forward rate predicts most of it.

Second: host work while the GPU idles. Nine blocking copies or reads per time
step drained the GPU's queue inside the forward loop (the own rows' targets,
the padded trunk's mask and actor index, which the recomputation built again,
and the own loss's sum); the loader's threads hold the GIL while they unpickle
records; garbage collection pauses the process. No record measured this
share; the timing stream (P0) does now.

## The items

| item | what changed | arithmetic | test |
|---|---|---|---|
| P0 | every steps.jsonl row carries the window's timing; `tools/bench_sequence_step.py` | unchanged | `test_every_steps_row_counts_the_tokens_its_trunk_ran`, `test_the_loader_clocks_its_reads_and_the_waits_for_them`, `test_the_benchmark_times_two_runs_and_compares_them` |
| P1 | `--packed-trunk` (default off) | changed: no pad token, flash varlen attention, no dropout | `test_the_packed_trunk_trains_what_the_padded_trunk_trains_without_dropout`, `test_a_pass_resumes_on_its_own_trunk_and_a_new_pass_starts_from_either`, `test_the_gradient_check_passes_the_packed_trunk`; box: the gradient check and the holdout acceptance below |
| P2 | `--activation-budget-gb` (default 0) | unchanged | `test_kept_activations_train_what_recomputed_ones_train`, on the corpus alone and with own game-sides |
| P3 | no wait for the device in the forward loop; one read per window | unchanged | `test_the_staged_own_batch_holds_each_field_as_defined`, `test_the_staged_belief_targets_are_the_ones_the_positions_define`, `test_a_prebuilt_trunk_index_gives_the_forward_the_index_built_inside_gives`, `test_a_trunk_index_for_another_batch_is_refused`, `test_a_non_finite_carried_memory_resets_its_slot_only`; CUDA (box only): `tests/test_trainer_speed_cuda.py`, with and without own game-sides |
| P4 | telemetry row every 100,000 positions (was 25,000) | unchanged | `test_the_telemetry_rows_leave_the_weights_untouched` |
| P5 | columnar records | unchanged | done in round 2 (R2): the preload arm ran 29% faster than the streaming one |

**P0.** Each row of `<out>.steps.jsonl` gains `seconds` (the window's wall
time on the host), `tokens_real` and `tokens_padded` (the trunk's real tokens
and the padded batch's, summed over the window's time steps), `peak_gb`
(`torch.cuda.max_memory_allocated`, reset each window; GiB), `gc_seconds`
(from `gc.callbacks`), `loader_wait_seconds` (the training thread blocked on a
record still being read), `decode_seconds` (the loader threads' CPU time in
`read_record`), and `kept_steps`, `kept_tokens`, `kept_gb` (P2's kept time
steps and the memory their forwards measured). `tools/sequence_timing.py`.

**P1.** With `--packed-trunk` every forward of the trainer (the steps, the
head shapes, the telemetry) runs `forward_embedded(packed=True)`: the real
tokens only, flash varlen attention on CUDA under bfloat16 (torch 2.5.1
defines its backward, `tools/autograd/derivatives.yaml:2862-2864`), q, k and v
as views of one projection, no dropout. The probe keeps the padded eval path.
The checkpoint's training meta records the trunk; `--resume` refuses another
trunk (a checkpoint without the field reads as padded), `--init-from` takes
either. The compiled packed trunk refuses training mode. `--packed-trunk` with
`--fp32` on CUDA is refused (flash takes bfloat16).
`scripts/memory_own_trajectories_box.sh` takes `PACKED_TRUNK=1`.

**P2.** Before a window's forward loop the trainer estimates each time step's
activations from its records' token counts (`ACTIVATION_BYTES_PER_TOKEN_LAYER`,
docs/design_constants.md, or the measured `--activation-bytes-per-token`) and
runs the longest suffix of time steps that fits the budget without the
checkpoint. Non-reentrant checkpointing restores the random state its forward
drew from and keeps the graph's shape, so the numbers are the same; the test
checks it with every dropout at 0.3. The default stays 0 until the bench
measures the bytes a token keeps.

**P3.** The own rows' masks, rows and targets share one pinned buffer and one
non-blocking copy (`own_loss.stage_own_batch`); the belief target and mask are
filled with numpy into a pinned buffer (`sequence_loss.step_labels`,
`StepLabels.belief_on`); the trunk's index arrays are built once per time step
(`WesnothModel.trunk_index`) and passed to the forward and its recomputation
(`forward_embedded(index=...)`), which refuses an index built for other
per-row counts; the clip's group norms stay on the device
(`GradientGroups.clip_on_device`); after the backward pass one transfer reads
the finiteness of the loss, the norm and every carried memory, the loss, the
group norms and every logged value (`tools/sequence_window.py`). The host's
sums take the same float32 values in the same order.

Before and after on the laptop (CPU, every dropout at 0.3, three windows of
two time steps, `bench_sequence_step.py trace`): the base commit bd5f33d and
this branch print the same SHA-256 of the returned values, every gradient
before it is zeroed, the parameters after each step and the carried
memories, at budgets 0, 1e-6 and 100 GiB. On the corpus fixture of
tests/helpers/sequence_pass.py: `0e1693af...`. With own game-sides on half
the streams (tests/helpers/own_pass.py; 4 streams, 2 of them own, a pass of
24 positions), the path production runs on 16 of its 32 streams:
`be996ef5...`; the own loss enters the digest (at weight 0.5 it reads
`7a4b9134...`).

**P4** is not measured by the bench, whose arms move every telemetry row
past their windows. Round 2's records give its share: 80 rows took 168 s of
the pass's 11,599 (1.9 to 2.5 s each); at the new cadence 20 rows take about
42 s, so it saves about 1.1% of a pass.

**P5** ran its gate on the box: the preload arm, on the packed trunk with kept
activations, ran 709.9 positions a second against the streaming arm's 548.9
(1.29 on the totals; the median window 1.13, 95% interval 1.07-1.25). Round 2
does it (R2).

## The benchmark

The first round's script (`scripts/trainer_speed_box.sh` at e2d2877; the file
now runs round 2's arms, below) on one 4090: the base code (a code stage of
bd5f33d, `BASE_STAGE`) and this stage, timed with this stage's
`tools/bench_sequence_step.py run` under round 2's training arguments from
own1 (its own sequences rebuilt once and cached on HF). Each arm: 20 untimed
windows, 80 timed ones between two device synchronizations, nvidia-smi every
100 ms, 3 windows under the profiler (kernel time by class, the GPU's idle
fraction). The arms, in order: base, this, base, this (valid when the two base
runs agree within 3%), the gradient check, the activation budget
(calibrated: a short arm measures the bytes a kept token takes, the budget
fits the worst window under 22 GiB), the packed trunk (when its gradient check
passed), the packed trunk with its budget, the fastest configuration with
every record preloaded (P5's gate), and this stage's defaults again.
`bench_sequence_step.py compare` gives each change's speedup: the median of
the per-window time ratios (the schedule depends only on the seed, so window
i holds the same positions in every arm) with a bootstrap 95% interval, and
the ratio of positions a second from the synchronized totals. It refuses two
runs whose first records differ in a field other than the code and the ones
named, and two runs whose windows ran other trunk tokens when both count
them (the base does not; the trace above covers it). `speed.json` gathers
them, with `valid` (the two base arms agree within 3%) and `flags_valid`
(this_1 and a last arm of this stage's defaults, this_3, which brackets the
flag arms, agree within 3%).

The order puts the base-against-this arms first: the gradient check runs
after them, and anything but a pass, a crash or a cut included, leaves only
the packed arms out. An arm whose process fails after its total row (the
profile, which records its own error) keeps its timed windows. Each arm's
first record names the code that ran: the git blob id of every source the
process loaded from the tree's `tools` and `wesnoth_ai`, so the base arms
check offline against `git ls-tree -r bd5f33d -- tools wesnoth_ai`.

## P1's acceptance (pending, a box)

Before any packed pass: the bench's gradient check (per-group gradient
cosine at least 0.99 and the loss within 0.5% of the padded trunk's at
dropout 0, over 3 production windows from own1). Then three passes from own1
under round 2's preset, the same seed, `--pass-positions 500000`: padded,
padded again (the run-to-run noise of today's GPU training), packed; each
ending with a probe at k0 and k64. Paired over holdout game-sides, the
packed-minus-padded difference in ce_all (k0, k64) and in belief must be no
larger than the repeat-minus-padded difference, within 2 paired standard
errors. A checkpoint trained with the flag still gets its own match.

## Round 2 (branch feature/trainer-speed-2)

### What binds after round 1

The first round's box run (training/metrics/trainer_speed_20261013/, one
RTX 4090, round 2's windows): base 249.6 positions a second, the packed
trunk with kept activations 548.9, the same with every record read first
709.9. On the packed arms the GPU is busy 10-13% of a window under the
profiler (27,677 kernels a window, 0.32 s of kernel time; the window takes
0.55 s at the median with the records preloaded): the step is host-bound.
Two costs: the loader (unpickling about 1,000 positions a step on threads
that hold the GIL, then the collector walking the objects: 0.30 s of GC a
window on average in the streaming arm, 0.06 s preloaded), and the number of
Python dispatches and kernel launches a time step (about 1,700 kernels).

### The items

| item | what changed | arithmetic | test |
|---|---|---|---|
| R1 | the memory batch gathered in one `index_select` (tools/sequence_memory.py) | unchanged | `test_the_routed_memories_train_what_per_row_memories_train` |
| R2 | columnar records and `--columns` (P5) | unchanged | tests/test_sequence_columns.py |
| R3 | `FlatLayout.torch_views` one view per field; targets copied to the device once; the heads' widths read off the modules | unchanged | the trace digests below; the staging tests |
| R4 | `--window-batched` (default off) | changed: rounding | `test_the_batched_window_trains_what_the_stepped_window_trains_within_rounding`; box: the gradient check |

**R1.** Each row's memory was a slice of the step that wrote it, padded by a
copy per row (`SlotMemory.batch`); the backward pass ran a zero fill, a copy
and an add per row: on the CPU counter below, R1 removes 195 backward ops and
30 forward ops a time step at 32 streams. The router
builds the padded batch with one gather from a pool of the previous step's
output, a copy of the initial memory per starting row and the carried
memories; its backward is one scatter. The previous output joins the pool
only when a row reads it: unread, it would get a zero gradient, and AdamW
would then decay the memory write's weights in a window that trained none of
them (the trace below caught it on the corpus fixture).

**R2.** `wesnoth_ai/sequence_columns.py` stores what the step reads as
columns per game-side: every field concatenated over the positions, offsets
where its length varies, every array in the pickle's dtype, one zlib blob per
side (inflate releases the GIL). Reading a game-side is one inflate and one
`np.frombuffer` per array; a position is a slice of each; no Python object
exists per position. On 40 real games the columns are 2.7 times smaller than
the pickles and read in about a twentieth of the time (laptop). What the step
does not read is left out (the `Position` objects, unit ids, recruit names,
the label's hexes); the pickles stay the records of the probe and the other
tools. `tools/columnize_sequences.py` writes them beside the pickles of a
corpus or own directory, reads every game back and compares each position
with the pickled one byte for byte; the trainer refuses columns of other
records (the manifest's fingerprint) or missing games. `tools/sequence_inputs.py`
stages a time step from either: the columns' path concatenates each field's
slices into the pinned buffer and writes the own rows' masks with one indexed
write per field; the staged buffers are the pickles' byte for byte.

**R3.** A staged field was three dispatched views (a slice, a dtype view, a
shape view); a step stages some 60 fields. `torch_views` now makes one view
of the buffer per dtype and one `as_strided` per field (the same tensors:
storage, offset, shape and strides). `ImitationTargets.views` copies to the
device once (a recomputed forward copied again). The trainer no longer runs a
forward to read the heads' widths.

**R4.** `--window-batched` (tools/sequence_batched.py, packed trunk only)
stages and embeds the window's positions at once, runs each step's packed
trunk and memory write on its rows of that embedding
(`WesnothModel.packed_step_trunk`), and runs the heads and the losses once over
every step's trunk outputs (`window_heads`, `packed_trunk.window_head_index`).
Its rounding differs from the stepped forward's: the embedding's and the
heads' matrix products run over more rows, each embedding table's gradient is
summed over the window in one kernel, the softmaxes run over rows padded to
the window's widths, the loss is summed in another order. On the CPU every
window's loss agrees within 1e-7 relative and every gradient within 1e-4 of
its largest element; the only gradient whose cosine is far from 1 is the actor
head's bias, which is zero but for rounding (1e-9). A network it trains needs
its own acceptance, as the packed trunk's.

Dispatched ops a window on the CPU (`TorchDispatchMode` count, 32 streams of
which 16 own, 8 time steps, d_model 32, 8 layers, attention counted as one op
as on CUDA; views counted apart; the pickles' path and the columns' dispatch
the same ops):

| code | forward ops + views | backward ops + views |
|---|---|---|
| round 1 (e2d2877) | 3,033 + 4,095 | 5,768 + 3,318 |
| this branch, stepped | 2,793 + 2,284 | 4,207 + 3,129 |
| this branch, `--window-batched` | 1,060 + 897 | 2,637 + 2,200 |

### Bit-exactness

`bench_sequence_step.py trace` (CPU, every dropout at 0.3, three windows)
prints the same digest from round 1's tree (e2d2877) and this branch, with and
without `--columns`, at budgets 0, 1e-6 and 100 GiB: the corpus fixture
`0e1693af...`; with own game-sides (tests/helpers/own_pass.py)
`607c8890...`; and on 40 real games beside the own fixture's game repeated,
8 streams of which 4 own, windows of 4, the padded trunk `764dd90b...`, the
packed trunk `3daea995...`, six packed windows (own game-sides ending and
starting inside windows) `0cf2145c...`. In the suite: the router against
per-row memories over whole passes (padded and packed, corpus and own),
columns against pickles (every position, every staged byte, whole passes,
hand-built game-sides with hidden tokens, sightings, timeouts and own
biases), each test broken once by hand to see it fail.

### The benchmark

`scripts/trainer_speed_box.sh` on one 4090. Every arm runs the first round's
fastest flags (the packed trunk, `--activation-budget-gb 19.18
--activation-bytes-per-token 102139`, its packed_budget arm's), so every arm
keeps the same time steps. The arms, in order: base_1, this_1, base_2, this_2
(round 1's code against this branch's: R1 and R3; valid when the two base runs
agree within 3%), the gradient check of `--window-batched` against the
stepped forward (a crash barrier for the batched arms), columns (R2),
batched_pickles (R4 alone), batched (R2 and R4), preload (the fastest of
columns and batched with every record read first: what the columnar loader
still costs), this_3 (brackets the flag arms). Expected wall about 1 hour
($0.45 at $0.42 an hour) with the own sequences restored from the first
round's HF cache.

### Not done

- Decoding or staging in Rust: no per-position decoding remains (one inflate
  and a few `np.frombuffer` per game-side), and the staging is one numpy
  concatenation per field; what binds is PyTorch's dispatch per op, which
  Rust cannot remove from the Python graph.
- CUDA graphs: the window's backward needs every time step's activations at
  once, so each time step would need its own captured graph per shape bucket,
  each holding its own activations (16 steps times the token buckets, about a
  GiB each, on a 24 GiB card); bucketing pads the token count, which changes
  the matrix products' tiling, and the varlen kernel's grid follows max_len.
- One cast of the trunk's weights a window instead of one a step (about 70
  cast kernels a time step forward): the weights' gradients would sum in
  bfloat16 across the window's steps.
- Host-built loss masks, skipping the value head's moments in training,
  int32 offsets built on the host: under 1% of a step's dispatches each.
- Prefetching the next window's inputs on a thread: the staging is a few
  milliseconds a window once the records are columns, and Python threads
  share the GIL with the dispatching thread.

## Rejected

- Length-sorted sub-batches on the padded trunk, because P1 supersedes them,
  they change the kernels' tiling (arithmetic), and they multiply launches.
- A bit-exact rewrite of the padded MHA block, because P1's packed loop already
  has that layout; revisit only if P1 fails its acceptance.
- A dropout-0 flag on its own, because P1 has no dropout; it stands alone only
  if P1 fails.
- torch.compile, fused cast/add/LayerNorm, a bfloat16 residual stream, because
  2.5.1 needs dynamic shapes under non-reentrant checkpointing, the packed
  attention op has no backward, and the gain is unknown until P1 lands;
  re-propose if P1's profile shows elementwise and norm kernels above a
  quarter of GPU time.
- Moving the losses out of the checkpoint, because it is 1-2% of the step and
  P2 removes most recomputations.
- `use_reentrant=True` or `determinism_check='none'`, because they are under
  2% and the first's gradient accumulation order is unverified.
- Fused AdamW, because it changes rounding for under 0.1%.
- Probe speedups, because the probe is outside the step (about 3% of a pass)
  and keeping the holdout resident adds millions of GC-tracked objects.
- `gc.freeze()` at window boundaries, because it can hold cyclic garbage
  indefinitely; P0 measures GC and P5 would remove the objects.
- Keeping a record until both its sides ran, because the partner side comes
  about 28,000 game-sides later in the shuffled order.
- Caching the memory-size draw, changing checkpoint saves: 0.1-0.2%.
- More streams per step, because it changes the batch size (a recipe change).
- CUDA graphs, because shapes change every time step and static buckets bring
  the padding back.
