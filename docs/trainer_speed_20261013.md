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
| P5 | columnar records | not done | its gate is the preload arm's reading |

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

**P5** is not done: the plan does it only if the bench's preload arm, on the
configuration after P1-P3, runs at least 5% faster than the streaming arm.

## The benchmark

`scripts/trainer_speed_box.sh` on one 4090: the base code (a code stage of
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
