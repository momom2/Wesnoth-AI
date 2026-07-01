# GPU perf patches — ready to apply on the first CUDA node

Branch `gpu-perf-fixes`. These target the CUDA-only D2H-sync stalls the CPU
laptop cannot measure (BACKLOG §2026-07-01). **Profile FIRST**
(`tools/profile_rollout.py --device cuda`), apply the ones whose stall
actually dominates, re-profile. Order below is by expected impact.

Split rationale: B3 is trivially correct (pin_memory only changes the
transfer *mechanism*, can't cause a device mismatch) so it is IMPLEMENTED on
this branch and CPU-tested. The sampler-on-CPU split and B2 move
device-resident tensors on the hot path — a wrong move point is a
**CUDA-only device-mismatch bug that CPU tests do NOT catch** (on CPU every
`.to('cpu')` is identity, so the suite stays green either way). Those are
therefore specified precisely here and must be validated ON the GPU node,
not merged blind.

---

## [IMPLEMENTED] B3 — pin host buffers for async H2D on CUDA

`encoder.py` `encode_from_raw._to_dev` and `encode_from_raw_batch._cat_to_dev`
now `.pin_memory()` before `.to(device, non_blocking=True)` when
`device.type=='cuda'`. From pageable numpy, `non_blocking=True` is silently
synchronous; pinning enables real DMA overlap. CPU path unchanged (guarded on
`device.type=='cuda'`). Full suite green on CPU.

**Validate on GPU:** confirm no regression, then check whether the H2D
overlaps compute (Nsight / `torch.cuda` timeline). `pin_memory()` is itself a
host→pinned copy — net win only if it overlaps real compute. If the encode
transfers aren't on the critical path, revert (it adds a copy for nothing).

---

## [SPEC] #1 — in-process rollout: forward on GPU, sampler on CPU (biggest win)

**Problem.** In `--actor-pool 0` (default) the inference model is on the GPU,
so `mcts._expand` / `_populate_leaf` call `enumerate_legal_actions_with_priors`
on GPU-resident `output` (+ `encoded`). That function does per-actor
`.item()`/`.tolist()` (action_sampler.py:545-547, 577, 607, 614, 625, 657,
721, 742) — dozens of serializing D2H syncs per leaf, ×n_simulations leaves
per move. The actor-pool path already avoids this: its `InferenceServer`
returns CPU outputs via `inference_seam.move_model_output`, and the actor's
`RemoteEncoder` is CPU, so enumeration is pure host work.

**Fix.** Give the in-process rollout the same forward-on-GPU / sampler-on-CPU
split. After the leaf forward, move BOTH `output` AND the device-resident
`encoded` fields to CPU **once**, then enumerate on CPU.

Precise touch points:
- `tools/mcts.py::_populate_leaf` (leaf.cliffness/value + enumerate) and
  `tools/mcts.py::_expand` (the single-leaf twin) — the two call sites that
  run `enumerate_legal_actions_with_priors` on a fresh forward's output.
- Add a helper, e.g. `_leaf_output_to_cpu(encoded, output)`:
  - `output_cpu = move_model_output(output, torch.device('cpu'))`
    (reuse `tools/inference_seam.move_model_output`).
  - Move the encoded fields that `action_sampler` reads as **tensors**:
    at minimum `encoded.unit_is_ours`, `encoded.recruit_is_ours` (used by
    `_build_legality_masks`, action_sampler.py:690-693). AUDIT the full set:
    grep `encoded\.` in `_build_legality_masks` and
    `enumerate_legal_actions_with_priors` and move every field that is a
    torch.Tensor; leave the host-side ones (`hex_positions`, `unit_positions`,
    `unit_ids`, `recruit_types`) as-is. Missing one field = CUDA device
    mismatch.
  - Return the CPU `encoded`+`output`; enumerate on those.
- Gate on `output.actor_logits.device.type != 'cpu'` so the CPU path is a
  no-op (keeps CPU behavior + tests byte-identical).
- The `.item()` on value/cliffness (mcts.py:1049, 1059) then reads CPU
  tensors (host op) — folds B2 in for free on this path.

**Why not blind-merged:** on CPU the moves are identity, so the whole suite
passes whether or not the encoded audit is complete — the device-mismatch
only manifests on CUDA. Must be run on the GPU node.

**Validate on GPU:** (1) the 1-2 iter smoke exits 0 with no device-mismatch;
(2) `profile_rollout` shows the per-leaf D2H time collapses; (3) combat parity
/ a short self-play run still produces sane actions (the enumeration output
must be identical to the pre-patch GPU path — compare a few leaves' priors).

---

## [SPEC] #2 — B2: batch the per-leaf value/cliffness read in `_run_sim_batch`

`tools/mcts.py::_run_sim_batch` runs one `forward_batch` (good) then loops
`_populate_leaf` per leaf, each doing `output.value.squeeze().item()` +
`output.cliffness.squeeze().item()` = 2 D2H syncs/leaf. Read them for the
whole batch in one transfer: after `forward_batch`, stack the per-leaf
`value`/`cliffness` and `.cpu().tolist()` once, then pass the plain floats
into `_populate_leaf` (add optional `value=`/`cliffness=` params so it skips
its own `.item()`). Values identical → CPU-testable for correctness, but the
throughput win is CUDA-only. Largely subsumed by #1 on the in-process path;
still relevant for the actor-pool serve loop. Note: the enumeration syncs (#1)
dominate — B2 alone is modest (do it with #1, not instead).

---

## [SPEC] #3 — misc micro-opts (low priority; do only if they show in the profile)

- **`model.py` `forward_batch` `actor_kind`** rebuilt per-`b` via
  `torch.tensor([...], device=device)` — precompute in `RawEncoded` at encode
  time (pure fn of U,R). NOTE it IS consumed on-device in
  `predict_priors` masking (action_sampler.py:898-901), so keep a device copy;
  don't naively move to CPU.
- **`tools/inference_seam.move_model_output`** issues ~9×B separate D2H (one
  per field per output). Coalesce: keep the batched head tensors, one `.cpu()`
  each, slice per-sample on the host. Requires deferring `forward_batch`'s
  per-sample slicing until after transfer (per-sample shapes differ) —
  nontrivial; only if the serve loop D2H shows up.
- **`trainer.py` Pass-1 value harvest** (`float(output.value.squeeze().item())`
  per transition, REINFORCE path only) — stack + single `.cpu().tolist()`.
  Off the `--mcts` campaign path; skip unless running REINFORCE.

---

## How to apply

```
git checkout gpu-perf-fixes      # B3 already here + this spec
# profile on the GPU node; implement #1 (and B2) guided by the spec;
# CPU-test (suite must stay green), then GPU-smoke + profile to confirm.
git checkout main && git merge gpu-perf-fixes   # once measured
```
