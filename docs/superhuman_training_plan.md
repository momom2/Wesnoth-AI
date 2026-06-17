# Plan: training & validating a superhuman Wesnoth model

**Status:** proposal / roadmap. Drafted 2026-06-17.
**Scope note:** this plan assumes Wesnoth_AI is *mutable* and calls for
incremental, scoped code changes (flagged inline as **[slight]**,
**[moderate]**, or **[new tooling]**). It deliberately avoids a full
redesign. Every prefixed change is optional in the sense that it can be
deferred; none requires rewriting the training core.

All compute prices are web-verified as of **2026-06-17** and will drift —
re-check before committing spend.

---

## 0. TL;DR

1. **Capacity first, compute second.** The current trained net is **~0.47M
   params** (`d_model=128, layers=3, heads=4, d_ff=256`; verified from
   `training/checkpoints/sim_selfplay.pt`). Reference points say that is
   *below Connect-4-grade capacity* — it is the binding constraint before
   GPU-hours are. Scale to **~3–10M** before any serious rented run, **10–30M**
   for the decisive campaign.
2. **The bottleneck is the Python rollout, not the GPU.** `sim.step` +
   encoder state-build + legal-action enumeration + MCTS bookkeeping are
   pure-Python/CPU. A bare GPU with 4–8 vCPUs **starves** and wastes money.
   The two highest-ROI engineering changes are **batched MCTS leaf eval +
   virtual loss** and a **`torch.multiprocessing` actor pool feeding one
   central batched-inference process**. Both are additive to the existing
   `--mcts-batch-size` / `--workers` seams.
3. **Buy a cheap consumer GPU on a high-vCPU host**, not a fast GPU on a
   starved one. Best picks: a CPU-filtered **Vast.ai RTX 4090** (~$0.30–0.40/hr),
   **Hyperstack A6000 spot** (~$0.40/hr, no egress), or **RunPod RTX 3090**
   ($0.46/hr, 16 vCPU). For VRAM headroom: **Azure NC24ads A100-80GB spot**
   ($0.82/hr). For sustained 24/7: a **Hetzner actor/learner split**
   (~€180–560/mo).
4. **"Superhuman" is not currently measurable** — the project only scores vs
   the built-in RCA AI and vs self-play. A **rating/validation bridge** (new
   tooling) is required to even define the target, let alone prove it.
5. **Spend in rungs, gated on measured results.** Free dev → ~$30–240
   calibration (Tier a) → ~$500–1,500 decisive campaign (Tier b) → stretch
   (Tier c). Re-estimate after each rung from *your* measured Elo-vs-compute
   curve.

---

## 1. Where the project is today

| Fact | Value | Source |
|---|---|---|
| Trained model size | **471,405 params** (`d=128/L=3/H=4/ff=256`) | `sim_selfplay.pt` `arch` + state_dict |
| Training so far | decision_step ≈ **5.68M** | checkpoint metadata |
| Method | AlphaZero-style MCTS (Gumbel root, PUCT, exact combat-outcome enumeration, tree reuse) + REINFORCE fallback; experience replay | `tools/sim_self_play.py`, `tools/mcts.py`, `trainer.py` |
| Simulator | pure-Python, bit-exact combat (731/731 strikes), ~1000× faster than real Wesnoth | `tools/wesnoth_sim.py` |
| Curriculum | ladder (21 maps) / mini (8) / drill (3) pools, mixed via `--mini-ratio`/`--drill-ratio` | `tools/scenario_pool.py` |
| Device | CPU-only locally (AMD RX 6600, DirectML dead-ended); **CUDA path wired but never exercised** | `docs/running_on_gpu.md` |
| Strength | **far from superhuman.** At iter-168, *zero* leaderkills on full ladder maps (army never threatens the enemy leader in 24 turns); decisive games only on mini/drill | `BACKLOG.md` |
| Validation | win-rate vs built-in **RCA AI** (`eval_vs_builtin.py`) + vs self-play (`eval_sim.py`); Wilson CIs. **No human-calibrated rating.** | eval tooling |

Two diagnoses from the BACKLOG matter here:
- The replay-buffer A/B showed the value head was starved of gradient steps
  (`--replay-updates` is the dominant lever: held-out value loss 4→3.28,
  8→3.09, 16→2.74). The plateau was a *training-signal* problem, not only a
  size problem — **so scaling the net without fixing the signal/throughput
  won't help.**
- Decisive games come only from mini/drill. On full maps the agent can't
  close distance — a **curriculum + turn-cap** issue, not (yet) a tactics
  issue.

---

## 2. The strategic frame

Three principles drive every choice below.

**(a) Capacity-first.** The goal is the *final capability*, not fitting the
0.47M net. That net is below the size used for even simple solved games
(AlphaZero.jl Connect-4 uses ~1.6M). Scale the net up front, then spend
compute on it.

**(b) Rollout-bound, not GPU-bound.** This is the single most important
hardware fact. Self-play throughput is gated by Python CPU work, so:
- the right rental maximizes **vCPU-per-GPU**, not GPU FLOPs;
- the right engineering keeps the GPU fed (batch leaf evals; many CPU actors);
- if you ignore this, a strong GPU sits idle and effective $/capability
  balloons 3–10×.

**(c) Measure, then scale.** No prior AlphaZero result exists for a
Wesnoth-class game (stochastic combat, ~1700-hex action space, multi-unit
turns, recruit economy). All compute estimates below are *scaled analogies*,
not measurements. The first paid run's real deliverable is **your** games/sec
and Elo-vs-compute curve — that re-prices everything downstream.

---

## 3. Required changes to Wesnoth_AI

Ordered by leverage. Nothing here is a redesign; the biggest item extends
seams that already exist.

### 3.1 Keep the GPU fed — **[moderate]** (highest ROI)
The CUDA path is untested and `--mcts-batch-size` is described in your own
BACKLOG as "wired, CPU-pessimal today." Finish that seam:
- **Batched MCTS leaf evaluation with virtual loss.** Collect N leaf states
  per search step, run **one** GPU forward, back them up; virtual loss keeps
  parallel descents from collapsing onto one path. Reported ~13× on
  accelerators. This is plain PyTorch + your existing tree code in
  `tools/mcts.py` — *not* a framework adoption.
- **Actor pool → central inference server.** `torch.multiprocessing` worker
  processes each run the pure-Python rollout (`sim.step`/encode/enumerate/
  MCTS) and ship leaf states over a `Queue`/`Pipe` to one process that owns
  the GPU and dynamically batches forwards. This is the SEED-RL / MonoBeast
  pattern; the `TorchDemon` PyPI package is almost exactly this component —
  **copy the ~100-line pattern, don't add a heavy dependency.** Extends your
  existing `--workers` path.
- **Do not** adopt Ray/RLlib, TorchRL, or Sample Factory for a single node
  (over-engineering), and **do not** consider EnvPool (needs a C++ env).

Refs: OpenSpiel AlphaZero docs; Oracle "Lessons From Alpha Zero pt5"; DeepMind
`mctx` (study its batched-tree layout; it's JAX, so don't port).

### 3.2 Model scaling — **[slight]**
Add CLI flags to set fresh-init architecture (today a no-checkpoint run uses
the `TransformerPolicy` constructor default of `512/8`, and a warm-start
*reads* arch from the checkpoint and **raises on mismatch** —
`sim_self_play.py:~1619`):
- Add `--d-model / --num-layers / --num-heads / --d-ff` to
  `tools/sim_self_play.py`, threaded into `TransformerPolicy`.
- **Scaling means fresh init.** The old 0.47M transformer weights won't
  load into a wider/deeper net (arch mismatch). That's fine — superhuman is
  the goal, not preserving 5.68M decision-steps. The curriculum + drills
  re-bootstrap a bigger net quickly. (A net2net/width-transfer warm-start is
  possible but **[moderate]** and optional.)

Param targets (transformer core; add ~0.2–0.5M for embeddings/heads):

| Tier | `d_model` | layers | heads | `d_ff` | ≈ params |
|---|---|---|---|---|---|
| current | 128 | 3 | 4 | 256 | 0.47M |
| **a** (beat RCA / strong amateur) | 256–384 | 6 | 8 | 1024–1536 | **~3–10M** |
| **b** (competitive human) | 512 | 8–12 | 8–16 | 2048 | **~10–30M** |
| **c** (superhuman, stretch) | 768 | 12 | 12 | 3072 | **~85–100M** |

### 3.3 Spatial reasoning — **[moderate]**, recommended for Tier a→b
The encoder has **no positional encoding** — only token-kind embeddings
distinguish streams (`model.py:TokenKind`). For a game decided by geography
(terrain, ZoC, chokepoints, leader distance), a position-blind transformer is
a real capability ceiling. Add **2D/relative positional encoding over the hex
grid** (axial learned embeddings keyed on hex `(x,y)`, or RoPE-2D). Localized
to `encoder.py` + `model.py`; not a redesign. Mark optional, but it is likely
necessary to fix the "never closes on the leader" failure on full maps.

Watch: attention is **O(seq_len²)** over ~1700 hex tokens. As `d_model` grows
this dominates. If forward cost rises faster than rollout, consider
local/windowed attention over hexes or hex-downsampling — **defer until
profiling shows it binds.**

### 3.4 Validation / rating bridge — **[new tooling]** (gates the whole goal)
You cannot currently measure "superhuman." Build, in rough priority:
1. **An Elo/anchor ladder.** Run round-robins among checkpoints + RCA AI +
   scripted `dummy_policy`, fit Elo (BayesElo/Whole-History-Rating). Gives a
   single internal strength axis over time. Reuse `eval_sim.py` for the
   sim-side games; add an Elo fitter. **[new]**
2. **A human anchor.** RCA AI is a weak, uncalibrated floor. Calibrate against
   *humans* by scoring the trained policy against **archived human ladder
   replays** (the download/extract tooling — `tools/download_replays.py`,
   `tools/replay_extract.py` — still exists even though the corpus was
   retired): measure top-1/top-k agreement with strong human moves, and
   "would-the-agent-have-blundered" rates. This is a teacher-forced
   evaluation, not live play, so it needs no human opponents. **[new]**
3. **(Optional, decisive) live ladder play.** The hardest, most convincing
   signal: have the agent play actual humans on the multiplayer ladder via the
   eval bridge (`wesnoth_interface.py`). Heavy, manual, and out of scope for
   the core plan — but it is the only *direct* proof. Treat as a Tier-c
   capstone, not a routine metric.

Define the capability tiers operationally (Section 6).

### 3.5 Training-efficiency tricks — **[slight]**, borrow from KataGo
KataGo reached superhuman Go at ~50× less compute than naive AlphaZero. The
portable, mostly-already-present levers:
- **Playout-cap randomization** (already in BACKLOG as 🟡 todo): most self-play
  moves get a tiny MCTS budget, a random subset gets the full budget and
  produces policy targets → 3–10× more games/GPU-hour.
- **Auxiliary prediction targets** (e.g. predict opponent reply, territory/
  gold-swing) to densify the learning signal.
- **Progressive net growth**: start Tier-a-sized, widen as the curve flattens.
- Keep tuning `--replay-updates` (the proven dominant value-head lever) and
  `--value-coef`.

### 3.6 Spot/interruptible orchestration — **[slight]**, ops glue
Your checkpoint/resume already makes the job preemption-tolerant. Add:
- **Atomic checkpointing** (temp file + `rename`) every N minutes to a mounted
  bucket; "load newest valid on boot."
- **SkyPilot** managed spot jobs (`sky jobs launch --use-spot`) for big-cloud
  auto-reprovision; **vast.ai CLI** directly for the marketplace (SkyPilot
  doesn't broker Vast).
- A **wall-clock kill + idle-GPU watchdog** that auto-destroys a hung/forgotten
  instance — the classic solo money leak.

---

## 4. Compute recommendation (verified 2026-06-17)

### 4.1 The shortlist, ranked for *this* job

| # | Option | Effective price | vCPU/GPU | Best for |
|---|---|---|---|---|
| 1 | **Vast.ai RTX 4090**, spot, *filtered* `cpu_cores_effective>=24 reliability>0.98` (or TensorDock à-la-carte 4090 + 32–64 vCPU) | **~$0.30–0.40/hr** (the $0.14 "floor" is a transient bid, not reliable) | you choose ≥24 | cheapest $/rollout-hr; main campaign |
| 2 | **Hyperstack RTX A6000 48GB**, spot | **$0.40/hr** (no egress) | up to 28 physical cores | first real run; clean (non-marketplace) campaign box; VRAM headroom |
| 3 | **Azure NC24ads A100-80GB**, spot | **$0.82/hr** (best big-cloud A100; 30s eviction) | 24 | decisive run once net >24GB; pairs with Founders-Hub credit |
| 4 | **Hetzner actor/learner split**: auction EPYC 48C/96T (~€150–200/mo) + Hostkey A5000 24GB (~€130–360/mo) or Hetzner GEX44 RTX-4000-Ada (€232/mo) | **~€180–560/mo all-in, 24/7** (unlimited free traffic) | many-core CPU box feeds GPU box | sustained multi-week 24/7 running |
| 5 | **RunPod RTX 3090**, Community | **$0.46/hr** (free egress) | 16 / 125GB RAM | low-friction Tier-a validation |
| 6 | **Lambda A100-40GB** on-demand ($1.99/hr, 30 vCPU, no egress) or **DataCrunch A100-40GB $0.72/hr / H100 spot $0.80/hr** (EU) | $0.72–1.99/hr | 30 (Lambda) | reliability-first run; no spot babysitting |
| 7 | **Free dev rung**: Lightning AI (32-core CPU Studio + GPU Studio, ~80 GPU-hr/mo, unlimited background exec) + Kaggle (4 vCPU, 2×T4, ~30 GPU-hr/wk) | $0 | low | dev/debug, batched-MCTS plumbing, Tier-a calibration |

**Avoid for this workload** (verifier-flagged starvation traps): Salad 4090
(~4 vCPU), RunPod's *cheaper* $0.34/hr 4090 (~6 vCPU — use their 3090
instead), AWS g5/g6.2xlarge & GCP a2/g2 (8–12 vCPU), all free notebook GPUs
for *campaigns* (Colab ~2 vCPU + idle-kills + bans automation; Kaggle
9h/session + 30h/wk). **OCI A10 is NOT available preemptible** (a tempting
"$1/hr 30-vCPU" listing turned out wrong — it's ~$2/hr on-demand only).

### 4.2 Free credits worth claiming
- **Microsoft for Startups Founders Hub** — **$1k email-verified, up to $5k**
  after business verification. The only big-three credit with *no* funding/VC/
  website-traction wall; the right one to apply for first. Funds Azure Spot
  (option 3) for a near-zero-cash decisive run.
- **GCP $300 / 90-day** and **Azure $200 / 30-day** new-account trials (stack-able).
- AWS Activate ($1k) needs a real company website; skip unless you incorporate.
- **TPU Research Cloud** (free TPUs) — *skip*: needs a JAX/XLA rewrite, XLA
  hates the dynamic ~1700-hex pointer-net shapes, and a TPU does nothing for a
  CPU-bound Python sim.

### 4.3 Egress / idle-storage gotchas
Vast bills disk continuously even while a spot instance is *stopped* (must
DELETE to stop charges) and egress is per-byte; RunPod *stopped* volumes cost
2×; GCP egress is the priciest. Free/no-egress: Lambda, Hyperstack, Hetzner
(unlimited), RunPod transfer. **Keep the replay buffer & checkpoints local to
the box; ship only small model+optimizer checkpoints off.**

---

## 5. Compute budget & timeline

Estimates normalized to one cheap modern GPU on a CPU-rich host ("rollout-
machine-hours"). Ranges are scaled analogies (see §2c); the intra-tier spread
is a real 5–10×.

| Tier | Capability | Net size | Rollout-machine-hrs | $ (spot) | Wall-clock |
|---|---|---|---|---|---|
| **a** | Decisively beat built-in RCA; strong-amateur play; closes on the leader on full maps | ~3–10M | ~150–600 | **~$30–240** | ~1–4 weeks, one box |
| **b** | Competitive with expert humans (microRTS-solo regime: a solo dev beat that game's best bots with ~5M params, ~70 GPU-days) | ~10–30M | ~1,500–6,000 | **~$500–1,500** | ~1–3 months one box, or weeks on a small fleet |
| **c** | Superhuman | ~30–100M | ~10,000–50,000+ | **~$3k–20k+** | multi-month; realistically a small fleet or KataGo-style long-haul |

**Anchors:** AlphaZero.jl Connect-4 strong-amateur ≈ 15–30 GPU-hr / 1.6M
params (RTX 2070); MiniZero superhuman 9×9 Go / 8×8 Othello (incl. Gumbel-AZ,
your exact family) ≈ hundreds of GPU-hr on one consumer GPU; KataGo scratch
superhuman 19×19 Go ≈ 12,000 V100-hr / 4.2M games / 20×256 net; microRTS
RAISocketAI (closest analogue) ≈ 70 GPU-days / ~5M params; AlphaStar (full
RTS, *not* a target) ≈ ~$3.2M of TPU.

**Honest read:** Tiers a and b are realistic on a hobbyist budget *if* you fix
throughput and size the net right. Tier c is a genuine stretch — historically
it needed distributed/crowd compute (KataGo public run: 87M+ games from 1,321
volunteers; Lc0: ~1 year crowd-sourced). Treat c as contingent on b being
convincingly clear of strong humans.

---

## 6. Phased execution plan

### Phase 0 — Engineering & sizing (free, ~$0)
**Goal: a pipeline that keeps a GPU busy, and a right-sized net.**
- Implement §3.1 (batched MCTS + actor pool) and §3.2 (arch flags). Develop on
  **Lightning AI free** (32-core CPU Studio mirrors the actor/learner split) +
  Kaggle bursts.
- Run the **required first CUDA smoke** from `docs/running_on_gpu.md` (exit 0,
  `nvidia-smi` busy during train_step, checkpoint round-trips).
- Add §3.4(1) Elo ladder so every later run has a strength axis.
- **Exit criteria:** on a rented GPU, sustained GPU utilization ≥ ~70% with
  your worker count; measured games/sec recorded.

### Phase 1 — Tier a calibration (~$30–240)
**Goal: strong-amateur + the real Elo-vs-compute curve.**
- Net ~3–10M (`--d-model 384 --num-layers 6 --num-heads 8 --d-ff 1536`),
  fresh init.
- Box: Hyperstack A6000 spot (option 2) or filtered Vast 4090 (option 1) or
  RunPod 3090 (option 5).
- Curriculum: start drill/mini-heavy, then raise ladder share and the
  **`--max-turns`** cap on full maps (directly targets the "never closes on
  the leader" failure). Keep `--replay-buffer --replay-updates 16
  --value-coef 1.0` (proven config) and re-tune `--replay-updates` on GPU.
- Add a per-game `scenario_id`+winner log line (BACKLOG todo) so leaderkill
  rate is attributable by map.
- **Exit criteria:** ≥ ~90% vs RCA on the full `eval_vs_builtin.py` grid;
  non-zero leaderkills on full ladder maps; a fitted Elo curve and measured
  $/Elo. **Re-estimate Tiers b/c from this curve before funding them.**

### Phase 2 — Tier b decisive campaign (~$500–1,500)
**Goal: competitive with expert humans.**
- Only after Phase 1 shows the pipeline learns *and* the bigger net helps.
- Net ~10–30M. If ≤24GB VRAM: keep cheapest CPU-rich spot 4090/A6000, or stand
  up the **Hetzner split** (option 4) for genuine 24/7 multi-week running. If
  >24GB: **Azure A100-80GB spot** (option 3), bankrolled by the **Founders Hub
  credit** (apply now — approval + Azure N-series quota ticket each take days).
- Turn on §3.5 efficiency tricks (playout-cap randomization, aux targets,
  progressive growth).
- Stand up §3.4(2) human-anchor eval (score vs archived ladder replays).
- **Exit criteria:** Elo plateau well above RCA; high move-agreement with
  strong human replays; low blunder rate; convincingly beats `dummy_policy`
  and all prior checkpoints.

### Phase 3 — Tier c superhuman (stretch, ~$3k–20k+)
**Goal: beats the strongest humans; prove it.**
- Net 30–100M. Small spot fleet (8–16× 4090 for 1–2 months) or KataGo-style
  long-haul; finance in stages off the measured curve, never up front.
- Capstone validation: §3.4(3) live ladder play vs humans — the only direct
  proof, manual and heavy.

---

## 7. How "superhuman" gets defined and proven

Because there is no live human rating pool wired in, use a ladder of
increasingly strong, increasingly convincing proxies:

1. **Internal Elo** (cheap, continuous): monotone improvement vs all prior
   checkpoints + RCA + scripted baselines.
2. **RCA dominance** (weak floor): ≥ ~95% vs built-in AI across the full grid.
3. **Human-replay agreement** (teacher-forced, no opponents needed): high
   top-k agreement with strong human moves; low rate of moves a strong human
   would call blunders; calibrated on the archived ladder corpus.
4. **Live ladder play** (direct, expensive): positive win-rate vs strong
   ranked human opponents. *This* is "superhuman" — the rest are necessary,
   not sufficient.

Be explicit in any claim about which rung was cleared; rungs 1–3 can be high
while rung 4 is unproven.

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **GPU starvation** from low vCPU (the #1 trap; inflates $/capability 3–10×) | Batched MCTS + actor pool (§3.1); rent only ≥16–24 vCPU/GPU; verify *allocated* vCPU before a long run |
| **Undersized net** wastes compute | Scale to 3–30M *before* the campaign (§3.2); keep it small enough that forward stays cheap vs rollout |
| **Position-blind model** can't reason about terrain/distance | Add 2D positional encoding (§3.3) |
| **Spot eviction** (Azure 30s notice) | Atomic checkpoints every N min; spot max-price = on-demand; SkyPilot managed jobs |
| **Forgotten idle instance** (classic money leak) | Wall-clock kill + idle-GPU watchdog auto-destroy (§3.6) |
| **Quota lead time** (AWS/Azure/GCP GPU quota defaults to 0) | File increase + Founders Hub app days ahead |
| **Estimates are analogies** | Phase 1 is a calibration experiment; re-price from your curve |
| **Tier c may exceed solo budget** | Treat as contingent stretch; consider distributed/crowd or stop at strong Tier b |
| **Stale prices** | All figures dated 2026-06-17; re-verify in console before spend |

---

## 9. Immediate next steps

1. **Apply to Microsoft for Startups Founders Hub** (lead time) — unlocks the
   cheapest decisive-run path.
2. **Implement batched MCTS leaf eval + virtual loss** in `tools/mcts.py`
   (§3.1) and the **actor-pool inference server** (extend `--workers`).
3. **Add arch CLI flags** to `tools/sim_self_play.py` (§3.2).
4. **Run the CUDA smoke** (`docs/running_on_gpu.md`) on a cheap rented box;
   record games/sec and GPU utilization.
5. **Add the Elo ladder** (§3.4-1) so Phase 1 has a strength axis.
6. **Launch Phase 1** (Tier a) on Hyperstack A6000 spot / filtered Vast 4090;
   produce the Elo-vs-compute curve; re-estimate Tiers b/c.

---

### Appendix: key sources (2026-06-17)
- Compute estimates: AlphaZero (arXiv:1712.01815), MuZero, KataGo
  (arXiv:1902.10565) + public run (katagotraining.org), Lc0, AlphaZero.jl
  Connect-4 tutorial, MiniZero (arXiv:2310.11305), microRTS RAISocketAI
  (arXiv:2402.08112), AlphaStar.
- Engineering: OpenSpiel AlphaZero docs; Oracle "Lessons From Alpha Zero pt5";
  DeepMind `mctx`; SEED RL; TorchBeast/MonoBeast (arXiv:1910.03552); TorchDemon;
  SkyPilot; vast.ai CLI.
- Prices: getdeploying, Spheron, RunPod, Vast, Lambda, Hyperstack, Azure,
  Hetzner price-adjustment doc (eff. 2026-06-15), OVHcloud, Scaleway,
  DataCrunch — all retrieved 2026-06-17.
