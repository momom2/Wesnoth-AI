# Tier-a re-measurement protocol (draft 2026-07-20)

## Why redo it

The 2026-07-05 Tier-a verdict ("curve flattened in the second half;
bottleneck is behavioral — fix before Tier-b") was measured on a sim
where **movement was silently impossible** (BACKLOG §2026-07-17):
42/66 ladder games ended by max_actions, 22 by turn cap, 2 by kills.
Every premise of that verdict is dead. The runbook's Tier-b gate
("does Elo-per-GPU-hour hold up?") needs a fresh answer on the
rebuilt sim before any Tier-b spend.

## What changed since the old measurement

- Movement/pathfinding rebuilt (true reachability, multi-hex orders);
  games are ~95% decisive by leader kill in 10-30 turns.
- MCTS demonstrably earns rent: mcts32 vs raw = 9-0-1 (+321 Elo).
- Value head decompressed (late-game AUC 0.85, E[V] ±0.4).
- Gate retired; oracle retired; draw_value_weight 0.25.
- All export/validation desyncs fixed (engine-verified CLEAN across
  every scenario class, 2026-07-19 sweep).

## Protocol

**Players** (all `mcts:32:<ckpt>` — the eval convention the rent
match used; PURE Elo, draws dropped):

1. `sl0` — supervised_5m_epoch3.pt (the SL prior, zero self-play).
2. `day1` — the overnight box campaign checkpoint
   (tier_a_campaign.pt @ HF, ~30 box-iterations ≈ $6).
3. `dayN` — the next campaign checkpoint after the resumed run
   (whenever the box next accumulates a comparable chunk).
4. `random` — random-init anchor at Elo 0 (gauge fix).
5. `dummy` — scripted floor (strength sanity, cheap).

**Games**: `--games-per-pair 12` (60 games for 5 players ≈
tolerable CPU-days locally at ~15 min/MCTS game, or ~2h on the box;
prefer the box). Ladder pool only, `--max-turns 60`, fog on
(training condition). `--prior-games 2` regularization as usual.

**Metric**: PURE Elo (no material tiebreak in scoring). Report
W-D-L matrices alongside; note draw rate (expect <10% now — if
draws exceed ~25% the pure ladder degrades and we revisit).

**The Tier-a answer**: Elo(day1) − Elo(sl0) per dollar (~$6 of the
crash-loop-discounted overnight ≈ the first honest
self-play-Elo-per-dollar point on working machinery), then
Elo(dayN) − Elo(day1) for the slope. GO for Tier-b if the slope is
clearly positive at these scales; NO-GO/redesign if self-play adds
nothing over the SL prior despite functional search.

## Cost

- Local: ~0 dollars, ~2-3 days of laptop CPU (background).
- Box: ~2-3h of a 4090 ≈ $1.50, plus it can run while the campaign
  is paused (same instance, before resuming training).

## Prereqs

- [x] Export/desync sweep green (2026-07-19).
- [x] Training-loop smoke green on current code.
- [ ] Box topped up + restarted (or accept the local CPU run).
- [ ] Decide fogless_ratio for the resumed campaign (separate).
