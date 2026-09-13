# relset self-pin (2026-09-12, box 50759583, RTX A4000)

The reference player against itself: `relset` (label `relset`) against
`relset` (label `relset_ref`), the same checkpoint
(`training/checkpoints/relset.pt`), at `raw:t0` through the shared
inference server, 20 workers, 40 seeds replayed in each of four match
arms (`scripts/core_sim_box.sh`, which ran them to time the
Rust-owned state). Every record carries `basis_a = basis_b = relset`
and `raw_temperature_a = raw_temperature_b = 0.0`, so both sides played
the reference player's own basis at temperature zero. (The records'
`relevant_set_a/b` field is the CLI override, unset here; the
effective basis is `basis_a`/`basis_b`.)

| quantity | value |
|---|---|
| games | 160 |
| decisive | 93 |
| at the turn cap | 67 |
| A's score | 39 / 93 = 0.419 +- 0.051 |
| as Elo | -57 +- 37 |
| A on side 1 | 23 / 61 = 0.377 |
| A on side 2 | 16 / 32 = 0.500 |
| per arm (wins / decisive) | P 10/23, P2 9/23, C 10/23, C2 10/24 |

Reading: no A/B asymmetry is detected. The score is 1.6 standard
errors below even (two-sided p about 0.13), and the four arms agree
with each other. The point estimate sits below 0.5 because the
decisive games split unevenly by side -- A held side 1 in 61 of the 93
-- and that subset went against side 1; the side-balanced estimate is
(0.377 + 0.500) / 2 = 0.439, still within noise at these counts.

What this does and does not settle: it rules out a gross asymmetry in
how the shared inference server serves the two labels (which is what
the pin exists to catch). It is not a tight pin: +- 37 Elo. A tight
one is an 800-game match against itself, about 18 min and $0.20 on a
4090 (plan 1.5's measured rate).

The high cap rate (67 of 160) is the known behaviour of `raw:t0`
against itself (CLAUDE.md: 17 of 40 at the cap in the 2026-09-04
measurement), not a property of this run.
