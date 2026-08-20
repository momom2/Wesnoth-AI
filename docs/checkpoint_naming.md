# Checkpoint naming scheme (user directive 2026-08-20)

Names encode a checkpoint's training history as a path through the
lineage tree, so ancestry and the right comparison points (nearest
common ancestor) are readable from the name alone.

## Grammar

    name   := steps ( "-" branch "-" steps )*
    steps  := <integer> "k"          # decision-steps SINCE the
                                     # preceding branch point,
                                     # rounded to the NEAREST thousand
    branch := [a-z][a-z0-9]*         # no "-" inside a slug; must not
                                     # parse as a steps token

Example: `2516k-b-294k-l4-430k` reads left to right as its history —
the trunk ran 2,516k steps, branched (`b`), ran 294k more, branched
(`l4`), ran 430k.

- The name is a LABEL; the exact step count lives in the catalog
  entry (`decision_step`). Rounding therefore cannot create real
  ambiguity, and collisions at the same rounded count are resolved
  by the catalog, not the name.
- A **branch** is any divergence: a new config restarted from a
  common parent, an architecture grow, a seed reused twice.
  Continuing the same campaign after a pause (leg 2 -> leg 3 on the
  same rolling checkpoint) is NOT a branch. Sibling slugs must be
  unique among siblings. Zero-step modifications at a branch point
  (e.g. a value-head reseed) belong to the child branch's config
  metadata, not the name.

## Ancestry rules (token-wise, never character-wise)

Tokenize on `-` into alternating steps/branch tokens. Character
prefixes mean nothing: `2516k-b-294k-tcs2` and `2516k-b-294k-tcs1`
share the character prefix `2516k-b-294k-tcs`, which names no
checkpoint.

- **Nearest common ancestor** = the longest shared TOKEN prefix,
  truncated back to a node boundary (a prefix ending in a steps
  token).
- **X is an ancestor of Y on the same branch** iff their token
  lists agree through Y's last branch slug and X's final steps
  token is <= Y's.

## Concatenation (name compression)

Only on the user's explicit say-so — typically when a branching has
yielded its results and all but one branch are definitely abandoned:
the surviving branch collapses into its parent and the step counts
ADD (`2516k-b-294k` -> `2810k`); every descendant name shortens with
it. The catalog records `aliases[old] = new` so historical names in
docs and edge records still resolve (see `tools/elo_catalog.py`;
alias resolution is applied when edges are recorded).

## Scope

Applies to IMMUTABLE identities: catalog labels, pinned eval
checkpoints, docs. Rolling operational files (`tier_b_l4.pt`, HF
escrow names) keep their operational names — they are mutable, so a
history-encoding name would lie within hours.

## Current tree (renamed 2026-08-20)

    2291k            ref (tier-a 5M, seed_20260718.pt)      Elo 0 (ref)
    2404k            tier-a trunk (campaign_live_20260729)
    2516k            tier-a trunk best (campaign_live_20260730)
    2516k-b-294k     imitation seed, 15M (imit_tierb_start;
                     `b` = net2net grow + tier-b + imitation retrain;
                     not yet Elo-rated)
    2516k-b-294k-tcs2-558k   leg-3 end (abandoned branch)
    2516k-b-294k-l4-<D>k     leg-4 pins (D = step - 2,809,659)

Old labels (`ref_2p29M`, `old_2p40M`, `new_2p52M`, `tcs3`) are
catalog aliases of the above.
