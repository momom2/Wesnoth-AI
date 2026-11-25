# The blind-spot audit (2026-11-25)

Readings and reports for `docs/blindspot_audit_20261125.md`. Nothing here
trains or changes a network. The scripts replay recorded games on the CPU.

- `reports/`: the brief given to the seven auditing agents and their
  reports as they wrote them. A report's finding is a claim; the document
  above says which were checked and how, and which did not hold.
- `hex_tokens.py`, `hex_tokens_round_0576.json`: the share of the board
  that has a hex token (40 games of round 576, two a map, one decision
  in five): 364 of 1,030 hexes a decision, and 49% of the hexes the side
  sees have none.
- `audit_counts.py`, `audit_counts_round_0576.json`: on the same games,
  the advancements (111, of which 42 from a type with a choice, the first
  option taken in all 42), the moves that bring an enemy into view (964 of
  13,349), and the enemy-faction input at each side's first decision
  (certain in 80 of 80).
- `count_refused.py`: the recruits that bounced on a hex a hidden unit
  holds, over rounds 561 to 576: 3 of 96,718, none refused.
- `new_term_reach.py`, `new_term_reach_round_0576.json`: whether the new
  projection's term reaches the policy served in fp16. On 2,206 decisions
  it does, in full: the served log probabilities carry 1.02 ± 0.02 of the
  term's effect, which is 0.0035 root mean square.

The checks ran at Idle priority beside the simple loop and cost its round
584 about three minutes.
