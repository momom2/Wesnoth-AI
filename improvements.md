Ideas for improvements:

Better modelization of the game:
- Provide defence caps
- Provide starting location of leaders
- Explicit gold-tracking (self and enemy)
- Explicit fog-of-war prediction

Better memory:
- Separate memory into tactical and strategic. Tactical memory gets wiped out at the beginning of each turn.
- Hierarchical memory with different time scales
- Attention-based selective updates (sounds complicated)
- Episodic memory for storing specific game situations (not convinced)

From EfficientZero:
https://www.lesswrong.com/posts/mRwJce3npmzbKfxws/efficientzero-how-it-works
- Self Supervised Consistency Loss
- Value Prefix
- Off-Policy Correction
- ReAnalyse https://arxiv.org/pdf/2104.06294 (???) (gotta understand it first)