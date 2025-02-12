To be fixed:

game_manager, batch_manager are unvalidated, unfinished and make abusive assumptions about data format.
server has been tested but I can't validate it since it's beyond my competence. Keep in mind there might be errors.
Need to add interaction with Wesnoth using API <- need to look into Wesnoth code to figure out how to do that
    Ref: https://wiki.wesnoth.org/LuaAI, https://wiki.wesnoth.org/Creating_Custom_AIs, 


Ideas for improvements:

Early training:
- More intrusive reward (for income, for killing, for recruiting) (to learn valid moves)


Better modelization of the game:
- Provide defence caps
(Hassle, might negatively impact compute requirements.)
- Provide starting location of leaders
(Hassle, not even sure that's provided to players, can be memorized since the map pool is so limited.)
- Explicit gold-tracking (self and enemy)
- Explicit fog-of-war prediction
(Let's see if it works unconstrained first)
- Add ability to pick advancements per the plan advancements mod.

Better memory:
- Separate memory into tactical and strategic. Tactical memory gets wiped out at the beginning of each turn.
- Hierarchical memory with different time scales
(Let's see if it works unconstrained first)
- Attention-based selective updates (sounds complicated)
- Episodic memory for storing specific game situations (i.e. tactical table) (suggested by Claude, not convinced it's worthwhile)

From EfficientZero:
https://www.lesswrong.com/posts/mRwJce3npmzbKfxws/efficientzero-how-it-works
- Self Supervised Consistency Loss
- Value Prefix
- Off-Policy Correction
- ReAnalyse https://arxiv.org/pdf/2104.06294 (???) (gotta understand it first)


- Look into Diplodocus to see what I can copy; looks like I could use something similar to the strategic module for Wesnoth tactics.
- That thing where the policy is corrected by the expected policy and expected opponent policy for finding Nash equilibria in mixed strategies (Nash learning?)