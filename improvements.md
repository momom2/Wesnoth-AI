To be fixed:

✅ server.py [request_choice] handling - FIXED (improved response format and logging)
✅ Wesnoth integration architecture - DESIGNED (Lua bridge + JSON server)
✅ action_selector.py index bug - FIXED (line 250 now uses actual map dimensions)
✅ classes.py validation mismatch - FIXED (changed defense count from 16 to 17)
✅ Map dataclass hashability - FIXED (changed hexes and units to List instead of Set)
✅ Action selection logic - IMPLEMENTED (action_selector.py with full recruit/move/attack logic)

⚠️ game_manager.py - needs completion:
    - Training loop (maybe_update_ai needs loss calculations)
    - Checkpoint loading/saving

⚠️ ai_server.py - functional but needs:
    - Checkpoint loading implementation (_load_checkpoint is TODO)
    - Better error handling for invalid states

✅ Wesnoth API integration - IMPLEMENTED via:
    - lua_ai_bridge.lua: Collects game state and executes actions
    - ai_server.py: Hosts transformer model and returns decisions
    - test_scenario.cfg: Test scenario configuration
    Ref: https://wiki.wesnoth.org/LuaAI, https://wiki.wesnoth.org/Creating_Custom_AIs

⏳ Dependencies needed for Lua integration:
    - LuaSocket (TCP communication)
    - dkjson or luajson (JSON parsing)
    - See INTEGRATION_GUIDE.md for installation instructions 


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