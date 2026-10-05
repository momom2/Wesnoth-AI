-- init_oracle.lua
-- The engine's half of the scenario-init oracle
-- (tools/scenario_init_oracle.py). Installed on side 1 by
-- init_oracle_ai.cfg, so it runs at side 1's first turn: after
-- prestart, start and side 1's turn-1 init. It reports the whole board
-- (board_report.lua), not what side 1 can see. Then the turn ends.

local json = wesnoth.require("~add-ons/wesnoth_ai/lua/json_encoder.lua")
local board = wesnoth.require("~add-ons/wesnoth_ai/lua/board_report.lua")

local FRAME_BEGIN = "===WESNOTH_AI_STATE_BEGIN==="
local FRAME_END   = "===WESNOTH_AI_STATE_END==="

local M = {}

function M:report()
    std_print(FRAME_BEGIN)
    std_print(json.encode(board.record("scenario_init")))
    std_print(FRAME_END)
end

return M
