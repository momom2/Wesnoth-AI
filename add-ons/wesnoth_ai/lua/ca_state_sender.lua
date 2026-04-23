-- ca_state_sender.lua
-- Candidate Action: serialize the game state and hand it off to Python
-- by emitting it via std_print.
--
-- Transport: std_print writes to Wesnoth's own .out.log file under
-- <userdata>/logs/. This is the only general-purpose outbound channel
-- available to Lua in Wesnoth 1.18 (io.open is sandboxed out). Python
-- tails that log and reassembles blocks framed between our marker lines.
--
-- Frame:
--     ===WESNOTH_AI_STATE_BEGIN===
--     meta: game_id=<id> turn=<n> side=<s>
--     <wml.tostring(state) output>
--     ===WESNOTH_AI_STATE_END===
--
-- The state_sender CA is scored higher than action_executor, so it runs
-- first in each AI turn and gets blacklisted after executing once — it
-- emits one state block per turn, then action_executor takes over.

local state_collector = wesnoth.require("~add-ons/wesnoth_ai/lua/state_collector.lua")
local json = wesnoth.require("~add-ons/wesnoth_ai/lua/json_encoder.lua")

local ca_state_sender = {}

local FRAME_BEGIN = "===WESNOTH_AI_STATE_BEGIN==="
local FRAME_END   = "===WESNOTH_AI_STATE_END==="

local function detect_game_over(state)
    local side1, side2 = false, false
    for _, leader in ipairs(wesnoth.units.find_on_map({ canrecruit = true })) do
        if leader.side == 1 then side1 = true end
        if leader.side == 2 then side2 = true end
    end
    if side1 and side2 then return end
    state.game_over = true
    if not side1 and not side2 then
        state.winner = 0
    elseif not side1 then
        state.winner = 2
    else
        state.winner = 1
    end
end

function ca_state_sender:evaluation()
    return 999990
end

function ca_state_sender:execution(cfg, data)
    local side_number = wesnoth.current.side
    local game_id = wml.variables.game_id or "game_0"
    local turn = wesnoth.current.turn

    local state = state_collector.collect_game_state(side_number, game_id)
    detect_game_over(state)

    std_print(FRAME_BEGIN)
    std_print(string.format("meta: game_id=%s turn=%d side=%d",
        game_id, turn, side_number))
    std_print(json.encode(state))
    std_print(FRAME_END)
end

return ca_state_sender
