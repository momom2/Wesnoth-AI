-- ca_state_sender.lua
-- Candidate Action that collects game state and sends to Python via stdout
-- This CA has max_score=999990 and gets blacklisted after first execution per turn

local state_collector = wesnoth.require("~add-ons/wesnoth_ai/lua/state_collector.lua")

local ca_state_sender = {}

function ca_state_sender:evaluation()
    -- Always return high score (will be blacklisted after execution)
    return 999990
end

function ca_state_sender:execution(cfg, data)
    local side_number = wesnoth.current.side
    local game_id = wml.variables.game_id or "game_0"
    
    std_print(string.format("[Turn %d, Side %d] Collecting game state...", 
        wesnoth.current.turn, side_number))
    
    -- Collect complete game state
    local game_state = state_collector.collect_game_state(side_number, game_id)
    
    -- Check for game over conditions
    local side1_has_leader = false
    local side2_has_leader = false
    
    local leaders = wesnoth.units.find_on_map({canrecruit = true})
    for _, leader in ipairs(leaders) do
        if leader.side == 1 then side1_has_leader = true end
        if leader.side == 2 then side2_has_leader = true end
    end
    
    if not side1_has_leader or not side2_has_leader then
        game_state.game_over = true
        if not side1_has_leader and not side2_has_leader then
            game_state.winner = 0  -- Draw
        elseif not side1_has_leader then
            game_state.winner = 2
        else
            game_state.winner = 1
        end
    end
    
    -- Convert to WML string format (since wesnoth.format_json doesn't exist)
    -- We'll use wml.tostring to serialize the state
    local wml_state = wml.tostring(game_state)
    
    -- Output WML state to stdout with markers
    std_print("===WML_STATE_BEGIN===")
    std_print(wml_state)
    std_print("===WML_STATE_END===")
    
    std_print(string.format("[Turn %d, Side %d] State sent successfully", 
        wesnoth.current.turn, side_number))
    
    -- This CA will be blacklisted after this execution
    -- The action executor CA will now take over
end

return ca_state_sender
