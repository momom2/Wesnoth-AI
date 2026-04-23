-- ca_action_executor.lua
-- Candidate Action that waits for action from Python and executes it
-- This CA has max_score=999980 and loops until end_turn is received

local action_executor = wesnoth.require("~add-ons/wesnoth_ai/lua/action_executor.lua")

local ca_action_executor = {}

-- Helper function to serialize a simple action table to string for comparison
local function action_to_string(action)
    if not action then return "nil" end
    
    local parts = {}
    table.insert(parts, "type=" .. tostring(action.type or ""))
    
    if action.start_x then table.insert(parts, "sx=" .. tostring(action.start_x)) end
    if action.start_y then table.insert(parts, "sy=" .. tostring(action.start_y)) end
    if action.target_x then table.insert(parts, "tx=" .. tostring(action.target_x)) end
    if action.target_y then table.insert(parts, "ty=" .. tostring(action.target_y)) end
    if action.weapon_index then table.insert(parts, "w=" .. tostring(action.weapon_index)) end
    if action.unit_type then table.insert(parts, "u=" .. tostring(action.unit_type)) end
    
    return table.concat(parts, "|")
end

function ca_action_executor:evaluation()
    local game_id = wml.variables.game_id or "game_0"
    local action_file = string.format("~add-ons/wesnoth_ai/games/%s/action_input.lua", game_id)
    
    -- Poll for action file changes with timeout
    local start_time = wesnoth.get_time_stamp()
    local timeout_ms = 30000  -- 30 seconds timeout
    
    -- Store previous action to detect changes
    local prev_action_str = wml.variables.prev_action_str or ""
    
    local action_changed = false
    local action_data = nil
    
    while (wesnoth.get_time_stamp() < start_time + timeout_ms) and not action_changed do
        local success, result = pcall(function()
            return wesnoth.dofile(action_file)
        end)
        
        if success and result then
            -- Convert action to string for comparison
            local action_str = action_to_string(result)
            
            if action_str ~= prev_action_str then
                action_data = result
                action_changed = true
                wml.variables.prev_action_str = action_str
            else
                -- No change yet, wait a bit
                wesnoth.interface.delay(50)  -- 50ms delay
            end
        else
            -- File read error, wait and retry
            wesnoth.interface.delay(50)
        end
    end
    
    if action_changed then
        -- Store action in WML variable (we'll retrieve it in execution)
        -- We can't store the Lua table directly, so we store a marker
        wml.variables.pending_action_ready = true
        wml.variables.cached_action_str = action_to_string(action_data)
        return 999980
    else
        std_print(string.format("[Turn %d, Side %d] WARNING: Action timeout, ending turn", 
            wesnoth.current.turn, wesnoth.current.side))
        -- Timeout - end turn by returning 0
        return 0
    end
end

function ca_action_executor:execution(cfg, data)
    -- NOTE: ai_context parameter removed - use global 'ai' table instead
    local side_number = wesnoth.current.side
    local game_id = wml.variables.game_id or "game_0"
    
    -- Check if action is ready
    if not wml.variables.pending_action_ready then
        std_print(string.format("[Turn %d, Side %d] ERROR: No pending action", 
            wesnoth.current.turn, side_number))
        return
    end
    
    -- Re-read the action file to get the actual action table
    local action_file = string.format("~add-ons/wesnoth_ai/games/%s/action_input.lua", game_id)
    local success, action = pcall(function()
        return wesnoth.dofile(action_file)
    end)
    
    if not success or not action then
        std_print(string.format("[Turn %d, Side %d] ERROR: Could not read action", 
            wesnoth.current.turn, side_number))
        return
    end
    
    -- Clear the marker
    wml.variables.pending_action_ready = false
    
    std_print(string.format("[Turn %d, Side %d] Executing action: %s", 
        wesnoth.current.turn, side_number, action.type))
    
    -- Check for end_turn action
    if action.type == "end_turn" then
        std_print(string.format("[Turn %d, Side %d] Ending turn", 
            wesnoth.current.turn, side_number))
        return  -- Execution completes, CA returns 0 next evaluation
    end
    
    -- Execute the action (passing global ai table)
    local result = action_executor.execute_action(action)
    
    if result.success then
        std_print(string.format("[Turn %d, Side %d] Action succeeded", 
            wesnoth.current.turn, side_number))
        
        -- If it's not end_turn, this CA will be evaluated again
        -- for the next action
    else
        std_print(string.format("[Turn %d, Side %d] Action failed: %s", 
            wesnoth.current.turn, side_number, result.error or "unknown"))
        
        -- Even if action fails, we continue to avoid blacklisting
        -- The Python side should handle this gracefully
    end
    
    -- Check if game is over
    local leaders = wesnoth.units.find_on_map({canrecruit = true})
    local side1_has_leader = false
    local side2_has_leader = false
    
    for _, leader in ipairs(leaders) do
        if leader.side == 1 then side1_has_leader = true end
        if leader.side == 2 then side2_has_leader = true end
    end
    
    if not side1_has_leader or not side2_has_leader then
        local winner = side1_has_leader and 1 or (side2_has_leader and 2 or 0)
        std_print(string.format("[Turn %d] ===GAME_OVER===", wesnoth.current.turn))
        std_print(tostring(winner))
        
        -- Actually end the scenario
        wesnoth.wml_actions.endlevel({
            result = "victory",
            bonus = false,
            carryover_percentage = 0
        })
    end
end

return ca_action_executor
