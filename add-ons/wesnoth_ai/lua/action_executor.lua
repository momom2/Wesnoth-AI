-- action_executor.lua
-- Helper functions to execute actions received from Python AI
-- FIXED: Removed ai_context parameter - uses global 'ai' table instead

local action_executor = {}

-- Find nearest valid destination if exact location is invalid
local function find_nearest_valid_hex(unit, target_x, target_y)
    -- Simple spiral search for nearest reachable hex
    local best_x, best_y = target_x, target_y
    local best_dist = 9999
    
    -- Get reachable hexes for this unit
    local reach = wesnoth.paths.find_reach(unit)
    
    for _, loc in ipairs(reach) do
        local dx = loc[1] - target_x
        local dy = loc[2] - target_y
        local dist = dx * dx + dy * dy
        
        if dist < best_dist then
            best_dist = dist
            best_x = loc[1]
            best_y = loc[2]
        end
    end
    
    return best_x, best_y
end

-- Execute a move action
function action_executor.execute_move(action)
    local start_x = action.start_x
    local start_y = action.start_y
    local target_x = action.target_x
    local target_y = action.target_y
    
    -- Get the unit
    local unit = wesnoth.units.get(start_x, start_y)
    if not unit then
        return {success = false, error = "no_unit_at_location"}
    end
    
    -- Check if unit belongs to current side
    if unit.side ~= wesnoth.current.side then
        return {success = false, error = "not_own_unit"}
    end
    
    -- Check if unit has moves left
    if unit.moves <= 0 then
        return {success = false, error = "no_moves_left"}
    end
    
    -- Check move validity using global ai table
    local check = ai.check_move(unit, target_x, target_y)
    if not check.ok then
        -- Try to find nearest valid hex
        target_x, target_y = find_nearest_valid_hex(unit, target_x, target_y)
        check = ai.check_move(unit, target_x, target_y)
        
        if not check.ok then
            return {success = false, error = "invalid_move"}
        end
    end
    
    -- Execute the move using global ai table
    local result = ai.move(unit, target_x, target_y)
    
    return {
        success = result.ok,
        gamestate_changed = result.gamestate_changed or false,
        error = result.status
    }
end

-- Execute an attack action
function action_executor.execute_attack(action)
    local attacker_x = action.start_x
    local attacker_y = action.start_y
    local defender_x = action.target_x
    local defender_y = action.target_y
    local weapon_index = action.weapon_index or 0
    
    -- Get units
    local attacker = wesnoth.units.get(attacker_x, attacker_y)
    local defender = wesnoth.units.get(defender_x, defender_y)
    
    if not attacker then
        return {success = false, error = "no_attacker"}
    end
    if not defender then
        return {success = false, error = "no_defender"}
    end
    
    -- Check if attacker belongs to current side
    if attacker.side ~= wesnoth.current.side then
        return {success = false, error = "not_own_unit"}
    end
    
    -- Check if defender is enemy
    if defender.side == wesnoth.current.side then
        return {success = false, error = "cannot_attack_own_unit"}
    end
    
    -- Check if attacker has attacks left
    if attacker.attacks_left <= 0 then
        return {success = false, error = "no_attacks_left"}
    end
    
    -- Validate weapon index (Lua is 1-indexed for arrays)
    if weapon_index < 0 or weapon_index >= #attacker.attacks then
        weapon_index = 0  -- Default to first weapon
    end
    
    -- Check attack validity using global ai table
    local check = ai.check_attack(attacker, defender, weapon_index)
    if not check.ok then
        return {success = false, error = "invalid_attack"}
    end
    
    -- Execute the attack using global ai table
    local result = ai.attack(attacker, defender, weapon_index)
    
    return {
        success = result.ok,
        gamestate_changed = result.gamestate_changed or false,
        error = result.status
    }
end

-- Execute a recruit action
function action_executor.execute_recruit(action)
    local unit_type = action.unit_type
    local recruit_x = action.target_x
    local recruit_y = action.target_y
    
    -- Get current side info
    local side = wesnoth.sides[wesnoth.current.side]
    
    -- Check if we can afford it
    local unit_cost = wesnoth.unit_types[unit_type].cost or 999
    if side.gold < unit_cost then
        return {success = false, error = "insufficient_gold"}
    end
    
    -- Check if unit type is in recruit list
    local can_recruit = false
    for _, recruit in ipairs(side.recruit) do
        if recruit == unit_type then
            can_recruit = true
            break
        end
    end
    
    if not can_recruit then
        return {success = false, error = "unit_not_in_recruit_list"}
    end
    
    -- Check recruit validity using global ai table
    local check = ai.check_recruit(unit_type, recruit_x, recruit_y)
    if not check.ok then
        return {success = false, error = "invalid_recruit_location"}
    end
    
    -- Execute the recruitment using global ai table
    local result = ai.recruit(unit_type, recruit_x, recruit_y)
    
    return {
        success = result.ok,
        gamestate_changed = result.gamestate_changed or false,
        error = result.status
    }
end

-- Execute a recall action
function action_executor.execute_recall(action)
    local unit_id = action.unit_id
    local recall_x = action.target_x
    local recall_y = action.target_y
    
    -- Get current side info
    local side = wesnoth.sides[wesnoth.current.side]
    
    -- Check if we can afford it (recall cost)
    local recall_cost = side.recall_cost or 20
    if side.gold < recall_cost then
        return {success = false, error = "insufficient_gold"}
    end
    
    -- Find unit in recall list
    local recall_list = wesnoth.units.find_on_recall({side = wesnoth.current.side})
    local unit_found = false
    for _, unit in ipairs(recall_list) do
        if unit.id == unit_id then
            unit_found = true
            break
        end
    end
    
    if not unit_found then
        return {success = false, error = "unit_not_in_recall_list"}
    end
    
    -- Check recall validity using global ai table
    local check = ai.check_recall(unit_id, recall_x, recall_y)
    if not check.ok then
        return {success = false, error = "invalid_recall_location"}
    end
    
    -- Execute the recall using global ai table
    local result = ai.recall(unit_id, recall_x, recall_y)
    
    return {
        success = result.ok,
        gamestate_changed = result.gamestate_changed or false,
        error = result.status
    }
end

-- Main action dispatcher
function action_executor.execute_action(action)
    local action_type = action.type
    
    if action_type == "move" then
        return action_executor.execute_move(action)
    elseif action_type == "attack" then
        return action_executor.execute_attack(action)
    elseif action_type == "recruit" then
        return action_executor.execute_recruit(action)
    elseif action_type == "recall" then
        return action_executor.execute_recall(action)
    elseif action_type == "end_turn" then
        return {success = true, end_turn = true}
    else
        return {success = false, error = "unknown_action_type"}
    end
end

return action_executor
