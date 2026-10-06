-- action_executor.lua
-- Executes the moves the hidden-units oracle (tools/hidden_units_oracle.py)
-- scripts through the turn stage, and its end of turn.

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
    
    -- Oracle mode: the route the engine will walk, computed the way
    -- ai.move computes it (the mover's side's view: hidden enemies do
    -- not exist for the pathfinder), so the Python driver can walk the
    -- SAME path through the simulator and compare where each stops.
    local oracle = nil
    if wml.variables.oracle_mode then
        local unit_id = unit.id
        local path, cost = wesnoth.paths.find_path(unit, { target_x, target_y })
        local steps = {}
        for i, loc in ipairs(path or {}) do
            steps[i] = { x = loc[1], y = loc[2] }
        end
        oracle = { unit_id = unit_id, path = steps, cost = cost }
    end

    -- Execute the move using global ai table
    local result = ai.move(unit, target_x, target_y)

    if oracle then
        local after = wesnoth.units.find_on_map({ id = oracle.unit_id })[1]
        if after then
            oracle.final = { x = after.x, y = after.y, moves = after.moves }
        end
        oracle.status = tostring(result.status)
        oracle.gamestate_changed = result.gamestate_changed or false
    end

    return {
        success = result.ok,
        gamestate_changed = result.gamestate_changed or false,
        error = result.status,
        oracle = oracle,
    }
end

-- Main action dispatcher
function action_executor.execute_action(action)
    local action_type = action.type
    
    if action_type == "move" then
        return action_executor.execute_move(action)
    elseif action_type == "end_turn" then
        return {success = true, end_turn = true}
    else
        return {success = false, error = "unknown_action_type"}
    end
end

return action_executor
