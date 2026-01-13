-- lua_ai_bridge.lua
-- Wesnoth Lua AI that communicates with external Python transformer model

local socket = require("socket")
local json = require("json") -- May need luajson or dkjson library

local H = wesnoth.require("helper")
local AH = wesnoth.require("ai/lua/ai_helper.lua")

-- Configuration
local HOST = "localhost"
local PORT = 15001  -- AI server port (15000 is for WML server)
local TIMEOUT = 5

-- Connection state
local ai_socket = nil
local connected = false
local game_id = nil

-- Initialize connection to Python AI server
local function connect_to_ai_server()
    if connected then return true end

    local sock = socket.tcp()
    sock:settimeout(TIMEOUT)

    local success, err = sock:connect(HOST, PORT)
    if not success then
        wesnoth.log("error", "Failed to connect to AI server: " .. tostring(err))
        return false
    end

    ai_socket = sock
    connected = true
    game_id = wesnoth.game_config.id or "game_" .. os.time()

    wesnoth.log("info", "Connected to AI server, game_id: " .. game_id)
    return true
end

-- Send game state to Python AI and receive action
local function get_ai_decision(game_state)
    if not connect_to_ai_server() then
        return nil
    end

    -- Serialize game state to JSON
    local state_json = json.encode(game_state)

    -- Send state to AI server
    local success, err = ai_socket:send(state_json .. "\n")
    if not success then
        wesnoth.log("error", "Failed to send state: " .. tostring(err))
        connected = false
        return nil
    end

    -- Receive action from AI server
    local response, err = ai_socket:receive("*l")
    if not response then
        wesnoth.log("error", "Failed to receive action: " .. tostring(err))
        connected = false
        return nil
    end

    -- Parse action
    local action = json.decode(response)
    return action
end

-- Convert Wesnoth unit to our format
local function serialize_unit(unit)
    local attacks = {}
    for i, attack in ipairs(unit.attacks) do
        table.insert(attacks, {
            name = attack.name,
            type = attack.type,
            damage = attack.damage,
            strikes = attack.num_blows,
            is_ranged = (attack.range == "ranged"),
            specials = attack.specials or {}
        })
    end

    return {
        name = unit.type,
        id = unit.id,
        side = unit.side,
        is_leader = unit.canrecruit,
        x = unit.x,
        y = unit.y,
        max_hp = unit.max_hitpoints,
        max_moves = unit.max_moves,
        max_exp = unit.max_experience,
        cost = unit.cost,
        alignment = tostring(unit.alignment),
        levelup_names = {}, -- TODO: Get advancement options
        current_hp = unit.hitpoints,
        current_moves = unit.moves,
        current_exp = unit.experience,
        has_attacked = unit.attacks_left == 0,
        attacks = attacks,
        resistances = unit.resistance,
        defenses = unit.defense,
        movement_costs = unit.movement_costs,
        abilities = unit.abilities or {},
        traits = unit.traits or {}
    }
end

-- Collect current game state
local function collect_game_state()
    local my_side = wesnoth.current.side
    local my_units = wesnoth.units.find_on_map({ side = my_side })
    local enemy_units = wesnoth.units.find_on_map({
        { "filter_side", { { "enemy_of", { side = my_side } } } }
    })

    -- Serialize units
    local units = {}
    for i, unit in ipairs(my_units) do
        table.insert(units, serialize_unit(unit))
    end
    for i, unit in ipairs(enemy_units) do
        table.insert(units, serialize_unit(unit))
    end

    -- Get map information
    local map_width, map_height = wesnoth.current.map.playable_width, wesnoth.current.map.playable_height

    -- Get visible hexes (simplified - would need fog of war handling)
    local hexes = {}
    for x = 1, map_width do
        for y = 1, map_height do
            local terrain = wesnoth.current.map[{x, y}]
            if terrain then
                table.insert(hexes, {
                    x = x,
                    y = y,
                    terrain_types = { terrain },
                    modifiers = {} -- TODO: Add terrain modifiers
                })
            end
        end
    end

    -- Get recruitment options
    local recruits = {}
    for i, unit_type in ipairs(wesnoth.sides[my_side].recruit) do
        local unit_info = wesnoth.unit_types[unit_type]
        if unit_info then
            table.insert(recruits, {
                name = unit_type,
                hp = unit_info.max_hitpoints,
                moves = unit_info.max_moves,
                exp = unit_info.max_experience,
                cost = unit_info.cost,
                alignment = tostring(unit_info.alignment),
                levelup_names = {},
                attacks = {}, -- TODO: Serialize attacks
                resistances = unit_info.resistance,
                defenses = unit_info.defense,
                movement_costs = unit_info.movement_costs,
                abilities = unit_info.abilities or {},
                traits = {}
            })
        end
    end

    return {
        game_id = game_id,
        turn = wesnoth.current.turn,
        side = my_side,
        gold = wesnoth.sides[my_side].gold,
        map = {
            width = map_width,
            height = map_height,
            mask = {}, -- TODO: Add map mask
            fog = {}, -- TODO: Add fog of war
            hexes = hexes,
            units = units
        },
        recruits = recruits,
        game_over = false
    }
end

-- Candidate Action: Main AI Decision
function transformer_ai_evaluation()
    -- Return high score to ensure this CA runs
    return 300000
end

function transformer_ai_execution()
    -- Collect current game state
    local game_state = collect_game_state()

    -- Get decision from Python AI
    local action = get_ai_decision(game_state)

    if not action then
        wesnoth.log("warn", "No action received from AI server, ending turn")
        return
    end

    wesnoth.log("info", "Received action: " .. action.type)

    -- Execute action based on type
    if action.type == "move" then
        local unit = wesnoth.units.get(action.unit_id)
        if unit then
            ai.move(unit, action.target_x, action.target_y)
        end

    elseif action.type == "attack" then
        local attacker = wesnoth.units.get(action.attacker_id)
        local defender = wesnoth.units.find_on_map({ x = action.target_x, y = action.target_y })[1]

        if attacker and defender then
            ai.attack(attacker, defender, action.weapon_index or 0)
        end

    elseif action.type == "recruit" then
        local recruit_hex = wesnoth.special_locations["recruit_" .. wesnoth.current.side]
        if recruit_hex then
            ai.recruit(action.unit_type, recruit_hex[1], recruit_hex[2])
        else
            -- Find a keep hex for recruitment
            local leader = wesnoth.units.find_on_map({ side = wesnoth.current.side, canrecruit = true })[1]
            if leader then
                ai.recruit(action.unit_type, leader.x, leader.y)
            end
        end

    elseif action.type == "end_turn" then
        -- Do nothing, let turn end naturally
        return
    end
end

return {
    evaluation = transformer_ai_evaluation,
    execution = transformer_ai_execution
}
