-- ai_plugin.lua
-- Wesnoth AI plugin that communicates with external Python AI via file I/O

local H = wesnoth.require "helper"
local AH = wesnoth.require "ai/lua/ai_helper.lua"
local LS = wesnoth.require "location_set"

local ai_python = {}

-- Configuration from command line arguments or AI config
local game_id = wesnoth.game_config.external_ai_game_id or "default"
local base_path = wesnoth.game_config.external_ai_path or "./games/" .. game_id

-- File paths for communication
local state_file = base_path .. "/state.json"
local action_file = base_path .. "/action.json"
local signal_file = base_path .. "/signal"

-- Ensure directory exists
os.execute("mkdir -p " .. base_path)

-- Helper function to write JSON (simplified, assumes data is already formatted)
local function write_file(path, content)
    local file = io.open(path, "w")
    if not file then
        wesnoth.log("error", "Failed to open file for writing: " .. path)
        return false
    end
    file:write(content)
    file:close()
    return true
end

-- Helper function to read JSON
local function read_file(path)
    local file = io.open(path, "r")
    if not file then
        return nil
    end
    local content = file:read("*all")
    file:close()
    return content
end

-- Convert Wesnoth unit to our JSON format
local function serialize_unit(unit)
    local attacks = {}
    for i, attack in ipairs(unit.attacks) do
        local specials = {}
        for special in attack:iter_specials() do
            table.insert(specials, special.id)
        end

        table.insert(attacks, {
            name = attack.name,
            type = attack.type,
            damage = attack.damage,
            strikes = attack.number,
            is_ranged = attack.range == "ranged",
            specials = specials
        })
    end

    local abilities = {}
    for ability in unit:iter_abilities() do
        table.insert(abilities, ability.id)
    end

    local traits = {}
    for trait in unit:iter_traits() do
        table.insert(traits, trait.id)
    end

    local resistances = {}
    for _, damage_type in ipairs({"blade", "pierce", "impact", "fire", "cold", "arcane"}) do
        resistances[damage_type] = unit:resistance_against(damage_type)
    end

    local defenses = {}
    for _, terrain in ipairs(wesnoth.terrain_types()) do
        defenses[terrain.id] = unit:defense_on(terrain.id)
    end

    local movement_costs = {}
    for _, terrain in ipairs(wesnoth.terrain_types()) do
        movement_costs[terrain.id] = unit:movement_on(terrain.id)
    end

    return {
        name = unit.type,
        side = unit.side,
        is_leader = unit.canrecruit,
        x = unit.x,
        y = unit.y,
        max_hp = unit.max_hitpoints,
        max_moves = unit.max_moves,
        max_exp = unit.max_experience,
        cost = wesnoth.unit_types[unit.type].cost,
        alignment = unit.alignment,
        levelup_names = unit.advances_to or {},
        current_hp = unit.hitpoints,
        current_moves = unit.moves,
        current_exp = unit.experience,
        has_attacked = unit.attacks_left == 0,
        attacks = attacks,
        resistances = resistances,
        defenses = defenses,
        movement_costs = movement_costs,
        abilities = abilities,
        traits = traits,
        statuses = {
            poisoned = unit.status.poisoned,
            slowed = unit.status.slowed,
            petrified = unit.status.petrified
        }
    }
end

-- Serialize current game state
local function serialize_game_state(side_number)
    local state = {
        game_id = game_id,
        turn = wesnoth.current.turn,
        side = side_number,
        gold = wesnoth.sides[side_number].gold,
        game_over = false,
        winner = nil,
        map = {
            width = wesnoth.current.map.playable_width,
            height = wesnoth.current.map.playable_height,
            mask = {},
            fog = {},
            hexes = {},
            units = {}
        },
        recruits = {}
    }

    -- Serialize map hexes
    for x = 1, state.map.width do
        for y = 1, state.map.height do
            local terrain = wesnoth.get_terrain(x, y)
            local terrain_info = wesnoth.get_terrain_info(terrain)

            -- Check if fogged
            if wesnoth.is_fogged(side_number, {x, y}) then
                table.insert(state.map.fog, {x = x, y = y})
            else
                local hex = {
                    x = x,
                    y = y,
                    terrain_types = {terrain_info.id},
                    modifiers = {}
                }

                -- Add terrain modifiers if any
                if terrain_info.light_modification then
                    if terrain_info.light_modification > 0 then
                        table.insert(hex.modifiers, "ILLUMINATED")
                    elseif terrain_info.light_modification < 0 then
                        table.insert(hex.modifiers, "SHADOWED")
                    end
                end

                table.insert(state.map.hexes, hex)
            end
        end
    end

    -- Serialize visible units
    for i, unit in ipairs(wesnoth.get_units{}) do
        if wesnoth.is_visible_to_side(unit.x, unit.y, side_number) then
            table.insert(state.map.units, serialize_unit(unit))
        end
    end

    -- Serialize recruitment options
    local side = wesnoth.sides[side_number]
    for _, unit_type_name in ipairs(side.recruit) do
        local unit_type = wesnoth.unit_types[unit_type_name]
        if unit_type then
            local partial_unit = {
                name = unit_type_name,
                hp = unit_type.max_hitpoints,
                moves = unit_type.max_moves,
                exp = unit_type.max_experience,
                cost = unit_type.cost,
                alignment = unit_type.alignment,
                levelup_names = unit_type.advances_to or {},
                attacks = {},
                resistances = {},
                defenses = {},
                movement_costs = {},
                abilities = {},
                traits = {}
            }
            -- TODO: Add attack, resistance, defense, movement info from unit_type
            table.insert(state.recruits, partial_unit)
        end
    end

    return state
end

-- Main AI evaluation function
function ai_python:evaluation(cfg, data)
    local side_number = wesnoth.current.side

    -- Serialize current game state
    local state = serialize_game_state(side_number)
    local state_json = wesnoth.json_encode(state)

    -- Write state to file
    if not write_file(state_file, state_json) then
        wesnoth.log("error", "Failed to write state file")
        return
    end

    -- Create signal file to notify Python AI
    write_file(signal_file, "")

    -- Wait for action file (with timeout)
    local max_wait = 30 -- 30 seconds timeout
    local wait_time = 0
    local action_json = nil

    while wait_time < max_wait do
        action_json = read_file(action_file)
        if action_json then
            -- Delete action file after reading
            os.remove(action_file)
            break
        end
        wesnoth.delay(100) -- Wait 100ms
        wait_time = wait_time + 0.1
    end

    if not action_json then
        wesnoth.log("error", "Timeout waiting for AI action")
        return
    end

    -- Parse action
    local action = wesnoth.json_decode(action_json)

    -- Execute action based on type
    if action.type == "move" then
        local unit = wesnoth.get_unit(action.start_x, action.start_y)
        if unit then
            wesnoth.extract_unit(unit)
            unit.x = action.target_x
            unit.y = action.target_y
            wesnoth.put_unit(unit)
        end
    elseif action.type == "attack" then
        local attacker = wesnoth.get_unit(action.start_x, action.start_y)
        local defender = wesnoth.get_unit(action.target_x, action.target_y)
        if attacker and defender then
            wesnoth.simulate_combat(attacker, action.attack_index, defender)
        end
    elseif action.type == "recruit" then
        local leader = wesnoth.get_units{side = side_number, canrecruit = true}[1]
        if leader then
            wesnoth.put_unit(action.target_x, action.target_y, {
                type = action.unit_type,
                side = side_number
            })
            wesnoth.sides[side_number].gold = wesnoth.sides[side_number].gold -
                wesnoth.unit_types[action.unit_type].cost
        end
    elseif action.type == "end_turn" then
        -- Do nothing, let turn end naturally
    end
end

return ai_python
