-- state_collector.lua
-- Helper functions to collect complete game state from Wesnoth
-- FIXED: Complete rewrite with proper syntax and error handling

local state_collector = {}

-- Collect attack information for a unit
function state_collector.collect_attack(attack)
    local specials = {}
    if attack.specials then
        for i = 1, #attack.specials do
            table.insert(specials, attack.specials[i][1])
        end
    end
    
    return {
        type = attack.type,
        strikes = attack.number,
        damage = attack.damage,
        is_ranged = (attack.range == "ranged"),
        specials = specials
    }
end

-- Collect complete unit information
function state_collector.collect_unit(unit)
    if not unit then 
        std_print("[ERROR] collect_unit called with nil unit")
        return nil 
    end
    
    std_print(string.format("[DEBUG] ===== Collecting unit: %s at (%d,%d) =====", 
        tostring(unit.type), tostring(unit.x), tostring(unit.y)))
    
    -- Collect attacks with error handling
    local attacks = {}
    local attack_status, attack_error = pcall(function()
        if unit.attacks then
            for i = 1, #unit.attacks do
                table.insert(attacks, state_collector.collect_attack(unit.attacks[i]))
            end
            std_print(string.format("[DEBUG]   Attacks: %d", #attacks))
        else
            std_print("[DEBUG]   No attacks")
        end
    end)
    if not attack_status then
        std_print(string.format("[ERROR] Failed to collect attacks: %s", attack_error))
    end
    
    -- Collect resistances with error handling
    local resistances = {}
    local resist_status, resist_error = pcall(function()
        if unit.resistance then
            resistances = {
                unit.resistance.blade or 0,
                unit.resistance.pierce or 0,
                unit.resistance.impact or 0,
                unit.resistance.fire or 0,
                unit.resistance.cold or 0,
                unit.resistance.arcane or 0
            }
            std_print("[DEBUG]   Resistances collected")
        else
            resistances = {0, 0, 0, 0, 0, 0}
            std_print("[DEBUG]   No resistance data, using defaults")
        end
    end)
    if not resist_status then
        std_print(string.format("[ERROR] Failed to collect resistances: %s", resist_error))
        resistances = {0, 0, 0, 0, 0, 0}
    end
    
    -- Collect defense values
    local defenses = {}
    local defense_keys = {
        "castle", "cave", "deep_water", "flat", "forest", "frozen",
        "fungus", "hills", "mountains", "reef", "sand", "shallow_water",
        "swamp", "unwalkable", "village", "impassable"
    }
    
    local defense_status, defense_error = pcall(function()
        if unit.defense then
            for _, terrain in ipairs(defense_keys) do
                defenses[terrain] = unit.defense[terrain] or 100
            end
            std_print("[DEBUG]   Defenses collected")
        else
            for _, terrain in ipairs(defense_keys) do
                defenses[terrain] = 100
            end
            std_print("[DEBUG]   No defense data, using defaults")
        end
    end)
    if not defense_status then
        std_print(string.format("[ERROR] Failed to collect defenses: %s", defense_error))
        for _, terrain in ipairs(defense_keys) do
            defenses[terrain] = 100
        end
    end
    
    -- Collect movement costs - use defaults only
    std_print("[DEBUG]   Using default movement costs (99 for all terrain)")
    local movement_costs = {}
    for _, terrain in ipairs(defense_keys) do
        movement_costs[terrain] = 99
    end
    
    -- Collect abilities
    local abilities = {}
    local ability_status, ability_error = pcall(function()
        if unit.abilities then
            for i = 1, #unit.abilities do
                table.insert(abilities, unit.abilities[i][1])
            end
            std_print(string.format("[DEBUG]   Abilities: %d", #abilities))
        else
            std_print("[DEBUG]   No abilities")
        end
    end)
    if not ability_status then
        std_print(string.format("[ERROR] Failed to collect abilities: %s", ability_error))
    end
    
    -- Collect traits
    local traits = {}
    local trait_status, trait_error = pcall(function()
        if unit.traits then
            for i = 1, #unit.traits do
                table.insert(traits, unit.traits[i].name)
            end
            std_print(string.format("[DEBUG]   Traits: %d", #traits))
        else
            std_print("[DEBUG]   No traits")
        end
    end)
    if not trait_status then
        std_print(string.format("[ERROR] Failed to collect traits: %s", trait_error))
    end
    
    -- Collect status
    local status = {}
    local status_status, status_error = pcall(function()
        if unit.status then
            if unit.status.poisoned then table.insert(status, "poisoned") end
            if unit.status.slowed then table.insert(status, "slowed") end
            if unit.status.petrified then table.insert(status, "petrified") end
            std_print(string.format("[DEBUG]   Status effects: %d", #status))
        else
            std_print("[DEBUG]   No status")
        end
    end)
    if not status_status then
        std_print(string.format("[ERROR] Failed to collect status: %s", status_error))
    end
    
    std_print("[DEBUG] ===== Unit collection complete =====")
    
    -- Build and return unit data
    return {
        id = unit.id or "",
        name = unit.type,
        side = unit.side,
        is_leader = unit.canrecruit,
        x = unit.x,
        y = unit.y,
        max_hp = unit.max_hitpoints,
        max_moves = unit.max_moves,
        max_exp = unit.max_experience,
        cost = unit.cost or 0,
        alignment = unit.alignment,
        levelup_names = unit.advances_to or {},
        current_hp = unit.hitpoints,
        current_moves = unit.moves,
        current_exp = unit.experience,
        has_attacked = (unit.attacks_left == 0),
        attacks = attacks,
        resistances = resistances,
        defenses = defenses,
        movement_costs = movement_costs,
        abilities = abilities,
        traits = traits,
        statuses = status
    }
end

-- Collect terrain information for a hex
function state_collector.collect_terrain(x, y, map_obj)
    local terrain_code = map_obj[{x, y}]
    local base, overlay = "", ""
    local caret_pos = terrain_code:find("%^")
    if caret_pos then
        base = terrain_code:sub(1, caret_pos - 1)
        overlay = terrain_code:sub(caret_pos + 1)
    else
        base = terrain_code
        overlay = ""
    end
    
    local terrain_types = {base}
    if overlay ~= "" then
        table.insert(terrain_types, overlay)
    end
    
    local modifiers = {}
    local village_owner = wesnoth.map.get_owner({x, y})
    if village_owner and village_owner ~= 0 then
        table.insert(modifiers, "village")
    end
    if terrain_code:find("K") then
        table.insert(modifiers, "keep")
    end
    if terrain_code:find("C") then
        table.insert(modifiers, "castle")
    end
    
    return {
        x = x,
        y = y,
        terrain_types = terrain_types,
        modifiers = modifiers,
        full_code = terrain_code
    }
end

-- Collect complete game state
function state_collector.collect_game_state(side_number, game_id)
    std_print(string.format("[DEBUG] Starting game state collection for side %d", side_number))
    
    local map_obj = wesnoth.current.map
    local width = map_obj.playable_width
    local height = map_obj.playable_height
    
    std_print(string.format("[DEBUG] Map size: %dx%d", width, height))
    
    -- Collect hexes
    local hexes = {}
    local hex_count = 0
    for y = 1, height do
        for x = 1, width do
            if wesnoth.current.map:on_board({x, y}) then
                table.insert(hexes, state_collector.collect_terrain(x, y, map_obj))
                hex_count = hex_count + 1
            end
        end
    end
    std_print(string.format("[DEBUG] Collected %d hexes", hex_count))
    
    -- Collect units
    local units = {}
    local all_units = wesnoth.units.find_on_map({})
    std_print(string.format("[DEBUG] Found %d total units on map", #all_units))
    
    for _, unit in ipairs(all_units) do
        if not wesnoth.sides.is_fogged(side_number, unit.x, unit.y) then
            local unit_data = state_collector.collect_unit(unit)
            if unit_data then
                table.insert(units, unit_data)
            end
        end
    end
    std_print(string.format("[DEBUG] Collected %d visible units", #units))
    
    -- Collect fog
    local fog = {}
    for y = 1, height do
        for x = 1, width do
            if wesnoth.current.map:on_board({x, y}) and 
               wesnoth.sides.is_fogged(side_number, x, y) then
                table.insert(fog, {x = x, y = y})
            end
        end
    end
    std_print(string.format("[DEBUG] Collected %d fogged hexes", #fog))
    
    -- Collect mask
    local mask = {}
    for y = 1, height do
        for x = 1, width do
            if not wesnoth.current.map:on_board({x, y}) then
                table.insert(mask, {x = x, y = y})
            end
        end
    end
    std_print(string.format("[DEBUG] Collected %d masked hexes", #mask))
    
    -- Collect sides
    local sides_info = {}
    for i = 1, #wesnoth.sides do
        local side = wesnoth.sides[i]
        local villages = {}
        for y = 1, height do
            for x = 1, width do
                local owner = wesnoth.map.get_owner({x, y})
                if owner == i then
                    table.insert(villages, {x = x, y = y})
                end
            end
        end
        
        sides_info[i] = {
            gold = side.gold,
            village_gold = side.village_gold,
            village_support = side.village_support or 1,
            base_income = side.base_income or 0,
            recruits = side.recruit or {},
            num_villages = #villages
        }
        std_print(string.format("[DEBUG] Side %d: gold=%d, villages=%d", i, side.gold, #villages))
    end
    
    local tod = wesnoth.current.schedule[wesnoth.current.schedule.id]
    std_print("[DEBUG] Game state collection completed successfully")
    
    return {
        game_id = game_id,
        current_side = side_number,
        turn_number = wesnoth.current.turn,
        time_of_day = tod.id or "morning",
        map = {
            width = width,
            height = height,
            hexes = hexes,
            units = units,
            fog = fog,
            mask = mask
        },
        sides = sides_info,
        game_over = false
    }
end

return state_collector
