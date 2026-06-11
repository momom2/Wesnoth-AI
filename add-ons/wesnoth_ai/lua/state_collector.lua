-- state_collector.lua
-- Build a per-turn game-state table that the Python side will parse.
--
-- Runs inside Wesnoth's sandboxed Lua (no io, no os.remove, etc.) — the
-- caller (ca_state_sender) emits the serialized result via std_print,
-- which reaches <userdata>/logs/wesnoth-*.out.log where Python tails it.

local state_collector = {}

-- Turn on when chasing a specific bug; leaves [DEBUG] traces in .out.log.
local DEBUG = false
local function dbg(s) if DEBUG then std_print(s) end end

local DEFENSE_TERRAINS = {
    "castle", "cave", "deep_water", "flat", "forest", "frozen",
    "fungus", "hills", "mountains", "reef", "sand", "shallow_water",
    "swamp", "unwalkable", "village", "impassable",
}

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
        specials = specials,
    }
end

function state_collector.collect_unit(unit)
    if not unit then return nil end

    -- Attacks.
    local attacks = {}
    pcall(function()
        if unit.attacks then
            for i = 1, #unit.attacks do
                table.insert(attacks, state_collector.collect_attack(unit.attacks[i]))
            end
        end
    end)

    -- Resistances (as an ordered list; order must match Python's DamageType enum).
    local resistances = { 0, 0, 0, 0, 0, 0 }
    pcall(function()
        if unit.resistance then
            resistances = {
                unit.resistance.blade or 0,
                unit.resistance.pierce or 0,
                unit.resistance.impact or 0,
                unit.resistance.fire or 0,
                unit.resistance.cold or 0,
                unit.resistance.arcane or 0,
            }
        end
    end)

    -- Defense per terrain. Initialize with defaults FIRST so the table
    -- has string keys even if the pcall below bails — an empty Lua table
    -- would JSON-encode as "[]" and break Python's dict lookups.
    local defenses = {}
    for _, t in ipairs(DEFENSE_TERRAINS) do defenses[t] = 100 end
    pcall(function()
        if unit.defense then
            for _, t in ipairs(DEFENSE_TERRAINS) do
                defenses[t] = unit.defense[t] or 100
            end
        end
    end)

    -- Movement costs: same defense-first pattern. unit.movement isn't
    -- reliably exposed in 1.18; Phase 2 can pull real per-terrain costs.
    local movement_costs = {}
    for _, t in ipairs(DEFENSE_TERRAINS) do movement_costs[t] = 99 end

    -- Abilities / traits / statuses.
    local abilities = {}
    pcall(function()
        if unit.abilities then
            for i = 1, #unit.abilities do
                table.insert(abilities, unit.abilities[i][1])
            end
        end
    end)

    local traits = {}
    pcall(function()
        if unit.traits then
            for i = 1, #unit.traits do
                table.insert(traits, unit.traits[i].name)
            end
        end
    end)

    local status = {}
    pcall(function()
        if unit.status then
            if unit.status.poisoned then table.insert(status, "poisoned") end
            if unit.status.slowed then table.insert(status, "slowed") end
            if unit.status.petrified then table.insert(status, "petrified") end
        end
    end)

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
        statuses = status,
    }
end

function state_collector.collect_terrain(x, y, map_obj)
    local terrain_code = map_obj[{ x, y }]
    local base, overlay = terrain_code, ""
    local caret = terrain_code:find("%^")
    if caret then
        base = terrain_code:sub(1, caret - 1)
        overlay = terrain_code:sub(caret + 1)
    end

    local terrain_types = { base }
    if overlay ~= "" then table.insert(terrain_types, overlay) end

    -- Only STATIC modifiers here; village-ownership (dynamic) is emitted
    -- separately per frame as `villages_owned` so we can skip hex
    -- re-collection on delta frames without losing capture/loss signals.
    local modifiers = {}
    if terrain_code:find("K") then table.insert(modifiers, "keep") end
    if terrain_code:find("C") then table.insert(modifiers, "castle") end

    return {
        x = x,
        y = y,
        terrain_types = terrain_types,
        modifiers = modifiers,
        full_code = terrain_code,
    }
end

-- Safely read the time-of-day id. The 1.18 Lua API changed here and the
-- old `wesnoth.current.schedule[wesnoth.current.schedule.id]` pattern
-- dereferences nil on some installs. Fall back to "morning" if anything
-- fails — the AI doesn't depend on ToD correctness yet.
local function safe_time_of_day()
    local tod_id = "morning"
    pcall(function()
        local sched = wesnoth.current.schedule
        if type(sched) == "table" and sched.id then
            tod_id = sched.id
        end
    end)
    return tod_id
end

-- `include_map` toggles hex+mask emission. On the first frame of a
-- Wesnoth process we send the static terrain data in full; on every
-- subsequent frame we skip it (hexes/mask shape doesn't change across
-- a Caves-of-the-Basilisk match, so re-sending is pure waste). Python
-- reconstructs each frame's hex set from cached terrain + the dynamic
-- villages_owned map below.
function state_collector.collect_game_state(side_number, game_id, include_map)
    local map_obj = wesnoth.current.map
    local width = map_obj.playable_width
    local height = map_obj.playable_height
    dbg(string.format("[DEBUG] Map %dx%d include_map=%s",
        width, height, tostring(include_map)))

    -- Hexes + off-board mask: only on the first (full) frame.
    -- Each hex visit is roughly:
    --   map_obj[{x,y}]   + find("%^") + find("K") + find("C")
    -- which sums to ~5-15 µs × 1700 hexes = ~20 ms. Skipping that on
    -- every delta frame is the main win of this protocol change.
    local hexes = nil
    local mask = nil
    if include_map then
        hexes = {}
        mask = {}
        for y = 1, height do
            for x = 1, width do
                if wesnoth.current.map:on_board({ x, y }) then
                    table.insert(hexes, state_collector.collect_terrain(x, y, map_obj))
                else
                    table.insert(mask, { x = x, y = y })
                end
            end
        end
    end

    -- Units visible to the current side.
    local units = {}
    local all_units = wesnoth.units.find_on_map({})
    for _, unit in ipairs(all_units) do
        if not wesnoth.sides.is_fogged(side_number, unit.x, unit.y) then
            local u = state_collector.collect_unit(unit)
            if u then table.insert(units, u) end
        end
    end

    -- Fog + village ownership: both dynamic, both emitted every frame.
    -- Collecting them in a single map-scan lets us count per-side
    -- villages at the same time — saves a 2×1700-call loop we used
    -- to run for that count alone.
    local fog = {}
    local villages_owned = {}
    local villages_count = { 0, 0, 0, 0, 0, 0, 0, 0 }  -- up to 8 sides
    for y = 1, height do
        for x = 1, width do
            if wesnoth.current.map:on_board({ x, y }) then
                if wesnoth.sides.is_fogged(side_number, x, y) then
                    table.insert(fog, { x = x, y = y })
                end
                local owner = wesnoth.map.get_owner({ x, y })
                if owner and owner ~= 0 then
                    villages_owned[tostring(x) .. "," .. tostring(y)] = owner
                    villages_count[owner] = (villages_count[owner] or 0) + 1
                end
            end
        end
    end

    -- Per-side info.
    local sides_info = {}
    for i = 1, #wesnoth.sides do
        local side = wesnoth.sides[i]
        -- Faction name for encoder conditioning. The attribute sits on
        -- the side's underlying WML config; some Wesnoth builds expose
        -- it as `side.faction`, others only via `side.__cfg.faction`.
        -- Wrap both paths in pcall and default to "" if neither works
        -- (encoder tolerates empty via its "" → id 0 reserved slot).
        local faction = ""
        pcall(function()
            if side.faction then faction = side.faction
            elseif side.__cfg and side.__cfg.faction then
                faction = side.__cfg.faction
            end
        end)
        sides_info[i] = {
            gold = side.gold,
            village_gold = side.village_gold,
            village_support = side.village_support or 1,
            base_income = side.base_income or 0,
            recruits = side.recruit or {},
            num_villages = villages_count[i] or 0,
            faction = faction,
        }
    end

    return {
        game_id = game_id,
        current_side = side_number,
        turn_number = wesnoth.current.turn,
        time_of_day = safe_time_of_day(),
        frame_type = include_map and "full" or "delta",
        map = {
            width = width,
            height = height,
            hexes = hexes,             -- nil on delta frames
            units = units,
            fog = fog,
            mask = mask,               -- nil on delta frames
            villages_owned = villages_owned,
        },
        sides = sides_info,
        game_over = false,
    }
end

return state_collector
