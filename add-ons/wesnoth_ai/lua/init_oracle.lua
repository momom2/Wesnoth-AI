-- init_oracle.lua
-- The engine's half of the scenario-init oracle
-- (tools/scenario_init_oracle.py). Installed on side 1 by
-- init_oracle_ai.cfg, so it runs at side 1's first turn: after
-- prestart, start and side 1's turn-1 init. It reports the whole board,
-- not what side 1 can see: every side's economy and settings, every
-- unit, every village owner, every terrain code and the lawful bonus of
-- every hex whose bonus differs from the global one. Then the turn ends.

local json = wesnoth.require("~add-ons/wesnoth_ai/lua/json_encoder.lua")

local FRAME_BEGIN = "===WESNOTH_AI_STATE_BEGIN==="
local FRAME_END   = "===WESNOTH_AI_STATE_END==="

local M = {}

local function trait_ids(unit)
    local ids = {}
    local mods = wml.get_child(unit.__cfg, "modifications")
    if mods then
        for _, trait in ipairs(wml.child_array(mods, "trait")) do
            table.insert(ids, trait.id or "")
        end
    end
    return ids
end

local function status_ids(unit)
    local ids = {}
    for key, on in pairs(wml.get_child(unit.__cfg, "status") or {}) do
        if on == true then table.insert(ids, key) end
    end
    table.sort(ids)
    return ids
end

local function collect_units()
    local units = {}
    for _, u in ipairs(wesnoth.units.find_on_map({})) do
        table.insert(units, {
            id = u.id, type = u.type, side = u.side, x = u.x, y = u.y,
            canrecruit = u.canrecruit,
            hitpoints = u.hitpoints, max_hitpoints = u.max_hitpoints,
            moves = u.moves, max_moves = u.max_moves,
            experience = u.experience, max_experience = u.max_experience,
            traits = trait_ids(u), status = status_ids(u),
        })
    end
    return units
end

local function collect_sides()
    local sides = {}
    for i = 1, #wesnoth.sides do
        local s = wesnoth.sides[i]
        table.insert(sides, {
            side = s.side, controller = s.controller,
            gold = s.gold, base_income = s.base_income, total_income = s.total_income,
            net_income = s.net_income, total_upkeep = s.total_upkeep,
            village_gold = s.village_gold, village_support = s.village_support,
            fog = s.fog, shroud = s.shroud,
            recruit = s.recruit or {}, faction = s.faction or "",
            num_villages = s.num_villages,
        })
    end
    return sides
end

-- Terrain codes row by row (rows[y][x], 1-based, playable area only),
-- village owners, and the hexes whose illuminated lawful bonus differs
-- from the global one (time areas, illuminated terrain).
local function collect_board(turn, global_bonus)
    local map = wesnoth.current.map
    local rows, owners, bonus = {}, {}, {}
    for y = 1, map.playable_height do
        local row = {}
        for x = 1, map.playable_width do
            row[x] = map[{ x, y }]
            local owner = wesnoth.map.get_owner({ x, y })
            if owner and owner ~= 0 then
                table.insert(owners, { x = x, y = y, side = owner })
            end
            local tod = wesnoth.schedule.get_illumination({ x, y }, turn)
            if tod.lawful_bonus ~= global_bonus then
                table.insert(bonus, { x = x, y = y, lawful_bonus = tod.lawful_bonus })
            end
        end
        rows[y] = row
    end
    return rows, owners, bonus
end

-- The experience modifier the game applies. mp_settings only holds the
-- host's parameter; a type's max_experience is (base * modifier + 50) /
-- 100 (src/units/types.cpp:577-589), so a type whose base is 100 reads
-- the applied modifier itself.
local EXPERIENCE_PROBE = "Dwarvish Berserker"

local function collect_settings()
    local mp = wesnoth.scenario.mp_settings
    local probe = wesnoth.unit_types[EXPERIENCE_PROBE]
    return {
        experience_probe = { type = EXPERIENCE_PROBE, base = probe.__cfg.experience,
                             applied = probe.max_experience },
        experience_modifier = mp.experience_modifier,
        village_gold = mp.mp_village_gold,
        village_support = mp.mp_village_support,
        fog = mp.mp_fog, shroud = mp.mp_shroud,
        random_start_time = mp.mp_random_start_time,
        use_map_settings = mp.mp_use_map_settings,
    }
end

function M:report()
    local turn = wesnoth.current.turn
    local tod = wesnoth.schedule.get_time_of_day(nil, turn)
    local rows, owners, bonus = collect_board(turn, tod.lawful_bonus)
    local record = {
        kind = "scenario_init",
        scenario_id = wesnoth.scenario.id,
        turns = wesnoth.scenario.turns,
        turn = turn,
        current_side = wesnoth.current.side,
        time_of_day = tod.id,
        lawful_bonus = tod.lawful_bonus,
        lawful_bonus_exceptions = bonus,
        settings = collect_settings(),
        sides = collect_sides(),
        units = collect_units(),
        village_owners = owners,
        terrain_rows = rows,
    }
    std_print(FRAME_BEGIN)
    std_print(json.encode(record))
    std_print(FRAME_END)
end

return M
