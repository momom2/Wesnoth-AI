-- oracle_setup.lua
-- Builds a scripted position for the hidden-units oracle
-- (tools/hidden_units_oracle.py) at prestart of the `ai_oracle` test
-- scenario: terrain edits, units, fog and the time of day come from
-- ~add-ons/wesnoth_ai/games/oracle/setup.lua, which Python writes
-- before launching Wesnoth. Everything is applied through the engine's
-- own API, so the position the Python driver then probes (moves through
-- the AI stage, visibility through [filter_vision]) is the engine's.

local M = {}

local SETUP_PATH = "~add-ons/wesnoth_ai/games/oracle/setup.lua"

local function read_setup()
    local src = wesnoth.read_file(SETUP_PATH)
    if not src or src == "" then
        std_print("[oracle] no setup file at " .. SETUP_PATH)
        return nil
    end
    local chunk, err = load(src, "oracle_setup", "t", {})
    if not chunk then
        std_print("[oracle] setup does not parse: " .. tostring(err))
        return nil
    end
    local ok, setup = pcall(chunk)
    if not ok or type(setup) ~= "table" then
        std_print("[oracle] setup did not return a table: " .. tostring(setup))
        return nil
    end
    return setup
end

local function set_terrain(x, y, code)
    local ok, err = pcall(function() wesnoth.current.map[{ x, y }] = code end)
    if not ok then
        std_print(string.format("[oracle] terrain %d,%d=%s failed: %s", x, y, code, tostring(err)))
    end
end

-- The default schedule's six times (data/core/macros/schedules.cfg
-- DAWN .. SECOND_WATCH), passed explicitly: at prestart
-- wesnoth.schedule.times is empty, so [replace_schedule] must carry
-- its own [time] children. `index` is 0-based (dawn=0 .. second_watch=5).
local DEFAULT_TIMES = {
    { id = "dawn", name = "Dawn", image = "misc/time-schedules/default/schedule-dawn.png",
      lawful_bonus = 0, red = -25, green = -15, blue = 0 },
    { id = "morning", name = "Morning", image = "misc/time-schedules/default/schedule-morning.png",
      lawful_bonus = 25 },
    { id = "afternoon", name = "Afternoon", image = "misc/time-schedules/default/schedule-afternoon.png",
      lawful_bonus = 25 },
    { id = "dusk", name = "Dusk", image = "misc/time-schedules/default/schedule-dusk.png",
      lawful_bonus = 0, red = 10, green = -20, blue = -35 },
    { id = "first_watch", name = "First Watch", image = "misc/time-schedules/default/schedule-firstwatch.png",
      lawful_bonus = -25, red = -75, green = -45, blue = -13 },
    { id = "second_watch", name = "Second Watch", image = "misc/time-schedules/default/schedule-secondwatch.png",
      lawful_bonus = -25, red = -75, green = -45, blue = -13 },
}

local function current_tod_id()
    local ok, tod = pcall(function() return wesnoth.schedule.get_time_of_day(nil, wesnoth.current.turn) end)
    if ok and type(tod) == "table" then return tostring(tod.id) end
    local ok2, tod2 = pcall(function() return wesnoth.current.schedule.time_of_day end)
    if ok2 and type(tod2) == "table" then return tostring(tod2.id) end
    if ok2 and type(tod2) == "string" then return tod2 end
    return "?"
end

local function set_time_of_day(index)
    local cfg = { current_time = index }
    for _, t in ipairs(DEFAULT_TIMES) do
        local copy = {}
        for k, v in pairs(t) do copy[k] = v end
        table.insert(cfg, { "time", copy })
    end
    local ok, err = pcall(function() wesnoth.schedule.replace(cfg) end)
    if not ok then
        std_print("[oracle] schedule.replace failed: " .. tostring(err))
        local ok2, err2 = pcall(function() wesnoth.current.schedule.time_of_day = DEFAULT_TIMES[index + 1].id end)
        if not ok2 then
            std_print("[oracle] time_of_day assignment failed too: " .. tostring(err2))
        end
    end
    std_print("[oracle] time of day requested " .. tostring(DEFAULT_TIMES[index + 1].id)
        .. ", engine reads " .. current_tod_id())
end

function M.apply()
    local setup = read_setup()
    if not setup then return end
    wml.variables.oracle_mode = true
    for _, t in ipairs(setup.terrain or {}) do
        set_terrain(t.x, t.y, t.code)
    end
    for _, u in ipairs(setup.units or {}) do
        local cfg = {
            type = u.type, side = u.side, id = u.id, name = u.id,
            canrecruit = u.canrecruit and true or false,
            random_traits = false, random_gender = false,
            generate_name = false,
        }
        local ok, err = pcall(function() wesnoth.units.to_map(cfg, u.x, u.y) end)
        if not ok then
            std_print(string.format("[oracle] unit %s at %d,%d failed: %s", u.id, u.x, u.y, tostring(err)))
        end
    end
    if setup.fog then
        for side_number, on in pairs(setup.fog) do
            local ok, err = pcall(function()
                wesnoth.sides[tonumber(side_number)].fog = on and true or false
                wesnoth.sides[tonumber(side_number)].shroud = false
            end)
            if not ok then
                std_print("[oracle] fog on side " .. tostring(side_number) .. " failed: " .. tostring(err))
            end
        end
    end
    if setup.tod_index ~= nil then
        set_time_of_day(setup.tod_index)
    end
    std_print(string.format("[oracle] setup applied: %d terrain edits, %d units, tod=%s",
        #(setup.terrain or {}), #(setup.units or {}), tostring(setup.tod_index)))
end

return M
