-- ca_action_executor.lua
-- Candidate Action: wait for Python to deliver an action, execute it.
--
-- Protocol: Python atomically writes "<game_dir>/action.lua". This CA
-- polls for that file in its evaluation pass, and when present, consumes
-- it (read → delete → execute) in the execution pass. Deleting ensures
-- the next evaluation waits for a *fresh* action rather than re-running
-- a stale one.
--
-- Action file format: a Lua chunk that `return`s an action table, e.g.
--     return { type = "move", start_x = 5, start_y = 7, target_x = 6, target_y = 7 }
--
-- On timeout, we return 0 from evaluation, which hands control back to
-- Wesnoth's AI scheduler — effectively ending the side's turn.

local action_executor = wesnoth.require("~add-ons/wesnoth_ai/lua/action_executor.lua")

local ca_action_executor = {}

local ACTION_POLL_MS     = 50      -- how often we poll for a fresh action
local ACTION_TIMEOUT_MS  = 30000   -- total time we'll wait before giving up

local function action_path(game_id)
    -- Wesnoth CWD is <userdata>/data/; relative paths work via the junction.
    return "add-ons/wesnoth_ai/games/" .. game_id .. "/action.lua"
end

local function read_action(path)
    -- Read the chunk ourselves (rather than wesnoth.dofile) so we can
    -- delete the file before executing — avoids re-reading a stale one
    -- on the next turn if execution errors partway through.
    local fh = io.open(path, "r")
    if not fh then return nil, "action file missing" end
    local source = fh:read("*a")
    fh:close()

    local chunk, err = load(source, path, "t")
    if not chunk then return nil, "parse error: " .. tostring(err) end

    local ok, result = pcall(chunk)
    if not ok then return nil, "execution error: " .. tostring(result) end
    if type(result) ~= "table" then return nil, "action is not a table" end
    return result, nil
end

local function detect_game_over_and_end()
    local side1, side2 = false, false
    for _, leader in ipairs(wesnoth.units.find_on_map({ canrecruit = true })) do
        if leader.side == 1 then side1 = true end
        if leader.side == 2 then side2 = true end
    end
    if side1 and side2 then return end

    wesnoth.wml_actions.endlevel({
        result = "victory",
        bonus = false,
        carryover_percentage = 0,
    })
end

function ca_action_executor:evaluation()
    local game_id = wml.variables.game_id or "game_0"
    local path = action_path(game_id)

    local start = wesnoth.get_time_stamp()
    local deadline = start + ACTION_TIMEOUT_MS

    while wesnoth.get_time_stamp() < deadline do
        -- have_file() accepts ~add-ons/ prefix; we also use a plain io.open
        -- probe as a belt-and-braces check in case have_file caches results.
        local probe = io.open(path, "r")
        if probe then
            probe:close()
            return 999980
        end
        wesnoth.interface.delay(ACTION_POLL_MS)
    end

    std_print(string.format("[Turn %d, Side %d] Action timeout after %dms",
        wesnoth.current.turn, wesnoth.current.side, ACTION_TIMEOUT_MS))
    return 0
end

function ca_action_executor:execution(cfg, data)
    local side_number = wesnoth.current.side
    local game_id = wml.variables.game_id or "game_0"
    local path = action_path(game_id)

    local action, err = read_action(path)
    os.remove(path)  -- always consume so next turn waits for a fresh one

    if not action then
        std_print(string.format("[Turn %d, Side %d] ERROR reading action: %s",
            wesnoth.current.turn, side_number, tostring(err)))
        return
    end

    std_print(string.format("[Turn %d, Side %d] Executing: %s",
        wesnoth.current.turn, side_number, tostring(action.type)))

    if action.type == "end_turn" then
        return  -- falling through returns control; no explicit call needed
    end

    local result = action_executor.execute_action(action)
    if not result.success then
        std_print(string.format("[Turn %d, Side %d] Action failed: %s",
            wesnoth.current.turn, side_number, tostring(result.error)))
    end

    detect_game_over_and_end()
end

return ca_action_executor
