-- ca_action_executor.lua
-- Candidate Action: wait for a fresh action from Python and execute it.
--
-- Python-to-Lua transport: Python atomically writes a Lua chunk to
-- <addon>/games/<game_id>/action.lua containing
--     return { seq = <monotonic>, type = "move"|"attack"|..., ... }
-- Lua cannot delete the file (no io / os.remove in the sandbox), so we
-- distinguish fresh from stale via the `seq` field. `wml.variables.
-- last_action_seq` tracks the highest seq we've already executed.
--
-- evaluation() polls the file via wesnoth.read_file every 50ms. It
-- returns the CA score if it finds an action with seq > last_action_seq,
-- or 0 after a 30s timeout (which hands control back to Wesnoth and
-- ends the side's turn).

local action_handler = wesnoth.require("~add-ons/wesnoth_ai/lua/action_executor.lua")

local ca_action_executor = {}

local POLL_MS    = 50
local TIMEOUT_MS = 30000

local function action_path(game_id)
    return "~add-ons/wesnoth_ai/games/" .. game_id .. "/action.lua"
end

-- Read + parse the action file. Returns the action table on success,
-- or nil if the file is absent / malformed.
local function read_action(path)
    if not wesnoth.have_file(path) then return nil end
    local src = wesnoth.read_file(path)
    if not src or src == "" then return nil end

    local chunk, err = load(src, path, "t")
    if not chunk then
        std_print("[action-exec] parse error: " .. tostring(err))
        return nil
    end

    local ok, result = pcall(chunk)
    if not ok then
        std_print("[action-exec] chunk error: " .. tostring(result))
        return nil
    end
    if type(result) ~= "table" then
        std_print("[action-exec] non-table action")
        return nil
    end
    return result
end

-- Is this action "fresh" relative to our last-executed seq?
local function is_fresh(action)
    if type(action.seq) ~= "number" then
        std_print("[action-exec] action missing numeric .seq — treating as stale")
        return false
    end
    local last = wml.variables.last_action_seq or 0
    return action.seq > last
end

local function detect_game_over_and_end()
    local s1, s2 = false, false
    for _, leader in ipairs(wesnoth.units.find_on_map({ canrecruit = true })) do
        if leader.side == 1 then s1 = true end
        if leader.side == 2 then s2 = true end
    end
    if s1 and s2 then return end
    wesnoth.wml_actions.endlevel({
        result = "victory",
        bonus = false,
        carryover_percentage = 0,
    })
end

function ca_action_executor:evaluation()
    local game_id = wml.variables.game_id or "game_0"
    local path = action_path(game_id)

    local deadline = wesnoth.get_time_stamp() + TIMEOUT_MS
    while wesnoth.get_time_stamp() < deadline do
        local action = read_action(path)
        if action and is_fresh(action) then
            return 999980
        end
        wesnoth.interface.delay(POLL_MS)
    end

    std_print(string.format(
        "[action-exec] timeout after %dms on turn %d side %d",
        TIMEOUT_MS, wesnoth.current.turn, wesnoth.current.side))
    return 0
end

function ca_action_executor:execution(cfg, data)
    local game_id = wml.variables.game_id or "game_0"
    local action = read_action(action_path(game_id))

    if not action or not is_fresh(action) then
        -- Shouldn't happen — evaluation would have returned 0.
        std_print("[action-exec] execution reached with no fresh action")
        return
    end

    wml.variables.last_action_seq = action.seq

    if action.type == "end_turn" then
        return  -- returning ends our execution; Wesnoth proceeds to next CA/turn
    end

    local result = action_handler.execute_action(action)
    if not result.success then
        std_print(string.format(
            "[action-exec] turn %d side %d action=%s failed: %s",
            wesnoth.current.turn, wesnoth.current.side,
            tostring(action.type), tostring(result.error)))
    end

    detect_game_over_and_end()
end

return ca_action_executor
