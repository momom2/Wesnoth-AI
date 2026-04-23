-- ca_turn_loop.lua
-- Single candidate action that drives one state→action cycle per AI
-- iteration. Replaces the old state_sender + action_executor pair.
--
-- Why one CA instead of two: Wesnoth blacklists a CA after it executes,
-- so a "state_sender" CA only runs ONCE per side-turn. That meant only
-- the first action per turn got fresh state; subsequent actions ran
-- against stale state, or the pipeline deadlocked when Python waited
-- for state that would never come.
--
-- Lifecycle:
--   evaluation():
--     * returns 0 if wml.variables.turn_done_flag is set (Python asked
--       us to end the turn) — Wesnoth's AI loop exits, the turn ends
--     * otherwise returns the CA score so execution() fires
--   execution():
--     * emits a JSON-framed state block to std_print (→ Wesnoth's
--       .out.log, which Python tails)
--     * waits up to ACTION_TIMEOUT_MS for a fresh action file to
--       appear (fresh = seq > wml.variables.last_action_seq)
--     * if action.type == "end_turn", sets turn_done_flag and returns
--     * otherwise dispatches the action and returns (the AI loop will
--       call evaluation() again for the next action)
--
-- The turn_done_flag is cleared at the start of each side's turn by
-- the "turn refresh" event in training_scenario.cfg.

local state_collector = wesnoth.require("~add-ons/wesnoth_ai/lua/state_collector.lua")
local action_handler  = wesnoth.require("~add-ons/wesnoth_ai/lua/action_executor.lua")
local json            = wesnoth.require("~add-ons/wesnoth_ai/lua/json_encoder.lua")

local CA = {}

local FRAME_BEGIN = "===WESNOTH_AI_STATE_BEGIN==="
local FRAME_END   = "===WESNOTH_AI_STATE_END==="

local POLL_MS            = 50       -- how often to check for action.lua
local ACTION_TIMEOUT_MS  = 30000    -- total wait before we bail

local function action_path(game_id)
    return "~add-ons/wesnoth_ai/games/" .. game_id .. "/action.lua"
end

local function leaders_alive()
    local s1, s2 = false, false
    for _, leader in ipairs(wesnoth.units.find_on_map({ canrecruit = true })) do
        if leader.side == 1 then s1 = true end
        if leader.side == 2 then s2 = true end
    end
    return s1, s2
end

local function emit_state(game_id, turn, side_number)
    local state = state_collector.collect_game_state(side_number, game_id)

    -- Detect game over inline so Python gets terminal-state signal.
    local s1, s2 = leaders_alive()
    if not (s1 and s2) then
        state.game_over = true
        if not s1 and not s2 then
            state.winner = 0
        elseif not s1 then
            state.winner = 2
        else
            state.winner = 1
        end
    end

    std_print(FRAME_BEGIN)
    std_print(string.format("meta: game_id=%s turn=%d side=%d",
        game_id, turn, side_number))
    std_print(json.encode(state))
    std_print(FRAME_END)
end

-- Read the action file; return (action_table, nil) on a fresh action,
-- (nil, "reason") otherwise.
local function try_read_action(path)
    if not wesnoth.have_file(path) then
        return nil, "missing"
    end
    local src = wesnoth.read_file(path)
    if not src or src == "" then
        return nil, "empty"
    end

    local chunk, err = load(src, path, "t")
    if not chunk then
        return nil, "parse: " .. tostring(err)
    end
    local ok, result = pcall(chunk)
    if not ok then
        return nil, "exec: " .. tostring(result)
    end
    if type(result) ~= "table" or type(result.seq) ~= "number" then
        return nil, "malformed"
    end

    local last = wml.variables.last_action_seq or 0
    if result.seq <= last then
        return nil, "stale"
    end
    return result, nil
end

local function wait_for_action(game_id)
    local path = action_path(game_id)
    local deadline = wesnoth.get_time_stamp() + ACTION_TIMEOUT_MS
    while wesnoth.get_time_stamp() < deadline do
        local action, _ = try_read_action(path)
        if action then
            wml.variables.last_action_seq = action.seq
            return action
        end
        wesnoth.interface.delay(POLL_MS)
    end
    return nil
end

function CA:evaluation()
    if wml.variables.turn_done_flag then
        return 0
    end
    return 999990
end

function CA:execution(cfg, data)
    local side_number = wesnoth.current.side
    local game_id = wml.variables.game_id or "game_0"
    local turn = wesnoth.current.turn

    emit_state(game_id, turn, side_number)

    local action = wait_for_action(game_id)
    if not action then
        std_print(string.format(
            "[turn-loop] action timeout on turn %d side %d — ending turn",
            turn, side_number))
        wml.variables.turn_done_flag = true
        return
    end

    if action.type == "end_turn" then
        wml.variables.turn_done_flag = true
        return
    end

    local result = action_handler.execute_action(action)
    if not result.success then
        std_print(string.format(
            "[turn-loop] turn %d side %d action=%s rejected: %s",
            turn, side_number, tostring(action.type), tostring(result.error)))
    end
end

return CA
