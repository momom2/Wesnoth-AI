-- turn_stage.lua
-- Custom Wesnoth AI stage that drives one side's whole turn in a loop.
-- Not a candidate action — the key property we need here is that stages
-- are NOT subject to the RCA blacklist-on-no-state-change rule.
-- Rejected actions (invalid_move, invalid_recruit_location, …) therefore
-- don't end the turn; the loop just emits fresh state and asks Python
-- for another action.
--
-- Wiring: the [ai] block in ai_config.cfg installs this module as the
-- side's Lua engine and runs `(...):run_turn()` as its sole stage.
--
-- Per-turn protocol (same as Phase 1):
--   - Emit state frame via std_print → <userdata>/logs/wesnoth-*.out.log
--     (Python tails this file).
--   - Wait for `action.lua` with a fresh `seq` to appear in the game's
--     IPC dir. Python atomically writes it; we read via
--     wesnoth.read_file and track last-seen seq in wml.variables.
--   - Dispatch the action through action_executor (ai.move / ai.attack
--     / ai.recruit). Failures are logged but don't exit the loop.
--   - Loop until Python sends { type = "end_turn" } or we time out.

local state_collector = wesnoth.require("~add-ons/wesnoth_ai/lua/state_collector.lua")
local action_handler  = wesnoth.require("~add-ons/wesnoth_ai/lua/action_executor.lua")
local json            = wesnoth.require("~add-ons/wesnoth_ai/lua/json_encoder.lua")

-- Training-mode preference overrides. Executed once per Wesnoth process
-- at module-load time. These are what turn a single action from
-- ~150ms (combat animation + unit walk cycle + SDL_Delay) into
-- something closer to the engine's actual compute time.
--
-- Setters confirmed in 1.18's src/scripting/lua_preferences.cpp
-- (generic key/value bindings for src/preferences/general.cpp keys).
-- `--nodelay` on the command line covers the SDL_Delay short-circuit
-- but not the display-pump animations; these keys handle the rest.
--
-- Wrapped in pcall so a future Wesnoth that drops one of these keys
-- doesn't take the whole training run down — we'd just run slower.
local function set_training_prefs()
    local function try(k, v)
        pcall(function() wesnoth.preferences[k] = v end)
    end
    try("turbo", true)
    try("turbo_speed", 10.0)
    try("animate_map", false)
    try("animate_water", false)
    try("idle_anim", false)
    try("show_combat", false)
    try("scroll_to_action", false)
end
set_training_prefs()

local M = {}

local FRAME_BEGIN = "===WESNOTH_AI_STATE_BEGIN==="
local FRAME_END   = "===WESNOTH_AI_STATE_END==="

-- Lowered from 50ms after profiling: the 50ms poll added ~25ms avg
-- latency per action, which at ~5 actions/s × 4 games ≈ 20% of wall
-- time spent just waiting on the polling granularity. 10ms gives
-- ~5ms avg wait and is still well below the ~20ms policy-decide time,
-- so Lua never wastes cycles past what Python needs to produce an
-- action. Cost: Wesnoth's event loop runs ~5× as often (all idle
-- pumps), but that's negligible on CPU.
local POLL_MS    = 10       -- how often to check for a fresh action
local TIMEOUT_MS = 30000    -- how long we'll wait for Python per action

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

-- Track whether this Wesnoth process has already emitted a full map
-- frame. Module-level so it persists across all side-turns and all
-- actions within a single process. Each training game spawns a fresh
-- Wesnoth process (see wesnoth_interface.start_wesnoth), so the flag
-- resets naturally between games.
local full_frame_emitted = false

local function emit_state(game_id, turn, side_number)
    local include_map = not full_frame_emitted
    local state = state_collector.collect_game_state(side_number, game_id, include_map)
    full_frame_emitted = true
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

-- Returns (action_table, nil) on a fresh action; (nil, reason) otherwise.
local function try_read_action(path)
    if not wesnoth.have_file(path) then return nil, "missing" end
    local src = wesnoth.read_file(path)
    if not src or src == "" then return nil, "empty" end

    local chunk, err = load(src, path, "t")
    if not chunk then return nil, "parse: " .. tostring(err) end
    local ok, result = pcall(chunk)
    if not ok then return nil, "exec: " .. tostring(result) end
    if type(result) ~= "table" or type(result.seq) ~= "number" then
        return nil, "malformed"
    end

    local last = wml.variables.last_action_seq or 0
    if result.seq <= last then return nil, "stale" end
    return result, nil
end

local function wait_for_action(game_id)
    local path = action_path(game_id)
    local deadline = wesnoth.get_time_stamp() + TIMEOUT_MS
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

-- Called once per side-turn by the [stage] Lua wiring. Runs until
-- Python ends the turn, a timeout, or the game ends.
function M:run_turn()
    local game_id = wml.variables.game_id or "game_0"
    local turn = wesnoth.current.turn
    local side_number = wesnoth.current.side

    while true do
        emit_state(game_id, turn, side_number)

        -- If a prior action killed a leader, bail out; the scenario's
        -- [event name=die] handles endlevel.
        local s1, s2 = leaders_alive()
        if not (s1 and s2) then return end

        local action = wait_for_action(game_id)
        if not action then
            std_print(string.format(
                "[turn-stage] action timeout on turn %d side %d", turn, side_number))
            return
        end

        if action.type == "end_turn" then return end

        local result = action_handler.execute_action(action)
        if not result.success then
            std_print(string.format(
                "[turn-stage] turn %d side %d action=%s rejected: %s",
                turn, side_number, tostring(action.type),
                tostring(result.error)))
            -- Stay in the loop. Next iteration: Python sees (unchanged)
            -- state and picks something else. Critical difference vs a
            -- CA: no blacklist kicks us out.
        end
    end
end

return M
