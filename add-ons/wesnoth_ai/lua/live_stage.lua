-- live_stage.lua
-- Our side's AI stage in a live game against the default AI
-- (tools/live_vs_rca.py), installed by live_ai.cfg. At each decision it
-- writes a sync marker into the engine log, the stream in which the
-- engine logs every command and draw (so the driver knows every earlier
-- one is logged), reports the whole board as a frame (board_report.lua),
-- then runs the batch of engine commands the driver sends for that
-- decision: a move, an attack, a recruit, or the end of the turn. The
-- driver mirrors the game in its simulator from the engine log; this
-- stage decides nothing.

local json = wesnoth.require("~add-ons/wesnoth_ai/lua/json_encoder.lua")
local board = wesnoth.require("~add-ons/wesnoth_ai/lua/board_report.lua")

local FRAME_BEGIN = "===WESNOTH_AI_STATE_BEGIN==="
local FRAME_END   = "===WESNOTH_AI_STATE_END==="
local SYNC_MARKER = "WESNOTH_AI_SYNC"
local IPC_DIR     = "~add-ons/wesnoth_ai/games/live/"
local POLL_MS     = 10
local TIMEOUT_MS  = 300000

-- The driver replaces the file atomically; a read that meets the moment
-- of the replacement fails and is tried again at the next poll.
local function load_table(path)
    if not wesnoth.have_file(path) then return nil end
    local read_ok, src = pcall(wesnoth.read_file, path)
    if not read_ok or not src or src == "" then return nil end
    local chunk = load(src, path, "t")
    if not chunk then return nil end
    local ok, result = pcall(chunk)
    if ok and type(result) == "table" then return result end
    return nil
end

-- The driver's speed, with animations when a person watches the game
-- (settings.animate), none for a check run.
local function set_prefs()
    local settings = load_table(IPC_DIR .. "settings.lua") or {}
    local animate = settings.animate ~= false
    local function try(k, v)
        pcall(function() wesnoth.preferences[k] = v end)
    end
    try("turbo", true)
    try("turbo_speed", settings.turbo_speed or 4.0)
    try("animate_map", animate)
    try("show_combat", animate)
    try("scroll_to_action", animate)
    try("idle_anim", false)
end
set_prefs()

local M = {}

-- Decisions are numbered for the whole process (one game per process).
local seq = 0

-- Wesnoth names every computer side after the default AI
-- (connect_engine.cpp:945-953, 1037), and with no human side its end
-- screen reads Defeat whoever wins (play_controller.cpp:1035): the
-- watcher is told which side the network plays, once.
local function announce()
    local settings = load_table(IPC_DIR .. "settings.lua") or {}
    local player = settings.player or "the Python driver"
    local side = wesnoth.current.side
    pcall(wesnoth.interface.add_chat_message, "Wesnoth AI", string.format(
        "Side %d is played by %s, the other side by the default AI. With no human side, "
        .. "the end screen reads Defeat whoever wins.", side, player))
end

local function emit_frame()
    local record = board.record("live")
    record.seq = seq
    std_print(FRAME_BEGIN)
    std_print(json.encode(record))
    std_print(FRAME_END)
end

local function wait_for_batch()
    local path = IPC_DIR .. "action.lua"
    local deadline = wesnoth.get_time_stamp() + TIMEOUT_MS
    while wesnoth.get_time_stamp() < deadline do
        local batch = load_table(path)
        if batch and batch.seq == seq then return batch end
        wesnoth.interface.delay(POLL_MS)
    end
    return nil
end

local function loc(x, y)
    return { x = x, y = y }
end

-- One engine command through the Lua AI's actions (ai/lua/core.cpp):
-- the weapon index is 1-based there.
local function execute(cmd)
    local result
    if cmd.type == "move" then
        result = ai.move(loc(cmd.from_x, cmd.from_y), loc(cmd.to_x, cmd.to_y))
    elseif cmd.type == "attack" then
        result = ai.attack(loc(cmd.from_x, cmd.from_y), loc(cmd.to_x, cmd.to_y), cmd.weapon)
    elseif cmd.type == "recruit" then
        result = ai.recruit(cmd.unit_type, loc(cmd.x, cmd.y))
    else
        return false, "unknown command " .. tostring(cmd.type)
    end
    return result.ok, result.status
end

function M:run_turn()
    if seq == 0 then announce() end
    while true do
        seq = seq + 1
        -- Error level reaches the log at any setting; `false` keeps it out
        -- of the chat (game_lua_kernel.cpp:4809-4822 reads the last
        -- argument as the chat flag).
        wesnoth.log("err", SYNC_MARKER .. " " .. seq, false)
        emit_frame()
        local batch = wait_for_batch()
        if not batch then
            std_print(string.format("[live-stage] no commands for decision %d", seq))
            return
        end
        for i, cmd in ipairs(batch.commands or {}) do
            if cmd.type == "end_turn" then return end
            local ok, status = execute(cmd)
            if not ok then
                std_print(string.format("[live-stage] decision %d command %d (%s) failed: %s",
                    seq, i, tostring(cmd.type), tostring(status)))
                break
            end
        end
    end
end

return M
