-- ca_state_sender.lua
-- Candidate Action: serialize the game state and hand it off to the
-- Python trainer by writing a file that Python is polling for.
--
-- Protocol: write to "<game_dir>/state.wml.tmp", then os.rename() it to
-- "state.wml" so Python never observes a partial write. Python reads the
-- file and deletes it, so this CA can safely overwrite next turn.
--
-- NOTE: paths here are relative. Wesnoth's CWD is <userdata>/data/, so
-- "add-ons/wesnoth_ai/..." resolves via the directory junction installed
-- by main.py's install_addon() — meaning Lua and Python both see the
-- same file on disk.

local state_collector = wesnoth.require("~add-ons/wesnoth_ai/lua/state_collector.lua")

local ca_state_sender = {}

local function game_dir(game_id)
    return "add-ons/wesnoth_ai/games/" .. game_id .. "/"
end

local function detect_game_over(game_state)
    -- Scenario ends when a side loses its leader. Flag it so Python can
    -- assign terminal rewards.
    local side1, side2 = false, false
    for _, leader in ipairs(wesnoth.units.find_on_map({ canrecruit = true })) do
        if leader.side == 1 then side1 = true end
        if leader.side == 2 then side2 = true end
    end

    if side1 and side2 then return end

    game_state.game_over = true
    if not side1 and not side2 then
        game_state.winner = 0  -- draw (both leaders died the same turn)
    elseif not side1 then
        game_state.winner = 2
    else
        game_state.winner = 1
    end
end

local function write_state_file(game_id, wml_state)
    local dir = game_dir(game_id)
    local tmp_path = dir .. "state.wml.tmp"
    local final_path = dir .. "state.wml"

    local fh, err = io.open(tmp_path, "w")
    if not fh then
        error("state-sender: could not open '" .. tmp_path .. "': " .. tostring(err))
    end
    fh:write(wml_state)
    fh:close()

    -- Rename over any stale final file. On Windows rename() does not always
    -- overwrite an existing file, so clear it first. By protocol Python
    -- should have already deleted it, but we defend against crashes.
    os.remove(final_path)
    local ok, rename_err = os.rename(tmp_path, final_path)
    if not ok then
        error("state-sender: rename failed: " .. tostring(rename_err))
    end
end

function ca_state_sender:evaluation()
    -- Always high score; Wesnoth blacklists us for the rest of this side's
    -- turn after we execute once, which is exactly the behavior we want.
    return 999990
end

function ca_state_sender:execution(cfg, data)
    local side_number = wesnoth.current.side
    local game_id = wml.variables.game_id or "game_0"

    std_print(string.format("[Turn %d, Side %d] Sending state to Python",
        wesnoth.current.turn, side_number))

    local game_state = state_collector.collect_game_state(side_number, game_id)
    detect_game_over(game_state)

    write_state_file(game_id, wml.tostring(game_state))

    std_print(string.format("[Turn %d, Side %d] State sent",
        wesnoth.current.turn, side_number))
end

return ca_state_sender
