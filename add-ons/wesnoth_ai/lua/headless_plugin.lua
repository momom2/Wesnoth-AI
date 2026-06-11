-- headless_plugin.lua
-- ----------------------------------------------------------------------
-- Plugin that drives Wesnoth from titlescreen → MP Create → launch → Game
-- → back to titlescreen, in a loop, so one Wesnoth process plays N games
-- back-to-back. Designed to pair with `--nogui` for true headless training.
--
-- USAGE:
--     wesnoth.exe --nogui --nosound --nomusic \
--                 --plugin "<absolute-path>\headless_plugin.lua"
--
-- Requires training_scenario_mp.cfg to be active in _main.cfg so the id
-- "ai_training_mp" is registered in the MP scenario browser (comment out
-- the [test] include, uncomment the [multiplayer] include).
--
-- DESIGN NOTES
-- ----------------------------------------------------------------------
-- Wesnoth plugins are Lua coroutines that run in parallel with the engine
-- and yield control whenever they want to let the game advance. Each
-- coroutine.yield() returns (events, context, info); `info.name` tells
-- you which UI context you're in (titlescreen, Multiplayer Lobby,
-- Multiplayer Create, Multiplayer Staging, Game, Dialog, …) and
-- `context` has methods that transition to the next one.
--
-- This is the same pattern used by mainline CI scripts (host.lua,
-- join.lua in data/test/plugin). See those files for a reference of the
-- available context methods.
--
-- LOOP STRUCTURE
-- ----------------------------------------------------------------------
-- 1. Wait for titlescreen.
-- 2. Click "Multiplayer" (context.play_multiplayer).
-- 3. Click "Create" in the lobby (context.create).
-- 4. Select our scenario by id (context.select_type + select_level).
-- 5. Settle settings and click "Create" again (context.create) — this
--    moves from Create to Staging.
-- 6. Click "Launch" to start the game (context.launch).
-- 7. Wait for info.name == "Game".
-- 8. Wait for the game to end (info.name ~= "Game").
-- 9. Quit back to titlescreen (context.quit repeated until there).
-- 10. Loop to step 2 for the next game.
--
-- Any modal dialog that appears mid-flow is dismissed with skip_dialog.

local SCENARIO_ID = "ai_training_mp"

-- Keep this high so a single Wesnoth process plays many games. -1
-- = unlimited; plugin exits when the Python training kills the process.
local MAX_GAMES   = -1

-- Safety-cap on yields while waiting for a specific context transition.
-- Plugin sandbox may not expose os.time(), so we count yields instead of
-- wall-clock seconds. Wesnoth yields its main loop many times per second;
-- a yield budget of ~5000 is order-seconds on typical hardware.
local MAX_YIELDS_PER_WAIT = 5000
-- Every N yields, log a status line so we can see progress in the
-- .out.log (which with --log-to-file IS written even under --nogui).
local LOG_EVERY_YIELDS    = 200

local function plugin()
    local events, context, info

    local function log(msg)
        std_print("[headless_plugin] " .. tostring(msg))
    end

    -- Dump every key of info and every method of context. Used when
    -- we're stuck in an unfamiliar UI state so we can see what options
    -- the engine is offering us.
    local function dump_current(tag)
        local i_keys, c_keys = {}, {}
        if info then
            for k, _ in pairs(info) do table.insert(i_keys, tostring(k)) end
        end
        if context then
            for k, _ in pairs(context) do table.insert(c_keys, tostring(k)) end
        end
        table.sort(i_keys); table.sort(c_keys)
        log(tag .. " info keys: " .. table.concat(i_keys, ","))
        log(tag .. " context methods: " .. table.concat(c_keys, ","))
    end

    -- Yield until predicate(info) is true. Returns true on success,
    -- false on exhausting MAX_YIELDS_PER_WAIT. Uses a yield counter
    -- (not os.time, which may not be in the plugin sandbox) so logging
    -- is guaranteed to fire.
    local function wait_until(pred, desc)
        local yields = 0
        local last_seen = ""
        local stuck_yields = 0
        local dumped = false
        while true do
            events, context, info = coroutine.yield()
            yields = yields + 1

            -- Auto-dismiss any modal dialog that appears. Wrap in pcall
            -- in case the method is present but errors on this context.
            if info and info.name == "Dialog" and context and context.skip_dialog then
                pcall(function() context.skip_dialog({}) end)
            end

            if pred(info) then return true end

            local cur = (info and info.name) or "<nil info>"
            if cur ~= last_seen then
                stuck_yields = 0
                dumped = false
                last_seen = cur
            else
                stuck_yields = stuck_yields + 1
            end

            if yields % LOG_EVERY_YIELDS == 0 then
                log("waiting for " .. desc .. " (yield " .. yields ..
                    "), currently in " .. cur)
            end

            -- Once we've been stuck in one context for a while, dump
            -- the full context+info shape so we can see what's being
            -- offered. Fire at ~2× the log interval so we get it once
            -- per stuck state.
            if stuck_yields == LOG_EVERY_YIELDS * 2 and not dumped then
                dumped = true
                dump_current("STUCK@" .. cur)
            end

            -- Escalation: if we've been in a Dialog context for a while,
            -- skip_dialog clearly isn't advancing it, try quit as a
            -- fallback (cancels the connecting/confirmation dialog).
            if cur == "Dialog" and stuck_yields == LOG_EVERY_YIELDS * 3 then
                if context and context.quit then
                    log("STUCK: trying context.quit on Dialog")
                    pcall(function() context.quit({}) end)
                end
            end

            if yields >= MAX_YIELDS_PER_WAIT then
                log("TIMEOUT (" .. MAX_YIELDS_PER_WAIT .. " yields) waiting for " ..
                    desc .. "; last context = " .. cur)
                return false
            end
        end
    end

    local function fatal(code, msg)
        log("FATAL: " .. msg)
        if context and context.exit then context.exit({code = code}) end
        coroutine.yield()
    end

    log("plugin loaded, scenario_id=" .. SCENARIO_ID)

    -- Initial yield to get the first (events, context, info).
    events, context, info = coroutine.yield()

    if not wait_until(function(i) return i and i.name == "titlescreen" end,
                      "titlescreen") then
        return fatal(1, "never reached titlescreen")
    end

    local game_count = 0
    while MAX_GAMES < 0 or game_count < MAX_GAMES do
        game_count = game_count + 1
        log("=== starting game " .. game_count .. " ===")

        -- 2: titlescreen → Multiplayer Lobby.
        if context.play_multiplayer == nil then
            return fatal(1, "no play_multiplayer in titlescreen context")
        end
        context.play_multiplayer({})

        if not wait_until(function(i) return i and i.name == "Multiplayer Lobby" end,
                          "Multiplayer Lobby") then
            return fatal(2, "failed to enter Multiplayer Lobby")
        end

        -- 3: Multiplayer Lobby → Multiplayer Create.
        context.create({})
        if not wait_until(function(i) return i and i.name == "Multiplayer Create" end,
                          "Multiplayer Create") then
            return fatal(3, "failed to enter Multiplayer Create")
        end

        -- 4: Select our scenario. select_type is required first — it
        -- switches between campaigns/scenarios tabs.
        if context.select_type then
            context.select_type({type = "scenario"})
        end
        -- Yield once so the UI model settles after select_type.
        events, context, info = coroutine.yield()

        local level = info.find_level and info.find_level({id = SCENARIO_ID})
        if not level or not level.index or level.index < 0 then
            return fatal(4, "scenario id=" .. SCENARIO_ID ..
                " not found in MP browser — is training_scenario_mp.cfg " ..
                "active in _main.cfg?")
        end
        context.select_level({index = level.index})
        log("selected scenario at index=" .. level.index)

        -- Registered-users=false means no account-based restrictions;
        -- matches host.lua's "no lobby accounts" local-play pattern.
        if context.update_settings then
            context.update_settings({registered_users = false})
        end
        events, context, info = coroutine.yield()

        -- 5: Multiplayer Create → Multiplayer Staging.
        context.create({})
        if not wait_until(function(i) return i and i.name == "Multiplayer Staging" end,
                          "Multiplayer Staging") then
            return fatal(5, "failed to enter Multiplayer Staging")
        end

        -- 6: Launch the game.
        context.launch({})
        if not wait_until(function(i) return i and i.name == "Game" end,
                          "Game") then
            return fatal(6, "failed to enter Game")
        end
        log("game " .. game_count .. " running")

        -- 7: Wait for the game to finish. The scenario's endlevel event
        -- or a Python-driven force-quit will exit the Game context.
        if not wait_until(function(i) return i and i.name ~= "Game" end,
                          "game end") then
            return fatal(7, "game never ended within timeout")
        end
        log("game " .. game_count .. " ended, now in " .. info.name)

        -- 8: Navigate back to titlescreen for the next iteration.
        -- quit may need calling multiple times to traverse nested contexts.
        local tries = 0
        while info and info.name ~= "titlescreen" and tries < 20 do
            if context.quit then context.quit({}) end
            events, context, info = coroutine.yield()
            tries = tries + 1
        end
        if info == nil or info.name ~= "titlescreen" then
            return fatal(8, "could not return to titlescreen after game " ..
                game_count .. " (stuck in " ..
                ((info and info.name) or "<nil>") .. ")")
        end
    end

    log("completed " .. game_count .. " games, clean exit")
    if context and context.exit then context.exit({code = 0}) end
    coroutine.yield()
end

return plugin
