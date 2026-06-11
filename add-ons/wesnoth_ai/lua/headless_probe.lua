-- headless_probe.lua
-- Diagnostic plugin: observes the context/info stream and dumps every
-- unique state seen. Purpose: figure out what Wesnoth's UI looks like
-- from inside a plugin under --nogui so we can write a working driver.
--
-- No mutations. Just observes. Exits after ~2000 yields.

local function plugin()
    local events, context, info
    local seen = {}
    local yields = 0
    local MAX = 2000
    local LOG_EVERY = 100

    local function dump_state(tag)
        local name = (info and info.name) or "<nil>"
        local i_keys, c_keys, evs = {}, {}, {}
        if info then
            for k in pairs(info) do table.insert(i_keys, tostring(k)) end
        end
        if context then
            for k in pairs(context) do table.insert(c_keys, tostring(k)) end
        end
        if events then
            for i, v in ipairs(events) do
                table.insert(evs, tostring(v[1]))
            end
        end
        table.sort(i_keys); table.sort(c_keys)
        std_print("[probe] " .. tag .. " name=" .. name)
        std_print("[probe]   info: " .. table.concat(i_keys, ","))
        std_print("[probe]   context: " .. table.concat(c_keys, ","))
        std_print("[probe]   events: " .. table.concat(evs, ","))
    end

    std_print("[probe] plugin loaded")
    events, context, info = coroutine.yield()

    -- First observation — unconditional dump.
    dump_state("yield=1")
    seen[(info and info.name) or "<nil>"] = true

    while yields < MAX do
        events, context, info = coroutine.yield()
        yields = yields + 1
        local name = (info and info.name) or "<nil>"
        if not seen[name] then
            seen[name] = true
            dump_state("yield=" .. yields .. " NEW")
        end
        if yields % LOG_EVERY == 0 then
            std_print("[probe] yield=" .. yields .. " still in " .. name)
        end

        -- Attempt ONE transition: from titlescreen, call play_multiplayer.
        -- Then observe what happens next.
        if name == "titlescreen" and context and context.play_multiplayer
                and not seen._played_mp then
            seen._played_mp = true
            std_print("[probe] calling context.play_multiplayer({})")
            local ok, err = pcall(function() context.play_multiplayer({}) end)
            std_print("[probe] play_multiplayer returned ok=" .. tostring(ok) ..
                " err=" .. tostring(err))
        end

        -- On Dialog: IMMEDIATELY try quit (skip_dialog didn't work). See
        -- if that returns us to titlescreen or elsewhere useful.
        if name == "Dialog" and context and context.quit
                and not seen._quit_dialog then
            seen._quit_dialog = true
            std_print("[probe] calling context.quit({}) on Dialog")
            local ok, err = pcall(function() context.quit({}) end)
            std_print("[probe] quit returned ok=" .. tostring(ok) ..
                " err=" .. tostring(err))
        end
    end

    std_print("[probe] exhausted " .. MAX .. " yields, exiting")
    if context and context.exit then context.exit({code = 0}) end
    coroutine.yield()
end

return plugin
