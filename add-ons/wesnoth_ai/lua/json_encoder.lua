-- json_encoder.lua
-- Minimal JSON encoder for Wesnoth Lua's sandbox.
--
-- Why we can't use wml.tostring: it requires WML-shaped tables
-- (numeric-indexed {tag, child} pairs for children) and rejects plain
-- Lua tables with list-valued string keys.
--
-- Why we roll our own: the Wesnoth Lua sandbox has no json library and
-- no way to load external Lua files outside the add-on tree. This is
-- small enough to live in-tree without being a maintenance burden.
--
-- Usage:
--     local json = wesnoth.require("~add-ons/wesnoth_ai/lua/json_encoder.lua")
--     local s = json.encode(some_table)
--
-- Arrays vs objects: a table with the same number of keys as its length
-- operator #t is encoded as a JSON array. Empty tables encode as "[]"
-- (we treat ambiguous empties as arrays — callers that need {} for an
-- empty object should pass `json.empty_object()` or structure their
-- data so the field isn't empty).

local M = {}

-- Escape the characters JSON requires escaped. Lua's %c class covers
-- control chars (0x00–0x1f); we handle the common ones explicitly and
-- use \uXXXX for the rest.
local escapes = {
    ['"']  = '\\"',
    ['\\'] = '\\\\',
    ['\b'] = '\\b',
    ['\f'] = '\\f',
    ['\n'] = '\\n',
    ['\r'] = '\\r',
    ['\t'] = '\\t',
}

local function escape_char(c)
    local e = escapes[c]
    if e then return e end
    return string.format("\\u%04x", string.byte(c))
end

local function escape_string(s)
    return (s:gsub('[%z\1-\31"\\]', escape_char))
end

local function is_array_like(t)
    local n = #t
    if n == 0 then
        -- Empty table: ambiguous. Treat as array.
        for _ in pairs(t) do return false end  -- has string keys → object
        return true
    end
    local count = 0
    for k in pairs(t) do
        count = count + 1
        if type(k) ~= "number" or k < 1 or k > n or k ~= math.floor(k) then
            return false
        end
    end
    return count == n
end

local encode  -- forward-declared for recursion

local function encode_table(t, visited)
    if visited[t] then
        error("json_encoder: cyclic reference")
    end
    visited[t] = true

    local result
    if is_array_like(t) then
        local parts = {}
        for i = 1, #t do
            parts[i] = encode(t[i], visited)
        end
        result = "[" .. table.concat(parts, ",") .. "]"
    else
        local parts = {}
        -- Stable-ish key order: sort so logs diff cleanly. Tiny overhead.
        local keys = {}
        for k in pairs(t) do table.insert(keys, k) end
        table.sort(keys, function(a, b) return tostring(a) < tostring(b) end)
        for _, k in ipairs(keys) do
            if type(k) == "string" or type(k) == "number" then
                table.insert(parts,
                    '"' .. escape_string(tostring(k)) .. '":' .. encode(t[k], visited))
            end
        end
        result = "{" .. table.concat(parts, ",") .. "}"
    end

    visited[t] = nil
    return result
end

encode = function(v, visited)
    local t = type(v)
    if v == nil then return "null" end
    if t == "boolean" then return v and "true" or "false" end
    if t == "number" then
        if v ~= v or v == math.huge or v == -math.huge then
            return "null"  -- JSON has no NaN / Infinity
        end
        return tostring(v)
    end
    if t == "string"  then return '"' .. escape_string(v) .. '"' end
    if t == "table"   then return encode_table(v, visited or {}) end
    error("json_encoder: cannot encode type '" .. t .. "'")
end

M.encode = function(v) return encode(v, {}) end

return M
