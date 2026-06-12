# PreToolUse hook: gate recursive/forced deletes behind an explicit
# link-safety acknowledgement.
#
# Why: on 2026-06-12 a `Remove-Item -Recurse` on a directory JUNCTION
# followed the link and deleted the TARGET directory's contents
# (the vendored wesnoth_src/ tree). PowerShell 5.1 recurses into
# reparse points; the safe way to remove a link is `cmd /c rmdir`.
#
# Behavior: if the shell command contains a recursive/forced delete
# (Remove-Item -Recurse/-r, rm -rf, rmdir /s, del /s) and does NOT
# carry the acknowledgement marker `[verified-not-a-link]`, deny with
# instructions. To proceed, first verify every deleted path:
#     (Get-Item <path> -Force).Attributes  -> must NOT contain ReparsePoint
# then append `# [verified-not-a-link]` to the command.
$raw = [Console]::In.ReadToEnd()
try { $payload = $raw | ConvertFrom-Json } catch { exit 0 }
$cmd = $payload.tool_input.command
if (-not $cmd) { exit 0 }

$patterns = @(
    'Remove-Item[^|;`n]*\s-Recurse',
    'Remove-Item[^|;`n]*\s-r\b',
    'ri\s[^|;`n]*-Recurse',
    '\brm\s+(-[a-zA-Z]*r[a-zA-Z]*\b)',
    '\brmdir\s+([^|;`n]*\s)?/s\b',
    '\bdel\s+([^|;`n]*\s)?/s\b'
)
$hit = $false
foreach ($p in $patterns) {
    if ($cmd -match $p) { $hit = $true; break }
}
if (-not $hit) { exit 0 }
if ($cmd -match '\[verified-not-a-link\]') { exit 0 }

$reason = ("Recursive/forced delete blocked pending link-safety check. " +
    "On Windows, Remove-Item -Recurse on a junction/symlink FOLLOWS the " +
    "link and deletes the TARGET's contents (this destroyed wesnoth_src/ " +
    "once). Before retrying: (1) for EVERY path being deleted, run " +
    "(Get-Item <path> -Force).Attributes and confirm ReparsePoint is " +
    "absent; (2) if it IS a link, remove the link itself with " +
    "cmd /c rmdir <path> instead; (3) once verified, append the marker " +
    "comment  # [verified-not-a-link]  to the command and re-run.")

$out = @{
    hookSpecificOutput = @{
        hookEventName            = "PreToolUse"
        permissionDecision       = "deny"
        permissionDecisionReason = $reason
    }
}
$out | ConvertTo-Json -Depth 5 -Compress
exit 0
