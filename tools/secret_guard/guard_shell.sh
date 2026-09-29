#!/usr/bin/env bash
# Shell prefix for Claude Code (CLAUDE_CODE_SHELL_PREFIX): runs the command
# line Claude Code assembled, given as $1, with its stdout and stderr each
# passed through secret_guard.py's filter, so that no credential reaches a
# Bash tool result, and exits with the command's status. The two streams stay
# apart because a stdio MCP server started through the prefix speaks its
# protocol on stdout.
guard="$(dirname "${BASH_SOURCE[0]}")/secret_guard.py"
if [ ! -f "$guard" ]; then
    echo "secret_guard: $guard is missing, so the command was not run" >&2
    exit 126
fi
bash -c "$1" 2> >(python -I -S "$guard" filter >&2) | python -I -S "$guard" filter
status=${PIPESTATUS[0]}
wait "$!" 2>/dev/null
exit "$status"
