#!/usr/bin/env bash
# Shell prefix for Claude Code (CLAUDE_CODE_SHELL_PREFIX): runs the command
# line Claude Code assembled, given as $1, with its stdout and stderr each
# passed through secret_guard.py's filter, so that no credential reaches a
# Bash tool result, and exits with the command's status. The two streams stay
# apart because a stdio MCP server started through the prefix speaks its
# protocol on stdout.
#
# The filter runs on $SECRET_GUARD_PYTHON when it is set, else on the first of
# python, python3 and py (the Windows launcher) that runs Python 3.7 or later.
# A name on PATH is not enough: on Windows, python3 and sometimes python are
# the Microsoft Store's alias, which prints an advert and exits without reading
# its input, so the command would run and its output vanish. With no working
# interpreter the command is not run.
guard="$(dirname "${BASH_SOURCE[0]}")/secret_guard.py"
if [ ! -f "$guard" ]; then
    echo "secret_guard: $guard is missing, so the command was not run" >&2
    exit 126
fi
if [ -n "${SECRET_GUARD_PYTHON:-}" ]; then
    candidates=("$SECRET_GUARD_PYTHON")
else
    candidates=(python python3 py)
fi
py=
for candidate in "${candidates[@]}"; do
    # stdin from /dev/null: the probe must not take the command's input.
    if "$candidate" -I -S -c 'import sys; sys.exit(sys.version_info < (3, 7))' </dev/null >/dev/null 2>&1; then
        py=$candidate
        break
    fi
done
if [ -z "$py" ]; then
    echo "secret_guard: none of ${candidates[*]} runs Python 3.7 or later (SECRET_GUARD_PYTHON names one)," \
         "so the command was not run" >&2
    exit 126
fi
"$BASH" -c "$1" 2> >("$py" -I -S "$guard" filter >&2) | "$py" -I -S "$guard" filter
status=${PIPESTATUS[0]}
wait "$!" 2>/dev/null
exit "$status"
