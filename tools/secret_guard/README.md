# secret_guard

Keeps credentials out of Claude Code transcripts. What a tool returns to
Claude is stored in the session's log and sent to the model, so every
tool result is redacted before Claude Code records it, whatever printed
the credential: a library's exception message, a verbose log, a dump of
the environment. Nothing about what the commands can do changes; the keys
stay usable, only their values never reach a result.

## Quickstart

```bash
python tools/secret_guard/install.py                   # copy to ~/.claude/secret_guard/, check, print the settings
python tools/secret_guard/install.py --write-settings  # the same, then add the settings to ~/.claude/settings.json
python tools/secret_guard/install.py --remove-settings # take them out again
```

The settings apply to running sessions as soon as the file is saved. After
changing `secret_guard.py` or `guard_shell.sh`, run the installer again:
sessions use the installed copy.

## What it covers

- **Bash**, including failed commands, background commands and subagents:
  the shell prefix (`CLAUDE_CODE_SHELL_PREFIX` = `guard_shell.sh`) runs
  each command with its stdout and stderr passed through
  `secret_guard.py filter` before Claude Code reads them.
- **The other tools that return text** (Read, Grep, the PowerShell tool,
  web and MCP tools): a `PostToolUse` hook rewrites a successful result
  (`updatedToolOutput`). A hook cannot rewrite a failed call's output, so
  for a failure holding a credential a `PostToolUseFailure` hook tells
  Claude to warn the user, and logs the event to
  `~/.claude/secret_guard/alerts.log`.

Redacted: the Vast API key (`~/.config/vastai/vast_api_key`), the Hugging
Face tokens (`~/.cache/huggingface/token`, `stored_tokens`), any value
listed one per line in `~/.claude/secret_guard/extra_secrets.txt`, and
every environment variable whose name marks a credential; each with its
URL-encoded form and any fragment of 10 characters or more. Strings shaped
like a credential are redacted whatever their value: Hugging Face tokens,
`api_key=` parameters, bearer tokens, `CONTAINER_API_KEY` assignments,
private-key blocks.

## Limits

- A hook that times out lets the result through; the shell prefix has no
  such failure mode. Claude Code's telemetry sees a hook-rewritten result
  before the rewrite.
- A background job (`&`) that inherits a command's output keeps the Bash
  call open until the job exits (the filter waits for the end of its
  input): redirect the job's output, or use `run_in_background`.
- Redaction is by value and by shape: a credential re-encoded (base64, hex)
  or cut into pieces shorter than 10 characters passes.
- Each Bash call starts two Python processes (about 0.2 s here).
