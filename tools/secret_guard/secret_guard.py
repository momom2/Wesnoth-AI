#!/usr/bin/env python3
"""Keep credentials out of Claude Code transcripts.

What a tool returns to Claude is the transcript: it is stored in the
session's log and sent to the model. This module redacts every credential
before that happens, whatever printed it.

    python secret_guard.py filter    # stdin to stdout, credentials redacted
    python secret_guard.py hook      # a PostToolUse / PostToolUseFailure hook

`filter` is what the shell prefix (guard_shell.sh, set as
CLAUDE_CODE_SHELL_PREFIX) pipes each Bash command's stdout and stderr
through, so a failed command's output is covered too. `hook` redacts the
result of the other tools (Read, Grep, the PowerShell tool, web and MCP
tools) through `updatedToolOutput`; a failed call's output cannot be
rewritten by a hook, so for those it adds a warning for Claude to relay and
appends a line to ~/.claude/secret_guard/alerts.log.

The credentials, read afresh by every run: the Vast API key, the Hugging
Face tokens, the values listed one per line in
~/.claude/secret_guard/extra_secrets.txt, and every environment variable
whose name marks a credential
(token, secret, password, API key...). Each is redacted wherever it
appears, with its URL-encoded form and any fragment of FRAGMENT characters
or more. Strings shaped like a credential are redacted whatever their value:
Hugging Face tokens, `api_key=` parameters, bearer tokens,
`CONTAINER_API_KEY` assignments and private-key blocks.
"""
from __future__ import annotations

import codecs
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import quote

PLACEHOLDER = "<redacted>"
# Shortest fragment of a known credential that is redacted: 10 hex
# characters are 40 bits, so an unrelated hex string (a commit hash, a
# digest) matches one of a key's ~60 fragments about once in 2**34 tries.
FRAGMENT = 10
# Shortest value counted as a credential (shorter environment values are
# flags and names, not keys).
MIN_SECRET = 16
# Characters of an unfinished line held back at least, so that a credential
# shape cut between two reads (a token is ~40 characters, a bearer token a
# few hundred at most) is still whole when redacted.
SHAPE_WINDOW = 512

CREDENTIAL_NAME = re.compile(r"token|secret|passw|api_?key|access_?key|private_?key|credential", re.I)

# Credential shapes, redacted whatever their value; group "v" is the value.
SHAPES = [
    re.compile(r"(?P<v>\bhf_[A-Za-z0-9]{30,})"),
    re.compile(r"api_key=(?P<v>[A-Za-z0-9_\-]{16,})"),
    re.compile(r"(?i:\bbearer)\s+(?P<v>[A-Za-z0-9_\-.~+/]{16,}=*)"),
    re.compile(r"CONTAINER_API_KEY['\"]?\s*[:=]\s*['\"]?(?P<v>[A-Za-z0-9_\-]{16,})"),
    re.compile(r"(?P<v>-----BEGIN [A-Z ]*PRIVATE KEY-----[A-Za-z0-9+/=\s:,\-\\]*?-----END [A-Z ]*PRIVATE KEY-----)"),
]
KEY_BEGIN = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
KEY_END = re.compile(r"-----END [A-Z ]*PRIVATE KEY-----")
# A line of a private key's body: base64, blank, or an encryption header.
KEY_BODY = re.compile(r"[A-Za-z0-9+/=]*|(Proc-Type|DEK-Info|Comment):.*")


# ---------------------------------------------------------------------
# The credentials
# ---------------------------------------------------------------------

def _home() -> Path:
    return Path(os.environ.get("SECRET_GUARD_HOME") or Path.home())


def data_dir() -> Path:
    """Where the extra credentials and the logs live: never next to a
    repository copy of this file."""
    return _home() / ".claude" / "secret_guard"


def _file_values(path: Path) -> List[str]:
    """The credential values in one file: its whole content, or for an
    INI-like store the value of every `name = value` line."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    values = [text.strip()]
    for line in text.splitlines():
        if "=" in line:
            values.append(line.split("=", 1)[1].strip().strip("'\""))
    return values


def secret_values(environ=None) -> List[str]:
    """Every known credential value, longest first."""
    environ = os.environ if environ is None else environ
    home = _home()
    candidates: List[str] = []
    for path in (home / ".config" / "vastai" / "vast_api_key", home / ".vast_api_key",
                 home / ".cache" / "huggingface" / "token",
                 home / ".cache" / "huggingface" / "stored_tokens"):
        candidates += _file_values(path)
    try:
        extra = (data_dir() / "extra_secrets.txt").read_text(encoding="utf-8")
        candidates += [line.strip() for line in extra.splitlines()
                       if line.strip() and not line.lstrip().startswith("#")]
    except OSError:
        pass
    for name, value in environ.items():
        value = value.strip()
        if CREDENTIAL_NAME.search(name) and not os.path.exists(value):
            candidates.append(value)
    values = {v for v in candidates if len(v) >= MIN_SECRET and "\n" not in v}
    return sorted(values, key=len, reverse=True)


# ---------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------

class Redactor:
    """Finds credentials in text: every fragment of FRAGMENT characters of
    a known value (or of its URL-encoded form), and the credential shapes."""

    def __init__(self, values: Iterable[str]):
        forms = set()
        for v in values:
            forms.add(v)
            forms.add(quote(v, safe=""))
        self.fragments = {f[i:i + FRAGMENT] for f in forms for i in range(len(f) - FRAGMENT + 1)}
        self.longest = max((len(f) for f in forms), default=0)

    def spans(self, text: str) -> List[Tuple[int, int]]:
        """The merged (start, end) spans of credentials in `text`."""
        found = []
        for frag in self.fragments:
            at = text.find(frag)
            while at != -1:
                found.append((at, at + FRAGMENT))
                at = text.find(frag, at + 1)
        for shape in SHAPES:
            found += [m.span("v") for m in shape.finditer(text)]
        return _merge(found)

    def redact(self, text: str) -> str:
        return _replace(text, self.spans(text))


def _merge(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _replace(text: str, spans: List[Tuple[int, int]]) -> str:
    if not spans:
        return text
    out, last = [], 0
    for start, end in spans:
        out += [text[last:start], PLACEHOLDER]
        last = end
    out.append(text[last:])
    return "".join(out)


class StreamRedactor:
    """Redacts a stream as it arrives. Complete lines are redacted and
    passed on at once; of an unfinished line, all but its last
    `redactor.longest` characters are, and never across a credential, so
    a credential cut between two reads is still whole when redacted. A
    private-key block spanning lines is redacted from its first line to
    its END marker, or to the first line that is not part of a key's body
    (a lone marker, as in code that names one, withholds nothing else)."""

    def __init__(self, redactor: Redactor):
        self.redactor = redactor
        self.pending = ""
        self.in_key = False

    def feed(self, text: str) -> str:
        self.pending += text
        cut = max(self.pending.rfind("\n"), self.pending.rfind("\r")) + 1
        out = self._lines(self.pending[:cut])
        tail = self.pending[cut:]
        if self.in_key:
            self.pending = tail
            return out
        keep = max(self.redactor.longest, SHAPE_WINDOW)
        if KEY_BEGIN.search(tail) or len(tail) <= keep:
            self.pending = tail
            return out
        split = len(tail) - keep
        spans = self.redactor.spans(tail)
        for start, end in spans:
            if start < split < end:
                split = start
        ready = [s for s in spans if s[1] <= split]
        self.pending = tail[split:]
        return out + _replace(tail[:split], ready)

    def close(self) -> str:
        rest, self.pending = self.pending, ""
        return self._lines(rest)

    def _lines(self, block: str) -> str:
        if not block:
            return ""
        if not self.in_key and "PRIVATE KEY-----" not in block:
            return self.redactor.redact(block)
        out = []
        for line in block.splitlines(keepends=True):
            body = line.rstrip("\r\n")
            ending = line[len(body):]
            if self.in_key:
                end = KEY_END.search(body)
                if end:
                    self.in_key = False
                    out.append(PLACEHOLDER + self.redactor.redact(body[end.end():]) + ending)
                    continue
                if KEY_BODY.fullmatch(body.strip()):
                    out.append(ending)
                    continue
                self.in_key = False
            begin = KEY_BEGIN.search(body)
            if begin and not KEY_END.search(body, begin.end()):
                self.in_key = True
                out.append(self.redactor.redact(body[:begin.start()]) + PLACEHOLDER + ending)
            else:
                out.append(self.redactor.redact(line))
        return "".join(out)


# ---------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------

def run_filter(src, dst, redactor: Optional[Redactor] = None) -> None:
    """Copy bytes from `src` to `dst` with every credential redacted.
    Invalid UTF-8 passes through unchanged (surrogateescape)."""
    stream = StreamRedactor(redactor or Redactor(secret_values()))
    decoder = codecs.getincrementaldecoder("utf-8")(errors="surrogateescape")
    try:
        while True:
            chunk = src.read1(65536) if hasattr(src, "read1") else src.read(65536)
            if not chunk:
                break
            _write(dst, stream.feed(decoder.decode(chunk)))
        _write(dst, stream.feed(decoder.decode(b"", final=True)) + stream.close())
    except Exception as e:  # noqa: BLE001 - withhold rather than pass unredacted text
        _log("errors.log", f"filter {type(e).__name__}")
        _write(dst, f"\n<secret_guard: filter error {type(e).__name__}; the rest of this output is withheld>\n")
        while src.read(65536):
            pass


def _write(dst, text: str) -> None:
    if text:
        dst.write(text.encode("utf-8", errors="surrogateescape"))
        dst.flush()


def _redact_tree(value, redactor: Redactor):
    """`value` with every string redacted, and whether any changed."""
    if isinstance(value, str):
        out = redactor.redact(value)
        return out, out != value
    if isinstance(value, list):
        pairs = [_redact_tree(v, redactor) for v in value]
        return [p[0] for p in pairs], any(p[1] for p in pairs)
    if isinstance(value, dict):
        pairs = {k: _redact_tree(v, redactor) for k, v in value.items()}
        return {k: p[0] for k, p in pairs.items()}, any(p[1] for p in pairs.values())
    return value, False


def hook_response(event: dict, redactor: Redactor) -> Optional[dict]:
    """The hook's JSON answer to one PostToolUse or PostToolUseFailure
    event, or None when the result holds no credential."""
    kind = event.get("hook_event_name")
    if kind == "PostToolUse":
        cleaned, changed = _redact_tree(event.get("tool_response"), redactor)
        if changed:
            return {"hookSpecificOutput": {"hookEventName": "PostToolUse", "updatedToolOutput": cleaned}}
    elif kind == "PostToolUseFailure":
        if redactor.spans(str(event.get("error") or "")):
            _log("alerts.log", f"{event.get('tool_name')} failure output held a credential "
                         f"(session {event.get('session_id')}, call {event.get('tool_use_id')})")
            return {"hookSpecificOutput": {
                "hookEventName": "PostToolUseFailure",
                "additionalContext": (
                    "secret_guard: this failed tool call's output contains a credential, and a hook cannot "
                    "rewrite a failure's output. Tell the user now that the credential reached the "
                    "transcript and must be rotated; do not repeat it.")}}
    return None


def run_hook(stdin, stdout) -> None:
    """The hook on byte streams: the event is UTF-8 JSON, and the answer
    ASCII JSON, whatever the console's code page."""
    try:
        answer = hook_response(json.loads(stdin.read().decode("utf-8")), Redactor(secret_values()))
    except Exception as e:  # noqa: BLE001 - a hook error must not print the result it held
        _log("errors.log", f"hook {type(e).__name__}")
        return
    if answer is not None:
        stdout.write(json.dumps(answer).encode("ascii"))
        stdout.flush()


def _log(name: str, line: str) -> None:
    try:
        folder = data_dir()
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / name).open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {line}\n")
    except OSError:
        pass


def main(argv: List[str]) -> int:
    if argv[1:] == ["filter"]:
        run_filter(sys.stdin.buffer, sys.stdout.buffer)
        return 0
    if argv[1:] == ["hook"]:
        run_hook(sys.stdin.buffer, sys.stdout.buffer)
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
