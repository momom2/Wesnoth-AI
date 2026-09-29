"""secret_guard keeps credentials out of what Claude Code records.

The redactor on text and on a stream cut at every possible point, the
credential sources, the hook's answers, the shell prefix end to end, and the
installer's settings merge. Every test plants a random sentinel credential
and asserts it never comes out.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote

import pytest

from tools.secret_guard import install as sg_install
from tools.secret_guard import secret_guard as sg

GUARD = Path(__file__).resolve().parent.parent / "tools" / "secret_guard"

SENTINEL = "SGtest" + "7d41c09a5be2f86314aa0d9b"   # not-a-secret
P = sg.PLACEHOLDER
FRAG = sg.FRAGMENT


def _redactor(*values):
    return sg.Redactor(values or (SENTINEL,))


def test_a_value_its_fragments_and_its_url_form_are_redacted():
    r = _redactor(SENTINEL, "k3y/with+odd=chars&more_0123")
    assert r.redact(f"x {SENTINEL} y") == f"x {P} y"
    assert r.redact(f"head {SENTINEL[:12]}") == f"head {P}"
    assert r.redact(f"{SENTINEL[-11:]} tail") == f"{P} tail"
    assert r.redact(f"url?k={quote('k3y/with+odd=chars&more_0123', safe='')}") == f"url?k={P}"
    assert r.redact(f"short {SENTINEL[:9]} kept") == f"short {SENTINEL[:9]} kept"
    digest = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert r.redact(digest) == digest


def test_credential_shapes_are_redacted_whatever_their_value():
    r = sg.Redactor([])
    cases = {
        "token hf_" + "Ab3" * 12: f"token {P}",                                  # not-a-secret
        "GET https://x/api?api_key=" + "f" * 40 + "&q=1": f"GET https://x/api?api_key={P}&q=1",  # not-a-secret
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.e30.abcdef": f"Authorization: Bearer {P}",  # not-a-secret
        'CONTAINER_API_KEY="' + "9" * 32 + '"': f'CONTAINER_API_KEY="{P}"',      # not-a-secret
        "a -----BEGIN RSA PRIVATE KEY-----\\nMIIE\\n-----END RSA PRIVATE KEY----- b": f"a {P} b",  # not-a-secret
    }
    for text, expected in cases.items():
        assert r.redact(text) == expected, text


def _stream(chunks, redactor):
    s = sg.StreamRedactor(redactor)
    return "".join(s.feed(c) for c in chunks) + s.close()


def test_a_credential_cut_between_two_reads_is_still_redacted():
    """Every cut point of a line holding the credential, the line long
    enough that its start is passed on before the rest arrives."""
    r = _redactor()
    line = "x" * 2000 + SENTINEL + "y" * 50
    for cut in range(1990, len(line)):
        out = _stream([line[:cut], line[cut:]], r)
        assert SENTINEL[:FRAG] not in out and out == "x" * 2000 + P + "y" * 50, cut


def test_a_private_key_block_across_lines_is_redacted_whole():
    key = ["-----BEGIN OPENSSH PRIVATE KEY-----", "b3BlbnNzaC1rZXktdjEAAAAA", "AAAAC3NzaC1lZDI1",  # not-a-secret
           "-----END OPENSSH PRIVATE KEY-----"]
    text = "before\n" + "\n".join(key) + "\nafter\n"
    out = _stream([text[:30], text[30:70], text[70:]], sg.Redactor([]))
    assert "b3BlbnNzaC1rZXktdjEAAAAA" not in out and "AAAAC3NzaC1lZDI1" not in out
    assert out.startswith("before\n") and out.endswith("after\n")


def test_a_lone_key_marker_withholds_nothing_after_it():
    """Code or a log line naming a private-key marker is not a key: the
    lines after it that are not a key's body pass through."""
    marker = "-----BEGIN " + "RSA PRIVATE KEY-----"
    text = f'x = "{marker}"\nprint("ok")\n3 passed\n'
    out = _stream([text], sg.Redactor([]))
    assert out == f'x = "{P}\nprint("ok")\n3 passed\n'


def test_the_filter_passes_invalid_utf8_and_carriage_returns_through():
    data = b"\xff\xfeok " + SENTINEL.encode() + b"\r50%\r100%\n\xc3\xa9t\xc3\xa9\n"
    out = io.BytesIO()
    sg.run_filter(io.BytesIO(data), out, _redactor())
    assert out.getvalue() == b"\xff\xfeok " + P.encode() + b"\r50%\r100%\n\xc3\xa9t\xc3\xa9\n"


def test_the_credentials_come_from_the_key_stores_and_named_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_GUARD_HOME", str(tmp_path))
    (tmp_path / ".config" / "vastai").mkdir(parents=True)
    (tmp_path / ".config" / "vastai" / "vast_api_key").write_text("vast" + "a1" * 30 + "\n")
    (tmp_path / ".cache" / "huggingface").mkdir(parents=True)
    (tmp_path / ".cache" / "huggingface" / "stored_tokens").write_text(
        "[laptop]\nhf_token = hf_" + "Q" * 34 + "\n")                              # not-a-secret
    (tmp_path / ".claude" / "secret_guard").mkdir(parents=True)
    (tmp_path / ".claude" / "secret_guard" / "extra_secrets.txt").write_text(
        "# a box token\nextra" + "b2" * 10 + "\n")
    env = {"MY_API_KEY": "c3" * 12, "TOKENIZERS_PARALLELISM": "false", "PATH": "x" * 40,
           "SOME_CREDENTIALS_FILE": str(tmp_path)}
    values = sg.secret_values(env)
    assert set(values) == {"vast" + "a1" * 30, "hf_" + "Q" * 34, "extra" + "b2" * 10, "c3" * 12}


def test_the_hook_rewrites_a_result_and_flags_a_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_GUARD_HOME", str(tmp_path))
    r = _redactor()
    event = {"hook_event_name": "PostToolUse", "tool_name": "Read",
             "tool_response": {"type": "text", "file": {"content": f"a={SENTINEL}\nb", "numLines": 2}}}
    answer = sg.hook_response(event, r)
    assert answer["hookSpecificOutput"]["updatedToolOutput"] == {
        "type": "text", "file": {"content": f"a={P}\nb", "numLines": 2}}
    event["tool_response"] = {"file": {"content": "nothing here"}}
    assert sg.hook_response(event, r) is None
    failure = {"hook_event_name": "PostToolUseFailure", "tool_name": "PowerShell", "session_id": "s1",
               "error": f"Exit code 1\n{SENTINEL}"}
    context = sg.hook_response(failure, r)["hookSpecificOutput"]["additionalContext"]
    assert SENTINEL not in context and "rotated" in context
    log = (tmp_path / ".claude" / "secret_guard" / "alerts.log").read_text()
    assert "PowerShell" in log and SENTINEL not in log


def test_the_hook_reads_utf8_and_answers_ascii(monkeypatch):
    monkeypatch.setenv("SG_TEST_TOKEN", SENTINEL)
    event = {"hook_event_name": "PostToolUse", "tool_name": "Grep",
             "tool_response": {"content": f"été {SENTINEL}"}}
    out = io.BytesIO()
    sg.run_hook(io.BytesIO(json.dumps(event, ensure_ascii=False).encode("utf-8")), out)
    answer = json.loads(out.getvalue().decode("ascii"))
    assert answer["hookSpecificOutput"]["updatedToolOutput"] == {"content": f"été {P}"}


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
def test_the_shell_prefix_redacts_both_streams_and_keeps_the_status():
    env = dict(os.environ, SG_TEST_TOKEN=SENTINEL)
    run = subprocess.run(
        [shutil.which("bash"), (GUARD / "guard_shell.sh").as_posix(),
         f"read line; echo \"in:$line\"; echo out {SENTINEL}; echo err {SENTINEL} >&2; exit 5"],
        input=b"hello\n", env=env, capture_output=True, timeout=60)
    assert run.returncode == 5
    assert run.stdout.decode() == f"in:hello\nout {P}\n"
    assert run.stderr.decode() == f"err {P}\n"


def test_the_installer_merges_its_settings_and_removes_exactly_them(tmp_path):
    path = tmp_path / "settings.json"
    mine = {"autoMode": True, "hooks": {"PostToolUse": [{"matcher": "Edit", "hooks": [
        {"type": "command", "command": "fmt"}]}]}}
    path.write_text(json.dumps(mine))
    target = tmp_path / "guard"
    sg_install.write_settings(path, target)
    merged = json.loads(path.read_text())
    assert merged["env"]["CLAUDE_CODE_SHELL_PREFIX"] == (target / "guard_shell.sh").as_posix()
    assert len(merged["hooks"]["PostToolUse"]) == 2 and merged["hooks"]["PostToolUseFailure"]
    sg_install.write_settings(path, target)                      # a second run replaces, never duplicates
    assert len(json.loads(path.read_text())["hooks"]["PostToolUse"]) == 2
    assert sg_install.remove_from(json.loads(path.read_text())) == mine


def test_no_checkpoint_load_unpickles_arbitrary_objects():
    """A leaked Hugging Face token can replace a checkpoint we download,
    and a full unpickling load would run whatever the replaced file holds.
    Every load takes tensors and plain containers only."""
    root = Path(__file__).resolve().parent.parent
    offenders = [str(p.relative_to(root)) for d in ("wesnoth_ai", "tools", "scripts", "tests")
                 for pattern in ("*.py", "*.sh") for p in (root / d).rglob(pattern)
                 if "weights_only=" + "False" in p.read_text(encoding="utf-8", errors="replace")]
    assert offenders == []
