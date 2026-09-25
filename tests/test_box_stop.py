"""scripts/box/box_stop.py: a box's stop counts only when Vast answers
`success: true`; the key moves from the Authorization header to the query
on a 4xx; failures are retried every interval up to the limit; the key never
reaches stdout, stderr or the outcome file. No test talks to Vast: the
transport is scripted, or a stub server on 127.0.0.1."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

import scripts.box.box_stop as box_stop

KEY = "SENTINEL0instance0key0never0printed0f00d"
IID = "4242"
ACCEPTED = (200, json.dumps({"success": True}))


class ScriptedVast:
    """Stands in for `send_put`: answers each request with the next reply
    (raising it when it is an exception) and records what was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.sent = []

    def __call__(self, url, headers, body, timeout):
        self.sent.append({"url": url, "headers": dict(headers), "body": body})
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    @property
    def forms(self):
        return ["bearer" if "Authorization" in s["headers"] else "query" for s in self.sent]


def run_stop(tmp_path, monkeypatch, vast, max_attempts=5):
    """(stopped, outcome records, the sleeps between attempts)."""
    monkeypatch.setattr(box_stop, "send_put", vast)
    naps = []
    path = tmp_path / "stop.jsonl"
    stopped = box_stop.stop_instance(IID, KEY, box_stop.Outcome(path, KEY), interval=7,
                                     max_attempts=max_attempts, sleep=naps.append)
    return stopped, [json.loads(line) for line in path.read_text().splitlines()], naps


def printed_and_recorded(capsys, tmp_path) -> str:
    out, err = capsys.readouterr()
    return out + err + (tmp_path / "stop.jsonl").read_text()


def test_the_header_form_accepted_stops_at_once(tmp_path, monkeypatch, capsys):
    vast = ScriptedVast(ACCEPTED)
    stopped, records, naps = run_stop(tmp_path, monkeypatch, vast)
    assert stopped and naps == []
    (sent,) = vast.sent
    assert sent["headers"]["Authorization"] == f"Bearer {KEY}"
    assert urlsplit(sent["url"]).path.endswith(f"/instances/{IID}/")
    assert KEY not in sent["url"]
    assert json.loads(sent["body"]) == {"state": "stopped"}
    assert records[-1]["event"] == "result" and records[-1]["stopped"] is True
    assert KEY not in printed_and_recorded(capsys, tmp_path)


def test_a_4xx_on_the_header_form_moves_the_key_to_the_query(tmp_path, monkeypatch, capsys):
    refusal = {"success": False, "msg": f"no instance for key {KEY}"}
    vast = ScriptedVast((404, json.dumps(refusal)), ACCEPTED)
    stopped, records, naps = run_stop(tmp_path, monkeypatch, vast)
    assert stopped and naps == []
    assert vast.forms == ["bearer", "query"]
    assert parse_qs(urlsplit(vast.sent[1]["url"]).query) == {"api_key": [KEY]}
    assert [r["status"] for r in records if r["event"] == "request"] == [404, 200]
    shown = printed_and_recorded(capsys, tmp_path)
    assert "no instance for key" in shown
    assert KEY not in shown


def test_success_false_is_no_stop_and_is_retried(tmp_path, monkeypatch, capsys):
    vast = ScriptedVast((200, json.dumps({"success": False, "msg": "instance is busy"})), ACCEPTED)
    stopped, records, naps = run_stop(tmp_path, monkeypatch, vast)
    assert stopped
    assert len(vast.sent) == 2 and naps == [7]
    assert [r["success"] for r in records if r["event"] == "request"] == [False, True]
    assert "instance is busy" in capsys.readouterr().out


def test_a_network_error_is_retried_and_its_message_never_shown(tmp_path, monkeypatch, capsys):
    leaky = ConnectionError(f"Max retries exceeded with url: /api/v0/instances/{IID}/?api_key={KEY}")
    vast = ScriptedVast((401, "{}"), leaky, ACCEPTED)
    stopped, records, naps = run_stop(tmp_path, monkeypatch, vast)
    assert stopped and naps == [7]
    assert vast.forms == ["bearer", "query", "query"]
    shown = printed_and_recorded(capsys, tmp_path)
    assert "ConnectionError" in shown
    assert KEY not in shown and "api_key" not in shown


def test_it_gives_up_after_the_limit_without_leaving_the_header_form(tmp_path, monkeypatch, capsys):
    vast = ScriptedVast((429, ""), (503, "<html>busy</html>"), (200, json.dumps({"success": False})))
    stopped, records, naps = run_stop(tmp_path, monkeypatch, vast, max_attempts=3)
    assert not stopped
    assert naps == [7, 7]
    assert vast.forms == ["bearer"] * 3
    assert records[-1]["event"] == "result" and records[-1]["stopped"] is False


def test_credentials_fall_back_to_pid1_environment(tmp_path):
    pid1 = tmp_path / "environ"
    pid1.write_bytes(b"PATH=/usr/bin\0CONTAINER_ID=777\0CONTAINER_API_KEY=" + KEY.encode() + b"\0")
    assert box_stop.instance_credentials({}, pid1) == ("777", KEY)
    assert box_stop.instance_credentials({"CONTAINER_ID": "5"}, pid1) == ("5", KEY)
    assert box_stop.instance_credentials({}, tmp_path / "absent") == (None, None)


@pytest.mark.parametrize("env, replies, code", [
    ({}, [], 2),
    ({"CONTAINER_ID": IID, "CONTAINER_API_KEY": KEY}, [ACCEPTED], 0),
    ({"CONTAINER_ID": IID, "CONTAINER_API_KEY": KEY}, [(200, json.dumps({"success": False}))], 1),
], ids=["no credentials", "accepted", "refused"])
def test_the_exit_status_says_whether_the_box_stopped(env, replies, code, tmp_path, monkeypatch,
                                                      capsys):
    for name in ("CONTAINER_ID", "CONTAINER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(box_stop, "PID1_ENVIRON", tmp_path / "absent")
    vast = ScriptedVast(*replies)
    monkeypatch.setattr(box_stop, "send_put", vast)
    path = tmp_path / "stop.jsonl"
    assert box_stop.main(["--outcome", str(path), "--max-attempts", "1"]) == code
    assert len(vast.sent) == len(replies)
    last = json.loads(path.read_text().splitlines()[-1])
    assert last["event"] == "result" and last["stopped"] is (code == 0)
    assert KEY not in printed_and_recorded(capsys, tmp_path)


class StubVast(BaseHTTPRequestHandler):
    """Refuses a key in the Authorization header with 404, as Vast answers a
    key it does not accept, and accepts the right key in the query."""

    def do_PUT(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        auth = self.headers.get("Authorization")
        self.server.seen.append({"path": url.path, "auth": auth, "query": query, "body": body})
        if auth is None and query.get("api_key") == [KEY]:
            self._answer(200, {"success": True, "msg": "stopping"})
        else:
            self._answer(404, {"success": False, "msg": "no such instance"})

    def _answer(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def stub_vast(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubVast)
    server.seen = []
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                     daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize("transport", ["_put_requests", "_put_urllib"])
def test_each_transport_against_a_stub_server(transport, stub_vast, tmp_path, monkeypatch, capsys):
    if transport == "_put_requests":
        pytest.importorskip("requests")
    monkeypatch.setattr(box_stop, "send_put", getattr(box_stop, transport))
    host, port = stub_vast.server_address
    path = tmp_path / "stop.jsonl"
    stopped = box_stop.stop_instance(IID, KEY, box_stop.Outcome(path, KEY), interval=0,
                                     max_attempts=2, api_base=f"http://{host}:{port}/api/v0",
                                     sleep=lambda seconds: None)
    assert stopped
    first, second = stub_vast.seen
    assert first["auth"] == f"Bearer {KEY}" and first["query"] == {}
    assert second["auth"] is None and second["query"] == {"api_key": [KEY]}
    assert first["path"] == second["path"] == f"/api/v0/instances/{IID}/"
    assert json.loads(second["body"]) == {"state": "stopped"}
    assert KEY not in printed_and_recorded(capsys, tmp_path)
