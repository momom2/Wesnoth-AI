#!/usr/bin/env python3
"""Stop the Vast instance this box runs on, with the credentials Vast puts
in the box's environment.

    python box_stop.py [--outcome PATH] [--interval SECONDS] [--max-attempts N]

Reads `CONTAINER_ID` and `CONTAINER_API_KEY` from the environment, each one
falling back to PID 1's environment (/proc/1/environ) when this process
lacks it, as a shell started over ssh can. Asks Vast to put the instance in
the STOPPED state: its GPU stops billing, its disk stays and bills storage.

Each attempt sends the key in an `Authorization: Bearer` header. When Vast
refuses that form with a 4xx other than 429, the same attempt sends the key
again as the `api_key` query parameter, the form the vastai SDK uses, and
later attempts keep that form. A stop counts only on a 2xx answer whose JSON
says `success: true`. Anything else (a refusal, `success: false`, a network
error) waits --interval seconds and tries again, for at most --max-attempts
attempts.

Output: stdout gets one line per request (the attempt, the key's form, and
the HTTP status and Vast's `msg`, or the exception's type name). The
--outcome file (appended, default `box_stop.jsonl` in the working directory)
gets the same as JSON lines, then one `{"event": "result", "stopped": ...}`
line; each line is written when it happens, so a killed run leaves what it
did. The key never reaches either: no exception message is printed (a
network error's message can quote the URL, which holds the key in the query
form), and the key is blanked from Vast's `msg`.

Exit status: 0 when Vast accepted the stop; 1 when every attempt failed (the
instance is still up and billing); 2 when the environment lacks the id or
the key.

In a box script: run it after the script's records are uploaded. Once Vast
accepts, the container is being stopped, so nothing after this command can
be relied on to run; when it exits non-zero the box is still up, and the
script can upload the outcome file:

    python /workspace/box_stop.py --outcome "$OUT/stop.jsonl" || upload_small

It needs only the standard library (it sends through `requests` when that is
installed, otherwise through urllib), so a script can download this file
from HF staging by itself, before its code stage, and still stop the box
when the stage fails to arrive.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus, urlencode

VAST_API = "https://console.vast.ai/api/v0"
PID1_ENVIRON = Path("/proc/1/environ")
REQUEST_TIMEOUT_S = 60
REDACTED = "<redacted>"


def _pid1_environment(path: Path) -> dict[str, str]:
    try:
        raw = path.read_bytes()
    except OSError:
        return {}
    pairs = (item.decode("utf-8", "replace").partition("=")
             for item in raw.split(b"\0") if item)
    return {name: value for name, _, value in pairs}


def instance_credentials(environ=None, pid1_environ: Path | None = None):
    """(instance id, key), each from `environ` (default: this process's)
    or else from PID 1's environment; None where neither has it."""
    environ = os.environ if environ is None else environ
    wanted = ("CONTAINER_ID", "CONTAINER_API_KEY")
    values = [environ.get(name, "").strip() for name in wanted]
    if not all(values):
        pid1 = _pid1_environment(pid1_environ or PID1_ENVIRON)
        values = [v or pid1.get(name, "").strip() for v, name in zip(values, wanted)]
    return tuple(v or None for v in values)


def _put_requests(url: str, headers: dict, body: bytes, timeout: float) -> tuple[int, str]:
    import requests
    response = requests.put(url, data=body, headers=headers, timeout=timeout)
    return response.status_code, response.text


def _put_urllib(url: str, headers: dict, body: bytes, timeout: float) -> tuple[int, str]:
    import urllib.error
    import urllib.request
    request = urllib.request.Request(
        url, data=body, method="PUT", headers={"User-Agent": "box_stop.py", **headers})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as refusal:
        return refusal.code, refusal.read().decode("utf-8", "replace")


def send_put(url: str, headers: dict, body: bytes, timeout: float) -> tuple[int, str]:
    """(HTTP status, answer text) of a PUT; raises on a network failure.
    `requests` when installed: the self-stop was first proven with it."""
    if importlib.util.find_spec("requests") is None:
        return _put_urllib(url, headers, body, timeout)
    return _put_requests(url, headers, body, timeout)


@dataclass
class Reply:
    """One request's outcome, holding nothing that can carry the key."""
    auth: str                   # "bearer" or "query": where the key went
    status: int | None = None   # None when the request raised
    success: object = None      # the answer's `success` field
    msg: str | None = None      # the answer's `msg` field
    error: str | None = None    # the exception's type name

    @property
    def accepted(self) -> bool:
        return (self.status is not None and 200 <= self.status < 300
                and self.success is True)

    @property
    def refused_form(self) -> bool:
        """A 4xx other than 429: Vast declined the request as sent."""
        return self.status is not None and 400 <= self.status < 500 and self.status != 429


def vast_answer(text: str) -> tuple[object, str | None]:
    """(`success`, `msg`) of Vast's JSON answer; (None, None) for anything
    that is not a JSON object."""
    try:
        data = json.loads(text)
    except ValueError:
        return None, None
    if not isinstance(data, dict):
        return None, None
    msg = data.get("msg")
    return data.get("success"), (None if msg is None else str(msg))


def request_stop(url: str, key: str, auth: str) -> Reply:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if auth == "bearer":
        headers["Authorization"] = f"Bearer {key}"
    else:
        url = f"{url}?{urlencode({'api_key': key})}"
    body = json.dumps({"state": "stopped"}).encode()
    try:
        status, text = send_put(url, headers, body, REQUEST_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 -- the type name only: the message can quote the keyed URL
        return Reply(auth, error=type(exc).__name__)
    success, msg = vast_answer(text)
    return Reply(auth, status, success, msg)


class Outcome:
    """Each request and the result, to stdout and as JSON lines to `path`,
    with `secret` blanked from both."""

    def __init__(self, path: Path, secret: str | None = None):
        self.path = path
        self._secrets = {s for s in (secret, secret and quote_plus(secret)) if s}

    def _scrub(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, REDACTED)
        return text

    def _write(self, record: dict, line: str) -> None:
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record}
        print(self._scrub(f"box_stop: {line}"), flush=True)
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(self._scrub(json.dumps(record)) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError as exc:
            print(f"box_stop: cannot write {self.path} ({type(exc).__name__})", flush=True)

    def request(self, attempt: int, reply: Reply) -> None:
        heard = (f"HTTP {reply.status}, success={reply.success}"
                 + (f", msg: {reply.msg}" if reply.msg else "")
                 if reply.status is not None else reply.error)
        self._write({"event": "request", "attempt": attempt, "auth": reply.auth,
                     "status": reply.status, "success": reply.success, "msg": reply.msg,
                     "error": reply.error},
                    f"attempt {attempt} ({reply.auth}): {heard}")

    def result(self, instance_id: str | None, stopped: bool, detail: str) -> None:
        self._write({"event": "result", "instance": instance_id, "stopped": stopped,
                     "detail": detail},
                    f"instance {instance_id}: {'STOPPED' if stopped else 'NOT stopped'} ({detail})")


def stop_instance(instance_id: str, key: str, outcome: Outcome, *, interval: float = 30.0,
                  max_attempts: int = 60, api_base: str = VAST_API, sleep=time.sleep) -> bool:
    """Ask Vast to stop `instance_id` until it accepts or `max_attempts`
    attempts have failed; True when it accepted."""
    url = f"{api_base}/instances/{instance_id}/"
    auth = "bearer"
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            sleep(interval)
        reply = request_stop(url, key, auth)
        outcome.request(attempt, reply)
        if auth == "bearer" and reply.refused_form:
            auth = "query"
            reply = request_stop(url, key, auth)
            outcome.request(attempt, reply)
        if reply.accepted:
            outcome.result(instance_id, True, f"accepted at attempt {attempt}")
            return True
    outcome.result(instance_id, False, f"{max_attempts} attempts failed; the instance is still up")
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stop this Vast instance (see the module docstring).")
    ap.add_argument("--outcome", default="box_stop.jsonl",
                    help="JSON-lines record of every request and the result, appended")
    ap.add_argument("--interval", type=float, default=30.0, help="seconds between attempts")
    ap.add_argument("--max-attempts", type=int, default=60,
                    help="attempts before giving up; one request each, two when the "
                         "header form is refused")
    args = ap.parse_args(argv)
    instance_id, key = instance_credentials()
    outcome = Outcome(Path(args.outcome), key)
    if not instance_id or not key:
        missing = " and ".join(name for name, v in (("CONTAINER_ID", instance_id),
                                                   ("CONTAINER_API_KEY", key)) if not v)
        outcome.result(instance_id, False, f"no {missing} in the environment")
        return 2
    print(f"box_stop: stopping instance {instance_id}", flush=True)
    stopped = stop_instance(instance_id, key, outcome, interval=args.interval,
                            max_attempts=args.max_attempts)
    return 0 if stopped else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- a message could quote a URL holding the key
        print(f"box_stop: stopped by an unexpected {type(exc).__name__}", flush=True)
        sys.exit(1)
