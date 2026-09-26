"""scripts/rent_box.py: a failed remote call prints neither the account key
nor a URL, Vast's `success: false` is a failure, and `create` rents only
when its preflight passes. No test talks to Vast or Hugging Face: the SDK
and HF staging are fakes, and one test drives the real SDK against a stub
server on 127.0.0.1."""
from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import scripts.rent_box as rent_box
from scripts.box.box_stage import LIBRARY_FILES, library_dir

KEY = "SENTINEL0account0key0never0printed0beef"
KEYED_PATH = f"/api/v0/instances/7/?owner=me&api_key={KEY}"
SCRIPT = "demo_box.sh"
STAGE = "tier-b/staging/stage_demo.tar.gz"
SCRIPT_TEXT = f'#!/usr/bin/env bash\nset -uo pipefail\nSTAGE="${{STAGE:-{STAGE}}}"\n'
ON_HF = {rent_box.STAGING + SCRIPT: SCRIPT_TEXT, STAGE: "<tarball>"}


@pytest.fixture(autouse=True)
def no_remembered_secrets(monkeypatch):
    """Each test starts with no secret to scrub, so a leak test sees exactly
    what would be printed without that second defence."""
    monkeypatch.setattr(rent_box, "_SECRETS", set())


def http_error():
    """What the SDK's raise_for_status() raises: its message quotes the URL."""
    requests = pytest.importorskip("requests")
    response = requests.Response()
    response.status_code, response.reason = 404, "Not Found"
    response.url = "https://console.vast.ai" + KEYED_PATH
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        return exc
    raise AssertionError("raise_for_status() did not raise")


def network_error():
    """What the SDK re-raises once its retries are spent: urllib3 quotes
    the URL."""
    requests = pytest.importorskip("requests")
    from urllib3.exceptions import MaxRetryError
    return requests.ConnectionError(MaxRetryError(None, KEYED_PATH, reason=OSError("refused")))


class RaisingSdk:
    """Every method raises `error`, as the SDK does on an HTTP or network
    failure."""

    def __init__(self, error):
        self.error = error

    def __getattr__(self, name):
        def method(*args, **kwargs):
            raise self.error
        return method


class AnsweringSdk:
    """Each method answers from `answers` and records its keyword arguments."""

    def __init__(self, **answers):
        self.answers = answers
        self.calls = []

    def __getattr__(self, name):
        if name not in self.__dict__.get("answers", {}):
            raise AttributeError(name)

        def method(*args, **kwargs):
            self.calls.append((name, kwargs))
            return self.answers[name]
        return method

    def called(self, name):
        return [kwargs for called, kwargs in self.calls if called == name]


class FakeStaging:
    """HF staging holding `files`, a map from path to text."""

    def __init__(self, files):
        self.files = files

    def exists(self, path):
        return path in self.files

    def read_text(self, path):
        return self.files[path]


def use(monkeypatch, sdk, on_hf=ON_HF):
    monkeypatch.setattr(rent_box, "_vast", lambda: sdk)
    monkeypatch.setattr(rent_box, "_hf_token", lambda: "hf_" + "t" * 34)
    monkeypatch.setattr(rent_box, "_staging", lambda token: FakeStaging(on_hf))


def market(price=0.5, hours_left=100.0, credit=20.0, balance=0.0):
    offer = {"id": 99, "dph_total": price, "end_date": time.time() + hours_left * 3600}
    return AnsweringSdk(search_offers=[offer], show_user={"credit": credit, "balance": balance},
                        create_instance={"success": True, "new_contract": 555})


def rent(monkeypatch, sdk, on_hf=ON_HF):
    """`create` for a 4-hour run: at $0.5/h the preflight needs $3.00 and a
    6-hour rental window."""
    use(monkeypatch, sdk, on_hf)
    return rent_box.main(["create", "99", "--onstart", SCRIPT, "--hours", "4"])


COMMANDS = [["search"], ["status"], ["status", "7"], ["logs", "7"], ["start", "7"],
            ["stop", "7"], ["destroy", "7"], ["create", "99", "--onstart", SCRIPT, "--hours", "2"]]


@pytest.mark.parametrize("make_error, shown", [(http_error, "HTTP 404"),
                                               (network_error, "ConnectionError")],
                         ids=["http", "network"])
@pytest.mark.parametrize("argv", COMMANDS, ids=" ".join)
def test_a_failed_call_prints_only_its_status_or_type(argv, make_error, shown, monkeypatch,
                                                      capsys):
    error = make_error()
    assert KEY in str(error), "the fake must quote the key as the SDK's errors do"
    use(monkeypatch, RaisingSdk(error))
    assert rent_box.main(argv) == 1
    out, err = capsys.readouterr()
    assert shown in err
    for leak in (KEY, "api_key", "console.vast.ai"):
        assert leak not in out + err


class RefusingServer(BaseHTTPRequestHandler):
    """Answers every request 404, as Vast does for an id it does not know."""

    def _refuse(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = b'{"success": false, "msg": "not found"}'
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_PUT = do_POST = do_DELETE = _refuse

    def log_message(self, *args):
        pass


def test_the_real_sdk_error_is_reduced_to_its_status(monkeypatch, capsys):
    vastai = pytest.importorskip("vastai")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    server = ThreadingHTTPServer(("127.0.0.1", 0), RefusingServer)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                     daemon=True).start()
    try:
        host, port = server.server_address
        sdk = vastai.VastAI(api_key=KEY, server_url=f"http://{host}:{port}")
        monkeypatch.setattr(rent_box, "_vast", lambda: sdk)
        assert rent_box.main(["stop", "7"]) == 1
    finally:
        server.shutdown()
        server.server_close()
    out, err = capsys.readouterr()
    assert "HTTP 404" in err
    assert KEY not in out + err


@pytest.mark.parametrize("command, method", [("start", "start_instance"),
                                             ("stop", "stop_instance"),
                                             ("destroy", "destroy_instance")])
@pytest.mark.parametrize("success", [True, False])
def test_only_success_true_counts(command, method, success, monkeypatch, capsys):
    sdk = AnsweringSdk(**{method: {"success": success, "msg": f"instance 7, key {KEY}"}})
    use(monkeypatch, sdk)
    rent_box.remember_secret(KEY)
    assert rent_box.main([command, "7"]) == (0 if success else 1)
    assert sdk.called(method) == [{"id": 7}]
    out, err = capsys.readouterr()
    assert "instance 7, key" in (out if success else err)
    assert KEY not in out + err


@pytest.mark.parametrize("credit, balance", [(20.0, 0.0), (20.0, 19.0), (20.0, -1.0)])
def test_a_covered_run_is_rented_under_its_script_name(credit, balance, monkeypatch):
    sdk = market(credit=credit, balance=balance)
    assert rent(monkeypatch, sdk) == 0
    (created,) = sdk.called("create_instance")
    assert created["id"] == 99 and created["label"] == SCRIPT
    (lookup,) = sdk.called("search_offers")
    assert lookup["query"] == "id=99" and lookup["storage"] == 40


@pytest.mark.parametrize("on_hf, account, reason", [
    ({STAGE: "<tarball>"}, {}, f"{SCRIPT} is not on HF"),
    ({rent_box.STAGING + SCRIPT: SCRIPT_TEXT}, {}, f"{STAGE} (the script's default) is not on HF"),
    ({rent_box.STAGING + SCRIPT: "#!/usr/bin/env bash\n"}, {}, "names no STAGE default"),
    (ON_HF, {"credit": 2.0}, "do not cover $3.00"),
    (ON_HF, {"credit": 10.0, "balance": -8.0}, "do not cover $3.00"),
    (ON_HF, {"credit": 0.0, "balance": -0.19}, "do not cover $3.00"),
    (ON_HF, {"hours_left": 5.0}, "leaves the market in 5.0 h"),
], ids=["script missing", "stage missing", "stage unknown", "credit short", "balance owed",
        "account in debt", "window short"])
def test_a_failed_check_refuses_to_rent(on_hf, account, reason, monkeypatch, capsys):
    sdk = market(**account)
    assert rent(monkeypatch, sdk, on_hf) == 1
    assert sdk.called("create_instance") == []
    err = capsys.readouterr().err
    assert "refusing to rent" in err and reason in err


def test_printed_answers_carry_no_secret(monkeypatch, capsys):
    sdk = market()
    sdk.answers["create_instance"] = {"success": True, "new_contract": 555,
                                      "instance_api_key": "k" * 40}
    sdk.answers["logs"] = f"container up\nexport SOME_TOKEN={KEY}\n"
    use(monkeypatch, sdk)
    rent_box.remember_secret(KEY)
    assert rent_box.main(["create", "99", "--onstart", SCRIPT, "--hours", "4"]) == 0
    assert rent_box.main(["logs", "7"]) == 0
    out, err = capsys.readouterr()
    assert "555" in out and "container up" in out
    assert "k" * 40 not in out + err
    assert KEY not in out + err


# ---- the box library ---------------------------------------------------------
LIBRARY = library_dir(STAGE)
LIBRARY_SCRIPT = ('#!/usr/bin/env bash\nset -uo pipefail\nSTAGE="${STAGE:-}"\n'
                  '. "${BOX_LIB:-/workspace/box}/boxlib.sh" || exit 1\nbox_init\n')
LIBRARY_ON_HF = {rent_box.STAGING + SCRIPT: LIBRARY_SCRIPT, STAGE: "<tarball>",
                 **{f"{LIBRARY}/{name}": "<file>" for name in LIBRARY_FILES}}


def bash_parses(command: str) -> bool:
    bash = shutil.which("bash")
    if bash is None or (sys.platform == "win32" and "system32" in bash.lower()):
        pytest.skip("no POSIX bash on this machine")
    return subprocess.run([bash, "-n", "-c", command]).returncode == 0


def test_a_library_script_boots_through_its_stages_library(monkeypatch):
    sdk = market()
    use(monkeypatch, sdk, LIBRARY_ON_HF)
    assert rent_box.main(["create", "99", "--onstart", SCRIPT, "--hours", "4", "--stage", STAGE]) == 0
    (created,) = sdk.called("create_instance")
    assert created["env"]["STAGE"] == STAGE
    onstart = created["onstart_cmd"]
    assert f"'{LIBRARY}/box_onstart.sh'" in onstart
    assert f"bash /workspace/box_onstart.sh {SCRIPT} {LIBRARY} >" in onstart
    assert bash_parses(onstart)


def test_any_other_script_is_fetched_and_run_as_before(monkeypatch):
    sdk = market()
    assert rent(monkeypatch, sdk) == 0
    (created,) = sdk.called("create_instance")
    onstart = created["onstart_cmd"]
    assert f"'{rent_box.STAGING}{SCRIPT}'" in onstart
    assert f"bash /workspace/{SCRIPT} >" in onstart and "box_onstart" not in onstart
    assert created["env"]["STAGE"] == STAGE          # the script's default, checked on HF
    assert bash_parses(onstart)


@pytest.mark.parametrize("on_hf, argv, reason", [
    ({k: v for k, v in LIBRARY_ON_HF.items() if not k.endswith("/boxlib.sh")},
     ["--stage", STAGE], "lacks boxlib.sh"),
    (LIBRARY_ON_HF, ["--stage", "none"], "runs on the box library"),
    (LIBRARY_ON_HF, [], "names no STAGE default"),
    (LIBRARY_ON_HF, ["--stage", STAGE, "--env", "STAGE=tier-b/staging/other.tar.gz"], "disagree"),
    ({**LIBRARY_ON_HF, "tier-b/staging/stage_$(reboot).tar.gz": "<tarball>"},
     ["--stage", "tier-b/staging/stage_$(reboot).tar.gz"], "cannot carry"),
], ids=["library file missing", "no stage", "no default", "two stages", "unsafe stage"])
def test_a_library_script_without_its_whole_library_is_refused(on_hf, argv, reason, monkeypatch,
                                                               capsys):
    sdk = market()
    use(monkeypatch, sdk, on_hf)
    assert rent_box.main(["create", "99", "--onstart", SCRIPT, "--hours", "4", *argv]) == 1
    assert sdk.called("create_instance") == []
    err = capsys.readouterr().err
    assert "refusing to rent" in err and reason in err
