#!/usr/bin/env python3
"""Rent, inspect, stop and destroy Vast boxes through the SDK (the `vastai`
CLI binary is blocked on the laptop).

    python scripts/rent_box.py search [--gpu "RTX 4090"] [--max-dph 0.8] [--min-cpu 16] [--min-hours 4]
    python scripts/rent_box.py create OFFER_ID --onstart HF_STAGING_SCRIPT [--hours N] [--stage PATH] [--disk 40] [--env K=V ...]
    python scripts/rent_box.py status [INSTANCE_ID]
    python scripts/rent_box.py logs INSTANCE_ID [--tail N] [--daemon-logs]
    python scripts/rent_box.py start INSTANCE_ID
    python scripts/rent_box.py stop INSTANCE_ID
    python scripts/rent_box.py destroy INSTANCE_ID

`create` starts the pytorch 2.5.1 image with ssh, labels the instance with
the script's name, passes HF_TOKEN (the laptop's huggingface token) and the
code stage as STAGE as environment variables, and an onstart that runs the
script detached under /workspace (unattended bring-up, docs/box_runbook.md).
For a script on the box library (it sources scripts/box/boxlib.sh) the
onstart fetches box_onstart.sh from the stage's library side copy on HF
(`<stage>.box/`, written by `tools/stage_code.py --upload`), and
box_onstart.sh fetches the rest of the library, then the script; for any
other script it fetches the script from HF `tier-b/staging/`. Offers are
filtered client-side: no VM hosts (they refuse ssh), a remaining rental
window of at least --min-hours. Prices are for --disk GB of storage, the
disk `create` rents.

Before renting, `create` checks the following, and refuses with the reasons
when any check fails:
  * the onstart script is on HF staging, and so is the code stage it
    downloads: --stage, else STAGE from --env, else the script's own
    `STAGE="${STAGE:-...}"` default (`--stage none` for a script that
    downloads no code); for a script on the box library, so is every file
    of the stage's library side copy;
  * the account's funds cover 1.5 x --hours of the offer's price (the
    project rule since a run was stopped by credit at 3.5 h of 6,
    2026-09-24), and the offer's rental window has 1.5 x --hours left.
    Without --hours it refuses only an account with no funds, and prints
    how many hours the funds and the window cover.

`start`, `stop` and `destroy` succeed only when Vast answers
`success: true`; otherwise they print Vast's `msg` and exit 1. A stopped
instance keeps its disk and pays for its storage; a destroyed one does not.

A create that answers `success: false` with a contract id has made the
instance in the STOPPED state (2026-09-14: three in a row showed
`intended_status: stopped` and sat in "loading" for half an hour);
`start` brings it up, and the onstart runs then, fetching the script
as it is on HF at that moment.

Secrets: the SDK puts the account key in every URL it requests, and
`requests` and urllib3 quote that URL in their exception messages. Every
remote call therefore goes through `api_call`, which reports a failure by
its HTTP status code or the exception's type name only, and every answer is
printed through `redacted`, which blanks key- and token-named fields and
the secrets this process holds.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts.box.box_stage import LIBRARY_FILES, library_dir
except ImportError:              # run as a script: scripts/ is on the path, the root is not
    from box.box_stage import LIBRARY_FILES, library_dir

IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
HF_REPO = "momom2/wesnoth-model-checkpoints"
STAGING = "tier-b/staging/"
# The onstart fetches one file from HF staging, writes the token file and
# starts the file detached: the script itself, or for a script on the box
# library its bring-up (scripts/box/box_onstart.sh), given the script and
# the library's HF folder.
ONSTART_FETCH = (
    "cd /workspace && python -m pip install -q huggingface_hub >/dev/null 2>&1; "
    "python -c \"from huggingface_hub import hf_hub_download as d; import shutil, os; "
    "shutil.copyfile(d('momom2/wesnoth-model-checkpoints', '{source}', "
    "token=os.environ['HF_TOKEN']), '/workspace/{target}')\" && "
    "printf '%s' \"$HF_TOKEN\" > /workspace/.hf_token && chmod 600 /workspace/.hf_token && "
    "(setsid nohup bash /workspace/{target}{arguments} > /workspace/onstart_script.log 2>&1 < /dev/null &)"
)
# Script names and HF paths go into the onstart's shell line unquoted.
_SHELL_SAFE = re.compile(r"^[A-Za-z0-9._/-]+$")
_SOURCES_LIBRARY = re.compile(r"^\s*(?:\.|source)\s+\S*boxlib\.sh", re.MULTILINE)
# The account's funds and the offer's rental window must both cover this
# multiple of the run's estimated hours.
MARGIN = 1.5
REDACTED = "<redacted>"
_SECRET_FIELD = re.compile(r"key|token|secret|password", re.IGNORECASE)
_API_KEY_PARAM = re.compile(r"(api_key=)[^&\s'\"]+")
_HF_TOKEN_TEXT = re.compile(r"\bhf_[A-Za-z0-9]{20,}")
_STAGE_DEFAULT = re.compile(
    r"""^\s*(?:export\s+)?STAGE=["']?(?:\$\{STAGE:-)?([^}"'\s$]+)""", re.MULTILINE)
_SECRETS: set[str] = set()


class ApiCallFailed(Exception):
    """A remote call raised. The message names the call and its HTTP status
    code or the exception's type name, nothing else."""


def failure_reason(exc: BaseException) -> str:
    """The HTTP status code of a failed call, else the exception's type
    name; never the message, which can quote the URL and the key in it."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return f"HTTP {status}" if isinstance(status, int) else type(exc).__name__


def api_call(what: str, fn, *args, **kwargs):
    """`fn(*args, **kwargs)`, or ApiCallFailed carrying only `what` and the
    failure's status code or type name. The original exception is not
    chained, so no traceback can print its message."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- any SDK exception may quote the keyed URL
        reason = failure_reason(exc)
    raise ApiCallFailed(f"{what} failed ({reason})")


def remember_secret(value) -> None:
    """Blank `value` wherever this process prints it."""
    if value:
        _SECRETS.add(str(value))


def scrub_text(text: str) -> str:
    for secret in _SECRETS:
        text = text.replace(secret, REDACTED)
    text = _API_KEY_PARAM.sub(r"\g<1>" + REDACTED, text)
    return _HF_TOKEN_TEXT.sub(REDACTED, text)


def redacted(obj):
    """`obj` with its key- and token-named fields blanked and every known
    secret removed from its text, at any depth: Vast's create answer holds
    the new instance's API key, and an instance's env holds HF_TOKEN."""
    if isinstance(obj, dict):
        return {k: (REDACTED if _SECRET_FIELD.search(str(k)) else redacted(v))
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redacted(v) for v in obj]
    if isinstance(obj, str):
        return scrub_text(obj)
    return obj


def _vast():
    from vastai import VastAI
    sdk = api_call("reading the Vast API key", VastAI)
    remember_secret(getattr(getattr(sdk, "client", None), "api_key", None))
    return sdk


def _hf_token() -> str | None:
    from huggingface_hub import get_token
    return get_token()


class HfStaging:
    """The two questions the preflight asks the model host."""

    def __init__(self, token: str):
        from huggingface_hub import HfApi
        self._api = HfApi(token=token)

    def exists(self, path: str) -> bool:
        return self._api.file_exists(HF_REPO, path)

    def read_text(self, path: str) -> str:
        local = self._api.hf_hub_download(HF_REPO, path)
        return Path(local).read_text(encoding="utf-8", errors="replace")


def _staging(token: str) -> HfStaging:
    return HfStaging(token)


def search(args) -> int:
    v = _vast()
    q = (f"gpu_name={args.gpu.replace(' ', '_')} num_gpus=1 cpu_cores_effective>={args.min_cpu} "
         f"dph_total<={args.max_dph} reliability>0.95 disk_space>={args.disk} "
         f"rentable=true rented=false verified=true inet_down>200 inet_up>50")
    offers = api_call("offer search", v.search_offers, query=q, order="dph_total",
                      storage=args.disk)
    if not isinstance(offers, list):
        print(redacted(offers))
        return 1
    now = time.time()
    rows = []
    for o in offers:
        if o.get("vms_enabled"):
            continue
        end = o.get("end_date") or 0
        hours = (end - now) / 3600 if end else 1e9
        if hours < args.min_hours:
            continue
        rows.append(o)
        print(f"{o['id']:>10}  {o.get('gpu_name'):12s}  ${o.get('dph_total', 0):.3f}/h  "
              f"{o.get('cpu_cores_effective', 0):5.1f} cores  {(o.get('cpu_name') or '')[:28]:28s}  "
              f"{o.get('geolocation', ''):18s}  up {o.get('inet_up', 0):6.0f} Mb/s  "
              f"disk {o.get('disk_space', 0):5.0f}  {hours:6.1f} h left")
        if len(rows) >= args.limit:
            break
    return 0


def stage_default(script_text: str) -> str | None:
    """The code stage a box script downloads unless STAGE overrides it:
    the default of its `STAGE="${STAGE:-...}"` line, or a plain
    `STAGE=...` value."""
    m = _STAGE_DEFAULT.search(script_text)
    return m.group(1) if m else None


def uses_library(script_text: str) -> bool:
    """The script sources the box library (scripts/box/boxlib.sh)."""
    return _SOURCES_LIBRARY.search(script_text) is not None


@dataclass
class OnstartPlan:
    """What the box will fetch, as the preflight found it on HF."""
    stage: str | None            # the code stage; None for --stage none
    library: str | None          # the stage's library side copy, for a script on the library


def staging_problems(staging, script: str, stage: str | None) -> tuple[list[str], OnstartPlan | None]:
    """(problems, plan): why the script, its code stage or its box library
    would be missing on the box, and otherwise what the onstart fetches. The
    onstart downloads the script (or, for a script on the box library, the
    library's bring-up), and the script downloads its stage."""
    if not _SHELL_SAFE.match(script):
        return [f"the script name {script!r} holds characters the onstart cannot carry"], None
    script_path = STAGING + script
    if not api_call(f"HF lookup of {script_path}", staging.exists, script_path):
        return [f"the onstart script {script_path} is not on HF (or this token cannot "
                f"read {HF_REPO}): upload it first"], None
    print(f"preflight: {script_path} is on HF")
    text = api_call(f"HF download of {script_path}", staging.read_text, script_path)
    on_library = uses_library(text)
    source = "given"
    if stage is None:
        stage, source = stage_default(text), "the script's default"
        if stage is None:
            return [f"{script} names no STAGE default: pass --stage PATH, "
                    f"or --stage none if it downloads no code"], None
    if stage == "none":
        if on_library:
            return [f"{script} runs on the box library, which comes with its code stage: "
                    f"pass --stage PATH"], None
        print("preflight: no code stage to check (--stage none)")
        return [], OnstartPlan(None, None)
    if not _SHELL_SAFE.match(stage):
        return [f"the code stage {stage!r} holds characters the onstart cannot carry"], None
    if not api_call(f"HF lookup of {stage}", staging.exists, stage):
        return [f"the code stage {stage} ({source}) is not on HF: build and upload it "
                f"(tools/stage_code.py --upload)"], None
    print(f"preflight: code stage {stage} ({source}) is on HF")
    if not on_library:
        return [], OnstartPlan(stage, None)
    try:
        library = library_dir(stage)
    except ValueError:
        return [f"the code stage {stage} is not a .tar.gz, so it has no box library"], None
    missing = [name for name in LIBRARY_FILES
               if not api_call(f"HF lookup of {library}/{name}", staging.exists, f"{library}/{name}")]
    if missing:
        return [f"the box library of {stage} lacks {', '.join(missing)} on HF ({library}/): "
                f"`tools/stage_code.py --upload` writes it beside the stage"], None
    print(f"preflight: the stage's box library is on HF ({library}/)")
    return [], OnstartPlan(stage, library)


def onstart_command(script: str, library: str | None) -> str:
    """The onstart for `script`: its box library's bring-up when `library`
    is the library's HF folder, otherwise the script from HF staging."""
    if library is None:
        return ONSTART_FETCH.format(source=STAGING + script, target=script, arguments="")
    return ONSTART_FETCH.format(source=f"{library}/box_onstart.sh", target="box_onstart.sh",
                                arguments=f" {script} {library}")


def spendable(user) -> float | None:
    """Dollars the account can spend before Vast stops its instances, or
    None when `show_user()` reports neither `credit` nor `balance`.

    The SDK does not say which of the two falls as charges accrue (the box
    Vast stopped on 2026-09-24 read credit 0, balance -0.19). This takes the
    lower of the readings those two fields allow: a negative balance is
    owed against the credit, and of two positive fields the smaller counts.
    """
    if not isinstance(user, dict):
        return None
    credit, balance = user.get("credit"), user.get("balance")
    numbers = [x for x in (credit, balance) if isinstance(x, (int, float))]
    if len(numbers) < 2:
        return numbers[0] if numbers else None
    return credit + balance if balance <= 0 else min(credit, balance)


def find_offer(v, offer_id: int, disk: int) -> dict | None:
    """The offer as it is on the market now, priced for `disk` GB."""
    offers = api_call("offer lookup", v.search_offers, query=f"id={offer_id}",
                      no_default=True, storage=disk)
    for o in offers if isinstance(offers, list) else []:
        if isinstance(o, dict) and o.get("id") == offer_id:
            return o
    return None


def budget_problems(v, offer_id: int, hours: float | None, disk: int) -> list[str]:
    """Why the account or the offer cannot carry the run: funds below
    MARGIN x hours x price, or a rental window shorter than MARGIN x hours.
    Without `hours`, only an account with no funds is refused."""
    offer = find_offer(v, offer_id, disk)
    if offer is None:
        return [f"offer {offer_id} is not on the market any more"]
    price = offer.get("dph_total")
    if not isinstance(price, (int, float)) or price <= 0:
        return [f"offer {offer_id} has no price"]
    user = api_call("show_user", v.show_user)
    funds = spendable(user)
    if funds is None:
        return ["show_user() reports neither credit nor balance"]
    end = offer.get("end_date")
    window = (end - time.time()) / 3600 if end else None
    print(f"preflight: offer {offer_id} at ${price:.3f}/h with {disk} GB; funds "
          f"${funds:.2f} (credit {user.get('credit')}, balance {user.get('balance')}) "
          f"cover {funds / (MARGIN * price):.1f} h at the {MARGIN}x margin; rental window "
          + (f"{window:.1f} h" if window is not None else "open-ended"))
    need = MARGIN * (hours or 0) * price
    problems = []
    if funds <= 0 or funds < need:
        cost = (f"${need:.2f} ({MARGIN} x {hours:g} h x ${price:.3f}/h)" if hours
                else "any time")
        problems.append(f"funds ${funds:.2f} do not cover {cost}: top up first")
    if hours is None:
        print("preflight: no --hours, so the run's cost and length are NOT checked")
    elif window is not None and window < MARGIN * hours:
        problems.append(f"offer {offer_id} leaves the market in {window:.1f} h, "
                        f"under {MARGIN} x {hours:g} h")
    return problems


def create(args) -> int:
    tok = _hf_token()
    if not tok:
        print("no huggingface token on this machine", file=sys.stderr)
        return 1
    remember_secret(tok)
    env = {"HF_TOKEN": tok}
    for kv in args.env or []:
        k, _, val = kv.partition("=")
        env[k] = val
        if _SECRET_FIELD.search(k):
            remember_secret(val)
    if args.stage and "STAGE" in env and env["STAGE"] != args.stage:
        print(f"refusing to rent: --stage {args.stage} and --env STAGE={env['STAGE']} disagree",
              file=sys.stderr)
        return 1
    v = _vast()
    stage = args.stage or env.get("STAGE") or None
    problems, plan = staging_problems(_staging(tok), args.onstart, stage)
    problems += budget_problems(v, args.offer_id, args.hours, args.disk)
    if problems:
        print("refusing to rent:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    if plan.stage:
        env["STAGE"] = plan.stage          # the box downloads the stage the preflight checked
    res = api_call("create", v.create_instance, id=args.offer_id, image=IMAGE, disk=args.disk,
                   runtype="ssh_direc", env=env, label=args.onstart,
                   onstart_cmd=onstart_command(args.onstart, plan.library))
    print(json.dumps(redacted(res), default=str))
    iid = res.get("new_contract") if isinstance(res, dict) else None
    if not (isinstance(res, dict) and res.get("success") is True):
        if iid:
            print(f"instance {iid} was made STOPPED; `rent_box.py start {iid}` brings it up",
                  file=sys.stderr)
        return 1
    print("instance", iid)
    return 0


def answered(action: str, res, instance_id: int) -> int:
    """0 when Vast answered `success: true`, else 1 with Vast's message."""
    if isinstance(res, dict) and res.get("success") is True:
        msg = res.get("msg")
        print(f"{action} instance {instance_id}: accepted"
              + (f" ({redacted(str(msg))})" if msg else ""))
        return 0
    msg = res.get("msg", res) if isinstance(res, dict) else res
    print(f"{action} instance {instance_id}: REFUSED by Vast: {redacted(msg)}", file=sys.stderr)
    return 1


def start(args) -> int:
    v = _vast()
    return answered("start", api_call("start", v.start_instance, id=args.instance_id),
                    args.instance_id)


def stop(args) -> int:
    v = _vast()
    return answered("stop", api_call("stop", v.stop_instance, id=args.instance_id),
                    args.instance_id)


def destroy(args) -> int:
    v = _vast()
    return answered("destroy", api_call("destroy", v.destroy_instance, id=args.instance_id),
                    args.instance_id)


def logs(args) -> int:
    v = _vast()
    res = api_call("logs", v.logs, args.instance_id,
                   tail=str(args.tail) if args.tail else None, daemon_logs=args.daemon_logs)
    if isinstance(res, dict) and res.get("success") is not True:
        return answered("logs of", res, args.instance_id)
    print(redacted(res))
    return 0


def status(args) -> int:
    v = _vast()
    if args.instance_id:
        i = api_call("show_instance", v.show_instance, id=args.instance_id)
        rows = [i] if isinstance(i, dict) else []
    else:
        i = api_call("show_instances", v.show_instances)
        rows = i if isinstance(i, list) else []
    if not rows:
        print(redacted(i))
        return 0
    for r in rows:
        direct = ((r.get("ports") or {}).get("22/tcp") or [{}])[0].get("HostPort")
        print(scrub_text(
            f"{r.get('id')}  {r.get('actual_status')}  {r.get('gpu_name')}  "
            f"${r.get('dph_total') or 0:.3f}/h  ssh {r.get('ssh_host')}:{r.get('ssh_port')}  "
            f"direct {r.get('public_ipaddr')}:{direct}  {(r.get('status_msg') or '')[:60]}"))
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Rent, inspect, stop and destroy Vast boxes.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search", help="rentable offers, cheapest first")
    s.add_argument("--gpu", default="RTX 4090")
    s.add_argument("--max-dph", type=float, default=0.8)
    s.add_argument("--min-cpu", type=int, default=16)
    s.add_argument("--min-hours", type=float, default=4.0)
    s.add_argument("--disk", type=int, default=40)
    s.add_argument("--limit", type=int, default=12)
    s.set_defaults(fn=search)
    c = sub.add_parser("create", help="rent an offer once the preflight passes")
    c.add_argument("offer_id", type=int)
    c.add_argument("--onstart", required=True, help="script name under tier-b/staging/")
    c.add_argument("--hours", type=float,
                   help="the run's estimated hours; the funds and the offer's rental "
                        "window must cover 1.5x this")
    c.add_argument("--stage",
                   help="HF path of the code tarball the script downloads, passed to the box "
                        "as STAGE (default: STAGE from --env, else the script's STAGE "
                        "default); none if it downloads none")
    c.add_argument("--disk", type=int, default=40)
    c.add_argument("--env", action="append")
    c.set_defaults(fn=create)
    st = sub.add_parser("status", help="one instance, or all of them")
    st.add_argument("instance_id", type=int, nargs="?")
    st.set_defaults(fn=status)
    lg = sub.add_parser("logs", help="an instance's container logs")
    lg.add_argument("instance_id", type=int)
    lg.add_argument("--tail", type=int, help="the last N lines only")
    lg.add_argument("--daemon-logs", action="store_true",
                    help="Vast's daemon logs instead of the container's")
    lg.set_defaults(fn=logs)
    go = sub.add_parser("start", help="start a stopped instance")
    go.add_argument("instance_id", type=int)
    go.set_defaults(fn=start)
    halt = sub.add_parser("stop", help="stop an instance: GPU billing ends, the disk stays")
    halt.add_argument("instance_id", type=int)
    halt.set_defaults(fn=stop)
    d = sub.add_parser("destroy", help="destroy an instance and its disk")
    d.add_argument("instance_id", type=int)
    d.set_defaults(fn=destroy)
    return ap


def main(argv) -> int:
    args = parser().parse_args(argv)
    try:
        return args.fn(args)
    except ApiCallFailed as exc:
        print(exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
