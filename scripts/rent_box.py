#!/usr/bin/env python3
"""Rent, inspect and destroy Vast boxes through the SDK (the `vastai`
CLI binary is blocked on the laptop).

    python scripts/rent_box.py search [--gpu "RTX 4090"] [--max-dph 0.8] [--min-cpu 16] [--min-hours 4]
    python scripts/rent_box.py create OFFER_ID --onstart HF_STAGING_SCRIPT [--disk 40] [--env K=V ...]
    python scripts/rent_box.py status [INSTANCE_ID]
    python scripts/rent_box.py start INSTANCE_ID
    python scripts/rent_box.py destroy INSTANCE_ID

`create` starts the pytorch 2.5.1 image with ssh, passes HF_TOKEN (the
laptop's huggingface token) as an environment variable, and an onstart
that fetches the named script from HF `tier-b/staging/` and runs it
detached under /workspace (unattended bring-up: docs/box_specs.md
"Operational facts"). Offers are filtered client-side: no VM hosts
(they refuse ssh), a remaining rental window of at least --min-hours.

A create that answers `success: false` with a contract id has made the
instance in the STOPPED state (2026-09-14: three in a row showed
`intended_status: stopped` and sat in "loading" for half an hour);
`start` brings it up, and the onstart runs then, fetching the script
as it is on HF at that moment.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
ONSTART = (
    "cd /workspace && python -m pip install -q huggingface_hub >/dev/null 2>&1; "
    "python -c \"from huggingface_hub import hf_hub_download as d; import shutil, os; "
    "shutil.copyfile(d('momom2/wesnoth-model-checkpoints', 'tier-b/staging/{script}', "
    "token=os.environ['HF_TOKEN']), '/workspace/{script}')\" && "
    "printf '%s' \"$HF_TOKEN\" > /workspace/.hf_token && chmod 600 /workspace/.hf_token && "
    "(setsid nohup bash /workspace/{script} > /workspace/onstart_script.log 2>&1 < /dev/null &)"
)


def _vast():
    from vastai import VastAI
    return VastAI()


def search(args) -> int:
    v = _vast()
    q = (f"gpu_name={args.gpu.replace(' ', '_')} num_gpus=1 cpu_cores_effective>={args.min_cpu} "
         f"dph_total<={args.max_dph} reliability>0.95 disk_space>={args.disk} "
         f"rentable=true rented=false verified=true inet_down>200 inet_up>50")
    offers = v.search_offers(query=q, order="dph_total")
    if not isinstance(offers, list):
        print(offers)
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


def create(args) -> int:
    from huggingface_hub import get_token
    tok = get_token()
    if not tok:
        print("no huggingface token on this machine", file=sys.stderr)
        return 1
    env = {"HF_TOKEN": tok}
    for kv in args.env or []:
        k, _, val = kv.partition("=")
        env[k] = val
    v = _vast()
    res = v.create_instance(id=args.offer_id, image=IMAGE, disk=args.disk, runtype="ssh_direc",
                            env=env, onstart_cmd=ONSTART.format(script=args.onstart))
    print(json.dumps(res, default=str))
    iid = res.get("new_contract") if isinstance(res, dict) else None
    if not (isinstance(res, dict) and res.get("success")):
        if iid:
            print(f"instance {iid} was made STOPPED; `rent_box.py start {iid}` brings it up",
                  file=sys.stderr)
        return 1
    print("instance", iid)
    return 0


def start(args) -> int:
    print(_vast().start_instance(id=args.instance_id))
    return 0


def status(args) -> int:
    v = _vast()
    if args.instance_id:
        i = v.show_instance(id=args.instance_id)
        rows = [i] if isinstance(i, dict) else []
    else:
        i = v.show_instances()
        rows = i if isinstance(i, list) else []
    if not rows:
        print(i)
        return 0
    for r in rows:
        direct = ((r.get("ports") or {}).get("22/tcp") or [{}])[0].get("HostPort")
        print(f"{r.get('id')}  {r.get('actual_status')}  {r.get('gpu_name')}  "
              f"${r.get('dph_total') or 0:.3f}/h  ssh {r.get('ssh_host')}:{r.get('ssh_port')}  "
              f"direct {r.get('public_ipaddr')}:{direct}  {(r.get('status_msg') or '')[:60]}")
    return 0


def destroy(args) -> int:
    print(_vast().destroy_instance(id=args.instance_id))
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--gpu", default="RTX 4090")
    s.add_argument("--max-dph", type=float, default=0.8)
    s.add_argument("--min-cpu", type=int, default=16)
    s.add_argument("--min-hours", type=float, default=4.0)
    s.add_argument("--disk", type=int, default=40)
    s.add_argument("--limit", type=int, default=12)
    s.set_defaults(fn=search)
    c = sub.add_parser("create")
    c.add_argument("offer_id", type=int)
    c.add_argument("--onstart", required=True, help="script name under tier-b/staging/")
    c.add_argument("--disk", type=int, default=40)
    c.add_argument("--env", action="append")
    c.set_defaults(fn=create)
    go = sub.add_parser("start")
    go.add_argument("instance_id", type=int)
    go.set_defaults(fn=start)
    st = sub.add_parser("status")
    st.add_argument("instance_id", type=int, nargs="?")
    st.set_defaults(fn=status)
    d = sub.add_parser("destroy")
    d.add_argument("instance_id", type=int)
    d.set_defaults(fn=destroy)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
