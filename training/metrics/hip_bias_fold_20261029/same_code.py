"""Whether the kernels without a bias compile to the same machine code as at
--old-ref: the attention, its log-sum-exp variant and the add and norm,
compiled by hiprtc for --arch from the checked-out sources and from the
old ref's hip_kernels.py, disassembled by the SDK's llvm-objdump, each
kernel's instructions compared (addresses, labels and the code object's
end padding dropped). No GPU is used.

    python same_code.py --root <repo> --objdump <.../_rocm_sdk_core/lib/llvm/bin/llvm-objdump.exe> \
        [--old-ref 041bd7ca] [--arch gfx1032] [--out same_code.json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path

KERNELS = (["varlen_attn_fwd_f16_d32", "varlen_attn_fwd_f16_d64", "varlen_attn_fwd_lse_f16_d32",
            "varlen_attn_fwd_lse_f16_d64"] + [f"add_ln_d{ni}" for ni in (1, 2, 3, 4, 6)])


def instructions(objdump: Path, code: bytes, name: str) -> list:
    """`name`'s instructions in the code object, without addresses or encodings."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "code.o"
        path.write_bytes(code)
        out = subprocess.run([str(objdump), "-d", "--no-show-raw-insn", "--no-leading-addr",
                              f"--disassemble-symbols={name}", str(path)],
                             check=True, capture_output=True, text=True).stdout
    lines = [ln.split("//")[0].strip() for ln in out.splitlines()]
    body = [ln for ln in lines if ln and not ln.endswith(":") and not ln.startswith("Disassembly")
            and "file format" not in ln]
    # After the kernel's last instruction: padding to the next kernel's
    # alignment (s_nop 0) or the code object's end (s_code_end), which
    # depends on what follows the kernel in its module, not on the kernel.
    while body and body[-1] in ("s_nop 0", "s_code_end"):
        body.pop()
    # Branch targets are printed as labels with offsets; keep the opcode and operands.
    return [re.sub(r"<[^>]*>", "", ln) for ln in body]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--objdump", type=Path, required=True)
    ap.add_argument("--old-ref", default="041bd7ca")
    ap.add_argument("--arch", default="gfx1032")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    sys.path.insert(0, str(args.root))
    import wesnoth_ai.hip_elementwise as he
    import wesnoth_ai.hip_kernels as hk

    src = subprocess.run(["git", "-C", str(args.root), "show", f"{args.old_ref}:wesnoth_ai/hip_kernels.py"],
                         check=True, capture_output=True, text=True).stdout
    old = types.ModuleType("hip_kernels_old")
    exec(compile(src, "hip_kernels_old", "exec"), old.__dict__)
    codes = {"old": hk.compile_source(args.arch, old._source()),
             "attention": hk.compile_source(args.arch, hk._source()),
             "elementwise": hk.compile_source(args.arch, he._source(), "wesnoth_elementwise.hip")}
    rec = {"old_ref": args.old_ref, "arch": args.arch, "kernels": {}}
    for name in KERNELS:
        new = codes["elementwise" if name.startswith("add_ln") else "attention"]
        a, b = instructions(args.objdump, codes["old"], name), instructions(args.objdump, new, name)
        rec["kernels"][name] = {"instructions": len(a), "identical": a == b and len(a) > 0}
        print(name, rec["kernels"][name])
    rec["all_identical"] = all(k["identical"] for k in rec["kernels"].values())
    print("all identical:", rec["all_identical"])
    if args.out:
        args.out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0 if rec["all_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
