import collections
import sys
sys.path.insert(0, ".")
import analyze as A
from tools.combat_dominance import admits
rows, dec, att = A.load_rows()
L = A.load.LOOSEST
for label, keep in (("tool classes", lambda r: r["cls"] != "Q2"), ("with Q2", lambda r: True)):
    for name, test in (("R0", lambda r: r["r0"]), ("loosest", lambda r: admits(r["cdims"], r["tier"], L)), ("any of 36", lambda r: r["any36"])):
        sel = [r for r in rows if keep(r) and test(r)]
        a = {(r["game"], r["anchor"]) for r in sel}
        print(f"{label:12s} {name:9s} rewrites {len(sel):4d} attacks {len(a):4d} ({1000*len(a)/dec:.2f}/1k dec, {100*len(a)/att:.2f}% of attacks) {dict(collections.Counter(r['cls'] for r in sel))}")
