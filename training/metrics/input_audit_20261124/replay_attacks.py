"""One-off: walk an exported replay's commands on the Rust core and print,
for every attack, what each weapon of the attacker would do against the
defender (the core's own fight statistics). Hexes are printed 1-indexed, as
Wesnoth shows them.

    python -I -X utf8 replay_attacks.py <replay.bz2> <out.json>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.replay_dataset import _stats_for, record_core  # noqa: E402
from tools.replay_extract import extract_replay  # noqa: E402
from wesnoth_ai.game_core import view_of  # noqa: E402


def level_of(name):
    try:
        return int(_stats_for(name).get("level", 1))
    except Exception:
        return 1


def fight_rows(core, att, dfd, ax, ay, dx, dy):
    rows = []
    a_stats = _stats_for(att.name)
    d_resist = _stats_for(dfd.name).get("resistance", {})
    for w in range(len(att.attacks)):
        base = a_stats["attacks"][w] if w < len(a_stats.get("attacks", [])) else {}
        cw = core.counter_weapon_choice(ax, ay, dx, dy, w)
        counter = int(cw[0]) if cw is not None else -1
        st = core.fight_stats(ax, ay, dx, dy, w, counter)
        if st is None:
            continue
        a_st, d_st = st
        rows.append({
            "w": w, "name": base.get("name"), "type": base.get("type"), "range": base.get("range"),
            "specials": base.get("specials", []), "resist_pct": d_resist.get(base.get("type")),
            "a_cth": a_st["cth"], "a_dmg": a_st["damage"], "a_n": a_st["n_attacks"],
            "d_cth": d_st["cth"] if d_st else None, "d_dmg": d_st["damage"] if d_st else None,
            "d_n": d_st["n_attacks"] if d_st else None,
            "dealt": a_st["cth"] / 100.0 * a_st["damage"] * a_st["n_attacks"],
            "taken": d_st["cth"] / 100.0 * d_st["damage"] * d_st["n_attacks"] if d_st else 0.0,
        })
    return rows


def main():
    data = extract_replay(Path(sys.argv[1]))
    if data is None:
        raise SystemExit("the replay did not extract")
    cs = record_core(data)
    gs = view_of(cs)
    out = []
    kinds = {}
    for idx, cmd in enumerate(data.get("commands", [])):
        kind = cmd[0] if cmd else "?"
        kinds[kind] = kinds.get(kind, 0) + 1
        if kind == "attack":
            ax, ay, dx, dy, w = (int(v) for v in cmd[1:6])
            at = {(u.position.x, u.position.y): u for u in gs.map.units}
            att, dfd = at.get((ax, ay)), at.get((dx, dy))
            if att is not None and dfd is not None:
                out.append({
                    "cmd": idx, "turn": int(gs.global_info.turn_number), "side": int(gs.global_info.current_side),
                    "att": att.name, "att_hex": [ax + 1, ay + 1], "att_hp": int(att.current_hp),
                    "att_level": level_of(att.name), "att_exp": int(att.current_exp), "att_max_exp": int(att.max_exp),
                    "dfd": dfd.name, "dfd_hex": [dx + 1, dy + 1], "dfd_hp": int(dfd.current_hp),
                    "dfd_level": level_of(dfd.name), "chosen_w": w,
                    "weapons": fight_rows(cs.core, att, dfd, ax, ay, dx, dy),
                })
        cs.apply_command(list(cmd))
        gs = view_of(cs)
    Path(sys.argv[2]).write_text(json.dumps({"attacks": out, "kinds": kinds}), encoding="utf-8")
    print("commands by kind:", kinds, "; attacks read:", len(out))
    print("end: turn", gs.global_info.turn_number, "leaders:", sorted((u.side, u.name, u.current_hp)
                                                                     for u in gs.map.units if u.is_leader))
    for a in out:
        c = next((r for r in a["weapons"] if r["w"] == a["chosen_w"]), None)
        if c is None:
            continue
        flag = ""
        if c["resist_pct"] is not None and c["resist_pct"] <= 60:
            flag += " RESISTED"
        for o in a["weapons"]:
            if o["w"] == c["w"]:
                continue
            more = o["dealt"] >= 1.25 * c["dealt"] and o["dealt"] > c["dealt"] + 0.5 and o["taken"] <= c["taken"] + 1e-9
            safer = o["dealt"] >= 0.9 * c["dealt"] and o["taken"] <= c["taken"] - 3.0
            if more or safer:
                flag += f" BETTER:{o['name']}({o['type']}) {o['dealt']:.1f}/{o['taken']:.1f}"
        if c["taken"] >= 1.5 * c["dealt"] and c["taken"] - c["dealt"] >= 3 and a["dfd_hp"] > c["dealt"]:
            flag += " BAD-TRADE"
        print(f"t{a['turn']} s{a['side']} {a['att']}@{a['att_hex'][0]},{a['att_hex'][1]} hp{a['att_hp']} -> "
              f"{a['dfd']}@{a['dfd_hex'][0]},{a['dfd_hex'][1]} hp{a['dfd_hp']}: {c['name']}({c['type']}) "
              f"{c['a_dmg']}x{c['a_n']} at {c['a_cth']}% takes {c['resist_pct']}% | back "
              f"{c['d_dmg']}x{c['d_n']} at {c['d_cth']}% | exp {c['dealt']:.1f} dealt / {c['taken']:.1f} taken{flag}")


if __name__ == "__main__":
    main()
