"""ASCII rendering of the gradient-amplitude tree."""
from __future__ import annotations


def render_tree(tree: dict) -> str:
    lines = []
    total = tree["terms"].get("total", {})
    lines.append(f"TOTAL gradient  norm={total.get('norm', 0):.4g}  "
                 f"({tree.get('n_experiences', '?')} experiences)")
    for g, info in sorted(tree.get("groups", {}).items(),
                          key=lambda kv: -kv[1]["total_norm"]):
        lines.append(f"  [{g:11s}] {info['total_norm']:.4g}")
    order = sorted(
        (t for t in tree["terms"] if t != "total"),
        key=lambda t: -abs(tree["terms"][t].get("proj_frac", 0.0)))
    for term in order:
        node = tree["terms"][term]
        lines.append(
            f"├─ {term:15s} norm={node['norm']:.4g}  "
            f"cos_total={node.get('cos_total', 0):+.3f}  "
            f"proj_frac={node.get('proj_frac', 0):+.3f}")
        for g, gi in sorted(node["groups"].items(),
                            key=lambda kv: -kv[1]["norm"]):
            if gi["norm"] <= 0:
                continue
            lines.append(
                f"│    {g:11s} norm={gi['norm']:.4g} "
                f"({gi['frac_of_total_group']:.2f} of group total, "
                f"cos {gi['cos_total']:+.2f})")
    return "\n".join(lines)
