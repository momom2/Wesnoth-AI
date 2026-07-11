"""Render trainer-history CSV(s) into a self-contained HTML dashboard.

Reusable check-in tool: pulls the latest escrowed CSV from HF (or takes
a local path) and emits one HTML file with inline-SVG line charts — no
external assets, so it can be published as a claude.ai Artifact or
opened directly in a browser. Panels:

  1. Value generalization — fresh_value_ce vs fresh_ce_floor vs
     holdout_value_loss (is the value head beating the state-blind
     predictor on never-seen states?)
  2. Memorization gap — train_value_loss vs fresh_value_ce
  3. Decisive rate — ladder and mini/drill, rolling d/n over a
     10-iteration window (fractions, per the reporting rules)
  4. Actions per side-turn — mean and median (MCTS depth calibration)

X-axis is decision_step (training units — the project's reporting
standard), falling back to row index for old-schema CSVs.

Usage:
    python tools/plot_training_curves.py --hf                # latest escrow
    python tools/plot_training_curves.py --csv path.csv
        [--out logs/training_curves.html] [--last N]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reference palette (dataviz skill, validated: worst adjacent CVD
# dE 24.2 light / 10.3 dark). Slots in fixed order.
_LIGHT = {"s1": "#2a78d6", "s2": "#1baf7a", "s3": "#eda100",
          "s4": "#008300",
          "surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e",
          "grid": "#e4e3df"}
_DARK = {"s1": "#3987e5", "s2": "#199e70", "s3": "#c98500",
         "s4": "#008300",
         "surface": "#1a1a19", "text": "#ffffff", "text2": "#c3c2b7",
         "grid": "#3a3936"}

W, H, PAD_L, PAD_R, PAD_T, PAD_B = 860, 260, 58, 16, 26, 34


def _f(row: dict, key: str) -> Optional[float]:
    v = (row.get(key) or "").strip()
    if not v:
        return None
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except ValueError:
        return None


def load_rows(path: Path) -> List[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def rolling_fraction(rows, num_key, den_key, window=10):
    """Rolling sum(num)/sum(den); None where the window has no games."""
    nums = [(_f(r, num_key) or 0.0) for r in rows]
    dens = [(_f(r, den_key) or 0.0) for r in rows]
    out = []
    for i in range(len(rows)):
        lo = max(0, i - window + 1)
        d = sum(dens[lo:i + 1])
        out.append(sum(nums[lo:i + 1]) / d if d > 0 else None)
    return out


def _svg_panel(pid, title, xs, series, y_label, y_fmt=".2f",
               y_min=None, y_max=None):
    """One line-chart panel. series: [(name, slot, [y|None])]."""
    ys_all = [y for _, _, ys in series for y in ys if y is not None]
    if not ys_all or not xs:
        return f"<h3>{title}</h3><p class='muted'>no data yet</p>"
    lo = min(ys_all) if y_min is None else y_min
    hi = max(ys_all) if y_max is None else y_max
    if hi - lo < 1e-9:
        hi = lo + 1.0
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    x0, x1 = xs[0], xs[-1]
    if x1 - x0 < 1e-9:
        x1 = x0 + 1.0

    def sx(x): return PAD_L + (x - x0) / (x1 - x0) * (W - PAD_L - PAD_R)
    def sy(y): return PAD_T + (hi - y) / (hi - lo) * (H - PAD_T - PAD_B)

    grid, labels = [], []
    for k in range(5):
        gy = lo + (hi - lo) * k / 4
        py = sy(gy)
        grid.append(f'<line x1="{PAD_L}" y1="{py:.1f}" x2="{W - PAD_R}"'
                    f' y2="{py:.1f}" class="grid"/>')
        labels.append(f'<text x="{PAD_L - 6}" y="{py + 3.5:.1f}"'
                      f' class="ax" text-anchor="end">'
                      f'{format(gy, y_fmt)}</text>')
    for k in range(4):
        gx = x0 + (x1 - x0) * k / 3
        px = sx(gx)
        lbl = (f"{gx / 1e6:.2f}M" if x1 > 2e6
               else f"{gx / 1e3:.0f}k" if x1 > 2e3 else f"{gx:.0f}")
        labels.append(f'<text x="{px:.1f}" y="{H - 10}" class="ax"'
                      f' text-anchor="middle">{lbl}</text>')

    paths, ends, legend = [], [], []
    for si, (name, slot, ys) in enumerate(series):
        pts = [(sx(x), sy(y)) for x, y in zip(xs, ys) if y is not None]
        if not pts:
            continue
        d = "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        paths.append(f'<path d="{d}" class="ln {slot}"/>')
        ex, ey = pts[-1]
        ends.append(f'<text x="{min(ex + 5, W - 2):.1f}" y="{ey + 3:.1f}"'
                    f' class="lbl {slot}-t">{name}</text>')
        legend.append(f'<span class="key"><i class="sw {slot}-bg"></i>'
                      f'{name}</span>')

    data = {"xs": xs,
            "series": [{"name": n, "slot": s, "ys": ys}
                       for n, s, ys in series],
            "fmt": y_fmt}
    return f"""
<h3>{title} <span class="muted">({y_label})</span></h3>
<div class="legend">{''.join(legend)}</div>
<div class="wrap">
<svg id="{pid}" viewBox="0 0 {W} {H}" data-plot='{json.dumps(data)}'
     data-x0="{x0}" data-x1="{x1}" data-lo="{lo}" data-hi="{hi}">
{''.join(grid)}{''.join(labels)}{''.join(paths)}{''.join(ends)}
<line class="cross" x1="0" y1="{PAD_T}" x2="0" y2="{H - PAD_B}"
      style="display:none"/>
</svg>
<div class="tip" style="display:none"></div>
</div>"""


def build_html(rows: List[dict], source_note: str) -> str:
    xs = []
    for i, r in enumerate(rows):
        ds = _f(r, "decision_step")
        xs.append(ds if ds is not None else float(i))
    x_is_steps = any(_f(r, "decision_step") is not None for r in rows)

    def col(key):
        return [_f(r, key) for r in rows]

    p1 = _svg_panel(
        "p1", "Value generalization on fresh games", xs,
        [("fresh CE", "s1", col("fresh_value_ce")),
         ("state-blind floor", "s3", col("fresh_ce_floor")),
         ("holdout CE", "s2", col("holdout_value_loss"))],
        "CE, nats — below the floor = reads the board")
    p2 = _svg_panel(
        "p2", "Memorization gap", xs,
        [("train value loss", "s2", col("train_value_loss")),
         ("fresh CE", "s1", col("fresh_value_ce"))],
        "CE, nats — gap = buffer fitting vs transfer")
    p3 = _svg_panel(
        "p3", "Decisive rate (rolling 10 iters)", xs,
        [("ladder", "s1",
          rolling_fraction(rows, "ladder_decisive", "ladder_games")),
         ("mini/drill", "s2",
          rolling_fraction(rows, "other_decisive", "other_games"))],
        "fraction of games decided", y_min=0.0, y_max=1.0)
    p4 = _svg_panel(
        "p4", "Actions per side-turn", xs,
        [("mean", "s1", col("actions_per_turn_mean")),
         ("median", "s2", col("actions_per_turn_median"))],
        "actions — spam pathology watch", y_fmt=".1f")
    p6 = _svg_panel(
        "p6", "Value-target composition (fresh batches)", xs,
        [("win (+1)", "s2", col("z_win_frac")),
         ("loss (-1)", "s1", col("z_loss_frac")),
         ("draw", "s3", col("z_draw_frac"))],
        "fraction of incoming targets", y_min=0.0, y_max=1.0)
    p7 = _svg_panel(
        "p7", "Human-anchor rehearsal loss", xs,
        [("anchor value CE", "s1", col("human_anchor_loss"))],
        "CE on pre-encoded human states")
    p5 = _svg_panel(
        "p5", "Trainer GPU memory", xs,
        [("allocated", "s1", col("gpu_mem_alloc_mb")),
         ("reserved", "s3", col("gpu_mem_reserved_mb"))],
        "MB — creep watch: linear=leak, staircase=ratchet",
        y_fmt=".0f")
    # Fogless-mixing experiment (2026-07-11): is full information
    # doing work on ladder maps? Compare capture activity and
    # decisiveness per fog condition.
    p8 = _svg_panel(
        "p8", "Village capture: fogged vs fogless ladder", xs,
        [("fogged vil/turn", "s1",
          col("ladder_fog_villages_per_turn")),
         ("fogless vil/turn", "s2",
          col("ladder_fogless_villages_per_turn")),
         ("fogged end", "s3", col("ladder_fog_villages_end")),
         ("fogless end", "s4", col("ladder_fogless_villages_end"))],
        "villages owned (both sides) — never-meet flatlines near 0")
    p9 = _svg_panel(
        "p9", "Ladder decisive rate by fog condition (rolling 10)", xs,
        [("fogged", "s1",
          rolling_fraction(rows, "ladder_fog_decisive",
                           "ladder_fog_games")),
         ("fogless", "s2",
          rolling_fraction(rows, "ladder_fogless_decisive",
                           "ladder_fogless_games"))],
        "fraction of ladder games decided", y_min=0.0, y_max=1.0)

    def css_vars(m):
        return (f"--surface:{m['surface']};--text:{m['text']};"
                f"--text2:{m['text2']};--grid:{m['grid']};"
                f"--s1:{m['s1']};--s2:{m['s2']};--s3:{m['s3']};"
                f"--s4:{m['s4']};")

    x_note = ("decision steps (training units)" if x_is_steps
              else "iteration index (old-schema CSV)")
    return f"""<title>Wesnoth self-play training curves</title>
<style>
.viz-root {{ {css_vars(_LIGHT)}
  background: var(--surface); color: var(--text);
  font: 14px/1.45 system-ui, sans-serif; padding: 20px;
  max-width: 940px; margin: 0 auto; }}
@media (prefers-color-scheme: dark) {{ .viz-root {{ {css_vars(_DARK)} }} }}
:root[data-theme="dark"] .viz-root {{ {css_vars(_DARK)} }}
:root[data-theme="light"] .viz-root {{ {css_vars(_LIGHT)} }}
.viz-root h2 {{ margin: 0 0 2px; font-size: 19px; }}
.viz-root h3 {{ margin: 26px 0 4px; font-size: 15px; }}
.muted {{ color: var(--text2); font-weight: 400; font-size: 12.5px; }}
.legend {{ margin: 2px 0 4px; }}
.key {{ margin-right: 14px; font-size: 12.5px; color: var(--text2); }}
.sw {{ display: inline-block; width: 10px; height: 10px;
      border-radius: 2px; margin-right: 5px; }}
.s1-bg {{ background: var(--s1); }} .s2-bg {{ background: var(--s2); }}
.s3-bg {{ background: var(--s3); }} .s4-bg {{ background: var(--s4); }}
.s1-t {{ fill: var(--s1); }} .s2-t {{ fill: var(--s2); }}
.s3-t {{ fill: var(--s3); }} .s4-t {{ fill: var(--s4); }}
.wrap {{ position: relative; overflow-x: auto; }}
svg {{ width: 100%; height: auto; display: block; }}
.grid {{ stroke: var(--grid); stroke-width: 1; }}
.ax {{ fill: var(--text2); font-size: 10.5px; }}
.lbl {{ font-size: 11px; font-weight: 600; }}
.ln {{ fill: none; stroke-width: 2; stroke-linejoin: round; }}
.ln.s1 {{ stroke: var(--s1); }} .ln.s2 {{ stroke: var(--s2); }}
.ln.s3 {{ stroke: var(--s3); }} .ln.s4 {{ stroke: var(--s4); }}
.cross {{ stroke: var(--text2); stroke-dasharray: 3 3; }}
.tip {{ position: absolute; pointer-events: none; background: var(--surface);
       color: var(--text); border: 1px solid var(--grid); border-radius: 6px;
       padding: 6px 9px; font-size: 12px; box-shadow: 0 2px 8px #0003;
       white-space: nowrap; z-index: 2; }}
</style>
<div class="viz-root">
<h2>Wesnoth self-play — training curves</h2>
<p class="muted">{source_note} &middot; {len(rows)} iterations &middot;
x-axis: {x_note}</p>
{p1}{p2}{p6}{p7}{p3}{p8}{p9}{p4}{p5}
</div>
<script>
for (const svg of document.querySelectorAll("svg[data-plot]")) {{
  const d = JSON.parse(svg.dataset.plot);
  const x0 = +svg.dataset.x0, x1 = +svg.dataset.x1;
  const tip = svg.parentElement.querySelector(".tip");
  const cross = svg.querySelector(".cross");
  const PL = {PAD_L}, PR = {PAD_R}, WW = {W};
  svg.addEventListener("mousemove", ev => {{
    const r = svg.getBoundingClientRect();
    const fx = (ev.clientX - r.left) / r.width * WW;
    const xv = x0 + (fx - PL) / (WW - PL - PR) * (x1 - x0);
    let best = 0, bd = Infinity;
    d.xs.forEach((x, i) => {{
      const dd = Math.abs(x - xv);
      if (dd < bd) {{ bd = dd; best = i; }}
    }});
    const px = PL + (d.xs[best] - x0) / ((x1 - x0) || 1) * (WW - PL - PR);
    cross.setAttribute("x1", px); cross.setAttribute("x2", px);
    cross.style.display = "";
    const lines = [`<b>x = ${{Math.round(d.xs[best]).toLocaleString()}}</b>`];
    for (const s of d.series) {{
      const y = s.ys[best];
      if (y !== null && y !== undefined)
        lines.push(`${{s.name}}: ${{y.toFixed(3)}}`);
    }}
    tip.innerHTML = lines.join("<br>");
    tip.style.display = "";
    const left = Math.min(px / WW * r.width + 12, r.width - 150);
    tip.style.left = left + "px";
    tip.style.top = "18px";
  }});
  svg.addEventListener("mouseleave", () => {{
    tip.style.display = "none"; cross.style.display = "none";
  }});
}}
</script>"""


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--hf", action="store_true",
                    help="Pull the latest escrowed CSV from the HF repo "
                         "(token file per project convention).")
    ap.add_argument("--out", type=Path,
                    default=Path("logs/training_curves.html"))
    ap.add_argument("--last", type=int, default=None,
                    help="Only the most recent N iterations.")
    args = ap.parse_args(argv[1:])

    if args.hf:
        from huggingface_hub import hf_hub_download
        tok = Path.home().joinpath(
            ".hf_token_wesnoth").read_text().strip() if Path.home(
            ).joinpath(".hf_token_wesnoth").exists() else Path(
            r"C:\Users\amaur\.hf_token_wesnoth").read_text().strip()
        src = Path(hf_hub_download("momom2/wesnoth-tier-a",
                                   "trainer_history_local.csv",
                                   token=tok, force_download=True))
        note = "source: HF escrow (<=30 min stale)"
    elif args.csv:
        src, note = args.csv, f"source: {args.csv.name}"
    else:
        ap.error("pass --csv PATH or --hf")

    rows = load_rows(src)
    if args.last:
        rows = rows[-args.last:]
    if not rows:
        print("no rows in CSV"); return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_html(rows, note), encoding="utf-8")
    last = rows[-1]
    print(f"wrote {args.out} ({len(rows)} iterations)")
    print("last row: " + " ".join(
        f"{k}={last.get(k)}" for k in
        ("iter", "decision_step", "fresh_value_ce", "fresh_ce_floor",
         "holdout_value_loss", "ladder_games", "ladder_decisive")
        if last.get(k)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
