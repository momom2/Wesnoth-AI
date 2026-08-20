"""General training-metrics visualizer (user directive 2026-08-20:
one reusable tool, lenient parsing, built for expansion -- no more
bespoke HTML per report).

Reads any trainer-history CSV(s) and holdout-probe CSV(s) --
LENIENTLY: unknown columns ignored, missing columns skip their
panel, unparseable cells dropped row-wise, schema drift tolerated
(old and new CSVs mix freely). Multiple files overlay as labeled
series, so cross-leg comparison (the leg-3 ghost) is a flag, not a
rewrite.

Output: one self-contained HTML file (validated palette, light/dark,
hover tooltips, no external assets).

Expansion = adding a PanelSpec to PANELS (or --panel key:title for
an ad-hoc raw column). Derived metrics (e.g. floor-relative fresh
CE) are small functions registered in DERIVED.

Usage:
    python tools/metrics_viz.py --out report.html \
        --history l4.csv:leg4 [--history l3.csv:leg3] \
        [--probe probe.csv:leg4] [--probe-t0 3.207] \
        [--title "Leg 4"] [--panel my_col:My Column]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------
# Lenient loading
# ---------------------------------------------------------------------

def _f(v) -> Optional[float]:
    try:
        x = float(v)
        return x if x == x else None          # drop NaN
    except (TypeError, ValueError):
        return None


def load_rows(path: Path) -> List[Dict[str, Optional[float]]]:
    """Every cell float-coerced or None; rows never rejected."""
    out = []
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                out.append({k: _f(v) for k, v in r.items()
                            if k is not None})
    except OSError as e:
        print(f"WARN: cannot read {path}: {e}", file=sys.stderr)
    return out


# ---------------------------------------------------------------------
# Derived metrics: name -> fn(row) -> Optional[float]
# ---------------------------------------------------------------------

def _fresh_rel(r):
    a, b = r.get("fresh_value_ce"), r.get("fresh_ce_floor")
    return a - b if a is not None and b is not None else None


def _decisive_share(r):
    lead, n = r.get("ended_leader"), r.get("n_games")
    return lead / n if lead is not None and n else None


DERIVED: Dict[str, Callable] = {
    "fresh_rel": _fresh_rel,
    "decisive_share": _decisive_share,
}


# ---------------------------------------------------------------------
# Panels (expansion point). x defaults to "iter" for history panels
# and "decision_step" for probe panels.
# ---------------------------------------------------------------------

@dataclass
class PanelSpec:
    key: str                  # column or DERIVED name
    title: str
    caption: str = ""
    source: str = "history"   # history | probe
    refs: List[Dict] = field(default_factory=list)   # {v,label,bad?}
    band: Optional[List[float]] = None               # [lo, hi]
    wide: bool = False


PANELS: List[PanelSpec] = [
    PanelSpec("actions_per_turn_mean", "Turn structure (actions/side-turn)",
              "The metric leg 3 died of. Band = pre-registered [8, 20].",
              band=[8, 20], wide=True),
    PanelSpec("ce", "Human-CE probe (prior retention)",
              "Rising = drifting from human play.", source="probe"),
    PanelSpec("value_auc", "Judge quality (probe AUC)",
              "Noisy by protocol; tripwire = 3 consecutive below floor.",
              source="probe",
              refs=[{"v": 0.60, "label": "gate 0.60"},
                    {"v": 0.52, "label": "floor 0.52", "bad": True}]),
    PanelSpec("fresh_rel", "Fresh value CE, floor-relative",
              "Default success metric: negative beats the state-blind "
              "floor on unseen games.",
              refs=[{"v": 0.0, "label": "floor"}]),
    PanelSpec("ended_max_turns", "Winnerless (cap) games per iteration",
              "Value-censored; measures behavior, not label poison."),
    PanelSpec("train_value_loss", "Train value loss", ""),
    PanelSpec("z_draw_frac_w", "Draw share of value gradient (weighted)",
              "Leg 3's poison channel; censoring keeps it ~0.",
              refs=[{"v": 0.5, "label": "leg-3 territory", "bad": True}]),
    PanelSpec("decisive_share", "Decisive-by-leader-kill share", "",
              refs=[{"v": 0.35, "label": "abort floor", "bad": True}]),
]


def series_for(panel: PanelSpec, sources: List[Dict]) -> List[Dict]:
    xkey = "iter" if panel.source == "history" else "decision_step"
    out = []
    for src in sources:
        if src["kind"] != panel.source:
            continue
        pts = []
        for r in src["rows"]:
            x = r.get(xkey)
            y = (DERIVED[panel.key](r) if panel.key in DERIVED
                 else r.get(panel.key))
            if x is not None and y is not None:
                pts.append([x, y])
        if pts:
            out.append({"label": src["label"], "points": pts})
    return out


# ---------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------

_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { color-scheme: light dark; }
body { margin:0; font-family: system-ui,-apple-system,"Segoe UI",sans-serif; }
.viz-root { color-scheme: light;
 --surface-1:#fcfcfb; --surface-2:#f3f2ef; --text-primary:#0b0b0b;
 --text-secondary:#52514e; --text-muted:#8b8a85; --grid:#e4e3df;
 --ref:#8b8a85; --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
 --bad:#c0392b;
 background:var(--surface-1); color:var(--text-primary);
 max-width:980px; margin:0 auto; padding:24px 20px 48px; }
@media (prefers-color-scheme: dark) {
 :root:where(:not([data-theme="light"])) .viz-root { color-scheme:dark;
  --surface-1:#1a1a19; --surface-2:#242423; --text-primary:#fff;
  --text-secondary:#c3c2b7; --text-muted:#8b8a85; --grid:#33332f;
  --ref:#8b8a85; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --bad:#e06050; } }
:root[data-theme="dark"] .viz-root { color-scheme:dark;
 --surface-1:#1a1a19; --surface-2:#242423; --text-primary:#fff;
 --text-secondary:#c3c2b7; --text-muted:#8b8a85; --grid:#33332f;
 --ref:#8b8a85; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
 --bad:#e06050; }
h1 { font-size:20px; margin:0 0 4px; }
.sub { color:var(--text-secondary); font-size:13px; margin-bottom:14px; }
h2 { font-size:14.5px; margin:22px 0 2px; }
.cap { color:var(--text-secondary); font-size:12.5px; margin:0 0 6px; }
svg { display:block; width:100%; height:auto; }
svg text { font-family:inherit; }
.axis { font-size:11px; fill:var(--text-muted); }
.refl { font-size:10.5px; fill:var(--text-secondary); }
.legend { display:flex; gap:16px; font-size:12px;
  color:var(--text-secondary); margin:2px 0; flex-wrap:wrap; }
.legend i { width:18px; border-top:2.5px solid currentColor;
  display:inline-block; margin-right:5px; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:22px; }
@media (max-width:720px){ .grid2 { grid-template-columns:1fr; } }
#tip { position:fixed; pointer-events:none; background:var(--surface-2);
 color:var(--text-primary); border:1px solid var(--grid);
 border-radius:6px; padding:5px 9px; font-size:12px; display:none;
 z-index:9; }
</style></head><body>
<div class="viz-root">
<h1>__TITLE__</h1>
<div class="sub">__SUB__</div>
<div id="panels"></div>
</div>
<div id="tip"></div>
<script>
const DATA = __DATA__;
const css = n => getComputedStyle(document.querySelector('.viz-root'))
  .getPropertyValue(n).trim();
const COLS = ['--s1','--s2','--s3','--s4'];
function sv(t,a,p){const e=document.createElementNS(
 'http://www.w3.org/2000/svg',t);
 for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e;}
function nice(lo,hi){const pad=(hi-lo)*0.08||1;return [lo-pad,hi+pad];}
const allPts=[];
DATA.panels.forEach((p,pi)=>{
 if(!p.series.length) return;
 const root=document.getElementById('panels');
 const h2=document.createElement('h2');h2.textContent=p.title;
 root.appendChild(h2);
 if(p.caption){const c=document.createElement('p');c.className='cap';
  c.textContent=p.caption;root.appendChild(c);}
 if(p.series.length>1){const lg=document.createElement('div');
  lg.className='legend';
  p.series.forEach((s,si)=>{const sp=document.createElement('span');
   sp.style.color=css(COLS[si%COLS.length]);
   sp.innerHTML='<i></i>'+s.label;lg.appendChild(sp);});
  root.appendChild(lg);}
 const W=p.wide?900:560,H=p.wide?300:250,L=52,R=90,T=12,B=30;
 const svg=sv('svg',{viewBox:`0 0 ${W} ${H}`,role:'img',
   'aria-label':p.title});
 root.appendChild(svg);
 let xs=[],ys=[];
 p.series.forEach(s=>s.points.forEach(([x,y])=>{xs.push(x);ys.push(y);}));
 (p.refs||[]).forEach(r=>ys.push(r.v));
 if(p.band){ys.push(p.band[0]);ys.push(p.band[1]);}
 const [ylo,yhi]=nice(Math.min(...ys),Math.max(...ys));
 const [xlo,xhi]=[Math.min(...xs),Math.max(...xs)];
 const x=v=>L+(v-xlo)*(W-L-R)/((xhi-xlo)||1);
 const y=v=>T+(yhi-v)*(H-T-B)/((yhi-ylo)||1);
 if(p.band) sv('rect',{x:L,y:y(p.band[1]),width:W-L-R,
   height:y(p.band[0])-y(p.band[1]),fill:css('--grid'),opacity:0.4},svg);
 const nt=5;
 for(let i=0;i<=nt;i++){const g=ylo+(yhi-ylo)*i/nt;
  sv('line',{x1:L,x2:W-R,y1:y(g),y2:y(g),stroke:css('--grid'),
   'stroke-width':1},svg);
  const t=sv('text',{x:L-7,y:y(g)+4,'text-anchor':'end','class':'axis'},svg);
  t.textContent=Math.abs(g)<10?g.toFixed(2):Math.round(g);}
 [xlo,(xlo+xhi)/2,xhi].forEach(g=>{const t=sv('text',{x:x(g),y:H-B+16,
  'text-anchor':'middle','class':'axis'},svg);
  t.textContent=g>1e5?(g/1e6).toFixed(2)+'M':Math.round(g);});
 (p.refs||[]).forEach(r=>{const col=r.bad?css('--bad'):css('--ref');
  sv('line',{x1:L,x2:W-R,y1:y(r.v),y2:y(r.v),stroke:col,
   'stroke-width':1.5,'stroke-dasharray':'3 4'},svg);
  const t=sv('text',{x:W-R+4,y:y(r.v)+4,'class':'refl',fill:col},svg);
  t.textContent=r.label;});
 p.series.forEach((s,si)=>{const col=css(COLS[si%COLS.length]);
  const d=s.points.map((pt,i)=>(i?'L':'M')+x(pt[0]).toFixed(1)+' '
   +y(pt[1]).toFixed(1)).join(' ');
  const at={d,fill:'none',stroke:col,'stroke-width':2.5,
   'stroke-linecap':'round'};
  if(si>0)at['stroke-dasharray']='6 5';
  sv('path',at,svg);
  s.points.forEach(pt=>{
   sv('circle',{cx:x(pt[0]),cy:y(pt[1]),r:2.6,fill:col,
    stroke:css('--surface-1'),'stroke-width':1.6},svg);
   allPts.push({X:x(pt[0]),Y:y(pt[1]),svg,label:s.label,
    x:pt[0],y:pt[1],title:p.title});});
  const last=s.points[s.points.length-1];
  const t=sv('text',{x:W-R+4,y:y(last[1])+4+si*13,'class':'refl',
   fill:col,'font-weight':600},svg);
  t.textContent=(Math.abs(last[1])<10?last[1].toFixed(2)
   :Math.round(last[1]));});
 const tip=document.getElementById('tip');
 svg.addEventListener('mousemove',ev=>{
  const r=svg.getBoundingClientRect();
  const mx=(ev.clientX-r.left)*W/r.width,my=(ev.clientY-r.top)*H/r.height;
  let best=null,bd=700;
  allPts.forEach(q=>{if(q.svg!==svg)return;
   const d=(q.X-mx)**2+(q.Y-my)**2;if(d<bd){bd=d;best=q;}});
  if(best){tip.style.display='block';
   tip.style.left=(ev.clientX+14)+'px';tip.style.top=(ev.clientY-10)+'px';
   tip.textContent=`${best.label} @ ${best.x}: ${best.y.toFixed(4)}`;}
  else tip.style.display='none';});
 svg.addEventListener('mouseleave',()=>{tip.style.display='none';});
});
</script></body></html>
"""


def build(histories, probes, title, sub, extra_panels) -> str:
    sources = []
    for path, label in histories:
        sources.append({"kind": "history", "label": label,
                        "rows": load_rows(Path(path))})
    for path, label in probes:
        sources.append({"kind": "probe", "label": label,
                        "rows": load_rows(Path(path))})
    panels = []
    for spec in PANELS + extra_panels:
        s = series_for(spec, sources)
        panels.append({"title": spec.title, "caption": spec.caption,
                       "refs": spec.refs, "band": spec.band,
                       "wide": spec.wide, "series": s})
    data = {"panels": panels}
    return (_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__SUB__", sub)
            .replace("__DATA__", json.dumps(data)))


def _split(spec: str):
    if ":" in spec:
        p, lab = spec.rsplit(":", 1)
        return p, lab
    return spec, Path(spec).stem


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--history", action="append", default=[],
                    metavar="CSV[:LABEL]")
    ap.add_argument("--probe", action="append", default=[],
                    metavar="CSV[:LABEL]")
    ap.add_argument("--probe-t0", type=float, default=None,
                    help="Adds t0 and t0+0.5 tripwire refs to the CE "
                         "panel.")
    ap.add_argument("--panel", action="append", default=[],
                    metavar="COLUMN[:TITLE]",
                    help="Ad-hoc extra panel from a raw history "
                         "column. Repeatable.")
    ap.add_argument("--title", default="Training metrics")
    ap.add_argument("--sub", default="")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv[1:])

    if args.probe_t0 is not None:
        for spec in PANELS:
            if spec.key == "ce":
                spec.refs = [
                    {"v": args.probe_t0, "label": f"t0 {args.probe_t0}"},
                    {"v": args.probe_t0 + 0.5, "label": "tripwire",
                     "bad": True}]
    extra = []
    for pd in args.panel:
        col, t = _split(pd)
        extra.append(PanelSpec(col, t))
    html = build([_split(h) for h in args.history],
                 [_split(p) for p in args.probe],
                 args.title, args.sub, extra)
    args.out.write_text(html, encoding="utf-8")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
