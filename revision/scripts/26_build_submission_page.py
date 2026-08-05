#!/usr/bin/env python3
"""Build the single-file review page for the submission package.

One HTML file with every figure and the full manuscript embedded, so a co-author can
read the submission without the repository, a PDF viewer, or a network connection.

The page it replaces was assembled by hand on 2026-07-30 and then silently went stale
three times over: the Arial/Liberation font pass, the Figure 4 restoration, and the
Figure S4 rebuild all changed figures it had already baked in. Hence this script — the
page is now regenerated from the same files the package ships.

Figures are down-sampled to `MAX_W` px and quantized before embedding: the ten PNGs are
about 2.6 MB on disk and would exceed 3.5 MB as base64, which makes the page slow to
open for no benefit on screen. The package still carries the full-resolution PDFs.

Usage:
    python 26_build_submission_page.py [--out 073026/index.html]
"""

from __future__ import annotations

import argparse
import base64
import io
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "073026"
MANUSCRIPT = ROOT / "manuscript_v3_plant_communications.md"
PANDOC = Path("/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASeq_postanalysis/bin/pandoc")

MAX_W = 900
COLORS = 96

FIGURES = [
    ("Figure 1", "Figure1", "Gene sentence format tokenization and model architecture"),
    ("Figure 2", "Figure2", "Training and evaluation datasets"),
    ("Figure 3", "Figure3", "De novo performance across ten plant species"),
    ("Figure 4", "Figure4", "Prompted isoforms confirmed by AtRTD3 long reads"),
    ("Figure 5", "Figure5", "Benchmark against deep-learning genome annotators"),
    ("Figure 6", "Figure6", "Alternative-splicing evaluation"),
    ("Figure S1", "FigureS1", "BUSCO completeness"),
    ("Figure S2", "FigureS2", "TSS and TES accuracy"),
    ("Figure S3", "FigureS3", "GSF vocabulary coverage"),
    ("Figure S4", "FigureS4", "Prompted isoforms absent from TAIR10, full scan"),
]

CSS = """
:root { --fg:#1a1a1a; --muted:#666; --rule:#e0e0e0; --accent:#00664a; --bg:#fff; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e8e8e8; --muted:#9a9a9a; --rule:#333; --accent:#4fd1a5; --bg:#141414; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font: 16px/1.65 -apple-system, "Segoe UI", Arial, sans-serif; }
.wrap { max-width: 900px; margin: 0 auto; padding: 2.5rem 1.25rem 6rem; }
h1 { font-size: 1.85rem; line-height:1.25; margin: 0 0 .4rem; }
h2 { font-size: 1.3rem; margin: 2.6rem 0 .8rem; padding-bottom:.3rem;
     border-bottom: 1px solid var(--rule); }
h3 { font-size: 1.05rem; margin: 1.8rem 0 .5rem; }
.byline { color: var(--muted); margin: 0 0 .3rem; }
.stamp { color: var(--muted); font-size: .85rem; margin-bottom: 2rem; }
.note { border-left: 3px solid var(--accent); padding: .6rem .9rem; margin: 1.2rem 0;
        background: color-mix(in srgb, var(--accent) 7%, transparent); font-size:.93rem; }
figure { margin: 2rem 0; }
figure img { width:100%; height:auto; border:1px solid var(--rule); border-radius:4px; }
figcaption { color: var(--muted); font-size: .88rem; margin-top: .5rem; }
figcaption b { color: var(--fg); }
table { border-collapse: collapse; width:100%; margin:1rem 0; font-size:.92rem;
        display:block; overflow-x:auto; }
th, td { border:1px solid var(--rule); padding:.4rem .6rem; text-align:left; }
th { background: color-mix(in srgb, var(--fg) 6%, transparent); }
pre { background: color-mix(in srgb, var(--fg) 5%, transparent); padding:.8rem;
      border-radius:4px; overflow-x:auto; font-size:.85rem; }
code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size:.9em; }
a { color: var(--accent); }
.toc { columns: 2; column-gap: 2rem; font-size:.93rem; }
.toc a { display:block; padding:.15rem 0; text-decoration:none; }
.toc a:hover { text-decoration: underline; }
hr { border:0; border-top:1px solid var(--rule); margin:3rem 0; }
"""


def embed(png: Path) -> str:
    """Down-sample, quantize, and return a data: URI."""
    im = Image.open(png).convert("RGB")
    if im.width > MAX_W:
        im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
    im = im.quantize(colors=COLORS, method=Image.MEDIANCUT)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def manuscript_html() -> str:
    if not PANDOC.exists():
        raise SystemExit(f"pandoc not found at {PANDOC}")
    out = subprocess.run([str(PANDOC), str(MANUSCRIPT), "-t", "html", "--wrap=none"],
                         capture_output=True, text=True, check=True)
    return out.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PKG / "index.html")
    ap.add_argument("--stamp", default="", help="date shown in the header, e.g. 2026-08-04")
    args = ap.parse_args()

    figs, total = [], 0
    for label, stem, caption in FIGURES:
        png = PKG / f"{stem}.png"
        if not png.exists():
            print(f"  missing: {png}", file=sys.stderr)
            continue
        uri = embed(png)
        total += len(uri)
        figs.append(
            f'<figure id="{stem.lower()}"><img alt="{label}" src="{uri}">'
            f'<figcaption><b>{label}.</b> {caption}. '
            f'Full resolution in <code>{stem}.pdf</code>.</figcaption></figure>')
        print(f"  {label:<11} {png.stat().st_size/1024:>7.0f} KB -> {len(uri)/1024:>6.0f} KB",
              file=sys.stderr)

    toc = "".join(f'<a href="#{s.lower()}">{l} — {c}</a>' for l, s, c in FIGURES)
    stamp = f"Generated {args.stamp}" if args.stamp else "Generated by 26_build_submission_page.py"

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TransGenic — submission package</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>TransGenic: a transformer-based framework for direct DNA-to-annotation translation</h1>
<p class="byline">Lomas, Ramazan, Cushman, Tang, Yim &middot; submitted to <i>Plant Communications</i></p>
<p class="stamp">{stamp} &middot; figures down-sampled to {MAX_W}px for the page; the package
carries full-resolution PDFs</p>

<div class="note">Internal review copy. The manuscript text is not public until after
submission; the code, figures, evaluation pipeline and summary tables are at
<a href="https://github.com/wyim-pgl/transgenic">github.com/wyim-pgl/transgenic</a>.</div>

<h2>Figures</h2>
<div class="toc">{toc}</div>
{"".join(figs)}

<hr>
<h2>Manuscript</h2>
{manuscript_html()}
</div></body></html>"""

    args.out.write_text(html)
    print(f"written: {args.out} ({args.out.stat().st_size/1024/1024:.2f} MB, "
          f"{len(figs)} figures, {total/1024/1024:.2f} MB of image data)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
