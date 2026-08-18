# Figure handoff — TransGenic manuscript

Everything needed to regenerate, edit, or re-style any figure in the manuscript without
re-deriving the decisions. Written for whoever picks this up next, including a fresh
session with no memory of how these were built.

Last verified 2026-08-04 against commit `abdbc6d` and the Figure S4 rebuild that followed it.

---

## 1. The figure set

Ten figures. They are **not** all in one directory, and two of them have near-identical
siblings that must not be submitted — see §5.

| # | Output stem | Script | Lives in |
|---|---|---|---|
| 1 | `figure1_architecture` | `Figures/make_figure1.py` | `Figures/` |
| 2 | `figure2_datasets` | `Figures/make_figure2.py` | `Figures/` |
| 3 | `figure3_performance_original` | `Figures/make_figure3_original.py` | `Figures/` |
| 4 | `figure4_example_loci_original` | `Figures/make_figure4_panelC.py` | `Figures/` |
| 5 | `figure5_gffcompare_f1` | `revision/scripts/15_revision_figures.py` | `revision/figures/` |
| 6 | `figure6_as_evaluation` | `revision/scripts/15_revision_figures.py` | `revision/figures/` |
| S1 | `figureS_busco` | `revision/scripts/15_revision_figures.py` | `revision/figures/` |
| S2 | `figureS2_tss_tes` | `revision/scripts/16_supplement_figures.py` | `revision/figures/` |
| S3 | `figureS3_vocabulary_coverage` | `revision/scripts/16_supplement_figures.py` | `revision/figures/` |
| S4 | `figure4_panelC_all` | `Figures/make_figure4_panelC.py` | `Figures/` |

Every script writes both `.pdf` and `.png`.

The submission package copies these to `073026/Figure1.pdf` … `073026/FigureS4.png`, renamed to
the manuscript numbering. That folder is deliberately outside the git repository — see
`073026/README.md`.

---

## 2. Shared conventions

Applied by every script. Keep them if you add a figure.

| Setting | Value |
|---|---|
| Resolution | `dpi=300` on save |
| Bounding box | `bbox_inches="tight"` |
| Background | `facecolor="white"` (Figure 1 uses `"none"` — it is a schematic meant to sit on the page) |
| Font | Arial, via `figstyle.py` — see below |
| PDF fonts | Type 42 (embedded TrueType), set by `figstyle.py` |
| Width | 7.0–7.6 in single column; the Figure 4 sheet widens to `FIG_W * 1.65` for two columns |

### Typography — `Figures/figstyle.py`

Every script calls it immediately after importing matplotlib:

```python
import figstyle
figstyle.apply(8)      # base point size; scripts pass 7 for denser panels
```

It does two things that are easy to get wrong silently.

**Arial.** The cluster has neither Arial nor Helvetica, so matplotlib was quietly falling back
to DejaVu Sans — a different face with different metrics, which only surfaces when a reviewer
compares figures. Liberation Sans is metric-compatible with Arial and openly licensed; the
TTFs live in `/data/gpfs/assoc/pgl/tools/fonts/` and are registered at import. `font.sans-serif`
still lists Arial first, so a machine with the real font uses it. Verify rather than assume:

```bash
python3 Figures/figstyle.py     # prints the resolved family; exits non-zero if it is neither
```

**Type 42 PDF fonts.** matplotlib defaults to Type 3, which several publishers reject and
which cannot be edited in Illustrator. `figstyle` sets `pdf.fonttype = 42`, embedding the
TrueType outlines and keeping text selectable.

**Neither Arial nor Liberation Sans contains U+2713 `✓`.** Figure 4 originally used a tick to
mark chain confirmation and it fell back to a last-resort face. It now uses `=`, which states
the actual relation (identical CDS intron chain) and is symmetric, so both the prediction and
the AtRTD3 row can carry it: `pred 3 = AT1G44575.2` ↔ `AT1G44575.2 = pred 3`. Before adding any
glyph beyond ASCII, check it:

```python
from fontTools.ttLib import TTFont
cmap = TTFont("/data/gpfs/assoc/pgl/tools/fonts/LiberationSans-Regular.ttf").getBestCmap()
ord("−") in cmap    # minus, en dash, em dash, middle dot, ≥ are all present
```

Confirm no fallback crept into a PDF:

```bash
python3 - <<'PY'
import re, pathlib
for f in list(pathlib.Path("Figures").glob("figure*.pdf")) + \
         list(pathlib.Path("revision/figures").glob("figure*.pdf")):
    fonts = {b.decode().split("+")[-1] for b in
             re.findall(rb"/BaseFont\s*/([A-Za-z0-9+\-]+)", f.read_bytes())}
    extra = {x for x in fonts if not x.startswith("Liberation")}
    if extra:
        print(f.name, sorted(extra))
PY
```

### Legends must not sit on the data

Wherever bars reach the top of the axes, an in-axes legend lands on them. Three panels had
this (Figure 6C, 6D, Figure S2). The fix used throughout is to put the legend above the axes
and give the title room:

```python
ax.legend(frameon=False, fontsize=6, ncol=3, loc="lower left",
          bbox_to_anchor=(0, 1.005), borderaxespad=0)
ax.set_title("D  ORF self-consistency", loc="left", fontweight="bold", pad=18)
```

Avoid `loc="best"`: it picks a spot from the current data, so a legend that clears the bars
today can land on them after a rerun with new numbers.

### Palette — Okabe-Ito, colour-vision safe

Do not introduce a colour outside this set without checking it against deuteranopia and
protanopia; the whole point of this palette is that the figures survive both.

| Hex | Name in code | Used for |
|---|---|---|
| `#8C8C8C` | `GREY` | TAIR10 reference |
| `#0072B2` | `BLUE` | TransGenic reproducing a TAIR10 chain |
| `#D55E00` | `ORANGE` | TransGenic chain absent from TAIR10 |
| `#00664A` | `DGREEN` | AtRTD3 transcript confirming the TransGenic chain |
| `#009E73` | `GREEN` | AtRTD3 carrying the marked feature |
| `#9AD5C0` | `PALE` | AtRTD3 other isoforms |
| `#F0C000` | `SHADE` | highlight band (drawn at alpha 0.20) |

Figures 5, 6 and S1–S3 use the same family plus `#E69F00`, `#56B4E9`, `#CC79A7` for extra
series, `#333333` for text, `#DDDDDD` for grid lines.

**Semantic rule that matters more than the hex values:** in Figure 4 the colour encodes the
*relationship between sources*, not the source alone. Dark green is reserved for the AtRTD3
transcript whose entire CDS intron-chain equals the prediction. Mid green is a transcript that
merely carries the highlighted feature. Collapsing those two makes a retained-intron panel
claim three "exact matches" whose chains in fact differ — this was a real bug, fixed 2026-07-30.

---

## 3. Figure 4 in detail

The most intricate figure and the one most likely to need edits. `make_figure4_panelC.py`
draws every panel of Figure 4 **and** Figure S4 from one code path.

### What each panel must communicate

TransGenic predicts an isoform; AtRTD3 long reads confirm it. That relationship is stated
in text on the tracks, not left to colour:

- a prediction is labelled with the AtRTD3 transcript whose CDS intron-chain it matches
  exactly — `novel = AT4G30510.2 +1`, with `+n` when more AtRTD3 transcripts share that chain
- each AtRTD3 row that confirms a prediction is labelled with the prediction it confirms
  (`AT4G30510.2 = novel`), so the equals sign reads in both directions
- **a prediction with no such label has no exact long-read match**, and that absence is meant
  to be as visible as a match

### Layout rules

- Track order: TAIR10 → TransGenic (novel first, then reproduced) → AtRTD3 (exact chain match,
  then feature support, then others)
- TAIR10 rows are labelled `t1, t2, …`, **never `.1, .2`**. AtRTD3 identifiers at these loci
  also end in `.1`/`.2` and denote different transcripts; sharing a label pattern made the two
  look like the same series.
- Glyphs: CDS = tall box (`H_CDS = 0.34`), UTR or non-coding exon = short box (`H_UTR = 0.17`),
  intron = line. Row pitch `ROW_H = 0.30` in, fixed so panels share a scale.
- x axis is **base pairs from the start of the drawn locus**, integer ticks. The absolute span
  goes in the axis label. Dividing by 1000 produced ticks like `15722.8`, where the decimal
  carried no information.
- Shaded bands are capped at `MAX_SHADE = 3`. Six overlapping bands read as a striped
  background; the remainder is stated in the amber line instead.
- The legend lists only the colours the panel actually draws.
- On the multi-panel sheet the title is just the locus name — the full sentence repeated five
  times and collided with the `(A)`/`(B)` markers.

### Highlight categories

Resolved in this order; the first that applies wins:

1. `exon` — a predicted exon overlaps no TAIR10 exon
2. `junction` — a junction no TAIR10 isoform uses
3. `retained` — the predicted CDS spans a TAIR10 intron **that no TAIR10 isoform leaves
   unspliced**. The second clause is essential: without it a locus whose isoforms differ by
   which intron they retain gets labelled "0 TAIR10 transcripts" while TAIR10 plainly
   contains that retention.
4. `unspliced_utr` — same, but inside a predicted UTR, which usually just means splicing
   stopped past the stop codon. Weak evidence, and labelled as such.
5. `combination` — every junction already occurs somewhere in TAIR10 and only the chain is
   new. Weakest; the full symmetric difference is shaded.
6. `reproduced` — no novel chain at all; the model returned ≥2 distinct TAIR10 chains that
   AtRTD3 also documents. This is panels A and B.

### Support counts

Never report a bare overlap count. AtRTD3 support is split into disjoint categories —
**identical / splice-site-shared / overlap-only** — because a longer exon that merely spans a
63 bp interval is not the same cassette exon. Conflating them once turned 1 supporting
transcript into a claimed 8.

### ⚠️ Two predictions exist, and they are not interchangeable

There are two prompted A. thaliana predictions in this repository. They draw **different
evaluation loci** — 4,875 and 3,328, sharing only 657 — so the same gene can carry a
different number of TAIR10 chains in each.

| File | Loci | Used for |
|---|---|---|
| `fig3_original/prompted/TAIR10_hyenaTest_prediction_noPost.gff3` | 3,328 | **Figure 4** — the inference the submitted figure was drawn from, recovered from the published model's own artefacts |
| `transgenic_comparison/standardized_results/A_thaliana_transgenic400Mprompt_beam1.gff3` | 27,413 | **Figure S4** — the prompted run the manuscript's isoform metrics come from |
| `fig4_forensics/raw_TAIR10_hyenaTest_prediction_noPost.gff3` | 4,875 | **nothing** — provenance never established; no figure reads it any more |

Extracts are kept in separate directories (`panelC_examples/original/` for Figure 4,
`panelC_examples/prompted_full/` for Figure S4) and `make_figure4_panelC.py` picks one per
locus in `source_dir()`: anything in `FIGURE4` reads from `original/`, everything else from
`prompted_full/`. **Never merge the two directories.** A panel drawn from one file with
support counts derived from the other describes an experiment that was never run — which
is exactly how the five-panel Figure 4 of 2026-07-30 came to claim "all four distinct
TAIR10 chains" for a locus that has two in the evaluation the manuscript reports.

### Adding or swapping a locus

```bash
cd revision/results/fig4_forensics
ATRTD=../../data/AtRTD3/atRTD3_TS_21Feb22_transfix.gtf

# Figure 4 (original inference) -> D=panelC_examples/original, SRC as below
D=panelC_examples/original
SRC=../fig3_original/prompted/TAIR10_hyenaTest

# Figure S4 (second inference)  -> D=panelC_examples, SRC=./raw_TAIR10_hyenaTest

for L in AT1G12345; do
  grep "GM=$L" ${SRC}_prediction_noPost.gff3 > $D/${L}_pred.gff3
  grep "GM=$L" ${SRC}_labels.gff3            > $D/${L}_tair.gff3
  grep "gene_id \"$L\"" $ATRTD               > $D/${L}_atrtd.gtf
done
python3 ../../../Figures/make_figure4_panelC.py AT1G12345    # single locus
```

Then add it to `FIGURE4` (manuscript figure), `STRONG`, or `LOCI` (supplementary sheet) at the
top of the script, and **update the manuscript legend**, which names each panel's locus.

To find candidates rather than guess: `panelC_examples/scan_panels.py <labels> <prediction>
<AtRTD3.gtf>` classifies every prompted A. thaliana locus into panel types A / B / C. Run it
on whichever prediction the panel belongs to, and keep the outputs apart — `panels.tsv`
(the unattributed 4,875-locus file: A 29 / B 3 / C 10),
`original/panels_original.tsv` (original inference, 3,328 loci: **A 25 / B 2 / C 1**) and
`prompted_full/panelC_prompted.tsv` (full prompted run, 27,413 loci: **45 Panel-C loci,
13 junction / 32 combination**, written by `revision/scripts/25_scan_panelC_prompted.py`).
The Methods counts come from the second and third files; `19_verify_manuscript_numbers.py`
re-derives them.

---

## 4. Regenerating everything

```bash
cd /data/gpfs/assoc/pgl/data/Transgenic/transgenic

python3 Figures/make_figure1.py
python3 Figures/make_figure2.py
python3 Figures/make_figure3_original.py          # Figure 3 — NOT make_figure3.py
python3 Figures/make_figure4_panelC.py            # Figure 4 + Figure S4 + per-locus panels
python3 revision/scripts/15_revision_figures.py   # Figures 5, 6, S1
python3 revision/scripts/16_supplement_figures.py # Figures S2, S3
```

Then verify Figure 4 against its source data:

```bash
cd revision/results/fig4_forensics/panelC_examples
python3 verify_panelC_figures.py     # exits non-zero if any track contradicts its colour
```

That check re-derives every claim a panel makes — TAIR10 lacks the feature, the novel track
carries it, the dark-green row's chain equals the prediction — from the GFF3/GTF files. It
currently passes 8/8 loci. **Run it after any edit to `make_figure4_panelC.py`.** It has
already caught two real errors that visual review missed.

---

## 5. Traps

**Two figures have superseded siblings.** Both live in `Figures/_deprecated/` (gitignored) with
a README explaining why:

| Do not submit | Submit instead | Why |
|---|---|---|
| `figure3_performance.*` | `figure3_performance_original.*` | the former is the reconstruction; the latter is the submission figure. Since 2026-08-17 (roadmap R11) its panel A plots the repaired-RC re-inference with runaway transcripts excluded (`revision/results/fig3a_divergence/gffcompare_noRunaway/nr_*.stats`, anchors A. thaliana 92.0 / Z. mays 74.0) and panel B the alt-only AtRTD3 comparison of Table S4a (de novo 72.9/74.4 vs prompted 76.0/76.2); panels C/D still come from the preserved original artifacts, which also remain the source of the disclosed historical anchors 92.2 / 71.2 |
| `figure4_example_loci.*` | `figure4_example_loci_original.*` | drawn from three **guessed** loci that appear nowhere in the authors' records |
| `figure4_example_loci_rebuilt.*` | `figure4_example_loci_original.*` | five sound panels drawn from the **wrong inference**: none of the loci J. Lomas confirmed on 2026-08-04 can be drawn from the file it used, and one of its panels claimed four TAIR10 chains for a locus that has two in the evaluation the manuscript reports |

**`revision/figures/` is gitignored** (`revision/.gitignore:8`). Figures 5, 6 and S1–S3 are
written there and were invisible in the repository until force-added on 2026-07-30. The ten
files now tracked stay tracked, so regenerating them shows up as a normal modification — but
**any new file in that directory is silently hidden**, including a renamed or newly added
figure. Add those with `git add -f`, and check `git status --porcelain revision/figures/`
rather than trusting a clean `git status`.

**Most scripts hardcode `/data/gpfs/assoc/pgl/data/Transgenic/...`.** Only
`make_figure4_panelC.py` and `verify_panelC_figures.py` resolve paths relative to the
repository. The rest will not run from a fresh clone until their `OUT`/`BASE` constants are
changed. Worth fixing if anyone runs them elsewhere.

**Editing the manuscript legend is part of editing the figure.** The Figure 4 legend names
every panel's locus, explains the tick convention, and defines the x axis. A panel swap that
skips the legend produces a caption that describes a different figure.

---

## 6. Checklist before calling a figure change done

- [ ] script rerun; both `.pdf` and `.png` regenerated at 300 dpi
- [ ] `verify_panelC_figures.py` passes, if Figure 4 or S4 changed
- [ ] manuscript legend updated if panels, loci, colours or axis changed
- [ ] `python3 revision/scripts/23_build_pc_manuscript.py` rerun, then
      `19_verify_manuscript_numbers.py` (123/123) and `22_check_citations.py` (56 = 56)
- [ ] `073026/` package refreshed if the figure is in the submission set
- [ ] new output under `revision/figures/` force-added, since that path is gitignored
