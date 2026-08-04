# Figure 4 example loci — reverse-engineered from J. Lomas's account

> ## ⛔ RESOLVED 2026-08-04 — the loci are known, and this document's conclusion was wrong
>
> J. Lomas supplied the three panels directly, as (TAIR10, TransGenic, AtRTD3) triples:
>
> | Panel | TAIR10 | TransGenic | AtRTD3 |
> |---|---|---|---|
> | A | AT4G10840 | `614ed78268b7` | AT4G10840 |
> | B | AT3G50550 | `884f69210f1f` | AT3G50550 |
> | C | AT1G19650 | `11e52f9e2918` | AT1G19650 |
>
> All three were confirmed against `fig3_original/prompted/TAIR10_hyenaTest_prediction_noPost.gff3`
> — the hex is the tail of each locus's prediction UUID (e.g. AT4G10840 =
> `9f6af336-6811-473b-81bc-614ed78268b7`), and the row counts match exactly. The panel types
> re-derived from that file reproduce the submitted legend: A and B recover both TAIR10 chains
> and predict nothing outside them, C carries a chain held by no TAIR10 transcript and matched
> exactly by AtRTD3 `AT1G19650.13`. AT1G19650 is the **only** Panel-C locus in that evaluation.
>
> **Two conclusions below are superseded:**
>
> 1. **"AT1G43770: high confidence — almost certainly one of the three Figure 4 panels."**
>    Wrong. AT1G43770 does not occur in the original prediction at all. The `grep AT1G43770
>    TAIR10_hyenaTest_prediction_noPost.gff3` in the shell history therefore ran against a
>    *different* file of the same name; a plot-prep artefact proves a locus was examined, not
>    that it was published.
> 2. **The data-driven scan below, and the five-panel figure built on it.** It ran over
>    `raw_TAIR10_hyenaTest_prediction_noPost.gff3` (4,875 loci), which shares only **657** loci
>    with the original prediction (3,328). Its panels were internally verified but describe a
>    different inference — AT1G44575 has four TAIR10 chains there and two in the original, so
>    "the only locus in the test set at which every TAIR10 chain was recovered" was true of a
>    file the manuscript does not report. That claim has been removed from the manuscript.
>
> **Still open**: what `raw_TAIR10_hyenaTest_prediction_noPost.gff3` actually is. It was copied
> here 2026-07-29 12:25, is prompted-like in shape (1.23 transcripts per locus), and matches
> neither the original prediction nor the Figure 3 regeneration (`fig3_regen`, 4,896 loci,
> 19% locus overlap). Figure S4 rests on it, so its provenance needs an answer from J. Lomas.
>
> The scan itself is kept: the Panel-C loci it found are Figure S4, whose legend now states
> that they come from a different inference than Figure 4.

**2026-07-29.** The original Figure 4 loci were not recorded in any manuscript file, so the
recreation (`transgenic/Figures/make_figure4.py`) guessed three loci from the revision tmap
data: AT1G02630, AT1G19350, AT1G01080. Access to J. Lomas's pgl-gpu account
(`jlomas`, `/home/pgl/scratch1/jlomas/transgenic`) allowed the real loci to be reverse-
engineered from his shell history and the files he prepared. **The recreation's three
guesses appear nowhere in his history or prep files** and are almost certainly wrong.

## Method

The figure itself was made in an external tool — no gene-model plotting library
(DNAFeaturesViewer, gggenes, GenomeDiagram) or plotting notebook exists in the repository,
and no script lists the loci together. So the loci were recovered two other ways:

1. **The plotting-prep file.** `bash_history` lines 1147–1152 show, in the published
   model's prompted-prediction directory
   (`.../Hyena_Gen9G_6144nt_768L12_E22-Completion/`):
   ```
   grep AT1G43770 TAIR10_hyenaTest_prediction_noPost.gff3 > temp.gff3
   agat_convert_sp_gxf2gxf.pl -g temp.gff3 -o temp.clean.gff3
   ```
   This is the exact "extract one locus and clean it for plotting" workflow. The resulting
   `temp.gff3`, `temp.clean.gff3`, and `temp.agat.log` are preserved (mtime 2025-04-13,
   during manuscript preparation) and copied here as
   `AT1G43770_temp{,.clean}.gff3` / `AT1G43770_temp.agat.log`.

2. **The locus-hunting session.** `bash_history` lines 482–508 show him opening the
   alternative-splicing-only test labels (`TAIR10_hyenaTest_labels.ASonly.gff3`) in vim and
   grepping candidate loci from both the ASonly and full labels — the way you audit loci
   before choosing figure examples.

## What was found

| Locus | Where in history | TAIR10 test isoforms | AtRTD3 isoforms | Prompted prediction | Read |
|---|---|---|---|---|---|
| **AT1G43770** | l.1149–1152, **AGAT-cleaned into a plotting file** | 2 | 5 | 7 raw → **3 distinct** | **Confirmed figure locus.** Model predicts isoforms beyond TAIR10's 2; AtRTD3 documents 5. Fits a panel showing recovered/novel AtRTD3-supported isoforms. |
| AT4G17760 | l.487–488, greps ASonly labels | (Chr4, `-rc`) | 2 | 2 | Candidate. Alternatively spliced, reverse-complement locus on Chr4. |
| AT1G53690 | l.505–506, greps ASonly + full labels | 1 in 768L12 label | 1 | — | Candidate from the earlier (SinusoidalDownsample) model iteration. |
| AT1G05770 | l.508, greps full labels | 1 in 768L12 label | 1 | — | Candidate from the earlier model iteration. |

`AT1G43770` structure from `temp.clean.gff3` (3 distinct predicted isoforms, Chr1
16,548,441–16,551,909, + strand): they differ in the 3′ CDS content — a 4-CDS form ending
at 16,549,907, a 6-CDS form ending at 16,550,329, and a longer form extending to
16,551,909. TAIR10 annotates 2 isoforms here; AtRTD3 documents 5. This is squarely an
alternative-splicing example with model-predicted isoforms beyond the primary annotation.

## Exon/CDS-count verification (2026-07-29)

Counting CDS segments per transcript across all sources sharpens the picture and, decisively,
**excludes two of the four candidates** — the figure's three panels are all about alternative
splicing, and two of the loci turn out to be single-isoform everywhere.

| Locus | TAIR10 (CDS/tx) | AtRTD3 (CDS/tx) | TransGenic prompted (CDS/tx) | Alternative splicing? |
|---|---|---|---|---|
| **AT4G17760** | 2 iso: **7, 7** | 2 iso: 7, 3 | 2 iso: **7, 7** | Yes — model reproduces both TAIR10 isoforms |
| **AT1G43770** | 2 iso: 6, 4 | 5 iso: 6, 6, 4, 2, 2 | 7 pred: 6×6, 4 | Yes — model predicts isoforms beyond TAIR10's two |
| AT1G53690 | 1 iso: 3 | 1 iso: 3 | (not predicted) | **No — single isoform, excluded** |
| AT1G05770 | 1 iso: 2 | 1 iso: 2 | (not predicted) | **No — single isoform, excluded** |

**AT4G17760 → Panel A** ("reproduced the alternative transcripts shared by TAIR10 and
AtRTD3"). TAIR10 has two 7-CDS isoforms differing only in their last two CDS (an alternative
3′ end); the model reproduced both — one an exact coordinate match, the other off by 3 bp in
one CDS boundary — and AtRTD3 documents two alternative transcripts at the locus. This is a
textbook "reproduced shared alternative transcripts" case.

**AT1G43770 → Panel B or C** ("recovered additional / AtRTD3-supported isoforms absent from
TAIR10"). TAIR10 annotates two isoforms (6-CDS and 4-CDS); AtRTD3 documents five (adding a
second 6-CDS form and two short 2-CDS forms); the model predicted seven transcripts (six
6-CDS variants and one 4-CDS), i.e. it recovered TAIR10's forms and proposed additional
6-CDS isoforms that AtRTD3 supports but TAIR10 omits. This is exactly a panel-B/C locus, and
it is the one with the preserved `temp.clean.gff3` plotting file.

**AT1G53690 and AT1G05770 are single-isoform in both TAIR10 and AtRTD3** — they cannot be
Figure 4 panels, whose every panel shows alternative splicing. J. Lomas likely grepped them
while checking single-isoform behavior, or rejected them as candidates; either way they are
out.

So exon counting turns "four candidates" into **two identified figure loci with assigned
panel types (AT4G17760 = A, AT1G43770 = B/C) and two exclusions**. The third panel's locus
is not in this shell history — it was probably chosen in an earlier session (`.bash_history`
keeps only the most recent) or on the machine where the figure was drawn.

## Confidence

- **AT1G43770: high.** It is the only locus with a dedicated, AGAT-cleaned single-locus
  file, created by the plotting-prep workflow, in the published model's prompted directory,
  dated to the manuscript period. It is almost certainly one of the three Figure 4 panels.
- **AT4G17760: high** (added by the exon-count verification). It is a genuine two-isoform AS
  locus in the final model; the model reproduced both isoforms; AtRTD3 documents two
  alternative transcripts. It fits Panel A precisely.
- **AT1G53690, AT1G05770: excluded** by exon counting — single-isoform in both TAIR10 and
  AtRTD3, so they cannot be panels in an alternative-splicing figure.
- **Third panel's locus: not recoverable** from this shell history.

## Data-driven scan for the third locus (2026-07-29)

With two panels forensically anchored (AT4G17760, AT1G43770), I scanned the original prompted
predictions for the remaining panel by classifying every A. thaliana locus on splice-junction
(intron-chain) matching between the prediction, TAIR10 labels, and AtRTD3
(`fig4_scan.py`, run over `raw_TAIR10_hyenaTest_prediction_noPost.gff3`):

- **Panel A** (model reproduced ≥2 isoforms shared by TAIR10 and AtRTD3): 33 loci (CDS-level).
- **Panel B** (model recovered ≥2 TAIR10 isoforms, ≥1 also in AtRTD3): 173 loci.
- **Panel C** (model predicted an AtRTD3-supported isoform absent from TAIR10): 10 loci
  (CDS+UTR level), including **AT1G43770** — confirming its panel-C placement independently
  of the prep-file evidence.

The scan places **AT1G43770 in panel C** (its extra isoforms are AtRTD3-supported and absent
from TAIR10) and confirms it is a genuine multi-isoform locus. It does **not** uniquely name
a third figure locus, because each panel criterion is satisfied by tens-to-hundreds of loci —
the three panels were a curatorial choice among many valid examples, not the only loci that
fit. AT4G17760 does not surface in the automated scan because AtRTD3 annotates that locus
with different CDS coordinates than TAIR10, so exact intron-chain identity to AtRTD3 fails
even though the exon count and the manual inspection support it — an illustration of why the
scan cannot substitute for the author's original choice.

Clean data-derived candidates for a "reproduced shared isoforms" panel (all with the model
reproducing 2 isoforms present in both TAIR10 and AtRTD3): AT1G03040, AT1G05890, AT1G15020,
AT1G73370, AT2G18030 — offered only as substitutes if an original locus cannot be recovered.

## What cannot be closed from artifacts

The **third panel's locus cannot be recovered**. The forensic sources are exhausted: no
original Figure 4 image exists on GPFS or in the `jlomas` account (only the recreation and
the manuscript text); no figure was ever committed to either git clone; no plotting notebook
or script names the loci; `.bash_history` (recent commands only) and `.python_history`
contain no further loci; and only AT1G43770 has a plotting-prep file. The figure was drawn in
an external tool (Illustrator/BioRender-style) on a machine not reachable here, and that file
is the only place the third locus is recorded.

The exact panel assignment for the two recovered loci is also not fully mechanical: automated
intron-chain matching against AtRTD3 is imperfect because AtRTD3 annotates coordinates
differently from TAIR10 (AT4G17760, for instance, has entirely different CDS junctions in
AtRTD3), so "supported by AtRTD3" in the figure is a visual/approximate judgment that exact
matching cannot fully reproduce.

## Recommendation

Put these four loci to J. Lomas with the evidence: "Figure 4 — was AT1G43770 one of the
three example loci (there's a `temp.clean.gff3` you prepared for it on 2025-04-13), and were
the other two among AT4G17760 / AT1G53690 / AT1G05770?" That is a far more answerable
question than "what were the loci?", and AT1G43770 alone already replaces one of the three
wrong guesses in the current recreation.

---

# Second forensic pass (2026-07-30) — two earlier conclusions corrected, sources now exhausted

Run because J. Lomas is not expected to answer. Five source classes that the first pass never
touched were searched on `jlomas@pgl-gpu`, and the two anchored loci were re-derived from the
data rather than from shell history. **Two conclusions above are wrong and are corrected here.**

## Correction 1 — AT1G43770 is Panel A, not "Panel B or C"

The first pass placed it in Panel C via `fig4_scan.py`, which matched at CDS+UTR level. At the
level that defines a splice isoform — the CDS intron chain — the locus contains no novel
isoform at all:

| Source | Distinct CDS intron chains |
|---|---|
| TAIR10 | 2 (3-intron, 5-intron) |
| TransGenic prediction (7 tx) | **exactly those 2**, nothing else |
| AtRTD3 (5 tx) | the same 2, plus two 1-intron forms |

Prediction ∩ AtRTD3 − TAIR10 = **0**, and prediction − TAIR10 − AtRTD3 = **0**. The model
reproduced both TAIR10 chains and invented nothing. That is the definition of Panel A,
"a locus at which TransGenic reproduced the alternative transcripts shared by TAIR10 and
AtRTD3" — not Panel B or C. The UTR-level match that produced the earlier call reflects UTR
boundary differences, which are not splice isoform differences.

Drawn to the common Panel-C rule set as `Figures/figure4_panelC_AT1G43770.{pdf,png}`
(mode `reproduced`), and checked by `verify_panelC_figures.py`.

## Correction 2 — AT4G17760 / AT1G53690 / AT1G05770 are weaker evidence than stated

Reading the full session around `.bash_history` lines 470–520 shows those greps were run in

    TestSetPerformanceAnalysis/Hyena_Gen9G_6144nt_SinusoidalDownsample_E15_Hyena_SegmentFocalDice_E13-21/

— the **superseded** model, not the published `768L12_E22` one. The first pass graded
AT4G17760 "high confidence" on exon counts alone; its only provenance is a locus-hunting
session against an earlier model. It remains a plausible candidate, not an anchored one.

By contrast AT1G43770 was prepared in `Hyena_Gen9G_6144nt_768L12_E22-Completion/`, the
published model's directory, and a full listing of that directory shows **`temp.gff3` and
`temp.clean.gff3` are the only single-locus extracts in it**. Exactly one locus was ever
pulled out for plotting from the final model.

## Sources searched this pass, all negative

| Source | Result |
|---|---|
| `~/.ipython/.../history.sqlite` (179 commands) | only `AT1G58150`, ×21 — the 160M demo notebook with dummy coordinates `Chr5, 1234`. Not a figure locus. No plotting calls at all. |
| `~/.viminfo` (31 KB) | file marks only reach 2025-08/09; the April session has aged out. No gene IDs, no search history. |
| `~/.vscode-server`, `~/.cursor-server`, `~/.cursor` | present but hold no `state.vscdb` — the workspace database lives on the client, not the remote host. |
| Account-wide images (`.png`/`.pdf`/`.svg`/`.ai`/`.pptx`) | `transgenic/temp.png` is an "Attention × Gradient" heatmap; `plot.png` is a 9 KB training curve. No gene-model figure anywhere. |
| `~/Desktop`, `~/Documents`, `~/Downloads`, `~/Pictures` | all empty. |
| `.bash_history` full sweep for `grep <locus> … > temp.gff3` | one occurrence only (AT1G43770). The filename was never reused for another locus. |

## Standing conclusion

**Panel A = AT1G43770**, anchored by both the plotting-prep file and the data. The other two
panels are not recoverable from any artifact reachable from here: the figure was assembled in
an external tool on a machine that is not this host, and no intermediate of it survives.

The practical path is therefore to finish Figure 4 from data rather than from recovery —
AT1G43770 for (A), and for (B) and (C) the loci selected and verified in
`panelC_examples/README.md`. Every panel then has a script, a source file, and an automated
check behind it, which is stronger provenance than the original figure ever had.
