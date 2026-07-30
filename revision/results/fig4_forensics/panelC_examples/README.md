# Panel-C examples: TransGenic predicts an AtRTD3-supported isoform absent from TAIR10

Found by scanning the original prompted test-set predictions
(`../raw_TAIR10_hyenaTest_prediction_noPost.gff3`) for a predicted transcript whose CDS
intron-chain exactly matches an AtRTD3 transcript but no TAIR10 transcript, with ≥2 introns
(`find_panelC.py`). Six loci qualify. All six are drawn to a single rule set by
`transgenic/Figures/make_figure4_panelC.py`.

At all six loci the model predicts two transcripts: one reproducing a TAIR10 chain (blue)
and one novel (orange). The novel chain is absent from TAIR10 everywhere — that part is not
in question. What differs between loci is **how visible that difference is**, and the table
grades exactly that.

| Locus | Chr/strand | Novel feature | Evidence | AtRTD3 chain match | AtRTD3 feature support |
|---|---|---|---|---|---|
| **AT2G37450** | Chr2 (−) | novel cassette exon<br>15,724,032–15,724,094 | **strong** | AT2G37450.7 | 1 identical, 3 site-shared, 4 overlap-only (of 11) |
| **AT4G22540** | Chr4 (−) | novel splice junction<br>11,864,127–11,865,712 | **strong** | AT4G22540.2 | 5 identical, 15 site-shared (of 36) |
| **AT3G56730** | Chr3 (+) | novel splice junction<br>21,014,509–21,014,635 | **strong** | AT3G56730.2 | 1 identical, 5 site-shared (of 6) |
| AT3G13740 | Chr3 (+) | unspliced intron in predicted UTR<br>4,505,796–4,505,914 | weak | AT3G13740.1 | 6 identical, 8 site-shared (of 14) |
| AT1G78940 | Chr1 (−) | chain combination only<br>2 junctions differ | weak | AT1G78940.1 | — (of 2) |
| AT3G29185 | Chr3 (−) | chain combination only<br>2 junctions differ | weak | AT3G29185.1 | — (of 20) |

**Recommended for the figure: AT2G37450** (categorical — TAIR10 splices straight over an exon
the model predicts), with **AT4G22540** as the best-supported alternative (5 independent
long-read transcripts carry the novel junction).

## Support categories — read these before quoting a number

A plain coordinate-overlap count overstates support and was the source of an earlier error in
this file (an "8 AtRTD3 transcripts" claim for AT2G37450 that was really 1). Every count is
now reported in disjoint categories:

- **identical** — carries the exact feature (same exon coordinates, or same junction)
- **splice-site-shared** — shares one boundary of the feature but not both
- **overlap-only** — spans the interval inside a *longer* exon; retained intron or alternative
  end, **not** the same cassette exon. Never support.
- **other** — the feature is absent

For AT2G37450 the breakdown over all 11 AtRTD3 transcripts is: identical **AT2G37450.7**;
splice-site-shared .3 / .5 / .8; overlap-only .1 / .4 / .10 / .11; absent .2 / .6 / .9.
Independently, the full 5-intron chain of the prediction matches AT2G37450.7 exactly at both
CDS and exon level, and each splice site of the novel exon is used by more than one long-read
transcript (acceptor 15,724,032 by .3/.7/.8; donor 15,724,094 by .5/.7).

## Why three loci are graded weak

- **AT3G13740** — the prediction's CDS stops early and the remaining 3′ region is emitted as
  one unspliced UTR that spans a TAIR10 intron. Real, but it reads as the model not splicing
  past the stop codon rather than as an alternative splicing event.
- **AT1G78940, AT3G29185** — every junction the novel chain uses (or omits) already occurs in
  some TAIR10 isoform; only the combination is new. At AT3G29185 both differences are intron
  retentions that TAIR10 already contains individually — `.2` retains the first, `.1` retains
  the second — so a panel claiming "TAIR10 does not retain this intron" would be false. The
  figure therefore shades the full symmetric difference and says "each junction occurs in
  TAIR10; this chain does not".

## Common figure rules

Track order TAIR10 → TransGenic (novel, then reproduced) → AtRTD3 (exact chain match, then
feature support, then others). CDS = tall box, UTR / non-coding exon = short box, intron =
line. Dark green marks **only** the transcript whose entire CDS chain equals the prediction;
transcripts that merely carry the highlighted feature are mid green. Fixed row height and
figure width across panels; x axis in kb with no offset notation.

## Files

| File | Contents |
|---|---|
| `<locus>_{pred,tair}.gff3`, `<locus>_atrtd.gtf` | per-locus extracts, all six loci |
| `find_panelC.py` | the original genome-wide scan that produced the candidate list |
| `audit_panelC.py` | prints the intron chains, feature and support breakdown per locus |
| `verify_panelC_figures.py` | asserts each drawn track matches its colour; non-zero exit on failure |

## Reproduce

```bash
cd /data/gpfs/assoc/pgl/data/Transgenic
python3 transgenic/Figures/make_figure4_panelC.py            # all six + combined sheet
python3 transgenic/Figures/make_figure4_panelC.py AT2G37450  # one locus

cd transgenic/revision/results/fig4_forensics/panelC_examples
python3 audit_panelC.py
python3 verify_panelC_figures.py     # exits 0 when every panel matches its data
```

Outputs `figure4_panelC_<locus>.{pdf,png}` for each locus plus `figure4_panelC_all.{pdf,png}`,
300 dpi, in `transgenic/Figures/`.

These are candidates independent of the original figure locus. **AT1G43770** remains the locus
confirmed (by its `temp.clean.gff3` prep file) to be in the published Figure 4; these six are
freshly verified alternatives if a new or replacement example is wanted.

---

## Genome-wide panel scan (2026-07-30) — `scan_panels.py`

All 4,875 prompted A. thaliana loci were classified on CDS intron chains against TAIR10 and
AtRTD3 (`panels.tsv`). Panel definitions follow the manuscript legend; A and B are separated
by how much was recovered, since the legend's wording overlaps.

| Panel | Criterion | Loci |
|---|---|---|
| A | 2 distinct TAIR10 chains reproduced, both in AtRTD3, nothing predicted outside them | 29 |
| B | **3+** distinct TAIR10 chains reproduced, 2+ in AtRTD3, nothing outside them | **3** |
| C | a predicted chain in AtRTD3 but in no TAIR10 transcript | 10 |

**AT1G43770 is classified A by this scan**, independently of the forensic evidence — the two
lines of evidence agree.

### Panel B: all three candidates, compared

| Locus | Chr | kb | TAIR10 chains | reproduced | AtRTD3-documented | shaded introns | verdict |
|---|---|---|---|---|---|---|---|
| **AT1G44575** | Chr1 (−) | 1.5 | 4 | **4 (all)** | 2 | 6 | **best** |
| AT5G01230 | Chr5 (+) | 2.7 | 4 | 3 | 2 | 14 | shading dominates |
| AT5G62890 | Chr5 (+) | 3.5 | 4 | 3 | 2 | 23 | unreadable — shading covers the panel |

**AT1G44575 is the only locus in the test set where the model recovered every distinct TAIR10
chain (4 of 4) while predicting nothing outside them.** That is precisely "recovered additional
TAIR10 isoforms supported by AtRTD3", and at 1.5 kb with six discriminating introns it stays
legible. The other two reproduce 3 of 4 chains but differ at 14 and 23 introns respectively,
so the highlight bands swamp the figure.

### Resulting Figure 4

| Panel | Locus | Basis |
|---|---|---|
| A | **AT1G43770** | forensically anchored (the only plot-prep file in the published model's directory) **and** independently classified A by the scan |
| B | **AT1G44575** | best of 3 candidates genome-wide |
| C | **AT2G37450** | cassette exon; AT4G22540 / AT3G56730 available as further panels |

Every panel now has a source file, a drawing script, and an automated check
(`verify_panelC_figures.py`, currently 10/10 loci).
