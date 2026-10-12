"""Shared MVP panel window derivation and restriction predicates (B6_MVP_PILOT_v1 §1.2/§2).

This module is the single source of the 30,720-nt evaluation-window definition.
Every consumer — the decoder, the pass-1 long-read prefilter, the junction table,
and the scoring restriction predicates — must derive windows from here so that the
window used for scoring is exactly the window that was decoded (Track B review
must-fix 1).

Coordinate conventions
----------------------
* Panel TSV `start`/`fin` are 1-based inclusive (spec §1.2). We convert once to
  half-open 0-based `[gene_start0, gene_end0)` and never mix conventions.
* Evaluation window: WINDOW_LENGTH = 30,720 nt CENTRED on the gene. With
  ``pad_total = WINDOW_LENGTH - gene_length`` the left pad is ``pad_total // 2``
  and the right pad is ``pad_total - pad_total // 2`` (the odd base goes to the
  genomic right). This is deterministic and documented here once.
* Prefilter interval: window +/- PREFILTER_PAD. The spec text says "panel
  windows ± 10 kb"; "10 kb" is resolved as the decimal 10,000 nt, NOT 10,240
  (the 10,240 figure appearing elsewhere is a tile-geometry constant, not the
  spec's prefilter pad). Frozen here as PREFILTER_PAD = 10_000.
* Junction intron coordinates are 0-based INCLUSIVE endpoints
  (`intron_start0`, `intron_end0`), matching the C0-lite `est_junction` table.

Two restriction scopes (must-fix 2) — do not conflate them:
* READ-SUPPORTED junctions are counted inside panel WINDOWS
  (`junction_within_windows`, containment in the 30,720-nt window).
* PREDICTED junctions and the recall denominator are restricted against panel
  GENES (`transcript_overlaps_gene` for selecting decoded transcripts,
  `junction_within_genes` for the read-supported recall denominator).

Fail-closed: a window that would extend beyond the chromosome (or below 0) is
an error, never a silent clamp/shift; that case needs an author decision.
"""

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

WINDOW_LENGTH = 30_720
PREFILTER_PAD = 10_000  # spec "10 kb" == 10,000 nt (decimal); NOT 10,240.

# Commit-bound trust anchor: sha256 of revision/protocols/b6_panels_v1.tsv at
# f847cb80c7750326f0781f81481891ab0c9ea309 (matches the committed .sha256
# companion; verified by the Track B adversarial review).
PANEL_SHA256 = "79367154bbf922fd0554285cb6cb47ae3d3c866b145aa1f51ead4f47b00cd243"

PANEL_COLUMNS = (
    "panel", "shuffle_rank", "species_id", "gene_id", "chromosome", "start",
    "fin", "strand", "length", "orthogroup_id", "n_transcripts", "exon_count",
    "intron_count", "est_molecules",
)
MVP_LABELS = ("mvp_core", "mvp")
MVP_EXPECTED_COUNTS = {"mvp_core": 20, "mvp": 30}


class WindowError(ValueError):
    """A panel/window inconsistency must stop scoring, not degrade it."""


def _require(condition, message):
    if not condition:
        raise WindowError(message)


@dataclass(frozen=True)
class PanelLocus:
    panel: str
    gene_id: str
    chromosome: str
    start1: int   # 1-based inclusive, as in the TSV
    fin1: int     # 1-based inclusive
    strand: str
    length: int

    @property
    def gene_start0(self):
        return self.start1 - 1

    @property
    def gene_end0(self):
        """Half-open end: [gene_start0, gene_end0)."""
        return self.fin1


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_panel(path, expected_sha256=PANEL_SHA256, labels=MVP_LABELS):
    """Load the MVP panel loci fail-closed.

    The default `expected_sha256` pins the commit-bound b6_panels_v1.tsv hash;
    tests may pass the hash of a fixture explicitly, but there is no way to
    skip the check.
    """
    path = Path(path)
    _require(isinstance(expected_sha256, str) and len(expected_sha256) == 64,
             "an explicit expected sha256 is required")
    actual = sha256_file(path)
    _require(actual == expected_sha256,
             f"panel TSV hash mismatch: {path} has {actual}, expected {expected_sha256}")
    loci = []
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = tuple(next(reader))
        _require(header == PANEL_COLUMNS,
                 f"panel TSV header mismatch: {header!r}")
        for row in reader:
            _require(len(row) == len(PANEL_COLUMNS), f"panel row has {len(row)} fields")
            record = dict(zip(PANEL_COLUMNS, row))
            if record["panel"] not in labels:
                continue
            start1, fin1, length = (int(record["start"]), int(record["fin"]),
                                    int(record["length"]))
            _require(1 <= start1 <= fin1, f"{record['gene_id']}: invalid start/fin")
            _require(length == fin1 - start1 + 1,
                     f"{record['gene_id']}: length column disagrees with start/fin")
            _require(record["strand"] in ("+", "-"),
                     f"{record['gene_id']}: invalid strand")
            loci.append(PanelLocus(record["panel"], record["gene_id"],
                                   record["chromosome"], start1, fin1,
                                   record["strand"], length))
    counts = {}
    for locus in loci:
        counts[locus.panel] = counts.get(locus.panel, 0) + 1
    if set(labels) == set(MVP_LABELS):
        _require(counts == MVP_EXPECTED_COUNTS,
                 f"MVP panel counts {counts} != {MVP_EXPECTED_COUNTS}")
    _require(len({(l.chromosome, l.gene_id) for l in loci}) == len(loci),
             "duplicate panel gene")
    return tuple(loci)


def evaluation_window(locus, chrom_length=None):
    """30,720-nt half-open window `[w_start0, w_end0)` centred on the gene.

    This is THE window the decoder decodes and the scorer restricts to.
    Fail-closed at chromosome edges: no silent clamping.
    """
    gene_len = locus.gene_end0 - locus.gene_start0
    _require(gene_len == locus.length, f"{locus.gene_id}: inconsistent gene length")
    _require(gene_len <= WINDOW_LENGTH,
             f"{locus.gene_id}: gene length {gene_len} exceeds window {WINDOW_LENGTH}")
    pad_total = WINDOW_LENGTH - gene_len
    pad_left = pad_total // 2          # odd base goes to the genomic right
    w_start0 = locus.gene_start0 - pad_left
    w_end0 = w_start0 + WINDOW_LENGTH
    _require(w_start0 >= 0,
             f"{locus.gene_id}: evaluation window would start before the chromosome "
             f"({w_start0}); author decision required, no silent clamp")
    if chrom_length is not None:
        _require(w_end0 <= chrom_length,
                 f"{locus.gene_id}: evaluation window would end past the chromosome; "
                 f"author decision required, no silent clamp")
    return (w_start0, w_end0)


def prefilter_interval(window):
    """Pass-1 candidate-read interval: evaluation window +/- 10,000 nt.

    The left edge is floored at 0 — the PREFILTER is a superset selector, so
    flooring can only keep extra reads, never drop evaluation-window evidence
    (unlike the window itself, which is fail-closed).
    """
    w_start0, w_end0 = window
    _require(w_end0 - w_start0 == WINDOW_LENGTH, "not an evaluation window")
    return (max(0, w_start0 - PREFILTER_PAD), w_end0 + PREFILTER_PAD)


def windows_by_chromosome(loci, chrom_lengths=None):
    """Map chromosome -> sorted list of evaluation windows (half-open)."""
    result = {}
    for locus in loci:
        length = None if chrom_lengths is None else chrom_lengths[locus.chromosome]
        result.setdefault(locus.chromosome, []).append(evaluation_window(locus, length))
    return {chrom: sorted(spans) for chrom, spans in result.items()}


def prefilter_by_chromosome(loci):
    """Map chromosome -> sorted list of pass-1 prefilter intervals (half-open)."""
    result = {}
    for locus in loci:
        result.setdefault(locus.chromosome, []).append(
            prefilter_interval(evaluation_window(locus)))
    return {chrom: sorted(spans) for chrom, spans in result.items()}


def _contained(span_start0, span_end0_incl, intervals):
    """0-based-inclusive span fully inside any half-open interval."""
    _require(span_start0 <= span_end0_incl, "invalid span")
    return any(start0 <= span_start0 and span_end0_incl < end0
               for start0, end0 in intervals)


def _overlapping(span_start0, span_end0_incl, intervals):
    """0-based-inclusive span overlapping any half-open interval by >= 1 base."""
    _require(span_start0 <= span_end0_incl, "invalid span")
    return any(span_start0 < end0 and span_end0_incl >= start0
               for start0, end0 in intervals)


def junction_within_windows(chrom, intron_start0, intron_end0, windows):
    """Scope 1 (must-fix 2): READ-SUPPORTED junctions count inside panel WINDOWS.

    `windows` is the mapping from `windows_by_chromosome`. Containment of the
    0-based inclusive intron span in the 30,720-nt window.
    """
    return _contained(intron_start0, intron_end0, windows.get(chrom, ()))


def genes_by_chromosome(loci):
    """Map chromosome -> sorted list of half-open gene spans."""
    result = {}
    for locus in loci:
        result.setdefault(locus.chromosome, []).append(
            (locus.gene_start0, locus.gene_end0))
    return {chrom: sorted(spans) for chrom, spans in result.items()}


def junction_within_genes(chrom, intron_start0, intron_end0, genes):
    """Scope 2a (must-fix 2): recall DENOMINATOR — read-supported junctions
    inside the panel GENES (containment in the gene span, spec §2 'inside the
    panel genes'). `genes` is the mapping from `genes_by_chromosome`."""
    return _contained(intron_start0, intron_end0, genes.get(chrom, ()))


def transcript_overlaps_gene(chrom, tx_start0, tx_end0_incl, genes):
    """Scope 2b (must-fix 2): PREDICTED junctions come from decoded transcripts
    OVERLAPPING the panel gene (spec §2 'introns of the decoded transcripts
    overlapping the panel gene'): >= 1 base of transcript-span overlap."""
    return _overlapping(tx_start0, tx_end0_incl, genes.get(chrom, ()))
