"""Tests for 36_filter_additions_structurally.py.

The filter's whole value is that it discards a lot while keeping the correct structures, so
these tests are written against the two ways that can go wrong: discarding something sound,
and keeping something unsound. The supplied prompt must survive unconditionally — filtering
it out would delete the annotation the user handed in.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "filter_additions", Path(__file__).resolve().parents[1] / "36_filter_additions_structurally.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["filter_additions"] = mod
spec.loader.exec_module(mod)


# --------------------------------------------------------------------------- fixtures

def _fasta(path: Path, sequences: dict) -> None:
    path.write_text("".join(f">{n}\n{s}\n" for n, s in sequences.items()))


def _gff3(path: Path, loci: list) -> None:
    lines = ["##gff-version 3"]
    for locus in loci:
        seq = locus.get("seq", "Chr1")
        strand = locus.get("strand", "+")
        gm = f";GM={locus['gm']}" if locus.get("gm") else ""
        segs = [s for _t, ss in locus["transcripts"] for s in ss]
        lines.append(f"{seq}\tsrc\tgene\t{min(s for s, _ in segs)}\t{max(e for _, e in segs)}"
                     f"\t.\t{strand}\t.\tID={locus['gene']}{gm}")
        for tx, ss in locus["transcripts"]:
            lines.append(f"{seq}\tsrc\tmRNA\t{min(s for s, _ in ss)}\t{max(e for _, e in ss)}"
                         f"\t.\t{strand}\t.\tID={tx};Parent={locus['gene']}{gm}")
            for n, (s, e) in enumerate(ss, 1):
                lines.append(f"{seq}\tsrc\tCDS\t{s}\t{e}\t.\t{strand}\t0\t"
                             f"ID={tx}.c{n};Parent={tx}{gm}")
    path.write_text("\n".join(lines) + "\n")


# Coordinate map of the fixture genome. Every case the filter distinguishes needs its own
# region, and each spliced case needs its exons to abut its intron exactly — an off-by-one
# there silently turns a "canonical intron" case into a non-canonical one.
#
#    1-6    ATGAAA        exon A1  ┐ spliced, complete ORF, canonical intron
#    7-16   GTCCCCCCAG    intron   │ GT..AG
#   17-19   TAA           exon A2  ┘ A1+A2 = ATGAAATAA
#   20-28   ATGAAATAA     single-exon complete ORF
#   29-37   AAAAAATAA     no start codon
#   38-43   ATGAAA        exon B1  ┐ spliced, complete ORF, NON-canonical intron
#   44-53   ATCCCCCCTC    intron   │ AT..TC
#   54-56   TAA           exon B2  ┘ B1+B2 = ATGAAATAA
#   57-70   C*14          padding
SPLICED_CANONICAL = [(1, 6), (17, 19)]
SINGLE_EXON_ORF = [(20, 28)]
NO_START_CODON = [(29, 37)]
SPLICED_NON_CANONICAL = [(38, 43), (54, 56)]


@pytest.fixture
def genome(tmp_path):
    # Every piece is joined with an explicit `+`. Mixing implicit literal concatenation
    # with `+` here silently regrouped the parts and produced a 135 nt sequence whose
    # regions no longer matched the map above.
    parts = [
        "ATGAAA",                 # 1-6    exon A1
        "GT" + "C" * 6 + "AG",    # 7-16   canonical intron
        "TAA",                    # 17-19  exon A2
        "ATGAAATAA",              # 20-28  single-exon ORF
        "AAAAAATAA",              # 29-37  no start codon
        "ATGAAA",                 # 38-43  exon B1
        "AT" + "C" * 6 + "TC",    # 44-53  NON-canonical intron
        "TAA",                    # 54-56  exon B2
        "C" * 14,                 # 57-70  padding
    ]
    seq = "".join(parts)
    assert len(seq) == 70, f"fixture genome is {len(seq)} nt, expected 70"
    path = tmp_path / "genome.fa"
    _fasta(path, {"Chr1": seq})
    return path


def _run(tmp_path, genome, prompt_loci, output_loci, filtered=None):
    prompt = tmp_path / "prompt.gff3"
    completed = tmp_path / "completed.gff3"
    _gff3(prompt, prompt_loci)
    _gff3(completed, output_loci)
    return mod.filter_predictions(prompt, completed, genome, filtered)


# ------------------------------------------------------------------- ORF completeness

def test_addition_with_a_complete_orf_is_kept(tmp_path, genome):
    stats = _run(tmp_path, genome,
                 [{"gene": "G1", "transcripts": [("G1.t1", SINGLE_EXON_ORF)]}],
                 [{"gene": "O1", "gm": "G1",
                   "transcripts": [("O1.t1", SINGLE_EXON_ORF),
                                   ("O1.t2", SPLICED_CANONICAL)]}])
    assert stats["additions"] == 1
    assert stats["additions_kept"] == 1
    assert stats["failed_orf"] == 0


def test_addition_without_a_start_codon_is_discarded(tmp_path, genome):
    stats = _run(tmp_path, genome,
                 [{"gene": "G1", "transcripts": [("G1.t1", SINGLE_EXON_ORF)]}],
                 [{"gene": "O1", "gm": "G1",
                   "transcripts": [("O1.t1", SINGLE_EXON_ORF), ("O1.t2", NO_START_CODON)]}])
    assert stats["additions"] == 1
    assert stats["additions_kept"] == 0
    assert stats["failed_orf"] == 1


def test_length_not_divisible_by_three_is_discarded(tmp_path, genome):
    stats = _run(tmp_path, genome,
                 [{"gene": "G1", "transcripts": [("G1.t1", SINGLE_EXON_ORF)]}],
                 [{"gene": "O1", "gm": "G1",
                   "transcripts": [("O1.t1", SINGLE_EXON_ORF), ("O1.t2", [(20, 27)])]}])
    assert stats["additions_kept"] == 0
    assert stats["failed_orf"] == 1


# ------------------------------------------------------------------------ splice sites

def test_canonical_gt_ag_intron_is_kept(tmp_path, genome):
    stats = _run(tmp_path, genome,
                 [{"gene": "G1", "transcripts": [("G1.t1", SINGLE_EXON_ORF)]}],
                 [{"gene": "O1", "gm": "G1",
                   "transcripts": [("O1.t1", SINGLE_EXON_ORF),
                                   ("O1.t2", SPLICED_CANONICAL)]}])
    assert stats["additions"] == 1
    assert stats["failed_splice"] == 0
    assert stats["additions_kept"] == 1


def test_non_canonical_intron_is_discarded_though_its_orf_is_complete(tmp_path, genome):
    """Isolates the splice test: this structure's CDS is a valid ORF, only the intron is not."""
    stats = _run(tmp_path, genome,
                 [{"gene": "G1", "transcripts": [("G1.t1", SINGLE_EXON_ORF)]}],
                 [{"gene": "O1", "gm": "G1",
                   "transcripts": [("O1.t1", SINGLE_EXON_ORF),
                                   ("O1.t2", SPLICED_NON_CANONICAL)]}])
    assert stats["additions"] == 1
    assert stats["failed_orf"] == 0
    assert stats["failed_splice"] == 1
    assert stats["additions_kept"] == 0


def test_single_exon_structure_is_not_failed_for_lacking_introns(tmp_path, genome):
    stats = _run(tmp_path, genome,
                 [{"gene": "G1", "transcripts": [("G1.t1", SPLICED_CANONICAL)]}],
                 [{"gene": "O1", "gm": "G1",
                   "transcripts": [("O1.t1", SPLICED_CANONICAL),
                                   ("O1.t2", SINGLE_EXON_ORF)]}])
    assert stats["failed_splice"] == 0
    assert stats["additions_kept"] == 1


# ------------------------------------------------------------- the prompt is never filtered

def test_the_supplied_structure_survives_even_if_it_would_fail(tmp_path, genome):
    """Filtering out the user's own annotation would be data loss, not curation."""
    out = tmp_path / "filtered.gff3"
    stats = _run(tmp_path, genome,
                 [{"gene": "G1", "transcripts": [("G1.t1", NO_START_CODON)]}],
                 [{"gene": "O1", "gm": "G1", "transcripts": [("O1.t1", NO_START_CODON)]}],
                 filtered=out)
    assert stats["prompt_re_emissions"] == 1
    assert stats["additions"] == 0
    assert "O1.t1" in out.read_text()


def test_prompt_re_emission_is_not_counted_as_an_addition(tmp_path, genome):
    stats = _run(tmp_path, genome,
                 [{"gene": "G1", "transcripts": [("G1.t1", SINGLE_EXON_ORF)]}],
                 [{"gene": "O1", "gm": "G1", "transcripts": [("O1.t1", SINGLE_EXON_ORF)]}])
    assert stats["additions"] == 0
    assert stats["additions_kept"] == 0
    assert stats["additions_discarded_pct"] is None   # N/A, never 0.0


# ------------------------------------------------------------------------ bookkeeping

def test_identical_emissions_count_once(tmp_path, genome):
    stats = _run(tmp_path, genome,
                 [{"gene": "G1", "transcripts": [("G1.t1", SINGLE_EXON_ORF)]}],
                 [{"gene": "O1", "gm": "G1",
                   "transcripts": [("O1.t1", SINGLE_EXON_ORF),
                                   ("O1.t2", SPLICED_CANONICAL),
                                   ("O1.t3", SPLICED_CANONICAL)]}])
    assert stats["additions"] == 1
    assert stats["additions_kept"] == 1


def test_unknown_seqid_is_reported_not_silently_dropped(tmp_path, genome):
    stats = _run(tmp_path, genome,
                 [{"gene": "G1", "seq": "ChrZ", "transcripts": [("G1.t1", SINGLE_EXON_ORF)]}],
                 [{"gene": "O1", "seq": "ChrZ", "gm": "G1",
                   "transcripts": [("O1.t1", SINGLE_EXON_ORF),
                                   ("O1.t2", SPLICED_CANONICAL)]}])
    assert stats["unknown_seqid"] == 1
    assert stats["additions_kept"] == 0


def test_discard_percentage_is_reported(tmp_path, genome):
    stats = _run(tmp_path, genome,
                 [{"gene": "G1", "transcripts": [("G1.t1", SINGLE_EXON_ORF)]}],
                 [{"gene": "O1", "gm": "G1",
                   "transcripts": [("O1.t1", SINGLE_EXON_ORF),
                                   ("O1.t2", SPLICED_CANONICAL),
                                   ("O1.t3", NO_START_CODON)]}])
    assert stats["additions"] == 2
    assert stats["additions_kept"] == 1
    assert stats["additions_discarded_pct"] == 50.0


def test_empty_output_raises_rather_than_reporting_a_clean_filter(tmp_path, genome):
    prompt, completed = tmp_path / "p.gff3", tmp_path / "c.gff3"
    _gff3(prompt, [{"gene": "G1", "transcripts": [("G1.t1", SINGLE_EXON_ORF)]}])
    completed.write_text("##gff-version 3\n")
    with pytest.raises(ValueError):
        mod.filter_predictions(prompt, completed, genome, None)


# --------------------------------------------------------------------- genome aliasing

def test_ensembl_named_genome_joins_to_tair_named_annotation(tmp_path):
    """The shipped TAIR10 FASTA is named 1..5,Mt,Pt while annotations use Chr1..ChrM."""
    path = tmp_path / "ensembl.fa"
    _fasta(path, {"1": "ATGAAATAA" + "C" * 51})
    genome = mod.load_genome(path)
    assert "Chr1" in genome and "1" in genome
    assert genome["Chr1"] == genome["1"]


def test_written_file_keeps_only_surviving_transcripts(tmp_path, genome):
    out = tmp_path / "filtered.gff3"
    _run(tmp_path, genome,
         [{"gene": "G1", "transcripts": [("G1.t1", SINGLE_EXON_ORF)]}],
         [{"gene": "O1", "gm": "G1",
           "transcripts": [("O1.t1", SINGLE_EXON_ORF), ("O1.t2", NO_START_CODON)]}],
         filtered=out)
    text = out.read_text()
    assert "O1.t1" in text
    assert "O1.t2" not in text


# ------------------------------------------------------------------------- ORF audit
#
# The audit exists to catch the failure that produced no exception: a genome whose sequence
# names do not join to the annotation's, which makes every predicate return False. So the
# tests below check that a sound annotation scores high, an unsound one scores low, and a
# non-joining genome is reported as skipped rather than as zero percent.

def test_audit_scores_a_sound_annotation_at_100_percent(tmp_path, genome):
    path = tmp_path / "sound.gff3"
    _gff3(path, [{"gene": "G1", "transcripts": [("G1.t1", SINGLE_EXON_ORF),
                                                ("G1.t2", SPLICED_CANONICAL)]}])
    stats = mod.orf_audit(path, genome)
    assert stats["transcripts_scored"] == 2
    assert stats["complete_orf_pct"] == 100.0
    assert stats["canonical_introns_pct"] == 100.0
    assert stats["both_pct"] == 100.0


def test_audit_separates_the_two_criteria(tmp_path, genome):
    """A transcript can have a complete ORF and a non-canonical intron, and vice versa."""
    path = tmp_path / "mixed.gff3"
    _gff3(path, [{"gene": "G1", "transcripts": [("G1.t1", SPLICED_NON_CANONICAL),
                                                ("G1.t2", NO_START_CODON)]}])
    stats = mod.orf_audit(path, genome)
    assert stats["complete_orf"] == 1        # the non-canonical one still reads ATG..TAA
    assert stats["canonical_introns"] == 1   # the single-exon one has no intron to fail
    assert stats["both"] == 0


def test_audit_reports_a_non_joining_genome_as_skipped_not_as_zero(tmp_path, genome):
    """The bug this exists for: names that do not match must not read as 'all invalid'."""
    path = tmp_path / "elsewhere.gff3"
    _gff3(path, [{"gene": "G1", "seq": "scaffold_99",
                  "transcripts": [("G1.t1", SINGLE_EXON_ORF)]}])
    stats = mod.orf_audit(path, genome)
    assert stats["transcripts_scored"] == 0
    assert stats["transcripts_skipped_unknown_seqid"] == 1
    assert stats["complete_orf_pct"] == 0.0
