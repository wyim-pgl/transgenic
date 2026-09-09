"""protein_accept.py: mapping_ambiguous proteins give family-level support only (protocol A19.2).

Support version 1 wrote every accepted alignment's CDS blocks and introns during the streaming pass and
named the ambiguous proteins afterwards, so their locus-specific support survived into <species>.tsv and
<species>.introns.tsv (Codex review 2026-09-09; 1.4-4.1 % of proteins per species on pronghorn).

The fixture is a three-protein miniprot GFF:
  org1:A  two equal-best loci (chr1 and chr3)   -> mapping_ambiguous, no locus-specific support
  org2:B  one locus, chr1, same intron as A      -> supports chr1
  org3:C  one locus, chr1, same intron as A      -> supports chr1 (intron now 2 organisms, not 3)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "revision" / "scripts" / "evidence" / "protein_accept.py"

pytestmark = pytest.mark.skipif(shutil.which("sort") is None, reason="coreutils sort is required")


def _paf(prot: str, chrom: str) -> str:
    # qname qlen qstart qend strand tname tlen tstart tend match blocklen mapq  tags...
    return (f"##PAF\t{prot}\t100\t0\t100\t+\t{chrom}\t100000\t99\t400\t95\t100\t60\t"
            f"AS:i:500\tfs:i:0\tcs:Z::150~gt100ag:150\n")


def _alignment(prot: str, chrom: str, mid: str, score: int = 500) -> str:
    return (_paf(prot, chrom)
            + f"{chrom}\tminiprot\tmRNA\t100\t400\t{score}\t+\t.\tID={mid};Rank=1;Identity=0.95;Target={prot} 1 100\n"
            + f"{chrom}\tminiprot\tCDS\t100\t200\t{score}\t+\t0\tParent={mid}\n"
            + f"{chrom}\tminiprot\tCDS\t301\t400\t{score}\t+\t0\tParent={mid}\n")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("accept")
    gff = out / "toy.gff"
    gff.write_text("##gff-version 3\n"
                   + _alignment("org1:A", "chr1", "MP1")
                   + _alignment("org1:A", "chr3", "MP2")          # equal-best second locus
                   + _alignment("org2:B", "chr1", "MP3")
                   + _alignment("org3:C", "chr1", "MP4"))
    cmd = [sys.executable, str(SCRIPT), "--gff", str(gff), "--species", "toy", "--out-dir", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return out


def _rows(path: Path):
    return [line.rstrip("\n").split("\t") for line in path.read_text().splitlines()[1:]]


def test_every_alignment_is_judged_and_accepted(run):
    summary = json.loads((run / "toy.summary.json").read_text())
    assert summary["alignments"] == 4 and summary["accepted"] == 4
    assert summary["proteins"] == 3


def test_ambiguous_protein_is_named(run):
    assert (run / "toy.mapping_ambiguous.txt").read_text().split() == ["org1:A"]
    summary = json.loads((run / "toy.summary.json").read_text())
    assert summary["mapping_ambiguous"] == 1


def test_ambiguous_protein_gives_no_locus_specific_support(run):
    coverage = _rows(run / "toy.tsv")
    assert coverage, "the unique proteins must still produce coverage"
    assert all(r[0] != "chr3" for r in coverage), "org1:A's chr3 locus leaked into the coverage table"
    # chr1 is covered by org2 and org3 only: the organism depth never reaches 3
    assert max(int(r[3]) for r in coverage) == 2
    assert {r[0] for r in coverage} == {"chr1"}


def test_intron_support_counts_only_unambiguous_organisms(run):
    introns = _rows(run / "toy.introns.tsv")
    assert len(introns) == 1
    chrom, start, end, strand, motif, orgs, n = introns[0]
    assert (chrom, start, end, strand, motif) == ("chr1", "201", "300", "+", "gt-ag")
    assert orgs == "org2,org3" and n == "2"


def test_summary_records_the_exclusion(run):
    summary = json.loads((run / "toy.summary.json").read_text())
    assert summary["support_version"] == 2
    assert summary["support_excludes_mapping_ambiguous"] is True
    assert summary["ambiguous_alignments_excluded"] == 2
    assert summary["ambiguous_blocks_excluded"] == 4       # 2 CDS blocks x 2 loci
    assert summary["ambiguous_introns_excluded"] == 2
    assert summary["introns_supported_by_2plus_organisms"] == 1


def test_alignments_table_still_lists_the_ambiguous_alignments(run):
    """Per-alignment statistics are unchanged: acceptance is judged per alignment, ambiguity per protein."""
    rows = _rows(run / "toy.alignments.tsv")
    assert sorted(r[0] for r in rows) == ["MP1", "MP2", "MP3", "MP4"]
    assert all(r[10] == "accepted" for r in rows)


def test_intermediates_are_removed(run):
    assert not list(run.glob("*.tmp"))
