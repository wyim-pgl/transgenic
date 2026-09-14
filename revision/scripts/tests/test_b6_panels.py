"""B6 locus panels (A46): eligibility helpers of 71_b6_panels.py, panel assignment of
72_mvp_select_loci.py and the samtools helper 72a_est_molecules_per_locus.sh."""
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


panels71 = load("panels71", "71_b6_panels.py")
select72 = load("select72", "72_mvp_select_loci.py")

GFF = """\
Chr1\tsrc\tgene\t101\t400\t.\t+\t.\tID=G1;Name=G1
Chr1\tsrc\tmRNA\t101\t400\t.\t+\t.\tID=G1.1;Parent=G1
Chr1\tsrc\texon\t101\t150\t.\t+\t.\tID=e1;Parent=G1.1
Chr1\tsrc\texon\t200\t250\t.\t+\t.\tID=e2;Parent=G1.1
Chr1\tsrc\texon\t300\t400\t.\t+\t.\tID=e3;Parent=G1.1
Chr1\tsrc\tmRNA\t101\t400\t.\t+\t.\tID=G1.2;Parent=G1
Chr1\tsrc\texon\t101\t150\t.\t+\t.\tID=e4;Parent=G1.2
Chr1\tsrc\texon\t300\t400\t.\t+\t.\tID=e5;Parent=G1.2
Chr1\tsrc\tgene\t501\t600\t.\t-\t.\tID=G2
Chr1\tsrc\tmRNA\t501\t600\t.\t-\t.\tID=G2.1;Parent=G2
Chr1\tsrc\tCDS\t501\t600\t.\t-\t0\tID=c1;Parent=G2.1
"""


def test_parse_gff_counts_max_exons_per_transcript_and_cds_fallback(tmp_path):
    gff = tmp_path / "g.gff3"
    gff.write_text(GFF)
    genes = panels71.parse_gff_genes(gff)
    assert genes["G1"] == {"chrom": "Chr1", "start0": 100, "end0": 400, "strand": "+", "n_transcripts": 2, "exon_count": 3}
    assert genes["G2"]["exon_count"] == 1 and genes["G2"]["n_transcripts"] == 1 and genes["G2"]["strand"] == "-"


def test_overlapping_genes_marks_both_partners_on_either_strand():
    # (gene_id, gene_id_original, chromosome, start0, end0, ...)
    genes = [("A", "a", "Chr1", 0, 100), ("B", "b", "Chr1", 99, 200), ("C", "c", "Chr1", 200, 300),
             ("D", "d", "Chr2", 0, 100)]
    assert panels71.overlapping_genes(genes, 0) == {"A", "B"}          # half-open: C touches B without overlap
    assert panels71.overlapping_genes(genes, 1) == {"A", "B", "C"}     # a 1-nt flank makes B and C overlap


def make_pool(n, long_ranks=()):
    rows = []
    for i in range(1, n + 1):
        length = 25000 if i in long_ranks else 1000 + i
        rows.append({"shuffle_rank": i, "species_id": "Athaliana", "gene_id": f"Ath{i:06d}", "chromosome": "Chr1",
                     "start": 10000 * i, "fin": 10000 * i + length - 1, "strand": "+", "length": length,
                     "orthogroup_id": "" if i % 7 == 0 else f"HOG{i}", "n_transcripts": 1, "exon_count": 3, "intron_count": 2})
    return rows


def test_select_panels_walks_rank_order_and_applies_sub_filter():
    pool = make_pool(30, long_ranks=(2,))
    counts = {r["gene_id"]: (1 if r["shuffle_rank"] == 4 else 5) for r in pool}
    report = []
    rows = select72.select_panels(pool, counts, mvp=6, core=2, pilot=10, max_len=20000, min_est=3, report=report)
    by_rank = {r["shuffle_rank"]: r for r in rows}
    assert [r["shuffle_rank"] for r in rows if r["panel"] == "mvp_core"] == [1, 3]      # rank 2 too long
    assert [r["shuffle_rank"] for r in rows if r["panel"] == "mvp"] == [5, 6, 7, 8]      # rank 4 too few ESTs
    assert [r["shuffle_rank"] for r in rows if r["panel"] == "pilot"] == [2, 4] + list(range(9, 17))
    assert all(r["panel"] == "final" for r in rows if r["shuffle_rank"] > 16)
    assert by_rank[1]["est_molecules"] == 5 and by_rank[2]["est_molecules"] == "" and by_rank[9]["est_molecules"] == ""
    assert [r["shuffle_rank"] for r in rows] == list(range(1, 31))
    assert "panel_counts=" in report[-1] and '"mvp_core": 2' in report[-1] and '"pilot": 10' in report[-1]


def test_select_panels_errors_when_walked_locus_has_no_count_or_pool_exhausted():
    pool = make_pool(5)
    with pytest.raises(select72.SelectError, match="no entry"):
        select72.select_panels(pool, {}, 2, 1, 1, 20000, 3, [])
    with pytest.raises(select72.SelectError, match="pool exhausted"):
        select72.select_panels(pool, {r["gene_id"]: 0 for r in pool}, 2, 1, 1, 20000, 3, [])


def test_cli_roundtrip_verifies_sha_and_writes_panels(tmp_path):
    pool = make_pool(12)
    eligible = tmp_path / "eligible.tsv"
    select72.write_tsv(pool, select72.IN_COLUMNS, eligible)  # also writes eligible.tsv.sha256
    counts = tmp_path / "est.tsv"
    counts.write_text("gene_id\test_molecules\n" + "".join(f"{r['gene_id']}\t4\n" for r in pool))
    out = tmp_path / "panels.tsv"
    argv = ["--eligible", str(eligible), "--est-counts", str(counts), "--out", str(out), "--mvp", "3", "--core", "1", "--pilot", "4"]
    assert select72.main(argv) == 0
    lines = out.read_text().splitlines()
    assert lines[0].split("\t") == list(select72.OUT_COLUMNS)
    assert [l.split("\t")[0] for l in lines[1:]] == ["mvp_core", "mvp", "mvp"] + ["pilot"] * 4 + ["final"] * 5
    assert (tmp_path / "panels.tsv.sha256").read_text().split()[1] == "panels.tsv"
    eligible.write_bytes(eligible.read_bytes() + b"\n")  # tamper -> recorded sha no longer matches
    assert select72.main(argv) == 1


@pytest.mark.skipif(shutil.which("samtools") is None, reason="samtools not installed")
def test_est_helper_counts_distinct_primary_read_names(tmp_path):
    seq = "A" * 100
    sam = ["@HD\tVN:1.6\tSO:coordinate", "@SQ\tSN:Chr1\tLN:100000"]
    reads = [("r1", 0, 1100, "100M"), ("r2", 16, 1200, "50M10N50M"), ("r2b", 16, 1200, "50M10N50M"),  # same placement as r2 -> collapses (§1.3 rule a)
             ("r2", 272, 1300, "100M"), ("r3", 2048, 1400, "100M"), ("r4", 0, 5000, "100M")]  # secondary, supplementary, outside
    for name, flag, pos, cigar in sorted(reads, key=lambda r: r[2]):
        sam.append(f"{name}\t{flag}\tChr1\t{pos}\t60\t{cigar}\t*\t0\t0\t{seq}\t*")
    (tmp_path / "t.sam").write_text("\n".join(sam) + "\n")
    subprocess.run(["samtools", "view", "-b", "-o", str(tmp_path / "t.bam"), str(tmp_path / "t.sam")], check=True)
    subprocess.run(["samtools", "index", str(tmp_path / "t.bam")], check=True)
    eligible = tmp_path / "eligible.tsv"
    select72.write_tsv(make_pool(2), select72.IN_COLUMNS, eligible)  # rank 1: Chr1:10000-11000, rank 2: 20000-21001
    # move locus 1 onto the reads: rewrite start/fin of rank 1 to 1000-1500
    text = eligible.read_text().splitlines()
    cols = text[1].split("\t"); cols[4], cols[5], cols[7] = "1000", "1500", "501"; text[1] = "\t".join(cols)
    eligible.write_text("\n".join(text) + "\n")
    out = tmp_path / "est.tsv"
    subprocess.run(["bash", str(SCRIPTS / "72a_est_molecules_per_locus.sh"), str(eligible), str(tmp_path / "t.bam"), str(out)], check=True)
    assert out.read_text() == "gene_id\test_molecules\nAth000001\t2\nAth000002\t0\n"
    assert (tmp_path / "est.tsv.sha256").exists()


def test_hard_flag_patterns_match_frozen_qc_flags_module():
    qc = load("qc_flags", Path(__file__).resolve().parents[3] / "src" / "transgenic" / "datasets" / "qc_flags.py")
    assert panels71.HARD_FLAG_PATTERNS == qc.HARD_FLAG_PATTERNS
    assert panels71.is_hard("missing_stop_codon") and panels71.is_hard("swissprot_caution_frameshift")
    assert not panels71.is_hard("missing_utr_5p") and not panels71.is_hard("swissprot_note_caution_resolved_in_reference")


def test_read_hard_flags_keeps_only_hard_flags_of_the_species(tmp_path):
    f = tmp_path / "flags.tsv"
    f.write_text("species_id\tgene_id\ttranscript_id\tflag\tstart\tend\n"
                 "Athaliana\tAT1.TAIR10\tAT1.1.TAIR10\tmissing_utr_5p\t0\t1\n"
                 "Athaliana\tAT1.TAIR10\tAT1.1.TAIR10\tmissing_stop_codon\t0\t1\n"
                 "Athaliana\tAT2.TAIR10\t\tswissprot_caution_frameshift\t0\t0\n"
                 "Gmax\tG1\t\tswissprot_caution_frameshift\t0\t0\n")
    hard = panels71.read_hard_flags(f)
    assert hard == {"AT1.TAIR10": {"AT1.1.TAIR10": {"missing_stop_codon"}}, "AT2.TAIR10": {"*": {"swissprot_caution_frameshift"}}}
    lst = tmp_path / "list.tsv"
    lst.write_text("species_id\tgene_id\tgene_name\tflag\nAthaliana\tAT3.TAIR10\tAT3\tx\nGmax\tG9\tG9\tx\n")
    assert panels71.read_gene_list(lst) == {"AT3.TAIR10"}
