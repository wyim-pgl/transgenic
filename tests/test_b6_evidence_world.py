"""Evidence world for WP2-prot / WP2b / WP2c tests: test_b6_c0_world's corpus with published EST, ONT, PacBio and
protein C0 DBs and their merged snapshot (b6_merge_evidence_db), all built by the production builders.

The protein (miniprot) GFF is written here.  Complete ORFs are designed on the world genome at build time
(``design_orf``: an ATG, a 61-nt first exon so the second CDS has GFF phase 2, a 100-nt intron and the first in-frame
stop codon >= 10 codons into exon 2), so start/stop codons and phases are real bases, not assumptions:

    4577_0:000011   +  complete ORF (stop = last three CDS bases)                   near Chr1:20000
    4577_0:000012   -  complete ORF                                                  near Chr1:26000
    39947_0:000013  +  partial query (qstart0 = 3)                                   near Chr1:31000
    4577_0:000014   +  complete ORF, plus a lower-scoring secondary at Chr1:240001   near Chr1:176500
    39947_0:000015  +  two equal-score placements: mapping_ambiguous, no accepted primary
    4577_0:000002   +  on the strict held-out gene gH: accepted primary, excluded_heldout in C0
    4577_0:000021 / 39947_0:000022  +  both on gA's first two CDS exons (two organisms, one intron boundary)

``evlabel_extras=True`` (WP2b) adds 39947_0:000016 (ORF 14's CDS with its start moved 3 nt: a conflicting CDS on
one read), 4577_0:000017 on gV2 (train_tiles_excluded: never a train-tile label or phase) and an ONT run SRR7 whose reads extend ORFs 11 / 12 / 14 by UTR bases (``UTR_EXT``) plus a read ``ri1``
whose single block spans gA's first intron (retained-intron-like).  ``variant=True`` builds on
test_b6_repair_world's corpus (rc="all": explicit RC tiles) instead of test_b6_c0_world's.
"""
import json
from pathlib import Path

import test_b6_c0_world as W
import test_b6_c0_build as TB
import b6_build_c0 as B
import b6_merge_evidence_db as M

SPECIES = W.SPECIES
UTR_EXT = {"4577_0:000011": (80, 120), "4577_0:000012": (80, 120), "4577_0:000014": (60, 60)}   # (5', 3') nt
EXTRA_ONT_RUN = "SRR7"
STOPS = {"TAA", "TAG", "TGA"}
_COMP = str.maketrans("ACGTN", "TGCAN")


def revcomp(s):
    return s.translate(_COMP)[::-1]


def design_orf(chrom_seq, strand, search_from, *, exon1_len=61, intron_len=100, convention="inside_last_block",
               min_exon2_codons=10):
    """1-based genomic CDS blocks [(s1, e1, phase), ...] (genomic order) and qlen of a designed complete ORF."""
    L = len(chrom_seq)
    T = chrom_seq if strand == "+" else revcomp(chrom_seq)
    s = search_from if strand == "+" else L - search_from
    while True:
        a = T.find("ATG", s)
        assert a >= 0, "no ATG left"
        s = a + 1
        e1, b = a + exon1_len, a + exon1_len + intron_len
        spliced = T[a:e1] + T[b:b + 3000]
        k = next(i for i in range(len(spliced) // 3) if spliced[3 * i:3 * i + 3] in STOPS)
        if 3 * k < exon1_len + 3 * min_exon2_codons:
            continue
        e2 = b + (3 * k + 3 - exon1_len if convention == "inside_last_block" else 3 * k - exon1_len)
        tx_blocks = [(a, e1, 0), (b, e2, (3 - exon1_len % 3) % 3)]
        if strand == "+":
            blocks = [(x + 1, y, ph) for x, y, ph in tx_blocks]
        else:
            blocks = sorted((L - y + 1, L - x, ph) for x, y, ph in tx_blocks)
        return blocks, k


def paf(q, qlen, qstart, qend, n_introns, strand="+"):
    cs = ":50" + "".join("~gt100ag:50" for _ in range(n_introns))
    return f"##PAF\t{q}\t{qlen}\t{qstart}\t{qend}\t{strand}\tChr1\t250000\t0\t1\t1\t1\t60\tfs:i:0\tcs:Z:{cs}"


def mrna(i, q, qlen, strand, score, blocks, chrom="Chr1"):
    lines = [f"{chrom}\tminiprot\tmRNA\t{blocks[0][0]}\t{blocks[-1][1]}\t{score}\t{strand}\t.\t"
             f"ID=MP{i};Identity=0.9;Target={q} 1 {qlen}"]
    lines += [f"{chrom}\tminiprot\tCDS\t{s}\t{e}\t{score}\t{strand}\t{ph}\tParent=MP{i}" for s, e, ph in blocks]
    return lines


def protein_records(genome, evlabel_extras=False):
    """(protein, qlen, qstart, qend, strand, score, blocks) of every miniprot record, in file order."""
    chr1 = genome["Chr1"]
    recs = []
    for q, strand, at in (("4577_0:000011", "+", 20000), ("4577_0:000012", "-", 26000), ("4577_0:000014", "+", 176500)):
        blocks, k = design_orf(chr1, strand, at)
        recs.append((q, k, 0, k, strand, 500, blocks))
        if q == "4577_0:000014":
            recs.append((q, k, 0, k, "+", 100, [(240001, 240061, 0), (240162, 240200, 2)]))
    blocks, k = design_orf(chr1, "+", 31000)
    recs.append(("39947_0:000013", k, 3, k, "+", 450, blocks))
    recs.append(("39947_0:000015", 60, 0, 60, "+", 300, [(22001, 22090, 0), (22191, 22280, 0)]))
    recs.append(("39947_0:000015", 60, 0, 60, "+", 300, [(24001, 24090, 0), (24191, 24280, 0)]))
    recs.append(("4577_0:000002", 133, 0, 133, "+", 400, [(40001, 40200, 0), (40301, 40500, 1)]))
    recs.append(("4577_0:000021", 133, 0, 133, "+", 480, [(10001, 10200, 0), (10301, 10500, 1)]))
    recs.append(("39947_0:000022", 133, 0, 133, "+", 470, [(10001, 10200, 0), (10301, 10500, 1)]))
    if evlabel_extras:
        (q, qlen, _, _, strand, _, blocks), = [r for r in recs if r[0] == "4577_0:000014" and r[5] == 500]
        shifted = [(blocks[0][0] + 3, blocks[0][1], 0)] + blocks[1:]
        recs.append(("39947_0:000016", qlen, 1, qlen, strand, 490, shifted))
        # on gV2 (overlaps the valid-orthogroup gene gV): train_track but train_tiles_excluded -- its CDS and phase
        # may label valid tiles only (review 1 H1 regression)
        recs.append(("4577_0:000017", 166, 0, 166, "+", 480, [(100001, 100300, 0), (100501, 100700, 0)]))
    return recs


def utr_reads(genome):
    """(read, 1-based exons, strand) of the SRR7 reads: each designed ORF's CDS blocks extended by UTR_EXT."""
    out = []
    for q, qlen, qs, qe, strand, score, blocks in protein_records(genome):
        if q not in UTR_EXT or score != 500:
            continue
        five, three = UTR_EXT[q]
        left, right = (five, three) if strand == "+" else (three, five)
        exons = [(blocks[0][0] - left, blocks[0][1]), (blocks[1][0], blocks[1][1] + right)]
        out.append(("u" + q[-2:], exons, strand))
    out.append(("ri1", [(10151, 10350)], "+"))
    return out


def write_utr_ont(tmp, genome, run=EXTRA_ONT_RUN):
    tmp = Path(tmp)
    align, audit_root = tmp / "align", tmp / "audit"
    reads = utr_reads(genome)
    lines = [W.sam(r, "Chr1", ex, flag=0 if st == "+" else 16, ts="+") for r, ex, st in reads]
    cl_f = "minimap2 -t 32 -ax splice -k14 -G 200000 --secondary=no --cs=long ref.fa reads.fq.gz"
    cl_a = "minimap2 -t 32 -ax splice -k14 -G 200000 --secondary=yes -N 50 -p 0.5 --cs=long ref.fa reads.fq.gz"
    d = align / SPECIES / "proj1" / run
    f = W._write_sam(d / f"{run}.sam", cl_f, lines)
    W._write_done(d, f, [("uf", "none")])
    W._write_meta(d, dict(library="SRX1000107", bioproject="PRJ2", genotype_stratum="reference", ingestion_qc_pass="true",
                          mapping_ambiguous="false", protocol="cDNA", umi_status="absent"), [dict(read=r) for r, _, _ in reads])
    ad = audit_root / SPECIES / "proj1" / run
    a = W._write_sam(ad / f"{run}.audit.sam", cl_a, lines)
    W._write_done(ad, a, [("source", "ont"), ("run", run)])
    return f"{W.ONT_DATASET}\t{run}\t{SPECIES}\treference\tPromethION\tRNA-Seq\t1\t{len(reads)}\t\t\t\tc2_training_eligible\tt\t"


def write_protein_gff(tmp, genome, evlabel_extras=False):
    d = Path(tmp) / "align" / SPECIES
    d.mkdir(parents=True, exist_ok=True)
    lines = ["##gff-version 3"]
    for i, (q, qlen, qs, qe, strand, score, blocks) in enumerate(protein_records(genome, evlabel_extras)):
        lines += [paf(q, qlen, qs, qe, len(blocks) - 1, strand)] + mrna(i, q, qlen, strand, score, blocks)
    gff = d / f"{SPECIES}.protein.gff"
    gff.write_text("\n".join(lines) + "\n")
    (d / "DONE").write_text(f"gff_md5          {W.md5(gff)}\n")
    return gff


def build_evidence_world(tmp, *, variant=False, evlabel_extras=False):
    """Corpus + EST/ONT/PacBio/protein C0 DBs + merged snapshot; returns a dict of paths."""
    tmp = Path(tmp)
    if variant:
        import test_b6_repair_world as RW

        world = RW.build_world(tmp)
    else:
        world = W.build_corpus(tmp)
    W.write_est(tmp)
    W.write_ont(tmp)
    W.write_pacbio(tmp)
    extra = [W.PACBIO_ROLE_ROW]
    if evlabel_extras:
        extra.append(write_utr_ont(tmp, world["genome"]))
    roles = W.write_roles(tmp, extra_rows=extra)
    dbs = {}
    for source in ("est", "ont", "pacbio"):
        args = W.cli_args(world, tmp, source=source, roles=roles)
        assert B.main(args) == 0, source
        dbs[source] = Path(args[args.index("--out") + 1])
    gff = write_protein_gff(tmp, world["genome"], evlabel_extras)
    est = (str(dbs["est"]), None)
    hsp = TB.write_hsp(tmp, TB._mols(est), state=TB.est_state(est))
    args = W.cli_args(world, tmp, source="protein", roles=roles, extra=TB._protein_extra(tmp, est, hsp))
    assert B.main(args) == 0, "protein"
    dbs["protein"] = Path(args[args.index("--out") + 1])
    merged = tmp / "evidence_c0.db"
    M.merge([str(dbs[s]) for s in ("est", "ont", "pacbio", "protein")], merged)
    return dict(world, dbs=dbs, merged=merged, protein_gff=gff, roles=roles, tmp=tmp)


def test_evidence_world_builds_every_source_and_the_merged_snapshot(tmp_path):
    ev = build_evidence_world(tmp_path)
    assert ev["merged"].is_file() and Path(str(ev["merged"]) + ".sha256").is_file()
    prov = json.loads(Path(str(ev["dbs"]["protein"]) + ".qc.json").read_text())
    assert prov["status"] == "PASS"
