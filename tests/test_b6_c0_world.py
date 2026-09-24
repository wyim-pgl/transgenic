"""Synthetic world for the B6 C0 tests: a real tile corpus (build_b5, tile6144-v3), planted splice motifs,
frozen + audit SAMs, read metadata, DATASET_ROLES rows, tool PROVENANCE, miniprot GFF and route-B HSPs.

Chr1 (250 kb) is one train block and Chr2 (60 kb) one test block under corpus seed 121.  Genes (1-based GFF):
    gA  +  train     10001-10200 / 10301-10500 / 10601-10800
    gH  +  STRICT held-out (test OG)   40001-40200 / 40301-40500
    gS  +  train, overlaps gH, shares gH's donor   40001-40200 / 40401-40600
    gT  -  test OG   70001-70300 / 70401-70600
    gV  +  valid OG  100001-100300 / 100401-100600
    gV2 +  train, overlaps gV, shares gV's donor   100001-100300 / 100501-100700
    gF  +  train, A22 hard flag (missing_start)    130001-130300 / 130401-130600
    gW  +  train, A30 flag only (swissprot_caution) 150001-150300 / 150401-150600
    gB  +  train     170001-170200 / 170301-170500
    gC  +  train, donor 'GTGT' (ambiguous ONT correction) 185001-185200 / 185301-185500
    gZ  +  train, on Chr2 (test block)             20001-20200 / 20301-20500
    gL  +  train, 35 kb: crosses every 30,720-nt tile edge   200001-200300 / 234701-235000
    gE  +  train, nested in gL's intron (keeps tiles holding a gL edge)  210001-210100 / 210201-210300
"""
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "revision" / "scripts", ROOT / "revision" / "scripts" / "evidence"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CORPUS_SEED = 121
SPECIES = "Athaliana"
MM_VERSION = "2.28-r1209"
GENES = {
    "gA": ("Chr1", "+", [(10001, 10200), (10301, 10500), (10601, 10800)], "train", False),
    "gH": ("Chr1", "+", [(40001, 40200), (40301, 40500)], "test", True),
    "gS": ("Chr1", "+", [(40001, 40200), (40401, 40600)], "train", False),
    "gT": ("Chr1", "-", [(70001, 70300), (70401, 70600)], "test", False),
    "gV": ("Chr1", "+", [(100001, 100300), (100401, 100600)], "valid", False),
    "gV2": ("Chr1", "+", [(100001, 100300), (100501, 100700)], "train", False),
    "gF": ("Chr1", "+", [(130001, 130300), (130401, 130600)], "train", False),
    "gW": ("Chr1", "+", [(150001, 150300), (150401, 150600)], "train", False),
    "gB": ("Chr1", "+", [(170001, 170200), (170301, 170500)], "train", False),
    "gC": ("Chr1", "+", [(185001, 185200), (185301, 185500)], "train", False),
    "gZ": ("Chr2", "+", [(20001, 20200), (20301, 20500)], "train", False),
    "gL": ("Chr1", "+", [(200001, 200300), (234701, 235000)], "train", False),
    "gE": ("Chr1", "+", [(210001, 210100), (210201, 210300)], "train", False),
}


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _plant(seq, s1, e1, strand, donor=None):
    c6 = "CCCCCC"
    motif_a, motif_b = ("GT", "AG") if strand == "+" else ("CT", "AC")
    seq[s1 - 7:s1 - 1] = c6
    seq[s1 - 1:s1 + 1] = motif_a
    seq[s1 + 1:s1 + 7] = c6
    seq[e1 - 8:e1 - 2] = c6
    seq[e1 - 2:e1] = motif_b
    seq[e1:e1 + 6] = c6
    if donor:
        seq[s1 - 1:s1 - 1 + len(donor)] = donor


def build_genome(tmp):
    rng = random.Random(5)
    chroms = {"Chr1": list("".join(rng.choice("ACGT") for _ in range(250000))),
              "Chr2": list("".join(rng.choice("ACGT") for _ in range(60000)))}
    for gid, (chrom, strand, exons, _, _) in GENES.items():
        for (_, e_prev), (s_next, _) in zip(exons, exons[1:]):
            _plant(chroms[chrom], e_prev + 1, s_next - 1, strand, donor="GTGT" if gid == "gC" else None)
    chroms["Chr1"][175097:175107] = list("CCCCCCCCCC")   # EST n1 intron 175101-175200: noncanonical donor
    fasta = tmp / "genome.fa"
    fasta.write_text("".join(f">{k}\n{''.join(v)}\n" for k, v in chroms.items()))
    return fasta, {k: "".join(v) for k, v in chroms.items()}


def build_corpus(tmp, *, with_a30=True):
    from transgenic.evidence import leakage as LK

    b5 = LK.load_build_b5()
    fasta, genome = build_genome(tmp)
    lines = ["##gff-version 3"]
    for gid, (chrom, strand, exons, _, _) in GENES.items():
        gs, ge = exons[0][0], exons[-1][1]
        lines.append(f"{chrom}\tt\tgene\t{gs}\t{ge}\t.\t{strand}\t.\tID={gid};Name={gid}")
        lines.append(f"{chrom}\tt\tmRNA\t{gs}\t{ge}\t.\t{strand}\t.\tID={gid}.1;Parent={gid}")
        for k, (s, e) in enumerate(exons):
            lines.append(f"{chrom}\tt\tCDS\t{s}\t{e}\t.\t{strand}\t0\tID={gid}.1.c{k};Parent={gid}.1")
    gff = tmp / "genes.gff3"
    gff.write_text("\n".join(lines) + "\n")
    split = tmp / "split.tsv"
    split.write_text("species_id\tgene_id\torthogroup_id\tsplit\tstrict_holdout\tseed\tsource_version\n" + "".join(
        f"{SPECIES}\t{gid}\tOG_{gid}\t{sp}\t{'true' if strict else 'false'}\t123\tv1\n"
        for gid, (_, _, _, sp, strict) in GENES.items()))
    a22 = tmp / "a22_flags.tsv"
    a22.write_text("species_id\tgene_id\ttranscript_id\tflag\tstart\tend\n" f"{SPECIES}\tgF\t\tmissing_start\t\t\n")
    a30 = tmp / "a30_flags.tsv"
    a30.write_text("species_id\tgene_id\ttranscript_id\tflag\tstart\tend\n"
                   f"{SPECIES}\tgW\t\tswissprot_caution_frameshift\t\t\n")
    manifest = tmp / "species.tsv"
    manifest.write_text("species_id\tspecies\ttable_s1_version\tfasta\tfasta_md5\tgff\tgff_md5\tnote\n"
                        f"{SPECIES}\tA\tT\t{fasta}\t\t{gff}\t\t\n")
    corpus = tmp / "corpus.db"
    b5.build_b5_database(str(corpus), str(manifest), str(split), rc="none", seed=CORPUS_SEED, verify_md5=False,
                         window_policy="tile6144-v3", qc_flags_path=[str(a22), str(a30)] if with_a30 else str(a22))
    return {"fasta": fasta, "genome": genome, "gff": gff, "split": split, "a22": a22, "a30": a30, "corpus": corpus}


# ------------------------------------------------------------------------------------------ SAM helpers
def sam(name, chrom, exons, *, flag=0, mapq=60, ts="+", clip=0, extra=(), as_score=100):
    if flag & 4:
        return f"{name}\t{flag}\t*\t0\t0\t*\t*\t0\t0\t*\t*"
    cig, cs = [], []
    if clip:
        cig.append(f"{clip}S")
    for i, (s, e) in enumerate(exons):
        if i:
            gap = s - exons[i - 1][1] - 1
            cig.append(f"{gap}N")
            cs.append(f"~gt{gap}ag")
        cig.append(f"{e - s + 1}M")
        cs.append(f":{e - s + 1}")
    tags = [f"cs:Z:{''.join(cs)}", f"AS:i:{as_score}"] + ([f"ts:A:{ts}"] if ts else []) + list(extra)
    return "\t".join([name, str(flag), chrom, str(exons[0][0]), str(mapq), "".join(cig), "*", "0", "0", "*", "*"] + tags)


def header(cl):
    return (f"@SQ\tSN:Chr1\tLN:250000\n@SQ\tSN:Chr2\tLN:60000\n"
            f"@PG\tID:minimap2\tPN:minimap2\tVN:{MM_VERSION}\tCL:{cl}\n")


FROZEN_CL = "minimap2 -t 32 -ax splice:hq -G 200000 --secondary=no --cs=long ref.fa est.fa"
AUDIT_CL = "minimap2 -t 32 -ax splice:hq -G 200000 --secondary=yes -N 50 -p 0.5 --cs=long ref.fa est.fa"

GA3 = [(10001, 10200), (10301, 10500), (10601, 10800)]
EST_READS = [
    # read, accession, library, clone, primary line kwargs
    ("e1", "EA1.1", "L1", "C1", dict(chrom="Chr1", exons=GA3)),
    ("e2", "EA1.2", "L1", "C1", dict(chrom="Chr1", exons=GA3)),
    ("e3", "EB2_part1", "L1", "", dict(chrom="Chr1", exons=[(10001, 10200), (10301, 10400)])),
    ("e4", "EB2_part2", "L1", "", dict(chrom="Chr1", exons=[(10601, 10800)])),
    ("e5", "EE5", "L1", "", dict(chrom="Chr1", exons=[(10301, 10500), (10601, 10800)])),
    ("e6", "EE6", "L1", "", dict(chrom="Chr1", exons=[(10301, 10500), (10601, 10800)])),
    ("e7", "EE7", "L2", "", dict(chrom="Chr1", exons=[(10301, 10500), (10601, 10800)])),
    ("e8", "EC8", "L1", "cloneX 5'", dict(chrom="Chr1", exons=[(10001, 10200), (10301, 10500)])),
    ("e9", "EC9", "L1", "cloneX 3'", dict(chrom="Chr1", exons=[(10301, 10500), (10601, 10800)])),
    ("h1", "EH1", "L1", "", dict(chrom="Chr1", exons=[(40001, 40200), (40301, 40500)])),
    ("t1", "ET1", "L1", "", dict(chrom="Chr1", exons=[(70001, 70300), (70401, 70600)], flag=16, ts="+")),
    ("v1", "EV1", "L1", "", dict(chrom="Chr1", exons=[(100001, 100300), (100501, 100700)])),
    ("f1", "EF1", "L1", "", dict(chrom="Chr1", exons=[(130001, 130300), (130401, 130600)])),
    ("w1", "EW1", "L1", "", dict(chrom="Chr1", exons=[(150001, 150300), (150401, 150600)])),
    ("z1", "EZ1", "L1", "", dict(chrom="Chr2", exons=[(20001, 20200), (20301, 20500)])),
    ("s1", "ES1", "L1", "", dict(chrom="Chr1", exons=[(10001, 10200)], extra=("SA:Z:Chr1,40001,+,200M,60,0;",))),
    ("m1", "EM1", "L1", "", dict(chrom="Chr1", exons=[(170001, 170200), (170301, 170500)])),
    ("n1", "EN1", "L1", "", dict(chrom="Chr1", exons=[(175001, 175100), (175201, 175300)])),
    ("g1", "EG1", "L1", "", dict(chrom="Chr1", exons=[(210001, 210100), (210201, 210300)])),
    ("q1", "EQ1", "L1", "", dict(chrom="Chr1", exons=[(10001, 10200), (10301, 10500)], ts=None)),
    ("u1", "EU1", "L1", "", dict(chrom="*", exons=[], flag=4)),
]
EST_EXTRA_FROZEN = {"s1": [sam("s1", "Chr1", [(40001, 40200)], flag=2048, extra=("SA:Z:Chr1,10001,+,200M,60,0;",))]}
EST_EXTRA_AUDIT = dict(EST_EXTRA_FROZEN)
EST_EXTRA_AUDIT["m1"] = [sam("m1", "Chr1", [(180001, 180200), (180301, 180500)], flag=256, mapq=0)]


def _write_sam(path, cl, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header(cl) + "\n".join(lines) + "\n")
    return path


def _write_done(directory, path, extra=()):
    (directory / "DONE").write_text(f"sam_md5={md5(path)}\n" + "".join(f"{k}={v}\n" for k, v in extra))


def _write_meta(directory, default, rows, extra_cols=()):
    cols = ["read", "accession", "library", "bioproject", "genotype_stratum", "clone", "ingestion_qc_pass",
            "mapping_ambiguous", "post_trim_length", "protocol", "umi_status", "source_molecule", *extra_cols]
    lines = ["\t".join(cols), "\t".join(["*"] + [str(default.get(c, "")) for c in cols[1:]])]
    for r in rows:
        lines.append("\t".join([r["read"]] + [str(r.get(c, "")) for c in cols[1:]]))
    p = directory / "c0_metadata.tsv"
    p.write_text("\n".join(lines) + "\n")
    (directory / "c0_metadata.tsv.sha256").write_text(f"{sha(p)}  c0_metadata.tsv\n")


def est_lines(extra):
    out = []
    for read, _, _, _, kw in EST_READS:
        kw = dict(kw)
        out.append(sam(read, kw.pop("chrom"), kw.pop("exons"), **kw))
        out += extra.get(read, [])
    return out


def write_est(tmp, *, reads=None, audit_extra=None, frozen_cl=FROZEN_CL, audit_cl=AUDIT_CL, audit=True, done_md5_ok=True):
    align, audit_root = tmp / "align", tmp / "audit"
    d = align / SPECIES
    frozen = _write_sam(d / f"{SPECIES}.est.sam", frozen_cl, est_lines(EST_EXTRA_FROZEN))
    _write_done(d, frozen, [("arm", "primary")])
    if not done_md5_ok:
        (d / "DONE").write_text("sam_md5=" + "0" * 32 + "\narm=primary\n")
    rows = []
    for read, acc, lib, clone, kw in EST_READS:
        qlen = sum(e - s + 1 for s, e in kw["exons"]) or 150
        rows.append(dict(read=read, accession=acc, library=lib, clone=clone, post_trim_length=qlen))
    _write_meta(d, dict(bioproject="PRJ1", genotype_stratum="reference", ingestion_qc_pass="true",
                        mapping_ambiguous="false"), rows)
    if audit:   # b6_audit_align.sbatch layout: $OUTROOT/$SP/est/est_primary/est_primary.audit.bam + DONE source/run
        a = _write_sam(est_audit_path(audit_root), audit_cl,
                       est_lines(audit_extra if audit_extra is not None else EST_EXTRA_AUDIT))
        _write_done(a.parent, a, [("source", "est"), ("run", "est_primary"), ("uf", "none")])
    return align, audit_root


def est_audit_path(audit_root):
    return audit_root / SPECIES / "est" / "est_primary" / "est_primary.audit.sam"


ONT_DATASET = "training/ont/Athaliana/proj1"
ONT_READS = [
    ("o1", [(170001, 170200), (170301, 170500)]),
    ("o2", [(170001, 170200), (170301, 170495)]),
    ("o3", [(170021, 170200), (170301, 170500)]),
    ("o4", [(170001, 170202), (170303, 170500)]),
    ("o5", [(185001, 185200), (185301, 185500)]),
]


def write_ont(tmp, run="SRR1"):
    align, audit_root = tmp / "align", tmp / "audit"
    lines = [sam(r, "Chr1", ex) for r, ex in ONT_READS]
    cl_f = "minimap2 -t 32 -ax splice -k14 -G 200000 --secondary=no --cs=long ref.fa reads.fq.gz"
    cl_a = "minimap2 -t 32 -ax splice -k14 -G 200000 --secondary=yes -N 50 -p 0.5 --cs=long ref.fa reads.fq.gz"
    d = align / SPECIES / "proj1" / run
    f = _write_sam(d / f"{run}.sam", cl_f, lines)
    _write_done(d, f, [("uf", "none")])
    _write_meta(d, dict(library="SRX1000101", bioproject="PRJ2", genotype_stratum="reference", ingestion_qc_pass="true",
                        mapping_ambiguous="false", protocol="cDNA", umi_status="absent"), [dict(read=r) for r, _ in ONT_READS])
    ad = audit_root / SPECIES / "proj1" / run          # b6_audit_align.sbatch shape: <run>.audit.*, no audit_n key
    a = _write_sam(ad / f"{run}.audit.sam", cl_a, lines)
    _write_done(ad, a, [("source", "ont"), ("run", run)])
    return align, audit_root


ONT_UMI_RUN = "SRR3"
ONT_UMI_ROLE_ROW = (f"{ONT_DATASET}\t{ONT_UMI_RUN}\t{SPECIES}\treference\tPromethION\tRNA-Seq\t1\t2\t\t\t\t"
                    "c2_training_eligible\tt\t")
ONT_UMI_READS = [
    # read, exons, MAPQ; one library, one sample/UMI/locus -> one A16 UMI unit; no ts tag and uf=none -> strand unknown
    ("k1", [(12001, 12300)], 60),          # accepted, clean intergenic span
    ("k2", [(40001, 40200)], 10),          # same UMI unit, MAPQ 10 on the held-out gene gH: rejected (mapq)
]


def write_ont_umi(tmp, run=ONT_UMI_RUN):
    """An ONT cDNA run WITH UMIs (A16: UMI = the unit) in write_ont's layout, no strand evidence on any read."""
    align, audit_root = tmp / "align", tmp / "audit"
    lines = [sam(r, "Chr1", ex, mapq=mq, ts=None) for r, ex, mq in ONT_UMI_READS]
    cl_f = "minimap2 -t 32 -ax splice -k14 -G 200000 --secondary=no --cs=long ref.fa reads.fq.gz"
    cl_a = "minimap2 -t 32 -ax splice -k14 -G 200000 --secondary=yes -N 50 -p 0.5 --cs=long ref.fa reads.fq.gz"
    d = align / SPECIES / "proj1" / run
    f = _write_sam(d / f"{run}.sam", cl_f, lines)
    _write_done(d, f, [("uf", "none")])
    _write_meta(d, dict(library="SRX1000102", bioproject="PRJ2", genotype_stratum="reference", ingestion_qc_pass="true",
                        mapping_ambiguous="false", protocol="cDNA", umi_status="present", sample="S1", umi="ACGTACGT",
                        locus="LOC1"), [dict(read=r) for r, _, _ in ONT_UMI_READS], extra_cols=("sample", "umi", "locus"))
    ad = audit_root / SPECIES / "proj1" / run
    a = _write_sam(ad / f"{run}.audit.sam", cl_a, lines)
    _write_done(ad, a, [("source", "ont"), ("run", run)])
    return align, audit_root


ONT_MANY_RUN = "SRR4"


def ont_many_role_row(n):
    return (f"{ONT_DATASET}\t{ONT_MANY_RUN}\t{SPECIES}\treference\tPromethION\tRNA-Seq\t1\t{n}\t\t\t\t"
            "c2_training_eligible\tt\t")


def write_ont_many(tmp, n, run=ONT_MANY_RUN):
    """``n`` mono-exonic ONT cDNA reads (no UMI) on the clean span Chr1 12001-38000, starts 20 nt apart, so every read
    is its own A16 PCR-equivalence unit (ends differ by > 10 nt) and its own audited molecule.  Used to give a species
    enough molecules for the species-level exhaustiveness rule (A46 rule 4(b)) to be tested end to end."""
    assert 20 * n <= 26000 - 200, n
    align, audit_root = tmp / "align", tmp / "audit"
    reads = [(f"r{i:05d}", [(12001 + 20 * i, 12200 + 20 * i)]) for i in range(n)]
    lines = [sam(r, "Chr1", ex) for r, ex in reads]
    cl_f = "minimap2 -t 32 -ax splice -k14 -G 200000 --secondary=no --cs=long ref.fa reads.fq.gz"
    cl_a = "minimap2 -t 32 -ax splice -k14 -G 200000 --secondary=yes -N 50 -p 0.5 --cs=long ref.fa reads.fq.gz"
    d = align / SPECIES / "proj1" / run
    f = _write_sam(d / f"{run}.sam", cl_f, lines)
    _write_done(d, f, [("uf", "none")])
    _write_meta(d, dict(library="SRX1000103", bioproject="PRJ2", genotype_stratum="reference", ingestion_qc_pass="true",
                        mapping_ambiguous="false", protocol="cDNA", umi_status="absent"), [dict(read=r) for r, _ in reads])
    ad = audit_root / SPECIES / "proj1" / run
    a = _write_sam(ad / f"{run}.audit.sam", cl_a, lines)
    _write_done(ad, a, [("source", "ont"), ("run", run)])
    return align, audit_root


PACBIO_DATASET = "training/pacbio/Athaliana/proj2"
PACBIO_RUN = "SRR2"
PACBIO_READS = [
    # read, A16 source molecule (movie/zmw), exons, MAPQ
    ("p1", "m64011/42", [(170001, 170200), (170301, 170500)], 60),   # accepted, clean locus (o1's)
    ("p2", "m64011/42", [(40001, 40200), (40301, 40500)], 10),       # same ZMW molecule, MAPQ 10 at the held-out gene: rejected
    ("p3", "m64011/43", [(175001, 175100), (175201, 175300)], 60),   # another molecule, accepted, clean
]
PACBIO_LIBRARY = "SRX1000201"          # the biological library (A11): an experiment accession, never the run


def pacbio_role_row(run, n_reads=3):
    return (f"{PACBIO_DATASET}\t{run}\t{SPECIES}\treference\tSequel II\tIso-Seq\t1\t{n_reads}\t\t\t\t"
            "c2_training_eligible\tt\t")


PACBIO_ROLE_ROW = pacbio_role_row(PACBIO_RUN)
# A second sequencing run of the SAME preparation (fix round 6, finding 1): p6 is accepted at a clean locus but shares
# ZMW m64011/42 with p1 of PACBIO_RUN, and its audit alignment carries a MAPQ-0 secondary on the held-out gene.
PACBIO_RUN2 = "SRR5"
PACBIO_READS2 = [("p6", "m64011/42", [(10001, 10200), (10301, 10500)], 60)]
PACBIO_AUDIT_EXTRA2 = {"p6": [sam("p6", "Chr1", [(40001, 40150), (40301, 40400)], flag=256, mapq=0, as_score=40)]}   # not equal-best


def write_pacbio(tmp, run=PACBIO_RUN, *, library=PACBIO_LIBRARY, reads=None, audit_extra=None):
    """A PacBio (source_molecule = A16 unit) run in the b6_audit_align.sbatch layout, like write_ont.  ``library`` is
    the metadata '*' row library (b6_c0_metadata --library); ``audit_extra`` adds audit-only lines per read."""
    align, audit_root = tmp / "align", tmp / "audit"
    reads = PACBIO_READS if reads is None else reads
    lines = [sam(r, "Chr1", ex, mapq=mq) for r, _, ex, mq in reads]
    audit_lines = []
    for (r, _, ex, mq) in reads:
        audit_lines.append(sam(r, "Chr1", ex, mapq=mq))
        audit_lines += (audit_extra or {}).get(r, [])
    cl_f = "minimap2 -t 32 -ax splice:hq -G 200000 --secondary=no --cs=long ref.fa reads.fq.gz"
    cl_a = "minimap2 -t 32 -ax splice:hq -G 200000 --secondary=yes -N 50 -p 0.5 --cs=long ref.fa reads.fq.gz"
    d = align / SPECIES / "proj2" / run
    f = _write_sam(d / f"{run}.sam", cl_f, lines)
    _write_done(d, f, [("uf", "none")])
    _write_meta(d, dict(library=library, bioproject="PRJ3", genotype_stratum="reference", ingestion_qc_pass="true",
                        mapping_ambiguous="false"), [dict(read=r, source_molecule=m) for r, m, _, _ in reads])
    ad = audit_root / SPECIES / "proj2" / run
    a = _write_sam(ad / f"{run}.audit.sam", cl_a, audit_lines)
    _write_done(ad, a, [("source", "pacbio"), ("run", run)])
    return align, audit_root


def write_roles(tmp, *, est_role="c2_training_eligible", extra_rows=(), drop_est=False):
    head = ("dataset\trun\tspecies\tgenotype_stratum\tinstrument\tdata_type\texpected_files\texpected_reads\t"
            "source_checksum\tsource_checksum_authority\tlocal_fa_md5\trole\tbasis\tnote\n")
    est = tmp / "roles_est.tsv"
    est.write_text(head + ("" if drop_est else f"est/{SPECIES}\test.fa.gz\t{SPECIES}\treference\tGenBank_EST\tEST\t1\t20\t\tnone\t\t{est_role}\tt\t\n"))
    lr = tmp / "roles_lr.tsv"
    rows = [f"{ONT_DATASET}\tSRR1\t{SPECIES}\treference\tPromethION\tRNA-Seq\t1\t5\t\t\t\tc2_training_eligible\tt\t",
            f"protein/{SPECIES}\tminiprot\t{SPECIES}\treference\tminiprot\tprotein\t1\t0\t\t\t\tc2_training_eligible\tt\t"]
    lr.write_text(head + "\n".join(rows + list(extra_rows)) + "\n")
    return lr, est


LINEAGE = {3702: [3702, 3701, 3700, 3398, 33090, 2759, 1], 4577: [4577, 4575, 4479, 4447, 3398, 33090, 2759, 1],
           39947: [39947, 4530, 4527, 4479, 4447, 3398, 33090, 2759, 1], 999001: [999001, 3702, 3701, 3700, 3398, 33090, 2759, 1]}   # an A. thaliana subtaxon


def write_lineage(tmp):
    p = tmp / "lineage.tsv"
    p.write_text("taxid\tlineage\n" + "".join(f"{t}\t{';'.join(map(str, l))}\n" for t, l in LINEAGE.items()))
    return p


def write_provenance(tmp, version=MM_VERSION):
    p = tmp / "PROVENANCE.cpu_tools_v1.json"
    p.write_text(json.dumps({"provenance_id": "t", "tools": {"minimap2_env": {"path": "/x", "sha256": "0" * 64, "version": version}}}))
    Path(str(p) + ".sha256").write_text(f"{sha(p)}  {p.name}\n")
    return p


def cli_args(world, tmp, *, source="est", out_name=None, align=None, audit=None, roles=None, extra=()):
    lr, est = roles or write_roles(tmp)
    prov = write_provenance(tmp)
    out = tmp / (out_name or f"{SPECIES}.{source}.c0.db")
    return ["--roles", str(lr), "--est-roles", str(est), "--species", SPECIES, "--source", source,
            "--align-root", str(align or tmp / "align"), "--audit-root", str(audit or tmp / "audit"),
            "--corpus", str(world["corpus"]), "--splits", str(world["split"]), "--flags", str(world["a22"]),
            "--build-flags", str(world["a22"]), str(world["a30"]), "--provenance", str(prov), "--seed", str(CORPUS_SEED),
            "--out", str(out), "--qc", str(out) + ".qc.json", *extra]


def test_world_corpus_has_train_and_test_blocks(tmp_path):
    import duckdb

    world = build_corpus(tmp_path)
    con = duckdb.connect(str(world["corpus"]), read_only=True)
    blocks = dict((c, s) for c, s in con.execute("SELECT chromosome, split FROM tile_blocks").fetchall())
    con.close()
    assert blocks == {"Chr1": "train", "Chr2": "test"}
