"""Synthetic world variant for the WP2 rule 5 repair tests (plan v4.1 §2 fixture); WP1's test_b6_c0_world is unedited.

It is test_b6_c0_world's gene set plus the genes below, built by build_b5 (tile6144-v3, corpus seed 121) with
``rc="all"`` so every tile with a labelled gene has an RC twin.  Chr1 tier-30,720 tile edges fall at
2860 + k * 30720 (the first draw of the build RNG; genes added only to non-empty tiles leave it unchanged).
Coordinates are 1-based GFF.

    gHE  +  STRICT held-out (test OG), crosses the 64,300 edge      64001-64200 / 64401-64600
    gM   +  train, overlaps gHE (component member, labelled in the edge tile by A40)  64401-64500 / 64801-65000
    gN   +  train, 20 nt past the component, reached only by the flank              65021-65200 / 65301-65400
    gK   +  train, far inside the edge tile (keeps its labelled gene set non-empty)  80001-80200 / 80301-80500
    gV3  +  valid OG, crosses the 95,020 edge (scope train_tiles)                   94801-95000 / 95201-95400
    gW2  +  train, A30 flag only, crosses the 156,460 edge (masked, never a C0 exclusion)  156201-156400 / 156601-156800
    gQ   +  test OG on Chr2 (test block): no train/valid tile ever masks it, so its flank is fallback_150  40001-40200 / 40401-40600

gHE and gM are the first two masked genes of the kept tier-61,440 tile Chr1:41006-102446, whose seeded flank draws
are 75 and 61, so the component's historical max flank is 75: a molecule at +40 nt past the component is masked and
excluded, one at +100 nt stays admitted and unmasked (a fresh draw of 140 would have masked it).

``refuse=True`` builds a second world with ``gBIG`` (test OG, 158001-190000) crossing the 187,180 edge: the tile
Chr1:156460-187180 is then over masked fraction 0.60 and loses its last labelled genes (gB, gC) although it has an
RC twin, so ``--apply`` must refuse.
"""
import json
from pathlib import Path

import test_b6_c0_world as W

SPECIES = W.SPECIES
SEED = W.CORPUS_SEED
EXTRA_GENES = {
    "gHE": ("Chr1", "+", [(64001, 64200), (64401, 64600)], "test", True),
    "gM": ("Chr1", "+", [(64401, 64500), (64801, 65000)], "train", False),
    "gN": ("Chr1", "+", [(65021, 65200), (65301, 65400)], "train", False),
    "gK": ("Chr1", "+", [(80001, 80200), (80301, 80500)], "train", False),
    "gV3": ("Chr1", "+", [(94801, 95000), (95201, 95400)], "valid", False),
    "gW2": ("Chr1", "+", [(156201, 156400), (156601, 156800)], "train", False),
    "gQ": ("Chr2", "+", [(40001, 40200), (40401, 40600)], "test", False),
}
REFUSE_GENES = {"gBIG": ("Chr1", "+", [(158001, 170000), (180001, 190000)], "test", False)}
A30_GENES = ("gW", "gW2")
HE_COMPONENT = (64000, 65000)          # 0-based half-open span of {gHE, gM}
HE_FLANK = 75
EDGE_TILE = (64300, 95020)             # the tier-30,720 tile whose left edge gHE crosses

# ONT reads (mono-exonic, 1-based) around the repair spans; every read is its own A16 unit.
ONT_READS = [
    ("mA", [(65041, 65090)]),          # +40 nt past the gHE component: inside the historical flank -> excluded, masked
    ("mB", [(65101, 65150)]),          # +100 nt: outside -> admitted, unmasked
    ("mV", [(95411, 95460)]),          # +10 nt past gV3: train_tiles_excluded, masked in the train edge tile
    ("mW", [(156811, 156860)]),        # +10 nt past gW2: A30-only mask, admitted (declared difference)
    ("mC", [(12001, 12200)]),          # clean
]


def genes(refuse=False):
    g = dict(W.GENES)
    g.update(EXTRA_GENES)
    if refuse:
        g.update(REFUSE_GENES)
    return g


def build_genome(tmp, gene_set):
    import random

    rng = random.Random(5)
    chroms = {"Chr1": list("".join(rng.choice("ACGT") for _ in range(250000))),
              "Chr2": list("".join(rng.choice("ACGT") for _ in range(60000)))}
    for gid, (chrom, strand, exons, _, _) in gene_set.items():
        for (_, e_prev), (s_next, _) in zip(exons, exons[1:]):
            W._plant(chroms[chrom], e_prev + 1, s_next - 1, strand, donor="GTGT" if gid == "gC" else None)
    chroms["Chr1"][175097:175107] = list("CCCCCCCCCC")
    fasta = tmp / "genome.fa"
    fasta.write_text("".join(f">{k}\n{''.join(v)}\n" for k, v in chroms.items()))
    return fasta, {k: "".join(v) for k, v in chroms.items()}


def build_world(tmp, *, refuse=False, rc="all"):
    from transgenic.evidence import leakage as LK

    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    b5 = LK.load_build_b5()
    gene_set = genes(refuse)
    fasta, genome = build_genome(tmp, gene_set)
    lines = ["##gff-version 3"]
    for gid, (chrom, strand, exons, _, _) in gene_set.items():
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
        for gid, (_, _, _, sp, strict) in gene_set.items()))
    a22 = tmp / "a22_flags.tsv"
    a22.write_text("species_id\tgene_id\ttranscript_id\tflag\tstart\tend\n" f"{SPECIES}\tgF\t\tmissing_start\t\t\n")
    a30 = tmp / "a30_flags.tsv"
    a30.write_text("species_id\tgene_id\ttranscript_id\tflag\tstart\tend\n" + "".join(
        f"{SPECIES}\t{g}\t\tswissprot_caution_frameshift\t\t\n" for g in A30_GENES))
    manifest = tmp / "species.tsv"
    manifest.write_text("species_id\tspecies\ttable_s1_version\tfasta\tfasta_md5\tgff\tgff_md5\tnote\n"
                        f"{SPECIES}\tA\tT\t{fasta}\t\t{gff}\t\t\n")
    corpus = tmp / "corpus.db"
    b5.build_b5_database(str(corpus), str(manifest), str(split), rc=rc, seed=SEED, verify_md5=False,
                         window_policy="tile6144-v3", qc_flags_path=[str(a22), str(a30)])
    return {"tmp": tmp, "fasta": fasta, "genome": genome, "gff": gff, "split": split, "a22": a22, "a30": a30,
            "corpus": corpus, "genes": gene_set}


def leakage_model(world):
    from transgenic.evidence import leakage as LK

    return LK.build_leakage_model(corpus_db=world["corpus"], species=SPECIES, splits_path=world["split"],
                                  flags_path=world["a22"], build_flags=[world["a22"], world["a30"]], seed=SEED)


def write_ont_reads(tmp, reads=ONT_READS, run="SRR1"):
    """ONT cDNA run in test_b6_c0_world.write_ont's layout with the reads above (role row SRR1 of write_roles)."""
    tmp = Path(tmp)
    align, audit_root = tmp / "align", tmp / "audit"
    lines = [W.sam(r, "Chr1", ex) for r, ex in reads]
    cl_f = "minimap2 -t 32 -ax splice -k14 -G 200000 --secondary=no --cs=long ref.fa reads.fq.gz"
    cl_a = "minimap2 -t 32 -ax splice -k14 -G 200000 --secondary=yes -N 50 -p 0.5 --cs=long ref.fa reads.fq.gz"
    d = align / SPECIES / "proj1" / run
    f = W._write_sam(d / f"{run}.sam", cl_f, lines)
    W._write_done(d, f, [("uf", "none")])
    W._write_meta(d, dict(library="SRX1000101", bioproject="PRJ2", genotype_stratum="reference", ingestion_qc_pass="true",
                          mapping_ambiguous="false", protocol="cDNA", umi_status="absent"), [dict(read=r) for r, _ in reads])
    ad = audit_root / SPECIES / "proj1" / run
    a = W._write_sam(ad / f"{run}.audit.sam", cl_a, lines)
    W._write_done(ad, a, [("source", "ont"), ("run", run)])
    return align, audit_root


def build_ont_c0(world, tmp):
    """The ONT C0 of the variant world (b6_build_c0 on the A40-equivalent corpus, as the real C0 is built)."""
    import b6_build_c0 as B

    tmp = Path(tmp)
    write_ont_reads(tmp)
    args = W.cli_args(world, tmp, source="ont")
    assert B.main(args) == 0, "variant-world ONT C0 build failed"
    return Path(args[args.index("--out") + 1]), json.loads(Path(args[args.index("--qc") + 1]).read_text())


def test_variant_world_places_the_held_out_component_as_designed(tmp_path):
    world = build_world(tmp_path)
    model = leakage_model(world)
    he = [iv for iv in model.intervals if iv.cause == "heldout" and iv.tile_rn is None and iv.gene_id == "gHE"]
    assert [(iv.start0, iv.end0) for iv in he] == [(HE_COMPONENT[0] - HE_FLANK, HE_COMPONENT[1] + HE_FLANK)]
