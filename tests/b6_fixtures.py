"""Shared B6 WP3/WP4 fixtures (owner: Writer B; A and C request changes through B, plan §7).

* ``stub_batch`` / ``stub_batch_torch``: a batch that satisfies the §7.7 contract exactly, for model work that must
  not depend on the evidence pipeline (CP0/CP1).  numpy by default; the torch variant imports torch lazily.
* the synthetic evidence world used by the WP3 tests lives further down (``make_world``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "revision" / "scripts", ROOT / "revision" / "scripts" / "evidence"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from transgenic.evidence import b6_track_spec as TS  # noqa: E402


def stub_batch(*, lengths=(60, 48), label_lengths=(7, 4), training=True, chains=True, seed=0):
    """A §7.7 batch of ``len(lengths)`` samples, left-padded to max(lengths) (must be multiples of 6).

    Sample b gets deterministic non-zero tracks on its real bases and, if ``chains``, two chains (3 + 1 elements)
    at real positions; the last sample has no chain element at all (a chain-less row, as C2/C7/dropped tiles).
    """
    rng = np.random.default_rng(seed)
    B, Lmax = len(lengths), max(lengths)
    if Lmax % TS.DOWNSAMPLE:
        raise ValueError("Lmax must be a multiple of 6")
    input_ids = np.zeros((B, Lmax), np.int64)
    attention = np.zeros((B, Lmax), bool)
    tracks = np.zeros((B, Lmax, TS.K), np.float16)
    idx = np.zeros((B, TS.N_TOK, 2), np.int64)
    feat = np.zeros((B, TS.N_TOK, TS.N_FEAT), np.int16)
    mask = np.zeros((B, TS.N_TOK), bool)
    T = max(label_lengths)
    labels = np.full((B, T), TS.LABEL_PAD, np.int64)
    for b, L in enumerate(lengths):
        pad = Lmax - L
        input_ids[b, pad:] = rng.integers(7, 11, size=L)
        attention[b, pad:] = True
        t = np.zeros((L, TS.K), np.float32)
        t[:, list(TS.MASK_CHANNELS)] = 1.0
        t[L // 4: L // 2, 0] = np.log1p(3)
        t[L // 4, 3] = np.log1p(2)
        t[L // 2 - 1, 4] = np.log1p(2)
        t[L // 4: L // 2, TS.phase_channel(0, "+")] = np.log1p(1)
        tracks[b, pad:] = t.astype(np.float16)
        labels[b, :label_lengths[b]] = rng.integers(4, 20, size=label_lengths[b])
        if chains and b < B - 1:
            pos = np.zeros((TS.N_TOK, 2), np.int32)
            f = np.zeros((TS.N_TOK, TS.N_FEAT), np.int16)
            m = np.zeros(TS.N_TOK, bool)
            elems = [((5, 10), 0, 2, 7, 1, 0, 3), ((14, 20), 1, 1, 7, 1, 0, 3), ((25, 30), 2, 0, 7, 1, 0, 3),
                     ((33, 40), 0, 0, 12, 0, 1, 1)]
            for e, ((d, a), of, orv, slot, src, strand, units) in enumerate(elems, start=1):
                pos[e] = (d, a)
                f[e] = (of, orv, slot, src, strand, units)
                m[e] = True
            idx[b] = TS.chain_idx(pos, m, pad)
            feat[b] = f
            mask[b] = m
    meta = {"row": [str(100 + b) for b in range(B)], "epoch": [0] * B, "arm": ["B6"] * B,
            "uniforms_sha256": ["0" * 64] * B, "resolved": [None] * B, "context_kind": ["corpus"] * B}
    batch = {"input_ids": input_ids, "attention_mask": attention, "evidence_tracks": tracks,
             "labels": labels if training else None, "chain_idx": idx, "chain_feat": feat, "chain_mask": mask,
             "meta": meta}
    TS.validate_batch(batch, training=training)
    return batch


def stub_batch_torch(**kw):
    import torch

    batch = stub_batch(**kw)
    out = {k: (torch.from_numpy(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else v)
           for k, v in batch.items()}
    TS.validate_batch(out, training=kw.get("training", True))
    return out


# =================================================================================== synthetic evidence world
import hashlib as _hashlib  # noqa: E402
import json as _json  # noqa: E402
import random as _random  # noqa: E402
from types import SimpleNamespace as _NS  # noqa: E402

SPECIES = "Athaliana"
CHROM = "Chr1"
GENOME_LEN = 32000
TILE = 600
# (rn, ws, is_rc, split); rn 2 is rn 1's RC twin; rn 4 a test tile; wid = f"{SPECIES}:{CHROM}:{ws}-{ws+TILE}"
TILES = [(1, 0, False, "train"), (2, 0, True, "train"), (3, 600, False, "valid"), (4, 1200, False, "test"),
         (5, 1800, False, "train"), (6, 2400, False, "valid"), (7, 3000, False, "train"), (8, 3000, True, "train")]
EST_DS, EST_RUN = "est/Athaliana", "est.fa.gz"
ONT_DS, ONT_RUN = "training/ont/Athaliana/proj1", "SRR1"
PB_DS, PB_RUN = "training/pacbio/Athaliana/proj2", "SRR2"
AONT2_DS, AONT2_RUN = "ont/Athaliana_cui2020_PRJNA594286", "SRR10611193"
AONT1_DS, AONT1_RUN = "ont/Athaliana_FLIC_PRJNA1087576", "SRR28341943"
PROT_DS, PROT_RUN = "protein/orthodb", "Athaliana.protein.gff"
LOCUS = ("gH", CHROM, 16000, 16250, "+")        # the panel locus (0-based half-open); centred window [765, 31485)

# molecule specs: (mid, source, dataset, run, library, state, train_tiles_excluded, role, strand, blocks)
MOLECULES = [
    ("e1", "est", EST_DS, EST_RUN, "LE1", "train_track", False, "training_eligible", "+", [(100, 200), (300, 400)]),
    ("e2", "est", EST_DS, EST_RUN, "LE2", "train_track", False, "training_eligible", "-",
     [(50, 100), (150, 250), (350, 450)]),
    ("e3", "est", EST_DS, EST_RUN, "LE1", "train_track", False, "training_eligible", "?", [(500, 550)]),
    ("e4", "est", EST_DS, EST_RUN, "LE1", "train_track", True, "training_eligible", "+", [(700, 760), (800, 860)]),
    ("e5", "est", EST_DS, EST_RUN, "LE1", "train_track", False, "training_eligible", "+", [(620, 650), (700, 760)]),
    ("o1", "ont", ONT_DS, ONT_RUN, "SRX1", "train_track", False, "training_eligible", "+",
     [(10, 40), (60, 90), (110, 140), (160, 190)]),
    ("p1", "pacbio", PB_DS, PB_RUN, "SRX2", "train_track", False, "training_eligible", "+",
     [(10, 40), (60, 90), (110, 140), (160, 190)]),
    ("o3", "ont", ONT_DS, ONT_RUN, "SRX1", "train_track", False, "training_eligible", "+", [(1850, 1900), (1950, 2000)]),
    ("eh", "est", EST_DS, EST_RUN, "LE1", "excluded_heldout", False, "training_eligible", "+",
     [(16010, 16050), (16150, 16200)]),
    ("oh", "ont", AONT2_DS, AONT2_RUN, "SRX9", "excluded_heldout", False, "training_eligible", "+",
     [(16020, 16050), (16100, 16130), (16170, 16190)]),
    ("oh_far", "ont", AONT2_DS, AONT2_RUN, "SRX9", "excluded_heldout", False, "training_eligible", "+",
     [(17000, 17040), (17060, 17070)]),
    ("s1", "ont", AONT1_DS, AONT1_RUN, "SRX8", "excluded_heldout", False, "validation_only", "+",
     [(16020, 16050), (16100, 16130)]),
    ("e6", "est", EST_DS, EST_RUN, "LE1", "train_track", True, "training_eligible", "+", [(2200, 2250), (2300, 2350)]),
    ("n1", "est", EST_DS, EST_RUN, "LE1", "never", False, "training_eligible", "+", [(210, 260), (270, 290)]),
    ("pr1", "protein", PROT_DS, PROT_RUN, "orthodb", "train_track", False, "training_eligible", "+", [(100, 160)]),
    ("pr2", "protein", PROT_DS, PROT_RUN, "orthodb", "train_track", False, "training_eligible", "+", [(140, 200)]),
    ("pr3", "protein", PROT_DS, PROT_RUN, "orthodb", "train_track", False, "training_eligible", "-", [(150, 170)]),
    ("prx", "protein", PROT_DS, PROT_RUN, "orthodb", "excluded_heldout", False, "training_eligible", "+", [(100, 400)]),
]


def cap_molecules():
    """Tile rn 7 [3000, 3600): 18 ONT chains x 27 introns (486 elements) + 70 one-intron EST chains."""
    out = []
    for k in range(18):
        s = 3010 + 3 * k
        blocks = [(s + 10 * j, s + 10 * j + 5) for j in range(28)]
        out.append((f"cap_o{k}", "ont", ONT_DS, ONT_RUN, "SRX1", "train_track", False, "training_eligible", "+",
                    blocks))
    for k in range(70):
        s = 3300 + 4 * k
        out.append((f"cap_e{k}", "est", EST_DS, EST_RUN, "LE1", "train_track", False, "training_eligible", "+",
                    [(s, s + 2), (s + 4, s + 6)]))
    return out


ORGANISM = {"pr1": "3701:0001", "pr2": "3701:0002", "pr3": "4577:0003", "prx": "3702:0009"}
# route-B HSP intervals: (hsp_id, molecule, start0, end0, strand, phase, leakage_excluded, train_tiles_excluded)
HSPS = [("h1", "e1", 100, 130, "+", 0, False, False), ("h2", "e2", 160, 190, "-", 2, False, False),
        ("h3", "eh", 16010, 16040, "+", 1, True, False)]
XJUNC = [("+", "donor", 40)]                         # rule 4(c): o1/p1 first intron donor (intron [40,60))
XINT = [(16000, 16250, "heldout", "all", "gH", None), (580, 620, "edge_partial", "all", "gP", 1),
        (2050, 2100, "valid_orthogroup", "train_tiles", "gV", None)]
ZERO = {"train_track": [(120, 125)], "valid_track": [(120, 125)], "eval_input": [(16020, 16022)]}


def introns_of(strand, blocks):
    out = []
    for (_, e_prev), (s_next, _) in zip(blocks, blocks[1:]):
        s0, e0 = e_prev, s_next
        d, a = (s0, e0 - 1) if strand != "-" else (e0 - 1, s0)
        out.append((s0, e0, d, a))
    return out[::-1] if strand == "-" else out


def _sha(path):
    return _hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_genome(tmp):
    rng = _random.Random(7)
    seq = [rng.choice("ACGT") for _ in range(GENOME_LEN)]
    seq[590:600] = "N" * 10
    genome = "".join(seq)
    fa = Path(tmp) / "genome.fa"
    fa.write_text(f">{CHROM}\n" + "\n".join(genome[i:i + 60] for i in range(0, len(genome), 60)) + "\n")
    return fa, genome


def _revcomp(s):
    return s.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


# Reference annotation of the fixture world (post-merge, review 3 M8): complete genes, one transcript each, GFF3
# 1-based features. gene_id -> (strand, [(type, start1, end1, phase)]). gE1 follows e1's exons [100,200)+[300,400)
# on '+'; gO3 follows o3's exons [1850,1900)+[1950,2000) (CDS 50 nt, so the second CDS has phase 1); gC1 is a '-'
# gene in tile [3000, 3600) (first CDS in transcription order is 3301-3390, 90 nt, so 3101-3200 has phase 0).
FIXTURE_GENES = {
    "gE1": ("+", [("five_prime_UTR", 101, 110, "."), ("CDS", 111, 200, "0"), ("CDS", 301, 390, "0"),
                  ("three_prime_UTR", 391, 400, ".")]),
    "gO3": ("+", [("CDS", 1851, 1900, "0"), ("CDS", 1951, 2000, "1")]),
    "gC1": ("-", [("three_prime_UTR", 3091, 3100, "."), ("CDS", 3101, 3200, "0"), ("CDS", 3301, 3390, "0"),
                  ("five_prime_UTR", 3391, 3400, ".")]),
}


def gsf_contract():
    """The gsf_contract module the B6 corpus builder uses (b6_corpus_repair._gc -> leakage.load_build_b5().gc), loaded
    without the torch-importing transgenic.utils package."""
    from transgenic.evidence import b6_corpus_repair as CR

    return CR._gc()


def tile_labels():
    """{rn: v3 GSF} for TILES. A forward tile's label is window_to_gsf_v3 of the FIXTURE_GENES that lie wholly inside
    it (EMPTY_LABEL when none); an RC twin's label is reverse_complement_v3(forward label, TILE), the derivation
    b6_corpus_repair and b6_validate_corpus use, so forward/RC twins are internally consistent."""
    gc = gsf_contract()
    forward = {}
    for rn, ws, is_rc, _ in TILES:
        if is_rc:
            continue
        genes = []
        for gid, (strand, feats) in FIXTURE_GENES.items():
            fs = [gc.Feature(*f) for f in feats]
            s0, e0 = min(f.start1 for f in fs) - 1, max(f.end1 for f in fs)
            if ws <= s0 and e0 <= ws + TILE:
                genes.append(gc.Gene(gid, CHROM, strand, s0, e0, {f"{gid}.1": fs}))
        forward[ws] = gc.window_to_gsf_v3(genes, ws)
    out = {}
    for rn, ws, is_rc, _ in TILES:
        out[rn] = gc.reverse_complement_v3(forward[ws], TILE) if is_rc else forward[ws]
        gc.check_caps_v3(out[rn], TILE)
    return out


def write_corpus(tmp, genome):
    import duckdb

    path = Path(tmp) / "b5_full_b6_v1.db"
    labels = tile_labels()
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE geneList (rn INT PRIMARY KEY, species_id VARCHAR, chromosome VARCHAR, start INT, "
                "fin INT, is_rc BOOLEAN, split VARCHAR, sequence VARCHAR, gene_id VARCHAR, gff VARCHAR)")
    for rn, ws, is_rc, split in TILES:
        seq = genome[ws:ws + TILE]
        wid = f"{SPECIES}:{CHROM}:{ws}-{ws + TILE}"
        con.execute("INSERT INTO geneList VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [rn, SPECIES, CHROM, ws, ws + TILE, is_rc, split, _revcomp(seq) if is_rc else seq, wid,
                     labels[rn]])
    con.close()
    return path


BASE_SHA = "b" * 64                                  # the A40 corpus sha the (fake) C0 was built against
MANIFEST_ROWS = [dict(manifest_version="b6-intervals-v2", species="Athaliana", chrom="Chr1", component_id="gH",
                      members=["gH"], start0=16000, end0=16250, cause="heldout", scope="all", flank=60,
                      flank_source="historical_max", c0_exclusion=True)]


def _canonical_manifest_sha(rows):
    from transgenic.evidence import b6_bindings as BD
    return BD.interval_manifest_sha256(rows)


IMAN_SHA = _canonical_manifest_sha(MANIFEST_ROWS)    # the b6-intervals-v2 manifest sha recorded in the freeze
FAKE_C0_SHA = {"est": "e" * 64, "ont": "0" * 63 + "1", "pacbio": "0" * 63 + "2", "protein": "0" * 63 + "3"}


def write_c0(tmp, *, molecules=MOLECULES, hsps=HSPS, xjunc=XJUNC, xint=XINT, provenance_extra=None,
             drop_provenance_keys=(), exhaustiveness=None, reference_sha="f" * 64, qc_rows=True):
    import duckdb
    from transgenic.evidence import c0_schema as CS

    path = Path(tmp) / "merged_c0.db"
    con = duckdb.connect(":memory:")            # file-backed DDL is slow on WSL; build in memory, copy once
    CS.create_schema(con, merged=True)
    prov = "0" * 64

    pending = {}

    def ins(table, **vals):
        pending.setdefault(table, []).append(vals)

    for mid, src, ds, run, lib, state, tte, role, strand, blocks in molecules:
        common = dict(species=SPECIES, source=src, dataset=ds, run=run, library=lib, genotype_stratum="reference",
                      role=role, validation_only=(role == "validation_only"), provenance_sha256=prov)
        member = f"m_{mid}"
        aid = f"a_{mid}"
        read = ORGANISM.get(mid, f"read_{mid}")
        ins("molecule_member", **common, member_id=member, molecule_id=mid, read_name=read, accession=read,
            accession_stem=read, unit_rule="test", superseded_version=False, molecule_state=state,
            train_tiles_excluded=tte)
        ins("evidence_alignment", **common, alignment_id=aid, member_id=member, molecule_id=mid, read_name=read,
            record_index=0, placement_kind="primary", flag=0, chrom=CHROM, start0=blocks[0][0], end0=blocks[-1][1],
            alignment_strand="+", transcript_strand=strand, mapq=60, accepted=True, mapping_ambiguous=False,
            arm="primary")
        order = blocks if strand != "-" else blocks[::-1]
        for k, (s, e) in enumerate(blocks):
            ins("aligned_block", **common, alignment_id=aid, molecule_id=mid, alignment_block_index=k,
                block_index=order.index((s, e)), chrom=CHROM, start0=s, end0=e, strand=strand)
        if src == "protein" or strand == "?":
            continue
        ints = introns_of(strand, blocks)
        for k, (s0, e0, d, a) in enumerate(ints):
            ins("junction_observation", **common, observation_id=f"{aid}.j{k}", alignment_id=aid, molecule_id=mid,
                intron_index=k, chrom=CHROM, strand=strand, raw_start0=s0, raw_end0=e0, raw_donor0=d,
                raw_acceptor0=a, start0=s0, end0=e0, donor0=d, acceptor0=a, correction_status="canonical",
                correction_radius=0, accepted=True, canonical=True, junction_key=f"{CHROM}:{strand}:{s0}-{e0}")
        if ints:
            ins("partial_chain", **common, chain_id=f"c_{mid}", alignment_id=aid, molecule_id=mid, member_id=member,
                placement_rank=0, chrom=CHROM, strand=strand, start0=blocks[0][0], end0=blocks[-1][1],
                n_introns=len(ints), intron_chain_key="k", all_junctions_accepted=True)
            for k, (s0, e0, d, a) in enumerate(ints):
                ins("partial_chain_intron", chain_id=f"c_{mid}", intron_index=k, tx_order=k,
                    observation_id=f"{aid}.j{k}", chrom=CHROM, strand=strand, start0=s0, end0=e0, donor0=d,
                    acceptor0=a, correction_status="canonical", accepted=True, **common)
    lib_of = {m[0]: m[4] for m in molecules}
    for hid, mid, s, e, strand, phase, lexc, ttexc in hsps:
        ins("est_protein_hsp", hsp_id=hid, species=SPECIES, library=lib_of[mid], molecule_id=mid,
            est_accession=mid, protein_id="3701:0001", protein_source="orthodb", protein_species="3701",
            est_start=0, est_end=e - s, frame=1, evalue=1e-9, bitscore=80.0, interval_index=0, chrom=CHROM,
            genomic_start0=s, genomic_end0=e, strand=strand, phase=phase, diamond_version="2.1.24",
            database_sha256="d" * 64, command_sha256="c" * 64, provenance_sha256=prov, source="protein",
            dataset=PROT_DS, run=PROT_RUN, genotype_stratum="not_applicable", role="training_eligible",
            validation_only=False, c0_provenance_sha256=prov, leakage_excluded=lexc, train_tiles_excluded=ttexc)
    for strand, kind, coord in xjunc:
        ins("excluded_junction_coordinate", species=SPECIES, source="est", chrom=CHROM, strand=strand, kind=kind,
            coordinate0=coord, excluded_gene_id="gX", neighbour_gene_id="gY")
    for s, e, cause, scope, gene, tile_rn in xint:
        ins("excluded_interval", species=SPECIES, source="est", chrom=CHROM, start0=s, end0=e, cause=cause,
            scope=scope, gene_id=gene, tile_rn=tile_rn)
    for src in sorted({m[1] for m in molecules}):
        pv = {"inputs": {"corpus": {"path": "a40.db", "sha256": BASE_SHA},
                         "reference": {"path": "genome.fa", "sha256": reference_sha}},
              "leakage_intervals_sha256": "l" * 64, "interval_manifest_sha256": IMAN_SHA,
              "completeness_codes": {"policy": "null_not_computed_at_c0"}}
        pv.update(provenance_extra.get(src, {}) if provenance_extra else {})
        for k in drop_provenance_keys:
            pv.pop(k, None)
        for k, v in sorted(pv.items()):
            ins("c0_provenance", species=SPECIES, source=src, key=k, value=_json.dumps(v, sort_keys=True))
        if qc_rows:
            ins("c0_qc", species=SPECIES, source=src, metric="leakage_nonzero", value=0.0)
        c0_path = Path(tmp) / f"{SPECIES}.{src}.c0.db"
        Path(str(c0_path) + ".PROVENANCE.txt").write_text(f"output\t{c0_path}\noutput_sha256\t{FAKE_C0_SHA[src]}\n")
        ins("merge_manifest", input_path=str(c0_path), input_sha256=FAKE_C0_SHA[src], species=SPECIES, source=src)
    ex = {SPECIES: {"by_source": {s: {"below_cap": 10000, "molecules": 10000} for s in ("est", "ont", "pacbio")}}}
    if exhaustiveness is not None:
        ex = exhaustiveness
    ins("c0_meta", key="exhaustiveness", value=_json.dumps(ex))
    import pyarrow as pa

    for table, rows in pending.items():
        cols = [c for c, _ in CS.table_columns(table)]
        staged = pa.table({c: [r.get(c) for r in rows] for c in cols})  # noqa: F841 (read by DuckDB by name)
        con.execute(f"INSERT INTO {table} SELECT * FROM staged")
    con.execute(f"ATTACH '{path}' AS out")
    con.execute("COPY FROM DATABASE memory TO out")
    con.execute("DETACH out")
    con.close()
    Path(str(path) + ".sha256").write_text(f"{_sha(path)}  {path.name}\n")
    return path


ROLE_ROWS = [(AONT2_DS, AONT2_RUN, "A-ONT2", True, False, "tracks+chains"),
             (EST_DS, EST_RUN, "A-EST", True, False, "tracks+chains"),
             (AONT1_DS, AONT1_RUN, "A-ONT1", False, True, "-")]


def make_roles(rows=ROLE_ROWS, sha="r" * 64):
    return _NS(sha256=sha, rows=tuple(_NS(dataset=d, run=r, species=SPECIES, set_label=lab, b6_input=i, b6_score=s,
                                          b6_input_scope=sc) for d, r, lab, i, s, sc in rows))


def write_zero_lists(tmp, zero=ZERO, upstream=None):
    out = {}
    for pop, ivs in zero.items():
        t = Path(tmp) / f"zero_{pop}.tsv"
        t.write_text("species\tchrom\tstart0\tend0\taction\treason\n" + "".join(
            f"{SPECIES}\t{CHROM}\t{s}\t{e}\tzero_all_route_b_phase_channels\troute_a_b_disagreement\n"
            for s, e in ivs))
        q = Path(tmp) / f"zero_{pop}.qc.json"
        q.write_text(_json.dumps({"population": pop, "species": SPECIES,
                                  "upstream": upstream or {"est_c0": FAKE_C0_SHA["est"],
                                                           "protein_c0": FAKE_C0_SHA["protein"]}}))
        out[pop] = {SPECIES: (t, q)}
    return out


def write_panel(tmp):
    p = Path(tmp) / "panel.tsv"
    gid, chrom, s0, e0, strand = LOCUS
    p.write_text("panel\tshuffle_rank\tspecies_id\tgene_id\tchromosome\tstart\tfin\tstrand\tlength\n"
                 f"pilot\t1\t{SPECIES}\t{gid}\t{chrom}\t{s0 + 1}\t{e0}\t{strand}\t{e0 - s0}\n")
    return p, _sha(p)


def writer_a_freeze(corpus, manifests):
    """A freeze record shaped exactly like b6_corpus_repair's producer output (/tmp/b6-wp2-corpus
    src/transgenic/evidence/b6_corpus_repair.py, freeze ``record``): interval_manifest_sha256 is {species: sha256}
    and interval_manifests is {species: {path, sha256}}.  Copied by shape; the module is not imported."""
    return {"rule_id": "rule5-leak-hard-c0interval-v1", "corpus_version": "b5_full_b6_v1", "base": "/x/a40.db",
            "base_sha256": BASE_SHA, "out": str(Path(corpus).absolute()), "out_sha256": _sha(corpus),
            "content_sha256": "c" * 64,
            "interval_manifest_sha256": {k: v["sha256"] for k, v in sorted(manifests.items())},
            "interval_manifests": manifests, "seed": 121, "code_sha256": {}, "argv": [], "inputs": {},
            "species_inputs": {}, "counts": {}, "species_qc": {}, "declared_differences": {}, "events": {},
            "row_counts": {}, "rows_repaired": 0, "b6_dna_repair_rows": 0, "b6_label_drop_rows": 0,
            "duckdb_version": "1.5.5", "geometry": "D1 default", "grandfathered": "A40 per-gene flanks"}


def write_interval_manifest(path, rows):
    """Canonical b6-intervals-v2 JSONL + sidecar (the byte format of leakage.write_interval_manifest)."""
    from transgenic.evidence import b6_bindings as BD

    rows = sorted(rows, key=lambda r: (r["species"], r["chrom"], r["start0"], r["end0"], r["cause"], r["scope"],
                                       r["component_id"]))
    Path(path).write_text("".join(_json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows))
    digest = BD.interval_manifest_sha256(rows)
    Path(str(path) + ".sha256").write_text(f"{digest}  {Path(path).name}\n")
    return digest


class Ref:
    def __init__(self, path):
        self.path, self.sha256 = Path(path), _sha(path)


def make_world(tmp, *, molecules=MOLECULES, hsps=HSPS, zero=ZERO, roles_rows=ROLE_ROWS, c0_kw=None):
    """Build genome, corpus, merged C0, roles, zero lists, panel and the training-family obs cache."""
    from transgenic.evidence import b6_obs_cache as OC
    import duckdb

    tmp = Path(tmp)
    fa, genome = write_genome(tmp)
    corpus = write_corpus(tmp, genome)
    c0 = write_c0(tmp, molecules=molecules, hsps=hsps, **dict({"reference_sha": _sha(fa)}, **(c0_kw or {})))
    roles = make_roles(roles_rows)
    zl = write_zero_lists(tmp, zero)
    con = duckdb.connect(str(c0), read_only=True)
    rows, drops, libs = OC.extract_rows(con, roles, "train")
    con.close()
    mpath = tmp / "intervals.Athaliana.jsonl"
    if not mpath.exists():
        assert write_interval_manifest(mpath, MANIFEST_ROWS) == IMAN_SHA
    freeze = writer_a_freeze(corpus, {SPECIES: {"path": str(mpath), "sha256": IMAN_SHA}})
    freeze_path = tmp / "freeze.json"         # same bytes as freeze_file(tmp, freeze): verify(w) copies stay bound
    freeze_path.write_text(_json.dumps(freeze, indent=1, sort_keys=True) + "\n")
    cache_dir = tmp / "cache_train"           # bound like b6_build_obs_cache: corpus AND freeze (verify_cache checks)
    OC.write_cache(cache_dir, rows, family="train", libraries=libs, drops=drops,
                   bindings={"reference_sha256": {SPECIES: _sha(fa)}, "roles_sha256": roles.sha256,
                             "evidence_sha256": _sha(c0), "corpus_sha256": _sha(corpus),
                             "freeze_sha256": _sha(freeze_path)}, zero_lists=zl)
    panel, panel_sha = write_panel(tmp)
    return _NS(tmp=tmp, fasta=fa, genome=genome, corpus=corpus, c0=c0, roles=roles, zero=zl, cache_dir=cache_dir,
               rows=rows, drops=drops, libraries=libs, panel=panel, panel_sha=panel_sha, freeze=freeze,
               freeze_path=freeze_path)


def load_b6_dataset():
    """Import src/transgenic/datasets/b6_dataset.py by full path (plan §0): the package __init__ imports torch."""
    import importlib.util

    name = "b6_dataset_bypath"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / "transgenic" / "datasets" / "b6_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def char_encoder(seq):
    """Test DNA encoder: one token per base (token offset 0), pad id 4."""
    table = {"A": 7, "C": 8, "G": 9, "T": 10, "N": 11}
    return np.asarray([table[c] for c in seq.upper()], np.int64), 4


def gsf_tokens(gff):
    return np.asarray([3] + [ord(c) % 50 + 5 for c in gff] + [2], np.int64)


def zmays_eval_c0(tmp):
    """A real eval-only C0 for 'Zmays' through b6_build_eval_c0 (b6_build_c0 ingestion, null leakage model) on the
    C0 test world's genome and EST SAMs, merged by b6_merge_evidence_db.  No corpus file is written."""
    import test_b6_c0_world as W
    import b6_build_eval_c0 as E
    import b6_merge_evidence_db as M

    tmp = Path(tmp)
    saved = W.SPECIES
    W.SPECIES = "Zmays"
    try:
        fasta, _genome = W.build_genome(tmp)
        lines = ["##gff-version 3"]
        for gid, (chrom, strand, exons, _, _) in W.GENES.items():
            gs, ge = exons[0][0], exons[-1][1]
            lines.append(f"{chrom}\tt\tgene\t{gs}\t{ge}\t.\t{strand}\t.\tID={gid};Name={gid}")
        gff = tmp / "genes.gff3"
        gff.write_text("\n".join(lines) + "\n")
        align, _ = W.write_est(tmp, audit=False)
        lr, est = W.write_roles(tmp, est_role="b1_validation_only")
        prov = W.write_provenance(tmp)
        out = tmp / "Zmays.est.eval_c0.db"
        rec = E.build_eval(["--roles", str(lr), "--est-roles", str(est), "--species", "Zmays", "--source", "est",
                            "--align-root", str(align), "--reference", str(fasta), "--gff", str(gff),
                            "--provenance", str(prov), "--out", str(out), "--qc", str(out) + ".qc.json"])
        assert rec["status"] == "PASS", rec
        merged = tmp / "Zmays.eval_merged.db"
        M.merge([str(out)], str(merged))
    finally:
        W.SPECIES = saved
    return _NS(fasta=fasta, gff=gff, c0=out, c0_sha=rec["output_sha256"], merged=merged, est_dataset="est/Zmays",
               est_run="est.fa.gz")


def freeze_file(tmp, freeze, name="freeze_variant.json"):
    p = Path(tmp) / name
    p.write_text(_json.dumps(freeze, indent=1, sort_keys=True) + "\n")
    return p
