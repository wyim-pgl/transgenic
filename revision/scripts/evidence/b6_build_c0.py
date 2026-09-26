#!/usr/bin/env python3
"""Full C0 builder for B6 evidence mode (A46 rules 2-4, SPEC-1).  One call = one species x one source.

    b6_build_c0.py --roles DATASET_ROLES.longread_v1.tsv --est-roles DATASET_ROLES.est_v1.tsv
                   --species Athaliana --source est|ont|pacbio|protein
                   --align-root <frozen alignments> --audit-root <b6_audit_align output>
                   --corpus <corpus DuckDB> --splits b5_orthogroup_split_v1.tsv --flags <A22 geenuff_flags.tsv>
                   [--build-flags <every flag file the corpus build used>] [--overlap <intervals TSV to verify>]
                   [--hsp <est_protein_hsp TSV/parquet> --est-c0 <same species' EST C0 DB> --lineage <frozen lineage TSV>]
                   --provenance PROVENANCE.<tools>.json [...] [--memory-limit 2GB --threads 4]
                   --out <species x source evidence DB, exclusive create> --qc <QC JSON, exclusive create>

SPEC-1 forced order as implemented here (steps 1, 3-rerun and 6 live elsewhere):
    [1  b6_audit_align.sbatch (part B) -- consumed: its minimap2 @PG CL must equal the frozen primary CL with only
        --secondary=no replaced by --secondary=yes -N <N> -p 0.5]
     2  rule 4(b) decision: molecule units are formed first (4(b) is molecule-wide), then every member's
        frozen primary/supplementary and audit placements are tested; a molecule with >= N secondary placements
        (summed over members) or a MAPQ < 5 placement outside every excluded interval -> unauditable.
     3  exhaustiveness (fraction of molecules with < N secondary placements): a SPECIES-level quantity ("reported
        per species and required to be >= 99.9 %", rule 4(b)) AND, by author decision of 2026-09-23, a per
        (species x source) floor.  This build records the per-source numerator / denominator (qc
        exhaustiveness_below_cap / exhaustiveness_molecules) and ABORTS below the per-source floor
        (leakage.MIN_SOURCE_EXHAUSTIVENESS, nothing published); b6_merge_evidence_db.py re-checks the per-source
        floor from the recorded counters and enforces the summed species threshold where all sources meet.
        Molecule units (step 2) are keyed on the metadata `library` (A11: runs do not create libraries; the
        producer's --library, never the run), so two runs of one preparation share their A16 units.
     4  C0 load: (a)/(b) decisions applied as molecule_state, merged molecule units, observation tables.
     5  rule 4(c): junction_support at boundary coordinates shared by an excluded gene and a neighbour.
    [6  b6_merge_evidence_db.py]
     7  QC gate: independent re-test of the WRITTEN rows (aligned blocks, junction observations and frozen
        alignments of train_track molecules against the written 'all' intervals; of train-tile-eligible rows
        against 'train_tiles' intervals; HSP rows; held-out loci) -> any non-zero count aborts; the DB is
        published (hard link, no clobber) only after the gate passes.

Protein source (miniprot, A19.2): one molecule per protein query.  A query at the miniprot ``-N 20`` cap
(21 placements) may have lost placements, so it is unauditable and excluded (the protein analogue of rule 4(b)).
Route-B HSP rows (``--hsp``, required for --source protein) must reference molecules of the same species' EST
C0 DB (``--est-c0``; ``b6_est_molecule_map.py`` exports those ids to part B); an HSP inherits its EST molecule's
exclusion and train_tiles flag.  A19 is enforced by NCBI lineage (``b6_hsp_project.check_a19``, frozen
``--lineage``) for every HSP row and every miniprot query.

Streaming (scale): alignments are read once, run by run, in a single pass; per-read metadata and audit facts are
looked up in DuckDB batches; every row is staged in column chunks to a scratch DuckDB (``<out>.partial-*.stg``)
and the final tables are produced there by SQL joins; molecule decisions stream molecule by molecule; protein
rows are emitted chromosome by chromosome.  Python memory is bounded by the batch size, the largest molecule and
the largest PCR-equivalence group; DuckDB memory by --memory-limit (it spills to a temp directory).

Row order: the final tables have NO guaranteed physical row order (rows are produced by parallel SQL joins with
``preserve_insertion_order = false``; the same holds after b6_merge_evidence_db.py).  Every reader must ``ORDER BY`` a
key (e.g. the table's primary key) when order matters; hashes over rows must sort first (as intervals_sha256 does).

Route-B inputs next to ``--hsp`` (b6_hsp_project.py writes both): ``<hsp>.molecule_state.tsv`` (cross-checked against
the EST C0 molecule_member) and ``<hsp>.provenance.json`` (its sha256 must be every row's provenance_sha256; its
species, phase_convention and alignment_sha256 must equal this build's species, c0_schema.PHASE_CONVENTION and the EST
C0's frozen primary alignment sha256).  ``--seed`` is required: the corpus build_manifest does not record it.

Input layout (mirrors est_align.sbatch / longread_align.sbatch / protein_merge.sbatch / b6_audit_align.sbatch):
    est      <align>/<species>/<species>.est.{bam,sam,sam.gz} + DONE (bam_md5=|sam_md5=, arm=primary; arm= required)
    ont/pb   <align>/<species>/<basename(dataset)>/<run>/<run>.{bam,sam,sam.gz} + DONE (bam_md5=|sam_md5=, uf=)
    protein  <align>/<species>/<species>.protein.gff[.gz] + DONE (gff_md5)
    audit    <audit>/<species>/<dsbase>/<run>/<run>.audit.bam + DONE (bam_md5=, source=, run=) exactly as
             b6_audit_align.sbatch writes it; EST: <audit>/<species>/est/est_primary/est_primary.audit.bam
             (b6_audit_runs.py row 'est <sp> est est_primary').  .sam/.sam.gz with sam_md5= are accepted.
    reads    c0_metadata.tsv (+ .sha256) next to each frozen alignment (b6_c0_metadata.py).  EVERY read of the
             alignment needs a row (a missing read fails closed); a first data row with read='*' carries the
             explicit run-level values of the producer; columns: see META_FIELDS.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve()
_SCRIPTS = _HERE.parents[1]
_REPO = _HERE.parents[3]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from transgenic.evidence import c0_schema as S  # noqa: E402
from transgenic.evidence import leakage as LK  # noqa: E402


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ES = _load("evidence_support_59", _SCRIPTS / "59_evidence_support.py")
PA = _load("protein_accept_c0", _SCRIPTS / "evidence" / "protein_accept.py")
GEOM = _load("est_phase_project_c0", _SCRIPTS / "evidence" / "est_phase_project.py")   # MULTI_HSP_RULE / resolve_hsps

# A46 rule 4(b) (protocol line 990): "Exhaustiveness = the fraction of molecules with fewer than N = 50 placements,
# reported per species and required to be >= 99.9 %".  Author decision 2026-09-23 (fix round 6): BOTH the species-level
# gate (the merge, where every source of a species meets) AND a per (species x source) floor, enforced fail-closed at
# build time (decide_all -> Exhaustiveness.check) and re-checked by the merge from the recorded counters.  The author
# gave no per-source number; the species value (0.999) is used and recorded here (assumption, stated in the report).
EXHAUSTIVENESS_RULE = {"scope": "per_source_floor_and_species",
                       "rule": "per-source floor 0.99 (A49 decision 2 / A49.2) + species >= 0.999 (rule 4(b))",
                       "min_exhaustiveness": LK.MIN_EXHAUSTIVENESS,
                       "min_source_exhaustiveness": LK.MIN_SOURCE_EXHAUSTIVENESS,
                       "per_source_counters": ["exhaustiveness_below_cap", "exhaustiveness_molecules", "audit_n"],
                       "enforced_by": "b6_build_c0.py decide_all (per-source floor, abort) + b6_merge_evidence_db.py "
                                      "(per-source floor from the recorded counters, then the species sum over est/ont/pacbio)"}

# A16 (protocol line 625): "Every accepted read alignment carries three codes stored in the C0 tables:
# `complete_5p_code` (5C/5I), `complete_3p_code` (3C/3I), `read_internal_code` (IC/IP/IM/IX), with the thresholds of
# the design document §2 ... direct-RNA reads default to 5I unless cap/CAGE clustering establishes 5C."  This build
# writes NULL into the three (nullable, c0-v1) columns for every row and records that policy here; the step-7 gate
# refuses a DB whose rows contradict it (a partial, silent fill).  Not computed because design §2 needs inputs this
# per-run streaming build does not have: 5C = cap/CAGE TSS clusters (>= 5 units from >= 2 libraries, medoid within
# +-50 nt of a CAGE peak), 3C = untemplated poly(A) >= 10 nt through the A3 internal-priming filter or a TES cluster
# with >= 2 tail-positive units, IC/IP = membership in the ADMITTED chain set (support thresholds over all molecules).
# AUTHOR DECISION PENDING (fix round 5, finding 6): computing them is a separate post-C0 pass over the merged DB or an
# A16 amendment stating the C0-time defaults; no value is invented here.
COMPLETENESS_CODES_POLICY = {
    "policy": "null_not_computed_at_c0",
    "columns": ["complete_5p_code", "complete_3p_code", "read_internal_code"],
    "clause": "A16 (PROTOCOL_B1_frozen_v1.md line 625): every accepted read alignment carries complete_5p_code (5C/5I), "
              "complete_3p_code (3C/3I), read_internal_code (IC/IP/IM/IX) with the thresholds of the design document §2; "
              "direct-RNA reads default to 5I unless cap/CAGE clustering establishes 5C",
    "missing_inputs": ["cap/CAGE peaks and TSS clusters (5C)", "untemplated poly(A) evidence and TES clusters (3C)",
                       "admitted chain set over all molecules of the species (IC/IP)"],
    "status": "AUTHOR_DECISION_PENDING_2026-09-23",
}
COMPLETENESS_CODES_POLICIES = ("null_not_computed_at_c0",)
MULTI_HSP_APPLIED_BY = "b6_build_c0.load_hsp"

ROLE_MAP = {"c2_training_eligible": "training_eligible", "b1_validation_only": "validation_only",
            "sensitivity_set": "validation_only"}
ES_SOURCE = {"est": "EST", "ont": "ONT", "pacbio": "PacBio"}
RADIUS = {"est": 0, "ont": 3, "pacbio": 1, "protein": 0}
NEVER_TRAINING_SPECIES = ("Zmays", "Slycopersicum")      # A46 rule 1
META_FIELDS = ("accession", "library", "bioproject", "genotype_stratum", "clone", "ingestion_qc_pass",
               "ingestion_qc_reason", "mapping_ambiguous", "post_trim_length", "protocol", "umi_status", "umi", "sample",
               "locus", "source_molecule", "oriented_genomic_strand", "paralog_competes",
               "allele_discriminating_matches_100nt")
BOOL_FIELDS = ("ingestion_qc_pass", "mapping_ambiguous", "paralog_competes")
INT_FIELDS = ("post_trim_length", "allele_discriminating_matches_100nt")
MINIPROT_MAX_PLACEMENTS = 21    # miniprot -N 20 (A19.2): primary + 20 secondaries; a query at the cap may be truncated
CHUNK = 50_000                  # staged rows per DuckDB insert / rows per fetch
BATCH = 20_000                  # read groups per metadata/audit lookup


class C0Error(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise C0Error(message)


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def md5_and_sha256(path):
    m, s = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            m.update(block)
            s.update(block)
    return m.hexdigest(), s.hexdigest()


def sid(*parts):
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()


def junction_key(species, chrom, strand, start0, end0):
    return sid("junction", species, chrom, strand, start0, end0)


def boundary(strand, start0, end0):
    """donor0/acceptor0: 0-based INCLUSIVE first/last intron base in transcript order (None for '?')."""
    if strand == "+":
        return start0, end0 - 1
    if strand == "-":
        return end0 - 1, start0
    return None, None


def lit(value):
    return "'" + str(value).replace("'", "''") + "'"


# ----------------------------------------------------------------------------------------------- gates
def read_roles(paths):
    rows = []
    seen = set()
    for path in paths:
        for row in ES.read_roles([path]).values():
            key = (row["dataset"], row["run"])
            require(key not in seen, f"duplicate dataset/run role across manifests: {key}")
            seen.add(key)
            rows.append(row)
    return rows


def dataset_source(dataset):
    parts = dataset.split("/")
    if parts[0] in ("est", "protein", "ont", "pacbio"):
        return parts[0]
    if parts[0] == "training" and len(parts) > 1 and parts[1] in ("ont", "pacbio"):
        return parts[1]
    return None


def select_runs(rows, species, source):
    chosen, excluded = [], []
    for row in rows:
        if row["species"] != species or dataset_source(row["dataset"]) != source:
            continue
        if row["role"] == "excluded":
            excluded.append(row)
            continue
        require(row["role"] in ROLE_MAP, f"{row['dataset']}/{row['run']}: role {row['role']!r} is not loadable")
        if species in NEVER_TRAINING_SPECIES:
            require(row["role"] != "c2_training_eligible",
                    f"{species} evidence can never be training-side (A46 rule 1): {row['dataset']}/{row['run']}")
        chosen.append(row)
    require(chosen, f"no DATASET_ROLES row for {species} x {source} (A18.3: a role row is required per run)")
    return chosen, excluded


def refuse_unroled(root, species, source, roles_rows):
    """A18.3: an alignment directory with no DATASET_ROLES row at all is refused, never skipped."""
    if source not in ("ont", "pacbio"):
        return
    known = {(Path(r["dataset"]).name, r["run"]) for r in roles_rows if r["species"] == species}
    base = Path(root) / species
    if not base.is_dir():
        return
    for done in sorted(base.glob("*/*/DONE")):
        key = (done.parent.parent.name, done.parent.name)
        require(key in known, f"alignment {done.parent} has no DATASET_ROLES row (A18.3 fail-closed)")


def read_done(path):
    """DONE / PROVENANCE files use 'key=value' or 'key<ws>value' lines."""
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition("=")
            if not sep or " " in key or "\t" in key:
                parts = line.split(None, 1)
                key, value = parts[0], (parts[1] if len(parts) > 1 else "")
            out[key.strip()] = value.strip()
    return out


def alignment_dir(root, species, source, row):
    if source in ("est", "protein"):
        return Path(root) / species
    return Path(root) / species / Path(row["dataset"]).name / row["run"]


EST_AUDIT_DATASET, EST_AUDIT_RUN = "est", "est_primary"   # b6_audit_runs.est_rows: ('est', sp, 'est', 'est_primary', ...)


def audit_run_name(source, row):
    return EST_AUDIT_RUN if source == "est" else row["run"]


def alignment_file(directory, species, source, row, audit=False):
    if audit:   # b6_audit_align.sbatch: BAM=$OUT/$RUN.audit.bam, RUN=est_primary for EST
        stems = [audit_run_name(source, row) + ".audit"]
    else:
        stems = [f"{species}.est"] if source == "est" else [f"{species}.protein"] if source == "protein" else [row["run"]]
    exts = (".gff.gz", ".gff") if source == "protein" else (".bam", ".sam", ".sam.gz")
    for stem in stems:
        for ext in exts:
            p = directory / (stem + ext)
            if p.is_file():
                return p
    raise C0Error(f"no {'audit ' if audit else ''}alignment {stems[0]}.* for {row['dataset']}/{row['run']} under {directory}")


def audit_dir(root, species, source, row):
    """b6_audit_align.sbatch: OUT=$OUTROOT/$SP/$DSBASE/$RUN; EST rows use DSBASE=est, RUN=est_primary."""
    if source == "est":
        return Path(root) / species / EST_AUDIT_DATASET / EST_AUDIT_RUN
    return Path(root) / species / Path(row["dataset"]).name / row["run"]


def verify_input(path, done_path):
    """Input hash vs its manifest record (DONE): the recorded md5 of this exact file must match (one read pass)."""
    require(done_path.is_file(), f"missing manifest record {done_path} for {path}")
    done = read_done(done_path)
    key = ("gff_md5" if ".gff" in path.name else "bam_md5" if path.suffix == ".bam" else "sam_md5")
    require(key in done, f"{done_path} records no {key}")
    md5, sha = md5_and_sha256(path)
    require(done[key] == md5, f"hash mismatch: {path} md5 {md5} != {done_path} {key}={done[key]}")
    return done, md5, sha


def load_tool_provenance(paths):
    versions, shas = defaultdict(set), {}
    for p in paths:
        p = Path(p)
        side = Path(str(p) + ".sha256")
        require(p.is_file() and side.is_file(), f"tool PROVENANCE {p} or its .sha256 sidecar missing")
        want = side.read_text().split()[0]
        got = digest(p)
        require(got == want, f"tool PROVENANCE {p} sha256 {got} != sidecar {want}")
        shas[str(p)] = got
        for name, tool in json.loads(p.read_text()).get("tools", {}).items():
            if isinstance(tool, dict) and tool.get("version"):
                versions[name.split("_")[0]].add(tool["version"])
    return versions, shas


def read_header(path, samtools):
    if str(path).endswith(".bam"):
        out = subprocess.run([samtools, "view", "-H", str(path)], capture_output=True, text=True)
        require(out.returncode == 0, f"samtools view -H failed ({out.returncode}): {path}")
        return out.stdout.splitlines()
    lines = []
    with ES.open_text(path) as fh:
        for line in fh:
            if not line.startswith("@"):
                break
            lines.append(line.rstrip("\n"))
    return lines


def minimap2_cl(pg_lines, versions, path):
    """The single minimap2 command line of an alignment header; VN must be a PROVENANCE-recorded version."""
    mm = [line for line in pg_lines if line.startswith("@PG") and ("\tPN:minimap2" in line or "\tID:minimap2" in line)]
    require(mm, f"{path}: no @PG minimap2 header; tool identity cannot be established")
    cls = set()
    for line in mm:
        f = dict(x.split(":", 1) for x in line.rstrip("\n").split("\t")[1:] if ":" in x)
        require(f.get("VN") in versions.get("minimap2", set()),
                f"{path}: minimap2 VN {f.get('VN')!r} not in the PROVENANCE-recorded versions {sorted(versions.get('minimap2', ()))}")
        cls.add(f.get("CL", ""))
    require(len(cls) == 1, f"{path}: {len(cls)} different minimap2 command lines in the header")
    return cls.pop()


def expected_audit_argv(frozen_cl, audit_n):
    """b6_audit_align.sbatch: AUDIT_CMD=${PRIMARY/ --secondary=no / --secondary=yes -N 50 -p 0.5 }; argv[0] (the
    program path) may differ between the two invocations and is not compared."""
    toks = frozen_cl.split()
    require(toks.count("--secondary=no") == 1, f"frozen @PG CL must carry --secondary=no exactly once: {frozen_cl!r}")
    i = toks.index("--secondary=no")
    return toks[1:i] + ["--secondary=yes", "-N", str(audit_n), "-p", "0.5"] + toks[i + 1:]


def check_audit_cl(audit_cl, frozen_cl, audit_n, path):
    require(audit_cl.split()[1:] == expected_audit_argv(frozen_cl, audit_n),
            f"{path}: audit @PG CL must equal the frozen primary CL with only --secondary=no -> "
            f"--secondary=yes -N {audit_n} -p 0.5 (A46 rule 4(b)); audit {audit_cl!r} vs frozen {frozen_cl!r}")


def raw_span(line):
    """1-based [pos, end] of a record parse_sam refuses (empty exon); used only for leakage placements."""
    f = line.split("\t")
    pos, n = int(f[3]), 0
    for k, op in re.findall(r"(\d+)([MIDNSHP=X])", f[5]):
        if op in "MDN=X":
            n += int(k)
    return pos, pos + max(n, 1) - 1


def parse_record(line):
    """(Alignment | None, raw | None, flag, SA tag | None); raw = (flag, chrom, start1, end1, mapq, AS)."""
    try:
        aln = ES.parse_sam(line)
        return aln, None, aln.flag, aln.tags.get("SA")
    except ES.EmptyExonError:
        f = line.rstrip("\n").split("\t")
        s, e = raw_span(line)
        tags = {t.split(":", 2)[0]: t.split(":", 2)[2] for t in f[11:] if t.count(":") >= 2}
        return None, (int(f[1]), f[2], s, e, int(f[4]), tags.get("AS")), int(f[1]), tags.get("SA")


def kind_of(flag):
    return "unmapped" if flag & 4 else "secondary" if flag & 0x100 else "supplementary" if flag & 0x800 else "primary"


def stream_groups(path, samtools, qc):
    """One pass over a --secondary=no alignment -> (read, [(record_index, Alignment | None, raw | None)]) per read.

    A record without SA tag that is not supplementary is a complete read group and is yielded at once (constant
    memory).  A chimeric read (SA tag / supplementary flag) is held until all 1 + |SA| parts are seen; it is
    rejected as supplementary_or_chimeric by 59 regardless.  Duplicate groups are refused later in SQL."""
    pending = {}
    for i, line in enumerate(ES.sam_lines(str(path), samtools)):
        if line.startswith("@") or not line.strip():
            continue
        aln, raw, flag, sa = parse_record(line)
        require(not flag & 0x100, f"{path}: frozen alignment carries a secondary record ({line.split(chr(9), 1)[0]}); "
                                  "it must be the --secondary=no primary alignment")
        read = aln.read if aln else line.split("\t", 1)[0]
        if not sa and not flag & 0x800:
            yield read, [(i, aln, raw)]
            continue
        parts = 1 + len([x for x in (sa or "").split(";") if x])
        entry = pending.setdefault(read, [parts, []])
        entry[0] = max(entry[0], parts)
        entry[1].append((i, aln, raw))
        if len(entry[1]) >= entry[0]:
            del pending[read]
            yield read, entry[1]
    for read in sorted(pending):
        qc["chimeric_groups_incomplete"] += 1
        yield read, pending[read][1]


# ----------------------------------------------------------------------------------------------- molecule units
_ACC = re.compile(r"(.+?)(?:_part(\d+))?")
_VER = re.compile(r"(.+?)(?:\.(\d+))?")


def parse_accession(value):
    m = _ACC.fullmatch(value)
    base, part = m[1], (int(m[2]) if m[2] else None)
    v = _VER.fullmatch(base)
    return v[1], (int(v[2]) if v[2] else None), part


def est_units(members):
    """A11 (+ A21 parts) over compact tuples (member_id, read, stem, library, clone, signature | None):
    accession stem (all versions, _partN pieces), normalised clone within a library, otherwise same library +
    identical exact alignment signature.  Returns member_id -> (molecule_id, rule).

    molecule_id = sha256(json.dumps(sorted(set(QNAMEs of every member)), separators=(',', ':'))) -- the formula of
    mvp_build_c0lite and of b6_hsp_project.MOLECULE_ID_RULE with member_uid = QNAME (b6_est_molecule_map.py)."""
    n = len(members)
    parent = list(range(n))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first = {}
    for i, (_, _, stem, library, clone, signature) in enumerate(members):
        keys = [("acc", stem)]
        if clone:
            keys.append(("clone", library, clone))
        elif signature is not None:
            keys.append(("sig", library, signature))
        for key in keys:
            j = first.setdefault(key, i)
            a, b = root(i), root(j)
            if a != b:
                parent[max(a, b)] = min(a, b)
    del first
    comps = defaultdict(list)
    for i in range(n):
        comps[root(i)].append(i)
    out = {}
    for ids in comps.values():
        names = sorted({members[i][1] for i in ids})
        molecule = hashlib.sha256(json.dumps(names, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
        for i in ids:
            _, _, _, _, clone, signature = members[i]
            out[members[i][0]] = (molecule, "clone" if clone else "signature" if signature is not None else "accession")
    return out


# ----------------------------------------------------------------------------------------------- staging
def _t(sqltype):
    return sqltype.split()[0]


DERIVED = {"molecule_id", "read_name", "arm", "complete_5p_code", "complete_3p_code", "read_internal_code", "block_index"}
# final data table -> (stage table, drop the rows of superseded EST versions)
STAGE_OF = {"evidence_alignment": ("ea", False), "aligned_block": ("blk", True), "junction_observation": ("jobs", True),
            "partial_chain": ("pc", True), "partial_chain_intron": ("pci", True), "chain_junction": ("cj", True)}


def _stage_cols(table):
    own = [(c, _t(t)) for c, t in S.DATA_TABLES[table][0] if c not in DERIVED]
    return own if any(c == "member_id" for c, _ in own) else [("member_id", "VARCHAR")] + own


STAGE_TABLES = {
    "run": [("run_idx", "INTEGER"), ("dataset", "VARCHAR"), ("run", "VARCHAR"), ("role", "VARCHAR"), ("training", "BOOLEAN")],
    "member": [("member_id", "VARCHAR"), ("run_idx", "INTEGER"), ("read", "VARCHAR"), ("accession", "VARCHAR"),
               ("stem", "VARCHAR"), ("version", "INTEGER"), ("part", "INTEGER"), ("clone", "VARCHAR"),
               ("bioproject", "VARCHAR"), ("library", "VARCHAR"), ("genotype", "VARCHAR"), ("training", "BOOLEAN"),
               ("accepted", "BOOLEAN"), ("signature", "VARCHAR"), ("unit_kind", "VARCHAR"), ("unit_key", "VARCHAR"),
               ("unit_rule", "VARCHAR"), ("aln_start", "BIGINT"), ("aln_end", "BIGINT")],
    "meta": [("run_idx", "INTEGER"), ("read", "VARCHAR")] + [
        (f, "BOOLEAN" if f in BOOL_FIELDS else "BIGINT" if f in INT_FIELDS else "VARCHAR") for f in META_FIELDS],
    "apl": [("run_idx", "INTEGER"), ("read", "VARCHAR"), ("kind", "VARCHAR"), ("chrom", "VARCHAR"), ("start0", "BIGINT"),
            ("end0", "BIGINT"), ("mapq", "INTEGER"), ("as_score", "VARCHAR")],
    "aread": [("run_idx", "INTEGER"), ("read", "VARCHAR"), ("n_secondary", "BIGINT"), ("equal_best", "BOOLEAN")],
    "pl": [("member_id", "VARCHAR"), ("origin", "VARCHAR"), ("kind", "VARCHAR"), ("chrom", "VARCHAR"), ("start0", "BIGINT"),
           ("end0", "BIGINT"), ("mapq", "INTEGER")],
    "sup": [("member_id", "VARCHAR"), ("junction_key", "VARCHAR"), ("chrom", "VARCHAR"), ("strand", "VARCHAR"),
            ("start0", "BIGINT"), ("end0", "BIGINT"), ("donor0", "BIGINT"), ("acceptor0", "BIGINT")],
    "unit": [("member_id", "VARCHAR"), ("molecule_id", "VARCHAR"), ("unit_rule", "VARCHAR"), ("superseded", "BOOLEAN")],
    "dec": [(c, _t(t)) for c, t in S.QC_TABLES["molecule_exclusion"][0] if c not in ("species", "source")]
    + [("cause", "VARCHAR")],
    "prec": [("gidx", "BIGINT"), ("protein", "VARCHAR"), ("organism", "VARCHAR"), ("chrom", "VARCHAR"), ("strand", "VARCHAR"),
             ("score", "BIGINT"), ("start1", "BIGINT"), ("end1", "BIGINT"), ("cds", "VARCHAR"), ("introns", "VARCHAR"),
             ("ok", "BOOLEAN"), ("reasons", "VARCHAR"), ("identity", "DOUBLE"), ("coverage", "DOUBLE")],
    "hspout": [(c, _t(t)) for c, t in S.table_columns("est_protein_hsp")],
}
STAGE_TABLES.update({stage: _stage_cols(table) for table, (stage, _) in STAGE_OF.items()})


class Stage:
    """Row buffers flushed in CHUNK-row column batches to tables of the attached scratch DB ``stg``."""

    def __init__(self, con):
        self.con, self.cols, self.buf = con, {}, {}
        for name, cols in STAGE_TABLES.items():
            con.execute(f"CREATE TABLE stg.{name} (" + ", ".join(f"{c} {t}" for c, t in cols) + ")")
            self.cols[name] = [c for c, _ in cols]
            self.buf[name] = []

    def add(self, name, **row):
        buf = self.buf[name]
        buf.append(tuple(row[c] for c in self.cols[name]))     # KeyError = a missing column (fail closed)
        if len(buf) >= CHUNK:
            self.flush(name)

    def flush(self, name=None):
        import pyarrow as pa

        for n in ([name] if name else list(self.buf)):
            rows = self.buf[n]
            if not rows:
                continue
            data = pa.table({c: pa.array([r[i] for r in rows]) for i, c in enumerate(self.cols[n])})
            self.buf[n] = []
            self.con.register("_stg_chunk", data)
            try:
                self.con.execute(f"INSERT INTO stg.{n} SELECT * FROM _stg_chunk")
            finally:
                self.con.unregister("_stg_chunk")


def fetch_stream(cursor, sql, params=()):
    rel = cursor.execute(sql, list(params))
    while True:
        rows = rel.fetchmany(CHUNK)
        if not rows:
            return
        yield from rows


# ----------------------------------------------------------------------------------------------- read metadata
def load_metadata(st, run_idx, directory):
    """Stage c0_metadata.tsv (sha256 sidecar verified).  The optional '*' row must be the first data row; its
    explicit values fill a read's empty cells.  Unknown columns fail closed; missing reads fail at lookup."""
    path = directory / "c0_metadata.tsv"
    side = directory / "c0_metadata.tsv.sha256"
    require(path.is_file() and side.is_file(), f"missing {path} or its .sha256 (A11/A16 read metadata)")
    sha = digest(path)
    require(side.read_text().split()[0] == sha, f"{path} sha256 differs from its sidecar")
    default = {}
    n = 0
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        require(header and header[0] == "read", f"{path}: first column must be 'read'")
        unknown = [c for c in header[1:] if c not in META_FIELDS]
        require(not unknown, f"{path}: unknown metadata column(s) {unknown}")
        for lineno, line in enumerate(fh, 2):
            if not line.strip():
                continue
            vals = line.rstrip("\n").split("\t")
            rec = {}
            for k, v in zip(header[1:], vals[1:]):
                if v == "":
                    continue
                if k in BOOL_FIELDS:
                    require(v in ("true", "false"), f"{path}:{lineno}: {k} must be true/false, got {v!r}")
                    rec[k] = v == "true"
                elif k in INT_FIELDS:
                    require(re.fullmatch(r"-?\d+", v) is not None, f"{path}:{lineno}: {k} must be an integer, got {v!r}")
                    rec[k] = int(v)
                else:
                    rec[k] = v
            if vals[0] == "*":
                require(n == 0 and not default, f"{path}:{lineno}: the '*' run-level row must be the first data row")
                default = rec
                continue
            n += 1
            merged = dict(default)
            merged.update(rec)
            st.add("meta", run_idx=run_idx, read=vals[0], **{f: merged.get(f) for f in META_FIELDS})
    st.flush("meta")
    dup = st.con.execute("SELECT read FROM stg.meta WHERE run_idx = ? GROUP BY read HAVING count(*) > 1 LIMIT 3",
                         [run_idx]).fetchall()
    require(not dup, f"{path}: duplicate read(s) {[d[0] for d in dup]}")
    return sha, n


def lookup(con, run_idx, reads):
    import pyarrow as pa

    con.register("_q", pa.table({"read": pa.array(reads, pa.string())}))
    try:
        cols = ", ".join(f"m.{f}" for f in META_FIELDS)
        rows = con.execute(f"""SELECT q.read, m.read IS NOT NULL, a.read IS NOT NULL, a.equal_best, {cols}
                               FROM _q q LEFT JOIN stg.meta m ON m.run_idx = ? AND m.read = q.read
                               LEFT JOIN stg.aread a ON a.run_idx = ? AND a.read = q.read""",
                           [run_idx, run_idx]).fetchall()
    finally:
        con.unregister("_q")
    return {r[0]: r[1:] for r in rows}


# ----------------------------------------------------------------------------------------------- audit
def stage_audit(st, run_idx, path, samtools):
    """Audit placements (primary/supplementary/secondary; unmapped kept for coverage) and per-read facts:
    n_secondary and A18.5 equal-best (a secondary with the primary's AS at a non-overlapping locus)."""
    for line in ES.sam_lines(str(path), samtools):
        if line.startswith("@") or not line.strip():
            continue
        aln, raw, flag, _ = parse_record(line)
        read = aln.read if aln else line.split("\t", 1)[0]
        kind = kind_of(flag)
        if kind == "unmapped":
            st.add("apl", run_idx=run_idx, read=read, kind=kind, chrom=None, start0=None, end0=None, mapq=None, as_score=None)
        elif aln:
            st.add("apl", run_idx=run_idx, read=read, kind=kind, chrom=aln.contig, start0=aln.start - 1, end0=aln.end,
                   mapq=aln.mapq, as_score=aln.tags.get("AS"))
        else:   # empty exon: a placement, but no AS for the equal-best test
            st.add("apl", run_idx=run_idx, read=read, kind=kind, chrom=raw[1], start0=raw[2] - 1, end0=raw[3], mapq=raw[4],
                   as_score=None)
    st.flush("apl")
    st.con.execute("""
        INSERT INTO stg.aread
        WITH p AS (SELECT read, any_value(chrom) AS chrom, any_value(start0) AS s, any_value(end0) AS e,
                          any_value(as_score) AS a, count(*) AS n
                   FROM stg.apl WHERE run_idx = $1 AND kind = 'primary' AND as_score IS NOT NULL GROUP BY read),
             eb AS (SELECT DISTINCT p.read FROM p JOIN stg.apl s ON s.run_idx = $1 AND s.read = p.read
                    AND s.kind = 'secondary' AND s.as_score = p.a AND NOT (s.chrom = p.chrom AND s.start0 < p.e AND s.end0 > p.s)
                    WHERE p.n = 1)
        SELECT $1, r.read, r.n_secondary, eb.read IS NOT NULL
        FROM (SELECT read, count(*) FILTER (WHERE kind = 'secondary') AS n_secondary FROM stg.apl WHERE run_idx = $1
              GROUP BY read) r LEFT JOIN eb ON eb.read = r.read""", [run_idx])


# ----------------------------------------------------------------------------------------------- read sources
def load_read_run(ctx, st, run_idx, row, args, versions):
    source, species = ctx["source"], ctx["species"]
    directory = alignment_dir(args.align_root, species, source, row)
    path = alignment_file(directory, species, source, row)
    done, md5, sha = verify_input(path, directory / "DONE")
    if source == "est":
        require("arm" in done, f"{directory / 'DONE'} records no arm= (est_align.sbatch writes it); fail closed")
        require(done["arm"] == "primary",
                f"{path}: arm={done['arm']!r}; only the A37 primary arm (>= 100 nt) is C0 input (A46 decision 3)")
    uf = done.get("uf", "none").lower() in ("-uf", "true", "yes", "1")
    meta_sha, n_meta = load_metadata(st, run_idx, directory)
    frozen_cl = minimap2_cl(read_header(path, args.samtools), ctx["versions"], path)
    require("--secondary=no" in frozen_cl.split(), f"{path}: frozen alignment @PG CL lacks --secondary=no")
    ctx["inputs"][f"alignment:{row['dataset']}/{row['run']}"] = {"path": str(path), "md5": md5, "sha256": sha,
                                                                 "metadata_sha256": meta_sha, "minimap2_cl": frozen_cl}
    training = ROLE_MAP[row["role"]] == "training_eligible"
    if training:
        adir = audit_dir(args.audit_root, species, source, row)
        apath = alignment_file(adir, species, source, row, audit=True)
        adone, amd5, asha = verify_input(apath, adir / "DONE")
        for key, want in (("source", source), ("run", audit_run_name(source, row))):
            require(adone.get(key) == want, f"{adir / 'DONE'}: {key}={adone.get(key)!r}, expected {want!r} "
                                            "(b6_audit_align.sbatch completion marker)")
        # N is taken from the audit alignment itself, never from a DONE key (b6_audit_align.sbatch writes none): its
        # @PG CL must carry exactly '-N <--audit-n>' (check_audit_cl), and a missing/other N fails closed.
        audit_cl = minimap2_cl(read_header(apath, args.samtools), ctx["versions"], apath)
        check_audit_cl(audit_cl, frozen_cl, args.audit_n, apath)          # defence in depth over the sbatch gate
        stage_audit(st, run_idx, apath, args.samtools)
        ctx["inputs"][f"audit:{row['dataset']}/{row['run']}"] = {"path": str(apath), "md5": amd5, "sha256": asha,
                                                                 "minimap2_cl": audit_cl}
    run = {"idx": run_idx, "dataset": row["dataset"], "run": row["run"], "role_raw": row["role"], "training": training,
           "uf": uf, "spec": {"species": species, "assembly": ctx["assembly"], "source": ES_SOURCE[source],
                              "dataset": row["dataset"], "run": row["run"], "arm": "primary", "uf": uf,
                              "annotation_independence": "n/a", "model_independent": True}}
    st.add("run", run_idx=run_idx, dataset=row["dataset"], run=row["run"], role=ROLE_MAP[row["role"]], training=training)
    missing = {"meta": [], "audit": []}
    batch = []
    for read, records in stream_groups(path, args.samtools, ctx["qc"]):
        batch.append((read, records))
        if len(batch) >= BATCH:
            process_batch(ctx, st, run, batch, missing)
            batch = []
    process_batch(ctx, st, run, batch, missing)
    require(not missing["meta"], f"{directory / 'c0_metadata.tsv'} lacks {len(missing['meta'])} read(s) of {path}, e.g. "
                                 f"{sorted(missing['meta'])[:3]} (no default is applied; fail closed)")
    require(not missing["audit"], f"audit alignment lacks {len(missing['audit'])} member(s) of the frozen alignment, e.g. "
                                  f"{sorted(missing['audit'])[:3]}; the closure would be incomplete")


def process_batch(ctx, st, run, batch, missing):
    if not batch:
        return
    info = lookup(st.con, run["idx"], [r for r, _ in batch])
    for read, records in batch:
        has_meta, has_audit, equal_best, *vals = info[read]
        if not has_meta:
            missing["meta"].append(read)
            continue
        if run["training"] and not has_audit:
            missing["audit"].append(read)
            continue
        meta = {k: v for k, v in zip(META_FIELDS, vals) if v is not None}
        process_read(ctx, st, run, read, records, meta, bool(equal_best))


def process_read(ctx, st, run, read, records, meta, equal_best):
    source, species, qc = ctx["source"], ctx["species"], ctx["qc"]
    member_id = sid("member", species, source, run["dataset"], run["run"], read)
    qc["reads_in"] += 1
    if run["training"] and equal_best:
        meta["mapping_ambiguous"] = True
        qc["audit_equal_best_reads"] += 1
    obs = reason = None
    if any(aln is None for _, aln, _ in records):
        reason = "empty_exon"
    else:
        try:
            obs, reason = ES.observation_from_records([a for _, a, _ in records], run["spec"], {read: meta}, ctx["genome"],
                                                      {"role": run["role_raw"]})
        except ES.EvidenceError as exc:
            raise C0Error(f"{run['dataset']}/{run['run']}/{read}: {exc}") from exc
    qc["accepted_reads" if obs is not None else "rejected:" + reason] += 1
    library = meta.get("library")
    require(library and library != "unknown", f"{run['dataset']}/{run['run']}/{read}: missing biological library identity")
    require(source == "est" or S.is_longread_library(library),
            f"{run['dataset']}/{run['run']}/{read}: long-read library {library!r} is not an A49.1 key (ENA experiment "
            "accession or '<BioSample>:<library_name>'); metadata written under the old run-as-library rule must be "
            "regenerated with b6_c0_metadata.py --library")
    genotype = meta.get("genotype_stratum")
    require(genotype in S.GENOTYPE_STRATA[:4], f"{read}: missing or invalid genotype stratum {genotype!r} (no default)")
    if source == "est":
        require(meta.get("accession"), f"{run['dataset']}/{read}: missing EST accession (no default)")
        accession = meta["accession"]
    else:
        accession = meta.get("accession") or read       # a long read has no accession: its read name is its identity
    stem, version, part = parse_accession(accession)
    clone = ES.normalize_clone(meta.get("clone", "")) if source == "est" else ""
    signature = unit_kind = unit_key = unit_rule = aln_start = aln_end = None
    if obs is not None:
        if source == "est":
            signature = json.dumps([obs.alignment.contig, obs.strand, obs.alignment.blocks], separators=(",", ":"))
        else:
            try:
                ES.assign_molecules([obs])      # 59's own unit rule and fail-closed metadata checks, per read
            except ES.EvidenceError as exc:
                raise C0Error(f"A16 molecule units: {exc}") from exc
            md = obs.metadata
            if source == "ont" and md.get("umi_status") != "present" and md.get("protocol") != "direct_RNA":
                unit_kind, unit_rule = "pcr", "pcr_equivalence"      # grouped per library/chain after all runs
                # (one molecule per connected 10-nt component, clique or not: 59.A16_PCR_LINKAGE, decision 9a)
                unit_key = json.dumps([obs.alignment.contig, obs.strand, obs.chain], separators=(",", ":"))
                aln_start, aln_end = obs.alignment.start, obs.alignment.end
            else:
                unit_kind, unit_key = "direct", obs.molecule
                unit_rule = ("source_molecule" if source == "pacbio" else "umi" if md.get("umi_status") == "present"
                             else "direct_rna")
    elif source != "est":
        unit_kind, unit_key, unit_rule = rejected_member_unit(ctx, run, read, records, meta)
        qc["rejected_members_in_a16_unit" if unit_kind else "rejected_members_per_read_unit"] += 1
    else:
        signature = rejected_est_signature(run, read, records, meta)
        qc["rejected_est_members_with_signature" if signature else "rejected_est_members_without_signature"] += 1
    st.add("member", member_id=member_id, run_idx=run["idx"], read=read, accession=accession, stem=stem, version=version,
           part=part, clone=clone, bioproject=meta.get("bioproject"), library=library, genotype=genotype,
           training=run["training"], accepted=obs is not None, signature=signature, unit_kind=unit_kind,
           unit_key=unit_key, unit_rule=unit_rule, aln_start=aln_start, aln_end=aln_end)
    for idx, aln, raw in records:
        flag = aln.flag if aln else raw[0]
        kind = kind_of(flag)
        accepted = obs is not None and aln is obs.alignment
        if aln is None:
            ts, strand = None, "?"
        else:
            ts = aln.tags.get("ts")
            try:
                strand = (obs.strand if accepted else ES.transcript_strand(aln, run["uf"], meta.get("oriented_genomic_strand"))) or "?"
            except ES.EvidenceError:
                strand = "?"
        ident = frac = None
        if aln is not None and kind == "primary" and aln.query_length:
            frac = aln.aligned_query / aln.query_length
            try:
                edits = ES.parse_cs(aln)
                mt = sum(e.length for e in edits if e.op in ":=")
                er = sum(e.length for e in edits if e.op in "*+-")
                ident = mt / (mt + er) if mt + er else None
            except ES.EvidenceError:
                ident = None
        mapped = kind != "unmapped"
        chrom, start0, end0, mapq = ((aln.contig, aln.start - 1, aln.end, aln.mapq) if aln else (raw[1], raw[2] - 1, raw[3], raw[4]))
        st.add("ea", member_id=member_id, alignment_id=sid("alignment", member_id, "frozen", idx), record_index=idx,
               placement_kind=kind, flag=flag, chrom=chrom if mapped else None, start0=start0 if mapped else None,
               end0=end0 if mapped else None, alignment_strand=(("-" if flag & 16 else "+") if mapped else None),
               transcript_strand=strand, ts_tag=ts, mapq=mapq, cigar=(aln.cigar if aln else None),
               query_length=(aln.query_length if aln else None), aligned_query=(aln.aligned_query if aln else None),
               aligned_fraction=frac, identity=ident, accepted=accepted,
               rejection_reason=None if accepted else (reason if kind == "primary" else "not_accepted_primary"),
               mapping_ambiguous=bool(meta.get("mapping_ambiguous")))
        if run["training"] and kind in ("primary", "supplementary"):
            st.add("pl", member_id=member_id, origin="frozen", kind=kind, chrom=chrom, start0=start0, end0=end0, mapq=mapq)
    if obs is None:
        return
    aln = obs.alignment
    alignment_id = sid("alignment", member_id, "frozen", next(i for i, a, _ in records if a is aln))
    strand = obs.strand or "?"
    for bi, (s1, e1) in enumerate(aln.blocks):
        st.add("blk", member_id=member_id, alignment_id=alignment_id, alignment_block_index=bi, chrom=aln.contig,
               start0=s1 - 1, end0=e1, strand=strand)
    # 59 relabels every non-canonical training call 'training_noncanonical'; the specific cause of a call that
    # could never be canonical (ambiguous correction, failed anchor, unknown strand) is kept from the raw call.
    raw_calls = ES.call_junctions(aln, ctx["genome"], obs.strand, ES_SOURCE[source])
    calls = []
    for j, r in zip(obs.junctions, raw_calls):
        status = r.status if (j.status == "training_noncanonical" and r.status in (
            "ambiguous_correction", "anchor_failed", "strand_ambiguous")) else j.status
        calls.append((j.raw, j.corrected, status, j.accepted, j.canonical))
    emit_chain(ctx, st, member_id, alignment_id, 0, aln.contig, strand, aln.start - 1, aln.end, calls, RADIUS[source])


def rejected_est_signature(run, read, records, meta):
    """A11 alignment signature of an EST member whose primary was NOT accepted (A18.5) -- Codex review 5.

    A11 joins clone-less ESTs of one library with an identical exact alignment signature (contig, transcript strand,
    blocks), and rule 4(b) closes a molecule over "any placement of any member": a rejected duplicate (e.g. MAPQ 10)
    whose audit secondary touches held-out sequence must exclude its accepted twin.  The signature is computed from
    the SAME frozen primary record and the SAME 59.transcript_strand call the accepted path uses
    (59.observation_from_records), so an accepted and a rejected twin produce identical keys.  An unmapped member, or
    one without a usable primary (no record without flags 4/0x900), has no signature (None; counted by the caller);
    a strand error fails closed as on the accepted path."""
    primary = next((aln for _, aln, _ in records if aln is not None and not aln.flag & (4 | 0x900)), None)
    if primary is None or not primary.contig or primary.contig == "*" or not primary.blocks:
        return None
    try:
        strand = ES.transcript_strand(primary, run["uf"], meta.get("oriented_genomic_strand"))
    except ES.EvidenceError as exc:
        raise C0Error(f"A11 signature (rejected EST member {run['dataset']}/{read}): {exc}") from exc
    return json.dumps([primary.contig, strand, primary.blocks], separators=(",", ":"))


def rejected_member_unit(ctx, run, read, records, meta):
    """A16 identity of a long-read member whose primary alignment was NOT accepted (A18.5).

    A46 rule 4(b) closes a molecule over "any placement of any member (primary, supplementary, secondary-equivalent,
    mapping_ambiguous)" and rule 3 makes the long-read unit the A16 unit ("PacBio: one source FLNC/ZMW molecule = one
    unit"), so a rejected PacBio read must join its accepted siblings' unit and carry its placements (stg.pl, audit
    secondaries) into the decision -- a per-read unit would let the accepted sibling stay train_track.  The key is
    computed by 59.assign_molecules itself (metadata-only for PacBio; a UMI key includes the transcript strand, read
    here from the primary record's orientation with 59.transcript_strand, the SAME call the accepted path makes).
    Unknown strand is ``None`` here exactly as on the accepted path: 59 keys on ``o.strand`` and hashes ``repr(key)``,
    so ``None`` and ``'?'`` would be two molecules (fix round 5, finding 1); ``'?'`` is a schema spelling and is written
    only to evidence_alignment.transcript_strand.  Direct RNA is one read = one unit already.  The ONT PCR-equivalence
    unit needs the corrected intron chain of an ACCEPTED alignment, so a rejected non-UMI cDNA read keeps a per-read
    unit; the caller counts both outcomes (qc rejected_members_*)."""
    source = ctx["source"]
    if source == "pacbio":
        rule = "source_molecule"
    elif meta.get("umi_status") == "present":
        rule = "umi"
    else:
        return None, None, None
    strand = None
    primary = next((aln for _, aln, _ in records if aln is not None and kind_of(aln.flag) == "primary"), None)
    if primary is not None:
        try:
            strand = ES.transcript_strand(primary, run["uf"], meta.get("oriented_genomic_strand"))
        except ES.EvidenceError as exc:
            raise C0Error(f"A16 molecule units (rejected member {run['dataset']}/{run['run']}/{read}): {exc}") from exc
    o = SimpleNamespace(species=ctx["species"], assembly=ctx["assembly"], source=ES_SOURCE[source], library=meta.get("library"),
                        run=run["run"], strand=strand, chain=(), molecule="", metadata=dict(meta),
                        alignment=SimpleNamespace(read=read, contig=None, start=None, end=None, blocks=()))
    try:
        ES.assign_molecules([o])            # the same key formula as the accepted path, fail-closed metadata checks
    except ES.EvidenceError as exc:
        raise C0Error(f"A16 molecule units (rejected member {run['dataset']}/{run['run']}/{read}): {exc}") from exc
    return "direct", o.molecule, rule


def emit_chain(ctx, st, member_id, alignment_id, rank, chrom, strand, start0, end0, junctions, radius):
    """junctions: [(raw 1-based incl, corrected 1-based incl | None, status, accepted, canonical)] in genomic order."""
    species = ctx["species"]
    chain_id = sid("chain", alignment_id)
    n = len(junctions)
    keyed = []
    for i, (raw, corr, status, accepted, canonical) in enumerate(junctions):
        rs0, re0 = raw[0] - 1, raw[1]                      # the single 1-based -> 0-based conversion
        cs0, ce0 = ((corr[0] - 1, corr[1]) if corr is not None else (None, None))
        rd, ra = boundary(strand, rs0, re0)
        d, a = boundary(strand, cs0, ce0) if corr is not None else (None, None)
        jkey = junction_key(species, chrom, strand, cs0, ce0) if accepted and corr is not None and strand in "+-" else None
        obs_id = sid("jobs", alignment_id, i)
        st.add("jobs", member_id=member_id, observation_id=obs_id, alignment_id=alignment_id, intron_index=i, chrom=chrom,
               strand=strand, raw_start0=rs0, raw_end0=re0, raw_donor0=rd, raw_acceptor0=ra, start0=cs0, end0=ce0,
               donor0=d, acceptor0=a, correction_status=status, correction_radius=radius,
               accepted=bool(accepted and jkey is not None), canonical=bool(canonical), junction_key=jkey)
        tx = (n - 1 - i) if strand == "-" else i
        ps0, pe0 = (cs0, ce0) if corr is not None else (rs0, re0)
        pd, pa = boundary(strand, ps0, pe0)
        st.add("pci", member_id=member_id, chain_id=chain_id, intron_index=i, tx_order=tx, observation_id=obs_id,
               chrom=chrom, strand=strand, start0=ps0, end0=pe0, donor0=pd, acceptor0=pa, correction_status=status,
               accepted=bool(accepted and jkey is not None))
        keyed.append((ps0, pe0))
        if jkey is not None:
            st.add("cj", member_id=member_id, chain_id=chain_id, tx_order=tx, junction_key=jkey)
            st.add("sup", member_id=member_id, junction_key=jkey, chrom=chrom, strand=strand, start0=cs0, end0=ce0,
                   donor0=d, acceptor0=a)
    st.add("pc", chain_id=chain_id, alignment_id=alignment_id, member_id=member_id, placement_rank=rank, chrom=chrom,
           strand=strand, start0=start0, end0=end0, n_introns=n, intron_chain_key=sid("chain_key", species, strand, keyed),
           all_junctions_accepted=all(bool(j[3]) for j in junctions))


def run_library_qc(ctx, cursor):
    """Information counters (A11 library vs sequencing run): libraries, libraries spanning several runs and molecules
    whose members come from several runs of one library.  Set-based SQL over the staged tables."""
    qc = ctx["qc"]
    qc["libraries"], qc["libraries_spanning_runs"] = cursor.execute("""
        SELECT count(*), count(*) FILTER (WHERE n_runs > 1)
        FROM (SELECT library, count(DISTINCT run_idx) AS n_runs FROM stg.member GROUP BY library)""").fetchone()
    qc["molecules_spanning_runs"] = cursor.execute("""
        SELECT count(*) FROM (SELECT u.molecule_id FROM stg.unit u JOIN stg.member m USING (member_id)
                              GROUP BY u.molecule_id HAVING count(DISTINCT m.run_idx) > 1)""").fetchone()[0]


def assign_read_units(ctx, st, cursor):
    """stg.unit for read sources.  EST: newest version per accession stem, then est_units (A11 + A21).
    ONT/PacBio: 59.assign_molecules; PCR-equivalence groups (library, contig, strand, chain) are streamed one at a
    time into 59.assign_molecules, so its 10-nt linkage and unit hash are applied unchanged: every connected component
    is one molecule, clique or not (59.A16_PCR_LINKAGE, author decision 2026-09-26, decision 9a), and the non-clique
    components collapsed are counted in qc pcr_nonclique_components / pcr_nonclique_reads (0 when none).  Every long-read
    unit key is 59's (species, assembly, source, LIBRARY, ...) -- the metadata library (the producer's --library =
    the biological preparation, A11), never the run -- so members of one ZMW / UMI / PCR-equivalence unit that were
    sequenced in two runs of one library form ONE molecule (fix round 6, finding 1); qc counts the libraries that
    span several runs and the molecules whose members come from several runs."""
    species, source = ctx["species"], ctx["source"]
    if source == "est":
        rows = list(fetch_stream(cursor, """
            SELECT member_id, read, stem, library, clone,
                   CASE WHEN superseded THEN NULL ELSE signature END, superseded
            FROM (SELECT *, coalesce(version, 0) < max(coalesce(version, 0)) OVER (PARTITION BY stem) AS superseded
                  FROM stg.member) ORDER BY member_id"""))
        superseded = {r[0] for r in rows if r[6]}
        units = est_units([r[:6] for r in rows])
        del rows
        for member_id, (molecule, rule) in sorted(units.items()):
            st.add("unit", member_id=member_id, molecule_id=molecule, unit_rule=rule, superseded=member_id in superseded)
        return
    group, key = [], None
    qc = ctx["qc"]
    for counter in ES.PCR_NONCLIQUE_COUNTERS:
        qc[counter] += 0                        # recorded as 0 in c0_qc when no non-clique component exists

    def flush_pcr():
        if not group:
            return
        try:
            ES.assign_molecules(group, pcr_stats=qc)
        except ES.EvidenceError as exc:
            raise C0Error(f"A16 molecule units: {exc}") from exc
        for o in group:
            st.add("unit", member_id=o.member_id, molecule_id=sid("molecule", species, source, o.molecule),
                   unit_rule="pcr_equivalence", superseded=False)
        group.clear()

    for member_id, read, library, dataset, run, unit_kind, unit_key, unit_rule, s1, e1 in fetch_stream(cursor, """
            SELECT m.member_id, m.read, m.library, r.dataset, r.run, m.unit_kind, m.unit_key, m.unit_rule,
                   m.aln_start, m.aln_end
            FROM stg.member m JOIN stg.run r USING (run_idx)
            ORDER BY m.unit_kind = 'pcr', m.library, m.unit_key, m.member_id"""):
        if unit_kind == "pcr":
            if key != (library, unit_key):
                flush_pcr()
                key = (library, unit_key)
            contig, strand, chain = json.loads(unit_key)
            group.append(SimpleNamespace(
                member_id=member_id, species=species, assembly=ctx["assembly"], source=ES_SOURCE[source], library=library,
                dataset=dataset, run=run, strand=strand, chain=tuple(tuple(c) for c in chain), molecule="",
                metadata={"protocol": "cDNA", "umi_status": "absent"},
                alignment=SimpleNamespace(read=read, contig=contig, start=s1, end=e1)))
        elif unit_kind == "direct":
            st.add("unit", member_id=member_id, molecule_id=sid("molecule", species, source, unit_key), unit_rule=unit_rule,
                   superseded=False)
        else:
            st.add("unit", member_id=member_id, molecule_id=sid("molecule", species, source, dataset, run, read),
                   unit_rule="read", superseded=False)
    flush_pcr()


# ----------------------------------------------------------------------------------------------- protein
def parse_miniprot(path):
    """Every mRNA alignment of the miniprot GFF with its A19.2 verdict (protein_accept.judge, unchanged); streamed."""
    paf, cur, cds = None, None, []

    def record():
        attrs, score, chrom, strand, s1, e1 = cur
        prot = (attrs.get("Target", "").split() or [""])[0]
        require(paf is not None and paf["qname"] == prot, f"{attrs.get('ID')}: no matching ##PAF record")
        ok, reasons, introns, identity, coverage = PA.judge(paf, attrs, cds)
        return {"protein": prot, "organism": PA.organism_of(prot), "chrom": chrom, "strand": strand, "score": score,
                "start1": s1, "end1": e1, "cds": sorted(cds), "introns": introns, "ok": ok, "reasons": reasons,
                "identity": identity, "coverage": coverage, "id": attrs.get("ID", "")}

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("##PAF\t"):
                if cur is not None:
                    yield record()
                    cur, cds = None, []
                paf = PA.parse_paf(line)
                continue
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            if f[2] == "mRNA":
                if cur is not None:
                    yield record()
                cur, cds = (dict(PA.ATTR.findall(f[8])), int(f[5]), f[0], f[6], int(f[3]), int(f[4])), []
            elif f[2] == "CDS" and cur is not None:
                cds.append((int(f[3]), int(f[4])))
    if cur is not None:
        yield record()


class A19:
    """A19 by NCBI lineage: b6_hsp_project.is_same_species_or_descendant over the frozen lineage table registered with
    b6_hsp_project.use_frozen_lineage (part B's logic, imported, not copied); unknown taxa fail closed."""

    def __init__(self, lineage_path, species):
        self.hp = _load("b6_hsp_project", _SCRIPTS / "evidence" / "b6_hsp_project.py")
        require(species in self.hp.SPECIES, f"A19 lineage check: {species} has no frozen root taxid in est_blastx.SPECIES")
        self.lineage_sha256 = self.hp.use_frozen_lineage(lineage_path)
        self.species, self.ok = species, set()

    def taxid_of_protein(self, protein_id):
        try:
            return self.hp.taxonomy.taxid_of_header(protein_id)
        except ValueError as exc:
            raise C0Error(f"A19: {exc}") from exc

    def check(self, taxid, what):
        if taxid in self.ok:
            return
        try:
            same = self.hp.is_same_species_or_descendant(taxid, self.species)
        except ValueError as exc:
            raise C0Error(f"{what}: A19 lineage check failed: {exc}") from exc
        require(not same, f"{what}: same-species protein (A19): taxid {taxid} is {self.species} or below it")
        self.ok.add(taxid)


def load_protein(ctx, st, cursor, rows, args):
    species, qc = ctx["species"], ctx["qc"]
    require(len(rows) == 1, f"protein source needs exactly one role row, got {len(rows)}")
    row = rows[0]
    directory = alignment_dir(args.align_root, species, "protein", row)
    path = alignment_file(directory, species, "protein", row)
    _, md5, sha = verify_input(path, directory / "DONE")
    ctx["inputs"]["alignment:protein"] = {"path": str(path), "md5": md5, "sha256": sha}
    a19 = ctx["a19"]
    for gidx, r in enumerate(parse_miniprot(path)):
        a19.check(a19.taxid_of_protein(r["protein"]), f"miniprot query {r['protein']}")
        st.add("prec", gidx=gidx, protein=r["protein"], organism=r["organism"], chrom=r["chrom"], strand=r["strand"],
               score=r["score"], start1=r["start1"], end1=r["end1"], cds=json.dumps(r["cds"]),
               introns=json.dumps(r["introns"]), ok=bool(r["ok"]), reasons=",".join(r["reasons"]), identity=r["identity"],
               coverage=r["coverage"])
    st.flush("prec")
    training = ROLE_MAP[row["role"]] == "training_eligible"
    st.add("run", run_idx=0, dataset=row["dataset"], run=row["run"], role=ROLE_MAP[row["role"]], training=training)
    st.con.execute("""CREATE TABLE stg.prank AS
        SELECT p.*, row_number() OVER (PARTITION BY protein ORDER BY gidx) - 1 AS i,
               row_number() OVER (PARTITION BY protein ORDER BY score DESC, gidx) - 1 AS rank,
               sum(CASE WHEN score = mx THEN 1 ELSE 0 END) OVER (PARTITION BY protein) > 1 AS ambiguous
        FROM (SELECT *, max(score) OVER (PARTITION BY protein) AS mx FROM stg.prec) p""")
    for prot, organism in fetch_stream(cursor, "SELECT protein, any_value(organism) FROM stg.prec GROUP BY protein ORDER BY protein"):
        member_id = sid("member", species, "protein", row["dataset"], row["run"], prot)
        st.add("member", member_id=member_id, run_idx=0, read=prot, accession=prot, stem=prot, version=None, part=None,
               clone="", bioproject=None, library=organism, genotype="not_applicable", training=training, accepted=True,
               signature=None, unit_kind=None, unit_key=None, unit_rule="protein_query", aln_start=None, aln_end=None)
        st.add("unit", member_id=member_id, molecule_id=sid("molecule", species, "protein", prot), unit_rule="protein_query",
               superseded=False)
    if training:
        for prot, chrom, s1, e1 in fetch_stream(cursor, "SELECT protein, chrom, start1, end1 FROM stg.prec ORDER BY protein, gidx"):
            st.add("pl", member_id=sid("member", species, "protein", row["dataset"], row["run"], prot), origin="frozen",
                   kind="primary", chrom=chrom, start0=s1 - 1, end0=e1, mapq=None)
    qc["protein_alignments_at_miniprot_cap_21"] = cursor.execute(
        "SELECT count(*) FROM (SELECT protein FROM stg.prec GROUP BY protein HAVING count(*) >= ?)",
        [MINIPROT_MAX_PLACEMENTS]).fetchone()[0]
    chroms = [c for (c,) in cursor.execute("SELECT DISTINCT chrom FROM stg.prank ORDER BY chrom").fetchall()]
    for chrom in chroms:                                        # rows are emitted chromosome by chromosome
        for (prot, i, rank, ambiguous, strand, s1, e1, cds, introns, ok, reasons, identity, coverage) in fetch_stream(
                cursor, """SELECT protein, i, rank, ambiguous, strand, start1, end1, cds, introns, ok, reasons, identity,
                                  coverage FROM stg.prank WHERE chrom = ? ORDER BY protein, i""", [chrom]):
            member_id = sid("member", species, "protein", row["dataset"], row["run"], prot)
            qc["reads_in"] += 1
            accepted = ok and not ambiguous
            reason = None if accepted else ("mapping_ambiguous_family_only" if ok else reasons)
            qc["accepted_reads" if accepted else "rejected:" + reason.split(",")[0]] += 1
            alignment_id = sid("alignment", member_id, "frozen", i)
            st.add("ea", member_id=member_id, alignment_id=alignment_id, record_index=i,
                   placement_kind="primary" if rank == 0 else "secondary", flag=None, chrom=chrom, start0=s1 - 1, end0=e1,
                   alignment_strand=strand, transcript_strand=strand, ts_tag=None, mapq=None, cigar=None, query_length=None,
                   aligned_query=None, aligned_fraction=coverage, identity=identity, accepted=accepted,
                   rejection_reason=reason, mapping_ambiguous=bool(ambiguous))
            if not accepted:
                continue
            for bi, (cs1, ce1) in enumerate(json.loads(cds)):
                st.add("blk", member_id=member_id, alignment_id=alignment_id, alignment_block_index=bi, chrom=chrom,
                       start0=cs1 - 1, end0=ce1, strand=strand)
            juncs = [((gs, ge), (gs, ge), "uncorrected", (d, a) in PA.CANONICAL, (d, a) in PA.CANONICAL)
                     for gs, ge, d, a in json.loads(introns)]
            emit_chain(ctx, st, member_id, alignment_id, rank, chrom, strand, s1 - 1, e1, juncs, 0)
    return row


# ----------------------------------------------------------------------------------------------- route-B HSP
def open_published_c0(path, species, source):
    """A published per-species x source C0 DB: exclusive PROVENANCE output_sha256, schema, c0_meta, leakage 0."""
    import duckdb

    path = Path(path).absolute()
    require(path.is_file() and not Path(str(path) + ".wal").exists(), f"{path}: missing or has a WAL")
    prov = Path(str(path) + ".PROVENANCE.txt")
    require(prov.is_file(), f"{prov} missing (only a published C0 DB is accepted)")
    recorded = read_done(prov).get("output_sha256")
    sha = digest(path)
    require(recorded == sha, f"{path}: sha256 {sha} != its PROVENANCE output_sha256 {recorded}")
    con = duckdb.connect(str(path), read_only=True)
    try:
        meta = S.verify_schema(con)
        require((meta.get("species"), meta.get("source")) == (species, source),
                f"{path}: c0_meta says {meta.get('species')}/{meta.get('source')}, expected {species}/{source}")
        leak = con.execute("SELECT value FROM c0_qc WHERE metric = 'leakage_nonzero'").fetchall()
        require(leak == [(0.0,)], f"{path}: leakage QC gate not passed ({leak})")
        provenance = dict(con.execute("SELECT key, value FROM c0_provenance").fetchall())
    finally:
        con.close()
    return sha, provenance


def verify_est_c0(ctx, args):
    """--est-c0 must be the published EST C0 DB of this species, built from the same excluded intervals."""
    est_sha, est_prov = open_published_c0(args.est_c0, ctx["species"], "est")
    require(json.loads(est_prov.get("leakage_intervals_sha256", "null")) == ctx["intervals_sha256"],
            f"--est-c0 {args.est_c0} was built from other excluded intervals than this build")
    ctx["inputs"]["est_c0"] = {"path": str(Path(args.est_c0).absolute()), "sha256": est_sha,
                               "provenance_sha256": est_prov.get("provenance_sha256")}
    inputs = json.loads(est_prov.get("inputs", "{}"))
    aln = {k: v for k, v in inputs.items() if k.startswith("alignment:")}
    require(len(aln) == 1, f"--est-c0 {args.est_c0} records {len(aln)} primary alignment inputs, expected exactly 1")
    ctx["est_c0_alignment_sha256"] = next(iter(aln.values()))["sha256"]


def hsp_provenance_path(hsp):
    """b6_hsp_project.py's default provenance record next to its TSV: <out>.provenance.json."""
    return Path(str(hsp) + ".provenance.json")


def verify_hsp_provenance(ctx, cursor, args):
    """The route-B provenance must be the one the rows cite and must match this build and its EST C0: every row's
    provenance_sha256 = sha256 of the record (as b6_hsp_project.py hashes it), species, phase_convention =
    c0_schema.PHASE_CONVENTION, and alignment_sha256 = the EST C0's frozen primary alignment (the projection must
    have used the very BAM/SAM the EST C0 was built from)."""
    path = hsp_provenance_path(args.hsp)
    require(path.is_file(), f"missing {path} (b6_hsp_project.py provenance record)")
    blob = path.read_bytes()
    want = hashlib.sha256(blob.rstrip(b"\n")).hexdigest()
    cited = [r[0] for r in cursor.execute("SELECT DISTINCT provenance_sha256 FROM stg.hsp ORDER BY 1").fetchall()]
    require(cited in ([want], []), f"--hsp rows cite provenance {cited[:3]} but {path.name} hashes to {want}")
    try:
        prov = json.loads(blob)
    except ValueError as exc:
        raise C0Error(f"{path}: not JSON ({exc})") from None
    require(isinstance(prov, dict), f"{path}: not a JSON object")
    require(prov.get("species") == ctx["species"], f"{path}: species {prov.get('species')!r} != {ctx['species']!r}")
    require(prov.get("phase_convention") == S.PHASE_CONVENTION,
            f"{path}: phase_convention {prov.get('phase_convention')!r} != c0_schema.PHASE_CONVENTION "
            f"{S.PHASE_CONVENTION!r}")
    require(prov.get("alignment_sha256") == ctx["est_c0_alignment_sha256"],
            f"{path}: alignment_sha256 {prov.get('alignment_sha256')!r} != the EST C0 primary alignment sha256 "
            f"{ctx['est_c0_alignment_sha256']} (route B must project through the same frozen alignment)")
    require(prov.get("multi_hsp_rule_applied_by") == MULTI_HSP_APPLIED_BY,
            f"{path}: multi_hsp_rule_applied_by {prov.get('multi_hsp_rule_applied_by')!r} != {MULTI_HSP_APPLIED_BY!r} "
            "(a projection that resolved phase conflicts before protein-query eligibility was known is refused; "
            "re-run b6_hsp_project.py)")
    ctx["qc"]["hsp_provenance_verified"] = 1


def hsp_state_path(hsp):
    """b6_hsp_project.py writes the per-HSP gating sidecar next to its TSV: <out>.molecule_state.tsv."""
    return Path(str(hsp) + ".molecule_state.tsv")


def cross_check_hsp_state(ctx, cursor, args):
    """The sidecar (hsp_id, molecule_id, qname, molecule_state, train_tiles_excluded) is a cross-check only: the
    state used here comes from the EST C0 molecule_member join; any disagreement fails closed."""
    path = hsp_state_path(args.hsp)
    cursor.execute(f"""CREATE TABLE stg.hspstate AS SELECT * FROM read_csv({lit(path)}, delim='\t', header=true,
                       all_varchar=true, quote='', escape='')""")
    cols = [d[0] for d in cursor.execute("SELECT * FROM stg.hspstate LIMIT 0").description]
    require(cols == ["hsp_id", "molecule_id", "qname", "molecule_state", "train_tiles_excluded"],
            f"{path}: columns {cols} != hsp_id molecule_id qname molecule_state train_tiles_excluded")
    bad = cursor.execute("""
        WITH h AS (SELECT hsp_id, any_value(molecule_id) AS molecule_id, count(DISTINCT molecule_id) AS n FROM stg.hsp GROUP BY hsp_id),
             e AS (SELECT molecule_id, any_value(molecule_state) AS state, bool_or(train_tiles_excluded) AS tte,
                          list(read_name) AS reads FROM estc0.molecule_member GROUP BY molecule_id),
             s AS (SELECT hsp_id, count(*) AS k, any_value(molecule_id) AS molecule_id, any_value(qname) AS qname,
                          any_value(molecule_state) AS state, lower(any_value(train_tiles_excluded)) AS tte
                   FROM stg.hspstate GROUP BY hsp_id)
        SELECT coalesce(h.hsp_id, s.hsp_id) FROM h FULL JOIN s USING (hsp_id) LEFT JOIN e ON e.molecule_id = h.molecule_id
        WHERE h.hsp_id IS NULL OR s.hsp_id IS NULL OR h.n <> 1 OR s.k <> 1 OR s.molecule_id <> h.molecule_id
           OR s.state IS DISTINCT FROM e.state OR s.tte NOT IN ('true', 'false') OR (s.tte = 'true') IS DISTINCT FROM e.tte
           OR NOT list_contains(e.reads, s.qname)
        LIMIT 3""").fetchall()
    require(not bad, f"{path} disagrees with the EST C0 molecule_member (or with --hsp) for HSP(s) {[b[0] for b in bad]}")
    ctx["qc"]["hsp_state_sidecar_rows"] = cursor.execute("SELECT count(*) FROM stg.hspstate").fetchone()[0]


def verify_hsp_placements(ctx, cursor):
    """Rule 6 projects an HSP "onto the genome through the EST's spliced alignment" and tracks are built "from retained
    observations", i.e. A18.5-accepted primary alignments.  The EST C0 keeps rejected members (molecule_member has no
    acceptance column and a rejected-only molecule can be train_track), so the molecule join above cannot tell; every
    route-B interval must therefore lie inside an ACCEPTED primary placement of its sidecar qname in --est-c0.  Part B
    (b6_hsp_project.py) never projects a qname the molecule map flags accepted=false; any such row here fails closed."""
    bad = cursor.execute("""
        SELECT h.hsp_id, h.interval_index, s.qname FROM stg.hsp h JOIN stg.hspstate s USING (hsp_id)
        WHERE NOT EXISTS (SELECT 1 FROM estc0.evidence_alignment a
                          WHERE a.species = h.species AND a.read_name = s.qname AND a.accepted AND a.placement_kind = 'primary'
                            AND a.chrom = h.chrom AND a.start0 <= h.genomic_start0 AND h.genomic_end0 <= a.end0)
        ORDER BY 1, 2 LIMIT 3""").fetchall()
    require(not bad, f"HSP interval(s) {bad} lie in no ACCEPTED primary EST placement of their qname in --est-c0 "
                     "(rule 6 / A18.5: route B projects only through accepted alignments; the qname's C0 primary was "
                     "rejected, unmapped or elsewhere)")
    ctx["qc"]["hsp_rows_inside_accepted_est_placement"] = cursor.execute("SELECT count(*) FROM stg.hsp").fetchone()[0]
    # The HSP's library / est_accession must be those of its EST member in THIS --est-c0 (a projection made before a
    # metadata correction keeps the old library; library withholding would then keep phase evidence of a withheld one).
    stale = cursor.execute("""
        SELECT h.hsp_id, h.interval_index, h.library, h.est_accession FROM stg.hsp h
        WHERE NOT EXISTS (SELECT 1 FROM estc0.molecule_member m
                          WHERE m.species = h.species AND m.molecule_id = h.molecule_id
                            AND m.accession = h.est_accession AND m.library = h.library)
        ORDER BY 1, 2 LIMIT 3""").fetchall()
    require(not stale, f"HSP row(s) {stale} carry a library / est_accession that differs from their EST member in "
                       "--est-c0 (stale projection: re-run b6_hsp_project.py against the current molecule map)")


def load_hsp(ctx, st, cursor, row, args):
    species, qc, index = ctx["species"], ctx["qc"], ctx["leakage"].index
    reader = ("read_parquet(?)" if str(args.hsp).endswith(".parquet")
              else "read_csv(?, delim='\t', header=true, all_varchar=true, quote='', escape='')")
    cols = [d[0] for d in cursor.execute(f"SELECT * FROM {reader} LIMIT 0", [str(args.hsp)]).description]
    require(tuple(cols) == S.HSP_COLUMN_NAMES, f"--hsp columns {cols} != est_protein_hsp contract {list(S.HSP_COLUMN_NAMES)}")
    casts = ", ".join(f"CAST({c} AS {_t(t)}) AS {c}" for c, t in S.HSP_COLUMNS)
    cursor.execute(f"CREATE TABLE stg.hsp AS SELECT {casts} FROM {reader}", [str(args.hsp)])
    q = lambda sql, p=(): cursor.execute(sql, list(p)).fetchall()  # noqa: E731
    bad = q("SELECT DISTINCT species FROM stg.hsp WHERE species <> ?", [species])
    require(not bad, f"--hsp rows of other species {bad} in a {species} build")
    verify_hsp_provenance(ctx, cursor, args)
    dup = q("SELECT hsp_id, interval_index FROM stg.hsp GROUP BY ALL HAVING count(*) > 1 LIMIT 3")
    require(not dup, f"--hsp duplicate (hsp_id, interval_index) rows, e.g. {dup}")
    for (ps,) in q("SELECT DISTINCT protein_species FROM stg.hsp"):
        require(re.fullmatch(r"\d+", ps or "") is not None, f"--hsp protein_species {ps!r} is not an NCBI taxid")
        ctx["a19"].check(int(ps), f"HSP protein_species {ps}")
    cursor.execute(f"ATTACH {lit(Path(args.est_c0).absolute())} AS estc0 (READ_ONLY)")
    try:
        orphans = q("""SELECT count(*), min(h.molecule_id) FROM stg.hsp h WHERE NOT EXISTS (
                           SELECT 1 FROM estc0.molecule_member m WHERE m.species = h.species AND m.molecule_id = h.molecule_id)""")
        require(orphans[0][0] == 0, f"{orphans[0][0]} est_protein_hsp row(s) reference no molecule of the {species} EST "
                                    f"molecule_member (e.g. {orphans[0][1]}); the molecule map is not the C0 one")
        cross_check_hsp_state(ctx, cursor, args)
        verify_hsp_placements(ctx, cursor)
        role = ROLE_MAP[row["role"]]
        n = n_excl = n_tt = n_pq = n_pq_tt = n_never = 0
        conflict = Counter()
        base = dict(source="protein", dataset=row["dataset"], run=row["run"], genotype_stratum="not_applicable", role=role,
                    validation_only=role == "validation_only", c0_provenance_sha256=ctx["provenance_sha256"])

        def resolve(population):
            """MULTI_HSP_RULE over ONE population's rows -> {hsp_id: reason} of the HSPs to drop."""
            hsps = {}
            for h in population:
                hsps.setdefault(h["hsp_id"], dict(key=h["hsp_id"], scope=h["protein_source"], intervals=[]))
                hsps[h["hsp_id"]]["intervals"].append(dict(contig=h["chrom"], start0=h["genomic_start0"],
                                                           end0=h["genomic_end0"], strand=h["strand"], phase=h["phase"]))
            if not hsps:
                return {}
            try:
                return GEOM.resolve_hsps(list(hsps.values()))["dropped"]
            except ValueError as exc:
                raise C0Error(f"--hsp molecule {population[0]['molecule_id']}: {exc}") from exc

        def emit_molecule(group):
            """MULTI_HSP_RULE (rule 6: "HSPs whose frame is inconsistent within one EST after projection are dropped")
            per (molecule, protein_source), applied HERE, after eligibility is known, separately within each of the
            two populations a consumer may read and never across them (fix round 6, finding 2):
              * training population = the HSPs that are not leakage_excluded (fix round 5: an excluded protein query's
                HSPs are "removed molecule-wide", rule 3/6, so they cannot delete an eligible HSP);
              * excluded population = the leakage_excluded HSPs of the molecule (the rule-10 held-out / evaluation
                evidence of an excluded_heldout or never molecule, and the protein-query-excluded rows of a train_track
                molecule -- the C0 has no finer notion of "evaluation input" than molecule_state, so ALL leakage-excluded
                rows of the molecule are resolved together; a row dropped here is counted under the *_excluded keys).
            b6_hsp_project.py emits every projected HSP unresolved; a row is never dropped silently."""
            nonlocal n, n_excl, n_tt, n_pq, n_pq_tt, n_never
            dropped = {}
            for population, suffix in ((False, ""), (True, "_excluded")):
                for hsp_id in resolve([h for h, leak, *_ in group if leak == population]):
                    dropped[hsp_id] = suffix
            counted = set()
            for h, leak, tt, cause, pq_only, pq_tt_only in group:
                if h["hsp_id"] in dropped:
                    suffix = dropped[h["hsp_id"]]
                    conflict["hsp_rows_dropped_phase_conflict_within_molecule" + suffix] += 1
                    if h["hsp_id"] not in counted:
                        counted.add(h["hsp_id"])
                        conflict["hsp_dropped_phase_conflict_within_molecule" + suffix] += 1
                        conflict["hsp_dropped_phase_conflict_within_molecule" + suffix + "_" + h["protein_source"]] += 1
                    continue
                st.add("hspout", **h, **base, leakage_excluded=leak, train_tiles_excluded=tt, exclusion_cause=cause)
                n += 1
                n_excl += leak
                n_tt += tt and not leak
                n_pq += pq_only
                n_pq_tt += pq_tt_only
                n_never += cause == "est_molecule_never"

        # Rule 3: "the leakage-closure unit is the protein query sequence with all its alignments ... A molecule is one
        # unit everywhere it appears"; rule 6: "protein queries touching excluded intervals are removed molecule-wide".
        # The route-B protein subject is that same query where it was miniprot-aligned (OrthoDB), so its rule 4(b)
        # decision of THIS build (stg.state, decided before load_hsp) is inherited by its HSPs.  Swiss-Prot subjects
        # were never miniprot-aligned and have no state row (LEFT JOIN -> NULL, no effect).  Rows stream molecule by
        # molecule (ORDER BY molecule_id); Python holds one molecule's HSP rows at a time.
        group, cur = [], None
        for r in fetch_stream(cursor, f"""
                SELECT {', '.join('h.' + c for c in S.HSP_COLUMN_NAMES)}, e.excl, e.tte, e.never, e.cause, p.state, p.tte
                FROM stg.hsp h JOIN (SELECT molecule_id, bool_or(molecule_state = 'excluded_heldout') AS excl,
                                            bool_or(train_tiles_excluded) AS tte, bool_and(molecule_state = 'never') AS never,
                                            string_agg(DISTINCT exclusion_cause, ',') AS cause
                                     FROM estc0.molecule_member WHERE species = ? GROUP BY molecule_id) e USING (molecule_id)
                LEFT JOIN (SELECT m.read AS protein, s.state, s.tte FROM stg.member m JOIN stg.unit u USING (member_id)
                           JOIN stg.state s ON s.molecule_id = u.molecule_id) p ON p.protein = h.protein_id
                ORDER BY h.molecule_id, h.hsp_id, h.interval_index""", [species]):
            h = dict(zip(S.HSP_COLUMN_NAMES, r[:len(S.HSP_COLUMN_NAMES)]))
            excl, tte, never, cause, p_state, p_tte = r[len(S.HSP_COLUMN_NAMES):]
            iv_all = index.causes(h["chrom"], h["genomic_start0"], h["genomic_end0"], "all")
            iv_tt = index.causes(h["chrom"], h["genomic_start0"], h["genomic_end0"], "train_tiles")
            # An HSP lies inside its EST molecule's primary placement, which rule 4 already tested: an 'all' hit on a
            # train_track molecule means the EST DB and this build disagree -> fail closed.
            require(not (iv_all and not excl and not never),
                    f"HSP {h['hsp_id']}/{h['interval_index']} overlaps {sorted(iv_all)} but EST molecule {h['molecule_id']} "
                    "is train_track")
            pq_excl = p_state == "excluded_heldout"
            # A 'never' EST molecule (validation_only run, rule 1 / rule 10) is not a training molecule: its HSP is no
            # training channel either (flagged, cause est_molecule_never; fix round 5).
            leak = bool(excl or iv_all or pq_excl or never)
            tt = bool(tte or iv_tt or p_tte)
            causes = ",".join(sorted((iv_all | iv_tt) | ({"protein_query"} if pq_excl or p_tte else set()))) or None
            if leak:
                cause_out = cause or causes or ("est_molecule_never" if never else "est_molecule")
            else:
                cause_out = causes if tt else None
            if h["molecule_id"] != cur and group:
                emit_molecule(group)
                group = []
            cur = h["molecule_id"]
            group.append((h, leak, tt, cause_out, bool(pq_excl and not (excl or iv_all or never)),
                          bool(p_tte and not (tte or iv_tt))))
        if group:
            emit_molecule(group)
    finally:
        st.flush("hspout")
        cursor.execute("DETACH estc0")
    qc["hsp_rows"], qc["hsp_rows_excluded"], qc["hsp_rows_train_tiles_excluded"] = n, n_excl, n_tt
    qc["hsp_rows_excluded_by_protein_query_only"], qc["hsp_rows_train_tiles_excluded_by_protein_query_only"] = n_pq, n_pq_tt
    qc["hsp_rows_excluded_never_est_molecule"] = n_never
    for suffix in ("", "_excluded"):                 # training population / excluded population (fix round 6)
        base_key = "hsp_dropped_phase_conflict_within_molecule" + suffix
        for key in (base_key, "hsp_rows_dropped_phase_conflict_within_molecule" + suffix,
                    base_key + "_orthodb", base_key + "_swissprot"):
            qc[key] = conflict[key]


# ----------------------------------------------------------------------------------------------- decisions and QC
def decide_all(ctx, st, cursor, *, audited, protein_cap=None):
    """Rule 4(b), molecule by molecule (streamed): decisions -> stg.dec, QC counts, independent held-out recount."""
    qc, index = ctx["qc"], ctx["leakage"].index
    cursor.execute("""CREATE TABLE stg.molsum AS
        SELECT u.molecule_id, count(*) AS n_members, bool_or(m.training) AS any_t, bool_and(m.training) AS all_t,
               coalesce(sum(a.n_secondary), 0) AS n_secondary
        FROM stg.unit u JOIN stg.member m USING (member_id)
        LEFT JOIN stg.aread a ON a.run_idx = m.run_idx AND a.read = m.read GROUP BY u.molecule_id""")
    mixed = cursor.execute("SELECT count(*), min(molecule_id) FROM stg.molsum WHERE any_t <> all_t").fetchone()
    require(mixed[0] == 0, f"molecule {mixed[1]} mixes training_eligible and validation_only members")
    heldout = LK.HeldoutLoci(ctx["leakage"].heldout_loci)
    ex = LK.Exhaustiveness(ctx["audit_n"], label=f"{ctx['species']}/{ctx['source']}")
    for key in ("molecules_in", "excluded_a", "excluded_a_direct_primary_supplementary", "excluded_b_secondary_cap",
                "excluded_b_low_mapq", "excluded_exclusive_a", "excluded_exclusive_b_secondary",
                "excluded_exclusive_b_low_mapq", "retained", "retained_train_tiles_excluded",
                "excluded_b_cause_secondary_mapq", "excluded_a_cause_edge_partial", "train_tiles_cause_edge_partial",
                "heldout_overlapping_molecules", "heldout_overlapping_excluded", "heldout_retained",
                "protein_excluded_b_miniprot_cap"):
        qc[key] += 0
    cur_id, info, pls = None, None, []

    def finish():
        d = LK.decide_molecule(cur_id, info[0], pls, info[1], index, audited=audited, n_cap=ctx["audit_n"])
        if protein_cap is not None and d.n_placements >= protein_cap:
            d.unauditable_secondary = True            # miniprot -N cap reached: placements may be missing
            d.n_secondary = d.n_placements - 1
            qc["protein_excluded_b_miniprot_cap"] += 1
        if audited:
            ex.add(d)
        cause = ",".join(list(d.cause_a) + (["b_secondary_cap"] if d.unauditable_secondary else [])
                         + (["b_low_mapq"] if d.unauditable_low_mapq else [])) or None
        st.add("dec", molecule_id=d.molecule_id, n_members=d.n_members, n_secondary=d.n_secondary,
               n_placements=d.n_placements, cause_a=",".join(d.cause_a) or None, cause_a_direct=d.cause_a_direct,
               cause_a_train_tiles=",".join(d.cause_a_train_tiles) or None, unauditable_secondary=d.unauditable_secondary,
               unauditable_low_mapq=d.unauditable_low_mapq, excluded=d.excluded, train_tiles_excluded=d.train_tiles_excluded,
               audited=d.audited, low_mapq_secondary_only=d.low_mapq_secondary_only, cause=cause)
        qc["molecules_in"] += 1
        qc["excluded_a"] += bool(d.cause_a)
        qc["excluded_a_direct_primary_supplementary"] += d.cause_a_direct
        qc["excluded_b_secondary_cap"] += d.unauditable_secondary
        qc["excluded_b_low_mapq"] += d.unauditable_low_mapq
        qc["excluded_b_cause_secondary_mapq"] += d.unauditable_low_mapq and d.low_mapq_secondary_only
        qc["train_tiles_cause_edge_partial"] += "edge_partial" in d.cause_a_train_tiles
        if d.primary_cause:
            qc["excluded_exclusive_" + d.primary_cause] += 1
        for c in d.cause_a:
            qc["excluded_a_cause_" + c] += 1
        qc["retained"] += not d.excluded
        qc["retained_train_tiles_excluded"] += (not d.excluded) and d.train_tiles_excluded
        if any(heldout.hits(p[2], p[3], p[4]) for p in pls):
            qc["heldout_overlapping_molecules"] += 1
            qc["heldout_overlapping_excluded" if d.excluded else "heldout_retained"] += 1

    for mid, n_members, n_sec, origin, kind, chrom, s0, e0, mapq in fetch_stream(cursor, """
            SELECT s.molecule_id, s.n_members, s.n_secondary, p.origin, p.kind, p.chrom, p.start0, p.end0, p.mapq
            FROM stg.molsum s JOIN stg.unit u USING (molecule_id) LEFT JOIN stg.pl p USING (member_id)
            WHERE s.all_t ORDER BY s.molecule_id"""):
        if mid != cur_id:
            if cur_id is not None:
                finish()
            cur_id, info, pls = mid, (n_members, n_sec), []
        if origin is not None:
            pls.append((origin, kind, chrom, s0, e0, mapq))
    if cur_id is not None:
        finish()
    st.flush("dec")
    if audited:
        # step 3: per-source counters (numerator / denominator for the merge's species sum) AND the per-source floor
        # (author 2026-09-23): below MIN_SOURCE_EXHAUSTIVENESS this raises ExhaustivenessError -> nothing is published.
        qc.update(ex.report())          # counters survive an abort (the QC record of a failed build keeps them)
        ex.check()                      # the per-source floor: raises ExhaustivenessError; qc is a Counter, so its
                                        # return value is NOT added again (that would double every counter)
    require(qc["molecules_in"] == qc["retained"] + sum(qc["excluded_exclusive_" + c] for c in ("a", "b_secondary", "b_low_mapq")),
            "QC accounting: molecules_in != retained + exclusive exclusions")
    cursor.execute("""CREATE TABLE stg.state AS
        SELECT s.molecule_id,
               CASE WHEN NOT s.all_t THEN 'never' WHEN d.excluded THEN 'excluded_heldout' ELSE 'train_track' END AS state,
               CASE WHEN s.all_t THEN d.cause END AS cause,
               coalesce(CASE WHEN s.all_t THEN d.train_tiles_excluded END, false) AS tte, d.molecule_id IS NULL AND s.all_t AS undecided
        FROM stg.molsum s LEFT JOIN stg.dec d USING (molecule_id)""")
    undecided = cursor.execute("SELECT count(*) FROM stg.state WHERE undecided").fetchone()[0]
    require(undecided == 0, f"{undecided} training molecule(s) without a rule 4(b) decision")


# ----------------------------------------------------------------------------------------------- final tables
def insert_rows(con, table, rows):
    import pyarrow as pa

    cols = [c for c, _ in S.table_columns(table)]
    for i in range(0, len(rows), CHUNK):
        chunk = rows[i:i + CHUNK]
        missing = set(cols) - set(chunk[0])
        require(not missing, f"{table}: row lacks columns {sorted(missing)}")
        con.register("_c0_rows", pa.table({c: pa.array([r[c] for r in chunk]) for c in cols}))
        try:
            con.execute(f"INSERT INTO {table} SELECT {', '.join(cols)} FROM _c0_rows")
        finally:
            con.unregister("_c0_rows")


def write_final(ctx, con):
    sp, src, prov = lit(ctx["species"]), lit(ctx["source"]), lit(ctx["provenance_sha256"])
    lk = ctx["leakage"]
    insert_rows(con, "excluded_interval", [dict(species=ctx["species"], source=ctx["source"], chrom=iv.chrom, start0=iv.start0,
                                                end0=iv.end0, cause=iv.cause, scope=iv.scope, gene_id=iv.gene_id,
                                                tile_rn=iv.tile_rn) for iv in lk.intervals])
    insert_rows(con, "heldout_locus", [dict(species=ctx["species"], source=ctx["source"], gene_id=g, chrom=c, start0=a, end0=b)
                                       for g, c, a, b in lk.heldout_loci])
    insert_rows(con, "excluded_junction_coordinate", [
        dict(species=ctx["species"], source=ctx["source"], chrom=c, strand=s, kind=k, coordinate0=x, excluded_gene_id=eg,
             neighbour_gene_id=ng) for (c, s, k, x), (eg, ng) in sorted(lk.coordinates.items())])
    base = {"species": sp, "source": src, "provenance_sha256": prov, "dataset": "r.dataset", "run": "r.run",
            "role": "r.role", "validation_only": "(r.role = 'validation_only')", "library": "m.library",
            "genotype_stratum": "m.genotype"}
    for table, (stage, drop_superseded) in STAGE_OF.items():
        exprs = []
        for c, _ in S.table_columns(table):
            e = base.get(c) or {"molecule_id": "u.molecule_id", "read_name": "m.read", "arm": "'primary'",
                                "complete_5p_code": "NULL", "complete_3p_code": "NULL", "read_internal_code": "NULL",
                                "block_index": "row_number() OVER (PARTITION BY u.molecule_id ORDER BY x.chrom, x.start0, "
                                               "x.end0, x.alignment_id, x.alignment_block_index) - 1"}.get(c, f"x.{c}")
            exprs.append(f"{e} AS {c}")
        cols = [c for c, _ in S.table_columns(table)]
        con.execute(f"""INSERT INTO {table} ({', '.join(cols)})
            SELECT {', '.join(exprs)} FROM stg.{stage} x JOIN stg.member m ON m.member_id = x.member_id
            JOIN stg.run r ON r.run_idx = m.run_idx JOIN stg.unit u ON u.member_id = x.member_id
            {'WHERE NOT u.superseded' if drop_superseded else ''} ORDER BY {', '.join(S.primary_key(table))}""")
    mm_cols = [c for c, _ in S.table_columns("molecule_member")]
    con.execute(f"""INSERT INTO molecule_member ({', '.join(mm_cols)})
        SELECT {sp}, {src}, r.dataset, r.run, m.library, m.genotype, r.role, r.role = 'validation_only', {prov},
               m.member_id, u.molecule_id, m.read, m.accession, m.stem, m.version, m.part, NULLIF(m.clone, ''), m.bioproject,
               u.unit_rule, u.superseded, s.state, s.tte, s.cause
        FROM stg.member m JOIN stg.run r ON r.run_idx = m.run_idx JOIN stg.unit u ON u.member_id = m.member_id
        JOIN stg.state s ON s.molecule_id = u.molecule_id ORDER BY m.member_id""")
    n_member = con.execute("SELECT count(*) FROM stg.member").fetchone()[0]
    require(con.execute("SELECT count(*) FROM molecule_member").fetchone()[0] == n_member,
            "molecule_member lost members (a member without a unit or state)")
    # rule 4(c) and junction counts
    con.execute(f"""CREATE TABLE stg.sup2 AS
        SELECT x.junction_key, r.dataset, r.run, m.library, r.role, m.genotype, u.molecule_id, s.state,
               x.chrom, x.strand, x.start0, x.end0, x.donor0, x.acceptor0
        FROM stg.sup x JOIN stg.member m ON m.member_id = x.member_id JOIN stg.run r ON r.run_idx = m.run_idx
        JOIN stg.unit u ON u.member_id = x.member_id JOIN stg.state s ON s.molecule_id = u.molecule_id
        WHERE NOT u.superseded""")
    con.execute("""CREATE TABLE stg.jflag AS
        SELECT j.junction_key, bool_or(cd.chrom IS NOT NULL OR ca.chrom IS NOT NULL) AS excluded_4c
        FROM (SELECT DISTINCT junction_key, chrom, strand, donor0, acceptor0 FROM stg.sup2) j
        LEFT JOIN (SELECT DISTINCT chrom, strand, coordinate0 FROM excluded_junction_coordinate WHERE kind = 'donor') cd
               ON cd.chrom = j.chrom AND cd.strand = j.strand AND cd.coordinate0 = j.donor0
        LEFT JOIN (SELECT DISTINCT chrom, strand, coordinate0 FROM excluded_junction_coordinate WHERE kind = 'acceptor') ca
               ON ca.chrom = j.chrom AND ca.strand = j.strand AND ca.coordinate0 = j.acceptor0
        GROUP BY j.junction_key""")
    multi = con.execute("""SELECT library, list(DISTINCT genotype) FROM stg.sup2
                           GROUP BY junction_key, dataset, run, library, role HAVING count(DISTINCT genotype) > 1 LIMIT 1""").fetchall()
    require(not multi, f"library {multi[0][0] if multi else ''} carries several genotype strata {multi[0][1] if multi else ''}")
    js_cols = [c for c, _ in S.table_columns("junction_support")]
    con.execute(f"""INSERT INTO junction_support ({', '.join(js_cols)})
        SELECT {sp}, {src}, dataset, run, library, any_value(genotype), role, role = 'validation_only', {prov},
               junction_key, molecule_id, count(*), any_value(state), any_value(f.excluded_4c)
        FROM stg.sup2 JOIN stg.jflag f USING (junction_key)
        GROUP BY junction_key, dataset, run, library, role, molecule_id
        ORDER BY junction_key, dataset, run, library, role, molecule_id""")
    capped = "least(10, count(DISTINCT molecule_id))" if ctx["source"] == "est" else "count(DISTINCT molecule_id)"
    j_cols = [c for c, _ in S.table_columns("junction")]
    con.execute(f"""INSERT INTO junction ({', '.join(j_cols)})
        SELECT {sp}, {src}, dataset, run, library, any_value(genotype), role, role = 'validation_only', {prov},
               junction_key, any_value(chrom), any_value(strand), any_value(start0), any_value(end0), any_value(donor0),
               any_value(acceptor0), count(DISTINCT molecule_id),
               CASE WHEN any_value(f.excluded_4c) THEN 0
                    ELSE count(DISTINCT molecule_id) FILTER (WHERE state = 'train_track') END,
               {capped}, count(*)
        FROM stg.sup2 JOIN stg.jflag f USING (junction_key)
        GROUP BY junction_key, dataset, run, library, role ORDER BY junction_key, dataset, run, library, role""")
    rows, mols = con.execute("""SELECT count(*), count(DISTINCT molecule_id) FROM junction_support
                                WHERE excluded_4c AND molecule_state = 'train_track'""").fetchone()
    ctx["qc"]["excluded_c_support_rows"], ctx["qc"]["excluded_c_molecules"] = rows, mols
    me_cols = [c for c, _ in S.QC_TABLES["molecule_exclusion"][0]]
    con.execute(f"""INSERT INTO molecule_exclusion ({', '.join(me_cols)})
        SELECT {sp}, {src}, {', '.join(me_cols[2:])} FROM stg.dec ORDER BY molecule_id""")
    con.execute(f"""INSERT INTO audit_placement
        SELECT {sp}, {src}, u.molecule_id, p.member_id, p.origin, p.kind, p.chrom, p.start0, p.end0, p.mapq
        FROM stg.pl p JOIN stg.unit u ON u.member_id = p.member_id JOIN stg.molsum s ON s.molecule_id = u.molecule_id
        WHERE s.all_t ORDER BY u.molecule_id, p.member_id, p.origin, p.kind, p.chrom, p.start0, p.end0""")
    con.execute("INSERT INTO est_protein_hsp SELECT * FROM stg.hspout ORDER BY hsp_id, interval_index")


# ----------------------------------------------------------------------------------------------- step 7 gate
# Two HSP intervals of one (species, molecule, protein source, chrom) AND ONE POPULATION (both retained, or both
# leakage_excluded -- never across) that overlap must share the strand and the codon frame.  Frame equality:
# est_phase_project.base_label(a, g) == b.phase for every shared base <=>
# '+': (a.phase - a.start0) == (b.phase - b.start0) (mod 3); '-': (a.phase + a.start0) == (b.phase + b.start0) (mod 3).
_HSP_CONFLICT_SQL = """
    SELECT count(*) FROM est_protein_hsp a JOIN est_protein_hsp b
      ON a.species = b.species AND a.molecule_id = b.molecule_id AND a.protein_source = b.protein_source
     AND a.chrom = b.chrom AND a.hsp_id < b.hsp_id
     AND a.genomic_start0 < b.genomic_end0 AND b.genomic_start0 < a.genomic_end0
    WHERE {population}
      AND (a.strand <> b.strand
           OR (a.strand = '+' AND ((a.phase - a.genomic_start0) % 3 + 3) % 3 <> ((b.phase - b.genomic_start0) % 3 + 3) % 3)
           OR (a.strand = '-' AND (a.phase + a.genomic_start0) % 3 <> (b.phase + b.genomic_start0) % 3))"""
HSP_CONFLICT_SQL = _HSP_CONFLICT_SQL.format(population="NOT a.leakage_excluded AND NOT b.leakage_excluded")
HSP_CONFLICT_EXCLUDED_SQL = _HSP_CONFLICT_SQL.format(population="a.leakage_excluded AND b.leakage_excluded")


# accepted rows; rows with all three A16 codes NULL; rows with all three codes set (a partially set row is neither).
COMPLETENESS_COUNTS_SQL = """
    SELECT count(*),
           count(*) FILTER (WHERE complete_5p_code IS NULL AND complete_3p_code IS NULL AND read_internal_code IS NULL),
           count(*) FILTER (WHERE complete_5p_code IS NOT NULL AND complete_3p_code IS NOT NULL AND read_internal_code IS NOT NULL)
    FROM evidence_alignment WHERE accepted"""


def completeness_policy_violation(policy, accepted_rows, null_rows, coded_rows):
    """Accepted rows that contradict the recorded A16 completeness-code policy (0 = consistent).  Under
    ``null_not_computed_at_c0`` every accepted row must be NULL in all three columns: any set code is drift."""
    require(policy in COMPLETENESS_CODES_POLICIES, f"unknown completeness-code policy {policy!r}")
    return int(accepted_rows - null_rows) if policy == "null_not_computed_at_c0" else int(accepted_rows - coded_rows)


def sql_gate(con, ctx):
    """Step 7: an independent re-test of the WRITTEN rows against the WRITTEN intervals / held-out loci (indices
    rebuilt from the DB tables, not from the in-memory model); rows streamed.  Every count must be 0."""
    qc = ctx["qc"]
    q = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731
    ivs = con.execute("SELECT chrom, start0, end0, cause, scope FROM excluded_interval").fetchall()
    index = LK.IntervalIndex([LK.ExcludedInterval(c, a, b, cause, scope) for c, a, b, cause, scope in ivs])
    loci = LK.HeldoutLoci(con.execute("SELECT gene_id, chrom, start0, end0 FROM heldout_locus").fetchall())
    cursor = con.cursor()
    state = "(SELECT DISTINCT molecule_id, molecule_state, train_tiles_excluded FROM molecule_member)"

    def count(sql, test):
        n, mols = 0, set()
        for mid, chrom, a, b in fetch_stream(cursor, sql):
            if chrom is not None and test(chrom, a, b):
                n += 1
                mols.add(mid)
        return n, len(mols)

    hit_all = lambda c, a, b: bool(index.causes(c, a, b, "all"))            # noqa: E731
    hit_tt = lambda c, a, b: bool(index.causes(c, a, b, "train_tiles"))     # noqa: E731
    rows = {
        "block": "SELECT x.molecule_id, x.chrom, x.start0, x.end0 FROM aligned_block x JOIN {s} s USING (molecule_id) WHERE {w}",
        "junction": "SELECT x.molecule_id, x.chrom, x.raw_start0, x.raw_end0 FROM junction_observation x "
                    "JOIN {s} s USING (molecule_id) WHERE {w}",
        "alignment": "SELECT x.molecule_id, x.chrom, x.start0, x.end0 FROM evidence_alignment x JOIN {s} s USING (molecule_id) "
                     "WHERE x.placement_kind IN ('primary', 'supplementary') AND {w}",
    }
    train = "s.molecule_state = 'train_track'"
    for name, sql in rows.items():
        qc[f"gate_train_track_{name}_rows_in_excluded_all"] = count(sql.format(s=state, w=train), hit_all)[0]
        if name != "alignment":
            qc[f"gate_train_tiles_{name}_rows_in_train_tiles_intervals"] = count(
                sql.format(s=state, w=train + " AND NOT s.train_tiles_excluded"), hit_tt)[0]
        qc[f"gate_train_track_{name}_rows_at_heldout_locus"] = count(sql.format(s=state, w=train), loci.hits)[0]
    qc["gate_hsp_retained_in_excluded_all"] = count(
        "SELECT molecule_id, chrom, genomic_start0, genomic_end0 FROM est_protein_hsp WHERE NOT leakage_excluded", hit_all)[0]
    qc["gate_hsp_train_tiles_in_train_tiles_intervals"] = count(
        "SELECT molecule_id, chrom, genomic_start0, genomic_end0 FROM est_protein_hsp WHERE NOT train_tiles_excluded", hit_tt)[0]
    # Rule 6 re-test of the WRITTEN route-B rows, per population: among retained (not leakage_excluded) HSPs, and
    # separately among leakage_excluded HSPs, of one molecule and protein source, no two HSPs may overlap with a
    # different strand or codon frame (independent of load_hsp's resolution; never tested across the populations).
    qc["hsp_phase_conflict_among_retained_recheck"] = q(HSP_CONFLICT_SQL)
    qc["hsp_phase_conflict_among_excluded_recheck"] = q(HSP_CONFLICT_EXCLUDED_SQL)
    # A16 completeness codes: count the written rows against the recorded policy (never a silent NULL).
    counts = con.execute(COMPLETENESS_COUNTS_SQL).fetchone()
    qc["completeness_codes_accepted_rows"], qc["completeness_codes_null_accepted_rows"], \
        qc["completeness_codes_coded_accepted_rows"] = counts
    qc["completeness_codes_policy_violation"] = completeness_policy_violation(COMPLETENESS_CODES_POLICY["policy"], *counts)
    placements = ("SELECT p.molecule_id, p.chrom, p.start0, p.end0 FROM audit_placement p "
                  "JOIN molecule_exclusion m USING (species, source, molecule_id) WHERE {w}")
    qc["gate_retained_overlapping_excluded_all"] = count(placements.format(w="NOT m.excluded"), hit_all)[1]
    qc["gate_state_mismatch"] = q("""SELECT count(*) FROM molecule_member mm JOIN molecule_exclusion m USING (species, source, molecule_id)
                                     WHERE (mm.molecule_state = 'train_track') = m.excluded""")
    overl = count(placements.format(w="true"), loci.hits)[1]
    excl = count(placements.format(w="m.excluded"), loci.hits)[1]
    qc["gate_heldout_overlapping"], qc["gate_heldout_excluded"] = overl, excl
    qc["gate_heldout_retained"] = overl - excl
    qc["gate_python_sql_heldout_disagreement"] = int(overl != qc["heldout_overlapping_molecules"])
    qc["gate_retained_4c_support_counted"] = q(
        "SELECT count(*) FROM junction j JOIN junction_support s USING (junction_key, dataset, run, library, role) "
        "WHERE s.excluded_4c AND j.n_molecules_train_track > 0")
    cursor.close()
    qc["leakage_nonzero"] = sum(v for k, v in qc.items() if k.startswith("gate_") and k not in (
        "gate_heldout_overlapping", "gate_heldout_excluded")) + qc["heldout_retained"]


# ----------------------------------------------------------------------------------------------- driver
def build(args, argv):
    species, source = args.species, args.source
    require((args.hsp is None) or source == "protein", "--hsp is only valid with --source protein")
    require(source != "protein" or (args.hsp and args.est_c0 and args.lineage),
            "--source protein requires --hsp, --est-c0 and --lineage (route-B rows and A19 by lineage)")
    require(args.est_c0 is None or source == "protein", "--est-c0 is only valid with --source protein")
    require(args.audit_n >= LK.AUDIT_N, f"--audit-n must be >= {LK.AUDIT_N} (A46 rule 4(b)); N may only be raised")
    out, qc_path = Path(args.out).absolute(), Path(args.qc).absolute()
    for p in (out, qc_path):
        if os.path.lexists(p):
            raise FileExistsError(p)
    roles_rows = read_roles([args.roles, args.est_roles])
    rows, excluded_rows = select_runs(roles_rows, species, source)
    refuse_unroled(args.align_root, species, source, roles_rows)
    versions, tool_shas = load_tool_provenance(args.provenance)
    leakage = LK.build_leakage_model(corpus_db=args.corpus, species=species, splits_path=args.splits, flags_path=args.flags,
                                     build_flags=args.build_flags, gff_path=args.gff, fasta_path=args.reference, seed=args.seed)
    if args.overlap:
        verify_overlap(args.overlap, leakage.intervals)
    ctx = {"species": species, "source": source, "genome": leakage.genome, "leakage": leakage, "qc": Counter(),
           "assembly": leakage.manifest["fasta_sha256"], "inputs": {}, "versions": versions, "audit_n": args.audit_n,
           "intervals_sha256": intervals_sha256(leakage.intervals),
           "interval_manifest_sha256": LK.interval_manifest_sha256(leakage.interval_manifest)}
    inputs = {"roles": args.roles, "est_roles": args.est_roles, "corpus": args.corpus, "splits": args.splits, "flags": args.flags,
              "reference": args.reference or leakage.manifest["fasta"], "gff": args.gff or leakage.manifest["gff"]}
    inputs.update({f"build_flags_{i}": p for i, p in enumerate(args.build_flags or [])})
    for k in ("hsp", "overlap", "lineage"):
        if getattr(args, k):
            inputs[k] = getattr(args, k)
    if args.hsp:
        require(hsp_state_path(args.hsp).is_file(), f"missing {hsp_state_path(args.hsp)} (b6_hsp_project.py state sidecar)")
        inputs["hsp_molecule_state"] = hsp_state_path(args.hsp)
        require(hsp_provenance_path(args.hsp).is_file(),
                f"missing {hsp_provenance_path(args.hsp)} (b6_hsp_project.py provenance record)")
        inputs["hsp_provenance"] = hsp_provenance_path(args.hsp)
    for k, p in inputs.items():
        ctx["inputs"][k] = {"path": str(p), "sha256": digest(p)}
    if source == "protein":
        ctx["a19"] = A19(args.lineage, species)
        verify_est_c0(ctx, args)
    code = {str(p): digest(p) for p in (_HERE, Path(S.__file__), Path(LK.__file__), _SCRIPTS / "59_evidence_support.py",
                                         _SCRIPTS / "evidence" / "protein_accept.py", _SCRIPTS / "evidence" / "b6_hsp_project.py",
                                         _SCRIPTS / "evidence" / "est_phase_project.py")}
    try:
        import duckdb
        duckdb_version = duckdb.__version__
    except ImportError:  # pragma: no cover
        duckdb_version = "missing"
    status = "ABORT"
    out.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(prefix=out.name + ".partial-", dir=out.parent)
    os.close(handle)
    os.unlink(tmp)
    scratch, spill = tmp + ".stg", tmp + ".spill"
    con = None
    try:
        con = duckdb.connect(tmp)
        con.execute(f"SET memory_limit = {lit(args.memory_limit)}")
        con.execute(f"SET threads = {int(args.threads)}")
        con.execute(f"SET temp_directory = {lit(spill)}")
        con.execute("SET preserve_insertion_order = false")
        con.execute(f"ATTACH {lit(scratch)} AS stg")
        st = Stage(con)
        cursor = con.cursor()
        if source == "protein":
            prow = load_protein(ctx, st, cursor, rows, args)
        else:
            for run_idx, row in enumerate(rows):                  # run by run, one streaming pass each
                load_read_run(ctx, st, run_idx, row, args, versions)
        st.flush()
        dup = con.execute("SELECT member_id FROM stg.member GROUP BY member_id HAVING count(*) > 1 LIMIT 3").fetchall()
        require(not dup, f"a read/query forms several groups (duplicate primary records?): {[d[0] for d in dup]}")
        # every input is verified and hashed; no final row exists yet
        ctx["provenance_sha256"] = provenance_hash(ctx, args, argv, tool_shas, code, duckdb_version, rows, excluded_rows)
        if source != "protein":
            assign_read_units(ctx, st, cursor)
            st.flush()
            run_library_qc(ctx, cursor)
            con.execute("""INSERT INTO stg.pl SELECT m.member_id, 'audit', a.kind, a.chrom, a.start0, a.end0, a.mapq
                           FROM stg.apl a JOIN stg.member m ON m.run_idx = a.run_idx AND m.read = a.read
                           WHERE m.training AND a.kind <> 'unmapped'""")
            decide_all(ctx, st, cursor, audited=True)               # steps 2-3 (may raise ExhaustivenessError)
        else:
            st.flush()
            decide_all(ctx, st, cursor, audited=False, protein_cap=MINIPROT_MAX_PLACEMENTS)
            load_hsp(ctx, st, cursor, prow, args)
        st.flush()
        cursor.close()
        S.create_schema(con)
        write_final(ctx, con)                                       # steps 4-5 (rule 4(c))
        con.executemany("INSERT INTO c0_meta VALUES (?, ?)", [("species", species), ("source", source)])
        sql_gate(con, ctx)                                          # step 7
        con.execute("DETACH stg")
        con.executemany("INSERT INTO c0_provenance VALUES (?, ?, ?, ?)",
                        [(species, source, k, json.dumps(v, sort_keys=True, default=str)) for k, v in sorted(ctx["provenance"].items())]
                        + [(species, source, "provenance_sha256", ctx["provenance_sha256"])])
        con.executemany("INSERT INTO c0_qc VALUES (?, ?, ?, ?)",
                        [(species, source, k, float(v)) for k, v in sorted(ctx["qc"].items()) if isinstance(v, (int, float))])
        S.verify_schema(con)
        con.execute("CHECKPOINT")
        con.close()
        con = None
        require(ctx["qc"]["leakage_nonzero"] == 0,
                f"QC gate (A46 rule 13): leakage count {ctx['qc']['leakage_nonzero']} != 0; nothing published")
        require(ctx["qc"]["hsp_phase_conflict_among_retained_recheck"] == 0,
                f"QC gate (A46 rule 6): {ctx['qc']['hsp_phase_conflict_among_retained_recheck']} retained route-B HSP "
                "pair(s) of one molecule and protein source still disagree on phase; nothing published")
        require(ctx["qc"]["hsp_phase_conflict_among_excluded_recheck"] == 0,
                f"QC gate (A46 rule 6, excluded population): {ctx['qc']['hsp_phase_conflict_among_excluded_recheck']} "
                "leakage-excluded route-B HSP pair(s) of one molecule and protein source still disagree on phase; "
                "nothing published")
        require(ctx["qc"]["completeness_codes_policy_violation"] == 0,
                f"QC gate (A16): {ctx['qc']['completeness_codes_policy_violation']} accepted alignment row(s) contradict "
                f"the recorded completeness-code policy {COMPLETENESS_CODES_POLICY['policy']!r}; nothing published")
        os.link(tmp, out)
        with open(str(out) + ".PROVENANCE.txt", "x") as fh:
            fh.write(f"date\t{datetime.datetime.now().isoformat(timespec='seconds')}\n")
            fh.write(f"provenance_sha256\t{ctx['provenance_sha256']}\n")
            for k, v in sorted(ctx["provenance"].items()):
                fh.write(f"{k}\t{json.dumps(v, sort_keys=True, default=str)}\n")
            fh.write(f"output\t{out}\noutput_sha256\t{digest(out)}\n")
        status = "PASS"
    finally:
        if con is not None:
            con.close()
        for p in (tmp, tmp + ".wal", scratch, scratch + ".wal"):
            Path(p).unlink(missing_ok=True)
        shutil.rmtree(spill, ignore_errors=True)
        record = {"species": species, "source": source, "status": status, "schema_version": S.C0_SCHEMA_VERSION,
                  "provenance_sha256": ctx.get("provenance_sha256"), "qc": dict(ctx["qc"]), "leakage_qc": leakage.qc,
                  "excluded_role_rows": [f"{r['dataset']}/{r['run']}" for r in excluded_rows],
                  "output": str(out) if status == "PASS" else None,
                  "output_sha256": digest(out) if status == "PASS" else None}
        with open(qc_path, "x") as fh:
            json.dump(record, fh, indent=1, sort_keys=True, default=str)
    return record


def intervals_sha256(intervals):
    return sid("intervals", sorted((iv.chrom, iv.start0, iv.end0, iv.cause, iv.scope, iv.gene_id or "",
                                    -1 if iv.tile_rn is None else iv.tile_rn) for iv in intervals))


def provenance_hash(ctx, args, argv, tool_shas, code, duckdb_version, rows, excluded_rows):
    prov = {"schema_version": S.C0_SCHEMA_VERSION, "schema_sha256": S.schema_sha256(), "species": ctx["species"],
            "source": ctx["source"], "argv": list(argv), "inputs": ctx["inputs"], "tool_provenance_sha256": tool_shas,
            "code_sha256": code, "duckdb_version": duckdb_version, "seed": args.seed, "audit_n": args.audit_n,
            "role_rows": sorted(f"{r['dataset']}/{r['run']}:{r['role']}" for r in rows),
            "excluded_role_rows": sorted(f"{r['dataset']}/{r['run']}" for r in excluded_rows),
            "leakage_intervals_sha256": ctx["intervals_sha256"],
            # plan v4.1 §2.5: the b6-intervals-v2 manifest (genome-level components + A30-only repair components)
            "interval_manifest_sha256": ctx["interval_manifest_sha256"], "phase_convention": S.PHASE_CONVENTION,
            "pending_author_rulings": {"low_mapq_applies_to_audit_secondaries": LK.LOW_MAPQ_APPLIES_TO_AUDIT_SECONDARIES,
                                       "edge_partial_molecule_wide": LK.EDGE_PARTIAL_MOLECULE_WIDE},
            "miniprot_max_placements": MINIPROT_MAX_PLACEMENTS,
            "exhaustiveness_rule": EXHAUSTIVENESS_RULE,
            "completeness_codes": COMPLETENESS_CODES_POLICY,
            "a16_pcr_linkage": ES.A16_PCR_LINKAGE_DECISION,
            "multi_hsp_rule": {"rule": GEOM.MULTI_HSP_RULE, "applied_by": MULTI_HSP_APPLIED_BY,
                               "scope": "per (molecule_id, protein_source), separately within each population and never "
                                        "across: training = HSPs that are not leakage_excluded; excluded = the "
                                        "leakage_excluded HSPs of the molecule (fix round 6, finding 2)"}}
    ctx["provenance"] = prov
    return hashlib.sha256(json.dumps(prov, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_overlap(path, intervals):
    """--overlap is optional: intervals are computed from --corpus; a supplied TSV must equal them exactly."""
    want = sorted((iv.chrom, iv.start0, iv.end0, iv.cause, iv.scope, iv.gene_id or "", "" if iv.tile_rn is None else str(iv.tile_rn))
                  for iv in intervals)
    got = []
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        require(header == ["chrom", "start0", "end0", "cause", "scope", "gene_id", "tile_rn"], f"{path}: bad header")
        for line in fh:
            c = line.rstrip("\n").split("\t")
            got.append((c[0], int(c[1]), int(c[2]), c[3], c[4], c[5], c[6]))
    require(sorted(got) == want, f"--overlap {path} differs from the intervals recomputed from --corpus")


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--est-roles", required=True)
    ap.add_argument("--species", required=True)
    ap.add_argument("--source", required=True, choices=S.SOURCES)
    ap.add_argument("--align-root", required=True)
    ap.add_argument("--audit-root", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--flags", required=True, help="A22 GeenuFF flag file (rule 4(a) hard flags; A30 is not a cause)")
    ap.add_argument("--build-flags", nargs="+", help="every flag file the corpus build used (A22 + A30), to reproduce its masks")
    ap.add_argument("--overlap", help="optional intervals TSV (b6_leakage_intervals.py) that must equal the recomputation")
    ap.add_argument("--hsp", help="est_protein_hsp input (TSV/parquet, part-B contract); required for --source protein")
    ap.add_argument("--est-c0", help="the same species' published EST C0 DB (HSP molecule ids and states)")
    ap.add_argument("--lineage", help="frozen NCBI lineage TSV (taxid<TAB>lineage;...) for A19 (--source protein)")
    ap.add_argument("--provenance", nargs="+", required=True, help="PROVENANCE.*.json tool records (sha256 sidecar verified)")
    ap.add_argument("--reference", help="reference FASTA (default: corpus build_manifest path; sha256 must match)")
    ap.add_argument("--gff", help="reference GFF (default: corpus build_manifest path; sha256 must match)")
    ap.add_argument("--seed", type=int, required=True,
                    help="corpus build seed (A33 flank / decoy seed strings).  Required: the corpus build_manifest does "
                         "not record it, so it cannot be defaulted or looked up; a wrong seed fails closed when the "
                         "per-tile masking recomputation disagrees with the corpus")
    ap.add_argument("--audit-n", type=int, default=LK.AUDIT_N)
    ap.add_argument("--samtools", default="samtools")
    ap.add_argument("--memory-limit", default="2GB", help="DuckDB memory limit (it spills to <out>.partial-*.spill)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--qc", required=True)
    return ap.parse_args(argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    try:
        record = build(args, argv)
    except (C0Error, LK.LeakageError, S.SchemaError, FileExistsError, OSError) as exc:
        print(f"b6_build_c0 failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": record["status"], "output_sha256": record["output_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
