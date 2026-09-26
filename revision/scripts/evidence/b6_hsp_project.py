#!/usr/bin/env python3
"""Route-B production projection -> `est_protein_hsp` rows (A46 rule 2 and rule 6).

Inputs (one species per call, A46 approved 2026-09-22):
  * DIAMOND RAW13 blastx output per protein source (`orthodb`, `swissprot`), with the
    exact blastx argv (commands.json from est_blastx.py) and the A19-filtered protein
    FASTA that the .dmnd was built from (its sha256 IS `database_sha256`);
  * the A21-screened EST FASTA that was the blastx query;
  * the EST spliced alignment (BAM via samtools, SAM, or PAF with cs:Z long form) --
    only the primary placement (FLAG & 0x904 == 0, PAF tp:A:P) is used;
  * the molecule map TSV written by `b6_est_molecule_map.py` (C0 side, required) with
    columns MOLECULE_MAP_COLUMNS and a `<map>.sha256` sidecar that is verified. Its
    `molecule_id` is used VERBATIM (this script never derives molecule ids), and its
    `molecule_state` / `train_tiles_excluded` are passed through per HSP in the sidecar
    `<out>.molecule_state.tsv` (STATE_COLUMNS) so the C0 builder can gate/cross-check
    leakage (`excluded_heldout` HSPs are emitted here and marked `leakage_excluded` by the
    builder, never dropped silently). The map lists superseded accession versions
    explicitly (`superseded` = true): their HSPs are dropped and counted
    (`superseded_qname_hsps`); likewise members whose C0 primary alignment was NOT
    accepted (`accepted` = false, A18.5): they belong to their molecule for rule 4(b)
    closure but supply no observation, so their HSPs are dropped and counted
    (`rejected_alignment_qname_hsps`, ACCEPTED_RULE). A RAW13 row whose qname is absent from the map is counted
    (`unmapped_qname_hsps`) and ALWAYS fails the run (no override: a name-scheme mismatch
    between the blastx query and the map must never be hidden). `library` /
    `est_accession` are free text (spaces allowed, no tab/CR/LF; GenBank /clone_lib);
    the id fields are whitespace-free;
  * the frozen NCBI lineage table used for the A19 filter.

Per HSP: RAW13 row validation -> whole DIAMOND-reported HSP length (col 9) >= 30 ->
approved contract (`adapt_raw13` + `classify_rich17`, exact codon/frameshift replay
against the real EST and protein residues) -> `est_phase_project.project` through the
primary placement (strand, introns by length mod 3, CIGAR I/D, clipping) -> one row per
merged genomic interval (0-based half-open). Same-species proteins abort the run (A19;
`is_same_species_or_descendant`).

MULTI_HSP_RULE (rule 6: "HSPs whose frame is inconsistent within one EST after projection
are dropped") is NOT applied here (fix round 5, finding 2): which HSPs exist for that test
depends on the rule 4(b) state of the PROTEIN query (rule 3 / rule 6: an excluded protein's
HSPs are removed molecule-wide), which only the protein C0 build knows.  Every projected HSP
is therefore emitted, and `b6_build_c0.load_hsp` resolves per (molecule, protein_source)
among the HSPs that are not leakage_excluded (`MULTI_HSP_APPLIED_BY`; the provenance record
says so).  QC here still reports, per source, what the rule WOULD drop if every HSP were
eligible (`phase_conflict_candidates_pre_eligibility`) and the pooled-scope outcome
(`cross_source_conflict_hsps`): both are information, neither drops a row.

Non-ACGT EST bases: NONCANONICAL_DNA_SCOPE = 'whole_est' (approved contract; the replay in
est_hsp_contract._validate and est_hsp_expanded.interpret requires the WHOLE query to be
ACGT, so an HSP-span-only rule would change that contract). QC counts, per source,
HSPs with non-ACGT inside their span (`noncanonical_dna_in_hsp_span`, dropped under either
rule) and HSPs dropped only because of bases outside their span
(`noncanonical_dna_outside_hsp_span_only`, what a span rule would recover); the top-level
`ests_noncanonical_whole_est_rule` counts mapped ESTs the whole-EST rule drops.

DIAMOND-masked BTOP 'X': MASKED_X_RULE (author decision 2026-09-25, option a). Production
DIAMOND 2.1.24 blastx runs with default masking (tantan), so BTOP letters carry 'X' where
the real EST codon / protein residue is ordinary (Sitalica swissprot: ~5.5% of 290,009
rows failed the strict replay on 'query codon truth' / 'subject truth' for this reason
alone). The contract is called with masked_x=True: a query X is accepted iff the real
oriented codon translates to a standard sense amino acid (not stop); a subject X iff the
subject index is inside the protein; the reconstructed/translated strings replayed by
classify_rich17 carry the TRUE residue, and the BTOP serialization check re-applies the
mask only at the X positions. Every other disagreement (non-X letters, shifts, bounds,
equal runs) still fails the run as before; NONCANONICAL_DNA_SCOPE and the noncanonical-
protein drop are unchanged. QC per source: `masked_x_hsps` (HSPs whose contract replay
relied on >=1 masked X), `masked_x_query_positions` / `masked_x_subject_positions`
(masked BTOP letters accepted), `projected_hsps_with_masked_x` (of those, projected).

hsp_id = sha256 of '<species>\t<protein_source>\t<raw13 line>'; (hsp_id, interval_index)
is unique. est_start/est_end = 0-based half-open span of the HSP on the screened EST record
(orientation is in `frame`). phase = est_phase_project.PHASE_CONVENTION.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from revision.scripts.evidence import est_phase_project as geometry  # noqa: E402
from revision.scripts.evidence.est_hsp_contract import (  # noqa: E402
    LENGTH_CONTRACT, adapt_raw13, classify_rich17)
from revision.scripts.evidence.est_hsp_expanded import CODE  # noqa: E402
from revision.scripts.evidence.est_phase_accept import parse_row  # noqa: E402
from revision.scripts.evidence.est_blastx import SPECIES, subject_taxid, taxonomy  # noqa: E402

COLUMNS = ('hsp_id species library molecule_id est_accession protein_id protein_source '
           'protein_species est_start est_end frame evalue bitscore interval_index chrom '
           'genomic_start0 genomic_end0 strand phase diamond_version database_sha256 '
           'command_sha256 provenance_sha256').split()   # == c0_schema.HSP_COLUMN_NAMES (exact)
# Per-HSP sidecar `<out>.molecule_state.tsv`: the molecule map's gating fields passed through
# (the 23-column contract above is what the C0 builder accepts; it derives the same state by
# joining the EST C0 molecule_member table, and can cross-check it against this sidecar).
STATE_COLUMNS = 'hsp_id molecule_id qname molecule_state train_tiles_excluded'.split()
MOLECULE_MAP_COLUMNS = ('qname member_uid molecule_id library est_accession molecule_state '
                        'train_tiles_excluded superseded accepted').split()
MAP_ID_FIELDS = ('qname member_uid molecule_id molecule_state train_tiles_excluded superseded accepted').split()
MAP_TEXT_FIELDS = ('library', 'est_accession')     # free text: spaces allowed, no tab/CR/LF
MOLECULE_STATES = ('train_track', 'excluded_heldout', 'never')   # c0_schema.MOLECULE_STATES
MOLECULE_ID_RULE = 'verbatim from b6_est_molecule_map.py (not derived here)'
SUPERSEDED_RULE = ('HSPs of qnames the molecule map flags superseded=true are dropped and counted '
                   '(superseded_qname_hsps); any other qname absent from the map fails the run')
ACCEPTED_RULE = ('HSPs of qnames the molecule map flags accepted=false (no A18.5-accepted primary alignment in the '
                 'EST C0: the read is in its molecule for rule 4(b) closure but supplies no observation) are dropped '
                 'and counted (rejected_alignment_qname_hsps); A46 rule 6 projects only through retained observations')
NONCANONICAL_DNA_SCOPE = 'whole_est'   # approved exact-replay contract; see module docstring
MASKED_X_RULE = ('author decision 2026-09-25 (option a): a DIAMOND BTOP X is a masked position; query X '
                 'accepted iff the real codon translates to a sense amino acid (not stop), subject X iff the '
                 'subject index is inside the protein; the TRUE residue is recorded in the replayed strings; '
                 'all other disagreements fail as before (est_hsp_contract masked_x=True)')
MULTI_HSP_APPLIED_BY = 'b6_build_c0.load_hsp'   # after protein-query eligibility; see module docstring
MULTI_HSP_SCOPE = ('per (molecule_id, protein_source) among HSPs that are not leakage_excluded, applied by '
                   + MULTI_HSP_APPLIED_BY + ' (not here)')
SOURCES = ('orthodb', 'swissprot')
RAW13 = 'qseqid qlen sseqid slen qstart qend sstart send length evalue bitscore qframe btop'.split()
FROZEN_FLAGS = (('--frameshift', '15'), ('--evalue', '1e-5'), ('--min-orf', '30'),
                ('--max-target-seqs', '25'))
MIN_DIAMOND = (2, 1, 24)
MIN_AA = 30
PROTEIN_ALPHABET = frozenset(CODE.values())


class SameSpeciesProtein(ValueError):
    """A19: a same-species (or descendant-taxon) protein reached route B."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _open(path):
    path = str(path)
    return gzip.open(path, 'rt') if path.endswith('.gz') else open(path)


def read_fasta(path, wanted=None):
    """{first header token: (full header, uppercase sequence)}; duplicates raise."""
    out, name, header, buf = {}, None, None, []

    def flush():
        if name is not None and (wanted is None or name in wanted):
            if name in out:
                raise ValueError(f'duplicate FASTA record {name!r} in {path}')
            out[name] = (header, ''.join(buf).upper())

    with _open(path) as f:
        for line in f:
            if line.startswith('>'):
                flush()
                header = line[1:].rstrip('\n')
                if not header.split():
                    raise ValueError(f'empty FASTA header in {path}')
                name, buf = header.split()[0], []
            else:
                buf.append(line.strip())
    flush()
    return out


# ---------------------------------------------------------------- molecules (A11, C0 side)

def verify_sidecar(path):
    """`<path>.sha256` holds '<hex>' or sha256sum's '<hex>  <name>'; it must match."""
    sidecar = Path(str(path) + '.sha256')
    if not sidecar.exists():
        raise ValueError(f'{path}: missing sha256 sidecar {sidecar.name}')
    tokens = sidecar.read_text().split()
    if not tokens or not re.fullmatch('[0-9a-f]{64}', tokens[0]) or len(tokens) > 2 \
            or (len(tokens) == 2 and Path(tokens[1].lstrip('*')).name != Path(path).name):
        raise ValueError(f'{sidecar}: not a sha256 sidecar for {Path(path).name}')
    actual = sha256_file(path)
    if actual != tokens[0]:
        raise ValueError(f'{path}: sha256 {actual} != sidecar {tokens[0]}')
    return actual


def load_molecule_map(path):
    verify_sidecar(path)
    with open(path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        if reader.fieldnames is None or any(c not in reader.fieldnames for c in MOLECULE_MAP_COLUMNS):
            raise ValueError(f'molecule map needs columns {MOLECULE_MAP_COLUMNS}')
        rows = list(reader)
    return validate_molecule_map(rows)


def _bool(value, name='train_tiles_excluded'):
    v = str(value).strip().lower()
    if v not in ('true', 'false'):
        raise ValueError(f'{name} must be true/false, got {value!r}')
    return v == 'true'


def _bad_map_field(row):
    for k in MAP_ID_FIELDS:
        if not row.get(k) or any(c.isspace() for c in row[k]):
            return k
    for k in MAP_TEXT_FIELDS:
        v = row.get(k)
        if not v or v != v.strip() or any(c in v for c in '\t\r\n'):
            return k
    return None


def validate_molecule_map(rows):
    """{qname: row} (every row, `superseded` as bool); molecule_id/state are authoritative and used verbatim."""
    by_qname, per_molecule = {}, defaultdict(set)
    for row in rows:
        bad = _bad_map_field(row)
        if bad is not None:
            raise ValueError(f'invalid molecule map row ({bad}): {row}')
        if row['qname'] in by_qname:
            raise ValueError(f'qname mapped twice: {row["qname"]}')
        if not re.fullmatch('[0-9a-f]{64}', row['molecule_id']):
            raise ValueError(f'molecule_id is not a sha256: {row["molecule_id"]}')
        if row['molecule_state'] not in MOLECULE_STATES:
            raise ValueError(f'unknown molecule_state {row["molecule_state"]!r}')
        row = dict(row, train_tiles_excluded=_bool(row['train_tiles_excluded']),
                   superseded=_bool(row['superseded'], 'superseded'), accepted=_bool(row['accepted'], 'accepted'))
        by_qname[row['qname']] = row
        per_molecule[row['molecule_id']].add((row['library'], row['molecule_state'],
                                              row['train_tiles_excluded']))
    for molecule, values in per_molecule.items():
        if len(values) != 1:
            raise ValueError(f'molecule {molecule} has inconsistent library/state/train_tiles_excluded')
    return by_qname


# ---------------------------------------------------------------- placements

CIGAR_RE = re.compile(r'([0-9]+)([MIDNSHP=X])')
CS_RE = re.compile(r'(=[ACGTNacgtn]+|:[0-9]+|\*[a-zA-Z][a-zA-Z]|\+[a-zA-Z]+|-[a-zA-Z]+|~[a-zA-Z]{2}([0-9]+)[a-zA-Z]{2})')


def _tag(fields, name):
    for field in fields:
        if field.startswith(name + ':'):
            return field.split(':', 2)[2]
    return None


def read_sam_primary(lines, wanted):
    """Primary placements from SAM text; contig lengths from @SQ."""
    lengths, out = {}, {}
    for line in lines:
        if line.startswith('@'):
            if line.startswith('@SQ'):
                tags = dict(t.split(':', 1) for t in line.rstrip('\n').split('\t')[1:] if ':' in t)
                lengths[tags['SN']] = int(tags['LN'])
            continue
        c = line.rstrip('\n').split('\t')
        if len(c) < 11:
            raise ValueError('malformed SAM record')
        flag = int(c[1])
        if flag & 0x904 or c[0] not in wanted:
            continue
        if c[0] in out:
            raise ValueError(f'two primary placements for {c[0]}')
        if c[2] not in lengths:
            raise ValueError(f'contig {c[2]} missing from SAM header')
        out[c[0]] = dict(flag=flag, contig=c[2], pos1=int(c[3]), cigar=c[5],
                         contig_length=lengths[c[2]], seq=None if c[9] == '*' else c[9],
                         ts=_tag(c[11:], 'ts'), mapq=int(c[4]))
    return out


def cs_to_cigar(cs):
    ops = []
    pos = 0
    for m in CS_RE.finditer(cs):
        if m.start() != pos:
            raise ValueError(f'unparsed cs at {pos}')
        pos = m.end()
        token = m.group(1)
        kind = token[0]
        if kind == '=':
            op, n = 'M', len(token) - 1
        elif kind == ':':
            op, n = 'M', int(token[1:])
        elif kind == '*':
            op, n = 'M', 1
        elif kind == '+':
            op, n = 'I', len(token) - 1
        elif kind == '-':
            op, n = 'D', len(token) - 1
        else:
            op, n = 'N', int(m.group(2))
        if ops and ops[-1][1] == op:
            ops[-1][0] += n
        else:
            ops.append([n, op])
    if pos != len(cs) or not ops:
        raise ValueError('invalid cs tag')
    return ops


def read_paf_primary(lines, wanted):
    out = {}
    for line in lines:
        c = line.rstrip('\n').split('\t')
        if len(c) < 12:
            raise ValueError('malformed PAF record')
        if c[0] not in wanted or _tag(c[12:], 'tp') != 'P':
            continue
        if c[0] in out:
            raise ValueError(f'two primary placements for {c[0]}')
        cs = _tag(c[12:], 'cs')
        if cs is None:
            raise ValueError('PAF placement lacks cs:Z (frozen command uses --cs=long)')
        qlen, qs, qe = int(c[1]), int(c[2]), int(c[3])
        left, right = (qs, qlen - qe) if c[4] == '+' else (qlen - qe, qs)
        ops = ([[left, 'S']] if left else []) + cs_to_cigar(cs) + ([[right, 'S']] if right else [])
        out[c[0]] = dict(flag=0 if c[4] == '+' else 16, contig=c[5], pos1=int(c[7]) + 1,
                         cigar=''.join(f'{n}{op}' for n, op in ops), contig_length=int(c[6]),
                         seq=None, ts=_tag(c[12:], 'ts'), mapq=int(c[11]))
    return out


def load_placements(path, wanted, samtools=None):
    name = str(path)
    if name.endswith('.bam'):
        if not samtools:
            raise ValueError('--samtools is required to read a BAM')
        proc = subprocess.Popen([samtools, 'view', '-h', '-F', '0x904', name],
                                stdout=subprocess.PIPE, text=True)
        try:
            out = read_sam_primary(proc.stdout, wanted)
        finally:
            proc.stdout.close()
            if proc.wait() != 0:
                raise ValueError(f'samtools view failed on {name}')
        return out
    reader = read_paf_primary if re.search(r'\.paf(\.gz)?$', name) else read_sam_primary
    with _open(path) as f:
        return reader(f, wanted)


# ---------------------------------------------------------------- blastx provenance

def blastx_argv(path):
    value = json.loads(Path(path).read_text())
    argvs = value if value and isinstance(value[0], list) else [value]
    found = [a for a in argvs if len(a) > 1 and a[1] == 'blastx']
    if len(found) != 1:
        raise ValueError(f'{path}: expected exactly one diamond blastx argv')
    return found[0]


def check_frozen_command(argv, est_fasta):
    for flag, value in FROZEN_FLAGS:
        if flag not in argv or argv[argv.index(flag) + 1] != value:
            raise ValueError(f'blastx command lacks frozen {flag} {value} (A46 rule 6)')
    i = argv.index('--outfmt') if '--outfmt' in argv else -1
    if i < 0 or argv[i + 1:i + 2 + len(RAW13)] != ['6'] + RAW13:
        raise ValueError('blastx command is not the RAW13 outfmt')
    if '--query' not in argv or Path(argv[argv.index('--query') + 1]).name != Path(est_fasta).name:
        raise ValueError('blastx query is not the screened EST FASTA given here')


def check_diamond_version(version):
    m = re.fullmatch(r'(\d+)\.(\d+)\.(\d+)(?:\.\d+)?', version or '')
    if not m or tuple(map(int, m.groups())) < MIN_DIAMOND:
        raise ValueError(f'diamond {version!r} < 2.1.24 (A46 rule 6)')


_FROZEN_LINEAGE = {}


def use_frozen_lineage(path):
    """Register the frozen NCBI lineage table (est_blastx lineage.tsv) for A19 checks.

    Returns its sha256 so callers can record which table decided.
    """
    digest = sha256_file(path)
    _FROZEN_LINEAGE.clear()
    _FROZEN_LINEAGE.update(path=str(path), sha256=digest, table=taxonomy.read_lineage(str(path)))
    return digest


def is_same_species_or_descendant(taxid, species, lineage=None):
    """A19: True iff protein `taxid` is the training `species` taxon or lies below it.

    `lineage` is a {taxid: [taxid, parent, ..., root]} table or a path to the frozen
    lineage TSV; None uses the table registered with use_frozen_lineage(). Fails closed:
    an unknown species, a non-numeric taxid or a taxid absent from the frozen table raises
    ValueError (never silently False). Importable by the C0 builder.
    """
    if species not in SPECIES:
        raise ValueError(f'unknown training species {species!r}')
    if isinstance(lineage, (str, os.PathLike)):
        lineage = taxonomy.read_lineage(str(lineage))
    elif lineage is None:
        if not _FROZEN_LINEAGE:
            raise ValueError('no frozen lineage table: pass lineage= or call use_frozen_lineage()')
        lineage = _FROZEN_LINEAGE['table']
    text = str(taxid).strip()
    if isinstance(taxid, bool) or not text.isdigit() or int(text) <= 0:
        raise ValueError(f'protein taxid {taxid!r} is not a positive integer')
    taxid, root = int(text), SPECIES[species]
    if taxid not in lineage:
        raise ValueError(f'protein taxid {taxid} missing from the frozen lineage')
    return taxid == root or root in lineage[taxid]


def check_a19(taxid, species, lineage):
    if is_same_species_or_descendant(taxid, species, lineage):
        raise SameSpeciesProtein(f'A19: protein taxid {taxid} is {species} ({SPECIES[species]}) or below it')


# ---------------------------------------------------------------- projection

def _pair(value):
    source, sep, path = value.partition('=')
    if not sep or source not in SOURCES or not path:
        raise argparse.ArgumentTypeError(f'expected SOURCE=PATH with SOURCE in {SOURCES}')
    return source, Path(path)


ACGT = frozenset('ACGT')


def hsp_span0(row):
    """0-based half-open span of the HSP on the ORIGINAL (unoriented) EST record."""
    return min(row['qs'], row['qe']) - 1, max(row['qs'], row['qe'])


def project_row(fields, source, species, est, proteins, placements, molecules, lineage,
                est_sha256, qc):
    """One RAW13 row -> dict(molecule_id, hsp_id, meta, intervals) or None (dropped)."""
    row = parse_row(0, '\t'.join(fields))
    qc['raw_hsps'] += 1
    if row['length'] < MIN_AA:
        qc['drop_below_30aa'] += 1
        return None
    header, protein = proteins[row['sseqid']]
    taxid = subject_taxid(header, source)
    check_a19(taxid, species, lineage)
    if row['qseqid'] not in molecules:
        qc['unmapped_qname_hsps'] += 1
        return None
    if molecules[row['qseqid']]['superseded']:
        qc['superseded_qname_hsps'] += 1
        return None
    if not molecules[row['qseqid']]['accepted']:          # ACCEPTED_RULE: rejected C0 primary -> no observation
        qc['rejected_alignment_qname_hsps'] += 1
        return None
    if row['qseqid'] not in est:
        raise ValueError(f'blastx query {row["qseqid"]} is not in the screened EST FASTA')
    dna = est[row['qseqid']][1]
    if len(dna) != row['qlen'] or len(protein) != row['slen']:
        raise ValueError(f'length mismatch for {row["qseqid"]}/{row["sseqid"]}')
    if not set(protein) <= PROTEIN_ALPHABET:
        qc['drop_noncanonical_protein'] += 1
        return None
    if not set(dna) <= ACGT:
        start0, end0 = hsp_span0(row)
        if not set(dna[start0:end0]) <= ACGT:
            qc['drop_noncanonical_dna_in_hsp_span'] += 1
            return None
        # NONCANONICAL_DNA_SCOPE == 'whole_est': the approved replay needs an all-ACGT query.
        qc['drop_noncanonical_dna_outside_hsp_span_only'] += 1
        return None
    # MASKED_X_RULE: DIAMOND default masking puts 'X' into BTOP letters (see module docstring).
    accepted = classify_rich17(adapt_raw13(fields, dna, protein, masked_x=True), dna, protein,
                               masked_x=True)
    if not accepted['accepted_by_hsp_rule']:
        raise ValueError('HSP contract disagrees with the RAW13 length column')
    masked = accepted['masked_x']
    if masked['query'] or masked['subject']:
        qc['masked_x_hsps'] += 1
        qc['masked_x_query_positions'] += masked['query']
        qc['masked_x_subject_positions'] += masked['subject']
    placement = placements.get(row['qseqid'])
    if placement is None:
        qc['drop_no_primary_placement'] += 1
        return None
    hsp_id = sha256_text(f'{species}\t{source}\t' + '\t'.join(fields))
    identity = dict(member=row['qseqid'], source_sha256=est_sha256, source_start0=0,
                    source_end0=len(dna), sequence_sha256=sha256_text(dna))
    hsp = dict(identity=identity, id=hsp_id, qstart=row['qs'], qend=row['qe'], frame=row['frame'],
               producer_contract_sha256=sha256_text(LENGTH_CONTRACT),
               label_convention=geometry.LABEL_CONVENTION)
    projected = geometry.project(hsp, accepted['codon_events'],
                                 dict(placement, identity=identity, sequence=dna))
    if projected['dropped']:
        qc['drop_' + projected['dropped']] += 1
        return None
    qc['projected_hsps'] += 1
    if masked['query'] or masked['subject']:
        qc['projected_hsps_with_masked_x'] += 1
    m = molecules[row['qseqid']]
    est_start, est_end = hsp_span0(row)
    meta = dict(hsp_id=hsp_id, species=species, library=m['library'], molecule_id=m['molecule_id'],
                est_accession=m['est_accession'], protein_id=row['sseqid'], protein_source=source,
                protein_species=str(taxid), est_start=est_start, est_end=est_end,
                frame=row['frame'], evalue=fields[9], bitscore=fields[10], qname=row['qseqid'],
                molecule_state=m['molecule_state'],
                train_tiles_excluded='true' if m['train_tiles_excluded'] else 'false')
    return dict(molecule_id=m['molecule_id'], hsp_id=hsp_id, meta=meta,
                intervals=projected['intervals'])


def iter_raw13(path):
    with _open(path) as f:
        for number, line in enumerate(f, 1):
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 13:
                raise ValueError(f'{path}:{number}: expected 13 RAW13 fields')
            yield fields


def run(args):
    if args.species not in SPECIES:
        raise ValueError(f'unknown training species {args.species}')
    check_diamond_version(args.diamond_version)
    raw = dict(args.raw13)
    fastas = dict(args.protein_fasta)
    commands = dict(args.blastx_command)
    if not raw or set(raw) != set(fastas) or set(raw) != set(commands):
        raise ValueError('each source needs --raw13, --protein-fasta and --blastx-command')
    for path in (args.out, args.qc, args.provenance, args.state_out):
        if path.exists():
            raise FileExistsError(path)
    molecule_map_sha256 = verify_sidecar(args.molecule_map)
    molecules = load_molecule_map(args.molecule_map)
    current = {q for q, r in molecules.items() if not r['superseded'] and r['accepted']}
    est = read_fasta(args.est_fasta, wanted=current)
    ests_noncanonical = sum(1 for _, seq in est.values() if not set(seq) <= ACGT)
    placements = load_placements(args.alignment, current, args.samtools)
    lineage = taxonomy.read_lineage(str(args.lineage))
    sources = {}
    for source in sorted(raw):
        argv = blastx_argv(commands[source])
        check_frozen_command(argv, args.est_fasta)
        sources[source] = dict(raw13=str(raw[source]), raw13_sha256=sha256_file(raw[source]),
                               protein_fasta=str(fastas[source]),
                               database_sha256=sha256_file(fastas[source]), command=argv,
                               command_sha256=sha256_text(json.dumps(argv, separators=(',', ':'))))
    code = {name: sha256_file(Path(__file__).parent / name) for name in
            ('b6_hsp_project.py', 'est_phase_project.py', 'est_hsp_contract.py',
             'est_hsp_expanded.py', 'est_phase_accept.py')}
    provenance = dict(schema='est_protein_hsp_v1', columns=COLUMNS, species=args.species,
                      approval=geometry.A46_APPROVAL, multi_hsp_rule=geometry.MULTI_HSP_RULE,
                      multi_hsp_rule_applied_by=MULTI_HSP_APPLIED_BY, multi_hsp_scope=MULTI_HSP_SCOPE,
                      phase_convention=geometry.PHASE_CONVENTION, length_contract=LENGTH_CONTRACT,
                      min_aa=MIN_AA, molecule_id_rule=MOLECULE_ID_RULE,
                      diamond_version=args.diamond_version, sources=sources,
                      est_fasta=str(args.est_fasta), est_fasta_sha256=sha256_file(args.est_fasta),
                      alignment=str(args.alignment), alignment_sha256=sha256_file(args.alignment),
                      molecule_map=str(args.molecule_map), molecule_map_sha256=molecule_map_sha256,
                      molecule_map_columns=MOLECULE_MAP_COLUMNS, superseded_rule=SUPERSEDED_RULE,
                      accepted_rule=ACCEPTED_RULE, state_columns=STATE_COLUMNS,
                      noncanonical_dna_scope=NONCANONICAL_DNA_SCOPE,
                      masked_x_rule=MASKED_X_RULE,
                      lineage_sha256=sha256_file(args.lineage), code_sha256=code)
    blob = json.dumps(provenance, sort_keys=True, separators=(',', ':')).encode()
    provenance_sha256 = hashlib.sha256(blob).hexdigest()

    qc = {source: Counter() for source in sources}
    scratch = tempfile.NamedTemporaryFile(dir=args.out.parent, prefix='.b6_hsp_', suffix='.sqlite',
                                          delete=False)
    scratch.close()
    db = sqlite3.connect(scratch.name)
    partial = args.out.with_name(args.out.name + '.partial')
    state_partial = args.state_out.with_name(args.state_out.name + '.partial')
    try:
        db.execute('CREATE TABLE hsp(molecule_id TEXT, hsp_id TEXT PRIMARY KEY, payload TEXT)')
        for source in sorted(sources):
            wanted = {fields[2] for fields in iter_raw13(raw[source])}
            proteins = read_fasta(fastas[source], wanted=wanted)
            missing = wanted - set(proteins)
            if missing:
                raise ValueError(f'{source}: {len(missing)} subjects absent from the protein FASTA')
            for fields in iter_raw13(raw[source]):
                item = project_row(fields, source, args.species, est, proteins, placements,
                                   molecules, lineage, provenance['est_fasta_sha256'], qc[source])
                if item:
                    try:
                        db.execute('INSERT INTO hsp VALUES (?,?,?)',
                                   (item['molecule_id'], item['hsp_id'],
                                    json.dumps(dict(meta=item['meta'], intervals=item['intervals']))))
                    except sqlite3.IntegrityError:
                        raise ValueError(f'duplicate RAW13 row {item["hsp_id"]}') from None
            del proteins
        unmapped = sum(c['unmapped_qname_hsps'] for c in qc.values())
        if unmapped:
            raise ValueError(f'{unmapped} HSPs have a qname absent from the molecule map '
                             f'(by source: { {s: c["unmapped_qname_hsps"] for s, c in qc.items()} }); '
                             'the blastx query and the map disagree on names (only superseded=true qnames '
                             'of the map may be dropped)')
        molecules_with_rows = set()
        with open(partial, 'x', newline='') as out, open(state_partial, 'x', newline='') as state_out:
            writer = csv.writer(out, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE, quotechar=None)
            writer.writerow(COLUMNS)
            state_writer = csv.writer(state_out, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE,
                                      quotechar=None)
            state_writer.writerow(STATE_COLUMNS)
            group, current = [], None
            cursor = db.execute('SELECT molecule_id, hsp_id, payload FROM hsp ORDER BY molecule_id, hsp_id')
            for molecule, _, payload in itertools.chain(cursor, [(None, None, None)]):
                if molecule != current and group:
                    _emit(group, (writer, state_writer), qc, args, sources, provenance_sha256,
                          molecules_with_rows)
                    group = []
                current = molecule
                if payload is not None:
                    group.append(json.loads(payload))
        report = dict(species=args.species, provenance_sha256=provenance_sha256,
                      multi_hsp_rule=geometry.MULTI_HSP_RULE, multi_hsp_rule_applied_by=MULTI_HSP_APPLIED_BY,
                      multi_hsp_scope=MULTI_HSP_SCOPE,
                      by_source={s: dict(sorted(c.items())) for s, c in qc.items()},
                      molecules_with_phase=len(molecules_with_rows),
                      noncanonical_dna_scope=NONCANONICAL_DNA_SCOPE,
                      masked_x_rule=MASKED_X_RULE,
                      ests_mapped=len(est), ests_noncanonical_whole_est_rule=ests_noncanonical,
                      superseded_rule=SUPERSEDED_RULE, accepted_rule=ACCEPTED_RULE,
                      note=('no HSP is dropped for phase conflict here: phase_conflict_candidates_pre_eligibility '
                            'counts HSPs the per-source MULTI_HSP_RULE would drop if every HSP were eligible, '
                            'cross_source_conflict_hsps those a pooled resolution would drop (swissprot is the A30 '
                            'sensitivity arm); ' + MULTI_HSP_APPLIED_BY + ' applies the rule after eligibility'))
        args.provenance.write_bytes(blob + b'\n')
        with open(args.qc, 'x') as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write('\n')
        os.replace(state_partial, args.state_out)
        os.replace(partial, args.out)
    finally:
        db.close()
        os.unlink(scratch.name)
        for leftover in (partial, state_partial):
            if leftover.exists():
                leftover.unlink()
    return report


def _emit(group, writers, qc, args, sources, provenance_sha256, molecules_with_rows):
    """One molecule: every projected HSP is written; MULTI_HSP_RULE outcomes are QC only (see module docstring)."""
    writer, state_writer = writers
    items = [dict(key=g['meta']['hsp_id'], scope=g['meta']['protein_source'],
                  intervals=g['intervals']) for g in group]
    would_drop = set(geometry.resolve_hsps(items)['dropped'])
    pooled_kept = set(geometry.resolve_hsps([dict(i, scope='pooled') for i in items])['kept'])
    for g in group:
        meta, source = g['meta'], g['meta']['protein_source']
        if meta['hsp_id'] in would_drop:
            qc[source]['phase_conflict_candidates_pre_eligibility'] += 1
        if meta['hsp_id'] not in pooled_kept:
            qc[source]['cross_source_conflict_hsps'] += 1
        qc[source]['kept_hsps'] += 1
        molecules_with_rows.add(meta['molecule_id'])
        qc[source]['kept_hsps_' + meta['molecule_state']] += 1
        state_writer.writerow([meta['hsp_id'], meta['molecule_id'], meta['qname'],
                               meta['molecule_state'], meta['train_tiles_excluded']])
        for index, iv in enumerate(sorted(g['intervals'], key=lambda i: (i['contig'], i['start0']))):
            qc[source]['rows'] += 1
            qc[source]['bases'] += iv['end0'] - iv['start0']
            writer.writerow([meta['hsp_id'], meta['species'], meta['library'], meta['molecule_id'],
                             meta['est_accession'], meta['protein_id'], meta['protein_source'],
                             meta['protein_species'], meta['est_start'], meta['est_end'],
                             meta['frame'], meta['evalue'], meta['bitscore'], index, iv['contig'],
                             iv['start0'], iv['end0'], iv['strand'], iv['phase'],
                             args.diamond_version, sources[source]['database_sha256'],
                             sources[source]['command_sha256'], provenance_sha256])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--species', required=True)
    p.add_argument('--raw13', type=_pair, action='append', default=[], metavar='SOURCE=PATH')
    p.add_argument('--protein-fasta', type=_pair, action='append', default=[], metavar='SOURCE=PATH')
    p.add_argument('--blastx-command', type=_pair, action='append', default=[], metavar='SOURCE=PATH',
                   help='commands.json written by est_blastx.py for that source')
    p.add_argument('--diamond-version', required=True)
    p.add_argument('--est-fasta', type=Path, required=True)
    p.add_argument('--alignment', type=Path, required=True, help='.bam, .sam[.gz] or .paf[.gz]')
    p.add_argument('--samtools')
    p.add_argument('--molecule-map', type=Path, required=True,
                   help='b6_est_molecule_map.py TSV (sidecar <map>.sha256 is verified)')
    p.add_argument('--lineage', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--qc', type=Path, required=True)
    p.add_argument('--provenance', type=Path)
    args = p.parse_args(argv)
    args.provenance = args.provenance or args.out.with_name(args.out.name + '.provenance.json')
    args.state_out = args.out.with_name(args.out.name + '.molecule_state.tsv')
    report = run(args)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
