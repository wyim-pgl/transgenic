"""Route-B phase geometry and, since A46 approval (2026-09-22), production projection.

`project_codons` / `project_observations` remain the diagnostic geometry API.
`project` (one accepted HSP -> merged genomic intervals) and `resolve_hsps`
(A46 rule 6 multi-HSP rule) are the production entrypoints, unblocked by the
A46 approval recorded in `A46_APPROVAL`. Labels MUST come from the approved
HSP contract (`est_hsp_contract.classify_rich17` codon events), never from
signed-frame extrapolation. Acceptance (whole reported HSP length >= 30 aa),
provenance binding and A11 molecule closure are the caller's job
(`b6_hsp_project.py`).
"""
import hashlib
import re
from bisect import bisect_right


LABEL_CONVENTION = 'c0_first_codon_base'


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'invalid {name}')
    return value


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)


def query_reference_runs(placement):
    """Return compact full-BAM-oriented u -> g runs and unmapped CIGAR spans.

    Input sequence is the full original screened FASTA string; seq is optional
    SAM/BAM-oriented SEQ (None or '*' if absent). Identity binds member, source
    file SHA256, source half-open interval, and uppercase sequence SHA256.
    The caller must bind contig/length to its validated reference dictionary.
    H advances this FULL cursor, never the stored SAM SEQ cursor.
    """
    p = placement
    sequence = p['sequence']
    if not isinstance(sequence, str) or not re.fullmatch('[ACGTRYSWKMBDHVNacgtryswkmbdhvn]+', sequence):
        raise ValueError('full screened sequence required')
    sequence = sequence.upper()
    length = len(sequence)
    identity = p['identity']
    if (not isinstance(identity.get('member'), str) or not identity['member']
            or not _hash(identity.get('source_sha256'))
            or identity.get('sequence_sha256') != hashlib.sha256(sequence.encode()).hexdigest()):
        raise ValueError('screened identity mismatch')
    a = _integer(identity.get('source_start0'), 'source identity start')
    b = _integer(identity.get('source_end0'), 'source identity end')
    if b-a != length:
        raise ValueError('source identity length mismatch')
    flag = _integer(p['flag'], 'flag')
    if flag > 65535:
        raise ValueError('invalid flag')
    if flag & 4:
        raise ValueError('unmapped placement')
    if p.get('ts') not in (None, '+', '-'):
        raise ValueError('invalid ts')
    g = _integer(p['pos1'], 'position', 1)-1
    contig_length = _integer(p['contig_length'], 'contig length', 1)
    if not isinstance(p['contig'], str) or p['contig'] in ('', '*') or g >= contig_length:
        raise ValueError('invalid contig position')
    cigar = p['cigar']
    if not isinstance(cigar, str) or not re.fullmatch(r'(?:[1-9][0-9]*[MIDNSHP=X])+', cigar):
        raise ValueError('invalid CIGAR')
    ops = [(int(n), op) for n, op in re.findall(r'([0-9]+)([MIDNSHP=X])', cigar)]
    letters = ''.join(op for _, op in ops)
    if not any(op in 'M=X' for _, op in ops):
        raise ValueError('no mapped CIGAR bases')
    if not re.fullmatch(r'H?S?[MIDNP=X]+S?H?', letters):
        raise ValueError('invalid terminal clipping')
    full_length = sum(n for n, op in ops if op in 'MISH=X')
    hard = sum(n for n, op in ops if op == 'H')
    if full_length != length:
        raise ValueError('full query length disagrees with CIGAR')
    seq = p.get('seq')
    if seq is not None and seq != '*':
        if not isinstance(seq, str) or len(seq) != length-hard:
            raise ValueError('SEQ length mismatch')
        oriented = sequence
        if flag & 16:
            oriented = sequence.translate(str.maketrans('ACGTRYSWKMBDHVN', 'TGCAYRSWMKVHDBN'))[::-1]
        left = ops[0][0] if ops[0][1] == 'H' else 0
        right = ops[-1][0] if ops[-1][1] == 'H' else 0
        if seq.upper() != oriented[left:length-right]:
            raise ValueError('SEQ identity mismatch')
    runs, gaps = [], []
    u = 0
    for n, op in ops:
        du = n if op in 'MISH=X' else 0
        dg = n if op in 'MDN=X' else 0
        row = dict(u0=u, u1=u+du, g0=g, g1=g+dg, op=op)
        (runs if op in 'M=X' else gaps).append(row)
        u += du
        g += dg
    if g > contig_length:
        raise ValueError('CIGAR exceeds contig bounds')
    return dict(runs=runs, gaps=gaps, query_length=length)


def _diagnostic(reasons, segments=(), **metadata):
    return dict(manifest=dict(schema='est_phase_geometry_v1', training_eligible=False,
                             input_label_convention=LABEL_CONVENTION,
                             channels=['+0', '+1', '+2', '-0', '-1', '-2']),
                reasons=reasons, segments=list(segments), **metadata)


def project_codons(hsp, codon_events, placement):
    """Project one producer-labelled HSP into compact diagnostic segments.

    Each event is an affine LABELLED run: q0 (original screened coordinate),
    length, direction (+1/-1), c0 (ordinal at q0). Labels advance by one in
    coding order. Split at producer discontinuities; never infer labels from
    frame. Single-base runs can encode noncontiguous codon tuples. Event index
    is retained. Producer contract SHA is a provenance reference, NOT approval.
    Self overlap and motif conflict refuse the entire diagnostic, not a chosen
    scientific rejection/masking scope. No observations are accepted here.
    """
    geometry = query_reference_runs(placement)
    length = geometry['query_length']
    if hsp['identity'] != placement['identity']:
        raise ValueError('HSP screened identity mismatch')
    if not isinstance(hsp.get('id'), str) or not hsp['id']:
        raise ValueError('HSP id required')
    if not _hash(hsp.get('producer_contract_sha256')):
        raise ValueError('producer contract SHA256 required')
    if hsp.get('label_convention') != LABEL_CONVENTION:
        raise ValueError('explicit input label convention required')
    frame = hsp['frame']
    if type(frame) is not int or frame not in (-3, -2, -1, 1, 2, 3):
        raise ValueError('invalid HSP frame')
    start = _integer(hsp['qstart'], 'HSP start', 1)-1
    end = _integer(hsp['qend'], 'HSP end', 1)-1
    direction = 1 if frame > 0 else -1
    if start >= length or end >= length or (end-start)*direction < 0:
        raise ValueError('HSP bounds/direction mismatch')
    normalized = []
    for i, event in enumerate(codon_events):
        q = _integer(event['q0'], 'event q0')
        n = _integer(event['length'], 'event length', 1)
        c = _integer(event['c0'], 'event c0')
        if type(event['direction']) is not int or event['direction'] != direction or c > 2:
            raise ValueError('invalid event direction/ordinal')
        last = q+direction*(n-1)
        lo, hi = min(q, last), max(q, last)+1
        if lo < min(start, end) or hi > max(start, end)+1:
            raise ValueError('event bounds outside HSP')
        normalized.append((lo, hi, i, q, c))
    normalized.sort()
    reasons = ['motif_strand_unavailable' if placement.get('ts') is None else 'motif_strand_available']
    if any(b[0] < a[1] for a, b in zip(normalized, normalized[1:])):
        reasons.append('self_overlap_policy_unresolved')
    if placement.get('ts') is not None and placement['ts'] != ('+' if direction == 1 else '-'):
        reasons.append('ts_hsp_conflict_policy_unresolved')
    if placement['flag'] & (256 | 2048):
        reasons.append('nonprimary_emission_policy_unresolved')
    meta = dict(hsp_id=hsp['id'], identity=dict(hsp['identity']),
                hsp_coordinates={key: hsp[key] for key in ('qstart', 'qend', 'frame')},
                placement_coordinates={key: placement.get(key) for key in
                                       ('contig', 'contig_length', 'pos1', 'cigar', 'flag', 'ts')},
                producer_contract_sha256=hsp['producer_contract_sha256'], geometry=geometry)
    if any(reason.endswith('_unresolved') for reason in reasons):
        return _diagnostic(reasons, **meta)
    reverse = bool(placement['flag'] & 16)
    step = direction * (-1 if reverse else 1)
    strand = '+' if step == 1 else '-'
    runs = geometry['runs']
    ends = [r['u1'] for r in runs]
    segments = []
    for lo, hi, i, q, c in normalized:
        ulo, uhi = (length-hi, length-lo) if reverse else (lo, hi)
        j = bisect_right(ends, ulo)
        while j < len(runs) and runs[j]['u0'] < uhi:
            r = runs[j]
            u0, u1 = max(ulo, r['u0']), min(uhi, r['u1'])
            if u0 < u1:
                q0 = length-1-u0 if reverse else u0
                c0 = (c+(q0-q)*direction) % 3
                g0 = r['g0']+u0-r['u0']
                segments.append(dict(contig=placement['contig'], start0=g0, end0=g0+u1-u0,
                                     strand=strand, c0=c0, step=step,
                                     anchor=(g0-step*c0) % 3, event_index=i,
                                     query_q0=q0, source_q0=placement['identity']['source_start0']+q0))
            j += 1
    if not segments:
        reasons.append('no_mapped_codon_bases')
    return _diagnostic(reasons, segments, **meta)


def project_observations(observations):
    """Diagnostic boundary; multi-HSP resolution deliberately not implemented."""
    observations = iter(observations)
    exhausted = object()
    first = next(observations, exhausted)
    if first is exhausted:
        return _diagnostic(['no_observations'])
    if first is None:
        raise ValueError('invalid observation: None')
    if next(observations, exhausted) is not exhausted:
        return _diagnostic(['multi_hsp_policy_unresolved'])
    return project_codons(*first)


def genomic_strand(flag, ts):
    """Pure FLAG16 x read-relative ts:A composition; not an alignment filter.

    Authority: .atl/quarantine.md §1m, ts:A/FLAG16 row and
    revision/scripts/67_orientation_audit.py:238-240 at base 950cf4c.
    """
    if type(flag) is not int or flag < 0 or ts not in ('+', '-'):
        raise ValueError('nonnegative SAM flag and explicit ts +/- required')
    return ('-' if ts == '+' else '+') if flag & 16 else ts


def reconcile_phases(route_a, route_b):
    """Compare already-labelled phases for the SAME base/strand/protein pair.

    None means no support (all phase-channel values zero), NOT phase label 0.
    Returns (A, B, disagreement). Does not resolve multilabel observations.
    """
    for phase in (route_a, route_b):
        if phase is not None and (type(phase) is not int or phase not in (0, 1, 2)):
            raise ValueError('phase must be 0, 1, 2 or absent')
    if route_a is not None and route_b is not None and route_a != route_b:
        return None, None, True
    return route_a, route_b, False


# A46 (PROTOCOL_B1_frozen_v1.md v1.36) was approved by the author on 2026-09-22
# 22:23 PDT; the approval record lists the protocol sha256 at approval. This is
# the ONLY production gate: changing A46 requires a new amendment (A48+).
A46_APPROVAL = dict(
    amendment='A46', protocol='revision/protocols/PROTOCOL_B1_frozen_v1.md', version='v1.36',
    approved='2026-09-22', record='notes/a46_approval/A46_APPROVAL_RECORD_20260922.md',
    protocol_sha256='e137c01edfa076d293c56d7af2b2f557e111a8389717312be89a97261e3c4284',
    rules=('rule 6 route-B phase channels', 'APPROVED_HSP_PHASE_A11_B5_20260917'))

# Interval label convention written to est_protein_hsp.phase: codon position
# (0 = first codon base, coding direction) of the base at genomic_start0, the
# LOWEST coordinate of the half-open interval. Base g then has codon position
# (phase + (g - start0)) % 3 on '+' and (phase - (g - start0)) % 3 on '-'.
# This is NOT the GFF3 CDS phase column (bases to skip to the next codon).
PHASE_CONVENTION = 'codon_position_at_genomic_start0_v1'

# Multi-HSP rule (A46 rule 6 "HSPs whose frame is inconsistent within one EST
# after projection are dropped"; approved 2026-09-17 "conflicting evidence is
# excluded, independently agreeing evidence is kept"):
#   * scope = one EST molecule (A11 closure unit, A46 rule 3 "a molecule is one
#     unit everywhere"), all members, all proteins, all protein sources;
#   * two HSPs conflict iff they label a common genomic base (same contig) with
#     a different (strand, codon position);
#   * every HSP that conflicts with at least one other HSP of the scope is
#     dropped whole (no winner selection, no partial trimming); all participants
#     are collected before any is removed, so dropping is order-independent;
#   * HSPs that only agree (or do not overlap) are kept, including duplicates;
#   * an HSP that is inconsistent with ITSELF (a -1 frameshift reusing a base,
#     reported by project_codons as self overlap) is dropped by `project`.
MULTI_HSP_RULE = 'A46_rule6_molecule_scope_pairwise_conflict_drop_v1'


def base_label(interval, g):
    """Codon position of genomic base g inside interval (PHASE_CONVENTION)."""
    step = 1 if interval['strand'] == '+' else -1
    return (interval['phase'] + step * (g - interval['start0'])) % 3


def merge_segments(segments):
    """project_codons segments -> maximal phase-continuous genomic intervals.

    Adjacent segments merge only when contiguous on the same contig/strand and
    the label continues without a jump; introns, EST insertions (CIGAR I) and
    frameshifts therefore start a new interval. Overlaps must agree.
    """
    merged = []
    for s in sorted(segments, key=lambda s: (s['contig'], s['start0'], s['end0'])):
        row = dict(contig=s['contig'], strand=s['strand'], start0=s['start0'],
                   end0=s['end0'], phase=s['c0'])
        if merged:
            last = merged[-1]
            if last['contig'] == row['contig'] and row['start0'] < last['end0']:
                if last['strand'] != row['strand'] or base_label(last, row['start0']) != row['phase']:
                    raise ValueError('self-inconsistent projected HSP')
                last['end0'] = max(last['end0'], row['end0'])
                continue
            if (last['contig'] == row['contig'] and last['strand'] == row['strand']
                    and last['end0'] == row['start0']
                    and base_label(last, row['start0']) == row['phase']):
                last['end0'] = row['end0']
                continue
        merged.append(row)
    return merged


def project(hsp, codon_events, placement):
    """Production projection of ONE accepted HSP through its primary placement.

    Returns dict(intervals=[...], dropped=None|reason, reasons=[...]). Any
    geometry reason ending in `_unresolved` (self overlap, transcript-strand vs
    HSP-strand conflict, non-primary placement) or no mapped codon base drops
    the HSP; malformed inputs still raise ValueError.
    """
    result = project_codons(hsp, codon_events, placement)
    blocking = [r for r in result['reasons']
                if r.endswith('_unresolved') or r == 'no_mapped_codon_bases']
    if blocking:
        return dict(intervals=[], dropped=blocking[0], reasons=result['reasons'])
    return dict(intervals=merge_segments(result['segments']), dropped=None,
                reasons=result['reasons'])


def resolve_hsps(items):
    """Apply MULTI_HSP_RULE. items: iterable of dict(key, scope, intervals).

    Returns dict(kept=[keys in input order], dropped={key: reason},
    conflict_pairs=int). Keys must be unique.
    """
    items = list(items)
    keys = [item['key'] for item in items]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate HSP key')
    by_scope = {}
    for item in items:
        for iv in item['intervals']:
            if iv['strand'] not in ('+', '-') or iv['phase'] not in (0, 1, 2) \
                    or not 0 <= iv['start0'] < iv['end0']:
                raise ValueError('invalid projected interval')
            by_scope.setdefault((item['scope'], iv['contig']), []).append((iv['start0'], iv['end0'], item['key'], iv))
    conflicted, pairs = set(), set()
    for rows in by_scope.values():
        rows.sort(key=lambda r: (r[0], r[1], r[2]))
        active = []
        for start, end, key, iv in rows:
            active = [a for a in active if a[1] > start]
            for a_start, a_end, a_key, a_iv in active:
                if a_key == key:
                    continue
                if a_iv['strand'] != iv['strand'] or base_label(a_iv, start) != iv['phase']:
                    conflicted.update((key, a_key))
                    pairs.add(tuple(sorted((key, a_key))))
            active.append((start, end, key, iv))
    return dict(kept=[k for k in keys if k not in conflicted],
                dropped={k: 'phase_conflict_within_molecule' for k in keys if k in conflicted},
                conflict_pairs=len(pairs))
