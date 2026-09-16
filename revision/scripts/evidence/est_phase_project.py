"""Diagnostic screened-query geometry; production project() remains blocked.

Labels MUST come from a separately validated event producer, not signed-frame
extrapolation or the candidate HSP decomposer. This API checks geometry, not
acceptance, provenance authenticity, leakage closure, or training eligibility.
No BAM reader, raster, molecule aggregator or route-A adapter is provided.
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
    first = next(observations, None)
    if first is None:
        return _diagnostic(['no_observations'])
    if next(observations, None) is not None:
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


def project():
    raise ValueError('UNRESOLVED phase anchor/HSP/BTOP semantics; production blocked by A46')


if __name__ == '__main__':
    project()
