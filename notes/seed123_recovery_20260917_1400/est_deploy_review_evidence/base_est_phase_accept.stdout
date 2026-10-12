#!/usr/bin/env python3
"""C2 acceptance gate and fixture-only candidate BTOP decomposition.

No accepted HSPs or phase evidence can be emitted. HSP >=30 aa may mean
reported alignment length, ungapped query residues or per-frame sub-span;
these readings MUST NOT be silently equated. Slash/backslash cursor shifts
below are inherited HYPOTHESES to challenge with real DIAMOND RICH17, not
verified semantics. CPU fixtures cannot approve them.

Both public acceptance entrypoints remain unconditionally blocked until a
reviewed real-RICH17 contract AND separate A46 production approval exist.
There is intentionally no CLI bypass switch or guessed approval schema.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import signal
import sys

MIN_AA = 30  # frozen; no CLI override
ACCEPTANCE_DEFINITION = {
    'chosen': None,
    'status': 'UNRESOLVED-until-reviewed-RICH17-fixture',
    'candidates': {
        'A_col9_alignment_length': 'raw13 column 9 as reported by DIAMOND (alignment columns, aa units, gaps included)',
        'B_ungapped_query_aa': 'BTOP columns consuming a query codon (match+mismatch), gaps excluded',
        'C_per_frame_subspan_aa': 'aa length of each constant-frame sub-span between frameshift ops'},
    'rule': 'UNRESOLVED: no accepted HSP output permitted',
    'terminology': 'candidate diagnostics only; no accepted HSP or phase evidence'}


def btop_tokens(btop):
    tokens = re.findall(r'\d+|[^\d][^\d]', btop)
    if ''.join(tokens) != btop:
        raise ValueError(f'invalid BTOP token stream: {btop!r}')
    return tokens


def candidate_decompose_hsp(qlen, qs, qe, frame, btop):
    """Fixture-only hypothesis; NOT a verified frameshift interpretation.

    Returns (subspans, ungapped_aa, n_frameshift_ops, subject_cols).  Each sub-span is a dict
    with ORIGINAL query coordinates (1-based inclusive), strand, and the signed
    frame recomputed at the sub-span start (sign = HSP strand; magnitude =
    oriented_start % 3 + 1, candidate convention checked for internal
    consistency with column 12, not independently verified).
    """
    reverse = frame < 0
    if (qe >= qs) == reverse:
        raise ValueError('qstart/qend orientation disagrees with frame sign')
    pos = qlen - qs if reverse else qs - 1        # 0-based, oriented query
    end = qlen - qe + 1 if reverse else qe        # exclusive, oriented query
    if abs(frame) != pos % 3 + 1:
        raise ValueError(f'frame {frame} disagrees with oriented start {pos}')
    subspans, shifts, ungapped_aa = [], 0, 0
    span_start, span_aa = pos, 0

    def close():
        nonlocal span_start, span_aa
        if pos > span_start and span_aa > 0:
            if reverse:  # oriented [span_start, pos) -> original 1-based inclusive
                q_start, q_end = qlen - pos + 1, qlen - span_start
            else:
                q_start, q_end = span_start + 1, pos
            subspans.append(dict(q_start=q_start, q_end=q_end, subspan_aa=span_aa,
                                 strand='-' if reverse else '+',
                                 signed_frame=(-1 if reverse else 1) * (span_start % 3 + 1)))
        span_start, span_aa = pos, 0

    subject_cols = 0
    for token in btop_tokens(btop):
        if token.isdigit():
            n = int(token)
            if n <= 0:
                raise ValueError('BTOP match run must be positive')
            pos += 3 * n
            ungapped_aa += n
            span_aa += n
            subject_cols += n
            continue
        q, s = token
        if q in ('/', '\\'):
            if s != '-':
                raise ValueError(f'frameshift op must not consume subject: {token!r}')
            close()
            shift = 1 if q == '\\' else -1  # '\\' = +1 nt, '/' = -1 nt (oriented)
            pos += shift
            shifts += 1
            if not 0 <= pos <= qlen:
                raise ValueError('frameshift outside query')
            span_start = pos
        else:
            if q == '-' and s == '-':
                raise ValueError('double gap in BTOP')
            if q != '-':
                pos += 3
                ungapped_aa += 1
                span_aa += 1
            if s != '-':
                subject_cols += 1
    close()
    if pos != end:
        raise ValueError(f'BTOP does not consume reported query span: {pos} != {end}')
    return subspans, ungapped_aa, shifts, subject_cols


def parse_row(number, line):
    c = line.rstrip('\n').split('\t')
    if len(c) != 13 or any(not x for x in c):
        raise ValueError(f'malformed raw13 row {number}')
    qlen, slen = int(c[1]), int(c[3])
    qs, qe, ss, se, length = map(int, c[4:9])
    evalue, bitscore, frame = float(c[9]), float(c[10]), int(c[11])
    # Mirrors est_blastx.validate_raw row contract (C1 already enforced it on
    # the whole file; re-checked here so acceptance is safe standalone too).
    if not (1 <= qs <= qlen and 1 <= qe <= qlen and 1 <= ss <= se <= slen
            and length > 0 and frame in (-3, -2, -1, 1, 2, 3)
            and (qe >= qs) == (frame > 0) and math.isfinite(evalue)
            and 0 <= evalue <= 1e-5 and math.isfinite(bitscore) and bitscore > 0):
        raise ValueError(f'invalid raw coordinates/score at row {number}')
    return dict(qseqid=c[0], qlen=qlen, sseqid=c[2], slen=slen, qs=qs, qe=qe,
                ss=ss, se=se, length=length, evalue=evalue, bitscore=bitscore,
                frame=frame, btop=c[12])


def accept_stream(raw13, source_kind, out_tsv):
    raise ValueError('UNRESOLVED HSP/BTOP semantics; production blocked by A46')


def run(args):
    raise ValueError('UNRESOLVED HSP/BTOP semantics; production blocked by A46')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw13', type=Path, required=True)
    p.add_argument('--raw13-sha256', required=True,
                   help='must match the C1 audit manifest hash for this unit')
    p.add_argument('--source-kind', choices=('orthodb', 'swissprot'), required=True)
    p.add_argument('--species', required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args(argv)

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    run(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
