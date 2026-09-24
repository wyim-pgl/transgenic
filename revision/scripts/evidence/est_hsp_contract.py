"""Approved whole-HSP rule, per-row diagnostics only; no production/training gate.

Caller MUST bind actual original-screened DNA/protein to row qseqid/sseqid.
Sequence hashes do not certify FASTA mapping, source files, lineage or A11.
Rows are field sequences; raw_row_sha256 hashes UTF-8 tab-joined fields without
line terminator, sequence identities hash exact supplied UTF-8 sequence text.
No files are read/written and historical NOT_ELICITED artifacts are untouched.
"""
import hashlib
from itertools import groupby

from .est_hsp_expanded import CODE, interpret, rc, require
from .est_phase_accept import btop_tokens, parse_row

LENGTH_CONTRACT = 'DIAMOND_REPORTED_HSP_M_PLUS_I_PLUS_D_GE30'
MIDF_LENGTH_CONTRACT = 'DIAMOND_REPORTED_HSP_M_PLUS_I_PLUS_D_PLUS_F_GE30'
SCOPE = 'DIAGNOSTIC_ONLY_NOT_PRODUCTION_TRAINING_OR_LINEAGE_A11_CERTIFICATION'


def _validate(row, dna, protein, width):
    require(isinstance(row, (list, tuple)) and len(row) == width, f'expected {width} fields')
    require(all(isinstance(x, str) and x and not any(c.isspace() for c in x) for x in row),
            'nonempty whitespace-free fields required')
    parsed = parse_row(1, '\t'.join(row[:13]))
    require(isinstance(dna, str) and dna and set(dna) <= set('ACGT'), 'noncanonical DNA')
    require(isinstance(protein, str) and protein and set(protein) <= set(CODE.values()),
            'noncanonical protein')
    require(parsed['qlen'] == len(dna) and parsed['slen'] == len(protein), 'input lengths')
    return parsed


def classify_rich17(row, dna, protein, *, expected_identities=None, length_contract=LENGTH_CONTRACT):
    """Validate truth and the explicit formula, then classify reported length >=30.

    A mathematical mode is NOT producer authority. Callers must prove the path.
    The default remains strict MID. Raises ValueError for invalid evidence.
    q0 is the first translated base in ORIGINAL screened query, zero-based;
    bases are q0 + direction * [0,1,2]. c0=0 anchors this codon only, NOT a
    projected phase inferred from qframe. Subject indices are zero-based.
    Optional expected_identities checks supplied hashes, not source authority.
    """
    require(length_contract in (LENGTH_CONTRACT, MIDF_LENGTH_CONTRACT), 'unknown length contract')
    _validate(row, dna, protein, 17)
    result = interpret(row, dna, protein)
    metric = 'columns_excluding_F' if length_contract == LENGTH_CONTRACT else 'columns_including_F'
    require(result['metrics']['reported'] == result['metrics'][metric],
            'reported length disagrees with length contract: ' + length_contract)
    identities = {key: hashlib.sha256(text.encode('utf-8')).hexdigest() for key, text in
                  (('raw_row_sha256', '\t'.join(row)), ('dna_sha256', dna), ('protein_sha256', protein))}
    if expected_identities is not None:
        require(isinstance(expected_identities, dict) and bool(expected_identities), 'missing source identities')
        for key, value in expected_identities.items():
            require(key in identities and identities[key] == value, f'source identity mismatch: {key}')
    direction = 1 if result['initial_frame'] > 0 else -1

    def original(at):
        return at if direction == 1 else len(dna) - 1 - at

    codons, subject, events = iter(result['codons']), int(row[6])-1, []
    for q, s in zip(row[14], row[15]):
        if q in '/\\':
            continue
        if q != '-':
            codon = next(codons)
            events.append(dict(q0=original(codon['at']), length=3, direction=direction, c0=0,
                               kind='I' if s == '-' else 'M',
                               subject_aa_index=None if s == '-' else subject))
        if s != '-':
            subject += 1
    shifts = [dict(shift, q0=original(shift['at'] if shift['delta'] == 1 else shift['at']-1),
                   length=1, direction=direction, effect='omitted' if shift['delta'] == 1 else 'reused')
              for shift in result['shifts']]
    accepted = int(row[8]) >= 30
    return dict(length_contract=length_contract, accepted_by_hsp_rule=accepted,
                diagnostic_disposition='HSP_RULE_ACCEPTED' if accepted else 'BELOW_HSP_LENGTH_30',
                scope=SCOPE, identities=identities, qseqid=row[0], sseqid=row[2],
                counts=result['counts'], codon_events=events, frameshifts=shifts)


def adapt_raw13(row, dna, protein, *, length_contract=LENGTH_CONTRACT):
    """Reconstruct diagnostic rich17 fields from actual residues, without engine.

    Same caller ID-binding obligation as classify_rich17. Returned fields are
    NOT production/training eligibility or lineage/A11 certification. Original
    raw13 fields are preserved exactly; caller can retain their separate hash.
    """
    require(length_contract in (LENGTH_CONTRACT, MIDF_LENGTH_CONTRACT), 'unknown length contract')
    parsed = _validate(row, dna, protein, 13)
    oriented = dna if parsed['frame'] > 0 else rc(dna)
    pos = parsed['qs']-1 if parsed['frame'] > 0 else len(dna)-parsed['qs']
    subject = parsed['ss']-1
    qg, sg, translated, ops = [], [], [], []

    def column(q, s):
        nonlocal pos, subject
        if q in '/\\':
            require(s == '-', 'shift must not consume subject')
            pos += 1 if q == '\\' else -1
            require(0 <= pos <= len(dna), 'shift outside DNA')
            op = q
        else:
            require(q != '-' or s != '-', 'double gap')
            if q != '-':
                require(0 <= pos and pos+3 <= len(dna) and CODE.get(oriented[pos:pos+3]) == q,
                        'query codon truth')
                translated.append(q)
                pos += 3
            if s != '-':
                require(subject < len(protein) and protein[subject] == s, 'subject truth')
                subject += 1
            op = 'D' if q == '-' else 'I' if s == '-' else 'M'
        qg.append(q)
        sg.append(s)
        ops.append(op)

    for token in btop_tokens(parsed['btop']):
        if token.isdigit():
            count = int(token)
            require(0 < count <= len(protein)-subject and count <= (len(dna)-pos)//3,
                    'BTOP equal run outside sequences')
            for _ in range(count):
                aa = protein[subject]
                column(aa, aa)
        else:
            column(*token)
    rich = list(row) + [''.join(str(len(list(g))) + op for op, g in groupby(ops)),
                        ''.join(qg), ''.join(sg), ''.join(translated)]
    # Gapped-string-driven replay cross-checks BTOP reconstruction; shared code
    # and genetic code mean this is not independent biological validation.
    classify_rich17(rich, dna, protein, length_contract=length_contract)
    return rich
