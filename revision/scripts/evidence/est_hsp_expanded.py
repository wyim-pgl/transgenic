#!/usr/bin/env python3
"""Bounded DIAMOND diagnostic, never acceptance or projection.

The truth interpreter uses oriented input DNA + the standard genetic code.
Aligned strings drive its cursor; BTOP/CIGAR are redundant serialization checks,
not independent biological evidence. No candidate decomposition or old replay.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import shutil
import signal
import sys

ENGINE_SHA256 = '0fb259d705815d44f6c667a976fb71db00ec415afc3cba9938f088b90bfed07d'
# NCBI standard code, T/C/A/G lexical order (including stop codons).
CODE = dict(zip((''.join(x) for x in itertools.product('TCAG', repeat=3)),
                'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'))
ENCODE = dict(zip('ACDEFGHIKLMNPQRSTVWY',
                  'GCC TGC GAC GAG TTC GGC CAC ATC AAG CTG ATG AAC CCC CAG CGC AGC ACC GTG TGG TAC'.split()))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def rc(dna):
    return dna.translate(str.maketrans('ACGT', 'TGCA'))[::-1]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


SENSE = frozenset(CODE.values()) - {'*'}


def masked_columns(btop, width):
    """Column indices where BTOP letters are 'X' ({'q': set, 's': set}); None if unparsable.

    Never expands equal runs (a huge digit run only advances the column cursor).
    """
    tokens = re.findall(r'\d+|[^\d][^\d]', btop)
    if ''.join(tokens) != btop:
        return None
    cols, masked = 0, dict(q=set(), s=set())
    for token in tokens:
        if cols > width:
            return None
        if token.isdigit():
            cols += int(token)
            continue
        for side, letter in zip('qs', token):
            if letter == 'X':
                masked[side].add(cols)
        cols += 1
    return masked if cols == width else None


def interpret(row, dna, protein, *, shift_sign=1, masked_x=False):
    """Interpret one rich17 row. shift_sign=-1 is a CPU negative control only.

    Unknown positive col9 semantics are reported, not assumed away. Caller must
    treat no consistent reading as inconclusive. M includes substitutions.

    masked_x=True (opt-in; author decision 2026-09-25): a BTOP 'X' letter is a
    DIAMOND-masked position whose aligned string carries the TRUE residue. The
    truth checks below run on the true residues unchanged; only the BTOP
    serialization check re-applies the mask: a query X is accepted iff the true
    aligned residue there is a sense amino acid, a subject X iff it is a protein
    residue (never a gap/shift), and a masked column never joins an equal run.
    Any other BTOP disagreement still fails. Counts are returned as 'masked_x'.
    """
    require(len(row) == 17 and all(row), 'expected 17 nonempty fields')
    qlen, slen = int(row[1]), int(row[3])
    qs, qe, ss, se, length = map(int, row[4:9])
    frame = int(row[11])
    require(qlen == len(dna) and slen == len(protein), 'input lengths')
    require(set(dna) <= set('ACGT'), 'noncanonical DNA')
    require(frame in (-3, -2, -1, 1, 2, 3), 'frame')
    require(1 <= qs <= qlen and 1 <= qe <= qlen and 1 <= ss <= se <= slen, 'coordinates')
    require((qs <= qe) == (frame > 0) and length > 0, 'orientation/length')
    require(math.isfinite(float(row[9])) and 0 <= float(row[9]) <= 1e-5
            and math.isfinite(float(row[10])) and float(row[10]) > 0, 'scores')
    dna = dna if frame > 0 else rc(dna)
    start = qs - 1 if frame > 0 else qlen - qs
    end = qe if frame > 0 else qlen - qe + 1
    require(abs(frame) == start % 3 + 1, 'initial frame disagrees with DNA start')
    pos, subject = start, ss - 1
    qg, sg = row[14:16]
    require(len(qg) == len(sg) and len(qg) > 0, 'aligned string lengths')
    counts = dict(M=0, I=0, D=0, F=0)
    blocks, frames, shifts, codons, ops, translated = [0], [frame], [], [], [], []
    btop, run = [], 0
    bq, bs = list(qg), list(sg)       # BTOP display letters; == aligned strings unless masked_x
    mask = masked_columns(row[12], len(qg)) if masked_x else None
    if mask:
        for i in mask['q']:
            if qg[i] in SENSE:
                bq[i] = 'X'
        for i in mask['s']:
            if sg[i] not in '-/\\':
                bs[i] = 'X'
    for q, s, dq, ds in zip(qg, sg, bq, bs):
        if dq == ds and dq not in '-/\\' and not (masked_x and dq == 'X'):
            run += 1
        else:
            if run:
                btop.append(str(run))
                run = 0
            btop.append(dq + ds)
        if q in '/\\':
            require(s == '-', 'shift must not consume subject')
            delta = (1 if q == '\\' else -1) * shift_sign
            shifts.append(dict(at=pos, delta=delta, marker=q))
            pos += delta
            require(0 <= pos <= qlen, 'shift outside DNA')
            counts['F'] += 1
            ops.append(q)
            blocks.append(0)
            frames.append((1 if frame > 0 else -1) * (pos % 3 + 1))
            continue
        require(q != '-' or s != '-', 'double gap')
        op = 'D' if q == '-' else ('I' if s == '-' else 'M')
        counts[op] += 1
        ops.append(op)
        if q != '-':
            codon = dna[pos:pos + 3]
            require(len(codon) == 3 and CODE.get(codon) == q, f'codon truth at {pos}: {codon} != {q}')
            codons.append(dict(at=pos, codon=codon, residue=q))
            translated.append(q)
            pos += 3
            blocks[-1] += 1
        if s != '-':
            require(subject < slen and protein[subject] == s, f'protein truth at {subject}')
            subject += 1
    if run:
        btop.append(str(run))
    cigar = ''.join(str(len(list(group))) + op for op, group in itertools.groupby(ops))
    require(''.join(btop) == row[12], 'BTOP serialization')
    require(cigar == row[13], 'CIGAR serialization')
    require(''.join(translated) == row[16], 'translated sequence')
    require(pos == end and subject == se, 'terminal coordinates')
    m, i, d, f = (counts[k] for k in 'MIDF')
    metrics = dict(reported=length, columns_excluding_F=m+i+d, columns_including_F=m+i+d+f,
                   query_residues=m+i, paired=m, max_block=max(blocks), sum_blocks=sum(blocks))
    extra = {}
    if masked_x:
        extra['masked_x'] = dict(query=sum(x == 'X' != y for x, y in zip(bq, qg)),
                                 subject=sum(x == 'X' != y for x, y in zip(bs, sg)))
    return dict(**extra, counts=counts, metrics=metrics, blocks=blocks, frames=frames,
                shifts=shifts, codons=codons, oriented_start=start, oriented_end=end,
                subject_start=ss, subject_end=se, initial_frame=frame,
                col9_matches=[key for key, value in metrics.items() if key != 'reported' and value == length],
                thresholds={key: value >= 30 for key, value in metrics.items()},
                block_thresholds=[n >= 30 for n in blocks],
                each_block_ge30=all(n >= 30 for n in blocks),
                qualifying_blocks=[n for n in blocks if n >= 30])


def make_cases():
    """24 predeclared shapes x 3 offsets x 2 strands; no combinatorial search.

    Edit coordinates are sequential, zero-based ORIENTED DNA coordinates.
    Flanks are independent stop-free 20-codon contexts, not target extensions.
    Every minus-one deletion reuses the last base of the preceding codon when
    interpreted, possibly changing the first residue of the following block.
    """
    shapes = []
    for n in (29, 30, 31):
        shapes.append((f'exact{n}', n, []))
    for n in (30, 31):
        shapes.append((f'D{n}', n, [(14, 'D', 1)]))
    for n in (29, 30):
        shapes.append((f'I{n}', n, [(14, 'I', 1)]))
    for n in (29, 30, 31):
        for delta in (1, -1):
            shapes.append((f'short{n}_{delta}', n, [(14, 'F', delta)]))
    for n in (29, 30, 31):
        for delta in (1, -1):
            shapes.append((f'blocks29_{n}_{delta}', 29+n, [(29, 'F', delta)]))
    shapes.extend([('long_plus', 180, [(90, 'F', 1)]),
                   ('long_minus', 180, [(90, 'F', -1)]),
                   ('long_I', 180, [(90, 'I', 3)]),
                   ('long_D', 180, [(90, 'D', 3)]),
                   ('cancel', 90, [(29, 'F', 1), (59, 'F', -1)])])
    cases = []
    rng = random.Random(202609161541)
    for name, n, edits in shapes:
        protein = ''.join(rng.choice(list(ENCODE)) for _ in range(n))
        left = ''.join(ENCODE[rng.choice(list(ENCODE))] for _ in range(20))
        right = ''.join(ENCODE[rng.choice(list(ENCODE))] for _ in range(20))
        coding = ''.join(ENCODE[a] for a in protein)
        for offset in range(3):
            baseline = 'C' * offset + left + coding + right
            dna, ledger, expected_shifts = baseline, [], []
            start, change = offset + len(left), 0
            counts = dict(M=n, I=0, D=0, F=0)
            blocks, previous = [], 0
            for aa, kind, amount in edits:
                at = start + aa * 3 + change
                if kind == 'F':
                    deleted = dna[at:at+1] if amount < 0 else ''
                    inserted = 'C' if amount > 0 else ''
                    expected_shifts.append(dict(at=at, delta=amount))
                    blocks.append(aa-previous)
                    previous = aa
                    counts['F'] += 1
                elif kind == 'I':
                    deleted, inserted = '', 'CAGTGGCCC'[:3*amount]
                    counts['I'] += amount
                else:
                    deleted, inserted = dna[at:at+3*amount], ''
                    counts['M'] -= amount
                    counts['D'] += amount
                ledger.append(dict(at=at, deleted=deleted, inserted=inserted, kind=kind))
                dna = dna[:at] + inserted + dna[at+len(deleted):]
                change += len(inserted)-len(deleted)
            blocks.append(n-previous + counts['I']-counts['D'])
            for sign in (1, -1):
                ident = f'{name}_o{offset}_{"fwd" if sign > 0 else "rev"}'
                cases.append(dict(id=ident, subject=name, family=name, protein=protein,
                                  baseline_dna=baseline, oriented_dna=dna, edits=ledger,
                                  dna=dna if sign > 0 else rc(dna), start=start,
                                  end=start+3*n+change, expected_frame=sign*(offset+1),
                                  expected_counts=counts, expected_blocks=blocks,
                                  expected_shifts=expected_shifts))
    return cases


def validate_case(case):
    """Reconstruct query independently from the written sequential edit ledger."""
    dna = case['baseline_dna']
    start = case['start']
    coding = dna[start:start + 3 * len(case['protein'])]
    require(''.join(CODE[coding[i:i+3]] for i in range(0, len(coding), 3))
            == case['protein'], 'baseline protein/codon ledger')
    for edit in case['edits']:
        at, deleted, inserted = edit['at'], edit['deleted'], edit['inserted']
        require(0 <= at <= len(dna) and dna[at:at+len(deleted)] == deleted,
                'edit ledger deleted bases')
        dna = dna[:at] + inserted + dna[at+len(deleted):]
    oriented = case['dna'] if case['expected_frame'] > 0 else rc(case['dna'])
    require(dna == oriented == case['oriented_dna'], 'edit ledger oriented DNA')


def elicited(result, case):
    return (result['counts'] == case['expected_counts']
            and result['blocks'] == case['expected_blocks']
            and [{k: s[k] for k in ('at', 'delta')} for s in result['shifts']] == case['expected_shifts']
            and result['oriented_start'] == case['start'] and result['oriented_end'] == case['end']
            and result['subject_start'] == 1 and result['subject_end'] == len(case['protein'])
            and result['initial_frame'] == case['expected_frame'])


def diagnose(raw, cases, out):
    """Persist all raw bytes and inventory BEFORE parsing; keep every failure/hit."""
    raw, out = Path(raw), Path(out)
    snapshot = out / 'raw-first.tsv'
    # Exclusive creation avoids silently replacing earlier evidence.
    with snapshot.open('xb') as target, raw.open('rb') as source:
        shutil.copyfileobj(source, target)
    write_json(out / 'raw-first.json', dict(raw_sha256=digest(snapshot),
               cases_expected=[c['id'] for c in cases], certified=False, production_authorized=False))
    by_id = {c['id']: c for c in cases}
    proteins = {c['subject']: c['protein'] for c in cases}
    rows, seen = [], Counter()
    for number, line in enumerate(snapshot.read_text().splitlines(), 1):
        record = dict(number=number, raw=line, status='INVALID',
                      row_sha256=hashlib.sha256(line.encode()).hexdigest())
        try:
            fields = line.split('\t')
            case = by_id[fields[0]]
            validate_case(case)
            record.update(query_sequence_sha256=hashlib.sha256(case['dna'].encode()).hexdigest(),
                          protein_sequence_sha256=hashlib.sha256(proteins[fields[2]].encode()).hexdigest())
            result = interpret(fields, case['dna'], proteins[fields[2]])
            same_subject = fields[2] == case['subject']
            record.update(result=result, status='ELICITED' if same_subject and elicited(result, case) else 'NOT_ELICITED')
            seen[case['id']] += record['status'] == 'ELICITED'
            if result['counts']['F']:
                try:
                    interpret(fields, case['dna'], proteins[fields[2]], shift_sign=-1)
                    record['wrong_sign_rejected'] = False
                except ValueError:
                    record['wrong_sign_rejected'] = True
        except (ValueError, KeyError, IndexError) as error:
            record['error'] = str(error)
        rows.append(record)
    inventory = [dict(id=c['id'], family=c['family'], expected_counts=c['expected_counts'],
                      expected_blocks=c['expected_blocks'],
                      status='ELICITED' if seen[c['id']] == 1 else 'NOT_ELICITED',
                      elicited_rows=seen[c['id']]) for c in cases]
    readings = [set(r['result']['col9_matches']) for r in rows if 'result' in r]
    consistent = sorted(set.intersection(*readings)) if readings else []
    complete = (bool(consistent) and bool(rows) and all(c['status'] == 'ELICITED' for c in inventory)
                and all(r['status'] == 'ELICITED' and r['result']['col9_matches']
                        and r.get('wrong_sign_rejected', True) for r in rows))
    report = dict(status='DIAGNOSTIC_COMPLETE_REVIEW_REQUIRED' if complete else 'INCONCLUSIVE',
                  certified=False, production_authorized=False, acceptance_policy='UNRESOLVED',
                  raw_sha256=digest(snapshot), consistent_col9_readings=consistent,
                  cases=inventory, rows=rows)
    write_json(out / 'diagnostic.json', report)
    return report


def run(diamond, out, tmpdir):
    # Only the runner imports the frozen launcher; interpreter remains standalone.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import est_blastx as launcher
    out = Path(out)
    out.mkdir()  # refuse reuse, including symlinks
    try:
        require(Path(diamond).is_absolute() and digest(diamond) == ENGINE_SHA256, 'engine hash pin')
        version = launcher.engine_version(diamond)
        cases = make_cases()
        write_json(out / 'manifest.json', dict(cases=cases, code=CODE))
        query, target = out / 'queries.fa', out / 'proteins.fa'
        query.write_text(''.join(f'>{c["id"]}\n{c["dna"]}\n' for c in cases))
        proteins = {c['subject']: c['protein'] for c in cases}
        target.write_text(''.join(f'>{k}\n{v}\n' for k, v in proteins.items()))
        provenance = dict(diamond_sha256=ENGINE_SHA256, version=version,
                          query_sha256=digest(query), protein_sha256=digest(target),
                          imports={str(p): digest(p) for p in
                                   (Path(__file__), Path(launcher.__file__), launcher.HELPER)},
                          production_authorized=False, certified=False)
        write_json(out / 'provenance.json', provenance)
        raw, db = out / 'expanded.raw17.tsv', out / 'expanded.dmnd'
        commands = [[diamond, 'makedb', '--in', str(target), '--db', str(db),
                     '--threads', '2', '--no-parse-seqids'],
                    launcher.search_command(diamond, query, db, raw, 2, tmpdir, rich=True)]
        write_json(out / 'commands.json', commands)
        launcher.execute(commands[0], out / 'makedb.log')
        try:
            launcher.execute(commands[1], out / 'blastx.log')
        except BaseException:
            if raw.exists():
                diagnose(raw, cases, out)
            raise
        report = diagnose(raw, cases, out)
        (out / 'DONE').write_text(report['status'] + '\nNOT CERTIFICATION\n')
        return report
    except BaseException as error:
        (out / 'FAILED').write_text(str(error) + '\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diamond', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--tmpdir', required=True)
    args = parser.parse_args()
    def interrupted(signum, _frame):
        raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    run(args.diamond, args.out_dir, args.tmpdir)


if __name__ == '__main__':
    main()
