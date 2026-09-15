#!/usr/bin/env python3
"""Tiny REAL DIAMOND integration gate. No mocks, no production data, no submission.

BTOP frameshifts are two-character pairs \\- /-; cursor shifts +1 / -1 nt.
Contract read from DIAMOND v2.1.24 src/basic/match.h, output/blast_tab_format.cpp
and output/sam_format.cpp. Replay is independently checked against input codons,
subject residues, raw coordinates, CIGAR and all three rich sequence fields.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from itertools import groupby
from pathlib import Path
import re
import signal
import sys
import tempfile

spec = importlib.util.spec_from_file_location('est_blastx', Path(__file__).with_name('est_blastx.py'))
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
BASES = 'TCAG'
AA_TABLE = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
CODE = dict(zip((a+b+c for a in BASES for b in BASES for c in BASES), AA_TABLE))
CODONS = dict(zip('ACDEFGHIKLMNPQRSTVWY',
                  'GCT TGT GAT GAA TTT GGT CAT ATT AAA CTG ATG AAT CCT CAA CGT TCT ACT GTT TGG TAT'.split()))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reverse_complement(dna):
    return dna.translate(str.maketrans('ACGT', 'TGCA'))[::-1]


def fixture_data():
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    protein = ''.join(alphabet[hashlib.sha256(f'B6-fixture:{i}'.encode()).digest()[0] % 20] for i in range(180))
    dna = ''.join(CODONS[a] for a in protein)
    queries = dict(ordinary=dna, reverse=reverse_complement(dna),
                   plus1=dna[:270] + 'C' + dna[270:], minus1=dna[:270] + dna[271:],
                   insertion=dna[:270] + 'CAATGGCCT' + dna[270:],
                   deletion=dna[:270] + dna[279:])
    return dict(protein=protein, queries=queries)


def replay(row, dna, protein):
    """Recover all aligned codons from BTOP; fail on any unsupported operation."""
    require(len(row) == 17, 'fixture requires rich17 output')
    qlen, slen, qs, qe, ss, se = map(int, (row[1], row[3], *row[4:8]))
    require(qlen == len(dna) and slen == len(protein), 'input sequence lengths differ')
    require(1 <= qs <= qlen and 1 <= qe <= qlen and 1 <= ss <= se <= slen, 'bad endpoints')
    reverse = qs > qe
    require(reverse == (int(row[11]) < 0), 'qframe strand differs')
    oriented = reverse_complement(dna) if reverse else dna
    pos = qlen - qs if reverse else qs - 1
    end = qlen - qe + 1 if reverse else qe
    require(abs(int(row[11])) == pos % 3 + 1, 'initial frame disagrees with coordinate')
    subject_pos = ss - 1
    qg, sg, translated, ops, shifts = [], [], [], [], []
    btop = row[12]
    tokens = re.findall(r'\d+|[^\d][^\d]', btop)
    require(''.join(tokens) == btop, 'invalid BTOP token')

    def column(q, s):
        nonlocal pos, subject_pos
        if q in ('\\', '/'):
            require(s == '-', 'frameshift must not consume subject')
            step = 1 if q == '\\' else -1
            shifts.append({'oriented_nt_offset': pos, 'delta_nt': step, 'marker': q})
            pos += step
            require(0 <= pos <= len(oriented), 'frameshift outside query')
            op = q
        else:
            require(q != '-' or s != '-', 'double gap')
            if q != '-':
                require(CODE.get(oriented[pos:pos+3]) == q, f'query codon mismatch at nt {pos}: {q}')
                translated.append(q)
                pos += 3
            if s != '-':
                require(subject_pos < len(protein) and protein[subject_pos] == s, 'subject residue mismatch')
                subject_pos += 1
            op = 'D' if q == '-' else ('I' if s == '-' else 'M')
        qg.append(q)
        sg.append(s)
        ops.append(op)

    for token in tokens:
        if token.isdigit():
            count = int(token)
            require(count > 0 and subject_pos + count <= len(protein), 'bad BTOP match run')
            for _ in range(count):
                aa = protein[subject_pos]
                column(aa, aa)
        else:
            column(*token)
    cigar = ''.join(f'{len(list(group))}{op}' for op, group in groupby(ops))
    require(pos == end and subject_pos == se, 'traceback does not consume reported endpoints')
    require(''.join(qg) == row[14] and ''.join(sg) == row[15], 'BTOP/gapped fields disagree')
    require(''.join(translated) == row[16], 'BTOP/translated field disagrees')
    require(cigar == row[13], 'BTOP/CIGAR disagree')
    return dict(net_frameshift_nt=sum(s['delta_nt'] for s in shifts), frameshifts=shifts,
                cigar=cigar, translated_residues=len(translated), query_reverse=reverse)


def run_fixture(diamond, out_dir, tmpdir, threads):
    require(threads > 0, 'threads must be positive')
    require(tmpdir.is_absolute() and tmpdir.is_dir(), 'explicit existing absolute tmpdir required')
    with launcher.work_directory(out_dir) as out:
        version = launcher.engine_version(diamond)  # missing engine is FAILURE, never a passing skip
        data = fixture_data()
        query = out / 'queries.fa'
        with open(query, 'x') as f:
            for name, dna in data['queries'].items():
                f.write(f'>{name}\n{dna}\n')
        lineage = out / 'lineage.tsv'
        with open(lineage, 'x') as f:
            f.write('taxid\tlineage\n3702\t3702;3701;1\n99\t99;3702;3701;1\n88\t88;99;3702;3701;1\n4530\t4530;4527;1\n')
        results = {}
        for kind in ('orthodb', 'swissprot'):
            with launcher.work_directory(out / kind) as sub:
                ids = ({3702: '3702_0:same.full', 99: '99_0:descendant.full',
                        88: '88_0:merged.full', 4530: '4530_0:keep.full'} if kind == 'orthodb' else
                       {3702: 'sp|SAME.1|SELF', 99: 'sp|DESC.2|CHILD',
                        88: 'sp|MERGED.3|OLD', 4530: 'sp|KEEP.4|CROSS'})
                source = sub / 'source.fa'
                with open(source, 'x') as f:
                    for taxid, name in ids.items():
                        f.write(f'>{name} fixture OX={taxid}\n{data["protein"]}\n')
                filtered = sub / f'{kind}.filtered.fa'
                qc = launcher.filter_source(source, kind, lineage, 3702, filtered)
                require(qc['sequences_out'] == 1 and qc['excluded_taxids'] == [88, 99, 3702], 'filter failed')
                with tempfile.TemporaryDirectory(prefix='fixture-', dir=tmpdir) as tmp:
                    qc.update(launcher.build_and_search(diamond, filtered, query, sub, kind, threads, tmp, rich=True))
                db_fasta = sub / 'database.getseq.fa'
                getseq = [str(diamond), 'getseq', '--db', str(sub / f'{kind}.dmnd'),
                          '--out', str(db_fasta)]
                launcher.write_json(sub / 'getseq.command.json', getseq)
                launcher.execute(getseq, sub / 'getseq.log')
                database = list(launcher.records(db_fasta))
                require([(h.split()[0], seq) for h, seq in database] == [(ids[4530], data['protein'])],
                        'filtered DB contains excluded subject, loses full ID, or changes sequence')
                checks = {}
                with open(sub / f'{kind}.raw17.tsv') as f:
                    for line in f:
                        row = line.rstrip('\n').split('\t')
                        qid = row[0]
                        require(qid in data['queries'] and qid not in checks, 'unexpected/duplicate query hit')
                        require(row[2] == ids[4530], 'excluded or parsed subject ID in search')
                        require(int(row[6]) <= 30 and int(row[7]) >= 150, 'HSP does not bridge both edit anchors')
                        result = replay(row, data['queries'][qid], data['protein'])
                        expected = {'plus1': 1, 'minus1': -1}.get(qid, 0)
                        require(result['net_frameshift_nt'] == expected, f'{qid}: wrong net frameshift')
                        require(len(result['frameshifts']) == (1 if expected else 0), f'{qid}: unexpected frameshift count')
                        require(result['query_reverse'] == (qid == 'reverse'), f'{qid}: wrong strand')
                        if qid == 'insertion':
                            require('I' in result['cigar'], 'insertion control has no insertion')
                        if qid == 'deletion':
                            require('D' in result['cigar'], 'deletion control has no deletion')
                        checks[qid] = result
                require(set(checks) == set(data['queries']), 'missing positive-control hits')
                qc.update(diamond_version=version, query_sha256=launcher.sha256(query),
                          traceback_checks=checks, database_getseq_sha256=launcher.sha256(db_fasta))
                launcher.write_json(sub / 'provenance.json', qc)
                results[kind] = qc
        launcher.write_json(out / 'provenance.json', dict(engine='real DIAMOND', diamond_version=version,
                           diamond_sha256=launcher.sha256(diamond),
                           fixture_sha256=launcher.sha256(__file__), launcher_sha256=launcher.sha256(launcher.__file__),
                           sources=results))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diamond', required=True, type=Path)
    p.add_argument('--out-dir', required=True, type=Path)
    p.add_argument('--tmpdir', required=True, type=Path)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args(argv)
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    run_fixture(args.diamond, args.out_dir, args.tmpdir, args.threads)
    return 0


if __name__ == '__main__':
    sys.exit(main())
