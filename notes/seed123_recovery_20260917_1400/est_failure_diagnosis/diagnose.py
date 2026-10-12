#!/usr/bin/env python3
"""Local retained-evidence diagnosis only. Prints JSON; never changes acceptance.
Run: ./diagnose.py > analysis.json (in this artifact directory).
No original raw/BAM access, subprocesses, network, or DIAMOND execution.
"""
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
SOURCE = Path('/tmp/est-real-source-binding-20260917')
sys.path.insert(0, str(SOURCE / 'revision/scripts'))
from evidence.est_hsp_contract import adapt_raw13
from evidence.est_hsp_expanded import CODE, rc
from evidence.est_phase_project import project_codons, query_reference_runs, LABEL_CONVENTION

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / 'est_result/raw/worker/sample/diagnostics.jsonl'
EXPECTED = '8d8c8be00102a6ba7e7b5cdd875dbeca5a99b18290c024f3690cd849e670fb1d'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def decode(raw):
    data = base64.b64decode(raw['bytes_base64'], validate=True)
    assert sha(data) == raw['sha256']
    return data.decode().rstrip('\r\n')

def replay(row, dna, protein, sign=1):
    """Instrument existing token interpretation, collecting rather than hiding errors.
    Numeric matches require both sources to agree. X is NOT a wildcard here.
    Alternative sign is a negative control, not an acceptance rule.
    """
    qs, qe, ss, se = map(int, row[4:8])
    reverse = int(row[11]) < 0
    dna = rc(dna) if reverse else dna
    pos = len(dna)-qs if reverse else qs-1
    end = len(dna)-qe+1 if reverse else qe
    subject = ss-1
    counts = Counter(dict(M=0, I=0, D=0, F=0))
    errors, shifts, events = [], [], []
    tokens = re.findall(r'\d+|[^\d][^\d]', row[12])
    assert ''.join(tokens) == row[12]
    def column(q, s, token):
        nonlocal pos, subject
        if q in '/\\':
            assert s == '-'
            delta = (1 if q == '\\' else -1) * sign
            shifts.append(dict(at=pos, marker=q, delta=delta))
            pos += delta
            counts['F'] += 1
            return
        counts['D' if q == '-' else 'I' if s == '-' else 'M'] += 1
        if q != '-':
            codon = dna[pos:pos+3] if pos >= 0 else ''
            actual = CODE.get(codon)
            events.append(dict(at=pos, codon=codon, actual=actual, expected=q))
            if actual != q:
                errors.append(dict(side='query', at=pos, codon=codon, actual=actual, expected=q, token=token))
            pos += 3
        if s != '-':
            actual = protein[subject] if 0 <= subject < len(protein) else None
            if actual != s:
                errors.append(dict(side='subject', at=subject, actual=actual, expected=s, token=token))
            subject += 1
    for token in tokens:
        if token.isdigit():
            for _ in range(int(token)):
                aa = protein[subject] if 0 <= subject < len(protein) else '?'
                column(aa, aa, token)
        else:
            column(*token, token)
    mid = sum(counts[k] for k in 'MID')
    return dict(counts=dict(counts), reported=int(row[8]), MID=mid, MIDF=mid+counts['F'],
                query_end=pos, expected_query_end=end, subject_end=subject, expected_subject_end=se,
                endpoints_match=(pos == end and subject == se), errors=errors,
                shifts=shifts, events=events)

def selftests():
    # Minimal frameshift counterexamples with canonical DNA and exact protein.
    for dna, protein, btop in [('ATGCAAAGCT', 'MKA', '1\\-2'),
                                ('ATGAAGCT', 'MEA', '1/-2')]:
        row = ['toy', str(len(dna)), 'protein', '3', '1', str(len(dna)), '1', '3', '4', '1e-9', '50', '1', btop]
        a = replay(row, dna, protein)
        assert not a['errors'] and a['endpoints_match'] and a['MID'] == 3 and a['MIDF'] == 4
        assert not replay(row, dna, protein, -1)['endpoints_match']
    # Literal X is not equivalent to proof of a canonical residue.
    row = ['toy', '3', 'protein', '1', '1', '3', '1', '1', '1', '1e-9', '50', '1', 'XT']
    assert replay(row, 'ACG', 'T')['errors'][0]['expected'] == 'X'
    row[12] = 'EX'
    assert replay(row, 'GAA', 'E')['errors'][0]['side'] == 'subject'
    row[12] = '1'
    assert replay(row, 'ACG', 'K')['errors'][0]['token'] == '1'


def main():
    selftests()
    data = INPUT.read_bytes()
    assert sha(data) == EXPECTED
    report_bytes = (INPUT.parent / 'report.json').read_bytes()
    assert sha(report_bytes) == '2e97691107440fd6d9e09341d3feb2588823cc28f9c9da5bf4f064a6d95728d8'
    assert sha((SOURCE / 'revision/scripts/evidence/est_hsp_contract.py').read_bytes()) == 'fcb8a4b2413f12dc5c5b0c4d55676cedeac21b3fcb26cf849c089657650a3954'
    rows = [json.loads(line) for line in data.splitlines()]
    members = {r['id']: r for r in rows if r['type'] == 'member'}
    proteins = {(r['source'], r['sseqid']): r for r in rows if r['type'] == 'protein'}
    placements = {r['id']: r for r in rows if r['type'] == 'placement'}
    for m in members.values():
        rel = m['relationship']
        assert m['screened']['sequence'] == m['original']['sequence'][rel['start0']:rel['end0']]
        for k in ('screened', 'original'):
            assert sha(m[k]['sequence'].encode()) == m[k]['sequence_sha256']
    for p in proteins.values():
        assert sha(p['sequence'].encode()) == p['sequence_sha256']
    output = []
    geometry_failures = []
    for h in (r for r in rows if r['type'] == 'hsp'):
        row = decode(h['raw']).split('\t')
        assert len(row) == 13
        m = members[h['member_id']]
        dna = m['screened']['sequence']
        protein = proteins[(h['preimage']['source'], h['sseqid'])]['sequence']
        try:
            adapt_raw13(row, dna, protein)
            observed = 'HSP_RULE_ACCEPTED' if int(row[8]) >= 30 else 'BELOW_HSP_LENGTH_30'
        except ValueError as e:
            observed = 'source_bound_replay_failure: ' + str(e)
        assert observed == h['disposition'], (h['id'], observed, h['disposition'])
        a = replay(row, dna, protein)
        negative = replay(row, dna, protein, -1) if a['shifts'] else None
        entry = dict(id=h['id'], qname=h['qname'], sseqid=h['sseqid'], source=h['preimage']['source'],
                     ordinal=h['raw']['ordinal'], raw_sha256=h['raw']['sha256'], raw13=row,
                     disposition=observed, analysis=a,
                     reversed_shift_sign=None if negative is None else dict(errors=len(negative['errors']), endpoints_match=negative['endpoints_match']))
        output.append(entry)
        for g in h.get('geometry', []):
            p = placements[g['placement_id']]
            sam = decode(p['raw']).split('\t')
            old = g['result']
            placement = dict(old['placement_coordinates'], sequence=dna, identity=m['identity'], seq=sam[9])
            assert int(sam[1]) == placement['flag'] and sam[5] == placement['cigar']
            query_reference_runs(placement)
            hs = dict(id=h['id'], identity=m['identity'], frame=int(row[11]), qstart=int(row[4]), qend=int(row[5]),
                      producer_contract_sha256=old['producer_contract_sha256'], label_convention=LABEL_CONVENTION)
            result = project_codons(hs, h['replay']['codon_events'], placement)
            assert result == old
            if not g['geometry_success']:
                intervals = sorted((min(e['q0'], e['q0']+e['direction']*(e['length']-1)),
                                    max(e['q0'], e['q0']+e['direction']*(e['length']-1))+1, i)
                                   for i,e in enumerate(h['replay']['codon_events']))
                overlaps = [(x,y) for x,y in zip(intervals, intervals[1:]) if y[0] < x[1]]
                geometry_failures.append(dict(id=h['id'], qname=h['qname'], sseqid=h['sseqid'],
                    placement_id=p['id'], reasons=result['reasons'], overlaps=overlaps, shifts=a['shifts'],
                    cigar=sam[5], sam_sha256=p['raw']['sha256']))
    counts = Counter(r['disposition'] for r in output)
    assert len(output) == 462 and counts['HSP_RULE_ACCEPTED'] == 385
    assert len(geometry_failures) == 5
    # Every observed source-specific relation is asserted, not generalized to other producers.
    for r in output:
        a = r['analysis']
        assert a['reported'] == (a['MIDF'] if r['source'] == 'orthodb' else a['MID'])
    inferred_mismatches = []
    for r in output:
        for e in r['analysis']['errors']:
            if e['expected'] != 'X':
                assert e['side'] == 'query' and e['token'].isdigit()
                witnesses = [s['id'] for s in output if s['qname'] == r['qname'] and s['raw13'][11] == r['raw13'][11]
                             and any(v['side'] == 'query' and v['at'] == e['at'] and v['expected'] == 'X'
                                     for v in s['analysis']['errors'])]
                assert witnesses
                inferred_mismatches.append(dict(id=r['id'], error=e, X_witness_ids=witnesses))
    summary = dict(selftests='PASS', dispositions=dict(counts), geometry_failures=len(geometry_failures),
        endpoints_match=sum(r['analysis']['endpoints_match'] for r in output),
        reported_equals_MID=sum(r['analysis']['reported']==r['analysis']['MID'] for r in output),
        reported_equals_MIDF=sum(r['analysis']['reported']==r['analysis']['MIDF'] for r in output),
        errors= dict(Counter((e['side']+':'+str(e['expected'])+':'+str(e['actual'])) for r in output for e in r['analysis']['errors'])),
        members_screened_equal_original_slice=len(members))
    print(json.dumps(dict(diagnostics_sha256=EXPECTED, source_hashes={p.name:sha(p.read_bytes()) for p in
        (SOURCE/'revision/scripts/evidence').glob('est_*.py')}, summary=summary,
        inferred_numeric_run_mismatches=inferred_mismatches,
        geometry_failures=geometry_failures, hsps=output), indent=2))

if __name__ == '__main__':
    main()
