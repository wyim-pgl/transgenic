#!/usr/bin/env python3
"""New diagnostic replay of pinned job6165875 retained bytes only.

No scan, subprocess, original FASTA/BAM/raw access or historical reclassification.
The byte-only replay API cannot read any input files. CLI reads only the fixed
local retained artifacts and current implementation source, writes a NEW directory.
"""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE / 'revision/scripts'))
from evidence import est_real_sample as binder
from evidence.est_hsp_contract import adapt_raw13, classify_rich17
from evidence.est_phase_project import LABEL_CONVENTION, project_codons

ROOT = Path('/mnt/f/scratch/develop/Transgenic/transgenic_local/notes/seed123_recovery_20260917_1400')
REPORT_SHA256 = '2e97691107440fd6d9e09341d3feb2588823cc28f9c9da5bf4f064a6d95728d8'
DIAGNOSTICS_SHA256 = '8d8c8be00102a6ba7e7b5cdd875dbeca5a99b18290c024f3690cd849e670fb1d'
# Read only implementation files at import, never paths from retained records.
SOURCE_FILES = [Path(__file__).resolve()] + [SOURCE/'revision/scripts/evidence'/name for name in
                ('est_real_sample.py', 'est_hsp_contract.py', 'est_hsp_expanded.py', 'est_phase_accept.py', 'est_phase_project.py')]
SOURCE_IDENTITIES = {str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE_FILES}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check(value, message):
    if not value:
        raise ValueError(message)


def verify_id(record, kind):
    expected = binder.bound_id(kind, record['preimage'])
    check(all(record[k] == expected[k] for k in ('id', 'domain')), 'parent identity mismatch: ' + kind)


def decode(record):
    data = base64.b64decode(record['bytes_base64'], validate=True)
    check(sha(data) == record['sha256'] and type(record['ordinal']) is int and record['ordinal'] > 0,
          'retained raw identity mismatch')
    return data.decode('ascii').rstrip('\r\n')


def verify_sequence(record):
    check(sha(record['sequence'].encode()) == record['sequence_sha256']
          and record['original_case_sequence'].upper() == record['sequence'], 'retained sequence identity mismatch')


def replay(report_bytes, diagnostics_bytes, proof):
    """Pure byte-only replay. Compiled parent and proof pins cannot be overridden."""
    check(isinstance(report_bytes, bytes) and len(report_bytes) <= 1024**2
          and sha(report_bytes) == REPORT_SHA256, 'parent report pin/size mismatch')
    check(isinstance(diagnostics_bytes, bytes) and len(diagnostics_bytes) <= 16*1024**2
          and sha(diagnostics_bytes) == DIAGNOSTICS_SHA256, 'parent diagnostics pin/size mismatch')
    report = json.loads(report_bytes)
    check(report['diagnostics_sha256'] == DIAGNOSTICS_SHA256 and report['status'] == 'COMPLETE', 'parent linkage')
    rows = [json.loads(line) for line in diagnostics_bytes.splitlines()]
    sources = report['sources']
    searches, members, proteins, placements, raw_rows, dictionary = {}, {}, {}, {}, {}, {}
    # Resolve ALL searches before any HSP, including no-shift rows.
    for r in rows:
        if r['type'] != 'search':
            continue
        verify_id(r, 'search')
        pre = r['preimage']
        path = binder.resolve_producer_path(proof, pre)
        for field, key in [('query', 'screened'), ('reference', 'reference'),
                           ('protein', pre['source']+'_protein'), ('raw', pre['source']+'_raw'),
                           ('c1_provenance', pre['source']+'_provenance'), ('c1_commands', pre['source']+'_commands')]:
            check(pre[field+'_sha256'] == sources[key]['sha256'], 'parent search/source mismatch')
        check(r['id'] not in searches, 'duplicate parent search')
        searches[r['id']] = dict(parent=r, resolved=path, new=binder.bound_id('search', dict(pre, **path)))
    check(len(searches) == 2, 'parent search count')
    scope = {k: next(iter(searches.values()))['parent']['preimage'][k] for k in ('species', 'arm', 'reference_sha256')}
    for r in rows:
        kind = r['type']
        if 'raw' in r:
            decode(r['raw'])
        if kind in ('member', 'placement', 'hsp'):
            verify_id(r, kind)
            check(all(r['preimage'][k] == v for k, v in scope.items()), 'parent scope mismatch')
        if kind == 'member':
            pre, rel = r['preimage'], r['relationship']
            for field in ('original', 'screened', 'report'):
                check(pre[field+'_sha256'] == sources[field]['sha256'], 'parent member/source mismatch')
            for field in ('screened', 'original'):
                verify_sequence(r[field])
                header = decode(r[field]['header'])
                expected = pre['qname'] if field == 'screened' else rel['original_id']
                check(header.split()[0] == '>'+expected, 'parent FASTA ID mismatch')
            decode(rel['report'])
            dna, original = r['screened']['sequence'], r['original']['sequence']
            check(rel == pre['relationship'] and dna == original[rel['start0']:rel['end0']]
                  and len(original) == rel['original_length'], 'parent member relationship mismatch')
            expected_identity = dict(member=r['id'], source_sha256=pre['original_sha256'],
                                     source_start0=rel['start0'], source_end0=rel['end0'], sequence_sha256=sha(dna.encode()))
            check(r['identity'] == expected_identity and r['id'] not in members, 'parent member identity mismatch')
            members[r['id']] = r
        elif kind == 'protein':
            verify_sequence(r)
            check(decode(r['header']).split()[0] == '>'+r['sseqid'], 'parent protein FASTA ID')
            check(r['source_sha256'] == sources[r['source']+'_protein']['sha256'], 'parent protein source')
            key = (r['source'], r['sseqid'])
            check(key not in proteins, 'duplicate parent protein')
            proteins[key] = r
        elif kind == 'placement':
            pre = r['preimage']
            check(pre['sam_sha256'] == r['raw']['sha256'] and pre['ordinal'] == r['raw']['ordinal']
                  and pre['bam_sha256'] == sources['bam']['sha256'] and r['id'] not in placements,
                  'parent placement identity mismatch')
            placements[r['id']] = r
        elif kind == 'raw_search_row':
            search = searches[r['search_id']]['parent']['preimage']
            fields = decode(r['raw']).split('\t')
            check(len(fields) == 13 and fields[0] == r['qname'] and fields[2] == r['sseqid']
                  and r['source'] == search['source'], 'parent raw row linkage')
            key = (r['search_id'], r['raw']['ordinal'])
            check(key not in raw_rows, 'duplicate retained raw row')
            raw_rows[key] = r
        elif kind == 'sam_header':
            check(r['bam_sha256'] == sources['bam']['sha256'], 'parent SAM header source')
            fields = decode(r['raw']).split('\t')
            if fields[0] == '@SQ':
                tags = dict(x.split(':', 1) for x in fields[1:])
                check(tags['SN'] not in dictionary, 'duplicate parent dictionary')
                dictionary[tags['SN']] = int(tags['LN'])
    check(sha(binder.canonical(dictionary)) == report['reference_dictionary']['sha256'], 'parent reference dictionary')
    output, seen = [], set()
    for h in (r for r in rows if r['type'] == 'hsp'):
        pre = h['preimage']
        check(h['search_id'] == pre['search_id'], 'parent HSP search identity')
        search = searches[h['search_id']]
        raw = raw_rows[(h['search_id'], pre['ordinal'])]
        check(raw['raw'] == h['raw'] and pre['raw_sha256'] == h['raw']['sha256']
              and raw['source'] == pre['source'] and h['id'] not in seen, 'parent HSP/raw linkage')
        seen.add(h['id'])
        fields = decode(h['raw']).split('\t')
        member = members[h['member_id']]
        check(fields[0] == h['qname'] == member['preimage']['qname'] and fields[2] == h['sseqid'], 'parent HSP sequence IDs')
        dna = member['screened']['sequence']
        protein = proteins[(pre['source'], fields[2])]['sequence']
        new = binder.bound_id('hsp', dict(pre, search_id=search['new']['id']))
        item = dict(new, parent_hsp_id=h['id'], parent_search_id=h['search_id'],
                    search_id=search['new']['id'], member_id=member['id'], raw=h['raw'],
                    **search['resolved'], length_accepted=None, geometry=[], biological_negative=False)
        try:
            contract = search['resolved']['length_contract']
            rich = adapt_raw13(fields, dna, protein, length_contract=contract)
            result = classify_rich17(rich, dna, protein, length_contract=contract)
            item.update(rich17=rich, replay=result, length_accepted=result['accepted_by_hsp_rule'],
                        disposition=result['diagnostic_disposition'])
            hs = dict(id=new['id'], identity=member['identity'], frame=int(fields[11]),
                      qstart=int(fields[4]), qend=int(fields[5]),
                      producer_contract_sha256=SOURCE_IDENTITIES['revision/scripts/evidence/est_hsp_contract.py'],
                      label_convention=LABEL_CONVENTION)
            for p in placements.values():
                if p['qname'] != fields[0]:
                    continue
                sam = decode(p['raw']).split('\t')
                check(sam[0] == p['qname'] == p['preimage']['qname'], 'parent SAM QNAME')
                flag = int(sam[1])
                g = dict(placement_id=p['id'], geometry_success=False)
                if flag & (4 | 256 | 2048):
                    g['disposition'] = 'nonprimary_or_unmapped_diagnostic'
                else:
                    try:
                        ts = [x[5:] for x in sam[11:] if x.startswith('ts:A:')]
                        check(len(ts) <= 1 and sam[9] != '*', 'retained SAM ts/SEQ unavailable')
                        placement = dict(identity=member['identity'], sequence=dna, contig=sam[2],
                                         contig_length=dictionary.get(sam[2]), flag=flag, pos1=int(sam[3]),
                                         cigar=sam[5], seq=sam[9], ts=ts[0] if ts else None)
                        projected = project_codons(hs, result['codon_events'], placement)
                        g.update(result=projected, geometry_success=bool(projected.get('segments')))
                    except (ValueError, KeyError, TypeError) as exc:
                        g['error'] = str(exc)
                item['geometry'].append(g)
        except (ValueError, KeyError, TypeError) as exc:
            item['disposition'] = 'source_bound_replay_failure: ' + str(exc)
        output.append(item)
    check(len(output) == len(raw_rows), 'parent row coverage')
    return dict(schema='retained-est-producer-length-replay-v1', parent_job='6165875',
                parent_report_sha256=REPORT_SHA256, parent_diagnostics_sha256=DIAGNOSTICS_SHA256,
                producer_path_proof_sha256=sha(proof), new_source_sha256=dict(SOURCE_IDENTITIES),
                status='DIAGNOSTIC_REPLAY_ONLY',
                training_eligible=False, full_c0=False, molecule_counts_complete=False,
                raster=False, route_a_complete=False, a11_status='unassessable',
                original_inputs_reopened=False, historical_result_reclassified=False,
                searches=list(searches.values()), hsps=output,
                counts=dict(hsps=len(output), dispositions=dict(Counter(h['disposition'] for h in output)),
                            geometry_success=sum(g['geometry_success'] for h in output for g in h['geometry']),
                            geometry_unsuccessful=sum(not g['geometry_success'] for h in output for g in h['geometry'])))


def bounded(path, cap):
    with path.open('rb') as stream:
        data = stream.read(cap+1)
    check(len(data) <= cap, 'retained/source size cap')
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='new directory only')
    args = parser.parse_args(argv)
    retained = ROOT / 'est_result/raw/worker/sample'
    result = replay(bounded(retained/'report.json', 1024**2),
                    bounded(retained/'diagnostics.jsonl', 16*1024**2),
                    bounded(ROOT/'est_failure_diagnosis/producer_path_proof.json', binder.PRODUCER_PATH_PROOF_MAX_BYTES))
    out = Path(args.out)
    out.mkdir(parents=False, exist_ok=False)
    data = binder.canonical(result) + b'\n'
    with (out/'replay.json').open('xb') as stream:
        stream.write(data)
    print(json.dumps(dict(replay_sha256=sha(data), counts=result['counts']), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
