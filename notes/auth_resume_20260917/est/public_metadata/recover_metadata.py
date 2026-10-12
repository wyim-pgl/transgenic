#!/usr/bin/env python3
"""Bounded one-time EFetch; default invocation is strictly offline validation.
Run: python recover_metadata.py --fetch  (once), then python recover_metadata.py
No FASTA, downstream source-acceptance changes, or molecule counts are produced.
"""
import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone
from urllib.parse import urlencode

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PARENT = ROOT / 'notes/seed123_recovery_20260917_1400/est_result/raw/worker/sample/diagnostics.jsonl'
RAW = HERE / 'nuccore_16.genbank.raw'
RECEIPT = HERE / 'fetch_receipt.json'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n')

def load_members():
    data = PARENT.read_bytes()
    rows = [json.loads(line) for line in data.splitlines()]
    members = [row for row in rows if row.get('type') == 'member']
    require(len(members) == 16, 'expected exactly 16 retained members')
    result = {}
    for row in members:
        require(row['preimage']['species'] == 'Sitalica', 'wrong species')
        accession = row['relationship']['original_id']
        require(re.fullmatch(r'[A-Z]+\d+\.\d+', accession), 'unversioned accession')
        require(accession not in result, 'duplicate retained accession')
        header = base64.b64decode(row['original']['header']['bytes_base64'], validate=True)
        require(sha(header) == row['original']['header']['sha256'], 'parent header hash mismatch')
        require(header.split()[0] == ('>' + accession).encode(), 'parent header ID mismatch')
        require(row['preimage']['relationship']['original_id'] == accession, 'parent relationship mismatch')
        require(row['original']['sequence'] == row['original']['original_case_sequence'].upper(), 'parent case mismatch')
        require(sha(row['original']['sequence'].encode()) == row['original']['sequence_sha256'], 'parent sequence hash mismatch')
        result[accession] = row
    return result, sha(data)

def section(text, name):
    match = re.search(r'^' + re.escape(name) + r'\b[^\n]*(?:\n[ \t][^\n]*)*\n?', text, re.M)
    return match.group() if match else ''

def validate_extract(raw, expected):
    require(0 < len(raw) <= 1048576, 'response size outside budget')
    # Include record terminators in byte spans; allow whitespace only outside records.
    matches = list(re.finditer(rb'^LOCUS [\s\S]*?^//\r?\n', raw, re.M))
    require(len(matches) == len(expected), 'wrong record count')
    cursor, validated = 0, {}
    for match in matches:
        require(not raw[cursor:match.start()].strip(), 'unparsed response bytes')
        cursor = match.end()
        record = match.group().decode('ascii')
        versions = re.findall(r'^VERSION\s+(\S+)', record, re.M)
        require(len(versions) == 1, 'missing or multiple VERSION')
        accession = versions[0]
        require(accession in expected and accession not in validated, 'extra, wrong-version, or duplicate ID')
        require(re.findall(r'^ACCESSION\s+(\S+)', record, re.M) == [accession.split('.')[0]], 'ACCESSION mismatch')
        locus = re.match(r'LOCUS\s+(\S+)\s+(\d+)\s+bp\s+[^\n]*?(\d{2}-[A-Z]{3}-\d{4})\n', record)
        require(locus is not None and locus[1] == accession.split('.')[0], 'LOCUS identity/date missing')
        datetime.strptime(locus[3], '%d-%b-%Y')
        origins = re.findall(r'^ORIGIN[^\n]*\n([\s\S]*?)^//', record, re.M)
        require(len(origins) == 1, 'missing/multiple ORIGIN')
        require(re.fullmatch(r'[\s\dACGTRYSWKMBDHVNacgtryswkmbdhvn]+', origins[0]), 'invalid ORIGIN bytes')
        seq = re.sub(r'[\s\d]', '', origins[0]).upper()
        parent = expected[accession]
        original = parent['original']
        require(seq == original['sequence'] == original['original_case_sequence'].upper(), 'sequence mismatch')
        require(sha(seq.encode()) == original['sequence_sha256'], 'retained sequence hash mismatch')
        require(len(seq) == int(locus[2]) == parent['relationship']['original_length'], 'length mismatch')
        validated[accession] = (record, match, locus[3], seq)
    require(not raw[cursor:].strip(), 'trailing response bytes')
    require(set(validated) == set(expected), 'ID set mismatch')
    # Extract metadata only after every sequence and the whole record set pass.
    rows = []
    for accession, parent in expected.items():
        record, match, date, seq = validated[accession]
        features = record[record.index('FEATURES'):record.index('\nORIGIN') + 1]
        sources = re.findall(r'^     source\s[^\n]*\n(?:^ {21}[^\n]*\n)*', features, re.M)
        require(len(sources) == 1, 'expected one source feature')
        qualifiers = {}
        for q in re.finditer(r'^ {21}/([A-Za-z_]+)(?:=([^\n]*(?:\n {21}(?!/)[^\n]*)*))?', sources[0], re.M):
            value = q[2] if q[2] is not None else ''
            # Remove qualifier quote delimiters only, preserving newlines and indentation.
            if value.startswith('"'):
                require(value.endswith('"'), 'unterminated quoted qualifier')
                value = value[1:-1]
            qualifiers.setdefault(q[1], []).append(value)
        dblink = section(record, 'DBLINK')
        rows.append(dict(accession=accession, member_id=parent['id'], record_date=date,
                         locus_raw=section(record, 'LOCUS'), accession_raw=section(record, 'ACCESSION'),
                         version_raw=section(record, 'VERSION'), definition_raw=section(record, 'DEFINITION'),
                         dblink_raw=dblink, biosample=re.findall(r'BioSample:\s*(SAM\w+)', dblink),
                         bioproject=re.findall(r'BioProject:\s*(PRJ\w+)', dblink),
                         comment_raw=section(record, 'COMMENT'), header_raw=record[:record.index('FEATURES')],
                         features_raw=features, source_feature_raw=sources[0], qualifiers=qualifiers,
                         sequence_length=len(seq), sequence_sha256=sha(seq.encode()),
                         parent_original_sequence_sha256=parent['original']['sequence_sha256'],
                         parent_source_sha256=parent['preimage']['original_sha256'],
                         sequence_exact_match=True, record_sha256=sha(match.group()),
                         raw_byte_start=match.start(), raw_byte_end=match.end(),
                         a11_status='unassessable', training_eligible=False))
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    members, parent_sha = load_members()
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?' + urlencode(dict(
        db='nuccore', id=','.join(members), rettype='gbwithparts', retmode='text'))
    if args.fetch:
        require(not RAW.exists() and not RECEIPT.exists(), 'fetch already attempted; refusing another batch')
        command = ['curl', '--proto', '=https', '--max-time', '30', '--connect-timeout', '5',
                   '--max-filesize', '1048576', '--fail', '--silent', '--show-error',
                   '--dump-header', str(HERE / 'nuccore_headers.raw'), '--output', str(RAW),
                   '--write-out', '%{json}', url]
        started = datetime.now(timezone.utc).isoformat()
        save(RECEIPT.name, dict(status='attempt_started', started_utc=started, url=url, command=command))
        proc = subprocess.run(command, capture_output=True, timeout=35)
        ended = datetime.now(timezone.utc).isoformat()
        (HERE / 'curl_stderr.txt').write_bytes(proc.stderr)
        (HERE / 'curl_transfer.json').write_bytes(proc.stdout)
        payload = RAW.read_bytes() if RAW.exists() else b''
        save(RECEIPT.name, dict(status='downloaded' if proc.returncode == 0 else 'failed',
             started_utc=started, ended_utc=ended, url=url, command=command, exit_code=proc.returncode,
             requested_accessions=list(members), raw_bytes=len(payload), raw_sha256=sha(payload),
             parent_path=str(PARENT), parent_sha256=parent_sha,
             header_sha256=sha((HERE / 'nuccore_headers.raw').read_bytes()),
             redirects_followed=False, retries=0, scope='existing 16 members only'))
        require(proc.returncode == 0, 'curl failed; retained evidence, no automatic retry')
    receipt = json.loads(RECEIPT.read_text())
    raw = RAW.read_bytes()
    require(receipt['status'] == 'downloaded' and receipt['exit_code'] == 0, 'unsuccessful fetch')
    require(receipt['url'] == url and receipt['requested_accessions'] == list(members), 'request identity mismatch')
    require(receipt['parent_sha256'] == parent_sha and receipt['raw_sha256'] == sha(raw), 'receipt hash mismatch')
    require(receipt['raw_bytes'] == len(raw), 'receipt length mismatch')
    rows = validate_extract(raw, members)
    save('metadata.json', rows)
    keys = ['accession', 'member_id', 'record_date', 'sequence_length', 'sequence_sha256', 'sequence_exact_match',
            'biosample', 'bioproject', 'clone_lib', 'clone', 'tissue_type', 'strain', 'cultivar', 'note',
            'comment_raw', 'source_feature_raw', 'record_sha256', 'raw_byte_start', 'raw_byte_end', 'a11_status', 'training_eligible']
    with (HERE / 'metadata.tsv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=keys, delimiter='\t')
        writer.writeheader()
        for row in rows:
            flat = {k: row.get(k, row['qualifiers'].get(k, [])) for k in keys}
            # JSON strings preserve exact multiline text in one physical TSV row.
            for key, value in flat.items():
                if isinstance(value, (list, dict)) or key.endswith('_raw'):
                    flat[key] = json.dumps(value, ensure_ascii=True)
            writer.writerow(flat)
    save('validation_receipt.json', dict(validated_utc=datetime.now(timezone.utc).isoformat(),
         parent_path=str(PARENT), parent_sha256=parent_sha, batch_sha256=sha(raw),
         requested_and_returned_exact_accessions=list(members), exact_sequence_matches=len(rows),
         no_extra_or_duplicate_records=True, parser_sha256=sha(Path(__file__).read_bytes()),
         outputs={name: sha((HERE / name).read_bytes()) for name in ['metadata.json', 'metadata.tsv']},
         source_acceptance_changed=False, actual_a11_audit_complete=False, training_eligible=False))
    print(json.dumps(dict(records=len(rows), exact_sequences=len(rows), bytes=len(raw), sha256=sha(raw))))

if __name__ == '__main__':
    main()
