"""Bound streaming EST sample. COMPLETE means input traversal, never A11/C0.

CLI: python -m evidence.est_real_sample --manifest FILE --sha SHA256 --out NEWDIR
Only the function's explicit sam_command_for_test hook accepts a synthetic SAM
reader. CLI always uses the manifest-pinned samtools executable, view -h BAM.
"""
import argparse
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import selectors
import subprocess
import sys
import time

from .est_hsp_contract import adapt_raw13, classify_rich17, LENGTH_CONTRACT, MIDF_LENGTH_CONTRACT
from .est_phase_project import LABEL_CONVENTION, project_codons
from .est_phase_accept import btop_tokens

LIMITS = dict(read_bytes=40 * 1024**3, output_bytes=256 * 1024**2,
              expanded_bases=1000000, line_bytes=1024**2, sequence_bytes=1024**2,
              retained_bytes=32 * 1024**2, records=10000, seconds=1740)


# Authority is compiled here, never supplied by an incoming record or manifest.
PRODUCER_PATH_PROOF_SHA256 = 'a2843aa36a006730c1fd3fe02a34b76afd772f1e01328162f7c0f9af88a2b6a9'
PRODUCER_PATH_PROOF_MAX_BYTES = 256 * 1024


def resolve_producer_path(proof, search_preimage):
    """Resolve only the exact reviewed registry, not a source label or row fit.

    Embedded source/log/package attribution is immutable under the compiled pin.
    This is not a reproducible-build attestation or scientific eligibility.
    """
    if (not isinstance(proof, bytes) or not 0 < len(proof) <= PRODUCER_PATH_PROOF_MAX_BYTES
            or hashlib.sha256(proof).hexdigest() != PRODUCER_PATH_PROOF_SHA256):
        raise ValueError('missing/invalid producer path proof bytes or compiled pin')
    try:
        registry = json.loads(proof)
        if (set(registry) != {'schema', 'scope', 'archive_sha256', 'package_record', 'recipe', 'producers'}
                or registry['schema'] != 'est-producer-path-registry-v1'
                or len(registry['producers']) != 2):
            raise ValueError('producer registry schema')
        def evidence(record):
            data = record['text'].encode('utf-8')
            if (hashlib.sha256(data).hexdigest() != record['sha256']
                    or ('size' in record and len(data) != record['size'])
                    or not os.path.isabs(record['path'])):
                raise ValueError('producer embedded evidence mismatch')
        evidence(registry['package_record'])
        for key in ('build.sh', 'meta.yaml'):
            evidence(registry['recipe'][key])
        if registry['archive_sha256'] not in registry['recipe']['meta.yaml']['text']:
            raise ValueError('producer archive/recipe mismatch')
        routes = {'diamond-2.1.24.178/legacy-direct': LENGTH_CONTRACT,
                  'diamond-2.1.24.178/legacy-blocked-transcript-join': MIDF_LENGTH_CONTRACT}
        matches, seen = [], set()
        fields = {'species', 'arm', 'source', 'reference_sha256', 'query_sha256',
                  'protein_sha256', 'raw_sha256', 'c1_provenance_sha256', 'c1_commands_sha256'}
        for record in registry['producers']:
            if set(record) != {'search_preimage', 'length_contract', 'provenance', 'commands', 'blastx_log', 'makedb_log'}:
                raise ValueError('producer record schema')
            pre = record['search_preimage']
            if (set(pre) != fields or any(not isinstance(v, str) or not v for v in pre.values())
                    or any(not re.fullmatch('[0-9a-f]{64}', pre[k]) for k in fields if k.endswith('_sha256'))):
                raise ValueError('producer search schema')
            identity = canonical(pre)
            if identity in seen or record['length_contract'] not in routes:
                raise ValueError('producer duplicate/conflicting/unknown path')
            seen.add(identity)
            for key in ('provenance', 'commands', 'blastx_log', 'makedb_log'):
                evidence(record[key])
            if any(pre['c1_' + k + '_sha256'] != record[k]['sha256'] for k in ('provenance', 'commands')):
                raise ValueError('producer search/evidence mismatch')
            if pre == search_preimage:
                matches.append(dict(producer_record_sha256=hashlib.sha256(canonical(record)).hexdigest(),
                                    producer_path_proof_sha256=PRODUCER_PATH_PROOF_SHA256,
                                    producer_path=record['length_contract'],
                                    length_contract=routes[record['length_contract']]))
        if len(matches) != 1:
            raise ValueError('producer search preimage not uniquely proven')
        return matches[0]
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('producer registry schema') from exc


class Incomplete(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def bound_id(domain, preimage):
    return {'id': hashlib.sha256(canonical({'domain': 'est-real-sample/v1/' + domain,
                                          'preimage': preimage})).hexdigest(),
            'domain': 'est-real-sample/v1/' + domain, 'preimage': preimage}


def stat(path):
    s = os.stat(path)
    return dict(device=s.st_dev, inode=s.st_ino, size=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def exact(raw, ordinal):
    return dict(ordinal=ordinal, bytes_base64=base64.b64encode(raw).decode(),
                sha256=hashlib.sha256(raw).hexdigest(), extracted_bytes_verified=True)


class Budget:
    def __init__(self, limits):
        self.limits = dict(LIMITS)
        for k, v in limits.items():
            if k not in LIMITS or type(v) is not int or not 0 < v <= LIMITS[k]:
                raise ValueError('invalid resource limit: ' + k)
            self.limits[k] = v
        if self.limits['output_bytes'] < 2 * 1024**2:
            raise ValueError('output_bytes must reserve at least 2 MiB for diagnostics/final status')
        self.used = dict(read_bytes=0, output_bytes=0, expanded_bases=0, retained_bytes=0, records=0)
        self.start = time.monotonic()

    def add(self, key, n):
        self.used[key] += n
        if self.used[key] > self.limits[key]:
            raise Incomplete('resource cap: ' + key)
        if time.monotonic() - self.start > self.limits['seconds']:
            raise Incomplete('wall timeout')

    def lines(self, stream):
        while True:
            raw = stream.readline(self.limits['line_bytes'] + 1)
            self.add('read_bytes', len(raw))
            if len(raw) > self.limits['line_bytes']:
                raise Incomplete('resource cap: line_bytes')
            if not raw:
                return
            yield raw


def hash_file(spec, budget):
    before = stat(spec['path'])
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with open(spec['path'], 'rb') as stream:
        while True:
            block = stream.read(1024**2)
            if not block:
                break
            budget.add('read_bytes', len(block))
            sha.update(block)
            md5.update(block)
    after = stat(spec['path'])
    if before != after:
        raise ValueError('source mutation: ' + spec['path'])
    digests = dict(sha256=sha.hexdigest(), md5=md5.hexdigest())
    for algorithm in ('sha256', 'md5'):
        if algorithm in spec and spec[algorithm] != digests[algorithm]:
            raise ValueError('expected hash mismatch: ' + spec['path'] + ' ' + algorithm)
    return dict(path=spec['path'], before=before, after=after, **digests,
                receipt=spec.get('receipt'), receipt_bound=False,
                fresh_whole_file_verified=True,
                expected_hash_validated=any(k in spec for k in ('sha256', 'md5')),
                extracted_bytes_verified=False)


def file_lines(spec, budget):
    opener = gzip.open if spec['path'].endswith('.gz') else open
    with opener(spec['path'], 'rb') as stream:
        yield from budget.lines(stream)


def fasta(spec, wanted, budget, *, lengths_only=False):
    """One record at a time; no whole protein/genome record list."""
    found, name, chunks, size, header, ordinal = {}, None, [], 0, None, 0

    def finish():
        if name is None or (not lengths_only and name not in wanted):
            return
        if name in found:
            raise ValueError('duplicate selected FASTA ID: ' + name + ' in ' + spec['path'])
        if lengths_only:
            found[name] = size
            budget.add('retained_bytes', len(name) + 64)
        else:
            sequence = b''.join(chunks).decode('ascii')
            budget.add('retained_bytes', len(sequence) + len(header))
            found[name] = dict(sequence=sequence.upper(), sequence_sha256=hashlib.sha256(sequence.upper().encode()).hexdigest(),
                               header=exact(header, ordinal), original_case_sequence=sequence)

    for n, raw in enumerate(file_lines(spec, budget), 1):
        if raw.startswith(b'>'):
            finish()
            tokens = raw[1:].split()
            if not tokens:
                raise ValueError('empty FASTA ID: ' + spec['path'])
            name, header, ordinal = tokens[0].decode('ascii'), raw, n
            chunks, size = [], 0
        else:
            if name is None:
                raise ValueError('FASTA sequence before header: ' + spec['path'])
            seq = raw.strip()
            if any(c in b' \t\r\n' for c in seq):
                raise ValueError('whitespace inside FASTA sequence')
            size += len(seq)
            if not lengths_only and name in wanted:
                if size > budget.limits['sequence_bytes']:
                    raise Incomplete('resource cap: sequence_bytes')
                chunks.append(seq)
    finish()
    if not lengths_only and set(found) != set(wanted):
        raise ValueError('missing FASTA IDs in ' + spec['path'] + ': ' + repr(sorted(set(wanted)-set(found))))
    return found


def relationships(spec, selected, budget):
    found = {}
    for n, raw in enumerate(file_lines(spec, budget), 1):
        f = raw.decode('ascii').rstrip('\r\n').split('\t')
        if n == 1:
            if f != ['accession', 'qlen', 'action', 'kept', 'categories']:
                raise ValueError('unrecognized UniVec report schema')
            continue
        if len(f) != 5:
            raise ValueError('invalid report row: ' + str(n))
        acc, qlen, action, kept, _ = f
        intervals = [] if not kept else [tuple(map(int, x.split('-'))) for x in kept.split(';')]
        for i, (a, b) in enumerate(intervals, 1):
            # Naming is derived from this report row, never suffix stripping.
            name = acc if len(intervals) == 1 else acc + '_part' + str(i)
            if name not in selected:
                continue
            if name in found:
                raise ValueError('ambiguous/duplicate report relationship: ' + name)
            if not 1 <= a <= b <= int(qlen) or action not in ('keep', 'trim', 'split'):
                raise ValueError('report interval/action mismatch: ' + name)
            found[name] = dict(original_id=acc, start0=a-1, end0=b, original_length=int(qlen),
                               action=action, report=exact(raw, n))
    if set(found) != set(selected):
        raise ValueError('missing report relationship: ' + repr(sorted(set(selected)-set(found))))
    return found


def run(manifest, out, *, sam_command_for_test=None, manifest_receipt=None, producer_path_proof=None):
    """Run on files, preserving partial diagnostics and FAILED/INCOMPLETE status.

    Test hook is deliberately not manifest-configurable and is absent from CLI.
    SIGALRM/SIGTERM handling requires the main thread (Linux compute target).
    """
    out = Path(out)
    out.mkdir(parents=False, exist_ok=False)
    report = dict(status='INCOMPLETE', training_eligible=False, full_c0=False,
                  a11_status='unassessable', molecule_counts_complete=False, raster=False,
                  route_a_complete=False, selected_search_rows_complete=False,
                  selected_bam_records_complete=False, sources={}, selected_qnames=[],
                  selection=manifest.get('selection'), runtime_declared=manifest.get('runtime'),
                  python_runtime=sys.version,
                  canonical_manifest_sha256=hashlib.sha256(canonical(manifest)).hexdigest(),
                  manifest_file=manifest_receipt)
    budget = None
    proc = None
    old_handlers = {}
    stderr_prefix = bytearray()
    stderr_sha = hashlib.sha256()
    stderr_total = 0
    stderr_eof = False

    def capture_stderr(block):
        nonlocal stderr_total, stderr_eof
        if not block:
            stderr_eof = True
            return
        stderr_total += len(block)
        stderr_sha.update(block)
        stderr_prefix.extend(block[:max(0, 65536 - len(stderr_prefix))])
        budget.add('read_bytes', len(block))

    def sam_lines():
        # Multiplex both pipes on the main thread: no blocked stderr writer and
        # no cross-thread Budget updates. Only one bounded SAM line is buffered.
        pending = bytearray()
        with selectors.DefaultSelector() as selector:
            for stream in (proc.stdout, proc.stderr):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ)
            while selector.get_map():
                for key, _ in selector.select():
                    block = os.read(key.fd, 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                    if key.fileobj is proc.stderr:
                        capture_stderr(block)
                        continue
                    budget.add('read_bytes', len(block))
                    pending.extend(block)
                    while b'\n' in pending:
                        end = pending.index(b'\n') + 1
                        if end > budget.limits['line_bytes']:
                            raise Incomplete('resource cap: line_bytes')
                        raw = bytes(pending[:end])
                        del pending[:end]
                        yield raw
                    if len(pending) > budget.limits['line_bytes']:
                        raise Incomplete('resource cap: line_bytes')
                    if not block and pending:
                        yield bytes(pending)
                        pending.clear()

    diag = None
    diag_sha = hashlib.sha256()
    def interrupt(signum, frame):
        raise Incomplete('signal/timeout: ' + str(signum))
    try:
        budget = Budget(manifest.get('limits', {}))
        for sig in (signal.SIGALRM, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, interrupt)
        signal.alarm(budget.limits['seconds'])
        diag = open(out / 'diagnostics.jsonl', 'xb')
        def emit(value):
            data = canonical(value) + b'\n'
            # Reserve 1 MiB for the final receipt and marker even on cap failure.
            if budget.used['output_bytes'] + len(data) > budget.limits['output_bytes'] - 1024**2:
                raise Incomplete('resource cap: output_bytes')
            budget.add('output_bytes', len(data))
            diag.write(data)
            diag_sha.update(data)
            diag.flush()

        files = manifest['files']
        if not manifest['species'] or not manifest['arm']:
            raise ValueError('species/arm required')
        searches = manifest['searches']
        if [s['source'] for s in searches] != ['orthodb', 'swissprot']:
            raise ValueError('both ordered search sources required')
        for search in searches:
            for field in ('c1_provenance', 'c1_commands'):
                key = search.get(field)
                spec = files.get(key) if isinstance(key, str) else None
                if (not isinstance(spec, dict) or not isinstance(spec.get('path'), str)
                        or not os.path.isabs(spec['path'])
                        or not isinstance(spec.get('sha256'), str)
                        or not re.fullmatch(r'[0-9a-f]{64}', spec['sha256'])):
                    raise ValueError('search contract requires explicit SHA256-pinned ' + field)
        for key, spec in files.items():
            if not os.path.isabs(spec['path']):
                raise ValueError('absolute input paths required')
            report['sources'][key] = hash_file(spec, budget)
        sources = report['sources']
        for key, spec in files.items():
            receipt = spec.get('receipt')
            if receipt:
                receipt_spec = files[receipt['file']]
                if not receipt_spec.get('sha256'):
                    raise ValueError('receipt file must be SHA256 pinned: ' + key)
                with open(receipt_spec['path'], 'rb') as stream:
                    text = stream.read(1024**2 + 1)
                budget.add('read_bytes', len(text))
                if len(text) > 1024**2:
                    raise Incomplete('receipt size cap')
                if 'field' in receipt:
                    value = json.loads(text)
                    for field in receipt['field']:
                        value = value[field]
                else:
                    matches = re.findall(r'^' + re.escape(receipt['text_key']) + r'\s+(\S+)', text.decode(), re.M)
                    if len(matches) != 1:
                        raise ValueError('ambiguous receipt text key: ' + key)
                    value = matches[0]
                algorithm = receipt.get('algorithm', 'sha256')
                if value != spec.get(algorithm) or value != sources[key][algorithm]:
                    raise ValueError('receipt digest mismatch: ' + key)
                sources[key]['receipt_bound'] = True
        scope = dict(species=manifest['species'], arm=manifest['arm'], reference_sha256=sources['reference']['sha256'])
        # Both searches must resolve before selection or any row eligibility,
        # even F=0. The old identity is retained, never reused for the new run.
        search_ids, producer_paths = {}, {}
        for search in searches:
            key = search['raw']
            pre = dict(scope, source=search['source'], raw_sha256=sources[key]['sha256'],
                       protein_sha256=sources[search['protein']]['sha256'], query_sha256=sources['screened']['sha256'],
                       c1_provenance_sha256=sources[search['c1_provenance']]['sha256'],
                       c1_commands_sha256=sources[search['c1_commands']]['sha256'])
            resolved = resolve_producer_path(producer_path_proof, pre)
            producer_paths[key] = resolved
            search_ids[key] = dict(bound_id('search', dict(pre, **resolved)), legacy_search=bound_id('search', pre))
        report['producer_paths'] = producer_paths
        producer = Path(__file__).with_name('est_hsp_contract.py')
        producer_sha = hash_file({'path': str(producer)}, budget)['sha256']
        report['producer_contract_sha256'] = producer_sha
        selection = manifest['selection']
        if selection['source'] != 'orthodb' or type(selection['count']) is not int or not 1 <= selection['count'] <= 16:
            raise ValueError('selection requires first distinct orthodb QNAMEs, count 1..16')
        if type(selection['prefix_lines']) is not int or not 1 <= selection['prefix_lines'] <= 100000:
            raise ValueError('invalid selection prefix cap')
        selected = []
        prefix_sha = hashlib.sha256()
        with open(files[searches[0]['raw']]['path'], 'rb') as stream:
            for n, raw in enumerate(budget.lines(stream), 1):
                prefix_sha.update(raw)
                report['selection_prefix_lines_read'] = n
                name = raw.split(b'\t', 1)[0].decode('ascii')
                if name not in selected:
                    selected.append(name)
                if len(selected) == selection['count'] or n == selection['prefix_lines']:
                    break
        if not selected:
            raise ValueError('empty selection')
        report['selection_prefix_sha256'] = prefix_sha.hexdigest()
        report['selected_qnames'] = selected
        report['selection_rule'] = 'first distinct exact QNAMEs in capped orthodb physical-line prefix; no outcome filtering'
        rows, targets = {}, {}
        for search in searches:
            key = search['raw']
            emit(dict(type='search', **search_ids[key], receipt=sources[key].get('receipt')))
            rows[key], targets[key] = [], set()
            count = 0
            for count, raw in enumerate(file_lines(files[key], budget), 1):
                f = raw.decode('ascii').rstrip('\r\n').split('\t')
                if f[0] in selected:
                    if len(f) != 13:
                        raise ValueError('selected raw13 width: ' + key + ':' + str(count))
                    budget.add('records', 1)
                    budget.add('retained_bytes', len(raw) * 3)
                    raw_binding = exact(raw, count)
                    rows[key].append((f, raw_binding))
                    emit(dict(type='raw_search_row', search_id=search_ids[key]['id'], source=search['source'],
                              qname=f[0], sseqid=f[2], raw=raw_binding))
                    targets[key].add(f[2])
            sources[key].update(records_at_eof=count, extracted_bytes_verified=True)
        rel = relationships(files['report'], selected, budget)
        dna = fasta(files['screened'], selected, budget)
        originals = fasta(files['original'], {r['original_id'] for r in rel.values()}, budget)
        members = {}
        for q in selected:
            r, seq = rel[q], dna[q]['sequence']
            original = originals[r['original_id']]['sequence']
            expected_tag = 'univec={}:{}-{}/{}'.format(r['action'], r['start0']+1, r['end0'], r['original_length'])
            header = base64.b64decode(dna[q]['header']['bytes_base64']).decode().split()
            if (len(original) != r['original_length'] or original[r['start0']:r['end0']] != seq
                    or len(seq) != r['end0']-r['start0'] or expected_tag not in header):
                raise ValueError('report interval/sequence/header mismatch: ' + q)
            member = bound_id('member', dict(scope, source='screened', qname=q,
                              screened_sha256=sources['screened']['sha256'], original_sha256=sources['original']['sha256'],
                              report_sha256=sources['report']['sha256'], relationship=r))
            identity = dict(member=member['id'], source_sha256=sources['original']['sha256'],
                            source_start0=r['start0'], source_end0=r['end0'], sequence_sha256=dna[q]['sequence_sha256'])
            members[q] = dict(identity=identity, sequence=seq)
            emit(dict(type='member', **member, identity=identity, relationship=r,
                      screened=dna[q], original=originals[r['original_id']], a11_status='unassessable'))
        proteins = {}
        for search in searches:
            proteins[search['raw']] = fasta(files[search['protein']], targets[search['raw']], budget)
            for name, protein in proteins[search['raw']].items():
                emit(dict(type='protein', source=search['source'], source_sha256=sources[search['protein']]['sha256'],
                          sseqid=name, **protein))
        reference = fasta(files['reference'], set(), budget, lengths_only=True)
        placements = {q: [] for q in selected}
        command = sam_command_for_test
        if command is None:
            if 'samtools' not in files or not files['samtools'].get('sha256'):
                raise ValueError('explicit SHA256-pinned samtools required')
            command = [files['samtools']['path'], 'view', '-h', files['bam']['path']]
        report['sam_command'] = command
        report['sam_test_helper'] = sam_command_for_test is not None
        # Account the external reader's full BAM pass as well as our own hash pass.
        budget.add('read_bytes', sources['bam']['before']['size'])
        # Reserve the bounded prefix and its base64/metadata report output before
        # spawning. The final-status reserve remains separate from this charge.
        budget.add('retained_bytes', 65536)
        budget.add('output_bytes', 90 * 1024)
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        dictionary, records, header_count = {}, 0, 0
        for ordinal, raw in enumerate(sam_lines(), 1):
            f = raw.decode('ascii').rstrip('\r\n').split('\t')
            if raw.startswith(b'@'):
                if records:
                    raise ValueError('SAM header after records')
                header_count += 1
                emit(dict(type='sam_header', bam_sha256=sources['bam']['sha256'], raw=exact(raw, ordinal)))
                if f[0] == '@SQ':
                    tags = dict(x.split(':', 1) for x in f[1:])
                    name, length = tags['SN'], int(tags['LN'])
                    if name in dictionary or reference.get(name) != length:
                        raise ValueError('SAM/reference dictionary mismatch: ' + name)
                    dictionary[name] = length
                continue
            records += 1
            if len(f) < 11:
                raise ValueError('malformed SAM record: ' + str(ordinal))
            if f[0] not in selected:
                continue
            budget.add('records', 1)
            budget.add('retained_bytes', len(raw)*3)
            q, flag = f[0], int(f[1])
            placement_id = bound_id('placement', dict(scope, source='bam', bam_sha256=sources['bam']['sha256'],
                                   qname=q, ordinal=ordinal, sam_sha256=hashlib.sha256(raw).hexdigest()))
            ts = [x[5:] for x in f[11:] if x.startswith('ts:A:')]
            p = dict(members[q], contig=f[2], contig_length=dictionary.get(f[2]), flag=flag,
                     pos1=int(f[3]), cigar=f[5], seq=f[9], ts=ts[0] if len(ts) == 1 else None)
            error = ('duplicate ts tags' if len(ts) > 1 else
                     'primary SAM SEQ unavailable; identity cannot be cross-checked' if f[9] == '*' else None)
            placements[q].append((placement_id['id'], p, error))
            emit(dict(type='placement', **placement_id, qname=q, raw=exact(raw, ordinal),
                      primary=not bool(flag & (4 | 256 | 2048)), error=error,
                      disposition='primary_geometry_candidate' if not flag & (4 | 256 | 2048) else 'nonprimary_or_unmapped_diagnostic'))
        if proc.wait() != 0:
            raise ValueError('samtools/helper nonzero exit: ' + str(proc.returncode))
        if not dictionary or dictionary != reference:
            raise ValueError('incomplete SAM/reference dictionary')
        report['bam_records_at_eof'] = records
        report['sam_header_count'] = header_count
        report['reference_dictionary'] = dict(entries=len(dictionary), sha256=hashlib.sha256(canonical(dictionary)).hexdigest(),
                                              exact_entries='diagnostics.jsonl sam_header @SQ records')
        for search in searches:
            key = search['raw']
            for f, raw in rows[key]:
                hsp_id = bound_id('hsp', dict(scope, source=search['source'], search_id=search_ids[key]['id'],
                                             ordinal=raw['ordinal'], raw_sha256=raw['sha256']))
                diagnostic = dict(type='hsp', **hsp_id, raw=raw, qname=f[0], sseqid=f[2],
                                  search_id=search_ids[key]['id'], member_id=members[f[0]]['identity']['member'],
                                  reported_length_at_least_30=(int(f[8]) >= 30) if f[8].isdigit() else None,
                                  length_accepted=None, geometry=[], biological_negative=False)
                try:
                    # Charge the actual trace before expansion, not untrusted col9.
                    # Include deletion columns conservatively; F events cost one.
                    for token in btop_tokens(f[12]):
                        occurrences = int(token)*3 if token.isdigit() else 1 if token[0] in '/\\' else 3
                        budget.add('expanded_bases', occurrences)
                    contract = producer_paths[key]['length_contract']
                    rich = adapt_raw13(f, members[f[0]]['sequence'], proteins[key][f[2]]['sequence'], length_contract=contract)
                    replay = classify_rich17(rich, members[f[0]]['sequence'], proteins[key][f[2]]['sequence'], length_contract=contract)
                    diagnostic.update(replay=replay, rich17=rich, length_accepted=replay['accepted_by_hsp_rule'],
                                      disposition=replay['diagnostic_disposition'])
                    hsp = dict(id=hsp_id['id'], identity=members[f[0]]['identity'], frame=int(f[11]),
                               qstart=int(f[4]), qend=int(f[5]), producer_contract_sha256=producer_sha,
                               label_convention=LABEL_CONVENTION)
                    for pid, p, error in placements[f[0]]:
                        geometry = dict(placement_id=pid, geometry_success=False)
                        if p['flag'] & (4 | 256 | 2048):
                            geometry['disposition'] = 'nonprimary_or_unmapped_diagnostic'
                        else:
                            try:
                                if error:
                                    raise ValueError(error)
                                budget.add('expanded_bases', sum(e['length'] for e in replay['codon_events']))
                                result = project_codons(hsp, replay['codon_events'], p)
                                geometry.update(result=result, geometry_success=bool(result.get('segments')))
                            except (ValueError, KeyError, TypeError) as exc:
                                geometry['error'] = str(exc)
                        diagnostic['geometry'].append(geometry)
                    if not placements[f[0]]:
                        diagnostic['placement_disposition'] = 'no_selected_qname_records_seen_not_biological_zero'
                except (ValueError, KeyError, TypeError) as exc:
                    diagnostic['disposition'] = 'source_bound_replay_failure: ' + str(exc)
                emit(diagnostic)
        for key, value in sources.items():
            value['after'] = stat(value['path'])
            if value['after'] != value['before']:
                raise ValueError('source mutation: ' + key)
            if key in {'screened', 'original', 'report', 'reference', 'bam'} | {s[k] for s in searches for k in ('raw', 'protein')}:
                value['extracted_bytes_verified'] = True
        report.update(status='COMPLETE', selected_search_rows_complete=True, selected_bam_records_complete=True)
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status='INCOMPLETE' if isinstance(exc, (Incomplete, KeyboardInterrupt)) else 'FAILED',
                      error=type(exc).__name__ + ': ' + str(exc))
    finally:
        signal.alarm(0)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if proc is not None:
            cleanup_errors = []
            # An exited group leader does not imply that its descendants exited.
            if report['status'] != 'COMPLETE':
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception as exc:
                    cleanup_errors.append('killpg: ' + str(exc))
            try:
                proc.wait(timeout=1)
            except Exception as exc:
                cleanup_errors.append('wait: ' + str(exc))
            report['sam_returncode'] = proc.returncode
            try:
                # Preserve bytes already in the pipe on abort, but never wait
                # indefinitely for a descendant after a failed kill attempt.
                os.set_blocking(proc.stderr.fileno(), False)
                deadline = time.monotonic() + .25
                while not stderr_eof and time.monotonic() < deadline:
                    try:
                        block = os.read(proc.stderr.fileno(), 65536)
                    except BlockingIOError:
                        break
                    capture_stderr(block)
            except Exception as exc:
                cleanup_errors.append('stderr drain: ' + str(exc))
            report['sam_stderr'] = dict(bytes_base64=base64.b64encode(stderr_prefix).decode(),
                                        total_bytes=stderr_total, sha256=stderr_sha.hexdigest(),
                                        truncated=stderr_total > len(stderr_prefix), eof=stderr_eof,
                                        captured_bytes=len(stderr_prefix), capture_limit_bytes=65536)
            for stream in (proc.stdout, proc.stderr):
                try:
                    stream.close()
                except Exception as exc:
                    cleanup_errors.append('pipe close: ' + str(exc))
            if cleanup_errors:
                report['sam_cleanup_errors'] = cleanup_errors
                if report['status'] == 'COMPLETE':
                    report.update(status='INCOMPLETE', error='SAM cleanup failed',
                                  selected_search_rows_complete=False, selected_bam_records_complete=False)
        if diag is not None:
            diag.flush()
            os.fsync(diag.fileno())
            diag.close()
        if budget is not None:
            report['resources'] = dict(limits=budget.limits, used=budget.used,
                                       elapsed_seconds=time.monotonic()-budget.start)
        report['diagnostics_sha256'] = diag_sha.hexdigest()
        if budget is not None:
            for value in report['sources'].values():
                try:
                    value['after'] = stat(value['path'])
                    if value['after'] != value['before'] and report['status'] == 'COMPLETE':
                        report.update(status='FAILED', error='source mutation during finalization: ' + value['path'],
                                      selected_search_rows_complete=False, selected_bam_records_complete=False)
                except OSError as exc:
                    value['after_error'] = str(exc)
                    if report['status'] == 'COMPLETE':
                        report.update(status='FAILED', error='source stat failed during finalization: ' + value['path'],
                                      selected_search_rows_complete=False, selected_bam_records_complete=False)
        data = canonical(report) + b'\n'
        if len(data) > 1024**2 - 128:
            report = dict(status='INCOMPLETE', error='final report size cap', training_eligible=False,
                          full_c0=False, a11_status='unassessable', selected_search_rows_complete=False,
                          selected_bam_records_complete=False, diagnostics_sha256=diag_sha.hexdigest())
            data = canonical(report) + b'\n'
        with open(out / 'report.json.tmp', 'xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(out / 'report.json.tmp', out / 'report.json')
        with open(out / report['status'], 'xb') as stream:
            stream.write(hashlib.sha256(data).hexdigest().encode() + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--sha', required=True, help='SHA256 of exact manifest file bytes')
    parser.add_argument('--out', required=True)
    parser.add_argument('--producer-path-proof', required=True)
    args = parser.parse_args(argv)
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('real-source CLI requires SLURM_JOB_ID; no login-node execution')
    with open(args.manifest, 'rb') as stream:
        data = stream.read(1024**2 + 1)
    if len(data) > 1024**2 or hashlib.sha256(data).hexdigest() != args.sha:
        parser.error('manifest size/hash mismatch')
    with open(args.producer_path_proof, 'rb') as stream:
        proof = stream.read(PRODUCER_PATH_PROOF_MAX_BYTES + 1)
    result = run(json.loads(data), args.out, producer_path_proof=proof,
                 manifest_receipt=dict(path=str(Path(args.manifest).resolve()), sha256=args.sha))
    return 0 if result['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
