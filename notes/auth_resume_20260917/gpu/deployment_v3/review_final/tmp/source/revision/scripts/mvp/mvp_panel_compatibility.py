#!/usr/bin/env python3
"""CPU-only proof connecting one immutable historical panel to seed123 consumption.

No producer migration, relocation fallback, training, or receipt-only fast path.
Run the CLI only in a reviewed compute allocation: preflight hashes the raw DBs.
"""
import argparse
import hashlib
import os
from pathlib import Path
import platform

import duckdb
import numpy as np
import torch

import mvp_prepare_panel_inputs as p
import mvp_produce_counts as producer
import mvp_verify_panel_inputs as v
from transgenic.evidence import mvp_decode as d

_ACCEPTED_MANIFEST_SHA256 = '7528c9bff5a9a9b177b1ad333ad76525a1d02d3c4cde4e0550865ff12e0acc0e'
_ACCEPTED_PANEL_ROOT = Path('/data/gpfs/assoc/pgl/data/Transgenic/evidence/mvp50_reference_map_20260916/outputs/job-6165628/panel')
_DATA_FIELDS = ('panels', 'fasta', 'reference', 'acceptance', 'inventory', 'receipts', 'corpus', 'evidence')
_SOURCE_NAMES = ('mvp_prepare_panel_inputs.py', 'mvp_tracks.py', 'mvp_windows.py',
                 'mvp_cli.py', 'mvp_cli_runtime.py', 'mvp_produce_counts.py')


def _canonical_path(value):
    path = Path(value)
    d.require(path.is_absolute() and '..' not in path.parts and path == path.resolve(strict=True),
              'canonical absolute non-symlink path required')
    return path


def _data_paths(payload):
    return [payload[k] for k in _DATA_FIELDS[:3]] + [payload['candidate'][k] for k in _DATA_FIELDS[3:]]


def _recheck(records):
    def paths_and_wal():
        for record in records:
            path = _canonical_path(record['path'])
            # Include both explicit DB identities even when their suffix is nonstandard.
            d.require(not os.path.lexists(str(path)+'.wal'), 'DB WAL appeared')
    paths_and_wal()
    producer.recheck_files(records)
    paths_and_wal()


def consumer_closure():
    """Exact current panel consumer/helper + preflight dependency source identities."""
    from transgenic.training import mvp_cli, mvp_cli_runtime, mvp_runtime, mvp_profiles
    modules = (v, p, p.mvp_tracks, p.w, mvp_cli, mvp_cli_runtime, producer, d, mvp_runtime, mvp_profiles)
    paths = [Path(__file__), *(Path(m.__file__) for m in modules),
             Path(__file__).with_name('mvp_decode_panel.py'),
             Path(__file__).with_name('mvp_decode_runtime.py')]
    return [p.file_identity(_canonical_path(path)) for path in paths]


def _profile(payload):
    d.require(payload.get('profile') == 'development-single', 'development-single profile required')
    for key, expected in (('seed',123), ('aug_seed',20261037)):
        if key in payload:
            d.require(type(payload[key]) is int and payload[key] == expected, 'compatibility '+key+' mismatch')


def _historical(root):
    root = _canonical_path(root)
    d.require(root == _ACCEPTED_PANEL_ROOT, 'unaccepted historical panel path')
    for name in ('manifest.json','COMPLETE'):
        _canonical_path(root/name)
    # Pin BEFORE parsing or touching any historical/request-supplied large path.
    manifest_id = d.checked_file({'path':str(root/'manifest.json'), 'sha256':_ACCEPTED_MANIFEST_SHA256})
    root, manifest, records = v._read_panel(root)
    d.require(records[1] == manifest_id, 'historical manifest changed')
    old = manifest['inputs']
    d.require(len(old) == 15 and tuple(Path(r['path']).name for r in old[8:14]) == _SOURCE_NAMES,
              'exact historical producer inventory required')
    for r in records + old:
        _canonical_path(r['path'])
    _recheck(records + old)
    request = d.read_json(old[-1]['path'])
    d.require([r['path'] for r in old[:8]] == _data_paths(request), 'historical ordered request inputs mismatch')
    d.require(manifest['rasterizer'] == old[9], 'historical rasterizer identity mismatch')
    return root, manifest, records + old, request


def _measure(root, payload):
    _profile(payload)
    root, manifest, records, old_request = _historical(root)
    closure = consumer_closure()
    records.extend(closure)
    for path in _data_paths(payload):
        _canonical_path(path)
    loci, authority, current, evidence = p.preflight(payload)
    d.require(len(current) == 14 and [r['path'] for r in current[:8]] == _data_paths(payload),
              'current ordered input inventory mismatch')
    d.require(tuple(Path(r['path']).name for r in current[8:]) == _SOURCE_NAMES,
              'current source inventory mismatch')
    # Explicit semantic fields, not "drop all code identities then compare lists".
    for field, old, new in zip(_DATA_FIELDS, manifest['inputs'][:8], current[:8]):
        d.require((old['sha256'],old['size']) == (new['sha256'],new['size']),
                  'semantic input identity mismatch: '+field)
    d.require(old_request['candidate']['acceptance_sha256'] == payload['candidate']['acceptance_sha256'],
              'candidate acceptance pin mismatch')
    # Authority file paths may name current copies; all non-path authority fields stay exact.
    historical_authority = manifest['authority']
    current_authority = dict(authority)
    if 'reference' in authority:
        ref = dict(authority['reference'])
        ref['files'] = {name:dict(record, path=historical_authority['reference']['files'][name]['path'])
                        for name, record in ref['files'].items()}
        current_authority['reference'] = ref
    d.require(current_authority == historical_authority, 'panel authority mismatch')
    records.extend(current)
    _recheck(records)
    sequences = {}
    for chrom, bases in p.fasta_sequences(payload['fasta']):
        for locus in loci:
            if locus.chromosome == chrom:
                start, end = p.w.evaluation_window(locus,len(bases))
                sequences[locus.gene_id] = bases[start:end]
    d.require(set(sequences) == {l.gene_id for l in loci}, 'missing reference sequence')
    # Shared strict artifact checks include all 150 hashes, historical-seed raster equality,
    # exact DB-selected JSONL (including members), FASTA bytes and support summaries.
    tracks, records = v._load_artifacts(root,manifest,loci,sequences,evidence,records)
    proofs = []
    with duckdb.connect(str(evidence), config={'access_mode':'READ_ONLY','threads':1}) as con:
        for locus, row in zip(loci,manifest['loci']):
            start, end = p.w.evaluation_window(locus)
            array = p.mvp_tracks.rasterise_est(con,sequence=sequences[locus.gene_id],
                chrom=locus.chromosome,offset=start,L=30720,is_rc=False,keep_fraction=1,
                aug_seed=20261037,rn=None,population='excluded_heldout',source_supplied=True)
            d.require(array.dtype == np.float16 and np.array_equal(array,tracks[locus.gene_id][0].numpy()),
                      'seed123 panel rasterization mismatch')
            selected = list(p.selected_provenance(con,locus.chromosome,start,end))
            proof = dict(row, selected_member_count=sum(len(r['members']) for r in selected),
                         selected_member_counts=[len(r['members']) for r in selected],
                         seed123_exact_equal=True, measured_aug_seeds=[20260915,20260916,20260917,20261037],
                         array_data_sha256=hashlib.sha256(array.tobytes()).hexdigest())
            d.require(hashlib.sha256(''.join(p.canonical(r) for r in selected).encode()).hexdigest() == row['provenance']['sha256'],
                      'seed123 provenance mismatch')
            proofs.append(proof)
    report = dict(schema_version=1, purpose='immutable_panel_seed123_compatibility_not_training',
        training_eligible=False, scientific_approval=False, profile='development-single',seed=123,aug_seed=20261037,
        keep_fraction=1,rn=None,population='excluded_heldout',panel_root=str(root),
        historical_manifest=records[1],historical_inputs=manifest['inputs'],historical_packages=manifest['packages'],
        authority=authority,current_inputs=current,consumer_closure=closure,
        current_packages={'python':platform.python_version(),'numpy':np.__version__,
                          'duckdb':duckdb.__version__,'torch':str(torch.__version__)},loci=proofs)
    _recheck(records)
    return report, tracks, records, loci, sequences


def verify_and_publish(input_path, out):
    """Read current request and publish ONE external no-replace JSON receipt; CPU only."""
    d.require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
              'reviewed CPU compute allocation and CUDA_VISIBLE_DEVICES= required')
    request_id = p.file_identity(_canonical_path(input_path))
    payload = d.read_json(input_path)
    _profile(payload)
    out = Path(out)
    d.require(out.is_absolute() and '..' not in out.parts, 'canonical absolute output required')
    _canonical_path(out.parent)
    if os.path.lexists(out) or os.path.lexists(str(out)+'.pending'):
        raise FileExistsError(out)
    root = _canonical_path(payload.get('panel_inputs',_ACCEPTED_PANEL_ROOT))
    d.require(not out.is_relative_to(root), 'receipt must be external to immutable panel')
    report, _, records, _, _ = _measure(root,payload)
    report['verification_request'] = request_id
    records.append(request_id)
    pending = producer.stage_json(out,report)
    pending_id = p.file_identity(_canonical_path(pending))
    d.require(d.read_json(pending) == report, 'staged compatibility receipt changed')
    def final_recheck():
        _canonical_path(root)
        _canonical_path(out.parent)
        _recheck(records + [pending_id])
    producer.publish_staged([out],final_recheck)
    return report


def load_compatible_panel(root, payload, panel, sequences):
    """Require independent receipt pin AND repeat live reference/DB/raster verification."""
    _profile(payload)
    spec = payload.get('panel_compatibility')
    d.require(isinstance(spec,dict) and 'path' in spec and 'sha256' in spec,
              'independently pinned panel_compatibility required')
    path = _canonical_path(spec['path'])
    d.require(not path.is_relative_to(_canonical_path(root)), 'receipt must be external to immutable panel')
    receipt_id = d.checked_file(spec)
    receipt = d.read_json(path)
    d.require(receipt.get('consumer_closure') == consumer_closure(), 'stale consumer closure')
    report, tracks, records, loci, actual_sequences = _measure(root,payload)
    request_id = d.checked_file(receipt['verification_request'])
    request = d.read_json(request_id['path'])
    _profile(request)
    d.require(_data_paths(request) == _data_paths(payload) and request['candidate'] == payload['candidate'],
              'compatibility request binding mismatch')
    report['verification_request'] = request_id
    d.require(receipt == report, 'compatibility receipt differs from live proof')
    d.require(loci == panel and sequences == actual_sequences, 'consumer reference slices mismatch')
    records.extend([receipt_id,request_id])
    _recheck(records)
    return tracks, records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args = parser.parse_args(argv)
    verify_and_publish(args.input,args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
