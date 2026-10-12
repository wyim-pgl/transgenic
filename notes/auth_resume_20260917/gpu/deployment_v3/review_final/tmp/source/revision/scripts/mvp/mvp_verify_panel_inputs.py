"""Consume only the COMPLETE schema-1 panel publication; no internal adapter API."""
import hashlib
from pathlib import Path

import duckdb
import numpy as np
import torch

from transgenic.evidence import mvp_decode as d, mvp_windows as w
import mvp_prepare_panel_inputs as p


def load_panel_inputs(root, payload, panel, sequences):
    if payload.get('profile') == 'development-single':
        from mvp_panel_compatibility import load_compatible_panel
        return load_compatible_panel(root, payload, panel, sequences)
    return _load_current_panel(root, payload, panel, sequences)


def _read_panel(root):
    root = Path(root).resolve(strict=True)
    for name in ('COMPLETE', 'manifest.json'):
        d.require(not (root/name).is_symlink(), 'panel publication marker symlink')
    records = [d.identity(root / name) for name in ('COMPLETE', 'manifest.json')]
    d.require((root / 'COMPLETE').read_text() == records[1]['sha256'] + '\n', 'COMPLETE manifest mismatch')
    manifest = d.read_json(root / 'manifest.json')
    policy = dict(schema_version=1, purpose='est_mvp_panel_inputs_not_training',
        training_eligible=False, orientation='genomic_forward', population='excluded_heldout',
        keep_fraction=1, source='EST', seeds=[1,2,3], aug_seeds=[20260915,20260916,20260917],
        seed_tracks_identical=True, off_policy='zeros_like(on); same sequence/window',
        zero_support_policy='report_not_fail')
    for key, value in policy.items():
        d.require(type(manifest[key]) is type(value) and manifest[key] == value, 'panel policy mismatch: ' + key)
    return root, manifest, records


def _load_current_panel(root, payload, panel, sequences):
    root, manifest, records = _read_panel(root)
    loci, authority, expected, evidence = p.preflight(payload)
    d.require(loci == panel and manifest['authority'] == authority, 'panel authority mismatch')
    # The producer adds exactly its request identity to its independently obtained inventory.
    d.require(len(manifest['inputs']) == len(expected) + 1 and manifest['inputs'][:-1] == expected,
              'panel input/code inventory mismatch')
    records.extend(d.checked_file(r) for r in manifest['inputs'])
    request = d.read_json(manifest['inputs'][-1]['path'])
    d.require(all(request[k] == payload[k] for k in ('candidate','panels','fasta','reference')),
              'panel request binding mismatch')
    d.require(manifest['rasterizer'] == d.identity(p.mvp_tracks.__file__), 'rasterizer mismatch')
    d.require(manifest['packages'] == {'numpy': np.__version__, 'duckdb': duckdb.__version__}, 'panel package mismatch')
    return _load_artifacts(root, manifest, panel, sequences, evidence, records)


def _load_artifacts(root, manifest, panel, sequences, evidence, records):
    """Shared byte/numerical checks, after the caller has bound producer identity."""
    rows = manifest['loci']
    d.require(len(rows) == len(panel) == 50 and [r['gene_id'] for r in rows] == [l.gene_id for l in panel],
              'exact ordered 50-locus inventory required')
    paths = set()

    def artifact(record):
        relative = Path(record['path'])
        path = root / relative
        d.require(not relative.is_absolute() and '..' not in relative.parts and
                  path.resolve(strict=True).is_relative_to(root), 'panel artifact path escape')
        d.require(path.resolve() == path.absolute(), 'panel artifact symlink is not permitted')
        d.require(str(relative) not in paths, 'duplicate panel artifact')
        paths.add(str(relative))
        actual = d.checked_file(dict(path=str(path), sha256=record['sha256'], size=record['bytes']))
        records.append(actual)
        return path

    result = {}
    with duckdb.connect(str(evidence), config={'access_mode':'READ_ONLY', 'threads':1}) as con:
        for locus, row in zip(panel, rows):
            start, end = w.evaluation_window(locus)
            sequence = sequences[locus.gene_id]
            d.require(row['window'] == [locus.chromosome,start,end] and
                      all(type(x) is int for x in row['window'][1:]), 'panel window mismatch')
            d.require(len(sequence) == 30720 and row['sequence_sha256'] == hashlib.sha256(sequence.encode('ascii')).hexdigest(), 'panel sequence digest mismatch')
            fa = artifact(row['sequence'])
            d.require(fa.read_text() == f'>{locus.gene_id} {locus.chromosome}:{start}-{end} genomic_forward\n{sequence}\n', 'panel sequence bytes mismatch')
            track = row['track']
            d.require(track['dtype'] == 'float16' and track['shape'] == [30720,8], 'panel track metadata mismatch')
            array = np.load(artifact(track), allow_pickle=False)
            d.require(array.dtype == np.float16 and array.shape == (30720,8), 'panel track shape mismatch')
            d.require(np.isfinite(array).all() and (array >= 0).all(), 'invalid tracks')
            masked = np.frombuffer(sequence.upper().encode('ascii'), dtype='S1') == b'N'
            d.require(not array[masked].any() and (array[~masked,7] == 1).all(), 'N mask/availability mismatch')
            d.require((array[:,:3] <= np.float16(np.log1p(64))).all(), 'coverage cap exceeded')
            # Independently reproduce actual excluded-heldout selection and all three seeds.
            d.require(np.array_equal(array, p.panel_tracks(con, sequence, locus.chromosome, start)), 'panel rasterization mismatch')
            provenance = list(p.selected_provenance(con, locus.chromosome, start, end))
            d.require(artifact(row['provenance']).read_text() == ''.join(p.canonical(r) for r in provenance), 'panel provenance mismatch')
            biological = np.count_nonzero(array[:,:7], axis=0).tolist()
            d.require(row['selected_molecule_count'] == len(provenance) and
                      row['biological_nonzero_per_channel'] == biological and
                      row['availability_nonzero'] == int(np.count_nonzero(array[:,7])) and
                      row['zero_biological_support'] is (not any(biological)), 'panel support summary mismatch')
            result[locus.gene_id] = torch.from_numpy(array.copy()).unsqueeze(0)
    p.unchanged(records)
    return result, records
