#!/usr/bin/env python3
"""Publish frozen-panel EST inputs, never training data. See MVP_PANEL_INPUT_CONTRACT.md."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from transgenic.evidence import mvp_tracks, mvp_windows as w
from mvp_produce_counts import file_identity, frozen_reference, read_json, require


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'


def unchanged(identities):
    for identity in identities:
        require(file_identity(identity['path']) == identity,
                'input/output mutated: ' + identity['path'])
        if identity['path'].endswith(('.db', '.duckdb')):
            require(not Path(identity['path'] + '.wal').exists(), 'DB WAL appeared')


def preflight(payload):
    from transgenic.training import mvp_cli, mvp_cli_runtime
    c = payload['candidate']
    paths = [payload[k] for k in ('panels', 'fasta', 'reference')]
    paths += [c[k] for k in ('acceptance', 'inventory', 'receipts', 'corpus', 'evidence')]
    paths += [__file__, mvp_tracks.__file__, w.__file__, mvp_cli.__file__, mvp_cli_runtime.__file__]
    import mvp_produce_counts
    paths.append(mvp_produce_counts.__file__)
    identities = [file_identity(path) for path in paths]
    loci = w.load_panel(payload['panels'])
    require(len(loci) == 50, 'exact frozen 50-locus panel required')
    authority = frozen_reference(payload['fasta'], payload['reference'])
    candidate = mvp_cli.validate_candidate_records(c['acceptance'], c['acceptance_sha256'], c['inventory'], c['receipts'])
    verified = mvp_cli_runtime.verify_candidate_artifacts(candidate, c['corpus'], c['evidence'], payload['panels'])
    unchanged(identities)
    return loci, {'reference': authority, 'accepted_data_sha256': verified['sha256']}, identities, Path(c['evidence']).resolve()


def panel_tracks(con, sequence, chrom, start):
    arrays = [mvp_tracks.rasterise_est(
        con, sequence=sequence, chrom=chrom, offset=start, L=len(sequence),
        is_rc=False, keep_fraction=1, aug_seed=20260914 + seed, rn=None,
        population='excluded_heldout', source_supplied=True) for seed in (1, 2, 3)]
    require(all(np.array_equal(arrays[0], a) for a in arrays[1:]), 'full-retention seed independence failed')
    return arrays[0]


def selected_provenance(con, chrom, start, end):
    """Same independent block/endpoint predicates as rasterise_est, no RN join."""
    def records(sql, params):
        cursor = con.execute(sql, params)
        names = [col[0] for col in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]
    policy = "m.excluded AND m.cause='heldout_locus' AND m.primary_count>0"
    blocks = records(f'''SELECT b.* FROM est_block b JOIN est_molecule m USING(molecule_id)
        WHERE b.chrom=? AND b.start0<? AND b.end0>? AND {policy}
        ORDER BY b.molecule_id,b.start0,b.end0''', [chrom, end, start])
    junctions = records(f'''SELECT j.* FROM est_junction j JOIN est_molecule m USING(molecule_id)
        WHERE j.chrom=? AND ((j.donor0>=? AND j.donor0<?) OR (j.acceptor0>=? AND j.acceptor0<?))
        AND {policy} ORDER BY j.molecule_id,j.donor0,j.acceptor0''', [chrom, start, end, start, end])
    grouped = {}
    for kind, rows in [('blocks', blocks), ('junctions', junctions)]:
        for row in rows:
            grouped.setdefault(row['molecule_id'], {'blocks': [], 'junctions': []})[kind].append(row)
    # ponytail: two metadata lookups per selected molecule; batch if real-panel profiling requires it.
    for mid, geometry in sorted(grouped.items()):
        molecule = records('SELECT * FROM est_molecule WHERE molecule_id=?', [mid])
        require(len(molecule) == 1, 'ambiguous molecule identity')
        members = records('SELECT * FROM est_member WHERE molecule_id=? ORDER BY member', [mid])
        require(members, 'selected molecule lacks member provenance')
        yield dict(molecule=molecule[0], members=members, **geometry)


def fasta_sequences(path):
    """Stream contigs without trusting or creating a FASTA index."""
    seen, chrom, parts = set(), None, []
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith('>'):
                if chrom is not None:
                    yield chrom, ''.join(parts)
                chrom = line[1:].split()[0]
                require(chrom not in seen, 'duplicate FASTA contig')
                seen.add(chrom)
                parts = []
            elif line.strip():
                require(chrom is not None, 'FASTA sequence before header')
                parts.append(line.strip())
        if chrom is not None:
            yield chrom, ''.join(parts)


def prepare(input_path, out):
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    request_identity = file_identity(input_path)
    payload = read_json(input_path)
    loci, authority, identities, evidence = preflight(payload)
    identities.append(request_identity)
    require(not Path(str(evidence) + '.wal').exists(), 'evidence DB has WAL')
    unchanged(identities)
    out.mkdir(parents=True, exist_ok=False)
    outputs, rows = [], {}

    def artifact(path):
        identity = file_identity(path)
        outputs.append(identity)
        return {'path': str(path.relative_to(out)), 'sha256': identity['sha256'], 'bytes': identity['size']}

    with duckdb.connect(str(evidence), config={'access_mode': 'READ_ONLY', 'threads': 1}) as con:
        for chrom, bases in fasta_sequences(payload['fasta']):
            for index, locus in enumerate(loci):
                if locus.chromosome != chrom:
                    continue
                start, end = w.evaluation_window(locus, len(bases))
                sequence = bases[start:end]
                tracks = panel_tracks(con, sequence, chrom, start)
                require(tracks.dtype == np.float16 and tracks.shape == (30720, 8), 'invalid track tensor')
                require(np.isfinite(tracks).all(), 'nonfinite tracks')
                prefix = out / f'{index:02d}'  # no gene identifiers in filesystem paths
                track_path = prefix.with_suffix('.npy')
                with track_path.open('xb') as handle:
                    np.save(handle, tracks, allow_pickle=False)
                track = dict(artifact(track_path), dtype='float16', shape=[30720, 8])
                sequence_path = prefix.with_suffix('.fa')
                with sequence_path.open('x') as handle:
                    handle.write(f'>{locus.gene_id} {chrom}:{start}-{end} genomic_forward\n{sequence}\n')
                provenance_path = prefix.with_suffix('.jsonl')
                count = 0
                with provenance_path.open('x') as handle:
                    for record in selected_provenance(con, chrom, start, end):
                        handle.write(canonical(record))
                        count += 1
                biological = np.count_nonzero(tracks[:, :7], axis=0).tolist()
                rows[index] = dict(gene_id=locus.gene_id, window=[chrom, start, end],
                    sequence_sha256=hashlib.sha256(sequence.encode('ascii')).hexdigest(),
                    sequence=artifact(sequence_path), track=track, provenance=artifact(provenance_path),
                    selected_molecule_count=count, biological_nonzero_per_channel=biological,
                    availability_nonzero=int(np.count_nonzero(tracks[:, 7])),
                    zero_biological_support=not any(biological))
    require(len(rows) == len(loci), 'panel chromosome absent from FASTA')
    manifest = dict(schema_version=1, purpose='est_mvp_panel_inputs_not_training',
        training_eligible=False, orientation='genomic_forward', population='excluded_heldout',
        keep_fraction=1, source='EST', seeds=[1, 2, 3], aug_seeds=[20260915, 20260916, 20260917],
        seed_tracks_identical=True, off_policy='zeros_like(on); same sequence/window',
        zero_support_policy='report_not_fail', inputs=identities, authority=authority,
        rasterizer=file_identity(mvp_tracks.__file__),
        packages={'numpy': np.__version__, 'duckdb': duckdb.__version__},
        loci=[rows[i] for i in range(len(loci))])
    unchanged(identities + outputs)
    require(not Path(str(evidence) + '.wal').exists(), 'evidence DB WAL appeared')
    manifest_path = out / 'manifest.json'
    with manifest_path.open('x') as handle:
        handle.write(canonical(manifest))
    published = file_identity(manifest_path)
    unchanged(identities + outputs + [published])
    require(not Path(str(evidence) + '.wal').exists(), 'evidence DB WAL appeared')
    with (out / 'COMPLETE').open('x') as handle:
        handle.write(published['sha256'] + '\n')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    prepare(args.input, args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
