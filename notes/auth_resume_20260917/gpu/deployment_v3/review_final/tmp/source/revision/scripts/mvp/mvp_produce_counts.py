#!/usr/bin/env python3
"""Local raw-GFF reference scoring; receipts are consistency evidence, not authorization."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from transgenic.evidence import mvp_windows as w
from transgenic.evidence.mvp_bootstrap import validate_paired_record

AUTHORITY_COMMIT = '92d8020ee79eeedd7f1d50167b1c8dc94700fb16'
AUTHORITY_PATH = 'data/manifests/b5_species_v1.tsv'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    # Duplicate keys must not silently replace a seed or a file identity.
    def unique(items):
        result = {}
        for k, v in items:
            require(k not in result, f'duplicate JSON key: {k}')
            result[k] = v
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def file_identity(path):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), f'not a file: {path}')
    return {'path': str(path), 'sha256': w.sha256_file(path), 'size': path.stat().st_size}


def check_file(path, expected):
    actual = file_identity(path)
    require(actual['sha256'] == expected['sha256'], f'file hash mismatch: {path}')
    if 'size' in expected:
        require(type(expected['size']) is int and actual['size'] == expected['size'], f'file size mismatch: {path}')
    return actual


def frozen_reference(fasta, reference):
    """Read authority from an existing immutable git object, never the working copy."""
    proc = subprocess.run(['git', '-C', str(ROOT), 'show', f'{AUTHORITY_COMMIT}:{AUTHORITY_PATH}'],
                          check=True, capture_output=True)
    rows = list(csv.DictReader(io.StringIO(proc.stdout.decode()), delimiter='\t'))
    rows = [r for r in rows if r['species_id'] == 'Athaliana']
    require(len(rows) == 1, 'missing/ambiguous frozen Athaliana manifest row')
    bound = {}
    for name, path in [('fasta', fasta), ('gff', reference)]:
        digest = hashlib.md5()
        with Path(path).open('rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                digest.update(block)
        require(digest.hexdigest() == rows[0][name + '_md5'], f'frozen {name} identity mismatch')
        bound[name] = file_identity(path)
    return {'commit': AUTHORITY_COMMIT, 'manifest': AUTHORITY_PATH,
            'manifest_sha256': hashlib.sha256(proc.stdout).hexdigest(), 'files': bound}


def verify_training_code(source, metadata):
    """Apply the existing training closure schema at its bound checkout, not ours."""
    require(isinstance(source, dict) and set(source) == {'files', 'external_packages'}, 'invalid consumed code manifest')
    files, packages = source['files'], source['external_packages']
    require(isinstance(files, dict) and files, 'consumed code files must be a nonempty object')
    required_packages = {'accelerate', 'duckdb', 'safetensors', 'torch', 'transformers'}
    require(isinstance(packages, dict) and set(packages) == required_packages and
            all(isinstance(v, str) and v.strip() for v in packages.values()), 'complete five-package code schema required')
    # duckdb is mandatory in the training code manifest but absent from the
    # existing finalizer runtime_packages schema. No scoring-runtime comparison.
    for name in required_packages - {'duckdb'}:
        require(packages[name] == metadata['runtime_packages'][name], 'code/runtime package mismatch')
    implementation = metadata['initialization_audit']['audit']['implementation']
    runtime_path = Path(implementation['b5_runtime.py']['path'])
    require(runtime_path.is_absolute() and runtime_path.parts[-4:] ==
            ('src', 'transgenic', 'training', 'b5_runtime.py'), 'invalid bound training checkout path')
    root = runtime_path.parents[3].resolve(strict=True)
    # Same closure as mvp_cli_runtime._expected_code_files, evaluated in the
    # historical training checkout. Additional scoring-checkout files are irrelevant.
    expected = {str(path) for path in (root/'src/transgenic').rglob('*.py')}
    expected.add(str(root/'train/train_HyenaTransgenic_MVP.py'))
    require(set(files) == expected, 'consumed code manifest must match exact bound training source closure')
    require(all(str(Path(path).resolve(strict=True)) == path for path in files), 'noncanonical consumed code path')
    identities = [check_file(path, {'sha256': digest}) for path, digest in files.items()]
    for name, relative in [('b5_runtime.py', 'src/transgenic/training/b5_runtime.py'),
                           ('configuration_transgenic.py', 'src/transgenic/model/configuration_transgenic.py')]:
        path = str(root/relative)
        require(implementation[name] == {'path': path, 'sha256': files.get(path)}, 'initialization implementation/code binding mismatch')
    # Reconstruct semantics only with the same project-owned implementation.
    # Do not import/execute arbitrary files from a supplied historical checkout.
    from transgenic.training import weights_init as wi
    for module, relative in [(wi, 'src/transgenic/training/weights_init.py'),
                             (wi.b5_runtime, 'src/transgenic/training/b5_runtime.py'),
                             (wi.configuration_transgenic, 'src/transgenic/model/configuration_transgenic.py')]:
        require(files.get(str(root/relative)) == w.sha256_file(module.__file__),
                'unsupported historical semantic implementation; reviewed verifier adapter required')
    return identities


def verify_initialization(audit, initialization, final_config):
    """Reconstruct directly checkable loader claims without loading a full model."""
    from transgenic.training import weights_init as wi
    from safetensors import safe_open, SafetensorError
    checkpoint = Path(audit['resolved_checkpoint_dir'])
    require(initialization['model.safetensors']['path'] == str(checkpoint/'model.safetensors') and
            initialization['run_config.json']['path'] == str(checkpoint.parent/'run_config.json'),
            'source checkpoint/run config canonical paths mismatch')
    for name in ('config.json', 'meta.json', 'runtime.json', 'runtime_identity.txt'):
        if name in initialization:
            parent = checkpoint.parent if name == 'runtime_identity.txt' else checkpoint
            require(initialization[name]['path'] == str(parent/name), 'source optional file canonical path mismatch')
    run_config = read_json(initialization['run_config.json']['path'])
    require(isinstance(run_config, dict) and type(run_config.get('seed')) is int and
            run_config['seed'] == 123, 'baseline source is not seed 123')
    recipe = run_config.get('recipe')
    require(isinstance(recipe, dict) and wi._REQUIRED_RECIPE_KEYS <= set(recipe), 'embedded source recipe missing required keys')
    require(audit['recipe'] == {'canonical_sha256': wi._sha256_bytes(wi._canonical(recipe))}, 'source recipe canonical digest mismatch')
    require(type(recipe.get('evidence_channels', 0)) is int and recipe.get('evidence_channels', 0) == 0,
            'source recipe must have zero evidence channels')
    source_config = wi.HyenaTransgenicConfig(**wi.model_kwargs(recipe))
    source_fields = wi._manifest(source_config, recipe)
    for name in wi._SEMANTIC_FIELDS:
        if name in recipe and name != 'evidence_channels' and hasattr(source_config, name):
            require(wi._canonical(wi._value(recipe[name])) == wi._canonical(wi._value(getattr(source_config, name))),
                    f'source recipe semantic contradiction: {name}')
    require(wi._manifest_record(source_fields) == audit['source_semantics'], 'source semantic manifest mismatch')
    if 'config.json' in initialization:
        recorded = read_json(initialization['config.json']['path'])
        require(isinstance(recorded, dict), 'source config must be a JSON object')
        for name, entry in source_fields.items():
            if name in recorded:
                require(wi._canonical(recorded[name]) == wi._canonical(entry['value']), f'source config semantic mismatch: {name}')
    require(isinstance(final_config, dict), 'final config must be a JSON object')
    require(set(audit['target_semantics']['fields']) <= set(final_config), 'final config missing audited semantic fields')
    target_fields = wi._manifest(wi.HyenaTransgenicConfig(**final_config), None)
    require(wi._manifest_record(target_fields) == audit['target_semantics'], 'final config target semantic manifest mismatch')
    for name, entry in target_fields.items():
        require(wi._canonical(final_config[name]) == wi._canonical(entry['value']), f'final config semantic normalization: {name}')
    wi._check_equal(source_fields, target_fields)
    try:
        with safe_open(initialization['model.safetensors']['path'], framework='pt', device='cpu') as model:
            require(bool(list(model.keys())) and wi._PROJECTION_KEY not in model.keys(), 'invalid zero-channel source weights inventory')
            require(model.metadata() == audit['load']['safetensors_metadata'], 'source safetensors metadata mismatch')
    except SafetensorError as error:
        raise ValueError('source weights are not readable safetensors') from error


def verify_final(final, seed, expected_rns, data_hashes, code_manifest, *, profile='historical-three-seed'):
    """Consume exactly mvp_finalize's existing publication format."""
    from transgenic.training.mvp_finalize import _validate_metadata
    from transgenic.training.mvp_metrics import aggregate_paired_records
    from transgenic.training.mvp_profiles import LEGACY_PROFILE, DEVELOPMENT_PROFILE, validate_seed_profile
    validate_seed_profile(profile, seed)
    final = Path(final).resolve(strict=True)
    require((final / 'TRAINING_DONE').read_bytes() == b'complete\n', 'final is not complete')
    manifest = read_json(final / 'manifest.json')
    names = {'metadata.json', 'paired_records.jsonl', 'paired_summary.json', 'model.safetensors', 'config.json'}
    require(set(manifest) == {'files'} and set(manifest['files']) == names, 'final manifest inventory mismatch')
    files = {name: check_file(final / name, manifest['files'][name]) for name in names}
    metadata = read_json(final / 'metadata.json')
    _validate_metadata(metadata, metadata.get('training_summary'))
    require(metadata.get('profile', LEGACY_PROFILE) == profile, 'final metadata profile mismatch')
    if profile == DEVELOPMENT_PROFILE:
        require(metadata.get('run_role') == 'mvp_finetune', 'final metadata run role mismatch')
    elif 'run_role' in metadata:
        require(metadata['run_role'] == 'mvp_finetune', 'final metadata run role mismatch')
    require(metadata['identity_scope'] == 'production', 'test final cannot establish MVP run identity')
    require(metadata['seed_augmentation'] == {'seed': seed, 'aug_seed': 20260914 + seed}, 'cross-seed final mismatch')
    for name, digest in data_hashes.items():
        require(metadata['data_identities'][name]['sha256'] == digest, f'final {name} identity mismatch')
    code = check_file(code_manifest, metadata['sif_code_identity']['code'])
    source = read_json(code_manifest)
    code_files = verify_training_code(source, metadata)
    audit_files = metadata['initialization_audit']['audit']['files']
    initialization = {name: check_file(record['path'], record['before'])
                      for name, record in audit_files.items() if record is not None}
    verify_initialization(metadata['initialization_audit']['audit'], initialization, read_json(final/'config.json'))
    records = [json.loads(line) for line in (final / 'paired_records.jsonl').read_text().splitlines()]
    for record in records:
        validate_paired_record(record)
    require(len(records) == len({r['rn'] for r in records}), 'final contains duplicate RN records')
    summary = aggregate_paired_records(records, expected_rns)
    require(read_json(final / 'paired_summary.json') == {k: v for k, v in summary.items() if k != 'records'},
            'paired summary mismatch')
    from safetensors import safe_open
    with safe_open(str(final / 'model.safetensors'), framework='pt') as model:
        require(bool(list(model.keys())), 'empty final checkpoint')
    return {'final': str(final), 'manifest': file_identity(final / 'manifest.json'),
            'done': file_identity(final / 'TRAINING_DONE'), 'files': files,
            'seed': seed, 'code_manifest': code, 'code_files': code_files,
            'initialization': initialization, 'valid_rns': list(expected_rns)}


def evaluation_profile(payload, requested=None):
    from transgenic.training.mvp_profiles import LEGACY_PROFILE, DEVELOPMENT_PROFILE, validate_seed_profile
    profile = payload.get('profile', LEGACY_PROFILE)
    validate_seed_profile(profile, 123 if profile == DEVELOPMENT_PROFILE else 1)
    require(requested is None or requested == profile, 'CLI/payload profile mismatch')
    metadata = payload.get('metadata', {})
    require(isinstance(metadata, dict) and metadata.get('profile', profile) == profile,
            'metadata profile mismatch')
    return profile


def verify_run_inventory(payload, artifacts):
    """One development final plus an independently pinned, unaugmented B5 source."""
    from transgenic.training.mvp_profiles import DEVELOPMENT_PROFILE, validate_seed_profile
    profile = evaluation_profile(payload)
    if profile != DEVELOPMENT_PROFILE:
        require(set(payload['runs']) == {'1', '2', '3'}, 'exact final seeds 1/2/3 required')
        runs = {s: verify_final(r['final'], int(s), artifacts['valid_rns'], artifacts['sha256'], r['code_manifest'])
                for s, r in payload['runs'].items()}
        require(len({r['code_manifest']['sha256'] for r in runs.values()}) == 1, 'cross-seed consumed code mismatch')
        sources = {tuple(sorted((name, value['sha256']) for name, value in r['initialization'].items()))
                   for r in runs.values()}
        require(len(sources) == 1, 'cross-seed initialization source inventory mismatch')
        return runs
    roles = {'mvp_seed123': 'mvp_finetune', 'b5_seed123': 'b5_baseline'}
    require(set(payload['runs']) == set(roles), 'exact development run identities required')
    for run_id, role in roles.items():
        spec = payload['runs'][run_id]
        validate_seed_profile(profile, spec.get('seed'))
        require(spec.get('run_role') == role and spec.get('profile', profile) == profile,
                'run profile/role mismatch')
        require(spec.get('run_id', run_id) == run_id, 'run identity mismatch')
        md = spec.get('metadata', {})
        require(isinstance(md, dict) and md.get('profile', profile) == profile and
                md.get('run_role', role) == role, 'run metadata profile/role mismatch')
        if 'seed' in md:
            validate_seed_profile(profile, md['seed'])
        for key in ('run_id', 'checkpoint_sha256'):
            require(key not in md or md[key] == spec.get(key, run_id if key == 'run_id' else None),
                    'run metadata identity mismatch')
    spec = payload['runs']['mvp_seed123']
    final = verify_final(spec['final'], spec['seed'], artifacts['valid_rns'], artifacts['sha256'],
                         spec['code_manifest'], profile=profile)
    require(spec['checkpoint_sha256'] == final['files']['model.safetensors']['sha256'],
            'fine-tuned checkpoint pin mismatch')
    base = payload['runs']['b5_seed123']
    require(set(base) <= {'run_id', 'run_role', 'seed', 'profile', 'metadata', 'checkpoint_sha256', 'weights'},
            'baseline must not carry MVP augmentation/final fields')
    require(not ({'seed_augmentation', 'aug_seed', 'final'} & set(base.get('metadata', {}))),
            'baseline metadata must not carry MVP augmentation')
    pinned = check_file(base['weights']['path'], base['weights'])
    source = final['initialization']['model.safetensors']
    require(pinned['sha256'] == source['sha256'] == base['checkpoint_sha256'],
            'baseline pinned weights disagree with final initialization audit')
    require(pinned['sha256'] != final['files']['model.safetensors']['sha256'],
            'baseline and fine-tuned checkpoints must be distinct')
    final.update(profile=profile, run_id='mvp_seed123', run_role='mvp_finetune')
    return {'mvp_seed123': final, 'b5_seed123': dict(profile=profile, run_id='b5_seed123',
            run_role='b5_baseline', seed=123, weights=pinned)}


def verify_evaluation_readiness(payload, candidate, runs):
    """Re-use training's trusted-readiness contract; never infer it from final files."""
    from transgenic.training.mvp_profiles import DEVELOPMENT_PROFILE
    from transgenic.training.mvp_cli_runtime import validate_launch_readiness
    from transgenic.training.mvp_runtime import MVPRecipe
    profile = evaluation_profile(payload)
    if profile != DEVELOPMENT_PROFILE:
        return []
    spec = payload['runs']['mvp_seed123']
    require(all(isinstance(spec.get(key), dict) and {'path', 'sha256'} <= set(spec[key])
                for key in ('launch_readiness', 'sif', 'config')), 'pinned launch readiness/SIF/config required')
    records = [check_file(spec[key]['path'], spec[key]) for key in ('launch_readiness', 'sif', 'config')]
    run = runs['mvp_seed123']
    metadata = read_json(run['files']['metadata.json']['path'])
    require(records[1]['sha256'] == metadata['sif_code_identity']['sif']['sha256'], 'readiness/final SIF mismatch')
    c = payload['candidate']
    paths = (c['corpus'], c['evidence'], payload['panels'], records[1]['path'], records[2]['path'],
             run['initialization']['model.safetensors']['path'], run['initialization']['run_config.json']['path'])
    readiness = validate_launch_readiness(records[0]['path'], records[0]['sha256'], candidate, paths,
                                          seed=123, recipe=MVPRecipe(), profile=profile)
    manifest = readiness['bindings']['consumed_code_manifest']
    require(check_file(manifest['path'], manifest) == run['code_manifest'], 'readiness/final code manifest mismatch')
    return records


def legacy_module(name):
    path = ROOT / 'revision/scripts' / name
    spec = importlib.util.spec_from_file_location('mvp_' + path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fasta_lengths(path):
    lengths = {}
    chrom = None
    with Path(path).open() as f:
        for line in f:
            if line.startswith('>'):
                chrom = line[1:].split()[0]
                require(chrom not in lengths, 'duplicate FASTA contig')
                lengths[chrom] = 0
            elif line.strip():
                require(chrom is not None, 'FASTA sequence before header')
                lengths[chrom] += len(line.strip())
    return lengths


def read_gff(path):
    """Strict preflight around the existing half-open GFF parser; never repair data."""
    genes, transcripts, features = {}, {}, []
    gff3_header = False
    with Path(path).open() as f:
        for line in f:
            if line.rstrip('\r\n') == '##gff-version 3':
                gff3_header = True
            if not line.strip() or line.startswith('#'):
                continue
            cols = line.rstrip('\n').split('\t')
            require(len(cols) == 9, 'malformed GFF: expected nine columns')
            chrom, _, kind, start, end, _, strand, phase, attrs = cols
            require(int(start) >= 1 and int(end) >= int(start) and strand in ('+', '-'), 'invalid GFF coordinates/strand')
            fields = [a.split('=', 1) for a in attrs.split(';') if a]
            require(all(len(a) == 2 for a in fields) and len(dict(fields)) == len(fields), 'malformed GFF attributes')
            attrs = dict(fields)
            require(all(key == key.strip() and not any(c.isspace() for c in key) for key in attrs), 'noncanonical GFF attribute key')
            for key in ('ID', 'Parent'):
                if key in attrs:
                    identifiers = attrs[key].split(',') if key == 'Parent' else [attrs[key]]
                    require(all(re.fullmatch(r'[A-Za-z0-9_.:-]+', value) for value in identifiers) and
                            len(identifiers) == len(set(identifiers)), 'noncanonical GFF ID/Parent')
            span = (chrom, int(start) - 1, int(end), strand)
            if kind == 'gene':
                gid = attrs.get('ID')
                require(gid and gid not in genes, 'missing/duplicate GFF gene ID')
                genes[gid] = span
            elif kind in ('mRNA', 'transcript'):
                tid, parent = attrs.get('ID'), attrs.get('Parent')
                require(tid and tid not in transcripts and parent in genes, 'missing/duplicate transcript or unordered/orphan gene')
                transcripts[tid] = (parent, span)
            elif kind in ('exon', 'CDS', 'five_prime_UTR', 'three_prime_UTR'):
                require(kind != 'CDS' or phase in ('0', '1', '2'), 'invalid CDS phase')
                for parent in attrs.get('Parent', '').split(','):
                    require(parent in transcripts, 'orphan/unordered GFF feature')
                    features.append((parent, span))
            else:
                raise ValueError(f'unsupported GFF feature {kind}; no silent omission')
    require(genes or gff3_header, 'completed empty prediction requires explicit ##gff-version 3 header')
    def contained(child, parent):
        return child[0] == parent[0] and child[3] == parent[3] and parent[1] <= child[1] < child[2] <= parent[2]
    for parent, span in transcripts.values():
        require(contained(span, genes[parent]), 'transcript outside parent gene')
    for parent, span in features:
        require(contained(span, transcripts[parent][1]), 'feature outside transcript')
    parsed = legacy_module('07_selfconsistency_stats.py').load_gff(str(path))
    require(set(parsed) == set(genes), 'GFF parser gene inventory changed')
    for gid, gene in parsed.items():
        require(set(gene['transcripts']) == {tid for tid, (parent, _) in transcripts.items() if parent == gid},
                'GFF parser transcript inventory changed')
    for gene in parsed.values():
        require(gene['transcripts'], 'GFF gene has no transcripts; only a header-only file represents zero predictions')
        for tx in gene['transcripts'].values():
            require(tx['exons'], 'explicit exon geometry required; CDS is not an exon substitute')
            for key in ('exons', 'cdss'):
                tx[key].sort()
                require(all(a[1] < b[0] for a, b in zip(tx[key], tx[key][1:])), 'duplicate/overlapping/adjacent GFF segments')
            require(all(any(a <= c < d <= b for a, b in tx['exons']) for c, d in tx['cdss']), 'CDS outside exons')
    return parsed


def resolve_reference_mapping(panel, corpus, authority, accepted_hashes, panel_path):
    """After candidate gates, bind all 50 builder keys to the corpus's original IDs.

    No aliases, ordinal reconstruction, coordinate lookup or label-table queries.
    The caller must recheck this receipt's sources immediately before publication.
    """
    import duckdb
    keys = [l.gene_id for l in panel]
    require(len(keys) == len(set(keys)) == 50, 'exact unique 50 panel keys required')
    sources = [check_file(corpus, {'sha256': accepted_hashes['corpus']}),
               check_file(panel_path, {'sha256': accepted_hashes['panel']})]
    sources.extend(check_file(r['path'], r) for r in authority['files'].values())
    sources.extend(file_identity(path) for path in
                   (Path(__file__), Path(w.__file__), ROOT/'revision/scripts/24_restrict_reference_to_test_genes.py',
                    ROOT/'revision/scripts/07_selfconsistency_stats.py'))
    corpus = sources[0]['path']
    require(not Path(corpus + '.wal').exists(), 'corpus WAL is not a frozen snapshot')
    with duckdb.connect(corpus, read_only=True, config={'threads': 1}) as con:
        manifest = con.execute('SELECT species_id,fasta_sha256,gff_sha256 FROM build_manifest WHERE species_id=?',
                               ['Athaliana']).fetchall()
        rows = con.execute('SELECT species_id,gene_id,gene_id_original,name_original,chromosome,start0,end0 '
                           'FROM gene_key_map WHERE species_id=? AND gene_id IN (' + ','.join('?' for _ in keys) + ')',
                           ['Athaliana', *keys]).fetchall()
    require(manifest == [('Athaliana', authority['files']['fasta']['sha256'], authority['files']['gff']['sha256'])],
            'corpus build_manifest/reference full SHA256 mismatch or duplicate species')
    require(len(rows) == 50 and len({r[1] for r in rows}) == 50 and {r[1] for r in rows} == set(keys),
            'missing/duplicate/wrong-species reference mapping')
    require(all(r[0] == 'Athaliana' and isinstance(r[2], str) and re.fullmatch(r'[A-Za-z0-9_.:-]+', r[2]) for r in rows),
            'null/empty/invalid original reference ID or wrong species')
    require(len({r[2] for r in rows}) == 50, 'many-to-one original reference mapping')
    by_key = {r[1]: r for r in rows}
    ordered = []
    for locus in panel:
        row = by_key[locus.gene_id]
        require(row[4:] == (locus.chromosome, locus.gene_start0, locus.gene_end0), 'corpus mapping/panel geometry mismatch')
        ordered.append(dict(zip(('species_id','gene_id','gene_id_original','name_original','chromosome','start0','end0'), row),
                            strand=locus.strand))
    receipt = dict(method='corpus_gene_key_map', rows=ordered, sources=sources, authority=authority)
    recheck_reference_mapping(receipt)
    return receipt


def recheck_files(records):
    # Catch a source changing even after its hash was read, during another hash.
    def stamps():
        stats = [Path(r['path']).stat() for r in records]
        return [(s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns) for s in stats]
    before = stamps()
    for record in records:
        check_file(record['path'], record)
    require(before == stamps(), 'source changed during hashing')


def recheck_reference_mapping(receipt):
    wal = receipt['sources'][0]['path'] + '.wal'
    require(not os.path.lexists(wal), 'corpus WAL appeared')
    recheck_files(receipt['sources'])
    require(not os.path.lexists(wal), 'corpus WAL appeared')


def stage_json(path, data):
    """Closed diagnostic staging file; never a consumable success receipt."""
    staging = Path(str(path) + '.pending')
    with staging.open('x') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    return staging


def publish_staged(paths, recheck):
    """Validate after serialization, then atomically link each fresh destination.

    Failed staging files are retained. Hard links provide stdlib no-replace
    publication on the same filesystem, without importing the decoder stack.
    """
    for path in paths:
        if os.path.lexists(path):
            raise FileExistsError(path)
        read_json(str(path) + '.pending')
    recheck()
    for path in paths:
        staging = Path(str(path) + '.pending')
        os.link(staging, path)
        staging.unlink()


def prepare_mapped_reference(panel, reference, restricted, mapping):
    """Shared strict preparation; subsets are only for CPU failure diagnostics."""
    rows = {r['gene_id']: r for r in mapping['rows']}
    originals = {rows[l.gene_id]['gene_id_original'] for l in panel}
    require(panel and len(originals) == len(panel), 'empty/duplicate reference selection')
    restrictor = legacy_module('24_restrict_reference_to_test_genes.py')
    # Tighten only the local legacy module, never alter raw GFF bytes/IDs.
    restrictor.ID_RE = re.compile(r'(?:^|;)ID=([^;\s]+)')
    restrictor.PARENT_RE = re.compile(r'(?:^|;)Parent=([^;\s]+)')
    counts = restrictor.restrict(Path(reference), originals, Path(restricted))
    require(counts[0] == len(originals), 'reference restriction gene inventory mismatch')
    genes = read_gff(restricted)
    require(set(genes) == originals, 'reference strict parser inventory mismatch')
    return {l.gene_id: select_transcripts(genes, l, reference=True,
            reference_id=rows[l.gene_id]['gene_id_original']) for l in panel}


def select_transcripts(genes, locus, reference=False, *, reference_id=None):
    if reference:
        key = locus.gene_id if reference_id is None else reference_id
        require(key in genes, f'reference missing panel gene {locus.gene_id} (raw {key})')
        gene = genes[key]
        require((gene['chrom'], gene['start'], gene['end'], gene['strand']) ==
                (locus.chromosome, locus.gene_start0, locus.gene_end0, locus.strand), 'panel/reference gene geometry mismatch')
        result = gene['transcripts']
        require(result, 'reference gene has no transcripts')
        return result
    return {tid: tx for gene in genes.values() for tid, tx in gene['transcripts'].items()
            if tx['chrom'] == locus.chromosome and tx['start'] < locus.gene_end0 and tx['end'] > locus.gene_start0}


def structure(tx):
    exons = tx['exons']
    if len(exons) == 1:
        # No CDS cannot be an exact-CDS match, including two noncoding transcripts.
        return (tx['chrom'], tx['strand'], 'mono', tuple(tx['cdss'])) if tx['cdss'] else None
    return (tx['chrom'], tx['strand'], 'multi', tuple((a[1], b[0]) for a, b in zip(exons, exons[1:])))


def write_gtf(path, transcripts):
    with Path(path).open('x') as out:
        for tid, tx in sorted(transcripts.items()):
            require(not any(c in tid for c in '\"\t\n;'), 'unsafe transcript ID for GTF')
            for a, b in tx['exons']:
                out.write(f'{tx["chrom"]}\tMVP\texon\t{a+1}\t{b}\t.\t{tx["strand"]}\t.\tgene_id "g"; transcript_id "{tid}";\n')


def reference_counts(predicted, reference, directory, executable):
    """One-to-one exact matches; ambiguous duplicate structures fail closed."""
    directory = Path(directory)
    directory.mkdir()
    for label, txs in [('predicted', predicted), ('reference', reference)]:
        keys = [structure(t) for t in txs.values() if structure(t) is not None]
        require(len(keys) == len(set(keys)), f'{label} duplicate isoform structures: author multiplicity policy needed')
    pm = {k: v for k, v in predicted.items() if len(v['exons']) > 1}
    rm = {k: v for k, v in reference.items() if len(v['exons']) > 1}
    write_gtf(directory / 'predicted.gtf', pm)
    write_gtf(directory / 'reference.gtf', rm)
    matches = set()
    command = None
    if pm and rm:
        command = [executable, '-r', str(directory / 'reference.gtf'), '-o', str(directory / 'compare'), str(directory / 'predicted.gtf')]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        (directory / 'stdout.txt').write_text(result.stdout)
        (directory / 'stderr.txt').write_text(result.stderr)
        maps = list(directory.glob('*.tmap'))
        require(len(maps) == 1, 'missing/ambiguous GFFCompare tmap')
        with maps[0].open() as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
        require(len(rows) == len(pm) and {r['qry_id'] for r in rows} == set(pm), 'GFFCompare query inventory mismatch')
        for row in rows:
            if row['class_code'] == '=':
                ref, pred = row['ref_id'], row['qry_id']
                require(ref in rm and structure(pm[pred]) == structure(rm[ref]), 'GFFCompare exact-chain mismatch')
                matches.add(ref)
    mono_ref = {structure(tx) for tx in reference.values() if len(tx['exons']) == 1 and structure(tx) is not None}
    mono_pred = {structure(tx) for tx in predicted.values() if len(tx['exons']) == 1 and structure(tx) is not None}
    tp = len(matches) + len(mono_ref & mono_pred)
    counts = {'f1_tp': tp, 'f1_predicted': len(predicted), 'f1_reference': len(reference)}
    return counts, {'command': command, 'counts': counts, 'TP': tp, 'FP': len(predicted)-tp,
                    'FN': len(reference)-tp, 'outputs': [file_identity(p) for p in sorted(directory.iterdir())]}


def prediction_inventory(rows, panel, checkpoints, *, profile='historical-three-seed'):
    from transgenic.training.mvp_profiles import DEVELOPMENT_PROFILE, validate_seed_profile
    development = profile == DEVELOPMENT_PROFILE
    validate_seed_profile(profile, 123 if development else 1)
    roles = {'mvp_seed123': 'mvp_finetune', 'b5_seed123': 'b5_baseline'}
    if development:
        require(len(panel) == 50 and len({l.gene_id for l in panel}) == 50, 'exact full50 panel required')
        require(set(checkpoints) == set(roles) and len(set(checkpoints.values())) == 2,
                'distinct role checkpoint keys required')
        expected = {(profile, run_id, fraction, l.gene_id) for l in panel
                    for run_id, fraction in [('mvp_seed123', 0), ('mvp_seed123', 1), ('b5_seed123', 0)]}
    else:
        expected = {(s, f, l.gene_id) for l in panel for s, f in [(1,0),(1,1),(2,0),(2,1),(3,0),(3,1),(123,0)]}
    actual, paths, stems = {}, set(), set()
    loci = {l.gene_id: l for l in panel}
    for row in rows:
        require(type(row['seed']) is int and type(row['fraction']) is int, 'seed/fraction must be integers')
        if development:
            validate_seed_profile(profile, row['seed'])
            require(row.get('profile') == profile and row.get('run_id') in roles and
                    row.get('run_role') == roles.get(row.get('run_id')), 'prediction profile/run role mismatch')
            metadata = row.get('metadata', {})
            require(isinstance(metadata, dict) and metadata.get('profile', profile) == profile and
                    metadata.get('run_role', row['run_role']) == row['run_role'], 'prediction metadata mismatch')
            if 'seed' in metadata:
                validate_seed_profile(profile, metadata['seed'])
            for field in ('run_id', 'checkpoint_sha256', 'fraction', 'gene_id'):
                require(field not in metadata or (type(metadata[field]) is type(row[field]) and
                        metadata[field] == row[field]), 'prediction metadata identity mismatch')
            key = profile, row['run_id'], row['fraction'], row['gene_id']
        else:
            require(row.get('profile', profile) == profile, 'prediction profile mismatch')
            key = row['seed'], row['fraction'], row['gene_id']
        require(key in expected and key not in actual, 'unexpected/duplicate prediction identity')
        locus = loci[row['gene_id']]
        require(row['window'] == [locus.chromosome, *w.evaluation_window(locus)], 'prediction panel window mismatch')
        require(row['checkpoint_sha256'] == checkpoints[row['run_id'] if development else row['seed']],
                'prediction checkpoint/seed mismatch')
        identity = check_file(row['path'], row)
        stat = Path(identity['path']).stat()
        inode = stat.st_dev, stat.st_ino
        require(inode not in paths, 'duplicate prediction file')
        paths.add(inode)
        if development:
            stem = Path(identity['path']).stem
            require(stem not in stems, 'duplicate prediction file stem')
            stems.add(stem)
        actual[key] = identity
    require(set(actual) == expected, 'exact full50 profile off/on plus baseline inventory required')
    return actual


def produce(input_path, out, *, profile=None):
    from transgenic.training.mvp_cli import validate_candidate_records
    from transgenic.training.mvp_cli_runtime import verify_candidate_artifacts
    import mvp_score
    out = Path(out).resolve()
    payload = read_json(input_path)
    from transgenic.training.mvp_profiles import DEVELOPMENT_PROFILE
    profile = evaluation_profile(payload, profile)
    development = profile == DEVELOPMENT_PROFILE
    inputs = [file_identity(input_path)]
    panel = w.load_panel(payload['panels'])
    inputs.append(file_identity(payload['panels']))
    authority = frozen_reference(payload['fasta'], payload['reference'])
    inputs.extend(authority['files'].values())
    lengths = fasta_lengths(payload['fasta'])
    for locus in panel:
        require(locus.chromosome in lengths, 'panel chromosome absent from FASTA')
        w.evaluation_window(locus, lengths[locus.chromosome])
    c = payload['candidate']
    candidate = validate_candidate_records(c['acceptance'], c['acceptance_sha256'], c['inventory'], c['receipts'])
    artifacts = verify_candidate_artifacts(candidate, c['corpus'], c['evidence'], payload['panels'])
    inputs.extend(file_identity(c[k]) for k in ('acceptance','inventory','receipts','corpus','evidence'))
    runs = verify_run_inventory(payload, artifacts)
    inputs.extend(verify_evaluation_readiness(payload, candidate, runs))
    if 'panel_compatibility' in payload:
        spec = payload['panel_compatibility']
        inputs.append(check_file(spec['path'], spec))
    finals = {'mvp_seed123': runs['mvp_seed123']} if development else runs
    if development:
        checkpoints = {'mvp_seed123': runs['mvp_seed123']['files']['model.safetensors']['sha256'],
                       'b5_seed123': runs['b5_seed123']['weights']['sha256']}
        inputs.append(runs['b5_seed123']['weights'])
    else:
        checkpoints = {int(s): r['files']['model.safetensors']['sha256'] for s, r in runs.items()}
        baselines = {r['initialization']['model.safetensors']['sha256'] for r in runs.values()}
        require(len(baselines) == 1, 'cross-seed baseline mismatch')
        checkpoints[123] = baselines.pop()
    for run in finals.values():
        inputs.extend([run['manifest'], run['done'], run['code_manifest'], *run['files'].values(), *run['code_files'], *run['initialization'].values()])
    decoder = payload['decoder']
    require(decoder['files'] and decoder['commands'] and all(isinstance(cmd, list) and cmd and all(isinstance(x,str) and x for x in cmd) for cmd in decoder['commands']), 'decoder code/commands missing')
    inputs.extend(check_file(p, {'sha256': h}) for p, h in decoder['files'].items())
    predictions = prediction_inventory(payload['predictions'], panel, checkpoints, profile=profile)
    inputs.extend(predictions.values())
    executable = shutil.which('gffcompare')
    require(executable, 'GFFCompare 0.12.6 required')
    executable = str(Path(executable).resolve())
    version = subprocess.run([executable, '--version'], check=True, capture_output=True, text=True)
    require(re.search(r'\bv?0\.12\.6\b', version.stdout + version.stderr), 'GFFCompare version must be 0.12.6')
    inputs.append(file_identity(executable))
    scorer_sources = [Path(__file__), ROOT/'revision/scripts/07_selfconsistency_stats.py',
                      ROOT/'revision/scripts/24_restrict_reference_to_test_genes.py',
                      Path(mvp_score.__file__), Path(mvp_score.b.__file__), Path(w.__file__)]
    scorer_sources.extend(ROOT/'src/transgenic/training'/name for name in
                          ('mvp_cli.py', 'mvp_cli_runtime.py', 'mvp_finalize.py', 'mvp_metrics.py', 'mvp_runtime.py', 'mvp_profiles.py', 'weights_init.py'))
    inputs.extend(file_identity(path) for path in scorer_sources)
    out.mkdir(parents=True, exist_ok=False)
    mapping = resolve_reference_mapping(panel, c['corpus'], authority, artifacts['sha256'], payload['panels'])
    restricted = out / 'reference.panel.gff3'
    reference = prepare_mapped_reference(panel, payload['reference'], restricted, mapping)
    restricted_identity = file_identity(restricted)
    loci, receipts, baseline = [], [], []
    jobs = [('mvp_seed123', 123, 'mvp_finetune', 0), ('mvp_seed123', 123, 'mvp_finetune', 1),
            ('b5_seed123', 123, 'b5_baseline', 0)] if development else [
            (str(s), s, 'mvp_finetune', f) for s in (1, 2, 3) for f in (0, 1)] + [('123', 123, 'b5_baseline', 0)]
    for locus in panel:
        refs = reference[locus.gene_id]
        row = {'gene_id': locus.gene_id, 'seeds': {str(s): {} for s in ((123,) if development else (1,2,3))}}
        for run_id, seed, role, fraction in jobs:
            key = (profile, run_id, fraction, locus.gene_id) if development else (seed, fraction, locus.gene_id)
            identity = predictions[key]
            genes = read_gff(identity['path'])
            start, end = w.evaluation_window(locus)
            require(all(g['chrom'] == locus.chromosome and start <= g['start'] < g['end'] <= end for g in genes.values()), 'prediction outside decoded window')
            predicted = select_transcripts(genes, locus)
            counts, receipt = reference_counts(predicted, refs, out / f'{len(receipts):04d}', executable)
            receipts.append(dict(receipt, seed=seed, fraction=fraction, gene_id=locus.gene_id, prediction=identity))
            if development:
                receipts[-1].update(profile=profile, run_id=run_id, run_role=role, checkpoint_sha256=checkpoints[run_id])
            if role == 'b5_baseline':
                baseline.append({'gene_id': locus.gene_id, **counts})
            else:
                row['seeds'][str(seed)]['on' if fraction else 'off'] = counts
        loci.append(row)
    counts = {'pairs': {s: r['files']['paired_records.jsonl']['path'] for s,r in finals.items()}, 'loci': loci}
    if development:
        counts['profile'] = profile
    result = mvp_score.score(counts, panel)
    result.update(pair_run_provenance_status='FINALIZER_BYTES_METADATA_INVENTORY_CONSISTENT',
                  three_seed_identity_verified=not development, three_seed_policy_verified=not development,
                  count_provenance_status='DERIVED_FROM_RAW_GFF_IN_THIS_INVOCATION',
                  reference_identity_status='COMMIT_BOUND_SPECIES_MANIFEST',
                  baseline_reference_counts=baseline,
                  provenance_caveat='Receipts establish local consistency, not historical execution or scientific approval. Seed-123 source bytes agree with initialization declarations; an external concrete pin for the intended historical best checkpoint remains required.')
    if development:
        result.pop('three_seed_identity_verified')
        result.pop('three_seed_policy_verified')
        result['provenance_caveat'] = 'Local byte/metadata consistency, not historical execution or scientific approval; baseline independently pinned and matched to final initialization audit.'
    baseline_rows = [{'gene_id': c['gene_id'], 'seeds': {'123': {'off': {k:v for k,v in c.items() if k != 'gene_id'}}}} for c in baseline]
    core_ids = {l.gene_id for l in panel if l.panel == 'mvp_core'}
    result['baseline_reference_F1_percent'] = {
        'all50': mvp_score.metric(baseline_rows, 'off', 'F1', (123,)),
        'core20': mvp_score.metric([r for r in baseline_rows if r['gene_id'] in core_ids], 'off', 'F1', (123,))}
    for identity in inputs:
        check_file(identity['path'], identity)
    recheck_reference_mapping(mapping)
    for name, obj in [('counts.json', counts), ('statistics.json', result)]:
        with (out/name).open('x') as f:
            json.dump(obj, f, indent=2, allow_nan=False)
    receipt = {'label': 'exploratory / MVP', 'authority': authority, 'reference_mapping': mapping, 'inputs': inputs, 'runs': runs,
               'decoder_recorded_commands': decoder['commands'], 'gffcompare_version': version.stdout + version.stderr,
               'gffcompare_version_command': [executable, '--version'],
               'comparisons': receipts, 'outputs': [restricted_identity, *[file_identity(out/n) for n in ('counts.json','statistics.json')]],
               'independent_read_splice_status': 'ABSENT_NOT_NEGATIVE', 'coverage_policy': 'PENDING_AUTHOR',
               'withholding_status': 'UNASSESSABLE', 'decision': None}
    if development:
        receipt.update(profile=profile, coverage_policy='APPROVED_SOURCE_OR_MISSING_NOT_NEGATIVE',
                       structural_status='MISSING_UNASSESSABLE_NOT_ZERO')
    stage_json(out/'receipt.json', receipt)
    def final_recheck():
        recheck_files(inputs + receipt['outputs'])
        recheck_reference_mapping(mapping)
    publish_staged([out/'receipt.json'], final_recheck)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    from transgenic.training.mvp_profiles import LEGACY_PROFILE, DEVELOPMENT_PROFILE
    parser.add_argument('--profile', choices=(LEGACY_PROFILE, DEVELOPMENT_PROFILE))
    args = parser.parse_args(argv)
    produce(args.input, args.out, profile=args.profile)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
