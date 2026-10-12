#!/usr/bin/env python3
"""TEST ONLY: unchanged frozen CLI bodies with explicit fixture input seams.

This is not the production wrapper. No submission, acquisition or canonical inputs.
The only nested CLI rewrite is a test bootstrap for the junction panel binding.
"""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import resource
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
RUN = 'SRR28341943'  # Frozen enum only, NOT the biological contents of that accession.
DATASET = 'ont/Athaliana_FLIC_PRJNA1087576'
GATE_SHA = '0b5be5e4863a348f0fb35a946e6d989980428483a72055e4d3f12492b7f7a17f'
PINS_SHA = '38c56c608336ef3fdcb83eaa5eb35a355ddc77ea1e8aff8387160603bf88ce34'
WRAPPER_SHA = '314d6027d9d146d63c6f1b2e2b71783ac5e1f63e95e6e31252614ddcf20728dd'
ROLE_COLUMNS = ('dataset run species genotype_stratum instrument data_type expected_files expected_reads source_checksum source_checksum_authority local_fa_md5 role basis note').split()
PANEL_COLUMNS = ('panel shuffle_rank species_id gene_id chromosome start fin strand length orthogroup_id n_transcripts exon_count intron_count est_molecules').split()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')


def inventory(root):
    result = {}
    for p in sorted(Path(root).rglob('*')):
        require(not p.is_symlink(), 'redirected inventory entry')
        if p.is_file():
            result[str(p.relative_to(root))] = dict(size=p.stat().st_size, sha256=digest(p))
    return result


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def verify_release(release):
    release = Path(release).resolve(strict=True)
    for name, sha in [('gate.py', GATE_SHA), ('source_pins.json', PINS_SHA), ('wrapper.py', WRAPPER_SHA)]:
        require(digest(release/name) == sha, 'release pin mismatch: '+name)
    pins = json.loads((release/'source_pins.json').read_text())
    w = module(release/'wrapper.py', 'fixture_original_wrapper')
    w.check_source_closure(release/'frozen', pins)
    for name, sha in pins['files'].items():
        require(digest(release/'frozen'/name) == sha, 'frozen source mismatch: '+name)
    return w


def generate(root):
    """Seven tiny reads, two fixed splice junctions, unspliced/outside controls."""
    root = Path(root)
    root.mkdir(mode=0o700, exist_ok=False)
    genome = root/'genomes'; genome.mkdir()
    rng = random.Random(20260917)
    seq = list(''.join(rng.choice('ACGT') for _ in range(60000)))
    for start in (20500, 30500):
        seq[start-1] = 'C'
        seq[start:start+2] = 'GT'
        seq[start+198:start+200] = 'AG'
        seq[start+200] = 'T'
    seq = ''.join(seq)
    outside = ''.join(rng.choice('ACGT') for _ in range(4000))
    ref = genome/'fixture.fa'
    ref.write_text('>fixture_chr\n'+seq+'\n>outside_chr\n'+outside+'\n')
    gff = genome/'fixture.gff3'
    lines = ['##gff-version 3']
    for i, start in enumerate((20000, 30000)):
        lines += [f'fixture_chr\tfixture\tgene\t{start+1}\t{start+1200}\t.\t+\t.\tID=g{i}',
                  f'fixture_chr\tfixture\tmRNA\t{start+1}\t{start+1200}\t.\t+\t.\tID=t{i};Parent=g{i}',
                  f'fixture_chr\tfixture\texon\t{start+1}\t{start+500}\t.\t+\t.\tParent=t{i}',
                  f'fixture_chr\tfixture\texon\t{start+701}\t{start+1200}\t.\t+\t.\tParent=t{i}']
    gff.write_text('\n'.join(lines)+'\n')
    reads = root/'evidence'/DATASET/(RUN+'.fa.gz')
    reads.parent.mkdir(parents=True)
    records = [(f'positive_{i}', seq[20000:20500]+seq[20700:21200]) for i in range(3)]
    records += [(f'below_threshold_{i}', seq[30000:30500]+seq[30700:31200]) for i in range(2)]
    records += [('unspliced', seq[40000:41000]), ('outside', outside[1000:2000])]
    with reads.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as out:
            out.write(''.join('>'+name+'\n'+sequence+'\n' for name, sequence in records).encode())
    row = dict.fromkeys(ROLE_COLUMNS, 'SYNTHETIC_TEST_ONLY')
    row.update(dataset=DATASET, run=RUN, species='Athaliana', genotype_stratum='reference',
               role='b1_validation_only', expected_reads='7', local_fa_md5=digest(reads, 'md5'))
    with (root/'evidence/DATASET_ROLES.longread_v1.tsv').open('x') as out:
        writer = csv.DictWriter(out, fieldnames=ROLE_COLUMNS, delimiter='\t', lineterminator='\n')
        writer.writeheader(); writer.writerow(row)
    with (root/'panel.tsv').open('x') as out:
        writer = csv.writer(out, delimiter='\t', lineterminator='\n'); writer.writerow(PANEL_COLUMNS)
        for i in range(50):
            writer.writerow(['mvp_core' if i < 20 else 'mvp', i, 'Athaliana', f'TEST_ONLY_{i}',
                             'fixture_chr', 20001, 41000, '+', 21000, '.', 1, 2, 1, 0])
    (root/'authority.json').write_text(json.dumps(dict(scope='SYNTHETIC_TEST_ONLY',
        fasta_md5=digest(ref, 'md5'), gff_md5=digest(gff, 'md5')), sort_keys=True)+'\n')
    rows = [dict(chrom='fixture_chr', intron_start0=start, intron_end0=start+199,
                 tx_strand='+', n_ONT=count, n_HiFi=0, supported=count >= 3)
            for start, count in ((20500, 3), (30500, 2))]
    write_json(root/'expected.json', dict(rows=rows, absent_decoy=[20509, 20708],
        selected_names=sorted(name for name, _ in records if name != 'outside'),
        orientation_status='UNRESOLVED', use_uf=False, read_records=7))


def panel_seam(driver, root):
    """Use frozen loader's explicit fixture-hash API, retaining 20/30 counts."""
    w = driver.w
    original = w.load_panel
    sha = digest(root/'panel.tsv')
    def load_panel(path, expected_sha256=None, labels=w.MVP_LABELS):
        require(Path(path).resolve() == root/'panel.tsv', 'fixture panel path mismatch')
        require(expected_sha256 in (None, sha), 'fixture panel hash mismatch')
        return original(path, expected_sha256=sha, labels=labels)
    w.load_panel = load_panel
    w.PANEL_SHA256 = sha
    return sha


def bindings(release, root):
    verify_release(release)
    gate = module(release/'gate.py', 'fixture_original_gate')
    driver = gate.load_driver(release/'frozen')
    panel_seam(driver, root)
    return gate, driver


def entry(release, root, kind, argv):
    """Test bootstrap. Frozen main parses the unchanged CLI argument tail."""
    require(json.loads((root/'authority.json').read_text())['scope'] == 'SYNTHETIC_TEST_ONLY', 'synthetic inputs only')
    require((root/'genomes/fixture.fa').stat().st_size < 100000, 'tiny reference only')
    require((root/'evidence'/DATASET/(RUN+'.fa.gz')).stat().st_size < 100000, 'tiny reads only')
    gate, driver = bindings(release, root)
    if kind == 'junction':
        j = module(release/'frozen/revision/scripts/mvp/mvp_junction_table.py', 'fixture_junction_cli')
        return j.main(argv)
    original_run = subprocess.run
    def run(command, *args, **kwargs):
        actual = list(map(str, command))
        logical = actual[:]
        if len(actual) > 1 and actual[1] == str(release/'frozen/revision/scripts/mvp/mvp_junction_table.py'):
            actual = [sys.executable, '-I', '-B', str(HERE/'fixture.py'), 'entry', str(release), str(root), 'junction', '--']+actual[2:]
        result = original_run(actual, *args, **kwargs)
        with (root.parent/'nested_execution.jsonl').open('a') as ledger:
            ledger.write(json.dumps(dict(logical_argv=logical, actual_argv=actual, exit_code=result.returncode))+'\n')
        return result
    subprocess.run = run
    driver.main(argv)
    return 0


def require_rows(actual, expected):
    require(actual == expected, 'fixed expected junction rows differ: '+repr(actual))


def postflight(release, root, receipt):
    gate, driver = bindings(release, root)
    authority_bytes = (root/'authority.json').read_bytes()
    # Test-only authority seam, not Git92 approval. No changes to read binding.
    def synthetic_reference(blob, fasta, gff):
        require(blob == authority_bytes, 'synthetic authority changed')
        authority = json.loads(blob)
        files = {}
        for key, p in [('fasta', fasta), ('gff', gff)]:
            require(digest(p, 'md5') == authority[key+'_md5'], 'synthetic reference changed')
            files[key] = dict(gate.identity(p), md5=authority[key+'_md5'])
        return dict(authority='SYNTHETIC_TEST_ONLY_NOT_GIT92', manifest_sha256=hashlib.sha256(blob).hexdigest(), files=files)
    gate.reference_binding = synthetic_reference
    args = SimpleNamespace(root=str(root), run=RUN, reads=str(root/'evidence'/DATASET/(RUN+'.fa.gz')),
        fasta=str(root/'genomes/fixture.fa'), gff=str(root/'genomes/fixture.gff3'),
        panels=str(root/'panel.tsv'), authority=str(root/'authority.json'), source=str(release/'frozen'), threads=4)
    result = gate.verify_alignment(args, receipt)
    out = driver.output_directory(root, RUN)
    j = module(release/'frozen/revision/scripts/mvp/mvp_junction_table.py', 'fixture_junction_postflight')
    expected = json.loads((root/'expected.json').read_text())
    require_rows(j.read_table(out/'junctions.tsv'), expected['rows'])
    require((out/'panel_reads.txt').read_text().splitlines() == expected['selected_names'], 'candidate identities differ')
    audit = json.loads((out/'orientation/audit.json').read_text())
    require(audit['status'] == 'UNRESOLVED' and audit['use_uf'] is False, 'tiny orientation must remain unresolved')
    commands = json.loads((out/'commands.json').read_text())
    require(commands[0] == [sys.executable, str(release/'frozen/revision/scripts/68_sample_fasta.py'),
        '--fasta', args.reads, '--out', str(out/'orientation/sample.fa'), '--n', '10000', '--md5', digest(args.reads, 'md5')], 'sample CLI drift')
    require(commands[1] == list(map(str, driver.pass2_command('ONT', False, args.fasta, out/'orientation/sample.fa', 4))), 'orientation CLI drift')
    ledger = [json.loads(line) for line in (root.parent/'nested_execution.jsonl').read_text().splitlines()]
    require(len(ledger) == 9, 'eight driver commands plus actual audit required')
    require([x['logical_argv'] for x in ledger if '67_orientation_audit.py' not in ' '.join(x['logical_argv'])] == commands, 'actual subprocess ledger differs')
    for item in ledger:
        expected_rc = 4 if '67_orientation_audit.py' in ' '.join(item['logical_argv']) else 0
        require(item['exit_code'] == expected_rc, 'nested CLI exit mismatch')
        if 'mvp_junction_table.py' not in ' '.join(item['logical_argv']):
            require(item['actual_argv'] == item['logical_argv'], 'unexpected CLI seam')
    result['fixture_oracle'] = expected
    return result


def run(release, out):
    require(sys.version_info >= (3, 9), 'Python >=3.9 required')
    require(os.environ.get('SLURM_JOB_ID', '').isdigit(), 'compute allocation required; never run on login')
    require(os.environ.get('SLURM_CPUS_PER_TASK') == '4', 'four CPUs required')
    require(len(os.sched_getaffinity(0)) == 4, 'bind exactly four CPUs')
    require(not str(out).startswith('/data/gpfs/assoc/pgl/data/Transgenic'), 'canonical tree forbidden')
    require(out.is_absolute() and out.resolve() == out, 'nonredirected fresh output required')
    require(not out.is_relative_to(release) and not release.is_relative_to(out), 'release/output overlap')
    out.mkdir(mode=0o700, exist_ok=False)
    wrapper = verify_release(release)
    before = inventory(release)
    runner_before = {name: digest(HERE/name) for name in ('fixture.py', 'runtime.json', 'fixture.sbatch', 'MANIFEST.sha256')}
    runtime = json.loads((HERE/'runtime.json').read_text())
    # Controller pins this separate approval file only after independent review.
    approval_path = Path(os.environ['ONT_FIXTURE_APPROVAL'])
    require(digest(approval_path) == os.environ['ONT_FIXTURE_APPROVAL_SHA256'], 'approval hash mismatch')
    approval = json.loads(approval_path.read_text())
    require(approval['scope'] == 'TEST_ONLY_REAL_TOOL_FIXTURE' and approval['runtime_closure_approved'] is True,
            'controller runtime approval required')
    require(approval['scratch_hard_limit_bytes'] == 256 << 20 and approval['output'] == str(out), 'quota-bound fresh output approval required')
    require(approval['cpus'] == 4 and approval['memory_mib'] == 4096 and approval['wall_seconds'] == 360,
            'reviewed budget mismatch')
    require(approval['manifest_sha256'] == digest(HERE/'MANIFEST.sha256'), 'approval/release mismatch')
    write_json(out/'controller_approval.json', approval)
    tools = runtime['tools']
    require(Path(sys.executable).resolve() == Path(tools['python']['path']), 'wrong interpreter')
    tool_inputs = {name: dict(entry, actual_sha256=digest(entry['path'])) for name, entry in tools.items()}
    for name, item in tool_inputs.items():
        require(item['actual_sha256'] == item['sha256'], 'tool hash mismatch: '+name)
    env = dict(PATH=str(Path(tools['minimap2']['path']).parent), HOME=str(out), TMPDIR=str(out),
        LANG='C', LC_ALL='C', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', GOMAXPROCS='4')
    versions = {}
    for name, item in tools.items():
        p = subprocess.run([item['path']]+item['version_args'], capture_output=True, text=True, timeout=10, env=env)
        require(p.returncode == 0 and p.stdout.splitlines()[0] == item['version'], 'version mismatch: '+name)
        versions[name] = dict(stdout=p.stdout, stderr=p.stderr, exit_code=p.returncode)
    write_json(out/'runtime_observed.json', dict(tools=tool_inputs, versions=versions))
    root = out/'synthetic'; generate(root)
    input_inventory = inventory(root)
    argv = [sys.executable, '-I', '-B', str(HERE/'fixture.py'), 'entry', str(release), str(root), 'driver', '--',
        '--root', str(root), '--run', RUN, '--reads', str(root/'evidence'/DATASET/(RUN+'.fa.gz')),
        '--ref', str(root/'genomes/fixture.fa'), '--gff', str(root/'genomes/fixture.gff3'),
        '--panels', str(root/'panel.tsv'), '--ref-md5', digest(root/'genomes/fixture.fa', 'md5'),
        '--gff-md5', digest(root/'genomes/fixture.gff3', 'md5'), '--threads', '4', '--execute']
    receipt = dict(scope='TEST_ONLY_REAL_TOOL_FIXTURE', status='INCOMPLETE', structural_status='UNASSESSABLE',
                   training_eligible=False, decision=None, real_fixture_controller_accepted=False)
    # Per-process safety nets. Slurm cgroup is the aggregate RAM/CPU containment.
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 << 20, 4 << 20))
    resource.setrlimit(resource.RLIMIT_CPU, (240, 240))
    start = time.monotonic()
    try:
        wrapper.child(argv, out, env, 240, 5, receipt)
        require(receipt['exit_code'] == 0, 'frozen driver CLI failed')
        gate = module(release/'gate.py', 'fixture_inventory_gate')
        scored = root/'evidence/mvp_scoring_20260916'/RUN
        receipt['outputs'] = gate.inventory(scored)
        streams = {name: dict(receipt[name]) for name in ('stdout', 'stderr')}
        receipt['postflight'] = postflight(release, root, receipt)
        for name in streams:
            require(wrapper.identity(Path(streams[name]['path'])) == streams[name], 'diagnostic changed')
        require(inventory(release) == before, 'release changed')
        require({name: digest(HERE/name) for name in runner_before} == runner_before, 'fixture launcher changed')
        for name, expected in input_inventory.items():
            require(dict(size=(root/name).stat().st_size, sha256=digest(root/name)) == expected, 'synthetic input changed')
        for name, item in tools.items():
            require(digest(item['path']) == item['sha256'], 'tool changed: '+name)
        gate.verify_inventory(scored, receipt['outputs'])
        require(sum(v['size'] for v in inventory(out).values()) < 256 << 20, 'aggregate output cap exceeded')
        receipt.update(status='FIXTURE_WORKLOAD_PASS_PENDING_CONTROLLER', elapsed_seconds=time.monotonic()-start,
                       seams=['synthetic enum-labelled role/input root, not canonical authority',
                              'explicit fixture panel hash; overlapping 50 test loci, not canonical panel',
                              'junction bootstrap actual argv recorded separately',
                              'synthetic reference binding, not Git92'])
        receipt['diagnostics'] = gate.inventory(out)
        for name, original in streams.items():
            require(wrapper.identity(Path(original['path'])) == original, 'final diagnostic changed')
            require(receipt['diagnostics']['driver.'+name] == {key: original[key] for key in ('size', 'sha256')},
                    'final diagnostic inventory differs from captured stream')
        receipt['runner_sha256'] = runner_before
        receipt['release_inventory'] = before
        wrapper.save(out/'receipt.json', receipt)
        (out/'COMPLETE').write_text(digest(out/'receipt.json')+'\n')
        print(json.dumps(receipt, sort_keys=True))
    except BaseException as error:
        receipt.update(status='INCOMPLETE', error=repr(error))
        wrapper.save(out/'receipt.json', receipt)
        write_json(out/'FAILED.json', dict(error=repr(error), training_eligible=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    g = sub.add_parser('generate'); g.add_argument('out', type=Path)
    r = sub.add_parser('run'); r.add_argument('release', type=Path); r.add_argument('out', type=Path)
    e = sub.add_parser('entry'); e.add_argument('release', type=Path); e.add_argument('root', type=Path)
    e.add_argument('kind', choices=['driver', 'junction']); e.add_argument('argv', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == 'generate':
        generate(args.out)
    elif args.mode == 'entry':
        return entry(args.release.resolve(), args.root.resolve(), args.kind,
                     args.argv[1:] if args.argv[:1] == ['--'] else args.argv)
    else:
        run(args.release, args.out)
    return 0

if __name__ == '__main__':
    sys.exit(main())
