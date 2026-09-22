#!/usr/bin/env python3
"""Scoring-only, local two-pass long-read alignment; never training evidence.

Canonical roles: ROOT/evidence/DATASET_ROLES.longread_v1.tsv (not repo copy).
Outputs only ROOT/evidence/mvp_scoring_20260916/RUN, fresh directory required.
External commands run only with --execute; this file submits no jobs.
Requires an EXTERNAL frozen-reference identity gate: caller MD5 values check
bytes only, not frozen reference/GFF identity. This partial preparatory driver
cannot establish scientific input validation, coverage approval, or training
authorization. Operational promotion also requires anchored producer receipts.
"""
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS.parents[1] / 'src'))
from transgenic.evidence import mvp_windows as w

ROLE_COLUMNS = ('dataset run species genotype_stratum instrument data_type expected_files '
                'expected_reads source_checksum source_checksum_authority local_fa_md5 role basis note').split()
ONT_RUNS = {'SRR28341943', 'SRR28341944', 'SRR28341945'}  # A-ONT1
# PacBio leg (labelled HiFi, PacBio threshold 2) is A-CCS23 per PROTOCOL_B1
# A47.2 (v1.38): PRJNA911826 Col-0 WT runs only, lima-processed full-length
# Iso-Seq (primers 0.000, 3' poly(A) only in 98.8-99.4 %, concatemer signal
# 0.4-0.9 %). splice:hq suits these CCS-level FL reads; 3' poly(A) tails are
# soft-clipped. A-CCS23 supports junctions / intron chains only, never TES.
HIFI_RUNS = {'SRR22719002', 'SRR22719003', 'SRR22719004',
             'SRR22719005', 'SRR22719006', 'SRR22719007'}  # A-CCS23
RUN_DATASETS = dict({run: 'A-ONT1' for run in ONT_RUNS}, **{run: 'A-CCS23' for run in HIFI_RUNS})
# A-HiFi (ERR13994458) publishes subreads only and no FLNC product exists:
# unavailable, missing per source (never zero), and refused here.
UNAVAILABLE_RUNS = {'ERR13994458': 'A-HiFi'}


def require_scoring_run(run):
    if run in UNAVAILABLE_RUNS:
        raise ValueError(f'{run} ({UNAVAILABLE_RUNS[run]}) is unavailable under A47.2: '
                         'subreads only, no FLNC product; missing, never zero')
    if run not in ONT_RUNS | HIFI_RUNS:
        raise ValueError('run is not A-ONT1 / A-CCS23')


def digest(path):
    h = hashlib.md5()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def validate_read(root, run, reads):
    require_scoring_run(run)
    roles = Path(root).resolve() / 'evidence/DATASET_ROLES.longread_v1.tsv'
    with roles.open() as f:
        reader = csv.DictReader(f, delimiter='\t')
        if reader.fieldnames != ROLE_COLUMNS:
            raise ValueError('canonical roles schema mismatch')
        rows = list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError('malformed canonical roles row')
    selected = [r for r in rows if r['run'] == run]
    if len(selected) != 1:
        raise ValueError('expected exactly one canonical run role')
    row = selected[0]
    if (row['role'], row['species'], row['genotype_stratum']) != ('b1_validation_only', 'Athaliana', 'reference'):
        raise ValueError('scoring requires reference Athaliana b1_validation_only; no training bypass')
    if not re.fullmatch('[0-9a-f]{32}', row['local_fa_md5']) or digest(Path(reads).resolve()) != row['local_fa_md5']:
        raise ValueError('actual consumed read file md5 mismatch')
    return row


def audit_key(reads, ref, gff):
    return ':'.join(digest(p) for p in (reads, ref, gff, SCRIPTS / '67_orientation_audit.py'))


def read_audit(out, key, run):
    marker = dict(line.split('=', 1) for line in (out / 'DONE').read_text().splitlines())
    if marker.get('key') != key or marker.get('audit_md5') != digest(out / 'audit.json'):
        raise ValueError('stale orientation DONE key or audit md5')
    result = json.loads((out / 'audit.json').read_text())
    if result['run'] != run or result['species'] != 'Athaliana':
        raise ValueError('orientation identity mismatch')
    if (result['min_mapq'], result['min_eligible'], result['threshold']) != (20, 1000, .95):
        raise ValueError('orientation parameters drifted')
    valid, sense = result['valid_for_orientation'], result['read_sense']
    if type(valid) is not int or type(sense) is not int or not 0 <= sense <= valid:
        raise ValueError('invalid orientation counts')
    status = 'UNRESOLVED' if valid < 1000 else ('PASS' if sense / valid >= .95 else 'FAIL')
    if result['status'] != status or result['use_uf'] is not (status == 'PASS'):
        raise ValueError('inconsistent orientation decision')
    return result


def audit_from_sam(sam, gff, out, run, dataset, key):
    """Legacy scorer accepts arbitrary output root; retain four-field DONE key."""
    if (out / 'audit.json').exists() or (out / 'DONE').exists():
        raise ValueError('refuse overwriting orientation artifact')
    cmd = [sys.executable, str(SCRIPTS / '67_orientation_audit.py'), '--sam', str(sam),
           '--gff', str(gff), '--out', str(out / 'audit.json'), '--run', run,
           '--species', 'Athaliana', '--dataset', dataset]
    with (out / 'audit.stdout').open('x') as stdout, (out / 'audit.stderr').open('x') as stderr:
        rc = subprocess.run(cmd, stdout=stdout, stderr=stderr).returncode
    if rc not in (0, 4):
        raise ValueError(f'orientation scorer failed: {rc}')
    marker = f'key={key}\nrun={run}\naudit_md5={digest(out / "audit.json")}\n'
    (out / 'DONE.tmp').write_text(marker)
    (out / 'DONE.tmp').rename(out / 'DONE')
    return read_audit(out, key, run)


def pass2_command(source, uf, ref, reads, threads):
    if source not in ('ONT', 'HiFi') or (source == 'HiFi' and uf):
        raise ValueError('invalid source/orientation combination')
    command = ['minimap2', '-t', str(threads), '-ax', 'splice' if source == 'ONT' else 'splice:hq']
    if source == 'ONT':
        command += (['-uf'] if uf else []) + ['-k14']
    return command + ['-G', '200000', '--secondary=no', '--cs=long', str(ref), str(reads)]


def select_names(paf_lines, loci):
    windows = w.prefilter_by_chromosome(loci)
    names = set()
    for line in paf_lines:
        f = line.rstrip('\n').split('\t')
        if len(f) < 12:
            raise ValueError('malformed pass1 PAF')
        start, end, quality = int(f[7]), int(f[8]), int(f[11])
        if not 0 <= start < end <= int(f[6]) or not 0 <= quality <= 255:
            raise ValueError('invalid pass1 coordinates/mapq')
        if quality >= 5 and any(start < b and end > a for a, b in windows.get(f[5], ())):
            names.add(f[0])
    return sorted(names)


def output_directory(root, run):
    require_scoring_run(run)
    out = Path(root).resolve() / 'evidence/mvp_scoring_20260916' / run
    if out.resolve() != out:
        raise ValueError('scoring namespace must not be redirected by symlinks')
    return out


def execute(args):
    root, reads, ref, gff = (Path(p).resolve() for p in (args.root, args.reads, args.ref, args.gff))
    row = validate_read(root, args.run, reads)
    for path, expected in [(ref, args.ref_md5), (gff, args.gff_md5)]:
        if not re.fullmatch('[0-9a-f]{32}', expected) or digest(path) != expected:
            raise ValueError('reference/GFF md5 mismatch')
    loci = w.load_panel(args.panels)
    # No dependence on an unverified .fai: stream actual FASTA lengths.
    lengths = {}
    chrom = None
    with ref.open() as f:
        for line in f:
            if line.startswith('>'):
                chrom = line[1:].split()[0]
                if chrom in lengths:
                    raise ValueError('duplicate reference contig')
                lengths[chrom] = 0
            elif line.strip():
                if chrom is None:
                    raise ValueError('invalid reference FASTA')
                lengths[chrom] += len(line.strip())
    w.windows_by_chromosome(loci, lengths)
    if args.threads < 1:
        raise ValueError('threads must be positive')
    boundary = dict(reference_identity_status='UNVERIFIED',
                    scientific_input_validation=False,
                    external_frozen_reference_gate_required=True,
                    training_eligible=False)
    if not args.execute:
        return dict(boundary, status='PREPARATORY_BYTE_CHECKS_ONLY',
                    role=row, coverage_policy='PENDING')
    out = output_directory(root, args.run)
    out.mkdir(parents=True, exist_ok=False)
    roles = root / 'evidence/DATASET_ROLES.longread_v1.tsv'
    inputs = {str(p): digest(p) for p in (reads, ref, gff, roles, Path(args.panels).resolve())}
    commands = []

    def run(command, output=None):
        commands.append([str(x) for x in command])
        (out / 'commands.json').write_text(json.dumps(commands, indent=2))
        with (out / 'commands.stderr').open('a') as err:
            if output is None:
                subprocess.run(command, check=True, stderr=err)
            else:
                with output.open('x') as f:
                    subprocess.run(command, check=True, stdout=f, stderr=err)
    try:
        source = 'ONT' if args.run in ONT_RUNS else 'HiFi'
        uf = False
        if source == 'ONT':
            audit = out / 'orientation'; audit.mkdir()
            run([sys.executable, str(SCRIPTS / '68_sample_fasta.py'), '--fasta', str(reads),
                 '--out', str(audit / 'sample.fa'), '--n', '10000', '--md5', digest(reads)], audit / 'sample.json')
            run(pass2_command('ONT', False, ref, audit / 'sample.fa', args.threads), audit / 'sample.sam')
            uf = audit_from_sam(audit / 'sample.sam', gff, audit, args.run, row['dataset'], audit_key(reads, ref, gff))['use_uf']
        run(['minimap2', '-t', str(args.threads), '-x', 'splice', str(ref), str(reads)], out / 'pass1.paf')
        with (out / 'pass1.paf').open() as f:
            names = select_names(f, loci)
        (out / 'panel_reads.txt').write_text(''.join(n + '\n' for n in names))
        if not names:
            raise ValueError('no candidate reads; independent scoring unavailable, not negative')
        run(['seqkit', 'grep', '-f', str(out / 'panel_reads.txt'), str(reads)], out / 'panel_reads.fa')
        run(pass2_command(source, uf, ref, out / 'panel_reads.fa', args.threads), out / 'panel.sam')
        run(['samtools', 'sort', '-o', str(out / 'panel.bam'), str(out / 'panel.sam')])
        run(['samtools', 'index', str(out / 'panel.bam')])
        run([sys.executable, str(SCRIPTS / 'mvp/mvp_junction_table.py'), '--alignment',
             f'{source}:uf={str(uf).lower()}:{out / "panel.bam"}', '--panels', str(Path(args.panels).resolve()),
             '--out', str(out / 'junctions.tsv'), '--qc-out', str(out / 'junctions.qc.json')])
        if any(digest(p) != d for p, d in inputs.items()):
            raise ValueError('input mutated during scoring')
        report = dict(boundary, label='exploratory / MVP', role=row, inputs_md5=inputs,
                      commands=commands, source=source, use_uf=uf, run=args.run,
                      dataset=RUN_DATASETS[args.run],
                      panel_sha256=w.PANEL_SHA256, coverage_policy='PENDING',
                      outputs_md5={p.name: digest(p) for p in out.iterdir() if p.is_file()},
                      status='PREPARATORY_COMMANDS_COMPLETED')
        (out / 'alignment.json.tmp').write_text(json.dumps(report, indent=2))
        (out / 'alignment.json.tmp').rename(out / 'alignment.json')
        return report
    except Exception as error:
        (out / 'FAILED.json').write_text(json.dumps({'error': str(error), 'training_eligible': False}))
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'run', 'reads', 'ref', 'gff', 'panels'):
        p.add_argument('--' + name, required=True)
    for name in ('ref-md5', 'gff-md5'):
        p.add_argument('--' + name, required=True,
                       help='caller-supplied byte checksum only; external frozen-reference identity gate still required')
    p.add_argument('--threads', type=int, default=32)
    p.add_argument('--execute', action='store_true')
    print(json.dumps(execute(p.parse_args(argv)), indent=2))


if __name__ == '__main__':
    main()
