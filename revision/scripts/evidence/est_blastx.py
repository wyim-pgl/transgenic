#!/usr/bin/env python3
"""Reviewed-input EST blastx preparation/search. Compute nodes only; never submits jobs.

prepare-lineage writes a NEW candidate cache; review its hash before run.
run only reads frozen lineages and starts from A19-prefiltered OrthoDB or reviewed
Swiss-Prot FASTA, never from the taxonomy-free historical .dmnd databases.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HELPER = Path(__file__).resolve().parents[1] / '64_orthodb_filter.py'
spec = importlib.util.spec_from_file_location('orthodb_taxonomy', HELPER)
taxonomy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(taxonomy)

SPECIES = dict(Athaliana=3702, Bdistachyon=15368, Gmax=3847, Osativa=4530,
               Ppatens=3218, Ptrichocarpa=3694, Sbicolor=4558,
               Sitalica=4555, Vvinifera=29760)
SOURCE_NAMES = dict(orthodb='odb12_Viridiplantae.filtered.fa.gz',
                    swissprot='swissprot_viridiplantae_reviewed.fa.gz')
RAW13 = 'qseqid qlen sseqid slen qstart qend sstart send length evalue bitscore qframe btop'.split()
RICH17 = RAW13 + 'cigar qseq_gapped sseq_gapped qseq_translated'.split()
VERSION = '2.1.24.178'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_hash(path, expected):
    if not re.fullmatch('[0-9a-f]{64}', expected) or sha256(path) != expected:
        raise ValueError(f'reviewed SHA256 mismatch: {path}')


def write_json(path, value):
    with open(path, 'x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')


@contextmanager
def work_directory(path):
    """Exclusive directory; failed artefacts remain for diagnosis, never reused."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    except BaseException as exc:
        write_json(path / 'FAILED.json', {'error': str(exc), 'type': type(exc).__name__})
        raise
    else:
        with open(path / 'DONE', 'x') as f:
            f.write('Outputs and QC complete; see provenance.json\n')


def subject_taxid(header, source_kind):
    if source_kind == 'orthodb':
        return taxonomy.taxid_of_header(header)
    if source_kind != 'swissprot':
        raise ValueError(f'unknown protein source: {source_kind}')
    values = re.findall(r'(?:^|\s)OX=([^\s]+)', header)
    if len(values) != 1 or not values[0].isdigit() or int(values[0]) <= 0:
        raise ValueError(f'expected exactly one Swiss-Prot OX taxid: {header!r}')
    return int(values[0])


def records(path):
    """Streaming records, retaining the entire header and accession version."""
    # Existing reader is used for source interoperability; guard malformed/empty records.
    with taxonomy._open(str(path)) as f:
        first = next((line for line in f if line.strip()), '')
    if not first.startswith('>'):
        raise ValueError(f'empty or malformed FASTA: {path}')
    for header, sequence in taxonomy.read_fasta(str(path)):
        if not header.split() or not sequence or any(c.isspace() for c in sequence):
            raise ValueError(f'empty/malformed FASTA record: {header!r}')
        yield header, sequence


def observed_taxids(source, source_kind):
    return Counter(subject_taxid(h, source_kind) for h, _ in records(source))


def frozen_lineage(path, present):
    lineages = taxonomy.read_lineage(str(path))
    # NCBI LineageEx commonly ends at cellular organisms (131567), omitting
    # root (1). Reject local descendant snippets, but accept either full format.
    # Endpoint/self checks do not replace review of complete taxonomy provenance.
    bad = [t for t in present if t not in lineages or t not in lineages[t]
           or not ({1, 131567} & set(lineages[t]))]
    if bad:
        raise ValueError(f'missing/incomplete frozen lineage (self and root/cellular ancestor required): {sorted(bad)[:10]}')
    return lineages


def filter_source(source, source_kind, lineage, species_root, output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    source_hash, lineage_hash = sha256(source), sha256(lineage)
    present = observed_taxids(source, source_kind)
    lineages = frozen_lineage(lineage, present)
    excluded = taxonomy.excluded_taxids({species_root}, lineages, set(present))
    if set(present) == excluded:
        raise ValueError('no cross-species proteins remain')
    kept = Counter()
    with open(output, 'x') as f:
        for header, sequence in records(source):
            tid = subject_taxid(header, source_kind)
            if tid not in excluded:
                f.write(f'>{header}\n{sequence}\n')
                kept[tid] += 1
    check_hash(source, source_hash)
    check_hash(lineage, lineage_hash)
    return dict(source=str(Path(source).resolve()), source_kind=source_kind,
                source_sha256=source_hash, lineage=str(Path(lineage).resolve()),
                lineage_sha256=lineage_hash, species_root=species_root,
                sequences_in=sum(present.values()), sequences_out=sum(kept.values()),
                sequences_excluded=sum(present.values()) - sum(kept.values()),
                retained_taxids=sorted(kept), excluded_taxids=sorted(excluded),
                counts_by_taxid={str(t): {'in': n, 'out': kept[t]} for t, n in sorted(present.items())},
                filtered_fasta_sha256=sha256(output))


def prepare_lineage(source, source_kind, source_sha256, cache, out_dir, fetch_missing):
    check_hash(source, source_sha256)
    with work_directory(out_dir) as out:
        present = observed_taxids(source, source_kind)
        frozen = out / 'lineage.tsv'
        if cache:
            # Never give the existing user's cache to the mutating resolver.
            with open(cache, 'rb') as src, open(frozen, 'xb') as dst:
                shutil.copyfileobj(src, dst)
        if fetch_missing:
            taxonomy.fetch_lineage(present, str(frozen))
        if not frozen.exists():
            raise ValueError('provide --cache or explicitly --fetch-missing')
        frozen_lineage(frozen, present)
        check_hash(source, source_sha256)
        write_json(out / 'provenance.json', dict(status='candidate; review hash before search',
                   source=str(Path(source).resolve()), source_kind=source_kind,
                   source_sha256=source_sha256, lineage_sha256=sha256(frozen),
                   observed_taxids=dict(sorted(present.items())), fetched=fetch_missing,
                   taxonomy_helper_sha256=sha256(HELPER)))


def production_inputs(*, species, source_kind, source, source_sha256, query,
                      query_sha256, lineage, lineage_sha256, roles, roles_sha256):
    if species not in SPECIES:
        raise ValueError('only the nine primary training species are permitted')
    if source_kind not in SOURCE_NAMES or Path(source).name != SOURCE_NAMES[source_kind]:
        raise ValueError('source must be the reviewed Swiss FASTA or A19-prefiltered OrthoDB FASTA')
    query = Path(query)
    if query.parts[-4:] != ('est', species, 'univec', 'est.univec.fa.gz'):
        raise ValueError('only the primary A21-screened est/SPECIES/univec/est.univec.fa.gz is permitted')
    for path, expected in ((source, source_sha256), (query, query_sha256),
                           (lineage, lineage_sha256), (roles, roles_sha256)):
        check_hash(path, expected)
    with open(roles) as f:
        rows = [r for r in csv.DictReader(f, delimiter='\t') if r.get('dataset') == f'est/{species}']
    if len(rows) != 1 or rows[0].get('role') != 'c2_training_eligible' or rows[0].get('species') != species:
        raise ValueError(f'one training-eligible est/{species} role row required')
    for name in ('DONE', 'summary.json'):
        if not (query.parent / name).is_file():
            raise ValueError(f'missing A21 screening artefact: {query.parent / name}')
    summary = json.loads((query.parent / 'summary.json').read_text())
    if not isinstance(summary, dict):
        raise ValueError('A21 summary must be a JSON object')
    count = 0
    for _, seq in records(query):
        if len(seq) < 100 or not re.fullmatch('[ACGTRYSWKMBDHVNacgtryswkmbdhvn]+', seq):
            raise ValueError('primary screened query must be nucleotide FASTA, each record >=100nt')
        count += 1
    return {'queries': count, 'screening_summary_sha256': sha256(query.parent / 'summary.json'),
            'screening_done_sha256': sha256(query.parent / 'DONE')}


def search_command(diamond, query, db, output, threads, tmpdir, rich=False):
    if threads < 1:
        raise ValueError('threads must be positive')
    return [str(diamond), 'blastx', '--query', str(query), '--db', str(db),
            '--out', str(output), '--threads', str(threads), '--tmpdir', str(tmpdir),
            '--frameshift', '15', '--evalue', '1e-5', '--min-orf', '30',
            '--max-target-seqs', '25', '--max-hsps', '1', '--no-parse-seqids',
            '--outfmt', '6', *(RICH17 if rich else RAW13)]


def validate_raw(path, rich=False):
    rows = short = 0
    with open(path) as f:
        for number, line in enumerate(f, 1):
            c = line.rstrip('\n').split('\t')
            if len(c) != (17 if rich else 13) or any(not x for x in c):
                raise ValueError(f'malformed raw row {number}')
            qlen, slen = int(c[1]), int(c[3])
            qs, qe, ss, se, length = map(int, c[4:9])
            evalue, bitscore, frame = float(c[9]), float(c[10]), int(c[11])
            if not (1 <= qs <= qlen and 1 <= qe <= qlen and 1 <= ss <= se <= slen
                    and length > 0 and frame in (-3, -2, -1, 1, 2, 3)
                    and (qe >= qs) == (frame > 0) and math.isfinite(evalue)
                    and 0 <= evalue <= 1e-5 and math.isfinite(bitscore) and bitscore > 0):
                raise ValueError(f'invalid raw coordinates/score at row {number}')
            rows += 1
            short += length < 30
    return dict(raw_hsps=rows, hsps_lt_30aa=short,
                note='Raw endpoints are 1-based; >=30aa acceptance and phase projection are downstream, not applied here.')


def engine_version(diamond):
    diamond = Path(diamond)
    if not diamond.is_absolute() or not diamond.is_file():
        raise ValueError('provide an absolute real DIAMOND executable path')
    version = subprocess.check_output([str(diamond), 'version'], text=True).strip()
    if version != f'diamond version {VERSION}':
        raise ValueError(f'unreviewed DIAMOND version: {version!r}; require {VERSION}')
    return version


def execute(command, log):
    with open(log, 'x') as f:
        process = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            # Slurm TERM/INT must not leave DIAMOND or its descendants working.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise


def build_and_search(diamond, fasta, query, out, source_kind, threads, tmpdir, rich=False):
    """Low-level engine path shared with synthetic fixture, no production bypass CLI."""
    db = Path(out) / f'{source_kind}.dmnd'
    raw = Path(out) / f'{source_kind}.raw{17 if rich else 13}.tsv'
    if db.exists() or db.is_symlink() or raw.exists() or raw.is_symlink():
        raise FileExistsError('database or search output already exists')
    commands = [[str(diamond), 'makedb', '--in', str(fasta), '--db', str(db),
                 '--threads', str(threads), '--no-parse-seqids'],
                search_command(diamond, query, db, raw, threads, tmpdir, rich)]
    write_json(Path(out) / 'commands.json', commands)
    execute(commands[0], Path(out) / 'makedb.log')
    if not db.is_file() or db.stat().st_size == 0:
        raise ValueError('makedb did not create a nonempty database')
    execute(commands[1], Path(out) / 'blastx.log')
    return dict(db_sha256=sha256(db), raw_sha256=sha256(raw),
                fields=RICH17 if rich else RAW13, **validate_raw(raw, rich))


def run(args):
    if args.threads < 1:
        raise ValueError('threads must be positive')
    if not args.tmpdir.is_absolute() or not args.tmpdir.is_dir():
        raise ValueError('explicit existing absolute --tmpdir required')
    keys = 'species source_kind source source_sha256 query query_sha256 lineage lineage_sha256 roles roles_sha256'.split()
    inputs = {k: getattr(args, k) for k in keys}
    with work_directory(args.out_dir) as out:
        qc = production_inputs(**inputs)
        version = engine_version(args.diamond)
        start = time.monotonic()
        fasta = out / f'{args.source_kind}.filtered.fa'
        qc.update(filter_source(args.source, args.source_kind, args.lineage, SPECIES[args.species], fasta))
        with tempfile.TemporaryDirectory(prefix='est-blastx-', dir=args.tmpdir) as tmp:
            qc.update(build_and_search(args.diamond, fasta, args.query, out,
                                       args.source_kind, args.threads, tmp))
        for name in ('source', 'query', 'lineage', 'roles'):
            check_hash(inputs[name], inputs[name + '_sha256'])
        qc.update(inputs={k: str(v) for k, v in inputs.items()}, diamond_version=version,
                  diamond_sha256=sha256(args.diamond),
                  elapsed_seconds=time.monotonic() - start,
                  launcher_sha256=sha256(__file__), taxonomy_helper_sha256=sha256(HELPER))
        write_json(out / 'provenance.json', qc)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage', required=True)
    prepare = sub.add_parser('prepare-lineage')
    search = sub.add_parser('run')
    for parser in (prepare, search):
        parser.add_argument('--source-kind', choices=SOURCE_NAMES, required=True)
        parser.add_argument('--source', type=Path, required=True)
        parser.add_argument('--source-sha256', required=True)
        parser.add_argument('--out-dir', type=Path, required=True)
    prepare.add_argument('--cache', type=Path)
    prepare.add_argument('--fetch-missing', action='store_true')
    search.add_argument('--species', choices=SPECIES, required=True)
    for key in ('query', 'lineage', 'roles'):
        search.add_argument('--' + key, type=Path, required=True)
        search.add_argument('--' + key + '-sha256', required=True)
    search.add_argument('--diamond', type=Path, required=True)
    search.add_argument('--threads', type=int, required=True)
    search.add_argument('--tmpdir', type=Path, required=True)
    args = p.parse_args(argv)
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if args.stage == 'prepare-lineage':
        prepare_lineage(args.source, args.source_kind, args.source_sha256,
                        args.cache, args.out_dir, args.fetch_missing)
    else:
        run(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
