#!/usr/bin/env python3
"""Run list for b6_audit_align.sbatch (A46 rule 4(b) audit alignment; SPEC-1 step 1).

One row per primary alignment unit, in array-task order:
  * EST     -- the nine training species, A37 primary arm (evidence/est_align/<sp>),
  * ONT     -- every run of evidence/longread_runs_v1.tsv with its section 3.3 item 2
               orientation verdict (PASS -> -uf, FAIL/UNRESOLVED -> none),
  * PacBio  -- every run of evidence/longread_runs_pacbio_v1.tsv (never -uf, A47.3).
The verdict is read from orientation_audit_v1/<sp>/<dataset>/<run>/audit.json and must
agree with the audit DONE marker and with the uf= the PRIMARY alignment recorded;
any disagreement refuses the whole list. Nothing is aligned or submitted here; the
sbatch re-checks every gate at run time.

Columns: index source species dataset run reads uf primary_dir (tab-separated,
no header; reads and primary_dir relative to --root).

Output layout (what b6_audit_align.sbatch writes and the C0 builder reads), per task:
  evidence/audit_align_v1/<species>/<dataset>/<run>/<run>.audit.bam
EST rows use dataset `est` and run `est_primary`, i.e.
  evidence/audit_align_v1/<sp>/est/est_primary/est_primary.audit.bam
`--list` runs every run-list check and prints `index source species <bam path>` for all
tasks (relative to --root) without writing anything.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SPECIES = ('Athaliana', 'Bdistachyon', 'Gmax', 'Osativa', 'Ppatens', 'Ptrichocarpa',
           'Sbicolor', 'Sitalica', 'Vvinifera')
DEFAULT_ROOT = '/data/gpfs/assoc/pgl/data/Transgenic'
EXPECTED = dict(est=9, ont=129, pacbio=49)


def kv(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        key, sep, value = line.partition('=')
        if sep:
            out[key] = value
    return out


def est_rows(root):
    rows = []
    for sp in SPECIES:
        primary = Path('evidence/est_align') / sp
        done = kv(root / primary / 'DONE')
        if done.get('arm') != 'primary' or not done.get('est_md5'):
            raise ValueError(f'{sp}: est_align DONE is not a primary-arm marker with est_md5')
        rows.append(('est', sp, 'est', 'est_primary',
                     f'evidence/est/{sp}/univec/est.univec.fa.gz', 'none', str(primary)))
    return rows


def ont_verdict(root, sp, dsbase, run):
    audit = root / 'evidence/orientation_audit_v1' / sp / dsbase / run
    status = json.loads((audit / 'audit.json').read_text())
    marker = kv(audit / 'DONE')
    verdict = {('PASS', True): '-uf', ('FAIL', False): 'none', ('UNRESOLVED', False): 'none'}.get(
        (status.get('status'), status.get('use_uf')))
    if verdict is None or marker.get('status') != status['status']:
        raise ValueError(f'{run}: inconsistent orientation verdict {status.get("status")}/'
                         f'{status.get("use_uf")} vs DONE {marker.get("status")}')
    return verdict


def longread_rows(root, source, runs_tsv):
    rows = []
    for line in (root / runs_tsv).read_text().splitlines():
        if not line.strip():
            continue
        sp, ds, run, reads = line.split('\t')
        dsbase = Path(ds).name
        primary = Path('evidence/longread_align_v1') / sp / dsbase / run
        recorded = kv(root / primary / 'DONE')
        uf = ont_verdict(root, sp, dsbase, run) if source == 'ont' else 'none'
        if source == 'ont' and recorded.get('uf') != uf:
            raise ValueError(f'{run}: primary alignment used uf={recorded.get("uf")}, verdict says {uf}')
        if source == 'pacbio' and recorded.get('preset') != 'splice:hq':
            raise ValueError(f'{run}: primary PacBio DONE is not the splice:hq preset')
        rows.append((source, sp, dsbase, run, reads, uf, str(primary)))
    return rows


def build(root, expected=EXPECTED):
    root = Path(root)
    rows = (est_rows(root) + longread_rows(root, 'ont', 'evidence/longread_runs_v1.tsv')
            + longread_rows(root, 'pacbio', 'evidence/longread_runs_pacbio_v1.tsv'))
    counts = {s: sum(r[0] == s for r in rows) for s in expected}
    if counts != expected:
        raise ValueError(f'run counts {counts} != expected {expected}')
    keys = [(r[1], r[2], r[3]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate species/dataset/run output directory')
    return [(i,) + r for i, r in enumerate(rows)]


OUT_ROOT = 'evidence/audit_align_v1'


def audit_bam(row):
    """Relative audit BAM path for one run-list row (index-prefixed tuple from build())."""
    _, _, sp, dsbase, run = row[:5]
    return f'{OUT_ROOT}/{sp}/{dsbase}/{run}/{run}.audit.bam'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=Path(DEFAULT_ROOT))
    p.add_argument('--out', type=Path, help='run list to write (required unless --list)')
    p.add_argument('--list', action='store_true',
                   help='dry run: print every task\'s audit BAM path, write nothing')
    p.add_argument('--expect', default='9,129,49', help='est,ont,pacbio run counts')
    args = p.parse_args(argv)
    expected = dict(zip(('est', 'ont', 'pacbio'), map(int, args.expect.split(','))))
    rows = build(args.root, expected)
    if args.list:
        for row in rows:
            print(f'{row[0]}\t{row[1]}\t{row[2]}\t{audit_bam(row)}')
        print(f'{len(rows)} tasks', file=sys.stderr)
        return 0
    if args.out is None:
        p.error('--out is required unless --list')
    with open(args.out, 'x') as f:
        f.writelines('\t'.join(map(str, r)) + '\n' for r in rows)
    print(f'{len(rows)} tasks -> {args.out}; submit with --array=0-{len(rows) - 1}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
