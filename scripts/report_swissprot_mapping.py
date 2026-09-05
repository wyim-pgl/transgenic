#!/usr/bin/env python3
"""Export A30 supplementary accounting from existing inputs; never regenerate flags.

Summary hard_flags counts caution events before deduplication. Flag TSV rows and
unique flagged genes are separate units. The supplied per-entry masked-gene audit
is copied byte-for-byte and need not cover every species in the flag files.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def export(summary_path, flags_dir, freeze_path, masked_path, out_dir):
    summary_path, flags_dir, freeze_path, masked_path, out_dir = map(
        Path, (summary_path, flags_dir, freeze_path, masked_path, out_dir))
    summary = json.loads(summary_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    rows, provenance = [], {}
    with masked_path.open() as fh:
        audit = list(csv.DictReader(fh, delimiter='\t'))
    for sp, values in sorted(summary['species'].items()):
        path = flags_dir / f'{sp}.swissprot_flags.tsv'
        data = path.read_bytes()
        digest = hashlib.md5(data).hexdigest()
        if digest != freeze['qc_flag_files'][path.name]:
            raise ValueError(f'{sp}: flags differ from frozen input MD5')
        provenance[path.name] = {'md5': digest, 'sha256': hashlib.sha256(data).hexdigest()}
        with path.open() as fh:
            flags = list(csv.DictReader(fh, delimiter='\t'))
        hard = [r for r in flags if r['flag'].startswith('swissprot_caution_')]
        detailed = [r for r in audit if r['species_id'] == sp]
        if not {r['gene_id'] for r in detailed} <= {r['gene_id'] for r in hard}:
            raise ValueError(f'{sp}: audit gene absent from frozen hard flags')
        row = {'species_id': sp, **{k: values.get(k, 0) for k in
            ('entries', 'mapped', 'mapped_by_sequence', 'unmapped', 'ambiguous', 'resolved_in_reference')}}
        if row['mapped'] + row['unmapped'] + row['ambiguous'] != row['entries']:
            raise ValueError(f'{sp}: mapping partition does not sum to entries')
        row.update(hard_caution_events=values.get('hard_flags', 0), hard_flag_rows=len(hard),
                   hard_flag_genes=len({r['gene_id'] for r in hard}),
                   detailed_audit_rows=len(detailed), detailed_audit_genes=len({r['gene_id'] for r in detailed}),
                   id_map_aliases=values.get('aliases_added', {}).get('id_map', 0),
                   note=('No RAP-MSU identifier map; all mapped entries are exact-sequence matches; zero hard flags is not absence of annotation errors.' if sp == 'Osativa' else ''))
        rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'TableS_swissprot_mapping.csv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shutil.copyfile(masked_path, out_dir / 'TableS_swissprot_hard_masked_genes.tsv')
    provenance.update(summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                      masked_audit_sha256=hashlib.sha256(masked_path.read_bytes()).hexdigest(),
                      freeze_manifest_sha256=hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
                      source_summary=str(summary_path), source_masked_audit=str(masked_path),
                      source_flags=str(flags_dir),
                      scope='Reporting only. Flag MD5s verified against frozen manifest; no database opened or input changed.')
    (out_dir / 'TableS_swissprot_provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True)+'\n')
    return {'species': len(rows), 'hard_flag_genes': sum(r['hard_flag_genes'] for r in rows),
            'detailed_audit_rows': len(audit), 'detailed_audit_genes': len({(r['species_id'],r['gene_id']) for r in audit}),
            'frozen_flag_hashes_verified': len(rows)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', required=True)
    p.add_argument('--flags-dir', required=True)
    p.add_argument('--freeze', required=True)
    p.add_argument('--masked-audit', required=True)
    p.add_argument('--out-dir', required=True)
    a = p.parse_args()
    print(json.dumps(export(a.summary, a.flags_dir, a.freeze, a.masked_audit, a.out_dir), sort_keys=True))


if __name__ == '__main__':
    main()
