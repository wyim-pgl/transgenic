import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('report_swissprot', Path(__file__).resolve().parents[1] / 'scripts/report_swissprot_mapping.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_mapping_units_copy_and_frozen_hash_guard(tmp_path):
    summary = tmp_path / 'summary.json'
    summary.write_text(json.dumps({'species': {'Osativa': {'entries': 5, 'mapped': 3, 'mapped_by_sequence': 3, 'unmapped': 2, 'hard_flags': 0}}}))
    flags = tmp_path / 'Osativa.swissprot_flags.tsv'
    flags.write_text('species_id\tgene_id\tflag\n')
    freeze = tmp_path / 'freeze.json'
    freeze.write_text(json.dumps({'qc_flag_files': {flags.name: hashlib.md5(flags.read_bytes()).hexdigest()}}))
    audit = tmp_path / 'audit.tsv'
    audit.write_text('species_id\tgene_id\tflag\n')
    out = tmp_path / 'out'
    r = report.export(summary, tmp_path, freeze, audit, out)
    assert r['frozen_flag_hashes_verified'] == 1
    assert (out / 'TableS_swissprot_hard_masked_genes.tsv').read_bytes() == audit.read_bytes()
    with (out / 'TableS_swissprot_mapping.csv').open() as fh:
        row = next(csv.DictReader(fh))
    assert row['hard_flag_genes'] == '0' and row['mapped_by_sequence'] == '3'
    assert 'No RAP-MSU' in row['note']
    flags.write_text(flags.read_text() + 'Osativa\tg\tswissprot_caution_frameshift\n')
    with pytest.raises(ValueError, match='frozen input MD5'):
        report.export(summary, tmp_path, freeze, audit, out)
