"""Synthetic corpus authority plus exact first complete gene from retained 128KiB prefix.
Not an actual 50-locus reference compatibility certification.
"""
from dataclasses import replace
from pathlib import Path
import sys
import importlib.util
from types import SimpleNamespace

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'revision/scripts/mvp'))
import mvp_produce_counts as p
from transgenic.utils import gsf_contract as gc


@pytest.fixture
def mapped(tmp_path):
    first = (ROOT/'tests/fixtures/mvp_reference_first_gene.gff3').read_text()
    gene = list(gc.parse_gff3(first.splitlines(), species_code=gc.species_code('Athaliana')))[0]
    assert gene.gene_id == 'Ath000001' and gene.gene_id_original == 'ATMG00010.TAIR10'
    assert gc.gene_key('Ath', 'ATMG00010.TAIR10', 'ATMG00010', 73) == 'Ath000073'
    assert gc.gene_key('Ath', 'shortID', 'otherName', 73) == 'shortID'
    panel = [p.w.PanelLocus('mvp_core', 'Ath000001', 'ChrM', 273, 734, '-', 462)]
    rows = [('Athaliana', 'Ath000001', 'ATMG00010.TAIR10', 'ATMG00010', 'ChrM', 272, 734)]
    text = first
    for i in range(2, 51):
        key = f'Ath{i:06d}'
        raw = key if i == 2 else f'RAW_{i}.LONG_REFERENCE_ID'
        panel.append(replace(panel[0], gene_id=key))
        rows.append(('Athaliana', key, raw, 'shortName', 'ChrM', 272, 734))
        text += first[first.index('ChrM'):].replace('ATMG00010.TAIR10', raw).replace('ATMG00010.1', f't{i}')
    reference = tmp_path/'ref.gff3'; reference.write_text(text)
    fasta = tmp_path/'ref.fa'; fasta.write_text('>ChrM\n'+'A'*1000+'\n')
    panel_file = tmp_path/'panel.tsv'; panel_file.write_text(repr(panel))
    corpus = tmp_path/'corpus.db'
    authority = {'SYNTHETIC_AUTHORITY_NOT_APPROVAL': True, 'files': {'fasta': p.file_identity(fasta), 'gff': p.file_identity(reference)}}
    with duckdb.connect(str(corpus)) as c:
        c.execute('CREATE TABLE gene_key_map(species_id VARCHAR,gene_id VARCHAR,gene_id_original VARCHAR,name_original VARCHAR,chromosome VARCHAR,start0 BIGINT,end0 BIGINT)')
        c.executemany('INSERT INTO gene_key_map VALUES (?,?,?,?,?,?,?)', rows)
        c.execute('CREATE TABLE build_manifest(species_id VARCHAR,fasta_sha256 VARCHAR,gff_sha256 VARCHAR)')
        c.execute('INSERT INTO build_manifest VALUES (?,?,?)', ['Athaliana', authority['files']['fasta']['sha256'], authority['files']['gff']['sha256']])
    def resolve():
        return p.resolve_reference_mapping(panel, corpus, authority,
            {'corpus': p.w.sha256_file(corpus), 'panel': p.w.sha256_file(panel_file)}, panel_file)
    return panel, corpus, reference, authority, resolve


def test_real_prefix_mapping_and_identity_case(mapped, tmp_path):
    panel, corpus, reference, authority, resolve = mapped
    before = reference.read_bytes()
    receipt = resolve()
    txs = p.prepare_mapped_reference(panel, reference, tmp_path/'subset.gff3', receipt)
    assert len(txs) == 50 and len(txs['Ath000001']) == 1
    assert receipt['method'] == 'corpus_gene_key_map'
    assert receipt['rows'][0]['gene_id_original'] == 'ATMG00010.TAIR10'
    assert receipt['rows'][1]['gene_id_original'] == 'Ath000002'
    assert reference.read_bytes() == before
    p.recheck_reference_mapping(receipt)
    corpus.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'): p.recheck_reference_mapping(receipt)


@pytest.mark.parametrize('sql', [
    "DELETE FROM gene_key_map WHERE gene_id='Ath000001'",
    "INSERT INTO gene_key_map SELECT * FROM gene_key_map WHERE gene_id='Ath000001'",
    "UPDATE gene_key_map SET species_id='Wrong' WHERE gene_id='Ath000001'",
    "UPDATE gene_key_map SET gene_id_original=NULL WHERE gene_id='Ath000001'",
    "UPDATE gene_key_map SET gene_id_original='' WHERE gene_id='Ath000001'",
    "UPDATE gene_key_map SET gene_id_original='Ath000002' WHERE gene_id='Ath000001'",
    "UPDATE gene_key_map SET start0=273 WHERE gene_id='Ath000001'",
    "UPDATE gene_key_map SET chromosome='Chr1' WHERE gene_id='Ath000001'",
    "UPDATE build_manifest SET fasta_sha256='wrong'",
    "UPDATE build_manifest SET gff_sha256='wrong'",
    "INSERT INTO build_manifest SELECT * FROM build_manifest",
])
def test_bad_authoritative_mapping_rejected(mapped, sql):
    _, corpus, _, _, resolve = mapped
    with duckdb.connect(str(corpus)) as c: c.execute(sql)
    with pytest.raises(ValueError): resolve()


@pytest.mark.parametrize('change', ['name', 'counter', 'strand', 'empty', 'duplicate'])
def test_no_alias_counter_geometry_or_empty_reference_fallback(mapped, tmp_path, change):
    panel, corpus, reference, authority, resolve = mapped
    if change in ('name', 'counter'):
        with duckdb.connect(str(corpus)) as c:
            c.execute("UPDATE gene_key_map SET gene_id_original=? WHERE gene_id='Ath000001'", ['ATMG00010' if change=='name' else 'Ath000001'])
    receipt = resolve()
    if change == 'strand': panel = [replace(panel[0], strand='+'), *panel[1:]]
    elif change == 'empty': reference.write_text('##gff-version 3\n')
    elif change == 'duplicate': reference.write_text(reference.read_text()+reference.read_text())
    with pytest.raises(ValueError): p.prepare_mapped_reference(panel, reference, tmp_path/'bad.gff3', receipt)
    empty = tmp_path/'empty.gff3'; empty.write_text('##gff-version 3\n')
    assert p.select_transcripts(p.read_gff(empty), panel[0]) == {}


@pytest.mark.parametrize('joint_failure', [False, True])
def test_cpu_wrapper_uses_shared_mapping_and_diagnostic_fallback(mapped, tmp_path, monkeypatch, joint_failure):
    panel, corpus, reference, authority, resolve = mapped
    from transgenic.training import mvp_cli, mvp_cli_runtime
    monkeypatch.setitem(sys.modules, 'authority', SimpleNamespace())  # packaging companion; not exercised
    spec = importlib.util.spec_from_file_location('reference_cpu_test', ROOT/'tasknotes/reference_mapping/run_panel_cpu.py')
    wrapper = importlib.util.module_from_spec(spec); spec.loader.exec_module(wrapper)
    panel_path = tmp_path/'panel.tsv'
    payload = dict(fasta=authority['files']['fasta']['path'], reference=str(reference), panels=str(panel_path),
        candidate=dict(corpus=str(corpus), acceptance='synthetic', acceptance_sha256='synthetic', inventory='synthetic', receipts='synthetic', evidence='synthetic'))
    monkeypatch.setattr(p, 'frozen_reference', lambda *a: authority)
    monkeypatch.setattr(p.w, 'load_panel', lambda *a: panel)
    monkeypatch.setattr(p, 'fasta_lengths', lambda *a: {'ChrM': 100000})
    # The real mitochondrial first gene is not an eligible nuclear MVP window.
    # This test exercises mapping/fallback only, not window eligibility.
    monkeypatch.setattr(p.w, 'evaluation_window', lambda *a: (0, 30720))
    monkeypatch.setattr(mvp_cli, 'validate_candidate_records', lambda *a: object())
    monkeypatch.setattr(mvp_cli_runtime, 'verify_candidate_artifacts', lambda *a: {'sha256':dict(corpus=p.w.sha256_file(corpus),panel=p.w.sha256_file(panel_path))})
    original = p.prepare_mapped_reference
    calls = []
    def preparation(loci, reference, restricted, receipt):
        calls.append([l.gene_id for l in loci])
        if joint_failure and len(loci) == 50: raise ValueError('explicit joint diagnostic failure fixture')
        return original(loci, reference, restricted, receipt)
    monkeypatch.setattr(p, 'prepare_mapped_reference', preparation)
    out = tmp_path/'cpu'; out.mkdir()
    report = wrapper.reference_audit(payload, out)
    assert report['compatible_locus_count'] == 50  # synthetic panel only
    assert report['panel_strict_parser']['ok'] is (not joint_failure)
    assert len(calls) == (51 if joint_failure else 1)
    assert report['loci'][0]['gene_id_original'] == 'ATMG00010.TAIR10'
    assert report['reference_mapping'] == resolve()


def test_accepted_hashes_and_exact_panel_gate(mapped):
    panel, corpus, reference, authority, resolve = mapped
    with pytest.raises(ValueError, match='hash mismatch'):
        p.resolve_reference_mapping(panel, corpus, authority, {'corpus':'0'*64,'panel':'0'*64}, corpus)
    with pytest.raises(ValueError, match='hash mismatch'):
        p.resolve_reference_mapping(panel, corpus, authority, {'corpus':p.w.sha256_file(corpus),'panel':'0'*64}, corpus)
    reference.write_text('##gff-version 3\n')
    with pytest.raises(ValueError, match='hash mismatch'): resolve()
    with pytest.raises(ValueError, match='50'):
        p.resolve_reference_mapping(panel[:-1], corpus, authority, {}, corpus)
