"""Local packaging tests, never actual accepted-corpus/50-locus certification."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import types
import pytest
import build_release as b

HERE=Path(__file__).resolve().parent
PACKAGE=HERE/'package'
SOURCE=PACKAGE/'source'
REPO=HERE.parents[3]
sys.path[:0]=[str(PACKAGE),str(SOURCE/'revision/scripts/mvp'),str(SOURCE/'src')]
job=b.load(PACKAGE/'run_panel_cpu.py')

def test_actual_staged_source_verification_and_exact_git_bytes():
    result=job.verify_source(SOURCE)
    assert result==json.loads((PACKAGE/'SOURCE_VERIFICATION.json').read_text())
    assert len(result['files'])==100
    for n,h in result['files'].items():
        assert (SOURCE/n).read_bytes()==b.authority.git(REPO,'show',b.COMMIT+':'+n)
    assert len(b.authority.verify(SOURCE)['objects'])==5
    assert not b.authority.git(REPO,'diff',b.OLD,b.COMMIT,'--','src','train','configs','revision/protocols','revision/scripts/mvp/real_probes.py','revision/scripts/mvp/run_bounded.py','revision/scripts/mvp/make_launch_readiness.py','revision/scripts/mvp/mvp_decode_runtime.py')

@pytest.mark.parametrize('commit,path',[(b.OLD,'notes/wait_work_20260916/panel_inputs/run_panel_cpu.py'),('9acd7dbb99304f193704ba6a174340cc540e4c68','tasknotes/reference_mapping/run_panel_cpu.py')])
def test_old_guards_reject_new_inventory(commit,path):
    if commit==b.OLD: raw=(HERE.parent/'final_release_bindfix1/run_panel_cpu.py').read_bytes()
    else: raw=b.authority.git(REPO,'show',commit+':'+path)
    module=types.ModuleType('old_guard'); exec(compile(raw,path,'exec'),module.__dict__)
    with pytest.raises(ValueError,match='inventory identity differs'): module.verify_source(SOURCE)

def test_only_three_literal_changes_and_archive_contents():
    template=b.authority.git(REPO,'show',b.COMMIT+':tasknotes/reference_mapping/run_panel_cpu.py').decode()
    for old,new in ((b.OLD,b.COMMIT),(b.OLD_COMPONENT,b.COMPONENT),(b.OLD_INVENTORY,b.sha(SOURCE/'source_pins.json'))): template=template.replace(old,new)
    assert template==(PACKAGE/'run_panel_cpu.py').read_text()
    old=HERE.parent/'final_release_bindfix1'
    for n in ('authority.py','cpu_site.tar','cpu_dependency.json','reference_authority.tar','panel_cpu.sbatch'):
        assert (PACKAGE/n).read_bytes()==(old/n).read_bytes()
    with tarfile.open(PACKAGE/'source.tar') as tar:
        members=tar.getmembers()
        for m in members:
            assert m.isfile() or m.isdir()
            assert not any(x in Path(m.name).parts for x in ('.venv','graft','cpu_site','site-packages','__pycache__'))
            if m.isfile(): assert tar.extractfile(m).read()==(PACKAGE/m.name).read_bytes()
    ledger=json.loads((PACKAGE/'transport.json').read_text())
    for n,h in ledger['files'].items(): assert b.sha(PACKAGE/n)==h
    assert ledger['expected_source_root']==b.REMOTE+'/source'
    assert ledger['decoder_code_sha256']==b.DECODER and ledger['evaluation_component_files']==86

def test_old_panel_publication_rejected(tmp_path,monkeypatch):
    import mvp_verify_panel_inputs as v
    oldhash=b.digest(b.authority.git(REPO,'show',b.OLD+':revision/scripts/mvp/mvp_produce_counts.py'))
    identity=v.p.file_identity(SOURCE/'revision/scripts/mvp/mvp_produce_counts.py')
    stale=dict(identity,sha256=oldhash)
    # Deliberately isolate the real consumer's code-inventory equality gate.
    monkeypatch.setattr(v.p,'preflight',lambda payload: ([],{},[identity],None))
    manifest=dict(schema_version=1,purpose='est_mvp_panel_inputs_not_training',training_eligible=False,
        orientation='genomic_forward',population='excluded_heldout',keep_fraction=1,source='EST',seeds=[1,2,3],
        aug_seeds=[20260915,20260916,20260917],seed_tracks_identical=True,
        off_policy='zeros_like(on); same sequence/window',zero_support_policy='report_not_fail',
        authority={},inputs=[stale,{}])
    b.dump(tmp_path/'manifest.json',manifest)
    (tmp_path/'COMPLETE').write_text(b.sha(tmp_path/'manifest.json')+'\n')
    with pytest.raises(ValueError,match='panel input/code inventory mismatch'):
        v.load_panel_inputs(tmp_path,{},[],{})

@pytest.fixture
def mapping_fixture(tmp_path):
    import duckdb
    import mvp_produce_counts as p
    panel=[p.w.PanelLocus('mvp_core',f'Ath{i:06d}','Chr1',40001,40500,'+',500) for i in range(1,51)]
    corpus=tmp_path/'synthetic.db'; panelpath=tmp_path/'panel.tsv'; panelpath.write_text('synthetic fixture only')
    fasta=tmp_path/'ref.fa'; fasta.write_text('>Chr1\n'+'A'*80000+'\n')
    gff=tmp_path/'ref.gff'; gff.write_text('##gff-version 3\n')
    authority={'SYNTHETIC_NOT_APPROVAL':True,'files':{'fasta':p.file_identity(fasta),'gff':p.file_identity(gff)}}
    with duckdb.connect(str(corpus)) as c:
        c.execute('CREATE TABLE gene_key_map(species_id VARCHAR,gene_id VARCHAR,gene_id_original VARCHAR,name_original VARCHAR,chromosome VARCHAR,start0 BIGINT,end0 BIGINT)')
        c.executemany('INSERT INTO gene_key_map VALUES (?,?,?,?,?,?,?)',[('Athaliana',l.gene_id,f'RAW_{i}.LONG_ID','name','Chr1',40000,40500) for i,l in enumerate(panel)])
        c.execute('CREATE TABLE build_manifest(species_id VARCHAR,fasta_sha256 VARCHAR,gff_sha256 VARCHAR)')
        c.execute('INSERT INTO build_manifest VALUES (?,?,?)',['Athaliana',p.w.sha256_file(fasta),p.w.sha256_file(gff)])
    def resolve(): return p.resolve_reference_mapping(panel,corpus,authority,{'corpus':p.w.sha256_file(corpus),'panel':p.w.sha256_file(panelpath)},panelpath)
    return corpus,resolve

def test_strict_map_receipt_fixture(mapping_fixture):
    import mvp_produce_counts as p
    corpus,resolve=mapping_fixture
    receipt=resolve()
    assert receipt['method']=='corpus_gene_key_map'
    assert len(receipt['rows'])==50
    assert set(receipt)=={'method','rows','sources','authority'}
    assert receipt['rows'][0]==dict(species_id='Athaliana',gene_id='Ath000001',gene_id_original='RAW_0.LONG_ID',name_original='name',chromosome='Chr1',start0=40000,end0=40500,strand='+')
    assert [r['gene_id'] for r in receipt['rows']]==[f'Ath{i:06d}' for i in range(1,51)]
    assert len(receipt['sources'])==8
    assert receipt['sources'][4]['sha256']=='6069fb5fe0771015d044d820a4cbbff313d9f3d2982991e1c58c17bc8aa726dc'
    assert receipt['authority']['SYNTHETIC_NOT_APPROVAL'] is True
    p.recheck_reference_mapping(receipt)

@pytest.mark.parametrize('sql',[
    "UPDATE gene_key_map SET gene_id_original=NULL WHERE gene_id='Ath000001'",
    "UPDATE gene_key_map SET gene_id_original='' WHERE gene_id='Ath000001'",
    "UPDATE gene_key_map SET start0=1 WHERE gene_id='Ath000001'",
    "UPDATE gene_key_map SET species_id='Wrong' WHERE gene_id='Ath000001'",
    "UPDATE gene_key_map SET gene_id_original='RAW_0.LONG_ID'",
    "UPDATE build_manifest SET gff_sha256='wrong'",
])
def test_strict_map_fields_rejected(mapping_fixture,sql):
    import duckdb
    corpus,resolve=mapping_fixture
    with duckdb.connect(str(corpus)) as c: c.execute(sql)
    with pytest.raises(ValueError): resolve()
