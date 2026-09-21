"""Tiny synthetic DB only; never evidence for the accepted actual50 publication."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import duckdb
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'revision/scripts/mvp'), str(ROOT/'src')]
import mvp_prepare_panel_inputs as p
import mvp_verify_panel_inputs as v


def test_explicit_profile_requires_independent_receipt(tmp_path):
    with pytest.raises(ValueError, match='panel_compatibility'):
        v.load_panel_inputs(tmp_path, {'profile':'development-single'}, [], {})


@pytest.fixture
def example(tmp_path, monkeypatch):
    import mvp_panel_compatibility as c
    monkeypatch.setenv('SLURM_JOB_ID','synthetic-test-only')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    db = tmp_path/'evidence.db'
    with duckdb.connect(str(db)) as con:
        con.execute('CREATE TABLE est_molecule(molecule_id VARCHAR, strand VARCHAR, excluded BOOLEAN, cause VARCHAR, primary_count INT)')
        con.execute("INSERT INTO est_molecule VALUES ('held','+',true,'heldout_locus',1),('flank','-',false,NULL,1)")
        con.execute('CREATE TABLE est_member(member VARCHAR, molecule_id VARCHAR)')
        con.execute("INSERT INTO est_member VALUES ('a','held'),('b','held')")
        con.execute('CREATE TABLE est_block(molecule_id VARCHAR,chrom VARCHAR,start0 INT,end0 INT)')
        con.execute("INSERT INTO est_block VALUES ('held','Chr1',1000,1100),('flank','Chr1',1000,1200)")
        con.execute('CREATE TABLE est_junction(molecule_id VARCHAR,chrom VARCHAR,donor0 INT,acceptor0 INT)')
        con.execute("INSERT INTO est_junction VALUES ('held','Chr1',1100,1200)")
    fasta = tmp_path/'ref.fa'
    fasta.write_text('>Chr1\n'+'A'*40000+'\n')
    payload = {'profile':'development-single', 'fasta':str(fasta), 'candidate':{}}
    for name in ('panels','reference','acceptance','inventory','receipts','corpus'):
        path = tmp_path/(name+'.txt')
        path.write_text(name)
        (payload if name in ('panels','reference') else payload['candidate'])[name] = str(path)
    payload['candidate']['evidence'] = str(db)
    payload['candidate']['acceptance_sha256'] = p.file_identity(payload['candidate']['acceptance'])['sha256']
    loci = [SimpleNamespace(gene_id=f'g{i}',chromosome='Chr1',gene_start0=15000,gene_end0=16000,length=1000) for i in range(50)]
    data = [p.file_identity(payload[k]) for k in ('panels','fasta','reference')]
    data += [p.file_identity(payload['candidate'][k]) for k in ('acceptance','inventory','receipts','corpus','evidence')]
    historical = []
    for i, name in enumerate(('mvp_prepare_panel_inputs.py','mvp_tracks.py','mvp_windows.py','mvp_cli.py','mvp_cli_runtime.py','mvp_produce_counts.py')):
        path = tmp_path/'old'/name
        path.parent.mkdir(exist_ok=True)
        path.write_text(f'historical source {i}')
        historical.append(p.file_identity(path))
    authority = {'synthetic_test_only':True}
    old = copy.deepcopy(payload)
    old.pop('profile')
    request = tmp_path/'old-request.json'
    request.write_text(p.canonical(old))
    monkeypatch.setattr(p,'preflight',lambda _: (loci,authority,data+historical,db))
    root = tmp_path/'panel'
    p.prepare(request,root)
    manifest = p.read_json(root/'manifest.json')
    manifest['rasterizer'] = historical[1]
    (root/'manifest.json').write_text(p.canonical(manifest))
    (root/'COMPLETE').write_text(p.file_identity(root/'manifest.json')['sha256']+'\n')
    # Isolated trust seam: patch private production constants, no request/CLI relaxation.
    monkeypatch.setattr(c,'_ACCEPTED_MANIFEST_SHA256',p.file_identity(root/'manifest.json')['sha256'])
    monkeypatch.setattr(c,'_ACCEPTED_PANEL_ROOT',root)
    from transgenic.training import mvp_cli, mvp_cli_runtime
    import mvp_produce_counts
    current = [p.file_identity(m.__file__) for m in (p,p.mvp_tracks,p.w,mvp_cli,mvp_cli_runtime,mvp_produce_counts)]
    monkeypatch.setattr(p,'preflight',lambda _: (loci,authority,data+current,db))
    payload['panel_inputs'] = str(root)
    return SimpleNamespace(c=c,root=root,payload=payload,loci=loci,db=db,historical=historical,tmp=tmp_path)


def publish(e):
    request = e.tmp/'current-request.json'
    request.write_text(p.canonical(e.payload))
    receipt = e.tmp/'compat.json'
    report = e.c.verify_and_publish(request, receipt)
    e.payload['panel_compatibility'] = p.file_identity(receipt)
    return receipt, report


def consume(e):
    sequences = {l.gene_id:'A'*30720 for l in e.loci}
    return v.load_panel_inputs(e.root,e.payload,e.loci,sequences)


def test_real_tiny_raster_compatibility_and_no_clobber(example):
    e = example
    before = p.file_identity(e.root/'manifest.json')
    receipt, report = publish(e)
    tracks, records = consume(e)
    assert len(tracks) == len(report['loci']) == 50
    assert report['seed'] == 123 and report['aug_seed'] == 20261037
    assert report['loci'][0]['selected_member_count'] == 2
    assert report['loci'][0]['seed123_exact_equal'] is True
    assert p.file_identity(e.root/'manifest.json') == before
    assert p.file_identity(receipt) in records
    with pytest.raises(FileExistsError):
        e.c.verify_and_publish(e.tmp/'current-request.json',receipt)


@pytest.mark.parametrize('target', ['historical','array','manifest'])
def test_mutated_trusted_bytes_fail(example, target):
    e = example
    publish(e)
    path = {'historical':Path(e.historical[0]['path']), 'array':e.root/'00.npy','manifest':e.root/'manifest.json'}[target]
    with path.open('ab') as handle:
        handle.write(b' ')
    with pytest.raises(ValueError):
        consume(e)


def test_forged_producer_with_self_consistent_markers_rejected(example):
    e = example
    manifest = p.read_json(e.root/'manifest.json')
    source = Path(e.historical[0]['path'])
    source.write_text('arbitrary other producer')
    manifest['inputs'][8] = p.file_identity(source)
    (e.root/'manifest.json').write_text(p.canonical(manifest))
    (e.root/'COMPLETE').write_text(p.file_identity(e.root/'manifest.json')['sha256']+'\n')
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        publish(e)
    assert not (e.tmp/'compat.json').exists()


def test_stale_consumer_receipt_fails(example):
    e = example
    receipt, report = publish(e)
    report['consumer_closure'][0]['sha256'] = '0'*64
    receipt.write_text(p.canonical(report))
    e.payload['panel_compatibility'] = p.file_identity(receipt)
    with pytest.raises(ValueError,match='consumer closure'):
        consume(e)


@pytest.mark.parametrize('change',['raster','provenance'])
def test_live_seed123_and_selection_must_match(example,monkeypatch,change):
    e = example
    publish(e)
    if change == 'raster':
        original = p.mvp_tracks.rasterise_est
        def different(con, **kwargs):
            a = original(con,**kwargs)
            if kwargs['aug_seed'] == 20261037:
                a[0,0] = 1
            return a
        monkeypatch.setattr(p.mvp_tracks,'rasterise_est',different)
    else:
        monkeypatch.setattr(p,'selected_provenance',lambda *args: iter([]))
    with pytest.raises(ValueError,match='rasterization|provenance'):
        consume(e)


@pytest.mark.parametrize('change',['wal','source'])
def test_late_mutation_after_closed_pending_no_publication(example,monkeypatch,change):
    e = example
    original = e.c.producer.stage_json
    def late(path, report):
        pending = original(path,report)
        if change == 'wal':
            Path(str(e.db)+'.wal').write_text('late')
        else:
            Path(e.historical[0]['path']).write_text('late')
        return pending
    monkeypatch.setattr(e.c.producer,'stage_json',late)
    with pytest.raises(ValueError,match='WAL|hash mismatch'):
        publish(e)
    assert not (e.tmp/'compat.json').exists()
    assert (e.tmp/'compat.json.pending').exists()


def test_api_requires_compute_allocation(example,monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID')
    with pytest.raises(ValueError,match='CPU compute allocation'):
        publish(example)


def test_missing_historical_source_stops(example):
    e = example
    Path(e.historical[0]['path']).unlink()
    with pytest.raises(FileNotFoundError):
        publish(e)
    assert not (e.tmp/'compat.json').exists()


def test_strict_current_producer_still_passes(example):
    e = example
    strict_root = e.tmp/'strict-panel'
    p.prepare(e.tmp/'old-request.json',strict_root)
    e.payload.pop('profile')
    tracks, _ = v.load_panel_inputs(strict_root,e.payload,e.loci,{l.gene_id:'A'*30720 for l in e.loci})
    assert len(tracks) == 50


def test_strict_branch_still_rejects_old_producer(example):
    e = example
    e.payload.pop('profile')
    with pytest.raises(ValueError,match='input/code inventory'):
        consume(e)


@pytest.mark.parametrize('kind',['relative','symlink'])
def test_receipt_path_strict(example,kind):
    e = example
    receipt, _ = publish(e)
    if kind == 'relative':
        e.payload['panel_compatibility']['path'] = 'compat.json'
    else:
        link = e.tmp/'link.json'
        link.symlink_to(receipt)
        e.payload['panel_compatibility']['path'] = str(link)
    with pytest.raises(ValueError,match='canonical'):
        consume(e)
