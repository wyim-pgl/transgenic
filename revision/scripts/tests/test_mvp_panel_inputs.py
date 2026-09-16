import hashlib
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'revision/scripts/mvp'))
sys.path.insert(0, str(ROOT / 'src'))
import mvp_prepare_panel_inputs as p
from transgenic.evidence.mvp_tracks import rasterise_est


@pytest.fixture
def db(tmp_path):
    path = tmp_path / 'evidence.db'
    with duckdb.connect(str(path)) as c:
        c.execute('CREATE TABLE est_molecule(molecule_id VARCHAR, strand VARCHAR, excluded BOOLEAN, cause VARCHAR, primary_count INT)')
        c.execute("INSERT INTO est_molecule VALUES ('held','+',true,'heldout_locus',1),('flank','-',false,NULL,1),('test','+',true,'test_block',1),('intron','-',true,'heldout_locus',1)")
        c.execute('CREATE TABLE est_member(member VARCHAR, molecule_id VARCHAR)')
        c.execute("INSERT INTO est_member VALUES ('read_a','held'),('read_b','held'),('read_c','intron')")
        c.execute('CREATE TABLE est_block(molecule_id VARCHAR,chrom VARCHAR,start0 INT,end0 INT)')
        c.execute("INSERT INTO est_block VALUES ('held','Chr1',10,14),('held','Chr1',12,16),('flank','Chr1',10,20),('test','Chr1',10,20),('intron','Chr1',0,1)")
        c.execute('CREATE TABLE est_junction(molecule_id VARCHAR,chrom VARCHAR,donor0 INT,acceptor0 INT)')
        c.execute("INSERT INTO est_junction VALUES ('held','Chr1',13,18),('held','Chr1',13,19),('intron','Chr1',17,11)")
        c.execute('CREATE TABLE tile_est(rn INT,molecule_id VARCHAR)')
    return path


def test_selection_union_n_and_intron_only(db):
    before = p.file_identity(db)
    with duckdb.connect(str(db), read_only=True) as c:
        tracks = p.panel_tracks(c, 'AANAAAAAAA', 'Chr1', 10)
        expected = np.zeros((10,8), np.float16)
        v = np.float16(np.log(2))
        expected[:,7] = 1
        expected[:6,0] = v
        expected[3,3] = v  # shared donor counts one molecule
        expected[[8,9],4] = v
        expected[7,5] = v
        expected[1,6] = v  # intron-only molecule retained
        expected[2] = 0
        np.testing.assert_array_equal(tracks, expected)
        provenance = list(p.selected_provenance(c, 'Chr1', 10, 20))
        assert [r['molecule']['molecule_id'] for r in provenance] == ['held','intron']
        assert [r['member'] for r in provenance[0]['members']] == ['read_a','read_b']
        assert provenance[1]['blocks'] == []
        assert len(provenance[1]['junctions']) == 1
        retained = rasterise_est(c, sequence='A'*10,chrom='Chr1',offset=10,L=10,is_rc=False,keep_fraction=1,aug_seed=1,population='retained')
        assert np.all(retained[:,1] == v)
        restricted = rasterise_est(c, sequence='A'*10,chrom='Chr1',offset=10,L=10,is_rc=False,keep_fraction=1,aug_seed=1,rn=1)
        assert not restricted[:,:7].any()
    assert p.file_identity(db) == before


def test_rc_small_handcomputed(db):
    with duckdb.connect(str(db), read_only=True) as c:
        a = rasterise_est(c,sequence='AAAAAAAANN',chrom='Chr1',offset=10,L=10,is_rc=True,keep_fraction=1,aug_seed=1,population='excluded_heldout')
    v = np.float16(np.log(2))
    assert a[6,5] == v and a[0,6] == v and a[1,6] == v
    assert a[2,3] == v
    assert np.all(a[4:8,1] == v)
    assert not a[8:].any()


def test_no_retained_flank_fallback(db):
    with duckdb.connect(str(db)) as c:
        c.execute("DELETE FROM est_molecule WHERE excluded")
    with duckdb.connect(str(db),read_only=True) as c:
        a = p.panel_tracks(c,'A'*10,'Chr1',10)
        assert not a[:,:7].any() and np.all(a[:,7] == 1)
        assert list(p.selected_provenance(c,'Chr1',10,20)) == []


def test_fresh_publication(tmp_path, db, monkeypatch):
    from types import SimpleNamespace
    request = tmp_path/'request.json'
    fasta = tmp_path/'ref.fa'
    fasta.write_text('>Chr1\n'+'A'*40000+'\n')
    request.write_text(json.dumps({'fasta':str(fasta)}))
    locus = SimpleNamespace(gene_id='gene',chromosome='Chr1',gene_start0=15000,gene_end0=16000,length=1000)
    monkeypatch.setattr(p, 'preflight', lambda payload: ([locus], {}, [p.file_identity(fasta),p.file_identity(db)], db))
    out = tmp_path/'out'
    result = p.prepare(request,out)
    assert result['training_eligible'] is False
    row = result['loci'][0]
    a = np.load(out/row['track']['path'],allow_pickle=False)
    assert a.dtype == np.float16 and a.shape == (30720,8)
    assert row['zero_biological_support']
    assert row['biological_nonzero_per_channel'] == [0]*7
    assert row['availability_nonzero'] == 30720
    assert (out/'COMPLETE').read_text().strip() == hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        p.prepare(request,out)


@pytest.mark.parametrize('phase', ['before_manifest', 'after_manifest'])
def test_input_mutation_no_complete(tmp_path, db, monkeypatch, phase):
    from types import SimpleNamespace
    fasta = tmp_path/'ref.fa'
    fasta.write_text('>Chr1\n'+'A'*40000+'\n')
    request = tmp_path/'request.json'
    request.write_text(json.dumps({'fasta':str(fasta)}))
    locus = SimpleNamespace(gene_id='gene',chromosome='Chr1',gene_start0=15000,gene_end0=16000,length=1000)
    monkeypatch.setattr(p,'preflight',lambda payload: ([locus],{},[p.file_identity(fasta),p.file_identity(db)],db))
    if phase == 'before_manifest':
        original = p.panel_tracks
        def mutate(*args):
            request.write_text('{}')
            return original(*args)
        monkeypatch.setattr(p,'panel_tracks',mutate)
    else:
        original = p.unchanged
        def mutate(identities):
            if any(Path(i['path']).name == 'manifest.json' for i in identities):
                request.write_text('{}')
            return original(identities)
        monkeypatch.setattr(p,'unchanged',mutate)
    with pytest.raises(ValueError,match='mutated'):
        p.prepare(request,tmp_path/'out')
    assert not (tmp_path/'out/COMPLETE').exists()


def test_all_frozen_50_loci(tmp_path, db, monkeypatch):
    loci = p.w.load_panel(ROOT/'revision/protocols/b6_panels_v1.tsv')
    request = tmp_path/'request.json'
    request.write_text(json.dumps({'fasta':'synthetic_stream_only'}))
    lengths = {}
    for locus in loci:
        lengths[locus.chromosome] = max(lengths.get(locus.chromosome,0), p.w.evaluation_window(locus)[1])
    monkeypatch.setattr(p,'preflight',lambda payload: (loci,{'synthetic_test_only':True},[p.file_identity(db)],db))
    monkeypatch.setattr(p,'fasta_sequences',lambda _: ((chrom,'A'*length) for chrom,length in lengths.items()))
    result = p.prepare(request,tmp_path/'out')
    assert len(result['loci']) == 50
    assert {row['gene_id'] for row in result['loci']} == {l.gene_id for l in loci}
    for locus,row in zip(loci,result['loci']):
        assert row['window'] == [locus.chromosome,*p.w.evaluation_window(locus)]
        for key in ('track','sequence','provenance'):
            actual = p.file_identity(tmp_path/'out'/row[key]['path'])
            assert actual['sha256'] == row[key]['sha256']
            assert actual['size'] == row[key]['bytes']


def test_seed_independence_failure(db, monkeypatch):
    original = p.mvp_tracks.rasterise_est
    def different(con, **kwargs):
        a = original(con,**kwargs)
        a[0,0] = kwargs['aug_seed'] - 20260914
        return a
    monkeypatch.setattr(p.mvp_tracks,'rasterise_est',different)
    with duckdb.connect(str(db),read_only=True) as c:
        with pytest.raises(ValueError,match='seed independence'):
            p.panel_tracks(c,'A'*10,'Chr1',10)


def test_frozen_reference_gate_rejects_synthetic(tmp_path):
    fasta = tmp_path/'ref.fa'
    fasta.write_text('>Chr1\nAAAA\n')
    gff = tmp_path/'ref.gff'
    gff.write_text('##gff-version 3\n')
    with pytest.raises(ValueError,match='frozen fasta identity mismatch'):
        p.frozen_reference(fasta,gff)


def test_preflight_rejects_unaccepted_database(tmp_path, db, monkeypatch):
    from transgenic.training import mvp_cli
    from types import SimpleNamespace
    panel = ROOT/'revision/protocols/b6_panels_v1.tsv'
    c = {key:str(db) for key in ('acceptance','inventory','receipts','corpus','evidence')}
    c['acceptance_sha256'] = '0'*64
    payload = dict(candidate=c,panels=str(panel),fasta=str(db),reference=str(db))
    monkeypatch.setattr(p,'frozen_reference',lambda *args: {})
    # Actual acceptance parser rejects the asserted digest before DB metadata queries.
    with pytest.raises(ValueError,match='SHA256'):
        p.preflight(payload)
    monkeypatch.setattr(mvp_cli,'validate_candidate_records',lambda *args: SimpleNamespace(acceptance={'inputs':{'corpus':{'sha256':'0'*64}},'artifact':{'sha256':'0'*64}}))
    with pytest.raises(ValueError,match='corpus SHA256 does not match'):
        p.preflight(payload)


def test_wal_rejected_before_output(tmp_path, db, monkeypatch):
    request = tmp_path/'request.json'
    request.write_text('{}')
    monkeypatch.setattr(p,'preflight',lambda payload: ([],{},[],db))
    Path(str(db)+'.wal').write_bytes(b'not a closed snapshot')
    with pytest.raises(ValueError,match='WAL'):
        p.prepare(request,tmp_path/'out')
    assert not (tmp_path/'out').exists()
