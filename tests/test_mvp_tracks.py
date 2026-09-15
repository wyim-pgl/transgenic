"""CPU-only, hand-checked nucleotide identities, not just channel totals."""
import hashlib
import importlib.util
from pathlib import Path
import time

import duckdb
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TRACKS = ROOT / 'src/transgenic/evidence/mvp_tracks.py'


def raster():
    assert TRACKS.exists(), 'eight-channel MVP rasterizer is not implemented'
    spec = importlib.util.spec_from_file_location('mvp_tracks',TRACKS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.rasterise_est


def database():
    con = duckdb.connect(':memory:',config={'threads':1})
    con.execute('CREATE TABLE est_molecule(molecule_id VARCHAR, strand VARCHAR, excluded BOOLEAN, cause VARCHAR, primary_count BIGINT)')
    con.execute('CREATE TABLE est_block(molecule_id VARCHAR, chrom VARCHAR, start0 BIGINT, end0 BIGINT)')
    con.execute('CREATE TABLE est_junction(molecule_id VARCHAR, chrom VARCHAR, intron_start0 BIGINT, intron_end0 BIGINT, tx_strand VARCHAR, donor0 BIGINT, acceptor0 BIGINT)')
    con.execute('CREATE TABLE tile_est(rn BIGINT,molecule_id VARCHAR)')
    return con


def add(con, mid='m', strand='-', blocks=((1050,1100),(1200,1250)), junctions=((1100,1199),), cause=None):
    con.execute('INSERT INTO est_molecule VALUES (?,?,?,?,1)',[mid,strand,cause is not None,cause])
    for a,b in blocks:
        con.execute('INSERT INTO est_block VALUES (?,?,?,?)',[mid,'Chr1',a,b])
    for a,b in junctions:
        donor, acceptor = (a,b) if strand=='+' else (b,a) if strand=='-' else (None,None)
        con.execute('INSERT INTO est_junction VALUES (?,?,?,?,?,?,?)',[mid,'Chr1',a,b,strand,donor,acceptor])
    con.execute('INSERT INTO tile_est VALUES (7,?)',[mid])


def tracks(con, **kwargs):
    defaults = dict(sequence='A'*300,chrom='Chr1',offset=1000,L=300,is_rc=False,keep_fraction=1,aug_seed=20260915)
    defaults.update(kwargs)
    return raster()(con,**defaults)


@pytest.mark.parametrize('rn', [None,7])
def test_minus_junction_identity_at_nonzero_offset_and_rc(rn):
    con = database()
    add(con)
    t = tracks(con,rn=rn)
    one = np.float16(np.log1p(1))
    assert t.shape == (300,8) and t.dtype == np.float16
    assert t[199,5] == one and t[100,6] == one
    assert t[199,6] == t[100,5] == 0
    assert np.all(t[50:100,1] == one) and t[100,1] == 0 and t[200,1] == one
    r = tracks(con,rn=rn,is_rc=True)
    assert r[100,3] == one and r[199,4] == one
    assert r[100,4] == r[199,3] == 0
    np.testing.assert_array_equal(r,t[::-1][:,[1,0,2,5,6,3,4,7]])


def test_junction_query_is_not_restricted_to_overlapping_blocks():
    con = database()
    add(con)
    # Window is intron-only; donor/acceptor must still rasterize in explicit-window mode.
    t = tracks(con,sequence='A'*100,L=100,offset=1100)
    assert not t[:,:3].any()
    assert t[99,5] == t[0,6] == np.float16(np.log(2))


def test_distinct_molecules_not_block_or_junction_rows_and_coverage_clipping():
    con = database()
    add(con,blocks=((1050,1100),(1070,1100),(1050,1100)),junctions=((1100,1199),(1100,1199),(1100,1209)))
    con.execute("INSERT INTO tile_est VALUES (7,'m')")
    t = tracks(con,rn=7)
    assert np.all(t[50:100,1] == np.float16(np.log(2)))
    assert t[100,6] == np.float16(np.log(2))  # one molecule at shared acceptor
    for i in range(1,70):
        add(con,mid=str(i),blocks=((1050,1100),),junctions=((1100,1199),))
    t = tracks(con)
    assert t[50,1] == np.float16(np.log1p(64))
    assert t[100,6] == np.float16(np.log1p(70))  # NO junction clipping


def test_unknown_and_conflicting_coverage_never_directional_junctions():
    con = database()
    add(con,strand='?')
    t = tracks(con)
    assert t[50,2] == np.float16(np.log(2)) and not t[:,3:7].any()


@pytest.mark.parametrize('rc', [False,True])
def test_source_empty_dropped_fraction_zero_and_output_oriented_mask(rc):
    con = database()
    seq = 'N' + 'A'*98 + 'n' + 'A'*200
    empty = tracks(con,sequence=seq,is_rc=rc)
    assert empty[:,7].sum() == 298 and not empty[:,:7].any()
    add(con)
    on = tracks(con,sequence=seq,is_rc=rc)
    assert not on[[0,99]].any() and on[299,7] == 1
    assert not tracks(con,sequence=seq,is_rc=rc,source_supplied=False).any()
    assert not tracks(con,sequence=seq,is_rc=rc,keep_fraction=0,population='excluded_heldout').any()
    assert not tracks(con,keep_fraction=0).any()


def test_retained_heldout_and_no_primary_populations_do_not_mix():
    con = database()
    add(con,'ret','+',blocks=((1000,1010),),junctions=())
    add(con,'held','-',blocks=((1010,1020),),junctions=(),cause='heldout_locus')
    add(con,'test','+',blocks=((1020,1030),),junctions=(),cause='test_block')
    add(con,'supp','+',blocks=((1030,1040),),junctions=(),cause='supplementary_only')
    con.execute("UPDATE est_molecule SET primary_count=0 WHERE molecule_id='supp'")
    on = tracks(con)
    held = tracks(con,population='excluded_heldout')
    assert on[0,0] > 0 and not on[10:,:7].any()
    assert held[10,1] > 0 and not held[:10,:7].any() and not held[20:,:7].any()
    assert not tracks(con,rn=999)[:,:7].any()
    assert not tracks(con,chrom='Chr2')[:,:7].any()


def test_exact_hash_nested_order_and_tile_stability():
    con = database()
    for i in reversed(range(60)):
        add(con,str(i),'+',blocks=((1000+i,1001+i),),junctions=())
    seed = 20260915
    expected = np.array([int.from_bytes(hashlib.blake2b(f'{seed}\tAthaliana\t{i}'.encode(),digest_size=8).digest(),'big')/2**64 for i in range(60)])
    prior = set()
    for fraction in (0.01,0.1,0.5,1):
        t = tracks(con,keep_fraction=fraction,aug_seed=seed)
        kept = set(np.flatnonzero(t[:60,0]))
        assert kept == set(np.flatnonzero(expected<fraction))
        assert prior <= kept
        prior = kept
        shifted = tracks(con,sequence='A'*100,L=100,offset=990,keep_fraction=fraction,aug_seed=seed,rn=7)
        np.testing.assert_array_equal(t[:60,0],shifted[10:70,0])
    # Exact threshold is strict '<', not '<='.
    rank = expected[0]
    assert tracks(con,keep_fraction=rank)[0,0] == 0
    assert tracks(con,keep_fraction=np.nextafter(rank,1))[0,0] > 0


@pytest.mark.parametrize('kwargs', [dict(L=299),dict(keep_fraction=-.1),dict(keep_fraction=1.1),
                                  dict(keep_fraction=float('nan')),dict(population='all'),dict(offset=-1)])
def test_bad_interface_arguments_fail_closed(kwargs):
    with pytest.raises(ValueError):
        tracks(database(),**kwargs)


def test_real_bam_to_tracks_and_129024_cpu_smoke(tmp_path):
    spec = importlib.util.spec_from_file_location('builder_tests',ROOT/'revision/scripts/tests/test_mvp_c0lite.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    rows = [fixture.record('c_F',500,'50M100N50M',0,'-'),
            fixture.record('c_R',510,'40M100N60M',16,'+')]
    con, qc = fixture.build(tmp_path,rows)
    start = time.perf_counter()
    t = tracks(con,sequence='A'*129024,L=129024,offset=0,rn=1)
    elapsed = time.perf_counter()-start
    assert t[649,5] == t[550,6] == np.float16(np.log(2))
    assert t[520,1] == np.float16(np.log(2)) and t[700,1] > 0
    assert qc['retained'] == 1 and qc['valid_tiles_with_evidence'] == 2
    assert t.nbytes == 129024*8*2
    print(f'CPU raster 129024 bases: {elapsed:.6f}s; output={t.nbytes} bytes; fixture only')
