"""Synthetic BAM tests: changing molecule identity/leakage policy must break these."""
import csv
import importlib.util
from pathlib import Path

import duckdb
import pysam
import pytest

ROOT = Path(__file__).resolve().parents[3]
BUILDER = ROOT / 'revision/scripts/mvp/mvp_build_c0lite.py'
COLUMNS = ('panel shuffle_rank species_id gene_id chromosome start fin strand length '
           'orthogroup_id n_transcripts exon_count intron_count est_molecules').split()


def load_builder():
    assert BUILDER.exists(), 'approved C0-lite builder is not implemented'
    spec = importlib.util.spec_from_file_location('mvp_builder', BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(name, start=500, cigar='20M', flag=0, ts='+', chrom='Chr1'):
    return name, start, cigar, flag, ts, chrom


def inputs(tmp_path, records, panels=None, tiles=None, blocks=None):
    bam = tmp_path / 'tiny.bam'
    header = {'HD': {'VN': '1.6', 'SO': 'unsorted'},
              'SQ': [{'SN': 'Chr1', 'LN': 200000}, {'SN': 'Chr2', 'LN': 1000}]}
    with pysam.AlignmentFile(bam, 'wb', header=header) as fh:
        for name, start, cigar, flag, ts, chrom in records:
            r = pysam.AlignedSegment(fh.header)
            r.query_name, r.flag = name, flag
            if not flag & 4:
                r.reference_id = fh.get_tid(chrom)
                r.reference_start = start
                r.cigarstring, r.mapping_quality = cigar, 60
            if ts is not None:
                r.set_tag('ts', ts, value_type='A')
            fh.write(r)
    panel = tmp_path / 'panels.tsv'
    with panel.open('w') as fh:
        writer = csv.writer(fh, delimiter='\t')
        writer.writerow(COLUMNS)
        for i, (label, chrom, start, end) in enumerate(panels or [('final', 'Chr1', 101, 200)], 1):
            writer.writerow([label, i, 'Athaliana', f'g{i}', chrom, start, end, '+', end-start+1, '', 1, 2, 1, ''])
    corpus = tmp_path / 'corpus.db'
    with duckdb.connect(str(corpus)) as con:
        con.execute('CREATE TABLE geneList(rn INT PRIMARY KEY, species_id VARCHAR, chromosome VARCHAR, start INT, fin INT, split VARCHAR, is_rc BOOLEAN)')
        con.executemany('INSERT INTO geneList VALUES (?,?,?,?,?,?,?)', tiles or [
            (1, 'Athaliana', 'Chr1', 350, 1000, 'train', False),
            (2, 'Athaliana', 'Chr1', 350, 1000, 'valid', False),
            (3, 'Athaliana', 'Chr1', 350, 1000, 'valid', True),
            (4, 'Athaliana', 'Chr1', 350, 1000, 'test', False),
            (5, 'Other', 'Chr1', 350, 1000, 'train', False)])
        con.execute('CREATE TABLE tile_blocks(species_id VARCHAR, chromosome VARCHAR, start0 INT, end0 INT, split VARCHAR)')
        con.executemany('INSERT INTO tile_blocks VALUES (?,?,?,?,?)', blocks or [
            ('Athaliana', 'Chr1', 2000, 2100, 'test'),
            ('Other', 'Chr1', 350, 1000, 'test')])
    return dict(bam=bam, panels=panel, corpus_db=corpus, out=tmp_path / 'evidence.db')


def build(tmp_path, records, **kwargs):
    args = inputs(tmp_path, records, **kwargs)
    qc = load_builder().build_c0lite(**args)
    return duckdb.connect(str(args['out']), read_only=True), qc


@pytest.mark.parametrize('reverse', [False, True])
def test_transitive_closure_late_supplementary_and_cause_priority(tmp_path, reverse):
    rows = [record('A'), record('clone_F'), record('clone_R', 800),
            record('clone_R', 100, flag=2048), record('A', 2000, flag=2048)]
    con, qc = build(tmp_path, rows[::-1] if reverse else rows)
    row = con.execute('SELECT members, excluded, cause, overlaps_panel, overlaps_test FROM est_molecule').fetchone()
    assert row == (['A', 'clone_F', 'clone_R'], True, 'heldout_locus', True, True)
    assert con.execute('SELECT count(*) FROM tile_est').fetchone()[0] == 0
    assert con.execute('SELECT count(*) FROM est_placement').fetchone()[0] == 5
    assert qc['collapsed_rule_a'] == 1 and qc['collapsed_rule_b'] == 1
    assert qc['retained_overlapping_panel'] == qc['retained_overlapping_test'] == 0
    # Identity is independent of traversal/root choice.
    assert con.execute('SELECT molecule_id FROM est_molecule').fetchone()[0] == __import__('hashlib').sha256(b'["A","clone_F","clone_R"]').hexdigest()


def test_exact_key_cigar_flag_chrom_not_ts_and_clone_suffixes(tmp_path):
    rows = [record('plain'), record('same', ts='-'), record('different_cigar', cigar='10M1I10M'),
            record('reverse', flag=16), record('chrom', chrom='Chr2'),
            record('ACC.1', 600), record('ACC.2', 650),
            record('mate.5', 700), record('mate.3', 750),
            record('pair_F', 800), record('pair_R', 850), record('unrelated', 900)]
    con, qc = build(tmp_path, rows)
    groups = con.execute('SELECT members, strand FROM est_molecule ORDER BY members').fetchall()
    assert len(groups) == 8
    assert (['plain', 'same'], '?') in groups
    assert (['ACC.1', 'ACC.2'], '+') in groups
    assert (['mate.3', 'mate.5'], '+') in groups
    assert (['pair_F', 'pair_R'], '+') in groups
    assert qc['strand_conflicts'] == 1


@pytest.mark.parametrize('flag,ts,want', [(0,'+','+'),(0,'-','-'),(16,'+','-'),(16,'-','+'),
                                         (0,None,'?'),(16,None,'?'),(0,'?','?'),(16,'?','?')])
def test_flag_ts_and_cigar_geometry(tmp_path, flag, ts, want):
    con, _ = build(tmp_path, [record('x', 400, '2S3M2=1X4D3I100N5M2H', flag, ts)])
    assert con.execute('SELECT strand,start0,end0 FROM est_molecule').fetchone() == (want,400,515)
    assert con.execute('SELECT start0,end0 FROM est_block ORDER BY start0').fetchall() == [(400,410),(510,515)]
    donor, acceptor = (410,509) if want == '+' else (509,410) if want == '-' else (None,None)
    assert con.execute('SELECT intron_start0,intron_end0,tx_strand,donor0,acceptor0 FROM est_junction').fetchone() == (410,509,want,donor,acceptor)


def test_boundaries_intronic_leakage_and_no_primary_precedence(tmp_path):
    con, qc = build(tmp_path, [record('touch',350), record('ends',349,'1M'),
        record('intron',0,'50M400N10M'), record('secondary_ok',600),
        record('secondary_ok',100,flag=256), record('supp_only',100,flag=2048),
        record('unmapped',flag=4), record('secondary_only',100,flag=256),
        record('test_hit',2000), record('test_touch',2100)])
    causes = dict(con.execute('SELECT members[1],cause FROM est_molecule').fetchall())
    assert causes == dict(touch=None, ends='heldout_locus', intron='heldout_locus', secondary_ok=None,
                         supp_only='supplementary_only', unmapped='unmapped', secondary_only='unmapped',
                         test_hit='test_block', test_touch=None)
    assert qc['reads_in'] == 10 and qc['secondary_ignored'] == 2
    assert con.execute("SELECT count(*) FROM est_block b JOIN est_molecule m USING(molecule_id) WHERE m.cause IN ('unmapped','supplementary_only')").fetchone()[0] == 0


def test_union_member_geometry_distinct_membership_and_no_intron_membership(tmp_path):
    tiles = [(1,'Athaliana','Chr1',510,520,'train',False),
             (2,'Athaliana','Chr1',510,520,'valid',False),
             (3,'Athaliana','Chr1',510,520,'valid',True),
             (4,'Athaliana','Chr1',510,520,'test',False),
             (5,'Other','Chr1',510,520,'train',False),
             (6,'Athaliana','Chr1',560,580,'train',False)]
    con, qc = build(tmp_path, [record('c_F',500,'50M100N50M'),record('c_R',510,'40M100N60M')], tiles=tiles)
    assert con.execute('SELECT start0,end0 FROM est_block ORDER BY start0').fetchall() == [(500,550),(650,710)]
    assert con.execute('SELECT intron_start0,intron_end0 FROM est_junction').fetchall() == [(550,649)]
    assert con.execute('SELECT rn FROM tile_est ORDER BY rn').fetchall() == [(1,),(2,),(3,)]
    assert qc['valid_tiles_total'] == qc['valid_tiles_with_evidence'] == 2


def test_safe_output_cli_and_invalid_inputs(tmp_path):
    args = inputs(tmp_path, [record('x')])
    module = load_builder()
    assert module.main(['--bam',str(args['bam']),'--panels',str(args['panels']),
                        '--db',str(args['corpus_db']),'--out',str(args['out'])]) == 0
    original = args['out'].read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        module.build_c0lite(**args)
    assert args['out'].read_bytes() == original
    with pytest.raises(ValueError):
        module.build_c0lite(**dict(args,out=args['corpus_db']))
    args['out'].unlink()
    args['panels'].write_text(args['panels'].read_text().replace('Athaliana','Other'))
    with pytest.raises(ValueError, match='Athaliana'):
        module.build_c0lite(**args)
    assert not args['out'].exists()
    assert not list(tmp_path.glob('*.partial*'))


def test_schema_and_checksum_fail_closed(tmp_path):
    args = inputs(tmp_path,[record('x')])
    module = load_builder()
    sidecar = Path(str(args['panels'])+'.sha256')
    sidecar.write_text('0'*64+'  panels.tsv\n')
    with pytest.raises(ValueError, match='sha256'):
        module.build_c0lite(**args)
    sidecar.unlink()
    with duckdb.connect(str(args['corpus_db'])) as con:
        con.execute('DROP TABLE tile_blocks')
    with pytest.raises(ValueError, match='tile_blocks'):
        module.build_c0lite(**args)
    assert not args['out'].exists()


def test_b5_short_contig_padded_tile_is_legal(tmp_path):
    # gsf_contract.tile_windows(1000,30720) returns [(0,30720)].
    con, qc = build(tmp_path,[record('short',500,chrom='Chr2')],
                    tiles=[(1,'Athaliana','Chr2',0,30720,'valid',False)])
    assert con.execute('SELECT rn FROM tile_est').fetchall() == [(1,)]
    assert qc['valid_tiles_with_evidence'] == 1


def test_bounded_batches_preserve_edges_and_all_member_geometry(tmp_path):
    args = inputs(tmp_path,[record('A'),record('clone_F'),record('clone_R',800),
                            record('clone_R',100,flag=2048),record('A',2000,flag=2048)])
    module = load_builder()
    module.BATCH_SIZE = 2
    qc = module.build_c0lite(**args)
    assert qc['max_stage_batch_rows'] == 2
    assert qc['molecules_after_dedup'] == 1 and qc['excluded_heldout_locus'] == 1
    with duckdb.connect(str(args['out']),read_only=True) as con:
        assert con.execute('SELECT count(*) FROM member_block').fetchone()[0] == 3
        assert con.execute('SELECT count(*) FROM est_member').fetchone()[0] == 3


def test_independent_qc_blocks_publication_of_corrupted_disposition(tmp_path):
    args = inputs(tmp_path,[record('hit',100)])
    module = load_builder()
    original = module.resolve_components
    def corrupt(con,qc):
        original(con,qc)
        con.execute('UPDATE est_molecule SET excluded=false,cause=NULL')
    module.resolve_components = corrupt
    with pytest.raises(ValueError, match='QC failed'):
        module.build_c0lite(**args)
    assert not args['out'].exists()
    assert not list(tmp_path.glob('*.partial*'))


def test_empty_bam_is_supplied_empty_source_with_provenance(tmp_path):
    con, qc = build(tmp_path,[])
    assert qc['reads_in'] == qc['molecules_after_dedup'] == qc['retained'] == 0
    assert qc['panel_rows_protected'] == 1
    assert con.execute('SELECT count(*) FROM est_block').fetchone()[0] == 0
    import json
    provenance = {k:json.loads(v) for k,v in con.execute('SELECT * FROM mvp_provenance').fetchall()}
    assert provenance['spec_sha256'] == '734387b0a3251e7b0f47cf36bb0edc0dc51549a02f0c2d2579765d5445879c74'
    assert provenance['flank'] == 150 and 'all supplied rows' in provenance['panel_universe']
