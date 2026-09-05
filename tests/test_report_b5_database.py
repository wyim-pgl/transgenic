"""Population accounting must deduplicate loci and detect illegal memberships."""
import importlib.util
from pathlib import Path

import duckdb
import pytest

spec = importlib.util.spec_from_file_location('report_b5', Path(__file__).resolve().parents[1] / 'scripts/report_b5_database.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.fixture
def con():
    c = duckdb.connect()
    c.sql('CREATE TABLE gene_split(species_id VARCHAR,gene_id VARCHAR,split VARCHAR,strict_holdout BOOLEAN)')
    c.sql('CREATE TABLE gene_key_map(species_id VARCHAR,gene_id VARCHAR,start0 INT,end0 INT)')
    c.sql('CREATE TABLE geneList(species_id VARCHAR,gene_id VARCHAR,split VARCHAR,is_rc BOOLEAN)')
    c.sql('CREATE TABLE window_genes(species_id VARCHAR,window_id VARCHAR,gene_id VARCHAR,is_rc BOOLEAN)')
    c.sql("INSERT INTO gene_split VALUES ('s','a','test',true),('s','b','test',false),('s','c','train',false)")
    c.sql("INSERT INTO gene_key_map VALUES ('s','a',0,100),('s','b',0,200),('s','c',0,50)")
    c.sql("INSERT INTO geneList VALUES ('s','w','test',false),('s','w','test',true),('s','v','test',false)")
    c.sql("INSERT INTO window_genes VALUES ('s','w','a',false),('s','w','a',true),('s','v','a',false),('s','w','c',false)")
    yield c
    c.close()


def test_population_deduplicates_rc_and_tiers_and_keeps_gene_split_distinct(con):
    p = report.evaluation_population(con)
    assert p['split_totals']['test'] == dict(assigned_genes=2, ever_labelled_genes=1, pct=50.0, strict_assigned=1, strict_ever_labelled=1)
    assert p['test_gene_length_nt']['s']['unlabelled']['median'] == 200
    assert {(r['gene_split'], r['tile_split'], r['unique_genes']) for r in p['gene_split_by_tile_split']} == {('train','test',1), ('test','test',1)}
    assert all(report.labelled_gene_checks(con).values())


@pytest.mark.parametrize('split', ['train','valid'])
def test_illegal_label_and_strict_holdout_are_detected(con, split):
    con.execute('UPDATE geneList SET split=?', [split])
    checks = report.labelled_gene_checks(con)
    assert not checks['no_gene_labelled_below_its_split']
    assert not checks['no_strict_holdout_labelled_outside_test']
    assert checks['no_orphan_window_genes']


def test_orphan_rc_membership_is_not_hidden_by_forward_row(con):
    con.sql('DELETE FROM geneList WHERE is_rc')
    assert not report.labelled_gene_checks(con)['no_orphan_window_genes']


def test_order_audit_distinguishes_ties_decreasing_coordinates_and_unknown(con):
    con.sql('CREATE TABLE rejected_records(species_id VARCHAR,gene_id VARCHAR,reason VARCHAR)')
    for reason in ['((12, 34) after (12, 34))', '((12, 34) after (56, 78))', 'unknown']:
        con.execute('INSERT INTO rejected_records VALUES (?,?,?)', ['s','w', 'window gene blocks out of canonical order '+reason])
    a = report.canonical_order_audit(con)
    assert (a['rejected_tiles'], a['equal_span_ties'], a['decreasing_spans'], a['unparsed_or_unexpected']) == (3,1,1,1)
