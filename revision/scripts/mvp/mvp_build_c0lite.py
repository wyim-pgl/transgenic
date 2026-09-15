#!/usr/bin/env python3
"""Exploratory Athaliana MVP only; no scoring filters or trainer dependencies.

Coordinates: spans/blocks half-open; junction endpoints inclusive. est_molecule's
primary placement is a deterministic representative, NOT the component geometry.
Use est_block/est_junction for tracks, est_placement for leakage/provenance.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import duckdb
import pysam

SPEC_SHA256 = '734387b0a3251e7b0f47cf36bb0edc0dc51549a02f0c2d2579765d5445879c74'
BATCH_SIZE = 4096
PANEL_COLUMNS = ('panel shuffle_rank species_id gene_id chromosome start fin strand length '
                 'orthogroup_id n_transcripts exon_count intron_count est_molecules').split()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def batches(con, sql):
    """Separate cursor allows bounded reads while the main connection writes."""
    cur = con.cursor()
    try:
        cur.execute(sql)
        while True:
            rows = cur.fetchmany(BATCH_SIZE)
            if not rows:
                break
            yield rows
    finally:
        cur.close()


def clone_key(name):
    return re.sub(r'(?:\.\d+|_[FR])+$', '', name)


def load_sources(con, corpus_db, panels, flank, references):
    sidecar = Path(str(panels) + '.sha256')
    if sidecar.exists():
        fields = sidecar.read_text().split()
        require(fields and fields[0] == sha256(panels), 'panel sha256 mismatch')
    con.execute('BEGIN')
    protected = []
    seen = set()
    with open(panels) as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        require(reader.fieldnames == PANEL_COLUMNS, 'invalid panel columns/order')
        for row in reader:
            require(row['species_id'] == 'Athaliana', 'panels must contain only Athaliana')
            require(row['panel'] in ('mvp_core', 'mvp', 'pilot', 'final'), 'invalid panel label')
            a, b = int(row['start']) - 1, int(row['fin'])
            chrom = row['chromosome']
            require(chrom in references and 0 <= a < b <= references[chrom], 'invalid panel coordinates/reference')
            require(row['strand'] in ('+', '-') and int(row['length']) == b-a, 'invalid panel strand/length')
            require(row['gene_id'] and row['gene_id'] not in seen, 'duplicate/empty panel gene')
            seen.add(row['gene_id'])
            protected.append((chrom, max(0, a-flank), b+flank, 'panel'))
    require(protected, 'empty panel universe')
    con.execute('CREATE TABLE protected_span(chrom VARCHAR, start0 BIGINT, end0 BIGINT, kind VARCHAR)')
    con.executemany('INSERT INTO protected_span VALUES (?,?,?,?)', protected)
    panel_count = len(protected)
    con.execute('CREATE TABLE corpus_tile(rn BIGINT PRIMARY KEY, chrom VARCHAR, start0 BIGINT, end0 BIGINT, split VARCHAR, is_rc BOOLEAN)')
    with duckdb.connect(str(corpus_db), read_only=True) as src:
        for table, columns in [('geneList', {'rn','species_id','chromosome','start','fin','split','is_rc'}),
                               ('tile_blocks', {'species_id','chromosome','start0','end0','split'})]:
            actual = {r[0] for r in src.execute('SELECT column_name FROM information_schema.columns WHERE table_name = ?', [table]).fetchall()}
            require(columns <= actual, f'{table}: missing required columns {sorted(columns-actual)}')
        n_tiles = 0
        for rows in batches(src, "SELECT rn,chromosome,start,fin,split,is_rc FROM geneList WHERE species_id='Athaliana'"):
            for rn, chrom, a, b, split, rc in rows:
                # B5 pads a short contig's [0,tier) tile beyond the reference.
                require(chrom in references and a is not None and b is not None
                        and 0 <= a < min(b,references[chrom])
                        and (b <= references[chrom] or (a == 0 and b in (30720,61440,129024))),
                        'invalid Athaliana tile coordinates/reference')
                require(rn is not None and split in ('train','valid','test') and rc is not None, 'invalid Athaliana tile metadata')
            con.executemany('INSERT INTO corpus_tile VALUES (?,?,?,?,?,?)', rows)
            n_tiles += len(rows)
        require(n_tiles > 0, 'corpus has no Athaliana tiles')
        for rows in batches(src, "SELECT chromosome,start0,end0,split FROM tile_blocks WHERE species_id='Athaliana'"):
            tests = []
            for chrom, a, b, split in rows:
                require(chrom in references and a is not None and b is not None and 0 <= a < b <= references[chrom], 'invalid Athaliana block coordinates/reference')
                require(split in ('train','valid','test'), 'invalid tile_blocks split')
                if split == 'test':
                    tests.append((chrom,a,b,'test'))
            if tests:
                con.executemany('INSERT INTO protected_span VALUES (?,?,?,?)', tests)
    con.execute('COMMIT')
    return panel_count


def stage_bam(con, fh):
    con.execute('CREATE TABLE est_placement(placement_id BIGINT PRIMARY KEY, member VARCHAR, flag INTEGER, kind VARCHAR, chrom VARCHAR, alignment_strand VARCHAR, start0 BIGINT, end0 BIGINT, cigar VARCHAR, ts VARCHAR, tx_strand VARCHAR, mapq INTEGER)')
    con.execute('CREATE TABLE member_block(placement_id BIGINT, chrom VARCHAR, start0 BIGINT, end0 BIGINT)')
    con.execute('CREATE TABLE member_junction(placement_id BIGINT, chrom VARCHAR, intron_start0 BIGINT, intron_end0 BIGINT)')
    tables = {'est_placement': ([],12), 'member_block': ([],4), 'member_junction': ([],4)}
    qc = {'reads_in': 0, 'secondary_ignored': 0, 'max_stage_batch_rows': 0}

    def flush(table):
        rows, width = tables[table]
        if rows:
            qc['max_stage_batch_rows'] = max(qc['max_stage_batch_rows'], len(rows))
            con.executemany(f'INSERT INTO {table} VALUES ({",".join("?" for _ in range(width))})', rows)
            rows.clear()

    def add(table, row):
        tables[table][0].append(row)
        if len(tables[table][0]) >= BATCH_SIZE:
            flush(table)

    con.execute('BEGIN')
    for i, r in enumerate(fh.fetch(until_eof=True)):
        qc['reads_in'] += 1
        require(r.query_name and r.query_name != '*', 'BAM member name required')
        strand = '-' if r.flag & 16 else '+'
        ts = r.get_tag('ts') if r.has_tag('ts') else None
        # A39 composition only. Deliberately do not reuse scorer exclusion/MAPQ policy.
        tx = strand if ts == '+' else ('+' if strand == '-' else '-') if ts == '-' else '?'
        kind = 'secondary' if r.is_secondary else 'unmapped' if r.is_unmapped else 'supplementary' if r.is_supplementary else 'primary'
        chrom = a = end = cigar = None
        if kind == 'secondary':
            qc['secondary_ignored'] += 1
        elif kind != 'unmapped':
            chrom, a, cigar = r.reference_name, r.reference_start, r.cigarstring
            require(r.cigartuples and chrom and a >= 0, 'mapped placement needs reference/CIGAR')
            pos = block_start = a
            for op, n in r.cigartuples:
                require(n > 0 and op in (0,1,2,3,4,5,6,7,8), 'invalid CIGAR operation')
                if op == 3:
                    require(pos > block_start, 'empty CIGAR block before N')
                    if kind == 'primary':
                        add('member_block', (i,chrom,block_start,pos))
                        add('member_junction', (i,chrom,pos,pos+n-1))
                    pos += n
                    block_start = pos
                elif op in (0,2,7,8):
                    pos += n
            require(pos > block_start and pos <= fh.get_reference_length(chrom), 'invalid placement span')
            end = pos
            if kind == 'primary':
                add('member_block', (i,chrom,block_start,pos))
        add('est_placement', (i,r.query_name,r.flag,kind,chrom,strand,a,end,cigar,ts,tx,r.mapping_quality))
    for table in tables:
        flush(table)
    con.execute('COMMIT')
    return qc


def resolve_components(con, qc):
    # ponytail: O(unique QNAMEs) union-find RAM; placements/geometry stay on disk.
    # Replace with external-memory connectivity only if the real name count needs it.
    parent = {}
    for rows in batches(con, 'SELECT DISTINCT member FROM est_placement'):
        parent.update((name,name) for name, in rows)

    def root(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(a, b):
        a, b = root(a), root(b)
        if a == b:
            return 0
        parent[max(a,b)] = min(a,b)
        return 1

    qc['members_in'] = len(parent)
    qc['collapsed_rule_a'] = 0
    last_key = last_name = None
    sql = "SELECT chrom,alignment_strand,start0,cigar,member FROM est_placement WHERE kind='primary' ORDER BY chrom,alignment_strand,start0,cigar,member"
    for rows in batches(con, sql):
        for *key, name in rows:
            if key == last_key:
                qc['collapsed_rule_a'] += union(last_name, name)
            last_key, last_name = key, name
    con.execute('CREATE TABLE est_member(member VARCHAR PRIMARY KEY, clone_key VARCHAR, component VARCHAR, molecule_id VARCHAR)')
    con.execute('BEGIN')
    pending = []
    for name in parent:
        pending.append((name,clone_key(name),None,None))
        if len(pending) == BATCH_SIZE:
            con.executemany('INSERT INTO est_member VALUES (?,?,?,?)', pending)
            pending.clear()
    if pending:
        con.executemany('INSERT INTO est_member VALUES (?,?,?,?)', pending)
    con.execute('COMMIT')
    qc['collapsed_rule_b'] = 0
    last_key = last_name = None
    for rows in batches(con, 'SELECT clone_key,member FROM est_member ORDER BY clone_key,member'):
        for key, name in rows:
            if key and key == last_key:
                qc['collapsed_rule_b'] += union(last_name,name)
            last_key, last_name = key, name
    con.execute('BEGIN')
    pending = []
    for name in parent:
        pending.append((root(name),name))
        if len(pending) == BATCH_SIZE:
            con.executemany('UPDATE est_member SET component=? WHERE member=?', pending)
            pending.clear()
    if pending:
        con.executemany('UPDATE est_member SET component=? WHERE member=?', pending)
    con.execute('COMMIT')
    con.execute('CREATE TEMP TABLE components AS SELECT component, list(member ORDER BY member) members, sha256(to_json(list(member ORDER BY member))) molecule_id FROM est_member GROUP BY component')
    con.execute('UPDATE est_member SET molecule_id=c.molecule_id FROM components c WHERE est_member.component=c.component')
    con.execute('ALTER TABLE est_member DROP COLUMN component')
    con.execute('''CREATE TEMP TABLE placement_hits AS
        SELECT p.placement_id, bool_or(s.kind='panel') overlaps_panel, bool_or(s.kind='test') overlaps_test
        FROM est_placement p JOIN protected_span s ON p.chrom=s.chrom AND p.start0<s.end0 AND p.end0>s.start0
        WHERE p.kind IN ('primary','supplementary') GROUP BY p.placement_id''')
    con.execute('''CREATE TEMP TABLE summary AS SELECT e.molecule_id,
        count(*) FILTER (WHERE p.kind='primary') primary_count,
        count(*) FILTER (WHERE p.kind='supplementary') supplementary_count,
        count(DISTINCT p.tx_strand) FILTER (WHERE p.kind='primary' AND p.tx_strand IN ('+','-')) known_strands,
        min(p.tx_strand) FILTER (WHERE p.kind='primary' AND p.tx_strand IN ('+','-')) known_strand,
        max(p.mapq) FILTER (WHERE p.kind='primary') mapq,
        coalesce(bool_or(h.overlaps_panel),false) overlaps_panel,
        coalesce(bool_or(h.overlaps_test),false) overlaps_test
        FROM est_member e JOIN est_placement p ON e.member=p.member
        LEFT JOIN placement_hits h USING(placement_id) GROUP BY e.molecule_id''')
    con.execute('''CREATE TABLE est_molecule AS WITH representative AS (
        SELECT e.molecule_id,p.chrom,p.start0,p.end0,
        row_number() OVER(PARTITION BY e.molecule_id ORDER BY p.chrom,p.start0,p.end0,p.alignment_strand,p.cigar,p.member) n
        FROM est_member e JOIN est_placement p ON e.member=p.member WHERE p.kind='primary')
        SELECT c.molecule_id,c.members,CASE WHEN s.known_strands=1 THEN s.known_strand ELSE '?' END strand,
        s.mapq,r.chrom,r.start0,r.end0,
        (s.primary_count=0 OR s.overlaps_panel OR s.overlaps_test) excluded,
        CASE WHEN s.primary_count=0 THEN CASE WHEN s.supplementary_count>0 THEN 'supplementary_only' ELSE 'unmapped' END
             WHEN s.overlaps_panel THEN 'heldout_locus' WHEN s.overlaps_test THEN 'test_block' ELSE NULL END cause,
        s.overlaps_panel,s.overlaps_test,s.primary_count,s.known_strands>1 strand_conflict
        FROM components c JOIN summary s USING(molecule_id)
        LEFT JOIN representative r ON c.molecule_id=r.molecule_id AND r.n=1''')
    con.execute('ALTER TABLE est_molecule ADD PRIMARY KEY(molecule_id)')
    # Running MAX (not lag) handles nested and repeated blocks before interval union.
    con.execute('''CREATE TABLE est_block AS WITH raw AS (
        SELECT DISTINCT e.molecule_id,b.chrom,b.start0,b.end0 FROM member_block b
        JOIN est_placement p USING(placement_id) JOIN est_member e ON p.member=e.member),
        prior AS (SELECT *,max(end0) OVER(PARTITION BY molecule_id,chrom ORDER BY start0,end0 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prev_end FROM raw),
        islands AS (SELECT *,sum(CASE WHEN prev_end IS NULL OR start0>prev_end THEN 1 ELSE 0 END)
            OVER(PARTITION BY molecule_id,chrom ORDER BY start0,end0) grp FROM prior)
        SELECT molecule_id,chrom,min(start0)::BIGINT start0,max(end0)::BIGINT end0 FROM islands GROUP BY molecule_id,chrom,grp''')
    con.execute('''CREATE TABLE est_junction AS SELECT DISTINCT e.molecule_id,j.chrom,j.intron_start0,j.intron_end0,m.strand tx_strand,
        CASE WHEN m.strand='+' THEN j.intron_start0 WHEN m.strand='-' THEN j.intron_end0 END donor0,
        CASE WHEN m.strand='+' THEN j.intron_end0 WHEN m.strand='-' THEN j.intron_start0 END acceptor0
        FROM member_junction j JOIN est_placement p USING(placement_id)
        JOIN est_member e ON p.member=e.member JOIN est_molecule m USING(molecule_id)''')
    con.execute('''CREATE TABLE tile_est AS SELECT DISTINCT t.rn,b.molecule_id FROM corpus_tile t
        JOIN est_block b ON t.chrom=b.chrom AND t.start0<b.end0 AND t.end0>b.start0
        JOIN est_molecule m USING(molecule_id) WHERE t.split IN ('train','valid') AND NOT m.excluded''')
    con.execute('ALTER TABLE tile_est ADD PRIMARY KEY(rn,molecule_id)')


def final_qc(con, qc):
    qc['molecules_after_dedup'] = con.execute('SELECT count(*) FROM est_molecule').fetchone()[0]
    qc['collapsed_total'] = qc['members_in'] - qc['molecules_after_dedup']
    for cause in ('heldout_locus','test_block','unmapped','supplementary_only'):
        qc['excluded_' + cause] = con.execute('SELECT count(*) FROM est_molecule WHERE cause=?',[cause]).fetchone()[0]
    qc['retained'] = con.execute('SELECT count(*) FROM est_molecule WHERE NOT excluded').fetchone()[0]
    qc['strand_conflicts'] = con.execute('SELECT count(*) FROM est_molecule WHERE strand_conflict').fetchone()[0]
    qc['valid_tiles_total'] = con.execute("SELECT count(*) FROM corpus_tile WHERE split='valid'").fetchone()[0]
    qc['valid_tiles_with_evidence'] = con.execute("SELECT count(DISTINCT t.rn) FROM corpus_tile t JOIN tile_est e USING(rn) WHERE t.split='valid'").fetchone()[0]
    # Independent rejoin: do NOT trust placement_hits or stored component overlap flags.
    for kind in ('panel','test'):
        value = con.execute('''SELECT count(DISTINCT m.molecule_id) FROM est_molecule m
            JOIN est_member e USING(molecule_id) JOIN est_placement p ON e.member=p.member
            JOIN protected_span s ON s.chrom=p.chrom AND p.start0<s.end0 AND p.end0>s.start0
            WHERE NOT m.excluded AND p.kind IN ('primary','supplementary') AND s.kind=?''',[kind]).fetchone()[0]
        qc['retained_overlapping_' + kind] = value
        require(value == 0, f'QC failed: retained overlap with {kind}')
    require(qc['collapsed_total'] == qc['collapsed_rule_a'] + qc['collapsed_rule_b'], 'dedup accounting mismatch')
    con.execute('CREATE TABLE mvp_qc(metric VARCHAR PRIMARY KEY, value BIGINT)')
    con.executemany('INSERT INTO mvp_qc VALUES (?,?)', list(qc.items()))


def build_c0lite(*, bam, panels, corpus_db, out, flank=150):
    """Build atomically, never overwrite; return numeric QC. Input corpus is read-only."""
    bam, panels, corpus_db, out = map(lambda p: Path(p).absolute(), (bam,panels,corpus_db,out))
    require(isinstance(flank,int) and flank >= 0, 'flank must be a nonnegative integer')
    for source in (bam,panels,corpus_db):
        require(source.is_file(), f'input missing: {source}')
        require(out.resolve() != source.resolve(), 'output/input collision')
    if os.path.lexists(out):
        raise FileExistsError(out)
    require(not Path(str(corpus_db)+'.wal').exists(), 'corpus has a WAL; supply a closed/checkpointed snapshot')
    out.parent.mkdir(parents=True,exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(prefix=out.name+'.partial-', dir=out.parent)
    os.close(handle)
    os.unlink(tmp_name)  # DuckDB must create its own file header.
    con = None
    try:
        con = duckdb.connect(tmp_name, config={'threads':1})
        with pysam.AlignmentFile(str(bam), 'rb') as fh:
            references = dict(zip(fh.references,fh.lengths))
            panel_count = load_sources(con,corpus_db,panels,flank,references)
            qc = stage_bam(con,fh)
        resolve_components(con,qc)
        qc['panel_rows_protected'] = panel_count
        final_qc(con,qc)
        provenance = {'schema_version':'mvp_c0lite_v1', 'spec_sha256':SPEC_SHA256,
            'bam':str(bam), 'panels':str(panels), 'corpus_db':str(corpus_db),
            'bam_sha256':sha256(bam), 'panels_sha256':sha256(panels), 'corpus_sha256':sha256(corpus_db),
            'flank':flank, 'panel_universe':'all supplied rows (production approved panel: 2079)',
            'strand_policy':'primary known strands: single known wins; conflicting known -> ?; all unknown -> ?',
            'geometry_policy':'union primary blocks; distinct primary junctions; representative primary is lexicographic chrom/start/end/FLAG-strand/CIGAR/member; mapq=max primary',
            'exclusion_policy':'full primary/supplementary spans including introns; no-primary disposition first; then panel before test; preserve both flags',
            'secondary_policy':'names retained; secondary geometry ignored; secondary-only cause=unmapped (no usable primary)',
            'molecule_id':'sha256 UTF-8 compact JSON array of sorted unique member names (DuckDB to_json)',
            'dedup_counts':'successful union edges: rule a first, then rule b; order within each rule does not change counts',
            'clone_policy':r'strip terminal runs of .digits and _F/_R; no library/UniVec inference',
            'stage_batch_rows':BATCH_SIZE, 'duckdb_version':duckdb.__version__, 'pysam_version':pysam.__version__}
        con.execute('CREATE TABLE mvp_provenance(key VARCHAR PRIMARY KEY,value VARCHAR)')
        con.executemany('INSERT INTO mvp_provenance VALUES (?,?)', [(k,json.dumps(v,sort_keys=True)) for k,v in provenance.items()])
        con.execute('CHECKPOINT')
        con.close()
        con = None
        os.link(tmp_name,out)  # Atomic no-clobber publish, same filesystem.
        return qc
    finally:
        if con is not None:
            con.close()
        for path in (Path(tmp_name),Path(tmp_name+'.wal')):
            path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bam',required=True)
    parser.add_argument('--panels',required=True)
    parser.add_argument('--db',dest='corpus_db',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--flank',type=int,default=150)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build_c0lite(**vars(args)),sort_keys=True))
        return 0
    except (ValueError,OSError,duckdb.Error) as exc:
        print(f'C0-lite build failed: {exc}',file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
