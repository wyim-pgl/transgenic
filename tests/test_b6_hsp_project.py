"""Route-B production projection (A46 rule 6) on synthetic, hand-checkable fixtures."""
import csv
import hashlib
import itertools
import json
import random

import pytest

from revision.scripts.evidence import b6_hsp_project as m
from revision.scripts.evidence import est_phase_project as g
from revision.scripts.evidence.est_blastx import RAW13, search_command
from revision.scripts.evidence.est_hsp_expanded import CODE, rc

# Codons whose first two bases avoid T: no stop in frame +1, and the +2 frame
# codon never starts with T, so it is stop-free too.
SAFE = [c for c in (''.join(x) for x in itertools.product('ACGT', repeat=3))
        if c[0] != 'T' and c[1] != 'T']
L = 303


def mol_id(uids):
    """Stand-in for b6_est_molecule_map.py ids; the projector must use them verbatim."""
    return hashlib.sha256(json.dumps(sorted(set(uids)), separators=(',', ':')).encode()).hexdigest()


def coding_dna(seed=7):
    rng = random.Random(seed)
    return ''.join(rng.choice(SAFE) for _ in range(L // 3))


def translate(dna):
    return ''.join(CODE[dna[i:i+3]] for i in range(0, len(dna) - 2, 3))


def raw_row(qname, protein_id, dna, start0, n_aa, reverse=False):
    """Perfect-match RAW13 row for codons [start0, start0+3n) of the ORIENTED EST."""
    oriented = rc(dna) if reverse else dna
    protein = translate(oriented[start0:start0 + 3 * n_aa])
    qlen = len(dna)
    frame = start0 % 3 + 1
    if reverse:
        qs, qe, frame = qlen - start0, qlen - start0 - 3 * n_aa + 1, -frame
    else:
        qs, qe = start0 + 1, start0 + 3 * n_aa
    row = [qname, str(qlen), protein_id, str(len(protein)), str(qs), str(qe), '1',
           str(len(protein)), str(n_aa), '1e-30', '200', str(frame), str(n_aa)]
    return row, protein


class Fixture:
    def __init__(self, tmp_path, species='Athaliana'):
        self.dir = tmp_path
        self.species = species
        self.est, self.rows, self.proteins, self.sam, self.paf, self.map = {}, {}, {}, [], [], []

    def add_est(self, qname, dna, cigar_blocks, intron, flag=0, ts='+', molecule_members=None,
                library='lib1', pos1=1001, state='train_track', tiles='false', mapped=True,
                superseded='false', accepted='true'):
        self.est[qname] = dna
        a, b = cigar_blocks
        cigar = f'{a}M{intron}N{b}M'
        seq = rc(dna) if flag & 16 else dna
        self.sam.append('\t'.join([qname, str(flag), 'chr1', str(pos1), '60', cigar, '*', '0', '0',
                                   seq, '*', f'ts:A:{ts}']))
        cs = f':{a}~gt{intron}ag:{b}'
        self.paf.append('\t'.join([qname, str(len(dna)), '0', str(len(dna)),
                                   '-' if flag & 16 else '+', 'chr1', '100000', str(pos1 - 1),
                                   str(pos1 - 1 + a + intron + b), str(a + b), str(a + b + intron),
                                   '60', 'tp:A:P', f'ts:A:{ts}', f'cs:Z:{cs}']))
        if mapped:
            self.map.append(dict(qname=qname, member_uid=qname,
                                 molecule_id=mol_id(molecule_members or [qname]), library=library,
                                 est_accession=qname + '.1', molecule_state=state,
                                 train_tiles_excluded=tiles, superseded=superseded, accepted=accepted))

    def add_hit(self, source, qname, protein_id, start0, n_aa, reverse=False, taxid=4081):
        row, protein = raw_row(qname, protein_id, self.est[qname], start0, n_aa, reverse)
        self.rows.setdefault(source, []).append(row)
        header = (f'{protein_id}\t{taxid}_0' if source == 'orthodb'
                  else f'{protein_id} test OS=x OX={taxid} GN=g')
        self.proteins.setdefault(source, {})[protein_id] = (header, protein)
        return row

    def run(self, alignment='sam', out='hsp.tsv', extra=(), sidecar=True):
        d = self.dir
        (d / 'est.univec.fa').write_text(''.join(f'>{q} desc\n{s}\n' for q, s in self.est.items()))
        (d / 'aln.sam').write_text('@SQ\tSN:chr1\tLN:100000\n' + '\n'.join(self.sam) + '\n')
        (d / 'aln.paf').write_text('\n'.join(self.paf) + '\n')
        with open(d / 'map.tsv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(self.map[0]), delimiter='\t', quoting=csv.QUOTE_NONE,
                               quotechar=None)       # as b6_est_molecule_map.py writes it (raw text)
            w.writeheader()
            w.writerows(self.map)
        if sidecar:
            (d / 'map.tsv.sha256').write_text(f'{m.sha256_file(d / "map.tsv")}  map.tsv\n')
        (d / 'lineage.tsv').write_text('taxid\tlineage\n4081\t4081;4107;33090;131567;1\n'
                                       '3702\t3702;3701;33090;131567;1\n'
                                       '999999\t999999;3702;3701;33090;131567;1\n')
        argv = ['--species', self.species, '--diamond-version', '2.1.24.178',
                '--est-fasta', str(d / 'est.univec.fa'), '--alignment', str(d / f'aln.{alignment}'),
                '--molecule-map', str(d / 'map.tsv'), '--lineage', str(d / 'lineage.tsv'),
                '--out', str(d / out), '--qc', str(d / (out + '.qc.json'))]
        for source, rows in self.rows.items():
            (d / f'{source}.raw13.tsv').write_text(''.join('\t'.join(r) + '\n' for r in rows))
            (d / f'{source}.fa').write_text(''.join(f'>{h}\n{s}\n' for h, s in self.proteins[source].values()))
            cmd = [['diamond', 'makedb'],
                   search_command('/bin/diamond', d / 'est.univec.fa', 'db.dmnd', 'o', 1, 't')]
            (d / f'{source}.commands.json').write_text(json.dumps(cmd))
            argv += ['--raw13', f'{source}={d / f"{source}.raw13.tsv"}',
                     '--protein-fasta', f'{source}={d / f"{source}.fa"}',
                     '--blastx-command', f'{source}={d / f"{source}.commands.json"}']
        m.main(argv + list(extra))
        with open(d / out, newline='') as f:
            header = f.readline().rstrip('\n').split('\t')
            rows = [dict(zip(header, line.rstrip('\n').split('\t'))) for line in f]
        self.state = list(csv.DictReader(open(d / (out + '.molecule_state.tsv')), delimiter='\t'))
        return header, rows, json.loads((d / (out + '.qc.json')).read_text())


def per_base(rows):
    out = {}
    for r in rows:
        iv = dict(strand=r['strand'], phase=int(r['phase']), start0=int(r['genomic_start0']))
        for pos in range(int(r['genomic_start0']), int(r['genomic_end0'])):
            out.setdefault((r['hsp_id'], pos), (r['strand'], g.base_label(iv, pos)))
    return out


def expected_labels(dna_len, blocks, intron, flag, start0, n_aa, reverse, pos0=1000):
    """Independent oracle: EST base -> genome through the blocks; label = codon position."""
    a, _ = blocks
    out = {}
    for k in range(3 * n_aa):
        oriented = start0 + k                       # coordinate on the coding (oriented) EST
        u = dna_len - 1 - oriented if reverse else oriented   # original EST coordinate
        i = dna_len - 1 - u if flag & 16 else u     # SAM query coordinate
        pos = pos0 + i + (intron if i >= a else 0)
        strand = '+' if reverse == bool(flag & 16) else '-'
        out[pos] = (strand, k % 3)
    return out


def test_schema_is_exact_and_shared(tmp_path):
    fx = Fixture(tmp_path)
    fx.add_est('e1', coding_dna(), (100, 203), 61)
    fx.add_hit('orthodb', 'e1', '4081_0:000001', 0, 60)
    header, rows, _ = fx.run()
    assert header == m.COLUMNS and len(m.COLUMNS) == 23
    assert header == ('hsp_id species library molecule_id est_accession protein_id protein_source '
                      'protein_species est_start est_end frame evalue bitscore interval_index chrom '
                      'genomic_start0 genomic_end0 strand phase diamond_version database_sha256 '
                      'command_sha256 provenance_sha256').split()
    r = rows[0]
    assert (r['species'], r['protein_source'], r['protein_species'], r['library']) == \
        ('Athaliana', 'orthodb', '4081', 'lib1')
    assert r['molecule_id'] == fx.map[0]['molecule_id']
    assert fx.state == [dict(hsp_id=r['hsp_id'], molecule_id=r['molecule_id'], qname='e1',
                             molecule_state='train_track', train_tiles_excluded='false')]
    assert (r['est_start'], r['est_end'], r['frame']) == ('0', '180', '1')
    assert [x['interval_index'] for x in rows] == ['0', '1']
    assert r['database_sha256'] == m.sha256_file(tmp_path / 'orthodb.fa')
    prov = (tmp_path / 'hsp.tsv.provenance.json').read_bytes()
    assert r['provenance_sha256'] == hashlib.sha256(prov.rstrip(b'\n')).hexdigest()
    assert json.loads(prov)['approval']['record'] == 'notes/a46_approval/A46_APPROVAL_RECORD_20260922.md'


@pytest.mark.parametrize('intron', [60, 61, 62])
@pytest.mark.parametrize('flag,reverse', [(0, False), (16, False), (0, True), (16, True)])
def test_projection_across_intron_by_length_mod3_and_strand(tmp_path, intron, flag, reverse):
    dna = coding_dna()
    if reverse:
        dna = rc(dna)            # the coding sense is now the reverse complement
    fx = Fixture(tmp_path)
    fx.add_est('e1', dna, (100, 203), intron, flag=flag, ts='-' if reverse else '+')
    fx.add_hit('orthodb', 'e1', '4081_0:000001', 3, 70, reverse=reverse)
    _, rows, qc = fx.run()
    got = {pos: v for (_, pos), v in per_base(rows).items()}
    assert got == expected_labels(L, (100, 203), intron, flag, 3, 70, reverse)
    assert len(rows) == 2 and qc['by_source']['orthodb']['kept_hsps'] == 1
    # The genomic frame anchor shifts by intron length mod 3 across the intron.
    first, second = sorted(rows, key=lambda r: int(r['genomic_start0']))
    step = 1 if first['strand'] == '+' else -1
    anchor = [(int(r['genomic_start0']) - step * int(r['phase'])) % 3 for r in (first, second)]
    assert (anchor[1] - anchor[0]) % 3 == intron % 3


def test_paf_and_sam_give_identical_rows(tmp_path):
    results = []
    for kind in ('sam', 'paf'):
        fx = Fixture(tmp_path / kind)
        (tmp_path / kind).mkdir()
        fx.add_est('e1', rc(coding_dna()), (100, 203), 61, flag=16, ts='-')
        fx.add_hit('orthodb', 'e1', '4081_0:000001', 0, 80, reverse=True)
        results.append([{k: v for k, v in r.items() if k not in ('provenance_sha256', 'command_sha256')}
                        for r in fx.run(alignment=kind)[1]])
    assert results[0] == results[1] and results[0]


def test_multi_hsp_conflicts_resolved_per_protein_source(tmp_path):
    fx = Fixture(tmp_path)
    fx.add_est('e1', coding_dna(), (100, 203), 61)
    fx.add_hit('orthodb', 'e1', '4081_0:00000a', 0, 70)       # frame +1 [0,210)
    fx.add_hit('swissprot', 'e1', 'sp|B|B_SOLLC', 121, 30)    # frame +2 [121,211): conflicts A
    fx.add_hit('orthodb', 'e1', '4081_0:00000c', 0, 40)       # frame +1 [0,120): agrees A only
    fx.add_hit('orthodb', 'e1', '4081_0:00000d', 213, 30)     # frame +1 [213,303): alone
    fx.add_hit('orthodb', 'e1', '4081_0:00000e', 0, 29)       # 29 aa: below the HSP rule
    _, rows, qc = fx.run()
    # Swiss-Prot (A30 sensitivity arm) never removes agreeing OrthoDB evidence; per-source outcome is QC only here.
    assert {r['protein_id'] for r in rows} == {'4081_0:00000a', '4081_0:00000c', '4081_0:00000d',
                                               'sp|B|B_SOLLC'}
    o, s = qc['by_source']['orthodb'], qc['by_source']['swissprot']
    assert o.get('phase_conflict_candidates_pre_eligibility', 0) == 0 and o['kept_hsps'] == 3
    assert s['kept_hsps'] == 1 and o['drop_below_30aa'] == 1
    assert o['cross_source_conflict_hsps'] == 1 and s['cross_source_conflict_hsps'] == 1
    assert qc['multi_hsp_rule_applied_by'] == 'b6_build_c0.load_hsp' and 'not here' in qc['multi_hsp_scope']
    prov = json.loads((tmp_path / 'hsp.tsv.provenance.json').read_text())
    assert prov['multi_hsp_rule_applied_by'] == 'b6_build_c0.load_hsp'


def test_multi_hsp_conflict_within_one_source_is_deferred_to_the_builder(tmp_path):
    """Fix round 5, finding 2: the projector no longer drops conflicting HSPs (which HSPs exist for the rule-6 test
    depends on the protein query's rule 4(b) state, known only to the protein C0 build); it emits them all and
    reports what the per-source rule would drop if every HSP were eligible."""
    fx = Fixture(tmp_path)
    fx.add_est('e1', coding_dna(), (100, 203), 61)
    fx.add_hit('orthodb', 'e1', '4081_0:00000a', 0, 70)       # frame +1 [0,210)
    fx.add_hit('orthodb', 'e1', '4081_0:00000f', 121, 30)     # frame +2 [121,211): conflicts A
    fx.add_hit('orthodb', 'e1', '4081_0:00000c', 0, 40)       # agrees A, overlaps F nowhere
    fx.add_hit('swissprot', 'e1', 'sp|S|S_SOLLC', 0, 70)      # agrees A; other source
    _, rows, qc = fx.run()
    assert {r['protein_id'] for r in rows} == {'4081_0:00000a', '4081_0:00000f', '4081_0:00000c', 'sp|S|S_SOLLC'}
    assert qc['by_source']['orthodb']['phase_conflict_candidates_pre_eligibility'] == 2
    assert qc['by_source']['orthodb']['kept_hsps'] == 3
    assert qc['by_source']['swissprot'].get('phase_conflict_candidates_pre_eligibility', 0) == 0
    assert 'drop_phase_conflict_within_molecule' not in qc['by_source']['orthodb']


def test_conflict_scope_is_the_molecule_across_members(tmp_path):
    fx = Fixture(tmp_path)
    dna = coding_dna()
    fx.add_est('e1', dna, (100, 203), 61, molecule_members=['e1', 'e2'])
    fx.add_est('e2', dna[1:] + 'A', (100, 203), 61, molecule_members=['e1', 'e2'], pos1=1002)
    fx.add_hit('orthodb', 'e1', '4081_0:000001', 0, 60)
    fx.add_hit('orthodb', 'e2', '4081_0:000002', 0, 60)   # same genome bases, frame shifted by one
    _, rows, qc = fx.run()
    assert {r['protein_id'] for r in rows} == {'4081_0:000001', '4081_0:000002'}    # emitted; the builder resolves
    assert qc['by_source']['orthodb']['phase_conflict_candidates_pre_eligibility'] == 2
    fx.map[1]['molecule_id'] = mol_id(['e2'])   # separate molecules: no candidate conflict
    fx.map[0]['molecule_id'] = mol_id(['e1'])
    _, rows, qc = fx.run(out='hsp2.tsv')
    assert {r['protein_id'] for r in rows} == {'4081_0:000001', '4081_0:000002'}
    assert qc['by_source']['orthodb'].get('phase_conflict_candidates_pre_eligibility', 0) == 0


@pytest.mark.parametrize('taxid', [3702, 999999])
def test_same_species_protein_refused(tmp_path, taxid):
    fx = Fixture(tmp_path)
    fx.add_est('e1', coding_dna(), (100, 203), 61)
    fx.add_hit('swissprot', 'e1', 'sp|P1|X_ARATH', 0, 60, taxid=taxid)
    with pytest.raises(m.SameSpeciesProtein):
        fx.run()
    assert not (tmp_path / 'hsp.tsv').exists()


def test_frozen_command_and_version_gates(tmp_path):
    good = search_command('/bin/diamond', tmp_path / 'est.univec.fa', 'db', 'o', 1, 't')
    m.check_frozen_command(good, tmp_path / 'est.univec.fa')
    bad = [x if x != '25' else '5' for x in good]
    with pytest.raises(ValueError, match='max-target-seqs'):
        m.check_frozen_command(bad, tmp_path / 'est.univec.fa')
    with pytest.raises(ValueError, match='query'):
        m.check_frozen_command(good, tmp_path / 'other.fa')
    assert good[-13:] == RAW13
    with pytest.raises(ValueError):
        m.check_diamond_version('2.1.23.1')


def test_resolve_hsps_rule_is_order_independent():
    iv = lambda s, e, strand, phase: dict(contig='c', start0=s, end0=e, strand=strand, phase=phase)
    items = [dict(key='A', scope='m', intervals=[iv(0, 30, '+', 0)]),
             dict(key='B', scope='m', intervals=[iv(10, 40, '+', 0)]),   # label 0 at 10 != A's 1
             dict(key='C', scope='m', intervals=[iv(3, 9, '+', 0)]),     # agrees with A
             dict(key='D', scope='m', intervals=[iv(35, 50, '-', 0)]),   # other strand, overlaps B
             dict(key='E', scope='other', intervals=[iv(0, 30, '-', 2)])]
    for order in itertools.permutations(items):
        r = g.resolve_hsps(order)
        assert set(r['kept']) == {'C', 'E'} and set(r['dropped']) == {'A', 'B', 'D'}
        assert r['conflict_pairs'] == 2


def test_molecule_id_is_verbatim_and_state_passes_through(tmp_path):
    fx = Fixture(tmp_path)
    fx.add_est('e1', coding_dna(), (100, 203), 61, state='excluded_heldout', tiles='true')
    fx.map[0]['molecule_id'] = 'f' * 64                   # not derivable from the members
    fx.add_hit('orthodb', 'e1', '4081_0:000001', 0, 60)
    _, rows, qc = fx.run()
    assert rows and {r['molecule_id'] for r in rows} == {'f' * 64}
    assert [(x['hsp_id'], x['molecule_id'], x['molecule_state'], x['train_tiles_excluded']) for x in fx.state] \
        == [(rows[0]['hsp_id'], 'f' * 64, 'excluded_heldout', 'true')]
    assert qc['by_source']['orthodb']['kept_hsps_excluded_heldout'] == 1
    assert not hasattr(m, 'molecule_map_from_closure') and not hasattr(m, 'molecule_id_for')


def test_molecule_map_sidecar_is_required_and_verified(tmp_path):
    fx = Fixture(tmp_path)
    fx.add_est('e1', coding_dna(), (100, 203), 61)
    fx.add_hit('orthodb', 'e1', '4081_0:000001', 0, 60)
    with pytest.raises(ValueError, match='sidecar'):
        fx.run(sidecar=False)
    (tmp_path / 'map.tsv.sha256').write_text('0' * 64 + '  map.tsv\n')
    with pytest.raises(ValueError, match='!= sidecar'):
        fx.run(sidecar=False)


MAP_ROW = dict(qname='q', member_uid='q', molecule_id='a' * 64, library='L', est_accession='A.1',
               molecule_state='train_track', train_tiles_excluded='false', superseded='false', accepted='true')


@pytest.mark.parametrize('bad', [dict(molecule_state='held'), dict(train_tiles_excluded='yes'),
                                 dict(molecule_id='abc'), dict(superseded='maybe'), dict(superseded=''),
                                 dict(accepted='maybe'), dict(accepted=''),
                                 dict(qname='q 1'), dict(molecule_state='train_track '),
                                 dict(library='Lib\tA'), dict(library='Lib\rA'), dict(library='Lib\nA'),
                                 dict(library=''), dict(library=' Lib'), dict(est_accession='A\t1')])
def test_molecule_map_rejects_invalid_rows(bad):
    with pytest.raises(ValueError):
        m.validate_molecule_map([dict(MAP_ROW, **bad)])


def test_molecule_map_accepts_a_multiword_library():
    lib = 'SAMN00739408 Vitis vinifera cv. Pinot Noir PN40024 leaves cDNA library'
    by = m.validate_molecule_map([dict(MAP_ROW, library=lib, est_accession='AB 1.1')])
    assert by['q']['library'] == lib and by['q']['superseded'] is False


def test_molecule_map_rejects_inconsistent_molecule_state():
    row = MAP_ROW
    with pytest.raises(ValueError, match='inconsistent'):
        m.validate_molecule_map([row, dict(row, qname='r', member_uid='r', molecule_state='excluded_heldout')])


def test_unmapped_qname_always_fails_and_has_no_override(tmp_path, capsys):
    fx = Fixture(tmp_path)
    fx.add_est('e1', coding_dna(), (100, 203), 61)
    fx.add_est('e2', coding_dna(8), (100, 203), 61, mapped=False)
    fx.add_hit('orthodb', 'e1', '4081_0:000001', 0, 60)
    fx.add_hit('orthodb', 'e2', '4081_0:000002', 0, 60)
    with pytest.raises(ValueError, match='1 HSPs have a qname absent'):
        fx.run()
    assert not (tmp_path / 'hsp.tsv').exists()
    with pytest.raises(SystemExit):                    # the old escape hatch no longer exists
        fx.run(extra=['--allow-unmapped', 'map_covers_training_members_only'])
    assert 'unrecognized arguments' in capsys.readouterr().err
    assert not hasattr(m, 'ALLOWED_UNMAPPED_REASONS')


def test_superseded_qnames_are_dropped_and_counted_only_when_the_map_says_so(tmp_path):
    fx = Fixture(tmp_path)
    fx.add_est('e1', coding_dna(), (100, 203), 61, molecule_members=['e0', 'e1'])
    fx.add_est('e0', coding_dna(8), (100, 203), 61, molecule_members=['e0', 'e1'], superseded='true')
    fx.add_hit('orthodb', 'e1', '4081_0:000001', 0, 60)
    fx.add_hit('orthodb', 'e0', '4081_0:000002', 0, 60)
    _, rows, qc = fx.run()
    assert {r['protein_id'] for r in rows} == {'4081_0:000001'}
    o = qc['by_source']['orthodb']
    assert o['superseded_qname_hsps'] == 1 and o.get('unmapped_qname_hsps', 0) == 0
    assert 'superseded_rule' in qc


def test_rejected_alignment_qnames_are_dropped_and_counted(tmp_path):
    """Finding 3 (A46 rule 6 / A18.5): a qname the map flags accepted=false (its C0 primary was rejected) is in its
    molecule for closure but supplies no observation -- its HSPs are dropped and counted, never projected."""
    fx = Fixture(tmp_path)
    fx.add_est('e1', coding_dna(), (100, 203), 61, molecule_members=['e1', 'e9'])
    fx.add_est('e9', coding_dna(8), (100, 203), 61, molecule_members=['e1', 'e9'], accepted='false')
    fx.add_hit('orthodb', 'e1', '4081_0:000001', 0, 60)
    fx.add_hit('orthodb', 'e9', '4081_0:000002', 0, 60)
    _, rows, qc = fx.run()
    assert {r['protein_id'] for r in rows} == {'4081_0:000001'}
    o = qc['by_source']['orthodb']
    assert o['rejected_alignment_qname_hsps'] == 1 and o.get('unmapped_qname_hsps', 0) == 0
    assert 'accepted_rule' in qc and 'accepted' in m.MOLECULE_MAP_COLUMNS and 'accepted' in m.MAP_ID_FIELDS
    prov = json.loads((tmp_path / 'hsp.tsv.provenance.json').read_text())
    assert prov['accepted_rule'] == m.ACCEPTED_RULE and 'accepted' in prov['molecule_map_columns']
    with pytest.raises(ValueError, match='molecule map needs columns'):      # a map without the column is refused
        map_path = tmp_path / 'map.tsv'
        lines = map_path.read_text().splitlines()
        map_path.write_text('\n'.join(l.rsplit('\t', 1)[0] for l in lines) + '\n')
        (tmp_path / 'map.tsv.sha256').write_text(f'{m.sha256_file(map_path)}  map.tsv\n')
        m.load_molecule_map(map_path)


def test_library_with_spaces_and_quotes_passes_through_verbatim(tmp_path):
    lib = 'SAMN00739408 Vitis vinifera cv. "Pinot Noir" PN40024 leaves cDNA library'
    fx = Fixture(tmp_path)
    fx.add_est('e1', coding_dna(), (100, 203), 61, library=lib)
    fx.add_hit('orthodb', 'e1', '4081_0:000001', 0, 60)
    _, rows, _ = fx.run()
    assert rows and {r['library'] for r in rows} == {lib}
    raw = (tmp_path / 'hsp.tsv').read_text().splitlines()[1].split('\t')
    assert raw[m.COLUMNS.index('library')] == lib           # no CSV quoting added


def test_projection_output_is_byte_deterministic_across_runs_and_input_order(tmp_path):
    fx = Fixture(tmp_path)
    fx.add_est('e1', coding_dna(), (100, 203), 61)
    fx.add_est('e2', coding_dna(9), (100, 203), 61, pos1=5001)
    fx.add_hit('orthodb', 'e1', '4081_0:000001', 0, 60)
    fx.add_hit('orthodb', 'e2', '4081_0:000002', 3, 50)
    fx.add_hit('swissprot', 'e1', 'sp|S|S_SOLLC', 0, 70)
    fx.run(out='a.tsv')
    fx.rows['orthodb'].reverse()
    fx.run(out='b.tsv')
    for suffix in ('', '.molecule_state.tsv'):
        assert (tmp_path / ('a.tsv' + suffix)).read_bytes() != b''
    # RAW13 order changes the raw13 sha in the provenance; rows must match modulo that one column.
    strip = lambda p: [l.rsplit('\t', 1)[0] for l in p.read_text().splitlines()]   # noqa: E731
    assert strip(tmp_path / 'a.tsv') == strip(tmp_path / 'b.tsv')
    assert (tmp_path / 'a.tsv.molecule_state.tsv').read_bytes() == (tmp_path / 'b.tsv.molecule_state.tsv').read_bytes()
    fx.rows['orthodb'].reverse()
    fx.run(out='c.tsv')
    assert (tmp_path / 'a.tsv').read_bytes() == (tmp_path / 'c.tsv').read_bytes()


def test_a19_predicate_is_importable_and_fails_closed(tmp_path):
    lineage = tmp_path / 'lineage.tsv'
    lineage.write_text('taxid\tlineage\n4081\t4081;4107;33090;131567;1\n3702\t3702;3701;33090;131567;1\n'
                       '999999\t999999;3702;3701;33090;131567;1\n')
    assert m.is_same_species_or_descendant(3702, 'Athaliana', lineage) is True
    assert m.is_same_species_or_descendant('999999', 'Athaliana', lineage) is True
    assert m.is_same_species_or_descendant(4081, 'Athaliana', lineage) is False
    assert m.use_frozen_lineage(lineage) == m.sha256_file(lineage)
    assert m.is_same_species_or_descendant('4081', 'Athaliana') is False
    for taxid, species in ((12345, 'Athaliana'), ('x', 'Athaliana'), (3702, 'Human')):
        with pytest.raises(ValueError):
            m.is_same_species_or_descendant(taxid, species)
    m._FROZEN_LINEAGE.clear()
    with pytest.raises(ValueError, match='no frozen lineage'):
        m.is_same_species_or_descendant(4081, 'Athaliana')


def test_plus_one_frameshift_is_exact_and_splits_the_interval(tmp_path):
    cd = coding_dna()
    dna = cd[:30] + 'A' + cd[30:-1]                 # one extra EST base after codon 10
    fx = Fixture(tmp_path)
    fx.add_est('e1', dna, (100, 203), 61)
    protein = translate(cd[:120])
    row = ['e1', str(L), '4081_0:000001', '40', '1', '121', '1', '40', '40', '1e-30', '200', '1',
           '10\\-30']
    fx.rows['orthodb'] = [row]
    fx.proteins['orthodb'] = {'4081_0:000001': ('4081_0:000001\t4081_0', protein)}
    _, rows, _ = fx.run()
    got = {pos: v for (_, pos), v in per_base(rows).items()}
    want = {1000 + u: ('+', u % 3) for u in range(30)}
    want.update({1000 + u: ('+', (u - 31) % 3) for u in range(31, 100)})
    want.update({1000 + 61 + u: ('+', (u - 31) % 3) for u in range(100, 121)})
    assert got == want and 1030 not in got
    assert [(r['genomic_start0'], r['genomic_end0']) for r in rows] == \
        [('1000', '1030'), ('1031', '1100'), ('1161', '1182')]


def test_noncanonical_est_is_dropped_and_counted(tmp_path):
    dna = coding_dna()
    fx = Fixture(tmp_path)
    fx.add_est('e1', dna[:200] + 'N' + dna[201:], (100, 203), 61)
    fx.add_hit('orthodb', 'e1', '4081_0:000001', 0, 40)
    fx.est['e1'] = dna[:200] + 'N' + dna[201:]
    _, rows, qc = fx.run()
    assert rows == [] and qc['by_source']['orthodb']['drop_noncanonical_dna_outside_hsp_span_only'] == 1
    assert qc['ests_noncanonical_whole_est_rule'] == 1 and qc['noncanonical_dna_scope'] == 'whole_est'


def test_noncanonical_inside_hsp_span_is_counted_separately(tmp_path):
    dna = coding_dna()
    bad = dna[:30] + 'N' + dna[31:]
    fx = Fixture(tmp_path)
    fx.add_est('e1', dna, (100, 203), 61)
    fx.add_hit('orthodb', 'e1', '4081_0:000001', 0, 40)   # protein from the clean sequence
    fx.add_hit('orthodb', 'e1', '4081_0:000002', 150, 40)
    fx.est['e1'] = bad
    _, rows, qc = fx.run()
    o = qc['by_source']['orthodb']
    assert rows == [] and o['drop_noncanonical_dna_in_hsp_span'] == 1
    assert o['drop_noncanonical_dna_outside_hsp_span_only'] == 1


def test_columns_match_the_c0_schema_contract():
    from transgenic.evidence import c0_schema
    assert tuple(m.COLUMNS) == c0_schema.HSP_COLUMN_NAMES
    assert m.MOLECULE_STATES == c0_schema.MOLECULE_STATES
    assert getattr(c0_schema, 'HSP_PRIMARY_KEY', ('hsp_id', 'interval_index')) == ('hsp_id', 'interval_index')


def test_diamond_masked_x_is_accepted_counted_and_recorded(tmp_path):
    # Production DIAMOND 2.1.24 (default tantan masking) writes 'X' into BTOP letters
    # where the real EST codon / protein residue is ordinary (MASKED_X_RULE, 2026-09-25).
    dna = coding_dna()
    fx = Fixture(tmp_path)
    fx.add_est('e1', dna, (100, 203), 61)
    row = fx.add_hit('orthodb', 'e1', '4081_0:000001', 0, 40)
    protein = fx.proteins['orthodb']['4081_0:000001'][1]
    row[12] = f'10X{protein[10]}19{protein[30]}X9'
    fx.add_hit('orthodb', 'e1', '4081_0:000002', 150, 40)     # unmasked control
    _, rows, qc = fx.run()
    got = {pos: v for (_, pos), v in per_base([r for r in rows if r['protein_id'] == '4081_0:000001']).items()}
    assert got == expected_labels(L, (100, 203), 61, 0, 0, 40, False)
    o = qc['by_source']['orthodb']
    assert o['kept_hsps'] == 2
    assert (o['masked_x_hsps'], o['masked_x_query_positions'], o['masked_x_subject_positions']) == (1, 1, 1)
    assert o['projected_hsps_with_masked_x'] == 1
    assert qc['masked_x_rule'] == m.MASKED_X_RULE and '2026-09-25' in m.MASKED_X_RULE
    prov = json.loads((tmp_path / 'hsp.tsv.provenance.json').read_text())
    assert prov['masked_x_rule'] == m.MASKED_X_RULE


def test_masked_x_over_a_stop_codon_fails_the_run(tmp_path):
    cd = coding_dna()
    dna = cd[:30] + 'TAA' + cd[33:]
    fx = Fixture(tmp_path)
    fx.add_est('e1', dna, (100, 203), 61)
    protein = translate(cd[:120])
    row = ['e1', str(L), '4081_0:000001', '40', '1', '120', '1', '40', '40', '1e-30', '200', '1',
           f'10X{protein[10]}29']
    fx.rows['orthodb'] = [row]
    fx.proteins['orthodb'] = {'4081_0:000001': ('4081_0:000001\t4081_0', protein)}
    with pytest.raises(ValueError, match='query codon truth'):
        fx.run()
