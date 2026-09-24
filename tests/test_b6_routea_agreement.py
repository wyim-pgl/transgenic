"""Route A/B per-base phase agreement QC (A46 rule 6) on the synthetic world.

Fix round 5 (findings 2 and 3): the script reads route B from the species' published PROTEIN C0 (the single
producer of the resolved, eligibility-aware HSP set) and binds every other input to that C0, so every case here
builds its own protein C0 from its route-B rows and its miniprot GFF (`build_case`).

Real loci of the world (test_b6_c0_world): the retained EST e2 (accession EA1.2, kind 'train') is placed over
Chr1 [10000,10800) = gA, a clean train gene; h1 ('held') over [40000,40500) = the strict held-out gene gH; v1 ('tte')
over [100000,100700) = gV (valid orthogroup -> train tiles only).  A protein whose miniprot CDS lies in gA is
train_track, in gH excluded_heldout, in gV train_track + train-tiles-excluded.  No excluded EST molecule of the world
has an accepted primary placement on clean bases (s1's primary is supplementary_or_chimeric), so a held-out HSP can
only ever lie on held-out bases here.
"""
import csv
import json

import pytest

import test_b6_c0_world as W
import test_b6_c0_build as TB
import b6_build_c0 as B
from revision.scripts.evidence import b6_routea_agreement as m

duckdb = pytest.importorskip("duckdb")

P, Q, R, Z, H, V, X = ('4577_0:P', '4577_0:Q', '4577_0:R', '4577_0:Z', '4577_0:H', '4577_0:V', '4577_0:X')
S, Y, Y2 = ('sp|S|S_SOLLC', 'sp|Y|Y_SOLLC', 'sp|Y2|Y2_SOLLC')


@pytest.fixture(scope="module")
def est(tmp_path_factory):
    """world, published EST C0 DB and the molecules by kind: {kind: (molecule_id, est_accession)}."""
    tmp = tmp_path_factory.mktemp("routea_est")
    world = W.build_corpus(tmp)
    W.write_est(tmp)
    args = W.cli_args(world, tmp)
    assert B.main(args) == 0
    db = args[args.index("--out") + 1]
    con = duckdb.connect(db, read_only=True)
    rows = con.execute("SELECT read_name, molecule_id, accession, molecule_state, train_tiles_excluded FROM molecule_member "
                       "WHERE read_name IN ('e2', 'h1', 'v1')").fetchall()
    con.close()
    by = {r[0]: r for r in rows}
    assert by["e2"][3:] == ("train_track", False) and by["h1"][3] == "excluded_heldout" and by["v1"][3:] == ("train_track", True)
    return dict(world=world, db=db, kinds={"train": by["e2"][1:3], "held": by["h1"][1:3], "tte": by["v1"][1:3]})


def cds_labels(strand, start0, end0, gff_phase):
    p = m.gff_cds_phase_at_start0(strand, start0, end0, gff_phase)
    anc = m.anchor(strand, start0, p)
    return {g: m.label(strand, anc, g) for g in range(start0, end0)}


def test_gff3_phase_real_miniprot_record_is_codon_continuous():
    # MP000001 (Athaliana.protein.gff.gz): CDS 6431865-6431989 phase 0, 6432069-6432405 phase 1.
    first = cds_labels('+', 6431864, 6431989, 0)
    second = cds_labels('+', 6432068, 6432405, 1)
    assert first[6431864] == 0 and first[6431988] == 1 and second[6432068] == 2


def test_gff3_phase_minus_strand_counts_from_the_high_end():
    first = cds_labels('-', 200, 300, 0)          # 5' base 299 -> codon position 0
    second = cds_labels('-', 100, 150, 2)         # 100 bases before: next base is position 1
    assert first[299] == 0 and first[200] == 0 and second[149] == 1 and second[147] == 0


def _hsp_rows(b_rows, kinds):
    """b_rows: (protein, source, kind, start0, end0, strand, phase) -> part-B contract rows (23 columns)."""
    rows = []
    for i, (protein, source, kind, s, e, strand, phase) in enumerate(b_rows):
        molecule, accession = kinds[kind]
        rows.append([f'h{i}', W.SPECIES, 'L1', molecule, accession, protein, source, '4577' if source == 'orthodb' else '39947',
                     '0', '150', '1', '1e-20', '200', '0', 'Chr1', str(s), str(e), strand, str(phase), '2.1.24',
                     'd' * 64, 'c' * 64, 'p' * 64])
    return rows


def _write_gff(d, gff):
    """gff: (mRNA id, protein, chrom, strand, [(start1, end1, gff_phase)], score, identity) in file order, written in the
    builder's miniprot layout (##PAF line before every mRNA; b6_build_c0.parse_miniprot / protein_accept.judge)."""
    lines = ['##gff-version 3']
    for mp, protein, chrom, strand, blocks, score, identity in gff:
        lines.append(TB._paf(protein, len(blocks) - 1))
        lines.append(f'{chrom}\tminiprot\tmRNA\t{blocks[0][0]}\t{blocks[-1][1]}\t{score}\t{strand}\t.\t'
                     f'ID={mp};Identity={identity};Target={protein} 1 180')
        lines += [f'{chrom}\tminiprot\tCDS\t{a}\t{b}\t{score}\t{strand}\t{p}\tParent={mp}' for a, b, p in blocks]
    ad = d / 'align' / W.SPECIES
    ad.mkdir(parents=True, exist_ok=True)
    path = ad / f'{W.SPECIES}.protein.gff'
    path.write_text('\n'.join(lines) + '\n')
    (ad / 'DONE').write_text(f'gff_md5          {W.md5(path)}\n')
    return path


def _accept_table(d, gff, ambiguous=None):
    """accept/<sp>.alignments.tsv and mapping_ambiguous.txt as protein_accept.py writes them: verdict accepted iff
    identity >= 0.30 (the other judge criteria hold by construction); ambiguous = equal-best score per protein."""
    best = {}
    for mp, protein, _c, _s, _b, score, _i in gff:
        best.setdefault(protein, []).append(score)
    if ambiguous is None:
        ambiguous = sorted(p for p, scores in best.items() if scores.count(max(scores)) > 1)
    (d / 'acc.tsv').write_text('alignment_id\tprotein\torganism\tverdict\n' + ''.join(
        f'{mp}\t{protein}\t4577_0\t{"accepted" if identity >= 0.30 else "rejected"}\n'
        for mp, protein, _c, _s, _b, _score, identity in gff))
    (d / 'amb.txt').write_text(''.join(p + '\n' for p in ambiguous))
    return d / 'acc.tsv', d / 'amb.txt'


def build_case(tmp_path, est, b_rows, gff, name='case', ambiguous=None):
    """Route-B TSV (+ sidecar, provenance) and miniprot GFF -> a published protein C0 of this species; returns the
    agreement argv pieces.  `gff` items: (mRNA id, protein, chrom, strand, [(start1, end1, phase)], score, identity)."""
    d = tmp_path / name
    d.mkdir()
    est_pair = (est['db'], None)
    hsp = TB.write_hsp(d, TB._mols(est_pair), rows=_hsp_rows(b_rows, est['kinds']), state=TB.est_state(est_pair))
    gff_path = _write_gff(d, gff)
    acc, amb = _accept_table(d, gff, ambiguous)
    pargs = W.cli_args(est['world'], d, source='protein', extra=TB._protein_extra(d, est_pair, hsp))
    assert B.main(pargs) == 0, name
    pdb = pargs[pargs.index('--out') + 1]
    return dict(dir=d, protein_c0=pdb, est_c0=est['db'], route_b=hsp, gff=gff_path, accepted=acc, ambiguous=amb)


def argv_of(case, out='zero.tsv', **override):
    c = dict(case, **override)
    d = case['dir']
    return ['--species', 'Athaliana', '--route-b', str(c['route_b']), '--miniprot-gff', str(c['gff']),
            '--accepted', str(c['accepted']), '--mapping-ambiguous', str(c['ambiguous']), '--est-c0', str(c['est_c0']),
            '--protein-c0', str(c['protein_c0']), '--out-intervals', str(d / out), '--qc', str(d / (out + '.qc.json'))]


def run_case(case, out='zero.tsv', **override):
    m.main(argv_of(case, out, **override))
    d = case['dir']
    with open(d / out) as f:
        zero = list(csv.DictReader(f, delimiter='\t'))
    return json.loads((d / (out + '.qc.json')).read_text()), zero


def _zero_set(zero):
    return {(z['reason'], z['chrom'], z['start0'], z['end0']) for z in zero}


def test_counts_and_disagreement_zeroing(tmp_path, est):
    b_rows = [(P, 'orthodb', 'train', 10000, 10100, '+', 0),     # A covers [10050,10200) in frame
              (Q, 'orthodb', 'train', 10300, 10330, '+', 0),     # A says the next frame: disagree
              (R, 'orthodb', 'train', 10300, 10330, '+', 0),     # agrees, but the base still zeroes
              (S, 'swissprot', 'train', 10600, 10660, '-', 1),   # no route A for Swiss-Prot
              (Z, 'orthodb', 'train', 10700, 10730, '+', 0)]     # A43-ambiguous: A ignored
    gff = [('MP1', P, 'Chr1', '+', [(10051, 10200, 1)], 500, 0.9),   # position 2 at 10050 == B's (0+50)%3
           ('MP2', Q, 'Chr1', '+', [(10301, 10330, 2)], 500, 0.9),   # position 1 at 10300 != B's 0
           ('MP3', R, 'Chr1', '+', [(10301, 10330, 0)], 500, 0.9),
           ('MP4', Z, 'Chr1', '+', [(10701, 10730, 1)], 500, 0.9),   # equal-best with MP4b: mapping_ambiguous
           ('MP4b', Z, 'Chr1', '+', [(170001, 170030, 1)], 500, 0.9),
           ('MP5', P, 'Chr1', '+', [(10401, 10430, 0)], 400, 0.2)]   # identity below 0.30: rejected, not used
    case = build_case(tmp_path, est, b_rows, gff)
    qc, zero = run_case(case)
    assert qc['bases'] == dict(agree=50, disagree=30, B_only=50 + 60 + 30, A_only=100)
    assert zero == [dict(species='Athaliana', chrom='Chr1', start0='10300', end0='10330',
                         action='zero_all_route_b_phase_channels', reason='route_a_b_disagreement')]
    assert qc['cross_source']['bases_labelled_by_both'] == 0 and qc['cross_source']['disagreement_intervals'] == 0
    assert qc['bases_by_route_b_source']['swissprot'] == dict(A_only=0, B_only=60, agree=0, disagree=0)
    assert qc['route_a']['A_mapping_ambiguous_alignments_removed'] == 2
    assert qc['route_a']['A_alignments_used'] == 3
    o, sp = qc['by_route_b_source']['orthodb'], qc['by_route_b_source']['swissprot']
    assert o['bases'] == dict(agree=50, disagree=30, B_only=50 + 30, A_only=100)
    assert (o['disagreement_intervals'], o['disagreement_bases'], o['route_b_proteins']) == (1, 30, 4)
    assert (sp['disagreement_intervals'], sp['route_a_proteins'], sp['route_b_rows']) == (0, 0, 1)
    assert qc['route_b_leakage_filter'] == {'rows_used': 5} and qc['gff_alignments_bound'] == 6
    assert set(qc['inputs_bound_to_protein_c0']) == {'protein_c0', 'est_c0', 'hsp', 'hsp_molecule_state', 'alignment:protein'}


def test_per_source_breakdown_has_both_sources_and_attributes_disagreement(tmp_path, est):
    b_rows = [(P, 'swissprot', 'train', 10000, 10030, '+', 0)]       # a Swiss-Prot id miniprot also saw
    gff = [('MP1', P, 'Chr1', '+', [(10001, 10030, 1)], 500, 0.9)]
    case = build_case(tmp_path, est, b_rows, gff)
    qc, zero = run_case(case)
    assert qc['by_route_b_source']['orthodb'] == dict(
        bases=dict(A_only=0, B_only=0, agree=0, disagree=0), route_b_rows=0, route_b_proteins=0,
        route_a_proteins=0, route_a_cds_blocks=0, disagreement_intervals=0, disagreement_bases=0)
    assert qc['by_route_b_source']['swissprot']['disagreement_bases'] == 30
    assert [(z['start0'], z['end0']) for z in zero] == [('10000', '10030')]


def test_same_protein_on_opposite_strands_disagrees(tmp_path, est):
    b_rows = [(P, 'orthodb', 'train', 10000, 10030, '+', 0), (Q, 'orthodb', 'train', 10300, 10330, '-', 2)]
    gff = [('MP1', P, 'Chr1', '-', [(10001, 10030, 0)], 500, 0.9),    # same protein, other strand
           ('MP2', R, 'Chr1', '+', [(10301, 10330, 0)], 500, 0.9)]    # other protein, other strand: no pairing
    case = build_case(tmp_path, est, b_rows, gff)
    qc, zero = run_case(case)
    assert qc['bases'] == dict(agree=0, disagree=30, B_only=30, A_only=0)
    assert [(z['start0'], z['end0']) for z in zero] == [('10000', '10030')]


def test_mismatched_gff_target_fails_closed(tmp_path, est):
    case = build_case(tmp_path, est, [(P, 'orthodb', 'train', 10000, 10030, '+', 0)],
                      [('MP1', Q, 'Chr1', '+', [(10001, 10030, 0)], 500, 0.9)])
    (case['dir'] / 'acc.tsv').write_text('alignment_id\tprotein\torganism\tverdict\nMP1\t' + P + '\t4577_0\taccepted\n')
    with pytest.raises(ValueError, match='Target'):
        m.main(argv_of(case))


def test_cross_source_disagreement_is_zeroed_agreement_is_not():
    # OrthoDB and Swiss-Prot label 100-130 with different frames on '+'; 200-230 with the same frame.
    b = [('P1', 'orthodb', 'Chr1', 100, 130, '+', 0), ('Q1', 'swissprot', 'Chr1', 100, 130, '+', 1),
         ('P2', 'orthodb', 'Chr1', 200, 230, '+', 0), ('Q2', 'swissprot', 'Chr1', 200, 230, '+', 0),
         ('P3', 'orthodb', 'Chr1', 300, 330, '+', 0)]
    rows = []
    for protein, source, chrom, s, e, strand, phase in b:
        rows.append((chrom, strand, protein, s, e, m.anchor(strand, s, phase), source))
    both, zero = m.compare_sources(rows)
    assert both == 60
    assert dict(zero) == {'Chr1': [[100, 130]]}


def test_cross_source_same_frame_different_start_agrees():
    # Same reading frame reached from different starts (anchor equality) is agreement.
    rows = [('Chr1', '+', P, 100, 160, m.anchor('+', 100, 0), 'orthodb'),
            ('Chr1', '+', Q, 103, 130, m.anchor('+', 103, 0), 'swissprot')]
    both, zero = m.compare_sources(rows)
    assert both == 27 and not zero


def test_excluded_est_molecule_hsps_take_part_in_no_zeroing_decision(tmp_path, est):
    """A46 rule 4: rows of an excluded_heldout EST molecule (h1, on the held-out gene) are flagged leakage_excluded by
    the protein C0 and dropped before any comparison; the protein H aligned there is excluded_heldout too (rule 3), so
    its route-A alignment is dropped as well.  Control rows on the retained molecule e2 zero both interval kinds."""
    b = [(H, 'orthodb', 'held', 40000, 40030, '+', 1),                                   # would disagree with route A
         (Q, 'orthodb', 'held', 40300, 40330, '+', 0), (S, 'swissprot', 'held', 40300, 40330, '+', 1),   # cross-source
         (P, 'orthodb', 'train', 10000, 10030, '+', 1),                                  # control: A/B disagreement
         (X, 'orthodb', 'train', 10300, 10330, '+', 0), (Y, 'swissprot', 'train', 10300, 10330, '+', 1)]  # control: cross
    gff = [('MP1', H, 'Chr1', '+', [(40001, 40030, 0)], 500, 0.9), ('MP2', P, 'Chr1', '+', [(10001, 10030, 0)], 500, 0.9)]
    case = build_case(tmp_path, est, b, gff)
    qc, zero = run_case(case)
    assert _zero_set(zero) == {('route_a_b_disagreement', 'Chr1', '10000', '10030'),
                               ('route_b_cross_source_disagreement', 'Chr1', '10300', '10330')}
    assert qc['route_b_leakage_filter'] == {'rows_dropped_excluded_heldout': 3, 'rows_used': 3}
    assert qc['route_a'] == {'A_leakage_excluded_alignments_removed': 1, 'A_alignments_used': 1, 'A_cds_blocks': 1}
    con = duckdb.connect(case['protein_c0'], read_only=True)
    flags = con.execute("SELECT hsp_id, leakage_excluded FROM est_protein_hsp ORDER BY hsp_id").fetchall()
    con.close()
    assert flags == [('h0', True), ('h1', True), ('h2', True), ('h3', False), ('h4', False), ('h5', False)]


def test_train_tiles_excluded_est_molecule_hsps_are_dropped(tmp_path, est):
    """A train-tiles-excluded EST molecule (v1, valid orthogroup) row is dropped -- the zero list has no tile scope --
    and the protein V placed there is train-tiles-excluded too, so its route-A alignment goes with it."""
    b = [(V, 'orthodb', 'tte', 100000, 100030, '+', 1), (P, 'orthodb', 'train', 10000, 10030, '+', 1)]
    gff = [('MP1', V, 'Chr1', '+', [(100001, 100030, 0)], 500, 0.9), ('MP2', P, 'Chr1', '+', [(10001, 10030, 0)], 500, 0.9)]
    case = build_case(tmp_path, est, b, gff)
    qc, zero = run_case(case)
    assert _zero_set(zero) == {('route_a_b_disagreement', 'Chr1', '10000', '10030')}
    assert qc['route_b_leakage_filter'] == {'rows_dropped_train_tiles_excluded': 1, 'rows_used': 1}
    assert qc['route_a'] == {'A_train_tiles_excluded_alignments_removed': 1, 'A_alignments_used': 1, 'A_cds_blocks': 1}


def test_excluded_protein_query_takes_part_in_no_zeroing_decision(tmp_path, est):
    """Finding 2 (rule 3 'one unit everywhere it appears'; rule 6 'protein queries touching excluded intervals are
    removed molecule-wide'): H has one placement on the held-out gene and one on the clean gene gA, so it is
    excluded_heldout as a whole; V is placed on gV (train-tiles-excluded).  H's clean route-A alignment and its
    route-B row on e2 (flagged protein_query by the protein C0), and V's row (train_tiles_excluded via the protein),
    are dropped before any comparison; the train_track query P still zeroes its disagreement."""
    gff = [('MP1', H, 'Chr1', '+', [(40001, 40030, 0)], 500, 0.9), ('MP2', H, 'Chr1', '+', [(10001, 10030, 0)], 400, 0.9),
           ('MP3', V, 'Chr1', '+', [(100001, 100030, 0)], 500, 0.9), ('MP4', P, 'Chr1', '+', [(10601, 10630, 0)], 500, 0.9)]
    b = [(H, 'orthodb', 'train', 10000, 10030, '+', 1),                                  # would disagree with MP2
         (V, 'orthodb', 'train', 10300, 10330, '+', 0), (S, 'swissprot', 'train', 10300, 10330, '+', 1),   # would cross-disagree
         (P, 'orthodb', 'train', 10600, 10630, '+', 1)]                                  # control -> zero
    case = build_case(tmp_path, est, b, gff)
    qc, zero = run_case(case)
    assert _zero_set(zero) == {('route_a_b_disagreement', 'Chr1', '10600', '10630')}
    assert qc['route_a'] == {'A_leakage_excluded_alignments_removed': 2, 'A_train_tiles_excluded_alignments_removed': 1,
                             'A_alignments_used': 1, 'A_cds_blocks': 1}
    assert qc['route_b_leakage_filter'] == {'rows_dropped_protein_query_excluded': 1,
                                            'rows_dropped_protein_query_train_tiles_excluded': 1, 'rows_used': 2}
    # Without the protein-C0 flags the same rows would have zeroed both intervals.
    b_all = [('Chr1', st, p, s, e, m.anchor(st, s, ph), src) for p, src, _k, s, e, st, ph in b]
    a_rows = [('Chr1', '+', H, 10000, 10030, m.anchor('+', 10000, m.gff_cds_phase_at_start0('+', 10000, 10030, 0)))]
    assert dict(m.compare(a_rows, b_all)[1]) == {'Chr1': [[10000, 10030]]} and dict(m.compare_sources(b_all)[1]) == {'Chr1': [[10300, 10330]]}


def test_inputs_are_bound_to_the_protein_c0(tmp_path, est):
    """Finding 3: an older protein C0 presented with other files (GFF, route-B TSV, sidecar, EST C0) or an accept table /
    ambiguous list that does not describe the C0's alignments is refused before anything is written."""
    b = [(P, 'orthodb', 'train', 10000, 10030, '+', 1), (Z, 'orthodb', 'train', 10700, 10730, '+', 0)]
    gff = [('MP1', P, 'Chr1', '+', [(10001, 10030, 0)], 500, 0.9), ('MP2', P, 'Chr1', '+', [(10401, 10430, 0)], 400, 0.2),
           ('MP3', Z, 'Chr1', '+', [(10701, 10730, 1)], 500, 0.9), ('MP3b', Z, 'Chr1', '+', [(170001, 170030, 1)], 500, 0.9)]
    case = build_case(tmp_path, est, b, gff)
    d = case['dir']
    run_case(case, out='ok.tsv')                                       # the bound set passes
    other = d / 'other.gff'
    other.write_text(case['gff'].read_text() + TB._paf(Q, 0) + '\n' + '\n'.join(
        TB._mrna(9, Q, 'Chr1', '+', 500, [(10301, 10330)])) + '\n')     # a newer miniprot GFF with one more protein
    with pytest.raises(ValueError, match='--miniprot-gff .* is not the input the protein C0'):
        m.main(argv_of(case, 'a.tsv', gff=other))
    tsv2 = d / 'hsp2.tsv'
    tsv2.write_bytes(case['route_b'].read_bytes() + b'\n')
    with pytest.raises(ValueError, match='--route-b .* is not the input the protein C0'):
        m.main(argv_of(case, 'b.tsv', route_b=tsv2))
    side = m.state_path(case['route_b'])
    original = side.read_bytes()
    side.write_bytes(original.replace(b'train_track', b'excluded_heldout', 1))
    with pytest.raises(ValueError, match='molecule_state.tsv .* is not the input the protein C0'):
        m.main(argv_of(case, 'c.tsv'))
    side.unlink()
    with pytest.raises(ValueError, match='missing <route-b>.molecule_state.tsv'):
        m.main(argv_of(case, 'd.tsv'))
    side.write_bytes(original)
    with pytest.raises(ValueError, match='--est-c0 .* is not the input the protein C0'):
        m.main(argv_of(case, 'e.tsv', est_c0=case['protein_c0']))
    acc = case['accepted'].read_text()
    case['accepted'].write_text(acc.replace('\trejected\n', '\taccepted\n'))        # MP2's verdict flipped
    with pytest.raises(ValueError, match='MP2: accepted=False in the protein C0 but the accept table says'):
        m.main(argv_of(case, 'f.tsv'))
    case['accepted'].write_text(acc)
    case['ambiguous'].write_text('')                                              # Z's ambiguity hidden
    with pytest.raises(ValueError, match='MP3: mapping_ambiguous True in the protein C0 but not in'):
        m.main(argv_of(case, 'g.tsv'))
    case['ambiguous'].write_text(Z + '\n')
    (d / 'short.gff').write_text('\n'.join(case['gff'].read_text().splitlines()[:-4]) + '\n')
    with pytest.raises(ValueError, match='is not the input the protein C0'):
        m.main(argv_of(case, 'h.tsv', gff=d / 'short.gff'))
    argv = argv_of(case, 'i.tsv')
    with pytest.raises(SystemExit):                                    # --protein-c0 is required
        m.main([a for a in argv if a not in ('--protein-c0', case['protein_c0'])])
    with pytest.raises(B.C0Error):                                     # the EST C0 is not a protein C0
        m.main(argv_of(case, 'j.tsv', protein_c0=case['est_c0']))
    assert not any((d / f'{n}.tsv').exists() for n in 'abcdefghij')     # every refusal happened before any output


def test_zero_intervals_are_byte_deterministic(tmp_path, est):
    b = [('P1', 'orthodb', 'train', 10000, 10030, '+', 0), ('Q1', 'swissprot', 'train', 10000, 10030, '+', 1),
         ('P2', 'orthodb', 'train', 10600, 10630, '+', 0), ('Q2', 'swissprot', 'train', 10600, 10630, '+', 2),
         (P, 'orthodb', 'train', 10300, 10330, '+', 1)]
    gff = [('MP1', P, 'Chr1', '+', [(10301, 10330, 0)], 500, 0.9)]
    one = build_case(tmp_path, est, b, gff, name='one')
    two = build_case(tmp_path, est, list(reversed(b)), gff, name='two')
    run_case(one)
    run_case(two)
    assert (one['dir'] / 'zero.tsv').read_bytes() == (two['dir'] / 'zero.tsv').read_bytes()
    assert len(list(csv.DictReader(open(one['dir'] / 'zero.tsv'), delimiter='\t'))) == 3
