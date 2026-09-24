"""Bounded per-row contract; synthetic truth, no engine or production gate."""
import hashlib
import importlib
import itertools

import pytest

from revision.scripts.evidence.est_hsp_expanded import ENCODE, rc


def api():
    # Assertion gives a useful RED rather than a collection/import error.
    from importlib.util import find_spec
    name = 'revision.scripts.evidence.est_hsp_contract'
    assert find_spec(name) is not None, 'approved HSP contract not implemented'
    return importlib.import_module(name)


def fixture(qg, sg, offset=0, reverse=False):
    dna, protein, btop, ops, translated = 'A' * offset, '', [], [], ''
    run = 0
    for q, s in zip(qg, sg):
        if q == s and q not in '-/\\':
            run += 1
        else:
            if run:
                btop.append(str(run))
                run = 0
            btop.append(q + s)
        if q in '/\\':
            # A-rich K codons make both shift directions independently replayable.
            dna = dna + 'A' if q == '\\' else dna[:-1]
            ops.append(q)
            continue
        ops.append('D' if q == '-' else 'I' if s == '-' else 'M')
        if q != '-':
            dna += ENCODE[q]
            translated += q
        if s != '-':
            protein += s
    if run:
        btop.append(str(run))
    length = sum(op in 'MID' for op in ops)
    qs, qe = offset + 1, len(dna)
    if reverse:
        qs, qe = len(dna) - qs + 1, len(dna) - qe + 1
        dna = rc(dna)
    row = ['q', str(len(dna)), 's', str(len(protein)), str(qs), str(qe), '1',
           str(len(protein)), str(length), '1e-20', '100',
           str((offset + 1) * (-1 if reverse else 1)), ''.join(btop),
           ''.join(str(len(list(g))) + op for op, g in itertools.groupby(ops)),
           qg, sg, translated]
    return row, dna, protein


@pytest.mark.parametrize('n', [29, 30, 31])
@pytest.mark.parametrize('offset', [0, 1, 2])
@pytest.mark.parametrize('reverse', [False, True])
def test_threshold_all_frames_and_original_codons(n, offset, reverse):
    row, dna, protein = fixture('K' * n, 'K' * n, offset, reverse)
    result = api().classify_rich17(row, dna, protein)
    assert result['accepted_by_hsp_rule'] is (n >= 30)
    assert result['length_contract'] == 'DIAMOND_REPORTED_HSP_M_PLUS_I_PLUS_D_GE30'
    assert result['diagnostic_disposition'] == ('HSP_RULE_ACCEPTED' if n >= 30 else 'BELOW_HSP_LENGTH_30')
    assert result['scope'] == 'DIAGNOSTIC_ONLY_NOT_PRODUCTION_TRAINING_OR_LINEAGE_A11_CERTIFICATION'
    assert result['identities']['raw_row_sha256'] == hashlib.sha256('\t'.join(row).encode()).hexdigest()
    assert result['identities']['dna_sha256'] == hashlib.sha256(dna.encode()).hexdigest()
    assert result['identities']['protein_sha256'] == hashlib.sha256(protein.encode()).hexdigest()
    for i, event in enumerate(result['codon_events']):
        assert event == dict(q0=len(dna)-1-offset-3*i if reverse else offset+3*i,
                             length=3, direction=-1 if reverse else 1, c0=0,
                             kind='M', subject_aa_index=i)
    assert api().adapt_raw13(row[:13], dna, protein) == row


@pytest.mark.parametrize('qg,sg,accepted', [
    ('K'*29+'-', 'K'*30, True),  # D30: query only29, whole30
    ('K'*29, 'K'*28+'-', False), # I29: paired28, whole29
    ('K'*30, 'K'*29+'-', True),
    ('K'*15+'\\'+'K'*15, 'K'*15+'-'+'K'*15, True),
    ('K'*14+'\\'+'K'*15, 'K'*14+'-'+'K'*15, False),
    ('K'*15+'/'+'K'*15, 'K'*15+'-'+'K'*15, True),
    ('K'*15+'R'+'K'*14, 'K'*30, True),
])
@pytest.mark.parametrize('reverse', [False, True])
def test_whole_hsp_indels_mismatch_and_shift(qg, sg, accepted, reverse):
    row, dna, protein = fixture(qg, sg, reverse=reverse)
    result = api().classify_rich17(row, dna, protein)
    assert result['accepted_by_hsp_rule'] is accepted
    assert len(result['codon_events']) == sum(q not in '-/\\' for q in qg)
    assert api().adapt_raw13(row[:13], dna, protein) == row
    if sg.endswith('-'):
        assert result['codon_events'][-1]['kind'] == 'I'
        assert result['codon_events'][-1]['subject_aa_index'] is None
    for shift in result['frameshifts']:
        assert shift['effect'] == ('omitted' if shift['delta'] == 1 else 'reused')
        oriented = shift['at'] if shift['delta'] == 1 else shift['at']-1
        assert shift['q0'] == (len(dna)-1-oriented if reverse else oriented)


@pytest.mark.parametrize('index,value', [(0,''),(0,'q x'),(2,''),(1,'89'),(3,'31'),(4,'0'),(5,'89'),(6,'0'),(7,'29'),(8,'31'),(8,'0'),(9,'nan'),(9,'inf'),(9,'0.01'),(10,'inf'),(10,'0'),(11,'0'),(12,'0'),(12,'999999999999'),(12,'30X'),(12,'KK29'),(12,'--30'),(13,'29M'),(14,'R'+'K'*29),(15,'R'+'K'*29),(16,'R'+'K'*29)])
def test_reject_invalid_rich(index, value):
    row, dna, protein = fixture('K'*30, 'K'*30)
    row[index] = value
    with pytest.raises(ValueError):
        api().classify_rich17(row, dna, protein)
    if index < 13:
        with pytest.raises(ValueError):
            api().adapt_raw13(row[:13], dna, protein)


@pytest.mark.parametrize('dna,protein', [('A'*89,'K'*30),('N'+'A'*89,'K'*30),('a'*90,'K'*30),('A'*90,'X'+'K'*29),('A'*90,'K'*29),('A'*90,'K'*29+' ')])
def test_reject_bad_actual_sequences(dna, protein):
    row, _, _ = fixture('K'*30, 'K'*30)
    for function in (api().classify_rich17, api().adapt_raw13):
        with pytest.raises(ValueError):
            function(row if function.__name__ == 'classify_rich17' else row[:13], dna, protein)


@pytest.mark.parametrize('reported', [29, 30, 31])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('shifts', ['/', '\\', '/\\'])
def test_explicit_midf_formula_boundaries(reported, reverse, shifts):
    n = reported - len(shifts)
    row, dna, protein = fixture('K'*10 + shifts + 'K'*(n-10),
                               'K'*10 + '-'*len(shifts) + 'K'*(n-10), reverse=reverse)
    row[8] = str(reported)
    mode = 'DIAMOND_REPORTED_HSP_M_PLUS_I_PLUS_D_PLUS_F_GE30'
    result = api().classify_rich17(row, dna, protein, length_contract=mode)
    assert result['accepted_by_hsp_rule'] is (reported >= 30)
    assert result['counts']['F'] == len(shifts)
    assert api().adapt_raw13(row[:13], dna, protein, length_contract=mode) == row
    with pytest.raises(ValueError):
        api().classify_rich17(row, dna, protein)
    row[8] = str(n)
    with pytest.raises(ValueError):
        api().adapt_raw13(row[:13], dna, protein, length_contract=mode)


@pytest.mark.parametrize('reverse', [False, True])
def test_midf_gaps_substitution_and_x_remain_exact(reverse):
    row, dna, protein = fixture('K'*10+'\\'+'K'*10+'R'+'-'+'K'*8,
                               'K'*10+'-'+'K'*10+'K'+'K'+'K'*7+'-', reverse=reverse)
    row[8] = '31'
    mode = 'DIAMOND_REPORTED_HSP_M_PLUS_I_PLUS_D_PLUS_F_GE30'
    result = api().classify_rich17(row, dna, protein, length_contract=mode)
    assert result['counts'] == {'M': 28, 'I': 1, 'D': 1, 'F': 1}
    assert api().adapt_raw13(row[:13], dna, protein, length_contract=mode) == row
    with pytest.raises(ValueError):
        api().adapt_raw13(row[:13], dna, 'X'+protein[1:], length_contract=mode)
    row[5] = str(int(row[5])+1)
    with pytest.raises(ValueError):
        api().classify_rich17(row, dna, protein, length_contract=mode)


@pytest.mark.parametrize('mode', ['unknown', '', None])
def test_unknown_length_mode_rejected_even_without_shifts(mode):
    row, dna, protein = fixture('K'*30, 'K'*30)
    for fn, fields in ((api().classify_rich17, row), (api().adapt_raw13, row[:13])):
        with pytest.raises(ValueError, match='length contract'):
            fn(fields, dna, protein, length_contract=mode)


def test_raw_equal_runs_checked_against_dna_and_source_hash():
    row, dna, protein = fixture('K'*30, 'K'*30)
    with pytest.raises(ValueError):
        api().adapt_raw13(row[:13], 'GCC'+dna[3:], protein)
    with pytest.raises(ValueError):
        api().classify_rich17(row, dna, protein, expected_identities={'dna_sha256': '0'*64})
    with pytest.raises(ValueError):
        api().classify_rich17(row, dna, protein, expected_identities={'unknown': '0'*64})
    for bad in ([], row[:-1], ['']*17):
        with pytest.raises(ValueError):
            api().classify_rich17(bad, dna, protein)
    with pytest.raises(ValueError):
        api().adapt_raw13(row, dna, protein)
