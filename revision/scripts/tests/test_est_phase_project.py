"""Hand-computed geometry only: no HSP acceptance or training claim."""
import hashlib
import importlib.util
from pathlib import Path
import pytest

P = Path(__file__).parents[1] / 'evidence' / 'est_phase_project.py'
spec = importlib.util.spec_from_file_location('geometry', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def inputs(cigar='2S4M4N5M1S', flag=0, frame=3, start=3, end=11, length=12, ts='+'):
    seq = ('ACGT' * (length // 4 + 1))[:length]
    identity = dict(member='piece_part1', source_sha256='a'*64,
                    source_start0=10, source_end0=10+length,
                    sequence_sha256=hashlib.sha256(seq.encode()).hexdigest())
    p = dict(identity=identity.copy(), sequence=seq, cigar=cigar, flag=flag,
             pos1=101, contig='chr', contig_length=1000, seq=None, ts=ts)
    h = dict(identity=identity.copy(), id='h1', qstart=start, qend=end, frame=frame,
             producer_contract_sha256='b'*64, label_convention='c0_first_codon_base')
    e = [dict(q0=start-1, length=abs(end-start)+1, direction=1 if frame>0 else -1, c0=0)]
    return p,h,e


def expand(result):
    return {g: (s['strand'], (s['c0'] + s['step']*(g-s['start0'])) % 3)
            for s in result['segments'] for g in range(s['start0'], s['end0'])}


@pytest.mark.parametrize('n', [3,4,5])
@pytest.mark.parametrize('flag,frame,start,end,ts,minus',
                         [(0,3,3,11,'+',False),(0,-2,11,3,'-',True),
                          (16,2,2,10,'+',True),(16,-3,10,2,'-',False)])
def test_f1_f2_f3_introns(n,flag,frame,start,end,ts,minus):
    p,h,e=inputs(f'2S4M{n}N5M1S',flag,frame,start,end,ts=ts)
    r=m.project_codons(h,e,p)
    low=[2,1,0,2] if minus else [0,1,2,0]
    high=[1,0,2,1,0] if minus else [1,2,0,1,2]
    strand='-' if minus else '+'
    assert expand(r)==dict(zip(list(range(100,104))+list(range(104+n,109+n)),
                               [(strand,c) for c in low+high]))
    anchors=[s['anchor'] for s in sorted(r['segments'],key=lambda s:s['start0'])]
    assert anchors == ([0,(n-3)%3] if minus else [1,(1+n)%3])
    split=(105+n,104+n,103) if minus else (103,104+n,105+n)
    assert [expand(r)[g][1] for g in split]==[0,1,2]
    assert r['hsp_coordinates']==dict(qstart=start,qend=end,frame=frame)
    assert r['placement_coordinates']['flag']==flag
    assert r['manifest']['training_eligible'] is False
    assert r['manifest']['channels']==['+0','+1','+2','-0','-1','-2']
    assert r['manifest']['input_label_convention']=='c0_first_codon_base'


@pytest.mark.parametrize('cigar,expected',[
    ('3M1I5M',{**dict(zip(range(100,103),[0,1,2])),**dict(zip(range(103,108),[1,2,0,1,2]))}),
    ('3M1D6M',{**dict(zip(range(100,103),[0,1,2])),**dict(zip(range(104,110),[0,1,2,0,1,2]))}),
    ('3M1P6M',dict(zip(range(100,109),[0,1,2]*3))),
    ('3=3X3M',dict(zip(range(100,109),[0,1,2]*3)))])
@pytest.mark.parametrize('flag',[0,16])
def test_f4_operations(cigar,expected,flag):
    p,h,e=inputs(cigar,flag,1,1,9,9)
    if flag:
        h.update(frame=-1,qstart=9,qend=1); e=[dict(q0=8,length=9,direction=-1,c0=0)]; p['ts']='-'
    assert expand(m.project_codons(h,e,p))=={g:('+',c) for g,c in expected.items()}


def test_hardclip_sequence_identity():
    p,h,e=inputs('2H4M4N5M1H')
    p['seq']=p['sequence'][2:-1]
    assert expand(m.project_codons(h,e,p))==expand(m.project_codons(*reorder(inputs())))
    p['seq']='A'*9
    with pytest.raises(ValueError,match='SEQ identity'): m.project_codons(h,e,p)


def reorder(x):
    p,h,e=x
    return h,e,p


def test_f7_boundaries_offsets_and_clips():
    p,h,e=inputs()
    r=m.project_codons(h,e,p)
    assert {g:v[1] for g,v in expand(r).items() if 102<=g<110}=={102:2,103:0,108:1,109:2}
    assert 112 in expand(r) and 113 not in expand(r)
    assert r['segments'][0]['source_q0']==12
    h.update(qstart=1,qend=1,frame=1)
    e=[dict(q0=0,length=1,direction=1,c0=2)]
    assert m.project_codons(h,e,p)['reasons']==['motif_strand_available','no_mapped_codon_bases']
    h.update(qstart=3,qend=3,frame=3); e[0]['q0']=2
    assert expand(m.project_codons(h,e,p))=={100:('+',2)}


def test_f8_rc_and_translation_invariance():
    p,h,e=inputs()
    original=expand(m.project_codons(h,e,p))
    rc={113-g:('-' if s=='+' else '+',c) for g,(s,c) in original.items()}
    assert rc==dict(zip([1,2,3,4,5,10,11,12,13],[('-',c) for c in [2,1,0,2,1,0,2,1,0]]))
    assert {113-g:('+' if s=='-' else '-',c) for g,(s,c) in rc.items()}==original
    reflected = dict(p, cigar='1S5M4N4M2S', flag=16, pos1=102)
    assert {g-100:v for g,v in expand(m.project_codons(h,e,reflected)).items()} == rc
    p['pos1']+=1
    moved=m.project_codons(h,e,p)
    assert {g-1:v for g,v in expand(moved).items()}==original
    assert moved['segments'][0]['anchor']==2


@pytest.mark.parametrize('change,reason',[
    ({'cigar':'2M1S9M'},'terminal'),({'cigar':'1M1H10M'},'terminal'),
    ({'cigar':'0M12M'},'CIGAR'),({'cigar':'12Q'},'CIGAR'),
    ({'cigar':'11M'},'length'),({'pos1':0},'position'),
    ({'contig_length':110},'contig'),({'flag':-1},'flag'),
    ({'flag':4},'unmapped'),({'seq':'A'},'SEQ length'),
    ({'ts':'?'},'ts'),({'cigar':'12S'},'mapped')])
def test_invalid_placement(change,reason):
    p,h,e=inputs(); p.update(change)
    with pytest.raises(ValueError,match=reason): m.project_codons(h,e,p)


def test_identity_and_hsp_validation():
    p,h,e=inputs(); h['identity']['source_start0']=11
    with pytest.raises(ValueError,match='identity'): m.project_codons(h,e,p)
    p,h,e=inputs(); h['qend']=2
    with pytest.raises(ValueError,match='direction'): m.project_codons(h,e,p)
    p,h,e=inputs(); h['label_convention']='genomic_mod3'
    with pytest.raises(ValueError,match='convention'): m.project_codons(h,e,p)
    p,h,e=inputs(); e[0]['length']=100
    with pytest.raises(ValueError,match='event bounds'): m.project_codons(h,e,p)


def test_ambiguity_refusals_not_policy():
    p,h,e=inputs(); p['ts']='-'
    r=m.project_codons(h,e,p)
    assert r['segments']==[] and 'ts_hsp_conflict_policy_unresolved' in r['reasons']
    p['ts']=None
    assert 'motif_strand_unavailable' in m.project_codons(h,e,p)['reasons']
    e.append(e[0].copy())
    r=m.project_codons(h,e,p)
    assert r['segments']==[] and 'self_overlap_policy_unresolved' in r['reasons']
    r=m.project_observations([(h,e,p),(h,e,p)])
    assert r['segments']==[] and r['reasons']==['multi_hsp_policy_unresolved']
    with pytest.raises(ValueError,match='UNRESOLVED'): m.project()


@pytest.mark.parametrize('cigar,reflected_cigar,span', [
    ('3M1I5M','5M1I3M',8), ('3M1D6M','6M1D3M',10),
    ('3M1P6M','6M1P3M',9)])
def test_f4_actual_minus_reflection(cigar,reflected_cigar,span):
    p,h,e=inputs(cigar,0,1,1,9,9)
    plus=expand(m.project_codons(h,e,p))
    p.update(cigar=reflected_cigar,flag=16)
    minus=expand(m.project_codons(h,e,p))
    assert minus == {199+span-g:('-',c) for g,(_,c) in plus.items()}


def test_reverse_hardclip_seq_not_double_complemented():
    p,h,e=inputs('2H4M4N5M1H',16,2,2,10)
    p['sequence']='AACCGGTTAACG'
    digest=hashlib.sha256(p['sequence'].encode()).hexdigest()
    p['identity']['sequence_sha256']=digest; h['identity']['sequence_sha256']=digest
    p['seq']='TTAACCGGT'  # RC CGTTAACCGGTT, minus 2 left and 1 right H
    assert len(expand(m.project_codons(h,e,p)))==9
    p['seq']=p['sequence'][2:-1]
    with pytest.raises(ValueError,match='SEQ identity'): m.project_codons(h,e,p)


@pytest.mark.parametrize('field,value', [('sequence_sha256','0'*64), ('source_sha256','bad'),
                                         ('source_end0',23),('member','')])
def test_invalid_screened_identity(field,value):
    p,h,e=inputs(); p['identity'][field]=value
    with pytest.raises(ValueError,match='identity'): m.project_codons(h,e,p)


@pytest.mark.parametrize('cigar', ['1S1H10M','10M1H1S','2H1H9M','1S1S10M'])
def test_bad_clip_order(cigar):
    p,h,e=inputs(cigar)
    with pytest.raises(ValueError,match='terminal'): m.project_codons(h,e,p)


def test_producer_discontinuity_not_extrapolated():
    p,h,e=inputs('7M',0,1,1,7,7)
    e=[dict(q0=0,length=3,direction=1,c0=0),dict(q0=4,length=3,direction=1,c0=0)]
    assert expand(m.project_codons(h,e,p))=={100:('+',0),101:('+',1),102:('+',2),
                                            104:('+',0),105:('+',1),106:('+',2)}


def test_compact_large_run():
    p,h,e=inputs('300000M',0,1,1,300000,300000); p['contig_length']=400000
    assert len(m.project_codons(h,e,p)['segments'])==1
