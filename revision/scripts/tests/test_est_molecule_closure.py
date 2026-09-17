"""Bounded normalized A11/A46 contracts; synthetic metadata, not source attestation."""
import importlib.util
from functools import lru_cache
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).parents[1] / 'evidence' / 'est_molecule_closure.py'


@lru_cache(None)
def api():
    assert PATH.exists(), 'A11 normalized closure primitive not implemented'
    spec = importlib.util.spec_from_file_location('est_molecule_closure', PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def member(m, uid='a', **kw):
    fields = dict(uid=uid, species='sp', assembly='asm', dataset='est/sp', arm='primary',
                  accession=uid.upper() + '.1', library='lib', clone='clone-' + uid,
                  clone_status='known', original_accession=uid.upper() + '.1',
                  original_length=200, original_interval=(0, 200), screened_sha256='a'*64,
                  provenance='metadata:receipt', bioproject='P', genotype='unknown',
                  role='train_track', support_accepted=True, report_id='report:'+uid)
    fields.update(kw)
    return m.Member(**fields)


def placement(m, uid='a', **kw):
    fields = dict(uid=uid+':p', member=uid, contig='chr1', strand='+', start=10, end=210,
                  blocks=((10, 110), (150, 210)), kind='primary', mapq=30)
    fields.update(kw)
    return m.Placement(**fields)


def run(m, members=None, placements=None, **kw):
    members = members if members is not None else [member(m)]
    placements = placements if placements is not None else [placement(m, x.uid) for x in members]
    audits = [m.Audit(x.uid, sum(p.member == x.uid for p in placements),
                      sum(p.member == x.uid and p.kind == 'secondary' for p in placements),
                      True, 'audit:receipt') for x in members]
    args = dict(species='sp', assembly='asm', members=members, placements=placements,
                relations=[], audits=audits, exclusions=[], library_aliases={'lib':'lib'},
                inventory_complete=True, identity_complete=True, exclusions_complete=True,
                context='tile1', policy='A11_A21_A46_normalized_v1')
    args.update(kw)
    return m.close_molecules(**args)


def test_normalization_library_aliases_and_no_inferred_chains():
    m = api()
    a, b = member(m, clone=' Clone-A / 5′ '), member(m, 'b', clone='clone_a-reverse', library='alias')
    r = run(m, [a,b], library_aliases={'lib':'lib','alias':'lib'})
    assert len(r['components']) == 1
    assert r['components'][0]['members'] == ('a','b')
    assert len(r['placements']) == 2
    assert 'junctions' not in r and 'chain' not in r
    assert r['diagnostic_ready'] and not r['training_eligible']
    assert len(run(m, [a,member(m,'b',clone=a.clone,library='other')],
                   library_aliases={'lib':'lib','other':'other'})['components']) == 2


def test_known_clone_not_signature_merged_and_unavailable_exact_blocks_only():
    m = api()
    assert len(run(m,[member(m),member(m,'b')])['components']) == 2
    rows = [member(m,x,clone='',clone_status='clone_unavailable') for x in ('a','b')]
    assert len(run(m,rows)['components']) == 1
    for changes in ({'strand':'-'},{'contig':'chr2'},{'blocks':((10,100),(150,210))}):
        assert len(run(m,rows,[placement(m),placement(m,'b',**changes)])['components']) == 2


def test_versions_kept_but_latest_support_and_old_secondary_leaks():
    m = api()
    rows = [member(m,accession='X.1',original_accession='X.1'),
            member(m,'b',accession='X.2',original_accession='X.2',clone='clone-a')]
    ps = [placement(m),placement(m,'b')]
    r = run(m,rows,ps)
    assert r['components'][0]['support_members'] == {'primary':('b',)}
    ps.append(m.Placement('late','a','chr9','-',100,200,((100,200),),'secondary',0))
    e = m.Exclusion('held','heldout','global',None,'chr9',150,160)
    r = run(m,rows,ps,exclusions=[e])
    assert r['components'][0]['disposition'] == 'excluded'
    assert ('overlap','a','late','held','heldout') in r['components'][0]['causes']


@pytest.mark.parametrize('dataset',['est/sp','other-download'])
@pytest.mark.parametrize('old_excluded',[False,True])
def test_accession_versions_reject_conflicting_canonical_libraries(dataset, old_excluded):
    m = api()
    rows = [member(m,'old',accession='X.1',original_accession='X.1',clone='clone-x'),
            member(m,'new',accession='X.2',original_accession='X.2',clone='clone-x',
                   library='lib2',dataset=dataset)]
    ps = [placement(m,'old'),placement(m,'new')]
    exclusions = []
    if old_excluded:
        ps.append(m.Placement('late','old','chr9','-',100,200,((100,200),),'secondary',30))
        exclusions = [m.Exclusion('held','heldout','global',None,'chr9',150,160)]
    with pytest.raises(ValueError,match='accession stem has conflicting canonical libraries'):
        run(m,rows,ps,library_aliases={'lib':'lib','lib2':'lib2'},exclusions=exclusions)


@pytest.mark.parametrize('old_excluded',[False,True])
def test_accession_version_library_alias_accepts_and_propagates(old_excluded):
    m = api()
    rows = [member(m,'old',accession='X.1',original_accession='X.1',clone='clone-x'),
            member(m,'new',accession='X.2',original_accession='X.2',clone='clone-x',
                   library='alias',dataset='other-download')]
    ps = [placement(m,'old'),placement(m,'new')]
    exclusions = []
    if old_excluded:
        ps.append(m.Placement('late','old','chr9','-',100,200,((100,200),),'secondary',30))
        exclusions = [m.Exclusion('held','heldout','global',None,'chr9',150,160)]
    r = run(m,rows,ps,library_aliases={'lib':'lib','alias':'lib'},exclusions=exclusions)
    assert len(r['components']) == 1
    c = r['components'][0]
    assert c['support_members'] == {'primary':('new',)}
    assert c['disposition'] == ('excluded' if old_excluded else 'retained')
    assert c['training_support_members'] == (() if old_excluded else ('new',))


def test_distinct_accessions_different_libraries_stay_independent():
    m = api()
    rows = [member(m,'old',accession='X.1',original_accession='X.1',clone='clone-x'),
            member(m,'new',accession='Y.2',original_accession='Y.2',clone='clone-x',library='lib2')]
    r = run(m,rows,library_aliases={'lib':'lib','lib2':'lib2'})
    assert len(r['components']) == 2 and all(c['ready'] for c in r['components'])


def test_unknown_library_of_accession_version_still_blocks_whole_view():
    m = api()
    rows = [member(m,'old',accession='X.1',original_accession='X.1',clone='clone-x'),
            member(m,'new',accession='X.2',original_accession='X.2',clone='clone-x',library='unknown')]
    r = run(m,rows)
    assert not r['diagnostic_ready'] and not any(c['ready'] for c in r['components'])
    assert all(not c['training_support_members'] for c in r['components'])


def test_report_parts_one_survivor_and_multiple_parts_arm_separation():
    m = api()
    rows = [member(m, original_length=400, original_interval=(0,100), report_id='R'),
            member(m,'b',accession='A.1',original_accession='A.1',original_length=400,
                   original_interval=(250,400),report_id='R',clone='clone-a'),
            member(m,'s',arm='min121',accession='A.1',original_accession='A.1',clone='clone-a')]
    r = run(m,rows)
    assert len(r['components']) == 2  # no implicit cross-arm physical linkage
    relation = m.Relation('a','s','parent','paired-source:receipt')
    r = run(m,rows,relations=[relation])
    assert len(r['components']) == 1
    assert r['components'][0]['support_members'] == {'min121':('s',),'primary':('a','b')}
    assert r['components'][0]['training_support_members'] == ('a','b')
    assert len(run(m,[rows[0]])['members']) == 1


@pytest.mark.parametrize('change', [dict(library='unknown'),dict(clone_status='uninspected',clone=''),dict(library='')])
def test_unknown_metadata_blocks_whole_input(change):
    m = api()
    r = run(m,[member(m),member(m,'b',**change)])
    assert not r['diagnostic_ready']
    assert all(not c['ready'] for c in r['components'])


@pytest.mark.parametrize('change', [dict(original_interval=(-1,100)),dict(original_interval=(0,201)),
                                    dict(original_accession='WRONG.1'),dict(screened_sha256='bad'),
                                    dict(species='other'),dict(assembly='other')])
def test_invalid_metadata_rejected(change):
    m = api()
    with pytest.raises(ValueError): run(m,[member(m,**change)])


def test_orphan_and_contradictory_links_rejected():
    m = api()
    with pytest.raises(ValueError): run(m,relations=[m.Relation('a','missing','mate','R')])
    with pytest.raises(ValueError):
        run(m,[member(m),member(m,'b')],relations=[m.Relation('a','b','mate','R')])
    with pytest.raises(ValueError): run(m,policy='guess')


@pytest.mark.parametrize('kind',['primary','secondary','supplementary','ambiguous'])
def test_mate_leakage_every_kind_rejected_support_and_intron_span(kind):
    m = api()
    rows = [member(m),member(m,'b',clone='clone-a',support_accepted=False)]
    ps = [placement(m),placement(m,'b',kind=kind)]
    e = m.Exclusion('gap','test_block','global',None,'chr1',120,130)
    r = run(m,rows,ps,exclusions=[e])
    assert r['components'][0]['disposition'] == 'excluded'
    assert len(r['components'][0]['causes']) == 2


def test_halfopen_and_context_scope_and_tamper_recheck():
    m = api()
    touch = m.Exclusion('touch','test_block','global',None,'chr1',210,220)
    assert run(m,exclusions=[touch])['components'][0]['disposition'] == 'retained'
    e = m.Exclusion('valid','valid_orthogroup','train_context','tile2','chr1',100,110)
    assert run(m,exclusions=[e])['components'][0]['disposition'] == 'retained'
    r = run(m,exclusions=[e],context='tile2')
    r['components'][0]['disposition'] = 'retained'
    with pytest.raises(ValueError,match='retained'):
        m.recheck_retained(r, exclusions=[e], context='tile2')
    with pytest.raises(ValueError):
        run(m,exclusions=[m.Exclusion('bad','valid_orthogroup','global',None,'chr1',1,2)])


@pytest.mark.parametrize('n,excluded',[(49,False),(50,True)])
def test_secondary_saturation_and_explicit_denominator(n,excluded):
    m = api()
    ps = [m.Placement(str(i),'a','chr1','+',10,210,((10,210),),'secondary',30) for i in range(n)]
    r = run(m,placements=ps)
    assert (r['components'][0]['disposition']=='excluded') == excluded
    assert r['exhaustiveness'] == {'molecules_lt50':int(n<50),'molecules_all':1,'fraction':float(n<50)}


def test_low_mapq_missing_audit_and_global_incomplete():
    m = api()
    assert run(m,placements=[placement(m,mapq=4)])['components'][0]['disposition']=='excluded'
    r = run(m,[member(m),member(m,'b')],audits=[m.Audit('a',1,0,True,'R')])
    assert not r['diagnostic_ready'] and all(not x['ready'] for x in r['components'])
    assert r['components'][1]['disposition']=='excluded'
    assert r['exhaustiveness'] == {'molecules_lt50':2,'molecules_all':2,'fraction':1.0}
    assert r['components'][1]['total_placements'] == 1
    for field in ('inventory_complete','identity_complete','exclusions_complete'):
        assert not run(m,**{field:False})['diagnostic_ready']


def test_order_invariance_and_duplicate_rejection():
    m = api()
    rows = [member(m),member(m,'b',clone='clone-a')]
    ps = [placement(m),placement(m,'b')]
    assert run(m,rows,ps) == run(m,list(reversed(rows)),list(reversed(ps)))
    with pytest.raises(ValueError): run(m,[rows[0],rows[0]])


@pytest.mark.parametrize('kind',['heldout','test_block','test_orthogroup','hard_flag',
                                  'overlap_component','decoy_masked','shared_junction',
                                  'valid_orthogroup','partial'])
def test_all_exclusion_types_preserve_causes(kind):
    m = api()
    contextual = kind in ('valid_orthogroup','partial')
    e = m.Exclusion(kind,kind,'train_context' if contextual else 'global',
                    'tile1' if contextual else None,'chr1',20,30)
    r = run(m,exclusions=[e])
    assert r['components'][0]['causes'] == (('overlap','a','a:p',kind,kind),)


def test_min121_is_never_training_and_preserves_input_audit():
    m = api()
    r = run(m,[member(m,arm='min121')])
    assert r['components'][0]['disposition'] == 'never'
    assert not r['components'][0]['training_support_members']
    assert r['audits'][0].member == 'a'
    assert r['library_aliases'] == (('lib','lib'),)


def test_supported_unmapped_and_report_disagreement_rejected():
    m = api()
    with pytest.raises(ValueError): run(m,placements=[])
    rows = [member(m),member(m,'b',accession='A.1',original_accession='A.1',clone='clone-a')]
    with pytest.raises(ValueError): run(m,rows)


def test_no_suffix_identity_or_sequence_hash_union_and_bad_alias_map():
    m = api()
    rows = [member(m,'x_part1'),member(m,'x_part2')]
    assert len(run(m,rows)['components']) == 2
    with pytest.raises(ValueError): run(m,library_aliases={'lib':'alias','alias':'lib'})


def test_recheck_context_cannot_be_changed():
    m = api()
    with pytest.raises(ValueError): m.recheck_retained(run(m),exclusions=[],context='other')


def test_recheck_refuses_changed_manifest():
    m = api()
    e = m.Exclusion('held','heldout','global',None,'chr1',20,30)
    r = run(m,exclusions=[e])
    r['components'][0]['disposition'] = 'retained'
    with pytest.raises(ValueError): m.recheck_retained(r,exclusions=[],context='tile1')


def test_normalizer_exact_existing_helper():
    import ast
    m = api()
    source = (PATH.parents[1] / '59_evidence_support.py').read_text()
    original = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='normalize_clone')
    current = next(n for n in ast.parse(PATH.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='normalize_clone')
    assert ast.dump(original) == ast.dump(current)
    assert m.normalize_clone(' Clone-X:forward ') == 'clonex'


def test_exhaustiveness_999_boundary():
    m = api()
    rows = [member(m,str(i),clone='unit'+str(i)+'x') for i in range(1000)]
    ps = [placement(m,x.uid) for x in rows]
    ps += [m.Placement('extra'+str(i),'0','chr1','+',10,210,((10,210),),'secondary',30) for i in range(49)]
    r = run(m,rows,ps)
    assert r['exhaustiveness']['fraction'] == .999 and r['diagnostic_ready']
    assert not run(m,rows[:-1],[p for p in ps if p.member != '999'])['diagnostic_ready']
