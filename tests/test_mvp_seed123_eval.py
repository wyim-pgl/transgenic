"""Development role identities never infer model role from the shared seed."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'revision/scripts/mvp'))
import mvp_produce_counts as p
import mvp_score as score
from transgenic.evidence import mvp_bootstrap as b, mvp_windows as w

PROFILE = 'development-single'


def development_rows(tmp_path):
    panel = w.load_panel(ROOT / 'revision/protocols/b6_panels_v1.tsv')
    checkpoints = {'mvp_seed123': 'a'*64, 'b5_seed123': 'b'*64}
    rows = []
    for locus in panel:
        for run_id, role, fractions in [('mvp_seed123', 'mvp_finetune', (0, 1)), ('b5_seed123', 'b5_baseline', (0,))]:
            for fraction in fractions:
                path = tmp_path / f'{run_id}.{fraction}.{locus.gene_id}.gff3'
                path.write_text('##gff-version 3\n')
                rows.append(dict(p.file_identity(path), profile=PROFILE, run_id=run_id,
                                 run_role=role, seed=123, fraction=fraction, gene_id=locus.gene_id,
                                 window=[locus.chromosome, *w.evaluation_window(locus)],
                                 checkpoint_sha256=checkpoints[run_id]))
    return rows, panel, checkpoints


def test_development_inventory_exact_role_qualified_150(tmp_path):
    rows, panel, checkpoints = development_rows(tmp_path)
    result = p.prediction_inventory(rows, panel, checkpoints, profile=PROFILE)
    assert len(result) == 150
    assert (PROFILE, 'mvp_seed123', 0, panel[0].gene_id) in result
    assert (PROFILE, 'b5_seed123', 0, panel[0].gene_id) in result


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'role', 'checkpoint', 'alias', 'seed_bool', 'fraction_bool', 'profile', 'run_id', 'same_checkpoint', 'same_stem', 'hardlink', 'metadata_profile', 'metadata_seed', 'metadata_checkpoint'])
def test_development_inventory_rejects_identity_mutations(tmp_path, mutation):
    rows, panel, checkpoints = development_rows(tmp_path)
    if mutation == 'missing': rows.pop()
    elif mutation == 'extra': rows.append(copy.deepcopy(rows[0]))
    elif mutation == 'role': rows[0]['run_role'] = 'b5_baseline'
    elif mutation == 'checkpoint': rows[0]['checkpoint_sha256'] = checkpoints['b5_seed123']
    elif mutation == 'alias': rows[1].update({k: rows[0][k] for k in ('path', 'sha256', 'size')})
    elif mutation == 'seed_bool': rows[0]['seed'] = True
    elif mutation == 'fraction_bool': rows[0]['fraction'] = False
    elif mutation == 'profile': rows[0]['profile'] = 'historical-three-seed'
    elif mutation == 'run_id': rows[0]['run_id'] = 'b5_seed123'
    elif mutation == 'same_checkpoint': checkpoints['mvp_seed123'] = checkpoints['b5_seed123']
    elif mutation in ('same_stem', 'hardlink'):
        directory = tmp_path / 'other'; directory.mkdir()
        path = directory / (Path(rows[0]['path']).name if mutation == 'same_stem' else 'alias.gff3')
        if mutation == 'hardlink': path.hardlink_to(rows[0]['path'])
        else: path.write_text('##gff-version 3\n')
        rows[1].update(p.file_identity(path))
    elif mutation == 'metadata_profile': rows[0]['metadata'] = {'profile': 'historical-three-seed'}
    elif mutation == 'metadata_seed': rows[0]['metadata'] = {'seed': 123.0}
    elif mutation == 'metadata_checkpoint': rows[0]['metadata'] = {'checkpoint_sha256': checkpoints['b5_seed123']}
    with pytest.raises(ValueError):
        p.prediction_inventory(rows, panel, checkpoints, profile=PROFILE)


def test_development_readiness_is_required_before_decoding():
    import mvp_decode_panel as decoder
    with pytest.raises(ValueError, match='readiness'):
        p.verify_evaluation_readiness({'profile': PROFILE, 'runs': {'mvp_seed123': {}}}, None, {})


def development_runs(tmp_path):
    from test_mvp_score_producer import finalized_fixture
    final, hashes, code = finalized_fixture(tmp_path, seed=123, profile=PROFILE)
    md = p.read_json(final / 'metadata.json')
    weights = md['initialization_audit']['audit']['files']['model.safetensors']
    baseline = p.file_identity(weights['path'])
    payload = dict(profile=PROFILE, runs={
        'mvp_seed123': dict(run_role='mvp_finetune', seed=123, final=str(final),
                            code_manifest=str(code), checkpoint_sha256=w.sha256_file(final / 'model.safetensors')),
        'b5_seed123': dict(run_role='b5_baseline', seed=123, weights=baseline,
                           checkpoint_sha256=baseline['sha256'])})
    return payload, hashes


def test_development_runs_verify_actual_finalizer_and_independent_baseline(tmp_path):
    payload, hashes = development_runs(tmp_path)
    runs = p.verify_run_inventory(payload, {'valid_rns': [1], 'sha256': hashes})
    assert runs['mvp_seed123']['seed'] == 123
    assert runs['mvp_seed123']['run_role'] == 'mvp_finetune'
    assert runs['b5_seed123']['run_role'] == 'b5_baseline'
    assert runs['b5_seed123']['weights']['sha256'] != runs['mvp_seed123']['files']['model.safetensors']['sha256']


@pytest.mark.parametrize('mutation', ['baseline_as_finetuned', 'seed_bool', 'missing_profile', 'metadata_profile', 'wrong_pin', 'baseline_aug', 'final_bytes', 'metadata_checkpoint'])
def test_development_runs_reject_role_profile_pin_mutations(tmp_path, mutation):
    payload, hashes = development_runs(tmp_path)
    if mutation == 'baseline_as_finetuned': payload['runs']['mvp_seed123']['run_role'] = 'b5_baseline'
    elif mutation == 'seed_bool': payload['runs']['mvp_seed123']['seed'] = True
    elif mutation == 'missing_profile': payload.pop('profile')
    elif mutation == 'metadata_profile': payload['metadata'] = {'profile': 'historical-three-seed'}
    elif mutation == 'wrong_pin': payload['runs']['b5_seed123']['weights']['sha256'] = 'c'*64
    elif mutation == 'baseline_aug': payload['runs']['b5_seed123']['seed_augmentation'] = {'seed': 123, 'aug_seed': 20261037}
    elif mutation == 'final_bytes': (Path(payload['runs']['mvp_seed123']['final']) / 'model.safetensors').write_bytes(b'wrong')
    elif mutation == 'metadata_checkpoint': payload['runs']['mvp_seed123']['metadata'] = {'checkpoint_sha256': 'e'*64}
    with pytest.raises(ValueError): p.verify_run_inventory(payload, {'valid_rns': [1], 'sha256': hashes})


def score_payload(tmp_path):
    panel = w.load_panel(ROOT / 'revision/protocols/b6_panels_v1.tsv')
    path = tmp_path / 'paired.jsonl'
    records = [dict(rn=i, has_evidence=True, loss_off=2., loss_on=1.,
                    composite_loss_off=4., composite_loss_on=2.,
                    ce_numerator_off=2.*i, ce_numerator_on=1.*i,
                    ce_denominator_off=i, ce_denominator_on=i) for i in (1, 2)]
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    loci = [dict(gene_id=l.gene_id, seeds={'123': {
        'off': dict(f1_tp=1, f1_predicted=2, f1_reference=2),
        'on': dict(f1_tp=2, f1_predicted=2, f1_reference=2)}}) for l in panel]
    return dict(profile=PROFILE, pairs={'mvp_seed123': str(path)}, loci=loci), panel


def test_development_metrics_are_single_seed_descriptive_and_missing_not_zero(tmp_path):
    payload, panel = score_payload(tmp_path)
    result = score.score(payload, panel)
    assert result['primary']['pooled_point'] == 1.
    assert result['primary']['per_seed_point'] == [1.]
    assert result['composite_delta_diagnostic'] == 2.
    assert result['readouts']['all50']['F1']['123']['point'] == 50.
    assert result['decision'] is None
    assert result['decision_status'] == 'DESCRIPTIVE_SINGLE_SEED_NO_FINAL_DECISION'
    assert result['structural_status'] == 'MISSING_UNASSESSABLE_NOT_ZERO'
    assert 'three_seed_identity_verified' not in result
    assert 'three_seed_policy_verified' not in result
    assert 'P' not in result['readouts']['all50']


@pytest.mark.parametrize('metadata', [{'run_role': 'b5_baseline'}, {'seed': True}, {'profile': 'historical-three-seed'}])
def test_development_scorer_rejects_conflicting_metadata(tmp_path, metadata):
    payload, panel = score_payload(tmp_path)
    payload['metadata'] = metadata
    with pytest.raises(ValueError): score.score(payload, panel)


@pytest.mark.parametrize('location', ['payload', 'metadata'])
@pytest.mark.parametrize('run_id', ['b5_seed123', '', None, 123])
def test_development_scorer_rejects_conflicting_run_id(tmp_path, location, run_id):
    payload, panel = score_payload(tmp_path)
    record = payload if location == 'payload' else payload.setdefault('metadata', {})
    record['run_id'] = run_id
    with pytest.raises(ValueError, match='run identity mismatch'):
        score.score(payload, panel)


@pytest.mark.parametrize('explicit', [False, True])
def test_development_scorer_accepts_matching_or_omitted_run_id(tmp_path, explicit):
    payload, panel = score_payload(tmp_path)
    if explicit:
        payload['run_id'] = 'mvp_seed123'
        payload['metadata'] = {'run_id': 'mvp_seed123'}
    result = score.score(payload, panel)
    assert result['run_id'] == 'mvp_seed123'
    assert result['decision'] is None


def test_development_partially_missing_structural_is_unassessable_not_negative(tmp_path):
    payload, panel = score_payload(tmp_path)
    for counts in payload['loci'][0]['seeds']['123'].values():
        counts.update(precision_tp=1, predicted=1, recall_tp=1, supported=1)
    result = score.score(payload, panel)
    assert result['structural_status'] == 'PARTIAL_UNASSESSABLE_NOT_ZERO'
    assert result['structural_available'] is False
    assert result['decision'] is None
    assert 'P' not in result['readouts']['all50']


@pytest.mark.parametrize('profile', [None, True, 'unknown', 'historical-three-seed'])
def test_single_seed_is_not_selected_without_explicit_profile(tmp_path, profile):
    payload, panel = score_payload(tmp_path)
    if profile is None: payload.pop('profile')
    else: payload['profile'] = profile
    with pytest.raises(ValueError): score.score(payload, panel)


@pytest.mark.parametrize('module', ['mvp_decode_panel', 'mvp_produce_counts', 'mvp_score'])
def test_cli_profile_must_match_explicit_payload(tmp_path, module):
    import importlib
    mod = importlib.import_module(module)
    path = tmp_path / 'request.json'
    path.write_text(json.dumps({'profile': PROFILE}))
    argv = ['--input', str(path), '--out', str(tmp_path / 'out'), '--profile', 'historical-three-seed']
    if module == 'mvp_score': argv += ['--panels', str(ROOT / 'revision/protocols/b6_panels_v1.tsv')]
    with pytest.raises(ValueError, match='profile mismatch'):
        mod.main(argv)


def test_bootstrap_single_seed_requires_explicit_profile(tmp_path):
    payload, _ = score_payload(tmp_path)
    rows = [json.loads(line) for line in Path(payload['pairs']['mvp_seed123']).read_text().splitlines()]
    with pytest.raises(ValueError): b.build_tiles({123: rows}, expected_seeds=(123,))
    rns, deltas = b.build_tiles({123: rows}, expected_seeds=(123,), profile=PROFILE)
    assert rns == (1, 2)
    assert deltas.tolist() == [[1.], [1.]]
