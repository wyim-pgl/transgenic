"""Publication-boundary regressions; all mutations are temporary synthetic inputs."""
import json
from pathlib import Path
import pytest
from test_mvp_reference_mapping import mapped, p, ROOT, test_cpu_wrapper_uses_shared_mapping_and_diagnostic_fallback as cpu_audit
from test_mvp_score_producer_integration import test_full_producer_actual_finalizer_frozen_panel_synthetic_biology as full350


@pytest.mark.parametrize('mutation', ['wal', 'source'])
@pytest.mark.parametrize('target', ['counts.json', 'statistics.json', 'receipt.json.pending'])
def test_producer_late_wal(tmp_path, monkeypatch, target, mutation):
    original = json.dump
    injected = []
    def dump(obj, handle, *args, **kwargs):
        result = original(obj, handle, *args, **kwargs)
        if Path(handle.name).name == target:
            path = tmp_path/('corpus.json.wal' if mutation == 'wal' else 'reference.gff')
            with path.open('ab') as changed: changed.write(b'late mutation')
            injected.append(target)
        return result
    monkeypatch.setattr(json, 'dump', dump)
    with pytest.raises(ValueError, match='WAL|hash mismatch'):
        full350(tmp_path, monkeypatch)
    assert injected == [target]
    assert not (tmp_path/'result/receipt.json').exists()
    assert path_mutated(tmp_path, mutation).read_bytes().endswith(b'late mutation')


def path_mutated(root, mutation):
    return root/('corpus.json.wal' if mutation == 'wal' else 'reference.gff')


@pytest.mark.parametrize('when', ['corpus', 'last_source'])
@pytest.mark.parametrize('mutation', ['wal', 'source'])
def test_mapping_mutation_during_hash(mapped, monkeypatch, mutation, when):
    _, corpus, _, _, resolve = mapped
    receipt = resolve()
    original = p.w.sha256_file
    injected = []
    def sha(path):
        result = original(path)
        trigger = corpus if when == 'corpus' else Path(receipt['sources'][-1]['path'])
        if Path(path) == trigger and not injected:
            if mutation == 'wal':
                Path(str(corpus)+'.wal').write_bytes(b'concurrent mutation')
            else:
                # Same-size change after hashing: size validation alone misses it.
                data = corpus.read_bytes()
                corpus.write_bytes(bytes([data[0] ^ 1]) + data[1:])
            injected.append(mutation)
        return result
    monkeypatch.setattr(p.w, 'sha256_file', sha)
    with pytest.raises(ValueError): p.recheck_reference_mapping(receipt)
    assert injected == [mutation]


@pytest.mark.parametrize('joint_failure', [False, True])
@pytest.mark.parametrize('mutation', ['wal', 'source'])
def test_cpu_audit_late_mutation(mapped, tmp_path, monkeypatch, joint_failure, mutation):
    original = json.dump
    injected = []
    def dump(obj, handle, *args, **kwargs):
        result = original(obj, handle, *args, **kwargs)
        if Path(handle.name).name == 'reference_compatibility.json.pending':
            target = Path(str(mapped[1])+'.wal') if mutation == 'wal' else mapped[2]
            with target.open('ab') as changed: changed.write(b'late mutation')
            injected.append(mutation)
        return result
    monkeypatch.setattr(json, 'dump', dump)
    with pytest.raises(ValueError): cpu_audit(mapped, tmp_path, monkeypatch, joint_failure)
    assert injected == [mutation]
    assert not (tmp_path/'cpu/reference_compatibility.json').exists()
    assert (tmp_path/'cpu/reference_compatibility.json.pending').exists()


@pytest.mark.parametrize('reference_only,target', [
    (True, 'reference_compatibility.json.pending'), (True, 'REFERENCE_ONLY.json.pending'),
    (False, 'reference_compatibility.json.pending'), (False, 'panel_counts.json'),
    (False, 'OBSERVATIONS_COMPLETE.json.pending'),
])
@pytest.mark.parametrize('mutation', ['none', 'wal', 'source'])
def test_cpu_completion(mapped, tmp_path, monkeypatch, reference_only, target, mutation):
    import importlib.util
    import sys
    from types import SimpleNamespace
    from transgenic.training import mvp_cli, mvp_cli_runtime
    panel, corpus, reference, authority, resolve = mapped
    root = tmp_path/'source'; (root/'notes').mkdir(parents=True)
    (root/'notes/b6_mvp_technical_candidate_20260915.json').write_text(json.dumps(
        {'inputs': {'corpus': {'path': str(corpus)}}, 'artifact': {'path': 'synthetic'}}))
    companion = SimpleNamespace(verify=lambda root: {}, SPEC='synthetic', git=lambda *a:
        ('species_id\tfasta\tgff\nAthaliana\t'+authority['files']['fasta']['path']+'\t'+str(reference)+'\n').encode())
    monkeypatch.setitem(sys.modules, 'authority', companion)
    spec = importlib.util.spec_from_file_location('cpu_completion_test', ROOT/'tasknotes/reference_mapping/run_panel_cpu.py')
    wrapper = importlib.util.module_from_spec(spec); spec.loader.exec_module(wrapper)
    monkeypatch.setattr(wrapper, 'verify_source', lambda root: {})
    # Keep genuine mapping/parser, but stub packaging/allocation and raster work.
    monkeypatch.setattr(p, 'ROOT', root)
    monkeypatch.setattr(p, 'frozen_reference', lambda *a: authority)
    monkeypatch.setattr(p.w, 'load_panel', lambda *a: panel)
    monkeypatch.setattr(p, 'fasta_lengths', lambda *a: {'ChrM': 100000})
    monkeypatch.setattr(p.w, 'evaluation_window', lambda *a: (0, 30720))
    monkeypatch.setattr(mvp_cli, 'validate_candidate_records', lambda *a: object())
    monkeypatch.setattr(mvp_cli_runtime, 'verify_candidate_artifacts', lambda *a: {'sha256': {}})
    # Mapping was resolved with the actual tiny DuckDB before changing ROOT.
    # resolve uses ROOT for code identities; restore only while obtaining receipt.
    monkeypatch.setattr(p, 'ROOT', ROOT); receipt = resolve(); monkeypatch.setattr(p, 'ROOT', root)
    monkeypatch.setattr(p, 'resolve_reference_mapping', lambda *a: receipt)
    panels = root/'revision/protocols/b6_panels_v1.tsv'; panels.parent.mkdir(parents=True); panels.write_text('synthetic')
    def prepare(request, out):
        out.mkdir(); (out/'manifest.json').write_text('{}')
        row = dict(gene_id='synthetic', window=[], selected_molecule_count=0,
                   biological_nonzero_per_channel=[], availability_nonzero=0, zero_biological_support=True)
        return {'loci': [row.copy() for _ in range(50)]}
    monkeypatch.setattr(wrapper, 'cpu_runtime', lambda *a: SimpleNamespace(prepare=prepare))
    for key, value in [('SLURM_JOB_ID','synthetic-local-test'), ('SLURM_CPUS_PER_TASK','1'), ('CUDA_VISIBLE_DEVICES','')]:
        monkeypatch.setenv(key, value)
    out = tmp_path/'completion'
    monkeypatch.setattr(sys, 'argv', ['cpu', '--root', str(root), '--out', str(out)] + (['--reference-only'] if reference_only else []))
    original = json.dump
    injected = []
    def dump(obj, handle, *args, **kwargs):
        result = original(obj, handle, *args, **kwargs)
        if Path(handle.name).name == target:
            injected.append(target)
            if mutation != 'none':
                changed = Path(str(corpus)+'.wal') if mutation == 'wal' else reference
                with changed.open('ab') as f: f.write(b'late mutation')
        return result
    monkeypatch.setattr(json, 'dump', dump)
    marker = out/('REFERENCE_ONLY.json' if reference_only else 'OBSERVATIONS_COMPLETE.json')
    if mutation == 'none':
        wrapper.main()
        assert marker.exists() and (out/'reference_compatibility.json').exists()
        assert not (out/'FAILED.json').exists()
    else:
        with pytest.raises(ValueError): wrapper.main()
        assert not marker.exists() and not (out/'reference_compatibility.json').exists()
        assert (out/'FAILED.json').exists()
    assert injected == [target]


def test_publication_closed_temp_and_no_replace(tmp_path):
    destination = tmp_path/'receipt.json'
    staging = p.stage_json(destination, {'ok': True})
    def recheck():
        assert json.loads(staging.read_text()) == {'ok': True}
        assert not destination.exists()
        destination.write_text('external data')
    with pytest.raises(FileExistsError): p.publish_staged([destination], recheck)
    assert destination.read_text() == 'external data'
    assert staging.exists()
