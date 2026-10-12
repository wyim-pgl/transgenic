#!/usr/bin/env python3
"""Local review snapshot only. No deployment, imports of project code, or authorization."""
import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

SOURCE = Path('/tmp/mvp-seed123-integration-20260917')
PROJECT = Path('/mnt/f/scratch/develop/Transgenic/transgenic_local')
HERE = Path(__file__).resolve().parent

def sha(data):
    return hashlib.sha256(data).hexdigest()

def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')

def main():
    out = HERE / 'candidate'
    out.mkdir()  # Never overwrite an earlier candidate.
    roots = ('src/', 'train/', 'revision/scripts/mvp/', 'configs/')
    extras = {'revision/protocols/b6_panels_v1.tsv',
              'revision/scripts/07_selfconsistency_stats.py',
              'revision/scripts/24_restrict_reference_to_test_genes.py'}
    names = subprocess.check_output(['git', '-C', str(SOURCE), 'ls-files', '-z', '--cached', '--others', '--exclude-standard']).decode().split('\0')
    names = sorted({n for n in names if n and (n.startswith(roots) or n in extras)})
    assert extras <= set(names)
    payload = {}
    for name in names:
        path = SOURCE / name
        assert path.is_file() and not path.is_symlink() and path.resolve() == path.absolute(), name
        assert '__pycache__' not in path.parts and path.suffix != '.pyc', name
        payload[name] = path.read_bytes()
    for n in ('src/transgenic/training/mvp_profiles.py',
              'revision/scripts/mvp/mvp_panel_compatibility.py',
              'revision/scripts/mvp/mvp_verify_pinned_panel.py'):
        assert n in payload
    source_files = {n: sha(b) for n, b in payload.items()}
    with tarfile.open(out / 'code.candidate.tar', 'x', format=tarfile.PAX_FORMAT) as tar:
        for n, b in payload.items():
            info = tarfile.TarInfo(n)
            info.size = len(b)
            info.mode = 0o644
            info.mtime = 0
            tar.addfile(info, io.BytesIO(b))
    with tarfile.open(out / 'code.candidate.tar') as tar:
        assert {m.name: sha(tar.extractfile(m).read()) for m in tar.getmembers()} == source_files
    assert all((SOURCE/n).read_bytes() == b for n, b in payload.items()), 'Source changed during snapshot'
    head = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD']).decode().strip()
    assert head == '9603e20fd4870a9f5b3d5f68e2df8ba0f8a49558'
    status = subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--short']).decode()
    (out/'source-status.txt').write_text(status)
    consumed = {('/work/repo/'+n): h for n,h in source_files.items()
                if (n.startswith('src/transgenic/') and n.endswith('.py')) or n == 'train/train_HyenaTransgenic_MVP.py'}
    save(out/'expected-project-files.json', consumed)
    # This is not the SIF-generated consumed manifest: installed package versions are unmeasured.
    save(out/'source-manifest.json', dict(status='LOCAL_CANDIDATE_NOT_RELEASE', base_commit=head,
         source_origin=str(SOURCE), provenance='dirty working-tree bytes including selected untracked helpers; NOT git archive HEAD',
         source_files=source_files, archive_sha256=sha((out/'code.candidate.tar').read_bytes())))
    old = PROJECT/'notes/mvp_execution_20260916/final_p1'
    pins = {}
    tree = ast.parse((old/'deploy.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='PINS' for t in node.targets):
            pins = ast.literal_eval(node.value)
    companions = json.loads((old/'release/release.json').read_text())['accepted_companions_not_in_source_commit']
    for n, digest in companions.items():
        b = (PROJECT/n).read_bytes()
        assert sha(b)==digest
        dest = out/n
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(b)
    config_name = 'configs/b5_400m_win_v3.json'
    (out/'model-config.candidate.json').write_bytes(payload[config_name])
    save(out/'binding.candidate.json', dict(status='RECIPE_ONLY_NOT_PREPARED_BINDINGS', profile='development-single', seed=123,
         run_role='mvp_finetune', baseline_run_id='b5_seed123', finetune_run_id='mvp_seed123',
         initialization_epoch='epoch_05', source_checkpoint_seed=123, immutable_input_pins=pins,
         companion_pins=companions, config_sha256=source_files[config_name],
         prepare_extra_argv=['--profile','development-single','--seed','123'],
         runtime_generated_required=['bindings_sha256','consumed_code_manifest','external_packages','assets','tiles','six measured receipts'],
         ready_for_finetuning=False))
    save(out/'budget.candidate.json', dict(status='INCOMPLETE_REQUEST_NOT_AUTHORIZATION', schema_version=2,
         authorized_scope='exploratory_mvp', profile='development-single', seeds=[123],
         resource_envelope=dict(nodes=1,ranks=4,microbatches_per_rank=24),
         campaign_diagnostic_cap_node_hours=5, campaign_diagnostic_cap_gpu_hours=20,
         proposed_first_allocation_seconds=1800, bindings_sha256=None, controller_record_reference=None,
         intent='Single seed123 exploratory MVP; preserve B5; no production authorization',
         max_training_node_hours=None,max_wall_seconds=None,max_probe_wall_seconds=None,
         prior_smoke_node_hours=None,safety_margin=None,memory_headroom_fraction=None,
         launcher_overhead_seconds=None,margin_rationale=None))
    scripts = [old/'build_release.py',old/'deploy.py',old/'p1.sbatch',old/'test_deploy.py',
               old/'controller_execution_scope.json',old/'release/release.sha256',
               PROJECT/'notes/wait_work_20260916/gpu_followup/followup.py',
               PROJECT/'notes/wait_work_20260916/gpu_followup/followup.sbatch']
    save(out/'existing-script-hashes.json',{str(p):sha(p.read_bytes()) for p in scripts})
    with (out/'candidate.sha256').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='candidate.sha256':
                f.write(sha(p.read_bytes())+'  '+str(p.relative_to(out))+'\n')
    print(json.dumps(dict(source_files=len(source_files), consumed_project_files=len(consumed),
                         archive_sha256=sha((out/'code.candidate.tar').read_bytes()),
                         manifest_sha256=sha((out/'source-manifest.json').read_bytes()),
                         ledger_sha256=sha((out/'candidate.sha256').read_bytes())),indent=2))

if __name__ == '__main__':
    main()
