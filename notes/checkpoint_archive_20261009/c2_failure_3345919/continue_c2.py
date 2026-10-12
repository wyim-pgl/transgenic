"""One-shot DeltaAI continuation: wait for the reviewed recovery job, preserve old data, submit C2.

Never retries a mutation or resumes an existing journal automatically. Inspect the journal/Slurm first after
an interrupted or failed attempt. Run on the login node with --watch; no training runs in this process.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def command(args, env=None):
    result = subprocess.run(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, timeout=120)
    if result.returncode:
        raise RuntimeError('%s failed (%s): %s' % (args[0], result.returncode, result.stderr.strip()))
    return result.stdout.strip()


def write_state(path, state):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(state, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))
    descriptor = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def promote_runs(original, clone, archived):
    os.rename(str(original), str(archived))
    try:
        os.rename(str(clone), str(original))
    except BaseException:
        os.rename(str(archived), str(original))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', default='/work/nvme/bilv/wcyim/transgenic')
    parser.add_argument('--job', type=int, default=3361386)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args(argv)
    work = Path(args.work).absolute()
    folder = work / 'ops/c2_recovery_20261011'
    folder.mkdir(parents=True, exist_ok=True)
    journal = folder / 'continuation_state.json'
    state = dict(phase='waiting', recovery_job=args.job, source_commit='d7b1a37', jobs=[],
                 started_unix=time.time(), pid=os.getpid())
    try:
        with journal.open('x') as stream:
            json.dump(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        print('REFUSED: existing continuation journal; inspect Slurm and journal before any retry', file=sys.stderr)
        return 2

    def save(phase, **details):
        state.update(phase=phase, updated_unix=time.time(), **details)
        write_state(journal, state)
        print(json.dumps(dict(phase=phase, jobs=state['jobs'], updated_unix=state['updated_unix'])), flush=True)

    try:
        deadline = time.monotonic() + 10 * 3600
        while True:
            try:
                accounting = command(['sacct', '-j', str(args.job), '-n', '-X',
                                      '--format=JobID,State%30,ExitCode', '-P'])
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                if not args.watch or time.monotonic() >= deadline:
                    raise
                save('waiting', last_read_error=str(exc))
                time.sleep(60)
                continue
            rows = [line.split('|') for line in accounting.splitlines() if line.split('|')[0].strip() == str(args.job)]
            observed = rows[0][1].strip() if rows else 'UNKNOWN'
            save('waiting', observed_job_state=observed)
            if observed == 'COMPLETED':
                if rows[0][2].strip() != '0:0':
                    raise RuntimeError('recovery exit code is not 0:0')
                break
            if observed not in ('UNKNOWN', 'PENDING', 'RUNNING', 'COMPLETING', 'CONFIGURING', 'REQUEUED', 'SUSPENDED'):
                raise RuntimeError('recovery job did not succeed: ' + observed)
            if not args.watch:
                return 0
            if time.monotonic() >= deadline:
                raise RuntimeError('bounded continuation watch expired before recovery completed')
            time.sleep(60)

        repo = work / 'repo_b6_d7b1a37'
        original = work / 'runs/b6_wp5_pilot_C2'
        clone = work / 'runs/b6_wp5_pilot_C2_recovery_20261011'
        archived = work / 'runs/b6_wp5_pilot_C2.failed_3345919_before_recovery_20261011'
        assert not archived.exists(), 'archive destination already exists'
        assert not original.is_symlink() and not clone.is_symlink()
        assert (original / 'TRAINING_FAILED').is_file() and not (original / 'TRAINING_DONE').exists()
        assert json.loads((original / 'latest_state/meta.json').read_text())['global_step'] == 200
        assert not (clone / 'TRAINING_FAILED').exists() and not (clone / 'TRAINING_DONE').exists()
        receipt = json.loads((clone / 'RECOVERY_VALIDATION_PASSED.json').read_text())
        meta = json.loads((clone / 'latest_state/meta.json').read_text())
        assert receipt['status'] == 'passed' and receipt['job_id'] == args.job
        assert receipt['source_commit'] == 'd7b1a37' and receipt['started_from_update'] == 200
        assert receipt['validation_job'] == 3361110 and receipt['final_checkpoint'] == meta
        assert meta['arm'] == 'C2' and 400 <= meta['global_step'] < 4099 and meta['pending_micro'] == 0
        assert meta['stop_after_updates'] == 4099
        assert (repo / 'SOURCE_COMMIT').read_text().strip() == 'd7b1a37'
        assert (repo / '.git').is_dir(), 'pilot launcher requires a Git checkout'
        assert command(['git', '-C', str(repo), 'rev-parse', '--is-inside-work-tree']) == 'true'
        snapshot_commit = command(['git', '-C', str(repo), 'log', '-1', '--format=%H'])
        assert len(snapshot_commit) == 40
        command(['git', '-C', str(repo), 'diff', '--quiet', 'HEAD'])
        # sha256sum's relative names are resolved in the snapshot directory.
        checked = subprocess.run(['sha256sum', '-c', 'validation_sources.sha256'], cwd=str(repo),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=120)
        if checked.returncode:
            raise RuntimeError('candidate source hash check failed: ' + checked.stdout + checked.stderr)
        identity_check = ("import json; from transgenic.training import b6_runtime as r; "
                          "from pathlib import Path; p=Path('/work/runs/b6_wp5_pilot_C2_recovery_20261011'); "
                          "first=json.loads((p/'run_config_b6.json').read_text()); "
                          "new=json.loads((p/'latest_state/identity.json').read_text()); "
                          "r.check_run_config_b6(first,new); r.check_checkpoint_identity(str(p/'latest_state'),new); "
                          "assert new['world_size']==4 and new['arm']=='C2'; print('IDENTITY_OK')")
        command(['apptainer', 'exec', '--bind', str(work) + ':/work', '--bind', str(repo) + ':/work/repo:ro',
                 str(work.parent / 'containers/transgenic-ngc-26.08.sif'), 'env', 'PYTHONPATH=/work/repo/src',
                 'python', '-c', identity_check])
        hashes = {}
        required = ['meta.json', 'identity.json', 'accelerate_state/model.safetensors',
                    'accelerate_state/optimizer.bin', 'accelerate_state/scheduler.bin']
        required += ['accelerate_state/random_states_%d.pkl' % rank for rank in range(4)]
        for name in required:
            path = clone / 'latest_state' / name
            assert path.is_file() and not path.is_symlink() and path.stat().st_size > 0, name
        for path in sorted((clone / 'latest_state').rglob('*')):
            if path.is_file():
                assert not path.is_symlink(), str(path)
                hashes[str(path.relative_to(clone))] = sha(path)
        held = command(['scontrol', 'show', 'job', '-o', '3345920'])
        assert 'UserId=wcyim(' + str(os.getuid()) + ')' in held
        assert 'JobState=PENDING' in held and 'Reason=JobHeldUser' in held
        save('validated', checkpoint_update=meta['global_step'], checkpoint_hashes=hashes,
             original_archive=str(archived), canonical_run=str(original), snapshot_commit=snapshot_commit)

        # Retire the obsolete held launcher before the canonical run path is changed.
        save('retire_intent')
        command(['scancel', '--state=PENDING', '--user=wcyim', '3345920'])
        for attempt in range(10):
            if '3345920' not in command(['squeue', '-u', 'wcyim', '-h', '-o', '%i']).split():
                break
            time.sleep(1)
        else:
            raise RuntimeError('obsolete C2 job still appears active; no promotion performed')
        save('retired_obsolete_job', retired_job=3345920)
        save('promotion_intent')
        promote_runs(original, clone, archived)
        promotion = dict(recovery_job=args.job, source_commit='d7b1a37', previous_run=str(archived),
                         recovered_from=str(clone), checkpoint_update=meta['global_step'],
                         checkpoint_hashes=hashes, promoted_unix=time.time())
        write_state(original / 'RECOVERY_PROMOTION.json', promotion)
        save('promoted')

        environment = {k: v for k, v in os.environ.items() if not k.startswith('SBATCH_')}
        for key in ('TRANSGENIC_NO_COMPILE', 'TRANSGENIC_NO_GRAD_CKPT', 'TRANSGENIC_CKPT_REENTRANT'):
            environment.pop(key, None)
            environment.pop('APPTAINERENV_' + key, None)
        environment.update(
            SCRATCH=str(work.parent), SIF=str(work.parent / 'containers/transgenic-ngc-26.08.sif'), DETERMINISTIC='1',
            APPTAINER_BIND=('/work/hdd/bilv/wcyim/transgenic/b6_prep_20260925:/hdd_prep:ro,' +
                            str(work / 'b6_prep_20260925/intervals') +
                            ':/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/intervals:ro'),
            B6_OBS_CACHE_REQUIRE_PREBUILT='1', APPTAINERENV_B6_OBS_CACHE_REQUIRE_PREBUILT='1',
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
            APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
            TORCH_LOGS='recompiles', APPTAINERENV_TORCH_LOGS='recompiles',
            TORCHINDUCTOR_CACHE_DIR='/work/runs/c2_recompile_diag/3360834/inductor_cache',
            APPTAINERENV_TORCHINDUCTOR_CACHE_DIR='/work/runs/c2_recompile_diag/3360834/inductor_cache')
        for link in (1, 2):
            submit = ['sbatch', '--parsable', '--nice=0', '--job-name=c2-wp5-recovery',
                      '--partition=ghx4', '--account=bilv-dtai-gh', '--nodes=1', '--gpus-per-node=4',
                      '--cpus-per-task=64', '--time=1-00:00:00', '--chdir=' + str(repo),
                      '--export=ALL,REPO_DIR=' + str(repo) + ',ARM=C2,BINDINGS=/work/b6_prep_20260925/bindings/train_v1.deltaai.json']
            if state['jobs']:
                submit.append('--dependency=afterok:' + str(state['jobs'][-1]))
            submit.append(str(repo / 'revision/scripts/b6/wp5_pilot.sbatch'))
            save('submission_intent', submitting_link=link)
            answer = command(submit, env=environment)
            identifier = answer.split(';')[0]
            if not identifier.isdigit():
                raise RuntimeError('uncertain sbatch response; inspect before retry: ' + answer)
            state['jobs'].append(int(identifier))
            save('link_submitted')
        save('continuation_submitted')
        return 0
    except BaseException as exc:
        save('error', error='%s: %s' % (type(exc).__name__, exc))
        print(state['error'], file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
