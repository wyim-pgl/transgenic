"""Read-only B6 pilot audit fault injections; all model outputs use TemporaryDirectory.
Run from /tmp/b6-compile-static with:
PYTHONPATH=/tmp/b6-spec-review-deps:src:. /home/wyim/miniconda3/envs/pdb/bin/python -u /tmp/b6-spec-review-20261009.py
"""
import os
import subprocess
from pathlib import Path
from unittest.mock import patch
from tests.test_b6_trainer import TrainerBase, rt

f = TrainerBase()
f.setUpClass()
f.setUp()
try:
    s = f.spec('B6', run='final_fault', stop_after_updates=6, save_every_n_steps=1)
    with patch.object(rt.CheckpointLayout, 'mark_done', side_effect=OSError('injected crash after publication')):
        try:
            f.train_arm(s)
            raise AssertionError('publication injection unexpectedly did not fail')
        except RuntimeError as exc:
            assert 'injected crash after publication' in str(exc)
            print('first_failure', type(exc).__name__, str(exc))
    markers = {k: os.path.exists(os.path.join(s.run_dir, k)) for k in
               ['step_00006/PILOT_FINAL', 'TRAINING_DONE', 'latest_state/meta.json']}
    print('first_markers', markers)
    assert markers == {'step_00006/PILOT_FINAL': True, 'TRAINING_DONE': False, 'latest_state/meta.json': True}
    try:
        f.train_arm(f.spec('B6', run='final_fault', stop_after_updates=6, save_every_n_steps=1, resume='auto'))
        raise AssertionError('resume unexpectedly succeeded')
    except RuntimeError as exc:
        assert 'refusing to replace published step checkpoint' in str(exc)
        print('resume_failure', type(exc).__name__, str(exc))
    assert not os.path.exists(os.path.join(s.run_dir, 'TRAINING_DONE'))
    print('publication_retry_bug', 'REPRODUCED')

    first = f.spec('B6', run='final_compile_fault', n_pilot=4, stop_after_updates=4,
                   compile_backend='eager', stop_at_step=2)
    assert first.compile_warmup_steps == 3
    first_result = f.train_arm(first)
    assert first_result['preempted'] and first_result['global_step'] == 2
    s = f.spec('B6', run='final_compile_fault', n_pilot=4, stop_after_updates=4,
               compile_backend='eager', resume='auto')
    print('final_link_updates_remaining', 2, 'compile_warmup_steps', s.compile_warmup_steps)
    with patch.object(rt, 'check_compiled', side_effect=RuntimeError('injected zero compiled graphs')):
        try:
            f.train_arm(s)
            raise AssertionError('compile verification unexpectedly did not fail')
        except RuntimeError as exc:
            assert 'injected zero compiled graphs' in str(exc)
            print('compile_failure', type(exc).__name__, str(exc))
    markers = {k: os.path.exists(os.path.join(s.run_dir, k)) for k in
               ['step_00004/PILOT_FINAL', 'TRAINING_DONE']}
    print('markers_after_compile_failure', markers)
    assert all(markers.values())
    print('done_before_verification_bug', 'REPRODUCED')
    launcher = Path('revision/scripts/b6/wp5_pilot.sbatch').read_text().splitlines()
    tail = '\n'.join(launcher[111:123])
    assert tail.startswith('STEP=$RUN_HOST/') and tail.endswith('exit 1')
    env = dict(os.environ, RUN_HOST=s.run_dir, N_PILOT='4', ARM='B6', TRAIN_RC='1')
    launch_result = subprocess.run(['bash', '-c', 'set -euo pipefail\n' + tail], env=env, text=True, capture_output=True)
    print('launcher_tail_stdout', launch_result.stdout.strip())
    print('launcher_trainer_rc', 1, 'launcher_exit', launch_result.returncode)
    assert launch_result.returncode == 0
    print('launcher_false_success_bug', 'REPRODUCED')
finally:
    f.doCleanups()
    f.tearDownClass()
