#!/usr/bin/env python3
"""Compute-only bounded launcher. No resume, checkpoint choice, or budget defaults.

Run inside the pinned SIF on one allocated four-GPU node. This does not submit
jobs. The trainer alone publishes training artifacts; PROCESS_EXIT.json is not
a completion certificate. A timeout/SIGTERM records FAILED.json, never DONE.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys

from make_launch_readiness import (_REPO_ROOT, _atomic_write, _canonical_bytes,
                                   _sha256, json_snapshot, check, require_allocation)
from run_bounded import run_command


def training_arguments(path, expected_sha):
    from transgenic.training.mvp_cli import build_parser, _required
    argv = json_snapshot(path, expected_sha)
    if not isinstance(argv, list) or any(not isinstance(value, str) for value in argv):
        raise ValueError('training arguments must be an argv list')
    args = build_parser().parse_args(argv)
    if args.validate_only:
        raise ValueError('validate-only bypasses readiness and is forbidden in this launcher')
    from transgenic.training.mvp_profiles import validate_seed_profile
    validate_seed_profile(args.profile, args.seed)
    _required(args, ('db', 'evidence_db', 'panel', 'acceptance_record', 'acceptance_sha256',
                     'inventory_record', 'receipts_record', 'config', 'init_from', 'output_dir',
                     'sif', 'seed', 'launch_readiness', 'launch_readiness_sha256'))
    args.readiness = args.launch_readiness
    args.readiness_sha256 = args.launch_readiness_sha256
    return argv, args


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arguments', required=True)
    parser.add_argument('--arguments-sha256', required=True)
    parser.add_argument('--budget-authorization', required=True)
    parser.add_argument('--budget-sha256', required=True)
    parser.add_argument('--timeout-seconds', required=True, type=int)
    parser.add_argument('--status-dir', required=True)
    args = parser.parse_args(argv)
    allocation = require_allocation()
    status_dir = Path(args.status_dir)
    status_dir.mkdir()
    status = {'allocation': allocation, 'resume_supported': False, 'pass': False,
              'output_use': 'EXPLORATORY_MVP_NOT_FOR_PRODUCTION'}

    def interrupted(number, frame):
        status['signal'] = number
        raise RuntimeError('launcher interrupted; no graceful save or resume')

    old = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        training_argv, training = training_arguments(args.arguments, args.arguments_sha256)
        budget = json_snapshot(args.budget_authorization, args.budget_sha256)
        from resource_probe import validate_budget
        validate_budget(budget, budget.get('bindings_sha256'),
                        bindings=dict(profile=training.profile, seed=training.seed))
        if (budget.get('authorized_scope') != 'exploratory_mvp' or
                training.seed not in budget.get('seeds', []) or
                type(budget.get('max_wall_seconds')) is not int or
                not 0 < args.timeout_seconds <= budget['max_wall_seconds']):
            raise ValueError('explicit seed/walltime authorization required')
        check(training)  # Direct readiness path; not --validate-only.
        ready = json_snapshot(training.readiness, training.readiness_sha256)
        import hashlib
        validate_budget(budget, hashlib.sha256(_canonical_bytes(ready['bindings'])).hexdigest(),
                        bindings=ready['bindings'])
        resource = ready['evidence']['resource_budget']
        measured = json_snapshot(resource['receipt_path'], resource['receipt_sha256'])['measurement']
        if (measured['authorization_sha256'] != args.budget_sha256 or
                measured['projected_run_seconds'] >= args.timeout_seconds):
            raise ValueError('measured budget does not fit this authorization/deadline')
        if os.path.lexists(training.output_dir):
            raise ValueError('training output exists; public resume is unsupported')
        # Retry CUDA startup only, never a training attempt. The probe touches all
        # visible devices and synchronizes; full topology is still the trainer gate.
        warmup = [sys.executable, '-c',
                  'import torch,sys\n'
                  'try:\n'
                  ' torch.cuda.init()\n'
                  'except RuntimeError as error:\n'
                  ' print(error, file=sys.stderr); sys.exit(75)\n'
                  'if torch.cuda.device_count()!=4: sys.exit(1)\n'
                  '[(torch.ones(1,device=f"cuda:{i}").sum().item(),torch.cuda.synchronize(i)) for i in range(4)]']
        for attempt in range(3):
            result = run_command(warmup, status_dir / f'warmup-{attempt}.log', 60)
            if result['exit_code'] == 0:
                break
            if result['exit_code'] != 75 or result['signal'] is not None or result['timed_out']:
                raise ValueError('CUDA warmup failed without retryable startup status')
        else:
            raise ValueError('CUDA warmup failed after three bounded attempts')
        # Recheck after warmup: launch only the exact prepared config bytes/path.
        from make_launch_readiness import accelerate_config_binding
        accelerate_config = ready['bindings']['accelerate_config']
        if accelerate_config_binding() != accelerate_config:
            raise ValueError('Accelerate config changed before launch')
        command = [sys.executable, '-m', 'accelerate.commands.launch', '--multi_gpu',
                   '--num_machines', '1', '--num_processes', '4', '--mixed_precision', 'bf16',
                   '--config_file', accelerate_config['path'], '--max_restarts', '0',
                   str(_REPO_ROOT / 'train/train_HyenaTransgenic_MVP.py'), *training_argv]
        status.update(run_command(command, status_dir / 'training.log', args.timeout_seconds))
        if status['exit_code'] != 0 or status['timed_out'] or status['signal'] is not None:
            raise ValueError('training failed/interrupted; do not resume or claim completion')
        status['pass'] = True
        status['meaning'] = 'process exited zero; inspect trainer artifact separately'
        _atomic_write(status_dir / 'PROCESS_EXIT.json', _canonical_bytes(status))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        for sig in old:
            signal.signal(sig, signal.SIG_IGN)
        (status_dir / 'PROCESS_EXIT.json').unlink(missing_ok=True)
        status['pass'] = False
        status['error'] = str(error)
        _atomic_write(status_dir / 'FAILED.json', _canonical_bytes(status))
        return 2
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    sys.exit(main())
