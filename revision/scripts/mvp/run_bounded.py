#!/usr/bin/env python3
"""Bound operator-reviewed probe argv; never substitute CPU fixtures for GPU receipts.

A probe writes measured JSON to MVP_MEASUREMENT_OUTPUT. This runner supplies
binding identity in MVP_BINDINGS_SHA256, requires successful measured checks,
and publishes a digest-bound DONE last. SIGTERM/timeout never means resume.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

from make_launch_readiness import (EVIDENCE_RECEIPTS, _atomic_write, _canonical_bytes,
                                   _sha256, json_snapshot, tooling_identity, require_allocation)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_step0(value):
    if (not isinstance(value, dict) or value.get('deterministic_algorithms') is not True or
            value.get('cublas_workspace_config') != ':4096:8' or
            value.get('comparison') != 'bitwise' or value.get('eval_mode') is not True or
            set(value.get('tiers', {})) != {'30720', '61440', '129024'}):
        raise ValueError('step0 requires deterministic eval bit-identity at all three tiers')
    for row in value['tiers'].values():
        if (not isinstance(row, dict) or not digest(row.get('input_sha256')) or
                row.get('finite') is not True or
                not digest(row.get('b5_logits_sha256')) or
                row.get('b5_logits_sha256') != row.get('mvp_off_logits_sha256') or
                row.get('b5_logits_sha256') != row.get('mvp_zero_projection_logits_sha256') or
                row.get('b5_logits_sha256') != row.get('repeat_logits_sha256')):
            raise ValueError('step0 missing, nonfinite or unequal logits')


def validate_measurement(name, value):
    if not isinstance(value, dict) or value.get('device') != 'cuda' or value.get('real_runtime') is not True:
        raise ValueError('measured real CUDA runtime required')
    if 'probe_identity' in value or 'mode' in value:
        from real_probes import validate_identity
        if value.get('mode') != name:
            raise ValueError('real probe mode/receipt mismatch')
        validate_identity(value)
    if name == 'real_hyena_step0_three_tiers':
        validate_step0(value)
    elif name == 'four_rank_smoke':
        from transgenic.training.mvp_cli_runtime import _validate_public_hardware
        _validate_public_hardware(value.get('hardware', []))
        if type(value.get('optimizer_steps')) is not int or value['optimizer_steps'] != 1 or value.get('finite') is not True:
            raise ValueError('four-rank smoke must complete one finite optimizer step')
    elif name == 'resource_budget':
        from resource_probe import validate_resource
        validate_resource(value)
    else:
        checks = {
            'real_tokenizer': ('tokenizer_roundtrip', 'hyena_loaded'),
            'generation_cache_parity': ('cached_uncached_equal', 'evidence_off_equal'),
            'bf16_compile': ('native_bf16', 'inductor', 'finite', 'no_eager_fallback'),
        }[name]
        if any(value.get(key) is not True for key in checks):
            raise ValueError(f'measured checks failed: {name}')


def run_command(argv, log_path, timeout, *, env=None):
    """Single attempt, bounded log, process-group teardown even after leader exits."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('positive bounded timeout required')
    if not isinstance(argv, list) or not argv or any(not isinstance(a, str) or not a for a in argv) or not Path(argv[0]).is_absolute():
        raise ValueError('nonempty argv list required')
    received = []
    old = {sig: signal.signal(sig, lambda number, frame: received.append(number))
           for sig in (signal.SIGTERM, signal.SIGINT)}
    child = None
    timed_out = False
    started = time.monotonic()
    try:
        with Path(log_path).open('xb') as log:
            child = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=env)
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                written = 0
                while child.poll() is None:
                    timed_out = time.monotonic() - started >= timeout
                    if received or timed_out:
                        break
                    for key, _ in selector.select(0.05):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                        if written < 1024 * 1024:
                            chunk = chunk[:1024 * 1024 - written]
                            log.write(chunk)
                            written += len(chunk)
            # A command cannot leave grandchildren running after any outcome.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        return {'exit_code': child.returncode if not received and not timed_out else
                -(received[0] if received else signal.SIGTERM),
                'signal': received[0] if received else (-child.returncode if child.returncode < 0 else None),
                'timed_out': timed_out, 'elapsed_seconds': time.monotonic() - started}
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        for sig, handler in old.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', choices=tuple(EVIDENCE_RECEIPTS), required=True)
    parser.add_argument('--bindings', required=True)
    parser.add_argument('--bindings-file-sha256', required=True)
    parser.add_argument('--command-file', required=True, help='reviewed JSON argv array')
    parser.add_argument('--command-sha256', required=True)
    parser.add_argument('--output-dir', required=True, help='fresh, per-probe directory')
    parser.add_argument('--timeout-seconds', type=float, required=True)
    parser.add_argument('--warmup-command-file')
    parser.add_argument('--warmup-sha256')
    parser.add_argument('--warmup-attempts', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--warmup-timeout-seconds', type=float, default=60)
    args = parser.parse_args(argv)
    allocation = require_allocation()
    bindings = json_snapshot(args.bindings, args.bindings_file_sha256)
    command = json_snapshot(args.command_file, args.command_sha256)
    if bindings['bindings'].get('tooling') != tooling_identity():
        raise ValueError('preflight tooling changed since prepare')
    binding_sha = hashlib.sha256(_canonical_bytes(bindings['bindings'])).hexdigest()
    if bindings['bindings_sha256'] != binding_sha:
        raise ValueError('binding identity mismatch')
    output = Path(args.output_dir).absolute()
    output.mkdir()  # Collision fails; no automatic resume or overwritten receipts.
    measurement = output / 'measurement.json'
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8',
               MVP_BINDINGS_SHA256=binding_sha, MVP_MEASUREMENT_OUTPUT=str(measurement),
               MVP_PROBE_TIMEOUT_SECONDS=str(args.timeout_seconds))
    receipt = {'schema_version': 1, 'preflight': EVIDENCE_RECEIPTS[args.name],
               'bindings_sha256': binding_sha, 'command_sha256': args.command_sha256,
               'allocation': allocation, 'pass': False, 'resume_supported': False}
    def interrupted(number, frame):
        receipt['signal'] = number
        raise RuntimeError('preflight interrupted before publication')

    old = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        if args.warmup_command_file:
            warmup = json_snapshot(args.warmup_command_file, args.warmup_sha256)
            receipt['warmup_attempts'] = []
            for attempt in range(args.warmup_attempts):
                status = run_command(warmup, output / f'warmup-{attempt}.log',
                                     args.warmup_timeout_seconds, env=dict(env, MVP_MEASUREMENT_OUTPUT=str(output / 'warmup.json')))
                receipt['warmup_attempts'].append(status)
                if status['exit_code'] == 0:
                    break
                # Only reviewed warmup's EX_TEMPFAIL (75) is retryable.
                if status['exit_code'] != 75:
                    raise ValueError('warmup failed without retryable status')
            else:
                raise ValueError('warmup retries exhausted')
        receipt.update(run_command(command,
                                   output / 'probe.log', args.timeout_seconds, env=env))
        if receipt['exit_code'] != 0 or receipt['timed_out'] or receipt['signal'] is not None:
            raise ValueError('probe interrupted/failed; no completion or resume')
        value = json.loads(measurement.read_text())
        if value.get('bindings_sha256') != binding_sha:
            raise ValueError('measurement binding mismatch')
        validate_measurement(args.name, value)
        receipt['measurement'] = value
        receipt['pass'] = True
        path = output / (args.name + '.json')
        _atomic_write(path, _canonical_bytes(receipt))
        _atomic_write(output / (args.name + '.DONE'), (_sha256(path) + '\n').encode())
        return 0
    except (Exception, KeyboardInterrupt) as error:
        for sig in old:
            signal.signal(sig, signal.SIG_IGN)
        (output / (args.name + '.DONE')).unlink(missing_ok=True)
        receipt['pass'] = False
        receipt['error'] = str(error)
        _atomic_write(output / (args.name + '.FAILED.json'), _canonical_bytes(receipt))
        return 2
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    sys.exit(main())
