#!/usr/bin/env python3
"""Bounded tiny-fixture TERM probes, no real data or runtime access."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch
BASE = Path(__file__).resolve().parent
PREP = Path('/mnt/f/scratch/develop/Transgenic/transgenic_local/notes/seed123_execution_20260917/est_deployment')
if len(sys.argv) == 2:
    phase = sys.argv[1]
    spec = importlib.util.spec_from_file_location('deployment_test', PREP / 'test_deployment.py')
    t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
    case = t.Deployment('test_exception_still_postchecks'); case.setUp(); case.build()
    calls = [0]
    def runtime(manifest):
        calls[0] += 1
        if calls[0] == (1 if phase == 'before' else 2):
            (BASE / ('signal-' + phase + '.ready')).write_text('ready\n')
            time.sleep(15)
        return {'fixture': True}
    with patch.object(case.runner, 'runtime_identity', side_effect=runtime):
        case.runner.run(case.release, case.sha, BASE / ('signal-' + phase + '-output'))
else:
    results = []
    for phase in ('before', 'after'):
        env = dict(os.environ, TMPDIR=str(BASE), PYTHONDONTWRITEBYTECODE='1')
        command = [sys.executable, '-B', str(Path(__file__).resolve()), phase]
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        deadline = time.monotonic() + 10
        while not (BASE / ('signal-' + phase + '.ready')).exists() and p.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert (BASE / ('signal-' + phase + '.ready')).exists(), 'fixture failed to become ready'
        p.send_signal(signal.SIGTERM)
        stdout, stderr = p.communicate(timeout=5)
        (BASE / ('signal-' + phase + '.stdout')).write_bytes(stdout)
        (BASE / ('signal-' + phase + '.stderr')).write_bytes(stderr)
        out = BASE / ('signal-' + phase + '-output')
        results.append(dict(phase=phase, command=command, returncode=p.returncode,
                            control_result_exists=(out / 'control/result.json').exists(),
                            worker_complete_exists=(out / 'sample/COMPLETE').exists(),
                            verification_exists=(out / 'control/verification.json').exists()))
    with (BASE / 'signal-probe.json').open('x') as f:
        json.dump(results, f, indent=2); f.write('\n')
    print(json.dumps(results, indent=2))
