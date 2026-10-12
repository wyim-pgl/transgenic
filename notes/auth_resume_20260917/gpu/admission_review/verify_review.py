"""Local-only review of immutable deployment_v2. No real scheduler or remote calls."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / 'deployment_v2'
sys.dont_write_bytecode = True
sys.path.insert(0, str(TARGET))
import admission as a
import test_admission_cases as cases

tempfile.tempdir = str(HERE / 'tmp')
Path(tempfile.tempdir).mkdir(exist_ok=True)
result = {'target_sha256': {n: hashlib.sha256((TARGET/n).read_bytes()).hexdigest()
          for n in ('admission.py', 'deploy.py', 'p1.sbatch', 'test_admission_cases.py', 'test_queued_gate.py')}}

# A non-timeout communication failure must not leave the child/group running.
with tempfile.TemporaryDirectory() as tmp:
    proc = Mock(pid=123456, returncode=None)
    proc.communicate.side_effect = OSError('synthetic pipe I/O failure')
    with patch.object(a.subprocess, 'Popen', return_value=proc), patch.object(a.os, 'killpg') as killpg:
        try:
            a.query(Path(tmp), 'communication-error', ['/fixture-only'])
        except ValueError as exc:
            result['communication_error'] = {'admission_rejected': str(exc),
                'killpg_calls': killpg.call_count, 'wait_calls': proc.wait.call_count,
                'communicate_calls': proc.communicate.call_count,
                'record': json.loads((Path(tmp)/'communication-error.json').read_text())}
    assert result['communication_error']['killpg_calls'] == 0
    assert result['communication_error']['wait_calls'] == 0

# The advertised output ceiling is checked only after full capture and persistence.
with tempfile.TemporaryDirectory() as tmp:
    proc = Mock(returncode=0)
    proc.communicate.return_value = ('x' * (16*1024*1024+1), '')
    with patch.object(a.subprocess, 'Popen', return_value=proc):
        try:
            a.query(Path(tmp), 'over-limit', ['/fixture-only'])
        except ValueError as exc:
            result['output_limit'] = {'rejection': str(exc),
                'persisted_bytes_before_rejection': (Path(tmp)/'over-limit.json').stat().st_size,
                'full_stdout_chars': len(json.loads((Path(tmp)/'over-limit.json').read_text())['stdout'])}
    assert result['output_limit']['full_stdout_chars'] == 16*1024*1024+1

# Check recorded source identity and conversion, without running its commands.
raw = json.loads((TARGET/'review/machine-probe.json').read_text())['stdout']
start = raw.index('#!/usr/bin/env perl\n')
end = raw.index("\nProject Summary for User 'wcyim':", start)
source = raw[start:end]
variants = [source, source.rstrip('\n')+'\n']
matched = next((s for s in variants if hashlib.sha256(s.encode()).hexdigest() == a.ACCOUNTS_SHA), None)
assert matched is not None
qos_line = next(s for s in raw.splitlines() if s.startswith('QOS='+a.ACCOUNT+'('))
result['finance_grounding'] = {'recorded_source_hash_matches_pin': True,
    'recorded_script_bytes': len(matched.encode()), 'qos_record': qos_line,
    'conversion_lines': [s.strip() for s in matched.splitlines()
                         if '$SLURM_XPF' in s or '$bal = floor' in s or '$dep = floor' in s]}

(HERE/'review_evidence.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'finance_grounding'}, indent=2))
