"""Exercise only the selected release's pure gate validator, never host()."""
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent / 'release_v2'
spec = importlib.util.spec_from_file_location('reviewed_deploy', RELEASE / 'deploy.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
release_sha = hashlib.sha256((RELEASE / 'release.json').read_bytes()).hexdigest()
gate = dict(scope='S0_P1_DIAGNOSTIC_ONLY', profile='development-single', seed=123,
            training_authorized=False, release_json_sha256=release_sha,
            measured_at_unix=1000, prior_diagnostic_node_hours=0,
            other_pending_reserved_node_hours=0,
            accounting_record_reference='TEST_FIXTURE_NOT_AUTHORITY',
            review_record_reference='TEST_FIXTURE_NOT_AUTHORITY',
            full_accounting_reviewed=True, no_overlap=True,
            new_package_independent_review=True, old_owned_pending_zero_confirmed_cancel=True,
            staging_ownership_mapping_reviewed=True, b5_exclusions_reviewed=True)
results = []
with patch.object(d.subprocess, 'run', side_effect=AssertionError('no external query permitted')):
    for age in (-1, 0, 300, 301, 3600, 86400):
        try:
            d.validate_controller_gate(gate, release_sha, now=1000+age)
            outcome = 'ACCEPTED'
        except ValueError as e:
            outcome = str(e)
        results.append(dict(age_seconds=age, outcome=outcome))
assert [r['age_seconds'] for r in results if r['outcome']=='ACCEPTED'] == [0,300]
canonical = lambda value: (json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode()
old_hash = hashlib.sha256(canonical(gate)).hexdigest()
new_hash = hashlib.sha256(canonical(dict(gate,measured_at_unix=1001))).hexdigest()
assert old_hash != new_hash
text = (RELEASE/'p1.sbatch').read_text()
assert text.index('> S0_P1_STARTED') < text.index('deploy.py host')
result = dict(fixture_only_not_authority=True, outcomes=results,
              changed_measurement_changes_external_pin=True,
              shell_claim_precedes_host_gate=True,
              pure_validator_uses_no_live_scheduler_or_accounting_query=True)
(HERE/'gate-boundary-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
