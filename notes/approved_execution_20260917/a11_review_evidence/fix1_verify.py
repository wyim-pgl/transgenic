"""Scoped readonly fix recheck. No implementation files are written."""
import hashlib
import importlib.util
import json
import runpy
from pathlib import Path

root = Path('/tmp/est-approved-contract-20260917')
source = root/'revision/scripts/evidence/est_molecule_closure.py'
tests = root/'revision/scripts/tests/test_est_molecule_closure.py'
text = source.read_text()
old = text.replace('    accession_libraries = defaultdict(set)\n','').replace('''        if libs[m.uid] is not None:
            accession_libraries[(m.species,m.assembly,m.arm,accessions[m.uid][0])].add(libs[m.uid])
    # Versions remain one accession across download datasets; library-scoped
    # union/latest keys must not hide contradictory resolved preparation IDs.
    require(all(len(libraries) <= 1 for libraries in accession_libraries.values()),
            'accession stem has conflicting canonical libraries')
''','')
assert len(text.splitlines())-len(old.splitlines()) == 7
assert hashlib.sha256(old.encode()).hexdigest() == '53d2f8fbf58f8851e669d25701f01d5ced02682351d7c4ea45cd6b6af144eb1f'
txt = tests.read_text()
start = txt.index("@pytest.mark.parametrize('dataset',['est/sp','other-download'])")
end = txt.index('def test_report_parts_one_survivor_and_multiple_parts_arm_separation():')
assert hashlib.sha256((txt[:start]+txt[end:]).encode()).hexdigest() == '88ddb85f0e262e7a8da1ed750f454e35228647dce7adb794299596ec6722583a'
msg = 'accession stem has conflicting canonical libraries'
try:
    runpy.run_path(str(Path(__file__).with_name('adversarial.py')))
except ValueError as exc:
    assert str(exc) == msg
else:
    raise AssertionError('Original reproduction did not fail closed')
spec = importlib.util.spec_from_file_location('fix1_fixtures', tests)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
m = t.api()
checks = ['original reproducer now rejects with expected ValueError',
          'source delta exactly seven additions; remainder original SHA256 matches',
          'tests delta one added block; all original tests byte-identical']
for dataset in ('est/sp','other-download'):
    for excluded in (False,True):
        t.test_accession_versions_reject_conflicting_canonical_libraries(dataset,excluded)
        checks.append(f'conflict rejected: dataset={dataset}, old_excluded={excluded}')
for excluded in (False,True):
    t.test_accession_version_library_alias_accepts_and_propagates(excluded)
    checks.append(f'canonical alias remains valid: old_excluded={excluded}')
t.test_distinct_accessions_different_libraries_stay_independent()
t.test_unknown_library_of_accession_version_still_blocks_whole_view()
checks += ['distinct accession libraries remain independent', 'unknown version library blocks whole input']
paths = [source, tests, root/'tasknotes/a11_molecule_closure_20260917.md',
         root/'tests/test_est_approved_contract_integration.py']
print(json.dumps({'checks':checks,'sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2))
