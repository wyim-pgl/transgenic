"""Local capture verification only; never opens captured remote paths."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

root = Path('/tmp/est-real-source-binding-20260917')
notes = Path('/mnt/f/scratch/develop/Transgenic/transgenic_local/notes/seed123_execution_20260917')
cap = notes / 'est_source_metadata.txt'
raw = json.loads(cap.read_text())
manifest_path = root / 'tasknotes/real_sample/sitalica.manifest.json'
m = json.loads(manifest_path.read_text())
assert runpy.run_path(str(root / 'tasknotes/real_sample/build_manifest.py'))['build'](cap) == m
assert hashlib.sha256(cap.read_bytes()).hexdigest() == m['metadata_capture_sha256']
for s in m['searches']:
    provspec = m['files'][s['c1_provenance']]
    cmdspec = m['files'][s['c1_commands']]
    p = json.loads(raw[provspec['path']])
    for spec in (provspec, cmdspec):
        assert hashlib.sha256(raw[spec['path']].encode()).hexdigest() == spec['sha256']
    assert m['files'][s['raw']]['sha256'] == p['raw_sha256']
    assert m['files'][s['protein']]['sha256'] == p['filtered_fasta_sha256']
    assert m['files']['screened']['sha256'] == p['inputs']['query_sha256']
    print(s['source'], json.dumps({k: p[k] for k in ('diamond_sha256', 'diamond_version', 'db_sha256', 'raw_sha256', 'filtered_fasta_sha256', 'launcher_sha256')}))
a = m['files']['align_provenance']
assert hashlib.sha256(raw[a['path']].encode()).hexdigest() == a['sha256']
fields = {line.split()[0]: line.split()[1] for line in raw[a['path']].splitlines() if len(line.split()) >= 2}
for role, name in [('original', 'est_raw'), ('reference', 'reference'), ('bam', 'bam'), ('screened', 'est')]:
    assert m['files'][role]['md5'] == fields[name + '_md5']
    assert m['files'][role]['path'] == fields[name]
print('manifest_file_sha256', hashlib.sha256(manifest_path.read_bytes()).hexdigest())
print('capture_sha256', hashlib.sha256(cap.read_bytes()).hexdigest())
print('ALL LOCAL CAPTURE BINDINGS MATCH; no remote bytes verified')
# Compute local static relative import closure, including package initializers if present.
todo = [root / 'revision/scripts/evidence/est_real_sample.py']
seen = set()
while todo:
    path = todo.pop()
    if path in seen:
        continue
    seen.add(path)
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            assert node.level == 1
            dep = path.parent / ((node.module or '') + '.py')
            if dep.exists():
                todo.append(dep)
    init = path.parent / '__init__.py'
    if init.exists() and init not in seen:
        todo.append(init)
for path in sorted(seen | {manifest_path, root / 'tests/test_est_real_sample.py', root / 'tasknotes/real_sample/build_manifest.py', root / 'tasknotes/real_sample/REPORT.md', root / 'tasknotes/real_sample/compute_proposal.sbatch'}):
    print(hashlib.sha256(path.read_bytes()).hexdigest(), path.relative_to(root))
print('This is review-time local inventory, NOT deployed package/runtime verification.')
