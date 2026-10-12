#!/usr/bin/env python3
"""Read-only local integrity preflight. NOT GPU readiness or submission approval."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def sha(data):
    return hashlib.sha256(data).hexdigest()

def canonical(mapping):
    return sha((json.dumps(mapping,sort_keys=True,separators=(',',':'))+'\n').encode())

def main():
    ledger = json.loads((HERE/'ledger.json').read_text())
    for r in ledger['evidence']:
        p = HERE / r['path']
        assert p.is_file() and HERE in p.resolve().parents
        data = p.read_bytes()
        assert len(data)==r['size_bytes'] and sha(data)==r['sha256'], r['path']
    for source in ledger['reference_documents'].values():
        assert (ROOT/source).exists(), source
    for job in ledger['jobs']:
        assert (HERE/job['evidence']).is_file()
    release = json.loads((HERE/'captured/release.json').read_text())
    assert sha((HERE/'captured/code.tar').read_bytes()) == ledger['source']['release_archive_sha256']
    assert sha((HERE/'captured/release.sha256').read_bytes()) == ledger['source']['release_ledger_sha256']
    with tarfile.open(HERE/'captured/code.tar') as archive:
        assert all(m.isfile() or m.isdir() for m in archive.getmembers())
        members = [m for m in archive.getmembers() if m.isfile()]
        assert len(members)==100
        blobs = {m.name:archive.extractfile(m).read() for m in members}
    assert set(blobs)==set(release['source_files'])
    for p,data in blobs.items():
        assert sha(data)==release['source_files'][p], p
        expected = subprocess.check_output(['git','-C',str(ROOT),'show',ledger['source']['commit']+':'+p])
        assert data == expected, p
    components = {p:sha(data) for p,data in blobs.items() if p.startswith(('src/transgenic/','revision/scripts/mvp/','configs/')) or p=='train/train_HyenaTransgenic_MVP.py'}
    assert len(components)==86 and canonical(components)==ledger['source']['relative_pin']
    tree = ast.parse(blobs['revision/scripts/mvp/mvp_decode_runtime.py'])
    paths = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DECODER_ONLY' for t in n.targets))
    assert len(paths)==13 and canonical({p:sha(blobs[p]) for p in paths})==ledger['source']['decoder_pin']
    assert ledger['budget']['actual_node_hours_at_snapshot']+ledger['budget']['reserved_node_hours'] <= ledger['budget']['cap_node_hours']
    assert ledger['native_crash']['reruns_this_task']==0
    print(json.dumps(dict(status='PASS_LOCAL_INTEGRITY_ONLY',captured_files=len(ledger['evidence']),archive_git_files=len(blobs),component_files=len(components),decoder_files=len(paths),remote_queries=0,gpu_readiness=False),indent=2))

if __name__=='__main__':
    main()
