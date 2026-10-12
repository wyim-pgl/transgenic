#!/usr/bin/env python3
"""LOCAL ONLY: produce a fresh controller-transfer bundle from exact git objects."""
import io
from pathlib import Path
import shutil
import subprocess
import tarfile
import hashlib
from deploy import COMMIT, ACCEPT, INVENTORY, RECEIPTS, CONFIG, PANEL, save, require, sha

HERE = Path(__file__).resolve().parent
PATHS = ['src', 'train', 'revision/scripts/mvp', 'configs', PANEL]
MAIN = HERE.parents[2]
REPO = str(MAIN)  # shared Git object database; no dependency on a temporary worktree
SMALL_PINS = {
    ACCEPT: 'faab8387cbbe87812c7f3faf5412d19004874a5cc439f5a3d182e2c976a5b51e',
    INVENTORY: '9f79019679bfdd25b0a2bdac7cae63d01fa6d874be03fa3ef3f81cfd0a880a98',
    RECEIPTS: '73204a4ed98624b1567a5eb26fd6b2bbafbb597e0bd4b807685e8c703c807bac',
    PANEL: '79367154bbf922fd0554285cb6cb47ae3d3c866b145aa1f51ead4f47b00cd243',
    CONFIG: '9b8bef9b642526835a0817e9b13ae50b7c7f274fe882256426cb78f90c080d79',
}


def main():
    out = HERE / 'release'
    out.mkdir()  # no overwrite, no mutable working-tree source reads
    archive = subprocess.check_output(['git', '-C', REPO, 'archive', '--format=tar', COMMIT, *PATHS])
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            require(member.isdir() or member.isfile(), 'archive contains a symlink/special file')
            if member.isfile():
                payload = tar.extractfile(member).read()
                git_bytes = subprocess.check_output(['git', '-C', REPO, 'show', COMMIT + ':' + member.name])
                require(payload == git_bytes, 'export attributes modified git source bytes')
                files[member.name] = hashlib.sha256(payload).hexdigest()
    for name in (CONFIG, PANEL):
        require(files[name] == SMALL_PINS[name], 'commit input pin changed: ' + name)
    # Accepted JSONs are untracked main artifacts, NOT present in d99b. Keep that provenance explicit.
    companions = {}
    for name in (ACCEPT, INVENTORY, RECEIPTS):
        require(sha(MAIN / name) == SMALL_PINS[name], 'accepted companion pin changed: ' + name)
        dest = out / name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(MAIN / name, dest)
        require(sha(dest) == SMALL_PINS[name], 'companion changed during copy')
        companions[name] = SMALL_PINS[name]
    (out / 'code.tar').write_bytes(archive)
    for name in ('deploy.py', 'p1.sbatch'):
        shutil.copyfile(HERE / name, out / name)
    release_files = {name: sha(out / name) for name in ('code.tar', 'deploy.py', 'p1.sbatch')}
    release_files.update(companions)
    save(out / 'release.json', {'commit': COMMIT, 'source_files': files, 'release_files': release_files,
         'accepted_companions_not_in_source_commit': companions,
         'small_input_pins': SMALL_PINS, 'scope': 'S0_P1_DIAGNOSTIC_ONLY', 'node_hours_max': 0.5})
    with (out / 'release.sha256').open('x') as f:
        for name in (*release_files, 'release.json'):
            f.write(f'{sha(out / name)}  {name}\n')
    print('Release:', out)
    print('code.tar SHA256:', sha(out / 'code.tar'))
    print('release.json SHA256:', sha(out / 'release.json'))
    print('release.sha256 SHA256:', sha(out / 'release.sha256'))
    print('source files:', len(files), 'archive bytes:', len(archive))


if __name__ == '__main__':
    main()
