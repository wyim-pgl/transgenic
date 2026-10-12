import hashlib,json,subprocess
from pathlib import Path
r=Path('/tmp/est-real-source-binding-20260917')
def git(*a): return subprocess.check_output(['git','-C',str(r),*a]).decode()
paths=set(git('ls-files').splitlines())
paths.update(str(p.relative_to(r)) for base in ('revision/scripts/evidence','tests','tasknotes/real_sample') for p in (r/base).rglob('*') if p.is_file() and '__pycache__' not in str(p))
print(json.dumps({'head':git('rev-parse','HEAD'),'status':git('status','--porcelain'),'index':hashlib.sha256(Path(git('rev-parse','--git-path','index').strip()).read_bytes()).hexdigest() if Path(git('rev-parse','--git-path','index').strip()).is_absolute() else 'relative-index','files':{p:hashlib.sha256((r/p).read_bytes()).hexdigest() for p in sorted(paths) if (r/p).is_file()}},sort_keys=True,indent=2))
