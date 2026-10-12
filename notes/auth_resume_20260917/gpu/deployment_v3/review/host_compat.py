"""Local driver: bounded remote STDIN-only Python 3.6 checks, no remote writes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent.parent
sources = {n: (HERE/n).read_text() for n in ('admission.py', 'deploy.py')}
payload = """import sys,types,json,subprocess,io
sources=json.loads(%r)
a=types.ModuleType('admission'); sys.modules['admission']=a
exec(compile(sources['admission.py'],'admission.py','exec'),a.__dict__)
d=types.ModuleType('deployment_host_test')
exec(compile(sources['deploy.py'],'deploy.py','exec'),d.__dict__)
assert sys.version_info[:2] == (3,6), sys.version
p=subprocess.Popen(['/bin/true'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,start_new_session=True)
assert p.communicate(timeout=5) == ('','') and p.returncode == 0
# Invoke real query API with an in-memory receipt sink; /bin/true's empty output must fail closed.
class Sink:
    def __truediv__(self, other): return self
    def open(self, mode): return io.StringIO()
try: a.query(Sink(),'host-api',['/bin/true'],timeout=5)
except ValueError as e: assert str(e) == 'failed/incomplete observation'
else: raise AssertionError('empty observation accepted')
x=a.example(); x.update(approved=True,issued_at_unix=1000,controller_reference='SYNTHETIC',review_reference='SYNTHETIC',terminal_recovery_reference='SYNTHETIC',staging_reference='SYNTHETIC',reservation_identity='SYNTHETIC',release_json_sha256='a'*64,transport_sha256='b'*64,ledger_sha256='c'*64)
a.validate_authorization(x,'a'*64,'b'*64,'c'*64,now=87400)
assert a.node_capacity('NodeName=gh018 CPUTot=288','gh018')['CPUTot']=='288'
assert d.EVIDENCE_BYTES+d.HF_COPY_MAX+d.CHECKPOINT_COPY_MAX < d.STAGING_MAX
print(json.dumps({'python':sys.version,'syntax_modules':list(sources),'stdlib_api':'Popen universal_newlines, timeout, session, query in-memory sink','durable_24h_authority':'synthetic pass','node_capacity':'synthetic pass','remote_files_written':False},sort_keys=True))
""" % json.dumps(sources)
argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'pronghorn',
        'ssh -o BatchMode=yes -o ConnectTimeout=10 dtai /usr/bin/python3 -B -']
r = subprocess.run(argv, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                   universal_newlines=True, timeout=45)
record = dict(argv=argv, stdin_sha256=hashlib.sha256(payload.encode()).hexdigest(),
              sources_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()},
              exit_code=r.returncode, stdout=r.stdout, stderr=r.stderr)
with (HERE/'review/host-compat-streaming-result.json').open('x') as out: json.dump(record,out,indent=2); out.write('\n')
print(r.stdout, r.stderr)
sys.exit(r.returncode)
