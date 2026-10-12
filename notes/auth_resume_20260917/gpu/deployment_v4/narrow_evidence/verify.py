"""Local, read-only package review; writes evidence only beside this script."""
from pathlib import Path
import hashlib, json, tarfile, difflib, sys, unittest, io, tempfile
E = Path(__file__).resolve().parent
D = E.parent
V = D / 'release_v6'
O = D.parent / 'deployment_v3/release_v5'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def files(p): return {str(x.relative_to(p)) for x in p.rglob('*') if x.is_file()}
assert files(V) == files(O) and len(files(V)) == 14
pins = {'release_v6.tar':'26c1dc0a233a95ab86977813e68310eccfce91b081de3a63586cfd7b9427628a', 'release_v6/release.json':'64a547b8ba5f54e87c1985f278564a9ea9fc7c8f3af07f6b6afe3bdf17f4f7d4', 'release_v6/release.sha256':'74a740ea154532aa3c0b807d052fb9c05bdf02fcfb6fd86845d348be72f70d9b', 'release_v6/code.candidate.tar':'da161088fb401dbe87382685c1b5faadc17c7255d4a89f705fe1126ecc5bcec7', 'scontrol-job-3163986.txt':'95d819c95d7665cb830f177c75f62db80585245d77dda26ac48159c80ad26679'}
for n,h in pins.items(): assert sha(D/n)==h, n
with tarfile.open(D/'release_v6.tar') as t:
    members=t.getmembers()
    assert len(members)==14 and {m.name for m in members}==files(V)
    for m in members:
        assert m.isfile() and not m.issym() and not m.islnk()
        assert t.extractfile(m).read()==(V/m.name).read_bytes()
ledger=[line.split('  ') for line in (V/'release.sha256').read_text().splitlines()]
assert len(ledger)==13 and {n for h,n in ledger}==files(V)-{'release.sha256'}
for h,n in ledger: assert sha(V/n)==h
old=json.loads((O/'release.json').read_text()); new=json.loads((V/'release.json').read_text())
assert set(new['release_files'])==files(V)-{'release.json','release.sha256'}
for n,h in new['release_files'].items(): assert sha(V/n)==h
assert {k:v for k,v in old.items() if k!='release_files'}=={k:v for k,v in new.items() if k!='release_files'}
production=['admission.py','deploy.py','build_release.py','p1.sbatch','authorization.example.json','controller_execution_scope.json']
diffs=[]
for n in sorted(files(V)):
    x=(O/n).read_bytes(); y=(V/n).read_bytes()
    if n in production:
        assert (D/n).read_bytes()==y
        normalized=y.decode().replace('v6','v5')
        if n=='admission.py': normalized=normalized.replace("k = 'CPUsPerTask' if m[1] == 'CPUs/Task' else m[1]", 'k = m[1]')
        assert normalized==x.decode(), n
        diffs.extend(difflib.unified_diff(x.decode().splitlines(True),y.decode().splitlines(True),fromfile='v5/'+n,tofile='v6/'+n))
    elif n not in ['release.json','release.sha256']: assert x==y,n
(E/'production.diff').write_text(''.join(diffs))
before=json.loads((D/'preservation-before.json').read_text()); after=json.loads((D/'preservation-after.json').read_text())
assert len(before)==305
assert after['unchanged_files']==305 and after['status']=='PASS' and after['transport_members']==14
for n,h in before.items(): assert sha(D.parent/n)==h,n
sys.path[:0]=[str(V),str(D)]
import admission as a
import test_field_alias as tests
import test_admission_cases as cases
assert Path(a.__file__).resolve()==V/'admission.py'
(E/'tmp').mkdir(exist_ok=True); tempfile.tempdir=str(E/'tmp')
stream=io.StringIO()
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(tests.FieldAliasTests))
(E/'unittest.txt').write_text(stream.getvalue())
assert result.wasSuccessful() and result.testsRun==5
raw=tests.RAW.decode().strip()
# Reproduce old parser without loading or modifying old deployment modules.
ns={}; exec('import re\ndef need(ok,msg):\n if not ok: raise ValueError(msg)\n'+__import__('inspect').getsource(a.fields).replace("k = 'CPUsPerTask' if m[1] == 'CPUs/Task' else m[1]",'k = m[1]'),ns)
assert 'CPUsPerTask' not in ns['fields'](raw)
# Historical PENDING output has NumCPUs64 and a predicted future start.
# Exercise resource guards with the existing RUNNING fixture, using real alias spelling.
_,current,jobs,_=cases.fixture()
j=a.fields(' '.join(('CPUs/Task' if k=='CPUsPerTask' else k)+'='+v for k,v in jobs[current].items()))
j['NumCPUs']='288'
j['AllocTRES']=j['AllocTRES'].replace('cpu=64','cpu=288')
cap={'NodeName':j['NodeList'],'CPUTot':'288'}
assert j['CPUsPerTask']=='64' and j['NumCPUs']=='288'
assert a.allocation(j,current,cases.NOW,cap)==j['NodeList']
checks=[]
def reject(label,job,capacity=cap):
    try: a.allocation(job,current,cases.NOW,capacity)
    except ValueError: checks.append(label)
    else: raise AssertionError(label)
for val in ['32','0','65','']:
    reject('task CPUs '+repr(val),dict(j,CPUsPerTask=val))
k=dict(j); k.pop('CPUsPerTask'); reject('missing task CPUs',k)
reject('requested CPU32',dict(j,ReqTRES=j['ReqTRES'].replace('cpu=64','cpu=32')))
reject('allocated mismatch',dict(j,NumCPUs='287'))
reject('untrusted capacity',j,None)
reject('wrong capacity',j,dict(cap,CPUTot='287'))
reject('wrong capacity node',j,dict(cap,NodeName='gh999'))
k=dict(j,NumCPUs='64',AllocTRES=j['AllocTRES'].replace('cpu=288','cpu=64'))
assert a.allocation(k,current,cases.NOW)==j['NodeList']
summary={'status':'PASS','pins':pins,'transport_members':14,'ledger_entries':13,'preserved_files':305,'new_unittest_pass':5,'cpu_rejections':checks,'accepted_cpu_allocations':[288,64],'controller_suite':'Existing controller_all_tests.txt: 47 PASS, 11.273s; not rerun','source_candidate_unchanged':True,'production_diff':'Only alias-before-duplicate and v5 to v6 identities; release metadata differs only in refreshed file hashes'}
(E/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
