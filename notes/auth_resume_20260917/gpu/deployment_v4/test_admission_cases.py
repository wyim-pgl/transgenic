"""Synthetic Slurm/accounting fixtures; no scheduler commands in this suite."""
import copy
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import admission as a
import deploy as d

NOW = 1800000000

def fixture():
    current = '4000000'
    job = dict(JobId=current, JobName='dev123-p1-v6-0917', UserId='wcyim(97434)',
               Account=a.ACCOUNT, QOS=a.ACCOUNT, Partition='ghx4', JobState='RUNNING',
               OverSubscribe='NO', NumNodes='1', NumCPUs='64', CPUsPerTask='64', NumTasks='1', Requeue='0', Restarts='0',
               Command=a.NAMESPACE+'/p1.sbatch', WorkDir=a.NAMESPACE, TimeLimit='00:30:00',
               RunTime='00:01:00', AllocTRES='billing=4000,cpu=64,gres/gpu=4,node=1',
               ReqTRES='billing=4000,cpu=64,gres/gpu=4,node=1', NodeList='gh018',
               ExcNodeList='gh[062,093]', SubmitTime='2026-09-17T00:00:00', StartTime='2026-09-18T00:00:00')
    row = dict(id=current,name=job['JobName'],user='wcyim',uid='97434',account=a.ACCOUNT,
               partition='ghx4',state='RUNNING',elapsed='60',nodes='1',cpus='64',
               alloc=job['AllocTRES'],req=job['ReqTRES'],limit='30',submit=job['SubmitTime'],
               start=job['StartTime'],end='Unknown',placement='gh018',exit='0:0')
    rows = {current:row}
    for ident in ['3163545','3163986']:
        old = dict(row, id=ident,name='mvp8-p1',state='CANCELLED',elapsed='0',nodes='0',cpus='0',
                   alloc='',start='None',end='2026-09-17T01:00:00',placement='None assigned')
        rows[ident] = old
    rows['3163986'].update(state='FAILED',elapsed='51',nodes='1',cpus='288',exit='1:0',
        alloc='billing=4000,cpu=288,gres/gpu=4,node=1',start='2026-09-17T18:47:46',
        end='2026-09-17T18:48:37',placement='gh077')
    b5 = dict(job, JobId='3149356', JobName='b5-seed', JobState='PENDING',
              Command='/work/nvme/bilv/wcyim/transgenic/repo_seeds/deploy/deltaai/train_b5.slurm',
              TimeLimit='11:00:00', AllocTRES='(null)', NodeList='')
    return a.example(), current, {current:job, '3149356':b5}, rows

class AdmissionCases(unittest.TestCase):
    def test_fresh_full_reconciliation_self_once_and_b5_maximum(self):
        auth, current, jobs, rows = fixture()
        r = a.reconcile(auth,current,jobs,rows,NOW)
        self.assertEqual((r['node_hours'],r['gpu_hours']), (1.5+51/3600,6+204/3600))
        self.assertEqual(r['b5_max_gpu_hours'],[44])

    def test_bad_actual_allocation_and_old_active_fail(self):
        auth, current, jobs, rows = fixture()
        for key,value in [('UserId','other(1)'),('Account','other'),('QOS','other'),
            ('Partition','ghx1'),('OverSubscribe','OK'),('NumNodes','2'),('NumCPUs','32'),
            ('Requeue','1'),('Restarts','1'),('TimeLimit','00:30:01'),('AllocTRES',''),
            ('NodeList','gh062'),('NodeList','gh093'),('Command','/old/p1.sbatch'),
            ('WorkDir','/old'),('StartTime','2099-01-01T00:00:00')]:
            changed=copy.deepcopy(jobs);changed[current][key]=value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                a.reconcile(auth,current,changed,rows,NOW)
        for key,value in [('state','RUNNING'),('elapsed','1'),('alloc','node=1')]:
            changed=copy.deepcopy(rows);changed['3163986'][key]=value
            with self.subTest(old=key), self.assertRaises(ValueError):
                a.reconcile(auth,current,jobs,changed,NOW)
        changed=copy.deepcopy(rows);del changed['3163545']
        with self.assertRaises(ValueError): a.reconcile(auth,current,jobs,changed,NOW)

    def test_unknown_diagnostic_or_pending_reservation_fails(self):
        auth,current,jobs,rows=fixture()
        jobs['9999']=dict(jobs[current],JobId='9999',JobName='mvp-extra')
        with self.assertRaises(ValueError): a.reconcile(auth,current,jobs,rows,NOW)
        jobs.pop('9999'); rows['9999']=dict(rows['3163545'],id='9999')
        with self.assertRaises(ValueError): a.reconcile(auth,current,jobs,rows,NOW)

    def test_b5_unbounded_or_same_node_fails(self):
        auth,current,jobs,rows=fixture()
        for key,value in [('TimeLimit','UNLIMITED'),('ReqTRES','cpu=64,node=1'),('NodeList','gh018')]:
            changed=copy.deepcopy(jobs);changed['3149356'][key]=value
            with self.assertRaises(ValueError): a.reconcile(auth,current,changed,rows,NOW)

    def test_step_dedup_and_duplicate_allocation_fail(self):
        auth,current,jobs,rows=fixture()
        line='|'.join(rows[current][k] for k in a.KEYS)
        step=line.replace(current,current+'.batch',1)
        self.assertEqual(set(a.accounting(line+'\n'+step)),{current})
        with self.assertRaises(ValueError): a.accounting(line+'\n'+line)
        with self.assertRaises(ValueError): a.accounting(line+'|')

    def test_reconciliation_enforces_node_and_gpu_caps_independently(self):
        auth,current,jobs,rows=fixture()
        auth['registered_prior_ids'].append('9999')
        for nodes,gpus,elapsed in [(10,4,1800),(1,40,1800)]:
            changed=copy.deepcopy(rows)
            changed['9999']=dict(rows['3163545'],id='9999',state='FAILED',elapsed=str(elapsed),
                nodes=str(nodes),cpus='64',alloc=f'billing={gpus*1000},node={nodes},gres/gpu={gpus}',
                start='2026-09-17T00:00:00')
            with self.assertRaises(ValueError): a.reconcile(auth,current,jobs,changed,NOW)

    def test_failed_admission_has_no_stage_hash_copy_rename_or_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); release=root/'release'; release.mkdir()
            (release/'release.json').write_text('{}')
            evidence=root/'staging/evidence';evidence.mkdir(parents=True)
            (evidence/'fullc0lite.db').write_text('fixture')
            with patch.object(d,'ROOT',root), patch.object(d,'SOURCE',root), \
                 patch.object(a,'NAMESPACE',str(release)), \
                 patch.dict('os.environ',SLURM_JOB_ID='4000000',SLURM_JOB_NUM_NODES='1'), \
                 patch.object(d,'controller_gate',return_value={'fixture':True}), \
                 patch.object(a,'live',side_effect=ValueError('admission blocked')), \
                 patch.object(d,'verify_source_manifest',side_effect=AssertionError('hash before admission')), \
                 patch.object(d,'copy_cache',side_effect=AssertionError('copy before admission')), \
                 patch.object(d,'run',side_effect=AssertionError('probe before admission')):
                with self.assertRaisesRegex(ValueError,'admission blocked'): d.host()
            self.assertEqual((evidence/'fullc0lite.db').read_text(),'fixture')
            job=root/'dev123-p1-v6-job-4000000'
            self.assertEqual({p.name for p in job.iterdir()},{'probe'})

class LivePathTests(unittest.TestCase):
    def run_live(self, mutation=None):
        auth,current,jobs,rows=fixture()
        auth.update(approved=True, issued_at_unix=NOW-86400, controller_reference='TEST ONLY',
            review_reference='TEST ONLY', terminal_recovery_reference='TEST ONLY',
            staging_reference='TEST ONLY', reservation_identity='TEST ONLY',
            release_json_sha256='a'*64, transport_sha256='b'*64, ledger_sha256='c'*64)
        pin=d.binding_digest(auth)
        a.validate_authorization(auth,'a'*64,'b'*64,'c'*64,now=NOW)
        queue='\n'.join('|'.join([i,j['JobName'],'wcyim',a.ACCOUNT,j['JobState'],j['NodeList']]) for i,j in jobs.items())+'\n'
        commands=[]
        real_popen=subprocess.Popen
        def launch(argv, **kwargs):
            commands.append(argv)
            if argv[0]=='/usr/bin/stat': output='67674|755|8705\n'
            elif argv[0]=='/usr/bin/sha256sum': output=a.ACCOUNTS_SHA+'  /sw/user/scripts/accounts\n'
            elif argv[0]=='/usr/bin/sacctmgr' and 'qos' in argv: output='bilv-dtai-gh|1.000000|NoDecay,UsageFactorSafe\n'
            elif argv[0]=='/usr/bin/sacctmgr': output='delta-gh|bilv-dtai-gh|wcyim|bilv-dtai-gh\n'
            elif argv==['/usr/bin/scontrol','show','config']: output='ClusterName             = delta-gh\n'
            elif argv[0]=='/usr/bin/squeue': output=queue
            elif argv[0]=='/usr/bin/sacct': output='\n'.join('|'.join(r[k] for k in a.KEYS) for r in rows.values())+'\n'
            elif argv[0]=='/sw/user/scripts/accounts': output="Project Summary for User 'wcyim':\n\nAccount                        Balance(Hours)   Deposited(Hours)  Project\n----------------------------  ----------------  ----------------  ----------------------\nbilv-dtai-gh                              1154              1499  example\n"
            elif argv[0]=='/usr/bin/scontrol' and 'node' in argv: output='NodeName=gh018 CPUTot=288\n'
            elif argv[0]=='/usr/bin/scontrol' and 'job' in argv:
                output=' '.join(k+'='+v for k,v in jobs[argv[-1]].items())+'\n'
            else: output='QOS=bilv-dtai-gh(1527) GrpTRESMins=cpu=N(0),billing=89940000(20667433),gres/gpu=N(20667)\n'
            stderr=''; code=0
            if mutation: output,stderr,code=mutation(argv,output,commands)
            return real_popen([sys.executable,'-c',
                'import sys;sys.stdout.write(sys.argv[1]);sys.stderr.write(sys.argv[2]);sys.exit(int(sys.argv[3]))',
                output,stderr,str(code)],**kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(a.os,'getuid',return_value=97434), \
             patch.object(a.time,'time',return_value=NOW), patch.object(a.subprocess,'Popen',side_effect=launch):
            root=Path(tmp)
            try: result=a.live(root,auth,current)
            except ValueError:
                self.assertFalse(json.loads((root/'admission-decision.json').read_text())['admitted'])
                raise
            self.assertEqual(pin,d.binding_digest(auth))
            self.assertEqual(result['operational_reserve_gpu_hours'],84)
            self.assertTrue(any(cmd[0]=='/usr/bin/sacct' and a.START in cmd for cmd in commands))
            self.assertTrue(result['admitted'])
        return commands

    def test_24_hour_queued_authority_with_real_runner_fixture_observations(self):
        self.run_live()

    def test_missing_account_failed_query_and_admission_race_stop_without_retry(self):
        def missing(argv,out,commands):
            return ('delta-gh|noalloc|wcyim|noalloc\n' if argv[0]=='/usr/bin/sacctmgr' else out),'',0
        def failed(argv,out,commands):
            return out,('failure' if argv[0]=='/usr/bin/sacctmgr' else ''),(1 if argv[0]=='/usr/bin/sacctmgr' else 0)
        def raced(argv,out,commands):
            n=sum(c[0]=='/usr/bin/squeue' for c in commands)
            return (out+'9999|unknown|wcyim|bilv-dtai-gh|PENDING|\n' if argv[0]=='/usr/bin/squeue' and n==2 else out),'',0
        for mutation in [missing,failed,raced]:
            with self.subTest(mutation=mutation.__name__),self.assertRaises(ValueError): self.run_live(mutation)

if __name__ == '__main__': unittest.main()
