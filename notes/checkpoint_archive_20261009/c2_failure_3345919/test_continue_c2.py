"""Offline operational tests; scheduler and container executables are isolated fixtures."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / 'transgenic'
        self.repo = self.work / 'repo_b6_d7b1a37'
        self.repo.mkdir(parents=True)
        (self.repo / 'SOURCE_COMMIT').write_text('d7b1a37\n')
        (self.repo / 'validation_sources.sha256').write_text(
            hashlib.sha256(b'd7b1a37\n').hexdigest() + '  SOURCE_COMMIT\n')
        subprocess.run(['git','init','-q',str(self.repo)],check=True)
        subprocess.run(['git','-C',str(self.repo),'add','.'],check=True)
        subprocess.run(['git','-C',str(self.repo),'-c','user.name=Fixture','-c','user.email=fixture@local',
                        'commit','-qm','fixture'],check=True)
        self.original = self.work / 'runs/b6_wp5_pilot_C2'
        self.clone = self.work / 'runs/b6_wp5_pilot_C2_recovery_20261011'
        for run, step, weights in ((self.original, 200, b'original'), (self.clone, 400, b'resumed')):
            (run / 'latest_state/accelerate_state').mkdir(parents=True)
            (run / 'latest_state/accelerate_state/model.safetensors').write_bytes(weights)
            meta = dict(global_step=step, epoch=0, step=step*24, pending_micro=0, arm='C2', stop_after_updates=4099)
            (run / 'latest_state/meta.json').write_text(json.dumps(meta))
            (run / 'latest_state/identity.json').write_text('{"fixture":true}\n')
            for name in ['optimizer.bin','scheduler.bin'] + ['random_states_%d.pkl'%r for r in range(4)]:
                (run / 'latest_state/accelerate_state'/name).write_bytes(b'fixture')
        (self.original / 'TRAINING_FAILED').touch()
        self.meta = json.loads((self.clone / 'latest_state/meta.json').read_text())
        self.receipt = dict(status='passed', job_id=42, source_commit='d7b1a37', started_from_update=200,
                            validation_job=3361110, final_checkpoint=self.meta)
        (self.clone / 'RECOVERY_VALIDATION_PASSED.json').write_text(json.dumps(self.receipt))
        self.scheduler = self.root / 'scheduler.json'
        self.scheduler.write_text(json.dumps(dict(smoke_state='COMPLETED', jobs=[], held=True, fail_second=False)))
        bindir = self.root / 'bin'
        bindir.mkdir()
        fake = bindir / 'fake'
        fake.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
p=Path(os.environ['FIXTURE_SCHEDULER']);s=json.loads(p.read_text());name=Path(sys.argv[0]).name
if name=='sacct': print('42|'+s['smoke_state']+'|'+('0:0' if s['smoke_state']=='COMPLETED' else '1:0'))
elif name=='scontrol':
 print('JobId=3345920 UserId=wcyim('+str(os.getuid())+') JobState=PENDING Reason=JobHeldUser')
elif name=='squeue':
 if s['held']:print('3345920')
elif name=='scancel':s['held']=False
elif name=='sbatch':
 if s['fail_second'] and s['jobs']:sys.exit(1)
 j=900001+len(s['jobs']);s['jobs'].append({'id':j,'args':sys.argv[1:],'env':{k:os.environ.get(k) for k in ['SCRATCH','SIF','DETERMINISTIC']}});print(j)
elif name=='apptainer': print('IDENTITY_OK')
else:sys.exit(99)
p.write_text(json.dumps(s))
''')
        fake.chmod(0o755)
        for name in ('sacct', 'scontrol', 'squeue', 'scancel', 'sbatch', 'apptainer'):
            (bindir / name).symlink_to(fake.name)
        self.env = dict(os.environ, PATH=str(bindir)+os.pathsep+os.environ['PATH'],
                        FIXTURE_SCHEDULER=str(self.scheduler),SCRATCH='/wrong',SIF='/wrong.sif',DETERMINISTIC='0')

    def run_controller(self):
        return subprocess.run([sys.executable, str(HERE/'continue_c2.py'), '--work', str(self.work),
                               '--job', '42'], env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)

    def state(self):
        return json.loads((self.work/'ops/c2_recovery_20261011/continuation_state.json').read_text())

    def test_verified_recovery_preserves_original_and_submits_a_single_two_link_chain(self):
        result = self.run_controller()
        self.assertEqual(result.returncode, 0, result.stderr+result.stdout)
        self.assertEqual((self.original/'latest_state/accelerate_state/model.safetensors').read_bytes(), b'resumed')
        archived = self.work/'runs/b6_wp5_pilot_C2.failed_3345919_before_recovery_20261011'
        self.assertEqual((archived/'latest_state/accelerate_state/model.safetensors').read_bytes(), b'original')
        self.assertFalse(self.clone.exists())
        state = self.state()
        self.assertEqual(state['phase'], 'continuation_submitted')
        self.assertEqual(state['jobs'], [900001, 900002])
        scheduler = json.loads(self.scheduler.read_text())
        self.assertFalse(scheduler['held'])
        self.assertIn('--dependency=afterok:900001', scheduler['jobs'][1]['args'])
        self.assertIn('--nice=0', scheduler['jobs'][0]['args'])
        self.assertEqual(scheduler['jobs'][0]['env'],dict(SCRATCH=str(self.work.parent),
                         SIF=str(self.work.parent/'containers/transgenic-ngc-26.08.sif'),DETERMINISTIC='1'))
        self.assertNotEqual(self.run_controller().returncode, 0)
        self.assertEqual(len(json.loads(self.scheduler.read_text())['jobs']), 2)

    def test_failed_smoke_never_moves_data_cancels_or_submits(self):
        data=json.loads(self.scheduler.read_text());data['smoke_state']='FAILED';self.scheduler.write_text(json.dumps(data))
        result = self.run_controller()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.clone.exists())
        self.assertEqual((self.original/'latest_state/accelerate_state/model.safetensors').read_bytes(), b'original')
        scheduler=json.loads(self.scheduler.read_text())
        self.assertTrue(scheduler['held']);self.assertEqual(scheduler['jobs'], [])
        self.assertEqual(self.state()['phase'], 'error')

    def test_partial_submission_is_recorded_and_never_blindly_retried(self):
        data=json.loads(self.scheduler.read_text());data['fail_second']=True;self.scheduler.write_text(json.dumps(data))
        result = self.run_controller()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.state()['jobs'], [900001])
        self.assertTrue((self.original/'RECOVERY_PROMOTION.json').exists())
        self.assertNotEqual(self.run_controller().returncode, 0)
        self.assertEqual(len(json.loads(self.scheduler.read_text())['jobs']), 1)

    def test_completed_job_without_validation_receipt_cannot_promote(self):
        (self.clone/'RECOVERY_VALIDATION_PASSED.json').unlink()
        self.assertNotEqual(self.run_controller().returncode, 0)
        self.assertEqual(self.state()['phase'], 'error')
        self.assertTrue(self.clone.exists())
        self.assertEqual((self.original/'latest_state/accelerate_state/model.safetensors').read_bytes(), b'original')
        scheduler=json.loads(self.scheduler.read_text())
        self.assertTrue(scheduler['held']);self.assertEqual(scheduler['jobs'], [])

    def test_changed_checkpoint_after_validation_cannot_promote(self):
        meta=dict(self.meta,global_step=401,step=9624)
        (self.clone/'latest_state/meta.json').write_text(json.dumps(meta))
        self.assertNotEqual(self.run_controller().returncode, 0)
        self.assertEqual(self.state()['phase'], 'error')
        scheduler=json.loads(self.scheduler.read_text())
        self.assertTrue(scheduler['held']);self.assertEqual(scheduler['jobs'], [])

    def test_second_rename_failure_rolls_back_and_submits_nothing(self):
        spec=importlib.util.spec_from_file_location('continuation_tested',str(HERE/'continue_c2.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        real_rename=os.rename
        count=[0]
        def failing_rename(source,destination):
            count[0]+=1
            if count[0]==2:raise OSError('injected second rename failure')
            return real_rename(source,destination)
        with patch.dict(os.environ,self.env), patch.object(module.os,'rename',side_effect=failing_rename):
            result=module.main(['--work',str(self.work),'--job','42'])
        self.assertEqual(result,1)
        self.assertEqual((self.original/'latest_state/accelerate_state/model.safetensors').read_bytes(),b'original')
        self.assertEqual((self.clone/'latest_state/accelerate_state/model.safetensors').read_bytes(),b'resumed')
        self.assertEqual(json.loads(self.scheduler.read_text())['jobs'],[])
        self.assertEqual(self.state()['phase'],'error')


if __name__ == '__main__':
    unittest.main()
