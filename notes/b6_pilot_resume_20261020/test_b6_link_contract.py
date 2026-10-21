"""410 -> epoch/pending-gradient preemption ->4099 through real producer/host schemas."""
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import test_continue_b6_history as fixtures
import continue_b6_history as host
import b6_production_link as producer
from test_continue_b6_history import put

class ChainTests(unittest.TestCase):
    setUpClass=classmethod(fixtures.ControllerTests.setUpClass.__func__)
    tearDownClass=classmethod(fixtures.ControllerTests.tearDownClass.__func__)
    setUp=fixtures.ControllerTests.setUp
    advance=fixtures.ControllerTests.advance
    batches=10001
    def draws(self,total):
        for rank in range(4):
            path=self.proof/'resumed/draw_log'/('rank%02d.tsv'%rank);path.parent.mkdir(exist_ok=True)
            path.write_text(''.join('%d\t%d\t%s\n'%(i,i//self.batches,'a'*64) for i in range(total)))
    def start(self):
        self.advance('waiting_production');self.entry=self.controller.j['links'][-1]
        self.pin=self.entry['intent']['request'];self.request=host.read_json(self.work/self.pin['relative_path']);self.job=self.entry['job_id']
        self.output=self.work/self.request['receipt_dir']/('execution_'+self.job);self.output.mkdir()
        self.draws(self.request['start_checkpoint']['epoch']*self.batches+self.request['start_checkpoint']['step'])
        producer.preflight(self.work,self.pin['relative_path'],self.pin['sha256'],self.job)
        intent=self.entry['intent'];created=time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime(intent['created']))
        self.scheduler.jobs[self.job]=dict(job_id=self.job,owner='wcyim',uid=97434,created=created,token=intent['token'],argv=intent['argv'],work_dir=intent['work_dir'],command=intent['argv'][-1],
            active=False,rows={self.job:['COMPLETED','0:0'],self.job+'.batch':['COMPLETED','0:0'],self.job+'.extern':['OUT_OF_MEMORY','0:125']})
    def checkpoint(self,name,epoch,step,global_step,pending,final=False):
        import torch
        from safetensors.torch import load_file,save_file
        run=self.proof/'resumed';path=run/name
        if path.exists():shutil.rmtree(path)
        source=self.proof/'continuous/latest_state'
        shutil.copytree(source,path)
        position=dict(epoch=epoch,step=step,global_step=global_step,pending_micro=pending)
        history=host.read_json(path/'compile_history.json');history['checkpoint']=position;put(path/'compile_history.json',history)
        meta=host.read_json(path/'meta.json');meta.update(position);meta['compile_history_sha256']=host.sha(path/'compile_history.json')
        if final:
            weights=load_file(str(path/'accelerate_state/model.safetensors'))
            save_file(weights,str(path/'model.safetensors'));shutil.rmtree(path/'accelerate_state')
            meta.update(pilot_final=True,stop_after_updates=4099,checkpoint_kind='fixed_step',selection='fixed optimizer-update count; never a best-validation checkpoint',model_sha256=producer.sha(path/'model.safetensors'))
            for marker,content in host.FINAL_MARKERS.items():(path/marker).write_text(content)
        elif pending:
            weights=load_file(str(path/'accelerate_state/model.safetensors'))
            for rank in range(4):torch.save({'weight':torch.ones_like(weights['weight'])},path/('pending_grads_rank%d.pt'%rank))
        put(path/'meta.json',meta)
        return path
    def emit(self,final=False):
        first_epoch=self.request['start_checkpoint']['epoch']+1;end_epoch=9 if final else 1
        rows=[];observations=[]
        for epoch in range(first_epoch,end_epoch+1):
            total=epoch*self.batches;global_step,pending=divmod(total,24)
            cp=self.checkpoint('epoch_%02d'%epoch,epoch,0,global_step,pending)
            row=dict(epoch=epoch,global_step=global_step,data_order_sha256='a'*64,train=2.,de_novo=3.,evidence=4.)
            observation=dict(epoch_end=epoch,arm='B6',data_order_first1000_sha256='a'*64,train=2.,de_novo=3.,evidence=4.)
            put(cp/'eval.json',dict(epoch=epoch,global_step=global_step,train_loss=2.,eval_loss=3.,eval_evidence_loss=4.))
            rows.append(row);observations.append(observation)
        if final:
            self.checkpoint('latest_state',9,4000*24-9*self.batches,4000,0)
            self.checkpoint('step_04099',9,4099*24-9*self.batches,4099,0,final=True)
            (self.proof/'resumed/TRAINING_DONE').write_text('done\n')
            result=dict(done=True,global_step=4099,history=rows,reason='stop_after_updates',pilot_final='/work/'+self.controller.active_run()+'/step_04099');total=4099*24
        else:result=dict(preempted=True,global_step=416,history=rows);total=self.batches
        self.draws(total)
        identity=host.read_json(self.proof/'resumed/run_config_b6.json')
        for rank in range(4):
            put(self.output/('rank%d.json'%rank),dict(rank=rank,job_id=self.job,request_sha256=self.pin['sha256'],
                input_verification=dict(job_id=self.job,bindings_sha256=self.request['bindings_sha256'],mmap_dir=self.request['mmap_dir'],record=self.binding),
                runtime_fingerprint=self.request['expected_runtime_fingerprint'],identity=identity,
                counts=dict(train_rows=self.batches*4,rank_batches=self.batches,per_rank_accumulation=24),result=result,epoch_end=observations))
        return producer.postflight(self.work,self.pin['relative_path'],self.pin['sha256'],self.job,0)
    def test_complete_real_chain_preserves_proof_and_accepts_evolved_checkpoint(self):
        self.advance('candidate_activated')
        fixed={str(p.relative_to(self.proof)):host.source_member_sha(p) for phase in ('continuous','at400') for p in (self.proof/phase).rglob('*') if p.is_file()}
        self.start();receipt=self.emit();self.controller.tick()
        self.assertEqual(self.controller.j['phase'],'link_accepted');self.assertEqual(receipt['checkpoint']['position']['pending_micro'],17)
        previous=host.sha(self.output/'link_receipt.json')
        self.start();self.assertEqual(self.request['predecessor']['sha256'],previous)
        self.emit(final=True);self.controller.tick();self.assertEqual(self.controller.j['phase'],'final_verified')
        self.controller.tick();self.assertEqual(self.controller.j['phase'],'complete')
        self.assertEqual(host.read_json(self.proof/'resumed/latest_state/meta.json')['global_step'],4000)
        self.assertEqual(len(self.scheduler.submissions),2)
        self.assertTrue(all(host.source_member_sha(self.proof/name)==digest for name,digest in fixed.items()))
        self.assertEqual(json.loads(host.read_json(self.state/'activation.json')['initial']['meta.json']['text'])['global_step'],410)
    def test_lost_submission_response_can_adopt_job_after_checkpoint_progress(self):
        self.start()
        # Crash left only the durable intent although the allocation is running.
        intent=copy.deepcopy(self.entry['intent'])
        self.controller.save('submission_uncertain',intent=intent,links=[])
        self.emit()
        latest=self.proof/'resumed/latest_state';shutil.rmtree(latest)
        shutil.copytree(self.proof/'resumed/epoch_01',latest)
        self.scheduler.matches=[dict(self.scheduler.jobs[self.job],created=intent['created'])]
        self.controller=host.Controller(self.state,self.scheduler);self.controller.tick()
        self.assertEqual(self.controller.j['phase'],'waiting_production')
        self.controller.tick();self.assertEqual(self.controller.j['phase'],'link_accepted')
        self.assertEqual(len(self.scheduler.submissions),1)
    def test_reactivated_accepted_job_prevents_next_submission(self):
        self.start();self.emit();self.controller.tick()
        self.assertEqual(self.controller.j['phase'],'link_accepted')
        self.scheduler.jobs[self.job].update(active=True,state='RUNNING')
        with self.assertRaises(host.Blocked):self.controller.tick()
        self.assertEqual(len(self.scheduler.submissions),1)
    def test_final_verified_rechecks_markers_before_complete(self):
        self.start();self.emit(final=True);self.controller.tick()
        self.assertEqual(self.controller.j['phase'],'final_verified')
        (self.proof/'resumed/step_04099/PILOT_FINAL').unlink()
        with self.assertRaises(host.Blocked):self.controller.tick()
        self.assertEqual(self.controller.j['phase'],'final_verified')
    def test_missing_rank_after_emitted_receipt_refuses_host_acceptance(self):
        self.start();self.emit();(self.output/'rank3.json').unlink()
        with self.assertRaises(host.Blocked):self.controller.tick()
    def test_rewritten_history_prefix_and_nonzero_exit_refuse(self):
        self.start();self.emit();(self.output/'link_receipt.json').unlink()
        cp=self.proof/'resumed/epoch_01';history=host.read_json(cp/'compile_history.json');history['entries'][0]['ranks'][0]['meta']={'changed':'row'};put(cp/'compile_history.json',history)
        meta=host.read_json(cp/'meta.json');meta['compile_history_sha256']=host.sha(cp/'compile_history.json');put(cp/'meta.json',meta)
        with self.assertRaisesRegex(ValueError,'history prefix'):producer.postflight(self.work,self.pin['relative_path'],self.pin['sha256'],self.job,0)
        with self.assertRaisesRegex(ValueError,'nonzero'):producer.postflight(self.work,self.pin['relative_path'],self.pin['sha256'],self.job,143)

if __name__=='__main__':unittest.main()
