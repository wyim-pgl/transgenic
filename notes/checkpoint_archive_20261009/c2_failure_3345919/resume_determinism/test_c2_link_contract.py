"""Real allocated postflight receipts consumed by the real host controller.

Only Slurm/quota are fake. Scientific proof and promotion are setup boundaries,
not claims made by these tiny fixture tests. Neither receipt validator is mocked.
"""
import copy
import json
from pathlib import Path
import shutil
import unittest

import torch
from safetensors.torch import save_file
from transgenic.training.b6_compile_history import CompileHistory
from transgenic.training import b6_runtime as rt
import production_link as producer
import continue_c2_history as host
import test_production_link as producer_fixtures
import test_continue_c2_history as host_fixtures


class LinkContractTests(unittest.TestCase):
    def setUp(self):
        # Reuse real tiny checkpoint/model/optimizer/four-rank RNG fixture setup.
        self.fixture = producer_fixtures.LinkTests(methodName='test_full_preempt_receipt_and_exclusive_publication')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.run, self.ck = self.fixture.root, self.fixture.run, self.fixture.ck
        template = host_fixtures.ControllerTests(methodName='test_init_is_exclusive_and_canonical_claim_blocks_second_operation')
        template.setUp()
        self.addCleanup(template.doCleanups)
        config = template.config()
        for name in ('proof', 'validation', 'production', 'runs', 'inputs'):
            shutil.copytree(template.root / name, self.root / name)
        request = self.fixture.request
        config.update(work=str(self.root), canonical='run', backup='runs/c2',
                      source_commit=request['source_commit'], source_snapshot_git=request['source_snapshot_git'],
                      source_dir=request['source_dir'], bindings=request['bindings'],
                      bindings_sha256=request['bindings_sha256'], config_relative=request['config_relative'],
                      config_sha256=request['config_sha256'], expected_runtime_fingerprint=self.fixture.fp)
        for pin in config['original_pins']:
            pin['relative_path'] = pin['relative_path'].replace('runs/c2/', 'run/', 1)
        # The preserved source marker is now included in the same genuine manifest.
        source = self.root / request['source_dir']
        (source / 'SOURCE_COMMIT').write_text(config['source_commit'] + '\n')
        with (source / 'validation_sources.sha256').open('a') as stream:
            stream.write(producer.sha(source / 'SOURCE_COMMIT') + '  SOURCE_COMMIT\n')
        config['source_manifest_sha256'] = producer.sha(source / 'validation_sources.sha256')
        config['phases']['production'].update(helper_dir=request['helper_dir'],
            helper_manifest_sha256=request['helper_manifest_sha256'], script='helpers/production_link.sbatch')
        bootstrap = {'slots': list(range(1, 16)) + [3915]}
        proof_root = self.root / config['phases']['proof']['root'] / config['proof_job_id']
        host_fixtures.put(proof_root / 'bootstrap.json', bootstrap)
        config['bootstrap_sha256'] = producer.sha(proof_root / 'bootstrap.json')
        # Use the real history schema/validator and rank descriptors from the tensor fixture.
        history = producer.read_json(self.ck / 'compile_history.json')
        example = history['entries'][0]
        entries = []
        for index, slot in enumerate(bootstrap['slots']):
            entry = copy.deepcopy(example)
            entry.update(ordinal=index, origin='reconstructed_legacy', position=dict(phase='train', epoch=0,
                batch_index=slot - 1, view='arm', training=True, grad_enabled=True, optimizer_step=(slot - 1) // 24))
            entries.append(entry)
        history.update(entries=entries, bootstrap=dict(manifest=bootstrap, sha256=config['bootstrap_sha256']))
        position = dict(epoch=0, step=9840, global_step=410, pending_micro=0)
        history['checkpoint'] = position
        CompileHistory._validate(history)
        producer_fixtures.write(self.ck / 'compile_history.json', history)
        meta = producer.read_json(self.ck / 'meta.json')
        meta.update(position, compile_history_sha256=producer.sha(self.ck / 'compile_history.json'))
        producer_fixtures.write(self.ck / 'meta.json', meta)
        optimizer = torch.load(self.ck / 'accelerate_state/optimizer.bin', weights_only=False)
        optimizer['state'][0]['step'] = torch.tensor(410.)
        torch.save(optimizer, self.ck / 'accelerate_state/optimizer.bin')
        scheduler = torch.load(self.ck / 'accelerate_state/scheduler.bin', weights_only=False)
        scheduler.update(last_epoch=410, _step_count=411)
        torch.save(scheduler, self.ck / 'accelerate_state/scheduler.bin')
        self.fixture.draws(position['step'])
        checkpoint = producer.check_checkpoint(self.ck, self.fixture.identity, self.fixture.fp)
        start = dict(checkpoint['position'], **{key: checkpoint[key] for key in ('relative_dir', 'meta_sha256',
            'identity_sha256', 'history_sha256', 'canonical_checkpoint_sha256')})
        request.update(operation_id=config['operation_id'], token='c2-test-operation-production-0001',
                       source_manifest_sha256=config['source_manifest_sha256'], start_checkpoint=start)
        producer_fixtures.write(self.root / 'request.json', request)
        self.request_sha = producer.sha(self.root / 'request.json')
        self.config = config
        self.state = self.root / 'operation'
        host.Controller.initialize(config, self.state)
        self.scheduler = host_fixtures.FakeSlurm()
        self.controller = host.Controller(self.state, self.scheduler)
        # Establish the already-promoted boundary; proof/promotion tests live elsewhere.
        self.controller.save('candidate_promoted')
        pin = dict(relative_path='request.json', sha256=self.request_sha)
        self.controller.prepare_submission('production', pin)
        self.controller.execute_submission()
        self.entry = self.controller.j['links'][-1]
        self.job = self.entry['job_id']
        self.output = producer.execution_dir(self.root, request, self.job)
        self.output.mkdir()
        producer.preflight(self.root, 'request.json', self.request_sha, self.job)

    def emit(self, final=False):
        step = 4099 if final else 420
        position = dict(epoch=0, step=step * 24, global_step=step, pending_micro=0)
        directory = self.run / ('step_04099' if final else 'latest_state')
        directory.mkdir(exist_ok=True)
        history = producer.read_json(self.ck / 'compile_history.json')
        history['checkpoint'] = position
        producer_fixtures.write(directory / 'compile_history.json', history)
        producer_fixtures.write(directory / 'identity.json', self.fixture.identity)
        meta = dict(position, arm='C2', seed=123,
                    compile_history_sha256=producer.sha(directory / 'compile_history.json'))
        result = dict(global_step=step, history=[])
        if final:
            save_file({'weight': torch.ones(2)}, str(directory / 'model.safetensors'))
            meta.update(stop_after_updates=4099, checkpoint_kind='fixed_step', selection=rt.FIXED_STEP_SELECTION,
                        pilot_final=True, model_sha256=producer.sha(directory / 'model.safetensors'))
            (directory / rt.PILOT_FINAL_MARKER).write_text(
                'WP5 pilot-final checkpoint at exactly 4099 optimizer updates (A46 rule 11); never a best-validation checkpoint\n')
            (directory / rt.PRUNE_EXEMPT_MARKER).write_text('A28/plan 9: step checkpoint, never pruned\n')
            (self.run / 'TRAINING_DONE').write_text('2026-10-12T00:00:00\n')
            result.update(done=True, reason='stop_after_updates', pilot_final='/work/run/step_04099')
        else:
            result['preempted'] = True
        producer_fixtures.write(directory / 'meta.json', meta)
        self.fixture.draws(position['step'])
        for rank in range(4):
            producer_fixtures.write(self.output / ('rank%d.json' % rank), dict(rank=rank, job_id=self.job,
                request_sha256=self.request_sha, input_verification=dict(job_id=self.job,
                bindings_sha256=self.fixture.request['bindings_sha256'], record=self.fixture.binding),
                runtime_fingerprint=self.fixture.fp, identity=self.fixture.identity,
                counts=dict(train_rows=400000, rank_batches=100000, per_rank_accumulation=24), result=result, epoch_end=[]))
        # This is the actual producer. No manually assembled link receipt is used.
        emitted = producer.postflight(self.root, 'request.json', self.request_sha, self.job, 0)
        receipt_path = self.output / 'link_receipt.json'
        self.assertEqual(host.read_json(receipt_path), emitted)
        intent = self.entry['intent']
        self.scheduler.jobs[self.job] = dict(job_id=self.job, owner=self.config['owner'], uid=self.config['uid'],
            active=False, token=intent['token'], argv=intent['argv'], rows={
                self.job: ['COMPLETED', '0:0'], self.job + '.batch': ['COMPLETED', '0:0']})
        return emitted

    def test_real_preemption_receipt_reaches_durable_link_accepted(self):
        emitted = self.emit()
        self.assertFalse(self.controller.validate_link_receipt(self.entry, emitted))
        self.controller.accept_link()
        durable = host.Controller(self.state, self.scheduler).status()
        self.assertEqual(durable['phase'], 'link_accepted')
        self.assertEqual(durable['links'][-1]['receipt']['sha256'], producer.sha(self.output / 'link_receipt.json'))
        self.assertEqual(emitted['checkpoint']['position']['global_step'], 420)

    def test_real_final4099_receipt_reaches_final_verified_without_partial_eval(self):
        emitted = self.emit(final=True)
        self.assertTrue(self.controller.validate_link_receipt(self.entry, emitted))
        self.controller.accept_link()
        self.assertEqual(host.Controller(self.state, self.scheduler).status()['phase'], 'final_verified')
        self.assertEqual(emitted['final']['expected_completed_epochs'], 0)
        self.assertEqual(emitted['epoch_events'], [])

    def test_post_emission_corruption_cannot_advance_controller(self):
        emitted = self.emit()
        checkpoint = self.run / emitted['checkpoint']['relative_dir']
        meta = producer.read_json(checkpoint / 'meta.json')
        meta['global_step'] = 421
        producer_fixtures.write(checkpoint / 'meta.json', meta)
        before = (self.state / 'journal.json').read_bytes()
        with self.assertRaises(host.Blocked):
            self.controller.validate_link_receipt(self.entry, emitted)
        with self.assertRaises(host.Blocked):
            self.controller.accept_link()
        self.assertEqual((self.state / 'journal.json').read_bytes(), before)
        self.assertEqual(self.controller.status()['phase'], 'waiting_production')


if __name__ == '__main__':
    unittest.main()
