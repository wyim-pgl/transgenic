"""Real producer receipts across controller v2 in-place activation and two links.

Only scientific observation inputs and Slurm/quota are fixtures. Activation,
request building, producer preflight/postflight and host acceptance are real.
"""
import copy
from pathlib import Path
import shutil
import unittest

import torch
from safetensors.torch import save_file
from transgenic.training.b6_compile_history import CompileHistory, input_signature
from transgenic.training import b6_runtime as rt
import production_link as producer
import test_production_link as producer_fixtures
import test_continue_c2_history as host_fixtures

# The fixture loads the actual controller once; keep its exception identity too.
host = host_fixtures.ctl


HERE = Path(__file__).resolve().parent


def draws(run, total):
    for rank in range(4):
        path = run / 'draw_log' / ('rank%02d.tsv' % rank)
        path.parent.mkdir(exist_ok=True)
        path.write_text(''.join('%d\t0\t%s\n' % (i, 'a' * 64) for i in range(total)))


def snapshot(paths):
    """Record temporary fixture bytes/inodes without following symlinks."""
    result = {}
    for root in paths:
        entries = [root] + list(root.rglob('*')) if root.is_dir() else [root]
        for path in entries:
            if path.is_symlink():
                raise AssertionError('unexpected fixture symlink')
            result[str(path)] = (path.stat().st_ino, producer.sha(path) if path.is_file() else None)
    return result


def scientific_fixture():
    # Local class avoids unittest discovering the inherited controller unit tests.
    class ScientificFixture(host_fixtures.ControllerTests):
        def config(self):
            config = super().config()
            self.tensor_fixture = producer_fixtures.LinkTests(
                methodName='test_full_preempt_receipt_and_exclusive_publication')
            self.tensor_fixture.setUp()
            self.addCleanup(self.tensor_fixture.doCleanups)
            self.binding = self.tensor_fixture.binding
            self.identity = self.tensor_fixture.identity
            source = self.root / config['source_dir']
            members = ['train/train_HyenaTransgenic_B6.py', 'src/transgenic/training/b6_runtime.py',
                       'src/transgenic/training/b6_compile_history.py', 'src/transgenic/training/b6_compile_replay.py']
            with (source / 'validation_sources.sha256').open('a') as manifest:
                for name in members:
                    path = source / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(self.tensor_fixture.source / name, path)
                    manifest.write(producer.sha(path) + '  ' + name + '\n')
            config['source_manifest_sha256'] = producer.sha(source / 'validation_sources.sha256')
            production = config['phases']['production']
            helper = self.root / production['helper_dir']
            with (helper / 'manifest.sha256').open('a') as manifest:
                for name in ('production_link.py', 'production_link.sbatch', 'probe.py'):
                    shutil.copyfile(HERE / name, helper / name)
                    manifest.write(producer.sha(helper / name) + '  ' + name + '\n')
            production['helper_manifest_sha256'] = producer.sha(helper / 'manifest.sha256')
            producer_fixtures.write(self.root / config['bindings'], self.binding)
            config['bindings_sha256'] = producer.sha(self.root / config['bindings'])
            return config

        def science_fixture(self, controller, kind, job='9001'):
            root, observation = super().science_fixture(controller, kind, job)
            report = producer.read_json(root / 'comparison.json')
            prototype = producer.read_json(self.tensor_fixture.ck / 'compile_history.json')['entries'][0]['ranks']
            stop, snap = (410, 400) if kind == 'validation' else (220, 210)
            checkpoints = {}
            for name in ('continuous', 'resumed', 'at%d' % snap):
                run = root / name
                checkpoint = run / 'latest_state'
                meta = producer.read_json(checkpoint / 'meta.json')
                history = producer.read_json(checkpoint / 'compile_history.json')
                history['identity'] = self.identity
                for entry in history['entries']:
                    entry['ranks'] = copy.deepcopy(prototype)
                    if entry['origin'] == 'observed':
                        signatures = input_signature({'input_ids': torch.zeros((1, 129024), dtype=torch.int64),
                                                      'labels': torch.zeros((1, 3924), dtype=torch.int64)})
                        for rank in entry['ranks']:
                            rank['inputs'] = copy.deepcopy(signatures)
                CompileHistory._validate(history)
                shutil.copytree(self.tensor_fixture.ck / 'accelerate_state', checkpoint / 'accelerate_state',
                                dirs_exist_ok=True)
                optimizer = torch.load(checkpoint / 'accelerate_state/optimizer.bin', weights_only=False)
                optimizer['state'][0]['step'] = torch.tensor(float(meta['global_step']))
                torch.save(optimizer, checkpoint / 'accelerate_state/optimizer.bin')
                scheduler = torch.load(checkpoint / 'accelerate_state/scheduler.bin', weights_only=False)
                scheduler.update(last_epoch=meta['global_step'], _step_count=meta['global_step'] + 1)
                torch.save(scheduler, checkpoint / 'accelerate_state/scheduler.bin')
                producer_fixtures.write(checkpoint / 'identity.json', self.identity)
                producer_fixtures.write(checkpoint / 'compile_history.json', history)
                meta.update(arm='C2', seed=123, identity_sha256=rt.canonical_sha256(self.identity),
                            compile_history_sha256=producer.sha(checkpoint / 'compile_history.json'))
                producer_fixtures.write(checkpoint / 'meta.json', meta)
                producer_fixtures.write(run / 'run_config_b6.json', self.identity)
                draws(run, meta['step'])
                checkpoints[name] = producer.check_checkpoint(checkpoint, self.identity, self.c['expected_runtime_fingerprint'])
                if name in ('continuous', 'resumed'):
                    report['checkpoint_history_sha256'][name] = checkpoints[name]['history_sha256']
            if kind == 'validation':
                report['checkpoint_digests'] = {name: checkpoints[name]['canonical_checkpoint_sha256']
                                                for name in ('continuous', 'resumed')}
            host_fixtures.put(root / 'comparison.json', report)
            if kind == 'validation':
                ready = producer.read_json(root / 'validation_receipt.json')
                ready['comparison_sha256'] = producer.sha(root / 'comparison.json')
                ready['candidate'].update({key: checkpoints['resumed'][key]
                    for key in ('meta_sha256', 'history_sha256', 'canonical_checkpoint_sha256')})
                host_fixtures.put(root / 'validation_receipt.json', ready)
            return root, observation

    fixture = ScientificFixture(methodName='test_init_is_exclusive_and_canonical_claim_blocks_second_operation')
    fixture.setUp()
    return fixture


class SequentialSlurm(host_fixtures.FakeSlurm):
    """Preserve accepted science observations, assign fresh IDs to production."""
    def __init__(self, previous):
        super().__init__()
        self.jobs = copy.deepcopy(previous.jobs)
        self.calls = list(previous.calls)
        self.next_job = 9002

    def submit(self, argv, env):
        result = super().submit(argv, env)
        result['stdout'] = str(self.next_job) + '\n'
        self.next_job += 1
        return result


class LinkContractTests(unittest.TestCase):
    def setUp(self):
        self.fixture = scientific_fixture()
        self.addCleanup(self.fixture.doCleanups)
        self.controller, self.validation = self.fixture.accepted_validation()
        self.fixture.retired(self.controller)
        self.root, self.state = self.fixture.root, self.fixture.state
        self.config = self.fixture.c
        self.active = str((self.validation / 'resumed').relative_to(self.root))
        self.run = self.root / self.active
        self.ck = self.run / 'latest_state'
        protected = [self.root / self.config['original_run'], self.root / self.config['diagnostic'],
                     self.root / self.config['phases']['proof']['root'], self.validation / 'continuous',
                     self.validation / 'at400', self.validation / 'comparison.json',
                     self.validation / 'validation_receipt.json']
        self.protected = protected
        self.before = snapshot(protected)
        active_before = snapshot([self.run])
        self.controller.activation_prepare()
        self.controller.activate()
        self.assertEqual(self.controller.status()['phase'], 'candidate_activated')
        self.assertEqual(self.controller.active_run(), self.active)
        self.assertEqual(snapshot([self.run]), active_before)
        self.assertEqual(snapshot(protected), self.before)
        self.activation_pin = copy.deepcopy(self.controller.status()['activation_receipt'])
        self.scheduler = SequentialSlurm(self.fixture.scheduler)
        self.controller.scheduler = self.scheduler

    def start_link(self):
        self.request_pin = self.controller.build_request()
        self.request = host.read_json(self.root / self.request_pin['relative_path'])
        self.assertEqual(self.request['run_dir'], self.active)
        self.assertNotEqual(self.request['run_dir'], self.config['original_run'])
        self.controller.prepare_submission('production', self.request_pin)
        self.controller.execute_submission()
        self.entry = self.controller.j['links'][-1]
        self.job = self.entry['job_id']
        self.output = producer.execution_dir(self.root, self.request, self.job)
        self.output.mkdir()
        producer.preflight(self.root, self.request_pin['relative_path'], self.request_pin['sha256'], self.job)

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
            result.update(done=True, reason='stop_after_updates', pilot_final='/work/' + self.active + '/step_04099')
        else:
            result['preempted'] = True
            optimizer = torch.load(directory / 'accelerate_state/optimizer.bin', weights_only=False)
            optimizer['state'][0]['step'] = torch.tensor(float(step))
            torch.save(optimizer, directory / 'accelerate_state/optimizer.bin')
            scheduler = torch.load(directory / 'accelerate_state/scheduler.bin', weights_only=False)
            scheduler.update(last_epoch=step, _step_count=step + 1)
            torch.save(scheduler, directory / 'accelerate_state/scheduler.bin')
        producer_fixtures.write(directory / 'meta.json', meta)
        draws(self.run, position['step'])
        for rank in range(4):
            producer_fixtures.write(self.output / ('rank%d.json' % rank), dict(rank=rank, job_id=self.job,
                request_sha256=self.request_pin['sha256'], input_verification=dict(job_id=self.job,
                bindings_sha256=self.request['bindings_sha256'], record=self.fixture.binding),
                runtime_fingerprint=self.config['expected_runtime_fingerprint'], identity=self.fixture.identity,
                counts=dict(train_rows=400000, rank_batches=100000, per_rank_accumulation=24), result=result, epoch_end=[]))
        # Never handcraft a production receipt or replace either acceptance validator.
        emitted = producer.postflight(self.root, self.request_pin['relative_path'], self.request_pin['sha256'], self.job, 0)
        self.assertEqual(host.read_json(self.output / 'link_receipt.json'), emitted)
        intent = self.entry['intent']
        self.scheduler.jobs[self.job] = dict(job_id=self.job, owner=self.config['owner'], uid=self.config['uid'],
            active=False, token=intent['token'], argv=intent['argv'], rows={
                self.job: ['COMPLETED', '0:0'], self.job + '.batch': ['COMPLETED', '0:0']})
        return emitted

    def unchanged_evidence(self):
        self.assertEqual(snapshot(self.protected), self.before)
        self.assertEqual(self.controller.status()['activation_receipt'], self.activation_pin)
        self.assertEqual(producer.sha(self.root / self.activation_pin['relative_path']), self.activation_pin['sha256'])
        self.assertTrue((self.root / self.config['original_run'] / 'TRAINING_FAILED').is_file())
        self.assertFalse((self.root / self.config['original_run'] / 'step_04099').exists())

    def test_preemption_second_request_and_final_follow_actual_activated_path(self):
        self.start_link()
        self.assertEqual(self.request['start_checkpoint']['global_step'], 410)
        first_predecessor = self.controller.j['receipts']['validation']
        self.assertEqual(self.request['predecessor']['sha256'], first_predecessor['ready_sha256'])
        emitted = self.emit()
        self.assertFalse(self.controller.validate_link_receipt(self.entry, emitted))
        self.controller.accept_link()
        self.assertEqual(host.Controller(self.state, self.scheduler).status()['phase'], 'link_accepted')
        first_pin = copy.deepcopy(self.controller.status()['links'][-1]['receipt'])
        self.unchanged_evidence()
        self.start_link()
        self.assertEqual(self.request['link_index'], 2)
        self.assertEqual(self.request['start_checkpoint']['global_step'], 420)
        self.assertEqual(self.request['predecessor'], first_pin)
        emitted = self.emit(final=True)
        self.assertTrue(self.controller.validate_link_receipt(self.entry, emitted))
        self.controller.accept_link()
        reopened = host.Controller(self.state, self.scheduler)
        self.assertEqual(reopened.status()['phase'], 'final_verified')
        self.assertEqual(reopened.active_run(), self.active)
        self.assertEqual(emitted['result']['pilot_final'], '/work/' + self.active + '/step_04099')
        self.assertEqual(emitted['final']['expected_completed_epochs'], 0)
        self.unchanged_evidence()

    def test_old_original_pilot_final_path_is_rejected_after_real_final_emission(self):
        self.start_link()
        emitted = self.emit(final=True)
        wrong = copy.deepcopy(emitted)
        wrong['result']['pilot_final'] = '/work/' + self.config['original_run'] + '/step_04099'
        producer_fixtures.write(self.output / 'link_receipt.json', wrong)
        before = (self.state / 'journal.json').read_bytes()
        with self.assertRaises(host.Blocked): self.controller.validate_link_receipt(self.entry, wrong)
        with self.assertRaises(host.Blocked): self.controller.accept_link()
        self.assertEqual((self.state / 'journal.json').read_bytes(), before)
        self.assertEqual(self.controller.status()['phase'], 'waiting_production')
        self.unchanged_evidence()

    def test_post_emission_checkpoint_corruption_cannot_advance_controller(self):
        self.start_link()
        emitted = self.emit()
        checkpoint = self.run / emitted['checkpoint']['relative_dir']
        meta = producer.read_json(checkpoint / 'meta.json')
        meta['global_step'] = 421
        producer_fixtures.write(checkpoint / 'meta.json', meta)
        before = (self.state / 'journal.json').read_bytes()
        with self.assertRaises(host.Blocked): self.controller.validate_link_receipt(self.entry, emitted)
        with self.assertRaises(host.Blocked): self.controller.accept_link()
        self.assertEqual((self.state / 'journal.json').read_bytes(), before)
        self.assertEqual(self.controller.status()['phase'], 'waiting_production')
        self.unchanged_evidence()


if __name__ == '__main__':
    unittest.main()
