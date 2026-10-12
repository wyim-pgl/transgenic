"""Offline tests of diagnostic refusal and canonical comparisons; no Slurm side effects."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest

import torch
from safetensors.torch import save_file

spec = importlib.util.spec_from_file_location('probe', Path(__file__).with_name('probe.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbeTests(unittest.TestCase):
    def test_clone_preserves_original_and_only_committed_draws(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'source', root / 'copy'
            (source / 'latest_state/accelerate_state').mkdir(parents=True)
            (source / 'latest_state/meta.json').write_text(json.dumps(dict(pending_micro=0, global_step=1, step=24)))
            (source / 'latest_state/accelerate_state/payload').write_bytes(b'original')
            for name in ('run_config_b6.json', 'runtime_identity.txt', 'data_order.jsonl'):
                (source / name).write_text('identity')
            (source / 'draw_log').mkdir()
            for rank in range(4):
                (source / 'draw_log' / ('rank%02d.tsv' % rank)).write_bytes(b'row\n' * 29)
            before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()}
            probe.clone(source, target)
            after = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual((target / 'draw_log/rank03.tsv').read_bytes(), b'row\n' * 24)
            with self.assertRaises(AssertionError):
                probe.clone(source, target)

    def test_tensor_digest_preserves_bits_dtype_and_is_order_independent(self):
        x = torch.tensor([1, 2], dtype=torch.bfloat16)
        self.assertEqual(probe.digest({'a': x, 'b': 1}), probe.digest({'b': 1, 'a': x.clone()}))
        self.assertNotEqual(probe.digest(x), probe.digest(x.float()))
        self.assertNotEqual(probe.digest(x), probe.digest(x + 1))
        probe.digest(torch.tensor(1.0))

    def test_complete_comparison_detects_resume_weight_and_input_differences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('A', 'B210', 'B', 'C'):
                state = root / name / 'latest_state/accelerate_state'
                state.mkdir(parents=True)
                step = 210 if name in ('A', 'B210') else 220
                (state.parent / 'meta.json').write_text(json.dumps(dict(global_step=step, step=step * 24, pending_micro=0)))
                save_file({'weight': torch.tensor([1., 2.])}, state / 'model.safetensors')
                for n in ['optimizer.bin', 'scheduler.bin'] + ['random_states_%d.pkl' % i for i in range(4)]:
                    torch.save({'payload': torch.tensor([1.])}, state / n)
                lo, hi = (5041, 5281) if name == 'C' else (4801, 5281)
                for rank in range(4):
                    rows = [json.dumps(dict(event='forward', micro=m, inputs='input', shapes={}, rng_before={}, rng_after={}, loss=1.)) for m in range(lo, hi)]
                    rows += [json.dumps(dict(event='accumulation_boundary', micro=m, gradients='equal')) for m in range(lo, hi) if m % 24 == 0]
                    (root / name / ('trace_rank%d.jsonl' % rank)).write_text('\n'.join(rows) + '\n')
            args = types.SimpleNamespace(root=str(root))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(probe.compare(args), 0)
                (root / 'comparison.json').unlink()
                p = root / 'C/trace_rank0.jsonl'
                p.write_text(p.read_text().replace('"input"', '"changed"', 1))
                self.assertEqual(probe.compare(args), 2)
                (root / 'comparison.json').unlink()
                p.write_text(p.read_text().replace('"changed"', '"input"', 1))
                save_file({'weight': torch.tensor([1., 3.])}, root / 'C/latest_state/accelerate_state/model.safetensors')
                self.assertEqual(probe.compare(args), 2)


if __name__ == '__main__':
    unittest.main()
