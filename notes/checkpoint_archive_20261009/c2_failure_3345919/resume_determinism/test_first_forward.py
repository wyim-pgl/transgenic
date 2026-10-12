import json
from pathlib import Path
import tempfile
import unittest

import torch
import first_forward as f


class StateRestorationTests(unittest.TestCase):
    def test_restores_parameters_and_complete_buffer_registry_in_place(self):
        m = torch.nn.Linear(3, 2)
        m.register_buffer('ephemeral', torch.tensor([2.]), persistent=False)
        m.register_buffer('empty', None)
        original_parameter, original_buffer = m.weight, m.ephemeral
        before = f.module_digest(m)
        state = f.snapshot_module(m)
        with torch.no_grad():
            m.weight.add_(7)
            m.ephemeral = torch.tensor([99.])
            m.empty = torch.tensor([1.])
            m.register_buffer('new_buffer', torch.tensor([5.]), persistent=False)
            m._non_persistent_buffers_set.clear()
        self.assertNotEqual(f.module_digest(m), before)
        f.restore_module(m, state)
        self.assertEqual(f.module_digest(m), before)
        self.assertIs(m.weight, original_parameter)
        self.assertIs(m.ephemeral, original_buffer)
        self.assertIsNone(m.empty)
        self.assertNotIn('new_buffer', m._buffers)
        self.assertIn('ephemeral', m._non_persistent_buffers_set)

    def test_replaced_parameter_refused(self):
        m = torch.nn.Linear(3, 2)
        state = f.snapshot_module(m)
        m.weight = torch.nn.Parameter(m.weight.detach().clone())
        with self.assertRaises(AssertionError):
            f.restore_module(m, state)


if __name__ == '__main__':
    unittest.main()
