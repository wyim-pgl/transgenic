"""Regression tests for compile replay's mutable training state boundary."""
import random
import unittest

import numpy as np
import torch


class ReplayInterrupted(BaseException):
    """A replay interruption that an Exception-only cleanup would miss."""


class ReplayFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1., 2.], [3., 4.]]))
        self.idle = torch.nn.Parameter(torch.tensor([5.]))
        self.register_parameter("absent", None)
        self.child = torch.nn.Module()
        self.child.register_parameter("tied", self.weight)
        self.register_buffer("running", torch.tensor([6., 7.]))
        self.child.register_buffer("alias", self.running)
        self.register_buffer("missing", None, persistent=False)
        self.register_buffer("temporary", torch.tensor([8.]), persistent=False)
        self.train()
        self.child.eval()
        self.weight.grad = torch.tensor([[9., 10.], [11., 12.]])


def mutate_fixture(model):
    with torch.no_grad():
        model.weight.fill_(-1)
        model.weight.grad.fill_(-2)
        model.idle.fill_(-3)
        model.running.fill_(-4)
        model.temporary.fill_(-5)
    model.weight.grad = torch.zeros_like(model.weight)
    model.idle.grad = torch.ones_like(model.idle)
    model.weight.requires_grad_(False)
    model.weight = torch.nn.Parameter(torch.zeros(3))
    del model.idle
    model.register_parameter("new_parameter", torch.nn.Parameter(torch.ones(1)))
    model.absent = torch.nn.Parameter(torch.ones(1))
    model.child.tied = torch.nn.Parameter(torch.ones(2, 2))
    del model.running
    model.temporary = torch.tensor([-6.])
    model.missing = torch.ones(1)
    model.register_buffer("new_buffer", torch.ones(1), persistent=False)
    model.child.register_buffer("alias", None, persistent=False)
    model.train(False)
    model.child.train(True)
    random.random()
    np.random.rand(7)
    torch.rand(7)


class ReplayStateTests(unittest.TestCase):
    def setUp(self):
        self.python_rng = random.getstate()
        self.numpy_rng = np.random.get_state()
        self.torch_rng = torch.get_rng_state()
        self.addCleanup(random.setstate, self.python_rng)
        self.addCleanup(np.random.set_state, self.numpy_rng)
        self.addCleanup(torch.set_rng_state, self.torch_rng)

    def test_parameter_mutation_is_rolled_back(self):
        from transgenic.training import b6_replay_state

        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(3)
        original = model.weight
        with b6_replay_state.preserve_training_state(model):
            with torch.no_grad():
                model.weight.fill_(9)
        self.assertIs(model.weight, original)
        self.assertTrue(torch.equal(model.weight, torch.tensor([[3., 3.]])))

    def assert_fixture_restored(self, model, weight, idle, grad, running, temporary):
        self.assertEqual(tuple(model._parameters), ("weight", "idle", "absent"))
        self.assertIs(model.weight, weight)
        self.assertIs(model.idle, idle)
        self.assertIsNone(model.absent)
        self.assertIs(model.child.tied, weight)
        self.assertTrue(torch.equal(model.weight, torch.tensor([[1., 2.], [3., 4.]])))
        self.assertTrue(torch.equal(model.idle, torch.tensor([5.])))
        self.assertTrue(model.weight.requires_grad)
        self.assertIs(model.weight.grad, grad)
        self.assertTrue(torch.equal(grad, torch.tensor([[9., 10.], [11., 12.]])))
        self.assertIsNone(model.idle.grad)
        self.assertEqual(tuple(model._buffers), ("running", "missing", "temporary"))
        self.assertEqual(tuple(model.child._buffers), ("alias",))
        self.assertIs(model.running, running)
        self.assertIs(model.child.alias, running)
        self.assertIsNone(model.missing)
        self.assertIs(model.temporary, temporary)
        self.assertTrue(torch.equal(running, torch.tensor([6., 7.])))
        self.assertTrue(torch.equal(temporary, torch.tensor([8.])))
        self.assertEqual(model._non_persistent_buffers_set, {"missing", "temporary"})
        self.assertEqual(model.child._non_persistent_buffers_set, set())
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)

    def exercise_fixture_restore(self, interrupt):
        from transgenic.training import b6_replay_state

        model = ReplayFixture()
        originals = (model.weight, model.idle, model.weight.grad, model.running, model.temporary)
        error = ReplayInterrupted("stop replay")
        try:
            with b6_replay_state.preserve_training_state(model):
                mutate_fixture(model)
                if interrupt:
                    raise error
        except ReplayInterrupted as caught:
            self.assertTrue(interrupt)
            self.assertIs(caught, error)
        else:
            self.assertFalse(interrupt, "the BaseException must propagate")
        self.assert_fixture_restored(model, *originals)
        self.assertEqual(random.getstate(), self.python_rng)
        np.testing.assert_equal(np.random.get_state(), self.numpy_rng)
        self.assertTrue(torch.equal(torch.get_rng_state(), self.torch_rng))

    def test_complete_registries_ties_gradients_and_modes_are_restored(self):
        self.exercise_fixture_restore(interrupt=False)

    def test_base_exception_propagates_after_complete_restore(self):
        self.exercise_fixture_restore(interrupt=True)

    def test_changed_tensor_metadata_preserves_original_objects(self):
        from transgenic.training import b6_replay_state

        model = ReplayFixture()
        original = model.weight
        grad = model.weight.grad
        running = model.running
        with b6_replay_state.preserve_training_state(model):
            model.weight.grad = None
            model.weight.data = torch.zeros(3, dtype=torch.float64)
            grad.data = torch.zeros(4, dtype=torch.float64)
            running.resize_(3).zero_()
        self.assertIs(model.weight, original)
        self.assertIs(model.weight.grad, grad)
        self.assertIs(model.running, running)
        self.assertTrue(torch.equal(model.weight, torch.tensor([[1., 2.], [3., 4.]])))
        self.assertTrue(torch.equal(grad, torch.tensor([[9., 10.], [11., 12.]])))
        self.assertTrue(torch.equal(running, torch.tensor([6., 7.])))
        self.assertEqual(model.weight.dtype, torch.float32)
        self.assertEqual(grad.dtype, torch.float32)

    def test_noncontiguous_tensor_strides_survive_metadata_changes(self):
        from transgenic.training import b6_replay_state

        model = torch.nn.Module()
        model.register_buffer("view", torch.tensor([[1., 2.], [3., 4.]]).t())
        original = model.view
        with b6_replay_state.preserve_training_state(model):
            original.data = torch.zeros(3)
        self.assertIs(model.view, original)
        self.assertEqual(model.view.stride(), (1, 2))
        self.assertTrue(torch.equal(model.view, torch.tensor([[1., 3.], [2., 4.]])))

    def test_sparse_parameter_buffer_or_gradient_fails_before_replay(self):
        from transgenic.training import b6_replay_state

        for registration in ("parameter", "buffer", "gradient"):
            with self.subTest(registration=registration):
                model = torch.nn.Module()
                sparse = torch.tensor([1., 0.]).to_sparse()
                if registration == "parameter":
                    model.register_parameter("weight", torch.nn.Parameter(sparse))
                elif registration == "buffer":
                    model.register_buffer("running", sparse)
                else:
                    model.weight = torch.nn.Parameter(torch.ones(2))
                    model.weight.grad = sparse
                entered = False
                with self.assertRaisesRegex(ValueError, "layout"):
                    with b6_replay_state.preserve_training_state(model):
                        entered = True
                self.assertFalse(entered)

    def test_overlapping_buffer_fails_before_replay(self):
        from transgenic.training import b6_replay_state

        model = torch.nn.Module()
        model.register_buffer("broadcast", torch.tensor([1.]).expand(3))
        entered = False
        with self.assertRaisesRegex(ValueError, "layout"):
            with b6_replay_state.preserve_training_state(model):
                entered = True
        self.assertFalse(entered)

    def test_rng_helpers_restore_independently_saved_draws(self):
        from transgenic.training import b6_replay_state

        random.seed(101)
        np.random.seed(102)
        torch.manual_seed(103)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        expected = (random.random(), np.random.rand(4), torch.rand(4))
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        saved = b6_replay_state.capture_rng()
        random.random()
        np.random.rand(25)
        torch.rand(25)
        b6_replay_state.restore_rng(saved)
        self.assertEqual(random.random(), expected[0])
        np.testing.assert_array_equal(np.random.rand(4), expected[1])
        self.assertTrue(torch.equal(torch.rand(4), expected[2]))

    def test_cpu_context_does_not_initialize_cuda(self):
        from transgenic.training import b6_replay_state

        if torch.cuda.is_initialized():
            self.skipTest("CUDA was already initialized")
        with b6_replay_state.preserve_training_state(torch.nn.Linear(2, 1)):
            torch.rand(2)
        self.assertFalse(torch.cuda.is_initialized())

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_initialized_cuda_rngs_and_gpu_values_are_restored(self):
        from transgenic.training import b6_replay_state

        model = torch.nn.Linear(2, 1, device="cuda")
        cuda_states = torch.cuda.get_rng_state_all()
        self.addCleanup(torch.cuda.set_rng_state_all, cuda_states)
        expected = [torch.rand(4, device=device) for device in range(torch.cuda.device_count())]
        torch.cuda.set_rng_state_all(cuda_states)
        original = model.weight
        expected_weight = original.detach().cpu().clone()
        with b6_replay_state.preserve_training_state(model):
            with torch.no_grad():
                original.zero_()
            for device in range(torch.cuda.device_count()):
                torch.rand(19, device=device)
        self.assertIs(model.weight, original)
        self.assertTrue(torch.equal(model.weight.cpu(), expected_weight))
        for device, values in enumerate(expected):
            self.assertTrue(torch.equal(torch.rand(4, device=device), values))


if __name__ == "__main__":
    unittest.main()
