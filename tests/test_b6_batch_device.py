"""Batch validation must preserve value checks after Accelerate moves tensors to CUDA."""
import numpy as np
import pytest

import b6_fixtures as FX
from transgenic.evidence import b6_track_spec as spec


@pytest.fixture(params=["numpy", "cpu", "cuda"])
def backend(request):
    if request.param != "numpy":
        torch = pytest.importorskip("torch")
        if request.param == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
    return request.param


def batch_on(backend, training):
    batch = FX.stub_batch(training=training)
    if backend != "numpy":
        import torch
        batch = {k: torch.as_tensor(v, device=backend) if isinstance(v, np.ndarray) else v
                 for k, v in batch.items()}
    return batch


@pytest.mark.parametrize("training", [True, False])
def test_valid_batch_on_device(backend, training):
    batch = batch_on(backend, training)
    assert spec.validate_batch(batch, training=training) == (2, 60)
    if backend != "numpy":
        from transgenic.training.b6_runtime import validate_b6_batch
        validate_b6_batch(batch, with_labels=training)
        assert batch["chain_mask"].device.type == backend


@pytest.mark.parametrize("key,index,value,message", [
    ("chain_mask", (0, 0), True, "column 0"),
    ("chain_idx", (1, 1, 0), 1, "masked chain_idx"),
    ("chain_idx", (0, 1, 0), -1, "out of"),
    ("chain_idx", (0, 1, 0), 10, "out of"),
    ("chain_feat", (1, 1, 0), 1, "masked chain_feat"),
])
def test_invalid_chain_values_still_rejected_on_device(backend, key, index, value, message):
    batch = batch_on(backend, True)
    batch[key][index] = value
    with pytest.raises(ValueError, match=message):
        spec.validate_batch(batch, training=True)
