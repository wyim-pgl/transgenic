"""Preserve mutable model and RNG state around compile-history replay.

Pass the original, unwrapped model. Replay may change registered tensors and
module modes, but must not add or remove submodules. Tensor value snapshots live
on CPU; optimizer, scheduler, counters and compiler state belong to the caller.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import random

import numpy as np
import torch


@dataclass
class _TensorState:
    original: torch.Tensor
    data_view: torch.Tensor
    value: torch.Tensor
    requires_grad: bool


@dataclass
class _ModuleState:
    original: torch.nn.Module
    parameters: dict
    buffers: dict
    nonpersistent: set
    training: bool


@dataclass
class _ModelState:
    modules: list
    tensors: list
    gradients: list


def capture_rng():
    """Return plain-container RNG state, without initializing CUDA.

    Keys: ``python`` (random state tuple), ``numpy`` (NumPy state tuple),
    ``torch_cpu`` (CPU ByteTensor), ``torch_cuda`` (CPU ByteTensor list, or None
    when CUDA was not initialized). The caller owns persistence and validation.
    """
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].copy(), *numpy_state[2:]),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": ([state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
                       if torch.cuda.is_initialized() else None),
    }


def restore_rng(state):
    """Restore a state from :func:`capture_rng`, without touching compilers."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def capture_model(model):
    """Retain original registrations and deduplicated CPU tensor snapshots."""
    modules = []
    tensors = {}
    gradients = {}

    def save_tensor(tensor):
        if tensor is None or id(tensor) in tensors:
            return
        if (type(tensor) not in (torch.Tensor, torch.nn.Parameter)
                or tensor.layout != torch.strided or tensor.is_quantized
                or tensor.device.type == "meta"
                or torch._debug_has_internal_overlap(tensor) != 0):
            raise ValueError("Unsupported tensor layout for training-state replay")
        tensors[id(tensor)] = _TensorState(
            tensor, tensor.detach(), tensor.detach().to(device="cpu", copy=True),
            tensor.requires_grad,
        )

    for module in model.modules():
        parameters = dict(module._parameters)
        buffers = dict(module._buffers)
        modules.append(_ModuleState(module, parameters, buffers,
                                    set(module._non_persistent_buffers_set), module.training))
        for parameter in parameters.values():
            save_tensor(parameter)
            if parameter is not None and id(parameter) not in gradients:
                gradients[id(parameter)] = (parameter, parameter.grad)
                save_tensor(parameter.grad)
        for buffer in buffers.values():
            save_tensor(buffer)
    return _ModelState(modules, list(tensors.values()), list(gradients.values()))


def restore_model(snapshot):
    """Restore original tensor objects, complete registries, gradients and modes."""
    with torch.no_grad():
        for saved in snapshot.tensors:
            original = saved.original
            data_view = saved.data_view
            if not (original.layout == torch.strided and not original.is_quantized
                    and original.shape == data_view.shape and original.stride() == data_view.stride()
                    and original.dtype == data_view.dtype and original.device == data_view.device
                    and original.storage_offset() == data_view.storage_offset()
                    and original.untyped_storage()._cdata == data_view.untyped_storage()._cdata):
                # This retained view allocates no tensor storage. Reattach its
                # original storage, offset and metadata before copying CPU values.
                original.data = data_view
            original.copy_(saved.value)
            original.requires_grad_(saved.requires_grad)
        for saved in snapshot.modules:
            module = saved.original
            # Deletion followed by a plain attribute assignment can shadow a
            # restored registration through ordinary Python attribute lookup.
            for name in saved.parameters.keys() | saved.buffers.keys():
                module.__dict__.pop(name, None)
            module._parameters.clear()
            module._parameters.update(saved.parameters)
            module._buffers.clear()
            module._buffers.update(saved.buffers)
            module._non_persistent_buffers_set.clear()
            module._non_persistent_buffers_set.update(saved.nonpersistent)
            # Calling train() would recursively overwrite heterogeneous child modes.
            module.training = saved.training
        for parameter, gradient in snapshot.gradients:
            parameter.grad = gradient


@contextmanager
def preserve_training_state(model):
    """Restore model and RNG state after replay, including BaseException exits."""
    snapshot = capture_model(model)
    rng = capture_rng()
    try:
        yield
    finally:
        try:
            restore_model(snapshot)
        finally:
            restore_rng(rng)
