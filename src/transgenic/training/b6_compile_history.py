"""Validated, checkpoint-bound union of every rank's real compiler inputs.

Persistence is rank-local. ``observe`` owns its separate collective preflight
and transport phases and must be called by every rank after the same forward
(and backward, when compiler activity can be lazy).
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch

from transgenic.training.mvp_finalize import (
    _collective_preflight, _gather_payload_bytes, _transport_lengths,
)

_POSITION_KEYS = {'phase', 'epoch', 'batch_index', 'view', 'training',
                  'grad_enabled', 'optimizer_step'}
_CHECKPOINT_KEYS = {'epoch', 'step', 'global_step', 'pending_micro'}
_RNG_KEYS = {'python', 'numpy', 'torch_cpu', 'torch_cuda'}
_DTYPES = {str(value) for value in vars(torch).values() if isinstance(value, torch.dtype)}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return type(value) is int and value >= 0


def _keys(value, keys, name):
    _require(type(value) is dict and set(value) == keys, f'invalid {name} fields')


def _json_value(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        _require(math.isfinite(value), 'JSON values must be finite')
    elif type(value) is list:
        for item in value:
            _json_value(item)
    elif type(value) is dict:
        _require(all(type(key) is str for key in value), 'JSON mapping keys must be strings')
        for item in value.values():
            _json_value(item)
    else:
        raise ValueError(f'unsupported JSON value: {type(value).__name__}')


def _canonical(value):
    _json_value(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f'duplicate JSON field: {key}')
        result[key] = value
    return result


def _copy(value):
    return json.loads(_canonical(value))


def _mapping(value, name):
    _require(type(value) is dict, f'{name} must be a JSON mapping')
    return _copy(value)


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _position(value):
    _keys(value, _POSITION_KEYS, 'position')
    _require(value['phase'] in ('train', 'eval'), 'invalid position phase')
    _require(all(_integer(value[key]) for key in ('epoch', 'batch_index', 'optimizer_step')),
             'position counters must be nonnegative integers')
    _require(type(value['training']) is bool and type(value['grad_enabled']) is bool,
             'position training and grad_enabled must be booleans')
    if value['phase'] == 'train':
        _require(value['view'] == 'arm' and value['training'] and value['grad_enabled'],
                 'invalid train position context')
    else:
        _require(value['view'] in ('de_novo', 'evidence') and not value['training'] and not value['grad_enabled'],
                 'invalid eval position context')
    return _copy(value)


def _checkpoint(value):
    _keys(value, _CHECKPOINT_KEYS, 'checkpoint')
    _require(all(_integer(item) for item in value.values()), 'checkpoint counters must be nonnegative integers')
    return _copy(value)


def input_signature(inputs):
    """Bind dense values, original layout and explicitly marked dynamic axes."""
    _require(type(inputs) is dict and all(type(name) is str for name in inputs),
             'inputs must be a mapping with string names')
    result = {}
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            _require(value.layout == torch.strided and not value.is_quantized and value.device.type != 'meta',
                     f'unsupported tensor input: {name}')
            _require(str(value.dtype) in _DTYPES, f'unsupported tensor dtype: {name}')
            dense = value.detach().resolve_conj().resolve_neg().cpu().contiguous().reshape(-1)
            raw = dense.view(torch.uint8).numpy().tobytes()
            result[name] = dict(kind='tensor', sha256=hashlib.sha256(raw).hexdigest(),
                                dtype=str(value.dtype), shape=list(value.shape), stride=list(value.stride()),
                                requires_grad=value.requires_grad,
                                dynamic_dims=sorted(getattr(value, '_dynamo_dynamic_indices', set())))
        elif value is None or type(value) in (str, bool, int, float):
            raw = _canonical(value)
            result[name] = dict(kind='scalar', type='null' if value is None else type(value).__name__,
                                value=value, sha256=hashlib.sha256(raw).hexdigest())
        else:
            raise ValueError(f'unsupported input value: {name} ({type(value).__name__})')
    _validate_signature(result)
    return result


def _validate_signature(signature):
    _mapping(signature, 'input signature')
    for descriptor in signature.values():
        _require(type(descriptor) is dict, 'invalid input descriptor')
        kind = descriptor.get('kind')
        if kind == 'tensor':
            _keys(descriptor, {'kind', 'sha256', 'dtype', 'shape', 'stride', 'requires_grad', 'dynamic_dims'}, 'tensor descriptor')
            _require(descriptor['dtype'] in _DTYPES, 'invalid input dtype')
            shape, stride, dynamic = (descriptor[key] for key in ('shape', 'stride', 'dynamic_dims'))
            _require(type(shape) is list and all(_integer(item) for item in shape), 'invalid input shape')
            _require(type(stride) is list and len(stride) == len(shape) and all(_integer(item) for item in stride), 'invalid input stride')
            _require(type(descriptor['requires_grad']) is bool, 'invalid input requires_grad')
            _require(type(dynamic) is list and all(_integer(item) and item < len(shape) for item in dynamic)
                     and dynamic == sorted(set(dynamic)), 'invalid input dynamic dimensions')
        elif kind == 'scalar':
            _keys(descriptor, {'kind', 'sha256', 'type', 'value'}, 'scalar descriptor')
            value = descriptor['value']
            _require(value is None or type(value) in (str, bool, int, float), 'invalid scalar value')
            _require(descriptor['type'] == ('null' if value is None else type(value).__name__), 'invalid scalar type')
            _require(descriptor['sha256'] == hashlib.sha256(_canonical(value)).hexdigest(), 'scalar digest mismatch')
        else:
            raise ValueError('invalid input descriptor kind')
        _require(_sha(descriptor['sha256']), 'invalid input SHA256')


def _rng_atom(value):
    if type(value) is tuple:
        return {'type': 'tuple', 'items': [_rng_atom(item) for item in value]}
    if isinstance(value, np.ndarray):
        _require(value.dtype == np.dtype('uint32') and value.ndim == 1, 'invalid NumPy RNG array')
        return dict(type='ndarray', dtype=value.dtype.str, shape=list(value.shape),
                    data=base64.b64encode(value.tobytes()).decode('ascii'))
    if isinstance(value, torch.Tensor):
        _require(value.device.type == 'cpu' and value.dtype == torch.uint8 and value.ndim == 1
                 and value.numel() > 0, 'invalid RNG ByteTensor')
        return dict(type='tensor', dtype='torch.uint8', shape=list(value.shape),
                    data=base64.b64encode(value.contiguous().numpy().tobytes()).decode('ascii'))
    _require(value is None or type(value) in (int, float), 'unsupported RNG state value')
    _json_value(value)
    return {'type': 'scalar', 'value': value}


def _unrng_atom(payload):
    _require(type(payload) is dict, 'invalid RNG typed value')
    kind = payload.get('type')
    if kind == 'tuple':
        _keys(payload, {'type', 'items'}, 'RNG tuple')
        _require(type(payload['items']) is list, 'invalid RNG tuple items')
        return tuple(_unrng_atom(item) for item in payload['items'])
    if kind == 'scalar':
        _keys(payload, {'type', 'value'}, 'RNG scalar')
        value = payload['value']
        _require(value is None or type(value) in (int, float), 'invalid RNG scalar')
        _json_value(value)
        return value
    _require(kind in ('tensor', 'ndarray'), 'invalid RNG value type')
    _keys(payload, {'type', 'dtype', 'shape', 'data'}, 'RNG bytes')
    shape = payload['shape']
    _require(type(shape) is list and len(shape) == 1 and _integer(shape[0]) and shape[0] > 0, 'invalid RNG byte shape')
    _require(type(payload['data']) is str, 'invalid RNG base64')
    try:
        raw = base64.b64decode(payload['data'], validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError('invalid RNG base64') from error
    if kind == 'tensor':
        _require(payload['dtype'] == 'torch.uint8' and len(raw) == shape[0], 'invalid RNG tensor bytes')
        return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy())
    _require(payload['dtype'] == np.dtype('uint32').str and len(raw) == shape[0] * 4, 'invalid NumPy RNG bytes')
    return np.frombuffer(raw, dtype=np.uint32).copy()


def _validate_rng(state):
    _keys(state, _RNG_KEYS, 'RNG state')
    python = state['python']
    _require(type(python) is tuple and len(python) == 3 and python[0] == 3
             and type(python[0]) is int and type(python[1]) is tuple and len(python[1]) == 625
             and all(_integer(item) and item < 2**32 for item in python[1][:-1])
             and _integer(python[1][-1]) and python[1][-1] <= 624
             and (python[2] is None or type(python[2]) is float and math.isfinite(python[2])), 'invalid Python RNG state')
    random.Random().setstate(python)
    numpy = state['numpy']
    _require(type(numpy) is tuple and len(numpy) == 5 and numpy[0] == 'MT19937'
             and isinstance(numpy[1], np.ndarray) and numpy[1].dtype == np.dtype('uint32')
             and numpy[1].shape == (624,) and _integer(numpy[2]) and numpy[2] <= 624
             and type(numpy[3]) is int and numpy[3] in (0, 1)
             and type(numpy[4]) is float and math.isfinite(numpy[4]), 'invalid NumPy RNG state')
    np.random.RandomState().set_state(numpy)
    cpu = state['torch_cpu']
    _rng_atom(cpu)
    try:
        torch.Generator(device='cpu').set_state(cpu)
    except (RuntimeError, TypeError) as error:
        raise ValueError('invalid CPU RNG state') from error
    cuda = state['torch_cuda']
    _require(cuda is None or type(cuda) is list and len(cuda) > 0, 'invalid CUDA RNG states')
    if cuda is not None:
        for item in cuda:
            _require(isinstance(item, torch.Tensor), 'invalid CUDA RNG state tensor')
            _rng_atom(item)
            # CUDAGeneratorImpl stores uint64 seed + int64 Philox offset.
            _require(item.numel() == 16, 'invalid CUDA RNG state size')
            offset = int.from_bytes(item.contiguous().numpy().tobytes()[8:], sys.byteorder)
            _require(offset % 4 == 0, 'invalid CUDA RNG Philox offset')


def encode_rng(state):
    """Encode the helper's RNG dictionary without pickle or global RNG mutation."""
    _validate_rng(state)
    numpy = state['numpy']
    # The algorithm is a string; encode it explicitly outside the numeric atoms.
    return dict(python=_rng_atom(state['python']),
                numpy=dict(algorithm=numpy[0], state=_rng_atom(numpy[1]), position=numpy[2],
                           has_gauss=numpy[3], cached_gaussian=numpy[4]),
                torch_cpu=_rng_atom(state['torch_cpu']),
                torch_cuda=None if state['torch_cuda'] is None else [_rng_atom(item) for item in state['torch_cuda']])


def decode_rng(payload):
    """Decode and validate safely on CPU; never initialize CUDA."""
    _keys(payload, _RNG_KEYS, 'encoded RNG')
    numpy = payload['numpy']
    _keys(numpy, {'algorithm', 'state', 'position', 'has_gauss', 'cached_gaussian'}, 'encoded NumPy RNG')
    cuda = payload['torch_cuda']
    _require(cuda is None or type(cuda) is list, 'invalid encoded CUDA RNG')
    state = dict(python=_unrng_atom(payload['python']),
                 numpy=(numpy['algorithm'], _unrng_atom(numpy['state']), numpy['position'], numpy['has_gauss'], numpy['cached_gaussian']),
                 torch_cpu=_unrng_atom(payload['torch_cpu']),
                 torch_cuda=None if cuda is None else [_unrng_atom(item) for item in cuda])
    _validate_rng(state)
    return state


def _validate_entry(entry, ordinal, world, bootstrap):
    _keys(entry, {'ordinal', 'position', 'origin', 'ranks'}, 'entry')
    _require(type(entry['ordinal']) is int and entry['ordinal'] == ordinal, 'invalid entry ordinal order')
    _position(entry['position'])
    _require(entry['origin'] in ('observed', 'reconstructed_legacy'), 'invalid entry origin')
    if entry['origin'] == 'reconstructed_legacy':
        _require(type(bootstrap) is dict and bool(bootstrap), 'reconstructed history requires explicit bootstrap provenance')
    _require(type(entry['ranks']) is list and len(entry['ranks']) == world, 'invalid rank cardinality')
    for rank, descriptor in enumerate(entry['ranks']):
        _keys(descriptor, {'rank', 'inputs', 'meta', 'rng_before', 'delta', 'activity_delta'}, 'rank descriptor')
        _require(type(descriptor['rank']) is int and descriptor['rank'] == rank, 'invalid rank order')
        _require(_integer(descriptor['delta']) and _integer(descriptor['activity_delta']), 'invalid compile delta')
        _validate_signature(descriptor['inputs'])
        _mapping(descriptor['meta'], 'batch meta')
        decode_rng(descriptor['rng_before'])
    if entry['origin'] == 'observed':
        _require(any(item['delta'] or item['activity_delta'] for item in entry['ranks']), 'observed entry has no compiler event')


def check_inputs(entry, inputs, meta, rank):
    """Reject altered data or metadata before replaying this rank's companion."""
    _require(_integer(rank) and type(entry) is dict and type(entry.get('ranks')) is list
             and rank < len(entry['ranks']), 'invalid replay rank')
    descriptor = entry['ranks'][rank]
    _require(descriptor.get('rank') == rank, 'invalid replay rank order')
    _require(_canonical(input_signature(inputs)) == _canonical(descriptor['inputs']), 'replay input signature mismatch')
    _require(_canonical(_mapping(meta, 'batch meta')) == _canonical(descriptor['meta']), 'replay batch metadata mismatch')


class CompileHistory:
    def __init__(self, identity, fingerprint, accelerator):
        self.identity = _mapping(identity, 'identity')
        self.fingerprint = _mapping(fingerprint, 'fingerprint')
        self.accelerator = accelerator
        self.world_size = accelerator.num_processes
        _require(_integer(self.world_size) and self.world_size > 0, 'invalid world size')
        _require(_integer(accelerator.process_index) and accelerator.process_index < self.world_size, 'invalid local rank')
        self.entries = []
        self.bootstrap = None
        self.checkpoint = None

    @property
    def bootstrap(self):
        return _copy(self._bootstrap)

    @bootstrap.setter
    def bootstrap(self, value):
        self._bootstrap = None if value is None else _mapping(value, 'bootstrap provenance')
        _require(self._bootstrap is None or bool(self._bootstrap), 'empty bootstrap provenance')

    def _gather(self, operation):
        local = {}

        def prepare():
            local['bytes'] = _canonical(operation())

        _collective_preflight(self.accelerator, prepare)
        raw = local['bytes']
        lengths = _transport_lengths(self.accelerator, None, raw)
        return [json.loads(item) for item in _gather_payload_bytes(self.accelerator, raw, lengths)]

    def observe(self, position, inputs, meta, rng_before, graphs_before, graphs_after,
                *, origin='observed', force=False, activity_before=None, activity_after=None):
        """All ranks join compiler deltas, then capture real companions on an event."""
        def control():
            context = _position(position)
            _require(_integer(graphs_before) and _integer(graphs_after) and graphs_after >= graphs_before,
                     'graph counters must be nonnegative and monotonic')
            _require((activity_before is None) == (activity_after is None), 'both activity counters are required')
            activity_delta = 0
            if activity_before is not None:
                _require(_integer(activity_before) and _integer(activity_after) and activity_after >= activity_before,
                         'activity counters must be nonnegative and monotonic')
                activity_delta = activity_after - activity_before
            _require(origin in ('observed', 'reconstructed_legacy') and type(force) is bool, 'invalid event origin or force')
            _require(not force or origin == 'reconstructed_legacy', 'forced entries cannot be observed history')
            if origin == 'reconstructed_legacy':
                _require(type(self.bootstrap) is dict and bool(self.bootstrap), 'reconstructed history requires explicit bootstrap provenance')
            binding = dict(identity=self.identity, fingerprint=self.fingerprint, world_size=self.world_size,
                           bootstrap=self.bootstrap, ordinal=len(self.entries))
            return dict(position=context, origin=origin, force=force, delta=graphs_after - graphs_before,
                        activity_delta=activity_delta, binding=hashlib.sha256(_canonical(binding)).hexdigest())

        controls = self._gather(control)
        agreement = ['position', 'origin', 'force', 'binding']
        _require(all(all(row[key] == controls[0][key] for key in agreement) for row in controls),
                 'compile history position, origin or binding mismatch across ranks')
        if not any(row['delta'] or row['activity_delta'] or row['force'] for row in controls):
            return False
        rank = self.accelerator.process_index
        descriptors = self._gather(lambda: dict(rank=rank, inputs=input_signature(inputs), meta=_mapping(meta, 'batch meta'),
                                                rng_before=encode_rng(rng_before), delta=controls[rank]['delta'],
                                                activity_delta=controls[rank]['activity_delta']))
        entry = dict(ordinal=len(self.entries), position=controls[0]['position'], origin=origin, ranks=descriptors)
        _validate_entry(entry, len(self.entries), self.world_size, self.bootstrap)
        self.entries.append(_copy(entry))
        return True

    def to_payload(self):
        payload = dict(version=1, identity=self.identity, fingerprint=self.fingerprint, world_size=self.world_size,
                       entries=self.entries, bootstrap=self.bootstrap, checkpoint=self.checkpoint)
        self._validate(payload)
        return _copy(payload)

    @staticmethod
    def _validate(payload):
        _keys(payload, {'version', 'identity', 'fingerprint', 'world_size', 'entries', 'bootstrap', 'checkpoint'}, 'history')
        _require(type(payload['version']) is int and payload['version'] == 1, 'unsupported compile history version')
        _mapping(payload['identity'], 'identity')
        _mapping(payload['fingerprint'], 'fingerprint')
        world = payload['world_size']
        _require(_integer(world) and world > 0, 'invalid history world size')
        if payload['checkpoint'] is not None:
            _checkpoint(payload['checkpoint'])
        if payload['bootstrap'] is not None:
            _require(bool(_mapping(payload['bootstrap'], 'bootstrap provenance')), 'empty bootstrap provenance')
        _require(type(payload['entries']) is list, 'invalid history entries')
        for ordinal, entry in enumerate(payload['entries']):
            _validate_entry(entry, ordinal, world, payload['bootstrap'])

    def save(self, path, *, checkpoint):
        """Write local canonical bytes; caller owns checkpoint transaction/rank selection."""
        context = _checkpoint(checkpoint)
        payload = self.to_payload()
        payload['checkpoint'] = context
        raw = _canonical(payload)
        Path(path).write_bytes(raw)
        self.checkpoint = context
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def load(cls, path, expected_sha256, identity, fingerprint, accelerator, *, checkpoint):
        """Local-only digest and schema validation before accepting a journal."""
        _require(_sha(expected_sha256), 'expected history SHA256 is required')
        raw = Path(path).read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == expected_sha256, 'compile history SHA256 mismatch')
        try:
            payload = json.loads(raw, object_pairs_hook=_unique_fields)
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError('invalid compile history JSON') from error
        cls._validate(payload)
        result = cls(identity, fingerprint, accelerator)
        _require(_canonical(payload['identity']) == _canonical(result.identity), 'compile history identity mismatch')
        _require(_canonical(payload['fingerprint']) == _canonical(result.fingerprint), 'compile history fingerprint mismatch')
        _require(payload['world_size'] == result.world_size, 'compile history world size mismatch')
        _require(payload['checkpoint'] == _checkpoint(checkpoint), 'compile history checkpoint mismatch')
        result.entries = _copy(payload['entries'])
        result.bootstrap = _copy(payload['bootstrap'])
        result.checkpoint = _copy(payload['checkpoint'])
        return result
