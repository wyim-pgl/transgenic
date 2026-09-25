"""Project pool quota, not df free space, limits CP4 checkpoint writes."""
import importlib.util
from pathlib import Path

import pytest

p = Path(__file__).parents[1] / 'revision/scripts/b6/storage_preflight.py'
spec = importlib.util.spec_from_file_location('b6_storage_preflight', p)
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)


@pytest.fixture
def storage_environment(monkeypatch):
    from types import SimpleNamespace
    root = '/work/hdd/bilv/wcyim/results'
    facts = dict(project='22169 P ' + root, layout='pool: ddn_hdd',
                 quota='/work/hdd/bilv 10884 1048576000 1153433600 - 14239 0 0 -',
                 available=2 ** 40)
    monkeypatch.setattr(storage.Path, 'resolve', lambda self, strict=True: Path(root))
    monkeypatch.setattr(storage.os, 'statvfs', lambda path: SimpleNamespace(f_bavail=facts['available'], f_frsize=1))
    def command(args, **kwargs):
        return facts[{'project': 'project', 'getstripe': 'layout', 'quota': 'quota'}[args[1]]]
    monkeypatch.setattr(storage.subprocess, 'check_output', command)
    return root, facts


def test_preflight_checks_real_pool_quota_not_only_filesystem_space(storage_environment):
    root, facts = storage_environment
    got = storage.check_storage(root)
    assert got['pool'] == 'ddn_hdd' and got['quota_remaining_bytes'] == 1073730678784
    facts['quota'] = '/work/hdd/bilv 1048575999 1048576000 1153433600 - 14239 0 0 -'
    with pytest.raises(RuntimeError, match='insufficient CP4 storage'):
        storage.check_storage(root)


@pytest.mark.parametrize('key,value', [('layout', 'pool: ddn_ssd'), ('project', '22169 - /wrong'),
                                       ('available', 1), ('quota', '/work/hdd/bilv 1 0 0 -')])
def test_preflight_refuses_wrong_pool_or_inadequate_space(storage_environment, key, value):
    root, facts = storage_environment
    facts[key] = value
    with pytest.raises((ValueError, RuntimeError)):
        storage.check_storage(root)


def test_full_nvme_pool_has_no_soft_quota_headroom():
    text = ' /work/nvme/bilv 574895272* 524288000 576716800 6d21h 14239* 0 0 -\n'
    assert storage.quota_headroom(text, '/work/nvme/bilv') == 0


def test_hdd_quota_uses_smaller_limit_and_kib_units():
    text = ' /work/hdd/bilv 10884 1048576000 1153433600 - 14239 0 0 -\n'
    assert storage.quota_headroom(text, '/work/hdd/bilv') == 1073730678784
    assert storage.quota_headroom(' /x 1 0 3 - 0 0 0 -', '/x') == 2048
    assert storage.quota_headroom(' /x 1 9 3 - 0 0 0 -', '/x') == 2048


@pytest.mark.parametrize('text', ['', '/wrong 1 2 3 -', '/x 1 0 0 -', '/x ? 2 3 -',
                                 '/x 1 2 3 -\n/x 1 2 3 -', '/x 1 2'])
def test_missing_or_ambiguous_quota_is_refused(text):
    with pytest.raises(ValueError):
        storage.quota_headroom(text, '/x')
