"""Bounded trusted-runtime metadata only. No alignment, installs or approval."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def identity(path):
    p = Path(path); resolved = p.resolve(strict=True); st = resolved.stat()
    h = hashlib.sha256()
    with resolved.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return dict(path=str(p), resolved=str(resolved), symlink=os.readlink(p) if p.is_symlink() else None,
                sha256=h.hexdigest(), size=st.st_size, uid=st.st_uid, gid=st.st_gid,
                mode=oct(st.st_mode & 0o7777), parents=[dict(path=str(a), uid=a.stat().st_uid,
                gid=a.stat().st_gid, mode=oct(a.stat().st_mode & 0o7777)) for a in resolved.parents])


# Explicit union of harness, reused gate/wrapper and executed frozen imports.
# No mvp_bootstrap/NumPy, no scientific CLI invocation. Includes loaded .pyc bytes.
PYTHON_INVENTORY = r'''
import argparse, bisect, collections, contextlib, csv, dataclasses, gzip, hashlib
import importlib.util, io, json, math, os, pathlib, random, re, resource
import shlex, signal, site, socket, stat, subprocess, sys, time, types
from pathlib import Path

def file(p):
    p = Path(p); q = p.resolve(strict=True); s = q.stat()
    return dict(path=str(p), resolved=str(q), sha256=hashlib.sha256(q.read_bytes()).hexdigest(),
                size=s.st_size, uid=s.st_uid, gid=s.st_gid, mode=oct(s.st_mode & 0o7777),
                parents=[dict(path=str(a),uid=a.stat().st_uid,gid=a.stat().st_gid,
                              mode=oct(a.stat().st_mode & 0o7777)) for a in q.parents])
modules = {}
for name, mod in sorted(list(sys.modules.items())):
    origin = getattr(mod, '__file__', None)
    entry = dict(origin=origin, spec_origin=getattr(getattr(mod, '__spec__', None), 'origin', None))
    if origin and Path(origin).is_file():
        entry['file'] = file(origin)
        cached = getattr(mod, '__cached__', None)
        if cached and Path(cached).is_file(): entry['cached'] = file(cached)
    modules[name] = entry
pth = {}
for directory in sorted(set(site.getsitepackages())):
    p = Path(directory)
    if p.exists():
        for item in sorted(p.glob('*.pth')):
            pth[str(item)] = dict(file(item), content=item.read_text())
print(json.dumps(dict(executable=sys.executable, version=sys.version, prefix=sys.prefix,
    base_prefix=sys.base_prefix, sys_path=sys.path, isolated=sys.flags.isolated,
    no_user_site=sys.flags.no_user_site, enable_user_site=site.ENABLE_USER_SITE,
    modules=modules, pth=pth), sort_keys=True))
'''


def linked(path, env):
    result = subprocess.run(['/usr/bin/ldd', str(path)], capture_output=True, text=True, timeout=15,
                            env=dict(env, PATH='/usr/bin:/bin'))  # ldd shell utilities, not workload PATH
    text = result.stdout+result.stderr
    if 'not a dynamic executable' in text or 'statically linked' in text:
        return dict(linkage='static', files={})
    if result.returncode or 'not found' in text:
        raise ValueError('unresolved dynamic linkage: '+str(path))
    paths = sorted(set(re.findall(r'(?:=>\s+|^\s*)(/\S+)', text, re.M)))
    if not paths: raise ValueError('missing loader/library metadata: '+str(path))
    return dict(linkage='dynamic', files={p:identity(p) for p in paths})


def collect(runtime, env):
    """Same code and clean env on login and compute. Hosts/times recorded separately."""
    tools = runtime['tools']; py = tools['python']['path']
    report = dict(tools={}, python={}, libraries={}, versions={},
                  path_resolution={name:shutil.which(name,path=env['PATH']) for name in ('minimap2','samtools','seqkit')})
    if any(report['path_resolution'][name] != tools[name]['path'] for name in report['path_resolution']):
        raise ValueError('tool PATH resolution mismatch')
    for name, item in sorted(tools.items()):
        report['tools'][name] = identity(item['path'])
        if report['tools'][name]['sha256'] != item['sha256']: raise ValueError('executable changed: '+name)
        report['libraries'][item['path']] = linked(item['path'], env)
        version = subprocess.run([item['path']]+item['version_args'], env=env, capture_output=True, text=True, timeout=10)
        if version.returncode or not version.stdout.splitlines() or version.stdout.splitlines()[0] != item['version']:
            raise ValueError('runtime version changed: '+name)
        report['versions'][name] = dict(stdout=version.stdout, stderr=version.stderr, exit_code=version.returncode)
    for mode, flags in [('isolated', ['-I', '-B']), ('ordinary_helper', ['-B'])]:
        result = subprocess.run([py]+flags+['-c', PYTHON_INVENTORY], env=env,
                                cwd=env['HOME'], capture_output=True, text=True, timeout=30)
        if result.returncode: raise ValueError('Python inventory failed: '+mode)
        data = json.loads(result.stdout)
        # -c ordinary startup uses empty path; actual frozen helpers use their script directory.
        data['sys_path'] = ['<script-directory>' if p == '' else p for p in data['sys_path']]
        report['python'][mode] = data
        for entry in data['modules'].values():
            path = entry.get('origin')
            if path and (path.endswith('.so') or '.so.' in path) and path not in report['libraries']:
                report['libraries'][path] = linked(path, env)
    for name, item in sorted(runtime.get('operations', {}).items()):
        report['tools'][name] = identity(item['path'])
        if report['tools'][name]['sha256'] != item['sha256']: raise ValueError('operational executable changed: '+name)
        report['libraries'][item['path']] = linked(item['path'], env)
    return report


def check_cgroup(approval):
    """Observe actual job memory hierarchy, fail closed on unsupported layouts."""
    if approval.get('scheduler_cgroup_reviewed') is not True:
        raise ValueError('controller scheduler cgroup review required')
    job = os.environ.get('SLURM_JOB_ID', '')
    text = Path('/proc/self/cgroup').read_text()
    if not job.isdigit(): raise ValueError('job ID required')
    v1 = [line.split(':', 2)[2] for line in text.splitlines() if 'memory' in line.split(':', 2)[1].split(',')]
    if v1:
        if len(v1) != 1: raise ValueError('ambiguous memory membership')
        root = Path('/sys/fs/cgroup/memory'); current = root / v1[0].lstrip('/')
        jobdir = next((p for p in [current]+list(current.parents) if p.name in ('job_'+job, 'job-'+job)), None)
        if jobdir is None: raise ValueError('memory cgroup does not identify current job')
        records = []
        for path in [current]+list(current.parents):
            if not path.is_relative_to(root): break
            record = dict(path=str(path))
            for name in ('memory.limit_in_bytes', 'memory.memsw.limit_in_bytes', 'memory.use_hierarchy',
                         'memory.failcnt', 'memory.oom_control', 'tasks'):
                p = path/name
                if p.exists(): record[name] = p.read_text().strip()
            records.append(record)
        jr = next(r for r in records if r['path'] == str(jobdir))
        if jr.get('memory.use_hierarchy') != '1' or int(jr.get('memory.limit_in_bytes', 0)) != 4294967296:
            raise ValueError('hierarchical job 4 GiB memory limit required')
        if int(jr.get('memory.memsw.limit_in_bytes', 0)) != 4294967296:
            raise ValueError('job combined memory/swap limit evidence required')
        if str(os.getpid()) not in (current/'tasks').read_text().split(): raise ValueError('process membership mismatch')
        return dict(version=1, membership=text, hierarchy=records,
                    scope='current process and inherited children; scheduler reaping separately reviewed')
    raise ValueError('only the reviewed site-v1 memory hierarchy is authorized for this fixture')
