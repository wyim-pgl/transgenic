#!/usr/bin/env python3
"""Controller-pinned, compute-only full ONT execution gate. Never submits jobs."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

GATE_SHA = '0b5be5e4863a348f0fb35a946e6d989980428483a72055e4d3f12492b7f7a17f'
PINS_SHA = '38c56c608336ef3fdcb83eaa5eb35a355ddc77ea1e8aff8387160603bf88ce34'
ROOT = Path('/data/gpfs/assoc/pgl/data/Transgenic')
SCOPE = 'full_alignment_descriptive'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identity(path):
    p = Path(path)
    require(p.is_absolute() and p.resolve(strict=True) == p, 'absolute nonredirected path required')
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return dict(path=str(p), size=p.stat().st_size, sha256=h.hexdigest())


def pinned(path, sha):
    entry = identity(path)
    require(entry['sha256'] == sha, 'pin mismatch: '+str(path))
    return entry


def check_source_closure(source, pins):
    """Exact frozen layout, not a snapshot: deployment must remain immutable."""
    require(source.is_absolute() and source.resolve(strict=True) == source
            and source.is_dir(), 'nonredirected frozen closure required')
    files = {Path(name) for name in pins['files']}
    require(len(files) == 9 and all(not p.is_absolute() and '..' not in p.parts for p in files),
            'invalid pinned source closure')
    directories = {parent for p in files for parent in p.parents if parent != Path('.')}
    seen = set()
    pending = [source]
    while pending:
        for p in pending.pop().iterdir():
            name = p.relative_to(source)
            mode = p.lstat().st_mode
            if name in directories:
                require(stat.S_ISDIR(mode), 'source closure directory redirected: '+str(name))
                pending.append(p)
            else:
                require(name in files and stat.S_ISREG(mode), 'unexpected source closure entry: '+str(name))
            seen.add(name)
    require(seen == files | directories, 'incomplete source closure')


def fresh(path):
    require(path.is_absolute() and path.resolve() == path, 'redirected namespace')
    path.mkdir(mode=0o700, exist_ok=False)  # Existing parent required, no implicit release creation.
    os.chmod(path, 0o700)


def save(path, data):
    # Atomic checkpoint, only within the exclusively owned private namespace.
    tmp = path.with_suffix('.tmp')
    with tmp.open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def stop_group(proc, grace):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    # Wait for group members too, even if the driver already exited.
    end = time.monotonic()+grace
    while time.monotonic() < end:
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(min(0.05, max(0, end-time.monotonic())))
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def child(argv, control, env, seconds, grace, receipt):
    receipt.update(argv=argv, exit_code=None, status='STARTING')
    proc = None
    try:
        with (control/'driver.stdout').open('x') as stdout, (control/'driver.stderr').open('x') as stderr:
            proc = subprocess.Popen(argv, stdout=stdout, stderr=stderr, env=env,
                                    cwd=control, start_new_session=True)
            receipt.update(pid=proc.pid, status='RUNNING')
            save(control/'receipt.json', receipt)
            try:
                receipt['exit_code'] = proc.wait(timeout=seconds)
                receipt['status'] = 'EXITED'
            except subprocess.TimeoutExpired as error:
                receipt['status'] = 'TIMED_OUT'
                raise TimeoutError('driver wall budget exceeded') from error
    finally:
        if proc is not None:
            # A second TERM or the outer alarm must not interrupt group cleanup.
            mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT, signal.SIGALRM})
            try:
                stop_group(proc, grace)
                receipt['exit_code'] = proc.returncode
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        for name in ('stdout', 'stderr'):
            p = control/('driver.'+name)
            if p.exists():
                receipt[name] = identity(p)


def check_boundary(receipt):
    require(receipt['scope'] == SCOPE and receipt['structural_status'] == 'UNASSESSABLE'
            and receipt['training_eligible'] is False and receipt['decision'] is None, 'scope drift')
    require(receipt['exit_code'] == 0, 'nonzero alignment exit')
    for stream in ('stdout', 'stderr'):
        require(identity(receipt[stream]['path']) == receipt[stream], stream+' mutated')


def slurm(request, control):
    """Live scheduler query plus local CPU affinity, not environment-only approval."""
    budget = request['budget']
    for key in ('cpus', 'memory_mib', 'wall_seconds', 'cleanup_seconds'):
        require(type(budget[key]) is int and budget[key] > 0, 'explicit positive budget required: '+key)
    require(budget['cpus'] >= 4, 'frozen seqkit default needs at least four allocated CPUs')
    require(budget['cleanup_seconds'] < budget['wall_seconds'], 'cleanup reserve must fit wall budget')
    job = os.environ.get('SLURM_JOB_ID', '')
    require(job.isdigit(), 'real SLURM allocation required')
    require(os.environ.get('SLURM_CPUS_PER_TASK') == str(budget['cpus']), 'CPU allocation mismatch')
    require(len(os.sched_getaffinity(0)) == budget['cpus'], 'explicit CPU binding required')
    sc = request['scontrol']['path']
    pinned(sc, request['scontrol']['sha256'])
    result = subprocess.run([sc, 'show', 'job', '-o', job], capture_output=True,
                            text=True, timeout=min(10, budget['wall_seconds']))
    with (control/'slurm.stdout').open('x') as f: f.write(result.stdout)
    with (control/'slurm.stderr').open('x') as f: f.write(result.stderr)
    require(result.returncode == 0, 'scheduler query failed')
    fields = dict(token.split('=', 1) for token in shlex.split(result.stdout) if '=' in token)
    require(fields.get('JobId') == job and fields.get('JobState') == 'RUNNING', 'job not running')
    require(fields.get('UserId', '').endswith('('+str(os.getuid())+')'), 'job not owned by caller')
    require(fields.get('NumNodes') == '1' and fields.get('NumCPUs') == str(budget['cpus'])
            and fields.get('NumTasks') == '1', 'single task single node allocation required')
    require(fields.get('NodeList') in (socket.gethostname(), socket.gethostname().split('.')[0]), 'wrong compute host')
    tres = dict(x.split('=', 1) for x in fields.get('AllocTRES', '').split(',') if '=' in x)
    require(not any('gpu' in k for k in tres), 'CPU-only allocation required')
    mem = tres.get('mem', '')
    require(mem.endswith(('M', 'G')) and float(mem[:-1]) * (1024 if mem.endswith('G') else 1) == budget['memory_mib'], 'reviewed memory allocation mismatch')
    runtime = fields.get('RunTime', '')
    limit = fields.get('TimeLimit', '')
    def seconds(value):
        days, _, clock = value.rpartition('-')
        h, m, s = map(int, clock.split(':'))
        return int(days or 0)*86400+h*3600+m*60+s
    require(seconds(limit)-seconds(runtime) >= budget['wall_seconds'], 'insufficient allocation time remaining')
    return dict(job=job, fields=fields, host=socket.gethostname())


def execute(request_path, request_sha, wrapper_sha):
    require(sys.version_info >= (3, 9), 'Python >=3.9 required; use the independently pinned deployment interpreter')
    sys.dont_write_bytecode = True
    wrapper = pinned(Path(__file__).resolve(), wrapper_sha)
    req_identity = pinned(Path(request_path), request_sha)
    request = json.loads(Path(request_path).read_text())
    require(request['scope'] == SCOPE, 'not a full alignment request')
    require(request['runtime_reviewed'] is True, 'runtime/source review still required')
    control = Path(request['control'])
    # Outside data/release, with a fresh private directory and no redirected ancestors.
    release = Path(request['release'])
    source = release/'frozen'
    for protected in (ROOT, release, Path(__file__).resolve().parent):
        require(not control.is_relative_to(protected) and not protected.is_relative_to(control), 'control overlaps protected namespace')
    fresh(control)
    receipt = dict(schema=1, scope=SCOPE, structural_status='UNASSESSABLE', decision=None,
                   training_eligible=False, status='INCOMPLETE', exit_code=None,
                   request=req_identity, wrapper=wrapper, outputs={})
    save(control/'receipt.json', receipt)
    old_handlers = {}
    def interrupted(signum, frame):
        raise TimeoutError('wall budget expired') if signum == signal.SIGALRM else InterruptedError('signal '+str(signum))
    try:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            old_handlers[sig] = signal.signal(sig, interrupted)
        budget = request['budget']
        require(type(budget['wall_seconds']) is int and budget['wall_seconds'] > 0, 'wall budget required')
        start = time.monotonic()
        signal.setitimer(signal.ITIMER_REAL, budget['wall_seconds'])
        receipt['scheduler'] = slurm(request, control)
        inputs = [wrapper, req_identity, pinned(release/'gate.py', GATE_SHA),
                  pinned(release/'source_pins.json', PINS_SHA)]
        pins = json.loads((release/'source_pins.json').read_text())
        check_source_closure(source, pins)  # Before ANY gate or driver import.
        for name, sha in pins['files'].items():
            inputs.append(pinned(source/name, sha))
        require(not list((release/'__pycache__').glob('gate.*.pyc')),
                'remove deployed bytecode in a separately reviewed fresh release')
        spec = importlib.util.spec_from_file_location('ont_pinned_gate', release/'gate.py')
        gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
        driver = gate.load_driver(source)
        run = request['run']
        require(run in gate.RUNS, 'only canonical ONT runs allowed')
        args = SimpleNamespace(root=str(ROOT), run=run,
            reads=str(ROOT/'evidence'/gate.DATASET/(run+'.fa.gz')),
            fasta=str(ROOT/'genomes/Athaliana_167_TAIR10.fa'),
            gff=str(ROOT/'genomes/Athaliana_167_TAIR10.gene.clean.gff3'),
            panels=str(source/'revision/protocols/b6_panels_v1.tsv'),
            authority=str(release/'authority.tsv'), source=str(source), threads=budget['cpus'])
        out = driver.output_directory(args.root, run)
        require(not out.exists() and not out.is_symlink(), 'driver output already exists; never reuse/delete')
        binding = gate.read_binding(args.root, run, args.reads)
        ref = gate.reference_binding(Path(args.authority).read_bytes(), args.fasta, args.gff)
        require(len(driver.w.load_panel(args.panels)) == 50, 'full frozen panel required')
        inputs += [identity(p) for p in (args.reads, args.fasta, args.gff, args.authority, binding['roles']['path'])]
        tools = request['tools']
        require(set(tools) == {'minimap2', 'samtools', 'seqkit', 'python'}, 'exact tool inventory required')
        for tool in tools.values():
            inputs.append(pinned(tool['path'], tool['sha256']))
            require(os.access(tool['path'], os.X_OK), 'tool not executable')
        require(Path(tools['python']['path']) == Path(sys.executable).resolve(), 'launch wrapper using reviewed interpreter')
        binary = Path(tools['minimap2']['path']).parent
        for name in ('minimap2', 'samtools', 'seqkit'):
            require(Path(tools[name]['path']) == binary/name, 'bare frozen commands need same pinned binary directory')
        inputs.append(pinned(request['scontrol']['path'], request['scontrol']['sha256']))
        receipt.update(inputs=inputs, args=vars(args), budget=budget, source_commit=gate.COMMIT,
                       output_directory=str(out))
        argv = [tools['python']['path'], '-I', '-B', str(source/'revision/scripts/mvp/mvp_align_panel_longreads.py'),
                '--root', args.root, '--run', run, '--reads', args.reads, '--ref', args.fasta,
                '--gff', args.gff, '--panels', args.panels, '--ref-md5', ref['files']['fasta']['md5'],
                '--gff-md5', ref['files']['gff']['md5'], '--threads', str(args.threads), '--execute']
        env = dict(PATH=str(binary), HOME=str(control), TMPDIR=str(control), LANG='C', LC_ALL='C',
                   PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS=str(args.threads), GOMAXPROCS=str(args.threads))
        require(not out.exists() and not out.is_symlink(), 'driver output appeared during preflight')
        save(control/'receipt.json', receipt)
        remaining = budget['wall_seconds']-budget['cleanup_seconds']-(time.monotonic()-start)
        require(remaining > 0, 'preflight exhausted wall budget')
        child(argv, control, env, remaining, budget['cleanup_seconds'], receipt)
        require(not (out/'FAILED.json').exists(), 'driver failure artifact')
        require(not (out/'COMPLETE').exists() and not (out/'receipt.json').exists(), 'unexpected excluded inventory artifact')
        receipt['outputs'] = gate.inventory(out)
        check_boundary(receipt)
        receipt['postflight'] = gate.verify_alignment(args, receipt)
        for entry in inputs:
            require(identity(entry['path']) == entry, 'input/code/tool mutated: '+entry['path'])
        check_source_closure(source, pins)
        gate.verify_inventory(out, receipt['outputs'])
        check_boundary(receipt)
        receipt.update(status='VERIFIED_DESCRIPTIVE', elapsed_seconds=time.monotonic()-start)
        receipt['diagnostics'] = {p.name: identity(p) for p in control.iterdir() if p.name not in ('receipt.json', 'COMPLETE')}
        for stream in ('stdout', 'stderr'):
            require(receipt['diagnostics'].get('driver.'+stream) == receipt[stream], stream+' diagnostics mutated')
        save(control/'receipt.json', receipt)
        # All expensive checks precede publication. Controller still requires Slurm accounting.
        with (control/'COMPLETE').open('x') as f:
            f.write(identity(control/'receipt.json')['sha256']+'\n')
        return receipt
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        # Only our publication marker, never any driver artifact, may be withdrawn.
        (control/'COMPLETE').unlink(missing_ok=True)
        receipt.update(status='INCOMPLETE', error=type(error).__name__+': '+str(error))
        save(control/'receipt.json', receipt)
        with (control/'FAILED.json').open('x') as f: json.dump(dict(error=receipt['error']), f)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in old_handlers.items(): signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('request', 'request-sha256', 'wrapper-sha256'):
        parser.add_argument('--'+key, required=True)
    options = parser.parse_args()
    print(json.dumps(execute(options.request, options.request_sha256, options.wrapper_sha256), indent=2))
