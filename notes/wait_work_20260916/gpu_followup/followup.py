#!/usr/bin/env python3
"""Fail-closed, single-mode post-P1 command builder/runner. Never submits Slurm jobs.

Run planning against actual small S0/P1 files; execute ONLY in the original
container namespace on compute. Local structural checks are not CUDA PASS.
The controller supplies paths, current complete accounting and wall caps.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

COMMIT = 'b660bf1e7dc48287ca34a0d6e3aeb53d45f599b6'
MODES = ('real_tokenizer', 'real_hyena_step0_three_tiers', 'generation_cache_parity',
         'bf16_compile', 'four_rank_smoke', 'resource_budget')
STAGES = ('p1_identity', 'p2_step0_determinism', 'p3_generation_parity',
          'p4_compile_topology', 'p4_compile_topology', 'p4_compile_topology')
HERE = Path(__file__).resolve().parent

def require(ok, why):
    if not ok: raise ValueError(why)

def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)+'\n').encode()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    with Path(path).open('xb') as stream: stream.write(canonical(value))

def wrapper(path, mode, binding):
    path = Path(path)
    require(not path.with_name(mode+'.FAILED.json').exists(), 'FAILED receipt present')
    require(path.with_suffix('.DONE').read_text().strip() == sha(path), 'DONE mismatch')
    r = load(path)
    require(r.get('schema_version') == 1 and r.get('preflight') == STAGES[MODES.index(mode)] and
            r.get('pass') is True and r.get('bindings_sha256') == binding and
            r.get('exit_code') == 0 and r.get('timed_out') is False and r.get('signal') is None and
            r.get('resume_supported') is False and bool(r.get('allocation')), 'wrapper execution/binding failure')
    m = r.get('measurement', {})
    require(m.get('bindings_sha256') == binding and m.get('mode') == mode and
            m.get('device') == 'cuda' and m.get('real_runtime') is True, 'real measurement required')
    identity = m.get('probe_identity', {})
    require(identity.get('schema_version') == 1 and identity.get('imports') and identity.get('asset_roots') and
            identity.get('probe_path') in identity['imports'], 'complete import identity required')
    if mode == 'real_tokenizer':
        require(m.get('tokenizer_roundtrip') is True and m.get('hyena_loaded') is True, 'P1 checks failed')
    return r

def budget(ledger, outer_seconds):
    now = time.time()
    require(type(outer_seconds) is int and outer_seconds > 0, 'positive integer outer cap')
    require(all(type(ledger[k]) in (int,float) and math.isfinite(ledger[k]) for k in ('observed_at','expires_at')), 'finite ledger clock required')
    require(ledger['observed_at'] <= now < ledger['expires_at'] and
            now + outer_seconds <= ledger['expires_at'], 'expired or insufficient clock authorization')
    require(ledger.get('controller_record_reference') and ledger.get('complete_diagnostic_history') is True and
            ledger.get('no_overlap_confirmed') is True, 'complete controller accounting/no-overlap required')
    jobs = ledger['jobs']; ids = [str(j['job_id']) for j in jobs]
    require(len(ids) == len(set(ids)) and '3163986' in ids and '3163545' in ids, 'missing/duplicate P1 accounting')
    spent = 0
    for j in jobs:
        require(j['state'] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'),
                'active/pending diagnostic reservation: no overlap permitted')
        n, seconds = j['nodes'], j['charged_seconds']
        require(type(n) is int and n > 0 and type(seconds) in (int,float) and
                math.isfinite(seconds) and seconds >= 0, 'invalid node accounting')
        spent += n * seconds
    require(next(j for j in jobs if str(j['job_id'])=='3163986')['state']=='COMPLETED', 'P1 not completed')
    require(spent + outer_seconds <= 18000, '5 node-hour cap exceeded')
    return spent

def source_binding(manifest, expected):
    require(bool(manifest.get('files')), 'empty source binding')
    for path, digest in manifest['files'].items():
        require(expected.get(path)==digest, 'b660bf1 source mismatch: '+path)

def build(args):
    cfg = load(args.inputs)
    require(cfg['commit'] == COMMIT, 'wrong source commit')
    spent = budget(load(cfg['ledger']), cfg['outer_seconds'])
    require(type(cfg['timeout_seconds']) is int and 0 < cfg['timeout_seconds'] <= cfg['outer_seconds'] - 120,
            'reserve >=120 seconds for gating/launch/teardown')
    control = Path(cfg['s0_control'])
    pins = load(control/'s0-inputs.json')
    require(set(pins)=={'prepared.json','consumed-code.json','assets.json','tiles.json','tile-availability.json'}, 'S0 inventory')
    for name, digest in pins.items(): require(sha(control/name)==digest, 'S0 changed: '+name)
    require(load(control/'tile-availability.json')['status']=='available_and_actual_measured_batches_checked','S0 tiles blocked')
    prepared = load(control/'prepared.json'); bindings = prepared['bindings']
    binding = hashlib.sha256(canonical(bindings)).hexdigest()
    require(prepared['bindings_sha256']==binding and bindings['seed'] in (1,2,3), 'prepared identity')
    manifest = load(control/'consumed-code.json')
    source_binding(manifest, load(HERE/'source-files.json'))
    require(bindings['consumed_code_manifest']['sha256']==pins['consumed-code.json'], 'consumed manifest binding mismatch')
    tooling = bindings['tooling']
    for path, digest in tooling.items():
        original = HERE/'source/revision/scripts/mvp'/Path(path).name
        require(original.is_file() and sha(original)==digest, 'b660bf1 tooling mismatch: '+path)
    expected_names = {'make_launch_readiness.py','stage_checkpoint.py','run_bounded.py','run_training.py',
                      'accelerate_mvp.json','real_probes.py','resource_probe.py'}
    require({Path(p).name for p in tooling}==expected_names, 'incomplete tooling binding')
    p1 = wrapper(cfg['receipts']['real_tokenizer'], 'real_tokenizer', binding)
    require(str(p1['allocation'])=='3163986', 'not actual final P1 allocation')
    p1argv = load(control/'p1.argv.json')
    require(sha(control/'p1.argv.json')==p1['command_sha256'], 'P1 argv digest mismatch')
    require(p1argv[1]=='/work/repo/revision/scripts/mvp/real_probes.py', 'unexpected P1 executable')
    options = dict(zip(p1argv[2::2],p1argv[3::2]))
    require(options['--mode']=='real_tokenizer' and options['--bindings-file-sha256']==pins['prepared.json'] and
            options['--assets-sha256']==pins['assets.json'] and options['--tile-rns-sha256']==pins['tiles.json'], 'P1/S0 argv mismatch')
    require(p1['measurement']['probe_identity']['imports'][p1argv[1]]==sha(HERE/'source/revision/scripts/mvp/real_probes.py'),
            'P1 source executable mismatch')
    mode = args.mode
    for previous in MODES[:MODES.index(mode)]: wrapper(cfg['receipts'][previous], previous, binding)
    require(Path(args.output).is_absolute(), 'absolute stable output path required')
    output = Path(args.output).absolute()
    require(not output.exists(), 'output must be fresh; no automatic resume')
    scratch = output/'cache'; receipt_dir = output/'receipt'
    options['--mode'] = mode; options['--scratch-root'] = str(scratch)
    if mode == 'resource_budget':
        authorization = cfg.get('resource', {})
        required = ('budget-authorization','budget-sha256','acceptance-record','inventory-record','receipts-record','panel','sif')
        require(all(authorization.get(k) for k in required), 'resource BLOCKED: actual controller training budget/inputs required')
        for key in required: options['--'+key] = authorization[key]
        options['--profile-dir'] = str(output/'profile')
        require(sha(authorization['budget-authorization'])==authorization['budget-sha256'], 'budget digest')
        b = load(authorization['budget-authorization'])
        require(b.get('bindings_sha256')==binding and b.get('prior_smoke_node_hours',-1)*3600 >= spent and
                b.get('max_probe_wall_seconds',0) <= cfg['timeout_seconds'], 'resource authorization binding/accounting')
    command = [p1argv[0], p1argv[1]] + [v for pair in options.items() for v in pair]
    if mode in ('bf16_compile','four_rank_smoke','resource_budget'):
        command = [p1argv[0], '-m','accelerate.commands.launch','--multi_gpu','--num_machines','1',
                   '--num_processes','4','--mixed_precision','bf16','--config_file',bindings['accelerate_config']['path'],
                   '--max_restarts','0'] + command[1:]
    if args.execute:
        require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_NUM_NODES')=='1', 'allocated single node required')
        require(os.environ.get('ALLOCATION_RESERVED_SECONDS')==str(cfg['outer_seconds']), 'outer allocation reservation mismatch')
        sys.path.insert(0, str(Path(p1argv[1]).parent))
        from make_launch_readiness import _load_receipt, tooling_identity
        require(tooling_identity()==tooling, 'actual runtime tooling mismatch')
        for previous in MODES[:MODES.index(mode)]:
            _load_receipt(Path(cfg['receipts'][previous]).parent, previous, binding)
        if mode=='resource_budget':
            from resource_probe import validate_budget
            validate_budget(b, binding)
        # Native receipt/import/tree validation may be slow; charge it too.
        budget(load(cfg['ledger']), cfg['outer_seconds'])
    output.mkdir()
    save(output/'inputs.snapshot.json',cfg)
    save(output/'command.json',command)
    runner = [p1argv[0], str(Path(p1argv[1]).with_name('run_bounded.py')), '--name',mode,
              '--bindings',options['--bindings'],'--bindings-file-sha256',options['--bindings-file-sha256'],
              '--command-file',str(output/'command.json'),'--command-sha256',sha(output/'command.json'),
              '--output-dir',str(receipt_dir),'--timeout-seconds',str(cfg['timeout_seconds'])]
    save(output/'runner.json',runner)
    save(output/'plan.json',dict(commit=COMMIT, seed=bindings['seed'], binding=binding, mode=mode,
         measured_pass=False, outer_seconds=cfg['outer_seconds'], prior_node_seconds=spent,
         retain_paths=[str(scratch),str(receipt_dir)], runner=runner))
    if not args.execute:
        print('STRUCTURAL PLAN ONLY; native validation and fresh --execute output still required')
        return
    for name in ('modules','inductor','triton','tmp','home'): (scratch/name).mkdir(parents=True,exist_ok=False)
    env = dict(os.environ, HF_MODULES_CACHE=str(scratch/'modules'), TORCHINDUCTOR_CACHE_DIR=str(scratch/'inductor'),
               TRITON_CACHE_DIR=str(scratch/'triton'), TMPDIR=str(scratch/'tmp'), HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(runner,env=env,check=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',required=True,help='actual controller JSON described in PLAN.md')
    p.add_argument('--mode',choices=MODES[1:],required=True)
    p.add_argument('--output',required=True,help='fresh absolute path, stable through decode')
    p.add_argument('--execute',action='store_true',help='compute/container only; never sbatch')
    args=p.parse_args()
    build(args)

if __name__=='__main__': main()
