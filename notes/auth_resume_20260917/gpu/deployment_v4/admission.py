#!/usr/bin/env python3
"""Durable controller permission plus bounded, read-only compute-start admission.
No submission, cancellation, refresh, retry, or incoming observation-file API.
"""
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import selectors
import subprocess
import time

ACCOUNT = 'bilv-dtai-gh'
USER = 'wcyim'
UID = 97434
CLUSTER = 'delta-gh'
START = '2026-09-16T00:00:00'
NAMESPACE = '/work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545/staging/dev123_p1_release_v6_20260917'
ACCOUNTS_SHA = '44302f2e6bd0773e1acd87c521947e3dac4ca9c581e9c79b7433489f22be35f5'
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'PREEMPTED', 'REVOKED'}
FIELDS = 'JobIDRaw,JobName%100,User,UID,Account,Partition,State%40,ElapsedRaw,AllocNodes,AllocCPUS,AllocTRES%200,ReqTRES%200,TimelimitRaw,Submit,Start,End,NodeList%100,ExitCode'
KEYS = 'id name user uid account partition state elapsed nodes cpus alloc req limit submit start end placement exit'.split()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def finite(x):
    return type(x) in (int, float) and math.isfinite(x) and x >= 0


def example():
    """Executable schema template, deliberately NOT approved and with no true facts."""
    return dict(schema=4, approved=False, issued_at_unix=None,
        release_json_sha256=None, transport_sha256=None, ledger_sha256=None,
        controller_reference=None, review_reference=None, terminal_recovery_reference=None,
        staging_reference=None, reservation_identity=None,
        attempt_basis='REVIEWED_3163986_FAILURE_RECOVERY_NOT_AUTOMATIC_RETRY',
        evidence_recovery=dict(source='/work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545/final-p1-job-3163986/inputs/evidence/fullc0lite.db',
            bytes=1068773376, sha256='ed2d0eef2b205626db0983ab9f029f84c30514babc77c330bdb250b2fdfe3d0d',
            owner=97434, read_only=True, preserve_source=True),
        terminal_history={'3163986': dict(state='FAILED', exit_code='1:0', elapsed_seconds=51,
            nodes=1, gpus=4, allocated_cpus=288, requested_cpus=64,
            node_hours=51/3600, gpu_hours=204/3600),
            '3163545': dict(state='CANCELLED', exit_code='0:0', elapsed_seconds=0)}, 
        scope='S0_P1_DIAGNOSTIC_ONLY', profile='development-single', seed=123,
        run_role='mvp_finetune', training_authorized=False, followup_authorized=False,
        namespace=NAMESPACE, cluster=CLUSTER, owner=USER, uid=UID, account=ACCOUNT,
        one_attempt=True, nodes=1, gpus=4, cpus=64, exclusive=True, seconds_max=1800,
        node_hours_max=0.5, gpu_hours_max=2, cumulative_node_hours_max=5,
        cumulative_gpu_hours_max=20, requeue=False, restart=False, resume=False,
        automatic_retry=False, serialized_through_terminal_reconciliation=True,
        campaign_start_local=START, timezone='America/Chicago',
        registered_prior_ids=['3163545', '3163986'],
        b5_registered_ids=['3131537', '3131538', *[str(i) for i in range(3149356, 3149364)]],
        other_reservations=[dict(identity='panel-candidate-not-submitted', job_id=None,
                                 node_hours=1, gpu_hours=4, submitted=False)],
        b5_policy='full_maximum_wall_times_allocated_or_requested_gpu_for_all_outstanding',
        diagnostic_account_reserve_gpu_hours=20, operational_margin_gpu_hours=20,
        excluded_nodes=['gh062', 'gh093'])


def validate_authorization(a, release_sha, transport_sha, ledger_sha, now=None):
    now = time.time() if now is None else now
    template = example()
    variable = {'approved', 'issued_at_unix', 'release_json_sha256', 'transport_sha256',
                'ledger_sha256', 'controller_reference', 'review_reference',
                'terminal_recovery_reference', 'staging_reference', 'reservation_identity',
                'registered_prior_ids', 'b5_registered_ids', 'other_reservations'}
    need(set(a) == set(template), 'unknown/missing authorization fields')
    for k in set(template) - variable:
        need(type(a[k]) is type(template[k]) and a[k] == template[k], 'authorization scope: '+k)
    need(a['approved'] is True, 'template is not approved')
    need(finite(a['issued_at_unix']) and a['issued_at_unix'] <= now, 'future/invalid issuance')
    for key, value in [('release_json_sha256', release_sha), ('transport_sha256', transport_sha),
                       ('ledger_sha256', ledger_sha)]:
        need(isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value) and a[key] == value,
             'external release identity: '+key)
    for k in ('controller_reference', 'review_reference', 'terminal_recovery_reference',
              'staging_reference', 'reservation_identity'):
        need(isinstance(a[k], str) and a[k].strip(), 'controller provenance missing: '+k)
    ids = a['registered_prior_ids']
    need(isinstance(ids, list) and len(ids) == len(set(ids)) and
         {'3163545', '3163986'} <= set(ids) and all(re.fullmatch(r'[0-9]+', i) for i in ids),
         'registered history incomplete')
    b5_ids = a['b5_registered_ids']
    need(isinstance(b5_ids, list) and len(b5_ids) == len(set(b5_ids)) and
         all(isinstance(i, str) and re.fullmatch(r'[0-9]+', i) for i in b5_ids) and
         not set(b5_ids) & set(ids), 'B5/diagnostic identities ambiguous')
    reservations = a['other_reservations']
    need(isinstance(reservations, list) and len(reservations) <= 32, 'reservation list')
    seen = {a['reservation_identity']}
    for r in reservations:
        need(set(r) == {'identity', 'job_id', 'node_hours', 'gpu_hours', 'submitted'}, 'reservation schema')
        need(isinstance(r['identity'], str) and r['identity'] and r['identity'] not in seen, 'duplicate reservation')
        seen.add(r['identity'])
        # This attempt admits no other pending diagnostic jobs. Future reservations remain unsubmitted.
        need(r['job_id'] is None and r['submitted'] is False and
             finite(r['node_hours']) and finite(r['gpu_hours']), 'unresolved other admission')
    need(any(r == template['other_reservations'][0] for r in reservations), 'panel ceiling missing')
    budget([], [(r['node_hours'], r['gpu_hours']) for r in reservations])


def fresh(records, mono=None, wall=None):
    mono = time.monotonic() if mono is None else mono
    wall = time.time() if wall is None else wall
    need(records and finite(mono) and finite(wall), 'no observations')
    for r in records:
        values = [r.get(k) for k in ('start_monotonic', 'end_monotonic', 'start_unix', 'end_unix')]
        need(all(finite(x) for x in values), 'invalid observation timing')
        sm, em, sw, ew = values
        need(0 <= sm <= em <= mono and mono-sm <= 300 and 0 <= sw <= ew <= wall and
             abs((ew-sw)-(em-sm)) <= 5, 'stale/future/clock-inconsistent observation')
        need(r.get('exit_code') == 0 and not r.get('timed_out') and
             r.get('stderr') == '' and bool(r.get('stdout', '').strip()), 'failed/incomplete observation')


def query(control, label, argv, timeout=15):
    """Bounded binary capture, Python 3.6. Any capture failure kills and reaps."""
    limit = 16 * 1024 * 1024
    r = dict(command=argv, start_monotonic=time.monotonic(), start_unix=time.time(),
             timed_out=False, stdout='', stderr='', exit_code=None, aggregate_limit_bytes=limit)
    p = None
    captured = {'stdout': bytearray(), 'stderr': bytearray()}
    total = 0
    try:
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, env={'PATH':'/usr/bin:/bin', 'LC_ALL':'C', 'TZ':'America/Chicago',
                                         'HOME':os.environ.get('HOME', '/u/wcyim')})
        deadline = r['start_monotonic'] + timeout
        with selectors.DefaultSelector() as selector:
            for name, stream in (('stdout', p.stdout), ('stderr', p.stderr)):
                selector.register(stream, selectors.EVENT_READ, name)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(argv, timeout)
                for key, _ in selector.select(remaining):
                    block = os.read(key.fd, min(65536, limit - total + 1))
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    room = limit - total
                    captured[key.data].extend(block[:room])
                    total += min(len(block), room)
                    need(len(block) <= room, 'aggregate output exceeds reviewed byte bound')
            p.wait(timeout=max(0, deadline - time.monotonic()))
        for name in captured:
            r[name] = captured[name].decode('utf-8')  # malformed text is not admitted
        r['exit_code'] = p.returncode
    except BaseException as e:
        r['error'] = repr(e)
        r['timed_out'] = isinstance(e, subprocess.TimeoutExpired)
        if p is not None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as cleanup:
                r['kill_error'] = repr(cleanup)
            try:
                p.wait(timeout=2)
            except Exception as cleanup:
                r['reap_error'] = repr(cleanup)
            r['exit_code'] = p.returncode
        # Bounded prefixes are evidence only, never an admissible truncated response.
        for name in captured:
            r[name] = captured[name].decode('utf-8', errors='replace')
    finally:
        if p is not None:
            for stream in (p.stdout, p.stderr):
                if stream is not None:
                    stream.close()
        r.update(end_monotonic=time.monotonic(), end_unix=time.time(), captured_bytes=total)
        r['stdout_sha256'] = hashlib.sha256(r['stdout'].encode()).hexdigest()
        r['stderr_sha256'] = hashlib.sha256(r['stderr'].encode()).hexdigest()
        with (control/(label+'.json')).open('x') as f:
            json.dump(r, f, sort_keys=True, allow_nan=False); f.write('\n')
    need('error' not in r, 'query capture failed: '+r.get('error', ''))
    fresh([r])
    return r


def budget(spent, reserved):
    need(all(finite(x) for pair in [*spent, *reserved] for x in pair), 'invalid accounting number')
    n = sum(p[0] for p in spent+reserved)+0.5
    g = sum(p[1] for p in spent+reserved)+2
    need(n <= 5 and g <= 20, 'cumulative node OR GPU cap exceeded')
    return n, g


def reserve(balance_hours, b5):
    need(finite(balance_hours) and all(finite(x) for x in b5), 'unknown finance reserve')
    required = sum(b5)+20+20
    need(balance_hours >= required, 'B5 plus entire diagnostic envelope plus operational margin unavailable')
    return required


def balance(text):
    lines = text.splitlines()
    need(lines.count("Project Summary for User 'wcyim':") == 1 and
         lines.count('Account                        Balance(Hours)   Deposited(Hours)  Project') == 1,
         'ambiguous account Balance(Hours) format')
    rows = [line.split() for line in lines if line.startswith(ACCOUNT)]
    need(len(rows) == 1 and len(rows[0]) >= 4 and all(re.fullmatch(r'[0-9]+', x) for x in rows[0][1:3]),
         'missing/ambiguous account balance')
    return int(rows[0][1])


def tres(text):
    if text in ('', '(null)'):
        return {}
    pairs = [p.split('=', 1) for p in text.split(',')]
    need(all(len(p) == 2 for p in pairs) and len({p[0] for p in pairs}) == len(pairs), 'invalid TRES')
    return dict(pairs)


def integer(text):
    need(isinstance(text, str) and re.fullmatch(r'[0-9]+', text), 'missing/noninteger resource')
    return int(text)


def gpu(t):
    n = integer(t.get('gres/gpu', ''))
    need(n > 0 and integer(t.get('billing', '')) == n*1000, 'billing units not GPU-consistent')
    return n


def seconds(text):
    m = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d\d):(\d\d)', text)
    need(m is not None, 'unbounded/unknown wall limit')
    days, h, m, s = [int(v or 0) for v in m.groups()]
    need(m < 60 and s < 60, 'bad wall clock')
    return days*86400+h*3600+m*60+s


def fields(text):
    matches = list(re.finditer(r'(?:^| )([A-Za-z][A-Za-z0-9_/:]*)=', text))
    out = {}
    for i, m in enumerate(matches):
        k = 'CPUsPerTask' if m[1] == 'CPUs/Task' else m[1]
        need(k not in out, 'duplicate scheduler field')
        out[k] = text[m.end():matches[i+1].start() if i+1 < len(matches) else len(text)].strip()
    return out


def accounting(text):
    rows = {}
    for line in text.splitlines():
        cols = line.split('|')
        need(len(cols) == len(KEYS), 'truncated accounting response')
        r = dict(zip(KEYS, cols)); ident = r['id']
        if '.' in ident:
            need(re.fullmatch(r'[0-9]+\.(?:batch|extern|[0-9]+)', ident), 'unknown accounting step')
            continue
        need(re.fullmatch(r'[0-9]+', ident) and ident not in rows, 'duplicate/array/heterogeneous allocation')
        r['state'] = r['state'].split(' by ')[0]
        rows[ident] = r
    need(rows, 'empty accounting')
    return rows


def timestamp(text, now, optional=False):
    if optional and text in ('Unknown', 'None', 'N/A'):
        return None
    need(re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d', text), 'unknown accounting timestamp')
    # Single-threaded host wrapper, Python 3.6 compatible. No guessed fixed UTC offset.
    previous = os.environ.get('TZ')
    try:
        os.environ['TZ'] = 'America/Chicago'; time.tzset()
        parsed = time.strptime(text, '%Y-%m-%dT%H:%M:%S')
        candidates = {time.mktime((*parsed[:8], dst)) for dst in (0, 1)}
        candidates = {v for v in candidates if time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(v)) == text}
        need(len(candidates) == 1, 'ambiguous/nonexistent local accounting time')
        value = candidates.pop()
    finally:
        if previous is None: os.environ.pop('TZ', None)
        else: os.environ['TZ'] = previous
        time.tzset()
    need(value <= now, 'future accounting timestamp')
    return value


def node_capacity(text, node):
    need(len(text.splitlines()) == 1, 'ambiguous node capacity response')
    record = fields(text.strip())
    need(record.get('NodeName') == node and re.fullmatch(r'gh\d{3}', node), 'node capacity identity')
    capacity = integer(record.get('CPUTot', ''))
    need(capacity >= 64, 'node capacity too small')
    return record


def allocation(j, current, now, capacity=None):
    need(j.get('JobId') == current and j.get('UserId') == 'wcyim(97434)' and
         j.get('Account') == ACCOUNT and j.get('QOS') == ACCOUNT and j.get('Partition') == 'ghx4' and
         j.get('JobState') == 'RUNNING' and j.get('OverSubscribe') == 'NO' and
         j.get('NumNodes') == '1' and j.get('CPUsPerTask') == '64' and j.get('NumTasks') == '1' and
         j.get('Requeue') == '0' and j.get('Restarts') == '0' and
         j.get('Command') == NAMESPACE+'/p1.sbatch' and j.get('WorkDir') == NAMESPACE,
         'actual allocation identity/geometry/restart/namespace mismatch')
    need(0 < seconds(j.get('TimeLimit', '')) <= 1800, 'actual time limit')
    need(0 <= seconds(j.get('RunTime', '')) <= 1800, 'elapsed allocation')
    t = tres(j.get('AllocTRES', ''))
    req = tres(j.get('ReqTRES', ''))
    need(gpu(req) == 4 and req.get('node') == '1' and req.get('cpu') == '64', 'requested TRES mismatch')
    cpus = integer(j.get('NumCPUs', ''))
    need(gpu(t) == 4 and t.get('node') == '1' and t.get('cpu') == str(cpus), 'allocated TRES mismatch')
    need(cpus == 64 or capacity is not None and capacity.get('NodeName') == j.get('NodeList') and
         cpus == integer(capacity.get('CPUTot', '')) and cpus >= 64,
         'allocated CPUs not requested count or trusted exclusive node capacity')
    node = j.get('NodeList', '')
    need(re.fullmatch(r'gh\d{3}', node) and node not in {'gh062', 'gh093'}, 'placement/exclusion mismatch')
    need(j.get('ExcNodeList') in ('gh[062,093]', 'gh062,gh093'), 'exclusion policy changed')
    timestamp(j.get('SubmitTime', ''), now); timestamp(j.get('StartTime', ''), now)
    return node


def reconcile(a, current, jobs, rows, now, capacity=None):
    need(current in jobs and current in rows, 'current allocation missing/lagging')
    node = allocation(jobs[current], current, now, capacity)
    known = set(a['registered_prior_ids'])
    need(current not in known and known <= set(rows), 'registered old history missing')
    b5 = []; spent = []
    for ident, j in jobs.items():
        if ident == current:
            continue
        # Only account/owner records are passed here, all must be known B5 or terminal history.
        if j.get('JobState') in TERMINAL:
            need(ident in rows and (ident in known or ident in a['b5_registered_ids']),
                 'unreconciled terminal scheduler allocation')
            continue
        need(j.get('JobName') == 'b5-seed' and j.get('UserId') == 'wcyim(97434)' and
             j.get('Account') == ACCOUNT and j.get('Partition') == 'ghx4' and
             j.get('Command') == '/work/nvme/bilv/wcyim/transgenic/repo_seeds/deploy/deltaai/train_b5.slurm',
             'other diagnostic/unknown active admission')
        need(j.get('JobState') in ('PENDING', 'RUNNING'), 'B5 lifecycle ambiguous')
        need(node not in expand_nodes(j.get('NodeList', '')), 'B5 shares current node')
        resources = tres(j.get('AllocTRES', '')) if j['JobState'] == 'RUNNING' else tres(j.get('ReqTRES', ''))
        wall = seconds(j.get('TimeLimit', ''))
        need(wall > 0, 'B5 maximum wall missing')
        b5.append(wall*gpu(resources)/3600)
    for ident, r in rows.items():
        need(r['user'] == USER and r['uid'] == str(UID) and r['account'] == ACCOUNT,
             'unexpected user/account campaign allocation')
        timestamp(r['submit'], now)
        timestamp(r['start'], now, optional=r['state'] == 'PENDING' or r['elapsed'] == '0')
        timestamp(r['end'], now, optional=r['state'] not in TERMINAL)
        elapsed = integer(r['elapsed']); alloc = tres(r['alloc'])
        if ident == current:
            need(r['state'] == 'RUNNING' and gpu(alloc) == 4 and r['nodes'] == '1' and
                 r['cpus'] == jobs[current]['NumCPUs'] and alloc.get('cpu') == r['cpus'] and
                 tres(r['req']).get('cpu') == '64',
                 'current accounting disagrees')
            continue  # Full 0.5/2 reservation below already includes startup elapsed.
        if r['name'] == 'b5-seed':
            need(ident in a['b5_registered_ids'] or ident in jobs and
                 jobs[ident].get('Command') == '/work/nvme/bilv/wcyim/transgenic/repo_seeds/deploy/deltaai/train_b5.slurm',
                 'unregistered B5 history cannot exempt a diagnostic allocation')
            if r['state'] not in TERMINAL:
                need(ident in jobs, 'B5 accounting/live conflict')
            continue
        need(ident in known and r['state'] in TERMINAL, 'unknown or active MVP diagnostic allocation')
        need(ident not in jobs or jobs[ident].get('JobState') == r['state'], 'old live/accounting conflict')
        if ident == '3163986':
            need(r['state'] == 'FAILED' and r.get('exit') == '1:0' and elapsed == 51 and
                 r['nodes'] == '1' and r['cpus'] == '288' and alloc.get('cpu') == '288' and
                 alloc.get('node') == '1' and gpu(alloc) == 4 and
                 tres(r['req']).get('cpu') == '64' and r['partition'] == 'ghx4' and
                 r['placement'] == 'gh077' and r['start'] == '2026-09-17T18:47:46' and
                 r['end'] == '2026-09-17T18:48:37', 'old3163986 not exact reviewed terminal failure')
        if ident == '3163545':
            need(r['state'] == 'CANCELLED' and r.get('exit') == '0:0' and elapsed == 0 and
                 not alloc and r['cpus'] == '0' and r['nodes'] == '0', 'old3163545 not cancelled zero')
        if elapsed:
            ng = gpu(alloc); nn = integer(alloc.get('node', ''))
            need(nn > 0 and nn == integer(r['nodes']), 'allocation node accounting mismatch')
            spent.append((elapsed*nn/3600, elapsed*ng/3600))
        else:
            need(not alloc, 'zero-time allocation requires controller reconciliation')
    totals = budget(spent, [(r['node_hours'], r['gpu_hours']) for r in a['other_reservations']])
    return dict(node_hours=totals[0], gpu_hours=totals[1], b5_max_gpu_hours=b5,
                current_reserved_node_hours=0.5, current_reserved_gpu_hours=2, node=node)


def expand_nodes(text):
    if text in ('', '(null)', 'None', 'None assigned'):
        return set()
    if re.fullmatch(r'gh\d{3}(?:,gh\d{3})*', text):
        return set(text.split(','))
    m = re.fullmatch(r'gh\[([0-9,\-]+)\]', text)
    need(m is not None, 'ambiguous node list')
    out = set()
    for part in m[1].split(','):
        bounds = part.split('-'); lo = int(bounds[0]); hi = int(bounds[-1])
        need(len(bounds) <= 2 and 1 <= lo <= hi <= 150, 'invalid node range')
        out.update('gh%03d'%i for i in range(lo, hi+1))
    return out


def live(control, a, current):
    """No observation data can enter here: all records originate in these commands."""
    records = []
    def q(name, argv):
        r = query(control, name, argv); records.append(r); return r['stdout']
    try:
        need(os.getuid() == UID, 'wrong compute UID')
        script_meta = q('account-script-metadata', ['/usr/bin/stat', '-c', '%u|%a|%s', '/sw/user/scripts/accounts'])
        need(script_meta.strip() == '67674|755|8705', 'account script ownership/mode/size changed')
        script_hash = q('account-script-identity', ['/usr/bin/sha256sum', '/sw/user/scripts/accounts'])
        need(script_hash == ACCOUNTS_SHA+'  /sw/user/scripts/accounts\n',
             'unreviewed account script/unit conversion')
        config = q('cluster-identity', ['/usr/bin/scontrol', 'show', 'config'])
        need(re.findall(r'^ClusterName\s*=\s*(\S+)\s*$', config, re.M) == [CLUSTER],
             'actual scheduler cluster identity mismatch')
        assoc = q('association', ['/usr/bin/sacctmgr', '-nP', 'show', 'assoc', 'where', 'user=wcyim',
                                  'format=Cluster,Account,User,QOS'])
        need(assoc.splitlines().count(CLUSTER+'|'+ACCOUNT+'|'+USER+'|'+ACCOUNT) == 1,
             'missing account/cluster association')
        queue = q('queue-before', ['/usr/bin/squeue', '-h', '-a', '-t', 'all', '-o', '%i|%j|%u|%a|%T|%N'])
        def queue_rows(text):
            out = {}
            for line in text.splitlines():
                v = line.split('|'); need(len(v) == 6 and v[0] not in out, 'ambiguous queue response')
                out[v[0]] = v
            return out
        queued = queue_rows(queue)
        relevant = {i:v for i,v in queued.items() if v[2] == USER or v[3] == ACCOUNT}
        need(current in relevant and len(relevant) <= 32, 'missing current/too many admissions')
        jobs = {}
        for ident in sorted(relevant):
            need(re.fullmatch(r'[0-9]+', ident), 'array/heterogeneous admission unsupported')
            raw = q('job-'+ident, ['/usr/bin/scontrol', '-o', 'show', 'job', ident])
            need(len(raw.splitlines()) == 1, 'incomplete/ambiguous scheduler job')
            jobs[ident] = fields(raw.strip())
        raw = q('campaign', ['/usr/bin/sacct', '-nPX', '--duplicates', '-u', USER, '-S', START, '--format='+FIELDS])
        rows = accounting(raw)
        node = jobs[current].get('NodeList', '')
        need(re.fullmatch(r'gh\d{3}', node), 'invalid current node')
        capacity = node_capacity(q('node-capacity', ['/usr/bin/scontrol', '-o', 'show', 'node', node]), node)
        decision = reconcile(a, current, jobs, rows, time.time(), capacity)
        factor = q('qos-factor', ['/usr/bin/sacctmgr', '-nP', 'show', 'qos', 'where', 'name='+ACCOUNT,
                                  'format=Name,UsageFactor,Flags'])
        need(factor.strip() == ACCOUNT+'|1.000000|NoDecay,UsageFactorSafe',
             'account usage factor/flags changed; finance units ambiguous')
        qos = q('qos-units', ['/usr/bin/scontrol', 'show', 'assoc', 'qos='+ACCOUNT, 'flags=qos', '-o'])
        matches = re.findall(r'GrpTRESMins=\S*?billing=(\d+)\((\d+)\)\S*?gres/gpu=N\((\d+)\)', qos)
        need(len(matches) == 1 and 'QOS='+ACCOUNT+'(' in qos, 'ambiguous account finance units')
        allocated, used, gpu_minutes = map(int, matches[0])
        # Site script divides expanded billing minutes by 1000 and 60. Existing
        # GPU-minute counters are integer-rounded, so at most one minute differs.
        need(abs(used/1000-gpu_minutes) <= 1, 'billing history not GPU-hour consistent')
        bal = balance(q('account-balance', ['/sw/user/scripts/accounts']))
        need(abs(bal-math.floor(allocated/60000-used/60000)) <= 1, 'finance balance inconsistent')
        decision['balance_hours_lower_bound'] = max(0, bal-1)  # report integer rounding conservatively
        decision['operational_reserve_gpu_hours'] = reserve(max(0, bal-1), decision['b5_max_gpu_hours'])
        # Second observations are race checks, never retries after a failed command.
        stable = ('JobId', 'UserId', 'Account', 'QOS', 'Partition', 'JobState', 'NumNodes',
                  'NumCPUs', 'CPUsPerTask', 'NumTasks', 'TimeLimit', 'Requeue', 'Restarts', 'Command', 'WorkDir',
                  'NodeList', 'AllocTRES', 'ReqTRES', 'OverSubscribe', 'ExcNodeList', 'StartTime')
        for ident, before_job in jobs.items():
            after_job = fields(q('job-final-'+ident, ['/usr/bin/scontrol', '-o', 'show', 'job', ident]).strip())
            need(all(before_job.get(k) == after_job.get(k) for k in stable), 'allocation changed during admission')
        after = queue_rows(q('queue-after', ['/usr/bin/squeue', '-h', '-a', '-t', 'all', '-o', '%i|%j|%u|%a|%T|%N']))
        after_relevant = {i:v for i,v in after.items() if v[2] == USER or v[3] == ACCOUNT}
        need(relevant == after_relevant, 'admission snapshot changed: no retry')
        for ident, row in after.items():
            if ident != current and row[4] not in TERMINAL:
                need(decision['node'] not in expand_nodes(row[5]), 'another allocation shares node')
        final_capacity = node_capacity(q('node-capacity-final', ['/usr/bin/scontrol', '-o', 'show', 'node', node]), node)
        need(capacity['CPUTot'] == final_capacity['CPUTot'], 'node capacity changed during admission')
        fresh(records)
        decision.update(admitted=True, decided_at_unix=time.time(), decided_at_monotonic=time.monotonic(),
                        reservation_identity=a['reservation_identity'],
                        limitation='Read-only snapshots do not guarantee future activity; serialized controller reservation remains required.')
    except Exception as e:
        decision = dict(admitted=False, reason=str(e), decided_at_unix=time.time(),
                        reservation_identity=a.get('reservation_identity'))
        with (control/'admission-decision.json').open('x') as f:
            json.dump(decision, f, sort_keys=True); f.write('\n')
        raise
    with (control/'admission-decision.json').open('x') as f:
        json.dump(decision, f, sort_keys=True); f.write('\n')
    return decision


if __name__ == '__main__':
    # Print an unapproved template only. Never manufacture an authorization.
    print(json.dumps(example(), indent=2, sort_keys=True))
