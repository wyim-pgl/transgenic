"""Bounded exact-job read-only observer; all writes are local evidence."""
import pathlib, subprocess, time
D=pathlib.Path(__file__).resolve().parent
start=time.monotonic()
command='date -Is; sacct -j 6165628 --format=JobID,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS,NodeList -P; squeue -j 6165628 -h -o "%i|%T|%M|%R"'
while time.monotonic()-start < 4200:
    p=subprocess.run(['ssh','-o','ConnectTimeout=20','pronghorn',command],capture_output=True,text=True,timeout=50)
    text=p.stdout+'\nSTDERR:\n'+p.stderr
    with (D/'observations.txt').open('a') as f: f.write(text+'\n')
    print(text,flush=True)
    states={r.split('|')[0]:r.split('|')[1] for r in p.stdout.splitlines() if r.startswith('6165628') and '|' in r}
    if p.returncode==0 and all(states.get(k,'RUNNING').split()[0] in {'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'} for k in ('6165628','6165628.batch')):
        print('TERMINAL_ALLOCATION_AND_BATCH',flush=True); break
    remaining=4200-(time.monotonic()-start)
    if remaining<120: break
    time.sleep(120)
else:
    print('OBSERVATION_BOUND_REACHED',flush=True)
