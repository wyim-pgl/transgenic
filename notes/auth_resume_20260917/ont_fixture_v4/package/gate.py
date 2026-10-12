#!/usr/bin/env python3
"""External CPU orientation gate. Not full-panel evidence or decision authorization.

Only pilot() writes; it never submits jobs. Reference authority is a fixed Git
blob's SHA256, not user-supplied expected MD5. Frozen driver labels are untouched.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
COMMIT = 'b660bf1e7dc48287ca34a0d6e3aeb53d45f599b6'
AUTHORITY = '92d8020ee79eeedd7f1d50167b1c8dc94700fb16:data/manifests/b5_species_v1.tsv'
AUTHORITY_SHA256 = '3ca7a3b9a5793f62d25f503c93cfd7167eb17f7c5b90de7830e6a8777117f783'
DATASET = 'ont/Athaliana_FLIC_PRJNA1087576'
RUNS = {'SRR28341943','SRR28341944','SRR28341945'}
ROLE_COLUMNS = ('dataset run species genotype_stratum instrument data_type expected_files expected_reads source_checksum source_checksum_authority local_fa_md5 role basis note').split()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(path, algorithm='sha256'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def identity(path):
    p=Path(path).resolve(strict=True)
    return dict(path=str(p),size=p.stat().st_size,sha256=digest(p))

def read_binding(root, run, reads):
    root=Path(root).resolve(strict=True)
    roles=root/'evidence/DATASET_ROLES.longread_v1.tsv'
    require(run in RUNS,'ONT FLIC run required; HiFi policy not selected')
    with roles.open() as f:
        reader=csv.DictReader(f,delimiter='\t')
        require(reader.fieldnames==ROLE_COLUMNS,'canonical roles schema mismatch')
        rows=list(reader)
    selected=[r for r in rows if r['run']==run]
    require(len(selected)==1,'unique canonical run role required')
    row=selected[0]
    require((row['dataset'],row['species'],row['genotype_stratum'],row['role']) == (DATASET,'Athaliana','reference','b1_validation_only'),'canonical role mismatch')
    expected=root/'evidence'/DATASET/(run+'.fa.gz')
    require(expected.resolve(strict=True)==expected,'canonical reads redirected by symlink')
    require(Path(reads).absolute()==expected,'wrong consumed read pathname')
    require(digest(expected,'md5')==row['local_fa_md5'],'fresh consumed FASTA MD5 mismatch')
    return dict(role=row,roles=identity(roles),reads=identity(expected),reads_md5=row['local_fa_md5'])

def reference_binding(blob, fasta, gff):
    require(hashlib.sha256(blob).hexdigest()==AUTHORITY_SHA256,'not fixed Git92 authority blob')
    rows=[r for r in csv.DictReader(io.StringIO(blob.decode()),delimiter='\t') if r['species_id']=='Athaliana']
    require(len(rows)==1,'unique Athaliana reference authority required')
    files={}
    for name,path in [('fasta',fasta),('gff',gff)]:
        require(digest(path,'md5')==rows[0][name+'_md5'],'frozen '+name+' bytes mismatch')
        files[name]=dict(identity(path),md5=rows[0][name+'_md5'])
    return dict(authority=AUTHORITY,manifest_sha256=AUTHORITY_SHA256,files=files)

def load_driver(source=None):
    source=Path(source or HERE/'frozen').resolve()
    pins=json.loads((HERE/'source_pins.json').read_text())
    require(pins['commit']==COMMIT,'wrong code commit')
    for name,sha in pins['files'].items():
        require(digest(source/name)==sha,'source mismatch: '+name)
    path=source/'revision/scripts/mvp/mvp_align_panel_longreads.py'
    spec=importlib.util.spec_from_file_location('frozen_longread_driver',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    require(Path(module.w.__file__).resolve()==source/'src/transgenic/evidence/mvp_windows.py','wrong imported windows source')
    return module

def inventory(out):
    out=Path(out)
    result={}
    for p in sorted(out.rglob('*')):
        require(not p.is_symlink(),'output symlink forbidden')
        if p.is_file() and p not in (out/'receipt.json',out/'COMPLETE'):
            result[str(p.relative_to(out))]=dict(size=p.stat().st_size,sha256=digest(p))
    return result

def verify_inventory(out, expected):
    require(inventory(out)==expected,'full output inventory/hash mismatch')

def verify_orientation(out,key,run,driver):
    out=Path(out)
    lines=(out/'DONE').read_text().splitlines()
    pairs=[line.split('=',1) for line in lines]
    require(all(len(x)==2 for x in pairs) and len({x[0] for x in pairs})==len(pairs),'malformed/duplicate DONE keys')
    require(dict(pairs).get('run')==run,'DONE run mismatch')
    return driver.read_audit(out,key,run)

def verify_alignment(args, outer_receipt):
    """Postflight consistency check; outer receipt must come from reviewed controller.

    Does not certify command execution authenticity, coverage policy or scores.
    An outer immutable receipt binds exit status, stdout and recursive inventory.
    """
    driver=load_driver(args.source)
    binding=read_binding(args.root,args.run,args.reads)
    reference=reference_binding(Path(args.authority).read_bytes(),args.fasta,args.gff)
    out=driver.output_directory(args.root,args.run)
    require(outer_receipt['exit_code']==0,'alignment driver nonzero exit')
    verify_inventory(out,outer_receipt['outputs'])
    report=json.loads((out/'alignment.json').read_text())
    require(json.loads(Path(outer_receipt['stdout']['path']).read_text())==report,'driver stdout/report mismatch')
    require(identity(outer_receipt['stdout']['path'])==outer_receipt['stdout'],'driver stdout changed')
    require(report['status']=='PREPARATORY_COMMANDS_COMPLETED' and report['reference_identity_status']=='UNVERIFIED' and report['training_eligible'] is False,'frozen driver boundary altered')
    require(report['role']==binding['role'] and report['run']==args.run and report['source']=='ONT','alignment run/role mismatch')
    expected={str(Path(p).resolve()):digest(p,'md5') for p in (args.reads,args.fasta,args.gff,args.panels,binding['roles']['path'])}
    require(report['inputs_md5']==expected,'alignment consumed-input inventory mismatch')
    require(set(report['outputs_md5'])=={'commands.json','commands.stderr','pass1.paf','panel_reads.txt','panel_reads.fa','panel.sam','panel.bam','panel.bam.bai','junctions.tsv','junctions.qc.json'},'missing alignment output inventory')
    for name,md5 in report['outputs_md5'].items():
        require(Path(name).name==name and digest(out/name,'md5')==md5,'alignment output mismatch')
    key=':'.join([binding['reads_md5'],reference['files']['fasta']['md5'],reference['files']['gff']['md5'],digest(Path(args.source)/'revision/scripts/67_orientation_audit.py','md5')])
    audit=verify_orientation(out/'orientation',key,args.run,driver)
    require(report['use_uf'] is audit['use_uf'],'orientation/alignment uf mismatch')
    commands=json.loads((out/'commands.json').read_text())
    require(report['commands']==commands and len(commands)==8,'driver command inventory mismatch')
    require(commands[2]==['minimap2','-t',str(args.threads),'-x','splice',str(Path(args.fasta).resolve()),str(Path(args.reads).resolve())],'pass1 must consume complete canonical run, not sample')
    require(commands[4]==list(map(str,driver.pass2_command('ONT',audit['use_uf'],Path(args.fasta).resolve(),out/'panel_reads.fa',args.threads))),'pass2 frozen command mismatch')
    expected_tail=[['samtools','sort','-o',str(out/'panel.bam'),str(out/'panel.sam')],['samtools','index',str(out/'panel.bam')], [commands[7][0],str(Path(args.source).resolve()/'revision/scripts/mvp/mvp_junction_table.py'),'--alignment','ONT:uf='+str(audit['use_uf']).lower()+':'+str(out/'panel.bam'),'--panels',str(Path(args.panels).resolve()),'--out',str(out/'junctions.tsv'),'--qc-out',str(out/'junctions.qc.json')]]
    require(commands[5:]==expected_tail,'alignment-to-junction command linkage mismatch')
    require(commands[3]==['seqkit','grep','-f',str(out/'panel_reads.txt'),str(Path(args.reads).resolve())],'selected reads must derive from canonical run')
    qc=json.loads((out/'junctions.qc.json').read_text())
    require(qc['thresholds']=={'ONT':3,'HiFi':2} and set(qc['per_input'])=={'ONT:'+str(out/'panel.bam')},'junction alignment/threshold mismatch')
    require(qc['scorer']=='59_evidence_support.py (sha256 '+digest(Path(args.source)/'revision/scripts/59_evidence_support.py')+')','junction scorer source mismatch')
    require(qc['panel_sha256']==driver.w.PANEL_SHA256==report['panel_sha256'],'panel mismatch')
    return dict(reference_gate=reference,read_binding=binding,orientation_key=key,alignment=identity(out/'alignment.json'),junctions=identity(out/'junctions.tsv'),qc=identity(out/'junctions.qc.json'),structural_status='UNASSESSABLE_PENDING_FULL_INVENTORY_AND_POLICY',decision=None)

def require_final_coverage(receipt):
    raise ValueError('UNASSESSABLE: full run/locus evidence and HiFi-versus-ONT policy are not certified; missing is not zero')

def verify_publication(out):
    out=Path(out)
    require(not (out/'FAILED.json').exists(),'failed pilot cannot be complete')
    require((out/'COMPLETE').read_text()==digest(out/'receipt.json')+'\n','publication not complete')
    receipt=json.loads((out/'receipt.json').read_text())
    require(receipt['scope']=='orientation_sample' and receipt['structural_status']=='UNASSESSABLE' and receipt['training_eligible'] is False,'pilot scope drift')
    require([c['stage'] for c in receipt['commands']]==['version_minimap2','version_samtools','version_seqkit','sample','sample_alignment','orientation_audit'],'incomplete stage inventory')
    require(all(c['exit_code'] in ((0,4) if c['stage']=='orientation_audit' else (0,)) for c in receipt['commands']),'external command failed')
    for c in receipt['commands']:
        require(c['argv'] and c['stdout'] in receipt['outputs'] and c['stderr'] in receipt['outputs'],'missing argv/stdout/stderr binding')
    driver=load_driver(receipt['source'])
    audit=verify_orientation(out/'orientation',receipt['orientation_key'],receipt['run'],driver)
    require(audit==receipt['audit'],'orientation publication disagreement')
    actual=read_binding(receipt['root'],receipt['run'],receipt['read_binding']['reads']['path'])
    require(actual==receipt['read_binding'],'canonical roles/read binding changed')
    ref=receipt['reference_gate']['files']
    require(reference_binding(Path(receipt['authority_file']).read_bytes(),ref['fasta']['path'],ref['gff']['path'])==receipt['reference_gate'],'reference authority binding changed')
    verify_inventory(out,receipt['outputs'])
    for entry in receipt['inputs']:
        require(identity(entry['path'])==entry,'input changed since pilot')
    return receipt

def pilot(args):
    require(os.environ.get('SLURM_JOB_ID'),'actual input hashing/sampling must run on compute')
    require(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'first pilot requires four CPUs')
    driver=load_driver(args.source)
    binding=read_binding(args.root,args.run,args.reads)
    reference=reference_binding(Path(args.authority).read_bytes(),args.fasta,args.gff)
    panel=driver.w.load_panel(args.panels)
    require(len(panel)==50,'full frozen 50-locus inventory required')
    out=Path(args.out).absolute()
    require(not out.is_symlink(),'symlink output forbidden')
    out.mkdir(parents=True,exist_ok=False)
    commands=[]
    inputs=[identity(p) for p in (args.reads,args.fasta,args.gff,args.panels,args.authority,__file__,HERE/'source_pins.json')]
    inputs.append(binding['roles'])
    source=Path(args.source).resolve()
    inputs.extend(identity(source/p) for p in json.loads((HERE/'source_pins.json').read_text())['files'])
    start=time.monotonic()
    def run(argv,stem):
        with (out/(stem+'.stdout')).open('x') as stdout,(out/(stem+'.stderr')).open('x') as stderr:
            result=subprocess.run(argv,stdout=stdout,stderr=stderr)
        commands.append(dict(stage=stem,argv=argv,exit_code=result.returncode,stdout=stem+'.stdout',stderr=stem+'.stderr'))
        (out/'commands.json').write_text(json.dumps(commands,indent=2))
        require(result.returncode in ((0,4) if stem=='orientation_audit' else (0,)),'failed command '+stem)
    try:
        binary=Path(args.bin).resolve()
        for tool in ('minimap2','samtools','seqkit'):
            path=binary/tool;inputs.append(identity(path))
            run([str(path),'version' if tool=='seqkit' else '--version'],'version_'+tool)
        scripts=source/'revision/scripts'
        run([sys.executable,str(scripts/'68_sample_fasta.py'),'--fasta',args.reads,'--out',str(out/'sample.fa'),'--n','10000','--md5',binding['reads_md5']],'sample')
        sample=json.loads((out/'sample.stdout').read_text())
        require(sample['sampled']==10000 and sample['run_records']==int(binding['role']['expected_reads']),'sample/run record inventory mismatch')
        cmd=driver.pass2_command('ONT',False,str(Path(args.fasta).resolve()),str(out/'sample.fa'),4)
        cmd[0]=str(binary/'minimap2')
        run(cmd,'sample_alignment')
        audit_dir=out/'orientation';audit_dir.mkdir()
        key=':'.join([binding['reads_md5'],reference['files']['fasta']['md5'],reference['files']['gff']['md5'],digest(scripts/'67_orientation_audit.py','md5')])
        run([sys.executable,str(scripts/'67_orientation_audit.py'),'--sam',str(out/'sample_alignment.stdout'),'--gff',str(Path(args.gff).resolve()),'--out',str(audit_dir/'audit.json'),'--run',args.run,'--species','Athaliana','--dataset',DATASET],'orientation_audit')
        with (audit_dir/'DONE').open('x') as f:
            f.write('key='+key+'\nrun='+args.run+'\naudit_md5='+digest(audit_dir/'audit.json','md5')+'\n')
        audit=verify_orientation(audit_dir,key,args.run,driver)
        for entry in inputs:require(identity(entry['path'])==entry,'input mutated')
        receipt=dict(schema=1,scope='orientation_sample',structural_status='UNASSESSABLE',training_eligible=False,decision=None,run=args.run,root=str(Path(args.root).resolve()),source=str(source),authority_file=str(Path(args.authority).resolve()),source_commit=COMMIT,read_binding=binding,reference_gate=reference,orientation_key=key,audit=audit,inputs=inputs,commands=commands,outputs=inventory(out),job=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),wall_seconds=time.monotonic()-start)
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2))
        with (out/'COMPLETE').open('x') as f:f.write(digest(out/'receipt.json')+'\n')
        verify_publication(out)
        return receipt
    except Exception as error:
        (out/'FAILED.json').write_text(json.dumps(dict(error=str(error),commands=commands)))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','run','reads','fasta','gff','panels','authority','source','bin','out'):p.add_argument('--'+name,required=True)
    print(json.dumps(pilot(p.parse_args()),indent=2))
