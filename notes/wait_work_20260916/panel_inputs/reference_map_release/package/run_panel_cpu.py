#!/usr/bin/env python3
"""Bounded CPU observations, no predictions, score, policy change or training."""
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
import authority

SOURCE_COMMIT='9603e20fd4870a9f5b3d5f68e2df8ba0f8a49558'
SOURCE_PIN='ad3e5a7aacbc5d71c3a59d9b1b6f6fd0afeff12efab10b7ed123f734e767380e'
ACCEPTANCE='faab8387cbbe87812c7f3faf5412d19004874a5cc439f5a3d182e2c976a5b51e'
COMPANIONS={'notes/b6_mvp_technical_candidate_20260915.json':ACCEPTANCE,
 'notes/b6_mvp_candidate_inventories_20260915.json':'9f79019679bfdd25b0a2bdac7cae63d01fa6d874be03fa3ef3f81cfd0a880a98',
 'notes/b6_mvp_candidate_receipts_20260915.json':'73204a4ed98624b1567a5eb26fd6b2bbafbb597e0bd4b807685e8c703c807bac'}

def dump(path,data):
    with Path(path).open('x') as f: json.dump(data,f,sort_keys=True,indent=2,allow_nan=False); f.write('\n')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def verify_source(root):
    if sha(root/'source_pins.json')!='5570f32378928ccd627494652988a0ec8cda3c657cff5d63e0c496ec8c97010b': raise ValueError('reviewed release inventory identity differs')
    pins=json.loads((root/'source_pins.json').read_text())
    if pins['commit']!=SOURCE_COMMIT: raise ValueError('source commit differs')
    for name,expected in {**pins['source_files'],**COMPANIONS}.items():
        p=root/name
        if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or sha(p)!=expected: raise ValueError('source/companion mismatch: '+name)
    expected_py={n for n in pins['source_files'] if n.endswith('.py') and n.startswith(('src/transgenic/','revision/scripts/mvp/'))}
    actual_py={str(p.relative_to(root)) for base in ('src/transgenic','revision/scripts/mvp') for p in (root/base).rglob('*.py')}
    if actual_py!=expected_py: raise ValueError('unexpected or missing executable source')
    components={n:sha(root/n) for n in pins['source_files'] if n.startswith(('src/transgenic/','configs/','revision/scripts/mvp/')) or n=='train/train_HyenaTransgenic_MVP.py'}
    # Reviewed pin enumerates all tracked component files, not a caller-selected subset.
    if hashlib.sha256((json.dumps(components,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()!=SOURCE_PIN: raise ValueError('component-source closure mismatch')
    return {'commit':SOURCE_COMMIT,'component_sha256':SOURCE_PIN,'files':pins['source_files'],'companions':COMPANIONS}

def cpu_runtime(root,out):
    dep=root.parent/'cpu_dependency.json'
    if sha(dep)!='cf4d509e8914d6bcf44e6a96a79fe127e4521f7592e4a768a7e2659760ec3435': raise ValueError('CPU wheel receipt changed')
    site=Path(os.environ['MVP_CPU_SITE']).resolve(strict=True)
    if site!=root.parent/'cpu_site': raise ValueError('unexpected CPU site path')
    records=json.loads(dep.read_text())
    for name,digest in records['files'].items():
        if sha(site/name)!=digest: raise ValueError('CPU dependency file mismatch: '+name)
    versions={n:importlib.metadata.version(n) for n in ('numpy','duckdb','torch')}
    if sys.version_info[:2]!=(3,10) or versions!={'numpy':'1.26.4','duckdb':'1.5.5','torch':'1.13.1+cpu'}: raise ValueError('CPU candidate runtime versions differ: '+str(versions))
    # The following are genuine imports, but no torch.cuda/GPU/model operations.
    import numpy,duckdb,torch
    from transgenic.training import mvp_cli_runtime
    import mvp_prepare_panel_inputs as panel
    if panel.ROOT!=root: raise ValueError('producer imported from wrong source root')
    container=os.environ.get('SINGULARITY_CONTAINER')
    if not container or Path(container).resolve()!=Path('/data/gpfs/assoc/pgl/singularity/micromamba.sif').resolve(): raise ValueError('expected CPU container absent')
    dump(out/'cpu_runtime.json',{'python':sys.version,'executable':sys.executable,'executable_sha256':sha(sys.executable),
         'container':{'path':container,'sha256':sha(container)},'packages':versions,'cpu_dependency_receipt_sha256':sha(dep),
         'module_files':{n:{'path':m.__file__,'sha256':sha(m.__file__)} for n,m in [('numpy',numpy),('duckdb',duckdb),('torch',torch)]},
         'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'scope':'CPU_utilities_not_decoder_SIF_certification'})
    return panel

def duplicate_groups(transcripts, producer):
    groups=defaultdict(list)
    for tid,tx in transcripts.items():
        key=producer.structure(tx)
        if key is not None: groups[key].append(tid)
    return [{'structure':key,'transcript_ids':ids} for key,ids in groups.items() if len(ids)>1]

def recheck_audit(payload, report):
    import mvp_produce_counts as p
    p.recheck_files(list(report['reference_identity']['files'].values()))
    if p.frozen_reference(payload['fasta'],payload['reference'])!=report['reference_identity']:
        raise ValueError('reference changed during audit')
    p.recheck_reference_mapping(report['reference_mapping'])


def reference_audit(payload,out, *, defer_publication=False):
    import mvp_produce_counts as p
    w=p.w
    binding=p.frozen_reference(payload['fasta'],payload['reference'])
    dump(out/'reference_identity.json',binding)
    loci=w.load_panel(payload['panels']); lengths=p.fasta_lengths(payload['fasta'])
    report={'scope':'raw_reference_compatibility_observations_not_scoring','reference_identity':binding,
            'panel':p.file_identity(payload['panels']),'fasta_lengths':lengths,'loci':[],
            'duplicate_policy':'REPORT_ONLY_NO_DEDUP_NO_CHANGED_F1_POLICY','scientific_verdict':None}
    # Candidate gates precede the read-only mapping query, including --reference-only.
    from transgenic.training.mvp_cli import validate_candidate_records
    from transgenic.training.mvp_cli_runtime import verify_candidate_artifacts
    c=payload['candidate']
    candidate=validate_candidate_records(c['acceptance'],c['acceptance_sha256'],c['inventory'],c['receipts'])
    accepted=verify_candidate_artifacts(candidate,c['corpus'],c['evidence'],payload['panels'])
    mapping=p.resolve_reference_mapping(loci,c['corpus'],binding,accepted['sha256'],payload['panels'])
    report['reference_mapping']=mapping
    mapped={r['gene_id']:r['gene_id_original'] for r in mapping['rows']}
    restricted=out/'reference.panel.gff3'
    try:
        genes=p.prepare_mapped_reference(loci,payload['reference'],restricted,mapping)
        report['panel_strict_parser']={'ok':True,'genes':len(genes),'transcripts':sum(len(t) for t in genes.values())}
    except Exception as e:
        genes=None; report['panel_strict_parser']={'ok':False,'stage':'mapped_reference_preparation','error':str(e),'type':type(e).__name__}
    if restricted.exists(): report['restricted_reference']=p.file_identity(restricted)
    # Individual files allow every locus failure to be observed even if the joint parser failed.
    for i,locus in enumerate(loci):
        row={'gene_id':locus.gene_id,'gene_id_original':mapped[locus.gene_id],'panel_strand':locus.strand,'panel_gene':[locus.chromosome,locus.gene_start0,locus.gene_end0]}
        try:
            if locus.chromosome not in lengths: raise ValueError('panel chromosome absent from FASTA')
            row['window']=[locus.chromosome,*w.evaluation_window(locus,lengths[locus.chromosome])]
            local=genes
            if local is None:
                path=out/('reference.%02d.gff3'%i)
                local=p.prepare_mapped_reference([locus],payload['reference'],path,mapping)
            txs=local[locus.gene_id]
            row.update(strict_parser_and_panel_geometry_ok=True,transcript_count=len(txs),duplicate_structures=duplicate_groups(txs,p),
                       noncoding_mono_transcripts=[tid for tid,t in txs.items() if len(t['exons'])==1 and not t['cdss']])
        except Exception as e: row.update(strict_parser_and_panel_geometry_ok=False,error=str(e),error_type=type(e).__name__)
        report['loci'].append(row)
    report['n_loci']=len(report['loci'])
    report['compatible_locus_count']=sum(r['strict_parser_and_panel_geometry_ok'] for r in report['loci'])
    report['duplicate_ambiguous_locus_count']=sum(bool(r.get('duplicate_structures')) for r in report['loci'])
    # Rehash raw authoritative bytes after parsing; no GFF repair or deduplication.
    if p.frozen_reference(payload['fasta'],payload['reference'])!=binding: raise ValueError('reference changed during audit')
    p.recheck_reference_mapping(mapping)
    p.stage_json(out/'reference_compatibility.json',report)
    if not defer_publication:
        p.publish_staged([out/'reference_compatibility.json'],lambda: recheck_audit(payload,report))
    return report

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--reference-only',action='store_true',help='explicit partial diagnostic; never a 50-raster completion')
    a=ap.parse_args(); root=a.root.resolve(strict=True); out=a.out.absolute()
    if not os.environ.get('SLURM_JOB_ID'): raise ValueError('large inputs may only be read in compute allocation')
    if not 1<=int(os.environ.get('SLURM_CPUS_PER_TASK','0'))<=4: raise ValueError('CPU allocation outside 1..4 bound')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise ValueError('CUDA_VISIBLE_DEVICES must be empty')
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    try:
        source=verify_source(root); objects=authority.verify(root)
        dump(out/'source_identity.json',source); dump(out/'git_authority.json',objects)
        sys.path[:0]=[str(root/'revision/scripts/mvp'),str(root/'src')]
        import mvp_produce_counts as p
        from transgenic.evidence import mvp_windows as w
        if p.ROOT!=root: raise ValueError('reference producer imported from wrong source root')
        panel=None if a.reference_only else cpu_runtime(root,out)
        # Paths and expected identities come from unchanged accepted records and original Git bytes.
        import csv,io
        rows=list(csv.DictReader(io.StringIO(authority.git(root,'show',authority.SPEC).decode()),delimiter='\t'))
        ref=[r for r in rows if r['species_id']=='Athaliana']
        if len(ref)!=1: raise ValueError('ambiguous authority species')
        accepted=json.loads((root/'notes/b6_mvp_technical_candidate_20260915.json').read_text())
        payload={'panels':str(root/'revision/protocols/b6_panels_v1.tsv'),'fasta':ref[0]['fasta'],'reference':ref[0]['gff'],
                 'candidate':{'acceptance':str(root/'notes/b6_mvp_technical_candidate_20260915.json'),'acceptance_sha256':ACCEPTANCE,
                 'inventory':str(root/'notes/b6_mvp_candidate_inventories_20260915.json'),'receipts':str(root/'notes/b6_mvp_candidate_receipts_20260915.json'),
                 'corpus':accepted['inputs']['corpus']['path'],'evidence':accepted['artifact']['path']}}
        dump(out/'request.json',payload)
        report=reference_audit(payload,out,defer_publication=True)
        def publish_completion(name, data):
            p.stage_json(out/name,data)
            def recheck():
                verify_source(root); authority.verify(root)
                recheck_audit(payload,report)
            p.publish_staged([out/'reference_compatibility.json',out/name],recheck)
        if a.reference_only:
            publish_completion('REFERENCE_ONLY.json',{'scope':'reference_only_raster_not_run','reference_report_sha256':sha(out/'reference_compatibility.json.pending')}); return
        manifest=panel.prepare(out/'request.json',out/'panel')
        verify_source(root); authority.verify(root)
        if len(manifest['loci'])!=50: raise ValueError('missing raster loci')
        counts=[{k:r[k] for k in ('gene_id','window','selected_molecule_count','biological_nonzero_per_channel','availability_nonzero','zero_biological_support')} for r in manifest['loci']]
        dump(out/'panel_counts.json',{'scope':'raw_EST_input_counts_only','population':'excluded_heldout','rn':None,'training_eligible':False,'loci':counts,
             'manifest_sha256':sha(out/'panel/manifest.json'),'reference_report_sha256':sha(out/'reference_compatibility.json.pending'),
             'reference_compatible_loci':report['compatible_locus_count'],'duplicate_ambiguous_loci':report['duplicate_ambiguous_locus_count'],'scientific_verdict':None})
        publish_completion('OBSERVATIONS_COMPLETE.json',{'job_id':os.environ['SLURM_JOB_ID'],'seconds':time.monotonic()-started,'counts_sha256':sha(out/'panel_counts.json'),'scope':'raw_CPU_observations_not_prediction_or_scientific_approval'})
    except Exception as e:
        dump(out/'FAILED.json',{'error':str(e),'type':type(e).__name__,'seconds':time.monotonic()-started,'scope':'incomplete_observations'}); raise

if __name__=='__main__': main()
