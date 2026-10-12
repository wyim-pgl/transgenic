"""Independent small-file validation only; never opens remote arrays or databases."""
import csv, hashlib, json, pathlib, stat
from collections import Counter, defaultdict
D=pathlib.Path(__file__).resolve().parent
R=D/'copied/outputs/job-6165628'
P=D.parents[1]/'wait_work_20260916/panel_inputs/reference_map_release/package'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text())
m=load(R/'panel/manifest.json'); c=load(R/'panel_counts.json'); r=load(R/'reference_compatibility.json'); s=load(R/'source_identity.json'); marker=load(R/'OBSERVATIONS_COMPLETE.json')
assert marker['job_id']=='6165628' and marker['counts_sha256']==sha(R/'panel_counts.json')
assert c['manifest_sha256']==sha(R/'panel/manifest.json')==(R/'panel/COMPLETE').read_text().strip()
assert c['reference_report_sha256']==sha(R/'reference_compatibility.json')
assert sha(R/'reference.panel.gff3')==r['restricted_reference']['sha256']
assert sha(D/'copied/transport.json')==sha(P/'transport.json')=='11d0550bd46c121c6b53e1cb51f0c528381ff41763e7e95e754d286a4d0448b5'
assert sha(D/'copied/source/source_pins.json')==sha(P/'source/source_pins.json')=='5570f32378928ccd627494652988a0ec8cda3c657cff5d63e0c496ec8c97010b'
pins=load(D/'copied/source/source_pins.json')
assert load(R/'git_authority.json')==load(P/'transport.json')['reference_authority']
assert s==load(P/'SOURCE_VERIFICATION.json')
assert s['files']==pins['source_files'] and len(s['files'])==100
assert s['files']['revision/scripts/mvp/mvp_produce_counts.py']=='6069fb5fe0771015d044d820a4cbbff313d9f3d2982991e1c58c17bc8aa726dc'
assert s['commit']=='9603e20fd4870a9f5b3d5f68e2df8ba0f8a49558'
assert s['component_sha256']=='ad3e5a7aacbc5d71c3a59d9b1b6f6fd0afeff12efab10b7ed123f734e767380e'
ref=load(R/'reference_identity.json')
assert ref==r['reference_identity']==r['reference_mapping']['authority']==m['authority']['reference']
assert ref['commit']=='92d8020ee79eeedd7f1d50167b1c8dc94700fb16'
assert ref['manifest_sha256']=='3ca7a3b9a5793f62d25f503c93cfd7167eb17f7c5b90de7830e6a8777117f783'
assert r['reference_mapping']['method']=='corpus_gene_key_map'
# Independent TSV geometry and reported mapping bijection, no DB re-query.
panel=P/'source/revision/protocols/b6_panels_v1.tsv'
assert sha(panel)==r['panel']['sha256']=='79367154bbf922fd0554285cb6cb47ae3d3c866b145aa1f51ead4f47b00cd243'
loci=[x for x in csv.DictReader(panel.open(),delimiter='\t') if x['panel'] in ('mvp_core','mvp')]
assert Counter(x['panel'] for x in loci)=={'mvp_core':20,'mvp':30}
ids=[x['gene_id'] for x in loci]
rows=r['reference_mapping']['rows']
assert len(ids)==len(set(ids))==len(rows)==len({x['gene_id_original'] for x in rows})==50
assert all([x['gene_id'] for x in rows2]==ids for rows2 in (rows,r['loci'],m['loci'],c['loci']))
# Parse actual copied restricted GFF independently, retaining original identifiers.
genes={}; txs={}; features=[]
for line in (R/'reference.panel.gff3').read_text().splitlines():
    if not line or line.startswith('#'): continue
    f=line.split('\t'); assert len(f)==9
    attrs=dict(x.split('=',1) for x in f[8].split(';') if x)
    feature={'chrom':f[0],'start':int(f[3])-1,'end':int(f[4]),'strand':f[6],'attrs':attrs,'type':f[2]}
    assert 0<=feature['start']<feature['end']
    if f[2]=='gene':
        assert attrs['ID'] not in genes; genes[attrs['ID']]=feature
    elif f[2]=='mRNA':
        assert attrs['ID'] not in txs; txs[attrs['ID']]=dict(feature,exons=[],cdss=[])
    else: features.append(feature)
for f in features:
    if f['type'] not in ('exon','CDS'): continue
    for tid in f['attrs']['Parent'].split(','):
        t=txs[tid]; assert (f['chrom'],f['strand'])==(t['chrom'],t['strand'])
        assert t['start']<=f['start']<f['end']<=t['end']
        t['exons' if f['type']=='exon' else 'cdss'].append((f['start'],f['end']))
bygene=defaultdict(dict)
for tid,t in txs.items():
    t['exons'].sort(); t['cdss'].sort(); assert t['exons']
    assert all(a[1]<b[0] for a,b in zip(t['exons'],t['exons'][1:]))
    assert all(any(e[0]<=cd[0]<cd[1]<=e[1] for e in t['exons']) for cd in t['cdss'])
    g=genes[t['attrs']['Parent']]
    assert (t['chrom'],t['strand'])==(g['chrom'],g['strand']) and g['start']<=t['start']<t['end']<=g['end']
    bygene[t['attrs']['Parent']][tid]=t
assert set(genes)=={x['gene_id_original'] for x in rows} and len(txs)==52
stats={x['path']:x for x in load(D/'remote_file_stats.json')}
assert not any('FAILED' in x or x.endswith('.pending') or 'REFERENCE_ONLY' in x for x in stats)
assert all(stat.S_ISREG(x['mode']) for x in stats.values())
duplicates=0
for p,maprow,rr,mr,cr in zip(loci,rows,r['loci'],m['loci'],c['loci']):
    geom=[p['chromosome'],int(p['start'])-1,int(p['fin'])]
    assert maprow['species_id']==p['species_id']=='Athaliana'
    assert [maprow['chromosome'],maprow['start0'],maprow['end0']]==geom==rr['panel_gene']
    assert maprow['strand']==p['strand']==rr['panel_strand']
    g=genes[maprow['gene_id_original']]
    assert [g['chrom'],g['start'],g['end']]==geom and g['strand']==p['strand']
    assert maprow['name_original']==g['attrs']['Name']
    start=geom[1]-(30720-(geom[2]-geom[1]))//2
    window=[geom[0],start,start+30720]
    assert 0<=start and start+30720<=r['fasta_lengths'][geom[0]]
    assert window==rr['window']==mr['window']==cr['window'] and rr['strict_parser_and_panel_geometry_ok'] is True
    structures=defaultdict(list); noncoding=[]
    for tid,t in bygene[maprow['gene_id_original']].items():
        ex=t['exons']
        key=('mono',tuple(t['cdss'])) if len(ex)==1 and t['cdss'] else (('multi',tuple((a[1],b[0]) for a,b in zip(ex,ex[1:]))) if len(ex)>1 else None)
        if key is not None: structures[key].append(tid)
        else: noncoding.append(tid)
    dup=[v for v in structures.values() if len(v)>1]
    assert not dup and rr['duplicate_structures']==[]
    assert sorted(noncoding)==sorted(rr['noncoding_mono_transcripts'])
    assert rr['transcript_count']==len(bygene[maprow['gene_id_original']])
    assert cr=={k:mr[k] for k in cr}
    assert len(mr['biological_nonzero_per_channel'])==7 and all(0<=n<=30720 for n in mr['biological_nonzero_per_channel'])
    assert mr['zero_biological_support']==(not any(mr['biological_nonzero_per_channel']))
    assert mr['track']['shape']==[30720,8] and mr['track']['dtype']=='float16'
    for kind in ('track','sequence','provenance'):
        a=mr[kind]; st=stats['outputs/job-6165628/panel/'+a['path']]
        assert st['size']==a['bytes'] and st['mtime']>=1789630863
assert r['compatible_locus_count']==c['reference_compatible_loci']==50
assert r['duplicate_ambiguous_locus_count']==c['duplicate_ambiguous_loci']==0
assert r['scientific_verdict'] is None and c['scientific_verdict'] is None
assert m['training_eligible'] is c['training_eligible'] is False
assert m['seeds']==[1,2,3] and m['aug_seeds']==[20260915,20260916,20260917]
assert c['rn'] is None and m['population']==c['population']=='excluded_heldout'
for i in m['inputs']+r['reference_mapping']['sources']:
    path=i['path']; assert '6165559' not in path
    if '/source/' in path:
        name=path.split('/source/',1)[1]; assert i['sha256']==sha(P/'source'/name)
    elif path.endswith('/request.json'): assert i['sha256']==sha(R/'request.json')
assert load(R/'cpu_runtime.json')['cuda_visible_devices']==''
result={'status':'PASS_SMALL_FILE_SCOPE_ONLY','mapping_bijection':50,'geometry_and_windows':50,'restricted_gff_genes':len(genes),'restricted_gff_transcripts':len(txs),'duplicate_ambiguous_loci':0,'array_stats_verified':50,'array_hashes_independently_verified':0,'sequence_provenance_hashes_independently_verified':0,'selected_molecule_count_sum_over_windows':sum(x['selected_molecule_count'] for x in m['loci']),'selected_molecule_count_range':[min(x['selected_molecule_count'] for x in m['loci']),max(x['selected_molecule_count'] for x in m['loci'])],'zero_biological_support_loci':sum(x['zero_biological_support'] for x in m['loci']),'biological_nonzero_sums':[sum(x['biological_nonzero_per_channel'][i] for x in m['loci']) for i in range(7)],'availability_nonzero_sum':sum(x['availability_nonzero'] for x in m['loci']),'hashes':{str(p.relative_to(R)):sha(p) for p in R.rglob('*') if p.is_file()}}
(D/'validation.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
