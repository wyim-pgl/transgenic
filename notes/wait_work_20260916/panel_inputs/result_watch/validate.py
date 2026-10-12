import pathlib,json,hashlib,tarfile,collections
w=pathlib.Path(__file__).resolve().parent; base=w.parent; meta=w/'metadata'; j=meta/'outputs/job-6165559'
def load(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
c=load(j/'panel_counts.json'); m=load(j/'panel/manifest.json'); r=load(j/'reference_compatibility.json'); s=load(j/'source_identity.json'); ref=load(j/'reference_identity.json'); rt=load(j/'cpu_runtime.json')
assert sha(meta/'transport.json')=='1883df3c32c5b560030dbcb89543b3d2730d7c08518d47e4f932b62f88b18c0a'
assert (meta/'transport.json').read_bytes()==(base/'final_release_bindfix1/transport.json').read_bytes()
assert s['commit']=='b660bf1e7dc48287ca34a0d6e3aeb53d45f599b6'
assert s['component_sha256']=='555c3618e792dbed06bfc2c392b99efe79fd16853e8c92abe252a5b34cbd0c44'
with tarfile.open(base/'final_release_bindfix1/source.tar') as t:
 for name,digest in dict(s['files'],**s['companions']).items():
  assert hashlib.sha256(t.extractfile('source/'+name).read()).hexdigest()==digest,name
assert load(j/'git_authority.json')==load(base/'authority.final_release.json')
assert r['reference_identity']==ref==m['authority']['reference']
assert load(j/'OBSERVATIONS_COMPLETE.json')['job_id']=='6165559'
assert load(j/'OBSERVATIONS_COMPLETE.json')['counts_sha256']==sha(j/'panel_counts.json')
assert c['manifest_sha256']==sha(j/'panel/manifest.json')==(j/'panel/COMPLETE').read_text().strip()
assert c['reference_report_sha256']==sha(j/'reference_compatibility.json')
assert r['restricted_reference']['sha256']==sha(j/'reference.panel.gff3')
assert len(c['loci'])==len(m['loci'])==len(r['loci'])==50
assert len({x['gene_id'] for x in c['loci']})==50
assert [x['gene_id'] for x in c['loci']]==[x['gene_id'] for x in r['loci']]
for a,b in zip(c['loci'],m['loci']):
 assert a=={k:b[k] for k in a}
 assert len(a['biological_nonzero_per_channel'])==7
 assert a['zero_biological_support']==(not any(a['biological_nonzero_per_channel']))
 assert all(0<=n<=30720 for n in a['biological_nonzero_per_channel'])
 assert 0<=a['availability_nonzero']<=30720
 assert a['window'][2]-a['window'][1]==30720
 assert b['track']['shape']==[30720,8] and b['track']['dtype']=='float16'
assert m['seeds']==[1,2,3] and m['aug_seeds']==[20260915,20260916,20260917] and m['seed_tracks_identical'] is True
assert m['packages']=={k:rt['packages'][k] for k in ('numpy','duckdb')}
assert c['rn'] is None and c['training_eligible'] is False and c['population']=='excluded_heldout'
assert m['training_eligible'] is False and m['population']=='excluded_heldout' and m['keep_fraction']==1
assert m['authority']['accepted_data_sha256']['evidence']=='ed2d0eef2b205626db0983ab9f029f84c30514babc77c330bdb250b2fdfe3d0d'
assert m['authority']['accepted_data_sha256']['panel']==s['files']['revision/protocols/b6_panels_v1.tsv']
assert m['rasterizer']['sha256']==s['files']['src/transgenic/evidence/mvp_tracks.py']
request=load(j/'request.json')
assert request['fasta']==ref['files']['fasta']['path'] and request['reference']==ref['files']['gff']['path']
assert m['inputs'][-1]['sha256']==sha(j/'request.json')
listing={x['path']:x for x in load(w/'collection.json')['listing']}
root=load(w/'collection.json')['root']
for row in m['loci']:
 for key in ('track','sequence','provenance'):
  a=row[key]; st=listing[root+'/outputs/job-6165559/panel/'+a['path']]
  assert st['regular'] and st['bytes']==a['bytes']
assert not any('FAILED' in p for p in listing)
summary={'validation':'all metadata assertions passed; no array/DB reread', 'source_files_compared_to_reviewed_tar':len(s['files']), 'companion_files_compared':len(s['companions']), 'loci':50,'selected_molecules_sum_over_windows_not_unique':sum(x['selected_molecule_count'] for x in c['loci']), 'selected_molecules_range':[min(x['selected_molecule_count'] for x in c['loci']),max(x['selected_molecule_count'] for x in c['loci'])], 'biological_nonzero_sums_channels_0_to_6':[sum(x['biological_nonzero_per_channel'][i] for x in c['loci']) for i in range(7)], 'zero_biological_support_loci':[x['gene_id'] for x in c['loci'] if x['zero_biological_support']], 'availability_nonzero_sum_channel_7':sum(x['availability_nonzero'] for x in c['loci']), 'availability_range':[min(x['availability_nonzero'] for x in c['loci']),max(x['availability_nonzero'] for x in c['loci'])], 'reference_compatible_loci':r['compatible_locus_count'], 'parser':r['panel_strict_parser'],'errors':dict(collections.Counter(x.get('error_type') for x in r['loci'])), 'all_missing_panel_gene_errors':all(x.get('error')=='reference missing panel gene '+x['gene_id'] for x in r['loci']), 'duplicate_ambiguous_loci':r['duplicate_ambiguous_locus_count'], 'seeds':m['seeds'],'aug_seeds':m['aug_seeds'],'seed_tracks_identical':m['seed_tracks_identical'],'packages':rt['packages']}
(w/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
hashes=[]
for p in sorted(meta.rglob('*')):
 if p.is_file(): hashes.append({'path':str(p.relative_to(w)),'bytes':p.stat().st_size,'sha256':sha(p)})
(w/'output_metadata_hashes.json').write_text(json.dumps({'files':hashes,'total_bytes':sum(x['bytes'] for x in hashes)},indent=2)+'\n')
print(json.dumps(summary,indent=2))
