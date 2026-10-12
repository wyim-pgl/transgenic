"""One-shot actual50 LOCAL publication verifier. No DB, Torch, network or source writes.
Checks adapted from mvp_verify_panel_inputs.py:13-84; independent provenance
raster arithmetic follows pinned mvp_tracks.py. Does NOT prove DB completeness
or compare window FASTA against the untransferred whole-genome reference.
"""
import copy
import csv
import hashlib
import json
import platform
import shutil
import tempfile
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent
ROOT = D / 'copied/panel'
OLD = D.parent / 'panel_result/copied/outputs/job-6165628'
PACKAGE = D.parents[1] / 'wait_work_20260916/panel_inputs/reference_map_release/package/source'
PIN = '7528c9bff5a9a9b177b1ad333ad76525a1d02d3c4cde4e0550865ff12e0acc0e'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load(p):
    return json.loads(p.read_text())

def artifact(root, record):
    rel = Path(record['path'])
    path = root / rel
    require(not rel.is_absolute() and '..' not in rel.parts and path.is_file()
            and not path.is_symlink() and path.resolve() == path.absolute(), 'artifact path')
    require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], 'artifact bytes/hash')
    return path

def check_row(root, row):
    chrom, start, end = row['window']
    require(type(start) is int and type(end) is int and start >= 0 and end-start == 30720, 'window')
    fa = artifact(root, row['sequence']).read_text(encoding='ascii')
    lines = fa.splitlines()
    require(len(lines) == 2 and lines[0] == f'>{row["gene_id"]} {chrom}:{start}-{end} genomic_forward', 'FASTA header/orientation')
    seq = lines[1]
    require(len(seq) == 30720 and set(seq.upper()) <= set('ACGTN') and fa == '\n'.join(lines)+'\n', 'FASTA sequence length/alphabet')
    require(hashlib.sha256(seq.encode('ascii')).hexdigest() == row['sequence_sha256'], 'sequence digest')
    require(row['track']['dtype'] == 'float16' and row['track']['shape'] == [30720,8], 'track metadata')
    with artifact(root, row['track']).open('rb') as f:
        array = np.load(f, allow_pickle=False)
        require(f.read() == b'', 'NPY trailing bytes')
    require(array.dtype == np.float16 and array.shape == (30720,8), 'track dtype/shape')
    require(np.isfinite(array).all() and (array >= 0).all(), 'finite nonnegative')
    masked = np.frombuffer(seq.upper().encode('ascii'), dtype='S1') == b'N'
    require(not array[masked].any() and (array[~masked,7] == 1).all(), 'N mask/availability')
    require((array[:,:3] <= np.float16(np.log1p(64))).all(), 'coverage cap')
    biological = np.count_nonzero(array[:,:7], axis=0).tolist()
    require(biological == row['biological_nonzero_per_channel'] and
            int(np.count_nonzero(array[:,7])) == row['availability_nonzero'] and
            row['zero_biological_support'] is (not any(biological)), 'support counts')
    text = artifact(root, row['provenance']).read_text()
    records = [json.loads(line) for line in text.splitlines()]
    require(text == ''.join(json.dumps(r, sort_keys=True, separators=(',',':'))+'\n' for r in records), 'canonical JSONL')
    ids = [r['molecule']['molecule_id'] for r in records]
    require(len(ids) == row['selected_molecule_count'] and len(set(ids)) == len(ids) and ids == sorted(ids), 'molecule count/uniqueness/order')
    coverage = np.zeros((30720,3), dtype=np.int64)
    junctions = np.zeros((30720,4), dtype=np.int64)
    for rec in records:
        m = rec['molecule']; mid = m['molecule_id']
        require(m['excluded'] is True and m['cause'] == 'heldout_locus' and m['primary_count'] > 0, 'heldout selection policy')
        require(m['chrom'] == chrom and m['strand'] in ('+','-','?') and 0 <= m['start0'] < m['end0'], 'molecule geometry')
        members = rec['members']
        names = [v['member'] for v in members]
        require(bool(members) and names == sorted(set(names)) and sorted(m['members']) == names and
                all(v['molecule_id'] == mid and v['clone_key'] for v in members), 'member identity')
        require(rec['blocks'] or rec['junctions'], 'empty selected molecule')
        require(rec['blocks'] == sorted(rec['blocks'], key=lambda b:(b['start0'],b['end0'])), 'block order')
        covered = np.zeros(30720, dtype=bool)
        for b in rec['blocks']:
            require(b['molecule_id'] == mid and b['chrom'] == chrom and
                    m['start0'] <= b['start0'] < b['end0'] <= m['end0'] and
                    b['start0'] < end and b['end0'] > start, 'block boundary/identity')
            covered[max(0,b['start0']-start):min(30720,b['end0']-start)] = True
        coverage[:,{'+':0,'-':1,'?':2}[m['strand']]] += covered
        seen = set()
        for j in rec['junctions']:
            donor, acceptor = j['donor0'], j['acceptor0']
            require(j['molecule_id'] == mid and j['chrom'] == chrom and j['tx_strand'] in ('+','-') and
                    m['start0'] <= j['intron_start0'] < j['intron_end0'] < m['end0'], 'junction identity/geometry')
            require((donor,acceptor) == ((j['intron_start0'],j['intron_end0']) if j['tx_strand']=='+' else (j['intron_end0'],j['intron_start0'])), 'junction orientation')
            require(start <= donor < end or start <= acceptor < end, 'junction endpoint boundary')
            if m['strand'] == '?':
                continue
            for pos,ch in ((donor,0 if m['strand']=='+' else 2),(acceptor,1 if m['strand']=='+' else 3)):
                if start <= pos < end:
                    seen.add((pos-start,ch))
        for pos,ch in seen:
            junctions[pos,ch] += 1
    expected = np.ones((30720,8), dtype=np.float32)
    expected[:,:3] = np.log1p(np.minimum(coverage,64))
    expected[:,3:7] = np.log1p(junctions)
    expected[masked] = 0
    require(np.array_equal(array, expected.astype(np.float16)), 'provenance numerical reconstruction')
    return {'gene_id':row['gene_id'], 'window':row['window'], 'selected_molecules':len(ids),
            'biological_nonzero':biological, 'availability_nonzero':int(np.count_nonzero(array[:,7])),
            'masked_N':int(masked.sum()), 'min':float(array.min()), 'max':float(array.max()),
            'molecule_ids':ids}

def tamper_checks(row):
    results = []
    cases = [('byte_hash','artifact bytes/hash'), ('dtype','track dtype/shape'),
             ('shape','track dtype/shape'), ('nan','finite nonnegative'),
             ('availability','N mask/availability'), ('orientation','FASTA header/orientation'),
             ('sequence','sequence digest'), ('duplicate','molecule count/uniqueness/order'),
             ('boundary','block boundary/identity'), ('policy','heldout selection policy'),
             ('value','provenance numerical reconstruction'), ('count','support counts')]
    for case, reason in cases:
        with tempfile.TemporaryDirectory(prefix='tamper-', dir=D) as tmp:
            root = Path(tmp); r = copy.deepcopy(row)
            for key in ('sequence','track','provenance'):
                shutil.copyfile(ROOT/r[key]['path'], root/r[key]['path'])
            key = 'track'; path = root/r[key]['path']
            if case == 'byte_hash':
                with path.open('ab') as f: f.write(b'x')
            elif case in ('dtype','shape','nan','availability','value'):
                a = np.load(path, allow_pickle=False)
                if case == 'dtype': a = a.astype(np.float32)
                elif case == 'shape': a = a[:-1]
                elif case == 'nan': a[0,0] = np.nan
                elif case == 'availability': a[0,7] = 0
                else:
                    i,j = np.argwhere(a[:,:3] > 0)[0]; a[i,j] *= np.float16(0.5)
                np.save(path,a,allow_pickle=False)
            elif case in ('orientation','sequence'):
                key='sequence'; path=root/r[key]['path']; text=path.read_text()
                if case=='orientation': text=text.replace('genomic_forward','reverse_complement')
                else:
                    header, seq = text.splitlines(); seq=('C' if seq[0]!='C' else 'A')+seq[1:]; text=header+'\n'+seq+'\n'
                path.write_text(text)
            elif case in ('duplicate','boundary','policy'):
                key='provenance'; path=root/r[key]['path']; records=[json.loads(x) for x in path.read_text().splitlines()]
                if case=='duplicate': records.append(records[0]); r['selected_molecule_count']+=1
                elif case=='boundary': records[0]['blocks'][0]['end0']=records[0]['blocks'][0]['start0']
                else: records[0]['molecule']['excluded']=False
                path.write_text(''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in records))
            elif case=='count': r['biological_nonzero_per_channel'][0]+=1
            if case != 'byte_hash':
                r[key].update(bytes=path.stat().st_size,sha256=sha(path))
            try:
                check_row(root,r)
            except ValueError as e:
                require(str(e)==reason, f'tamper {case}: unexpected rejection {e}')
                results.append({'case':case,'rejected_by':str(e)})
            else:
                raise ValueError('tamper accepted: '+case)
    return results

def main():
    manifest=load(ROOT/'manifest.json'); counts=load(OLD/'panel_counts.json')
    require(sha(ROOT/'manifest.json') == sha(OLD/'panel/manifest.json') == PIN, 'original manifest pin')
    require((ROOT/'COMPLETE').read_bytes() == (PIN+'\n').encode() == (OLD/'panel/COMPLETE').read_bytes(), 'COMPLETE link')
    marker=load(OLD/'OBSERVATIONS_COMPLETE.json')
    require(marker['job_id']=='6165628' and marker['counts_sha256']==sha(OLD/'panel_counts.json'), 'original observations marker')
    require(counts['manifest_sha256']==PIN and counts['reference_report_sha256']==sha(OLD/'reference_compatibility.json'), 'counts publication links')
    policy=dict(schema_version=1,purpose='est_mvp_panel_inputs_not_training',training_eligible=False,
                orientation='genomic_forward',population='excluded_heldout',keep_fraction=1,source='EST',
                seeds=[1,2,3],aug_seeds=[20260915,20260916,20260917],seed_tracks_identical=True,
                off_policy='zeros_like(on); same sequence/window',zero_support_policy='report_not_fail')
    require(all(type(manifest[k]) is type(v) and manifest[k]==v for k,v in policy.items()), 'manifest policy')
    panel=PACKAGE/'revision/protocols/b6_panels_v1.tsv'
    require(sha(panel)==manifest['authority']['accepted_data_sha256']['panel'], 'frozen TSV digest')
    with panel.open() as f: loci=[x for x in csv.DictReader(f,delimiter='\t') if x['panel'] in ('mvp','mvp_core')]
    rows=manifest['loci']; require(len(rows)==len(loci)==50 and len({r['gene_id'] for r in rows})==50, 'exact50')
    expected={'COMPLETE','manifest.json'}
    for row,p,c in zip(rows,loci,counts['loci']):
        start=int(p['start'])-1; length=int(p['fin'])-start; offset=start-(30720-length)//2
        require(row['gene_id']==p['gene_id'] and row['window']==[p['chromosome'],offset,offset+30720], 'frozen window binding')
        require(c=={k:row[k] for k in c}, 'count row linkage')
        for key in ('sequence','track','provenance'):
            require(row[key]['path'] not in expected,'duplicate artifact'); expected.add(row[key]['path'])
    require({p.name for p in ROOT.iterdir()}==expected and len(expected)==152, 'exact file inventory')
    stats=load(D.parent/'panel_result/remote_file_stats.json')
    stats={Path(x['path']).name:x['size'] for x in stats if x['path'].startswith('outputs/job-6165628/panel/')}
    require(sum(stats.values())<=100_000_000 and stats=={p.name:p.stat().st_size for p in ROOT.iterdir()}, 'transfer byte inventory')
    before={p.name:sha(p) for p in ROOT.iterdir()}
    measured=[check_row(ROOT,row) for row in rows]
    negatives=tamper_checks(rows[0])
    require(before=={p.name:sha(p) for p in ROOT.iterdir()}, 'copied originals unchanged')
    ids=set().union(*(set(x['molecule_ids']) for x in measured))
    for x in measured: del x['molecule_ids']
    report=dict(status='PASS_LOCAL_ACTUAL50_PUBLICATION_NUMERICAL_SCOPE', scientific_approval=False,
                externOOM='unresolved; unrelated scheduler-step finding', python=platform.python_version(),numpy=np.__version__,
                transferred_files=152,transferred_bytes=sum(stats.values()),artifact_hashes_verified=150,
                marker_manifest_linkage=True,whole_reference_sequence_equality_verified=False,
                source_DB_selection_completeness_verified=False,source_inputs_freshness_verified=False,
                seed_independence='manifest assertion; one shared array per locus, not three independently sampled runs',
                provenance_numerical_equality_loci=50,selected_molecule_sum_over_windows=sum(x['selected_molecules'] for x in measured),
                distinct_molecule_ids_across_copied_windows=len(ids),
                biological_nonzero_sums=np.sum([x['biological_nonzero'] for x in measured],axis=0).tolist(),
                availability_nonzero_sum=sum(x['availability_nonzero'] for x in measured),
                negative_checks=negatives,files_sha256=before,loci=measured)
    (D/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('files_sha256','loci')},indent=2))

if __name__=='__main__': main()
