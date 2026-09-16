"""CPU fixtures only: no biological measurements or external alignments."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'revision/scripts/mvp'))
import mvp_produce_counts as p
from transgenic.evidence import mvp_windows as w


def dump(path, value):
    path.write_text(json.dumps(value))
    return path


def locus():
    return w.PanelLocus('mvp_core', 'g', 'Chr1', 30001, 31000, '+', 1000)


def gff(path, entries):
    lines = ['##gff-version 3', 'Chr1\tx\tgene\t30001\t31000\t.\t+\t.\tID=g']
    for tid, strand, exons, cds in entries:
        # Separate minus-strand gene to keep valid parent strand geometry.
        parent = 'g'
        if strand == '-':
            parent = 'gm' + tid
            lines.append(f'Chr1\tx\tgene\t30001\t31000\t.\t-\t.\tID={parent}')
        lines.append(f'Chr1\tx\tmRNA\t{exons[0][0]}\t{exons[-1][1]}\t.\t{strand}\t.\tID={tid};Parent={parent}')
        for kind, spans in [('exon',exons),('CDS',cds)]:
            for a,b in spans:
                lines.append(f'Chr1\tx\t{kind}\t{a}\t{b}\t.\t{strand}\t{0 if kind == "CDS" else "."}\tParent={tid}')
    path.write_text('\n'.join(lines)+'\n')
    return path


def txs(path):
    return p.select_transcripts(p.read_gff(path), locus())


def fake_gffcompare(tmp_path):
    """Explicit stand-in: exact-chain matching fixture, NOT tool certification."""
    path = tmp_path / 'gffcompare'
    path.write_text('''#!/usr/bin/env python3
import sys
from pathlib import Path
if '--version' in sys.argv:
    print('gffcompare v0.12.6'); raise SystemExit()
def read(path):
    tx = {}
    for line in Path(path).read_text().splitlines():
        f = line.split('\\t'); tid = f[8].split('transcript_id "')[1].split('"')[0]
        tx.setdefault(tid, [f[0],f[6],[]])[2].append((int(f[3]),int(f[4])))
    return {k:(v[0],v[1],tuple((a[1],b[0]) for a,b in zip(v[2],v[2][1:]))) for k,v in tx.items()}
r=read(sys.argv[sys.argv.index('-r')+1]); q=read(sys.argv[-1])
out=Path(sys.argv[-1]).parent/'compare.predicted.gtf.tmap'
with out.open('w') as f:
    f.write('ref_gene_id\\tref_id\\tclass_code\\tqry_gene_id\\tqry_id\\n')
    for tid,key in q.items():
        hit=next((k for k,v in r.items() if v==key),'-')
        f.write('g\\t'+hit+'\\t'+('=' if hit!='-' else 'u')+'\\tg\\t'+tid+'\\n')
''')
    path.chmod(0o755)
    return str(path)


def test_hand_counts_chain_mono_strand_and_gene_restriction(tmp_path):
    multi = [(30001,30100),(30201,30300)]
    mono = [(30501,30600)]
    ref = txs(gff(tmp_path/'ref.gff', [('r','+',multi,multi),('mono','+',mono,[(30510,30590)])]))
    pred = txs(gff(tmp_path/'pred.gff', [('p','+',multi,multi),('wrongstrand','-',mono,[(30510,30590)])]))
    counts, receipt = p.reference_counts(pred, ref, tmp_path/'result', fake_gffcompare(tmp_path))
    assert counts == {'f1_tp':1,'f1_predicted':2,'f1_reference':2}
    assert (receipt['TP'],receipt['FP'],receipt['FN']) == (1,1,1)
    assert receipt['command'] and receipt['outputs']
    pred['wrongstrand']['strand'] = '+'
    counts, _ = p.reference_counts(pred, ref, tmp_path/'result2', str(tmp_path/'gffcompare'))
    assert counts['f1_tp'] == 2
    pred['wrongstrand']['cdss'] = [(30510,30591)]
    assert p.reference_counts(pred, ref, tmp_path/'result3', str(tmp_path/'gffcompare'))[0]['f1_tp'] == 1
    pred['p']['start'], pred['p']['end'] = 40000,41000
    genes = {'x':{'transcripts':pred}}
    assert set(p.select_transcripts(genes,locus())) == {'wrongstrand'}


def test_empty_predictions_are_zero_not_missing(tmp_path):
    path = tmp_path/'empty.gff'; path.write_text('##gff-version 3\n')
    assert p.read_gff(path) == {}
    mono = [(30001,30100)]
    ref = txs(gff(tmp_path/'ref.gff',[('r','+',mono,mono)]))
    counts, receipt = p.reference_counts({},ref,tmp_path/'out','not-invoked')
    assert counts == {'f1_tp':0,'f1_predicted':0,'f1_reference':1}
    assert receipt['FN'] == 1
    with pytest.raises(FileNotFoundError): p.read_gff(tmp_path/'absent')


def test_duplicate_isoform_policy_fails_closed(tmp_path):
    mono = [(30001,30100)]
    pred = txs(gff(tmp_path/'p.gff',[('p','+',mono,mono),('duplicate','+',mono,mono)]))
    with pytest.raises(ValueError,match='duplicate isoform'):
        p.reference_counts(pred,{},tmp_path/'out','unused')


@pytest.mark.parametrize('change', ['columns','strand','orphan','duplicate','overlap','phase','missing_exon'])
def test_malformed_gff_rejected(tmp_path, change):
    path = gff(tmp_path/'p.gff',[('p','+',[(30001,30100)],[(30001,30100)])])
    text = path.read_text()
    if change == 'columns': text += 'broken\n'
    elif change == 'strand': text = text.replace('\t+\t','\t.\t')
    elif change == 'orphan': text = text.replace('Parent=p','Parent=unknown')
    elif change == 'duplicate': text += text.splitlines()[2]+'\n'
    elif change == 'overlap': text += text.splitlines()[3]+'\n'
    elif change == 'phase': text = text.replace('\t0\tParent=p','\t.\tParent=p')
    else: text = '\n'.join(l for l in text.splitlines() if '\texon\t' not in l)
    path.write_text(text)
    with pytest.raises(ValueError): p.read_gff(path)


def prediction_rows(tmp_path, panel):
    checkpoints = {s: str(s)*64 for s in (1,2,3,123)}
    rows = []
    for l in panel:
        for s,f in [(1,0),(1,1),(2,0),(2,1),(3,0),(3,1),(123,0)]:
            path = tmp_path/f'{len(rows)}.gff'; path.write_text('##gff-version 3\n')
            rows.append(dict(p.file_identity(path), seed=s, fraction=f, gene_id=l.gene_id,
                             window=[l.chromosome,*w.evaluation_window(l)],checkpoint_sha256=checkpoints[s]))
    return rows, checkpoints


def test_full_frozen_panel_inventory_350_and_cross_seed(tmp_path):
    panel = w.load_panel(ROOT/'revision/protocols/b6_panels_v1.tsv')
    rows, checkpoints = prediction_rows(tmp_path,panel)
    assert len(p.prediction_inventory(rows,panel,checkpoints)) == 350
    with pytest.raises(ValueError,match='full50'): p.prediction_inventory(rows[:-1],panel,checkpoints)
    bad = copy.deepcopy(rows); bad[0]['checkpoint_sha256'] = checkpoints[2]
    with pytest.raises(ValueError,match='checkpoint'): p.prediction_inventory(bad,panel,checkpoints)
    bad = copy.deepcopy(rows); bad[0]['window'][1] += 1
    with pytest.raises(ValueError,match='window'): p.prediction_inventory(bad,panel,checkpoints)
    bad = copy.deepcopy(rows); bad[1].update({k:bad[0][k] for k in ('path','sha256','size')})
    with pytest.raises(ValueError,match='duplicate prediction file'): p.prediction_inventory(bad,panel,checkpoints)


def test_actual_commit_anchor_wrong_reference_and_missing_anchor(tmp_path,monkeypatch):
    fasta = tmp_path/'fake.fa'; fasta.write_text('>Chr1\nA\n')
    with pytest.raises(ValueError,match='frozen fasta'):
        p.frozen_reference(fasta,fasta)
    monkeypatch.setattr(p,'AUTHORITY_COMMIT','0'*40)
    with pytest.raises(subprocess.CalledProcessError): p.frozen_reference(fasta,fasta)


def finalized_fixture(tmp_path, seed=1, hashes=None):
    import test_mvp_finalize as f
    md = f.metadata('production')
    md['seed_augmentation'] = {'seed':seed, 'aug_seed':20260914+seed}
    if hashes is not None:
        md['data_identities'] = {k:{'sha256':v,'verification':'recorded'} for k,v in hashes.items()}
    source = tmp_path/'source'; source.mkdir()
    (source/'model.safetensors').write_bytes(b'explicit synthetic source')
    dump(source/'run_config.json',{'seed':123})
    audit = md['initialization_audit']['audit']
    audit['resolved_checkpoint_dir'] = str(source)
    audit['resolved_weights_path'] = str(source/'model.safetensors')
    for name in ('model.safetensors','run_config.json'):
        identity = p.file_identity(source/name)
        snap = {'sha256':identity['sha256'],'size':identity['size'],'device':1,'inode':1,'mtime_ns':1}
        audit['files'][name] = {'path':identity['path'],'before':snap,'after':dict(snap)}
    md['initialization_audit']['audit_sha256'] = hashlib.sha256(f.finalize._canonical_json(audit)).hexdigest()
    code = dump(tmp_path/'code.json',{'files':{str(Path(__file__)):w.sha256_file(__file__)},
                                    'external_packages':{'torch':'test-version','duckdb':'test-version'}})
    md['sif_code_identity']['code']['sha256'] = w.sha256_file(code)
    f.publish(tmp_path,md=md,train=f.summary('production'))
    hashes = {k:v['sha256'] for k,v in md['data_identities'].items()}
    return tmp_path/'final',hashes,code


def test_reuses_actual_finalizer_publication_and_complete_inventory(tmp_path):
    final, hashes, code = finalized_fixture(tmp_path)
    receipt = p.verify_final(final,1,[1],hashes,code)
    assert receipt['seed'] == 1 and receipt['valid_rns'] == [1]
    for seed in (2,3):
        with pytest.raises(ValueError,match='cross-seed'): p.verify_final(final,seed,[1],hashes,code)
    with pytest.raises(ValueError,match='inventory mismatch'): p.verify_final(final,1,[1,2],hashes,code)
    (final/'paired_records.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='hash mismatch'): p.verify_final(final,1,[1],hashes,code)


def test_final_code_input_and_source_byte_bindings(tmp_path):
    final, hashes, code = finalized_fixture(tmp_path)
    bad = dict(hashes,panel='0'*64)
    with pytest.raises(ValueError,match='panel identity'): p.verify_final(final,1,[1],bad,code)
    original = code.read_text(); code.write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'): p.verify_final(final,1,[1],hashes,code)
    code.write_text(original)
    (tmp_path/'source/model.safetensors').write_bytes(b'mutated')
    with pytest.raises(ValueError,match='hash mismatch'): p.verify_final(final,1,[1],hashes,code)


@pytest.mark.parametrize('mutation', ['summary','duplicate_rn','test_scope','marker'])
def test_rehashed_final_does_not_bypass_semantic_checks(tmp_path, mutation):
    final, hashes, code = finalized_fixture(tmp_path)
    if mutation == 'summary':
        dump(final/'paired_summary.json',{'primary_delta':999})
    elif mutation == 'duplicate_rn':
        path=final/'paired_records.jsonl'; path.write_text(path.read_text()*2)
    elif mutation == 'test_scope':
        metadata=p.read_json(final/'metadata.json'); metadata['identity_scope']='test'
        dump(final/'metadata.json',metadata)
    else:
        (final/'TRAINING_DONE').write_text('incomplete\n')
    manifest=p.read_json(final/'manifest.json')
    for name in manifest['files']:
        identity=p.file_identity(final/name)
        manifest['files'][name]={k:identity[k] for k in ('sha256','size')}
    dump(final/'manifest.json',manifest)
    with pytest.raises(ValueError): p.verify_final(final,1,[1],hashes,code)


def test_json_duplicate_seed_rejected(tmp_path):
    path=tmp_path/'x.json'; path.write_text('{"1":1,"1":2}')
    with pytest.raises(ValueError,match='duplicate JSON'): p.read_json(path)
