"""CPU stand-ins only: synthetic authority, remote Hyena/tokenizer, GPU generation,
and GFFCompare. Actual candidate, finalizer, adapter, decoder and producer gates.
No actual SIF or scientific pass is asserted by these tests.
"""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import duckdb
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'revision/scripts/mvp'))
import mvp_decode_panel as runner
import mvp_decode_runtime as runtime
import mvp_prepare_panel_inputs as panel_producer
import mvp_verify_panel_inputs as adapter
from transgenic.evidence import mvp_decode as d
from test_mvp_score_producer import p, w, dump, fake_gffcompare


@pytest.fixture
def panel_publication(tmp_path, monkeypatch):
    from transgenic.training import mvp_cli as cli
    panel_path = ROOT/'revision/protocols/b6_panels_v1.tsv'
    panel = w.load_panel(panel_path)
    fasta, reference = tmp_path/'ref.fa', tmp_path/'ref.gff3'
    lengths = {}
    for locus in panel:
        lengths[locus.chromosome] = max(lengths.get(locus.chromosome,0), w.evaluation_window(locus)[1]+100)
    with fasta.open('w') as handle:
        for chrom, length in lengths.items():
            sequence = bytearray(b'A'*length)
            for locus in panel:
                if locus.chromosome == chrom:
                    start,_ = w.evaluation_window(locus)
                    sequence[start:start+2] = b'Nn'
            handle.write('>'+chrom+'\n'+sequence.decode()+'\n')
    lines = ['##gff-version 3']
    for l in panel:
        for kind, attrs in [('gene',f'ID={l.gene_id}'),('mRNA',f'ID=t{l.gene_id};Parent={l.gene_id}'),('exon',f'Parent=t{l.gene_id}')]:
            lines.append(f'{l.chromosome}\tfixture\t{kind}\t{l.start1}\t{l.fin1}\t.\t{l.strand}\t.\t{attrs}')
    reference.write_text('\n'.join(lines)+'\n')
    # Only the unavailable biological authority is a stand-in; data/candidate and
    # all finalized publication inventories below are genuinely validated.
    def synthetic_authority(a,b):
        return {'CPU_SYNTHETIC_NOT_REFERENCE_APPROVAL':True,'files':{'fasta':d.identity(a),'gff':d.identity(b)}}
    monkeypatch.setattr(p,'frozen_reference',synthetic_authority)
    monkeypatch.setattr(panel_producer,'frozen_reference',synthetic_authority)
    corpus, evidence = tmp_path/'corpus.db', tmp_path/'evidence.db'
    with duckdb.connect(str(corpus)) as c:
        c.execute('CREATE TABLE geneList(rn INT, chromosome VARCHAR, start INT, fin INT, split VARCHAR, is_rc BOOLEAN, species_id VARCHAR)')
        c.execute("INSERT INTO geneList VALUES (1,'Chr1',10,20,'valid',false,'Athaliana'),(2,'Chr2',10,20,'train',false,'Athaliana')")
    with duckdb.connect(str(evidence)) as c:
        c.execute('CREATE TABLE corpus_tile(rn INT, chrom VARCHAR, start0 INT, end0 INT, split VARCHAR, is_rc BOOLEAN)')
        c.execute("INSERT INTO corpus_tile VALUES (1,'Chr1',10,20,'valid',false),(2,'Chr2',10,20,'train',false)")
        c.execute('CREATE TABLE est_molecule(molecule_id VARCHAR, strand VARCHAR, excluded BOOLEAN, cause VARCHAR, primary_count INT)')
        c.execute('CREATE TABLE est_member(member VARCHAR, molecule_id VARCHAR)')
        c.execute('CREATE TABLE est_block(molecule_id VARCHAR,chrom VARCHAR,start0 INT,end0 INT)')
        c.execute('CREATE TABLE est_junction(molecule_id VARCHAR,chrom VARCHAR,donor0 INT,acceptor0 INT)')
        c.execute('CREATE TABLE tile_est(rn INT,molecule_id VARCHAR)')
        start, _ = w.evaluation_window(panel[0])
        c.execute('BEGIN TRANSACTION')
        for i in range(70):
            c.execute("INSERT INTO est_molecule VALUES (?,'+',true,'heldout_locus',1)",[str(i)])
            c.execute('INSERT INTO est_member VALUES (?,?)',[f'read{i}',str(i)])
            c.execute('INSERT INTO est_block VALUES (?,?,?,?)',[str(i),panel[0].chromosome,start+10,start+20])
            c.execute('INSERT INTO est_junction VALUES (?,?,?,?)',[str(i),panel[0].chromosome,start+12,start+30])
        c.execute('COMMIT')
    inventory = dict(schema_version=1,scope=cli._INVENTORY_SCOPE,
        records=[dict(rn=1,chromosome='Chr1',start0=10,end0=20,split='valid',is_rc=False),dict(rn=2,chromosome='Chr2',start0=10,end0=20,split='train',is_rc=False)],
        counts=dict(total=2,train=1,valid=1,evidence_corpus_tile_all_splits=dict(train=1,valid=1,test=0)),
        comparison=dict(exact_ordered_identity_match=True,expected_count=2,source_count=2))
    inv = dump(tmp_path/'inventory.json',inventory)
    receipts = dump(tmp_path/'receipts.json',dict(schema_version=1,scope=cli._RECEIPTS_SCOPE,receipts=[],receipt_count=0))
    hashes = dict(corpus=w.sha256_file(corpus),evidence=w.sha256_file(evidence),panel=w.sha256_file(panel_path))
    acceptance = dict(schema_version=1,scope=cli._SCOPE,
        decision=dict(technical_acceptance=True,production_training_accepted=False,production_authorized=False,author_protocol_freeze=False,scientific_approval=False),
        checks={k:True for k in cli._ACCEPTANCE_CHECKS},
        qc=dict(valid_tiles_total=1,valid_tiles_with_evidence=1,retained_overlapping_panel=0,retained_overlapping_test=0),
        source_counts=dict(members=70,collapsed_rule_a=0,collapsed_rule_b=0,molecules_after_dedup=70,retained=0,excluded_heldout_locus=70,excluded_test_block=0,excluded_unmapped=0),
        companions=dict(inventory={'sha256':w.sha256_file(inv)},receipts={'sha256':w.sha256_file(receipts)}),
        inventory_summary=dict(companion_sha256=w.sha256_file(inv),exact_match_to_evidence_corpus_tile=True,total=2,train=1,valid=1),
        inputs=dict(corpus={'sha256':hashes['corpus']},panel={'sha256':hashes['panel']}),artifact={'sha256':hashes['evidence']})
    accepted = dump(tmp_path/'acceptance.json',acceptance)
    candidate = dict(acceptance=str(accepted),acceptance_sha256=w.sha256_file(accepted),inventory=str(inv),receipts=str(receipts),corpus=str(corpus),evidence=str(evidence))
    payload = dict(candidate=candidate,panels=str(panel_path),fasta=str(fasta),reference=str(reference))
    request = dump(tmp_path/'request.json',payload)
    out = tmp_path/'panel'
    manifest = panel_producer.prepare(request,out)
    payload['panel_inputs'] = str(out)
    sequences = runner.sequences_from_fasta(fasta,panel)
    return payload,panel,sequences,manifest,hashes


def test_real_panel_high_junction_and_biological_zero(panel_publication):
    payload,panel,sequences,manifest,_ = panel_publication
    tracks, records = adapter.load_panel_inputs(payload['panel_inputs'],payload,panel,sequences)
    a = tracks[panel[0].gene_id][0].numpy()
    assert a[12,3] == np.float16(np.log1p(70)) > np.float16(np.log1p(64))
    assert a[12,0] == np.float16(np.log1p(64))
    assert not a[:2].any() and a[2,7] == 1
    assert any(r['zero_biological_support'] for r in manifest['loci'])
    assert len(records) >= 152
    root=Path(payload['panel_inputs'])
    track=manifest['loci'][0]['track']; path=root/track['path']
    for index,value,message in [((12,3),float('nan'),'invalid tracks'),((12,3),-1,'invalid tracks'),
                                 ((0,0),1,'N mask'),((2,7),0,'availability')]:
        changed=a.copy(); changed[index]=value; np.save(path,changed)
        track['sha256']=w.sha256_file(path)
        (root/'manifest.json').write_text(panel_producer.canonical(manifest))
        (root/'COMPLETE').write_text(w.sha256_file(root/'manifest.json')+'\n')
        with pytest.raises(ValueError,match=message):
            adapter.load_panel_inputs(root,payload,panel,sequences)


@pytest.mark.parametrize('mutation', ['missing_marker','changed_jsonl','symlink_escape','coverage','wrong_sequence','wrong_request','wrong_authority','internal_schema'])
def test_panel_rejects_mutations(panel_publication, mutation, tmp_path):
    payload,panel,sequences,manifest,_ = panel_publication
    root = Path(payload['panel_inputs'])
    row = manifest['loci'][0]
    if mutation == 'missing_marker':
        (root/'COMPLETE').unlink()
    elif mutation == 'changed_jsonl':
        (root/row['provenance']['path']).write_text('{}\n')
    elif mutation == 'symlink_escape':
        path = root/row['track']['path']
        outside = tmp_path/'outside.npy'; outside.write_bytes(path.read_bytes())
        path.unlink(); path.symlink_to(outside)
    else:
        if mutation == 'coverage':
            path = root/row['track']['path']; a = np.load(path); a[12,0]=5; np.save(path,a)
            row['track']['sha256'] = w.sha256_file(path)
        elif mutation == 'wrong_sequence': row['sequence_sha256']='0'*64
        elif mutation == 'wrong_request': payload['candidate']=dict(payload['candidate'],acceptance_sha256='0'*64)
        elif mutation == 'wrong_authority': manifest['authority']={}
        else: manifest={'schema':'mvp_panel_batch_internal_v1'}
        (root/'manifest.json').write_text(panel_producer.canonical(manifest))
        (root/'COMPLETE').write_text(w.sha256_file(root/'manifest.json')+'\n')
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):
        adapter.load_panel_inputs(root,payload,panel,sequences)


@pytest.mark.parametrize('phase', ['serialization','flush','fsync','close','final_rehash'])
def test_atomic_final_failure_never_publishes(tmp_path, monkeypatch, phase):
    target = tmp_path/'evaluation.json'
    if phase == 'serialization':
        def broken(value, handle, **kwargs):
            handle.write('{"partial":')
            raise OSError('serialization')
        monkeypatch.setattr(runner.json,'dump',broken)
    elif phase == 'fsync':
        monkeypatch.setattr(runner.os,'fsync',lambda fd: (_ for _ in ()).throw(OSError('fsync')))
    elif phase in ('flush','close'):
        real = runner.tempfile.NamedTemporaryFile
        class Broken:
            def __init__(self, *a, **k): self.file=real(*a,**k)
            def __enter__(self): return self
            def __getattr__(self,name): return getattr(self.file,name)
            def flush(self):
                if phase == 'flush': raise OSError('flush')
                self.file.flush()
            def __exit__(self,*args):
                self.file.close()
                if phase == 'close': raise OSError('close')
        monkeypatch.setattr(runner.tempfile,'NamedTemporaryFile',Broken)
    def recheck():
        if phase == 'final_rehash': raise ValueError('mutated input')
    with pytest.raises((OSError,ValueError)):
        runner.write_json(target,{'rows':[]},before_publish=recheck)
    assert not target.exists()
    assert list(tmp_path.glob('.evaluation.json.*.tmp'))


def test_atomic_no_replace(tmp_path):
    target = tmp_path/'evaluation.json'
    runner.write_json(target,{'original':1})
    with pytest.raises(FileExistsError): runner.write_json(target,{'replacement':2})
    assert d.read_json(target) == {'original':1}


def test_cpu_cannot_claim_runtime_receipt(tmp_path):
    record = dump(tmp_path/'receipt.json',dict(mode='generation_cache_parity',device='cuda',real_runtime=True,runtime={'versions':runtime.VERSIONS},cached_uncached_equal=True,evidence_off_equal=True,nonzero_projection=True,reorder_indices_sha256=['stand-in-not-a-real-receipt']))
    with pytest.raises(ValueError,match='version pin'):
        runtime.verify_runtime({'receipt':d.identity(record)}, {})
    with pytest.raises(KeyError): runtime.verify_runtime({}, {})
    with pytest.raises(ValueError,match='SHA256'):
        runtime.verify_runtime({'receipt':dict(d.identity(record),sha256='0'*64)}, {})


@pytest.mark.parametrize('mutation', ['unbound_dependency','changed_dependency','remote_not_asset','missing_file'])
def test_loaded_closure_negative_standins(tmp_path, monkeypatch, mutation):
    source=tmp_path/'module.py'; source.write_text('fixture = 1\n')
    sha=w.sha256_file(source)
    identity={'imports':{str(source):sha}}
    name='dependency'
    remote_hashes={sha}
    if mutation == 'unbound_dependency': identity['imports']={}
    elif mutation == 'changed_dependency': source.write_text('fixture = 2\n')
    elif mutation == 'remote_not_asset': name='transformers_modules.fixture'; remote_hashes=set()
    else: source.unlink()
    monkeypatch.setattr(runtime,'sys',SimpleNamespace(modules={name:SimpleNamespace(__file__=str(source))}))
    with pytest.raises(ValueError): runtime.verify_loaded_modules(identity,remote_hashes)


def test_asset_tree_rechecks_additions_and_link_target(tmp_path):
    root=tmp_path/'assets'; root.mkdir()
    path=root/'config.json'; path.write_text('{}')
    before=runtime.tree_identity(root)
    (root/'dependency.py').write_text('x=1')
    assert runtime.tree_identity(root) != before
    (root/'dependency.py').unlink()
    assert runtime.tree_identity(root) == before
    external=tmp_path/'external'; external.write_text('{}')
    path.unlink(); path.symlink_to(external)
    assert runtime.tree_identity(root) != before


def test_full_publication_panel_decoder_counts(panel_publication, tmp_path, monkeypatch):
    import test_mvp_finalize as f
    from test_weights_init import WeightsInitTests, modeling, HyenaTransgenicConfig
    from test_mvp_scoring import record
    from safetensors.torch import save_model
    from transgenic.training.b5_runtime import model_kwargs
    from transgenic.training.weights_init import load_weights_only
    from transgenic.training.mvp_cli_runtime import _expected_code_files
    from transgenic.training.mvp_metrics import aggregate_paired_records
    from transgenic.model.tokenization_transgenic import GFFTokenizer
    payload,panel,sequences,manifest,hashes = panel_publication
    case=WeightsInitTests(); case.setUp() # explicitly substitutes only remote Hyena factories
    try:
        recipe=dict(case.recipe,vocab_size=290,max_decoder_position_embeddings=8192,gff_vocab_version='v3',max_label_tokens=8192)
        cfg=dict(model_kwargs(recipe),load_pretrained_encoder=False)
        source=tmp_path/'source'; source.mkdir()
        dump(tmp_path/'run_config.json',dict(case.run_config,recipe=recipe))
        base=modeling.transgenicForConditionalGeneration(HyenaTransgenicConfig(**cfg))
        save_model(base,source/'model.safetensors')
        dump(source/'config.json',base.config.to_dict())
        code=dump(tmp_path/'code.json',dict(files={str(path):w.sha256_file(path) for path in _expected_code_files()},external_packages={name:'test-version' for name in ('accelerate','duckdb','safetensors','torch','transformers')}))
        payload['runs']={}
        for seed in (1,2,3):
            model=modeling.transgenicForConditionalGeneration(HyenaTransgenicConfig(**cfg,evidence_channels=8))
            audit=load_weights_only(model,source)
            with torch.no_grad(): model.get_encoder().evidence_proj.weight.fill_(0.25)
            md=f.metadata('production')
            md['seed_augmentation']={'seed':seed,'aug_seed':20260914+seed}
            md['data_identities']={k:{'sha256':v,'verification':'recorded'} for k,v in hashes.items()}
            md['initialization_audit']={'audit':audit,'verification':'recorded','audit_sha256':hashlib.sha256(f.finalize._canonical_json(audit)).hexdigest()}
            md['sif_code_identity']['code']['sha256']=w.sha256_file(code)
            final=tmp_path/f'final{seed}'
            f.finalize._publish_final(f.LocalAccelerator(),model,aggregate_paired_records([record(1,1)],[1]),final,metadata=md,training_summary=f.summary('production'))
            payload['runs'][str(seed)]={'final':str(final),'code_manifest':str(code)}
        # GPU runtime and expensive generation are explicit CPU stand-ins, never
        # a receipt generated by verify_runtime. All actual decoding gates remain.
        monkeypatch.setattr(runtime,'verify_runtime',lambda *a,**k:[])
        payload.update(runtime_preflight={},encoder_tokenizer={})
        class EncoderTokenizer:
            sep_token_id=2; pad_token_id=4
            def __call__(self, seq, **kwargs): return {'input_ids':torch.tensor([[1]*len(seq)+[2]])}
        tokenizer=GFFTokenizer(vocab_version='v3')
        monkeypatch.setattr(runner,'load_tokenizers',lambda spec:(EncoderTokenizer(),tokenizer,[]))
        seen=[]
        fixture_gsf = '2|CDS1|11|+|A>CDS1<gene>20|CDS1|32|-|A>CDS1'
        def cpu_generate(self, **kw):
            tr=kw['evidence_tracks']; seen.append(tr.detach().clone())
            assert kw['num_beams'] == 2 and kw['max_length'] == 8192
            assert 'max_new_tokens' not in kw and kw['use_cache'] and not kw['do_sample']
            self.get_encoder()(kw['input_ids'], evidence_tracks=tr)
            return torch.tensor([tokenizer(fixture_gsf)['input_ids']])
        monkeypatch.setattr(modeling.transgenicForConditionalGeneration,'generate',cpu_generate)
        out=tmp_path/'decode'
        result=runner.decode_panel(payload,out,['CPU_SYNTHETIC_INTEGRATION_NOT_APPROVAL'])
        assert len(result['predictions']) == len(seen) == 350
        normalized_ids = set()
        for prediction in result['predictions']:
            path = Path(prediction['path'])
            text = path.read_text()
            ids = d.validate_gff(text)
            assert text.count('\tgene\t') == text.count('\tmRNA\t') == 2
            assert not normalized_ids.intersection(ids)
            normalized_ids.update(ids)
            start = prediction['window'][1]
            assert f'\t{start+3}\t{start+11}\t.\t+' in text
            assert f'\t{start+21}\t{start+32}\t.\t-' in text
            raw = d.read_json(path.with_suffix('.raw.json'))
            assert raw['grammar_plan'] == ['<tx1>', '<tx1>']
            assert raw['raw_tokens'][-1] == '</s>'
        for seed in range(3):
            for locus in range(50):
                off,on=seen[seed*100+locus*2:seed*100+locus*2+2]
                assert not off.any()
                assert torch.equal(on,seen[locus*2+1])
        assert not any(t.any() for t in seen[300:])
        assert seen[1][0,12,3] == float(np.float16(np.log1p(70)))
        executable=fake_gffcompare(tmp_path)
        monkeypatch.setattr(p.shutil,'which',lambda _: executable)
        counts=p.produce(out/'evaluation.json',tmp_path/'counts')
        assert len(counts['baseline_reference_counts']) == 50
        assert counts['structural_available'] is False and counts['decision'] is None
        assert len(d.read_json(tmp_path/'counts/receipt.json')['comparisons']) == 350
    finally:
        case.doCleanups()
