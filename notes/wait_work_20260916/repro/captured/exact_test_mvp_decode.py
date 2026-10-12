"""Decoder boundary tests; CPU only, no claim about production GPU runtime."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from transgenic.evidence import mvp_decode as d
from transgenic.model.tokenization_transgenic import GFFTokenizer


def test_raw_empty_and_truncation():
    tok = GFFTokenizer(vocab_version='v3')
    ids = tok('<empty>')['input_ids']
    result = d.validate_decode(ids, tok)
    assert result['gsf'] == '<empty>'
    assert d.gsf_gff(result['gsf'], 'Chr1', 100, 'p') == '##gff-version 3\n'
    for bad in (ids[:-1], [], ids + [tok.eos_token_id]):
        with pytest.raises(ValueError):
            d.validate_decode(bad, tok)


def test_multi_gene_minus_deterministic_and_no_dedup():
    gsf = '2|CDS1|11|+|A>CDS1<gene>20|CDS1|32|-|A>CDS1'
    out = d.gsf_gff(gsf, 'Chr1', 100, 'p')
    assert out == d.gsf_gff(gsf, 'Chr1', 100, 'p')
    assert '\t103\t111\t.\t+' in out
    assert '\t121\t132\t.\t-' in out
    assert out.count('\tgene\t') == 2
    assert out.count('\tmRNA\t') == 2
    for bad in ('', ' ', '2|CDS1|11|+|A>CDS1;CDS1', '2|CDS1|2|+|A>CDS1'):
        with pytest.raises(ValueError):
            d.gsf_gff(bad, 'Chr1', 100, 'p')


def test_generation_kwargs_tracks_and_rejection():
    tok = GFFTokenizer(vocab_version='v3')
    class Encoder(torch.nn.Module):
        def forward(self, ids, evidence_tracks=None): return ids
    class Model:
        encoder = Encoder()
        drop_tracks = False
        def get_encoder(self): return self.encoder
        config = SimpleNamespace(decoder_start_token_id=tok.bos_token_id,
                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        def eval(self): return self
        def generate(self, **kwargs):
            self.kwargs = kwargs
            self.encoder(kwargs['input_ids'], evidence_tracks=None if self.drop_tracks else kwargs['evidence_tracks'])
            return torch.tensor([tok('<empty>')['input_ids']])
    model = Model()
    ids = torch.ones((1,30720), dtype=torch.long)
    tracks = torch.ones((1,30720,8), dtype=torch.float16)
    d.generate_one(model, tok, ids, ids.bool(), tracks)
    assert model.kwargs['max_length'] == 8192
    assert 'max_new_tokens' not in model.kwargs
    assert model.kwargs['num_beams'] == 2
    assert model.kwargs['do_sample'] is False and model.kwargs['use_cache'] is True
    assert model.kwargs['evidence_tracks'] is tracks
    with pytest.raises(ValueError):
        d.generate_one(model, tok, ids, ids.bool(), tracks[:,:,:7])
    model.drop_tracks = True
    with pytest.raises(ValueError,match='forward evidence'):
        d.generate_one(model, tok, ids, ids.bool(), tracks)


def test_sequence_sep_and_no_rc():
    class Tok:
        sep_token_id = 9
        pad_token_id = 0
        def __call__(self, seq, **kwargs):
            assert seq == 'A' * 30720
            return {'input_ids': torch.tensor([[1]*30720+[9]])}
    ids, mask = d.encode_window('A'*30720, Tok())
    assert ids.shape == mask.shape == (1,30720)


def test_normalize_rejects_bad_identifiers():
    for text in ('', ' \n', '##gff-version 3\nChr1\tx\tgene\t1\t3\t.\t+\t.\tID=t \n'):
        with pytest.raises(ValueError):
            d.validate_gff(text)
    d.validate_gff('##gff-version 3\n')


def test_raw_planning_validation_and_beam_reorder():
    from transgenic.model.constrained_decoding import GSFGrammarLogitsProcessor
    tok = GFFTokenizer(vocab_version='v3')
    gsf = '2|CDS1|11|+|A>CDS1<gene>20|CDS1|32|-|A>CDS1'
    ids = tok(gsf)['input_ids']
    result = d.validate_decode(ids, tok)
    assert result['grammar_plan'] == ['<tx1>','<tx1>']
    assert result['raw_token_ids'] == ids and result['gsf'] == gsf
    bad = list(ids)
    bad[bad.index(tok.vocab['<tx1>'])] = tok.vocab['<tx2>']
    with pytest.raises(ValueError):
        d.validate_decode(bad, tok)
    processor = GSFGrammarLogitsProcessor(tok, [30720], num_beams=2, v2=True, v3=True)
    prefixes = torch.tensor([[0, tok.vocab['2']], [0,tok.vocab['<empty>']]])
    scores = torch.zeros((2,290))
    expected = processor(prefixes, scores)
    actual = processor(prefixes.flip(0), scores)
    assert torch.equal(expected.flip(0), actual)


def test_strict_final_load_preserves_projection_and_aliases(tmp_path, monkeypatch):
    from safetensors.torch import save_model, save_file, load_file
    from test_mvp_model import tiny_config, TinyHyena, modeling
    monkeypatch.setattr(modeling.AutoConfig, 'from_pretrained', lambda *a,**k: SimpleNamespace())
    monkeypatch.setattr(modeling.AutoModel, 'from_config', lambda c,**k: TinyHyena(c))
    config = tiny_config(evidence_channels=8)
    config.vocab_size = 290
    config.max_decoder_position_embeddings = 8192
    model = modeling.transgenicForConditionalGeneration(config).eval()
    with torch.no_grad():
        model.get_encoder().evidence_proj.weight.copy_(torch.eye(8))
    config.save_pretrained(tmp_path)
    save_model(model, str(tmp_path/'model.safetensors'))
    run = {'files': {n:d.identity(tmp_path/n) for n in ('config.json','model.safetensors')}}
    loaded = d.load_final_model(run)
    assert torch.equal(loaded.get_encoder().evidence_proj.weight, torch.eye(8))
    assert loaded.lm_head.weight is loaded.transgenic.decoder_embed_tokens.weight
    for key,value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
    metadata = tmp_path/'metadata.json'
    metadata.write_text(json.dumps({'initialization_audit':{'audit':{'target_semantics':{'fields':{'d_model':{'value':999}}}}}}))
    run['files']['metadata.json'] = d.identity(metadata)
    with pytest.raises(ValueError,match='config/audit'):
        d.load_final_model(run)
    del run['files']['metadata.json']
    state = load_file(str(tmp_path/'model.safetensors'))
    del state['transgenic.encoder.evidence_proj.weight']
    save_file(state, str(tmp_path/'model.safetensors'))
    run['files']['model.safetensors'] = d.identity(tmp_path/'model.safetensors')
    with pytest.raises(RuntimeError):
        d.load_final_model(run)


def test_tiny_real_beam_cache_tracks_and_negative(tmp_path, monkeypatch):
    from test_mvp_model import tiny_config, TinyHyena, modeling
    monkeypatch.setattr(modeling.AutoConfig, 'from_pretrained', lambda *a,**k: SimpleNamespace())
    monkeypatch.setattr(modeling.AutoModel, 'from_config', lambda c,**k: TinyHyena(c))
    model = modeling.transgenicForConditionalGeneration(tiny_config(evidence_channels=8)).eval()
    with torch.no_grad():
        model.get_encoder().evidence_proj.weight.copy_(torch.eye(8))
    ids = torch.tensor([[3,4,5,6]*6])
    tracks = torch.rand(1,24,8)
    seen = []
    hook = model.get_encoder().register_forward_pre_hook(lambda m,a,k: seen.append(k.get('evidence_tracks')), with_kwargs=True)
    kwargs = dict(input_ids=ids,evidence_tracks=tracks,num_beams=2,do_sample=False,max_length=8,eos_token_id=None)
    with torch.inference_mode():
        a = model.generate(**kwargs,use_cache=True)
        b = model.generate(**kwargs,use_cache=False)
    hook.remove()
    assert torch.equal(a,b) and len(seen) == 2 and all(t is tracks for t in seen)
    with pytest.raises(ValueError):
        model.generate(**dict(kwargs,evidence_tracks=tracks[:,:,:7]))


def test_runner_exact_350_no_overwrite_and_partial_failure(tmp_path, monkeypatch):
    import importlib.util
    import sys
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('decode_runner_test',root/'revision/scripts/mvp/mvp_decode_panel.py')
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    import mvp_produce_counts as producer
    panel_path = root/'revision/protocols/b6_panels_v1.tsv'
    panel = runner.w.load_panel(panel_path)
    pin = tmp_path/'pin'; pin.write_text('fake CPU fixture')
    inp = tmp_path/'batch.json'; inp.write_text('{"rows":[]}')
    runs = {s: {'files':{'model.safetensors':d.identity(pin)},'initialization':{'model.safetensors':d.identity(pin)}} for s in (1,2,3)}
    payload = dict(panels=str(panel_path),fasta=str(pin),reference=str(pin),panel_inputs=str(inp),
        candidate={'evidence':str(pin)},runs={},encoder_tokenizer={},runtime_preflight={})
    monkeypatch.setattr(producer,'frozen_reference',lambda *a: None)
    monkeypatch.setattr(runner,'verify_runs',lambda *a:runs)
    monkeypatch.setattr(runner,'sequences_from_fasta',lambda *a:{p.gene_id:'A'*30720 for p in panel})
    import mvp_verify_panel_inputs as adapter
    import mvp_decode_runtime as runtime
    monkeypatch.setattr(adapter,'load_panel_inputs',lambda *a:({p.gene_id:torch.ones((1,30720,8),dtype=torch.float16) for p in panel}, []))
    monkeypatch.setattr(runtime,'verify_runtime',lambda *a,**k:[]) # explicit CPU orchestration stand-in
    monkeypatch.setattr(runner,'load_tokenizers',lambda *a:(None,None,[]))
    monkeypatch.setattr(d,'load_final_model',lambda *a:object())
    monkeypatch.setattr(d,'load_baseline_model',lambda *a:(object(),{}))
    monkeypatch.setattr(d,'encode_window',lambda *a:(torch.ones((1,30720),dtype=torch.long),torch.ones((1,30720),dtype=torch.bool)))
    def generated(*a,raw_sink):
        raw = dict(gsf='<empty>',raw_token_ids=[0,289,2],grammar_plan=[])
        raw_sink(raw)
        return raw
    monkeypatch.setattr(d,'generate_one',generated)
    out = tmp_path/'success'
    transport = runner.decode_panel(payload,out,['python','fixture'])
    assert len(transport['predictions']) == 350
    assert len({row['path'] for row in transport['predictions']}) == 350
    assert len(list(out.glob('*.gff3'))) == len(list(out.glob('*.raw.json'))) == 350
    assert all(Path(r['path']).read_text() == '##gff-version 3\n' for r in transport['predictions'])
    with pytest.raises(FileExistsError): runner.decode_panel(payload,out,['python','fixture'])
    def fail(*a,**k): raise ValueError('truncated')
    monkeypatch.setattr(d,'generate_one',fail)
    out = tmp_path/'failure'
    with pytest.raises(ValueError,match='truncated'): runner.decode_panel(payload,out,['python','fixture'])
    assert (out/'FAILED.json').exists() and not (out/'evaluation.json').exists()
    monkeypatch.setattr(d,'generate_one',generated)
    real_dump=runner.json.dump
    def fail_final(value, handle, **kwargs):
        if Path(handle.name).name.startswith('.evaluation.json.'):
            handle.write('{"predictions":[')
            raise OSError('injected final serialization')
        return real_dump(value,handle,**kwargs)
    monkeypatch.setattr(runner.json,'dump',fail_final)
    out=tmp_path/'final-write-failure'
    with pytest.raises(OSError,match='final serialization'):
        runner.decode_panel(payload,out,['python','fixture'])
    assert not (out/'evaluation.json').exists() and (out/'FAILED.json').exists()
    assert list(out.glob('.evaluation.json.*.tmp'))
    def mutate_after_serialization(value, handle, **kwargs):
        result=real_dump(value,handle,**kwargs)
        if Path(handle.name).name.startswith('.evaluation.json.'):
            pin.write_text('changed after the initial final_recheck')
        return result
    monkeypatch.setattr(runner.json,'dump',mutate_after_serialization)
    out=tmp_path/'late-input-mutation'
    with pytest.raises(ValueError,match='mutated'):
        runner.decode_panel(payload,out,['python','fixture'])
    assert not (out/'evaluation.json').exists() and (out/'FAILED.json').exists()


def test_baseline_approved_loader_actual_roundtrip(tmp_path):
    from test_weights_init import WeightsInitTests, modeling, HyenaTransgenicConfig
    from transgenic.training.b5_runtime import model_kwargs
    from safetensors.torch import save_model
    fixture = WeightsInitTests()
    fixture.setUp()
    try:
        recipe = dict(fixture.recipe,vocab_size=290,max_decoder_position_embeddings=8192,gff_vocab_version='v3',max_label_tokens=8192)
        run_config = dict(fixture.run_config,recipe=recipe)
        fixture._write_json(fixture.run/'run_config.json',run_config)
        cfg = dict(model_kwargs(recipe),load_pretrained_encoder=False)
        source = modeling.transgenicForConditionalGeneration(HyenaTransgenicConfig(**cfg)).eval()
        save_model(source, fixture.epoch/'model.safetensors')
        final = modeling.transgenicForConditionalGeneration(HyenaTransgenicConfig(**cfg,evidence_channels=8)).eval()
        with torch.no_grad(): final.get_encoder().evidence_proj.weight.fill_(0.25)
        final.config.save_pretrained(tmp_path)
        save_model(final,tmp_path/'model.safetensors')
        run = dict(files={n:d.identity(tmp_path/n) for n in ('config.json','model.safetensors')},
                   initialization={'model.safetensors':d.identity(fixture.epoch/'model.safetensors'),'run_config.json':d.identity(fixture.run/'run_config.json')})
        baseline,audit = d.load_baseline_model(run)
        assert baseline.get_encoder().evidence_proj.weight.count_nonzero() == 0
        for key,value in source.state_dict().items():
            assert torch.equal(value,baseline.state_dict()[key])
        assert audit['resolved_weights_path'] == str(fixture.epoch/'model.safetensors')
    finally:
        fixture.doCleanups()


def test_reviewed_verifier_requires_actual_candidate_records():
    import mvp_decode_panel as runner
    import mvp_produce_counts as producer
    with pytest.raises(KeyError, match='candidate'):
        runner.verify_runs({}, producer)
