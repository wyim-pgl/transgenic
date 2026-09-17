#!/usr/bin/env python3
"""Decode exact 350-row transport from COMPLETE panel inputs and pinned runtime.

--input adds panel_inputs (publication directory), encoder_tokenizer and
runtime_preflight to the producer contract. See MVP_DECODER_INTEGRATION_REPORT.md.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# No accidental network fetch from model's external Hyena config/tokenizer loader.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from transgenic.evidence import mvp_decode as d, mvp_windows as w


def write_json(path, value, *, before_publish=None):
    """Close durable sibling temp before atomic no-replace publication.

    Hard-link publication is the POSIX no-overwrite equivalent of rename. Failed
    temporary files are intentionally retained for diagnostics, never consumed.
    """
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.' + path.name + '.',
                                     suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    if before_publish is not None:
        before_publish()
    os.link(temporary, path)  # atomic, fails if destination already exists
    temporary.unlink()


def verify_runs(payload, producer):
    from transgenic.training.mvp_cli import validate_candidate_records
    from transgenic.training.mvp_cli_runtime import verify_candidate_artifacts
    c = payload['candidate']
    candidate = validate_candidate_records(c['acceptance'], c['acceptance_sha256'], c['inventory'], c['receipts'])
    artifacts = verify_candidate_artifacts(candidate, c['corpus'], c['evidence'], payload['panels'])
    d.require(set(payload['runs']) == {'1','2','3'}, 'exact seeds required')
    runs = {int(s): producer.verify_final(r['final'], int(s), artifacts['valid_rns'], artifacts['sha256'], r['code_manifest']) for s,r in payload['runs'].items()}
    d.require(len({r['code_manifest']['sha256'] for r in runs.values()}) == 1, 'cross-seed code mismatch')
    sources = {tuple(sorted((name, value['sha256']) for name,value in r['initialization'].items())) for r in runs.values()}
    d.require(len(sources) == 1, 'cross-seed initialization mismatch')
    return runs


def sequences_from_fasta(path, panel):
    from Bio import SeqIO
    result = {}
    for record in SeqIO.parse(str(path), 'fasta'):
        for locus in panel:
            if record.id == locus.chromosome:
                d.require(locus.gene_id not in result, 'duplicate FASTA chromosome')
                start,end = w.evaluation_window(locus, len(record))
                result[locus.gene_id] = str(record.seq[start:end])
    d.require(set(result) == {l.gene_id for l in panel}, 'missing FASTA chromosome')
    return result


def load_tokenizers(spec):
    from transformers import AutoTokenizer
    from transgenic.model.tokenization_transgenic import GFFTokenizer
    root = Path(spec['path']).resolve(strict=True)
    d.require(root.is_dir() and spec['files'], 'pinned local encoder tokenizer required')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    d.require(actual == set(spec['files']), 'tokenizer snapshot inventory mismatch')
    identities = []
    for name, digest in spec['files'].items():
        path = root / name
        d.require(path.resolve().is_relative_to(root), 'tokenizer snapshot path escape')
        identities.append(d.checked_file({'path': str(path), 'sha256': digest}))
    encoder = AutoTokenizer.from_pretrained(str(root), local_files_only=True, trust_remote_code=True)
    return encoder, GFFTokenizer(vocab_version='v3'), identities


def decode_panel(payload, out, command):
    import torch
    import mvp_produce_counts as producer
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        panel = w.load_panel(payload['panels'])
        d.require(len(panel) == 50, 'exact 50 loci required')
        producer.frozen_reference(payload['fasta'], payload['reference'])
        runs = verify_runs(payload, producer)
        from mvp_verify_panel_inputs import load_panel_inputs
        from mvp_decode_runtime import verify_runtime
        inputs = [d.identity(payload[k]) for k in ('panels','fasta','reference')]
        if '_request_identity' in payload:
            inputs.append(d.checked_file(payload['_request_identity']))
        inputs.extend(verify_runtime(payload['runtime_preflight'], payload['encoder_tokenizer']))
        files = [Path(__file__).resolve(), Path(d.__file__).resolve(), Path(w.__file__).resolve(),
                 Path(__file__).with_name('mvp_verify_panel_inputs.py'),
                 Path(__file__).with_name('mvp_decode_runtime.py'),
                 Path(__file__).with_name('mvp_prepare_panel_inputs.py'),
                 Path(producer.__file__),
                 ROOT/'src/transgenic/evidence/mvp_tracks.py',
                 ROOT/'src/transgenic/model/tokenization_transgenic.py',
                 ROOT/'src/transgenic/model/constrained_decoding.py',
                 ROOT/'src/transgenic/model/modeling_HyenaTransgenic.py',
                 ROOT/'src/transgenic/model/configuration_transgenic.py',
                 ROOT/'src/transgenic/training/weights_init.py', ROOT/'src/transgenic/training/b5_runtime.py',
                 ROOT/'src/transgenic/utils/gsf.py', ROOT/'src/transgenic/utils/gsf_contract.py',
                 ROOT/'src/transgenic/utils/gsf_grammar.py']
        code_inputs = [d.identity(p) for p in files]
        inputs.extend(code_inputs)
        sequences = sequences_from_fasta(payload['fasta'], panel)
        panel_root = Path(payload['panel_inputs']).resolve(strict=True)
        tracks, panel_records = load_panel_inputs(panel_root, payload, panel, sequences)
        inputs.extend(panel_records)
        encoder, tokenizer, tokenizer_files = load_tokenizers(payload['encoder_tokenizer'])
        inputs.extend(tokenizer_files)
        device = payload.get('device','cpu')
        predictions, audits = [], []
        for seed in (1,2,3,123):
            if seed == 123:
                model, audit = d.load_baseline_model(runs[1], device)
                checkpoint = runs[1]['initialization']['model.safetensors']['sha256']
                write_json(out / 'baseline_initialization.json', audit)
                inputs.append(d.identity(out / 'baseline_initialization.json'))
            else:
                model = d.load_final_model(runs[seed], device)
                checkpoint = runs[seed]['files']['model.safetensors']['sha256']
            inputs.extend(verify_runtime(payload['runtime_preflight'], payload['encoder_tokenizer'], model=model, tokenizer=encoder))
            for index,locus in enumerate(panel):
                ids,mask = d.encode_window(sequences[locus.gene_id], encoder)
                ids,mask = ids.to(device),mask.to(device)
                on = tracks[locus.gene_id].to(device)
                for fraction in ((0,) if seed == 123 else (0,1)):
                    stem = f's{seed}.f{fraction}.p{index:02d}'
                    # Persist even malformed/truncated output before tokenizer cleanup.
                    raw = d.generate_one(model, tokenizer, ids, mask, on if fraction else torch.zeros_like(on),
                        raw_sink=lambda value: write_json(out / (stem+'.raw.json'), value))
                    start,end = w.evaluation_window(locus)
                    text = d.gsf_gff(raw['gsf'], locus.chromosome, start, stem)
                    path = out / (stem+'.gff3')
                    with path.open('x') as handle:
                        handle.write(text)
                    predictions.append(dict(seed=seed, fraction=fraction, gene_id=locus.gene_id,
                        window=[locus.chromosome,start,end], checkpoint_sha256=checkpoint, **d.identity(path)))
                    audits.append(d.identity(out / (stem+'.raw.json')))
            del model
        checkpoints = {s:r['files']['model.safetensors']['sha256'] for s,r in runs.items()}
        checkpoints[123] = runs[1]['initialization']['model.safetensors']['sha256']
        producer.prediction_inventory(predictions, panel, checkpoints)
        runtime_records = verify_runtime(payload['runtime_preflight'], payload['encoder_tokenizer'])
        inputs.extend(runtime_records)
        def final_recheck():
            # Includes COMPLETE, every FASTA/npy/JSONL, request/code and live imports.
            from mvp_prepare_panel_inputs import unchanged
            d.require(Path(payload['panel_inputs']).resolve(strict=True) == panel_root, 'panel directory identity changed')
            unchanged(inputs + audits)
            for prediction in predictions:
                d.checked_file(prediction)
            d.require(verify_runs(payload, producer) == runs, 'run inputs mutated during decoding')
            d.require(verify_runtime(payload['runtime_preflight'], payload['encoder_tokenizer']) == runtime_records,
                      'runtime import closure changed before publication')
        final_recheck()
        transport = {k:payload[k] for k in ('panels','fasta','reference','candidate','runs')}
        transport.update(decoder={'files':{r['path']:r['sha256'] for r in code_inputs}, 'commands':[command]}, predictions=predictions)
        write_json(out/'decode_audit.json', dict(status='DECODE_COMPLETE_NOT_APPROVAL', label='exploratory / MVP', inputs=inputs,
            raw_outputs=audits, generation=dict(num_beams=2, do_sample=False, use_cache=True, max_length=8192),
            panel_input_contract='COMPLETE_schema_1_est_mvp_panel_inputs_not_training',
            runtime_preflight=payload['runtime_preflight']))
        # Last file only: never publish transport on a partial/failed decode.
        inputs.append(d.identity(out/'decode_audit.json'))
        write_json(out/'evaluation.json', transport, before_publish=final_recheck)
        return transport
    except Exception as error:
        write_json(out/'FAILED.json', {'error':str(error), 'status':'PARTIAL_NOT_A_RECEIPT'})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    import mvp_produce_counts as producer
    request_identity = d.identity(args.input)
    payload = producer.read_json(args.input)
    payload['_request_identity'] = request_identity
    decode_panel(payload, args.out, [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
