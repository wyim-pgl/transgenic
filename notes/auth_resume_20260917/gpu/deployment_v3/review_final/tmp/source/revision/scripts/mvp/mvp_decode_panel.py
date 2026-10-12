#!/usr/bin/env python3
"""Decode legacy 350 or explicit development-single 150 role-qualified rows.

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
    from transgenic.training.mvp_profiles import DEVELOPMENT_PROFILE
    runs = producer.verify_run_inventory(payload, artifacts)
    readiness_inputs = producer.verify_evaluation_readiness(payload, candidate, runs)
    if readiness_inputs:
        runs['mvp_seed123']['readiness_inputs'] = readiness_inputs
    return runs if producer.evaluation_profile(payload) == DEVELOPMENT_PROFILE else {int(s): r for s, r in runs.items()}


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
        from transgenic.training.mvp_profiles import DEVELOPMENT_PROFILE
        profile = producer.evaluation_profile(payload)
        development = profile == DEVELOPMENT_PROFILE
        from mvp_verify_panel_inputs import load_panel_inputs
        from mvp_decode_runtime import verify_runtime, require_cuda_device
        device = require_cuda_device(payload.get('device', 'cuda'))
        inputs = [d.identity(payload[k]) for k in ('panels','fasta','reference')]
        if '_request_identity' in payload:
            inputs.append(d.checked_file(payload['_request_identity']))
        if development:
            inputs.extend(runs['mvp_seed123']['readiness_inputs'])
            inputs.append(runs['b5_seed123']['weights'])
        inputs.extend(verify_runtime(payload['runtime_preflight'], payload['encoder_tokenizer']))
        files = [Path(__file__).resolve(), Path(d.__file__).resolve(), Path(w.__file__).resolve(),
                 Path(__file__).with_name('mvp_verify_panel_inputs.py'),
                 Path(__file__).with_name('mvp_panel_compatibility.py'),
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
        predictions, audits = [], []
        jobs = [('mvp_seed123', 123, 'mvp_finetune'), ('b5_seed123', 123, 'b5_baseline')] if development else [
            (s, s, 'mvp_finetune') for s in (1, 2, 3)] + [(123, 123, 'b5_baseline')]
        for run_id, seed, role in jobs:
            if role == 'b5_baseline':
                source_run = runs['mvp_seed123'] if development else runs[1]
                model, audit = d.load_baseline_model(source_run, device)
                checkpoint = source_run['initialization']['model.safetensors']['sha256']
                write_json(out / 'baseline_initialization.json', audit)
                inputs.append(d.identity(out / 'baseline_initialization.json'))
            else:
                model = d.load_final_model(runs[run_id], device)
                checkpoint = runs[run_id]['files']['model.safetensors']['sha256']
            inputs.extend(verify_runtime(payload['runtime_preflight'], payload['encoder_tokenizer'], model=model, tokenizer=encoder))
            for index,locus in enumerate(panel):
                ids,mask = d.encode_window(sequences[locus.gene_id], encoder)
                ids,mask = ids.to(device),mask.to(device)
                on = tracks[locus.gene_id].to(device)
                for fraction in ((0,) if role == 'b5_baseline' else (0,1)):
                    stem = f'{run_id}.f{fraction}.p{index:02d}' if development else f's{seed}.f{fraction}.p{index:02d}'
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
                    if development:
                        predictions[-1].update(profile=profile, run_id=run_id, run_role=role)
                    audits.append(d.identity(out / (stem+'.raw.json')))
            del model
        if development:
            checkpoints = {'mvp_seed123': runs['mvp_seed123']['files']['model.safetensors']['sha256'],
                           'b5_seed123': runs['b5_seed123']['weights']['sha256']}
        else:
            checkpoints = {s:r['files']['model.safetensors']['sha256'] for s,r in runs.items()}
            checkpoints[123] = runs[1]['initialization']['model.safetensors']['sha256']
        producer.prediction_inventory(predictions, panel, checkpoints, profile=profile)
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
        for key in ('profile', 'metadata', 'panel_compatibility'):
            if key in payload:
                transport[key] = payload[key]
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
    from transgenic.training.mvp_profiles import LEGACY_PROFILE, DEVELOPMENT_PROFILE
    parser.add_argument('--profile', choices=(LEGACY_PROFILE, DEVELOPMENT_PROFILE))
    args = parser.parse_args(argv)
    import mvp_produce_counts as producer
    request_identity = d.identity(args.input)
    payload = producer.read_json(args.input)
    producer.evaluation_profile(payload, args.profile)
    payload['_request_identity'] = request_identity
    decode_panel(payload, args.out, [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
