#!/usr/bin/env python3
"""Decode exact 350-row MVP transport. Internal panel-input adapter; no approval receipt.

--input contains producer contract fields plus panel_inputs, panel_input_pins,
and encoder_tokenizer:{path,files:{relative_path:sha256}}. Snapshot must be local.
Requires the independently reviewed producer verifier at integration time.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# No accidental network fetch from model's external Hyena config/tokenizer loader.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from transgenic.evidence import mvp_decode as d, mvp_windows as w


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write('\n')


def verify_runs(payload, producer):
    # Controller reported incomplete checks in this exact base implementation.
    # A different hash is NOT approval; integration still requires owner review.
    d.require(d.identity(producer.__file__)['sha256'] !=
              'a3b731c217e2da92eb044e1c7815a310fbd78b18a707487b12eb20a897576f1a',
              'known-incomplete producer verifier: integrate reviewed owner fixes first')
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
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        panel = w.load_panel(payload['panels'])
        d.require(len(panel) == 50, 'exact 50 loci required')
        producer.frozen_reference(payload['fasta'], payload['reference'])
        runs = verify_runs(payload, producer)
        inputs = [d.identity(payload[k]) for k in ('panels','fasta','reference','panel_inputs')]
        files = [Path(__file__).resolve(), Path(d.__file__).resolve(), Path(w.__file__).resolve(),
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
        pins = payload['panel_input_pins']
        d.require(set(pins) == {'evidence_db','panels','fasta','rasterizer_code'}, 'exact internal provenance pins required')
        for key,path in (('evidence_db',payload['candidate']['evidence']),('panels',payload['panels']),('fasta',payload['fasta'])):
            d.require(d.checked_file(pins[key]) == d.identity(path), 'worker input identity mismatch')
        inputs.extend(d.checked_file(record) for record in pins.values())
        rasterizer = ROOT / 'src/transgenic/evidence/mvp_tracks.py'
        d.require(d.checked_file(pins['rasterizer_code'])['sha256'] == d.identity(rasterizer)['sha256'], 'rasterizer code mismatch')
        batch = producer.read_json(payload['panel_inputs'])
        tracks = d.load_panel_batch(batch, panel, sequences, pins)
        inputs.extend(d.checked_file(r['tracks']) for r in batch['rows'])
        encoder, tokenizer, tokenizer_files = load_tokenizers(payload['encoder_tokenizer'])
        inputs.extend(tokenizer_files)
        device = payload.get('device','cpu')
        predictions, audits = [], []
        for seed in (1,2,3,123):
            if seed == 123:
                model, audit = d.load_baseline_model(runs[1], device)
                checkpoint = runs[1]['initialization']['model.safetensors']['sha256']
                write_json(out / 'baseline_initialization.json', audit)
            else:
                model = d.load_final_model(runs[seed], device)
                checkpoint = runs[seed]['files']['model.safetensors']['sha256']
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
        for record in inputs:
            d.checked_file(record)
        # Recheck producer publication inputs after the long decode, not only at entry.
        d.require(verify_runs(payload, producer) == runs, 'run inputs mutated during decoding')
        transport = {k:payload[k] for k in ('panels','fasta','reference','candidate','runs')}
        transport.update(decoder={'files':{r['path']:r['sha256'] for r in code_inputs}, 'commands':[command]}, predictions=predictions)
        write_json(out/'decode_audit.json', dict(status='DECODE_COMPLETE_NOT_APPROVAL', label='exploratory / MVP', inputs=inputs,
            raw_outputs=audits, generation=dict(num_beams=2, do_sample=False, use_cache=True, max_length=8192),
            panel_input_contract='INTERNAL_PENDING_CONTROLLER_INTEGRATION'))
        # Last file only: never publish transport on a partial/failed decode.
        write_json(out/'evaluation.json', transport)
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
    decode_panel(producer.read_json(args.input), args.out, [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
