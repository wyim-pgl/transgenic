"""Fail-closed MVP decode boundaries. Panel-input schema is INTERNAL, not approval."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

from . import mvp_windows as w

LENGTH = 30720
MAX_LENGTH = 8192


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, f'duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def identity(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=w.sha256_file(path), size=path.stat().st_size)


def checked_file(record):
    actual = identity(record['path'])
    require(actual['sha256'] == record['sha256'], 'input SHA256 mismatch')
    if 'size' in record:
        require(actual['size'] == record['size'], 'input size mismatch')
    return actual


def load_panel_batch(payload, panel, sequences, pins):
    """Normalize internal worker output; caller supplies independently checked pins.

    payload: {schema, provenance, rows}; rows: gene_id, window,
    sequence_sha256, tracks:{path,sha256,size}, dtype, shape. No ONT/HiFi.
    pins: exact identities for evidence_db, panels, fasta, rasterizer_code;
    provenance also carries population=excluded_heldout, keep_fraction=1, is_rc=False.
    This is byte/shape validation, NOT a scientific approval receipt.
    """
    require(payload['schema'] == 'mvp_panel_batch_internal_v1', 'unknown internal schema')
    expected = dict(pins, population='excluded_heldout', keep_fraction=1, is_rc=False)
    require(payload['provenance'] == expected, 'panel evidence provenance mismatch')
    require(type(payload['provenance']['keep_fraction']) is int and payload['provenance']['is_rc'] is False, 'invalid fraction/orientation types')
    for record in pins.values():
        checked_file(record)
    rows = payload['rows']
    require(len(rows) == len(panel) == 50, 'exact 50 panel inputs required')
    by_id = {r['gene_id']: r for r in rows}
    require(len(by_id) == 50 and set(by_id) == {p.gene_id for p in panel}, 'panel input inventory mismatch')
    result = {}
    for locus in panel:
        row = by_id[locus.gene_id]
        start, end = w.evaluation_window(locus)
        seq = sequences[locus.gene_id]
        require(row['window'] == [locus.chromosome, start, end] and all(type(v) is int for v in row['window'][1:]), 'input window mismatch')
        require(len(seq) == LENGTH and row['sequence_sha256'] == hashlib.sha256(seq.encode('ascii')).hexdigest(), 'sequence mismatch')
        require(row['dtype'] == 'float16' and row['shape'] == [LENGTH, 8], 'track metadata mismatch')
        checked_file(row['tracks'])
        tracks = np.load(row['tracks']['path'], allow_pickle=False)
        require(tracks.dtype == np.float16 and tracks.shape == (LENGTH, 8), 'track array mismatch')
        require(np.isfinite(tracks).all() and (tracks >= 0).all(), 'invalid track values')
        masked = np.frombuffer(seq.upper().encode('ascii'), dtype='S1') == b'N'
        require(not tracks[masked].any(), 'N bases must have zero evidence')
        require((tracks[~masked, 7] == 1).all(), 'on availability must be one')
        require((tracks[:, :7] <= np.float16(np.log1p(64))).all(), 'track count cap exceeded')
        result[locus.gene_id] = torch.from_numpy(tracks.copy()).unsqueeze(0)
    return result


def encode_window(sequence, tokenizer):
    require(len(sequence) == LENGTH, 'evaluation sequence must be 30720 genomic-forward bases')
    ids = tokenizer(sequence, return_tensors='pt', truncation=False)['input_ids']
    require(ids.shape == (1, LENGTH + 1), 'encoder must yield one base per token plus SEP')
    require(tokenizer.sep_token_id is not None and ids[0,-1].item() == tokenizer.sep_token_id, 'missing trailing SEP')
    ids = ids[:, :-1]
    return ids, ids != tokenizer.pad_token_id


def validate_decode(ids, tokenizer):
    from transgenic.utils import gsf_contract as gc, gsf_grammar
    ids = list(ids)
    require(3 <= len(ids) <= MAX_LENGTH and all(type(i) is int for i in ids), 'invalid or truncated raw IDs')
    vocab = tokenizer.get_vocab()
    inverse = {v:k for k,v in vocab.items()}
    require(all(i in inverse for i in ids), 'unknown token ID')
    tokens = [inverse[i] for i in ids]
    require(tokens[0] == '<s>' and tokens[-1] == '</s>' and '</s>' not in tokens[1:-1], 'missing/extra EOS or invalid start')
    for i in range(1,len(tokens)):
        require(tokens[i] in gsf_grammar.allowed_next(tokens[:i], LENGTH, v2=True, v3=True), f'grammar violation at token {i}')
    gsf = tokenizer.convert_tokens_to_string(tokens[1:-1])
    gc.check_caps_v3(gsf, window_len=LENGTH)
    # Compare, never replace: canonicalization merges duplicate isoforms.
    require(gc.canonicalize_v3(gsf) == gsf, 'noncanonical GSF (including duplicate isoforms)')
    return dict(raw_token_ids=ids, raw_tokens=tokens, grammar_plan=[t for t in tokens if re.fullmatch(r'<tx\d+>', t)], gsf=gsf)


def validate_gff(text):
    require(text.startswith('##gff-version 3\n'), 'completed GFF requires exact header')
    ids, parents = {}, []
    for line in text.splitlines()[1:]:
        require(bool(line) and not line.startswith('#'), 'unexpected empty/comment GFF row')
        cols = line.split('\t')
        require(len(cols) == 9, 'invalid GFF columns')
        attrs = {}
        for item in cols[8].split(';'):
            require('=' in item, 'invalid GFF attribute')
            key, value = item.split('=', 1)
            require(key not in attrs and re.fullmatch(r'[A-Za-z0-9_.:-]+', value), 'noncanonical GFF ID/attribute')
            attrs[key] = value
        require('ID' in attrs and attrs['ID'] not in ids, 'missing/duplicate GFF ID')
        require(cols[2] in ('gene','mRNA','exon','CDS','five_prime_UTR','three_prime_UTR'), 'unsupported GFF feature')
        require(cols[6] in ('+','-') and 1 <= int(cols[3]) <= int(cols[4]), 'invalid genomic coordinates/strand')
        ids[attrs['ID']] = cols
        if cols[2] == 'gene':
            require('Parent' not in attrs, 'gene cannot have Parent')
        else:
            require('Parent' in attrs, 'missing Parent')
            parents.append((attrs['ID'],attrs['Parent']))
    for child, parent in parents:
        require(parent in ids, 'missing parent ID')
        c, p = ids[child], ids[parent]
        require(p[2] == ('gene' if c[2] == 'mRNA' else 'mRNA'), 'invalid hierarchy')
        require(c[0] == p[0] and c[6] == p[6] and int(p[3]) <= int(c[3]) <= int(c[4]) <= int(p[4]), 'parent containment mismatch')
    return ids


def gsf_gff(gsf, chromosome, start, prefix):
    from transgenic.utils import gsf_contract as gc
    from transgenic.utils.gsf import gffString2GFF3
    require(re.fullmatch(r'[A-Za-z0-9_.:-]+', prefix) and re.fullmatch(r'[A-Za-z0-9_.:-]+', chromosome), 'unsafe identifier')
    try:
        gc.check_caps_v3(gsf, window_len=LENGTH)
        require(gc.canonicalize_v3(gsf) == gsf, 'noncanonical/duplicate GSF')
        lines = []
        for n, block in enumerate(gc.split_v3(gsf), 1):
            features, txs, strand = gc._parse(block)
            require(strand in ('+','-') and all(0 <= s < e <= LENGTH and (kind != 'CDS' or e-s >= 3) for kind,s,e,ph in features.values()), 'invalid feature coordinates')
            rows = gffString2GFF3(block, chromosome, start, 'source=mvp')
            require(all(rows) and sum('\tmRNA\t' in row for row in rows) == len(txs), 'legacy converter dropped transcript')
            old = rows[0].split('\t')[8].split(';')[0].removeprefix('ID=')
            new = f'{prefix}.g{n}'
            for row in rows:
                cols = row.split('\t')
                attrs = cols[8].split(';')
                cols[8] = ';'.join(k+'='+ (new+v[len(old):] if k in ('ID','Parent') and (v == old or v.startswith(old+'.')) else v) for k,v in (a.split('=',1) for a in attrs))
                lines.append('\t'.join(cols))
        text = '##gff-version 3\n' + ''.join(line+'\n' for line in lines)
        validate_gff(text)
        return text
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError('malformed GSF') from error


def generate_one(model, tokenizer, input_ids, attention_mask, tracks, *, raw_sink=None):
    from transformers import GenerationConfig, LogitsProcessorList
    from transgenic.model.constrained_decoding import GSFGrammarLogitsProcessor
    require(input_ids.shape == attention_mask.shape == (1,LENGTH), 'invalid sequence batch')
    require(tracks.shape == (1,LENGTH,8) and tracks.is_floating_point() and torch.isfinite(tracks).all(), 'invalid evidence batch')
    require(tracks.device == input_ids.device == attention_mask.device, 'batch device mismatch')
    processor = GSFGrammarLogitsProcessor(tokenizer, [LENGTH], num_beams=2, v2=True, v3=True)
    model.eval()
    received = []
    hook = model.get_encoder().register_forward_pre_hook(
        lambda module, args, kwargs: received.append(kwargs.get('evidence_tracks')), with_kwargs=True)
    try:
        with torch.inference_mode():
            output = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                evidence_tracks=tracks, decoder_input_ids=torch.tensor([[tokenizer.get_vocab()['<s>']]], device=input_ids.device),
                decoder_start_token_id=tokenizer.get_vocab()['<s>'],
                num_beams=2, do_sample=False, use_cache=True, max_length=MAX_LENGTH,
                generation_config=GenerationConfig(), num_return_sequences=1,
                eos_token_id=tokenizer.get_vocab()['</s>'], pad_token_id=tokenizer.get_vocab()['<pad>'],
                logits_processor=LogitsProcessorList([processor]))
    finally:
        hook.remove()
    require(len(received) == 1 and received[0] is not None and torch.equal(received[0], tracks),
            'generation did not forward evidence exactly once to encoder')
    require(output.ndim == 2 and output.shape[0] == 1, 'expected one top-beam sequence')
    ids = output[0].cpu().tolist()
    if raw_sink is not None:
        inverse = {v:k for k,v in tokenizer.get_vocab().items()}
        tokens = [inverse.get(i, '<unknown-ID>') for i in ids]
        raw_sink(dict(raw_token_ids=ids, raw_tokens=tokens,
                      grammar_plan=[t for t in tokens if re.fullmatch(r'<tx\d+>', t)]))
    return validate_decode(ids, tokenizer)


def load_final_model(verified_run, device='cpu'):
    """Caller must verify publication; independently parse config and strictly load aliases."""
    from safetensors.torch import load_model
    from transgenic.model.configuration_transgenic import HyenaTransgenicConfig
    from transgenic.model.modeling_HyenaTransgenic import transgenicForConditionalGeneration
    files = verified_run['files']
    config_file = checked_file(files['config.json'])
    weights = checked_file(files['model.safetensors'])
    raw = read_json(config_file['path'])
    require(raw.get('evidence_channels') == 8 and type(raw.get('evidence_channels')) is int, 'final must have eight evidence channels')
    require(raw.get('vocab_size') == 290 and raw.get('max_decoder_position_embeddings') == MAX_LENGTH, 'final must use v3/A27 config')
    config = HyenaTransgenicConfig(**raw)
    if 'metadata.json' in files:
        metadata = read_json(checked_file(files['metadata.json'])['path'])
        target = metadata['initialization_audit']['audit']['target_semantics']['fields']
        for name, record in target.items():
            require(getattr(config, name, None) == record['value'], f'final config/audit mismatch: {name}')
    # This construction-only switch prevents external pretrained loading. All
    # learned tensors (including projection) then come from strict final bytes.
    pretrained = config.load_pretrained_encoder
    config.load_pretrained_encoder = False
    model = transgenicForConditionalGeneration(config)
    config.load_pretrained_encoder = pretrained
    load_model(model, weights['path'], strict=True, device='cpu')
    checked_file(files['model.safetensors'])
    checked_file(files['config.json'])
    return model.to(device).eval()


def load_baseline_model(verified_run, device='cpu'):
    from transgenic.training.weights_init import load_weights_only
    model = load_final_model(verified_run, 'cpu')
    for record in verified_run['initialization'].values():
        checked_file(record)
    source = checked_file(verified_run['initialization']['model.safetensors'])
    # Resolved audit path, not a caller's mutable best symlink.
    audit = load_weights_only(model, Path(source['path']).parent)
    require(not model.get_encoder().evidence_proj.weight.detach().count_nonzero(), 'baseline projection not zero')
    checked_file(verified_run['initialization']['model.safetensors'])
    return model.to(device).eval(), audit
