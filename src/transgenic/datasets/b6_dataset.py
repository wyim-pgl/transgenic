"""B6 dataset, epoch sampler and collate (plan §7.3, §7.7).

``B6Dataset[(rn, epoch)]`` -> one sample dict; ``B6EpochSampler`` yields ``(rn, epoch)`` so a worker holds no RNG
state (every draw is a pure function of (aug_seed, row, epoch)); ``b6_collate_fn`` builds the exact §7.7 batch.

The collate is numpy-native (validated by ``b6_track_spec.validate_batch``); ``b6_collate_fn_torch`` converts the
same arrays to tensors.  Nothing here imports torch at module import time, and nothing here is part of the MVP
byte pin (new file, imported by full path).
"""
from __future__ import annotations

import numpy as np

from transgenic.evidence import b6_aug as AG
from transgenic.evidence import b6_chains as CH
from transgenic.evidence import b6_context as CX
from transgenic.evidence import b6_track_spec as TS
from transgenic.evidence import b6_tracks as TR


def hyena_char_encoder(encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf"):
    """The B5 encoder path (datasets.py isoformDataHyena): HF tokenizer, trailing [SEP] removed."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(encoder_model, cache_dir="./HFmodels", trust_remote_code=True)

    def encode(seq):
        ids = tok(seq)["input_ids"]
        if len(ids) > 1:
            ids = ids[:-1]
        return np.asarray(ids, np.int64), tok.pad_token_id
    return encode


def gff_label_tokenizer(vocab_version="v3"):
    from transgenic.model.tokenization_transgenic import GFFTokenizer

    dt = GFFTokenizer(vocab_version=vocab_version)
    maxlength = 8192 if vocab_version == "v3" else 2048

    def tokenize(gff):
        ids = [dt._convert_token_to_id(t) for t in dt._tokenize(gff)]
        if len(ids) > maxlength:
            ids = ids[:maxlength - 1] + [dt.vocab["</s>"]]
        return np.asarray(ids, np.int64)
    return tokenize


class B6Dataset:
    """Corpus-tile samples for training/validation under one arm.

    ``mode``: 'train_track' | 'valid_track'.  ``arm`` in TS.ARMS.  ``c3_donors``: {rn: donor wid} (required for C3,
    from ``b6_aug.c3_donor_map``).  ``encode_dna(seq) -> (ids int64[L], pad_id)``; ``tokenize_gsf(gff) -> int64[T]``.
    """

    def __init__(self, corpus, cache, *, mode, arm, aug_seed, encode_dna, tokenize_gsf=None, training=True,
                 c3_donors=None):
        if arm not in TS.ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        if arm == "C3" and not c3_donors:
            raise ValueError("C3 needs the fixed donor map (b6_aug.c3_donor_map)")
        if training and tokenize_gsf is None:
            raise ValueError("training samples need a GSF tokenizer")
        self.corpus, self.cache, self.mode, self.arm = corpus, cache, mode, arm
        self.aug_seed = int(aug_seed)
        self.encode_dna, self.tokenize_gsf, self.training = encode_dna, tokenize_gsf, training
        self.c3_donors = dict(c3_donors or {})

    def evidence(self, ctx, epoch):
        draw = AG.make_draw(self.aug_seed, ctx.row, epoch)
        ev_ctx = ctx
        if self.arm == "C3":
            ev_ctx = CX.donor_context(self.corpus, ctx, self.c3_donors[int(ctx.row)])
        owner = AG.Owner.from_pairs(ev_ctx.species, self.cache.owner_libraries(ctx.population, ev_ctx.species))
        resolved = AG.resolve(draw, owner)
        tracks = TR.rasterise(self.cache, ev_ctx, resolved)
        tile = CH.build_chains(self.cache, ev_ctx, resolved)
        tracks = AG.apply_arm_tracks(tracks, self.arm)
        if AG.arm_masks_chains(self.arm):
            tile = CH.empty_tile(tile.qc)
        return draw, resolved, tracks, tile

    def __getitem__(self, key):
        rn, epoch = key
        ctx = CX.corpus_context(self.corpus, rn, self.mode)
        draw, resolved, tracks, tile = self.evidence(ctx, epoch)
        ids, pad_id = self.encode_dna(ctx.dna)
        if len(ids) != ctx.L:
            raise ValueError(f"rn {rn}: {len(ids)} DNA tokens != {ctx.L} bases (token offset must be 0)")
        labels = None
        if self.training:
            labels = self.tokenize_gsf(self.corpus.row(rn)[9])
        return {"input_ids": ids, "attention_mask": ids != pad_id, "evidence_tracks": tracks, "labels": labels,
                "chain_pos": tile.chain_pos, "chain_feat": tile.chain_feat, "chain_mask": tile.chain_mask,
                "meta": {"row": ctx.row, "epoch": int(epoch), "arm": self.arm, "uniforms_sha256": draw.sha256(),
                         "resolved": resolved.record(), "context_kind": ctx.kind}}


class B6EpochSampler:
    """Yields (rn, epoch).  ``order(epoch) -> sequence of rn`` is the B5 data order (owned by the trainer); resume
    starts at (epoch, index) exactly, with no state besides those two integers."""

    def __init__(self, order, *, epochs, start_epoch=0, start_index=0):
        self.order, self.epochs = order, int(epochs)
        self.start_epoch, self.start_index = int(start_epoch), int(start_index)

    def __iter__(self):
        for epoch in range(self.start_epoch, self.epochs):
            rns = list(self.order(epoch))
            first = self.start_index if epoch == self.start_epoch else 0
            for rn in rns[first:]:
                yield (int(rn), epoch)

    def __len__(self):
        n = 0
        for epoch in range(self.start_epoch, self.epochs):
            n += len(self.order(epoch)) - (self.start_index if epoch == self.start_epoch else 0)
        return n


def b6_collate_fn(samples):
    """Exact §7.7 batch (numpy).  DNA, masks and tracks left-padded; labels right-padded -100."""
    if not samples:
        raise ValueError("empty batch")
    B = len(samples)
    Ls = [len(s["input_ids"]) for s in samples]
    Lmax = max(Ls)
    if Lmax % TS.DOWNSAMPLE:
        raise ValueError(f"Lmax {Lmax} is not a multiple of {TS.DOWNSAMPLE}")
    training = samples[0]["labels"] is not None
    if any((s["labels"] is not None) != training for s in samples):
        raise ValueError("mixed training/inference samples")
    ids = np.zeros((B, Lmax), np.int64)
    att = np.zeros((B, Lmax), bool)
    tracks = np.zeros((B, Lmax, TS.K), np.float16)
    idx = np.zeros((B, TS.N_TOK, 2), np.int64)
    feat = np.zeros((B, TS.N_TOK, TS.N_FEAT), np.int16)
    mask = np.zeros((B, TS.N_TOK), bool)
    labels = None
    if training:
        T = max(len(s["labels"]) for s in samples)
        labels = np.full((B, T), TS.LABEL_PAD, np.int64)
    meta = {k: [] for k in TS.META_KEYS}
    for b, s in enumerate(samples):
        L = Ls[b]
        pad = Lmax - L
        if s["evidence_tracks"].shape != (L, TS.K) or s["evidence_tracks"].dtype != np.float16:
            raise ValueError("evidence tracks must be float16 [L, 31]")
        ids[b, pad:] = s["input_ids"]
        att[b, pad:] = s["attention_mask"]
        tracks[b, pad:] = s["evidence_tracks"]
        m = np.asarray(s["chain_mask"], bool)
        if m[0]:
            raise ValueError("chain element 0 must be the masked dummy")
        pos = np.asarray(s["chain_pos"])
        if m.any() and (pos[m].min() < 0 or pos[m].max() >= L):
            raise ValueError("chain position outside the sample")
        idx[b] = TS.chain_idx(pos, m, pad)
        feat[b] = np.where(m[:, None], s["chain_feat"], 0)
        mask[b] = m
        if training:
            labels[b, :len(s["labels"])] = s["labels"]
        for k in TS.META_KEYS:
            meta[k].append(s["meta"][k])
    batch = {"input_ids": ids, "attention_mask": att, "evidence_tracks": tracks, "labels": labels,
             "chain_idx": idx, "chain_feat": feat, "chain_mask": mask, "meta": meta}
    TS.validate_batch(batch, training=training)
    return batch


def b6_collate_fn_torch(samples):
    import torch

    batch = b6_collate_fn(samples)
    out = {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in batch.items()}
    TS.validate_batch(out, training=batch["labels"] is not None)
    return out
