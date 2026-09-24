"""B6 chain memory tiles (plan §7.2, §7.6; A46 rule 6a; design v2 §2.4).

``build_chains(cache, ctx, draw) -> ChainTile`` with ``chain_pos`` int32[512, 2], ``chain_feat`` int16[512, 6],
``chain_mask`` bool[512] and QC.  Admission and thinning are the rasteriser's (same ``Context.admit``, same
``ResolvedDraw``), plus the rule 6a restriction that ``tracks_only`` datasets never feed chains, so the chain
molecules are always a subset of the track molecules.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from transgenic.evidence import b6_track_spec as TS


@dataclass
class ChainTile:
    chain_pos: np.ndarray
    chain_feat: np.ndarray
    chain_mask: np.ndarray
    qc: dict = field(default_factory=dict)


def empty_tile(qc=None):
    return ChainTile(np.zeros((TS.N_TOK, 2), np.int32), np.zeros((TS.N_TOK, TS.N_FEAT), np.int16),
                     np.zeros(TS.N_TOK, bool), qc or {})


def _parse(introns, oks):
    pairs = [tuple(int(x) for x in p.split("-")) for p in introns.split(",")] if introns else []
    flags = [f == "1" for f in oks.split(",")] if oks else []
    if len(pairs) != len(flags):
        raise ValueError(f"chain introns/introns_ok length mismatch: {introns!r} / {oks!r}")
    return pairs, flags


def split_chain(ctx, strand, pairs, flags):
    """Fragments (lists of (donor0, acceptor0)) after rule 6a splitting for this context."""
    xj = ctx.excluded_junctions
    frags, cur = [], []
    for (d, a), ok in zip(pairs, flags):
        lo, hi = min(d, a), max(d, a) + 1
        keep = ok and ctx.ws <= d < ctx.we and ctx.ws <= a < ctx.we
        if keep and ctx.kind == "corpus":
            if (strand, "donor", d) in xj or (strand, "acceptor", a) in xj:
                keep = False
            elif any(lo < e and hi > s for s, e, _c, _sc in ctx.tile_exclusions):
                keep = False
        if keep:
            cur.append((d, a))
        else:
            if cur:
                frags.append(cur)
            cur = []
    if cur:
        frags.append(cur)
    return frags


def candidate_chains(cache, ctx, draw):
    """Merged pre-cap chains: {(source, strand, introns tuple): set(molecule_id)}."""
    if draw.dropped:
        return {}
    sh, idx = cache.chain_rows(ctx.species, ctx.chrom, ctx.ws, ctx.we)
    cols = sh.cols
    if len(idx):
        idx = idx[ctx.admit(cols, idx, chains=True)]
    merged = {}
    for i in idx:
        src, lib, mid = cols["source"][i], cols["library"][i], cols["molecule_id"][i]
        if src not in TS.CHAIN_SOURCES or not draw.library_admitted(src, lib) or not draw.keeps(ctx.species, mid):
            continue
        strand = cols["strand"][i]
        if strand not in TS.CHAIN_STRANDS:
            continue
        pairs, flags = _parse(cols["introns"][i], cols["introns_ok"][i])
        for frag in split_chain(ctx, strand, pairs, flags):
            merged.setdefault((src, strand, tuple(frag)), set()).add(mid)
    return merged


def build_chains(cache, ctx, draw):
    """ChainTile for ``ctx`` under ``draw``.  The rule 6a slot permutation is keyed by (draw.aug_seed, ctx.row,
    draw.epoch): the recipient's training aug_seed (C3 draws carry the recipient's), or 20260914 + d in dose-response,
    with epoch 0 at evaluation.  It is a pure function and consumes no augmentation stream."""
    qc = {"candidates": {s: 0 for s in TS.CHAIN_SOURCES}, "filled": {s: 0 for s in TS.CHAIN_SOURCES},
          "truncated": {s: 0 for s in TS.CHAIN_SOURCES}, "elements": 0}
    merged = candidate_chains(cache, ctx, draw)
    tile = empty_tile(qc)
    if not merged:
        qc["precap_keys"] = []
        return tile
    order = []
    for (src, strand, introns), mols in merged.items():
        units = len(mols)
        key = TS.tie_key(ctx.species, strand, introns)
        order.append((TS.fill_sort_key(src, units, len(introns), key), src, strand, introns, units))
        qc["candidates"][src] += 1
    order.sort(key=lambda x: x[0])
    qc["precap_keys"] = sorted(f"{src}\t{strand}\t{TS.introns_text(introns)}" for _, src, strand, introns, _ in order)
    pi = TS.slot_permutation(draw.aug_seed, ctx.row, draw.epoch)
    used, rank = 1, 0
    for _, src, strand, introns, units in order:
        n = len(introns)
        if rank >= TS.MAX_CHAINS or used + n > TS.N_TOK:
            qc["truncated"][src] += 1
            continue
        slot = pi[rank]
        rank += 1
        for i, (d, a) in enumerate(introns):
            e = used + i
            tile.chain_pos[e] = (d - ctx.ws, a - ctx.ws)
            tile.chain_feat[e] = (min(i, TS.ORDER_CAP), min(n - 1 - i, TS.ORDER_CAP), slot,
                                  TS.CHAIN_SOURCES.index(src), TS.CHAIN_STRANDS.index(strand), min(units, TS.UNITS_CAP))
            tile.chain_mask[e] = True
        used += n
        qc["filled"][src] += 1
    qc["elements"] = used - 1
    if ctx.is_rc:
        tile.chain_pos, tile.chain_feat, tile.chain_mask = TS.rc_chain(tile.chain_pos, tile.chain_feat,
                                                                       tile.chain_mask, ctx.L)
    return tile
