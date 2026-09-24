"""B6 K = 31 track rasteriser (plan §7.1, §7.6; A46 rule 6).

``rasterise(cache, ctx, draw) -> float16 [L, 31]`` in output orientation.  Construction order (fixed, part of
TRACK_SPEC): admit rows (``Context.admit``) -> library withholding -> rule 8 thinning -> rasterise in genomic
orientation -> zero the phase channels over the population's A48 zero list -> set masks -> RC iff ``ctx.is_rc``.
"""
from __future__ import annotations

import numpy as np

from transgenic.evidence import b6_track_spec as TS


class LeakageError(RuntimeError):
    """A retained observation touches a rule 4(a) interval of the tile: abort (rule 13)."""


def _retained(cache, ctx, draw, kinds):
    sh, idx = cache.track_rows(ctx.species, ctx.chrom, ctx.ws, ctx.we)
    cols = sh.cols
    if len(idx):
        idx = idx[np.isin(cols["kind"][idx], kinds)]
    if len(idx):
        idx = idx[ctx.admit(cols, idx)]
    if draw.dropped or not len(idx):
        return cols, np.zeros(0, np.int64)
    admitted = draw.admitted_set
    keep = []
    for i in idx:
        src, lib = cols["source"][i], cols["library"][i]
        if src == "protein":
            qid = cols["query_id"][i]
            if not qid:
                raise ValueError(f"protein row of {cols['molecule_id'][i]} has no query_id (D7 thinning key)")
            if draw.keeps(ctx.species, qid):          # D7: rank over the query id; never withheld
                keep.append(i)
        elif (src, lib) in admitted and draw.keeps(ctx.species, cols["molecule_id"][i]):
            keep.append(i)
    return cols, np.asarray(keep, dtype=np.int64)


def _check_leakage(ctx, cols, idx):
    if not ctx.tile_exclusions or not len(idx):
        return
    s, e = cols["start0"][idx], cols["end0"][idx]
    for a, b, cause, scope in ctx.tile_exclusions:
        hit = (s < b) & (e > a)
        if hit.any():
            j = idx[np.nonzero(hit)[0][0]]
            raise LeakageError(f"{ctx.row}: retained {cols['kind'][j]} of molecule {cols['molecule_id'][j]} overlaps "
                               f"rule 4(a) interval {cause}/{scope} [{a}, {b})")


def _union_add(diff, intervals):
    """Add +1 over the union of ``intervals`` (local, clipped, sorted by start) to a diff array."""
    lo = hi = None
    for a, b in intervals:
        if lo is None or a > hi:
            if lo is not None:
                diff[lo] += 1
                diff[hi] -= 1
            lo, hi = a, b
        else:
            hi = max(hi, b)
    if lo is not None:
        diff[lo] += 1
        diff[hi] -= 1


def rasterise_with_qc(cache, ctx, draw):
    L, ws, we = ctx.L, ctx.ws, ctx.we
    out = np.zeros((L, TS.K), np.float32)
    qc = {"retained_molecules": {}, "dropped": bool(draw.dropped)}
    if draw.dropped:
        return out.astype(np.float16), qc
    cols, idx = _retained(cache, ctx, draw, ("block", "junction", "protein", "phase"))
    _check_leakage(ctx, cols, idx)
    kind = cols["kind"][idx] if len(idx) else np.zeros(0, object)
    for s in TS.TRACK_SOURCES:
        qc["retained_molecules"][s] = sorted({cols["molecule_id"][i] for i in idx[kind != "protein"]
                                              if cols["source"][i] == s}) if len(idx) else []
    qc["retained_molecules"]["protein"] = sorted({cols["molecule_id"][i] for i in idx[kind == "protein"]}) \
        if len(idx) else []

    # coverage: one count per molecule per base, per (source, strand)
    groups = {}
    for i in idx[kind == "block"]:
        a, b = max(int(cols["start0"][i]) - ws, 0), min(int(cols["end0"][i]) - ws, L)
        if a < b:
            ch = TS.source_base(cols["source"][i]) + {"+": 0, "-": 1, "?": 2}[cols["strand"][i]]
            groups.setdefault((ch, cols["molecule_id"][i]), []).append((a, b))
    diffs = {}
    for (ch, _mid), iv in groups.items():
        if ch not in diffs:
            diffs[ch] = np.zeros(L + 1, np.int64)
        _union_add(diffs[ch], sorted(iv))
    for ch, d in diffs.items():
        out[:, ch] = np.log1p(np.minimum(np.cumsum(d[:-1]), TS.COV_CLIP_MOLECULES))

    # donor / acceptor: distinct molecules per boundary base; rule 4(c) junctions never counted
    seen = set()
    xj = ctx.excluded_junctions
    for i in idx[kind == "junction"]:
        strand = cols["strand"][i]
        s0, e0 = int(cols["start0"][i]), int(cols["end0"][i])
        donor, acceptor = (s0, e0 - 1) if strand == "+" else (e0 - 1, s0)
        if (strand, "donor", donor) in xj or (strand, "acceptor", acceptor) in xj:
            continue
        base = TS.source_base(cols["source"][i])
        off_d, off_a = ((TS.OFF_DONOR_PLUS, TS.OFF_ACCEPTOR_PLUS) if strand == "+"
                        else (TS.OFF_DONOR_MINUS, TS.OFF_ACCEPTOR_MINUS))
        for pos, off in ((donor, off_d), (acceptor, off_a)):
            if ws <= pos < we:
                seen.add((pos - ws, base + off, cols["molecule_id"][i]))
    if seen:
        p, ch = np.array([(p, c) for p, c, _m in seen], np.int64).T
        cnt = np.bincount(p * TS.K + ch, minlength=L * TS.K).reshape(L, TS.K)
        dm = [TS.source_base(s) + o for s in TS.TRACK_SOURCES for o in range(3, 7)]
        out[:, dm] = np.log1p(cnt[:, dm])

    # protein: distinct OrthoDB organisms among retained queries covering the base (D7)
    org = {}
    for i in idx[kind == "protein"]:
        a, b = max(int(cols["start0"][i]) - ws, 0), min(int(cols["end0"][i]) - ws, L)
        if a < b:
            o = cols["organism"][i]
            if not o:
                raise ValueError(f"protein row of {cols['molecule_id'][i]} has no organism")
            org.setdefault(o, []).append((a, b))
    if org:
        d = np.zeros(L + 1, np.int64)
        for iv in org.values():
            _union_add(d, sorted(iv))
        out[:, TS.PROTEIN_CHANNEL] = np.log1p(np.cumsum(d[:-1]))

    # phase: distinct EST molecules per (base, codon position, strand); only when EST is supplied
    if draw.phase_enabled:
        per_mol = {}
        for i in idx[kind == "phase"]:
            g0, g1 = int(cols["start0"][i]), int(cols["end0"][i])
            a, b = max(g0, ws), min(g1, we)
            if a >= b:
                continue
            strand, ph = cols["strand"][i], int(cols["phase"][i])
            g = np.arange(a, b, dtype=np.int64)
            cpos = (ph + (g - g0)) % 3 if strand == "+" else (ph - (g - g0)) % 3
            ch = TS.PHASE_CHANNELS[0] + (0 if strand == "+" else 3) + cpos
            per_mol.setdefault(cols["molecule_id"][i], set()).update(zip((g - ws).tolist(), ch.tolist()))
        if per_mol:
            flat = np.fromiter((p * TS.K + c for cells in per_mol.values() for p, c in cells), np.int64)
            cnt = np.bincount(flat, minlength=L * TS.K).reshape(L, TS.K)
            pc = list(TS.PHASE_CHANNELS)
            out[:, pc] = np.log1p(cnt[:, pc])
    for a, b in ctx.zero_intervals:
        out[max(a - ws, 0):min(b - ws, L), list(TS.PHASE_CHANNELS)] = 0
    for s in TS.TRACK_SOURCES:
        out[:, TS.source_base(s) + TS.OFF_MASK] = draw.mask(s)
    res = out.astype(np.float16)
    if ctx.is_rc:
        res = TS.rc_tracks(res)
    return np.ascontiguousarray(res), qc


def rasterise(cache, ctx, draw):
    """float16 [L, 31] (plan §7.6)."""
    return rasterise_with_qc(cache, ctx, draw)[0]
