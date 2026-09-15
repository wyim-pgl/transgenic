"""CPU-only eight-channel EST tracks for the exploratory Athaliana MVP.

The caller owns a read-only DuckDB connection. Sequence is already output-oriented;
offset is always genomic, zero-based. No corpus, torch, sampler or model imports.
"""
from functools import lru_cache
from hashlib import blake2b
import math
from numbers import Integral

import numpy as np


def rasterise_est(con, *, sequence, chrom, offset, L, is_rc,
                  keep_fraction, aug_seed, rn=None,
                  population='retained', source_supplied=True):
    """Return float16 [L,8]; dropped/zero-fraction source has availability zero.

    rn restricts to tile_est membership (retained train/valid only). Evaluation
    windows must omit rn and select population='excluded_heldout'. Separate block
    and junction queries prevent losing boundaries in intron-only windows.
    """
    if (not isinstance(L,Integral) or L < 0 or not isinstance(offset,Integral) or offset < 0
            or not isinstance(sequence,str) or len(sequence) != L):
        raise ValueError('sequence length must equal nonnegative L; offset must be nonnegative integer')
    if not math.isfinite(keep_fraction) or not 0 <= keep_fraction <= 1:
        raise ValueError('keep_fraction must be finite and in [0,1]')
    if population not in ('retained','excluded_heldout'):
        raise ValueError('population must be retained or excluded_heldout')
    result = np.zeros((L,8),dtype=np.float32)
    if not source_supplied or keep_fraction == 0 or L == 0:
        return result.astype(np.float16)
    result[:,7] = 1

    @lru_cache(maxsize=None)
    def keep(molecule_id):
        rank = int.from_bytes(blake2b(f'{aug_seed}\tAthaliana\t{molecule_id}'.encode(),digest_size=8).digest(),'big') / 2**64
        return rank < keep_fraction

    policy = 'NOT m.excluded' if population == 'retained' else "m.excluded AND m.cause='heldout_locus'"
    policy += ' AND m.primary_count>0'
    params = [chrom,offset+L,offset]
    if rn is not None:
        policy += ' AND EXISTS (SELECT 1 FROM tile_est t WHERE t.molecule_id=m.molecule_id AND t.rn=?)'
        params.append(rn)

    def rows(sql, args):
        cur = con.execute(sql,args)
        while True:
            batch = cur.fetchmany(4096)
            if not batch:
                break
            yield from batch

    diff = np.zeros((L+1,3),dtype=np.int64)
    # Defensively union each molecule's clipped blocks even for a hand-built DB.
    previous = None
    lo = hi = channel = None
    sql = f'''SELECT m.molecule_id,m.strand,b.start0,b.end0 FROM est_block b
        JOIN est_molecule m USING(molecule_id)
        WHERE b.chrom=? AND b.start0<? AND b.end0>? AND {policy}
        ORDER BY m.molecule_id,b.start0,b.end0'''
    for mid,strand,a,b in rows(sql,params):
        if not keep(mid):
            continue
        a,b = max(a-offset,0),min(b-offset,L)
        if mid == previous and a <= hi:
            hi = max(hi,b)
        else:
            if previous is not None:
                diff[lo,channel] += 1
                diff[hi,channel] -= 1
            previous,lo,hi,channel = mid,a,b,{'+':0,'-':1,'?':2}[strand]
    if previous is not None:
        diff[lo,channel] += 1
        diff[hi,channel] -= 1
    result[:,:3] = np.log1p(np.minimum(np.cumsum(diff[:-1],axis=0),64))

    # Endpoint overlap, independent of blocks. Distinct molecule counts at a
    # boundary even when two different introns share that donor or acceptor.
    counts = np.zeros((L,4),dtype=np.int64)
    jparams = [chrom,offset,offset+L,offset,offset+L]
    if rn is not None:
        jparams.append(rn)
    sql = f'''SELECT m.molecule_id,m.strand,j.donor0,j.acceptor0 FROM est_junction j
        JOIN est_molecule m USING(molecule_id)
        WHERE j.chrom=? AND ((j.donor0>=? AND j.donor0<?) OR (j.acceptor0>=? AND j.acceptor0<?))
        AND {policy} ORDER BY m.molecule_id'''
    previous,seen = None,set()
    for mid,strand,donor,acceptor in rows(sql,jparams):
        if strand == '?' or not keep(mid):
            continue
        if previous != mid:
            seen.clear()
            previous = mid
        for position,channel in ((donor,0 if strand=='+' else 2),(acceptor,1 if strand=='+' else 3)):
            if position is not None and offset <= position < offset+L:
                key = (position-offset,channel)
                if key not in seen:
                    np.add.at(counts,key,1)
                    seen.add(key)
    result[:,3:7] = np.log1p(counts)
    if is_rc:
        result = result[::-1][:,[1,0,2,5,6,3,4,7]]
    # Mask LAST: sequence is already RC-oriented when is_rc=True.
    bases = np.frombuffer(sequence.encode('ascii'),dtype=np.uint8)
    result[(bases==ord('N')) | (bases==ord('n'))] = 0
    return np.ascontiguousarray(result,dtype=np.float16)
