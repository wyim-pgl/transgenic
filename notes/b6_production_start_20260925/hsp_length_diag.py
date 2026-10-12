"""Diagnose 'reported length disagrees' under masked_x=True (read-only)."""
import sys, collections, gzip
sys.path[:0] = [sys.argv[1] + '/revision/scripts', sys.argv[1]]
from evidence import est_hsp_contract as C
from evidence import est_hsp_expanded as E
from pathlib import Path
bx, estfa, source, limit = Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], int(sys.argv[5])
def fasta(path, opener):
    out, name, buf = {}, None, []
    with opener(path) as f:
        for line in f:
            if line.startswith('>'):
                if name: out[name] = ''.join(buf).upper()
                name, buf = line[1:].split()[0], []
            else: buf.append(line.strip())
    if name: out[name] = ''.join(buf).upper()
    return out
est = fasta(estfa, lambda p: gzip.open(p, 'rt')); prot = fasta(bx / source / f'{source}.filtered.fa', open)
def first_bad(fl, dna, p):
    frame, qs, ss = int(fl[11]), int(fl[4]), int(fl[6])
    o = dna if frame > 0 else E.rc(dna); pos = qs - 1 if frame > 0 else len(dna) - qs; sub = ss - 1
    for t in C.btop_tokens(fl[12]):
        pairs = [(p[sub + i], p[sub + i]) for i in range(int(t))] if t.isdigit() else [tuple(t)]
        for q, s_ in pairs:
            if q in '/\\':
                pos += 1 if q == '\\' else -1; continue
            if q != '-':
                aa = E.CODE.get(o[pos:pos + 3])
                if q == 'X':
                    if aa not in E.SENSE: return ('qX_over', aa, 'frame' + fl[11])
                elif aa != q: return ('q', q, aa, s_)
                pos += 3
            if s_ != '-': sub += 1
    return ('none',)
cnt = collections.Counter(); shown = 0
with open(bx / source / f'{source}.raw13.tsv') as f:
    for line in f:
        fl = line.rstrip('\n').split('\t'); cnt['rows'] += 1
        dna, p = est.get(fl[0]), prot.get(fl[2])
        if dna is None or p is None or set(dna) - set('ACGT'): cnt['skip'] += 1; continue
        hasx = 'X' in fl[12]; hasf = ('/' in fl[12]) or ('\\' in fl[12])
        try:
            C.adapt_raw13(fl, dna, p, masked_x=True); cnt[('ok', hasx, hasf)] += 1
        except ValueError as e:
            msg = str(e).split(':')[0]; cnt[(msg, 'X' if hasx else '-', 'F' if hasf else '-')] += 1
            if msg.startswith('query codon truth'):
                cnt[('detail',) + first_bad(fl, dna, p)] += 1
            if msg.startswith('reported length'):
                try:
                    C.adapt_raw13(fl, dna, p, masked_x=True, length_contract=C.MIDF_LENGTH_CONTRACT); cnt['len_fail_but_MIDF_ok'] += 1
                except ValueError as e2:
                    cnt['len_fail_MIDF_also:' + str(e2)[:40]] += 1
                if shown < 4:
                    shown += 1; print('EX', fl[:12], fl[12][:80])
        if cnt['rows'] >= limit: break
for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])[:30]: print(v, k)
