"""Diagnose 'query codon truth' failures for one species' raw13 rows (read-only)."""
import sys, collections, gzip
sys.path[:0] = [sys.argv[1] + '/revision/scripts/evidence', sys.argv[1] + '/revision/scripts', sys.argv[1] + '/src']
from evidence import est_hsp_contract as C
from pathlib import Path
bx, estfa, source = Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
est = {}
with gzip.open(estfa, 'rt') as f:
    name = None
    for line in f:
        if line.startswith('>'):
            name = line[1:].split()[0]; est[name] = []
        else:
            est[name].append(line.strip())
est = {k: ''.join(v) for k, v in est.items()}
prot = {}
with open(bx / source / f'{source}.filtered.fa') as f:
    name = None
    for line in f:
        if line.startswith('>'):
            name = line[1:].split()[0]; prot[name] = []
        else:
            prot[name].append(line.strip())
prot = {k: ''.join(v) for k, v in prot.items()}
def first_bad(fields, dna, p):
    frame, qs, ss = int(fields[11]), int(fields[4]), int(fields[6])
    o = dna if frame > 0 else C.rc(dna)
    pos = qs - 1 if frame > 0 else len(dna) - qs
    sub = ss - 1
    for t in C.btop_tokens(fields[12]):
        pairs = [(p[sub + i], p[sub + i]) for i in range(int(t))] if t.isdigit() else [tuple(t)]
        for q, s_ in pairs:
            if q in '/\\':
                pos += 1 if q == '\\' else -1; continue
            if q != '-':
                cod = o[pos:pos + 3]
                if C.CODE.get(cod) != q: return ('q', q, s_, 'codon=' + cod + '->' + str(C.CODE.get(cod)))
                pos += 3
            if s_ != '-':
                if sub >= len(p) or p[sub] != s_: return ('s', q, s_, 'actual=' + (p[sub] if sub < len(p) else 'END'))
                sub += 1
    return ('none',)
cnt = collections.Counter(); ex = collections.Counter(); shown = 0
with open(bx / source / f'{source}.raw13.tsv') as f:
    for line in f:
        fields = line.rstrip('\n').split('\t')
        cnt['rows'] += 1
        q, s = fields[0], fields[2]
        dna, p = est.get(q), prot.get(s)
        if dna is None or p is None:
            cnt['missing_seq'] += 1; continue
        dna, p = dna.upper(), p.upper()
        if set(dna) - set('ACGT'):
            cnt['skip_noncanonical_whole_est'] += 1; continue
        try:
            C.adapt_raw13(fields, dna, p)
            cnt['ok'] += 1
        except ValueError as e:
            cnt[str(e)] += 1
            if str(e) in ('query codon truth', 'subject truth'):
                tok = first_bad(fields, dna, p)
                ex[(str(e), tok)] += 1
                bad = set(dna.upper()) - set('ACGT')
                ex['query_has_nonACGT' if bad else 'query_ACGT_only'] += 1
                ex['lowercase' if dna != dna.upper() else 'uppercase'] += 1
                if shown < 3 and not bad:
                    shown += 1; print('EXAMPLE', fields[:12], fields[12][:60])
        if cnt['rows'] >= int(sys.argv[5]): break
print(dict(cnt))
for k, v in ex.most_common(25): print(v, k)
