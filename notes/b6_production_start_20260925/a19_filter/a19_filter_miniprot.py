"""A19 (protocol: "Same-species proteins are never used") applied to a species' miniprot GFF: drop every ##PAF-led
alignment block whose query protein is of the species or below it, using the C0 builder's own A19 class
(b6_build_c0.A19 -> b6_hsp_project.is_same_species_or_descendant over the frozen lineage). Unknown taxa abort.
Each query is aligned independently by miniprot, so dropping its blocks equals not having queried it.
Usage: a19_filter_miniprot.py <species> <in.gff.gz> <lineage.tsv> <out_dir>"""
import gzip, hashlib, json, sys
from collections import Counter
from pathlib import Path
import b6_build_c0 as B
sp, src, lineage, out_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
out_dir.mkdir(parents=True, exist_ok=False)
a19 = B.A19(str(lineage), sp)
cache, qc = {}, Counter()
def same(protein):
    taxid = a19.taxid_of_protein(protein)
    if taxid not in cache:
        cache[taxid] = a19.hp.is_same_species_or_descendant(taxid, sp)   # ValueError (unknown taxon) aborts
    return cache[taxid], taxid
out = out_dir / f'{sp}.protein.gff.gz'
tmp = out.with_name(out.name + '.partial')
keep, dropped_tax = True, Counter()
with gzip.open(src, 'rt') as fi, gzip.open(tmp, 'wt', compresslevel=6) as fo:
    for line in fi:
        if line.startswith('##PAF\t'):
            s, taxid = same(line.split('\t', 2)[1])
            keep = not s
            qc['blocks_kept' if keep else 'blocks_dropped_same_species'] += 1
            if not keep: dropped_tax[taxid] += 1
        elif line.startswith('#') and qc['blocks_kept'] + qc['blocks_dropped_same_species'] == 0:
            fo.write(line); continue           # file header before the first block
        if keep: fo.write(line)
        if not keep and not line.startswith('#'): qc['lines_dropped'] += 1
tmp.rename(out)
def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()
def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
src_done = dict(l.split('=', 1) for l in (src.parent / 'DONE').read_text().splitlines() if '=' in l)
gmd5 = md5(out); smd5 = md5(src)
assert smd5 == src_done['gff_md5'], 'source GFF md5 differs from its DONE'
(out_dir / 'DONE').write_text(''.join(f'{k}={v}\n' for k, v in [
    ('ref_md5', src_done['ref_md5']), ('faa_md5', src_done['faa_md5']), ('params', src_done['params']), ('gff_md5', gmd5),
    ('source_gff_md5', smd5), ('a19_filter', 'same_species_or_descendant_dropped'), ('lineage_sha256', a19.lineage_sha256),
    ('blocks_kept', qc['blocks_kept']), ('blocks_dropped_same_species', qc['blocks_dropped_same_species'])]))
(out_dir / 'PROVENANCE.txt').write_text(Path(src.parent / 'PROVENANCE.txt').read_text() +
    f'a19_filter       author/protocol A19 "Same-species proteins are never used"; source {src} md5 {smd5}; '
    f'lineage sha256 {a19.lineage_sha256}; dropped taxids {dict(dropped_tax)}\n')
print(json.dumps(dict(species=sp, **qc, dropped_taxids=dict(dropped_tax), gff_md5=gmd5), sort_keys=True))
