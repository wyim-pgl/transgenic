"""A19 applied to a species' miniprot accept tables (companion of a19_filter_miniprot.py, same predicate): drop every
`<sp>.alignments.tsv` row and every `<sp>.mapping_ambiguous.txt` protein whose query is of the species or below it,
using the C0 builder's own A19 class over the frozen lineage. Unknown taxa abort.
protein_accept.py judges each alignment on its own (identity/coverage/frameshift/introns) and ambiguity per protein
over that protein's own alignments, so dropping same-species rows equals running acceptance on the A19-filtered GFF.
The kept/dropped row counts must equal the GFF filter's blocks_kept/blocks_dropped_same_species (one ##PAF block per
alignment). Only the two tables b6_routea_agreement.py reads are written; blocks/introns/support tables are not.
Usage: a19_filter_accept.py <species> <accept_dir> <lineage.tsv> <a19_species_dir>   (writes <a19_species_dir>/accept)"""
import csv, hashlib, json, sys
from collections import Counter
from pathlib import Path
import b6_build_c0 as B

sp, src, lineage, a19_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
out_dir = a19_dir / 'accept'
gff_done = dict(l.split('=', 1) for l in (a19_dir / 'DONE').read_text().splitlines() if '=' in l)
acc_done = dict(l.split('=', 1) for l in (src / 'ACCEPT_DONE').read_text().splitlines() if '=' in l)
assert acc_done['gff_md5'] == gff_done['source_gff_md5'], 'accept tables are not from the GFF the A19 filter read'
out_dir.mkdir(parents=False, exist_ok=False)
a19 = B.A19(str(lineage), sp)
assert a19.lineage_sha256 == gff_done['lineage_sha256'], 'lineage differs from the GFF filter'
cache, qc, dropped_tax = {}, Counter(), Counter()


def same(protein):
    taxid = a19.taxid_of_protein(protein)
    if taxid not in cache:
        cache[taxid] = a19.hp.is_same_species_or_descendant(taxid, sp)   # ValueError (unknown taxon) aborts
    return cache[taxid], taxid


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


files = {}
for name, kind in ((f'{sp}.alignments.tsv', 'table'), (f'{sp}.mapping_ambiguous.txt', 'list')):
    fi_p, fo_p = src / name, out_dir / name
    tmp = fo_p.with_name(fo_p.name + '.partial')
    with open(fi_p) as fi, open(tmp, 'w') as fo:
        if kind == 'table':
            header = fi.readline()
            fo.write(header)
            cols = header.rstrip('\n').split('\t')
            col, vcol = cols.index('protein'), cols.index('verdict')
            for line in fi:
                c = line.rstrip('\n').split('\t')
                s, taxid = same(c[col])
                if s:
                    qc['rows_dropped_same_species'] += 1
                    dropped_tax[taxid] += 1
                    if c[vcol] == 'accepted':
                        qc['accepted_rows_dropped_same_species'] += 1
                else:
                    qc['rows_kept'] += 1
                    fo.write(line)
        else:
            for line in fi:
                if not line.strip():
                    continue
                if same(line.strip())[0]:
                    qc['ambiguous_dropped_same_species'] += 1
                else:
                    qc['ambiguous_kept'] += 1
                    fo.write(line)
    tmp.rename(fo_p)
    files[name] = dict(source_sha256=sha(fi_p), sha256=sha(fo_p))

assert qc['rows_kept'] == int(gff_done['blocks_kept']), (qc['rows_kept'], gff_done['blocks_kept'])
assert qc['rows_dropped_same_species'] == int(gff_done['blocks_dropped_same_species']), \
    (qc['rows_dropped_same_species'], gff_done['blocks_dropped_same_species'])
(out_dir / 'ACCEPT_DONE').write_text(''.join(f'{k}={v}\n' for k, v in [
    ('species', sp), ('gff_md5', gff_done['gff_md5']), ('source_gff_md5', acc_done['gff_md5']),
    ('source_accept_dir', src), ('a19_filter', 'same_species_or_descendant_dropped'),
    ('lineage_sha256', a19.lineage_sha256), *sorted(qc.items()),
    *[(f'{n}.{k}', v) for n, d in files.items() for k, v in d.items()]]))
print(json.dumps(dict(species=sp, **qc, dropped_taxids=dict(dropped_tax), files=files), sort_keys=True))
