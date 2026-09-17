# revision/scripts/52_decoder_token_cap_audit.py

- load_tokenizers · function · L109-L116 — def load_tokenizers()
- extraction_window · function · L120-L148 — def extraction_window(gene_start, gene_end, static_size=STATIC_SIZE)
- GeneBuilder · class · L155-L206 — class GeneBuilder
- __init__ · method · L163-L178 — def __init__(self, gene_id, chrom, strand, gene_start, gene_end, region_start, region_end)
- open_transcript · method · L180-L184 — def open_transcript(self): # preprocess.py:404 -- drop the trailing '|' of the previous transcript # and start a new one. For the first mRNA this seeds mrna_list with ';'.
- add_feature · method · L186-L198 — def add_feature(self, typ, start, fin, strand, phase, table)
- gsf · method · L200-L206 — def gsf(self)
- iter_genes_gff3 · function · L209-L259 — def iter_genes_gff3(path)
- iter_genes_gtf · function · L270-L319 — def iter_genes_gtf(path)
- pct · function · L323-L330 — def pct(sorted_vals, q)
- audit_species · function · L333-L476 — def audit_species(species, path, tokenizers, out_dir, is_gtf=False)
- block · function · L428-L433 — def block(arr)
- validate_window_math · function · L480-L520 — def validate_window_math(db_path, gff_path)
- main · function · L524-L607 — def main()
