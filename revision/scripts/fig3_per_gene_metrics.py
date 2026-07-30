#!/usr/bin/env python3
"""Figure 3C/D regeneration: per-gene base-level F1 vs gene length and CDS count.

For every test-set prediction (GM=<geneModel> in the prediction GFF3), compute
base-level recall/precision/F1 against the corresponding reference gene model:
  recall    = |pred_exonic ∩ ref_exonic| / |ref_exonic|
  precision = |pred_exonic ∩ ref_exonic| / |pred_exonic|
Exonic bases = union of exon features (references) and union of CDS+UTR+exon
features (predictions; predictions carry CDS/exon/UTR).

Outputs: fig3_per_gene_metrics.csv (species,gene_model,gene_len,n_cds,recall,precision,f1)
"""
import re, sys, csv
from collections import defaultdict
from pathlib import Path

GPFS = Path("/data/gpfs/assoc/pgl/data/Transgenic")
GENOMES = GPFS / "genomes"
PREDS = GPFS / "transgenic/revision/results/fig3_regen/preds"
OUT = GPFS / "transgenic/revision/results/fig3_regen/fig3_per_gene_metrics.csv"

REFS = {
    "A_thaliana": "Athaliana_167_TAIR10.gene.clean.gff3",
    "G_max": "Gmax_880_Wm82.a6.v1.gene_exons.clean.gff3",
    "P_patens": "Ppatens_318_v3.3.gene_exons.clean.gff3",
    "P_trichocarpa": "Ptrichocarpa_533_v4.1.gene_exons.clean.gff3",
    "S_bicolor": "Sbicolor_730_v5.1.gene_exons.clean.gff3",
    "B_distachyon": "Bdistachyon_314_v3.1.gene_exons.clean.gff3",
    "S_italica": "Sitalica_312_v2.2.gene_exons.clean.gff3",
    "V_vinifera": "Vvinifera_PN40024_5.1_on_T2T_ref.exon.gff3",
    "O_sativa": "Osativa_323_v7.0.gene_exons.exon.gff3",
    "Z_mays": "Zmays_493_RefGen_V4.gene_exons.exon.gff3",
}

def merge(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]

def span(iv):
    return sum(e - s + 1 for s, e in iv)

def overlap_bp(a, b):
    i = j = tot = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0]); e = min(a[i][1], b[j][1])
        if s <= e:
            tot += e - s + 1
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot

def load_ref(path):
    """gene_id -> [chr, start, end, exon_iv, n_cds] (exons merged per gene)"""
    genes = {}
    tx2gene = {}
    for line in open(path):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 9:
            continue
        feat, s, e = f[2], int(f[3]), int(f[4])
        attrs = f[8]
        m = re.search(r'ID=([^;]+)', attrs)
        mp = re.search(r'Parent=([^;]+)', attrs)
        if feat == "gene" and m:
            genes.setdefault(m.group(1), [f[0], s, e, [], 0])
        elif feat in ("mRNA", "transcript") and m and mp:
            parent = mp.group(1)
            if parent in genes:
                tx2gene[m.group(1)] = parent
        elif feat in ("exon", "CDS") and mp:
            g = tx2gene.get(mp.group(1))
            if g is None and mp.group(1) in genes:  # direct gene parentage
                g = mp.group(1)
            if g is None:
                continue
            if feat == "exon":
                genes[g][3].append((s, e))
            else:
                genes[g][4] += 1
                genes[g][3].append((s, e))
    for g, v in genes.items():
        v[3] = merge(v[3])
    return genes

# transcript->gene map needed for refs with mRNA between; handled via cur above (best effort)
rows_out = []
for sp, refname in REFS.items():
    pred_file = PREDS / f"{sp}_test400M.gff3"
    if not pred_file.exists():
        print("skip", sp)
        continue
    ref = load_ref(GENOMES / refname)
    # predictions grouped by GM
    pred = defaultdict(list)
    for line in open(pred_file):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 9 or f[2] not in ("CDS", "exon", "five_prime_UTR", "three_prime_UTR"):
            continue
        m = re.search(r'GM=([^;]+)', f[8])
        if not m:
            continue
        pred[(f[0], m.group(1))].append((int(f[3]), int(f[4])))
    for (chrom, gm), iv in pred.items():
        gm_base = gm.replace("-rc", "")
        if gm_base not in ref:
            continue
        chr_, gs, ge, rex, ncds = ref[gm_base]
        if chr_ != chrom or not rex:
            continue
        piv = merge(iv)
        ov = overlap_bp(piv, rex)
        rec = ov / span(rex)
        prec = ov / span(piv) if span(piv) else 0.0
        f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0.0
        rows_out.append((sp, gm, ge - gs + 1, ncds, round(rec, 4), round(prec, 4), round(f1, 4)))
    print(sp, "done", flush=True)

with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["species", "gene_model", "gene_len", "n_cds", "recall", "precision", "f1"])
    w.writerows(rows_out)
print("wrote", OUT, len(rows_out), "rows")
