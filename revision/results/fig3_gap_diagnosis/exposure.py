import re, sys
from pathlib import Path
sys.path.insert(0, "/data/gpfs/assoc/pgl/tmp/scratch")
from diagnose import write_ref_subset, load_ref_genes, OUT, SPLIT_TSV, GM_RE

ref_genes = load_ref_genes()
cat = {}
for line in SPLIT_TSV.read_text().splitlines()[1:]:
    p = line.split("\t")
    if len(p) >= 2:
        cat[p[0]] = p[1]
heldout = {g for g in ref_genes if cat.get(g) == "test"}
exposed = {g for g in ref_genes if cat.get(g) in ("train", "validation")}

lines = (OUT / "pred_all_repaired.gff3").read_text().splitlines(keepends=True)
def gene_of(l):
    gm = GM_RE.search(l).group(1)
    return gm[:-3] if gm.endswith("-rc") else gm

h = [l for l in lines if gene_of(l) in heldout]
e = [l for l in lines if gene_of(l) in exposed]
(OUT / "pred_all_repaired_heldout.gff3").write_text("".join(h))
(OUT / "pred_all_repaired_exposed.gff3").write_text("".join(e))
print("heldout genes", len(heldout), "exposed genes", len(exposed))
print("heldout lines", len(h), "exposed lines", len(e))
for name, gs in (("ref_heldout_all", heldout), ("ref_exposed_all", exposed)):
    print(name, write_ref_subset(gs, OUT / f"{name}.gff3"))

# how many rc-only genes are exposed?
rc_only = {g.strip() for g in Path("/data/gpfs/assoc/pgl/tmp/scratch/rc_only_genes.txt").read_text().split()}
print("rc-only genes that are train/val-exposed:", len(rc_only & exposed), "of", len(rc_only))
