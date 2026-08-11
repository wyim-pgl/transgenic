#!/usr/bin/env python3
"""Reconstruct torch.utils.data.random_split(seed=123) over the training dataset
without torch, and assign every A. thaliana gene to train / validation / test.

torch.manual_seed(123) seeds the CPU generator's MT19937 with init_genrand(123).
random_split calls torch.randperm(n, generator=g), whose CPU kernel is a
Fisher-Yates shuffle drawing one raw uint32 per step:

    for i in 0..n-2:  z = gen.random() % (n - i);  swap(r[i], r[i+z])

then slices the permutation into the requested lengths in order.

The reconstruction is validated against revision/results/fig3_regen/fig3_test_genes.tsv,
the A. thaliana test-split gene list produced by the real torch on the real database.
"""
import json
import sys
from pathlib import Path

N_MT = 624
MATRIX_A = 0x9908B0DF
UPPER_MASK = 0x80000000
LOWER_MASK = 0x7FFFFFFF


class MT19937:
    """Reference MT19937 with init_genrand seeding, matching at::mt19937."""

    def __init__(self, seed):
        self.mt = [0] * N_MT
        self.mt[0] = seed & 0xFFFFFFFF
        for j in range(1, N_MT):
            self.mt[j] = (1812433253 * (self.mt[j - 1] ^ (self.mt[j - 1] >> 30)) + j) & 0xFFFFFFFF
        self.idx = N_MT

    def _twist(self):
        mt = self.mt
        for i in range(N_MT):
            y = (mt[i] & UPPER_MASK) | (mt[(i + 1) % N_MT] & LOWER_MASK)
            nxt = y >> 1
            if y & 1:
                nxt ^= MATRIX_A
            mt[i] = mt[(i + 397) % N_MT] ^ nxt
        self.idx = 0

    def random(self):
        if self.idx >= N_MT:
            self._twist()
        y = self.mt[self.idx]
        self.idx += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF


def randperm(n, seed=123):
    g = MT19937(seed)
    r = list(range(n))
    rnd = g.random
    for i in range(n - 1):
        z = rnd() % (n - i)
        j = i + z
        r[i], r[j] = r[j], r[i]
    return r


SCRATCH = Path(__file__).resolve().parent
REG = Path("/data/gpfs/assoc/pgl/data/Transgenic/transgenic/revision/results/fig3_regen/fig3_test_genes.tsv")

# Row counts replicated from the real FASTA+GFF3 with the real preprocess.py.
COUNTS = json.loads((SCRATCH / "allsp" / "row_counts.json").read_text())
# 176 legacy GRMZM maize rows carry no "Zm" prefix and so survive exclude_prefix="Zm";
# they are documented in revision/scripts/fig3_infer.py and sit in the maize block,
# which was appended last. They only extend the index space past A. thaliana.
GRMZM_ROWS = 176


def main():
    order = sys.argv[1].split(",") if len(sys.argv) > 1 else [
        "TAIR10", "MSUv7", "Glyma", "Sobic", "Potri", "Bradi", "Vitvi", "Seita", "Pp3"]
    total = sum(COUNTS[s] for s in order) + GRMZM_ROWS
    train_size = int(total * 0.75)
    eval_size = int(total * 0.10)
    test_size = total - train_size - eval_size
    print(f"total={total} train={train_size} eval={eval_size} test={test_size}", flush=True)

    offset = 0
    for s in order:
        if s == "TAIR10":
            break
        offset += COUNTS[s]
    at_rows = (SCRATCH / "allsp" / "rows_TAIR10.txt").read_text().split()
    assert len(at_rows) == COUNTS["TAIR10"], (len(at_rows), COUNTS["TAIR10"])

    print("building permutation...", flush=True)
    perm = randperm(total, 123)

    # slice index -> split label, for the A. thaliana index window only
    label = {}
    for pos, idx in enumerate(perm):
        if offset <= idx < offset + len(at_rows):
            if pos < train_size:
                label[idx] = "train"
            elif pos < train_size + eval_size:
                label[idx] = "validation"
            else:
                label[idx] = "test"
    assert len(label) == len(at_rows)

    # per-gene: a gene is "seen in training" if ANY of its rows (forward or rc) is in train
    gene_splits = {}
    row_split_counts = {"train": 0, "validation": 0, "test": 0}
    for i, gm in enumerate(at_rows):
        sp = label[offset + i]
        row_split_counts[sp] += 1
        gene = gm[:-3] if gm.endswith("-rc") else gm
        gene_splits.setdefault(gene, set()).add(sp)

    def category(sps):
        if "train" in sps:
            return "train"
        if "validation" in sps:
            return "validation"
        return "test"

    gene_cat = {g: category(s) for g, s in gene_splits.items()}
    cat_counts = {}
    for c in gene_cat.values():
        cat_counts[c] = cat_counts.get(c, 0) + 1

    # ---- validation against the real torch run -------------------------------
    real_test = set()
    with REG.open() as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) == 2 and f[0] == "A_thaliana":
                real_test.add(f[1])
    mine_any_test = {g for g, s in gene_splits.items() if "test" in s}
    ok = mine_any_test == real_test
    print(f"reconstructed genes with >=1 test row: {len(mine_any_test)}")
    print(f"fig3_regen A_thaliana test genes:      {len(real_test)}")
    print(f"EXACT SET MATCH: {ok}")
    if not ok:
        print(f"  only mine: {len(mine_any_test - real_test)}  only real: {len(real_test - mine_any_test)}")

    out = {
        "species_order_assumed": order,
        "total_rows_non_Zm_prefixed": total,
        "split_sizes": {"train": train_size, "validation": eval_size, "test": test_size},
        "A_thaliana_rows": len(at_rows),
        "A_thaliana_row_split_counts": row_split_counts,
        "A_thaliana_gene_categories_any_train_wins": cat_counts,
        "validation_against_fig3_regen": {
            "reconstructed_genes_with_test_row": len(mine_any_test),
            "fig3_regen_test_genes": len(real_test),
            "exact_set_match": ok,
        },
    }
    (SCRATCH / "split_reconstruction.json").write_text(json.dumps(out, indent=1))
    with (SCRATCH / "at_gene_split.tsv").open("w") as fh:
        fh.write("gene\tcategory\trow_splits\n")
        for g in sorted(gene_cat):
            fh.write(f"{g}\t{gene_cat[g]}\t{','.join(sorted(gene_splits[g]))}\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
