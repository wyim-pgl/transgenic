import sys
sys.path.insert(0, "/home/framazan/transgenic/src")
import torch
from transgenic.datasets.datasets import isoformDataHyena

DB = "/home/framazan/data/Generation_10G_static6144_addExtra200_addRCIsoOnly_clean.db"

torch.manual_seed(123)
ds = isoformDataHyena(DB, mode="train", exclude_prefix="Zm")
total = len(ds)
train_size = int(total * 0.75)
eval_size = int(total * 0.10)
test_size = total - train_size - eval_size
train_data, eval_data, test_data = torch.utils.data.random_split(ds, [train_size, eval_size, test_size])
print(f"total={total} train={train_size} eval={eval_size} test={test_size} | test.indices={len(test_data.indices)}")

import duckdb, collections
con = duckdb.connect(DB, config={"access_mode": "READ_ONLY"})
test_rns = sorted(ds._index_map[i] for i in test_data.indices)
print("test_rns:", len(test_rns), "first/last:", test_rns[:3], test_rns[-3:])

# species distribution of test rows (by geneModel prefix heuristic via chromosome)
import csv
with open("/home/framazan/fig3_test_rows.tsv", "w") as out:
    w = csv.writer(out, delimiter="\t")
    w.writerow(["rn","geneModel","start","fin","strand","chromosome"])
    # chunk the rn list to avoid huge SQL
    B = 5000
    for i in range(0, len(test_rns), B):
        chunk = test_rns[i:i+B]
        q = f"SELECT rn, geneModel, start, fin, strand, chromosome FROM geneList WHERE rn IN ({','.join(map(str,chunk))})"
        for row in con.sql(q).fetchall():
            w.writerow(row)

# Z. mays rows (held out entirely)
zm = con.sql("SELECT COUNT(*) FROM geneList WHERE geneModel LIKE 'Zm%'").fetchone()[0]
print("Zm rows in db:", zm)
cnt = collections.Counter()
for rn in test_rns[:0]:
    pass
con.close()
print("WROTE /home/framazan/fig3_test_rows.tsv")
