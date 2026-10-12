"""Row counts per table for every C0 DB to be merged (read-only), to size the production merge."""
import sys, json, duckdb
from transgenic.evidence import c0_schema as S
tot = {}
for path in sys.argv[1:]:
    con = duckdb.connect(path, read_only=True)
    try:
        row = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in S.ALL_TABLES if t not in ("merge_manifest",)}
    finally:
        con.close()
    for t, n in row.items():
        tot[t] = tot.get(t, 0) + n
    print(json.dumps({"db": path.split('/c0/')[1], **{t: n for t, n in row.items() if n}}), flush=True)
print(json.dumps({"TOTAL": tot, "pk": {t: S.primary_key(t) for t in S.ALL_TABLES}}))
