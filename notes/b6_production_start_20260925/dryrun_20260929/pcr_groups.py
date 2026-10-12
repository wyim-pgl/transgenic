"""Largest A16 PCR-equivalence groups (unit_kind='pcr', per library + unit_key) in a C0 staging DB (read-only)."""
import sys, json, duckdb
path = sys.argv[1]
con = duckdb.connect(path, read_only=True)
con.execute("SET memory_limit = '24GB'")
tabs = [r[0] for r in con.execute("SELECT table_name FROM information_schema.tables ORDER BY 1").fetchall()]
print(json.dumps({"tables": tabs}), flush=True)
t = "member"
print(json.dumps({"members": con.execute(f"SELECT count(*), count(*) FILTER (WHERE unit_kind = 'pcr') FROM {t}").fetchone()}), flush=True)
top = con.execute(f"""SELECT library, unit_key, count(*) n FROM {t} WHERE unit_kind = 'pcr'
                      GROUP BY library, unit_key ORDER BY n DESC LIMIT 10""").fetchall()
print(json.dumps({"top_groups": [[l, k[:80], n] for l, k, n in top]}), flush=True)
dist = con.execute(f"""SELECT count(*) groups, max(n), sum(n) FILTER (WHERE n > 100000), count(*) FILTER (WHERE n > 100000)
                       FROM (SELECT count(*) n FROM {t} WHERE unit_kind = 'pcr' GROUP BY library, unit_key)""").fetchone()
print(json.dumps({"groups": dist[0], "max": dist[1], "reads_in_groups_gt_100k": dist[2], "groups_gt_100k": dist[3]}), flush=True)
