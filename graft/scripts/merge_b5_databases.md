# scripts/merge_b5_databases.py

- table_columns · function · L38-L40 — def table_columns(con, table, schema="main")
- stream_hash · function · L43-L60 — def stream_hash(con, table, columns)
- read_excluded · function · L63-L67 — def read_excluded(path)
- preflight · function · L70-L102 — def preflight(sources, excluded)
- merge · function · L105-L174 — def merge(out, sources, frozen, commits, split_hash, split_table, qc_flags, species_manifest=None)
- md5 · function · L177-L182 — def md5(path)
- completion_from_status · function · L185-L229 — def completion_from_status(sources, status_path)
- main · function · L232-L266 — def main()
