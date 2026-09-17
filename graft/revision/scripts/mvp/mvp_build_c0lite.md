# revision/scripts/mvp/mvp_build_c0lite.py

- sha256 · function · L27-L32 — def sha256(path)
- require · function · L35-L37 — def require(ok, message)
- batches · function · L40-L51 — def batches(con, sql)
- clone_key · function · L54-L55 — def clone_key(name)
- load_sources · function · L58-L111 — def load_sources(con, corpus_db, panels, flank, references)
- stage_bam · function · L114-L168 — def stage_bam(con, fh)
- flush · function · L121-L126 — def flush(table)
- add · function · L128-L131 — def add(table, row)
- resolve_components · function · L171-L274 — def resolve_components(con, qc): # ponytail: O(unique QNAMEs) union-find RAM; placements/geometry stay on disk. # Replace with external-memory connectivity only if the real name count needs it.
- root · function · L178-L182 — def root(name)
- union · function · L184-L189 — def union(a, b)
- final_qc · function · L277-L296 — def final_qc(con, qc)
- build_c0lite · function · L299-L346 — def build_c0lite(*, bam, panels, corpus_db, out, flank=150)
- main · function · L349-L362 — def main(argv=None)
