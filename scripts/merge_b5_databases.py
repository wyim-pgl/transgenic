#!/usr/bin/env python3
"""Merge the per-species B5 tile databases into the single frozen B5 database (issue #50).

One DuckDB file cannot take parallel writers, so the nine species were built into separate files.
This merges them back preserving `rn` uniqueness and every table, verifies that the nine builds agree
on the frozen inputs, and writes a freeze manifest with a *content* hash that does not depend on the
DuckDB file layout (the file md5 is recorded too, but it is not reproducible across rebuilds).
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
import types

import duckdb

# loaded by path, the way tests/conftest.py does: importing the transgenic package pulls in torch.
_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "transgenic", "datasets", "build_b5.py")
_b5 = types.ModuleType("build_b5")
_b5.__file__ = _PATH
sys.modules["build_b5"] = _b5
with open(_PATH) as _fh:
    exec(compile(_fh.read(), _PATH, "exec"), _b5.__dict__)
ensure_schema, sha256 = _b5.ensure_schema, _b5.sha256

# Copied verbatim, no rewriting: the per-species tables carry species_id already.
VERBATIM_TABLES = ("gene_key_map", "build_manifest", "rejected_records", "window_genes", "tile_blocks")
# Identical in all nine builds (the frozen split table loaded into each): copied once.
ONCE_TABLES = ("gene_split",)
# build_manifest fields that must agree across the nine builds or the merge is refused.
FROZEN_FIELDS = ("split_file_sha256", "rc_mode", "build_version", "ordering_version", "window_policy", "duckdb_version")


def table_columns(con, table, schema="main"):
    """information_schema does not see an ATTACHed catalog, so ask DESCRIBE (works for both)."""
    return [r[0] for r in con.sql(f'DESCRIBE {schema}."{table}"').fetchall()]


def stream_hash(con, table, columns):
    """A content sha256 that does not depend on the DuckDB file layout, on `rn`, or on row order.

    Each row is reduced to an md5 inside DuckDB (one streaming pass; the 16 GB `sequence` column never
    reaches Python), the digests are sorted (272k x 32 B, trivial) and folded into one sha256. Sorting the
    digests rather than the rows is what keeps this affordable: `ORDER BY <cols> LIMIT/OFFSET` would re-sort
    the whole table, sequence column included, once per chunk.
    """
    parts = ", ".join(f"coalesce(\"{c}\"::VARCHAR, '\\N')" for c in columns)
    n = con.sql(f'SELECT count(*) FROM "{table}"').fetchone()[0]
    h = hashlib.sha256()
    cur = con.execute(f'SELECT md5(concat_ws(chr(1), {parts})) AS d FROM "{table}" ORDER BY d')
    while True:
        batch = cur.fetchmany(50000)
        if not batch:
            break
        h.update("".join(r[0] for r in batch).encode())
    return h.hexdigest(), n


def preflight(sources):
    """Refuse to merge builds that disagree on a frozen input, and report the git commits used."""
    frozen, commits, split_hashes = {}, {}, {}
    for path in sources:
        sp = os.path.basename(path)[:-3]
        con = duckdb.connect(path, read_only=True)
        rows = con.sql(f"SELECT species_id, {', '.join(FROZEN_FIELDS)}, git_commit FROM build_manifest").fetchall()
        if len(rows) != 1:
            raise SystemExit(f"{sp}: build_manifest has {len(rows)} rows, expected 1")
        if rows[0][0] != sp:
            raise SystemExit(f"{sp}: build_manifest species_id is {rows[0][0]}")
        frozen.setdefault(tuple(rows[0][1:-1]), []).append(sp)
        commits.setdefault(rows[0][-1], []).append(sp)
        gs_cols = table_columns(con, "gene_split")
        split_hashes[sp] = stream_hash(con, "gene_split", gs_cols)
        con.close()
    if len(frozen) != 1:
        raise SystemExit("the nine builds disagree on a frozen input:\n" +
                         "\n".join(f"  {dict(zip(FROZEN_FIELDS, k))} -> {v}" for k, v in frozen.items()))
    if len(set(split_hashes.values())) != 1:
        raise SystemExit("gene_split differs between builds:\n" +
                         "\n".join(f"  {k}: {v}" for k, v in sorted(split_hashes.items())))
    return dict(zip(FROZEN_FIELDS, next(iter(frozen)))), commits, next(iter(split_hashes.values()))


def merge(out, sources, frozen, commits, split_hash, split_table, qc_flags):
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; refusing to overwrite a frozen artifact")
    con = duckdb.connect(out)
    ensure_schema(con)
    per_species, rn_ranges = {}, {}
    for i, path in enumerate(sources):
        sp = os.path.basename(path)[:-3]
        con.sql(f"ATTACH '{path}' AS src (READ_ONLY)")
        gl_cols = [c for c in table_columns(con, "geneList", "src") if c != "rn"]
        quoted = ", ".join(f'"{c}"' for c in gl_cols)
        before = con.sql("SELECT count(*) FROM geneList").fetchone()[0]
        # NOT nextval(): a sequence in the SELECT list is evaluated during the (parallel) scan, before the
        # sort, so `INSERT ... SELECT nextval(...) ... ORDER BY rn` assigns row numbers in thread arrival
        # order. Measured on this box with duckdb 1.5.5 at 16 threads: 300,000 of 300,000 rows landed in the
        # wrong position even with the source physically stored in rn order. row_number() OVER (ORDER BY rn)
        # is defined to follow its own ORDER BY, and measured 0 of 300,000 wrong.
        con.sql(f"INSERT INTO geneList (rn, {quoted}) "
                f"SELECT row_number() OVER (ORDER BY rn) + {before}, {quoted} FROM src.\"geneList\"")
        after = con.sql("SELECT count(*) FROM geneList").fetchone()[0]
        src_n = con.sql("SELECT count(*) FROM src.\"geneList\"").fetchone()[0]
        if after - before != src_n:
            raise SystemExit(f"{sp}: inserted {after - before} geneList rows, source has {src_n}")
        per_species[sp] = src_n
        rn_ranges[sp] = con.sql("SELECT min(rn), max(rn) FROM geneList WHERE species_id = ?", params=[sp]).fetchone()
        for t in VERBATIM_TABLES:
            # window_genes / tile_blocks / rejected_records are created by the tile builder, not ensure_schema:
            # inherit their exact schema from the first source instead of restating the DDL here.
            con.sql(f'CREATE TABLE IF NOT EXISTS "{t}" AS SELECT * FROM src."{t}" LIMIT 0')
            q = ", ".join(f'"{c}"' for c in table_columns(con, t, "src"))
            con.sql(f'INSERT INTO "{t}" ({q}) SELECT {q} FROM src."{t}"')
        if i == 0:
            for t in ONCE_TABLES:
                cols = ", ".join(f'"{c}"' for c in table_columns(con, t, "src"))
                con.sql(f'INSERT INTO "{t}" ({cols}) SELECT {cols} FROM src."{t}"')
        con.sql("DETACH src")
        print(f"[{sp}] {src_n} rows, rn {rn_ranges[sp][0]}..{rn_ranges[sp][1]}", flush=True)
    # the frozen artifact keeps a usable sequence: leaving row_id at 1 would collide on any later append
    n_rows_total = con.sql("SELECT max(rn) FROM geneList").fetchone()[0] or 0
    con.sql("DROP SEQUENCE IF EXISTS row_id")
    con.sql(f"CREATE SEQUENCE row_id START {n_rows_total + 1}")
    con.sql("CHECKPOINT")
    gl_cols = table_columns(con, "geneList")
    content_hash, n_rows = stream_hash(con, "geneList", [c for c in gl_cols if c != "rn"])
    table_hashes = {t: stream_hash(con, t, table_columns(con, t))
                    for t in ("gene_key_map", "gene_split", "window_genes", "tile_blocks", "rejected_records", "build_manifest")}
    con.close()
    manifest = {
        "database": os.path.abspath(out),
        "merged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sources": [os.path.abspath(p) for p in sources],
        "source_md5": {os.path.basename(p)[:-3]: md5(p) for p in sources},
        "frozen_inputs": frozen,
        "git_commits": commits,
        "split_table": {"path": split_table, "sha256": sha256(split_table)} if split_table else None,
        "qc_flag_files": {os.path.basename(p): md5(p) for p in qc_flags},
        "rows_by_species": per_species,
        "rn_ranges": {k: list(v) for k, v in rn_ranges.items()},
        "geneList_rows": n_rows,
        "geneList_content_sha256": content_hash,
        "table_content_sha256": {k: {"sha256": v[0], "rows": v[1]} for k, v in table_hashes.items()},
        "gene_split_content_sha256": split_hash[0],
        "file_md5": md5(out),
        "file_bytes": os.path.getsize(out),
    }
    return manifest


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", required=True, help="directory holding <species>.db and <species>.DONE")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True, help="freeze manifest JSON")
    ap.add_argument("--split-table", default=None, help="data/splits/b5_orthogroup_split_v1.tsv, hashed into the manifest")
    ap.add_argument("--qc-flags", nargs="*", default=[], help="flag TSVs used by the builds, md5'd into the manifest")
    ap.add_argument("--expect-species", type=int, default=9)
    a = ap.parse_args()
    sources = sorted(glob.glob(os.path.join(a.src_dir, "*.db")))
    if len(sources) != a.expect_species:
        raise SystemExit(f"found {len(sources)} databases in {a.src_dir}, expected {a.expect_species}")
    for p in sources:
        done = p[:-3] + ".DONE"
        if not os.path.exists(done):
            raise SystemExit(f"{p} has no .DONE marker")
    frozen, commits, split_hash = preflight(sources)
    print(json.dumps({"frozen_inputs": frozen, "git_commits": commits}, indent=1), flush=True)
    manifest = merge(a.out, sources, frozen, commits, split_hash, a.split_table, a.qc_flags)
    with open(a.manifest, "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("sources", "source_md5")}, indent=1))


if __name__ == "__main__":
    main()
