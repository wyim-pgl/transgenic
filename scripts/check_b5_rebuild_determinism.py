#!/usr/bin/env python3
"""Does building the B5 database twice from identical inputs give the same database? (#15)

The third acceptance criterion of #15 - "rebuilding identical inputs produces an equivalent manifest
and row inventory" - had never been exercised. This checks it.

What "equivalent" means here matters. The DuckDB *file* md5 is not reproducible: page layout, free
lists and insertion threading all vary between runs, which is why the freeze records content hashes
alongside it. So the comparison is on content, using the same `stream_hash` the merge already uses -
an order-independent, layout-independent, `rn`-independent digest.

    python3 scripts/check_b5_rebuild_determinism.py <reference.db> <rebuilt.db>

Exits non-zero on any difference, and prints which one.
"""
import hashlib
import json
import os
import sys
import types

import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))

# Reuse the merge's hash rather than writing a second one: a determinism check that computes its own
# digest differently from the freeze would be answering a different question than the freeze asked.
_src = os.path.join(HERE, "merge_b5_databases.py")
_m = types.ModuleType("merge_b5_databases")
_m.__file__ = _src
sys.modules["merge_b5_databases"] = _m
exec(compile(open(_src).read(), _src, "exec"), _m.__dict__)
stream_hash = _m.stream_hash

# Fields that are expected to differ between two runs of the same build and say nothing about
# whether the data is the same.
VOLATILE = {"built", "built_at", "finished_at", "started", "wall_seconds", "file_md5", "file_bytes",
            "host", "hostname", "pid", "date"}

# Provenance, not content: these SHOULD differ between two runs made at different times or commits,
# and a determinism check that fails on them is answering the wrong question. They are reported so a
# reader can see what the second build actually was, but they do not make the result a failure.
# `git_commit` is the one that matters: if it differs, say so loudly enough that someone checks
# whether the builder itself changed between the two commits, because that is what would invalidate
# the comparison. (For the 2026-09-04 run it differed, 0aa5fab vs e36f051, and the four builder files
# were byte-identical across them.)
PROVENANCE = {"git_commit", "git_commits", "build_driver", "species_manifest"}


def cols(con, table):
    return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()]


def snapshot(path):
    con = duckdb.connect(path, read_only=True)
    con.execute("PRAGMA disable_progress_bar")
    tables = sorted(r[0] for r in con.execute("SHOW TABLES").fetchall())
    out = {"tables": tables, "rows": {}, "hashes": {}}
    for t in tables:
        n = con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
        out["rows"][t] = n
        # rn is assigned at merge time, not content. build_manifest additionally carries the build's
        # own provenance (commit, timestamp, file md5), which is exactly what two runs are supposed to
        # disagree on - hashing it would make the table permanently "different" and hide whether the
        # recorded *inputs* changed.
        drop = {"rn"} | (VOLATILE | PROVENANCE if t == "build_manifest" else set())
        c = [x for x in cols(con, t) if x not in drop]
        out["hashes"][t] = stream_hash(con, t, c)[0] if n else None
    if "build_manifest" in tables:
        rows = con.execute("SELECT * FROM build_manifest").fetchall()
        names = cols(con, "build_manifest")
        out["manifest"] = [{k: v for k, v in zip(names, r)
                            if k not in VOLATILE and k not in PROVENANCE} for r in rows]
        out["provenance"] = [{k: v for k, v in zip(names, r) if k in PROVENANCE} for r in rows]
    con.close()
    return out


def main(ref, new):
    a, b = snapshot(ref), snapshot(new)
    bad = []
    if a["tables"] != b["tables"]:
        bad.append(f"tables differ: {a['tables']} vs {b['tables']}")
    for t in sorted(set(a["tables"]) & set(b["tables"])):
        if a["rows"][t] != b["rows"][t]:
            bad.append(f"{t}: {a['rows'][t]} rows vs {b['rows'][t]}")
        elif a["hashes"][t] != b["hashes"][t]:
            bad.append(f"{t}: {a['rows'][t]} rows match but content differs "
                       f"({str(a['hashes'][t])[:16]} vs {str(b['hashes'][t])[:16]})")
    if a.get("manifest") != b.get("manifest"):
        bad.append("build_manifest differs on a content field once volatile and provenance are dropped")
    prov = []
    for x, y in zip(a.get("provenance", []), b.get("provenance", [])):
        for k in sorted(set(x) | set(y)):
            if str(x.get(k)) != str(y.get(k)):
                prov.append(f"{k}: {x.get(k)} -> {y.get(k)}")
    for t in sorted(set(a["tables"]) & set(b["tables"])):
        mark = "=" if a["hashes"][t] == b["hashes"][t] and a["rows"][t] == b["rows"][t] else "!"
        print(f"  {mark} {t:<20} {a['rows'][t]:>8} rows  {str(a['hashes'][t])[:16]}")
    if prov:
        print("\nprovenance (expected to differ; not a failure):")
        for x in prov:
            print("  ~", x)
        if any(x.startswith("git_commit") for x in prov):
            print("  NOTE: the two builds are from different commits. Confirm the builder itself did not")
            print("        change between them, or this comparison proves nothing:")
            print("        git diff --name-only <a> <b> -- scripts/build_b5_database.py \\")
            print("            src/transgenic/datasets/build_b5.py src/transgenic/utils/gsf_contract.py \\")
            print("            src/transgenic/datasets/qc_flags.py")
    if bad:
        print("\nDIFFERENT:")
        for x in bad:
            print("  -", x)
        return 1
    print("\nIDENTICAL: every table matches by row count and content hash, "
          "and build_manifest agrees on everything but volatile fields.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
