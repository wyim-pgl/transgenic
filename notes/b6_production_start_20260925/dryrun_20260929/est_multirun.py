"""Codex finding 1 prevalence: EST molecules whose members span more than one (dataset, run) / role / arm (read-only)."""
import sys, json, duckdb
for path in sys.argv[1:]:
    con = duckdb.connect(path, read_only=True)
    try:
        r = con.execute("""
            WITH m AS (SELECT species, source, member_id, any_value(molecule_id) molecule_id FROM molecule_member
                       GROUP BY species, source, member_id)
            SELECT count(*) molecules,
                   count(*) FILTER (WHERE n_dsrun > 1) multi_dataset_run,
                   count(*) FILTER (WHERE n_role > 1) multi_role,
                   count(*) FILTER (WHERE n_arm > 1) multi_arm
            FROM (SELECT m.molecule_id, count(DISTINCT (a.dataset, a.run)) n_dsrun, count(DISTINCT a.role) n_role,
                         count(DISTINCT a.arm) n_arm
                  FROM evidence_alignment a JOIN m ON m.species = a.species AND m.source = a.source
                   AND m.member_id = a.member_id WHERE a.source = 'est' GROUP BY m.molecule_id)""").fetchone()
        ex = con.execute("""
            WITH m AS (SELECT species, source, member_id, any_value(molecule_id) molecule_id FROM molecule_member
                       GROUP BY species, source, member_id)
            SELECT m.molecule_id, list(DISTINCT a.dataset || '/' || a.run) FROM evidence_alignment a JOIN m
              ON m.species = a.species AND m.source = a.source AND m.member_id = a.member_id WHERE a.source = 'est'
            GROUP BY m.molecule_id HAVING count(DISTINCT (a.dataset, a.run)) > 1 LIMIT 2""").fetchall()
    finally:
        con.close()
    print(json.dumps({"db": path.split('/c0/')[1], "molecules": r[0], "multi_dataset_run": r[1], "multi_role": r[2],
                      "multi_arm": r[3], "examples": ex}, default=str), flush=True)
