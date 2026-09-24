#!/usr/bin/env python3
"""Independent checks of an ``evlabel_b6_v1.db`` (plan v4.1 §5).

    b6_validate_evlabel.py --evlabel evlabel_b6_v1.db --merged evidence_c0.db --corpus b5_full_b6_v1.db
                           --roles DATASET_ROLES.b6_v1.tsv [--protein-placement <sp>.db ...] [--report out.json]

* bindings: output sidecar + PROVENANCE; merged / corpus / roles / supplement shas equal the recorded inputs;
  ``consumer_policy`` and ``boundary_policy`` present.
* junction ``n``: recomputed per (species, source, scope, junction) in SQL written independently of the builder
  (distinct eligible molecules per library, EST capped at 10, summed; protein = distinct organisms, >= 2) and equal
  to ``evlabel_junction``; and n <= sum of the C0 ``junction.n_molecules_train_track`` of that junction.
* cells: class index in range, 0 < weight <= 4, inside the tile, never on an N base, never in a test tile; every
  intron-family cell lies on a recorded junction; RC tiles are the mirror image of their forward tiles.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from transgenic.evidence import b6_evlabel as EV  # noqa: E402
from transgenic.evidence import b6_protein_placement as PP  # noqa: E402
from transgenic.evidence import b6_roles as R  # noqa: E402
from transgenic.evidence import leakage as LK  # noqa: E402


class EvlabelValidationError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise EvlabelValidationError(message)


# Independent SQL for junction n (does not reuse the builder's Python): eligible molecule = all members train_track
# and training_eligible and no b6_score (dataset, run); EST from primary-arm accepted primaries; train scope drops
# train_tiles_excluded molecules.
N_SQL = """
WITH score(dataset, run) AS (SELECT * FROM (VALUES {score_values}) v(dataset, run)),
mm AS (
  SELECT x.*, (sc.dataset IS NOT NULL) AS scored
  FROM m.molecule_member x LEFT JOIN score sc ON sc.dataset = x.dataset AND sc.run = x.run),
mol AS (
  SELECT species, source, molecule_id, min(library) AS lib, bool_or(train_tiles_excluded) AS tte
  FROM mm GROUP BY species, source, molecule_id
  HAVING bool_and(molecule_state = 'train_track') AND bool_and(role = 'training_eligible') AND NOT bool_or(scored)),
prim AS (
  SELECT alignment_id FROM m.evidence_alignment
  WHERE accepted AND placement_kind = 'primary' AND (source <> 'est' OR arm = 'primary')
    AND (source <> 'protein' OR alignment_id IN (SELECT alignment_id FROM sup))),
obs AS (
  SELECT DISTINCT o.species, o.source, o.junction_key, o.molecule_id
  FROM m.junction_observation o
  JOIN m.junction_support s USING (species, source, junction_key, dataset, run, library, role, molecule_id)
  WHERE o.accepted AND o.canonical AND o.strand IN ('+', '-') AND NOT s.excluded_4c
    AND o.alignment_id IN (SELECT alignment_id FROM prim)
    AND NOT EXISTS (SELECT 1 FROM m.excluded_junction_coordinate x WHERE x.species = o.species AND x.chrom = o.chrom
                    AND x.strand = o.strand AND ((x.kind = 'donor' AND x.coordinate0 = o.donor0)
                                                 OR (x.kind = 'acceptor' AND x.coordinate0 = o.acceptor0)))),
scoped AS (
  SELECT sc.scope, obs.species, obs.source, obs.junction_key, obs.molecule_id, mol.lib
  FROM obs JOIN mol USING (species, source, molecule_id)
  CROSS JOIN (VALUES ('train'), ('valid')) sc(scope)
  WHERE sc.scope = 'valid' OR NOT mol.tte),
per_lib AS (
  SELECT scope, species, source, junction_key, lib, count(DISTINCT molecule_id) AS k FROM scoped GROUP BY ALL)
SELECT species, source, scope, junction_key,
       CASE WHEN source = 'protein' THEN count(*) WHEN source = 'est' THEN sum(least(10, k)) ELSE sum(k) END AS n
FROM per_lib GROUP BY species, source, scope, junction_key
HAVING source <> 'protein' OR count(*) >= 2
ORDER BY ALL
"""


def validate(a):
    import duckdb

    qc = Counter()
    ev = Path(a.evlabel)
    sha = LK.sha256(ev)
    require(Path(str(ev) + ".sha256").read_text().split()[:1] == [sha], "evlabel sidecar != file sha256")
    prov = json.loads(Path(str(ev) + ".PROVENANCE.json").read_text())
    require(prov["output_sha256"] == sha, "PROVENANCE output_sha256 != file")
    require(prov["consumer_policy"] == EV.CONSUMER_POLICY and prov["boundary_policy"] == EV.BOUNDARY_POLICY,
            "consumer/boundary policy missing")
    inp = prov["inputs"]
    require(LK.sha256(a.merged) == inp["merged"]["sha256"], "merged sha256 != recorded")
    require(LK.sha256(a.corpus) == inp["corpus"]["sha256"], "corpus sha256 != recorded")
    roles = R.load_roles(a.roles)
    require(roles.sha256 == inp["roles"]["sha256"], "roles sha256 != recorded")
    sups = []
    for p in a.protein_placement or []:
        s, sp = PP.open_supplement(p)
        require(inp["protein_supplements"].get(sp["species"], {}).get("sha256") == s, f"{p}: sha256 != recorded")
        sups.append(str(Path(p).absolute()))
    require(len(sups) == len(inp["protein_supplements"]), "not every recorded protein supplement was supplied")
    con = duckdb.connect(str(ev), read_only=True)
    try:
        con.execute(f"ATTACH '{Path(a.merged).absolute()}' AS m (READ_ONLY)")
        con.execute(f"ATTACH '{Path(a.corpus).absolute()}' AS c (READ_ONLY)")
        if sups:
            parts = []
            for i, p in enumerate(sups):
                con.execute(f"ATTACH '{p}' AS s{i} (READ_ONLY)")
                parts.append(f"SELECT alignment_id, molecule_id FROM s{i}.protein_placement")
            con.execute("CREATE TEMP VIEW sup AS " + " UNION ALL ".join(parts))
        else:
            con.execute("CREATE TEMP VIEW sup AS SELECT NULL::VARCHAR AS alignment_id, NULL::VARCHAR AS molecule_id "
                        "WHERE false")
        score = [(r.dataset, r.run) for r in roles.rows if r.b6_score] or [("", "")]
        values = ", ".join("(" + ", ".join("'" + x.replace("'", "''") + "'" for x in t) + ")" for t in score)
        want = con.execute(N_SQL.format(score_values=values)).fetchall()
        got = con.execute("SELECT species, source, scope, junction_key, n FROM evlabel_junction ORDER BY ALL").fetchall()
        require(sorted(got) == sorted(want), f"junction n differs from the independent SQL: "
                                             f"{sorted(set(got) ^ set(want))[:5]}")
        qc["junctions_n_verified"] = len(got)
        over = con.execute("""
            SELECT e.species, e.source, e.junction_key, e.n, sum(j.n_molecules_train_track) FROM evlabel_junction e
            JOIN m.junction j ON j.species = e.species AND j.source = e.source AND j.junction_key = e.junction_key
            GROUP BY ALL HAVING e.n > sum(j.n_molecules_train_track) AND e.source <> 'protein'""").fetchall()
        require(not over, f"junction n exceeds sum(n_molecules_train_track): {over[:3]}")
        n_cls = len(LK.load_build_b5().gc.SEG_CLASSES)
        bad = con.execute(f"""SELECT count(*) FROM evlabel_cell x JOIN evlabel_tile t USING (rn)
                              WHERE x.class_index < 0 OR x.class_index >= {n_cls} OR x.end0 > t.L OR x.weight <= 0
                                 OR x.weight > 4 OR t.split NOT IN ('train', 'valid')""").fetchone()[0]
        require(bad == 0, f"{bad} evlabel cells out of range, weight or split")
        tiles = con.execute("SELECT rn, L, is_rc, split FROM evlabel_tile ORDER BY rn").fetchall()
        seqs = dict(con.execute("SELECT rn, sequence FROM c.geneList WHERE rn IN (SELECT rn FROM evlabel_tile) "
                                "ORDER BY rn").fetchall())
        wids = dict(con.execute("SELECT rn, geneModel FROM c.geneList ORDER BY rn").fetchall())
        by_wid = {w: rn for rn, w in wids.items()}
        test_rns = {r for (r,) in con.execute("SELECT rn FROM c.geneList WHERE split = 'test' ORDER BY rn").fetchall()}
        require(not (test_rns & {t[0] for t in tiles}), "a test tile is labelled")
        cells_by = {}
        for rn, L, is_rc, _ in tiles:
            cells = con.execute("SELECT class_index, start0, end0, weight, source, rule FROM evlabel_cell WHERE rn = ? "
                                "ORDER BY ALL", [rn]).fetchall()
            seq = seqs[rn]
            for cls, s, e, *_ in cells:
                require("N" not in seq[s:e] and "n" not in seq[s:e], f"rn {rn}: cell {cls} {s}-{e} lies on N")
            cells_by[rn] = cells
            qc["cells"] += len(cells)
        for rn, L, is_rc, _ in tiles:
            if not is_rc:
                continue
            f = by_wid.get(wids[rn][:-3])
            require(f in cells_by, f"RC tile {rn} has no labelled forward tile")
            mirrored = sorted((c, L - e, L - s, w, src, rule) for c, s, e, w, src, rule in cells_by[f])
            require(mirrored == cells_by[rn], f"RC tile {rn} is not the mirror of forward tile {f}")
            qc["rc_tiles_mirrored"] += 1
        junc = con.execute("""SELECT count(*) FROM evlabel_cell x JOIN evlabel_tile t USING (rn)
                              WHERE x.family = 'intron' AND x.rule = 'junction' AND NOT EXISTS (
                                SELECT 1 FROM evlabel_junction j WHERE j.species = t.species AND j.chrom = t.chrom
                                  AND j.source = x.source AND j.scope = t.scope
                                  AND least(j.donor0, j.acceptor0) <= CASE WHEN t.is_rc THEN t.we - x.end0 ELSE t.ws + x.start0 END
                                  AND greatest(j.donor0, j.acceptor0) + 1 >= CASE WHEN t.is_rc THEN t.we - x.start0 ELSE t.ws + x.end0 END)
                           """).fetchone()[0]
        require(junc == 0, f"{junc} intron-family cells lie on no recorded junction")
        # review 1 H1, independent of the builder: every phase row comes from a supplement alignment whose protein
        # molecule is eligible (train_track, training_eligible) and, on a train tile, not train_tiles_excluded.
        phase_bad = con.execute("""
            WITH mm AS (SELECT species, molecule_id, bool_and(molecule_state = 'train_track') AS tt,
                               bool_and(role = 'training_eligible') AS te, bool_or(train_tiles_excluded) AS tte
                        FROM m.molecule_member WHERE source = 'protein' GROUP BY species, molecule_id)
            SELECT count(*) FROM evlabel_phase ph JOIN evlabel_tile t USING (rn)
            LEFT JOIN sup ON sup.alignment_id = ph.alignment_id
            LEFT JOIN mm ON mm.species = t.species AND mm.molecule_id = sup.molecule_id
            WHERE sup.alignment_id IS NULL OR mm.molecule_id IS NULL OR NOT (mm.tt AND mm.te)
               OR (t.scope = 'train' AND mm.tte)""").fetchone()[0]
        require(phase_bad == 0, f"{phase_bad} evlabel_phase rows come from a protein not eligible for their tile scope")
        qc["phase_rows_eligible"] = con.execute("SELECT count(*) FROM evlabel_phase").fetchone()[0]
    finally:
        con.close()
    return {"status": "PASS", "qc": dict(sorted(qc.items()))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evlabel", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--protein-placement", nargs="*", default=[])
    ap.add_argument("--report")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        out = validate(a)
    except (EvlabelValidationError, EV.EvlabelError, PP.PlacementError, R.RolesError, OSError, KeyError) as exc:
        print(f"b6_validate_evlabel FAILED: {exc}", file=sys.stderr)
        return 1
    if a.report:
        with open(a.report, "x") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
            fh.write("\n")
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
