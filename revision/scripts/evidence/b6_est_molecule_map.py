#!/usr/bin/env python3
"""Export the C0 EST molecule units (A11 + A21 parts, b6_build_c0.est_units) as the ONE molecule map part B uses.

    b6_est_molecule_map.py --est-c0 Athaliana.est.c0.db --species Athaliana --out Athaliana.est_molecule_map.tsv

There is a single producer of EST molecule ids: ``b6_build_c0.est_units`` inside the EST C0 build.  This script only
reads the published EST C0 DB (its PROVENANCE output_sha256, schema c0-v1, species/source and a zero leakage gate are
verified) and writes, per EST record of the frozen primary alignment:

    qname                 the SAM QNAME / screened-FASTA record name (molecule_member.read_name)
    member_uid            = qname (so b6_hsp_project.MOLECULE_ID_RULE over member_uids is the C0 formula)
    molecule_id           molecule_member.molecule_id = sha256(json.dumps(sorted(set(QNAMEs of all members))))
    library               molecule_member.library (a molecule spanning libraries is refused)
    est_accession         molecule_member.accession (versioned accession, A21 '_partN' kept)
    molecule_state        train_track | excluded_heldout | never (rule 4 decision of the C0 build)
    train_tiles_excluded  true | false
    superseded            true | false (molecule_member.superseded_version)
    accepted              true | false: the member has an ACCEPTED (A18.5) primary alignment in evidence_alignment

Field text: the id fields (qname, member_uid, molecule_id, molecule_state, train_tiles_excluded, superseded, accepted)
must be
non-empty and whitespace-free; ``library`` and ``est_accession`` must be non-empty and free of tab, CR and LF only
-- ``library`` is the GenBank /clone_lib (or esummary clone_lib) text, which routinely holds spaces
(b6_c0_metadata.py joins wrapped qualifier lines with one space).

Superseded accession versions (superseded = true) supply no observation in C0, as part B's closure keeps only the
newest version.  They ARE exported, explicitly flagged, so that b6_hsp_project.py can drop exactly those qnames (and
count them) while any other qname absent from the map still fails its run.  Every molecule id is recomputed from ALL
its members (including superseded ones) and must equal the stored id.  Rejected members (accepted = false: the C0
primary was refused by A18.5 -- mapq, identity, chimeric, unmapped ...) are exported too, because rule 4(b) keeps them
in their molecule, but they supply NO observation: A46 rule 6 projects an HSP "through the EST's spliced alignment"
from retained observations only, so b6_hsp_project.py drops and counts their HSPs (rejected_alignment_qname_hsps)
and b6_build_c0.verify_hsp_placements refuses any HSP outside an accepted primary.  Rows are written in read_name order (the
file is byte-deterministic for one C0 DB).  Output: the TSV (exclusive create) and ``<out>.sha256``
('<hex>  <name>', sha256sum format) plus ``<out>.PROVENANCE.txt`` naming the EST C0 DB and its sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
if str(_HERE.parents[3] / "src") not in sys.path:
    sys.path.insert(0, str(_HERE.parents[3] / "src"))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import b6_build_c0 as B  # noqa: E402

COLUMNS = ("qname", "member_uid", "molecule_id", "library", "est_accession", "molecule_state", "train_tiles_excluded",
           "superseded", "accepted")
ID_FIELDS = ("qname", "member_uid", "molecule_id", "molecule_state", "train_tiles_excluded", "superseded", "accepted")
TEXT_FIELDS = ("library", "est_accession")          # free text: no tab / CR / LF (spaces allowed)


def check_fields(row):
    """None when the map row is valid text, else the reason (shared rule with b6_hsp_project.validate_molecule_map)."""
    for k in ID_FIELDS:
        if not row[k] or any(c.isspace() for c in row[k]):
            return f"{k} is empty or holds whitespace"
    for k in TEXT_FIELDS:
        if not row[k] or row[k] != row[k].strip() or any(c in row[k] for c in "\t\r\n"):
            return f"{k} is empty, padded or holds a tab/CR/LF"
    return None
STATES = ("train_track", "excluded_heldout", "never")


def export(est_c0, species, out):
    import duckdb

    out = Path(out).absolute()
    for p in (out, Path(str(out) + ".sha256"), Path(str(out) + ".PROVENANCE.txt")):
        if os.path.lexists(p):
            raise FileExistsError(p)
    db_sha, prov = B.open_published_c0(est_c0, species, "est")
    con = duckdb.connect(str(Path(est_c0).absolute()), read_only=True)
    partial = out.with_name(out.name + ".partial")
    try:
        q = lambda sql: con.execute(sql).fetchall()  # noqa: E731
        B.require(not q("SELECT read_name FROM molecule_member GROUP BY read_name HAVING count(*) > 1 LIMIT 1"),
                  "a QNAME is a member twice")
        multi = q("SELECT molecule_id FROM molecule_member GROUP BY molecule_id HAVING count(DISTINCT library) > 1 LIMIT 3")
        B.require(not multi, f"molecule(s) {[m[0] for m in multi]} span libraries; part B's map needs one library each")
        mixed = q("""SELECT molecule_id FROM molecule_member GROUP BY molecule_id
                     HAVING count(DISTINCT molecule_state) > 1 OR count(DISTINCT train_tiles_excluded) > 1 LIMIT 3""")
        B.require(not mixed, f"molecule(s) {[m[0] for m in mixed]} carry several states")
        for molecule, names in B.fetch_stream(con.cursor(), "SELECT molecule_id, list(read_name ORDER BY read_name) "
                                                            "FROM molecule_member GROUP BY molecule_id"):
            want = hashlib.sha256(json.dumps(sorted(set(names)), separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            B.require(want == molecule, f"molecule {molecule} != the A11 formula over its members {names[:3]}")
        n = n_superseded = n_rejected = 0
        with open(partial, "x") as fh:
            fh.write("\t".join(COLUMNS) + "\n")
            for row in B.fetch_stream(con.cursor(), """
                    SELECT m.read_name, m.molecule_id, m.library, m.accession, m.molecule_state, m.train_tiles_excluded,
                           m.superseded_version, coalesce(a.accepted, false)
                    FROM molecule_member m
                    LEFT JOIN (SELECT member_id, bool_or(accepted AND placement_kind = 'primary') AS accepted
                               FROM evidence_alignment GROUP BY member_id) a USING (member_id)
                    ORDER BY m.read_name"""):
                qname, molecule, library, accession, state, tte, superseded, accepted = row
                vals = [qname, qname, molecule, library, accession, state, "true" if tte else "false",
                        "true" if superseded else "false", "true" if accepted else "false"]
                B.require(state in STATES, f"{qname}: molecule_state {state!r}")
                bad = check_fields(dict(zip(COLUMNS, [v or "" for v in vals])))
                B.require(bad is None, f"{qname}: invalid map field ({bad}): {vals}")
                fh.write("\t".join(vals) + "\n")
                n += 1
                n_superseded += bool(superseded)
                n_rejected += not accepted
        B.require(n > n_superseded, f"{est_c0}: no current (non-superseded) EST member to export")
        os.link(partial, out)
    finally:
        con.close()
        partial.unlink(missing_ok=True)
    digest = B.digest(out)
    with open(str(out) + ".sha256", "x") as fh:
        fh.write(f"{digest}  {out.name}\n")
    with open(str(out) + ".PROVENANCE.txt", "x") as fh:
        fh.write(f"est_c0\t{Path(est_c0).absolute()}\nest_c0_sha256\t{db_sha}\n"
                 f"est_c0_provenance_sha256\t{prov.get('provenance_sha256')}\nrows\t{n}\nsuperseded_rows\t{n_superseded}\n"
                 f"rejected_rows\t{n_rejected}\noutput_sha256\t{digest}\n")
    return {"output": str(out), "sha256": digest, "rows": n, "superseded_rows": n_superseded, "rejected_rows": n_rejected,
            "est_c0_sha256": db_sha}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--est-c0", required=True)
    ap.add_argument("--species", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        result = export(args.est_c0, args.species, args.out)
    except (B.C0Error, B.S.SchemaError, FileExistsError, OSError) as exc:
        print(f"b6_est_molecule_map failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
