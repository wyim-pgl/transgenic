#!/usr/bin/env python3
"""Eval-only C0 (plan §7.4, §8): Z. mays evaluation evidence without a leakage model.

``b6_build_c0.py`` needs ``--corpus`` to build the rule 4 leakage model (``:1707``) and refuses Z. mays training
roles (``:150``, ``:240``).  Z. mays has no B5/B6 corpus rows, and its evidence is evaluation input only (A46 rule 10,
decision 2: M-EST + M-HQ18 in, M-FLNC + root-tip ONT scoring).  This wrapper reuses the builder's ingestion unchanged
(the same parsers, molecule units, junction correction, schema, QC and publication) with a NULL leakage model:

* no excluded interval, no held-out locus, no 4(c) coordinate (there is no split for the species);
* every selected DATASET_ROLES row must be non-training (``b1_validation_only`` / ``sensitivity_set``), so no audit
  alignment is read (the builder audits training rows only);
* the corpus / splits / flags inputs are replaced by one stub JSON ``<out>.no_leakage_model.json`` recording the
  policy, so the C0 provenance never names a corpus it was not built from;
* ``c0_provenance.policy = "eval_only_no_leakage_model"`` (the ``eval_only`` family of ``b6_bindings``), which every
  training consumer rejects.

Usage (per species x source, then ``b6_merge_evidence_db.py`` as for training C0s):
    b6_build_eval_c0.py --roles DATASET_ROLES.longread_v1.tsv --est-roles DATASET_ROLES.est_v1.tsv --species Zmays \
        --source est --align-root <align> --reference Zmays.fa --gff Zmays.gff3 --provenance PROVENANCE.*.json \
        --out Zmays.est.eval_c0.db --qc Zmays.est.eval_c0.qc.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for p in (_HERE.parent, _HERE.parents[1], _HERE.parents[3] / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import b6_build_c0 as B  # noqa: E402
from transgenic.evidence import leakage as LK  # noqa: E402

EVAL_ONLY_POLICY = "eval_only_no_leakage_model"
EVAL_ONLY_SPECIES = ("Zmays",)


class EvalC0Error(RuntimeError):
    pass


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_fasta(path):
    genome, name, parts = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if line.startswith(">"):
                if name is not None:
                    genome[name] = "".join(parts).upper()
                name, parts = line[1:].split()[0], []
            else:
                parts.append(line)
    if name is not None:
        genome[name] = "".join(parts).upper()
    return genome


def null_leakage_model(species, reference, gff):
    manifest = {"fasta": str(reference), "fasta_sha256": _sha(reference), "gff": str(gff),
                "gff_sha256": _sha(gff), "policy": EVAL_ONLY_POLICY}
    return LK.LeakageModel(species, [], LK.IntervalIndex([]), [], {}, read_fasta(reference), manifest,
                           {"policy": EVAL_ONLY_POLICY})


def build_eval(argv, *, eval_species=EVAL_ONLY_SPECIES):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("--roles", "--est-roles", "--species", "--source", "--align-root", "--reference", "--gff", "--out",
                 "--qc"):
        ap.add_argument(name, required=True)
    ap.add_argument("--provenance", nargs="+", required=True)
    ap.add_argument("--samtools", default="samtools")
    ap.add_argument("--memory-limit", default="2GB")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    if a.species not in eval_species:
        raise EvalC0Error(f"{a.species} is not an eval-only species {eval_species}; training species use "
                          "b6_build_c0.py with their corpus")
    rows, _excluded = B.select_runs(B.read_roles([a.roles, a.est_roles]), a.species, a.source)
    training = [f"{r['dataset']}/{r['run']}" for r in rows if B.ROLE_MAP[r["role"]] == "training_eligible"]
    if training:
        raise EvalC0Error(f"eval-only C0 refuses training role rows {training}")
    out = Path(a.out).absolute()
    stub = Path(str(out) + ".no_leakage_model.json")
    stub.write_text(json.dumps({"policy": EVAL_ONLY_POLICY, "species": a.species,
                                "reference_sha256": _sha(a.reference)}, sort_keys=True) + "\n")
    inner = ["--roles", a.roles, "--est-roles", a.est_roles, "--species", a.species, "--source", a.source,
             "--align-root", a.align_root, "--audit-root", a.align_root, "--corpus", str(stub), "--splits", str(stub),
             "--flags", str(stub), "--reference", a.reference, "--gff", a.gff, "--provenance", *a.provenance,
             "--seed", "0", "--samtools", a.samtools, "--memory-limit", a.memory_limit, "--threads", str(a.threads),
             "--out", str(out), "--qc", a.qc]
    args = B.parse_args(inner)
    orig_model, orig_prov = B.LK.build_leakage_model, B.provenance_hash

    def model(**kw):
        return null_leakage_model(kw["species"], a.reference, a.gff)

    def prov_hash(ctx, args_, argv_, tool_shas, code, duckdb_version, rows_, excluded_rows):
        orig_prov(ctx, args_, argv_, tool_shas, code, duckdb_version, rows_, excluded_rows)
        ctx["provenance"]["policy"] = EVAL_ONLY_POLICY
        ctx["provenance"]["eval_c0_code_sha256"] = _sha(_HERE)
        return hashlib.sha256(json.dumps(ctx["provenance"], sort_keys=True, separators=(",", ":"))
                              .encode()).hexdigest()

    B.LK.build_leakage_model, B.provenance_hash = model, prov_hash
    try:
        return B.build(args, argv)
    finally:
        B.LK.build_leakage_model, B.provenance_hash = orig_model, orig_prov


def main(argv=None):
    rec = build_eval(sys.argv[1:] if argv is None else argv)
    print(json.dumps({k: rec[k] for k in ("species", "source", "status", "output", "output_sha256")}, sort_keys=True))
    return 0 if rec["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
