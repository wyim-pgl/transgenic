#!/usr/bin/env python3
"""Build data/splits/b5_orthogroup_split_v1.tsv from OrthoFinder orthogroups and the reference GFF3s (issue #14; spec §7).

Inputs
  --orthogroups   OrthoFinder Orthogroups/Orthogroups.tsv (columns = species names = manifest species_id; cells = gene ids
                  of the primary proteomes, i.e. the GFF gene ID attribute)
  --unassigned    Orthogroups/Orthogroups_UnassignedGenes.tsv (singletons)
  --species-manifest  data/manifests/b5_species_v1.tsv (gff column; override paths with --gff SP=path)
  --strict-holdout    text file of A. thaliana held-out loci (gene Name or ID, one per line); their whole orthogroups -> test
Rules
  * one row per gene of every training annotation; gene_id is the builder key (gsf_contract.gene_key: a generated code for
    ids longer than 10 characters or with two dots), so the table matches geneList.geneModel of the B5 database;
  * orthogroup-level 75/10/15 with --seed through gsf_contract.make_split; singleton genes (OrthoFinder unassigned, or
    genes without a protein) are their own group and get an empty orthogroup_id;
  * held-out loci and their orthogroups are forced to test (strict_holdout = true on the loci themselves);
  * gsf_contract.validate_split must return no violation (Z. mays absent, groups never span splits, held-out in test).
Outputs: the TSV, a JSON summary (per-species split counts, singletons, unmapped OrthoFinder names, held-out found/missing,
key_of maps GFF id -> builder key per species, sha256 of the TSV).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import types
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRACTIONS = (0.75, 0.10, 0.15)


def _load_contract():
    path = os.path.join(ROOT, "src", "transgenic", "utils", "gsf_contract.py")
    mod = types.ModuleType("gsf_contract")
    mod.__file__ = path
    sys.modules.setdefault("gsf_contract", mod)
    with open(path) as fh:
        exec(compile(fh.read(), path, "exec"), mod.__dict__)
    return mod


def read_manifest(path: str) -> List[Dict[str, str]]:
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def gene_keys(gc, species_id: str, gff_path: str) -> Tuple[List[str], Dict[str, str]]:
    """Builder keys in GFF order and the alias map (GFF ID and Name -> key) for one annotation."""
    keys: List[str] = []
    alias: Dict[str, str] = {}
    with open(gff_path) as fh:
        for gene in gc.parse_gff3(fh, species_code=gc.species_code(species_id)):
            keys.append(gene.gene_id)
            alias[gene.gene_id] = gene.gene_id
            if getattr(gene, "gene_id_original", ""):
                alias.setdefault(gene.gene_id_original, gene.gene_id)
            if getattr(gene, "name_original", ""):
                alias.setdefault(gene.name_original, gene.gene_id)
    return keys, alias


def read_orthogroups(path: str) -> Dict[str, Dict[str, List[str]]]:
    """{orthogroup: {species: [gene ids]}} from an OrthoFinder table."""
    out: Dict[str, Dict[str, List[str]]] = {}
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if not c or not c[0]:
                continue
            og = c[0]
            out[og] = {}
            for sp, cell in zip(header[1:], c[1:]):
                ids = [x.strip() for x in cell.split(",") if x.strip()]
                if ids:
                    out[og][sp] = ids
    return out


def build_rows(gc, manifest_rows, gff_override: Dict[str, str], orthogroups: Dict[str, Dict[str, List[str]]],
               unassigned: Dict[str, Dict[str, List[str]]], holdout_names: Set[str], holdout_species: str, seed: int,
               source_version: str, fractions=FRACTIONS):
    key_of: Dict[str, Dict[str, str]] = {}
    keys_by_species: Dict[str, List[str]] = {}
    for r in manifest_rows:
        sp = r["species_id"]
        keys, alias = gene_keys(gc, sp, gff_override.get(sp, r["gff"]))
        keys_by_species[sp] = keys
        key_of[sp] = alias
    unmapped: Dict[str, List[str]] = defaultdict(list)
    group_of: Dict[Tuple[str, str], str] = {}          # (species, key) -> orthogroup id
    for og, cells in orthogroups.items():
        for sp, ids in cells.items():
            if sp not in key_of:
                unmapped[sp].extend(ids)
                continue
            for gid in ids:
                k = key_of[sp].get(gid)
                if k is None:
                    unmapped[sp].append(gid)
                    continue
                group_of[(sp, k)] = og
    singletons: Dict[str, int] = defaultdict(int)
    for og, cells in unassigned.items():
        for sp, ids in cells.items():
            for gid in ids:
                k = key_of.get(sp, {}).get(gid)
                if k is None:
                    unmapped[sp].append(gid)
                    continue
                group_of.setdefault((sp, k), "")
    no_entry: Dict[str, int] = defaultdict(int)
    for sp, keys in keys_by_species.items():
        for k in keys:
            if (sp, k) not in group_of:
                group_of[(sp, k)] = ""
                no_entry[sp] += 1
    composite = {f"{sp}\t{k}": (og or f"{sp}\t{k}") for (sp, k), og in group_of.items()}
    for (sp, k), og in group_of.items():
        if not og:
            singletons[sp] += 1
    held_ids: Set[str] = set()
    found = 0
    missing: List[str] = []
    for name in sorted(holdout_names):
        k = key_of.get(holdout_species, {}).get(name)
        if k is None:
            missing.append(name)
        else:
            held_ids.add(f"{holdout_species}\t{k}")
            found += 1
    split_of = gc.make_split(composite, seed, tuple(fractions), held_ids)
    rows = []
    for sp in keys_by_species:
        for k in keys_by_species[sp]:
            cid = f"{sp}\t{k}"
            rows.append({"species_id": sp, "gene_id": k, "orthogroup_id": group_of[(sp, k)], "split": split_of[cid],
                         "strict_holdout": cid in held_ids, "seed": seed, "source_version": source_version})
    violations = gc.validate_split(rows)
    per_species: Dict[str, Dict[str, int]] = {}
    for r in rows:
        per_species.setdefault(r["species_id"], {"train": 0, "valid": 0, "test": 0})[r["split"]] += 1
    groups = {r["orthogroup_id"] for r in rows if r["orthogroup_id"]}
    summary = {"seed": seed, "fractions": list(fractions), "source_version": source_version, "rows": len(rows),
               "orthogroups_used": len(groups), "singletons": dict(singletons), "genes_without_orthofinder_entry": dict(no_entry),
               "unmapped_orthofinder_genes": {sp: v for sp, v in unmapped.items() if v},
               "strict_holdout": {"species": holdout_species, "requested": len(holdout_names), "found": found, "missing": missing,
                                  "groups_forced_to_test": len({composite[c] for c in held_ids})},
               "per_species": per_species, "validation_violations": violations, "key_of": key_of}
    return rows, summary


def _parse_kv(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orthogroups", required=True)
    ap.add_argument("--unassigned", required=True)
    ap.add_argument("--species-manifest", required=True)
    ap.add_argument("--gff", action="append", default=[], help="SPECIES=<gff3 path> override of the manifest path (repeatable)")
    ap.add_argument("--strict-holdout", required=True, help="held-out loci list (one gene Name or ID per line)")
    ap.add_argument("--strict-holdout-species", default="Athaliana")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--fractions", type=float, nargs=3, default=list(FRACTIONS))
    ap.add_argument("--source-version", required=True, help="e.g. orthofinder-3.1.5-run1-2026-09-02")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--allow-violations", action="store_true")
    a = ap.parse_args(argv)
    gc = _load_contract()
    manifest_rows = read_manifest(a.species_manifest)
    holdout = {l.strip() for l in open(a.strict_holdout) if l.strip() and not l.startswith("#")}
    rows, summary = build_rows(gc, manifest_rows, _parse_kv(a.gff), read_orthogroups(a.orthogroups), read_orthogroups(a.unassigned),
                               holdout, a.strict_holdout_species, a.seed, a.source_version, tuple(a.fractions))
    if summary["validation_violations"] and not a.allow_violations:
        print("\n".join(summary["validation_violations"][:20]), file=sys.stderr)
        raise SystemExit(f"{len(summary['validation_violations'])} split violation(s); nothing written")
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["species_id", "gene_id", "orthogroup_id", "split", "strict_holdout", "seed", "source_version"])
        for r in rows:
            w.writerow([r["species_id"], r["gene_id"], r["orthogroup_id"], r["split"], "true" if r["strict_holdout"] else "false", r["seed"], r["source_version"]])
    with open(a.out, "rb") as fh:
        summary["sha256"] = hashlib.sha256(fh.read()).hexdigest()
    with open(a.summary, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: summary[k] for k in ("rows", "orthogroups_used", "singletons", "genes_without_orthofinder_entry", "strict_holdout", "per_species", "sha256")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
